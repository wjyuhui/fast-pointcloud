#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <Eigen/Geometry>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <tf2_eigen/tf2_eigen.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

using PointT = pcl::PointXYZ;
using CloudT = pcl::PointCloud<PointT>;

class CloudWorkspaceNode : public rclcpp::Node
{
public:
  CloudWorkspaceNode()
  : Node("cloud_workspace_node"),
    tf_buffer_(this->get_clock()),
    tf_listener_(tf_buffer_)
  {
    declare_parameter<std::string>("depth_topic", "/camera/depth/image_raw");
    declare_parameter<std::string>("camera_info_topic", "/camera/color/camera_info");
    declare_parameter<std::string>("output_cloud_topic", "/perception/cloud_workspace");
    declare_parameter<std::string>("target_frame", "base_link");
    declare_parameter<int>("depth_stride", 4);
    declare_parameter<double>("min_depth_m", 0.2);
    declare_parameter<double>("max_depth_m", 5.0);
    declare_parameter<double>("passthrough_x_min", 0.2);
    declare_parameter<double>("passthrough_x_max", 5.0);
    declare_parameter<double>("passthrough_z_min", -1.0);
    declare_parameter<double>("passthrough_z_max", 2.0);
    declare_parameter<double>("voxel_leaf_m", 0.05);
    declare_parameter<double>("tf_timeout_sec", 0.1);
    declare_parameter<bool>("projection_enable_openmp", false);
    declare_parameter<int>("projection_openmp_threads", 4);
    declare_parameter<bool>("projection_enable_neon", false);

    depth_topic_ = get_parameter("depth_topic").as_string();
    info_topic_ = get_parameter("camera_info_topic").as_string();
    output_topic_ = get_parameter("output_cloud_topic").as_string();
    target_frame_ = get_parameter("target_frame").as_string();
    projection_enable_openmp_ = get_parameter("projection_enable_openmp").as_bool();
    projection_openmp_threads_ = get_parameter("projection_openmp_threads").as_int();
    projection_enable_neon_ = get_parameter("projection_enable_neon").as_bool();

    core_samples_.reserve(kMeasureFrames);
    voxel_samples_.reserve(kMeasureFrames);
    callback_samples_.reserve(kMeasureFrames);

    rclcpp::SensorDataQoS qos;
    info_sub_ = create_subscription<sensor_msgs::msg::CameraInfo>(
      info_topic_, qos,
      [this](const sensor_msgs::msg::CameraInfo::SharedPtr msg) {
        camera_info_ = *msg;
        have_info_ = true;
      });
    depth_sub_ = create_subscription<sensor_msgs::msg::Image>(
      depth_topic_, qos,
      std::bind(&CloudWorkspaceNode::depthCallback, this, std::placeholders::_1));
    cloud_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>(output_topic_, qos);

    RCLCPP_INFO(get_logger(),
      "cloud_workspace_node depth=%s info=%s -> %s (%s) openmp=%s threads=%d neon=%s",
      depth_topic_.c_str(), info_topic_.c_str(), output_topic_.c_str(),
      target_frame_.c_str(),
      projection_enable_openmp_ ? "true" : "false",
      projection_openmp_threads_,
      projection_enable_neon_ ? "true" : "false");
  }

  ~CloudWorkspaceNode() override
  {
    if (core_samples_.size() >= kMinDumpFrames) {
      logPerf();
    }
  }

private:
  static constexpr size_t kWarmupFrames = 30;
  static constexpr size_t kMeasureFrames = 1000;
  static constexpr size_t kMinDumpFrames = 200;

  static double calculateP95(std::vector<double> samples)
  {
    if (samples.empty()) {
      return 0.0;
    }

    std::sort(samples.begin(), samples.end());

    const size_t index =
      static_cast<size_t>(std::ceil(0.95 * static_cast<double>(samples.size()))) - 1;
    return samples[std::min(index, samples.size() - 1)];
  }

  void logPerf() const
  {
    RCLCPP_INFO(
      get_logger(),
      "PERF V0 samples=%zu core_p95=%.3f ms voxel_p95=%.3f ms callback_p95=%.3f ms",
      core_samples_.size(),
      calculateP95(core_samples_),
      calculateP95(voxel_samples_),
      calculateP95(callback_samples_));
  }

  void recordTiming(double core_ms, double voxel_ms, double callback_ms)
  {
    ++successful_frames_;
    if (successful_frames_ <= kWarmupFrames) {
      return;
    }

    core_samples_.push_back(core_ms);
    voxel_samples_.push_back(voxel_ms);
    callback_samples_.push_back(callback_ms);

    if (core_samples_.size() < kMeasureFrames) {
      return;
    }

    logPerf();
    core_samples_.clear();
    voxel_samples_.clear();
    callback_samples_.clear();
  }

  void depthCallback(const sensor_msgs::msg::Image::SharedPtr msg)
  {
    const auto callback_begin = std::chrono::steady_clock::now();

    if (!have_info_) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "Waiting for camera_info on %s",
        info_topic_.c_str());
      return;
    }

    const auto core_begin = std::chrono::steady_clock::now();

    CloudT::Ptr cloud_cam(new CloudT);
    if (!depthToCloudSparse(*msg, *cloud_cam) || cloud_cam->empty()) {
      return;
    }

    const std::string optical_frame = camera_info_.header.frame_id.empty()
      ? msg->header.frame_id
      : camera_info_.header.frame_id;

    double tf_lookup_ms = 0.0;
    CloudT::Ptr cloud_base(new CloudT);
    if (!transformCloud(
          *cloud_cam, optical_frame, msg->header.stamp, *cloud_base, tf_lookup_ms))
    {
      return;
    }

    CloudT::Ptr cloud_cut(new CloudT);  // 下面整个步骤是做点云切割，去掉无效点
    {
      pcl::PassThrough<PointT> pass;
      pass.setInputCloud(cloud_base);
      pass.setFilterFieldName("x");
      pass.setFilterLimits(
        get_parameter("passthrough_x_min").as_double(),
        get_parameter("passthrough_x_max").as_double());
      pass.filter(*cloud_cut);

      CloudT::Ptr tmp(new CloudT);
      pass.setInputCloud(cloud_cut);
      pass.setFilterFieldName("z");
      pass.setFilterLimits(
        get_parameter("passthrough_z_min").as_double(),
        get_parameter("passthrough_z_max").as_double());
      pass.filter(*tmp);
      cloud_cut.swap(tmp);
    }

    const auto core_end = std::chrono::steady_clock::now();
    const double core_with_lookup_ms =
      std::chrono::duration<double, std::milli>(core_end - core_begin).count();
    const double core_ms = std::max(0.0, core_with_lookup_ms - tf_lookup_ms);

    const auto voxel_begin = std::chrono::steady_clock::now();
    CloudT::Ptr cloud_voxel(new CloudT);
    {
      pcl::VoxelGrid<PointT> vg;
      vg.setInputCloud(cloud_cut);
      const float leaf = static_cast<float>(get_parameter("voxel_leaf_m").as_double());
      vg.setLeafSize(leaf, leaf, leaf);
      vg.filter(*cloud_voxel);
    }
    const auto voxel_end = std::chrono::steady_clock::now();
    const double voxel_ms =
      std::chrono::duration<double, std::milli>(voxel_end - voxel_begin).count();

    sensor_msgs::msg::PointCloud2 out;
    pcl::toROSMsg(*cloud_voxel, out);
    out.header.stamp = msg->header.stamp;
    out.header.frame_id = target_frame_;
    cloud_pub_->publish(out);

    const auto callback_end = std::chrono::steady_clock::now();
    const double callback_ms =
      std::chrono::duration<double, std::milli>(callback_end - callback_begin).count();
    recordTiming(core_ms, voxel_ms, callback_ms);
  }

  bool depthToCloudSparse(const sensor_msgs::msg::Image & depth, CloudT & cloud)
  {
    // 在这里稀疏反投影depthToCloudSparse
    // depth_stride 设置多少个像素取一个点
    const int stride = std::max(1, static_cast<int>(get_parameter("depth_stride").as_int()));
    const float zmin = static_cast<float>(get_parameter("min_depth_m").as_double());
    const float zmax = static_cast<float>(get_parameter("max_depth_m").as_double());
    const double fx = camera_info_.k[0];
    const double fy = camera_info_.k[4];
    const double cx = camera_info_.k[2];
    const double cy = camera_info_.k[5];
    if (!(fx > 1e-6 && fy > 1e-6)) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "Invalid camera_info intrinsics");
      return false;
    }

    const int width = static_cast<int>(depth.width);
    const int height = static_cast<int>(depth.height);
    if (width <= 0 || height <= 0 || depth.data.empty()) {
      return false;
    }

    cloud.clear();
    cloud.reserve(static_cast<size_t>((width / stride) * (height / stride)));

    // 在这里稀疏反投影depthToCloudSparse  核心函数
    auto push_z = [&](int u, int v, float z) {
      if (!std::isfinite(z) || z < zmin || z > zmax) {
        return;
      }
      PointT p;
      p.x = static_cast<float>((static_cast<double>(u) - cx) * static_cast<double>(z) / fx);
      p.y = static_cast<float>((static_cast<double>(v) - cy) * static_cast<double>(z) / fy);
      p.z = z;
      cloud.push_back(p);
    };

    if (depth.encoding == "16UC1" || depth.encoding == "mono16") {
      if (depth.step < static_cast<size_t>(width) * sizeof(uint16_t)) {
        return false;
      }

      // 遍历一次深度图像
      for (int v = 0; v < height; v += stride) {
        const auto * row = reinterpret_cast<const uint16_t *>(
          depth.data.data() + static_cast<size_t>(v) * depth.step);
        for (int u = 0; u < width; u += stride) {
          push_z(u, v, static_cast<float>(row[u]) * 0.001f);
        }
      }
    } else if (depth.encoding == "32FC1") {
      if (depth.step < static_cast<size_t>(width) * sizeof(float)) {
        return false;
      }

      // 遍历一次深度图像
      for (int v = 0; v < height; v += stride) {
        const auto * row = reinterpret_cast<const float *>(
          depth.data.data() + static_cast<size_t>(v) * depth.step);
        for (int u = 0; u < width; u += stride) {
          push_z(u, v, row[u]);
        }
      }
    } else {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
        "Unsupported depth encoding: %s", depth.encoding.c_str());
      return false;
    }

    cloud.width = cloud.size();
    cloud.height = 1;
    cloud.is_dense = true;
    return !cloud.empty();
  }

  bool transformCloud(
    const CloudT & cloud_in, const std::string & frame_in,
    const rclcpp::Time & stamp, CloudT & cloud_out, double & tf_lookup_ms)
  {
    tf_lookup_ms = 0.0;
    cloud_out.clear();
    if (frame_in == target_frame_) {
      cloud_out = cloud_in;
      return true;
    }

    geometry_msgs::msg::TransformStamped tf;
    const auto tf_lookup_begin = std::chrono::steady_clock::now();
    try {
      tf = tf_buffer_.lookupTransform(
        target_frame_, frame_in, stamp,
        tf2::durationFromSec(get_parameter("tf_timeout_sec").as_double()));
    } catch (const tf2::TransformException & ex) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
        "TF %s <- %s failed: %s", target_frame_.c_str(), frame_in.c_str(), ex.what());
      return false;
    }
    tf_lookup_ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - tf_lookup_begin).count();

    const Eigen::Affine3d T = tf2::transformToEigen(tf.transform);
    cloud_out.reserve(cloud_in.size());
    for (const auto & p : cloud_in.points) {
      const Eigen::Vector3d q = T * Eigen::Vector3d(p.x, p.y, p.z);
      PointT o;
      o.x = static_cast<float>(q.x());
      o.y = static_cast<float>(q.y());
      o.z = static_cast<float>(q.z());
      cloud_out.push_back(o);
    }
    cloud_out.width = cloud_out.size();
    cloud_out.height = 1;
    cloud_out.is_dense = true;
    return true;
  }

  std::string depth_topic_;
  std::string info_topic_;
  std::string output_topic_;
  std::string target_frame_;
  bool projection_enable_openmp_{false};
  int projection_openmp_threads_{4};
  bool projection_enable_neon_{false};
  size_t successful_frames_{0};  // 成功处理并 publish 的帧计数。
  std::vector<double> core_samples_;  // 预热之后，每一成功帧的三段耗时（ms）各存一份：
  std::vector<double> voxel_samples_;
  std::vector<double> callback_samples_;
  bool have_info_{false};
  sensor_msgs::msg::CameraInfo camera_info_;
  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr depth_sub_;
  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr info_sub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_pub_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<CloudWorkspaceNode>());
  rclcpp::shutdown();
  return 0;
}
