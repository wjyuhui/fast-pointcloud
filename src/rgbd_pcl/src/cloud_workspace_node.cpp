#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <Eigen/Geometry>
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
      "PERF V2 samples=%zu core_p95=%.3f ms voxel_p95=%.3f ms callback_p95=%.3f ms",
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

    const std::string optical_frame = camera_info_.header.frame_id.empty()
      ? msg->header.frame_id
      : camera_info_.header.frame_id;

    Eigen::Affine3d transform = Eigen::Affine3d::Identity();
    double tf_lookup_ms = 0.0;
    if (!lookupTransformMatrix(optical_frame, msg->header.stamp, transform, tf_lookup_ms)) {
      return;
    }

    const auto core_begin = std::chrono::steady_clock::now();
    CloudT::Ptr cloud_roi(new CloudT);
    if (!depthToWorkspaceScalar(*msg, transform, *cloud_roi)) {
      return;
    }
    const auto core_end = std::chrono::steady_clock::now();
    const double core_ms =
      std::chrono::duration<double, std::milli>(core_end - core_begin).count();

    const auto voxel_begin = std::chrono::steady_clock::now();
    CloudT::Ptr cloud_voxel(new CloudT);
    {
      pcl::VoxelGrid<PointT> vg;
      vg.setInputCloud(cloud_roi);
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

  bool lookupTransformMatrix(
    const std::string & frame_in,
    const rclcpp::Time & stamp,
    Eigen::Affine3d & transform,
    double & tf_lookup_ms)
  {
    tf_lookup_ms = 0.0;
    if (frame_in == target_frame_) {
      transform = Eigen::Affine3d::Identity();
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
    transform = tf2::transformToEigen(tf.transform);
    return true;
  }

  bool depthToWorkspaceScalar(
    const sensor_msgs::msg::Image & depth,
    const Eigen::Affine3d & transform,
    CloudT & cloud_roi)
  {
    const int stride = std::max(1, static_cast<int>(get_parameter("depth_stride").as_int()));
    const float zmin = static_cast<float>(get_parameter("min_depth_m").as_double());
    const float zmax = static_cast<float>(get_parameter("max_depth_m").as_double());
    const float xmin = static_cast<float>(get_parameter("passthrough_x_min").as_double());
    const float xmax = static_cast<float>(get_parameter("passthrough_x_max").as_double());
    const float zmin_roi = static_cast<float>(get_parameter("passthrough_z_min").as_double());
    const float zmax_roi = static_cast<float>(get_parameter("passthrough_z_max").as_double());

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

    if (!ensureRayCache(width, height, stride, fx, fy, cx, cy)) {
      return false;
    }

    const Eigen::Matrix4d & M = transform.matrix();
    const double r00 = M(0, 0);
    const double r01 = M(0, 1);
    const double r02 = M(0, 2);
    const double tx = M(0, 3);
    const double r10 = M(1, 0);
    const double r11 = M(1, 1);
    const double r12 = M(1, 2);
    const double ty = M(1, 3);
    const double r20 = M(2, 0);
    const double r21 = M(2, 1);
    const double r22 = M(2, 2);
    const double tz = M(2, 3);

    cloud_roi.clear();
    cloud_roi.reserve(ray_x_.size() * ray_y_.size());

    auto process_z = [&](int ui, int vi, float z) {
      if (!std::isfinite(z) || z < zmin || z > zmax) {
        return;
      }

      const double x_camera = static_cast<double>(ray_x_[static_cast<size_t>(ui)]) * z;
      const double y_camera = static_cast<double>(ray_y_[static_cast<size_t>(vi)]) * z;
      const double z_camera = static_cast<double>(z);

      const float x_target = static_cast<float>(r00 * x_camera + r01 * y_camera + r02 * z_camera + tx);
      const float y_target = static_cast<float>(r10 * x_camera + r11 * y_camera + r12 * z_camera + ty);
      const float z_target = static_cast<float>(r20 * x_camera + r21 * y_camera + r22 * z_camera + tz);

      if (x_target < xmin || x_target > xmax || z_target < zmin_roi || z_target > zmax_roi) {
        return;
      }

      PointT p;
      p.x = x_target;
      p.y = y_target;
      p.z = z_target;
      cloud_roi.push_back(p);
    };

    if (depth.encoding == "16UC1" || depth.encoding == "mono16") {
      if (depth.step < static_cast<size_t>(width) * sizeof(uint16_t)) {
        return false;
      }

      for (int vi = 0, v = 0; v < height; v += stride, ++vi) {
        const auto * row = reinterpret_cast<const uint16_t *>(
          depth.data.data() + static_cast<size_t>(v) * depth.step);
        for (int ui = 0, u = 0; u < width; u += stride, ++ui) {
          process_z(ui, vi, static_cast<float>(row[u]) * 0.001f);
        }
      }
    } else if (depth.encoding == "32FC1") {
      if (depth.step < static_cast<size_t>(width) * sizeof(float)) {
        return false;
      }

      for (int vi = 0, v = 0; v < height; v += stride, ++vi) {
        const auto * row = reinterpret_cast<const float *>(
          depth.data.data() + static_cast<size_t>(v) * depth.step);
        for (int ui = 0, u = 0; u < width; u += stride, ++ui) {
          process_z(ui, vi, row[u]);
        }
      }
    } else {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
        "Unsupported depth encoding: %s", depth.encoding.c_str());
      return false;
    }

    cloud_roi.width = cloud_roi.size();
    cloud_roi.height = 1;
    cloud_roi.is_dense = true;
    return !cloud_roi.empty();
  }

  bool ensureRayCache(
    int width, int height, int stride,
    double fx, double fy, double cx, double cy)
  {
    if (camera_info_.width > 0 && camera_info_.height > 0 &&
        (static_cast<int>(camera_info_.width) != width ||
         static_cast<int>(camera_info_.height) != height))
    {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 2000,
        "depth size %dx%d != camera_info %ux%u",
        width, height, camera_info_.width, camera_info_.height);
      return false;
    }

    if (cached_width_ == width && cached_height_ == height && cached_stride_ == stride &&
        cached_fx_ == fx && cached_fy_ == fy && cached_cx_ == cx && cached_cy_ == cy)
    {
      return true;
    }

    const int nx = (width + stride - 1) / stride;
    const int ny = (height + stride - 1) / stride;
    ray_x_.resize(static_cast<size_t>(nx));
    ray_y_.resize(static_cast<size_t>(ny));
    for (int ui = 0, u = 0; u < width; u += stride, ++ui) {
      ray_x_[static_cast<size_t>(ui)] =
        static_cast<float>((static_cast<double>(u) - cx) / fx);
    }
    for (int vi = 0, v = 0; v < height; v += stride, ++vi) {
      ray_y_[static_cast<size_t>(vi)] =
        static_cast<float>((static_cast<double>(v) - cy) / fy);
    }

    cached_width_ = width;
    cached_height_ = height;
    cached_stride_ = stride;
    cached_fx_ = fx;
    cached_fy_ = fy;
    cached_cx_ = cx;
    cached_cy_ = cy;
    return true;
  }

  std::string depth_topic_;
  std::string info_topic_;
  std::string output_topic_;
  std::string target_frame_;
  bool projection_enable_openmp_{false};
  int projection_openmp_threads_{4};
  bool projection_enable_neon_{false};
  size_t successful_frames_{0};
  std::vector<double> core_samples_;
  std::vector<double> voxel_samples_;
  std::vector<double> callback_samples_;
  bool have_info_{false};
  std::vector<float> ray_x_;
  std::vector<float> ray_y_;
  int cached_width_{0};
  int cached_height_{0};
  int cached_stride_{0};
  double cached_fx_{0.0};
  double cached_fy_{0.0};
  double cached_cx_{0.0};
  double cached_cy_{0.0};
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
