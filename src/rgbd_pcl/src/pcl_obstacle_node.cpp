#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include <Eigen/Geometry>
#include <geometry_msgs/msg/point.hpp>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl_conversions/pcl_conversions.h>
#include <rclcpp/rclcpp.hpp>
#include <rgbd_perception_msgs/msg/obstacle.hpp>
#include <rgbd_perception_msgs/msg/obstacle_array.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <tf2_eigen/tf2_eigen.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

using PointT = pcl::PointXYZ;
using CloudT = pcl::PointCloud<PointT>;

class PclObstacleNode : public rclcpp::Node
{
public:
  PclObstacleNode()
  : Node("pcl_obstacle_node"),
    tf_buffer_(this->get_clock()),
    tf_listener_(tf_buffer_)
  {
    declare_parameter<std::string>("depth_topic", "/camera/depth/image_raw");
    declare_parameter<std::string>("camera_info_topic", "/camera/color/camera_info");
    declare_parameter<std::string>("target_frame", "base_link");
    declare_parameter<int>("depth_stride", 4);
    declare_parameter<double>("min_depth_m", 0.2);
    declare_parameter<double>("max_depth_m", 5.0);
    declare_parameter<double>("passthrough_x_min", 0.2);
    declare_parameter<double>("passthrough_x_max", 5.0);
    declare_parameter<double>("passthrough_z_min", -1.0);
    declare_parameter<double>("passthrough_z_max", 2.0);
    declare_parameter<double>("voxel_leaf_m", 0.05);
    declare_parameter<bool>("enable_sor", false);
    declare_parameter<int>("sor_mean_k", 20);
    declare_parameter<double>("sor_stddev", 2.0);
    declare_parameter<std::string>("ground_method", "ransac");  // ransac | height
    declare_parameter<double>("ground_distance_thresh_m", 0.03);
    declare_parameter<int>("ransac_max_iterations", 200);
    declare_parameter<double>("height_thresh_m", 0.12);
    declare_parameter<double>("cluster_tolerance_m", 0.15);
    declare_parameter<int>("cluster_min_points", 20);
    declare_parameter<int>("cluster_max_points", 25000);
    declare_parameter<int>("max_clusters", 30);
    declare_parameter<double>("tf_timeout_sec", 0.1);

    depth_topic_ = get_parameter("depth_topic").as_string();
    info_topic_ = get_parameter("camera_info_topic").as_string();
    target_frame_ = get_parameter("target_frame").as_string();

    rclcpp::SensorDataQoS qos;
    info_sub_ = create_subscription<sensor_msgs::msg::CameraInfo>(
      info_topic_, qos,
      [this](const sensor_msgs::msg::CameraInfo::SharedPtr msg) {
        camera_info_ = *msg;
        have_info_ = true;
      });
    depth_sub_ = create_subscription<sensor_msgs::msg::Image>(
      depth_topic_, qos,
      std::bind(&PclObstacleNode::depthCallback, this, std::placeholders::_1));

    obstacles_pub_ = create_publisher<rgbd_perception_msgs::msg::ObstacleArray>(
      "/perception/obstacles", 10);
    cloud_obstacles_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>(
      "/perception/cloud_obstacles", 10);
    cloud_ground_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>(
      "/perception/cloud_ground", 10);
    markers_pub_ = create_publisher<visualization_msgs::msg::MarkerArray>(
      "/perception/markers/obstacles", 10);

    RCLCPP_INFO(get_logger(),
      "pcl_obstacle_node depth=%s info=%s -> %s (sparse backproject)",
      depth_topic_.c_str(), info_topic_.c_str(), target_frame_.c_str());
  }

private:
  void depthCallback(const sensor_msgs::msg::Image::SharedPtr msg)
  {
    const auto t0 = std::chrono::steady_clock::now();

    if (!have_info_) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "Waiting for camera_info on %s",
        info_topic_.c_str());
      return;
    }

    CloudT::Ptr cloud_cam(new CloudT);
    if (!depthToCloudSparse(*msg, *cloud_cam) || cloud_cam->empty()) {
      return;
    }

    const std::string optical_frame = camera_info_.header.frame_id.empty()
      ? msg->header.frame_id
      : camera_info_.header.frame_id;

    CloudT::Ptr cloud_base(new CloudT);
    if (!transformCloud(*cloud_cam, optical_frame, msg->header.stamp, *cloud_base)) {
      return;
    }

    // Passthrough in base_link: x forward, z up
    CloudT::Ptr cloud_cut(new CloudT);
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

    CloudT::Ptr cloud_voxel(new CloudT);
    {
      pcl::VoxelGrid<PointT> vg;
      vg.setInputCloud(cloud_cut);
      const float leaf = static_cast<float>(get_parameter("voxel_leaf_m").as_double());
      vg.setLeafSize(leaf, leaf, leaf);
      vg.filter(*cloud_voxel);
    }

    if (get_parameter("enable_sor").as_bool() && !cloud_voxel->empty()) {
      CloudT::Ptr cloud_sor(new CloudT);
      pcl::StatisticalOutlierRemoval<PointT> sor;
      sor.setInputCloud(cloud_voxel);
      sor.setMeanK(get_parameter("sor_mean_k").as_int());
      sor.setStddevMulThresh(get_parameter("sor_stddev").as_double());
      sor.filter(*cloud_sor);
      cloud_voxel.swap(cloud_sor);
    }

    CloudT::Ptr obstacles(new CloudT);
    CloudT::Ptr ground(new CloudT);
    segmentGround(cloud_voxel, obstacles, ground);

    publishCloud(cloud_obstacles_pub_, obstacles, msg->header.stamp, target_frame_);
    publishCloud(cloud_ground_pub_, ground, msg->header.stamp, target_frame_);

    auto clusters = euclideanCluster(obstacles);
    publishObstacles(clusters, msg->header.stamp);

    const auto ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - t0).count();
    RCLCPP_DEBUG(get_logger(),
      "pcl pipeline %.1f ms, cam_pts=%zu obstacles=%zu clusters=%zu",
      ms, cloud_cam->size(), obstacles->size(), clusters.size());
  }

  bool depthToCloudSparse(const sensor_msgs::msg::Image & depth, CloudT & cloud)
  {
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
    const rclcpp::Time & stamp, CloudT & cloud_out)
  {
    cloud_out.clear();
    if (frame_in == target_frame_) {
      cloud_out = cloud_in;
      return true;
    }

    geometry_msgs::msg::TransformStamped tf;
    try {
      tf = tf_buffer_.lookupTransform(
        target_frame_, frame_in, stamp,
        tf2::durationFromSec(get_parameter("tf_timeout_sec").as_double()));
    } catch (const tf2::TransformException & ex) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
        "TF %s <- %s failed: %s", target_frame_.c_str(), frame_in.c_str(), ex.what());
      return false;
    }

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

  void splitByHeight(
    const CloudT::Ptr & input, CloudT::Ptr & obstacles, CloudT::Ptr & ground) const
  {
    const float thresh = static_cast<float>(get_parameter("height_thresh_m").as_double());
    ground->clear();
    obstacles->clear();
    ground->reserve(input->size() / 2);
    obstacles->reserve(input->size());
    for (const auto & p : input->points) {
      if (p.z <= thresh) {
        ground->push_back(p);
      } else {
        obstacles->push_back(p);
      }
    }
    ground->width = ground->size();
    ground->height = 1;
    ground->is_dense = true;
    obstacles->width = obstacles->size();
    obstacles->height = 1;
    obstacles->is_dense = true;
  }

  void segmentGround(const CloudT::Ptr & input, CloudT::Ptr & obstacles, CloudT::Ptr & ground)
  {
    obstacles->clear();
    ground->clear();
    if (input->empty()) {
      return;
    }

    const std::string method = get_parameter("ground_method").as_string();
    if (method == "height") {
      splitByHeight(input, obstacles, ground);
      return;
    }

    // Need enough near-floor points or RANSAC will spam "No solution found".
    const float probe_z = static_cast<float>(
      std::max(0.2, get_parameter("height_thresh_m").as_double() * 2.0));
    size_t near_ground = 0;
    for (const auto & p : input->points) {
      if (p.z <= probe_z) {
        ++near_ground;
      }
    }
    const size_t min_ground = static_cast<size_t>(
      std::max(30, static_cast<int>(get_parameter("cluster_min_points").as_int())));
    if (near_ground < min_ground) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 3000,
        "Too few near-ground points (%zu); fallback to height split", near_ground);
      splitByHeight(input, obstacles, ground);
      return;
    }

    pcl::SACSegmentation<PointT> seg;
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PERPENDICULAR_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setDistanceThreshold(get_parameter("ground_distance_thresh_m").as_double());
    seg.setMaxIterations(get_parameter("ransac_max_iterations").as_int());
    Eigen::Vector3f axis(0.0f, 0.0f, 1.0f);
    seg.setAxis(axis);
    seg.setEpsAngle(25.0 * M_PI / 180.0);
    seg.setInputCloud(input);

    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    pcl::ModelCoefficients::Ptr coeffs(new pcl::ModelCoefficients);
    seg.segment(*inliers, *coeffs);

    if (inliers->indices.empty() ||
      inliers->indices.size() < min_ground)
    {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 3000,
        "RANSAC ground failed (inliers=%zu); fallback to height split",
        inliers->indices.size());
      splitByHeight(input, obstacles, ground);
      return;
    }

    pcl::ExtractIndices<PointT> extract;
    extract.setInputCloud(input);
    extract.setIndices(inliers);
    extract.setNegative(false);
    extract.filter(*ground);
    extract.setNegative(true);
    extract.filter(*obstacles);
  }

  std::vector<CloudT::Ptr> euclideanCluster(const CloudT::Ptr & obstacles)
  {
    std::vector<CloudT::Ptr> out;
    if (!obstacles || obstacles->empty()) {
      return out;
    }

    pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>);
    tree->setInputCloud(obstacles);

    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<PointT> ec;
    ec.setClusterTolerance(get_parameter("cluster_tolerance_m").as_double());
    ec.setMinClusterSize(get_parameter("cluster_min_points").as_int());
    ec.setMaxClusterSize(get_parameter("cluster_max_points").as_int());
    ec.setSearchMethod(tree);
    ec.setInputCloud(obstacles);
    ec.extract(cluster_indices);

    const int max_clusters = get_parameter("max_clusters").as_int();
    int count = 0;
    for (const auto & indices : cluster_indices) {
      if (count >= max_clusters) {
        break;
      }
      CloudT::Ptr cluster(new CloudT);
      cluster->reserve(indices.indices.size());
      for (int idx : indices.indices) {
        cluster->push_back(obstacles->points[idx]);
      }
      cluster->width = cluster->size();
      cluster->height = 1;
      cluster->is_dense = true;
      out.push_back(cluster);
      ++count;
    }
    return out;
  }

  void publishCloud(
    const rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr & pub,
    const CloudT::Ptr & cloud, const rclcpp::Time & stamp, const std::string & frame)
  {
    sensor_msgs::msg::PointCloud2 msg;
    pcl::toROSMsg(*cloud, msg);
    msg.header.stamp = stamp;
    msg.header.frame_id = frame;
    pub->publish(msg);
  }

  void publishObstacles(const std::vector<CloudT::Ptr> & clusters, const rclcpp::Time & stamp)
  {
    rgbd_perception_msgs::msg::ObstacleArray arr;
    arr.header.stamp = stamp;
    arr.header.frame_id = target_frame_;

    visualization_msgs::msg::MarkerArray markers;
    visualization_msgs::msg::Marker del;
    del.action = visualization_msgs::msg::Marker::DELETEALL;
    markers.markers.push_back(del);

    int id = 0;
    for (const auto & c : clusters) {
      if (!c || c->empty()) {
        continue;
      }
      float min_x = std::numeric_limits<float>::max();
      float min_y = std::numeric_limits<float>::max();
      float min_z = std::numeric_limits<float>::max();
      float max_x = std::numeric_limits<float>::lowest();
      float max_y = std::numeric_limits<float>::lowest();
      float max_z = std::numeric_limits<float>::lowest();
      double sx = 0, sy = 0, sz = 0;
      for (const auto & p : c->points) {
        min_x = std::min(min_x, p.x); min_y = std::min(min_y, p.y); min_z = std::min(min_z, p.z);
        max_x = std::max(max_x, p.x); max_y = std::max(max_y, p.y); max_z = std::max(max_z, p.z);
        sx += p.x; sy += p.y; sz += p.z;
      }
      const float n = static_cast<float>(c->size());

      rgbd_perception_msgs::msg::Obstacle obs;
      obs.header = arr.header;
      obs.cluster_id = id;
      obs.center.x = sx / n;
      obs.center.y = sy / n;
      obs.center.z = sz / n;
      obs.aabb_min.x = min_x; obs.aabb_min.y = min_y; obs.aabb_min.z = min_z;
      obs.aabb_max.x = max_x; obs.aabb_max.y = max_y; obs.aabb_max.z = max_z;
      obs.num_points = static_cast<int>(c->size());
      arr.obstacles.push_back(obs);

      visualization_msgs::msg::Marker m;
      m.header = arr.header;
      m.ns = "obstacles_aabb";
      m.id = id;
      m.type = visualization_msgs::msg::Marker::CUBE;
      m.action = visualization_msgs::msg::Marker::ADD;
      m.pose.position.x = 0.5 * (min_x + max_x);
      m.pose.position.y = 0.5 * (min_y + max_y);
      m.pose.position.z = 0.5 * (min_z + max_z);
      m.pose.orientation.w = 1.0;
      m.scale.x = std::max(0.02f, max_x - min_x);
      m.scale.y = std::max(0.02f, max_y - min_y);
      m.scale.z = std::max(0.02f, max_z - min_z);
      m.color.r = 0.1f;
      m.color.g = 0.8f;
      m.color.b = 0.2f;
      m.color.a = 0.45f;
      m.lifetime = rclcpp::Duration::from_seconds(0.3);
      markers.markers.push_back(m);
      ++id;
    }

    obstacles_pub_->publish(arr);
    markers_pub_->publish(markers);
  }

  std::string depth_topic_;
  std::string info_topic_;
  std::string target_frame_;
  bool have_info_{false};
  sensor_msgs::msg::CameraInfo camera_info_;
  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr depth_sub_;
  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr info_sub_;
  rclcpp::Publisher<rgbd_perception_msgs::msg::ObstacleArray>::SharedPtr obstacles_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_obstacles_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_ground_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr markers_pub_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<PclObstacleNode>());
  rclcpp::shutdown();
  return 0;
}
