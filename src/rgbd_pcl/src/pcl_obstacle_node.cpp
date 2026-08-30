#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include <Eigen/Geometry>
#include <geometry_msgs/msg/point.hpp>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl_conversions/pcl_conversions.h>
#include <rclcpp/rclcpp.hpp>
#include <rgbd_perception_msgs/msg/obstacle.hpp>
#include <rgbd_perception_msgs/msg/obstacle_array.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

using PointT = pcl::PointXYZ;
using CloudT = pcl::PointCloud<PointT>;

namespace {

double elapsedMs(const std::chrono::steady_clock::time_point & begin)
{
  return std::chrono::duration<double, std::milli>(
    std::chrono::steady_clock::now() - begin).count();
}

struct Percentiles
{
  double p50{0.0};
  double p95{0.0};
  double p99{0.0};
};

struct SizePercentiles
{
  size_t p50{0};
  size_t p95{0};
  size_t p99{0};
};

size_t percentileIndex(size_t n, double p)
{
  if (n == 0) {
    return 0;
  }
  const size_t index = static_cast<size_t>(std::ceil(p * static_cast<double>(n))) - 1;
  return std::min(index, n - 1);
}

Percentiles calculatePercentiles(std::vector<double> samples)
{
  Percentiles out;
  if (samples.empty()) {
    return out;
  }
  std::sort(samples.begin(), samples.end());
  out.p50 = samples[percentileIndex(samples.size(), 0.50)];
  out.p95 = samples[percentileIndex(samples.size(), 0.95)];
  out.p99 = samples[percentileIndex(samples.size(), 0.99)];
  return out;
}

SizePercentiles calculateSizePercentiles(std::vector<size_t> samples)
{
  SizePercentiles out;
  if (samples.empty()) {
    return out;
  }
  std::sort(samples.begin(), samples.end());
  out.p50 = samples[percentileIndex(samples.size(), 0.50)];
  out.p95 = samples[percentileIndex(samples.size(), 0.95)];
  out.p99 = samples[percentileIndex(samples.size(), 0.99)];
  return out;
}

}  // namespace

class PclObstacleNode : public rclcpp::Node
{
public:
  PclObstacleNode()
  : Node("pcl_obstacle_node")
  {
    declare_parameter<std::string>("input_cloud_topic", "/perception/cloud_workspace");
    declare_parameter<std::string>("target_frame", "base_link");
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
    declare_parameter<bool>("enable_clustering", false);
    declare_parameter<bool>("ransac_enable_openmp", false);
    declare_parameter<int>("ransac_openmp_threads", 4);
    declare_parameter<bool>("ransac_enable_neon", false);
    declare_parameter<bool>("profiling_enable", true);
    declare_parameter<int>("profiling_warmup_frames", 30);
    declare_parameter<int>("profiling_window_frames", 1000);

    input_topic_ = get_parameter("input_cloud_topic").as_string();
    target_frame_ = get_parameter("target_frame").as_string();
    enable_clustering_ = get_parameter("enable_clustering").as_bool();
    ransac_enable_openmp_ = get_parameter("ransac_enable_openmp").as_bool();
    ransac_openmp_threads_ = get_parameter("ransac_openmp_threads").as_int();
    ransac_enable_neon_ = get_parameter("ransac_enable_neon").as_bool();
    profiling_enable_ = get_parameter("profiling_enable").as_bool();
    profiling_warmup_frames_ = get_parameter("profiling_warmup_frames").as_int();
    profiling_window_frames_ = get_parameter("profiling_window_frames").as_int();
    if (profiling_warmup_frames_ < 0) {
      profiling_warmup_frames_ = 0;
    }
    if (profiling_window_frames_ < 1) {
      profiling_window_frames_ = 1;
    }
    window_.reserve(static_cast<size_t>(profiling_window_frames_));

    rclcpp::SensorDataQoS qos;
    cloud_sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
      input_topic_, qos,
      std::bind(&PclObstacleNode::cloudCallback, this, std::placeholders::_1));

    obstacles_pub_ = create_publisher<rgbd_perception_msgs::msg::ObstacleArray>(
      "/perception/obstacles", 10);
    cloud_obstacles_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>(
      "/perception/cloud_obstacles", 10);
    cloud_ground_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>(
      "/perception/cloud_ground", 10);
    markers_pub_ = create_publisher<visualization_msgs::msg::MarkerArray>(
      "/perception/markers/obstacles", 10);

    RCLCPP_INFO(get_logger(),
      "pcl_obstacle_node cloud=%s frame=%s (ground/cluster) clustering=%s "
      "openmp=%s threads=%d neon=%s",
      input_topic_.c_str(), target_frame_.c_str(),
      enable_clustering_ ? "true" : "false",
      ransac_enable_openmp_ ? "true" : "false",
      ransac_openmp_threads_,
      ransac_enable_neon_ ? "true" : "false");
    RCLCPP_INFO(get_logger(),
      "pcl_obstacle_node profiling=%s warmup=%d window=%d",
      profiling_enable_ ? "true" : "false",
      profiling_warmup_frames_,
      profiling_window_frames_);
  }

  ~PclObstacleNode() override
  {
    if (profiling_enable_ && window_.callback_ms.size() >= 200) {
      logProfilingWindow();
    }
  }

private:
  enum class GroundPath
  {
    kUnknown,
    kDirectHeight,
    kTooFewNearGround,
    kRansacFailed,
    kRansacOk
  };

  struct FrameProfile
  {
    double from_ros_ms{0.0};
    double sor_ms{0.0};
    double near_ground_scan_ms{0.0};
    double ransac_ms{0.0};
    double ground_extract_ms{0.0};
    double height_split_ms{0.0};
    double ground_total_ms{0.0};
    double cloud_to_ros_ms{0.0};
    double cloud_publish_call_ms{0.0};
    double kdtree_ms{0.0};
    double cluster_extract_ms{0.0};
    double cluster_materialize_ms{0.0};
    double cluster_total_ms{0.0};
    double bbox_message_ms{0.0};
    double obstacle_publish_call_ms{0.0};
    double callback_ms{0.0};

    size_t input_points{0};
    size_t near_ground_points{0};
    size_t ground_points{0};
    size_t obstacle_points{0};
    size_t cluster_candidates{0};
    size_t published_clusters{0};
    size_t clustered_points{0};

    bool sor_executed{false};
    bool near_ground_scan_executed{false};
    bool ransac_executed{false};
    bool ground_extract_executed{false};
    bool height_split_executed{false};
    bool kdtree_executed{false};
    bool cluster_extract_executed{false};
    bool cluster_materialize_executed{false};
    GroundPath ground_path{GroundPath::kUnknown};
  };

  struct PublishTiming
  {
    double conversion_ms{0.0};
    double publish_call_ms{0.0};
  };

  struct ProfilingWindow
  {
    std::vector<double> from_ros_ms;
    std::vector<double> sor_ms;
    std::vector<double> near_ground_scan_ms;
    std::vector<double> ransac_ms;
    std::vector<double> ground_extract_ms;
    std::vector<double> height_split_ms;
    std::vector<double> ground_total_ms;
    std::vector<double> cloud_to_ros_ms;
    std::vector<double> cloud_publish_call_ms;
    std::vector<double> kdtree_ms;
    std::vector<double> cluster_extract_ms;
    std::vector<double> cluster_materialize_ms;
    std::vector<double> cluster_total_ms;
    std::vector<double> bbox_message_ms;
    std::vector<double> obstacle_publish_call_ms;
    std::vector<double> callback_ms;
    std::vector<size_t> input_points;
    std::vector<size_t> near_ground_points;
    std::vector<size_t> ground_points;
    std::vector<size_t> obstacle_points;
    std::vector<size_t> cluster_candidates;
    std::vector<size_t> published_clusters;
    std::vector<size_t> clustered_points;
    size_t path_direct_height{0};
    size_t path_too_few{0};
    size_t path_ransac_failed{0};
    size_t path_ransac_ok{0};
    size_t empty_input_frames{0};
    size_t failed_frames{0};

    void reserve(size_t n)
    {
      from_ros_ms.reserve(n);
      sor_ms.reserve(n);
      near_ground_scan_ms.reserve(n);
      ransac_ms.reserve(n);
      ground_extract_ms.reserve(n);
      height_split_ms.reserve(n);
      ground_total_ms.reserve(n);
      cloud_to_ros_ms.reserve(n);
      cloud_publish_call_ms.reserve(n);
      kdtree_ms.reserve(n);
      cluster_extract_ms.reserve(n);
      cluster_materialize_ms.reserve(n);
      cluster_total_ms.reserve(n);
      bbox_message_ms.reserve(n);
      obstacle_publish_call_ms.reserve(n);
      callback_ms.reserve(n);
      input_points.reserve(n);
      near_ground_points.reserve(n);
      ground_points.reserve(n);
      obstacle_points.reserve(n);
      cluster_candidates.reserve(n);
      published_clusters.reserve(n);
      clustered_points.reserve(n);
    }

    void clearKeepCapacity()
    {
      from_ros_ms.clear();
      sor_ms.clear();
      near_ground_scan_ms.clear();
      ransac_ms.clear();
      ground_extract_ms.clear();
      height_split_ms.clear();
      ground_total_ms.clear();
      cloud_to_ros_ms.clear();
      cloud_publish_call_ms.clear();
      kdtree_ms.clear();
      cluster_extract_ms.clear();
      cluster_materialize_ms.clear();
      cluster_total_ms.clear();
      bbox_message_ms.clear();
      obstacle_publish_call_ms.clear();
      callback_ms.clear();
      input_points.clear();
      near_ground_points.clear();
      ground_points.clear();
      obstacle_points.clear();
      cluster_candidates.clear();
      published_clusters.clear();
      clustered_points.clear();
      path_direct_height = 0;
      path_too_few = 0;
      path_ransac_failed = 0;
      path_ransac_ok = 0;
      empty_input_frames = 0;
      failed_frames = 0;
    }
  };

  // 回调函数，也是主要的工作函数，收到点云数据后，进行处理， 每一个步骤几乎封装成另一个函数在此回调函数中进行调用
  void cloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
  {
    const auto t0 = std::chrono::steady_clock::now();
    FrameProfile * profile = nullptr;
    FrameProfile frame_profile;
    if (profiling_enable_) {
      profile = &frame_profile;
    }

    CloudT::Ptr cloud(new CloudT);
    if (profile) {
      const auto stage_begin = std::chrono::steady_clock::now();
      pcl::fromROSMsg(*msg, *cloud);  // 从Msg中取出点云数据，存入cloud中
      profile->from_ros_ms = elapsedMs(stage_begin);
      profile->input_points = cloud->size();
    } else {
      pcl::fromROSMsg(*msg, *cloud);  // 从Msg中取出点云数据，存入cloud中
    }
    if (cloud->empty()) {
      if (profiling_enable_ &&
        successful_frames_ > static_cast<size_t>(profiling_warmup_frames_))
      {
        ++window_.empty_input_frames;
      }
      return;
    }

    const std::string frame = msg->header.frame_id.empty()
      ? target_frame_
      : msg->header.frame_id;

    if (get_parameter("enable_sor").as_bool()) {  // 如果开启SOR后，下面进行SOR操作
      /**
        原理：每个点看周围 K 个邻居的平均距离；整片点云统计出μ、σ；距离超过μ+倍数×σ的点丢掉。
      */
      CloudT::Ptr cloud_sor(new CloudT);
      pcl::StatisticalOutlierRemoval<PointT> sor;  // 仍然是创建一个滤波器（类实例），然后设置参数，然后开始过滤，结果存入新建变量中。
      if (profile) {
        const auto stage_begin = std::chrono::steady_clock::now();
        sor.setInputCloud(cloud);
        sor.setMeanK(get_parameter("sor_mean_k").as_int());
        sor.setStddevMulThresh(get_parameter("sor_stddev").as_double());
        sor.filter(*cloud_sor);
        profile->sor_ms = elapsedMs(stage_begin);
        profile->sor_executed = true;
      } else {
        sor.setInputCloud(cloud);
        sor.setMeanK(get_parameter("sor_mean_k").as_int());
        sor.setStddevMulThresh(get_parameter("sor_stddev").as_double());
        sor.filter(*cloud_sor);
      }
      cloud.swap(cloud_sor);
    }

    CloudT::Ptr obstacles(new CloudT);
    CloudT::Ptr ground(new CloudT);
    if (profile) {
      const auto stage_begin = std::chrono::steady_clock::now();
      segmentGround(cloud, obstacles, ground, profile);  // 地面分割， 分割出地面和障碍物
      profile->ground_total_ms = elapsedMs(stage_begin);
      profile->ground_points = ground->size();
      profile->obstacle_points = obstacles->size();
    } else {
      segmentGround(cloud, obstacles, ground, nullptr);  // 地面分割， 分割出地面和障碍物
    }

    // 发布地面和障碍物点云
    const PublishTiming obstacles_cloud_timing =
      publishCloud(cloud_obstacles_pub_, obstacles, msg->header.stamp, frame, profile);
    const PublishTiming ground_cloud_timing =
      publishCloud(cloud_ground_pub_, ground, msg->header.stamp, frame, profile);
    if (profile) {
      profile->cloud_to_ros_ms =
        obstacles_cloud_timing.conversion_ms + ground_cloud_timing.conversion_ms;
      profile->cloud_publish_call_ms =
        obstacles_cloud_timing.publish_call_ms + ground_cloud_timing.publish_call_ms;
    }

    std::vector<CloudT::Ptr> clusters;
    if (enable_clustering_) {
      if (profile) {
        const auto stage_begin = std::chrono::steady_clock::now();
        clusters = euclideanCluster(obstacles, profile);  // 欧式聚类， 将障碍物点云分割成多个聚类
        profile->cluster_total_ms = elapsedMs(stage_begin);
      } else {
        clusters = euclideanCluster(obstacles, nullptr);  // 欧式聚类， 将障碍物点云分割成多个聚类
      }
      publishObstacles(clusters, msg->header.stamp, frame, profile);  // 发布障碍物聚类
    }

    if (profile) {
      profile->callback_ms = elapsedMs(t0);
      recordProfile(*profile);
    }

    const auto ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - t0).count();
    RCLCPP_DEBUG(get_logger(),
      "pcl pipeline %.1f ms, in_pts=%zu obstacles=%zu clusters=%zu",
      ms, cloud->size(), obstacles->size(), clusters.size());
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

  void segmentGround(
    const CloudT::Ptr & input,
    CloudT::Ptr & obstacles,
    CloudT::Ptr & ground,
    FrameProfile * profile)
  {
    obstacles->clear();
    ground->clear();
    if (input->empty()) {
      return;
    }

    const std::string method = get_parameter("ground_method").as_string();
    if (method == "height") {  // 如果选择高度分割，则调用splitByHeight函数 这个太死板了，地面不一定完美额的平，矮一点的障碍物容易被忽略
      if (profile) {
        profile->ground_path = GroundPath::kDirectHeight;
        const auto stage_begin = std::chrono::steady_clock::now();
        splitByHeight(input, obstacles, ground);
        profile->height_split_ms = elapsedMs(stage_begin);
        profile->height_split_executed = true;
      } else {
        splitByHeight(input, obstacles, ground);
      }
      return;
    }

    // Need enough near-floor points or RANSAC will spam "No solution found".
    const float probe_z = static_cast<float>(
      std::max(0.2, get_parameter("height_thresh_m").as_double() * 2.0));
    size_t near_ground = 0;
    if (profile) {
      const auto stage_begin = std::chrono::steady_clock::now();
      for (const auto & p : input->points) {  // 判断有多少的点在地面附近
        if (p.z <= probe_z) {
          ++near_ground;
        }
      }
      profile->near_ground_scan_ms = elapsedMs(stage_begin);
      profile->near_ground_points = near_ground;
      profile->near_ground_scan_executed = true;
    } else {
      for (const auto & p : input->points) {  // 判断有多少的点在地面附近
        if (p.z <= probe_z) {
          ++near_ground;
        }
      }
    }
    const size_t min_ground = static_cast<size_t>(
      std::max(30, static_cast<int>(get_parameter("cluster_min_points").as_int())));
    if (near_ground < min_ground) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 3000,
        "Too few near-ground points (%zu); fallback to height split", near_ground);
      if (profile) {
        profile->ground_path = GroundPath::kTooFewNearGround;
        const auto stage_begin = std::chrono::steady_clock::now();
        splitByHeight(input, obstacles, ground);
        profile->height_split_ms = elapsedMs(stage_begin);
        profile->height_split_executed = true;
      } else {
        splitByHeight(input, obstacles, ground);
      }
      return;
    }
    /**
    SAC是Sample Consensus 采样一致性缩写
    SACSegmentation 本身是一个通用框架，需要继续告诉它：
      - 要找什么几何模型
      - 用什么算法寻找
      - 多远算内点
      - 最多尝试多少次
      - 有没有方向限制
      - 输入点云是什么
    */
    pcl::SACSegmentation<PointT> seg;  // 创建分割器
    seg.setOptimizeCoefficients(true);  // 是否优化最终平面系数  开启后地层是怎么优化的呢？
    seg.setModelType(pcl::SACMODEL_PERPENDICULAR_PLANE);  // 指定几何模型  寻找一个与指定轴线近似垂直的平面。
    seg.setMethodType(pcl::SAC_RANSAC);  // 指定 RANSAC 算法
    seg.setDistanceThreshold(get_parameter("ground_distance_thresh_m").as_double());  // 设置点到平面的距离阈值
    seg.setMaxIterations(get_parameter("ransac_max_iterations").as_int());  // RANSAC 最多尝试*组随机样本。
    Eigen::Vector3f axis(0.0f, 0.0f, 1.0f);  // 定义期望轴线
    seg.setAxis(axis);
    seg.setEpsAngle(25.0 * M_PI / 180.0);  // 设置最大角度误差  PCL接受的是弧度，所以需要将25转换一下
    seg.setInputCloud(input);

    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);  // 用于存落在平面上的点的下标
    pcl::ModelCoefficients::Ptr coeffs(new pcl::ModelCoefficients);  // 用于存平面系数ax+by+cz+d=0 的 a b c d 四个系数
    if (profile) {
      const auto stage_begin = std::chrono::steady_clock::now();
      seg.segment(*inliers, *coeffs);
      profile->ransac_ms = elapsedMs(stage_begin);
      profile->ransac_executed = true;
    } else {
      seg.segment(*inliers, *coeffs);
    }

    if (inliers->indices.empty() ||
      inliers->indices.size() < min_ground)
    {  // 如果内点数小于最小地面点数，则回退到高度分割
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 3000,
        "RANSAC ground failed (inliers=%zu); fallback to height split",
        inliers->indices.size());
        if (profile) {
          profile->ground_path = GroundPath::kRansacFailed;
          const auto stage_begin = std::chrono::steady_clock::now();
          splitByHeight(input, obstacles, ground);
          profile->height_split_ms = elapsedMs(stage_begin);
          profile->height_split_executed = true;
        } else {
          splitByHeight(input, obstacles, ground);
        }
        return;
      }
  
      pcl::ExtractIndices<PointT> extract;
      if (profile) {
        profile->ground_path = GroundPath::kRansacOk;
        const auto stage_begin = std::chrono::steady_clock::now();
        extract.setInputCloud(input);
        extract.setIndices(inliers);
        extract.setNegative(false);
        extract.filter(*ground);
        extract.setNegative(true);
        extract.filter(*obstacles);
        profile->ground_extract_ms = elapsedMs(stage_begin);
        profile->ground_extract_executed = true;
      } else {
        extract.setInputCloud(input);
        extract.setIndices(inliers);
        extract.setNegative(false);
        extract.filter(*ground);
        extract.setNegative(true);
        extract.filter(*obstacles);
      }
    }
  
  /**
  大致流程：
  障碍物点云
  建立 KD-Tree
  查找每个点附近的邻居
  通过邻接关系组成聚类
  过滤过小、过大的聚类
  根据索引复制出每个聚类点云
  */
  std::vector<CloudT::Ptr> euclideanCluster(
    const CloudT::Ptr & obstacles,
    FrameProfile * profile)
  {
    std::vector<CloudT::Ptr> out;
    if (!obstacles || obstacles->empty()) {
      return out;
    }

    std::chrono::steady_clock::time_point kdtree_begin{};
    if (profile) {
      kdtree_begin = std::chrono::steady_clock::now();
    }
    pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>);  // 创建 KD-Tree； 欧式聚类需要不断回答： 距离当前点15厘米以内有哪些点？ 如果没有空间索引，每查询一个点，都遍历整片点云。 KD-Tree 会按照空间位置组织点，使邻域查询更快。
    tree->setInputCloud(obstacles);  // 把点云交给 KD-Tree
    if (profile) {
      profile->kdtree_ms = elapsedMs(kdtree_begin);
      profile->kdtree_executed = true;
    }

    std::vector<pcl::PointIndices> cluster_indices;  // 创建聚类索引数组； 用于保存每个聚类有哪些点
    pcl::EuclideanClusterExtraction<PointT> ec;  // 创建欧式聚类器
    ec.setClusterTolerance(get_parameter("cluster_tolerance_m").as_double());  // 设置邻接距离，这也是左墙-尽头墙-右墙连接的原因
    ec.setMinClusterSize(get_parameter("cluster_min_points").as_int());  // 设置最小聚类点数
    ec.setMaxClusterSize(get_parameter("cluster_max_points").as_int());  // 设最大聚类点数（目前是25000）
    ec.setSearchMethod(tree);  // 设置搜索方法， 现在用的是KD-Tree
    ec.setInputCloud(obstacles);  // 设置输入点云
    if (profile) {
      const auto stage_begin = std::chrono::steady_clock::now();
      ec.extract(cluster_indices);  // 正式运行聚类
      profile->cluster_extract_ms = elapsedMs(stage_begin);
      profile->cluster_extract_executed = true;
      profile->cluster_candidates = cluster_indices.size();
    } else {
      ec.extract(cluster_indices);  // 正式运行聚类
    }

    const int max_clusters = get_parameter("max_clusters").as_int();
    int count = 0;
    size_t clustered_points = 0;
    std::chrono::steady_clock::time_point materialize_begin{};
    if (profile) {
      materialize_begin = std::chrono::steady_clock::now();
    }
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
      clustered_points += cluster->size();
      out.push_back(cluster);
      ++count;
    }
    if (profile) {
      profile->cluster_materialize_ms = elapsedMs(materialize_begin);
      profile->cluster_materialize_executed = true;
      profile->published_clusters = out.size();
      profile->clustered_points = clustered_points;
    }
    return out;
  }

  PublishTiming publishCloud(
    const rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr & pub,
    const CloudT::Ptr & cloud, const rclcpp::Time & stamp, const std::string & frame,
    FrameProfile * profile)
  {
    PublishTiming timing;
    sensor_msgs::msg::PointCloud2 msg;
    std::chrono::steady_clock::time_point conv_begin;
    if (profile) {
      conv_begin = std::chrono::steady_clock::now();
    }
    pcl::toROSMsg(*cloud, msg);
    msg.header.stamp = stamp;
    msg.header.frame_id = frame;
    if (profile) {
      timing.conversion_ms = elapsedMs(conv_begin);
    }
    std::chrono::steady_clock::time_point pub_begin;
    if (profile) {
      pub_begin = std::chrono::steady_clock::now();
    }
    pub->publish(msg);
    if (profile) {
      timing.publish_call_ms = elapsedMs(pub_begin);
    }
    return timing;
  }

  void publishObstacles(
    const std::vector<CloudT::Ptr> & clusters, const rclcpp::Time & stamp,
    const std::string & frame,
    FrameProfile * profile)
  {
    std::chrono::steady_clock::time_point msg_begin;
    if (profile) {
      msg_begin = std::chrono::steady_clock::now();
    }
    rgbd_perception_msgs::msg::ObstacleArray arr;
    arr.header.stamp = stamp;
    arr.header.frame_id = frame;

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
    if (profile) {
      profile->bbox_message_ms = elapsedMs(msg_begin);
    }

    std::chrono::steady_clock::time_point pub_begin;
    if (profile) {
      pub_begin = std::chrono::steady_clock::now();
    }
    obstacles_pub_->publish(arr);
    markers_pub_->publish(markers);
    if (profile) {
      profile->obstacle_publish_call_ms = elapsedMs(pub_begin);
    }
  }

  void recordProfile(const FrameProfile & profile)
  {
    ++successful_frames_;
    if (successful_frames_ <= static_cast<size_t>(profiling_warmup_frames_)) {
      return;
    }

    window_.callback_ms.push_back(profile.callback_ms);
    window_.from_ros_ms.push_back(profile.from_ros_ms);
    window_.ground_total_ms.push_back(profile.ground_total_ms);
    window_.cloud_to_ros_ms.push_back(profile.cloud_to_ros_ms);
    window_.cloud_publish_call_ms.push_back(profile.cloud_publish_call_ms);
    window_.input_points.push_back(profile.input_points);
    window_.ground_points.push_back(profile.ground_points);
    window_.obstacle_points.push_back(profile.obstacle_points);

    if (enable_clustering_) {
      window_.cluster_total_ms.push_back(profile.cluster_total_ms);
      window_.bbox_message_ms.push_back(profile.bbox_message_ms);
      window_.obstacle_publish_call_ms.push_back(profile.obstacle_publish_call_ms);
      window_.cluster_candidates.push_back(profile.cluster_candidates);
      window_.published_clusters.push_back(profile.published_clusters);
      window_.clustered_points.push_back(profile.clustered_points);
    }

    if (profile.sor_executed) {
      window_.sor_ms.push_back(profile.sor_ms);
    }
    if (profile.near_ground_scan_executed) {
      window_.near_ground_scan_ms.push_back(profile.near_ground_scan_ms);
      window_.near_ground_points.push_back(profile.near_ground_points);
    }
    if (profile.ransac_executed) {
      window_.ransac_ms.push_back(profile.ransac_ms);
    }
    if (profile.ground_extract_executed) {
      window_.ground_extract_ms.push_back(profile.ground_extract_ms);
    }
    if (profile.height_split_executed) {
      window_.height_split_ms.push_back(profile.height_split_ms);
    }
    if (profile.kdtree_executed) {
      window_.kdtree_ms.push_back(profile.kdtree_ms);
    }
    if (profile.cluster_extract_executed) {
      window_.cluster_extract_ms.push_back(profile.cluster_extract_ms);
    }
    if (profile.cluster_materialize_executed) {
      window_.cluster_materialize_ms.push_back(profile.cluster_materialize_ms);
    }

    switch (profile.ground_path) {
      case GroundPath::kDirectHeight:
        ++window_.path_direct_height;
        break;
      case GroundPath::kTooFewNearGround:
        ++window_.path_too_few;
        break;
      case GroundPath::kRansacFailed:
        ++window_.path_ransac_failed;
        break;
      case GroundPath::kRansacOk:
        ++window_.path_ransac_ok;
        break;
      case GroundPath::kUnknown:
        ++window_.failed_frames;
        break;
    }

    if (window_.callback_ms.size() >= static_cast<size_t>(profiling_window_frames_)) {
      logProfilingWindow();
      window_.clearKeepCapacity();
    }
  }

  static std::string formatMsStage(const char * name, const std::vector<double> & samples)
  {
    char buf[256];
    if (samples.empty()) {
      std::snprintf(buf, sizeof(buf), "%s_n=0", name);
      return buf;
    }
    const Percentiles p = calculatePercentiles(samples);
    std::snprintf(
      buf, sizeof(buf),
      "%s_n=%zu %s_p50=%.3f %s_p95=%.3f %s_p99=%.3f",
      name, samples.size(), name, p.p50, name, p.p95, name, p.p99);
    return buf;
  }

  static std::string formatSizeStage(const char * name, const std::vector<size_t> & samples)
  {
    char buf[256];
    if (samples.empty()) {
      std::snprintf(buf, sizeof(buf), "%s_n=0", name);
      return buf;
    }
    const SizePercentiles p = calculateSizePercentiles(samples);
    std::snprintf(
      buf, sizeof(buf),
      "%s_n=%zu %s_p50=%zu %s_p95=%zu %s_p99=%zu",
      name, samples.size(), name, p.p50, name, p.p95, name, p.p99);
    return buf;
  }

  void logProfilingWindow() const
  {
    const Percentiles callback = calculatePercentiles(window_.callback_ms);
    RCLCPP_INFO(
      get_logger(),
      "PERF PCL samples=%zu callback_p50=%.3f callback_p95=%.3f callback_p99=%.3f",
      window_.callback_ms.size(), callback.p50, callback.p95, callback.p99);
    RCLCPP_INFO(
      get_logger(),
      "PERF PCL_STAGE %s %s %s %s %s %s %s %s %s %s %s %s %s",
      formatMsStage("from_ros", window_.from_ros_ms).c_str(),
      formatMsStage("sor", window_.sor_ms).c_str(),
      formatMsStage("near_ground_scan", window_.near_ground_scan_ms).c_str(),
      formatMsStage("ransac", window_.ransac_ms).c_str(),
      formatMsStage("ground_extract", window_.ground_extract_ms).c_str(),
      formatMsStage("height_split", window_.height_split_ms).c_str(),
      formatMsStage("ground_total", window_.ground_total_ms).c_str(),
      formatMsStage("cloud_to_ros", window_.cloud_to_ros_ms).c_str(),
      formatMsStage("cloud_publish_call", window_.cloud_publish_call_ms).c_str(),
      formatMsStage("kdtree", window_.kdtree_ms).c_str(),
      formatMsStage("cluster_extract", window_.cluster_extract_ms).c_str(),
      formatMsStage("cluster_materialize", window_.cluster_materialize_ms).c_str(),
      formatMsStage("cluster_total", window_.cluster_total_ms).c_str());
    RCLCPP_INFO(
      get_logger(),
      "PERF PCL_WORK %s %s %s %s %s %s %s "
      "%s %s "
      "ground_paths ransac_ok=%zu too_few=%zu ransac_failed=%zu direct_height=%zu "
      "empty_input=%zu failed=%zu",
      formatSizeStage("input_pts", window_.input_points).c_str(),
      formatSizeStage("near_ground_pts", window_.near_ground_points).c_str(),
      formatSizeStage("ground_pts", window_.ground_points).c_str(),
      formatSizeStage("obstacle_pts", window_.obstacle_points).c_str(),
      formatSizeStage("cluster_candidates", window_.cluster_candidates).c_str(),
      formatSizeStage("published_clusters", window_.published_clusters).c_str(),
      formatSizeStage("clustered_pts", window_.clustered_points).c_str(),
      formatMsStage("bbox_message", window_.bbox_message_ms).c_str(),
      formatMsStage("obstacle_publish_call", window_.obstacle_publish_call_ms).c_str(),
      window_.path_ransac_ok,
      window_.path_too_few,
      window_.path_ransac_failed,
      window_.path_direct_height,
      window_.empty_input_frames,
      window_.failed_frames);
  }

  std::string input_topic_;
  std::string target_frame_;
  bool enable_clustering_{false};
  bool ransac_enable_openmp_{false};
  int ransac_openmp_threads_{4};
  bool ransac_enable_neon_{false};
  bool profiling_enable_{true};
  int profiling_warmup_frames_{30};
  int profiling_window_frames_{1000};
  size_t successful_frames_{0};
  ProfilingWindow window_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_sub_;
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
