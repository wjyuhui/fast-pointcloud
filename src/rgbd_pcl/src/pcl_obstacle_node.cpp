#include <algorithm>
#include <chrono>
#include <cmath>
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

    input_topic_ = get_parameter("input_cloud_topic").as_string();
    target_frame_ = get_parameter("target_frame").as_string();

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
      "pcl_obstacle_node cloud=%s frame=%s (ground/cluster)",
      input_topic_.c_str(), target_frame_.c_str());
  }

private:

  // 回调函数，也是主要的工作函数，收到点云数据后，进行处理， 每一个步骤几乎封装成另一个函数在此回调函数中进行调用
  void cloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
  {
    const auto t0 = std::chrono::steady_clock::now();

    CloudT::Ptr cloud(new CloudT);
    pcl::fromROSMsg(*msg, *cloud);  // 从Msg中取出点云数据，存入cloud中
    if (cloud->empty()) {
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
      sor.setInputCloud(cloud);
      sor.setMeanK(get_parameter("sor_mean_k").as_int());
      sor.setStddevMulThresh(get_parameter("sor_stddev").as_double());
      sor.filter(*cloud_sor);
      cloud.swap(cloud_sor);
    }

    CloudT::Ptr obstacles(new CloudT);
    CloudT::Ptr ground(new CloudT);
    segmentGround(cloud, obstacles, ground);  // 地面分割， 分割出地面和障碍物

    // 发布地面和障碍物点云
    publishCloud(cloud_obstacles_pub_, obstacles, msg->header.stamp, frame);
    publishCloud(cloud_ground_pub_, ground, msg->header.stamp, frame);

    auto clusters = euclideanCluster(obstacles);  // 欧式聚类， 将障碍物点云分割成多个聚类
    publishObstacles(clusters, msg->header.stamp, frame);  // 发布障碍物聚类

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

  void segmentGround(const CloudT::Ptr & input, CloudT::Ptr & obstacles, CloudT::Ptr & ground)
  {
    obstacles->clear();
    ground->clear();
    if (input->empty()) {
      return;
    }

    const std::string method = get_parameter("ground_method").as_string();
    if (method == "height") {  // 如果选择高度分割，则调用splitByHeight函数 这个太死板了，地面不一定完美额的平，矮一点的障碍物容易被忽略
      splitByHeight(input, obstacles, ground);
      return;
    }

    // Need enough near-floor points or RANSAC will spam "No solution found".
    const float probe_z = static_cast<float>(
      std::max(0.2, get_parameter("height_thresh_m").as_double() * 2.0));
    size_t near_ground = 0;
    for (const auto & p : input->points) {  // 判断有多少的点在地面附近
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
    seg.segment(*inliers, *coeffs);

    if (inliers->indices.empty() ||
      inliers->indices.size() < min_ground)
    {  // 如果内点数小于最小地面点数，则回退到高度分割
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
  
  /**
  大致流程：
  障碍物点云
  建立 KD-Tree
  查找每个点附近的邻居
  通过邻接关系组成聚类
  过滤过小、过大的聚类
  根据索引复制出每个聚类点云
  */
  std::vector<CloudT::Ptr> euclideanCluster(const CloudT::Ptr & obstacles)
  {
    std::vector<CloudT::Ptr> out;
    if (!obstacles || obstacles->empty()) {
      return out;
    }

    pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>);  // 创建 KD-Tree； 欧式聚类需要不断回答： 距离当前点15厘米以内有哪些点？ 如果没有空间索引，每查询一个点，都遍历整片点云。 KD-Tree 会按照空间位置组织点，使邻域查询更快。
    tree->setInputCloud(obstacles);  // 把点云交给 KD-Tree

    std::vector<pcl::PointIndices> cluster_indices;  // 创建聚类索引数组； 用于保存每个聚类有哪些点
    pcl::EuclideanClusterExtraction<PointT> ec;  // 创建欧式聚类器
    ec.setClusterTolerance(get_parameter("cluster_tolerance_m").as_double());  // 设置邻接距离，这也是左墙-尽头墙-右墙连接的原因
    ec.setMinClusterSize(get_parameter("cluster_min_points").as_int());  // 设置最小聚类点数
    ec.setMaxClusterSize(get_parameter("cluster_max_points").as_int());  // 设最大聚类点数（目前是25000）
    ec.setSearchMethod(tree);  // 设置搜索方法， 现在用的是KD-Tree
    ec.setInputCloud(obstacles);  // 设置输入点云
    ec.extract(cluster_indices);  // 正式运行聚类

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

  void publishObstacles(
    const std::vector<CloudT::Ptr> & clusters, const rclcpp::Time & stamp,
    const std::string & frame)
  {
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

    obstacles_pub_->publish(arr);
    markers_pub_->publish(markers);
  }

  std::string input_topic_;
  std::string target_frame_;
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
