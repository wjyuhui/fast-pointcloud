#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include <nav_msgs/msg/occupancy_grid.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

class BevNode : public rclcpp::Node
{
public:
  BevNode()
  : Node("bev_node")
  {
    declare_parameter<std::string>("input_cloud_topic", "/perception/cloud_obstacles");
    declare_parameter<std::string>("frame_id", "base_link");
    declare_parameter<double>("x_min_m", 0.0);
    declare_parameter<double>("x_max_m", 5.0);
    declare_parameter<double>("y_min_m", -2.5);
    declare_parameter<double>("y_max_m", 2.5);
    declare_parameter<double>("resolution_m", 0.05);
    declare_parameter<double>("z_min_m", 0.05);
    declare_parameter<double>("z_max_m", 1.8);
    declare_parameter<int>("occupied_count", 2);
    declare_parameter<bool>("publish_markers", true);

    frame_id_ = get_parameter("frame_id").as_string();  // OccupancyGrid 的坐标系
    x_min_ = get_parameter("x_min_m").as_double();    
    x_max_ = get_parameter("x_max_m").as_double();  // x_min_m / x_max_m 前方距离 0～5m
    y_min_ = get_parameter("y_min_m").as_double();
    y_max_ = get_parameter("y_max_m").as_double();  //  同x 
    res_ = get_parameter("resolution_m").as_double();  // 地图分辨率 一格边长 5cm
    z_min_ = get_parameter("z_min_m").as_double();  // 
    z_max_ = get_parameter("z_max_m").as_double();  // 同x
    occ_count_ = get_parameter("occupied_count").as_int();  // 障碍物阈值 超过这个数就标占用

    rows_ = static_cast<int>(std::ceil((x_max_ - x_min_) / res_));  // 计算行数和列数 格子数
    cols_ = static_cast<int>(std::ceil((y_max_ - y_min_) / res_));

    sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
      get_parameter("input_cloud_topic").as_string(), rclcpp::SensorDataQoS(),
      std::bind(&BevNode::cloudCallback, this, std::placeholders::_1));

    grid_pub_ = create_publisher<nav_msgs::msg::OccupancyGrid>(
      "/perception/bev/occupancy_grid", 10);
    markers_pub_ = create_publisher<visualization_msgs::msg::MarkerArray>(
      "/perception/markers/bev", 10);

    RCLCPP_INFO(get_logger(), "bev_node grid %dx%d res=%.3f", rows_, cols_, res_);
  }

private:

  /** 
  把一帧障碍点云压成一张鸟瞰占用图：3D 点 → 数每个格子里有几个点 → 超过阈值就标占用 → 发出去。
  大致流程：
    PointCloud2
    → 遍历每个点的 x/y/z
    → 丢掉无效 / 盒子外的点
    → 投到格子 (ix, iy)，counts++
    → 按 OccupancyGrid 规则写入 data（0 或 100）
    → 发布；可选再画 Marker
  */ 
  void cloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
  {
    std::vector<int> counts(static_cast<size_t>(rows_ * cols_), 0);  // 创建一个大小为 rows_ * cols_ 的数组，用于存储每个格子里的点数

    sensor_msgs::PointCloud2ConstIterator<float> iter_x(*msg, "x");
    sensor_msgs::PointCloud2ConstIterator<float> iter_y(*msg, "y");
    sensor_msgs::PointCloud2ConstIterator<float> iter_z(*msg, "z");  // 这是用来直接遍历ROS PointCloud2的z值的   x， y 同理
    for (; iter_x != iter_x.end(); ++iter_x, ++iter_y, ++iter_z) {
      const float x = *iter_x;
      const float y = *iter_y;
      const float z = *iter_z;
      if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z)) {   // isfinite 判断一个浮点数是不是普通有限值
        continue;
      }
      if (x < x_min_ || x >= x_max_ || y < y_min_ || y >= y_max_ ||
        z < z_min_ || z > z_max_)  // 判断一个浮点数是不是在指定范围内
      {
        continue;
      }
      const int ix = static_cast<int>((x - x_min_) / res_);  // 计算一个点在地图上的哪个格子
      const int iy = static_cast<int>((y - y_min_) / res_);
      if (ix < 0 || ix >= rows_ || iy < 0 || iy >= cols_) {  // 格子外的不要了
        continue;
      }
      // OccupancyGrid: row-major, index = iy + ix * cols? 
      // nav_msgs: data[i] corresponds to map cell at
      //   x = origin.x + (i % width) * res
      //   y = origin.y + (i / width) * res
      // We set origin at (x_min, y_min), width=cols (y), height=rows (x)
      // so cell (ix along x / height, iy along y / width): index = iy + ix * width
      counts[static_cast<size_t>(iy + ix * cols_)] += 1;
    }

    nav_msgs::msg::OccupancyGrid grid;
    grid.header.stamp = msg->header.stamp;
    grid.header.frame_id = frame_id_;
    grid.info.resolution = static_cast<float>(res_);
    grid.info.width = static_cast<uint32_t>(cols_);
    grid.info.height = static_cast<uint32_t>(rows_);
    // OccupancyGrid x axis = map width direction; we map width->y, height->x
    // Rotate convention: treat map x as robot y, map y as robot x by setting origin
    // and indexing carefully.
    // Simpler: publish with origin at (x_min, y_min), and interpret:
    //   map x increases with robot +y (left)
    //   map y increases with robot +x (forward)
    // Then RViz Map display in base_link looks rotated 90 deg.
    // Prefer: width along +x (forward), height along +y (left) by swapping.
    // Rebuild with width=rows (x), height=cols (y):

    // Rebuild correctly: width along x (forward), height along y (left)
    grid.info.width = static_cast<uint32_t>(rows_);
    grid.info.height = static_cast<uint32_t>(cols_);
    grid.info.origin.position.x = x_min_;
    grid.info.origin.position.y = y_min_;
    grid.info.origin.position.z = 0.0;
    grid.info.origin.orientation.w = 1.0;
    grid.data.assign(static_cast<size_t>(rows_ * cols_), 0);  // free in ROI

    for (int ix = 0; ix < rows_; ++ix) {
      for (int iy = 0; iy < cols_; ++iy) {
        const int c = counts[static_cast<size_t>(iy + ix * cols_)];
        // index: i = mx + my * width, mx along x, my along y
        const size_t idx = static_cast<size_t>(ix + iy * rows_);
        if (c >= occ_count_) {
          grid.data[idx] = 100;  // occupied
        } else {
          grid.data[idx] = 0;    // free
        }
      }
    }

    grid_pub_->publish(grid);

    if (get_parameter("publish_markers").as_bool()) {
      publishOccupiedMarkers(grid);
    }
  }

  void publishOccupiedMarkers(const nav_msgs::msg::OccupancyGrid & grid)
  {
    visualization_msgs::msg::MarkerArray arr;
    visualization_msgs::msg::Marker del;
    del.action = visualization_msgs::msg::Marker::DELETEALL;
    arr.markers.push_back(del);

    visualization_msgs::msg::Marker cubes;
    cubes.header = grid.header;
    cubes.ns = "bev_occupied";
    cubes.id = 0;
    cubes.type = visualization_msgs::msg::Marker::CUBE_LIST;
    cubes.action = visualization_msgs::msg::Marker::ADD;
    cubes.scale.x = res_ * 0.9;
    cubes.scale.y = res_ * 0.9;
    cubes.scale.z = 0.05;
    cubes.color.r = 0.9f;
    cubes.color.g = 0.2f;
    cubes.color.b = 0.1f;
    cubes.color.a = 0.6f;
    cubes.pose.orientation.w = 1.0;
    cubes.lifetime = rclcpp::Duration::from_seconds(0.3);

    const int width = static_cast<int>(grid.info.width);
    const int height = static_cast<int>(grid.info.height);
    for (int my = 0; my < height; ++my) {
      for (int mx = 0; mx < width; ++mx) {
        const size_t idx = static_cast<size_t>(mx + my * width);
        if (grid.data[idx] < 50) {
          continue;
        }
        geometry_msgs::msg::Point p;
        p.x = grid.info.origin.position.x + (mx + 0.5) * res_;
        p.y = grid.info.origin.position.y + (my + 0.5) * res_;
        p.z = 0.05;
        cubes.points.push_back(p);
      }
    }
    arr.markers.push_back(cubes);
    markers_pub_->publish(arr);
  }

  std::string frame_id_;
  double x_min_{0}, x_max_{5}, y_min_{-2.5}, y_max_{2.5}, res_{0.05};
  double z_min_{0.05}, z_max_{1.8};
  int occ_count_{2};
  int rows_{0}, cols_{0};

  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_;
  rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr grid_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr markers_pub_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<BevNode>());
  rclcpp::shutdown();
  return 0;
}
