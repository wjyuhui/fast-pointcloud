方案设计（已落地 C++ / ROS2 Humble）：

节点划分：
orbbec_camera（官方）
    │  color + D2C depth + camera_info + 相机内 TF
    │
    │  + bringup: static TF  base_link -> camera_link
    ├──────────────────────┐
    ▼                      ▼
┌─────────────────┐   ┌──────────────────┐
│ yolo_3d_node    │   │ pcl_obstacle_node│
│ 2D检测 + ROI抬升 │   │ 裁剪/体素/SOR/   │
│ → Detection3D   │   │ RANSAC/聚类/AABB │
└────────┬────────┘   └────────┬─────────┘
         │ 语义旁路             │ 几何主链路
         ▼                     ▼
   detections_3d          cloud_obstacles
   markers/detections          │
                               ▼
                         ┌──────────────────┐
                         │ bev_node         │
                         │ OccupancyGrid +  │
                         │ Marker           │
                         └──────────────────┘

包结构（src/）：
  rgbd_perception_msgs  自定义消息
  rgbd_pcl              pcl_obstacle_node
  rgbd_bev              bev_node
  rgbd_detection        yolo_3d_node (ONNX Runtime + OpenCV 预处理)
  rgbd_bringup          launch / config
  OrbbecSDK_ROS2        官方相机驱动

数据流动：

orbbec_camera:
  发布:
    /camera/color/image_raw
    /camera/color/camera_info
    /camera/depth/image_raw          # depth_registration:=true 后对齐到 color
    /camera/depth_registered/points

bringup:
  发布 TF: base_link -> camera_link （安装高度默认 0.8m）

yolo_3d_node：
  订阅: color + depth + camera_info + TF
  发布:
    /perception/detections_3d
    /perception/markers/detections
  （语义旁路：跟人/停靠/测距；不进 BEV）

pcl_obstacle_node：
  订阅: /camera/depth_registered/points
  发布:
    /perception/obstacles
    /perception/cloud_obstacles
    /perception/cloud_ground
    /perception/markers/obstacles

bev_node：
  订阅: /perception/cloud_obstacles
  发布:
    /perception/bev/occupancy_grid
    /perception/markers/bev

────────────────────────────────────────
编译 / 运行

# 依赖：系统 apt 装 libpcl-dev / g++ 最省事；当前也可用工作空间本地解压依赖：
source /opt/ros/humble/setup.bash
source /home/yuhui/ros2_ws/setup_local_deps.bash   # PCL/Boost/g++/ONNXRuntime

cd /home/yuhui/ros2_ws
colcon build --cmake-args -DCMAKE_BUILD_TYPE=Release -DONNXRUNTIME_ROOT=$ONNXRUNTIME_ROOT
source install/setup.bash

# YOLO ONNX（已放 models/yolov8n.onnx；需重导出时）：
#   bash scripts/export_yolo_onnx.sh

# 全栈（含 Orbbec）：
ros2 launch rgbd_bringup perception.launch.py

# 仅算法节点（相机已在别处启动）：
ros2 launch rgbd_bringup perception.launch.py use_orbbec:=false

# 先跑几何链路、暂不跑 YOLO：
ros2 launch rgbd_bringup perception.launch.py enable_yolo:=false
