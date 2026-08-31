方案设计（已落地 C++ / ROS2 Humble）：

节点划分：
orbbec_camera（官方）
    │  color + D2C depth + camera_info + 相机内 TF
    │
    │  + bringup: static TF  base_link -> camera_link
    ├──────────────────────┐
    ▼                      ▼
┌─────────────────┐   ┌──────────────────────┐
│ yolo_3d_node    │   │ cloud_workspace_node │
│ 2D检测 + ROI抬升 │   │ 稀疏反投影/TF/       │
│ → Detection3D   │   │ 裁剪/体素            │
└────────┬────────┘   └──────────┬───────────┘
         │ 语义旁路              │ cloud_workspace
         ▼                       ▼
   detections_3d          ┌──────────────────┐
   markers/detections     │ pcl_obstacle_node│
                          │ SOR/RANSAC/聚类/ │
                          │ AABB             │
                          └────────┬─────────┘
                                   │ cloud_obstacles
                                   ▼
                             ┌──────────────────┐
                             │ bev_node         │
                             │ OccupancyGrid +  │
                             │ Marker           │
                             └──────────────────┘

包结构（src/）：
  rgbd_perception_msgs  自定义消息
  rgbd_pcl              cloud_workspace_node + pcl_obstacle_node
  rgbd_bev              bev_node
  rgbd_detection        yolo_3d_node (RKNN YOLOv8 person + OpenCV 预处理)
  rgbd_bringup          launch / config
  OrbbecSDK_ROS2        官方相机驱动

数据流动：

orbbec_camera:
  发布:
    /camera/color/image_raw
    /camera/color/camera_info
    /camera/depth/image_raw          # depth_registration:=true 后对齐到 color
  （launch 关闭 enable_point_cloud / enable_colored_point_cloud，不出稠密点云）

bringup:
  发布 TF: base_link -> camera_link （安装高度默认 0.8m）

yolo_3d_node：
  订阅: color + depth + camera_info + TF
  发布:
    /perception/detections_3d
    /perception/markers/detections
  （语义旁路：跟人/停靠/测距；不进 BEV）

cloud_workspace_node：
  订阅: /camera/depth/image_raw + /camera/color/camera_info + TF
  （stride 稀疏反投影 → TF → PassThrough x/z → VoxelGrid）
  发布:
    /perception/cloud_workspace          # base_link，含地面

pcl_obstacle_node：
  订阅: /perception/cloud_workspace
  （可选 SOR → 地面分割 → 欧式聚类 → AABB）
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
source /home/yuhui/ros2_ws/setup_local_deps.bash   # RKNN runtime (librknnrt)

cd /home/yuhui/ros2_ws
colcon build --cmake-args -DCMAKE_BUILD_TYPE=Release
source install/setup.bash

# YOLO RKNN：models/yolov8n_rk3588_fp16.rknn（仅检测 person）

# 全栈（含 Orbbec）：
ros2 launch rgbd_bringup perception.launch.py

# 仅算法节点（相机已在别处启动）：
ros2 launch rgbd_bringup perception.launch.py use_orbbec:=false

# 先跑几何链路、暂不跑 YOLO：
ros2 launch rgbd_bringup perception.launch.py enable_yolo:=false


────────────────────────────────────────
启动实时程序：
cd /home/yuhui/ros2_ws
source /home/yuhui/ros2_ws/setup_local_deps.bash
source install/setup.bash
ros2 launch rgbd_bringup perception.launch.py

启动包非实时程序：
cd /home/yuhui/ros2_ws
source /home/yuhui/ros2_ws/setup_local_deps.bash
source install/setup.bash
ros2 launch rgbd_bringup perception.launch.py use_orbbec:=false

────────────────────────────────────────
录制算法需要的包：
cd ~/ros2_ws/bags
source /opt/ros/humble/setup.bash
ros2 bag record -o cam_$(date +%Y%m%d_%H%M%S) \
  /camera/color/image_raw \
  /camera/color/camera_info \
  /camera/depth/image_raw \
  /camera/depth/camera_info \
  /tf_static

启动包： ros2 bag play ./ --start-offset 0.5 -l

相机节点分析：
最原始的信息：
  /camera/color/image_raw 
  /camera/color/camera_info 
  /camera/depth/image_raw 
  /camera/depth/camera_info 
  /tf_static（相机内参坐标系）
第二层：
  /camera/depth/points 
  /camera/depth_registered/points 

--------------------------------------------------
2026.8.30 启动命令
ros2 launch rgbd_bringup perception.launch.py \
  use_orbbec:=false \
  enable_yolo:=false \
  projection_enable_openmp:=true \
  projection_openmp_threads:=4 \
  projection_enable_neon:=false \
  projection_omp_places:='{4},{5},{6},{7}'

ros2 launch rgbd_bringup perception.launch.py \
  use_orbbec:=false \
  enable_yolo:=false \
  projection_enable_openmp:=false \
  projection_enable_neon:=false \





  