cmake_minimum_required(VERSION 3.8)
project(rgbd_pcl)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_CXX_STANDARD 17)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_eigen REQUIRED)
find_package(tf2_sensor_msgs REQUIRED)
find_package(rgbd_perception_msgs REQUIRED)

# Boost must be found before PCL when using a non-system prefix.
if(DEFINED ENV{BOOST_ROOT})
  set(BOOST_ROOT $ENV{BOOST_ROOT})
  set(Boost_NO_SYSTEM_PATHS ON)
endif()
find_package(Boost REQUIRED COMPONENTS system filesystem date_time iostreams serialization)
find_package(PCL REQUIRED COMPONENTS common filters segmentation sample_consensus search kdtree)
find_package(pcl_conversions REQUIRED)
find_package(OpenMP REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable(cloud_workspace_node src/cloud_workspace_node.cpp)
target_include_directories(cloud_workspace_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${PCL_INCLUDE_DIRS}
)
ament_target_dependencies(cloud_workspace_node
  rclcpp sensor_msgs geometry_msgs
  tf2 tf2_ros tf2_eigen pcl_conversions
)
target_link_libraries(cloud_workspace_node ${PCL_LIBRARIES} OpenMP::OpenMP_CXX)

add_executable(pcl_obstacle_node src/pcl_obstacle_node.cpp)
target_include_directories(pcl_obstacle_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${PCL_INCLUDE_DIRS}
)
ament_target_dependencies(pcl_obstacle_node
  rclcpp sensor_msgs visualization_msgs geometry_msgs
  pcl_conversions rgbd_perception_msgs
)
target_link_libraries(pcl_obstacle_node ${PCL_LIBRARIES})

install(TARGETS cloud_workspace_node pcl_obstacle_node DESTINATION lib/${PROJECT_NAME})

ament_package()
