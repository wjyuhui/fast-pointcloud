cmake_minimum_required(VERSION 3.8)
project(rgbd_bev)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_CXX_STANDARD 17)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)

add_executable(bev_node src/bev_node.cpp)
ament_target_dependencies(bev_node rclcpp sensor_msgs nav_msgs visualization_msgs)

install(TARGETS bev_node DESTINATION lib/${PROJECT_NAME})
ament_package()
