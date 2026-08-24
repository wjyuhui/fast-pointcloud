cmake_minimum_required(VERSION 3.8)
project(rgbd_detection)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_CXX_STANDARD 17)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(message_filters REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(rgbd_perception_msgs REQUIRED)
find_package(OpenCV REQUIRED)

set(_rknn_hints)
if(RKNN_ROOT)
  list(APPEND _rknn_hints "${RKNN_ROOT}/lib")
endif()
if(DEFINED ENV{RKNN_ROOT})
  list(APPEND _rknn_hints "$ENV{RKNN_ROOT}/lib")
endif()
list(APPEND _rknn_hints
  "${CMAKE_CURRENT_SOURCE_DIR}/../../../.deps/rknpu2/lib"
  "${CMAKE_SOURCE_DIR}/../../.deps/rknpu2/lib"
  /usr/lib
  /usr/lib64
  /usr/lib/aarch64-linux-gnu
)

find_library(RKNN_RT_LIB NAMES rknnrt HINTS ${_rknn_hints})
if(NOT RKNN_RT_LIB)
  message(FATAL_ERROR
    "librknnrt.so not found. Install the RKNN runtime (rknpu2) or place "
    "librknnrt.so under .deps/rknpu2/lib and set RKNN_ROOT.")
endif()
get_filename_component(RKNN_RT_DIR "${RKNN_RT_LIB}" DIRECTORY)
message(STATUS "Using RKNN runtime: ${RKNN_RT_LIB}")

add_executable(yolo_3d_node src/yolo_3d_node.cpp)
target_include_directories(yolo_3d_node PUBLIC
  ${OpenCV_INCLUDE_DIRS}
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party/rknn/include
)
ament_target_dependencies(yolo_3d_node
  rclcpp sensor_msgs visualization_msgs geometry_msgs
  cv_bridge message_filters
  tf2 tf2_ros tf2_geometry_msgs rgbd_perception_msgs
)
target_link_libraries(yolo_3d_node
  ${OpenCV_LIBS}
  ${RKNN_RT_LIB}
)
set_target_properties(yolo_3d_node PROPERTIES
  BUILD_RPATH "${RKNN_RT_DIR}"
  INSTALL_RPATH "${RKNN_RT_DIR}"
)

install(TARGETS yolo_3d_node DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY models DESTINATION share/${PROJECT_NAME})

ament_package()
