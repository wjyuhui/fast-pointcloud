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

# ONNX Runtime (local unpack under workspace .deps preferred)
if(NOT ONNXRUNTIME_ROOT)
  if(DEFINED ENV{ONNXRUNTIME_ROOT})
    set(ONNXRUNTIME_ROOT $ENV{ONNXRUNTIME_ROOT})
  else()
    file(GLOB _ort_candidates
      "${CMAKE_SOURCE_DIR}/../../.deps/onnxruntime-linux-x64-*"
      "${CMAKE_CURRENT_SOURCE_DIR}/../../../.deps/onnxruntime-linux-x64-*")
    if(_ort_candidates)
      list(GET _ort_candidates 0 ONNXRUNTIME_ROOT)
    endif()
  endif()
endif()

if(NOT ONNXRUNTIME_ROOT OR NOT EXISTS "${ONNXRUNTIME_ROOT}/include/onnxruntime_cxx_api.h")
  message(FATAL_ERROR
    "ONNX Runtime not found. Set ONNXRUNTIME_ROOT or unpack "
    "onnxruntime-linux-x64-* under ros2_ws/.deps/")
endif()

message(STATUS "Using ONNX Runtime at ${ONNXRUNTIME_ROOT}")

add_executable(yolo_3d_node src/yolo_3d_node.cpp)
target_include_directories(yolo_3d_node PUBLIC
  ${OpenCV_INCLUDE_DIRS}
  ${ONNXRUNTIME_ROOT}/include
)
ament_target_dependencies(yolo_3d_node
  rclcpp sensor_msgs visualization_msgs geometry_msgs
  cv_bridge message_filters
  tf2 tf2_ros tf2_geometry_msgs rgbd_perception_msgs
)
target_link_libraries(yolo_3d_node
  ${OpenCV_LIBS}
  ${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so
)
set_target_properties(yolo_3d_node PROPERTIES
  BUILD_RPATH "${ONNXRUNTIME_ROOT}/lib"
  INSTALL_RPATH "${ONNXRUNTIME_ROOT}/lib"
)

install(TARGETS yolo_3d_node DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY models DESTINATION share/${PROJECT_NAME})

ament_package()
