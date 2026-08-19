#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

#include <cv_bridge/cv_bridge.h>
#include <geometry_msgs/msg/point_stamped.hpp>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
#include <onnxruntime_cxx_api.h>
#include <opencv2/imgproc.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rgbd_perception_msgs/msg/detection3_d.hpp>
#include <rgbd_perception_msgs/msg/detection3_d_array.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

namespace
{

const char * kCocoNames[80] = {
  "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
  "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
  "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
  "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
  "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
  "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
  "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
  "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
  "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
  "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
};

struct Det2D
{
  int class_id{0};
  float score{0.f};
  float x1{0}, y1{0}, x2{0}, y2{0};
};

float iou(const Det2D & a, const Det2D & b)
{
  const float xx1 = std::max(a.x1, b.x1);
  const float yy1 = std::max(a.y1, b.y1);
  const float xx2 = std::min(a.x2, b.x2);
  const float yy2 = std::min(a.y2, b.y2);
  const float w = std::max(0.f, xx2 - xx1);
  const float h = std::max(0.f, yy2 - yy1);
  const float inter = w * h;
  const float area_a = std::max(0.f, a.x2 - a.x1) * std::max(0.f, a.y2 - a.y1);
  const float area_b = std::max(0.f, b.x2 - b.x1) * std::max(0.f, b.y2 - b.y1);
  return inter / (area_a + area_b - inter + 1e-6f);
}

std::vector<Det2D> nms(std::vector<Det2D> dets, float iou_thresh)
{
  std::sort(dets.begin(), dets.end(),
    [](const Det2D & a, const Det2D & b) { return a.score > b.score; });
  std::vector<Det2D> keep;
  std::vector<char> removed(dets.size(), 0);
  for (size_t i = 0; i < dets.size(); ++i) {
    if (removed[i]) {
      continue;
    }
    keep.push_back(dets[i]);
    for (size_t j = i + 1; j < dets.size(); ++j) {
      if (!removed[j] && dets[i].class_id == dets[j].class_id &&
        iou(dets[i], dets[j]) > iou_thresh)
      {
        removed[j] = 1;
      }
    }
  }
  return keep;
}

}  // namespace

class Yolo3dNode : public rclcpp::Node
{
public:
  Yolo3dNode()
  : Node("yolo_3d_node"),
    tf_buffer_(this->get_clock()),
    tf_listener_(tf_buffer_),
    env_(ORT_LOGGING_LEVEL_WARNING, "yolo_3d"),
    memory_info_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault))
  {
    declare_parameter<std::string>("model_path", "");
    declare_parameter<double>("conf_thresh", 0.35);
    declare_parameter<double>("iou_thresh", 0.45);
    declare_parameter<int>("imgsz", 640);
    declare_parameter<std::vector<int64_t>>("class_whitelist", {0, 56, 57, 59, 60, 62, 63});
    declare_parameter<double>("depth_percentile", 50.0);
    declare_parameter<double>("trim_low", 0.1);
    declare_parameter<double>("trim_high", 0.9);
    declare_parameter<int>("min_valid_pixels", 30);
    declare_parameter<double>("center_crop_ratio", 0.6);
    declare_parameter<double>("min_depth_m", 0.2);
    declare_parameter<double>("max_depth_m", 5.0);
    declare_parameter<std::string>("target_frame", "base_link");
    declare_parameter<std::string>("color_topic", "/camera/color/image_raw");
    declare_parameter<std::string>("depth_topic", "/camera/depth/image_raw");
    declare_parameter<std::string>("camera_info_topic", "/camera/color/camera_info");
    declare_parameter<bool>("enable_detection", true);

    target_frame_ = get_parameter("target_frame").as_string();
    conf_thresh_ = static_cast<float>(get_parameter("conf_thresh").as_double());
    iou_thresh_ = static_cast<float>(get_parameter("iou_thresh").as_double());
    imgsz_ = get_parameter("imgsz").as_int();

    for (auto id : get_parameter("class_whitelist").as_integer_array()) {
      whitelist_.insert(static_cast<int>(id));
    }

    const std::string model_path = get_parameter("model_path").as_string();
    enable_detection_ = get_parameter("enable_detection").as_bool();
    if (enable_detection_) {
      if (model_path.empty() || !loadModel(model_path)) {
        RCLCPP_WARN(get_logger(),
          "YOLO model not loaded (path='%s'). Node will skip detection until model is available.",
          model_path.c_str());
        enable_detection_ = false;
      } else {
        RCLCPP_INFO(get_logger(), "Loaded YOLO ONNX via ORT: %s", model_path.c_str());
      }
    }

    rclcpp::SensorDataQoS qos;
    color_sub_.subscribe(this, get_parameter("color_topic").as_string(), qos.get_rmw_qos_profile());
    depth_sub_.subscribe(this, get_parameter("depth_topic").as_string(), qos.get_rmw_qos_profile());

    sync_ = std::make_shared<message_filters::Synchronizer<SyncPolicy>>(
      SyncPolicy(10), color_sub_, depth_sub_);
    sync_->registerCallback(
      std::bind(&Yolo3dNode::syncCallback, this, std::placeholders::_1, std::placeholders::_2));

    info_sub_ = create_subscription<sensor_msgs::msg::CameraInfo>(
      get_parameter("camera_info_topic").as_string(), qos,
      [this](const sensor_msgs::msg::CameraInfo::SharedPtr msg) {
        camera_info_ = *msg;
        have_info_ = true;
      });

    det_pub_ = create_publisher<rgbd_perception_msgs::msg::Detection3DArray>(
      "/perception/detections_3d", 10);
    marker_pub_ = create_publisher<visualization_msgs::msg::MarkerArray>(
      "/perception/markers/detections", 10);
  }

private:
  using SyncPolicy = message_filters::sync_policies::ApproximateTime<
    sensor_msgs::msg::Image, sensor_msgs::msg::Image>;

  bool loadModel(const std::string & path)
  {
    try {
      Ort::SessionOptions opts;
      opts.SetIntraOpNumThreads(2);
      opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
      session_ = std::make_unique<Ort::Session>(env_, path.c_str(), opts);

      Ort::AllocatorWithDefaultOptions allocator;
      {
        auto in = session_->GetInputNameAllocated(0, allocator);
        auto out = session_->GetOutputNameAllocated(0, allocator);
        input_name_ = in.get();
        output_name_ = out.get();
      }
      auto in_info = session_->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
      // expect [1,3,H,W]
      if (in_info.size() == 4 && in_info[2] > 0) {
        imgsz_ = static_cast<int>(in_info[2]);
      }
      return true;
    } catch (const Ort::Exception & e) {
      RCLCPP_ERROR(get_logger(), "ORT load failed: %s", e.what());
      return false;
    }
  }

  void syncCallback(
    const sensor_msgs::msg::Image::ConstSharedPtr & color_msg,
    const sensor_msgs::msg::Image::ConstSharedPtr & depth_msg)
  {
    if (!have_info_) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "Waiting for camera_info...");
      return;
    }
    if (!enable_detection_ || !session_) {
      return;
    }

    cv_bridge::CvImageConstPtr color_cv;
    try {
      color_cv = cv_bridge::toCvShare(color_msg, "bgr8");
    } catch (const cv_bridge::Exception & e) {
      RCLCPP_ERROR(get_logger(), "cv_bridge color: %s", e.what());
      return;
    }

    cv::Mat depth_m;
    if (!depthToMeters(depth_msg, depth_m)) {
      return;
    }

    auto dets = inferYolo(color_cv->image);
    publishDetections3d(dets, depth_m, color_msg);
  }

  bool depthToMeters(const sensor_msgs::msg::Image::ConstSharedPtr & depth_msg, cv::Mat & depth_m)
  {
    try {
      if (depth_msg->encoding == "16UC1" || depth_msg->encoding == "mono16") {
        auto d = cv_bridge::toCvShare(depth_msg);
        d->image.convertTo(depth_m, CV_32FC1, 0.001);
        return true;
      }
      if (depth_msg->encoding == "32FC1") {
        auto d = cv_bridge::toCvShare(depth_msg);
        depth_m = d->image;
        return true;
      }
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
        "Unsupported depth encoding: %s", depth_msg->encoding.c_str());
      return false;
    } catch (const cv_bridge::Exception & e) {
      RCLCPP_ERROR(get_logger(), "cv_bridge depth: %s", e.what());
      return false;
    }
  }

  std::vector<Det2D> inferYolo(const cv::Mat & bgr)
  {
    std::vector<Det2D> out;
    const int img_w = bgr.cols;
    const int img_h = bgr.rows;

    const float scale = std::min(
      static_cast<float>(imgsz_) / img_w, static_cast<float>(imgsz_) / img_h);
    const int new_w = static_cast<int>(std::round(img_w * scale));
    const int new_h = static_cast<int>(std::round(img_h * scale));
    const int pad_left = (imgsz_ - new_w) / 2;
    const int pad_top = (imgsz_ - new_h) / 2;

    cv::Mat resized;
    cv::resize(bgr, resized, cv::Size(new_w, new_h));
    cv::Mat input(imgsz_, imgsz_, CV_8UC3, cv::Scalar(114, 114, 114));
    resized.copyTo(input(cv::Rect(pad_left, pad_top, new_w, new_h)));
    cv::cvtColor(input, input, cv::COLOR_BGR2RGB);
    input.convertTo(input, CV_32FC3, 1.0 / 255.0);

    // HWC -> CHW
    std::vector<float> blob(static_cast<size_t>(3 * imgsz_ * imgsz_));
    std::vector<cv::Mat> channels(3);
    cv::split(input, channels);
    for (int c = 0; c < 3; ++c) {
      std::memcpy(
        blob.data() + static_cast<size_t>(c * imgsz_ * imgsz_),
        channels[c].data,
        static_cast<size_t>(imgsz_ * imgsz_) * sizeof(float));
    }

    std::array<int64_t, 4> input_shape{1, 3, imgsz_, imgsz_};
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
      memory_info_, blob.data(), blob.size(), input_shape.data(), input_shape.size());

    const char * input_names[] = {input_name_.c_str()};
    const char * output_names[] = {output_name_.c_str()};
    auto outputs = session_->Run(
      Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);

    float * data = outputs[0].GetTensorMutableData<float>();
    auto shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
    // YOLOv8: [1, 84, N] typically
    int64_t attrs = 84;
    int64_t num = 8400;
    bool transposed = false;  // false means [1,84,N]
    if (shape.size() == 3) {
      if (shape[1] < shape[2]) {
        attrs = shape[1];
        num = shape[2];
        transposed = false;
      } else {
        attrs = shape[2];
        num = shape[1];
        transposed = true;
      }
    }

    std::vector<Det2D> cand;
    const int num_classes = static_cast<int>(attrs - 4);
    for (int64_t i = 0; i < num; ++i) {
      float cx, cy, w, h;
      std::vector<float> scores(static_cast<size_t>(num_classes));
      if (!transposed) {
        cx = data[0 * num + i];
        cy = data[1 * num + i];
        w = data[2 * num + i];
        h = data[3 * num + i];
        for (int c = 0; c < num_classes; ++c) {
          scores[static_cast<size_t>(c)] = data[(4 + c) * num + i];
        }
      } else {
        const float * row = data + i * attrs;
        cx = row[0]; cy = row[1]; w = row[2]; h = row[3];
        for (int c = 0; c < num_classes; ++c) {
          scores[static_cast<size_t>(c)] = row[4 + c];
        }
      }
      const auto max_it = std::max_element(scores.begin(), scores.end());
      const float score = *max_it;
      if (score < conf_thresh_) {
        continue;
      }
      const int class_id = static_cast<int>(std::distance(scores.begin(), max_it));
      if (!whitelist_.empty() && whitelist_.count(class_id) == 0) {
        continue;
      }
      Det2D d;
      d.class_id = class_id;
      d.score = score;
      d.x1 = (cx - 0.5f * w - pad_left) / scale;
      d.y1 = (cy - 0.5f * h - pad_top) / scale;
      d.x2 = (cx + 0.5f * w - pad_left) / scale;
      d.y2 = (cy + 0.5f * h - pad_top) / scale;
      d.x1 = std::clamp(d.x1, 0.f, static_cast<float>(img_w - 1));
      d.y1 = std::clamp(d.y1, 0.f, static_cast<float>(img_h - 1));
      d.x2 = std::clamp(d.x2, 0.f, static_cast<float>(img_w - 1));
      d.y2 = std::clamp(d.y2, 0.f, static_cast<float>(img_h - 1));
      if (d.x2 > d.x1 && d.y2 > d.y1) {
        cand.push_back(d);
      }
    }
    return nms(cand, iou_thresh_);
  }

  bool roiDepth(
    const cv::Mat & depth_m, const Det2D & d, float & z_out, int & n_valid) const
  {
    const double crop = get_parameter("center_crop_ratio").as_double();
    const float cx = 0.5f * (d.x1 + d.x2);
    const float cy = 0.5f * (d.y1 + d.y2);
    const float bw = (d.x2 - d.x1) * static_cast<float>(crop);
    const float bh = (d.y2 - d.y1) * static_cast<float>(crop);
    int x1 = static_cast<int>(std::floor(cx - 0.5f * bw));
    int y1 = static_cast<int>(std::floor(cy - 0.5f * bh));
    int x2 = static_cast<int>(std::ceil(cx + 0.5f * bw));
    int y2 = static_cast<int>(std::ceil(cy + 0.5f * bh));
    x1 = std::clamp(x1, 0, depth_m.cols - 1);
    y1 = std::clamp(y1, 0, depth_m.rows - 1);
    x2 = std::clamp(x2, 0, depth_m.cols - 1);
    y2 = std::clamp(y2, 0, depth_m.rows - 1);

    const float zmin = static_cast<float>(get_parameter("min_depth_m").as_double());
    const float zmax = static_cast<float>(get_parameter("max_depth_m").as_double());
    std::vector<float> zs;
    for (int y = y1; y <= y2; ++y) {
      const float * row = depth_m.ptr<float>(y);
      for (int x = x1; x <= x2; ++x) {
        const float z = row[x];
        if (std::isfinite(z) && z >= zmin && z <= zmax) {
          zs.push_back(z);
        }
      }
    }
    n_valid = static_cast<int>(zs.size());
    if (n_valid < get_parameter("min_valid_pixels").as_int()) {
      return false;
    }
    std::sort(zs.begin(), zs.end());
    const double trim_low = get_parameter("trim_low").as_double();
    const double trim_high = get_parameter("trim_high").as_double();
    const size_t i0 = static_cast<size_t>(trim_low * (zs.size() - 1));
    const size_t i1 = static_cast<size_t>(trim_high * (zs.size() - 1));
    if (i1 <= i0) {
      return false;
    }
    std::vector<float> trimmed(zs.begin() + static_cast<long>(i0),
      zs.begin() + static_cast<long>(i1) + 1);
    const double percentile = get_parameter("depth_percentile").as_double();
    const size_t pi = static_cast<size_t>(
      std::clamp(percentile / 100.0, 0.0, 1.0) * (trimmed.size() - 1));
    z_out = trimmed[pi];
    return z_out > 0.f;
  }

  void publishDetections3d(
    const std::vector<Det2D> & dets, const cv::Mat & depth_m,
    const sensor_msgs::msg::Image::ConstSharedPtr & color_msg)
  {
    const double fx = camera_info_.k[0];
    const double fy = camera_info_.k[4];
    const double cx = camera_info_.k[2];
    const double cy = camera_info_.k[5];
    const std::string optical_frame = camera_info_.header.frame_id.empty()
      ? color_msg->header.frame_id
      : camera_info_.header.frame_id;

    rgbd_perception_msgs::msg::Detection3DArray arr;
    arr.header.stamp = color_msg->header.stamp;
    arr.header.frame_id = target_frame_;

    visualization_msgs::msg::MarkerArray markers;
    visualization_msgs::msg::Marker del;
    del.action = visualization_msgs::msg::Marker::DELETEALL;
    markers.markers.push_back(del);

    int mid = 0;
    for (const auto & d : dets) {
      float z = 0.f;
      int n_valid = 0;
      if (!roiDepth(depth_m, d, z, n_valid)) {
        continue;
      }
      const float u = 0.5f * (d.x1 + d.x2);
      const float v = 0.5f * (d.y1 + d.y2);
      geometry_msgs::msg::PointStamped p_cam;
      p_cam.header.stamp = color_msg->header.stamp;
      p_cam.header.frame_id = optical_frame;
      p_cam.point.x = (u - cx) * z / fx;
      p_cam.point.y = (v - cy) * z / fy;
      p_cam.point.z = z;

      geometry_msgs::msg::PointStamped p_base;
      try {
        p_base = tf_buffer_.transform(p_cam, target_frame_, tf2::durationFromSec(0.1));
      } catch (const tf2::TransformException & ex) {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "TF lift failed: %s", ex.what());
        continue;
      }

      rgbd_perception_msgs::msg::Detection3D det;
      det.header = arr.header;
      det.class_id = d.class_id;
      det.class_name = (d.class_id >= 0 && d.class_id < 80) ? kCocoNames[d.class_id] : "unknown";
      det.score = d.score;
      det.x1 = d.x1; det.y1 = d.y1; det.x2 = d.x2; det.y2 = d.y2;
      det.center_cam = p_cam.point;
      det.center_base = p_base.point;
      det.distance_m = z;
      det.num_valid_depth = n_valid;
      arr.detections.push_back(det);

      visualization_msgs::msg::Marker m;
      m.header.stamp = arr.header.stamp;
      m.header.frame_id = target_frame_;
      m.ns = "detections_3d";
      m.id = mid++;
      m.type = visualization_msgs::msg::Marker::SPHERE;
      m.action = visualization_msgs::msg::Marker::ADD;
      m.pose.position = p_base.point;
      m.pose.orientation.w = 1.0;
      m.scale.x = m.scale.y = m.scale.z = 0.15;
      m.color.r = 0.2f; m.color.g = 0.4f; m.color.b = 1.0f; m.color.a = 0.9f;
      m.lifetime = rclcpp::Duration::from_seconds(0.3);
      markers.markers.push_back(m);

      visualization_msgs::msg::Marker text = m;
      text.id = mid++;
      text.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
      text.scale.z = 0.12;
      text.pose.position.z += 0.15;
      char buf[64];
      std::snprintf(buf, sizeof(buf), "%s %.1fm", det.class_name.c_str(), z);
      text.text = buf;
      markers.markers.push_back(text);
    }

    det_pub_->publish(arr);
    marker_pub_->publish(markers);
  }

  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;
  message_filters::Subscriber<sensor_msgs::msg::Image> color_sub_;
  message_filters::Subscriber<sensor_msgs::msg::Image> depth_sub_;
  std::shared_ptr<message_filters::Synchronizer<SyncPolicy>> sync_;
  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr info_sub_;
  rclcpp::Publisher<rgbd_perception_msgs::msg::Detection3DArray>::SharedPtr det_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;

  Ort::Env env_;
  Ort::MemoryInfo memory_info_;
  std::unique_ptr<Ort::Session> session_;
  std::string input_name_;
  std::string output_name_;

  sensor_msgs::msg::CameraInfo camera_info_;
  bool have_info_{false};
  bool enable_detection_{true};
  std::string target_frame_;
  float conf_thresh_{0.35f};
  float iou_thresh_{0.45f};
  int imgsz_{640};
  std::unordered_set<int> whitelist_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<Yolo3dNode>());
  rclcpp::shutdown();
  return 0;
}
