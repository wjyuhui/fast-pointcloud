#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <omp.h>
#include <sched.h>

#if defined(__aarch64__) && defined(__ARM_NEON)
#include <arm_neon.h>
#define RGBD_PCL_HAS_NEON 1
#else
#define RGBD_PCL_HAS_NEON 0
#endif

#include <Eigen/Geometry>
#include <pcl/filters/voxel_grid.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <tf2_eigen/tf2_eigen.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

using PointT = pcl::PointXYZ;
using CloudT = pcl::PointCloud<PointT>;

namespace {

std::string formatCpuList(const std::vector<int> & cpus)
{
  std::string text;
  for (size_t i = 0; i < cpus.size(); ++i) {
    if (i != 0) {
      text += ',';
    }
    text += std::to_string(cpus[i]);
  }
  return text;
}

void tryAppendPoint(
  CloudT & cloud,
  float z,
  size_t ui,
  size_t vi,
  const float * ray_x,
  const float * ray_y,
  float zmin,
  float zmax,
  float xmin,
  float xmax,
  float zmin_roi,
  float zmax_roi,
  double r00, double r01, double r02, double tx,
  double r10, double r11, double r12, double ty,
  double r20, double r21, double r22, double tz)
{
  if (!std::isfinite(z) || z < zmin || z > zmax) {
    return;
  }

  const double x_camera = static_cast<double>(ray_x[ui]) * z;
  const double y_camera = static_cast<double>(ray_y[vi]) * z;
  const double z_camera = static_cast<double>(z);

  const float x_target = static_cast<float>(r00 * x_camera + r01 * y_camera + r02 * z_camera + tx);
  const float y_target = static_cast<float>(r10 * x_camera + r11 * y_camera + r12 * z_camera + ty);
  const float z_target = static_cast<float>(r20 * x_camera + r21 * y_camera + r22 * z_camera + tz);

  if (x_target < xmin || x_target > xmax || z_target < zmin_roi || z_target > zmax_roi) {
    return;
  }

  PointT p;
  p.x = x_target;
  p.y = y_target;
  p.z = z_target;
  cloud.push_back(p);
}

#if RGBD_PCL_HAS_NEON
struct NeonWorkspaceParams
{
  float zmin;
  float zmax;
  float xmin;
  float xmax;
  float zmin_roi;
  float zmax_roi;

  float r00, r01, r02, tx;
  float r10, r11, r12, ty;
  float r20, r21, r22, tz;
};

inline uint32x4_t finiteMask(const float32x4_t z)
{
  const uint32x4_t bits = vreinterpretq_u32_f32(z);
  const uint32x4_t exponent = vandq_u32(bits, vdupq_n_u32(0x7F800000u));
  return vmvnq_u32(vceqq_u32(exponent, vdupq_n_u32(0x7F800000u)));
}

inline void appendTransformedPointsNeon4(
  CloudT & cloud,
  const float32x4_t z,
  const size_t ui,
  const size_t vi,
  const float * ray_x,
  const float * ray_y,
  const NeonWorkspaceParams & params)
{
  const float32x4_t ray_x_vec = vld1q_f32(ray_x + ui);
  const float32x4_t ray_y_vec = vdupq_n_f32(ray_y[vi]);

  float32x4_t x_direction = vfmaq_n_f32(vdupq_n_f32(params.r02), ray_x_vec, params.r00);
  x_direction = vfmaq_n_f32(x_direction, ray_y_vec, params.r01);
  float32x4_t y_direction = vfmaq_n_f32(vdupq_n_f32(params.r12), ray_x_vec, params.r10);
  y_direction = vfmaq_n_f32(y_direction, ray_y_vec, params.r11);
  float32x4_t z_direction = vfmaq_n_f32(vdupq_n_f32(params.r22), ray_x_vec, params.r20);
  z_direction = vfmaq_n_f32(z_direction, ray_y_vec, params.r21);

  const float32x4_t x_target = vfmaq_f32(vdupq_n_f32(params.tx), z, x_direction);
  const float32x4_t y_target = vfmaq_f32(vdupq_n_f32(params.ty), z, y_direction);
  const float32x4_t z_target = vfmaq_f32(vdupq_n_f32(params.tz), z, z_direction);

  const uint32x4_t depth_valid = vandq_u32(
    vandq_u32(finiteMask(z), vcgeq_f32(z, vdupq_n_f32(params.zmin))),
    vcleq_f32(z, vdupq_n_f32(params.zmax)));
  const uint32x4_t roi_valid = vandq_u32(
    vandq_u32(
      vcgeq_f32(x_target, vdupq_n_f32(params.xmin)),
      vcleq_f32(x_target, vdupq_n_f32(params.xmax))),
    vandq_u32(
      vcgeq_f32(z_target, vdupq_n_f32(params.zmin_roi)),
      vcleq_f32(z_target, vdupq_n_f32(params.zmax_roi))));
  const uint32x4_t valid_mask = vandq_u32(depth_valid, roi_valid);

  alignas(16) float xs[4];
  alignas(16) float ys[4];
  alignas(16) float zs[4];
  alignas(16) uint32_t valid[4];
  vst1q_f32(xs, x_target);
  vst1q_f32(ys, y_target);
  vst1q_f32(zs, z_target);
  vst1q_u32(valid, valid_mask);

  for (size_t lane = 0; lane < 4; ++lane) {
    if (valid[lane] != 0U) {
      PointT p;
      p.x = xs[lane];
      p.y = ys[lane];
      p.z = zs[lane];
      cloud.push_back(p);
    }
  }
}

inline void tryAppendPointsNeon4(
  CloudT & cloud,
  const uint16_t * row,
  const size_t ui,
  const size_t vi,
  const int stride,
  const float * ray_x,
  const float * ray_y,
  const NeonWorkspaceParams & params)
{
  const size_t u = ui * static_cast<size_t>(stride);
  const size_t step = static_cast<size_t>(stride);
  alignas(16) uint32_t depth_values[4] = {
    row[u],
    row[u + step],
    row[u + 2U * step],
    row[u + 3U * step]
  };
  const uint32x4_t raw = vld1q_u32(depth_values);
  const float32x4_t z = vmulq_n_f32(vcvtq_f32_u32(raw), 0.001f);
  appendTransformedPointsNeon4(cloud, z, ui, vi, ray_x, ray_y, params);
}

inline void tryAppendPointsNeon4(
  CloudT & cloud,
  const float * row,
  const size_t ui,
  const size_t vi,
  const int stride,
  const float * ray_x,
  const float * ray_y,
  const NeonWorkspaceParams & params)
{
  const size_t u = ui * static_cast<size_t>(stride);
  const size_t step = static_cast<size_t>(stride);
  alignas(16) float depth_values[4] = {
    row[u],
    row[u + step],
    row[u + 2U * step],
    row[u + 3U * step]
  };
  const float32x4_t z = vld1q_f32(depth_values);
  appendTransformedPointsNeon4(cloud, z, ui, vi, ray_x, ray_y, params);
}
#endif

}  // namespace

class CloudWorkspaceNode : public rclcpp::Node
{
public:
  CloudWorkspaceNode()
  : Node("cloud_workspace_node"),
    tf_buffer_(this->get_clock()),
    tf_listener_(tf_buffer_)
  {
    declare_parameter<std::string>("depth_topic", "/camera/depth/image_raw");
    declare_parameter<std::string>("camera_info_topic", "/camera/color/camera_info");
    declare_parameter<std::string>("output_cloud_topic", "/perception/cloud_workspace");
    declare_parameter<std::string>("target_frame", "base_link");
    declare_parameter<int>("depth_stride", 4);
    declare_parameter<double>("min_depth_m", 0.2);
    declare_parameter<double>("max_depth_m", 5.0);
    declare_parameter<double>("passthrough_x_min", 0.2);
    declare_parameter<double>("passthrough_x_max", 5.0);
    declare_parameter<double>("passthrough_z_min", -1.0);
    declare_parameter<double>("passthrough_z_max", 2.0);
    declare_parameter<double>("voxel_leaf_m", 0.05);
    declare_parameter<double>("tf_timeout_sec", 0.1);
    declare_parameter<bool>("projection_enable_openmp", false);
    declare_parameter<int>("projection_openmp_threads", 4);
    declare_parameter<std::vector<int64_t>>("projection_openmp_cpus", std::vector<int64_t>{});
    declare_parameter<bool>("projection_enable_neon", false);
    declare_parameter<int>("projection_compute_cpu", 7);

    depth_topic_ = get_parameter("depth_topic").as_string();
    info_topic_ = get_parameter("camera_info_topic").as_string();
    output_topic_ = get_parameter("output_cloud_topic").as_string();
    target_frame_ = get_parameter("target_frame").as_string();
    projection_enable_openmp_ = get_parameter("projection_enable_openmp").as_bool();
    projection_openmp_threads_ = get_parameter("projection_openmp_threads").as_int();
    const std::vector<int64_t> configured_cpus =
      get_parameter("projection_openmp_cpus").as_integer_array();
    projection_openmp_cpus_.reserve(configured_cpus.size());
    for (const int64_t cpu : configured_cpus) {
      if (cpu < 0 || cpu >= CPU_SETSIZE) {
        throw std::invalid_argument(
                "projection_openmp_cpus contains an invalid CPU: " + std::to_string(cpu));
      }
      const int cpu_id = static_cast<int>(cpu);
      if (std::find(
          projection_openmp_cpus_.begin(), projection_openmp_cpus_.end(), cpu_id) !=
        projection_openmp_cpus_.end())
      {
        throw std::invalid_argument(
                "projection_openmp_cpus must not contain duplicate CPUs");
      }
      projection_openmp_cpus_.push_back(cpu_id);
    }
    projection_enable_neon_ = get_parameter("projection_enable_neon").as_bool();
    projection_compute_cpu_ = get_parameter("projection_compute_cpu").as_int();

    if (projection_enable_openmp_ && projection_enable_neon_) {
      throw std::invalid_argument(
        "V4 does not support projection OpenMP and NEON at the same time");
    }
#if !RGBD_PCL_HAS_NEON
    if (projection_enable_neon_) {
      throw std::invalid_argument(
        "projection_enable_neon=true requires an AArch64 NEON build");
    }
#endif
    if (projection_openmp_threads_ < 1 || projection_openmp_threads_ > kMaxOpenMPThreads) {
      throw std::invalid_argument(
        "projection_openmp_threads must be in [1, " + std::to_string(kMaxOpenMPThreads) + "]");
    }
    if (projection_enable_openmp_ &&
      projection_openmp_cpus_.size() != static_cast<size_t>(projection_openmp_threads_))
    {
      throw std::invalid_argument(
              "projection_openmp_cpus count must match projection_openmp_threads");
    }
    if (projection_compute_cpu_ < 0 || projection_compute_cpu_ >= CPU_SETSIZE) {
      throw std::invalid_argument(
        "projection_compute_cpu must be in [0, " + std::to_string(CPU_SETSIZE - 1) + "]");
    }
    if (projection_enable_openmp_) {
      projection_compute_cpu_ = projection_openmp_cpus_.front();
    }

    // omp_set_dynamic(0);
    // omp_set_nested(0);

    core_samples_.reserve(kMeasureFrames);
    voxel_samples_.reserve(kMeasureFrames);
    callback_samples_.reserve(kMeasureFrames);
    queue_wait_samples_.reserve(kMeasureFrames);

    rclcpp::SensorDataQoS qos;
    info_sub_ = create_subscription<sensor_msgs::msg::CameraInfo>(
      info_topic_, qos,
      [this](const sensor_msgs::msg::CameraInfo::SharedPtr msg) {
        std::lock_guard<std::mutex> lock(camera_info_mutex_);
        camera_info_ = *msg;
        have_info_ = true;
      });
    depth_sub_ = create_subscription<sensor_msgs::msg::Image>(
      depth_topic_, qos,
      std::bind(&CloudWorkspaceNode::depthCallback, this, std::placeholders::_1));
    cloud_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>(output_topic_, qos);

    RCLCPP_INFO(get_logger(),
      "cloud_workspace_node depth=%s info=%s -> %s (%s) openmp=%s requested_threads=%d neon=%s compute_cpu=%d",
      depth_topic_.c_str(), info_topic_.c_str(), output_topic_.c_str(),
      target_frame_.c_str(),
      projection_enable_openmp_ ? "true" : "false",
      projection_openmp_threads_,
      projection_enable_neon_ ? "true" : "false",
      projection_compute_cpu_);
    if (projection_enable_openmp_) {
      const char * bind = std::getenv("OMP_PROC_BIND");
      RCLCPP_INFO(
        get_logger(),
        "OpenMP manual affinity cpus=%s OMP_PROC_BIND=%s OMP_DYNAMIC=FALSE",
        formatCpuList(projection_openmp_cpus_).c_str(),
        bind != nullptr ? bind : "(unset)");
    }

    compute_thread_ = std::thread(&CloudWorkspaceNode::computeLoop, this);
  }

  ~CloudWorkspaceNode() override
  {
    {
      std::lock_guard<std::mutex> lock(job_mutex_);
      stop_requested_ = true;
      pending_job_.reset();
    }
    job_cv_.notify_all();

    if (compute_thread_.joinable()) {
      compute_thread_.join();
    }

    if (core_samples_.size() >= kMinDumpFrames) {
      logPerf();
    }
  }

private:
  static constexpr size_t kWarmupFrames = 30;
  static constexpr size_t kMeasureFrames = 1000;
  static constexpr size_t kMinDumpFrames = 200;
  static constexpr int kMaxOpenMPThreads = 16;

  struct FrameJob
  {
    sensor_msgs::msg::Image::SharedPtr depth;
    sensor_msgs::msg::CameraInfo camera_info;
    std::chrono::steady_clock::time_point pipeline_begin;
    uint64_t sequence{0};
  };

  static double calculateP95(std::vector<double> samples)
  {
    if (samples.empty()) {
      return 0.0;
    }

    std::sort(samples.begin(), samples.end());

    const size_t index =
      static_cast<size_t>(std::ceil(0.95 * static_cast<double>(samples.size()))) - 1;
    return samples[std::min(index, samples.size() - 1)];
  }

  void logPerf() const
  {
    const char * affinity_ok = (compute_affinity_valid_ && omp_affinity_valid_) ? "true" : "false";
    if (projection_enable_openmp_) {
      RCLCPP_INFO(
        get_logger(),
        "PERF V3_OMP requested_threads=%d actual_threads=%d affinity_ok=%s samples=%zu "
        "queue_wait_p95=%.3f ms core_p95=%.3f ms voxel_p95=%.3f ms callback_p95=%.3f ms "
        "submitted=%lu processed=%lu dropped_pending=%lu",
        projection_openmp_threads_,
        omp_actual_threads_,
        affinity_ok,
        core_samples_.size(),
        calculateP95(queue_wait_samples_),
        calculateP95(core_samples_),
        calculateP95(voxel_samples_),
        calculateP95(callback_samples_),
        static_cast<unsigned long>(submitted_frames_.load()),
        static_cast<unsigned long>(processed_frames_.load()),
        static_cast<unsigned long>(dropped_pending_frames_.load()));
      if (!compute_affinity_valid_ || !omp_affinity_valid_) {
        RCLCPP_WARN(
          get_logger(),
          "PERF V3_OMP result is invalid because compute/OpenMP affinity failed");
      }
    } else if (projection_enable_neon_) {
      RCLCPP_INFO(
        get_logger(),
        "PERF V4_NEON affinity_ok=%s samples=%zu "
        "queue_wait_p95=%.3f ms core_p95=%.3f ms voxel_p95=%.3f ms callback_p95=%.3f ms "
        "submitted=%lu processed=%lu dropped_pending=%lu",
        affinity_ok,
        core_samples_.size(),
        calculateP95(queue_wait_samples_),
        calculateP95(core_samples_),
        calculateP95(voxel_samples_),
        calculateP95(callback_samples_),
        static_cast<unsigned long>(submitted_frames_.load()),
        static_cast<unsigned long>(processed_frames_.load()),
        static_cast<unsigned long>(dropped_pending_frames_.load()));
      if (!compute_affinity_valid_) {
        RCLCPP_WARN(
          get_logger(),
          "PERF V4_NEON result is invalid because compute-thread affinity failed");
      }
    } else {
      RCLCPP_INFO(
        get_logger(),
        "PERF V2_SCALAR affinity_ok=%s samples=%zu "
        "queue_wait_p95=%.3f ms core_p95=%.3f ms voxel_p95=%.3f ms callback_p95=%.3f ms "
        "submitted=%lu processed=%lu dropped_pending=%lu",
        affinity_ok,
        core_samples_.size(),
        calculateP95(queue_wait_samples_),
        calculateP95(core_samples_),
        calculateP95(voxel_samples_),
        calculateP95(callback_samples_),
        static_cast<unsigned long>(submitted_frames_.load()),
        static_cast<unsigned long>(processed_frames_.load()),
        static_cast<unsigned long>(dropped_pending_frames_.load()));
      if (!compute_affinity_valid_) {
        RCLCPP_WARN(
          get_logger(),
          "PERF V2_SCALAR result is invalid because compute-thread affinity failed");
      }
    }
  }

  void logOmpAffinity(int actual_threads)
  {
    RCLCPP_INFO(
      get_logger(),
      "OMP_AFFINITY requested=%d actual=%d manual_cpus=%s",
      projection_openmp_threads_,
      actual_threads,
      formatCpuList(projection_openmp_cpus_).c_str());

    omp_affinity_valid_ = actual_threads == projection_openmp_threads_;
    if (!omp_affinity_valid_) {
      if (actual_threads != projection_openmp_threads_) {
        RCLCPP_WARN(
          get_logger(),
          "OMP_AFFINITY actual_threads=%d != requested_threads=%d; this run is invalid",
          actual_threads, projection_openmp_threads_);
      }
    }

    for (int tid = 0; tid < actual_threads; ++tid) {
      const int cpu = omp_cpu_by_thread_[static_cast<size_t>(tid)];
      const int target_cpu = projection_openmp_cpus_[static_cast<size_t>(tid)];
      const int affinity_error = omp_affinity_errors_[static_cast<size_t>(tid)];
      RCLCPP_INFO(
        get_logger(),
        "OMP_AFFINITY tid=%d target_cpu=%d cpu=%d set_error=%d",
        tid, target_cpu, cpu, affinity_error);
      if (affinity_error != 0 || cpu != target_cpu) {
        omp_affinity_valid_ = false;
        RCLCPP_WARN(
          get_logger(),
          "OMP_AFFINITY tid=%d target_cpu=%d cpu=%d set_error=%d; this run is invalid",
          tid, target_cpu, cpu, affinity_error);
      }
    }
  }

  void recordTiming(double queue_wait_ms, double core_ms, double voxel_ms, double callback_ms)
  {
    ++successful_frames_;
    if (successful_frames_ <= kWarmupFrames) {
      return;
    }

    queue_wait_samples_.push_back(queue_wait_ms);
    core_samples_.push_back(core_ms);
    voxel_samples_.push_back(voxel_ms);
    callback_samples_.push_back(callback_ms);

    if (core_samples_.size() < kMeasureFrames) {
      return;
    }

    logPerf();
    queue_wait_samples_.clear();
    core_samples_.clear();
    voxel_samples_.clear();
    callback_samples_.clear();
  }

  void depthCallback(const sensor_msgs::msg::Image::SharedPtr msg)
  {
    const auto pipeline_begin = std::chrono::steady_clock::now();

    bool have_info = false;
    sensor_msgs::msg::CameraInfo camera_info_snapshot;
    {
      std::lock_guard<std::mutex> lock(camera_info_mutex_);
      have_info = have_info_;
      if (have_info) {
        camera_info_snapshot = camera_info_;
      }
    }
    if (!have_info) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "Waiting for camera_info on %s",
        info_topic_.c_str());
      return;
    }

    FrameJob job;
    job.depth = msg;
    job.camera_info = std::move(camera_info_snapshot);
    job.pipeline_begin = pipeline_begin;
    job.sequence = submitted_frames_.fetch_add(1) + 1;

    {
      std::lock_guard<std::mutex> lock(job_mutex_);
      if (stop_requested_) {
        return;
      }
      if (pending_job_.has_value()) {
        dropped_pending_frames_.fetch_add(1);
      }
      pending_job_ = std::move(job);
    }

    job_cv_.notify_one();
  }

  bool bindCurrentThreadToCpu(int cpu, int & error)
  {
    cpu_set_t mask;
    CPU_ZERO(&mask);
    CPU_SET(cpu, &mask);

    if (sched_setaffinity(0, sizeof(mask), &mask) != 0) {
      error = errno;
      return false;
    }

    error = 0;
    return true;
  }

  void computeLoop()
  {
    const int coordinator_cpu = projection_enable_openmp_
      ? projection_openmp_cpus_.front()
      : projection_compute_cpu_;

    int bind_error = 0;
    if (!bindCurrentThreadToCpu(coordinator_cpu, bind_error)) {
      compute_affinity_valid_ = false;
      RCLCPP_ERROR(
        get_logger(),
        "COMPUTE_THREAD bind failed target_cpu=%d actual_cpu=%d errno=%d",
        coordinator_cpu, sched_getcpu(), bind_error);
    } else {
      RCLCPP_INFO(
        get_logger(),
        "COMPUTE_THREAD target_cpu=%d actual_cpu=%d",
        coordinator_cpu, sched_getcpu());
    }

    while (true) {
      FrameJob job;
      {
        std::unique_lock<std::mutex> lock(job_mutex_);
        job_cv_.wait(lock, [this]() {
          return stop_requested_ || pending_job_.has_value();
        });

        if (stop_requested_) {
          return;
        }

        job = std::move(*pending_job_);
        pending_job_.reset();
      }

      const auto compute_begin = std::chrono::steady_clock::now();
      const double queue_wait_ms =
        std::chrono::duration<double, std::milli>(
          compute_begin - job.pipeline_begin).count();

      try {
        if (processFrame(job, queue_wait_ms)) {
          processed_frames_.fetch_add(1);
        }
      } catch (const std::exception & ex) {
        RCLCPP_ERROR_THROTTLE(
          get_logger(), *get_clock(), 2000,
          "Compute thread frame failed: %s", ex.what());
      }
    }
  }

  bool processFrame(const FrameJob & job, double queue_wait_ms)
  {
    const std::string optical_frame = job.camera_info.header.frame_id.empty()
      ? job.depth->header.frame_id
      : job.camera_info.header.frame_id;

    Eigen::Affine3d transform = Eigen::Affine3d::Identity();
    double tf_lookup_ms = 0.0;
    if (!lookupTransformMatrix(optical_frame, job.depth->header.stamp, transform, tf_lookup_ms)) {
      return false;
    }

    const auto core_begin = std::chrono::steady_clock::now();
    CloudT::Ptr cloud_roi(new CloudT);
    bool ok = false;
    if (projection_enable_openmp_) {
      ok = depthToWorkspaceOpenMP(*job.depth, job.camera_info, transform, *cloud_roi);
    } else if (projection_enable_neon_) {
#if RGBD_PCL_HAS_NEON
      ok = depthToWorkspaceNeon(*job.depth, job.camera_info, transform, *cloud_roi);
#else
      ok = false;
#endif
    } else {
      ok = depthToWorkspaceScalar(*job.depth, job.camera_info, transform, *cloud_roi);
    }
    if (!ok) {
      return false;
    }
    const auto core_end = std::chrono::steady_clock::now();
    const double core_ms =
      std::chrono::duration<double, std::milli>(core_end - core_begin).count();

    const auto voxel_begin = std::chrono::steady_clock::now();
    CloudT::Ptr cloud_voxel(new CloudT);  // 通过PCL做的体素滤波
    {
      pcl::VoxelGrid<PointT> vg;
      vg.setInputCloud(cloud_roi);
      const float leaf = static_cast<float>(get_parameter("voxel_leaf_m").as_double());
      vg.setLeafSize(leaf, leaf, leaf);
      vg.filter(*cloud_voxel);
    }
    const auto voxel_end = std::chrono::steady_clock::now();
    const double voxel_ms =
      std::chrono::duration<double, std::milli>(voxel_end - voxel_begin).count();

    sensor_msgs::msg::PointCloud2 out;
    pcl::toROSMsg(*cloud_voxel, out);
    out.header.stamp = job.depth->header.stamp;
    out.header.frame_id = target_frame_;
    cloud_pub_->publish(out);

    const auto callback_end = std::chrono::steady_clock::now();
    const double callback_ms =
      std::chrono::duration<double, std::milli>(callback_end - job.pipeline_begin).count();
    recordTiming(queue_wait_ms, core_ms, voxel_ms, callback_ms);
    return true;
  }

  bool lookupTransformMatrix(
    const std::string & frame_in,
    const rclcpp::Time & stamp,
    Eigen::Affine3d & transform,
    double & tf_lookup_ms)
  {
    tf_lookup_ms = 0.0;
    if (frame_in == target_frame_) {
      transform = Eigen::Affine3d::Identity();
      return true;
    }

    geometry_msgs::msg::TransformStamped tf;
    const auto tf_lookup_begin = std::chrono::steady_clock::now();
    try {
      tf = tf_buffer_.lookupTransform(
        target_frame_, frame_in, stamp,
        tf2::durationFromSec(get_parameter("tf_timeout_sec").as_double()));
    } catch (const tf2::TransformException & ex) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
        "TF %s <- %s failed: %s", target_frame_.c_str(), frame_in.c_str(), ex.what());
      return false;
    }
    tf_lookup_ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - tf_lookup_begin).count();
    transform = tf2::transformToEigen(tf.transform);
    return true;
  }

  bool prepareWorkspaceInputs(
    const sensor_msgs::msg::Image & depth,
    const sensor_msgs::msg::CameraInfo & camera_info,
    const Eigen::Affine3d & transform,
    int & stride,
    float & zmin,
    float & zmax,
    float & xmin,
    float & xmax,
    float & zmin_roi,
    float & zmax_roi,
    double & r00, double & r01, double & r02, double & tx,
    double & r10, double & r11, double & r12, double & ty,
    double & r20, double & r21, double & r22, double & tz,
    bool & is_u16)
  {
    stride = std::max(1, static_cast<int>(get_parameter("depth_stride").as_int()));
    zmin = static_cast<float>(get_parameter("min_depth_m").as_double());
    zmax = static_cast<float>(get_parameter("max_depth_m").as_double());
    xmin = static_cast<float>(get_parameter("passthrough_x_min").as_double());
    xmax = static_cast<float>(get_parameter("passthrough_x_max").as_double());
    zmin_roi = static_cast<float>(get_parameter("passthrough_z_min").as_double());
    zmax_roi = static_cast<float>(get_parameter("passthrough_z_max").as_double());

    const double fx = camera_info.k[0];
    const double fy = camera_info.k[4];
    const double cx = camera_info.k[2];
    const double cy = camera_info.k[5];
    if (!(fx > 1e-6 && fy > 1e-6)) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "Invalid camera_info intrinsics");
      return false;
    }

    const int width = static_cast<int>(depth.width);
    const int height = static_cast<int>(depth.height);
    if (width <= 0 || height <= 0 || depth.data.empty()) {
      return false;
    }

    if (!ensureRayCache(camera_info, width, height, stride, fx, fy, cx, cy)) {
      return false;
    }

    is_u16 = (depth.encoding == "16UC1" || depth.encoding == "mono16");
    const bool is_f32 = (depth.encoding == "32FC1");
    if (!is_u16 && !is_f32) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
        "Unsupported depth encoding: %s", depth.encoding.c_str());
      return false;
    }
    if (is_u16 && depth.step < static_cast<size_t>(width) * sizeof(uint16_t)) {
      return false;
    }
    if (is_f32 && depth.step < static_cast<size_t>(width) * sizeof(float)) {
      return false;
    }

    const Eigen::Matrix4d & M = transform.matrix();
    r00 = M(0, 0); r01 = M(0, 1); r02 = M(0, 2); tx = M(0, 3);
    r10 = M(1, 0); r11 = M(1, 1); r12 = M(1, 2); ty = M(1, 3);
    r20 = M(2, 0); r21 = M(2, 1); r22 = M(2, 2); tz = M(2, 3);
    return true;
  }

  bool depthToWorkspaceScalar(
    const sensor_msgs::msg::Image & depth,
    const sensor_msgs::msg::CameraInfo & camera_info,
    const Eigen::Affine3d & transform,
    CloudT & cloud_roi)
  {
    int stride = 1;
    float zmin = 0.0f, zmax = 0.0f, xmin = 0.0f, xmax = 0.0f, zmin_roi = 0.0f, zmax_roi = 0.0f;
    double r00 = 0, r01 = 0, r02 = 0, tx = 0;
    double r10 = 0, r11 = 0, r12 = 0, ty = 0;
    double r20 = 0, r21 = 0, r22 = 0, tz = 0;
    bool is_u16 = false;
    if (!prepareWorkspaceInputs(
          depth, camera_info, transform, stride, zmin, zmax, xmin, xmax, zmin_roi, zmax_roi,
          r00, r01, r02, tx, r10, r11, r12, ty, r20, r21, r22, tz, is_u16))
    {
      return false;
    }

    const int width = static_cast<int>(depth.width);
    const int height = static_cast<int>(depth.height);
    const float * ray_x = ray_x_.data();
    const float * ray_y = ray_y_.data();

    cloud_roi.clear();
    cloud_roi.reserve(ray_x_.size() * ray_y_.size());

    if (is_u16) {
      for (int vi = 0, v = 0; v < height; v += stride, ++vi) {
        const auto * row = reinterpret_cast<const uint16_t *>(
          depth.data.data() + static_cast<size_t>(v) * depth.step);
        for (int ui = 0, u = 0; u < width; u += stride, ++ui) {
          tryAppendPoint(
            cloud_roi, static_cast<float>(row[u]) * 0.001f,
            static_cast<size_t>(ui), static_cast<size_t>(vi),
            ray_x, ray_y, zmin, zmax, xmin, xmax, zmin_roi, zmax_roi,
            r00, r01, r02, tx, r10, r11, r12, ty, r20, r21, r22, tz);
        }
      }
    } else {
      for (int vi = 0, v = 0; v < height; v += stride, ++vi) {
        const auto * row = reinterpret_cast<const float *>(
          depth.data.data() + static_cast<size_t>(v) * depth.step);
        for (int ui = 0, u = 0; u < width; u += stride, ++ui) {
          tryAppendPoint(
            cloud_roi, row[u],
            static_cast<size_t>(ui), static_cast<size_t>(vi),
            ray_x, ray_y, zmin, zmax, xmin, xmax, zmin_roi, zmax_roi,
            r00, r01, r02, tx, r10, r11, r12, ty, r20, r21, r22, tz);
        }
      }
    }

    cloud_roi.width = cloud_roi.size();
    cloud_roi.height = 1;
    cloud_roi.is_dense = true;
    return !cloud_roi.empty();
  }

  void prepareOpenMPBuffers(int requested_threads, size_t sampled_rows, size_t sampled_cols)
  {
    const size_t n = static_cast<size_t>(requested_threads);
    if (omp_thread_clouds_.size() < n) {  // 如果设置的线程数大于数组大小，则扩容
      omp_thread_clouds_.reserve(n);
      while (omp_thread_clouds_.size() < n) {
        omp_thread_clouds_.emplace_back(new CloudT);
      }
    }

    const size_t cap = sampled_rows * sampled_cols;
    for (size_t i = 0; i < n; ++i) {
      omp_thread_clouds_[i]->clear();
      if (omp_thread_clouds_[i]->points.capacity() < cap) {
        omp_thread_clouds_[i]->points.reserve(cap);
      }
    }

    if (omp_cpu_by_thread_.size() != n) {
      omp_cpu_by_thread_.assign(n, -1);
      omp_affinity_errors_.assign(n, 0);
    }
  }

  bool depthToWorkspaceOpenMP(
    const sensor_msgs::msg::Image & depth,
    const sensor_msgs::msg::CameraInfo & camera_info,
    const Eigen::Affine3d & transform,
    CloudT & cloud_roi)
  {
    int stride = 1;
    float zmin = 0.0f, zmax = 0.0f, xmin = 0.0f, xmax = 0.0f, zmin_roi = 0.0f, zmax_roi = 0.0f;
    double r00 = 0, r01 = 0, r02 = 0, tx = 0;
    double r10 = 0, r11 = 0, r12 = 0, ty = 0;
    double r20 = 0, r21 = 0, r22 = 0, tz = 0;
    bool is_u16 = false;
    if (!prepareWorkspaceInputs(  // 给需要的参数赋值
          depth, camera_info, transform, stride, zmin, zmax, xmin, xmax, zmin_roi, zmax_roi,
          r00, r01, r02, tx, r10, r11, r12, ty, r20, r21, r22, tz, is_u16))
    {
      return false;
    }

    const size_t sampled_rows = ray_y_.size();  // 行数
    const size_t sampled_cols = ray_x_.size();  // 列数
    if (sampled_rows == 0 || sampled_cols == 0) {
      return false;
    }

    prepareOpenMPBuffers(projection_openmp_threads_, sampled_rows, sampled_cols);  // 给用于装点云的数组进行扩容

    const float * ray_x = ray_x_.data();
    const float * ray_y = ray_y_.data();
    const uint8_t * depth_data = depth.data.data();
    const size_t depth_step = depth.step;
    std::vector<CloudT::Ptr> * thread_clouds = &omp_thread_clouds_;
    const int requested_threads = projection_openmp_threads_;
    int actual_threads = 0;
    bool capture_affinity = !omp_affinity_logged_;
    int * cpu_by_thread = omp_cpu_by_thread_.data();
    int * affinity_errors = omp_affinity_errors_.data();
    const int * target_cpus = projection_openmp_cpus_.data();

    /**
    #pragma omp parallel 表示后面的操作开始并行执行
    num_threads(requested_threads) 设置并行线程数
    每个线程通过 sched_setaffinity 只绑定自身，不改变 ROS/FastDDS 线程亲和性。
    tid0 是持久计算协调线程，只在第一次进入并行区时绑定到目标大核。
    default(none) 显式声明外部变量是 shared 还是 private，有利于避免数据竞争。
    shared（。。。。。。） 表示这些变量在并行区域内共享，可以被所有线程访问和修改。
    #pragma omp single  会保证只有一个线程执行后面的操作，其他线程等待。
    */
#pragma omp parallel num_threads(requested_threads) default(none) \
    shared(actual_threads, is_u16, sampled_rows, sampled_cols, stride, \
           depth_data, depth_step, ray_x, ray_y, thread_clouds, \
           zmin, zmax, xmin, xmax, zmin_roi, zmax_roi, \
           r00, r01, r02, tx, r10, r11, r12, ty, r20, r21, r22, tz, \
           capture_affinity, cpu_by_thread, affinity_errors, target_cpus)
    {
      const int tid = omp_get_thread_num();  // 获取当前线程的编号
#pragma omp single
      {
        actual_threads = omp_get_num_threads();  // 当前这个并行区里实际有多少个工作线程
      }

      cpu_set_t compute_affinity;
      CPU_ZERO(&compute_affinity);
      CPU_SET(target_cpus[tid], &compute_affinity);
      // libgomp keeps all team threads alive between parallel regions. Bind
      // each one once and only rebind it when the configured target changes.
      static thread_local int bound_compute_cpu = -1;
      if (bound_compute_cpu == target_cpus[tid]) {
        affinity_errors[tid] = 0;
      } else if (sched_setaffinity(0, sizeof(compute_affinity), &compute_affinity) != 0) {
        affinity_errors[tid] = errno;
      } else {
        affinity_errors[tid] = 0;
        bound_compute_cpu = target_cpus[tid];
      }

      if (capture_affinity) {
        cpu_by_thread[tid] = sched_getcpu();
      }

      const int team_size = actual_threads;
      const size_t vi_begin = sampled_rows * static_cast<size_t>(tid) /
        static_cast<size_t>(team_size);
      const size_t vi_end = sampled_rows * static_cast<size_t>(tid + 1) /
        static_cast<size_t>(team_size);
      CloudT & local_cloud = *(*thread_clouds)[static_cast<size_t>(tid)];

      if (is_u16) {
        for (size_t vi = vi_begin; vi < vi_end; ++vi) {
          const int v = static_cast<int>(vi) * stride;
          const auto * row = reinterpret_cast<const uint16_t *>(
            depth_data + static_cast<size_t>(v) * depth_step);
          for (size_t ui = 0; ui < sampled_cols; ++ui) {
            const int u = static_cast<int>(ui) * stride;
            tryAppendPoint(
              local_cloud, static_cast<float>(row[u]) * 0.001f, ui, vi,
              ray_x, ray_y, zmin, zmax, xmin, xmax, zmin_roi, zmax_roi,
              r00, r01, r02, tx, r10, r11, r12, ty, r20, r21, r22, tz);
          }
        }
      } else {
        for (size_t vi = vi_begin; vi < vi_end; ++vi) {
          const int v = static_cast<int>(vi) * stride;
          const auto * row = reinterpret_cast<const float *>(
            depth_data + static_cast<size_t>(v) * depth_step);
          for (size_t ui = 0; ui < sampled_cols; ++ui) {
            const int u = static_cast<int>(ui) * stride;
            tryAppendPoint(
              local_cloud, row[u], ui, vi,
              ray_x, ray_y, zmin, zmax, xmin, xmax, zmin_roi, zmax_roi,
              r00, r01, r02, tx, r10, r11, r12, ty, r20, r21, r22, tz);
          }
        }
      }
    }  // 并行区结束

    if (!omp_affinity_logged_) {
      logOmpAffinity(actual_threads);
      omp_affinity_logged_ = true;
    }

    // 后面是合并的逻辑
    if (actual_threads != omp_actual_threads_) {
      omp_actual_threads_ = actual_threads;
      RCLCPP_INFO(
        get_logger(),
        "OpenMP team requested_threads=%d actual_threads=%d",
        projection_openmp_threads_, omp_actual_threads_);
    }

    size_t total_points = 0;
    for (int tid = 0; tid < actual_threads; ++tid) {
      total_points += omp_thread_clouds_[static_cast<size_t>(tid)]->size();
    }

    cloud_roi.clear();
    cloud_roi.reserve(total_points);
    for (int tid = 0; tid < actual_threads; ++tid) {
      const auto & points = omp_thread_clouds_[static_cast<size_t>(tid)]->points;
      cloud_roi.points.insert(cloud_roi.points.end(), points.begin(), points.end());
    }
    cloud_roi.width = cloud_roi.size();
    cloud_roi.height = 1;
    cloud_roi.is_dense = true;
    return !cloud_roi.empty();
  }

#if RGBD_PCL_HAS_NEON
  bool depthToWorkspaceNeon(
    const sensor_msgs::msg::Image & depth,
    const sensor_msgs::msg::CameraInfo & camera_info,
    const Eigen::Affine3d & transform,
    CloudT & cloud_roi)
  {
    int stride = 1;
    float zmin = 0.0f, zmax = 0.0f, xmin = 0.0f, xmax = 0.0f, zmin_roi = 0.0f, zmax_roi = 0.0f;
    double r00 = 0, r01 = 0, r02 = 0, tx = 0;
    double r10 = 0, r11 = 0, r12 = 0, ty = 0;
    double r20 = 0, r21 = 0, r22 = 0, tz = 0;
    bool is_u16 = false;
    if (!prepareWorkspaceInputs(
          depth, camera_info, transform, stride, zmin, zmax, xmin, xmax, zmin_roi, zmax_roi,
          r00, r01, r02, tx, r10, r11, r12, ty, r20, r21, r22, tz, is_u16))
    {
      return false;
    }

    const size_t sampled_rows = ray_y_.size();
    const size_t sampled_cols = ray_x_.size();
    if (sampled_rows == 0 || sampled_cols == 0) {
      return false;
    }

    const float * ray_x = ray_x_.data();
    const float * ray_y = ray_y_.data();
    const NeonWorkspaceParams params{
      zmin, zmax, xmin, xmax, zmin_roi, zmax_roi,
      static_cast<float>(r00), static_cast<float>(r01), static_cast<float>(r02), static_cast<float>(tx),
      static_cast<float>(r10), static_cast<float>(r11), static_cast<float>(r12), static_cast<float>(ty),
      static_cast<float>(r20), static_cast<float>(r21), static_cast<float>(r22), static_cast<float>(tz)
    };

    cloud_roi.clear();
    cloud_roi.reserve(sampled_rows * sampled_cols);

    if (is_u16) {
      for (size_t vi = 0; vi < sampled_rows; ++vi) {
        const int v = static_cast<int>(vi) * stride;
        const auto * row = reinterpret_cast<const uint16_t *>(
          depth.data.data() + static_cast<size_t>(v) * depth.step);

        size_t ui = 0;
        for (; ui + 3 < sampled_cols; ui += 4) {
          tryAppendPointsNeon4(cloud_roi, row, ui, vi, stride, ray_x, ray_y, params);
        }
        for (; ui < sampled_cols; ++ui) {
          const int u = static_cast<int>(ui) * stride;
          tryAppendPoint(
            cloud_roi, static_cast<float>(row[u]) * 0.001f, ui, vi,
            ray_x, ray_y, zmin, zmax, xmin, xmax, zmin_roi, zmax_roi,
            r00, r01, r02, tx, r10, r11, r12, ty, r20, r21, r22, tz);
        }
      }
    } else {
      for (size_t vi = 0; vi < sampled_rows; ++vi) {
        const int v = static_cast<int>(vi) * stride;
        const auto * row = reinterpret_cast<const float *>(
          depth.data.data() + static_cast<size_t>(v) * depth.step);

        size_t ui = 0;
        for (; ui + 3 < sampled_cols; ui += 4) {
          tryAppendPointsNeon4(cloud_roi, row, ui, vi, stride, ray_x, ray_y, params);
        }
        for (; ui < sampled_cols; ++ui) {
          const int u = static_cast<int>(ui) * stride;
          tryAppendPoint(
            cloud_roi, row[u], ui, vi,
            ray_x, ray_y, zmin, zmax, xmin, xmax, zmin_roi, zmax_roi,
            r00, r01, r02, tx, r10, r11, r12, ty, r20, r21, r22, tz);
        }
      }
    }

    cloud_roi.width = cloud_roi.size();
    cloud_roi.height = 1;
    cloud_roi.is_dense = true;
    return !cloud_roi.empty();
  }
#endif

  bool ensureRayCache(
    const sensor_msgs::msg::CameraInfo & camera_info,
    int width, int height, int stride,
    double fx, double fy, double cx, double cy)
  {
    if (camera_info.width > 0 && camera_info.height > 0 &&
        (static_cast<int>(camera_info.width) != width ||
         static_cast<int>(camera_info.height) != height))
    {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 2000,
        "depth size %dx%d != camera_info %ux%u",
        width, height, camera_info.width, camera_info.height);
      return false;
    }

    if (cached_width_ == width && cached_height_ == height && cached_stride_ == stride &&
        cached_fx_ == fx && cached_fy_ == fy && cached_cx_ == cx && cached_cy_ == cy)
    {
      return true;
    }

    const int nx = (width + stride - 1) / stride;
    const int ny = (height + stride - 1) / stride;
    ray_x_.resize(static_cast<size_t>(nx));
    ray_y_.resize(static_cast<size_t>(ny));
    for (int ui = 0, u = 0; u < width; u += stride, ++ui) {
      ray_x_[static_cast<size_t>(ui)] =
        static_cast<float>((static_cast<double>(u) - cx) / fx);
    }
    for (int vi = 0, v = 0; v < height; v += stride, ++vi) {
      ray_y_[static_cast<size_t>(vi)] =
        static_cast<float>((static_cast<double>(v) - cy) / fy);
    }

    cached_width_ = width;
    cached_height_ = height;
    cached_stride_ = stride;
    cached_fx_ = fx;
    cached_fy_ = fy;
    cached_cx_ = cx;
    cached_cy_ = cy;
    return true;
  }

  std::string depth_topic_;
  std::string info_topic_;
  std::string output_topic_;
  std::string target_frame_;
  bool projection_enable_openmp_{false};
  int projection_openmp_threads_{4};
  std::vector<int> projection_openmp_cpus_;
  bool projection_enable_neon_{false};
  int projection_compute_cpu_{7};
  int omp_actual_threads_{0};
  std::vector<CloudT::Ptr> omp_thread_clouds_;
  std::vector<int> omp_cpu_by_thread_;
  std::vector<int> omp_affinity_errors_;
  bool omp_affinity_logged_{false};
  bool omp_affinity_valid_{true};
  bool compute_affinity_valid_{true};
  size_t successful_frames_{0};
  std::vector<double> queue_wait_samples_;
  std::vector<double> core_samples_;
  std::vector<double> voxel_samples_;
  std::vector<double> callback_samples_;

  std::mutex camera_info_mutex_;
  bool have_info_{false};
  sensor_msgs::msg::CameraInfo camera_info_;

  std::mutex job_mutex_;
  std::condition_variable job_cv_;
  std::optional<FrameJob> pending_job_;
  bool stop_requested_{false};  // 只在job_mutex_保护下访问
  std::thread compute_thread_;

  std::atomic<uint64_t> submitted_frames_{0};
  std::atomic<uint64_t> processed_frames_{0};
  std::atomic<uint64_t> dropped_pending_frames_{0};

  std::vector<float> ray_x_;
  std::vector<float> ray_y_;
  int cached_width_{0};
  int cached_height_{0};
  int cached_stride_{0};
  double cached_fx_{0.0};
  double cached_fy_{0.0};
  double cached_cx_{0.0};
  double cached_cy_{0.0};
  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr depth_sub_;
  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr info_sub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_pub_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<CloudWorkspaceNode>());
  rclcpp::shutdown();
  return 0;
}
