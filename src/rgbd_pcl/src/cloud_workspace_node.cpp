#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <algorithm>
#include <array>
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
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
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

using VoxelKey = uint64_t;

struct VoxelAccum
{
  VoxelKey key{0};
  double sum_x{0.0};
  double sum_y{0.0};
  double sum_z{0.0};
  uint32_t count{0};
  uint32_t generation{0};
};

struct FusedTiming
{
  double accumulate_ms{0.0};
  double merge_ms{0.0};
  double emit_ms{0.0};
  size_t sampled_pixels{0};
  size_t roi_valid_points{0};
  size_t local_voxel_entries{0};
  size_t output_voxels{0};
  uint32_t max_probe{0};
  uint32_t table_grows{0};
  uint32_t invalid_keys{0};
};

struct ParallelVoxelTiming
{
  double minmax_ms{0.0};
  double index_ms{0.0};
  double sort_ms{0.0};
  double group_ms{0.0};
  double centroid_ms{0.0};
  size_t input_points{0};
  size_t output_voxels{0};
};

struct VoxelIndexRecord
{
  uint32_t voxel_index{0};
  uint32_t point_index{0};
};

struct VoxelGroupRange
{
  uint32_t first{0};
  uint32_t last{0};
};

constexpr int32_t kVoxelIndexOffset = 1 << 20;
constexpr int32_t kVoxelIndexMin = -(1 << 20);
constexpr int32_t kVoxelIndexMax = (1 << 20) - 1;
constexpr size_t kVoxelTableMaxCapacity = 1u << 24;

inline uint64_t mixVoxelKey(VoxelKey key)
{
  key ^= key >> 30;
  key *= 0xbf58476d1ce4e5b9ULL;
  key ^= key >> 27;
  key *= 0x94d049bb133111ebULL;
  key ^= key >> 31;
  return key;
}

bool makeVoxelKey(float x, float y, float z, float inverse_leaf, VoxelKey & key)
{
  const double ix = std::floor(static_cast<double>(x) * static_cast<double>(inverse_leaf));
  const double iy = std::floor(static_cast<double>(y) * static_cast<double>(inverse_leaf));
  const double iz = std::floor(static_cast<double>(z) * static_cast<double>(inverse_leaf));
  if (ix < kVoxelIndexMin || ix > kVoxelIndexMax ||
      iy < kVoxelIndexMin || iy > kVoxelIndexMax ||
      iz < kVoxelIndexMin || iz > kVoxelIndexMax)
  {
    return false;
  }

  const uint64_t ux = static_cast<uint64_t>(static_cast<int32_t>(ix) + kVoxelIndexOffset);
  const uint64_t uy = static_cast<uint64_t>(static_cast<int32_t>(iy) + kVoxelIndexOffset);
  const uint64_t uz = static_cast<uint64_t>(static_cast<int32_t>(iz) + kVoxelIndexOffset);
  key = (ux << 42) | (uy << 21) | uz;
  return true;
}

class FlatVoxelTable
{
public:
  explicit FlatVoxelTable(size_t capacity)
  {
    if (capacity < 256 || (capacity & (capacity - 1)) != 0) {
      throw std::invalid_argument("FlatVoxelTable capacity must be a power of two >= 256");
    }
    slots_.assign(capacity, VoxelAccum{});
    occupied_slots_.reserve(capacity / 2);
    mask_ = capacity - 1;
  }

  void beginFrame()
  {
    ++generation_;
    if (generation_ == 0) {
      std::fill(slots_.begin(), slots_.end(), VoxelAccum{});
      generation_ = 1;
    }
    occupied_slots_.clear();
    max_probe_ = 0;
  }

  bool accumulate(VoxelKey key, float x, float y, float z)
  {
    return insertOrAdd(key, static_cast<double>(x), static_cast<double>(y), static_cast<double>(z), 1U);
  }

  bool merge(VoxelKey key, const VoxelAccum & src)
  {
    return insertOrAdd(key, src.sum_x, src.sum_y, src.sum_z, src.count);
  }

  size_t size() const
  {
    return occupied_slots_.size();
  }

  const std::vector<uint32_t> & occupiedSlots() const
  {
    return occupied_slots_;
  }

  const VoxelAccum & slot(uint32_t index) const
  {
    return slots_[index];
  }

  uint32_t maxProbe() const
  {
    return max_probe_;
  }

  uint32_t growCount() const
  {
    return grow_count_;
  }

private:
  bool insertOrAdd(VoxelKey key, double sum_x, double sum_y, double sum_z, uint32_t count)
  {
    if (occupied_slots_.size() * 10U >= slots_.size() * 7U) {
      if (!grow()) {
        return false;
      }
    }
    return insertNoGrow(key, sum_x, sum_y, sum_z, count);
  }

  bool insertNoGrow(VoxelKey key, double sum_x, double sum_y, double sum_z, uint32_t count)
  {
    uint32_t idx = static_cast<uint32_t>(mixVoxelKey(key) & mask_);
    uint32_t probe = 0;
    while (probe < slots_.size()) {
      VoxelAccum & s = slots_[idx];
      if (s.generation != generation_) {
        s.key = key;
        s.sum_x = sum_x;
        s.sum_y = sum_y;
        s.sum_z = sum_z;
        s.count = count;
        s.generation = generation_;
        occupied_slots_.push_back(idx);
        if (probe > max_probe_) {
          max_probe_ = probe;
        }
        return true;
      }
      if (s.key == key) {
        s.sum_x += sum_x;
        s.sum_y += sum_y;
        s.sum_z += sum_z;
        s.count += count;
        if (probe > max_probe_) {
          max_probe_ = probe;
        }
        return true;
      }
      idx = (idx + 1U) & static_cast<uint32_t>(mask_);
      ++probe;
    }
    return false;
  }

  bool grow()
  {
    const size_t new_capacity = slots_.size() * 2U;
    if (new_capacity > kVoxelTableMaxCapacity) {
      return false;
    }

    std::vector<VoxelAccum> old_slots = std::move(slots_);
    std::vector<uint32_t> old_occupied = std::move(occupied_slots_);
    slots_.assign(new_capacity, VoxelAccum{});
    occupied_slots_.clear();
    occupied_slots_.reserve(old_occupied.size());
    mask_ = new_capacity - 1;
    ++grow_count_;

    for (const uint32_t old_idx : old_occupied) {
      const VoxelAccum & src = old_slots[old_idx];
      if (src.generation != generation_ || src.count == 0U) {
        continue;
      }
      if (!insertNoGrow(src.key, src.sum_x, src.sum_y, src.sum_z, src.count)) {
        return false;
      }
    }
    return true;
  }

  std::vector<VoxelAccum> slots_;
  std::vector<uint32_t> occupied_slots_;
  size_t mask_{0};
  uint32_t generation_{0};
  uint32_t max_probe_{0};
  uint32_t grow_count_{0};
};

inline bool tryProjectRoiPoint(
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
  double r20, double r21, double r22, double tz,
  float & x_target,
  float & y_target,
  float & z_target)
{
  if (!std::isfinite(z) || z < zmin || z > zmax) {
    return false;
  }

  const double x_camera = static_cast<double>(ray_x[ui]) * z;
  const double y_camera = static_cast<double>(ray_y[vi]) * z;
  const double z_camera = static_cast<double>(z);

  x_target = static_cast<float>(r00 * x_camera + r01 * y_camera + r02 * z_camera + tx);
  y_target = static_cast<float>(r10 * x_camera + r11 * y_camera + r12 * z_camera + ty);
  z_target = static_cast<float>(r20 * x_camera + r21 * y_camera + r22 * z_camera + tz);

  if (x_target < xmin || x_target > xmax || z_target < zmin_roi || z_target > zmax_roi) {
    return false;
  }
  return true;
}

inline int projectAndAccumulate(
  FlatVoxelTable & table,
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
  double r20, double r21, double r22, double tz,
  float inverse_leaf)
{
  float x_target = 0.0f;
  float y_target = 0.0f;
  float z_target = 0.0f;
  if (!tryProjectRoiPoint(
        z, ui, vi, ray_x, ray_y, zmin, zmax, xmin, xmax, zmin_roi, zmax_roi,
        r00, r01, r02, tx, r10, r11, r12, ty, r20, r21, r22, tz,
        x_target, y_target, z_target))
  {
    return 0;
  }

  VoxelKey key = 0;
  if (!makeVoxelKey(x_target, y_target, z_target, inverse_leaf, key)) {
    return 1;
  }
  if (!table.accumulate(key, x_target, y_target, z_target)) {
    return 2;
  }
  return 3;
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

struct PreciseNeonWorkspaceParams
{
  float zmin;
  float zmax;
  float xmin;
  float xmax;
  float zmin_roi;
  float zmax_roi;
  float voxel_leaf;

  double r00, r01, r02, tx;
  double r10, r11, r12, ty;
  double r20, r21, r22, tz;
};

#if RGBD_PCL_HAS_NEON
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
  const NeonWorkspaceParams & params,
  const PreciseNeonWorkspaceParams * precise_params = nullptr)
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

  uint32x4_t voxel_boundary_mask = vdupq_n_u32(0U);
  if (precise_params != nullptr && precise_params->voxel_leaf > 0.0f) {
    const float inverse_leaf = 1.0f / precise_params->voxel_leaf;
    constexpr float kPreciseVoxelBoundaryBand = 0.0001f;
    const float threshold_in_voxels = kPreciseVoxelBoundaryBand * inverse_leaf;
    const float32x4_t x_scaled = vmulq_n_f32(x_target, inverse_leaf);
    const float32x4_t y_scaled = vmulq_n_f32(y_target, inverse_leaf);
    const float32x4_t z_scaled = vmulq_n_f32(z_target, inverse_leaf);
    voxel_boundary_mask = vorrq_u32(
      vcleq_f32(vabdq_f32(x_scaled, vrndnq_f32(x_scaled)),
        vdupq_n_f32(threshold_in_voxels)),
      vorrq_u32(
        vcleq_f32(vabdq_f32(y_scaled, vrndnq_f32(y_scaled)),
          vdupq_n_f32(threshold_in_voxels)),
        vcleq_f32(vabdq_f32(z_scaled, vrndnq_f32(z_scaled)),
          vdupq_n_f32(threshold_in_voxels))));
  }

  alignas(16) float xs[4];
  alignas(16) float ys[4];
  alignas(16) float zs[4];
  alignas(16) float depths[4];
  alignas(16) uint32_t valid[4];
  alignas(16) uint32_t near_voxel_boundary[4];
  vst1q_f32(xs, x_target);
  vst1q_f32(ys, y_target);
  vst1q_f32(zs, z_target);
  vst1q_f32(depths, z);
  vst1q_u32(valid, valid_mask);
  vst1q_u32(near_voxel_boundary, voxel_boundary_mask);

  for (size_t lane = 0; lane < 4; ++lane) {
    constexpr float kPreciseBoundaryBand = 0.002f;
    const bool needs_precise_recheck =
      std::abs(xs[lane] - params.xmin) <= kPreciseBoundaryBand ||
      std::abs(xs[lane] - params.xmax) <= kPreciseBoundaryBand ||
      std::abs(zs[lane] - params.zmin_roi) <= kPreciseBoundaryBand ||
      std::abs(zs[lane] - params.zmax_roi) <= kPreciseBoundaryBand ||
      near_voxel_boundary[lane] != 0U;
    if (precise_params != nullptr && needs_precise_recheck) {
      const float depth = depths[lane];
      if (!std::isfinite(depth) || depth < params.zmin || depth > params.zmax) {
        continue;
      }
      const double x_camera = static_cast<double>(ray_x[ui + lane]) * depth;
      const double y_camera = static_cast<double>(ray_y[vi]) * depth;
      const double z_camera = static_cast<double>(depth);
      const float exact_x = static_cast<float>(
        precise_params->r00 * x_camera + precise_params->r01 * y_camera +
        precise_params->r02 * z_camera + precise_params->tx);
      const float exact_y = static_cast<float>(
        precise_params->r10 * x_camera + precise_params->r11 * y_camera +
        precise_params->r12 * z_camera + precise_params->ty);
      const float exact_z = static_cast<float>(
        precise_params->r20 * x_camera + precise_params->r21 * y_camera +
        precise_params->r22 * z_camera + precise_params->tz);
      if (exact_x < precise_params->xmin || exact_x > precise_params->xmax ||
        exact_z < precise_params->zmin_roi || exact_z > precise_params->zmax_roi)
      {
        continue;
      }
      PointT p;
      p.x = exact_x;
      p.y = exact_y;
      p.z = exact_z;
      cloud.push_back(p);
      continue;
    }
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
  const NeonWorkspaceParams & params,
  const PreciseNeonWorkspaceParams * precise_params = nullptr)
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
  appendTransformedPointsNeon4(
    cloud, z, ui, vi, ray_x, ray_y, params, precise_params);
}

inline void tryAppendPointsNeon4(
  CloudT & cloud,
  const float * row,
  const size_t ui,
  const size_t vi,
  const int stride,
  const float * ray_x,
  const float * ray_y,
  const NeonWorkspaceParams & params,
  const PreciseNeonWorkspaceParams * precise_params = nullptr)
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
  appendTransformedPointsNeon4(
    cloud, z, ui, vi, ray_x, ray_y, params, precise_params);
}

inline void appendTransformedPointsNeon2Precise(
  CloudT & cloud,
  const float32x2_t z_float,
  const size_t ui,
  const size_t vi,
  const float * ray_x,
  const float * ray_y,
  const PreciseNeonWorkspaceParams & params)
{
  const float64x2_t z = vcvt_f64_f32(z_float);
  const float64x2_t x_camera = vmulq_f64(
    vcvt_f64_f32(vld1_f32(ray_x + ui)), z);
  const float64x2_t y_camera = vmulq_f64(
    vdupq_n_f64(static_cast<double>(ray_y[vi])), z);

  float64x2_t x_target = vmulq_n_f64(x_camera, params.r00);
  x_target = vaddq_f64(
    x_target, vmulq_n_f64(y_camera, params.r01));
  x_target = vaddq_f64(
    x_target, vmulq_n_f64(z, params.r02));
  x_target = vaddq_f64(x_target, vdupq_n_f64(params.tx));

  float64x2_t y_target = vmulq_n_f64(x_camera, params.r10);
  y_target = vaddq_f64(
    y_target, vmulq_n_f64(y_camera, params.r11));
  y_target = vaddq_f64(
    y_target, vmulq_n_f64(z, params.r12));
  y_target = vaddq_f64(y_target, vdupq_n_f64(params.ty));

  float64x2_t z_target = vmulq_n_f64(x_camera, params.r20);
  z_target = vaddq_f64(
    z_target, vmulq_n_f64(y_camera, params.r21));
  z_target = vaddq_f64(
    z_target, vmulq_n_f64(z, params.r22));
  z_target = vaddq_f64(z_target, vdupq_n_f64(params.tz));

  alignas(16) float depths[2];
  alignas(16) double xs[2];
  alignas(16) double ys[2];
  alignas(16) double zs[2];
  vst1_f32(depths, z_float);
  vst1q_f64(xs, x_target);
  vst1q_f64(ys, y_target);
  vst1q_f64(zs, z_target);

  for (size_t lane = 0; lane < 2U; ++lane) {
    const float depth = depths[lane];
    if (!std::isfinite(depth) || depth < params.zmin || depth > params.zmax) {
      continue;
    }
    const float x = static_cast<float>(xs[lane]);
    const float y = static_cast<float>(ys[lane]);
    const float target_z = static_cast<float>(zs[lane]);
    if (x < params.xmin || x > params.xmax ||
      target_z < params.zmin_roi || target_z > params.zmax_roi)
    {
      continue;
    }
    PointT p;
    p.x = x;
    p.y = y;
    p.z = target_z;
    cloud.push_back(p);
  }
}

inline void tryAppendPointsNeon2Precise(
  CloudT & cloud,
  const uint16_t * row,
  const size_t ui,
  const size_t vi,
  const int stride,
  const float * ray_x,
  const float * ray_y,
  const PreciseNeonWorkspaceParams & params)
{
  const size_t u = ui * static_cast<size_t>(stride);
  const size_t step = static_cast<size_t>(stride);
  const float depths[2] = {
    static_cast<float>(row[u]) * 0.001f,
    static_cast<float>(row[u + step]) * 0.001f
  };
  appendTransformedPointsNeon2Precise(
    cloud, vld1_f32(depths), ui, vi, ray_x, ray_y, params);
}

inline void tryAppendPointsNeon2Precise(
  CloudT & cloud,
  const float * row,
  const size_t ui,
  const size_t vi,
  const int stride,
  const float * ray_x,
  const float * ray_y,
  const PreciseNeonWorkspaceParams & params)
{
  const size_t u = ui * static_cast<size_t>(stride);
  const size_t step = static_cast<size_t>(stride);
  const float depths[2] = {row[u], row[u + step]};
  appendTransformedPointsNeon2Precise(
    cloud, vld1_f32(depths), ui, vi, ray_x, ray_y, params);
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
    declare_parameter<bool>("projection_fuse_voxel", false);
    declare_parameter<bool>("projection_fused_voxel_openmp", false);
    declare_parameter<int>("projection_voxel_chunk_rows", 8);
    declare_parameter<int>("projection_voxel_table_capacity", 16384);
    declare_parameter<bool>("projection_fused_voxel_verify", false);

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
    projection_fuse_voxel_ = get_parameter("projection_fuse_voxel").as_bool();
    projection_fused_voxel_openmp_ = get_parameter("projection_fused_voxel_openmp").as_bool();
    projection_voxel_chunk_rows_ = get_parameter("projection_voxel_chunk_rows").as_int();
    projection_voxel_table_capacity_ = get_parameter("projection_voxel_table_capacity").as_int();
    projection_fused_voxel_verify_ = get_parameter("projection_fused_voxel_verify").as_bool();

    if (projection_enable_openmp_ && projection_enable_neon_ && !projection_fuse_voxel_) {
      throw std::invalid_argument(
        "V4 does not support projection OpenMP and NEON at the same time");
    }
    if (projection_fuse_voxel_ && projection_enable_openmp_) {
      throw std::invalid_argument(
        "fused voxel and V3 OpenMP cannot be enabled together");
    }
    if (projection_fused_voxel_openmp_ && !projection_fuse_voxel_) {
      throw std::invalid_argument(
        "projection_fused_voxel_openmp requires projection_fuse_voxel=true");
    }
    if (projection_fuse_voxel_ && !projection_fused_voxel_openmp_) {
      throw std::invalid_argument(
        "V5 PCL-equivalent voxel requires projection_fused_voxel_openmp=true");
    }
    if (projection_voxel_chunk_rows_ < 1 || projection_voxel_chunk_rows_ > 128) {
      throw std::invalid_argument("projection_voxel_chunk_rows must be in [1, 128]");
    }
    if (projection_voxel_table_capacity_ < 256 ||
        (projection_voxel_table_capacity_ & (projection_voxel_table_capacity_ - 1)) != 0 ||
        static_cast<size_t>(projection_voxel_table_capacity_) > kVoxelTableMaxCapacity)
    {
      throw std::invalid_argument(
        "projection_voxel_table_capacity must be a power of two in [256, 16777216]");
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
    const bool uses_openmp_team = projection_enable_openmp_ || projection_fused_voxel_openmp_;
    if (uses_openmp_team &&
      projection_openmp_cpus_.size() != static_cast<size_t>(projection_openmp_threads_))
    {
      throw std::invalid_argument(
              "projection_openmp_cpus count must match projection_openmp_threads");
    }
    if (projection_compute_cpu_ < 0 || projection_compute_cpu_ >= CPU_SETSIZE) {
      throw std::invalid_argument(
        "projection_compute_cpu must be in [0, " + std::to_string(CPU_SETSIZE - 1) + "]");
    }
    if (uses_openmp_team) {
      projection_compute_cpu_ = projection_openmp_cpus_.front();
    }

    // omp_set_dynamic(0);
    // omp_set_nested(0);

    core_samples_.reserve(kMeasureFrames);
    voxel_samples_.reserve(kMeasureFrames);
    callback_samples_.reserve(kMeasureFrames);
    queue_wait_samples_.reserve(kMeasureFrames);
    compute_total_samples_.reserve(kMeasureFrames);
    fused_accumulate_samples_.reserve(kMeasureFrames);
    fused_merge_samples_.reserve(kMeasureFrames);
    fused_emit_samples_.reserve(kMeasureFrames);
    fused_sampled_pixels_samples_.reserve(kMeasureFrames);
    fused_roi_valid_samples_.reserve(kMeasureFrames);
    fused_local_entries_samples_.reserve(kMeasureFrames);
    fused_output_voxels_samples_.reserve(kMeasureFrames);
    fused_max_probe_samples_.reserve(kMeasureFrames);
    v5_minmax_samples_.reserve(kMeasureFrames);
    v5_index_samples_.reserve(kMeasureFrames);
    v5_sort_samples_.reserve(kMeasureFrames);
    v5_group_samples_.reserve(kMeasureFrames);
    v5_centroid_samples_.reserve(kMeasureFrames);

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
      "cloud_workspace_node depth=%s info=%s -> %s (%s) openmp_team=%s "
      "requested_threads=%d neon=%s compute_cpu=%d v5_pcl_equiv=%s verify=%s",
      depth_topic_.c_str(), info_topic_.c_str(), output_topic_.c_str(),
      target_frame_.c_str(),
      uses_openmp_team ? "true" : "false",
      projection_openmp_threads_,
      projection_enable_neon_ ? "true" : "false",
      projection_compute_cpu_,
      projection_fuse_voxel_ ? "true" : "false",
      projection_fused_voxel_verify_ ? "true" : "false");
    if (projection_enable_openmp_ || projection_fused_voxel_openmp_) {
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

    if (core_samples_.size() >= kMinDumpFrames ||
        compute_total_samples_.size() >= kMinDumpFrames)
    {
      logPerf();
    }
  }

private:
  static constexpr size_t kWarmupFrames = 30;
  static constexpr size_t kMeasureFrames = 1000;
  static constexpr size_t kMinDumpFrames = 200;
  static constexpr int kMaxOpenMPThreads = 16;
  static constexpr unsigned kRadixBits = 11U;
  static constexpr unsigned kRadixPassCount = 3U;
  static constexpr size_t kRadixBucketCount = 1U << kRadixBits;
  static constexpr uint32_t kRadixBucketMask =
    static_cast<uint32_t>(kRadixBucketCount - 1U);

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
    if (projection_fuse_voxel_) {
      RCLCPP_INFO(
        get_logger(),
        "PERF V5_PCL_EQUIV_OMP neon=%s requested_threads=%d actual_threads=%d "
        "affinity_ok=%s samples=%zu core_p95=%.3f ms minmax_p95=%.3f ms "
        "index_p95=%.3f ms sort_p95=%.3f ms group_p95=%.3f ms "
        "centroid_p95=%.3f ms voxel_p95=%.3f ms compute_total_p95=%.3f ms "
        "callback_p95=%.3f ms queue_wait_p95=%.3f ms input_pts_p95=%.0f "
        "output_voxels_p95=%.0f submitted=%lu processed=%lu dropped_pending=%lu",
        projection_enable_neon_ ? "true" : "false",
        projection_openmp_threads_,
        omp_actual_threads_,
        affinity_ok,
        compute_total_samples_.size(),
        calculateP95(core_samples_),
        calculateP95(v5_minmax_samples_),
        calculateP95(v5_index_samples_),
        calculateP95(v5_sort_samples_),
        calculateP95(v5_group_samples_),
        calculateP95(v5_centroid_samples_),
        calculateP95(voxel_samples_),
        calculateP95(compute_total_samples_),
        calculateP95(callback_samples_),
        calculateP95(queue_wait_samples_),
        calculateP95(fused_roi_valid_samples_),
        calculateP95(fused_output_voxels_samples_),
        static_cast<unsigned long>(submitted_frames_.load()),
        static_cast<unsigned long>(processed_frames_.load()),
        static_cast<unsigned long>(dropped_pending_frames_.load()));
      if (!compute_affinity_valid_ || !omp_affinity_valid_) {
        RCLCPP_WARN(
          get_logger(),
          "PERF V5_PCL_EQUIV_OMP result is invalid because compute/OpenMP affinity failed");
      }
    } else if (projection_enable_openmp_) {
      RCLCPP_INFO(
        get_logger(),
        "PERF V3_OMP requested_threads=%d actual_threads=%d affinity_ok=%s samples=%zu "
        "queue_wait_p95=%.3f ms core_p95=%.3f ms voxel_p95=%.3f ms compute_total_p95=%.3f ms callback_p95=%.3f ms "
        "submitted=%lu processed=%lu dropped_pending=%lu",
        projection_openmp_threads_,
        omp_actual_threads_,
        affinity_ok,
        core_samples_.size(),
        calculateP95(queue_wait_samples_),
        calculateP95(core_samples_),
        calculateP95(voxel_samples_),
        calculateP95(compute_total_samples_),
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
        "queue_wait_p95=%.3f ms core_p95=%.3f ms voxel_p95=%.3f ms compute_total_p95=%.3f ms callback_p95=%.3f ms "
        "submitted=%lu processed=%lu dropped_pending=%lu",
        affinity_ok,
        core_samples_.size(),
        calculateP95(queue_wait_samples_),
        calculateP95(core_samples_),
        calculateP95(voxel_samples_),
        calculateP95(compute_total_samples_),
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
        "queue_wait_p95=%.3f ms core_p95=%.3f ms voxel_p95=%.3f ms compute_total_p95=%.3f ms callback_p95=%.3f ms "
        "submitted=%lu processed=%lu dropped_pending=%lu",
        affinity_ok,
        core_samples_.size(),
        calculateP95(queue_wait_samples_),
        calculateP95(core_samples_),
        calculateP95(voxel_samples_),
        calculateP95(compute_total_samples_),
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

  void recordTiming(
    double queue_wait_ms,
    double core_ms,
    double voxel_ms,
    double compute_total_ms,
    double callback_ms,
    const FusedTiming * fused,
    const ParallelVoxelTiming * parallel_voxel = nullptr)
  {
    ++successful_frames_;
    if (successful_frames_ <= kWarmupFrames) {
      return;
    }

    queue_wait_samples_.push_back(queue_wait_ms);
    core_samples_.push_back(core_ms);
    voxel_samples_.push_back(voxel_ms);
    compute_total_samples_.push_back(compute_total_ms);
    callback_samples_.push_back(callback_ms);
    if (fused != nullptr) {
      fused_accumulate_samples_.push_back(fused->accumulate_ms);
      fused_merge_samples_.push_back(fused->merge_ms);
      fused_emit_samples_.push_back(fused->emit_ms);
      fused_sampled_pixels_samples_.push_back(static_cast<double>(fused->sampled_pixels));
      fused_roi_valid_samples_.push_back(static_cast<double>(fused->roi_valid_points));
      fused_local_entries_samples_.push_back(static_cast<double>(fused->local_voxel_entries));
      fused_output_voxels_samples_.push_back(static_cast<double>(fused->output_voxels));
      fused_max_probe_samples_.push_back(static_cast<double>(fused->max_probe));
      fused_table_grows_ += fused->table_grows;
    }
    if (parallel_voxel != nullptr) {
      v5_minmax_samples_.push_back(parallel_voxel->minmax_ms);
      v5_index_samples_.push_back(parallel_voxel->index_ms);
      v5_sort_samples_.push_back(parallel_voxel->sort_ms);
      v5_group_samples_.push_back(parallel_voxel->group_ms);
      v5_centroid_samples_.push_back(parallel_voxel->centroid_ms);
      fused_roi_valid_samples_.push_back(
        static_cast<double>(parallel_voxel->input_points));
      fused_output_voxels_samples_.push_back(
        static_cast<double>(parallel_voxel->output_voxels));
    }

    if (compute_total_samples_.size() < kMeasureFrames) {
      return;
    }

    logPerf();
    queue_wait_samples_.clear();
    core_samples_.clear();
    voxel_samples_.clear();
    compute_total_samples_.clear();
    callback_samples_.clear();
    fused_accumulate_samples_.clear();
    fused_merge_samples_.clear();
    fused_emit_samples_.clear();
    fused_sampled_pixels_samples_.clear();
    fused_roi_valid_samples_.clear();
    fused_local_entries_samples_.clear();
    fused_output_voxels_samples_.clear();
    fused_max_probe_samples_.clear();
    v5_minmax_samples_.clear();
    v5_index_samples_.clear();
    v5_sort_samples_.clear();
    v5_group_samples_.clear();
    v5_centroid_samples_.clear();
    fused_table_grows_ = 0;
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
    const int coordinator_cpu = (projection_enable_openmp_ || projection_fused_voxel_openmp_)
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

    const float voxel_leaf = static_cast<float>(get_parameter("voxel_leaf_m").as_double());
    CloudT transient_cloud_voxel;
    CloudT & cloud_voxel = projection_fuse_voxel_ ?
      v5_cloud_voxel_buffer_ : transient_cloud_voxel;
    bool ok = false;
    double core_ms = 0.0;
    double voxel_ms = 0.0;
    const FusedTiming * fused_ptr = nullptr;
    ParallelVoxelTiming parallel_voxel_timing;
    const ParallelVoxelTiming * parallel_voxel_ptr = nullptr;
    CloudT * v5_cloud_roi = nullptr;

    const auto compute_total_begin = std::chrono::steady_clock::now();
    if (projection_fuse_voxel_) {
      v5_cloud_roi = &v5_cloud_roi_buffer_;
      const auto core_begin = std::chrono::steady_clock::now();
      ok = depthToWorkspaceOpenMP(
        *job.depth, job.camera_info, transform, *v5_cloud_roi, projection_enable_neon_);
      core_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - core_begin).count();
      if (ok) {
        const auto voxel_begin = std::chrono::steady_clock::now();
        ok = voxelGridEquivalentOpenMP(
          *v5_cloud_roi, voxel_leaf, cloud_voxel,
          false, parallel_voxel_timing);
        voxel_ms = std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - voxel_begin).count();
        parallel_voxel_ptr = &parallel_voxel_timing;
      }
    } else {
      CloudT::Ptr cloud_roi(new CloudT);
      const auto core_begin = std::chrono::steady_clock::now();
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
      const auto core_end = std::chrono::steady_clock::now();
      core_ms = std::chrono::duration<double, std::milli>(core_end - core_begin).count();
      if (ok) {
        const auto voxel_begin = std::chrono::steady_clock::now();
        pcl::VoxelGrid<PointT> vg;
        vg.setInputCloud(cloud_roi);
        vg.setLeafSize(voxel_leaf, voxel_leaf, voxel_leaf);
        vg.filter(cloud_voxel);
        const auto voxel_end = std::chrono::steady_clock::now();
        voxel_ms = std::chrono::duration<double, std::milli>(voxel_end - voxel_begin).count();
      }
    }
    const auto compute_total_end = std::chrono::steady_clock::now();
    const double compute_total_ms =
      std::chrono::duration<double, std::milli>(compute_total_end - compute_total_begin).count();
    if (!ok) {
      return false;
    }

    if (projection_fuse_voxel_ && projection_fused_voxel_verify_ && v5_cloud_roi) {
      verifyParallelVoxelAgainstPcl(
        *job.depth, job.camera_info, transform, voxel_leaf, cloud_voxel);
    }

    sensor_msgs::msg::PointCloud2 out;
    pcl::toROSMsg(cloud_voxel, out);
    out.header.stamp = job.depth->header.stamp;
    out.header.frame_id = target_frame_;
    cloud_pub_->publish(out);

    const auto callback_end = std::chrono::steady_clock::now();
    const double callback_ms =
      std::chrono::duration<double, std::milli>(callback_end - job.pipeline_begin).count();
    if (!(projection_fuse_voxel_ && projection_fused_voxel_verify_)) {
      recordTiming(
        queue_wait_ms, core_ms, voxel_ms, compute_total_ms, callback_ms,
        fused_ptr, parallel_voxel_ptr);
    }
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

    // Each worker owns one contiguous row range, so reserving one team share is
    // sufficient. The previous full-frame reserve per worker multiplied memory
    // footprint by the thread count and put avoidable pressure on the caches.
    const size_t total_candidates = sampled_rows * sampled_cols;
    const size_t cap =
      (total_candidates + static_cast<size_t>(requested_threads) - 1U) /
      static_cast<size_t>(requested_threads);
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
    omp_cloud_offsets_.assign(n + 1U, 0U);
  }

  void prepareParallelVoxelBuffers(size_t point_count, int requested_threads)
  {
    voxel_indices_a_.resize(point_count);
    voxel_indices_b_.resize(point_count);
    voxel_radix_counts_.resize(
      static_cast<size_t>(requested_threads) * kRadixBucketCount);
    voxel_radix_offsets_.resize(
      static_cast<size_t>(requested_threads) * kRadixBucketCount);
    voxel_groups_.clear();
    if (voxel_groups_.capacity() < point_count) {
      voxel_groups_.reserve(point_count);
    }
  }

  bool voxelGridEquivalentOpenMP(
    const CloudT & input,
    float voxel_leaf,
    CloudT & output,
    bool use_neon,
    ParallelVoxelTiming & timing)
  {
    timing = ParallelVoxelTiming{};
    timing.input_points = input.size();
    if (input.empty() || !(voxel_leaf > 0.0f) || !std::isfinite(voxel_leaf)) {
      output.clear();
      return false;
    }
    if (input.size() > static_cast<size_t>(std::numeric_limits<uint32_t>::max())) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 2000,
        "Parallel VoxelGrid input exceeds uint32 index range");
      output = input;
      return true;
    }

    const int requested_threads = projection_openmp_threads_;
    prepareParallelVoxelBuffers(input.size(), requested_threads);

    const size_t point_count = input.size();
    const float inverse_leaf = 1.0f / voxel_leaf;
    const PointT * input_points = input.points.data();
    VoxelIndexRecord * sort_src = voxel_indices_a_.data();
    VoxelIndexRecord * sort_dst = voxel_indices_b_.data();
    uint32_t * radix_counts = voxel_radix_counts_.data();
    uint32_t * radix_offsets = voxel_radix_offsets_.data();
    const int * target_cpus = projection_openmp_cpus_.data();

    std::array<float, kMaxOpenMPThreads> local_min_x{};
    std::array<float, kMaxOpenMPThreads> local_min_y{};
    std::array<float, kMaxOpenMPThreads> local_min_z{};
    std::array<float, kMaxOpenMPThreads> local_max_x{};
    std::array<float, kMaxOpenMPThreads> local_max_y{};
    std::array<float, kMaxOpenMPThreads> local_max_z{};

    float min_x = 0.0f, min_y = 0.0f, min_z = 0.0f;
    float max_x = 0.0f, max_y = 0.0f, max_z = 0.0f;
    int min_b_x = 0, min_b_y = 0, min_b_z = 0;
    int div_x = 0, div_y = 0, div_xy = 0;
    bool index_overflow = false;
    int actual_threads = 0;
    auto phase_begin = std::chrono::steady_clock::now();

#pragma omp parallel num_threads(requested_threads) default(shared)
    {
      const int tid = omp_get_thread_num();
      const int team_size = omp_get_num_threads();
#pragma omp single
      {
        actual_threads = team_size;
      }

      cpu_set_t compute_affinity;
      CPU_ZERO(&compute_affinity);
      CPU_SET(target_cpus[tid], &compute_affinity);
      static thread_local int bound_voxel_cpu = -1;
      if (bound_voxel_cpu != target_cpus[tid]) {
        if (sched_setaffinity(0, sizeof(compute_affinity), &compute_affinity) == 0) {
          bound_voxel_cpu = target_cpus[tid];
        }
      }

      const size_t begin = point_count * static_cast<size_t>(tid) /
        static_cast<size_t>(team_size);
      const size_t end = point_count * static_cast<size_t>(tid + 1) /
        static_cast<size_t>(team_size);

      float thread_min_x = std::numeric_limits<float>::infinity();
      float thread_min_y = std::numeric_limits<float>::infinity();
      float thread_min_z = std::numeric_limits<float>::infinity();
      float thread_max_x = -std::numeric_limits<float>::infinity();
      float thread_max_y = -std::numeric_limits<float>::infinity();
      float thread_max_z = -std::numeric_limits<float>::infinity();
      for (size_t i = begin; i < end; ++i) {
        const PointT & p = input_points[i];
        thread_min_x = std::min(thread_min_x, p.x);
        thread_min_y = std::min(thread_min_y, p.y);
        thread_min_z = std::min(thread_min_z, p.z);
        thread_max_x = std::max(thread_max_x, p.x);
        thread_max_y = std::max(thread_max_y, p.y);
        thread_max_z = std::max(thread_max_z, p.z);
      }
      local_min_x[static_cast<size_t>(tid)] = thread_min_x;
      local_min_y[static_cast<size_t>(tid)] = thread_min_y;
      local_min_z[static_cast<size_t>(tid)] = thread_min_z;
      local_max_x[static_cast<size_t>(tid)] = thread_max_x;
      local_max_y[static_cast<size_t>(tid)] = thread_max_y;
      local_max_z[static_cast<size_t>(tid)] = thread_max_z;

#pragma omp barrier
#pragma omp single
      {
        min_x = local_min_x[0];
        min_y = local_min_y[0];
        min_z = local_min_z[0];
        max_x = local_max_x[0];
        max_y = local_max_y[0];
        max_z = local_max_z[0];
        for (int worker = 1; worker < actual_threads; ++worker) {
          min_x = std::min(min_x, local_min_x[static_cast<size_t>(worker)]);
          min_y = std::min(min_y, local_min_y[static_cast<size_t>(worker)]);
          min_z = std::min(min_z, local_min_z[static_cast<size_t>(worker)]);
          max_x = std::max(max_x, local_max_x[static_cast<size_t>(worker)]);
          max_y = std::max(max_y, local_max_y[static_cast<size_t>(worker)]);
          max_z = std::max(max_z, local_max_z[static_cast<size_t>(worker)]);
        }

        const int64_t dx = static_cast<int64_t>((max_x - min_x) * inverse_leaf) + 1;
        const int64_t dy = static_cast<int64_t>((max_y - min_y) * inverse_leaf) + 1;
        const int64_t dz = static_cast<int64_t>((max_z - min_z) * inverse_leaf) + 1;
        index_overflow = dx * dy * dz >
          static_cast<int64_t>(std::numeric_limits<int32_t>::max());
        if (!index_overflow) {
          min_b_x = static_cast<int>(std::floor(min_x * inverse_leaf));
          min_b_y = static_cast<int>(std::floor(min_y * inverse_leaf));
          min_b_z = static_cast<int>(std::floor(min_z * inverse_leaf));
          const int max_b_x = static_cast<int>(std::floor(max_x * inverse_leaf));
          const int max_b_y = static_cast<int>(std::floor(max_y * inverse_leaf));
          div_x = max_b_x - min_b_x + 1;
          div_y = max_b_y - min_b_y + 1;
          div_xy = div_x * div_y;
        }
        timing.minmax_ms = std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - phase_begin).count();
        phase_begin = std::chrono::steady_clock::now();
      }
#pragma omp barrier

      if (!index_overflow) {
        size_t i = begin;
#if RGBD_PCL_HAS_NEON
        if (use_neon) {
          const int32x4_t min_x_vec = vdupq_n_s32(min_b_x);
          const int32x4_t min_y_vec = vdupq_n_s32(min_b_y);
          const int32x4_t min_z_vec = vdupq_n_s32(min_b_z);
          for (; i + 3U < end; i += 4U) {
            const float32x4x4_t xyzw = vld4q_f32(
              reinterpret_cast<const float *>(input_points + i));
            const int32x4_t ix = vsubq_s32(
              vcvtq_s32_f32(vrndmq_f32(vmulq_n_f32(xyzw.val[0], inverse_leaf))),
              min_x_vec);
            const int32x4_t iy = vsubq_s32(
              vcvtq_s32_f32(vrndmq_f32(vmulq_n_f32(xyzw.val[1], inverse_leaf))),
              min_y_vec);
            const int32x4_t iz = vsubq_s32(
              vcvtq_s32_f32(vrndmq_f32(vmulq_n_f32(xyzw.val[2], inverse_leaf))),
              min_z_vec);
            int32x4_t linear = vmlaq_n_s32(ix, iy, div_x);
            linear = vmlaq_n_s32(linear, iz, div_xy);
            alignas(16) uint32_t keys[4];
            vst1q_u32(keys, vreinterpretq_u32_s32(linear));
            for (size_t lane = 0; lane < 4U; ++lane) {
              sort_src[i + lane] = VoxelIndexRecord{
                keys[lane], static_cast<uint32_t>(i + lane)};
            }
          }
        }
#endif
        for (; i < end; ++i) {
          const PointT & p = input_points[i];
          const int ix = static_cast<int>(std::floor(p.x * inverse_leaf)) - min_b_x;
          const int iy = static_cast<int>(std::floor(p.y * inverse_leaf)) - min_b_y;
          const int iz = static_cast<int>(std::floor(p.z * inverse_leaf)) - min_b_z;
          const int linear = ix + iy * div_x + iz * div_xy;
          sort_src[i] = VoxelIndexRecord{
            static_cast<uint32_t>(linear), static_cast<uint32_t>(i)};
        }
      }

#pragma omp barrier
#pragma omp single
      {
        timing.index_ms = std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - phase_begin).count();
        phase_begin = std::chrono::steady_clock::now();
      }
#pragma omp barrier

      if (!index_overflow) {
        for (unsigned pass = 0; pass < kRadixPassCount; ++pass) {
          uint32_t * thread_counts = radix_counts +
            static_cast<size_t>(tid) * kRadixBucketCount;
          std::fill(thread_counts, thread_counts + kRadixBucketCount, 0U);
          const unsigned shift = pass * kRadixBits;
          for (size_t record = begin; record < end; ++record) {
            const uint32_t bucket =
              (sort_src[record].voxel_index >> shift) & kRadixBucketMask;
            ++thread_counts[bucket];
          }

#pragma omp barrier
#pragma omp single
          {
            uint32_t global_offset = 0U;
            for (size_t bucket = 0; bucket < kRadixBucketCount; ++bucket) {
              uint32_t bucket_offset = global_offset;
              for (int worker = 0; worker < actual_threads; ++worker) {
                const size_t offset_index =
                  static_cast<size_t>(worker) * kRadixBucketCount + bucket;
                radix_offsets[offset_index] = bucket_offset;
                bucket_offset += radix_counts[offset_index];
              }
              global_offset = bucket_offset;
            }
          }
#pragma omp barrier

          uint32_t * thread_offsets = radix_offsets +
            static_cast<size_t>(tid) * kRadixBucketCount;
          for (size_t record = begin; record < end; ++record) {
            const VoxelIndexRecord value = sort_src[record];
            const uint32_t bucket =
              (value.voxel_index >> shift) & kRadixBucketMask;
            sort_dst[thread_offsets[bucket]++] = value;
          }

#pragma omp barrier
#pragma omp single
          {
            std::swap(sort_src, sort_dst);
          }
#pragma omp barrier
        }
      }

#pragma omp single
      {
        timing.sort_ms = std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - phase_begin).count();
        phase_begin = std::chrono::steady_clock::now();
        voxel_groups_.clear();
        if (!index_overflow) {
          size_t first = 0U;
          while (first < point_count) {
            size_t last = first + 1U;
            while (last < point_count &&
              sort_src[last].voxel_index == sort_src[first].voxel_index)
            {
              ++last;
            }
            voxel_groups_.push_back(VoxelGroupRange{
              static_cast<uint32_t>(first), static_cast<uint32_t>(last)});
            first = last;
          }
          output.clear();
          output.resize(voxel_groups_.size());
        }
        timing.group_ms = std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - phase_begin).count();
        phase_begin = std::chrono::steady_clock::now();
      }
#pragma omp barrier

      if (!index_overflow) {
#pragma omp for schedule(static)
        for (size_t group_index = 0; group_index < voxel_groups_.size(); ++group_index) {
          const VoxelGroupRange range = voxel_groups_[group_index];
          Eigen::Vector4f centroid = Eigen::Vector4f::Zero();
          for (uint32_t record = range.first; record < range.last; ++record) {
            centroid += input_points[sort_src[record].point_index].getVector4fMap();
          }
          centroid /= static_cast<float>(range.last - range.first);
          output.points[group_index].getVector4fMap() = centroid;
        }
      }

#pragma omp single
      {
        timing.centroid_ms = std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - phase_begin).count();
      }
    }

    if (actual_threads != projection_openmp_threads_) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 2000,
        "Parallel VoxelGrid actual_threads=%d requested_threads=%d",
        actual_threads, projection_openmp_threads_);
    }
    if (index_overflow) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 2000,
        "Parallel VoxelGrid leaf index would overflow; returning input like PCL");
      output = input;
    }

    output.width = output.size();
    output.height = 1;
    output.is_dense = true;
    timing.output_voxels = output.size();
    return !output.empty();
  }

  void prepareFusedVoxelTables(int requested_threads)
  {
    const size_t n = static_cast<size_t>(requested_threads);
    const size_t cap = static_cast<size_t>(projection_voxel_table_capacity_);
    if (!global_voxel_table_) {
      global_voxel_table_ = std::make_unique<FlatVoxelTable>(cap);
    }
    while (omp_local_voxel_tables_.size() < n) {
      omp_local_voxel_tables_.emplace_back(std::make_unique<FlatVoxelTable>(cap));
    }
    if (omp_cpu_by_thread_.size() != n) {
      omp_cpu_by_thread_.assign(n, -1);
      omp_affinity_errors_.assign(n, 0);
    }
  }

  bool depthToVoxelFusedOpenMP(
    const sensor_msgs::msg::Image & depth,
    const sensor_msgs::msg::CameraInfo & camera_info,
    const Eigen::Affine3d & transform,
    float voxel_leaf,
    CloudT & cloud_voxel,
    FusedTiming & timing)
  {
    timing = FusedTiming{};
    if (!(voxel_leaf > 0.0f) || !std::isfinite(voxel_leaf)) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "Invalid voxel_leaf_m");
      return false;
    }

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
    timing.sampled_pixels = sampled_rows * sampled_cols;

    prepareFusedVoxelTables(projection_openmp_threads_);
    for (int tid = 0; tid < projection_openmp_threads_; ++tid) {
      omp_local_voxel_tables_[static_cast<size_t>(tid)]->beginFrame();
    }

    const float inverse_leaf = 1.0f / voxel_leaf;
    const float * ray_x = ray_x_.data();
    const float * ray_y = ray_y_.data();
    const uint8_t * depth_data = depth.data.data();
    const size_t depth_step = depth.step;
    std::vector<std::unique_ptr<FlatVoxelTable>> * local_tables = &omp_local_voxel_tables_;
    const int requested_threads = projection_openmp_threads_;
    const int chunk_rows = projection_voxel_chunk_rows_;
    int actual_threads = 0;
    bool capture_affinity = !omp_affinity_logged_;
    int * cpu_by_thread = omp_cpu_by_thread_.data();
    int * affinity_errors = omp_affinity_errors_.data();
    const int * target_cpus = projection_openmp_cpus_.data();
    size_t roi_valid_points = 0;
    uint32_t invalid_keys = 0;
    uint32_t overflow_count = 0;

    uint32_t grow_before = global_voxel_table_->growCount();
    for (int tid = 0; tid < requested_threads; ++tid) {
      grow_before += omp_local_voxel_tables_[static_cast<size_t>(tid)]->growCount();
    }

    const auto accumulate_begin = std::chrono::steady_clock::now();
#pragma omp parallel num_threads(requested_threads) default(none) \
    shared(actual_threads, is_u16, sampled_rows, sampled_cols, stride, \
           depth_data, depth_step, ray_x, ray_y, local_tables, \
           zmin, zmax, xmin, xmax, zmin_roi, zmax_roi, \
           r00, r01, r02, tx, r10, r11, r12, ty, r20, r21, r22, tz, \
           inverse_leaf, chunk_rows, \
           capture_affinity, cpu_by_thread, affinity_errors, target_cpus) \
    reduction(+:roi_valid_points, invalid_keys, overflow_count)
    {
      const int tid = omp_get_thread_num();
#pragma omp single
      {
        actual_threads = omp_get_num_threads();
      }

      cpu_set_t compute_affinity;
      CPU_ZERO(&compute_affinity);
      CPU_SET(target_cpus[tid], &compute_affinity);
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

      FlatVoxelTable & local = *(*local_tables)[static_cast<size_t>(tid)];

#pragma omp for schedule(static, chunk_rows)
      for (size_t vi = 0; vi < sampled_rows; ++vi) {
        const int v = static_cast<int>(vi) * stride;
        if (is_u16) {
          const auto * row = reinterpret_cast<const uint16_t *>(
            depth_data + static_cast<size_t>(v) * depth_step);
          for (size_t ui = 0; ui < sampled_cols; ++ui) {
            const int u = static_cast<int>(ui) * stride;
            const int status = projectAndAccumulate(
              local, static_cast<float>(row[u]) * 0.001f, ui, vi,
              ray_x, ray_y, zmin, zmax, xmin, xmax, zmin_roi, zmax_roi,
              r00, r01, r02, tx, r10, r11, r12, ty, r20, r21, r22, tz,
              inverse_leaf);
            if (status != 0) {
              ++roi_valid_points;
              if (status == 1) {
                ++invalid_keys;
              } else if (status == 2) {
                ++overflow_count;
              }
            }
          }
        } else {
          const auto * row = reinterpret_cast<const float *>(
            depth_data + static_cast<size_t>(v) * depth_step);
          for (size_t ui = 0; ui < sampled_cols; ++ui) {
            const int u = static_cast<int>(ui) * stride;
            const int status = projectAndAccumulate(
              local, row[u], ui, vi,
              ray_x, ray_y, zmin, zmax, xmin, xmax, zmin_roi, zmax_roi,
              r00, r01, r02, tx, r10, r11, r12, ty, r20, r21, r22, tz,
              inverse_leaf);
            if (status != 0) {
              ++roi_valid_points;
              if (status == 1) {
                ++invalid_keys;
              } else if (status == 2) {
                ++overflow_count;
              }
            }
          }
        }
      }
    }

    timing.accumulate_ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - accumulate_begin).count();
    timing.roi_valid_points = roi_valid_points;
    timing.invalid_keys = invalid_keys;

    if (!omp_affinity_logged_) {
      logOmpAffinity(actual_threads);
      omp_affinity_logged_ = true;
    }
    if (actual_threads != omp_actual_threads_) {
      omp_actual_threads_ = actual_threads;
      RCLCPP_INFO(
        get_logger(),
        "OpenMP fused voxel team requested_threads=%d actual_threads=%d",
        projection_openmp_threads_, omp_actual_threads_);
    }

    uint32_t max_probe = 0;
    size_t local_entries = 0;
    for (int tid = 0; tid < actual_threads; ++tid) {
      const FlatVoxelTable & local = *omp_local_voxel_tables_[static_cast<size_t>(tid)];
      max_probe = std::max(max_probe, local.maxProbe());
      local_entries += local.size();
    }

    const auto merge_begin = std::chrono::steady_clock::now();
    global_voxel_table_->beginFrame();
    for (int tid = 0; tid < actual_threads; ++tid) {
      const FlatVoxelTable & local = *omp_local_voxel_tables_[static_cast<size_t>(tid)];
      for (const uint32_t slot_idx : local.occupiedSlots()) {
        const VoxelAccum & src = local.slot(slot_idx);
        if (!global_voxel_table_->merge(src.key, src)) {
          ++overflow_count;
        }
      }
    }
    timing.merge_ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - merge_begin).count();
    timing.local_voxel_entries = local_entries;
    max_probe = std::max(max_probe, global_voxel_table_->maxProbe());

    const auto emit_begin = std::chrono::steady_clock::now();
    cloud_voxel.clear();
    cloud_voxel.reserve(global_voxel_table_->size());
    for (const uint32_t slot_idx : global_voxel_table_->occupiedSlots()) {
      const VoxelAccum & accum = global_voxel_table_->slot(slot_idx);
      if (accum.count == 0U) {
        continue;
      }
      const double inv_count = 1.0 / static_cast<double>(accum.count);
      PointT p;
      p.x = static_cast<float>(accum.sum_x * inv_count);
      p.y = static_cast<float>(accum.sum_y * inv_count);
      p.z = static_cast<float>(accum.sum_z * inv_count);
      cloud_voxel.push_back(p);
    }
    cloud_voxel.width = cloud_voxel.size();
    cloud_voxel.height = 1;
    cloud_voxel.is_dense = true;
    timing.emit_ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - emit_begin).count();

    uint32_t grow_after = global_voxel_table_->growCount();
    for (int tid = 0; tid < actual_threads; ++tid) {
      grow_after += omp_local_voxel_tables_[static_cast<size_t>(tid)]->growCount();
    }
    timing.output_voxels = cloud_voxel.size();
    timing.max_probe = max_probe;
    timing.table_grows = grow_after - grow_before;

    if (overflow_count != 0U || invalid_keys != 0U) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 2000,
        "Fused voxel dropped overflow=%u invalid_keys=%u; increase projection_voxel_table_capacity",
        overflow_count, invalid_keys);
    }

    return !cloud_voxel.empty();
  }

  void verifyFusedVoxelAgainstPcl(
    const sensor_msgs::msg::Image & depth,
    const sensor_msgs::msg::CameraInfo & camera_info,
    const Eigen::Affine3d & transform,
    float voxel_leaf,
    const CloudT & fused_cloud)
  {
    CloudT::Ptr cloud_roi(new CloudT);
    if (!depthToWorkspaceScalar(depth, camera_info, transform, *cloud_roi)) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 2000,
        "VERIFY fused voxel skipped: scalar projection produced an empty cloud");
      return;
    }

    CloudT cloud_pcl;
    {
      pcl::VoxelGrid<PointT> vg;
      vg.setInputCloud(cloud_roi);
      vg.setLeafSize(voxel_leaf, voxel_leaf, voxel_leaf);
      vg.filter(cloud_pcl);
    }

    struct VoxelStats
    {
      uint32_t count{0};
      double sum_x{0.0};
      double sum_y{0.0};
      double sum_z{0.0};
    };

    const float inverse_leaf = 1.0f / voxel_leaf;
    std::unordered_map<VoxelKey, VoxelStats> fused_map;
    if (global_voxel_table_) {
      fused_map.reserve(global_voxel_table_->size() * 2U);
      for (const uint32_t slot_idx : global_voxel_table_->occupiedSlots()) {
        const VoxelAccum & accum = global_voxel_table_->slot(slot_idx);
        VoxelStats stats;
        stats.count = accum.count;
        stats.sum_x = accum.sum_x;
        stats.sum_y = accum.sum_y;
        stats.sum_z = accum.sum_z;
        fused_map.emplace(accum.key, stats);
      }
    }

    std::unordered_map<VoxelKey, VoxelStats> group_map;
    group_map.reserve(fused_map.size() * 2U);
    uint32_t group_invalid_keys = 0;
    for (const PointT & p : cloud_roi->points) {
      VoxelKey key = 0;
      if (!makeVoxelKey(p.x, p.y, p.z, inverse_leaf, key)) {
        ++group_invalid_keys;
        continue;
      }
      VoxelStats & stats = group_map[key];
      stats.count += 1U;
      stats.sum_x += static_cast<double>(p.x);
      stats.sum_y += static_cast<double>(p.y);
      stats.sum_z += static_cast<double>(p.z);
    }

    size_t missing_keys = 0;
    size_t extra_keys = 0;
    size_t count_mismatches = 0;
    double centroid_max_error = 0.0;
    for (const auto & entry : group_map) {
      const auto it = fused_map.find(entry.first);
      if (it == fused_map.end()) {
        ++missing_keys;
        continue;
      }
      if (it->second.count != entry.second.count) {
        ++count_mismatches;
      }
      const double inv_count = 1.0 / static_cast<double>(entry.second.count);
      const double fx = it->second.sum_x / static_cast<double>(it->second.count);
      const double fy = it->second.sum_y / static_cast<double>(it->second.count);
      const double fz = it->second.sum_z / static_cast<double>(it->second.count);
      const double gx = entry.second.sum_x * inv_count;
      const double gy = entry.second.sum_y * inv_count;
      const double gz = entry.second.sum_z * inv_count;
      centroid_max_error = std::max(centroid_max_error, std::abs(fx - gx));
      centroid_max_error = std::max(centroid_max_error, std::abs(fy - gy));
      centroid_max_error = std::max(centroid_max_error, std::abs(fz - gz));
    }
    for (const auto & entry : fused_map) {
      if (group_map.find(entry.first) == group_map.end()) {
        ++extra_keys;
      }
    }

    size_t pcl_missing = 0;
    size_t pcl_extra = 0;
    double pcl_centroid_max_error = 0.0;
    std::unordered_map<VoxelKey, PointT> pcl_map;
    pcl_map.reserve(cloud_pcl.size() * 2U);
    uint32_t pcl_invalid_keys = 0;
    for (const PointT & p : cloud_pcl.points) {
      VoxelKey key = 0;
      if (!makeVoxelKey(p.x, p.y, p.z, inverse_leaf, key)) {
        ++pcl_invalid_keys;
        continue;
      }
      pcl_map.emplace(key, p);
    }
    for (const auto & entry : fused_map) {
      const auto it = pcl_map.find(entry.first);
      if (it == pcl_map.end()) {
        ++pcl_missing;
        continue;
      }
      const double inv_count = 1.0 / static_cast<double>(entry.second.count);
      const double fx = entry.second.sum_x * inv_count;
      const double fy = entry.second.sum_y * inv_count;
      const double fz = entry.second.sum_z * inv_count;
      pcl_centroid_max_error = std::max(
        pcl_centroid_max_error, static_cast<double>(std::abs(it->second.x - static_cast<float>(fx))));
      pcl_centroid_max_error = std::max(
        pcl_centroid_max_error, static_cast<double>(std::abs(it->second.y - static_cast<float>(fy))));
      pcl_centroid_max_error = std::max(
        pcl_centroid_max_error, static_cast<double>(std::abs(it->second.z - static_cast<float>(fz))));
    }
    for (const auto & entry : pcl_map) {
      if (fused_map.find(entry.first) == fused_map.end()) {
        ++pcl_extra;
      }
    }

    const bool grouping_ok =
      missing_keys == 0 && extra_keys == 0 && count_mismatches == 0 &&
      centroid_max_error <= 1e-4 && group_invalid_keys == 0;
    const bool pcl_ok =
      pcl_missing == 0 && pcl_extra == 0 && pcl_centroid_max_error <= 1e-4 &&
      pcl_invalid_keys == 0 && fused_cloud.size() == cloud_pcl.size();

    if (!grouping_ok || !pcl_ok) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 1000,
        "VERIFY fused voxel mismatch fused=%zu pcl=%zu group=%zu "
        "missing=%zu extra=%zu count_mismatch=%zu centroid_max=%.6f "
        "pcl_missing=%zu pcl_extra=%zu pcl_centroid_max=%.6f "
        "group_invalid=%u pcl_invalid=%u",
        fused_cloud.size(), cloud_pcl.size(), group_map.size(),
        missing_keys, extra_keys, count_mismatches, centroid_max_error,
        pcl_missing, pcl_extra, pcl_centroid_max_error,
        group_invalid_keys, pcl_invalid_keys);
    } else {
      RCLCPP_INFO_THROTTLE(
        get_logger(), *get_clock(), 2000,
        "VERIFY fused voxel ok fused=%zu pcl=%zu centroid_max=%.6f pcl_centroid_max=%.6f",
        fused_cloud.size(), cloud_pcl.size(), centroid_max_error, pcl_centroid_max_error);
    }
  }

  void verifyParallelVoxelAgainstPcl(
    const sensor_msgs::msg::Image & depth,
    const sensor_msgs::msg::CameraInfo & camera_info,
    const Eigen::Affine3d & transform,
    float voxel_leaf,
    const CloudT & parallel_cloud)
  {
    CloudT::Ptr input(new CloudT);
    if (!depthToWorkspaceScalar(depth, camera_info, transform, *input)) {
      RCLCPP_ERROR_THROTTLE(
        get_logger(), *get_clock(), 2000,
        "VERIFY V5 PCL_EQUIV could not build scalar reference input");
      return;
    }
    CloudT pcl_cloud;
    pcl::VoxelGrid<PointT> vg;
    vg.setInputCloud(input);
    vg.setLeafSize(voxel_leaf, voxel_leaf, voxel_leaf);
    vg.filter(pcl_cloud);

    bool order_ok = parallel_cloud.size() == pcl_cloud.size();
    double max_coordinate_error = 0.0;
    size_t first_key_mismatch = std::numeric_limits<size_t>::max();
    const float inverse_leaf = 1.0f / voxel_leaf;
    const size_t compare_count = std::min(parallel_cloud.size(), pcl_cloud.size());
    for (size_t i = 0; i < compare_count; ++i) {
      VoxelKey parallel_key = 0;
      VoxelKey pcl_key = 0;
      const PointT & actual = parallel_cloud.points[i];
      const PointT & reference = pcl_cloud.points[i];
      const bool parallel_key_ok = makeVoxelKey(
        actual.x, actual.y, actual.z, inverse_leaf, parallel_key);
      const bool pcl_key_ok = makeVoxelKey(
        reference.x, reference.y, reference.z, inverse_leaf, pcl_key);
      if (!parallel_key_ok || !pcl_key_ok || parallel_key != pcl_key) {
        order_ok = false;
        if (first_key_mismatch == std::numeric_limits<size_t>::max()) {
          first_key_mismatch = i;
        }
      }
      max_coordinate_error = std::max(
        max_coordinate_error,
        static_cast<double>(std::abs(actual.x - reference.x)));
      max_coordinate_error = std::max(
        max_coordinate_error,
        static_cast<double>(std::abs(actual.y - reference.y)));
      max_coordinate_error = std::max(
        max_coordinate_error,
        static_cast<double>(std::abs(actual.z - reference.z)));
    }

    constexpr double kCentroidTolerance = 1e-5;
    if (order_ok && max_coordinate_error <= kCentroidTolerance) {
      RCLCPP_INFO_THROTTLE(
        get_logger(), *get_clock(), 2000,
        "VERIFY V5 PCL_EQUIV ok input=%zu output=%zu max_abs_error=%.9f",
        input->size(), parallel_cloud.size(), max_coordinate_error);
    } else {
      RCLCPP_ERROR_THROTTLE(
        get_logger(), *get_clock(), 2000,
        "VERIFY V5 PCL_EQUIV mismatch input=%zu parallel=%zu pcl=%zu "
        "first_key_mismatch=%zu max_abs_error=%.9f tolerance=%.9f",
        input->size(), parallel_cloud.size(), pcl_cloud.size(),
        first_key_mismatch, max_coordinate_error, kCentroidTolerance);
    }
  }

  bool depthToWorkspaceOpenMP(
    const sensor_msgs::msg::Image & depth,
    const sensor_msgs::msg::CameraInfo & camera_info,
    const Eigen::Affine3d & transform,
    CloudT & cloud_roi,
    bool use_neon = false)
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
    size_t * cloud_offsets = omp_cloud_offsets_.data();
    const NeonWorkspaceParams neon_params{
      zmin, zmax, xmin, xmax, zmin_roi, zmax_roi,
      static_cast<float>(r00), static_cast<float>(r01), static_cast<float>(r02),
      static_cast<float>(tx),
      static_cast<float>(r10), static_cast<float>(r11), static_cast<float>(r12),
      static_cast<float>(ty),
      static_cast<float>(r20), static_cast<float>(r21), static_cast<float>(r22),
      static_cast<float>(tz)
    };
    const PreciseNeonWorkspaceParams precise_neon_params{
      zmin, zmax, xmin, xmax, zmin_roi, zmax_roi,
      static_cast<float>(get_parameter("voxel_leaf_m").as_double()),
      r00, r01, r02, tx,
      r10, r11, r12, ty,
      r20, r21, r22, tz
    };

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
           capture_affinity, cpu_by_thread, affinity_errors, target_cpus, \
           use_neon, neon_params, precise_neon_params, cloud_roi, cloud_offsets)
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
          size_t ui = 0;
#if RGBD_PCL_HAS_NEON
          if (use_neon) {
            for (; ui + 3U < sampled_cols; ui += 4U) {
              tryAppendPointsNeon4(
                local_cloud, row, ui, vi, stride, ray_x, ray_y,
                neon_params, &precise_neon_params);
            }
          }
#endif
          for (; ui < sampled_cols; ++ui) {
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
          size_t ui = 0;
#if RGBD_PCL_HAS_NEON
          if (use_neon) {
            for (; ui + 3U < sampled_cols; ui += 4U) {
              tryAppendPointsNeon4(
                local_cloud, row, ui, vi, stride, ray_x, ray_y,
                neon_params, &precise_neon_params);
            }
          }
#endif
          for (; ui < sampled_cols; ++ui) {
            const int u = static_cast<int>(ui) * stride;
            tryAppendPoint(
              local_cloud, row[u], ui, vi,
              ray_x, ray_y, zmin, zmax, xmin, xmax, zmin_roi, zmax_roi,
              r00, r01, r02, tx, r10, r11, r12, ty, r20, r21, r22, tz);
          }
        }
      }

      // Preserve the scalar row-major point order without a serial
      // concatenation: contiguous row ownership makes thread-id order equal to
      // input order, and every worker copies directly to its final range.
#pragma omp barrier
#pragma omp single
      {
        cloud_offsets[0] = 0U;
        for (int worker = 0; worker < actual_threads; ++worker) {
          cloud_offsets[static_cast<size_t>(worker + 1)] =
            cloud_offsets[static_cast<size_t>(worker)] +
            (*thread_clouds)[static_cast<size_t>(worker)]->size();
        }
        cloud_roi.clear();
        cloud_roi.resize(cloud_offsets[static_cast<size_t>(actual_threads)]);
      }
#pragma omp barrier
      std::copy(
        local_cloud.points.begin(), local_cloud.points.end(),
        cloud_roi.points.begin() + static_cast<std::ptrdiff_t>(
          cloud_offsets[static_cast<size_t>(tid)]));
    }  // 并行区结束

    if (!omp_affinity_logged_) {
      logOmpAffinity(actual_threads);
      omp_affinity_logged_ = true;
    }

    if (actual_threads != omp_actual_threads_) {
      omp_actual_threads_ = actual_threads;
      RCLCPP_INFO(
        get_logger(),
        "OpenMP team requested_threads=%d actual_threads=%d",
        projection_openmp_threads_, omp_actual_threads_);
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
  bool projection_fuse_voxel_{false};
  bool projection_fused_voxel_openmp_{false};
  int projection_voxel_chunk_rows_{8};
  int projection_voxel_table_capacity_{16384};
  bool projection_fused_voxel_verify_{false};
  int omp_actual_threads_{0};
  std::vector<CloudT::Ptr> omp_thread_clouds_;
  std::vector<size_t> omp_cloud_offsets_;
  std::vector<VoxelIndexRecord> voxel_indices_a_;
  std::vector<VoxelIndexRecord> voxel_indices_b_;
  std::vector<uint32_t> voxel_radix_counts_;
  std::vector<uint32_t> voxel_radix_offsets_;
  std::vector<VoxelGroupRange> voxel_groups_;
  CloudT v5_cloud_roi_buffer_;
  CloudT v5_cloud_voxel_buffer_;
  std::unique_ptr<FlatVoxelTable> global_voxel_table_;
  std::vector<std::unique_ptr<FlatVoxelTable>> omp_local_voxel_tables_;
  std::vector<int> omp_cpu_by_thread_;
  std::vector<int> omp_affinity_errors_;
  bool omp_affinity_logged_{false};
  bool omp_affinity_valid_{true};
  bool compute_affinity_valid_{true};
  size_t successful_frames_{0};
  std::vector<double> queue_wait_samples_;
  std::vector<double> core_samples_;
  std::vector<double> voxel_samples_;
  std::vector<double> compute_total_samples_;
  std::vector<double> callback_samples_;
  std::vector<double> fused_accumulate_samples_;
  std::vector<double> fused_merge_samples_;
  std::vector<double> fused_emit_samples_;
  std::vector<double> fused_sampled_pixels_samples_;
  std::vector<double> fused_roi_valid_samples_;
  std::vector<double> fused_local_entries_samples_;
  std::vector<double> fused_output_voxels_samples_;
  std::vector<double> fused_max_probe_samples_;
  std::vector<double> v5_minmax_samples_;
  std::vector<double> v5_index_samples_;
  std::vector<double> v5_sort_samples_;
  std::vector<double> v5_group_samples_;
  std::vector<double> v5_centroid_samples_;
  uint32_t fused_table_grows_{0};

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
