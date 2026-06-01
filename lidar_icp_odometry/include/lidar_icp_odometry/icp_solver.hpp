#pragma once

#include <vector>
#include <utility>
#include <unordered_map>
#include <cmath>

#include <Eigen/Dense>
#include <sophus/se3.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>

#include "lidar_icp_odometry/third_party/nanoflann.hpp"

namespace lidar_icp {

// ---------------------------------------------------------------------------
// Aliases matching ekf_se3_localization style
// ---------------------------------------------------------------------------
using Vector3d = Eigen::Vector3d;
using Matrix3d = Eigen::Matrix3d;
using Matrix6d = Eigen::Matrix<double, 6, 6>;
using Vector6d = Eigen::Matrix<double, 6, 1>;

// ---------------------------------------------------------------------------
// A single point with its estimated surface normal.
// ---------------------------------------------------------------------------
struct PointNormal {
  Vector3d point;   ///< 3D position in the target scan frame
  Vector3d normal;  ///< Unit surface normal oriented toward the sensor origin
};

// ---------------------------------------------------------------------------
// Diagnostics returned by IcpSolver::align().
// ---------------------------------------------------------------------------
struct IcpResult {
  Sophus::SE3d delta_T;   ///< ΔT ∈ SE(3): maps source frame → target frame
  double       rmse;      ///< Point-to-plane RMSE of the final correspondence set
  int          iterations;///< Number of linearisation iterations executed
  bool         converged; ///< True when ‖ξ‖₂ < convergence_tolerance
};

// ---------------------------------------------------------------------------
// Nanoflann adaptor — wraps std::vector<Vector3d> so nanoflann can build
// a KD-tree over it without copying data.
// ---------------------------------------------------------------------------
struct EigenCloud {
  const std::vector<Vector3d> & pts;

  // Required nanoflann interface
  inline size_t kdtree_get_point_count() const { return pts.size(); }
  inline double kdtree_get_pt(const size_t idx, const size_t dim) const
  {
    return pts[idx][static_cast<int>(dim)];
  }
  template<class BBOX>
  bool kdtree_get_bbox(BBOX &) const { return false; }
};

using KdTree3d = nanoflann::KDTreeSingleIndexAdaptor<
  nanoflann::L2_Simple_Adaptor<double, EigenCloud>,
  EigenCloud, 3>;

// ---------------------------------------------------------------------------
// Voxel key used by the hash-map downsampler.
// ---------------------------------------------------------------------------
struct VoxelKey {
  int x, y, z;
  bool operator==(const VoxelKey & o) const
  {
    return x == o.x && y == o.y && z == o.z;
  }
};

struct VoxelKeyHash {
  size_t operator()(const VoxelKey & k) const
  {
    // Boost-style hash combining
    size_t h = 0;
    auto combine = [&](int v) {
      h ^= std::hash<int>{}(v) + 0x9e3779b9u + (h << 6) + (h >> 2);
    };
    combine(k.x);
    combine(k.y);
    combine(k.z);
    return h;
  }
};

// ---------------------------------------------------------------------------
/**
 * @brief Point-to-Plane ICP solver.
 *
 * Implements the algorithm described in SLAM Handbook Chapter 8, §8.2.1
 * (Foundations of Scan Registration). Given a source point cloud P and a
 * target point cloud Q (with per-point surface normals), it finds
 *
 *   ΔT* = arg min_{ΔT ∈ SE(3)} Σᵢ ( nᵢᵀ (ΔR pᵢ + Δt − qᵢ) )²
 *
 * Rotation is linearised as ΔR ≈ I + [θ]× (small-angle), converting the
 * non-linear problem into a 6×6 linear system solved with LDLT:
 *
 *   (AᵀA) ξ = Aᵀb,   ξ = [Δt; θ] ∈ ℝ⁶
 *
 * The tangent vector ξ uses Sophus ordering (translation first, then
 * rotation) matching Sophus::SE3d::exp(). After each solve, ξ is retracted:
 *
 *   ΔT ← ΔT · Sophus::SE3d::exp(ξ)
 *
 * Iteration repeats until ‖ξ‖₂ < convergence_tolerance or max_iterations
 * is reached.
 *
 * This class has zero rclcpp dependencies and is independently testable.
 * Its static helpers (from_ros_msg, voxel_downsample, estimate_normals)
 * perform all preprocessing before align() is called.
 */
class IcpSolver
{
public:
  /**
   * @param max_iterations        Maximum linearisation iterations per call.
   * @param convergence_tolerance Stop when ‖ξ‖₂ < tol.
   * @param max_correspondence_dist  Reject source–target pairs farther apart.
   */
  IcpSolver(int max_iterations,
            double convergence_tolerance,
            double max_correspondence_dist);

  // -------------------------------------------------------------------------
  // Preprocessing — called by the ROS node before align()
  // -------------------------------------------------------------------------

  /**
   * @brief Parse a sensor_msgs::PointCloud2 into a flat vector of 3D points.
   *
   * Uses sensor_msgs::PointCloud2ConstIterator (standard ROS2, zero PCL).
   * NaN and Inf returns are silently dropped.
   */
  static std::vector<Vector3d> from_ros_msg(
    const sensor_msgs::msg::PointCloud2 & msg);

  /**
   * @brief Voxel-grid downsample — replace each occupied voxel with its centroid.
   *
   * Partitions space into axis-aligned cubes of side leaf_size. Each occupied
   * voxel contributes one output point (the mean of all input points inside).
   * Implemented with std::unordered_map<VoxelKey, ...>; zero PCL dependency.
   *
   * @param pts       Input point cloud.
   * @param leaf_size Voxel side length in metres.
   */
  static std::vector<Vector3d> voxel_downsample(
    const std::vector<Vector3d> & pts,
    double leaf_size);

  /**
   * @brief Estimate a unit surface normal for every point via k-NN PCA.
   *
   * For each point p:
   *  1. Find k nearest neighbours using nanoflann KD-tree.
   *  2. Build 3×3 covariance C = (1/k) Σ (pⱼ − p̄)(pⱼ − p̄)ᵀ.
   *  3. Normal = eigenvector of C with smallest eigenvalue
   *              (Eigen::SelfAdjointEigenSolver, column 0).
   *  4. Flip sign so that n · (−p) > 0 (normal points toward sensor origin).
   *
   * @param pts Input point cloud.
   * @param k   Number of nearest neighbours (typically 10–30).
   */
  static std::vector<PointNormal> estimate_normals(
    const std::vector<Vector3d> & pts,
    int k);

  // -------------------------------------------------------------------------
  // Core ICP
  // -------------------------------------------------------------------------

  /**
   * @brief Align source_pts → target_normals using Point-to-Plane ICP.
   *
   * Builds a nanoflann KD-tree over the target point positions once per call
   * (O(N log N)), then iterates: find 1-NN correspondence → solve 6×6 linear
   * system → retract → repeat.
   *
   * @param source_pts      Downsampled source cloud (current scan).
   * @param target_normals  Downsampled target cloud with normals (previous scan).
   * @param T_init          Initial guess for ΔT (use SE3d::Identity() when unknown).
   * @return IcpResult with final ΔT, RMSE, iteration count, convergence flag.
   */
  IcpResult align(
    const std::vector<Vector3d>    & source_pts,
    const std::vector<PointNormal> & target_normals,
    const Sophus::SE3d             & T_init = Sophus::SE3d()) const;

private:
  /**
   * @brief One linearised Point-to-Plane iteration.
   *
   * Builds the normal equations (AᵀA)ξ = Aᵀb where each row of A is
   *   aᵢᵀ = [nᵢᵀ,  (pᵢ × nᵢ)ᵀ]   (translation first — Sophus ordering)
   * and solves with LDLT (same numerical pattern as the EKF Kalman gain).
   *
   * @param src_transformed  Source points after current ΔT is applied.
   * @param correspondences  Matched (target_point, target_normal) pairs.
   * @param xi_out           Solved correction ξ = [Δt; θ] ∈ ℝ⁶.
   * @return Point-to-plane RMSE of the residuals before this correction.
   */
  double solve_linear_system(
    const std::vector<Vector3d>                          & src_transformed,
    const std::vector<std::pair<Vector3d, Vector3d>>     & correspondences,
    Vector6d                                             & xi_out) const;

  int    max_iterations_;
  double convergence_tolerance_;
  double max_correspondence_dist_;
};

}  // namespace lidar_icp
