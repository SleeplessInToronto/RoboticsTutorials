#include "lidar_icp_odometry/icp_solver.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace lidar_icp {

// ---------------------------------------------------------------------------
IcpSolver::IcpSolver(int max_iterations,
                     double convergence_tolerance,
                     double max_correspondence_dist)
: max_iterations_(max_iterations),
  convergence_tolerance_(convergence_tolerance),
  max_correspondence_dist_(max_correspondence_dist)
{}

// ---------------------------------------------------------------------------
// Parse PointCloud2 → std::vector<Vector3d>
// Uses sensor_msgs::PointCloud2ConstIterator — zero PCL.
// ---------------------------------------------------------------------------
std::vector<Vector3d> IcpSolver::from_ros_msg(
  const sensor_msgs::msg::PointCloud2 & msg)
{
  std::vector<Vector3d> pts;
  pts.reserve(msg.width * msg.height);

  sensor_msgs::PointCloud2ConstIterator<float> it_x(msg, "x");
  sensor_msgs::PointCloud2ConstIterator<float> it_y(msg, "y");
  sensor_msgs::PointCloud2ConstIterator<float> it_z(msg, "z");

  for (; it_x != it_x.end(); ++it_x, ++it_y, ++it_z) {
    if (std::isfinite(*it_x) && std::isfinite(*it_y) && std::isfinite(*it_z)) {
      pts.emplace_back(
        static_cast<double>(*it_x),
        static_cast<double>(*it_y),
        static_cast<double>(*it_z));
    }
  }
  return pts;
}

// ---------------------------------------------------------------------------
// Voxel grid downsampling
//
// Partition space into axis-aligned voxels of side leaf_size.
// Each occupied voxel is replaced by the centroid of its points.
// Implemented with std::unordered_map<VoxelKey,...>; no PCL.
// ---------------------------------------------------------------------------
std::vector<Vector3d> IcpSolver::voxel_downsample(
  const std::vector<Vector3d> & pts,
  double leaf_size)
{
  if (pts.empty() || leaf_size <= 0.0) return pts;

  const double inv_leaf = 1.0 / leaf_size;

  std::unordered_map<VoxelKey, Vector3d, VoxelKeyHash> voxel_sum;
  std::unordered_map<VoxelKey, int,      VoxelKeyHash> voxel_cnt;
  voxel_sum.reserve(pts.size() / 4);
  voxel_cnt.reserve(pts.size() / 4);

  for (const auto & p : pts) {
    VoxelKey k{
      static_cast<int>(std::floor(p.x() * inv_leaf)),
      static_cast<int>(std::floor(p.y() * inv_leaf)),
      static_cast<int>(std::floor(p.z() * inv_leaf))};
    voxel_sum[k] += p;
    voxel_cnt[k]++;
  }

  std::vector<Vector3d> out;
  out.reserve(voxel_sum.size());
  for (const auto & [k, sum] : voxel_sum) {
    out.push_back(sum / static_cast<double>(voxel_cnt.at(k)));
  }
  return out;
}

// ---------------------------------------------------------------------------
// Normal estimation via k-NN PCA
//
// For each point p:
//   1. Find k nearest neighbours with nanoflann.
//   2. Compute covariance C = (1/k) Σ (pⱼ − p̄)(pⱼ − p̄)ᵀ.
//   3. Normal = eigenvector of C with smallest eigenvalue (col 0 of
//      SelfAdjointEigenSolver, which sorts eigenvalues in ascending order).
//   4. Flip normal so it points toward the sensor origin (0,0,0):
//      if n · (−p) < 0  →  n = −n.
// ---------------------------------------------------------------------------
std::vector<PointNormal> IcpSolver::estimate_normals(
  const std::vector<Vector3d> & pts,
  int k)
{
  if (pts.empty()) return {};

  // Build KD-tree over the cloud itself for k-NN queries
  EigenCloud adaptor{pts};
  KdTree3d tree(3, adaptor, nanoflann::KDTreeSingleIndexAdaptorParams(10));
  tree.buildIndex();

  const int k_actual = std::min(k, static_cast<int>(pts.size()));

  std::vector<PointNormal> normals;
  normals.reserve(pts.size());

  std::vector<size_t>  nn_idx(k_actual);
  std::vector<double>  nn_sq_dist(k_actual);

  for (const auto & p : pts) {
    nanoflann::KNNResultSet<double> result(k_actual);
    result.init(nn_idx.data(), nn_sq_dist.data());
    tree.findNeighbors(result, p.data(), nanoflann::SearchParameters());

    // Compute centroid of the neighbourhood
    Vector3d centroid = Vector3d::Zero();
    for (int j = 0; j < k_actual; ++j) {
      centroid += pts[nn_idx[j]];
    }
    centroid /= static_cast<double>(k_actual);

    // Build 3×3 covariance matrix C = (1/k) Σ (pⱼ − p̄)(pⱼ − p̄)ᵀ
    Matrix3d C = Matrix3d::Zero();
    for (int j = 0; j < k_actual; ++j) {
      const Vector3d d = pts[nn_idx[j]] - centroid;
      C += d * d.transpose();
    }
    C /= static_cast<double>(k_actual);

    // Normal = eigenvector of C with smallest eigenvalue
    // SelfAdjointEigenSolver returns eigenvalues in ascending order,
    // so column 0 corresponds to the smallest eigenvalue = surface normal.
    Eigen::SelfAdjointEigenSolver<Matrix3d> solver(C);
    Vector3d normal = solver.eigenvectors().col(0);

    // Flip toward viewpoint (sensor at origin of sensor frame)
    if (normal.dot(-p) < 0.0) {
      normal = -normal;
    }

    normals.push_back(PointNormal{p, normal});
  }
  return normals;
}

// ---------------------------------------------------------------------------
// IcpSolver::align — iterative Point-to-Plane ICP
//
// Algorithm (SLAM Handbook §8.2.1):
//   Build KD-tree over target point positions (O(N log N), once per call).
//   For each iteration k:
//     1. Transform source points by current ΔT.
//     2. For each transformed source point, find 1-NN in target via nanoflann.
//        Reject pairs whose squared distance exceeds max_correspondence_dist_².
//     3. Solve the 6×6 linear system (AᵀA)ξ = Aᵀb (see solve_linear_system).
//     4. Retract: ΔT ← ΔT · Exp(ξ),  ΔT.so3().normalize().
//     5. Check convergence: ‖ξ‖₂ < convergence_tolerance_.
// ---------------------------------------------------------------------------
IcpResult IcpSolver::align(
  const std::vector<Vector3d>    & source_pts,
  const std::vector<PointNormal> & target_normals,
  const Sophus::SE3d             & T_init) const
{
  IcpResult result;
  result.delta_T   = T_init;
  result.rmse      = 0.0;
  result.iterations = 0;
  result.converged  = false;

  if (source_pts.empty() || target_normals.empty()) {
    return result;
  }

  // --- Build KD-tree over target point positions (once per align() call) ---
  // Extract positions from PointNormal for the adaptor
  std::vector<Vector3d> target_pts;
  target_pts.reserve(target_normals.size());
  for (const auto & pn : target_normals) {
    target_pts.push_back(pn.point);
  }

  EigenCloud target_cloud{target_pts};
  KdTree3d target_tree(3, target_cloud,
                       nanoflann::KDTreeSingleIndexAdaptorParams(10));
  target_tree.buildIndex();

  const double max_dist_sq =
    max_correspondence_dist_ * max_correspondence_dist_;

  // --- Iterative linearisation loop ---
  std::vector<Vector3d> src_transformed(source_pts.size());

  for (int iter = 0; iter < max_iterations_; ++iter) {
    // 1. Apply current ΔT to all source points
    for (size_t i = 0; i < source_pts.size(); ++i) {
      src_transformed[i] = result.delta_T * source_pts[i];
    }

    // 2. Find correspondences (1-NN, distance filtered)
    std::vector<std::pair<Vector3d, Vector3d>> correspondences;
    std::vector<Vector3d> src_valid;
    correspondences.reserve(source_pts.size());
    src_valid.reserve(source_pts.size());

    size_t nn_idx;
    double nn_sq_dist;
    nanoflann::KNNResultSet<double> nn_result(1);

    for (size_t i = 0; i < src_transformed.size(); ++i) {
      nn_result.init(&nn_idx, &nn_sq_dist);
      target_tree.findNeighbors(nn_result,
                                src_transformed[i].data(),
                                nanoflann::SearchParameters());
      if (nn_sq_dist <= max_dist_sq) {
        correspondences.emplace_back(target_normals[nn_idx].point,
                                     target_normals[nn_idx].normal);
        src_valid.push_back(src_transformed[i]);
      }
    }

    if (correspondences.empty()) {
      // No valid correspondences — ICP cannot proceed
      break;
    }

    // 3. Solve the linearised 6×6 system
    Vector6d xi;
    result.rmse = solve_linear_system(src_valid, correspondences, xi);

    // Guard: degenerate geometry can make AtA rank-deficient, causing LDLT
    // to return NaN/inf. Stop early rather than letting SE3d::exp() abort.
    if (!xi.allFinite() || xi.norm() > 10.0) {
      break;
    }

    // 4. Retract: ΔT ← ΔT · Exp(ξ)
    result.delta_T = result.delta_T * Sophus::SE3d::exp(xi);
    // Quaternion drift guard (identical pattern to ekf_se3.cpp)
    result.delta_T.so3().normalize();

    result.iterations = iter + 1;

    // 5. Convergence check
    if (xi.norm() < convergence_tolerance_) {
      result.converged = true;
      break;
    }
  }

  return result;
}

// ---------------------------------------------------------------------------
// solve_linear_system — one linearised Point-to-Plane iteration
//
// Objective after linearising ΔR ≈ I + [θ]× :
//
//   E = Σᵢ ( nᵢᵀ ([θ]× pᵢ + Δt − (qᵢ − pᵢ)) )²
//
// Using the scalar triple product identity  nᵀ([θ]× p) = (p × n)ᵀ θ :
//
//   Row i of A:   aᵢᵀ = [ nᵢᵀ ,  (pᵢ × nᵢ)ᵀ ]   ∈ ℝ⁶
//   b[i]       = nᵢᵀ (qᵢ − pᵢ)
//
// Sophus ordering: ξ = [Δt (0:2); θ (3:5)]  →  translation block first.
//
// Normal equations: (AᵀA) ξ = Aᵀb
// Solved with LDLT (symmetric positive semi-definite system, same as EKF gain).
// ---------------------------------------------------------------------------
double IcpSolver::solve_linear_system(
  const std::vector<Vector3d>                       & src_pts,
  const std::vector<std::pair<Vector3d, Vector3d>>  & correspondences,
  Vector6d                                          & xi_out) const
{
  Matrix6d AtA = Matrix6d::Zero();
  Vector6d Atb = Vector6d::Zero();
  double   sse  = 0.0;

  const int N = static_cast<int>(src_pts.size());

  for (int i = 0; i < N; ++i) {
    const Vector3d & p = src_pts[i];                    // transformed source point
    const Vector3d & q = correspondences[i].first;     // target point
    const Vector3d & n = correspondences[i].second;    // target normal (unit)

    // Row of A — translation block: nᵀ, rotation block: (p × n)ᵀ
    // This ordering matches Sophus::SE3d::exp(ξ) where ξ = [translation; rotation]
    Vector6d a;
    a.head<3>() = n;              // ∂residual/∂Δt = nᵀ
    a.tail<3>() = p.cross(n);    // ∂residual/∂θ  = (p × n)ᵀ

    const double b_i = n.dot(q - p);

    AtA += a * a.transpose();
    Atb += a * b_i;

    // Accumulate squared residual before correction (for RMSE reporting)
    const double res = n.dot(p - q);
    sse += res * res;
  }

  // Solve (AᵀA) ξ = Aᵀb with LDLT (numerically stable for symmetric PSD matrices)
  xi_out = AtA.ldlt().solve(Atb);

  return std::sqrt(sse / static_cast<double>(N));
}

}  // namespace lidar_icp
