#pragma once

// ---------------------------------------------------------------------------
// Gauss-Newton pose-graph optimiser on SE(3) — §8.4.2 of the SLAM Handbook.
//
// Minimises the total cost
//
//   C = ½ Σ_{(i,j) ∈ E}  e_ij^T  Ω_ij  e_ij
//
// where
//   e_ij  = Log( T_ij^{-1} · T_i^{-1} · T_j )  ∈ ℝ⁶          (residual)
//   T_ij  = measured relative transform for edge (i→j)
//   Ω_ij  = 6×6 information matrix (inverse covariance)
//
// Linearisation via right perturbation (ε_i on T_i, ε_j on T_j):
//
//   J_i  =  −Adj( T_j^{-1} · T_i )   ∈ ℝ^{6×6}   (Jacobian w.r.t. δξ_i)
//   J_j  =   I₆                       ∈ ℝ^{6×6}   (Jacobian w.r.t. δξ_j)
//
// where Adj(T) is the 6×6 SE(3) adjoint in Sophus translation-first ordering:
//
//   Adj(T) = | R    t̂R |     t̂ = skew-symmetric matrix of translation t
//            | 0    R  |
//
// The dense 6n×6n Gauss-Newton normal equations H·δ = −b are solved with
// Eigen::LDLT.  Node 0 is anchored by a large prior on H[0:6,0:6] to remove
// the gauge freedom.
//
// For classroom scenarios (~50–100 keyframes) the dense solver is accurate and
// requires no extra dependencies.  For large-scale graphs a sparse solver
// (Eigen::SparseMatrix or g2o) would be substituted here without changing
// any other code.
// ---------------------------------------------------------------------------

#include "lidar_icp_odometry/pose_graph.hpp"
#include <Eigen/Dense>

namespace lidar_icp {

class PoseGraphOptimizer
{
public:
  /**
   * @param max_iterations  Maximum Gauss-Newton outer iterations.
   * @param tolerance       Stop when max absolute update element < tol.
   */
  explicit PoseGraphOptimizer(int max_iterations = 20, double tolerance = 1e-4);

  /**
   * @brief Optimise in place — updates PoseGraphNode::T_world for all nodes.
   *
   * After the call, graph.nodes[i].T_world holds the optimised pose T*_i.
   * The input edge set is not modified; call this again if more edges are
   * added (incremental PGO pattern).
   *
   * @return Number of Gauss-Newton iterations executed.
   */
  int optimise(PoseGraph & graph) const;

private:
  /**
   * @brief 6×6 SE(3) adjoint in Sophus translation-first ordering.
   *
   *   Adj(T) = | R    t̂·R |
   *            | 0₃   R   |
   *
   * where t̂ is the 3×3 skew-symmetric matrix of T.translation().
   */
  static Eigen::Matrix<double, 6, 6> adjoint(const Sophus::SE3d & T);

  int    max_iterations_;
  double tolerance_;
};

}  // namespace lidar_icp
