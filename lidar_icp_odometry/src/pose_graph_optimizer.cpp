#include "lidar_icp_odometry/pose_graph_optimizer.hpp"

#include <Eigen/Dense>
#include <sophus/se3.hpp>
#include <algorithm>
#include <cmath>

namespace lidar_icp {

PoseGraphOptimizer::PoseGraphOptimizer(int max_iterations, double tolerance)
  : max_iterations_(max_iterations), tolerance_(tolerance)
{}

// ---------------------------------------------------------------------------
// SE(3) adjoint — Sophus translation-first (ρ, θ) ordering.
//
//   Adj(T) = | R    t̂·R |
//            | 0₃   R   |
//
// Derivation: for ξ ∈ ℝ⁶ and T ∈ SE(3),
//   T · Exp(ξ) · T^{-1} = Exp(Adj(T) · ξ)
// This lets us "move" a right perturbation past T to become a left perturbation,
// which is the key step in deriving J_i in the Gauss-Newton system.
// ---------------------------------------------------------------------------
Eigen::Matrix<double, 6, 6>
PoseGraphOptimizer::adjoint(const Sophus::SE3d & T)
{
  const Eigen::Matrix3d     R = T.rotationMatrix();
  const Eigen::Vector3d   & t = T.translation();

  // Skew-symmetric matrix of t  (t̂)
  Eigen::Matrix3d t_hat;
  t_hat <<    0.0, -t.z(),  t.y(),
           t.z(),    0.0,  -t.x(),
          -t.y(),  t.x(),    0.0;

  Eigen::Matrix<double, 6, 6> adj = Eigen::Matrix<double, 6, 6>::Zero();
  adj.topLeftCorner<3, 3>()     = R;
  adj.topRightCorner<3, 3>()    = t_hat * R;
  adj.bottomRightCorner<3, 3>() = R;
  return adj;
}

// ---------------------------------------------------------------------------
// Gauss-Newton optimisation loop
// ---------------------------------------------------------------------------
int PoseGraphOptimizer::optimise(PoseGraph & graph) const
{
  using Matrix6d = Eigen::Matrix<double, 6, 6>;
  using Vector6d = Eigen::Matrix<double, 6, 1>;

  const int n = static_cast<int>(graph.nodes.size());
  if (n < 2) return 0;

  int iter = 0;
  for (; iter < max_iterations_; ++iter) {

    // -----------------------------------------------------------------------
    // Accumulate 6n×6n Gauss-Newton system   H·δ = −b
    //
    // For each edge (i→j):
    //   e   = Log( T_ij^{-1} · T_i^{-1} · T_j )
    //   J_j =  I₆
    //   J_i = −Adj( T_j^{-1} · T_i )   =  −Adj( T_pred^{-1} )
    //
    //   H_ii += J_i^T Ω J_i
    //   H_jj += J_j^T Ω J_j
    //   H_ij += J_i^T Ω J_j   (symmetrised)
    //   b_i  += J_i^T Ω e
    //   b_j  += J_j^T Ω e
    // -----------------------------------------------------------------------
    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(6 * n, 6 * n);
    Eigen::VectorXd b = Eigen::VectorXd::Zero(6 * n);

    for (const PoseGraphEdge & edge : graph.edges) {
      const int i = edge.from;
      const int j = edge.to;

      const Sophus::SE3d & Ti  = graph.nodes[i].T_world;
      const Sophus::SE3d & Tj  = graph.nodes[j].T_world;
      const Sophus::SE3d & Tij = edge.T_from_to;

      // Predicted relative transform and residual
      const Sophus::SE3d T_pred = Ti.inverse() * Tj;              // T_i^{-1}·T_j
      const Sophus::SE3d T_err  = Tij.inverse() * T_pred;        // T_ij^{-1}·T_pred
      const Vector6d     e      = T_err.log();                   // ∈ ℝ⁶, Sophus ordering

      // Jacobians
      const Matrix6d J_j =  Matrix6d::Identity();
      const Matrix6d J_i = -adjoint(T_pred.inverse());

      const Matrix6d & Omega = edge.information;

      // Pre-compute Omega*J products
      const Matrix6d OmegaJ_i = Omega * J_i;
      const Matrix6d OmegaJ_j = Omega * J_j;

      // Accumulate — skip node 0 (anchored)
      if (i != 0) {
        H.block<6, 6>(6 * i, 6 * i) += J_i.transpose() * OmegaJ_i;
        b.segment<6>(6 * i)          += J_i.transpose() * Omega * e;
      }
      if (j != 0) {
        H.block<6, 6>(6 * j, 6 * j) += J_j.transpose() * OmegaJ_j;
        b.segment<6>(6 * j)          += J_j.transpose() * Omega * e;
      }
      if (i != 0 && j != 0) {
        H.block<6, 6>(6 * i, 6 * j) += J_i.transpose() * OmegaJ_j;
        H.block<6, 6>(6 * j, 6 * i) += J_j.transpose() * OmegaJ_i;
      }
    }

    // -----------------------------------------------------------------------
    // Anchor node 0 — prior factor: effectively fixes the first pose
    // (removes gauge freedom = one SE(3) DOF = 6 DOF)
    // -----------------------------------------------------------------------
    H.block<6, 6>(0, 0) += Matrix6d::Identity() * 1e12;

    // -----------------------------------------------------------------------
    // Solve:  H·δ = −b   (note sign: gradient ascent on b, descent on −b)
    // -----------------------------------------------------------------------
    const Eigen::VectorXd delta = H.ldlt().solve(-b);

    // -----------------------------------------------------------------------
    // Retract:  T_i ← T_i · Exp(δ_i)
    // -----------------------------------------------------------------------
    double max_delta = 0.0;
    for (int k = 1; k < n; ++k) {
      const Vector6d xi = delta.segment<6>(6 * k);
      graph.nodes[k].T_world =
        graph.nodes[k].T_world * Sophus::SE3d::exp(xi);
      graph.nodes[k].T_world.so3().normalize();   // guard against SO(3) drift
      max_delta = std::max(max_delta, xi.cwiseAbs().maxCoeff());
    }

    if (max_delta < tolerance_) {
      ++iter;
      break;
    }
  }
  return iter;
}

}  // namespace lidar_icp
