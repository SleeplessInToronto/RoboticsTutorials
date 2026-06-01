#pragma once

// ---------------------------------------------------------------------------
// Pose-graph data structures — §8.4.2 of the SLAM Handbook.
//
// A pose graph encodes the SLAM problem as a factor graph:
//
//   Nodes  — keyframe poses T_i ∈ SE(3)
//   Edges  — relative-transform measurements T_ij (odometry or loop closure)
//            each weighted by a 6×6 information matrix Ω_ij = Σ_ij^{−1}
//
// The optimiser minimises
//
//   C = ½ Σ_{(i,j)}  e_ij^T  Ω_ij  e_ij
//
// where  e_ij = Log( T_ij^{−1} · T_i^{−1} · T_j )  ∈ ℝ⁶.
//
// This header is pure data — no ROS, no Eigen solvers.  All optimisation
// logic lives in PoseGraphOptimizer.
// ---------------------------------------------------------------------------

#include <vector>
#include <Eigen/Dense>
#include <sophus/se3.hpp>

#include "lidar_icp_odometry/icp_solver.hpp"    // Vector3d, PointNormal
#include "lidar_icp_odometry/scan_context.hpp"  // ScanContext::Descriptor, RingKey

namespace lidar_icp {

// ---------------------------------------------------------------------------
/// One keyframe node in the pose graph.
// ---------------------------------------------------------------------------
struct PoseGraphNode {
  int          id;          ///< Sequential identifier (0-based)
  Sophus::SE3d T_world;     ///< Estimated world pose T_world_sensor at keyframe time

  // Point cloud and normals retained for loop-closure ICP verification
  std::vector<Vector3d>    pts;      ///< Downsampled points (sensor frame)
  std::vector<PointNormal> normals;  ///< Estimated surface normals

  // Scan Context fields (§8.3.2.2)
  ScanContext::Descriptor  sc_desc;  ///< Full N_RINGS × N_SECTORS descriptor
  ScanContext::RingKey     ring_key; ///< N_RINGS-dim summary for fast rejection
};

// ---------------------------------------------------------------------------
/// One edge in the pose graph.
// ---------------------------------------------------------------------------
struct PoseGraphEdge {
  int          from;            ///< Index of the "from" node i
  int          to;              ///< Index of the "to"   node j
  Sophus::SE3d T_from_to;       ///< Measured transform: T_i^{−1} · T_j
  Eigen::Matrix<double, 6, 6> information; ///< Ω = Σ^{−1}, precision matrix
  bool         is_loop_closure; ///< True for loop-closure edges
};

// ---------------------------------------------------------------------------
/// Lightweight container — add nodes/edges, then hand to PoseGraphOptimizer.
// ---------------------------------------------------------------------------
struct PoseGraph {
  std::vector<PoseGraphNode> nodes;
  std::vector<PoseGraphEdge> edges;
};

}  // namespace lidar_icp
