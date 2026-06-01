#pragma once

// ---------------------------------------------------------------------------
// SlamNode — full LiDAR SLAM pipeline (§8.4 of the SLAM Handbook)
//
// Extends the raw ICP odometry pipeline (§8.2) with:
//   • Keyframe selection  — add a new graph node every ~1 m or ~17°
//   • Scan Context        — §8.3.2.2 ring-key pre-filter + full SC distance
//   • ICP loop verification — re-align query ↔ candidate to reject false matches
//   • Gauss-Newton PGO    — §8.4.2 in-place optimisation on the pose graph
//   • Drift correction    — between keyframes, propagate raw ICP delta from
//                           the optimised last-keyframe pose
//
// Published topics
//   /slam/odom   nav_msgs/Odometry  — PGO-corrected pose
//   /icp/odom    nav_msgs/Odometry  — raw ICP odometry (for drift comparison)
//   /slam/path   nav_msgs/Path
//   /icp/path    nav_msgs/Path
//   TF: odom → base_link  (SLAM-corrected)
// ---------------------------------------------------------------------------

#include <memory>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <visualization_msgs/msg/marker_array.hpp>
#include <std_msgs/msg/header.hpp>

#include "lidar_icp_odometry/icp_solver.hpp"
#include "lidar_icp_odometry/scan_context.hpp"
#include "lidar_icp_odometry/pose_graph.hpp"
#include "lidar_icp_odometry/pose_graph_optimizer.hpp"

namespace lidar_icp {

class SlamNode : public rclcpp::Node
{
public:
  explicit SlamNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());

private:
  // ---------------------------------------------------------------------------
  // Callbacks
  // ---------------------------------------------------------------------------
  void lidar_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg);

  // ---------------------------------------------------------------------------
  // Internal pipeline steps
  // ---------------------------------------------------------------------------

  /// Add a new keyframe to the pose graph (sequential odometry edge + SC descriptor).
  void add_keyframe(const rclcpp::Time      & stamp,
                    const Sophus::SE3d      & T_world,
                    std::vector<Vector3d>     pts,
                    std::vector<PointNormal>  normals);

  /// Two-stage loop-closure search (ring-key → SC distance → ICP verify).
  /// Returns true if a loop-closure edge was added to the graph.
  bool detect_and_add_loop_closure(int query_id);

  /// Update the drift-correction transform after PGO.
  void apply_pgo_correction();

  // ---------------------------------------------------------------------------
  // Publishing helpers
  // ---------------------------------------------------------------------------
  void publish_slam_odom(const rclcpp::Time & stamp, const Sophus::SE3d & T);
  void publish_raw_odom (const rclcpp::Time & stamp, const Sophus::SE3d & T);
  void publish_graph_markers(const rclcpp::Time & stamp);
  void publish_map(const rclcpp::Time & stamp);

  // ---------------------------------------------------------------------------
  // ICP solver (shared for both odometry and loop-closure verification)
  // ---------------------------------------------------------------------------
  std::unique_ptr<IcpSolver> solver_;

  // ---------------------------------------------------------------------------
  // Pose graph
  // ---------------------------------------------------------------------------
  PoseGraph          graph_;
  PoseGraphOptimizer optimizer_;

  // ---------------------------------------------------------------------------
  // Running state
  // ---------------------------------------------------------------------------
  bool          first_scan_{true};
  Sophus::SE3d  T_world_raw_;   ///< Raw ICP odometry (no loop closure)

  // Previous scan kept for scan-to-scan ICP
  std::vector<Vector3d>    prev_pts_;
  std::vector<PointNormal> prev_normals_;

  // ---------------------------------------------------------------------------
  // Keyframe tracking
  // ---------------------------------------------------------------------------
  double       keyframe_min_dist_;   ///< [m] — min translation since last KF
  double       keyframe_min_angle_;  ///< [rad] — min rotation since last KF
  Sophus::SE3d T_last_keyframe_;     ///< Raw pose when last KF was added

  // ---------------------------------------------------------------------------
  // Drift correction (applied between keyframes)
  //
  //   T_corrected = T_last_kf_opt · T_last_kf_raw^{-1} · T_world_raw
  //
  // T_last_kf_opt  : optimised pose of the last keyframe (updated after PGO)
  // T_last_kf_raw  : raw ICP pose at the last keyframe   (stored when added)
  // T_world_raw    : current raw ICP accumulation
  // ---------------------------------------------------------------------------
  Sophus::SE3d T_last_kf_opt_;
  Sophus::SE3d T_last_kf_raw_;

  // ---------------------------------------------------------------------------
  // Loop-closure parameters
  // ---------------------------------------------------------------------------
  double sc_dist_threshold_;     ///< Accept if SC distance < this
  double ring_key_threshold_;    ///< Pre-filter: skip if ring-key dist > this
  int    min_loop_gap_;          ///< Min node-index gap before testing
  int    lc_icp_max_iter_;       ///< ICP max iterations for LC verification
  double lc_icp_max_corr_dist_;  ///< ICP max correspondence distance [m]
  double lc_icp_rmse_threshold_; ///< Reject LC if ICP RMSE > this [m]
  double lc_cycle_t_threshold_;  ///< Eq.(3.6): reject if translation cycle error > this [m]
  double lc_cycle_R_threshold_;  ///< Eq.(3.6): reject if rotation cycle error > this [rad]

  // ---------------------------------------------------------------------------
  // Pre-processing parameters
  // ---------------------------------------------------------------------------
  double      voxel_leaf_;
  int         normal_k_;
  double      min_point_z_;  ///< [m] sensor-frame z cutoff — drops ground returns
  std::string odom_frame_;
  std::string base_link_frame_;

  // ---------------------------------------------------------------------------
  // ROS interfaces
  // ---------------------------------------------------------------------------
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr lidar_sub_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr              slam_odom_pub_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr              raw_odom_pub_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr                  slam_path_pub_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr                  raw_path_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr graph_marker_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr        map_pub_;
  rclcpp::Publisher<std_msgs::msg::Header>::SharedPtr                loop_closure_pub_;
  std::unique_ptr<tf2_ros::TransformBroadcaster>                     tf_broadcaster_;

  nav_msgs::msg::Path slam_path_;
  nav_msgs::msg::Path raw_path_;
};

}  // namespace lidar_icp
