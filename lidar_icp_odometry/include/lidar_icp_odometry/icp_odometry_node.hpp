#pragma once

#include <memory>
#include <mutex>
#include <vector>
#include <string>

#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <tf2_ros/transform_broadcaster.h>

#include "lidar_icp_odometry/icp_solver.hpp"

namespace lidar_icp {

/**
 * @brief ROS2 node: scan-to-scan Point-to-Plane ICP LiDAR odometry.
 *
 * Pipeline per /lidar/points callback (SLAM Handbook §8.2.2):
 *  1. Parse PointCloud2 → std::vector<Vector3d>   (PointCloud2ConstIterator)
 *  2. Voxel-grid downsample                        (std::unordered_map)
 *  3. Estimate per-point normals                   (nanoflann k-NN + Eigen PCA)
 *  4. IcpSolver::align(current, previous)  →  ΔT ∈ SE(3)
 *  5. T_world ← T_world · ΔT              (pose accumulation on manifold)
 *  6. so3().normalize()                    (quaternion drift guard)
 *  7. Publish /icp/odom, /icp/path, TF odom → base_link
 *
 * The previous scan's PointNormal vector and its nanoflann KD-tree are rebuilt
 * by IcpSolver::align() each call; no stale state is kept in the node.
 */
class IcpOdometryNode : public rclcpp::Node
{
public:
  explicit IcpOdometryNode(
    const rclcpp::NodeOptions & options = rclcpp::NodeOptions());

private:
  void declare_and_load_params();

  void cloud_callback(
    const sensor_msgs::msg::PointCloud2::ConstSharedPtr & msg);

  void publish_odometry(const rclcpp::Time & stamp);

  // --- ICP solver (pure math, no ROS) ---
  std::unique_ptr<IcpSolver> icp_solver_;

  // --- ROS interfaces ---
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_sub_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr          odom_pub_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr              path_pub_;
  std::unique_ptr<tf2_ros::TransformBroadcaster>                 tf_broadcaster_;

  // --- State (mutex-guarded) ---
  std::mutex             state_mutex_;
  Sophus::SE3d           T_world_;       ///< Accumulated world pose T_world_lidar
  bool                   first_scan_{true};

  // --- Previous scan kept for next align() call ---
  std::vector<PointNormal> prev_normals_;

  // --- Path history ---
  nav_msgs::msg::Path path_;

  // --- Parameters ---
  double      voxel_leaf_size_{0.3};
  int         normal_k_neighbors_{20};
  int         max_iterations_{50};
  double      convergence_tolerance_{1.0e-4};
  double      max_correspondence_distance_{1.0};
  std::string odom_frame_{"odom"};
  std::string base_link_frame_{"base_link"};
  double      min_point_z_{-0.5};  ///< [m] sensor-frame z cutoff — drops ground returns
};

}  // namespace lidar_icp
