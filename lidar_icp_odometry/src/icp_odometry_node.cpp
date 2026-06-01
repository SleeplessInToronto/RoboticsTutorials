#include "lidar_icp_odometry/icp_odometry_node.hpp"

#include <geometry_msgs/msg/transform_stamped.hpp>
#include <tf2_eigen/tf2_eigen.hpp>

namespace lidar_icp {

// ---------------------------------------------------------------------------
IcpOdometryNode::IcpOdometryNode(const rclcpp::NodeOptions & options)
: Node("icp_odometry_node", options)
{
  declare_and_load_params();

  icp_solver_ = std::make_unique<IcpSolver>(
    max_iterations_,
    convergence_tolerance_,
    max_correspondence_distance_);

  // Subscriber — raw LiDAR point cloud
  cloud_sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
    "/lidar/points",
    rclcpp::SensorDataQoS(),
    [this](const sensor_msgs::msg::PointCloud2::ConstSharedPtr & msg) {
      cloud_callback(msg);
    });

  odom_pub_ = create_publisher<nav_msgs::msg::Odometry>("/icp/odom", 10);
  path_pub_ = create_publisher<nav_msgs::msg::Path>("/icp/path", 10);
  tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

  path_.header.frame_id = odom_frame_;

  RCLCPP_INFO(get_logger(),
    "ICP odometry node ready | voxel=%.2f m | k=%d | max_iter=%d | "
    "tol=%.1e | max_corr=%.2f m",
    voxel_leaf_size_, normal_k_neighbors_, max_iterations_,
    convergence_tolerance_, max_correspondence_distance_);
}

// ---------------------------------------------------------------------------
void IcpOdometryNode::declare_and_load_params()
{
  declare_parameter("voxel_leaf_size",           voxel_leaf_size_);
  declare_parameter("normal_k_neighbors",        normal_k_neighbors_);
  declare_parameter("max_iterations",            max_iterations_);
  declare_parameter("convergence_tolerance",     convergence_tolerance_);
  declare_parameter("max_correspondence_distance", max_correspondence_distance_);
  declare_parameter("odom_frame",                odom_frame_);
  declare_parameter("base_link_frame",           base_link_frame_);
  declare_parameter("min_point_z",               min_point_z_);

  voxel_leaf_size_            = get_parameter("voxel_leaf_size").as_double();
  normal_k_neighbors_         = get_parameter("normal_k_neighbors").as_int();
  max_iterations_             = get_parameter("max_iterations").as_int();
  convergence_tolerance_      = get_parameter("convergence_tolerance").as_double();
  max_correspondence_distance_= get_parameter("max_correspondence_distance").as_double();
  odom_frame_                 = get_parameter("odom_frame").as_string();
  base_link_frame_            = get_parameter("base_link_frame").as_string();
  min_point_z_                = get_parameter("min_point_z").as_double();
}

// ---------------------------------------------------------------------------
void IcpOdometryNode::cloud_callback(
  const sensor_msgs::msg::PointCloud2::ConstSharedPtr & msg)
{
  std::lock_guard<std::mutex> lk(state_mutex_);

  // 1. Parse PointCloud2 → flat vector (PointCloud2ConstIterator, zero PCL)
  const auto raw_pts = IcpSolver::from_ros_msg(*msg);
  if (raw_pts.empty()) {
    RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000,
      "Received empty point cloud — skipping");
    return;
  }

  // Drop ground returns (sensor-frame z < min_point_z_)
  std::vector<lidar_icp::Vector3d> pts_filtered;
  pts_filtered.reserve(raw_pts.size());
  for (const auto & p : raw_pts)
    if (p.z() >= min_point_z_) pts_filtered.push_back(p);

  // 2. Voxel-grid downsample (std::unordered_map, zero PCL)
  auto curr_pts = IcpSolver::voxel_downsample(pts_filtered, voxel_leaf_size_);
  if (static_cast<int>(curr_pts.size()) < normal_k_neighbors_) {
    RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000,
      "Too few points after downsampling (%zu) — need at least %d",
      curr_pts.size(), normal_k_neighbors_);
    return;
  }

  // 3. Estimate per-point normals (nanoflann k-NN + Eigen PCA)
  auto curr_normals = IcpSolver::estimate_normals(curr_pts, normal_k_neighbors_);

  // 4. First scan bootstrap — store and return (nothing to align against yet)
  if (first_scan_) {
    prev_normals_ = std::move(curr_normals);
    first_scan_   = false;
    RCLCPP_INFO(get_logger(), "First scan stored (%zu pts after voxel grid). "
      "ICP will start on next scan.", curr_pts.size());
    return;
  }

  // 5. ICP: align current scan (source) → previous scan (target)
  //    prev_normals_ is the target; curr_pts is the source.
  const IcpResult res = icp_solver_->align(curr_pts, prev_normals_);

  RCLCPP_DEBUG(get_logger(),
    "ICP: iter=%d  rmse=%.4f  converged=%s  |xi|≈%.2e",
    res.iterations, res.rmse,
    res.converged ? "yes" : "no",
    res.delta_T.log().norm());

  // Accept the result if RMSE is below half the correspondence distance —
  // the alignment is good even if ‖ξ‖₂ hasn't dropped below tolerance yet.
  const bool good_rmse = res.rmse < max_correspondence_distance_ * 0.5;
  if (!res.converged && !good_rmse) {
    RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
      "ICP did not converge (rmse=%.4f, iters=%d) — skipping pose update",
      res.rmse, res.iterations);
    prev_normals_ = std::move(curr_normals);
    return;
  }
  if (!res.converged) {
    RCLCPP_DEBUG(get_logger(),
      "ICP tolerance not met but RMSE=%.4f m is acceptable — using result",
      res.rmse);
  }

  // Safety check: reject implausibly large increments (degenerate geometry)
  const double xi_norm = res.delta_T.log().norm();
  if (xi_norm > 2.0) {
    RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
      "ICP increment too large (‖ξ‖=%.2f) — skipping pose update", xi_norm);
    prev_normals_ = std::move(curr_normals);
    return;
  }

  // 6. Accumulate pose on SE(3) manifold (SLAM Handbook §8.2.2.4)
  //    res.delta_T = T_prev_curr  (maps current frame → previous frame)
  //    T_world = T_world · T_prev_curr
  T_world_ = T_world_ * res.delta_T;
  T_world_.so3().normalize();   // quaternion drift guard (same as ekf_se3.cpp)

  // 7. Publish
  publish_odometry(msg->header.stamp);

  // 8. Store current scan as reference for the next callback
  prev_normals_ = std::move(curr_normals);
}

// ---------------------------------------------------------------------------
void IcpOdometryNode::publish_odometry(const rclcpp::Time & stamp)
{
  const Eigen::Vector3d    t = T_world_.translation();
  const Eigen::Quaterniond q(T_world_.rotationMatrix());

  // --- nav_msgs/Odometry ---
  nav_msgs::msg::Odometry odom;
  odom.header.stamp    = stamp;
  odom.header.frame_id = odom_frame_;
  odom.child_frame_id  = base_link_frame_;
  odom.pose.pose.position.x    = t.x();
  odom.pose.pose.position.y    = t.y();
  odom.pose.pose.position.z    = t.z();
  odom.pose.pose.orientation.x = q.x();
  odom.pose.pose.orientation.y = q.y();
  odom.pose.pose.orientation.z = q.z();
  odom.pose.pose.orientation.w = q.w();
  odom_pub_->publish(odom);

  // --- nav_msgs/Path ---
  geometry_msgs::msg::PoseStamped ps;
  ps.header = odom.header;
  ps.pose   = odom.pose.pose;
  path_.header.stamp = stamp;
  path_.poses.push_back(ps);
  path_pub_->publish(path_);

  // --- TF: odom → base_link ---
  geometry_msgs::msg::TransformStamped tf_msg;
  tf_msg.header.stamp    = stamp;
  tf_msg.header.frame_id = odom_frame_;
  tf_msg.child_frame_id  = base_link_frame_;
  tf_msg.transform.translation.x = t.x();
  tf_msg.transform.translation.y = t.y();
  tf_msg.transform.translation.z = t.z();
  tf_msg.transform.rotation.x    = q.x();
  tf_msg.transform.rotation.y    = q.y();
  tf_msg.transform.rotation.z    = q.z();
  tf_msg.transform.rotation.w    = q.w();
  tf_broadcaster_->sendTransform(tf_msg);
}

}  // namespace lidar_icp

// ---------------------------------------------------------------------------
int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<lidar_icp::IcpOdometryNode>());
  rclcpp::shutdown();
  return 0;
}
