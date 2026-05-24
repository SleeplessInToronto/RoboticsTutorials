#pragma once

#include <memory>
#include <mutex>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose_array.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <geometry_msgs/msg/vector3_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <tf2_ros/transform_broadcaster.h>

#include "ekf_se3_localization/ekf_se3.hpp"

namespace ekf_se3 {

class EkfNode : public rclcpp::Node
{
public:
  explicit EkfNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());

private:
  void declare_and_load_params();

  void cmd_vel_callback(const geometry_msgs::msg::Twist::ConstSharedPtr msg);
  void beacons_detected_callback(const geometry_msgs::msg::PoseArray::SharedPtr msg);
  void ground_truth_callback(const nav_msgs::msg::Odometry::SharedPtr msg);
  void odom_callback(const nav_msgs::msg::Odometry::SharedPtr msg);
  void prediction_timer_callback();

  void publish_all();

  // EKF core
  std::unique_ptr<EkfSe3> ekf_;

  // ROS interfaces
  rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr          cmd_vel_sub_;
  rclcpp::Subscription<geometry_msgs::msg::PoseArray>::SharedPtr     beacons_sub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr           gt_sub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr           odom_sub_;
  rclcpp::Publisher<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr pose_pub_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr                  path_pub_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr                  gt_path_pub_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr                  odom_path_pub_;
  rclcpp::Publisher<geometry_msgs::msg::Vector3Stamped>::SharedPtr   error_pub_;
  rclcpp::TimerBase::SharedPtr                                       pred_timer_;
  std::unique_ptr<tf2_ros::TransformBroadcaster>                     tf_broadcaster_;

  // State (all guarded by ekf_mutex_)
  std::mutex       ekf_mutex_;
  Vector6d         last_u_{Vector6d::Zero()};
  rclcpp::Time     last_pred_time_;
  bool             ekf_initialized_{false};

  // Ground truth (for error computation only, not fed to filter)
  Sophus::SE3d     gt_pose_;
  bool             gt_received_{false};

  // Beacon map (world frame)
  std::vector<Eigen::Vector3d> beacon_map_;

  // Path history
  nav_msgs::msg::Path ekf_path_;
  nav_msgs::msg::Path gt_path_;
  nav_msgs::msg::Path odom_path_;

  // Params
  std::string map_frame_;
  std::string base_link_frame_;
  Matrix6d    P0_;
  double      ransac_inlier_thresh_{0.5};
};

}  // namespace ekf_se3
