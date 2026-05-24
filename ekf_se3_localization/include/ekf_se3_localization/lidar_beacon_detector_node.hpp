#pragma once

#include <mutex>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose_array.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <Eigen/Dense>
#include <sophus/se3.hpp>

namespace ekf_se3 {

/**
 * Extracts beacon centroids from a 3D LiDAR point cloud.
 *
 * For each known beacon b_k (world frame), the node predicts its body-frame
 * location using the EKF estimate (falling back to /odom if unavailable),
 * collects cloud points within cluster_radius, and publishes the centroid.
 *
 * Using the EKF estimate instead of raw odometry avoids drift-induced
 * mismatch between the predicted and true beacon body-frame position.
 */
class LidarBeaconDetectorNode : public rclcpp::Node
{
public:
  explicit LidarBeaconDetectorNode(
    const rclcpp::NodeOptions & options = rclcpp::NodeOptions());

private:
  void odom_callback(const nav_msgs::msg::Odometry::SharedPtr msg);
  void ekf_callback(
    const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr msg);
  void gt_callback(const nav_msgs::msg::Odometry::SharedPtr msg);
  void cloud_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg);

  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr                        odom_sub_;
  rclcpp::Subscription<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr  ekf_sub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr                        gt_sub_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr                  cloud_sub_;
  rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr                     beacons_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr                     cluster_pub_;

  std::mutex         odom_mutex_;
  Sophus::SE3d       odom_pose_;
  bool               odom_received_{false};

  std::mutex         ekf_mutex_;
  Sophus::SE3d       ekf_pose_;
  bool               ekf_received_{false};

  // Ground truth pose — used for data association to bootstrap EKF.
  // The GT pose is only used to predict where to look in the cloud;
  // the actual measurement fed to the EKF is always the raw LiDAR centroid.
  std::mutex         gt_mutex_;
  Sophus::SE3d       gt_pose_;
  bool               gt_received_{false};

  std::vector<Eigen::Vector3d> beacon_map_;
  double   cluster_radius_{2.0};
  int      min_cluster_pts_{1};
  std::string base_link_frame_{"base_link"};
};

}  // namespace ekf_se3
