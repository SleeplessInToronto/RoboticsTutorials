#include "ekf_se3_localization/ekf_node.hpp"

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>

namespace ekf_se3 {

// ---------------------------------------------------------------------------
EkfNode::EkfNode(const rclcpp::NodeOptions & options)
: Node("ekf_node", options)
{
  declare_and_load_params();

  cmd_vel_sub_ = create_subscription<geometry_msgs::msg::Twist>(
    "/cmd_vel", 10,
    std::bind(&EkfNode::cmd_vel_callback, this, std::placeholders::_1));

  beacons_sub_ = create_subscription<geometry_msgs::msg::PoseArray>(
    "/beacons_detected", 10,
    std::bind(&EkfNode::beacons_detected_callback, this, std::placeholders::_1));

  gt_sub_ = create_subscription<nav_msgs::msg::Odometry>(
    "/ground_truth/odom", 10,
    std::bind(&EkfNode::ground_truth_callback, this, std::placeholders::_1));

  odom_sub_ = create_subscription<nav_msgs::msg::Odometry>(
    "/odom_noisy", 10,
    std::bind(&EkfNode::odom_callback, this, std::placeholders::_1));

  pose_pub_      = create_publisher<geometry_msgs::msg::PoseWithCovarianceStamped>("/ekf/pose", 10);
  path_pub_      = create_publisher<nav_msgs::msg::Path>("/ekf/path", 10);
  gt_path_pub_   = create_publisher<nav_msgs::msg::Path>("/ground_truth/path", 10);
  odom_path_pub_ = create_publisher<nav_msgs::msg::Path>("/odom/path", 10);
  error_pub_     = create_publisher<geometry_msgs::msg::Vector3Stamped>("/ekf/error", 10);

  tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

  const double rate = get_parameter("prediction_rate").as_double();
  pred_timer_ = create_wall_timer(
    std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::duration<double>(1.0 / rate)),
    std::bind(&EkfNode::prediction_timer_callback, this));

  ekf_path_.header.frame_id  = map_frame_;
  gt_path_.header.frame_id   = map_frame_;
  odom_path_.header.frame_id = map_frame_;
  last_pred_time_ = now();

  RCLCPP_INFO(get_logger(), "EKF SE(3) node ready at %.0f Hz", rate);
}

// ---------------------------------------------------------------------------
void EkfNode::declare_and_load_params()
{
  declare_parameter<double>("prediction_rate", 50.0);
  declare_parameter<std::vector<double>>("Q_diag",
    {0.01, 0.01, 0.001, 0.001, 0.001, 0.005});
  declare_parameter<std::vector<double>>("R_diag", {0.05, 0.05, 0.05});
  declare_parameter<std::vector<double>>("init_translation", {0.0, 0.0, 0.0});
  declare_parameter<std::vector<double>>("init_quaternion",  {0.0, 0.0, 0.0, 1.0});
  declare_parameter<std::vector<double>>("P0_diag",
    {1.0, 1.0, 0.1, 0.1, 0.1, 0.5});
  declare_parameter<std::string>("map_frame",       "map");
  declare_parameter<std::string>("base_link_frame", "base_link");
  declare_parameter<double>("ransac_inlier_thresh", 0.5);
  declare_parameter<std::vector<double>>("beacon_positions", {});

  map_frame_             = get_parameter("map_frame").as_string();
  base_link_frame_       = get_parameter("base_link_frame").as_string();
  ransac_inlier_thresh_  = get_parameter("ransac_inlier_thresh").as_double();

  auto diag_to_matrix6 = [](const std::vector<double> & v) {
    Matrix6d M = Matrix6d::Zero();
    for (int i = 0; i < 6; ++i) M(i, i) = v[i];
    return M;
  };

  const Matrix6d Q = diag_to_matrix6(get_parameter("Q_diag").as_double_array());
  P0_ = diag_to_matrix6(get_parameter("P0_diag").as_double_array());

  const auto r = get_parameter("R_diag").as_double_array();
  Eigen::Matrix3d R = Eigen::Matrix3d::Zero();
  for (int i = 0; i < 3; ++i) R(i, i) = r[i];

  const auto t = get_parameter("init_translation").as_double_array();
  const auto q = get_parameter("init_quaternion").as_double_array();
  Eigen::Quaterniond quat(q[3], q[0], q[1], q[2]);
  quat.normalize();
  const Sophus::SE3d X0(quat, Eigen::Vector3d(t[0], t[1], t[2]));

  ekf_ = std::make_unique<EkfSe3>(Q, R, X0, P0_);
  ekf_initialized_ = true;

  // Load beacon map: flat list [x0,y0,z0, x1,y1,z1, ...]
  const auto bv = get_parameter("beacon_positions").as_double_array();
  for (size_t i = 0; i + 2 < bv.size(); i += 3) {
    beacon_map_.emplace_back(bv[i], bv[i + 1], bv[i + 2]);
  }
  RCLCPP_INFO(get_logger(), "Loaded %zu beacons", beacon_map_.size());
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
void EkfNode::cmd_vel_callback(const geometry_msgs::msg::Twist::ConstSharedPtr msg)
{
  std::lock_guard<std::mutex> lk(ekf_mutex_);
  last_u_ << msg->linear.x, 0.0, 0.0, 0.0, 0.0, msg->angular.z;
}

// ---------------------------------------------------------------------------
void EkfNode::beacons_detected_callback(
  const geometry_msgs::msg::PoseArray::SharedPtr msg)
{
  if (!ekf_initialized_ || !gt_received_) return;
  std::lock_guard<std::mutex> lk(ekf_mutex_);

  // Collect valid detections (orientation.w == 0 → not detected this scan).
  struct Detection { size_t k; Eigen::Vector3d y_k; };
  std::vector<Detection> detections;
  const size_t n = std::min(msg->poses.size(), beacon_map_.size());
  for (size_t k = 0; k < n; ++k) {
    if (msg->poses[k].orientation.w == 0.0) continue;
    detections.push_back({k, {msg->poses[k].position.x,
                               msg->poses[k].position.y,
                               msg->poses[k].position.z}});
  }
  if (detections.empty()) return;

  // RANSAC outlier rejection.
  // For each detection i, compute the hypothetical pose that a single-beacon
  // Kalman correction would produce, then count how many other detections j
  // are consistent (residual < inlier threshold) at that hypothesis.
  // Apply EKF corrections only for the largest inlier set.
  std::vector<size_t> best_inliers;
  for (size_t i = 0; i < detections.size(); ++i) {
    const Sophus::SE3d X_hyp = ekf_->hypothetical_pose(
      beacon_map_[detections[i].k], detections[i].y_k);

    std::vector<size_t> inliers;
    for (size_t j = 0; j < detections.size(); ++j) {
      const Eigen::Vector3d z_j =
        detections[j].y_k - X_hyp.inverse() * beacon_map_[detections[j].k];
      if (z_j.norm() < ransac_inlier_thresh_) {
        inliers.push_back(j);
      }
    }
    if (inliers.size() > best_inliers.size()) {
      best_inliers = inliers;
    }
  }

  RCLCPP_DEBUG(get_logger(), "RANSAC: %zu/%zu beacons accepted",
    best_inliers.size(), detections.size());

  for (size_t idx : best_inliers) {
    const auto & d = detections[idx];
    ekf_->correct(beacon_map_[d.k], d.y_k);
  }

  publish_all();
}

// ---------------------------------------------------------------------------
void EkfNode::ground_truth_callback(const nav_msgs::msg::Odometry::SharedPtr msg)
{
  const auto & p = msg->pose.pose.position;
  const auto & o = msg->pose.pose.orientation;
  Eigen::Quaterniond q(o.w, o.x, o.y, o.z);
  q.normalize();
  {
    std::lock_guard<std::mutex> lk(ekf_mutex_);
    gt_pose_ = Sophus::SE3d(q, Eigen::Vector3d(p.x, p.y, p.z));
    if (!gt_received_) {
      ekf_->reset_pose(gt_pose_, P0_);
      gt_received_ = true;
    }
  }
  geometry_msgs::msg::PoseStamped ps;
  ps.header.stamp    = msg->header.stamp;
  ps.header.frame_id = map_frame_;
  ps.pose = msg->pose.pose;
  gt_path_.header.stamp = msg->header.stamp;
  gt_path_.poses.push_back(ps);
  gt_path_pub_->publish(gt_path_);
}

// ---------------------------------------------------------------------------
void EkfNode::odom_callback(const nav_msgs::msg::Odometry::SharedPtr msg)
{
  geometry_msgs::msg::PoseStamped ps;
  ps.header.stamp    = msg->header.stamp;
  ps.header.frame_id = map_frame_;
  ps.pose = msg->pose.pose;
  odom_path_.header.stamp = msg->header.stamp;
  odom_path_.poses.push_back(ps);
  odom_path_pub_->publish(odom_path_);
}

// ---------------------------------------------------------------------------
void EkfNode::prediction_timer_callback()
{
  if (!ekf_initialized_) return;
  std::lock_guard<std::mutex> lk(ekf_mutex_);

  const rclcpp::Time now_t = now();
  const double dt = (now_t - last_pred_time_).seconds();
  last_pred_time_ = now_t;  // always advance clock so first real dt is ~20 ms

  // Don't predict until Gazebo has started publishing (avoids 5-second
  // dead-reckoning gap between node launch and physics start).
  if (!gt_received_) return;
  if (dt <= 0.0 || dt > 1.0) return;

  ekf_->predict(last_u_, dt);
  publish_all();
}

// ---------------------------------------------------------------------------
void EkfNode::publish_all()
{
  // Called while ekf_mutex_ is held.
  const rclcpp::Time stamp = now();
  const Sophus::SE3d & X   = ekf_->pose();
  const Matrix6d &     P   = ekf_->covariance();

  const Eigen::Vector3d    t = X.translation();
  const Eigen::Quaterniond q(X.rotationMatrix());

  // --- PoseWithCovarianceStamped ---
  geometry_msgs::msg::PoseWithCovarianceStamped pose_msg;
  pose_msg.header.stamp    = stamp;
  pose_msg.header.frame_id = map_frame_;
  pose_msg.pose.pose.position.x    = t.x();
  pose_msg.pose.pose.position.y    = t.y();
  pose_msg.pose.pose.position.z    = t.z();
  pose_msg.pose.pose.orientation.x = q.x();
  pose_msg.pose.pose.orientation.y = q.y();
  pose_msg.pose.pose.orientation.z = q.z();
  pose_msg.pose.pose.orientation.w = q.w();
  for (int r = 0; r < 6; ++r)
    for (int c = 0; c < 6; ++c)
      pose_msg.pose.covariance[r * 6 + c] = P(r, c);
  pose_pub_->publish(pose_msg);

  // --- Path ---
  geometry_msgs::msg::PoseStamped ps;
  ps.header = pose_msg.header;
  ps.pose   = pose_msg.pose.pose;
  ekf_path_.header.stamp = stamp;
  ekf_path_.poses.push_back(ps);
  path_pub_->publish(ekf_path_);

  // --- TF map → base_link ---
  geometry_msgs::msg::TransformStamped tf_msg;
  tf_msg.header.stamp    = stamp;
  tf_msg.header.frame_id = map_frame_;
  tf_msg.child_frame_id  = base_link_frame_;
  tf_msg.transform.translation.x = t.x();
  tf_msg.transform.translation.y = t.y();
  tf_msg.transform.translation.z = t.z();
  tf_msg.transform.rotation.x = q.x();
  tf_msg.transform.rotation.y = q.y();
  tf_msg.transform.rotation.z = q.z();
  tf_msg.transform.rotation.w = q.w();
  tf_broadcaster_->sendTransform(tf_msg);

  // --- Error vs ground truth ---
  if (!gt_received_) return;
  const Eigen::Vector3d t_gt  = gt_pose_.translation();
  const double pos_err = (t - t_gt).norm();

  // Yaw error: extract yaw from both rotations
  auto yaw_from_rot = [](const Sophus::SE3d & X_) {
    const Eigen::Matrix3d R = X_.rotationMatrix();
    return std::atan2(R(1, 0), R(0, 0));
  };
  const double dyaw = yaw_from_rot(X) - yaw_from_rot(gt_pose_);
  const double yaw_err = std::atan2(std::sin(dyaw), std::cos(dyaw));

  geometry_msgs::msg::Vector3Stamped err_msg;
  err_msg.header.stamp    = stamp;
  err_msg.header.frame_id = map_frame_;
  err_msg.vector.x = pos_err;
  err_msg.vector.y = yaw_err;
  err_msg.vector.z = 0.0;
  error_pub_->publish(err_msg);
}

}  // namespace ekf_se3

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ekf_se3::EkfNode>());
  rclcpp::shutdown();
  return 0;
}
