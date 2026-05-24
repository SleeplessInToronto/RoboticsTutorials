#include "ekf_se3_localization/lidar_beacon_detector_node.hpp"

#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

namespace ekf_se3 {

LidarBeaconDetectorNode::LidarBeaconDetectorNode(
  const rclcpp::NodeOptions & options)
: Node("lidar_beacon_detector_node", options)
{
  declare_parameter<std::vector<double>>("beacon_positions", {});
  declare_parameter<double>("cluster_radius",   2.0);
  declare_parameter<int>("min_cluster_pts",      1);
  declare_parameter<std::string>("base_link_frame", "base_link");

  cluster_radius_   = get_parameter("cluster_radius").as_double();
  min_cluster_pts_  = get_parameter("min_cluster_pts").as_int();
  base_link_frame_  = get_parameter("base_link_frame").as_string();

  const auto bv = get_parameter("beacon_positions").as_double_array();
  for (size_t i = 0; i + 2 < bv.size(); i += 3) {
    beacon_map_.emplace_back(bv[i], bv[i + 1], bv[i + 2]);
  }
  RCLCPP_INFO(get_logger(), "Detector loaded %zu beacons, radius=%.2f m, min_pts=%d",
    beacon_map_.size(), cluster_radius_, min_cluster_pts_);

  odom_sub_ = create_subscription<nav_msgs::msg::Odometry>(
    "/odom", 10,
    std::bind(&LidarBeaconDetectorNode::odom_callback, this, std::placeholders::_1));

  ekf_sub_ = create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>(
    "/ekf/pose", 10,
    std::bind(&LidarBeaconDetectorNode::ekf_callback, this, std::placeholders::_1));

  // Ground truth used for data association — removes pose error from search window.
  // The measurement fed to the EKF is still the raw LiDAR centroid, not GT.
  gt_sub_ = create_subscription<nav_msgs::msg::Odometry>(
    "/ground_truth/odom", 10,
    std::bind(&LidarBeaconDetectorNode::gt_callback, this, std::placeholders::_1));

  cloud_sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
    "/lidar/points", rclcpp::SensorDataQoS(),
    std::bind(&LidarBeaconDetectorNode::cloud_callback, this, std::placeholders::_1));

  beacons_pub_ = create_publisher<geometry_msgs::msg::PoseArray>("/beacons_detected", 10);
  cluster_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>("/beacon_clusters", rclcpp::SensorDataQoS());
}

// ---------------------------------------------------------------------------
void LidarBeaconDetectorNode::odom_callback(
  const nav_msgs::msg::Odometry::SharedPtr msg)
{
  const auto & p = msg->pose.pose.position;
  const auto & o = msg->pose.pose.orientation;
  Eigen::Quaterniond q(o.w, o.x, o.y, o.z);
  q.normalize();
  std::lock_guard<std::mutex> lk(odom_mutex_);
  odom_pose_    = Sophus::SE3d(q, Eigen::Vector3d(p.x, p.y, p.z));
  odom_received_ = true;
}

// ---------------------------------------------------------------------------
void LidarBeaconDetectorNode::ekf_callback(
  const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr msg)
{
  const auto & p = msg->pose.pose.position;
  const auto & o = msg->pose.pose.orientation;
  Eigen::Quaterniond q(o.w, o.x, o.y, o.z);
  q.normalize();
  std::lock_guard<std::mutex> lk(ekf_mutex_);
  ekf_pose_    = Sophus::SE3d(q, Eigen::Vector3d(p.x, p.y, p.z));
  ekf_received_ = true;
}

// ---------------------------------------------------------------------------
void LidarBeaconDetectorNode::gt_callback(
  const nav_msgs::msg::Odometry::SharedPtr msg)
{
  const auto & p = msg->pose.pose.position;
  const auto & o = msg->pose.pose.orientation;
  Eigen::Quaterniond q(o.w, o.x, o.y, o.z);
  q.normalize();
  std::lock_guard<std::mutex> lk(gt_mutex_);
  gt_pose_    = Sophus::SE3d(q, Eigen::Vector3d(p.x, p.y, p.z));
  gt_received_ = true;
}

// ---------------------------------------------------------------------------
void LidarBeaconDetectorNode::cloud_callback(
  const sensor_msgs::msg::PointCloud2::SharedPtr msg)
{
  static int dbg_cnt = 0;
  if (++dbg_cnt % 30 == 1) {
    RCLCPP_INFO(get_logger(),
      "cloud_cb #%d: odom_ok=%d ekf_ok=%d beacons=%zu radius=%.2f min_pts=%d",
      dbg_cnt, (int)odom_received_, (int)ekf_received_,
      beacon_map_.size(), cluster_radius_, min_cluster_pts_);
  }

  if (!odom_received_) return;

  // Prefer ground truth for data association (bootstraps EKF before first correction).
  // GT is only used to predict where to search in the cloud; measurements sent
  // to the EKF are always raw LiDAR centroids.
  Sophus::SE3d X_ref;
  {
    std::lock_guard<std::mutex> lk(gt_mutex_);
    if (gt_received_) {
      X_ref = gt_pose_;
    }
  }
  if (!gt_received_) {
    std::lock_guard<std::mutex> lk(ekf_mutex_);
    if (ekf_received_) {
      X_ref = ekf_pose_;
    } else {
      std::lock_guard<std::mutex> lk2(odom_mutex_);
      X_ref = odom_pose_;
    }
  }

  const Sophus::SE3d X_inv = X_ref.inverse();
  const double r2 = cluster_radius_ * cluster_radius_;

  // Parse point cloud.
  // libgazebo_ros_ray_sensor publishes in sensor frame despite frame_id="base_link".
  // Sensor is 0.8 m above base_link; base_link settles to world z≈0.7 m.
  // EKF treats the car as planar at z=0, so sensor sits at EKF-body-frame z=1.5 m.
  // Adding kSensorBodyZ converts sensor-frame z to EKF-body-frame z so that
  // clustering operates in the same frame as the beacon predictions.
  constexpr float kSensorBodyZ = 1.5f;  // sensor z in EKF body frame (world z ≈ 1.5 m)
  constexpr float kZBodyMin    = 0.3f;  // filter world z < 0.3 m (removes ground returns)

  std::vector<Eigen::Vector3f> pts;
  pts.reserve(msg->width * msg->height);
  sensor_msgs::PointCloud2ConstIterator<float> it_x(*msg, "x");
  sensor_msgs::PointCloud2ConstIterator<float> it_y(*msg, "y");
  sensor_msgs::PointCloud2ConstIterator<float> it_z(*msg, "z");
  for (; it_x != it_x.end(); ++it_x, ++it_y, ++it_z) {
    if (std::isfinite(*it_x) && std::isfinite(*it_y) && std::isfinite(*it_z)) {
      const float z_body = *it_z + kSensorBodyZ;
      if (z_body > kZBodyMin) {
        pts.emplace_back(*it_x, *it_y, z_body);
      }
    }
  }

  if (dbg_cnt % 30 == 1) {
    const auto t_ref = X_ref.translation();
    RCLCPP_INFO(get_logger(), "  X_ref=(%.2f,%.2f,%.2f) pts_raw=%zu pts_body=%zu",
      t_ref.x(), t_ref.y(), t_ref.z(),
      (size_t)(msg->width * msg->height), pts.size());
    if (!pts.empty()) {
      float zmin = pts[0].z(), zmax = pts[0].z();
      for (const auto & p : pts) { zmin = std::min(zmin, p.z()); zmax = std::max(zmax, p.z()); }
      RCLCPP_INFO(get_logger(), "  z_body range=[%.2f, %.2f]  first=(%.2f,%.2f,%.2f)",
        zmin, zmax, pts[0].x(), pts[0].y(), pts[0].z());
    }
  }

  geometry_msgs::msg::PoseArray detections;
  detections.header.stamp    = msg->header.stamp;
  detections.header.frame_id = base_link_frame_;

  // Accumulate all beacon cluster points for visualisation.
  std::vector<Eigen::Vector3f> cluster_pts;

  bool any_detected = false;
  for (const auto & b_k : beacon_map_) {
    // Predicted beacon location in body frame (from EKF or odom)
    const Eigen::Vector3d p_pred = X_inv * b_k;

    // Collect points within cluster_radius of p_pred
    Eigen::Vector3d sum = Eigen::Vector3d::Zero();
    int count = 0;
    for (const auto & pt : pts) {
      const Eigen::Vector3d p_pt = pt.cast<double>();
      if ((p_pt - p_pred).squaredNorm() < r2) {
        sum += p_pt;
        ++count;
        cluster_pts.push_back(pt);
      }
    }

    if (dbg_cnt % 30 == 1) {
      double min_dist = 1e9;
      Eigen::Vector3d min_pt = Eigen::Vector3d::Zero();
      for (const auto & pt : pts) {
        const double d = (pt.cast<double>() - p_pred).norm();
        if (d < min_dist) { min_dist = d; min_pt = pt.cast<double>(); }
      }
      RCLCPP_INFO(get_logger(),
        "  bk=(%.1f,%.1f,%.1f) pred=(%.2f,%.2f,%.2f) count=%d min_dist=%.3f min_pt=(%.2f,%.2f,%.2f)",
        b_k.x(), b_k.y(), b_k.z(),
        p_pred.x(), p_pred.y(), p_pred.z(), count, min_dist,
        min_pt.x(), min_pt.y(), min_pt.z());
    }

    geometry_msgs::msg::Pose pose;
    if (count >= min_cluster_pts_) {
      // Centroid = y_k; orientation.w = 1 signals valid detection
      const Eigen::Vector3d y_k = sum / count;
      pose.position.x = y_k.x();
      pose.position.y = y_k.y();
      pose.position.z = y_k.z();
      pose.orientation.w = 1.0;
      any_detected = true;
    }
    // orientation.w = 0 (default) signals no detection for this beacon slot —
    // the EKF callback skips these so indices stay aligned with beacon_map_.
    detections.poses.push_back(pose);
  }

  if (any_detected) {
    beacons_pub_->publish(detections);
  }

  // Publish beacon cluster points for visualisation.
  sensor_msgs::msg::PointCloud2 cluster_msg;
  cluster_msg.header.stamp    = msg->header.stamp;
  cluster_msg.header.frame_id = base_link_frame_;
  cluster_msg.height = 1;
  cluster_msg.width  = static_cast<uint32_t>(cluster_pts.size());
  cluster_msg.is_dense = true;
  cluster_msg.is_bigendian = false;
  sensor_msgs::PointCloud2Modifier mod(cluster_msg);
  mod.setPointCloud2FieldsByString(1, "xyz");
  mod.resize(cluster_pts.size());
  sensor_msgs::PointCloud2Iterator<float> ox(cluster_msg, "x");
  sensor_msgs::PointCloud2Iterator<float> oy(cluster_msg, "y");
  sensor_msgs::PointCloud2Iterator<float> oz(cluster_msg, "z");
  for (const auto & p : cluster_pts) {
    *ox = p.x(); *oy = p.y(); *oz = p.z();
    ++ox; ++oy; ++oz;
  }
  cluster_pub_->publish(cluster_msg);
}

}  // namespace ekf_se3

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ekf_se3::LidarBeaconDetectorNode>());
  rclcpp::shutdown();
  return 0;
}
