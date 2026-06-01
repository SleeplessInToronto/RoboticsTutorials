#include "lidar_icp_odometry/slam_node.hpp"

#include <geometry_msgs/msg/transform_stamped.hpp>
#include <std_msgs/msg/header.hpp>
#include <cmath>
#include <algorithm>
#include <limits>

namespace lidar_icp {

// ===========================================================================
// Constructor — declare and load parameters, wire up pubs/subs
// ===========================================================================
SlamNode::SlamNode(const rclcpp::NodeOptions & options)
  : Node("slam_node", options)
{
  // ---- Pre-processing ----
  voxel_leaf_   = declare_parameter("voxel_leaf_size",           0.3);
  normal_k_     = declare_parameter("normal_k_neighbors",        20);
  min_point_z_  = declare_parameter("min_point_z",              -0.5);

  // ---- ICP solver ----
  const int    max_iter  = declare_parameter("max_iterations",              50);
  const double tol       = declare_parameter("convergence_tolerance",       1.0e-4);
  const double max_corr  = declare_parameter("max_correspondence_distance", 1.0);

  // ---- Keyframe selection ----
  keyframe_min_dist_  = declare_parameter("keyframe_min_dist",  1.5);   // [m]
  keyframe_min_angle_ = declare_parameter("keyframe_min_angle", 0.3);   // [rad]

  // ---- Loop closure (Scan Context) ----
  sc_dist_threshold_    = declare_parameter("sc_dist_threshold",     0.15);
  ring_key_threshold_   = declare_parameter("ring_key_threshold",    0.2);
  min_loop_gap_         = declare_parameter("min_loop_gap",          20);
  lc_icp_max_iter_      = declare_parameter("lc_icp_max_iter",       100);
  lc_icp_max_corr_dist_ = declare_parameter("lc_icp_max_corr_dist",  1.5);
  lc_icp_rmse_threshold_= declare_parameter("lc_icp_rmse_threshold", 0.3);
  lc_cycle_t_threshold_ = declare_parameter("lc_cycle_t_threshold",  3.0);
  lc_cycle_R_threshold_ = declare_parameter("lc_cycle_R_threshold",  0.3);

  // ---- Frame IDs ----
  odom_frame_      = declare_parameter("odom_frame",      std::string("odom"));
  base_link_frame_ = declare_parameter("base_link_frame", std::string("base_link"));

  // ---- Construct ICP solver ----
  solver_ = std::make_unique<IcpSolver>(max_iter, tol, max_corr);

  // ---- Pose-graph optimiser ----
  optimizer_ = PoseGraphOptimizer(20, 1e-4);

  // ---- Initialise poses ----
  T_world_raw_     = Sophus::SE3d();
  T_last_keyframe_ = Sophus::SE3d();
  T_last_kf_raw_   = Sophus::SE3d();
  T_last_kf_opt_   = Sophus::SE3d();

  // ---- Publishers ----
  slam_odom_pub_    = create_publisher<nav_msgs::msg::Odometry>("/slam/odom",   10);
  raw_odom_pub_     = create_publisher<nav_msgs::msg::Odometry>("/icp/odom",    10);
  slam_path_pub_    = create_publisher<nav_msgs::msg::Path>("/slam/path",       10);
  raw_path_pub_     = create_publisher<nav_msgs::msg::Path>("/icp/path",        10);
  graph_marker_pub_ = create_publisher<visualization_msgs::msg::MarkerArray>(
                        "/slam/graph", 10);
  map_pub_           = create_publisher<sensor_msgs::msg::PointCloud2>(
                        "/slam/map", rclcpp::SensorDataQoS());
  loop_closure_pub_  = create_publisher<std_msgs::msg::Header>(
                        "/slam/loop_closures", 10);
  tf_broadcaster_    = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

  slam_path_.header.frame_id = odom_frame_;
  raw_path_.header.frame_id  = odom_frame_;

  // ---- Subscriber ----
  lidar_sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
    "/lidar/points", 10,
    std::bind(&SlamNode::lidar_callback, this, std::placeholders::_1));

  RCLCPP_INFO(get_logger(),
    "SlamNode started: voxel=%.2f  normal_k=%d  kf_dist=%.1f m  kf_angle=%.2f rad",
    voxel_leaf_, normal_k_, keyframe_min_dist_, keyframe_min_angle_);
  RCLCPP_INFO(get_logger(),
    "  SC: dist_thr=%.3f  rk_thr=%.3f  min_gap=%d",
    sc_dist_threshold_, ring_key_threshold_, min_loop_gap_);
}

// ===========================================================================
// lidar_callback — main per-scan pipeline
// ===========================================================================
void SlamNode::lidar_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
{
  const rclcpp::Time stamp = msg->header.stamp;

  // ------------------------------------------------------------------
  // 1.  Parse → voxel downsample → estimate normals
  // ------------------------------------------------------------------
  const auto pts_raw = IcpSolver::from_ros_msg(*msg);

  // Drop ground returns — in sensor frame the floor is ~0.8 m below the lidar,
  // so z < min_point_z_ are floor hits that add nothing to x/y/yaw estimation.
  std::vector<Vector3d> pts_filtered;
  pts_filtered.reserve(pts_raw.size());
  for (const auto & p : pts_raw)
    if (p.z() >= min_point_z_) pts_filtered.push_back(p);

  auto pts = IcpSolver::voxel_downsample(pts_filtered, voxel_leaf_);

  if (static_cast<int>(pts.size()) < 50) {
    RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
      "Too few points after downsampling (%zu); skipping scan", pts.size());
    return;
  }
  auto normals = IcpSolver::estimate_normals(pts, normal_k_);

  // ------------------------------------------------------------------
  // 2.  Bootstrap on first scan
  // ------------------------------------------------------------------
  if (first_scan_) {
    first_scan_   = false;
    prev_pts_     = pts;
    prev_normals_ = normals;
    add_keyframe(stamp, T_world_raw_, pts, normals);
    publish_raw_odom(stamp, T_world_raw_);
    publish_slam_odom(stamp, T_world_raw_);
    return;
  }

  // ------------------------------------------------------------------
  // 3.  ICP: align current (source) to previous scan (target)
  //          source pts → target prev_normals_
  //          result.delta_T: T_prev_frame ← T_curr_frame
  //          accumulation: T_world_curr = T_world_prev · delta_T
  // ------------------------------------------------------------------
  const IcpResult result = solver_->align(pts, prev_normals_);

  if (!result.converged) {
    RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000,
      "ICP did not converge (iter=%d  RMSE=%.4f m)", result.iterations, result.rmse);
  }

  // ------------------------------------------------------------------
  // 4.  Accumulate raw odometry
  // ------------------------------------------------------------------
  T_world_raw_ = T_world_raw_ * result.delta_T;
  T_world_raw_.so3().normalize();

  // ------------------------------------------------------------------
  // 5.  Publish raw odometry
  // ------------------------------------------------------------------
  publish_raw_odom(stamp, T_world_raw_);

  // ------------------------------------------------------------------
  // 6.  Corrected (SLAM) pose: propagate PGO correction from last KF
  //
  //     T_corrected = T_last_kf_opt · T_last_kf_raw^{-1} · T_world_raw
  //
  //     The ratio  T_last_kf_raw^{-1} · T_world_raw  is the raw motion
  //     since the last keyframe.  Premultiplying by  T_last_kf_opt
  //     applies the PGO correction as if motion since the last KF is
  //     unbiased (a first-order approximation between loop closures).
  // ------------------------------------------------------------------
  Sophus::SE3d T_corrected = T_last_kf_opt_ * T_last_kf_raw_.inverse() * T_world_raw_;
  T_corrected.so3().normalize();
  publish_slam_odom(stamp, T_corrected);

  // ------------------------------------------------------------------
  // 7.  Advance previous-scan buffer
  // ------------------------------------------------------------------
  prev_pts_     = pts;
  prev_normals_ = normals;

  // ------------------------------------------------------------------
  // 8.  Keyframe check
  // ------------------------------------------------------------------
  const Sophus::SE3d delta = T_last_keyframe_.inverse() * T_world_raw_;
  const double dist        = delta.translation().norm();
  const double angle       = delta.so3().log().norm();

  if (dist >= keyframe_min_dist_ || angle >= keyframe_min_angle_) {
    add_keyframe(stamp, T_world_raw_, pts, normals);
  }
}

// ===========================================================================
// add_keyframe — create PoseGraphNode, odometry edge, try loop closure
// ===========================================================================
void SlamNode::add_keyframe(const rclcpp::Time      & stamp,
                             const Sophus::SE3d      & T_world,
                             std::vector<Vector3d>     pts,
                             std::vector<PointNormal>  normals)
{
  const int new_id = static_cast<int>(graph_.nodes.size());

  // ---- Build Scan Context descriptor ----
  const ScanContext::Descriptor sc  = ScanContext::compute(pts);
  const ScanContext::RingKey    rk  = ScanContext::ring_key(sc);

  // ---- Create node ----
  PoseGraphNode node;
  node.id       = new_id;
  node.T_world  = T_world;
  node.pts      = std::move(pts);
  node.normals  = std::move(normals);
  node.sc_desc  = sc;
  node.ring_key = rk;
  graph_.nodes.push_back(std::move(node));

  // ---- Sequential odometry edge (from previous KF to this one) ----
  if (new_id > 0) {
    PoseGraphEdge edge;
    edge.from          = new_id - 1;
    edge.to            = new_id;
    edge.T_from_to     = graph_.nodes[new_id - 1].T_world.inverse() * T_world;
    // Odometry precision: moderate (σ ~ 0.1 per DOF)
    edge.information   = Eigen::Matrix<double, 6, 6>::Identity() * 100.0;
    edge.is_loop_closure = false;
    graph_.edges.push_back(edge);
  }

  // ---- Update keyframe tracking ----
  // Propagate the current correction ratio to estimate the optimised position
  // of this new keyframe.  If PGO fires below, apply_pgo_correction() will
  // overwrite T_last_kf_opt_ with the true optimised value.
  // Without this, T_last_kf_opt_ stays at Identity between PGO runs, which
  // makes the correction formula collapse T_corrected back toward the origin.
  T_last_kf_opt_   = T_last_kf_opt_ * T_last_kf_raw_.inverse() * T_world;
  T_last_keyframe_ = T_world;
  T_last_kf_raw_   = T_world;

  RCLCPP_INFO(get_logger(),
    "Keyframe %d  pos=(%.2f, %.2f, %.2f)",
    new_id,
    T_world.translation().x(),
    T_world.translation().y(),
    T_world.translation().z());

  // ---- Loop-closure search ----
  if (new_id >= min_loop_gap_) {
    const bool found = detect_and_add_loop_closure(new_id);
    if (found) {
      RCLCPP_INFO(get_logger(),
        "Loop closure at KF %d → running PGO ...", new_id);
      const int iters = optimizer_.optimise(graph_);
      RCLCPP_INFO(get_logger(),
        "PGO done in %d iteration(s)", iters);
      apply_pgo_correction();

      std_msgs::msg::Header lc_msg;
      lc_msg.stamp    = stamp;
      lc_msg.frame_id = odom_frame_;
      loop_closure_pub_->publish(lc_msg);
    }
  }

  publish_graph_markers(stamp);
  publish_map(stamp);
}

// ===========================================================================
// detect_and_add_loop_closure — two-stage Scan Context + ICP verification
// ===========================================================================
bool SlamNode::detect_and_add_loop_closure(int query_id)
{
  const PoseGraphNode & query = graph_.nodes[query_id];

  // ------------------------------------------------------------------
  // Stage 1: ring-key pre-filter
  //   Compute ring-key distance from the query to every earlier node
  //   (stopping min_loop_gap_ before the query so we skip recent scans).
  // ------------------------------------------------------------------
  struct Candidate { int id; double rkd; };
  std::vector<Candidate> candidates;

  for (int i = 0; i <= query_id - min_loop_gap_; ++i) {
    const double rkd = ScanContext::ring_key_distance(
      query.ring_key, graph_.nodes[i].ring_key);
    if (rkd < ring_key_threshold_) {
      candidates.push_back({i, rkd});
    }
  }

  if (candidates.empty()) return false;

  // Keep the top-5 ring-key candidates for the (more expensive) full SC check
  std::sort(candidates.begin(), candidates.end(),
    [](const Candidate & a, const Candidate & b) { return a.rkd < b.rkd; });
  if (candidates.size() > 5u) candidates.resize(5);

  // ------------------------------------------------------------------
  // Stage 2: full Scan Context distance (rotation-invariant)
  // ------------------------------------------------------------------
  int    best_cand  = -1;
  double best_sc    = std::numeric_limits<double>::max();
  int    best_shift = 0;

  for (const Candidate & c : candidates) {
    int shift = 0;
    const double sc_d = ScanContext::distance(
      query.sc_desc, graph_.nodes[c.id].sc_desc, &shift);
    if (sc_d < best_sc) {
      best_sc    = sc_d;
      best_cand  = c.id;
      best_shift = shift;
    }
  }

  if (best_sc >= sc_dist_threshold_) return false;

  RCLCPP_INFO(get_logger(),
    "SC candidate: query=%d  cand=%d  sc_dist=%.4f  shift=%d",
    query_id, best_cand, best_sc, best_shift);

  // ------------------------------------------------------------------
  // Stage 3: ICP verification
  //   Initial guess from SC column shift: each shift unit corresponds to
  //   2π/N_SECTORS radians of yaw around the z-axis.  Translation is left
  //   at zero — for a genuine loop closure the robot has physically returned
  //   to near the candidate location, so zero translation is the correct
  //   starting point.  Using the odometry chain translation would offset
  //   the query scan by the full accumulated drift, moving it away from
  //   the candidate rather than toward it.
  //   ICP(source=query.pts, target=cand.normals) returns delta_T such that
  //     query pts aligned in cand frame: x_cand ≈ delta_T * x_query
  //   Which in world terms gives:  T_world_query ≈ T_world_cand · delta_T
  //   → for edge (cand → query):  T_from_to = delta_T
  // ------------------------------------------------------------------
  const PoseGraphNode & cand = graph_.nodes[best_cand];

  const double yaw_init = static_cast<double>(best_shift)
                          * (2.0 * M_PI / ScanContext::N_SECTORS);
  const Eigen::AngleAxisd aa(yaw_init, Eigen::Vector3d::UnitZ());
  const Sophus::SE3d T_init(Sophus::SO3d(aa.toRotationMatrix()),
                             Eigen::Vector3d::Zero());

  IcpSolver lc_solver(lc_icp_max_iter_, 1e-4, lc_icp_max_corr_dist_);
  const IcpResult lc_result  = lc_solver.align(query.pts, cand.normals, T_init);

  if (lc_result.rmse > lc_icp_rmse_threshold_) {
    RCLCPP_WARN(get_logger(),
      "Loop closure ICP rejected: RMSE=%.4f m > threshold=%.4f m",
      lc_result.rmse, lc_icp_rmse_threshold_);
    return false;
  }

  // ------------------------------------------------------------------
  // Handbook §3.2.2 eq. (3.6) — cycle consistency check
  //
  //   dist( delta_T · T̄_odom^{-1} , I ) ≤ γ
  //
  // where T̄_odom = T_world_cand^{-1} · T_world_query  is the odometry
  // chain from the candidate to the query node.  Composing the loop
  // closure with the inverse odometry chain gives the cycle error: if
  // it is near identity, the loop closure is consistent with odometry.
  // Translation (metres) and rotation (radians) are checked separately
  // because they have different units.
  // ------------------------------------------------------------------
  const Sophus::SE3d T_odom_chain = cand.T_world.inverse() * query.T_world;
  const Sophus::SE3d T_cycle_err  = lc_result.delta_T * T_odom_chain.inverse();
  const double t_err = T_cycle_err.translation().norm();
  const double R_err = T_cycle_err.so3().log().norm();

  RCLCPP_INFO(get_logger(),
    "LC eq.(3.6) cycle check: cand=%d query=%d  RMSE=%.4f m  "
    "t_err=%.3f m (thr=%.2f)  R_err=%.3f rad (thr=%.2f)",
    best_cand, query_id, lc_result.rmse,
    t_err, lc_cycle_t_threshold_, R_err, lc_cycle_R_threshold_);

  if (t_err > lc_cycle_t_threshold_ || R_err > lc_cycle_R_threshold_) {
    RCLCPP_WARN(get_logger(),
      "Loop closure rejected by eq.(3.6): t_err=%.3f m  R_err=%.3f rad",
      t_err, R_err);
    return false;
  }

  RCLCPP_INFO(get_logger(),
    "Loop closure verified: cand=%d  query=%d  RMSE=%.4f m  "
    "t_err=%.3f m  R_err=%.3f rad",
    best_cand, query_id, lc_result.rmse, t_err, R_err);

  // ------------------------------------------------------------------
  // Add loop-closure edge to the pose graph
  // ------------------------------------------------------------------
  PoseGraphEdge edge;
  edge.from            = best_cand;
  edge.to              = query_id;
  edge.T_from_to       = lc_result.delta_T;
  // Loop closures carry higher precision than odometry (5× weight)
  edge.information     = Eigen::Matrix<double, 6, 6>::Identity() * 500.0;
  edge.is_loop_closure = true;
  graph_.edges.push_back(edge);

  return true;
}

// ===========================================================================
// apply_pgo_correction — refresh T_last_kf_opt_ from optimised graph
//                        and rebuild /slam/path from corrected node poses
// ===========================================================================
void SlamNode::apply_pgo_correction()
{
  const int last_kf_id = static_cast<int>(graph_.nodes.size()) - 1;
  T_last_kf_opt_ = graph_.nodes[last_kf_id].T_world;

  // Rebuild the SLAM path from the now-optimised keyframe positions so RViz
  // shows the corrected trajectory without the pre-PGO starburst artifact.
  slam_path_.poses.clear();
  const rclcpp::Time now = get_clock()->now();
  slam_path_.header.stamp = now;

  for (const auto & node : graph_.nodes) {
    const Eigen::Vector3d    t = node.T_world.translation();
    const Eigen::Quaterniond q = node.T_world.unit_quaternion();

    geometry_msgs::msg::PoseStamped ps;
    ps.header.stamp    = now;
    ps.header.frame_id = odom_frame_;
    ps.pose.position.x    = t.x();
    ps.pose.position.y    = t.y();
    ps.pose.position.z    = t.z();
    ps.pose.orientation.w = q.w();
    ps.pose.orientation.x = q.x();
    ps.pose.orientation.y = q.y();
    ps.pose.orientation.z = q.z();
    slam_path_.poses.push_back(ps);
  }
  slam_path_pub_->publish(slam_path_);

  // Republish map — keyframe poses have changed so world-frame points move
  publish_map(get_clock()->now());
}

// ===========================================================================
// Publish helpers
// ===========================================================================
namespace {

nav_msgs::msg::Odometry make_odom(
  const rclcpp::Time & stamp,
  const std::string  & frame_id,
  const std::string  & child_frame_id,
  const Sophus::SE3d & T)
{
  nav_msgs::msg::Odometry odom;
  odom.header.stamp    = stamp;
  odom.header.frame_id = frame_id;
  odom.child_frame_id  = child_frame_id;

  const Eigen::Vector3d   & t = T.translation();
  const Eigen::Quaterniond  q = T.unit_quaternion();

  odom.pose.pose.position.x    = t.x();
  odom.pose.pose.position.y    = t.y();
  odom.pose.pose.position.z    = t.z();
  odom.pose.pose.orientation.w = q.w();
  odom.pose.pose.orientation.x = q.x();
  odom.pose.pose.orientation.y = q.y();
  odom.pose.pose.orientation.z = q.z();
  return odom;
}

}  // anonymous namespace

void SlamNode::publish_slam_odom(const rclcpp::Time & stamp, const Sophus::SE3d & T)
{
  const auto odom = make_odom(stamp, odom_frame_, base_link_frame_, T);
  slam_odom_pub_->publish(odom);

  // Path
  geometry_msgs::msg::PoseStamped ps;
  ps.header = odom.header;
  ps.pose   = odom.pose.pose;
  slam_path_.header.stamp = stamp;
  slam_path_.poses.push_back(ps);
  slam_path_pub_->publish(slam_path_);

  // TF: odom → base_link (SLAM-corrected — gives RViz a consistent frame)
  geometry_msgs::msg::TransformStamped tf_msg;
  tf_msg.header         = odom.header;
  tf_msg.child_frame_id = base_link_frame_;
  const Eigen::Vector3d   & t = T.translation();
  const Eigen::Quaterniond  q = T.unit_quaternion();
  tf_msg.transform.translation.x = t.x();
  tf_msg.transform.translation.y = t.y();
  tf_msg.transform.translation.z = t.z();
  tf_msg.transform.rotation.w    = q.w();
  tf_msg.transform.rotation.x    = q.x();
  tf_msg.transform.rotation.y    = q.y();
  tf_msg.transform.rotation.z    = q.z();
  tf_broadcaster_->sendTransform(tf_msg);
}

void SlamNode::publish_raw_odom(const rclcpp::Time & stamp, const Sophus::SE3d & T)
{
  const auto odom = make_odom(stamp, odom_frame_, base_link_frame_, T);
  raw_odom_pub_->publish(odom);

  // Path (raw — for drift comparison in the plot script)
  geometry_msgs::msg::PoseStamped ps;
  ps.header = odom.header;
  ps.pose   = odom.pose.pose;
  raw_path_.header.stamp = stamp;
  raw_path_.poses.push_back(ps);
  raw_path_pub_->publish(raw_path_);
}

// ===========================================================================
// publish_graph_markers — visualise pose graph in RViz
//   /slam/graph  MarkerArray
//     ns="nodes"     SPHERE_LIST  blue  — one sphere per keyframe
//     ns="odom_edges" LINE_LIST   grey  — sequential odometry edges
//     ns="lc_edges"   LINE_LIST   yellow — loop-closure edges
// ===========================================================================
void SlamNode::publish_graph_markers(const rclcpp::Time & stamp)
{
  using Marker      = visualization_msgs::msg::Marker;
  using MarkerArray = visualization_msgs::msg::MarkerArray;
  using Point       = geometry_msgs::msg::Point;

  MarkerArray ma;

  auto make_marker = [&](const std::string & ns, int type) {
    Marker m;
    m.header.frame_id = odom_frame_;
    m.header.stamp    = stamp;
    m.ns              = ns;
    m.id              = 0;
    m.type            = type;
    m.action          = Marker::ADD;
    return m;
  };

  // ---- Keyframe nodes (blue spheres) ----
  Marker nodes = make_marker("nodes", Marker::SPHERE_LIST);
  nodes.scale.x = nodes.scale.y = nodes.scale.z = 0.25;
  nodes.color.r = 0.2f; nodes.color.g = 0.5f;
  nodes.color.b = 1.0f; nodes.color.a = 1.0f;
  for (const auto & n : graph_.nodes) {
    Point p;
    p.x = n.T_world.translation().x();
    p.y = n.T_world.translation().y();
    p.z = n.T_world.translation().z();
    nodes.points.push_back(p);
  }
  ma.markers.push_back(nodes);

  // ---- Edges ----
  Marker odom_edges = make_marker("odom_edges", Marker::LINE_LIST);
  odom_edges.scale.x = 0.05;
  odom_edges.color.r = odom_edges.color.g = odom_edges.color.b = 0.5f;
  odom_edges.color.a = 0.8f;

  Marker lc_edges = make_marker("lc_edges", Marker::LINE_LIST);
  lc_edges.scale.x = 0.12;
  lc_edges.color.r = 1.0f; lc_edges.color.g = 0.85f;
  lc_edges.color.b = 0.0f; lc_edges.color.a = 1.0f;

  for (const auto & e : graph_.edges) {
    Point p_from, p_to;
    const auto & tf = graph_.nodes[e.from].T_world.translation();
    const auto & tt = graph_.nodes[e.to].T_world.translation();
    p_from.x = tf.x(); p_from.y = tf.y(); p_from.z = tf.z();
    p_to.x   = tt.x(); p_to.y   = tt.y(); p_to.z   = tt.z();

    if (e.is_loop_closure) {
      lc_edges.points.push_back(p_from);
      lc_edges.points.push_back(p_to);
    } else {
      odom_edges.points.push_back(p_from);
      odom_edges.points.push_back(p_to);
    }
  }
  ma.markers.push_back(odom_edges);
  ma.markers.push_back(lc_edges);

  graph_marker_pub_->publish(ma);
}

// ===========================================================================
// publish_map — accumulated keyframe clouds transformed to world frame
// ===========================================================================
void SlamNode::publish_map(const rclcpp::Time & stamp)
{
  // Count total points across all keyframes
  size_t total = 0;
  for (const auto & node : graph_.nodes) total += node.pts.size();
  if (total == 0) return;

  sensor_msgs::msg::PointCloud2 cloud;
  cloud.header.stamp    = stamp;
  cloud.header.frame_id = odom_frame_;
  cloud.height       = 1;
  cloud.is_dense     = true;
  cloud.is_bigendian = false;
  cloud.point_step   = 12;           // 3 × float32

  sensor_msgs::msg::PointField fx, fy, fz;
  fx.name = "x"; fx.offset =  0; fx.datatype = sensor_msgs::msg::PointField::FLOAT32; fx.count = 1;
  fy.name = "y"; fy.offset =  4; fy.datatype = sensor_msgs::msg::PointField::FLOAT32; fy.count = 1;
  fz.name = "z"; fz.offset =  8; fz.datatype = sensor_msgs::msg::PointField::FLOAT32; fz.count = 1;
  cloud.fields = {fx, fy, fz};
  cloud.data.resize(cloud.row_step);

  // Pre-pass: collect only above-ground points (z > 0.1 m filters floor returns)
  std::vector<Eigen::Vector3f> world_pts;
  world_pts.reserve(total);
  for (const auto & node : graph_.nodes) {
    for (const auto & p_sensor : node.pts) {
      const Eigen::Vector3d p_world = node.T_world * p_sensor;
      if (p_world.z() < 0.1) continue;   // skip ground-plane returns
      world_pts.emplace_back(p_world.cast<float>());
    }
  }

  cloud.width    = static_cast<uint32_t>(world_pts.size());
  cloud.row_step = cloud.point_step * cloud.width;
  cloud.data.resize(cloud.row_step);

  uint8_t * dst = cloud.data.data();
  for (const auto & p : world_pts) {
    std::memcpy(dst,     &p.x(), 4);
    std::memcpy(dst + 4, &p.y(), 4);
    std::memcpy(dst + 8, &p.z(), 4);
    dst += 12;
  }

  map_pub_->publish(cloud);
}

}  // namespace lidar_icp

// ---------------------------------------------------------------------------
// main — ROS2 component-style entry point
// ---------------------------------------------------------------------------
int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<lidar_icp::SlamNode>());
  rclcpp::shutdown();
  return 0;
}
