# Robotics Tutorials

A collection of hands-on robotics projects.

---

## 1. [EKF SE(3) Map-Based Localization](ekf_se3_localization/)

**Click on the link above and look at the README** ->  A full ROS2 + Gazebo Fortress simulation implementing an Extended Kalman Filter on the SE(3) Lie group for map-based localization of a ground vehicle. A roof-mounted 3D LiDAR (VLP-16 style) detects cylindrical beacon poles; the EKF uses velocity commands for dead reckoning and corrects with LiDAR beacon detections to maintain a bounded-error pose estimate.

**What you will learn:**

- **Lie group & algebra** — SE(3) as a smooth manifold, the exponential and logarithm maps, right-perturbation convention, and representing pose uncertainty in the Lie algebra
- **Data association** — matching LiDAR point clusters to a known beacon map using nearest-neighbour search
- **RANSAC** — robust outlier rejection to handle false beacon detections before the EKF correction step
- **Sensor simulation** — 3D LiDAR in Gz Fortress, beacon centroid extraction without the Point Cloud Library (PCL)
- **Floating-point drift** — how accumulated rounding error causes rotation matrices to leave SE(3), and how to re-project back onto the manifold to keep the filter numerically sound
- **ROS2 architecture** — multi-node design with bridged Gz topics and ground truth comparison

**Stack:** ROS2 Humble · Gz Fortress · Sophus · Eigen · C++17 · Python
