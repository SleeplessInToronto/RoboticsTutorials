# Robotics Tutorials

A collection of hands-on robotics projects.

---

## 1. [EKF SE(3) Map-Based Localization](ekf_se3_localization/)

A full ROS2 + Gazebo Fortress simulation implementing an **Extended Kalman Filter on the SE(3) Lie group** for map-based localization of a ground vehicle. A roof-mounted 3D LiDAR (VLP-16 style) detects cylindrical beacon poles; the EKF uses velocity commands for dead reckoning and corrects with LiDAR beacon detections to maintain a bounded-error pose estimate.

**What you will learn:**

- **Lie group & algebra** — SE(3) as a smooth manifold, the exponential and logarithm maps, right-perturbation convention, and representing pose uncertainty in the Lie algebra
- **Data association** — matching LiDAR point clusters to a known beacon map using nearest-neighbour search
- **RANSAC** — robust outlier rejection to handle false beacon detections before the EKF correction step
- **Sensor simulation** — 3D LiDAR in Gz Fortress, beacon centroid extraction without the Point Cloud Library (PCL)
- **Floating-point drift** — how accumulated rounding error causes rotation matrices to leave SE(3), and how to re-project back onto the manifold to keep the filter numerically sound
- **ROS2 architecture** — multi-node design with bridged Gz topics and ground truth comparison

**Stack:** ROS2 Humble · Gz Fortress · Sophus · Eigen · C++17 · Python

---

## 2. [LiDAR SLAM — ICP Odometry, Scan Context Loop Closure, and Pose-Graph Optimisation](lidar_icp_odometry/)

A fully-standalone ROS2 + Gazebo Fortress simulation implementing the **full LiDAR SLAM pipeline** following Chapter 8 of the [SLAM Handbook](https://github.com/SLAM-Handbook-contributors/slam-handbook-public-release). A VLP-16 style LiDAR on a ground vehicle scans an obstacle-filled world. The pipeline progresses through three stages: scan-to-scan ICP odometry that accumulates drift, Scan Context place recognition that detects loop closures, and Gauss-Newton pose-graph optimisation that corrects the accumulated drift across all keyframe poses simultaneously.

**What you will learn:**

**ICP Odometry — per-scan pipeline:**

1. **Parse point cloud** ([README §3.1](lidar_icp_odometry/README.md#31-the-scan-registration-problem))
   - `PointCloud2` stores LiDAR data as a flat byte buffer
   - `PointCloud2ConstIterator` walks the buffer extracting the `"x"`, `"y"`, `"z"` fields into an `Eigen::Vector3d` per point
   - NaN points (beams with no return) and ground returns (z below threshold) are filtered out
   - `std::vector<Vector3d>` (~11 500 filtered points)

2. **Voxel grid downsample** ([README §3.2](lidar_icp_odometry/README.md#32-voxel-grid-downsampling))
   - Divide 3D space into a regular grid of cubes (voxels) of side length L=0.3 m
   - For each cube that contains at least one point, discard all points inside it and replace them with a single point at their average position (centroid)
   - Empty cubes are ignored
   - `std::vector<Vector3d>` (~400–800 evenly spaced representative points)

3. **Estimate normals on the target scan** ([README §3.3](lidar_icp_odometry/README.md#33-normal-estimation-via-pca)) — the target is the previous scan, held fixed as the reference; the source is the current (newest) scan that will be aligned onto it. For each target point **q_i**:
   - **k-NN** — find the k=20 nearest neighbouring points in the target cloud (using nanoflann KD-tree)
   - **PCA** — fit a plane to that local patch; the eigenvector with the smallest eigenvalue is the surface normal **n_i**
   - `std::vector<PointNormal>` (one unit surface normal **n_i** per target point **q_i**)

4. **Find correspondences** ([README §3.4](lidar_icp_odometry/README.md#34-kd-tree-nearest-neighbour-search))
   - Transform each source point by the current ΔT: **p_i' = ΔT · p_i**
   - Query the KD-tree to find the nearest target point **q_i** to **p_i'**
   - Each pair carries **n_i** (the normal at **q_i** from step 3)
   - As ΔT improves across iterations, correspondences become more accurate

5. **Solve point-to-plane linear system** ([README §3.5](lidar_icp_odometry/README.md#35-point-to-plane-icp--linearised-system))
   - For each triple **(p_i', q_i, n_i)** compute residual = **n_i · (p_i' − q_i)**
   - Stack all residuals into AᵀA ξ = Aᵀb and solve with LDLT
   - ξ ∈ ℝ⁶ (6-DOF tangent vector: 3 translation + 3 rotation)

6. **Check convergence** ([README §3.6](lidar_icp_odometry/README.md#36-convergence-criteria))
   - If ‖ξ‖ < ε, max iterations reached, or correspondences empty: stop and return ΔT
   - Otherwise apply ΔT ← ΔT · Exp(ξ) and repeat from step 4

7. **Accumulate pose** ([README §3.7](lidar_icp_odometry/README.md#37-pose-accumulation-on-se3))
   - Right-compose ΔT onto the running world pose and renormalise SO(3)
   - **T_world** ∈ SE(3) (updated absolute pose estimate)

**Loop Closure Detection**
- **Keyframe selection** — triggering a new graph node only when the robot has moved a minimum distance or angle, keeping the graph sparse
- **Scan Context descriptor** — a bird's-eye-view polar height grid that fingerprints each LiDAR scan for place recognition; ring-key pre-filter for fast rejection and column-shifted cosine distance for rotation-invariant matching
- **Loop closure detection pipeline** — four-stage rejection: ring-key pre-filter → SC cosine distance → ICP RMSE threshold → cycle consistency check

**Pose Graph Optimisation**
- **Pose graph construction** — nodes as keyframe poses T_i ∈ SE(3), edges as relative transform measurements T_ij with information matrices Ω_ij weighting trust
- **Pose Graph Optimization** — linearising residuals e_ij = Log(T_ij⁻¹ T_i⁻¹ T_j), assembling the 6n×6n normal equations H·δ = −b, and retracting with T_k ← T_k · Exp(δξ_k)
- **Repeated traversal limitation** — why pose-graph SLAM degrades when repeatedly traversing the same small environment, as noted in the SLAM Handbook

**Stack:** ROS2 Humble · Gz Fortress · Sophus · Eigen · nanoflann · C++17 · Python · **zero PCL**
