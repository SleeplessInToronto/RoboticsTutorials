#pragma once

#include <Eigen/Dense>
#include <sophus/se3.hpp>

namespace ekf_se3 {

using Matrix6d = Eigen::Matrix<double, 6, 6>;
using Vector6d = Eigen::Matrix<double, 6, 1>;
using Matrix3d = Eigen::Matrix3d;
using Vector3d = Eigen::Vector3d;

// Skew-symmetric matrix from a 3-vector: v^× such that v^× w = v × w
inline Matrix3d skew(const Vector3d & v)
{
  Matrix3d S;
  S <<  0.0,  -v.z(),  v.y(),
       v.z(),   0.0,  -v.x(),
      -v.y(),  v.x(),   0.0;
  return S;
}

/**
 * EKF on SE(3) — right-perturbation convention.
 *
 * State:      X̂ ∈ SE(3)        (Sophus::SE3d)
 * Covariance: P ∈ ℝ⁶ˣ⁶        (in Lie algebra, tangent ordering [ρ; φ])
 *
 * Motion model:      X_i = X_{i-1} · Exp(u_i·dt + w),  w ~ N(0,Q)
 * Measurement model: y_k = X⁻¹·b_k + v,                v ~ N(0,R)
 */
class EkfSe3
{
public:
  EkfSe3(
    const Matrix6d & Q,
    const Matrix3d & R,
    const Sophus::SE3d & X0,
    const Matrix6d & P0);

  /**
   * Prediction step.
   * @param u   Body-frame velocity twist [vx,vy,vz, wx,wy,wz] ∈ ℝ⁶
   * @param dt  Time step (seconds)
   *
   * X̂ ← X̂ · Exp(u·dt)
   * F  = Adj(Exp(−u·dt))     [6×6]
   * P ← F·P·Fᵀ + Q
   */
  void predict(const Vector6d & u, double dt);

  /**
   * Correction step for one beacon.
   * @param b_k  Beacon position in world frame  (ℝ³)
   * @param y_k  Observed beacon in robot frame  (ℝ³)
   *
   * p   = X̂⁻¹·b_k
   * z_k = y_k − p
   * H   = [−I₃ | skew(p)]   [3×6]
   * Z_k = H·P·Hᵀ + R
   * K   = P·Hᵀ·Z_k⁻¹
   * X̂  ← X̂ · Exp(K·z_k)
   * P  ← P − K·Z_k·Kᵀ
   */
  void correct(const Vector3d & b_k, const Vector3d & y_k);

  /** Returns the pose that would result from correcting with one beacon,
   *  without committing any change to the filter state. Used by RANSAC. */
  Sophus::SE3d hypothetical_pose(const Vector3d & b_k, const Vector3d & y_k) const;

  const Sophus::SE3d & pose()       const { return X_; }
  const Matrix6d &     covariance() const { return P_; }

  void reset_pose(const Sophus::SE3d & X0, const Matrix6d & P0) { X_ = X0; P_ = P0; }

private:
  Sophus::SE3d X_;
  Matrix6d     P_;
  Matrix6d     Q_;
  Matrix3d     R_;
};

}  // namespace ekf_se3
