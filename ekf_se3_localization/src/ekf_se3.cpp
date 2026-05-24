#include "ekf_se3_localization/ekf_se3.hpp"

namespace ekf_se3 {

EkfSe3::EkfSe3(
  const Matrix6d & Q,
  const Matrix3d & R,
  const Sophus::SE3d & X0,
  const Matrix6d & P0)
: X_(X0), P_(P0), Q_(Q), R_(R)
{}

void EkfSe3::predict(const Vector6d & u, double dt)
{
  // Propagate mean on the manifold: X̂ ← X̂ · Exp(u·dt)
  const Vector6d u_dt = u * dt;
  X_ = X_ * Sophus::SE3d::exp(u_dt);

  // F = Adj(Exp(−u·dt))  — linearised motion Jacobian w.r.t. right perturbation
  const Matrix6d F = Sophus::SE3d::exp(-u_dt).Adj();

  // P ← F·P·Fᵀ + Q  (G = I so G·Q·Gᵀ = Q)
  P_ = F * P_ * F.transpose() + Q_;

  // Re-normalize the quaternion inside SO3 — repeated SE3 multiplications
  // accumulate floating-point drift that makes ||q|| drift from 1, so
  // R·Rᵀ ≠ I.  Dividing by ||q|| projects back onto the unit hypersphere.
  X_.so3().normalize();
}

Sophus::SE3d EkfSe3::hypothetical_pose(const Vector3d & b_k, const Vector3d & y_k) const
{
  const Vector3d p = X_.inverse() * b_k;
  const Vector3d z_k = y_k - p;
  Eigen::Matrix<double, 3, 6> H;
  H.leftCols<3>()  = -Matrix3d::Identity();
  H.rightCols<3>() =  skew(p);
  const Eigen::Matrix3d Z_k = H * P_ * H.transpose() + R_;
  const Eigen::Matrix<double, 6, 3> K =
    P_ * H.transpose() * Z_k.ldlt().solve(Eigen::Matrix3d::Identity());
  return X_ * Sophus::SE3d::exp(K * z_k);
}

void EkfSe3::correct(const Vector3d & b_k, const Vector3d & y_k)
{
  // Predicted beacon position in body frame
  const Vector3d p = X_.inverse() * b_k;

  // Innovation
  const Vector3d z_k = y_k - p;

  // Measurement Jacobian H = [−I₃ | skew(p)]  (3×6)
  Eigen::Matrix<double, 3, 6> H;
  H.leftCols<3>()  = -Matrix3d::Identity();
  H.rightCols<3>() =  skew(p);

  // Innovation covariance Z_k = H·P·Hᵀ + R  (3×3)
  const Eigen::Matrix3d Z_k = H * P_ * H.transpose() + R_;

  // Kalman gain K = P·Hᵀ·Z_k⁻¹  (6×3) — ldlt for numerical stability
  const Eigen::Matrix<double, 6, 3> K =
    P_ * H.transpose() * Z_k.ldlt().solve(Eigen::Matrix3d::Identity());

  // State update on manifold: X̂ ← X̂ · Exp(K·z_k)
  X_ = X_ * Sophus::SE3d::exp(K * z_k);
  X_.so3().normalize();  // keep quaternion on unit hypersphere after correction

  P_ = P_ - K * Z_k * K.transpose();
  P_ = 0.5 * (P_ + P_.transpose());  // enforce exact symmetry
}

}  // namespace ekf_se3
