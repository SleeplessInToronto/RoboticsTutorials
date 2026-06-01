#pragma once

// ---------------------------------------------------------------------------
// Scan Context descriptor — Kim & Kim 2018 (IROS), §8.3.2.2 of the SLAM
// Handbook.  A 2-D polar bird's-eye-view (BEV) grid where each cell stores
// the maximum height of all points whose range falls in ring r and whose
// azimuth falls in sector s.  The resulting N_RINGS × N_SECTORS matrix is
// highly distinctive and, crucially, shift-invariant in the column direction
// (which corresponds to rotation about the z-axis), enabling rotation-
// invariant place recognition.
//
// Two-stage loop-closure candidate retrieval:
//   Stage 1 (fast)   — ring key L1-distance pre-filter (O(N_RINGS) per pair)
//   Stage 2 (precise) — column-shifted cosine distance  (O(N_RINGS × N_SECTORS²))
// ---------------------------------------------------------------------------

#include <vector>
#include <Eigen/Dense>

// Pull in Vector3d from the ICP solver type aliases so we share the same type.
#include "lidar_icp_odometry/icp_solver.hpp"

namespace lidar_icp {

class ScanContext
{
public:
  // ---- Grid dimensions (tuned for outdoor ground vehicles, VLP-16 style) ----
  static constexpr int    N_RINGS   = 20;    ///< Radial bins
  static constexpr int    N_SECTORS = 60;    ///< Azimuth bins
  static constexpr double MAX_RANGE = 20.0;  ///< [m] — farther points ignored

  using Descriptor = Eigen::Matrix<double, N_RINGS, N_SECTORS>;
  using RingKey    = Eigen::Matrix<double, N_RINGS, 1>;

  // ---------------------------------------------------------------------------
  /**
   * @brief Build a Scan Context descriptor from a downsampled point cloud.
   *
   * Each cell (r, s) stores max{z : range(p) ∈ ring r, azimuth(p) ∈ sector s}.
   * Empty cells are 0.  Complexity O(|pts|).
   *
   * @param pts Downsampled 3-D points in the sensor frame.
   * @return N_RINGS × N_SECTORS descriptor matrix.
   */
  static Descriptor compute(const std::vector<Vector3d> & pts);

  // ---------------------------------------------------------------------------
  /**
   * @brief Ring key: column-wise mean of the descriptor (N_RINGS-dim vector).
   *
   * Condenses the full descriptor to a compact summary used for O(N_RINGS)
   * pre-filtering.  Unlike the full descriptor, the ring key is NOT
   * rotation-invariant, but is cheap to compare.
   */
  static RingKey ring_key(const Descriptor & desc);

  // ---------------------------------------------------------------------------
  /**
   * @brief L1 distance between ring keys, normalised to [0, 1].
   *
   * Candidates with ring_key_distance > ring_key_threshold can be rejected
   * before computing the (more expensive) full SC distance.
   */
  static double ring_key_distance(const RingKey & a, const RingKey & b);

  // ---------------------------------------------------------------------------
  /**
   * @brief Full Scan Context distance — rotation-invariant.
   *
   * Finds the integer column shift s* ∈ {0 … N_SECTORS−1} that minimises the
   * average per-ring cosine distance between descriptor a and column-shifted
   * descriptor b:
   *
   *   d(a, b) = min_{s} (1/N_rings_valid) Σ_r  (1 − cos_sim(a[r], b[r, shift s]))
   *
   * Returns a value in [0, 1]: 0 = identical, 1 = maximally dissimilar.
   *
   * @param a          Query descriptor.
   * @param b          Candidate descriptor.
   * @param best_shift If non-null, written with the shift achieving d_min.
   *                   Can seed an ICP initial guess (each shift = 2π/N_SECTORS rad).
   */
  static double distance(const Descriptor & a, const Descriptor & b,
                          int * best_shift = nullptr);
};

}  // namespace lidar_icp
