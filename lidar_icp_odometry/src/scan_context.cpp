#include "lidar_icp_odometry/scan_context.hpp"

#include <cmath>
#include <algorithm>
#include <limits>

namespace lidar_icp {

// ---------------------------------------------------------------------------
// compute — §8.3.2.2 "Scan Context" descriptor
// ---------------------------------------------------------------------------
ScanContext::Descriptor
ScanContext::compute(const std::vector<Vector3d> & pts)
{
  Descriptor desc = Descriptor::Zero();

  for (const auto & p : pts) {
    // 2-D range in the ground plane
    const double range = std::sqrt(p.x() * p.x() + p.y() * p.y());
    if (range < 1e-3 || range > MAX_RANGE) continue;

    // Ring index: linearly partition [0, MAX_RANGE) into N_RINGS bins
    int r = static_cast<int>(range / MAX_RANGE * static_cast<double>(N_RINGS));
    if (r >= N_RINGS) r = N_RINGS - 1;

    // Sector index: azimuth in (−π, π] → [0, N_SECTORS)
    const double azimuth      = std::atan2(p.y(), p.x());           // (−π, π]
    const double azimuth_norm = (azimuth + M_PI) / (2.0 * M_PI);   // [0, 1)
    int s = static_cast<int>(azimuth_norm * static_cast<double>(N_SECTORS));
    if (s >= N_SECTORS) s = N_SECTORS - 1;

    // Cell value = maximum z observed in this voxel column
    desc(r, s) = std::max(desc(r, s), p.z());
  }
  return desc;
}

// ---------------------------------------------------------------------------
// ring_key — column-wise mean of descriptor (N_RINGS-dim summary)
// ---------------------------------------------------------------------------
ScanContext::RingKey
ScanContext::ring_key(const Descriptor & desc)
{
  return desc.rowwise().mean();
}

// ---------------------------------------------------------------------------
// ring_key_distance — normalised L1
// ---------------------------------------------------------------------------
double
ScanContext::ring_key_distance(const RingKey & a, const RingKey & b)
{
  return (a - b).cwiseAbs().sum() / static_cast<double>(N_RINGS);
}

// ---------------------------------------------------------------------------
// distance — minimum column-shifted cosine distance (rotation-invariant)
// ---------------------------------------------------------------------------
double
ScanContext::distance(const Descriptor & a, const Descriptor & b,
                       int * best_shift)
{
  double min_dist = std::numeric_limits<double>::max();
  int    best_s   = 0;

  for (int s = 0; s < N_SECTORS; ++s) {
    // Accumulate mean cosine distance across all rings for this shift
    double sum_dist = 0.0;
    int    n_valid  = 0;

    for (int r = 0; r < N_RINGS; ++r) {
      // Build the shifted row of b
      Eigen::Matrix<double, 1, N_SECTORS> row_b;
      for (int c = 0; c < N_SECTORS; ++c) {
        row_b(c) = b(r, (c + s) % N_SECTORS);
      }

      const auto row_a = a.row(r);
      const double norm_a = row_a.norm();
      const double norm_b = row_b.norm();

      // Skip all-zero rows (no returns in this ring)
      if (norm_a < 1e-9 || norm_b < 1e-9) continue;

      const double cos_sim = row_a.dot(row_b) / (norm_a * norm_b);
      sum_dist += 1.0 - cos_sim;
      ++n_valid;
    }

    if (n_valid == 0) continue;

    const double dist = sum_dist / static_cast<double>(n_valid);
    if (dist < min_dist) {
      min_dist = dist;
      best_s   = s;
    }
  }

  if (best_shift) *best_shift = best_s;
  return (min_dist < std::numeric_limits<double>::max()) ? min_dist : 1.0;
}

}  // namespace lidar_icp
