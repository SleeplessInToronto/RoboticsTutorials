#!/usr/bin/env python3
"""
Analyze a rosbag recorded from the SLAM simulation.

Topics read
-----------
/ground_truth/odom  nav_msgs/Odometry  Gazebo ground truth (gz_ground_truth.py)
/icp/odom           nav_msgs/Odometry  Raw scan-to-scan ICP odometry (no loop closure)
/slam/odom          nav_msgs/Odometry  PGO-corrected SLAM pose

Plots produced (slam_analysis.png saved to <bag_dir>/)
------------------------------------------------------
Left  — XY trajectories for all three sources over the full run.
Right — Per-sample 2-D position error vs ground truth over time for ICP and SLAM.

RMSE calculation
----------------
All three topics are interpolated onto the ICP time axis (the common time window
where all three overlap). For each sample at time t:

    icp_err(t)  = sqrt( (icp_x(t)  - gt_x(t))^2 + (icp_y(t)  - gt_y(t))^2 )
    slam_err(t) = sqrt( (slam_x(t) - gt_x(t))^2 + (slam_y(t) - gt_y(t))^2 )

RMSE is then the root-mean-square of those per-sample errors:

    ICP  RMSE = sqrt( mean( icp_err^2  ) )
    SLAM RMSE = sqrt( mean( slam_err^2 ) )

A lower SLAM RMSE relative to ICP RMSE quantifies how much loop closure and
PGO reduced the accumulated drift.

Usage:
  python3 analyze_slam_bag.py <bag_dir>
"""
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

import rclpy
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
import rosbag2_py


def load_topic(bag_dir: str, topic: str):
    reader = rosbag2_py.SequentialReader()
    storage_options = rosbag2_py.StorageOptions(uri=bag_dir, storage_id='sqlite3')
    converter_options = rosbag2_py.ConverterOptions('', '')
    reader.open(storage_options, converter_options)

    topic_types = reader.get_all_topics_and_types()
    type_map = {t.name: t.type for t in topic_types}

    if topic not in type_map:
        print(f"  WARNING: topic '{topic}' not found")
        return np.array([]), np.array([]), np.array([])

    msg_type = get_message(type_map[topic])

    filter_ = rosbag2_py.StorageFilter(topics=[topic])
    reader.set_filter(filter_)

    stamps, xs, ys = [], [], []
    while reader.has_next():
        _, data, bag_ts_ns = reader.read_next()
        msg = deserialize_message(data, msg_type)
        # Always use the bag recording timestamp so all topics share the same
        # time axis (ground truth header stamps are zero from the Gz bridge).
        stamps.append(bag_ts_ns * 1e-9)
        xs.append(msg.pose.pose.position.x)
        ys.append(msg.pose.pose.position.y)

    return np.array(stamps), np.array(xs), np.array(ys)


def load_timestamps(bag_dir: str, topic: str):
    """Return bag recording timestamps (seconds) for every message on topic."""
    reader = rosbag2_py.SequentialReader()
    storage_options = rosbag2_py.StorageOptions(uri=bag_dir, storage_id='sqlite3')
    converter_options = rosbag2_py.ConverterOptions('', '')
    reader.open(storage_options, converter_options)

    topic_types = reader.get_all_topics_and_types()
    if topic not in {t.name for t in topic_types}:
        return np.array([])

    filter_ = rosbag2_py.StorageFilter(topics=[topic])
    reader.set_filter(filter_)

    stamps = []
    while reader.has_next():
        _, _, bag_ts_ns = reader.read_next()
        stamps.append(bag_ts_ns * 1e-9)
    return np.array(stamps)


def interp_to(ref_t, src_t, src_x, src_y):
    x = np.interp(ref_t, src_t, src_x)
    y = np.interp(ref_t, src_t, src_y)
    return x, y


def rmse(err):
    return np.sqrt(np.mean(err ** 2))


def main():
    if len(sys.argv) < 2:
        print("Usage: analyze_slam_bag.py <bag_dir>")
        sys.exit(1)

    bag_dir = str(Path(sys.argv[1]).resolve())
    print(f"Reading: {bag_dir}")

    rclpy.init()

    print("Loading /ground_truth/odom ...")
    gt_t,   gt_x,   gt_y   = load_topic(bag_dir, '/ground_truth/odom')
    print("Loading /icp/odom ...")
    icp_t,  icp_x,  icp_y  = load_topic(bag_dir, '/icp/odom')
    print("Loading /slam/odom ...")
    slam_t, slam_x, slam_y = load_topic(bag_dir, '/slam/odom')
    print("Loading /slam/loop_closures ...")
    lc_t = load_timestamps(bag_dir, '/slam/loop_closures')

    rclpy.shutdown()

    if len(gt_t) == 0 or len(icp_t) == 0 or len(slam_t) == 0:
        print("Missing data — check topic names.")
        sys.exit(1)

    print(f"\n  Ground truth : {len(gt_t)} msgs  t=[{gt_t[0]:.1f} .. {gt_t[-1]:.1f}]")
    print(f"  ICP odom     : {len(icp_t)} msgs  t=[{icp_t[0]:.1f} .. {icp_t[-1]:.1f}]")
    print(f"  SLAM odom    : {len(slam_t)} msgs  t=[{slam_t[0]:.1f} .. {slam_t[-1]:.1f}]")

    # Common time window
    t0    = max(gt_t[0], icp_t[0], slam_t[0])
    t1    = min(gt_t[-1], icp_t[-1], slam_t[-1])
    ref_t = icp_t[(icp_t >= t0) & (icp_t <= t1)]

    if len(ref_t) == 0:
        print("ERROR: no overlapping timestamps between topics.")
        sys.exit(1)

    gt_xi,   gt_yi   = interp_to(ref_t, gt_t,   gt_x,   gt_y)
    icp_xi,  icp_yi  = interp_to(ref_t, icp_t,  icp_x,  icp_y)
    slam_xi, slam_yi = interp_to(ref_t, slam_t, slam_x, slam_y)

    icp_err  = np.hypot(icp_xi  - gt_xi, icp_yi  - gt_yi)
    slam_err = np.hypot(slam_xi - gt_xi, slam_yi - gt_yi)

    icp_rmse  = rmse(icp_err)
    slam_rmse = rmse(slam_err)
    improvement = (icp_rmse - slam_rmse) / icp_rmse * 100

    print(f"\n  ICP  RMSE vs ground truth : {icp_rmse:.4f} m")
    print(f"  SLAM RMSE vs ground truth : {slam_rmse:.4f} m")
    if improvement > 0:
        print(f"  SLAM improved over ICP by  : {improvement:.1f}%")
    else:
        print(f"  SLAM is WORSE than ICP by  : {-improvement:.1f}%")

    # ---- Plot ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"SLAM vs ICP Odometry vs Ground Truth   [ICP RMSE={icp_rmse:.3f}m  SLAM RMSE={slam_rmse:.3f}m]",
                 fontsize=12)

    ax = axes[0]
    ax.plot(gt_x,   gt_y,   'g-',  lw=1.5, label='Ground truth')
    ax.plot(icp_x,  icp_y,  'r-',  lw=1.0, label=f'ICP Odometry   RMSE={icp_rmse:.3f} m')
    ax.plot(slam_x, slam_y, 'b-',  lw=1.0, label=f'SLAM      RMSE={slam_rmse:.3f} m')
    ax.set_aspect('equal')
    ax.legend(fontsize=9)
    ax.set_title("XY Trajectories")
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    t_rel = ref_t - ref_t[0]
    ax2.plot(t_rel, icp_err,  'r-', lw=1.0, label=f'ICP Odometry  RMSE={icp_rmse:.3f} m')
    ax2.plot(t_rel, slam_err, 'b-', lw=1.0, label=f'SLAM RMSE={slam_rmse:.3f} m')

    # Vertical markers for each loop closure event
    for i, lc_abs in enumerate(lc_t):
        lc_rel = lc_abs - ref_t[0]
        if t_rel[0] <= lc_rel <= t_rel[-1]:
            ax2.axvline(lc_rel, color='orange', lw=1.0, linestyle='--',
                        label='Loop closure' if i == 0 else None)

    ax2.set_title("Position Error vs Ground Truth")
    ax2.set_xlabel("Time [s]"); ax2.set_ylabel("Error [m]")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    out = Path(bag_dir) / "slam_analysis.png"
    plt.tight_layout()
    plt.savefig(str(out), dpi=150)
    print(f"\nPlot saved to: {out}")


if __name__ == '__main__':
    main()
