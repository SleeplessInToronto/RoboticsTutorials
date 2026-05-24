#!/usr/bin/env python3
"""Read a ros2 bag containing /ekf/error and /ekf/pose and produce a 2-panel
error + 1-sigma consistency plot."""

import sys
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from rclpy.serialization import deserialize_message
from geometry_msgs.msg import Vector3Stamped, PoseWithCovarianceStamped


def read_bag(bag_path: str):
    reader = SequentialReader()
    reader.open(
        StorageOptions(uri=bag_path, storage_id='sqlite3'),
        ConverterOptions('', ''))

    t_err, trans_err, rot_err = [], [], []
    t_cov, sigma_pos, sigma_yaw = [], [], []

    while reader.has_next():
        topic, data, stamp_ns = reader.read_next()
        t_sec = stamp_ns * 1e-9

        if topic == '/ekf/error':
            msg = deserialize_message(data, Vector3Stamped)
            t_err.append(t_sec)
            trans_err.append(msg.vector.x)
            rot_err.append(msg.vector.y)

        elif topic == '/ekf/pose':
            msg = deserialize_message(data, PoseWithCovarianceStamped)
            c = msg.pose.covariance          # flat 36-element row-major 6x6
            t_cov.append(t_sec)
            sigma_pos.append(math.sqrt(max(c[0] + c[7], 0.0)))
            sigma_yaw.append(math.sqrt(max(c[35], 0.0)))

    return (np.array(t_err),  np.array(trans_err), np.array(rot_err),
            np.array(t_cov),  np.array(sigma_pos),  np.array(sigma_yaw))


def normalise_time(*arrays):
    t0 = min(a[0] for a in arrays if len(a))
    return [a - t0 for a in arrays]


def make_plot(bag_path: str, out_path: str):
    t_err, trans_err, rot_err, t_cov, sigma_pos, sigma_yaw = read_bag(bag_path)

    t_err, t_cov = normalise_time(t_err, t_cov)

    blue   = '#2563EB'
    shade  = '#D1D5DB'

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    fig.subplots_adjust(hspace=0.08)

    # ── top: translational error ──────────────────────────────────────────
    ax = axes[0]
    ax.fill_between(t_cov, sigma_pos, 0,
                    color=shade, alpha=0.45, label='1σ position')
    ax.plot(t_err, trans_err, color=blue, linewidth=1.2, label='EKF error')
    ax.axhline(0, color='black', linewidth=0.6, linestyle='--')
    ax.set_ylabel('Translational error (m)', fontsize=11)
    ax.set_ylim(bottom=0)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.spines[['top', 'right']].set_visible(False)

    # ── bottom: rotational error ──────────────────────────────────────────
    ax = axes[1]
    ax.fill_between(t_cov,  sigma_yaw, -sigma_yaw,
                    color=shade, alpha=0.45, label='±1σ yaw')
    ax.plot(t_err, rot_err,   color=blue, linewidth=1.2, label='EKF error')
    ax.axhline(0, color='black', linewidth=0.6, linestyle='--')
    ax.set_ylabel('Rotational error (rad)', fontsize=11)
    ax.set_xlabel('Time (s)', fontsize=11)
    ymax = np.abs(rot_err).max() * 2.0
    ax.set_ylim(-ymax, ymax)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.spines[['top', 'right']].set_visible(False)

    fig.suptitle('EKF SE(3) Localization — Error & 1σ Consistency', fontsize=13)

    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'Saved → {out_path}')


if __name__ == '__main__':
    bag  = sys.argv[1] if len(sys.argv) > 1 else '/home/gui/ekf_data'
    out  = sys.argv[2] if len(sys.argv) > 2 else \
           '/home/gui/Robotics/ekf_se3_localization/docs/ekf_error_plot.png'
    make_plot(bag, out)
