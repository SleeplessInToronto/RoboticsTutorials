#!/usr/bin/env python3
"""
Generate an animated video recreating the RViz SLAM visualisation from a rosbag.

Renders a top-down XY view matching the slam_viz.rviz config:
  - Dark background (RGB 25, 25, 25)
  - Ground truth path  — green
  - SLAM path          — blue  (0, 180, 255)
  - SLAM map           — light blue point cloud (180, 220, 255)

Topics read
-----------
/ground_truth/odom   nav_msgs/Odometry       — ground truth robot path
/slam/odom           nav_msgs/Odometry       — PGO-corrected SLAM path
/slam/map            sensor_msgs/PointCloud2  — accumulated keyframe map

Usage:
  python3 animate_slam_bag.py <bag_dir>

Output: <bag_dir>/slam_animation.mp4
"""
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path

import rclpy
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
import rosbag2_py


# ---------------------------------------------------------------------------
# Bag reading helpers
# ---------------------------------------------------------------------------

def read_all_messages(bag_dir: str, topics: list):
    """Read all messages from bag for the given topics, sorted by bag timestamp."""
    reader = rosbag2_py.SequentialReader()
    storage_options = rosbag2_py.StorageOptions(uri=bag_dir, storage_id='sqlite3')
    converter_options = rosbag2_py.ConverterOptions('', '')
    reader.open(storage_options, converter_options)

    topic_types = reader.get_all_topics_and_types()
    type_map = {t.name: t.type for t in topic_types}

    for topic in topics:
        if topic not in type_map:
            print(f"  WARNING: topic '{topic}' not found in bag")

    filter_ = rosbag2_py.StorageFilter(topics=topics)
    reader.set_filter(filter_)

    messages = []
    while reader.has_next():
        topic, data, bag_ts_ns = reader.read_next()
        if topic not in type_map:
            continue
        msg_type = get_message(type_map[topic])
        msg = deserialize_message(data, msg_type)
        messages.append((bag_ts_ns * 1e-9, topic, msg))

    return sorted(messages, key=lambda x: x[0])


def parse_point_cloud2(msg):
    """Extract x, y arrays from a sensor_msgs/PointCloud2 (3×float32 layout)."""
    n = msg.width
    if n == 0:
        return np.array([]), np.array([])
    data = np.frombuffer(bytes(msg.data), dtype=np.float32)
    points = data.reshape(n, 3)
    return points[:, 0], points[:, 1]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) < 2:
        print("Usage: animate_slam_bag.py <bag_dir>")
        sys.exit(1)

    bag_dir = str(Path(sys.argv[1]).resolve())
    print(f"Reading: {bag_dir}")

    rclpy.init()
    messages = read_all_messages(bag_dir, [
        '/ground_truth/odom',
        '/slam/odom',
        '/slam/map',
    ])
    rclpy.shutdown()

    if not messages:
        print("No messages found.")
        sys.exit(1)

    print(f"  Loaded {len(messages)} messages")

    # -----------------------------------------------------------------------
    # Build per-topic event lists
    # -----------------------------------------------------------------------
    events = []   # (t, topic, msg)
    for t, topic, msg in messages:
        events.append((t, topic, msg))

    t0 = events[0][0]

    # -----------------------------------------------------------------------
    # Set up figure — dark background matching RViz
    # -----------------------------------------------------------------------
    bg = np.array([25, 25, 25]) / 255.0

    fig, ax = plt.subplots(figsize=(8, 8))
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)

    ax.set_aspect('equal')
    ax.set_xlim(-18, 18)
    ax.set_ylim(-10, 26)
    ax.tick_params(colors='white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    for spine in ax.spines.values():
        spine.set_edgecolor('#444444')
    ax.grid(True, color='#444444', linewidth=0.5, alpha=0.5)
    ax.set_xlabel('x [m]', color='white')
    ax.set_ylabel('y [m]', color='white')

    # Artists — map uses plot with dot marker (scatter doesn't work with blit)
    map_line,  = ax.plot([], [], '.', color='#B4DCFF', ms=1.5, alpha=0.8)
    gt_line,   = ax.plot([], [], color='#00ff00', lw=1.5, label='Ground truth')
    slam_line, = ax.plot([], [], color='#00B4FF', lw=1.2, label='SLAM')

    time_text = ax.text(0.02, 0.97, '', transform=ax.transAxes,
                        color='white', fontsize=10, va='top')

    ax.legend(loc='upper right', facecolor='#333333', labelcolor='white',
              edgecolor='#555555', fontsize=9)

    # -----------------------------------------------------------------------
    # Pre-process all events into per-frame state snapshots
    # -----------------------------------------------------------------------
    target_fps     = 10
    total_duration = events[-1][0] - t0
    frame_times    = np.arange(0, total_duration, 1.0 / target_fps)

    # Walk through events once, building cumulative state at each frame time
    gt_xs,   gt_ys   = [], []
    slam_xs, slam_ys = [], []
    cur_map_x, cur_map_y = np.array([]), np.array([])

    frames = []   # list of (t_rel, gt_x, gt_y, slam_x, slam_y, map_x, map_y)
    ei = 0
    for ft in frame_times:
        abs_ft = t0 + ft
        while ei < len(events) and events[ei][0] <= abs_ft:
            _, topic, msg = events[ei]
            if topic == '/ground_truth/odom':
                gt_xs.append(msg.pose.pose.position.x)
                gt_ys.append(msg.pose.pose.position.y)
            elif topic == '/slam/odom':
                slam_xs.append(msg.pose.pose.position.x)
                slam_ys.append(msg.pose.pose.position.y)
            elif topic == '/slam/map':
                mx, my = parse_point_cloud2(msg)
                if len(mx) > 0:
                    cur_map_x, cur_map_y = mx, my
            ei += 1
        frames.append((ft,
                        list(gt_xs), list(gt_ys),
                        list(slam_xs), list(slam_ys),
                        cur_map_x.copy(), cur_map_y.copy()))

    print(f"  Rendering {len(frames)} frames at {target_fps} fps ...")

    def update(fi):
        ft, gx, gy, sx, sy, mx, my = frames[fi]
        gt_line.set_data(gx, gy)
        slam_line.set_data(sx, sy)
        if len(mx) > 0:
            map_line.set_data(mx, my)
        time_text.set_text(f't = {ft:.1f} s')
        return gt_line, slam_line, map_line, time_text

    frame_indices = list(range(len(frames)))

    print(f"  Rendering {len(frame_indices)} frames at {target_fps} fps ...")

    ani = animation.FuncAnimation(
        fig, update,
        frames=frame_indices,
        interval=1000 // target_fps,
        blit=True)

    out = Path(bag_dir) / 'slam_animation.mp4'
    writer = animation.FFMpegWriter(fps=target_fps, bitrate=2000,
                                     extra_args=['-vcodec', 'libx264'])
    ani.save(str(out), writer=writer)
    print(f"  Saved: {out}")


if __name__ == '__main__':
    main()
