#!/usr/bin/env python3
"""
Wraps perfect Gazebo odometry with realistic wheel-encoder drift.

Noise model (Thrun et al., Probabilistic Robotics):
  v_noisy = v + N(0, alpha_v * |v|)       proportional velocity noise
  w_noisy = w + N(0, alpha_w * |w|) + drift_w   proportional + constant heading bias

Integrating these over time produces unbounded position drift — exactly what
real odometry does.  The EKF (corrected by beacon measurements) should stay
bounded near ground truth while this path diverges.
"""
import numpy as np
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry


def _quat_to_yaw(x, y, z, w):
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


class NoisyOdomNode(Node):
    def __init__(self):
        super().__init__('noisy_odom_node')

        self.declare_parameter('alpha_v',  0.04)   # velocity noise std (fraction of |v|)
        self.declare_parameter('alpha_w',  0.04)   # angular noise std  (fraction of |w|)
        self.declare_parameter('drift_w',  0.003)  # constant heading bias [rad/s]

        self.alpha_v = self.get_parameter('alpha_v').value
        self.alpha_w = self.get_parameter('alpha_w').value
        self.drift_w = self.get_parameter('drift_w').value

        self.sub = self.create_subscription(Odometry, '/odom',       self.odom_cb, 10)
        self.pub = self.create_publisher(   Odometry, '/odom_noisy', 10)

        self.initialized = False
        self.x = self.y = self.yaw = 0.0
        self.last_t = 0.0

        self.get_logger().info(
            f'Noisy odom: alpha_v={self.alpha_v}, alpha_w={self.alpha_w}, '
            f'drift_w={self.drift_w} rad/s')

    def odom_cb(self, msg: Odometry):
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        q = msg.pose.pose.orientation

        if not self.initialized:
            self.x   = msg.pose.pose.position.x
            self.y   = msg.pose.pose.position.y
            self.yaw = _quat_to_yaw(q.x, q.y, q.z, q.w)
            self.last_t = t
            self.initialized = True
            return

        dt = t - self.last_t
        self.last_t = t
        if dt <= 0.0 or dt > 1.0:
            return

        v = msg.twist.twist.linear.x
        w = msg.twist.twist.angular.z

        # Proportional noise + constant heading drift
        v_n = v + np.random.normal(0.0, self.alpha_v * max(abs(v), 0.01))
        w_n = w + np.random.normal(0.0, self.alpha_w * max(abs(w), 0.01)) + self.drift_w

        # Euler integration of noisy velocities
        self.yaw += w_n * dt
        self.x   += v_n * np.cos(self.yaw) * dt
        self.y   += v_n * np.sin(self.yaw) * dt

        out = Odometry()
        out.header.stamp    = msg.header.stamp
        out.header.frame_id = 'odom'
        out.child_frame_id  = 'base_link'
        out.pose.pose.position.x = self.x
        out.pose.pose.position.y = self.y
        out.pose.pose.position.z = 0.0
        half = self.yaw * 0.5
        out.pose.pose.orientation.z = np.sin(half)
        out.pose.pose.orientation.w = np.cos(half)
        self.pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    rclpy.spin(NoisyOdomNode())
    rclpy.shutdown()


if __name__ == '__main__':
    main()
