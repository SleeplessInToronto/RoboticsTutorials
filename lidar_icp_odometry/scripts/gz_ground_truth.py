#!/usr/bin/env python3
"""
Converts /world/icp_world/dynamic_pose/info (bridged as tf2_msgs/TFMessage)
to /ground_truth/odom (nav_msgs/Odometry) for comparison with /icp/odom.

Adapted from ekf_se3_localization/scripts/gz_ground_truth.py.
Only change: world name icp_world, model name lidar_car.
"""
import rclpy
from rclpy.node import Node
from tf2_msgs.msg import TFMessage
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped


class GzGroundTruth(Node):
    def __init__(self):
        super().__init__('gz_ground_truth')
        self.sub = self.create_subscription(
            TFMessage,
            '/world/icp_world/dynamic_pose/info',
            self.cb,
            10)
        self.pub      = self.create_publisher(Odometry, '/ground_truth/odom',  10)
        self.path_pub = self.create_publisher(Path,     '/ground_truth/path',  10)
        self.path     = Path()
        self.path.header.frame_id = 'world'

    def cb(self, msg: TFMessage):
        for tf in msg.transforms:
            if tf.child_frame_id == 'lidar_car':
                odom = Odometry()
                odom.header.stamp    = tf.header.stamp
                odom.header.frame_id = 'world'
                odom.child_frame_id  = 'base_link'
                odom.pose.pose.position.x  = tf.transform.translation.x
                odom.pose.pose.position.y  = tf.transform.translation.y
                odom.pose.pose.position.z  = tf.transform.translation.z
                odom.pose.pose.orientation = tf.transform.rotation
                self.pub.publish(odom)

                ps = PoseStamped()
                ps.header = odom.header
                ps.pose   = odom.pose.pose
                self.path.header.stamp = tf.header.stamp
                self.path.poses.append(ps)
                self.path_pub.publish(self.path)
                return


def main(args=None):
    rclpy.init(args=args)
    rclpy.spin(GzGroundTruth())
    rclpy.shutdown()


if __name__ == '__main__':
    main()
