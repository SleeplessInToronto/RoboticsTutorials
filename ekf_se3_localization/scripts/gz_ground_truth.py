#!/usr/bin/env python3
"""
Converts /world/beacons_world/dynamic_pose/info (bridged as tf2_msgs/TFMessage)
to /ground_truth/odom (nav_msgs/Odometry) for the EKF node.

The dynamic_pose/info topic is published by the Gz SceneBroadcaster system and
contains the exact physics-engine pose of every entity in the world frame —
equivalent to the Classic Gazebo libgazebo_ros_p3d plugin.
"""
import rclpy
from rclpy.node import Node
from tf2_msgs.msg import TFMessage
from nav_msgs.msg import Odometry


class GzGroundTruth(Node):
    def __init__(self):
        super().__init__('gz_ground_truth')
        self.sub = self.create_subscription(
            TFMessage,
            '/world/beacons_world/dynamic_pose/info',
            self.cb,
            10)
        self.pub = self.create_publisher(Odometry, '/ground_truth/odom', 10)

    def cb(self, msg: TFMessage):
        for tf in msg.transforms:
            if tf.child_frame_id == 'ekf_car':
                odom = Odometry()
                odom.header.stamp    = tf.header.stamp
                odom.header.frame_id = 'world'
                odom.child_frame_id  = 'base_link'
                odom.pose.pose.position.x  = tf.transform.translation.x
                odom.pose.pose.position.y  = tf.transform.translation.y
                odom.pose.pose.position.z  = tf.transform.translation.z
                odom.pose.pose.orientation = tf.transform.rotation
                self.pub.publish(odom)
                return


def main(args=None):
    rclpy.init(args=args)
    rclpy.spin(GzGroundTruth())
    rclpy.shutdown()


if __name__ == '__main__':
    main()
