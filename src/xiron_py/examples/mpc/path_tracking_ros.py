#!/usr/bin/env python3

from time import sleep
import numpy as np
from xiron_py.controller.mpc import ModelPredictiveController
from xiron_py.controller import cutils
from geometry_msgs.msg import Twist, PoseStamped
from tf_transformations import euler_from_quaternion as efq, quaternion_from_euler as qfe
from nav_msgs.msg import Odometry, Path

from ppmt_nav_common.trajectory_utils import (
    catmull_rom_spline_path,
    align_orientation_along_path,
)

import matplotlib.pyplot as plt

import rclpy
from rclpy.node import Node

from enum import IntEnum


class PathType(IntEnum):
    Spiral = 1
    StraightLine = 2
    Circle = 3
    Waypoints = 4


class PathGenerator:
    def __init__(self):
        pass

    @staticmethod
    def generate_path(path_type: PathType = PathType.Spiral):
        if path_type == PathType.Spiral:

            def figure_eight_spiral(radius, num_points, rotations):
                theta = np.linspace(0, rotations * 2 * np.pi, num_points)

                x1 = -2.0 + radius * np.cos(theta)
                y1 = radius * np.sin(2 * theta)
                orientation = theta + np.pi / 2

                return x1, y1, orientation

            # Parameters for the figure-eight double spiral
            radius = 3
            num_points = 500
            rotations = 1

            # Generate figure-eight double spiral points
            x1, y1, t1 = figure_eight_spiral(radius, num_points, rotations)

            return x1, y1, t1

        elif path_type == PathType.Circle:

            def generate_circular_path(radius, num_points):
                theta = np.linspace(0, 2 * np.pi, num_points)

                x1 = -3 + radius * np.cos(theta)
                y1 = radius * np.sin(theta)
                orientation = theta + np.pi / 2

                return x1, y1, orientation

            radius = 3
            num_points = 500

            # Generate figure-eight double spiral points
            x1, y1, t1 = generate_circular_path(radius, num_points)

            return x1, y1, t1

        elif path_type == PathType.StraightLine:
            x1 = np.linspace(0, 10, 500)
            y1 = np.zeros_like(x1)
            t1 = np.zeros(x1)

            return x1, y1, t1

        elif path_type == PathType.Waypoints:
            x1 = [
                0.5, 1.0, 1.5, 2.5, 4.0, 5.0
            ]
            y1 = [
                0.5, 1.5, 2.5, 2.0, 0.2, 0.0
            ]
            t1 = [0.0 for _ in range(len(x1))]

            path = Path()
            for x, y, t in zip(x1, y1, t1):
                pose = PoseStamped()
                pose.pose.position.x = x #- x1[0]
                pose.pose.position.y = y #- y1[0]

                path.poses.append(pose)

            path_smooth = align_orientation_along_path(
                catmull_rom_spline_path(path, path_points_per_second=20, auto_calculate_path_points=False, path_points_resolution=0.025)
            )

            x1 = []
            y1 = []
            t1 = []

            for pose in path_smooth.poses:
                x1.append(pose.pose.position.x)
                y1.append(pose.pose.position.y)

                quat = pose.pose.orientation
                t1.append(efq([quat.x, quat.y, quat.z, quat.w])[2])

            return x1, y1, t1


class MPCNode(Node):
    def __init__(self):
        super().__init__("mpc_node")
        self.controller = ModelPredictiveController(
            timesteps=20,
            dt=(1 / 20),
            max_linear_speed=1.5,
            max_angular_speed=1.0,
            reversing_allowed=False,
            max_horizon_distance=1.2,
            move_ahead_by=0,
            rotate_to_heading=True,
        )

        self.last_control = np.array([0.0, 0.0]).reshape(-1, 1)
        self.robot_poses = []

        self.path_pub = self.create_publisher(Path, "/SD0451002/path", 10)
        self.lookahead_path_pub = self.create_publisher(
            Path, "/SD0451002/lookahead_path", 10
        )
        self.mpc_path_pub = self.create_publisher(Path, "/SD0451002/mpc_path", 10)
        self.vel_pub = self.create_publisher(Twist, "/SD0451002/cmd_vel", 10)
        self.odom_sub = self.create_subscription(
            Odometry, "/SD0451002/odom", self.pose_callback, 10
        )

        x1, y1, theta1 = PathGenerator.generate_path(PathType.Spiral)

        plan = []
        for x, y, t in zip(x1, y1, theta1):
            plan.append([x, y, t])

        plan = np.array(plan).T
        print("Starting Pose: ", plan[:, 0])

        x1, y1 = plan[0, :], plan[1, :]

        self.path_msg = Path()
        self.path_msg.header.stamp = self.get_clock().now().to_msg()
        self.path_msg.header.frame_id = "odom"

        for x, y in zip(x1, y1):
            pose = PoseStamped()
            pose.pose.position.x = x
            pose.pose.position.y = y

            pose.header.stamp = self.get_clock().now().to_msg()
            pose.header.frame_id = "odom"

            self.path_msg.poses.append(pose)

        self.path_pub.publish(self.path_msg)

        sleep(3)
        self.controller.set_plan(plan)

    def pose_callback(self, msg: Odometry):
        # self.path_pub.publish(self.path_msg)
        if self.controller.lookahead_plan is not None:
            path_msg = Path()
            path_msg.header.frame_id = "odom"
            path_msg.header.stamp = self.get_clock().now().to_msg()
            for i in range(self.controller.lookahead_plan.shape[-1]):
                pose_array = self.controller.lookahead_plan[:, i]
                pose = PoseStamped()
                pose.pose.position.x = pose_array[0]
                pose.pose.position.y = pose_array[1]

                quat_list = qfe(0.0, 0.0, pose_array[2])
                pose.pose.orientation.x = quat_list[0]
                pose.pose.orientation.y = quat_list[1]
                pose.pose.orientation.z = quat_list[2]
                pose.pose.orientation.w = quat_list[3]

                pose.header.stamp = self.get_clock().now().to_msg()
                pose.header.frame_id = "odom"

                path_msg.poses.append(pose)
            self.lookahead_path_pub.publish(path_msg)

        if self.controller.plan is not None:
            yaw = efq(
                [
                    msg.pose.pose.orientation.x,
                    msg.pose.pose.orientation.y,
                    msg.pose.pose.orientation.z,
                    msg.pose.pose.orientation.w,
                ]
            )[2]
            pose_array = np.array(
                [msg.pose.pose.position.x, msg.pose.pose.position.y, yaw]
            ).reshape(-1, 1)
            self.robot_poses.append(pose_array)

            control, path = self.controller.compute_contol(pose_array, self.last_control)

            if control is not None:
                twist_message = Twist()
                twist_message.linear.x = control[0][0]
                twist_message.angular.z = control[1][0]
                self.vel_pub.publish(twist_message)
                self.last_control = control

                path_msg = Path()
                path_msg = Path()
                path_msg.header.frame_id = "odom"
                path_msg.header.stamp = self.get_clock().now().to_msg()
                if path is not None:
                    for i in range(path.shape[1]):
                        pose_array = path[:, i]
                        pose = PoseStamped()
                        pose.pose.position.x = pose_array[0]
                        pose.pose.position.y = pose_array[1]

                        quat_list = qfe(0.0, 0.0, pose_array[2])
                        pose.pose.orientation.x = quat_list[0]
                        pose.pose.orientation.y = quat_list[1]
                        pose.pose.orientation.z = quat_list[2]
                        pose.pose.orientation.w = quat_list[3]

                        pose.header.stamp = self.get_clock().now().to_msg()
                        pose.header.frame_id = "odom"

                        path_msg.poses.append(pose)
                    self.mpc_path_pub.publish(path_msg)

                # self.get_logger().info(str(twist_message))

            else:
                twist_message = Twist()
                twist_message.linear.x = control[0][0]
                twist_message.angular.z = control[1][0]

                self.vel_pub.publish(twist_message)
                self.get_logger().info(
                    "Got None out from the controller. Setting to zero velocity"
                )
                self.last_control = np.array([0.0, 0.0]).reshape(-1, 1)


if __name__ == "__main__":
    rclpy.init()
    node = MPCNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt as e:
        rclpy.shutdown()
