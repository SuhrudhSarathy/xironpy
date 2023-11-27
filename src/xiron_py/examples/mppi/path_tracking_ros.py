#!/usr/bin/env python3

from time import sleep
import numpy as np
from xiron_py.controller.mppi import MPPIController

from geometry_msgs.msg import Twist, PoseStamped
from tf_transformations import euler_from_quaternion as efq
from nav_msgs.msg import Odometry, Path

import matplotlib.pyplot as plt

import rclpy
from rclpy.node import Node

from enum import IntEnum


class PathType(IntEnum):
    Spiral = 1
    StraightLine = 2
    Circle = 3


class PathGenerator:
    def __init__(self):
        pass

    @staticmethod
    def generate_path(path_type: PathType = PathType.Spiral):
        if path_type == PathType.Spiral:

            def figure_eight_spiral(radius, num_points, rotations):
                theta = np.linspace(0, rotations * 2 * np.pi, num_points)

                x1 = radius * np.cos(theta)
                y1 = radius * np.sin(2 * theta)

                return x1, y1

            # Parameters for the figure-eight double spiral
            radius = 3
            num_points = 100
            rotations = 1

            # Generate figure-eight double spiral points
            x1, y1 = figure_eight_spiral(radius, num_points, rotations)

            return x1, y1

        elif path_type == PathType.Circle:

            def generate_circular_path(radius, num_points):
                theta = np.linspace(0, 2 * np.pi, num_points)

                x1 = radius * np.cos(theta)
                y1 = radius * np.sin(theta)

                return x1, y1

            radius = 3
            num_points = 100

            # Generate figure-eight double spiral points
            x1, y1 = generate_circular_path(radius, num_points)

            return x1, y1

        elif path_type == PathType.StraightLine:
            x1 = np.linspace(0, 10, 1000)
            y1 = np.zeros_like(x1)

            return x1, y1


class MPPINode(Node):
    def __init__(self):
        super().__init__("mppi_node")
        critics = [
            # "PathLengthCritic",
            "GoalReachingCritic",
            "AngularVelocityCritic",
            # "AlignToPathCritic",
        ]

        max_control = [0.5, 1.0]
        min_control = [0.0, -1.0]
        self.controller = MPPIController(
            device="cpu",
            min_control=min_control,
            max_control=max_control,
            dt=(1 / 30),
            no_of_samples=4000,
            timesteps=40,
            critics=critics,
            temperature=0.3,
            control_std_dev=[0.5, 0.7],
            max_horizon_distance=1.2,
        )

        self.last_control = np.array([0.0, 0.0]).reshape(-1, 1)
        self.robot_poses = []

        self.path_pub = self.create_publisher(Path, "/SD0451002/path", 10)
        self.vel_pub = self.create_publisher(Twist, "/SD0451002/cmd_vel", 10)
        self.odom_sub = self.create_subscription(
            Odometry, "/SD0451002/odom", self.pose_callback, 10
        )

        x1, y1 = PathGenerator.generate_path(PathType.Spiral)

        plan = []
        for x, y in zip(x1, y1):
            plan.append([x, y, 0])

        plan = np.array(plan).T
        print("Starting Pose: ", plan[:, 0])

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
        self.path_pub.publish(self.path_msg)
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

            control = self.controller.compute_contol(pose_array, self.last_control)

            if control is not None:
                twist_message = Twist()
                twist_message.linear.x = control[0][0]
                twist_message.angular.z = control[1][0]
                self.vel_pub.publish(twist_message)
                self.last_control = control

            else:
                twist_message = Twist()
                twist_message.linear.x = control[0][0]
                twist_message.angular.z = control[1][0]

                self.vel_pub.publish(twist_message)
                print("Got None out from the controller. Setting to zero velocity")
                self.last_control = np.array([0.0, 0.0]).reshape(-1, 1)


if __name__ == "__main__":
    rclpy.init()
    node = MPPINode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt as e:
        rclpy.shutdown()
