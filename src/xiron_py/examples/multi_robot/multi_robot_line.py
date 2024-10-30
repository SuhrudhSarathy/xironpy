#!/usr/bin/env python3

import numpy as np
from threading import Thread
import time

from xiron_py.comms import XironContext
from xiron_py.data import Twist, LaserScan, Pose
from xiron_py.controller.pid import PIDConfig, PIDController


class RobotControl:
    def __init__(self, robot_id: str, ctx: XironContext):
        self.robot_id = robot_id
        self.ctx = ctx

        # Class Variables
        self.current_pose = None
        self.nearest_range = None

        self.vel_pub = self.ctx.create_vel_publisher(self.robot_id)
        self.odom_sub = self.ctx.create_pose_subscriber(self.robot_id, self.odom_cb)
        self.scan_sub = self.ctx.create_scan_subscriber(self.robot_id, self.scan_cb)

        # Initialise the controller
        self.path_to_follow = [
            [-5.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
            [5.0, 5.0, 0.0],
            [-5.0, 5.0, 0.0],
            [-5.0, 0.0, 0.0],
        ]

        self.last_pose = np.array(self.path_to_follow[0]).reshape(-1, 1)
        self.last_control = np.array([0.0, 0.0]).reshape(-1, 1)

        # Initialise PID controller using default config parameters
        control_config = PIDConfig()
        self.controller = PIDController(control_config)
        self.controller.set_plan(self.path_to_follow)

        # Timer for velocity control
        self.vel_pub_thread = Thread(target=self.vel_timer)

        print(f"Intialised controller for {self.robot_id}")

        self.vel_pub_thread.start()

    def odom_cb(self, msg: Pose):
        self.current_pose = np.array(
            [msg.position[0], msg.position[1], msg.orientation]
        ).reshape(-1, 1)

    def scan_cb(self, msg: LaserScan):
        mid_value = int(len(msg.values) / 2)
        self.nearest_range = min(msg.values[mid_value - 25 : mid_value + 25])

    def send_vel(self, v: float, w: float):
        twist = Twist(self.robot_id, [v, 0.0], w)

        self.vel_pub.publish(twist)

    def vel_timer(self):
        while True:
            # Do nothing, if we dont have the current pose
            if self.current_pose is None or self.nearest_range is None:
                print(f"{self.robot_id}: Did not get scan or pose")
                continue

            # If the nearest range is less than a safety distance, stop moving
            if self.nearest_range < 0.8:
                self.send_vel(0.0, 0.0)
                print(f"{self.robot_id}: Not safe to move. Stopping")
                continue

            if np.linalg.norm(self.current_pose - self.last_pose) < 0.05:
                print(f"{self.robot_id}: Reached Final destination")

                continue

            # Compute the velocity command to send using the PID controller
            vel = self.controller.compute_contol(self.current_pose, self.last_control)
            self.last_control = vel
            print(f"{self.robot_id}: Last control: {self.last_control}")

            self.send_vel(self.last_control[0][0], self.last_control[1][0])

            time.sleep(0.05)


class MultiRobotControlNode:
    def __init__(self):
        self.ctx = XironContext()
        self.max_robots = 3
        self.robot_control_nodes = []
        for i in range(self.max_robots):
            self.robot_control_nodes.append(RobotControl(f"robot{i}", self.ctx))

        print("Starting Control")


if __name__ == "__main__":
    node = MultiRobotControlNode()
    while True:
        time.sleep(0.1)
