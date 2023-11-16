from time import sleep

import numpy as np

from xiron_py.comms import XironContext
from xiron_py.controller.mpc import ModelPredictiveController
from xiron_py.data import Pose, Twist


def sample_points_in_a_circle(center, radius):
    points = []
    thetas = np.linspace(0, 2 * np.pi, 100)
    for t in thetas:
        x = center[0] + radius * np.cos(t)
        y = center[1] + radius * np.sin(t)

        theta = np.pi / 2 + t

        points.append([x, y, theta])

    return np.array(points).T


def distance(pose1, pose2):
    return np.linalg.norm(pose1[:2, :] - pose2[:2, :])


if __name__ == "__main__":
    last_control = np.array([0.0, 0.0]).reshape(-1, 1)

    def pose_callback(msg: Pose):
        global last_control, controller, path

        pose_array = np.array(
            [msg.position[0], msg.position[1], msg.orientation]
        ).reshape(-1, 1)

        nearest_pose_in_array = np.argsort(np.linalg.norm(path - pose_array, axis=0))
        plan = np.ones((3, 10)) * path[:, nearest_pose_in_array[0]].reshape(-1, 1)

        controller.set_plan(plan)
        control = controller.compute_contol(pose_array, last_control)

        if control is not None:
            twist_message = Twist("robot0", [control[0][0], 0.0], control[1][0])
            # print(twist_message)
            vel_pub.publish(twist_message)
            last_control = control
        else:
            twist_message = Twist("robot0", [0.0, 0.0], 0.0)
            vel_pub.publish(twist_message)
            print("Got None out from the controller. Setting to zero velocity")
            last_control = np.array([0.0, 0.0]).reshape(-1, 1)

    # Create a context object
    ctx = XironContext()

    # Create the Velocity publisher for robot0
    vel_pub = ctx.create_vel_publisher("robot0")

    controller = ModelPredictiveController(timesteps=9)
    path = sample_points_in_a_circle([0, 0], 3.0)

    # Create the Pose Subscriber and add callback function
    ctx.create_pose_subscriber("robot0", pose_callback)

    while True:
        sleep(0.1)
