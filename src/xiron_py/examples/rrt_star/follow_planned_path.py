from time import sleep

import matplotlib.pyplot as plt
import numpy as np

from xiron_py.comms import XironContext
from xiron_py.controller.pid import PIDConfig, PIDController
from xiron_py.data import Pose, Twist
from xiron_py.env import EnvironmentManager
from xiron_py.planner.rrt_star import RRTStar, RRTStarConfig

PLOT = True


if __name__ == "__main__":
    last_control = np.array([0.0, 0.0]).reshape(-1, 1)

    def pose_callback(msg: Pose):
        global last_control
        pose_array = np.array(
            [msg.position[0], msg.position[1], msg.orientation]
        ).reshape(-1, 1)

        control = controller.compute_contol(pose_array, last_control)

        if control is not None:
            twist_message = Twist("robot0", [control[0][0], 0.0], control[1][0])
            print(twist_message)
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

    controller = PIDController(
        PIDConfig(angular_kp=0.5, angular_kd=0.0, angular_ki=0.0)
    )

    env = EnvironmentManager("src/xiron_py/examples/rrt/config.yaml")

    planner = RRTStar(env, RRTStarConfig(0.01, expand_dist=2.5, stay_away_dist=0.0))
    start = np.array([-5.0, 2.5]).reshape(-1, 1)
    goal = np.array([4.0, 7.0]).reshape(-1, 1)
    path_found, path = planner.compute_plan(start, goal)
    if not path_found:
        fig, ax = plt.subplots()
        env.plot(ax)
        planner.plot(ax, [])

        fig.suptitle("RRT")
        plt.show()

    if path_found:
        if PLOT:
            fig, ax = plt.subplots()
            env.plot(ax)
            planner.plot(ax, path)

            fig.suptitle("RRT")
            plt.savefig("media/planner_results/rrt.png")

        controller.set_plan(path)

        # Create the Pose Subscriber and add callback function
        ctx.create_pose_subscriber("robot0", pose_callback)

        while True:
            sleep(0.1)
