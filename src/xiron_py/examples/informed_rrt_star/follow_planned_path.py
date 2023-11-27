from time import sleep

import matplotlib.pyplot as plt
import numpy as np

from xiron_py.comms import XironContext
from xiron_py.controller.mppi import *
from xiron_py.data import Pose, Twist
from xiron_py.env import EnvironmentManager
from xiron_py.planner.informed_rrt_star import InformedRRTStar, IRRTStarConfig

PLOT = True


if __name__ == "__main__":
    last_control = np.array([0.0, 0.0]).reshape(-1, 1)
    robot_poses = []

    def pose_callback(msg: Pose):
        global last_control, robot_poses
        pose_array = np.array(
            [msg.position[0], msg.position[1], msg.orientation]
        ).reshape(-1, 1)
        robot_poses.append(pose_array)

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

    dt = 1 / 30

    critics = [
        # "PathLengthCritic",
        "GoalReachingCritic",
        "AngularVelocityCritic",
        # "AlignToPathCritic",
    ]

    max_control = [0.5, 1.0]
    min_control = [0.0, -1.0]
    controller = MPPIController(
        device="cpu",
        min_control=min_control,
        max_control=max_control,
        dt=dt,
        no_of_samples=3000,
        timesteps=20,
        critics=critics,
        temperature=0.3,
        control_std_dev=[0.5, 0.9],
        max_horizon_distance=1.0,
    )

    env = EnvironmentManager("src/xiron_py/examples/informed_rrt_star/config.yaml")

    planner = InformedRRTStar(
        env,
        IRRTStarConfig(
            EXPAND_DIST=2.5,
            MAX_ITERS=5000,
            GOAL_DIST=0.5,
            BIAS_THRESHOLD=0.25,
            MAX_PLANNING_TIME=5,
        ),
    )
    start = np.array([-5.0, 2.5]).reshape(-1, 1)
    goal = np.array([4.0, 7.0]).reshape(-1, 1)
    path_found, path = planner.compute_plan(start, goal)
    print([p[0][0] for p in path])
    print([p[1][0] for p in path])
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

            fig.suptitle("InformedRRTStar")
            plt.savefig("media/planner_results/irrt_following_path.png")

        path_array = []
        for pose in path:
            path_array.append([pose[0][0], pose[1][0], 0])

        path_array = np.array(path_array)
        controller.set_plan(path_array.T)

        # Create the Pose Subscriber and add callback function
        ctx.create_pose_subscriber("robot0", pose_callback)

        try:
            while True:
                sleep(dt)
        except KeyboardInterrupt as e:
            fig, ax = plt.subplots(figsize=(10, 10))
            env.plot(ax)

            X_path = [rp[0][0] for rp in path]
            Y_path = [rp[1][0] for rp in path]

            ax.scatter(X_path, Y_path, color="green", marker="*", label="Target Path")
            ax.plot(X_path, Y_path, "g--", alpha=0.5)

            X_real = [rp[0][0] for rp in robot_poses]
            Y_real = [rp[1][0] for rp in robot_poses]

            ax.plot(
                X_real, Y_real, color="red", alpha=0.5, label="Actual Followed Path"
            )
            ax.set_title("MPPI Controller with IRRT* Planner")

            plt.legend()
            plt.axis("equal")
            # plt.show()

            plt.savefig("media/controller_results/mppi_irrt_follow_path.png")
