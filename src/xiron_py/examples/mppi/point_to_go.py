from xiron_py.comms import XironContext
from xiron_py.controller.mppi import MPPIController
from xiron_py.data import Twist, Pose

import numpy as np
from time import sleep


if __name__ == "__main__":
    last_control = np.array([0.0, 0.0]).reshape(-1, 1)

    def pose_callback(msg: Pose):
        global last_control, goal
        pose_array = np.array(
            [msg.position[0], msg.position[1], msg.orientation]
        ).reshape(-1, 1)

        control = mppi.compute_contol(pose_array, last_control)

        if control is not None:
            twist_message = Twist(
                "robot0", [control[0][0].item(), 0.0], control[1][0].item()
            )
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
    dt = 0.05

    max_control = [0.5, 1.0]
    min_control = [0.0, -1.0]
    mppi = MPPIController(
        device="cpu",
        min_control=min_control,
        max_control=max_control,
        dt=dt,
        no_of_samples=2000,
    )

    plan = np.array([[0.0, 0.0, 0.0]])
    mppi.set_plan(plan)

    # Create the Pose Subscriber and add callback function
    ctx.create_pose_subscriber("robot0", pose_callback)

    while True:
        sleep(dt)