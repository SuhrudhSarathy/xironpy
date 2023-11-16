from time import sleep

import numpy as np

from xiron_py.comms import XironContext
from xiron_py.controller.pid import PIDController
from xiron_py.data import Pose, Twist

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

    controller = PIDController()
    goal = [[0.0, 0.0, 0.0], [1.0, 4.0, 0.0], [3.3, -2.0, 0.0], [0.0, 0.0, 0.0]]

    controller.set_plan(goal)

    # Create the Pose Subscriber and add callback function
    ctx.create_pose_subscriber("robot0", pose_callback)

    while True:
        sleep(0.1)
