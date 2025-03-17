import time
from threading import Thread
from time import sleep

from xiron_py.comms import XironContext
from xiron_py.data import Twist


def scan_callback(msg):
    print(f"Recieved Scan message: {msg}")


def pose_callback(msg):
    print(f"Recieved Pose message: {msg}")


def robot0_vel_cb():
    twist_message0 = Twist(ctx.now(), "robot0", [0.5, 0.0], 0.0)
    ctx.publish_velocity(twist_message0)


def robot1_vel_cb():
    twist_message1 = Twist(ctx.now(), "robot1", [0.5, 0.0], 0.0)
    ctx.publish_velocity(twist_message1)


def robot2_vel_cb():
    twist_message2 = Twist(ctx.now(), "robot2", [0.5, 0.0], 0.0)
    ctx.publish_velocity(twist_message2)


if __name__ == "__main__":
    # Create a context object
    ctx = XironContext()

    # Create the Scan Subscriber and add callback function
    ctx.create_scan_subscriber("robot0", scan_callback)

    # Create the Pose Subscriber and add callback function
    ctx.create_pose_subscriber("robot0", pose_callback)

    # Create velocity timers for multi robots
    ctx.create_timer(10, robot0_vel_cb)
    ctx.create_timer(10, robot1_vel_cb)
    ctx.create_timer(10, robot2_vel_cb)

    # Keep the context alive
    ctx.run()
