from time import sleep

from xiron_py.comms import XironContext
from xiron_py.data import Twist


def scan_callback(msg):
    print(f"Recieved Scan message: {msg}")


def pose_callback(msg):
    print(f"Recieved Pose message: {msg}")


def vel_cb():
    msg = Twist(ctx.now(), "robot0", [0.1, 0.0], 0.1)
    ctx.publish_velocity(msg)


def timer_cb():
    print("Timer cb")


if __name__ == "__main__":
    # Create a context object
    ctx = XironContext()

    # Create the Scan Subscriber and add callback function
    ctx.create_scan_subscriber("robot0", scan_callback)

    # Create the Pose Subscriber and add callback function
    ctx.create_pose_subscriber("robot0", pose_callback)

    ctx.create_timer(10, vel_cb)
    ctx.create_timer(10, timer_cb)

    ctx.run()
