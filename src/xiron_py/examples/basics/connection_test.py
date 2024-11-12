from time import sleep

from xiron_py.comms import XironContext
from xiron_py.data import Twist


def scan_callback(msg):
    print(f"Recieved Scan message: {msg}")
    # pass


def pose_callback(msg):
    print(f"Recieved Pose message: {msg}")
    # pass


def vel_cb():
    msg = Twist(ctx.now(), "robot0", [0.5, 0.0], 0.1)
    ctx.publish_velocity(msg)


if __name__ == "__main__":
    # Create a context object
    ctx = XironContext()

    # Create the Scan Subscriber and add callback function
    ctx.create_scan_subscriber("robot0", scan_callback)

    # Create the Pose Subscriber and add callback function
    ctx.create_pose_subscriber("robot0", pose_callback)

    # Create a timer to publish velocity.
    ctx.create_timer(10, vel_cb)

    # Keep the context alive
    ctx.run()
