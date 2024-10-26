from time import sleep

from xiron_py.comms import XironContext
from xiron_py.data import Twist


def scan_callback(msg):
    # print(f"Recieved Scan message: {msg}")
    pass


def pose_callback(msg):
    # print(f"Recieved Pose message: {msg}")
    pass


if __name__ == "__main__":
    # Create a context object
    ctx = XironContext()

    # Create the Velocity publisher for robot0
    vel_pub = ctx.create_vel_publisher("robot0")

    # Create the Scan Subscriber and add callback function
    ctx.create_scan_subscriber("robot0", scan_callback)

    # Create the Pose Subscriber and add callback function
    ctx.create_pose_subscriber("robot0", pose_callback)

    twist_message = Twist("robot0", [0.1, 0.0], 0.1)
    for i in range(20):
        vel_pub.publish(twist_message)
        print("Publihsed vel: ", i)
        sleep(0.1)

    twist_message = Twist("robot0", [0.0, 0.0], 0.0)
    vel_pub.publish(twist_message)

    print("Done!")

    reset_input = input("Reset simulation? [y/n]:")

    if reset_input.lower() == "y":
        ctx.reset_simulation()
