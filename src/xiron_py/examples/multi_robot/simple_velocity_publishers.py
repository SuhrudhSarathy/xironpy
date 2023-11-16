from time import sleep

from xiron_py.comms import XironContext
from xiron_py.data import Twist


def scan_callback(msg):
    print(f"Recieved Scan message: {msg}")


def pose_callback(msg):
    print(f"Recieved Pose message: {msg}")


if __name__ == "__main__":
    # Create a context object
    ctx = XironContext()

    # Create the Velocity publisher for robot0
    vel_pub_robot0 = ctx.create_vel_publisher("robot0")
    vel_pub_robot1 = ctx.create_vel_publisher("robot1")
    vel_pub_robot2 = ctx.create_vel_publisher("robot2")

    # Create the Scan Subscriber and add callback function
    ctx.create_scan_subscriber("robot0", scan_callback)

    # Create the Pose Subscriber and add callback function
    ctx.create_pose_subscriber("robot0", pose_callback)

    twist_message0 = Twist("robot0", [0.5, 0.0], 0.0)
    twist_message1 = Twist("robot1", [0.5, 0.0], 0.0)
    twist_message2 = Twist("robot2", [0.5, 0.0], 0.0)

    try:
        while True:
            vel_pub_robot0.publish(twist_message0)
            vel_pub_robot1.publish(twist_message1)
            vel_pub_robot2.publish(twist_message2)
            sleep(0.1)
    except KeyboardInterrupt as e:
        twist_message0 = Twist("robot0", [0.5, 0.0], 0.0)
        twist_message1 = Twist("robot1", [0.5, 0.0], 0.0)
        twist_message2 = Twist("robot2", [0.5, 0.0], 0.0)

        vel_pub_robot0.publish(twist_message0)
        vel_pub_robot1.publish(twist_message1)
        vel_pub_robot2.publish(twist_message2)

        print("Closing Fine")
