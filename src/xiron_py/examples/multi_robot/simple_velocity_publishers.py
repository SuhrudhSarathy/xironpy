import time
from time import sleep

from xiron_py.comms import XironContext
from xiron_py.data import Twist

from threading import Thread


def scan_callback(msg):
    print(f"Recieved Scan message: {msg}")


def pose_callback(msg):
    print(f"Recieved Pose message: {msg}")


def tcb(pub, msg):
    while True:
        pub.publish(msg)
        sleep(0.1)


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

    start = time.time()

    t1 = Thread(target=tcb, args=[vel_pub_robot0, twist_message0], daemon=True)
    t2 = Thread(target=tcb, args=[vel_pub_robot1, twist_message1], daemon=True)
    t3 = Thread(target=tcb, args=[vel_pub_robot2, twist_message2], daemon=True)

    t1.start()
    t2.start()
    t3.start()

    try:
        t1.join()
        t2.join()
        t3.join()
    except KeyboardInterrupt:
        twist_message0 = Twist("robot0", [0.5, 0.0], 0.0)
        twist_message1 = Twist("robot1", [0.5, 0.0], 0.0)
        twist_message2 = Twist("robot2", [0.5, 0.0], 0.0)

        vel_pub_robot0.publish(twist_message0)
        vel_pub_robot1.publish(twist_message1)
        vel_pub_robot2.publish(twist_message2)

        print("Closing Fine")
