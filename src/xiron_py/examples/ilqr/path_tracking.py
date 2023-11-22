# TODO : Add Path tracking API to the MPPI Controller
from time import sleep
import numpy as np
from xiron_py.comms import XironContext
from xiron_py.controller.ilqr import ILQR
from xiron_py.data import Pose, Twist

import matplotlib.pyplot as plt

last_control = np.array([0.001, 0.001]).reshape(-1, 1)
robot_poses = []

def pose_callback(msg: Pose):
    global last_control, robot_poses
    pose_array = np.array([msg.position[0], msg.position[1], msg.orientation]).reshape(
        -1, 1
    )
    robot_poses.append(pose_array)

    control = controller.compute_contol(pose_array, last_control)

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
        last_control = np.array([0.001, 0.001]).reshape(-1, 1)


# Create a context object
ctx = XironContext()

# Create the Velocity publisher for robot0
vel_pub = ctx.create_vel_publisher("robot0")
dt = (1/30)

critics = [
    # "PathLengthCritic",
    "GoalReachingCritic",
    "AngularVelocityCritic",
    # "AlignToPathCritic",
]

max_control = [0.5, 1.0]
min_control = [0.0, -1.0]
controller = ILQR(np.diag([10.0, 10.0, 1.0]), np.diag([0.1, 0.1]))

def figure_eight_spiral(radius, num_points, rotations):
    theta = np.linspace(0, rotations * 2 * np.pi, num_points)
    
    x1 = radius * np.cos(theta)
    y1 = radius * np.sin(2*theta)
    
    return x1, y1

# Parameters for the figure-eight double spiral
radius = 3
num_points = 100
rotations = 1

# Generate figure-eight double spiral points
x1, y1 = figure_eight_spiral(radius, num_points, rotations)

plan = []
for (x, y) in zip(x1, y1):
    plan.append([x, y, 0])

plan = np.array(plan).T
print("Starting Pose: ", plan[:, 0])

controller.set_plan(plan)

# Create the Pose Subscriber and add callback function
ctx.create_pose_subscriber("robot0", pose_callback)

try:
    while True:
        sleep(dt)
except KeyboardInterrupt as e:
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(x1, y1, color="green", marker="*", label="Target Path")
    ax.plot(x1, y1, "g--", alpha=0.5)

    X_real = [rp[0][0] for rp in robot_poses]
    Y_real = [rp[1][0] for rp in robot_poses]

    ax.plot(X_real, Y_real, color="red", alpha=0.5, label="Actual Followed Path")
    ax.set_title("MPPI Controller")
    
    plt.legend()
    plt.axis('equal')
    # plt.show()

    plt.savefig("media/controller_results/ilqr_diff_drive.png")