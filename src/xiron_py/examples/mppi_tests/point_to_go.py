from xiron_py.comms import XironContext
from xiron_py.controller.mppi import MPPI, DynamicsModel, Critic
from xiron_py.data import Twist, Pose

import torch
from time import sleep


class DiffDriveModel(DynamicsModel):
    def __init__(self, dt: float = 0.1, device="cpu") -> None:
        super().__init__(device)
        self.dt = dt

    def forward(self, state, control):
        theta = state[:, 2] + control[:, 1] * self.dt
        x = state[:, 0] + control[:, 0] * torch.cos(theta) * self.dt
        y = state[:, 1] + control[:, 0] * torch.sin(theta) * self.dt

        return torch.vstack([x, y, theta]).T


class PathLengthCritic(Critic):
    def __init__(self, device: str = "cpu") -> None:
        super().__init__(device)

        self.device = torch.device(device)

    def forward(self, trajectory, goal_pose):
        cost_vec = torch.zeros((trajectory.shape[0], 1))
        cost_vec += cost_vec

        # loop 20
        for t in range(1, trajectory.shape[2]):
            cost_vec += (
                (
                    (trajectory[:, 0, t] - trajectory[:, 0, t - 1]) ** 2
                    + (trajectory[:, 1, t] - trajectory[:, 1, t - 1]) ** 2
                )
                .reshape(-1, 1)
                .to(self.device)
            )

        return cost_vec / torch.max(cost_vec)


class GoalCritic(Critic):
    def __init__(self, device: str = "cpu") -> None:
        super().__init__(device, weight=1.0)

    def forward(self, trajectory: torch.Tensor, goal_pose: torch.Tensor):
        cost_vec = torch.zeros((trajectory.shape[0]))
        diff_vector = trajectory - goal_pose

        # loop 20
        for t in range(0, trajectory.shape[2]):
            x_items = diff_vector[:, 0, t]
            y_items = diff_vector[:, 1, t]

            dist = x_items**2 + y_items**2
            cost_vec += dist

        cost_vec = torch.sqrt(cost_vec)
        cost_vec /= torch.max(cost_vec)
        return cost_vec.reshape(-1, 1)


if __name__ == "__main__":
    last_control = torch.tensor([0.0, 0.0]).reshape(-1, 1)

    def pose_callback(msg: Pose):
        global last_control, goal
        pose_array = torch.tensor(
            [msg.position[0], msg.position[1], msg.orientation]
        ).reshape(-1, 1)

        control = mppi(pose_array, last_control, goal)

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
            last_control = torch.tensor([0.0, 0.0]).reshape(-1, 1)

    # Create a context object
    ctx = XironContext()

    # Create the Velocity publisher for robot0
    vel_pub = ctx.create_vel_publisher("robot0")
    dt = 0.05

    max_control = [0.5, 1.0]
    min_control = [0.0, -1.0]
    mppi = MPPI(
        device="cpu",
        min_control=min_control,
        max_control=max_control,
        dt=dt,
        no_of_samples=2000,
    )
    dynamics_model = DiffDriveModel(device="cpu")
    mppi.register_dynamics_model(dynamics_model)
    mppi.register_critic(PathLengthCritic())
    mppi.register_critic(GoalCritic())

    goal = torch.tensor([0.0, 0.0, 0.0]).reshape(-1, 1)

    # Create the Pose Subscriber and add callback function
    ctx.create_pose_subscriber("robot0", pose_callback)

    while True:
        sleep(dt)
