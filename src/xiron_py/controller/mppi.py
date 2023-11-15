import torch
import torch.nn as nn


class Critic(nn.Module):
    def __init__(
        self, device: str = "cpu", weight: float = 1.0, power: int = 1.0
    ) -> None:
        super().__init__()
        self.device = torch.device(device)
        self.weight = weight
        self.power = power

    def forward(self, trajectory: torch.Tensor, goal_pose: torch.Tensor):
        raise NotImplementedError

    def __call__(self, trajectory: torch.Tensor, goal_pose: torch.Tensor):
        return torch.pow(self.weight * self.forward(trajectory, goal_pose), self.power)


class DynamicsModel(nn.Module):
    def __init__(self, device: str = "cpu") -> None:
        super().__init__()
        self.device = torch.device(device)

    def __call__(self, state: torch.Tensor, control: torch.Tensor) -> torch.Tensor:
        return self.forward(state, control)

    def forward(self, state: torch.Tensor, control: torch.Tensor):
        raise NotImplementedError


class MPPI(nn.Module):
    def __init__(
        self,
        control_dims: int = 2,
        state_dims: int = 3,
        timesteps: int = 20,
        dt: float = 0.1,
        temperature: float = 0.1,
        no_of_samples: int = 1000,
        device: str = "mps",
        max_control: list[float] = None,
        min_control: list[float] = None,
    ) -> None:
        super().__init__()
        self.control_dims = control_dims
        self.state_dims = state_dims
        self.timesteps = timesteps
        self.dt = dt
        self.no_of_samples = no_of_samples
        self.device = torch.device(device)
        self.critics: list[Critic] = []
        self.model = DynamicsModel()
        self.temperature = temperature

        self.max_control = torch.zeros((self.control_dims, self.timesteps))
        if max_control is None:
            for i in range(self.control_dims):
                self.max_control[i, :] = 0.5
        else:
            for i in range(self.control_dims):
                self.max_control[i, :] = max_control[i]

        self.min_control = torch.zeros((self.control_dims, self.timesteps))
        if min_control is None:
            for i in range(self.control_dims):
                self.min_control[i, :] = 0.5
        else:
            for i in range(self.control_dims):
                self.min_control[i, :] = min_control[i]

    def register_critic(self, critic: Critic):
        """Registers Critic functions"""
        self.critics.append(critic)

    def register_dynamics_model(self, model: DynamicsModel):
        self.model = model

    @torch.no_grad()
    def sample(
        self,
    ) -> torch.Tensor:
        """Samples Random Noise on the GPU"""

        noise_tensor = torch.randn(
            size=[self.no_of_samples, self.control_dims, self.timesteps],
            device=self.device,
        )

        return noise_tensor

    @torch.no_grad()
    def evaluate_trajectories(
        self, samples: torch.Tensor, goal_pose: torch.Tensor
    ) -> torch.Tensor:
        cost_of_trajectory = torch.zeros((self.no_of_samples, 1))
        for critic in self.critics:
            name = critic.__class__.__name__
            cost = critic(samples, goal_pose)
            print(f"Name: {name}. Mean cost: {torch.mean(cost)}")
            cost_of_trajectory += cost

        # Convert cost of trajectory to weights
        adjusted_cost = cost_of_trajectory - torch.min(cost_of_trajectory)

        return torch.exp(-(1 / self.temperature) * adjusted_cost)

    @torch.no_grad()
    def compute_average_control(
        self,
        start_control: torch.Tensor,
        weights: torch.Tensor,
        noise_samples: torch.Tensor,
    ) -> torch.Tensor:
        weights_tensor = weights.reshape(-1, 1, 1)
        weighted_sum_of_noise = torch.sum(weights_tensor * noise_samples, dim=0)
        return start_control + weighted_sum_of_noise / torch.sum(weights)

    @torch.no_grad()
    def rollout(
        self, current_state: torch.Tensor, perturbed_control: torch.Tensor
    ) -> torch.Tensor:
        rolled_out_trajectory = torch.empty(
            (
                self.no_of_samples,
                current_state.shape[0],
                self.timesteps + 1,
            )
        ).to(self.device)

        rolled_out_trajectory[:, :, 0] = current_state.reshape(
            -1,
        )

        # Do the timestep loop
        for i in range(1, self.timesteps + 1):
            next_state = self.model(
                rolled_out_trajectory[:, :, i - 1],
                perturbed_control[:, :, i - 1],
            )

            rolled_out_trajectory[:, :, i] = next_state

        return rolled_out_trajectory

    @torch.no_grad()
    def forward(
        self,
        current_state: torch.Tensor,
        current_control: torch.Tensor,
        goal_pose: torch.Tensor,
    ) -> torch.Tensor:
        # Make sure they are on the same device
        if current_state.device != self.device:
            current_state.to(self.device)

        if current_control.device != self.device:
            current_control.to(self.device)

        noise_sample = self.sample()
        current_control = current_control.to(self.device)

        # Converts a [control_dims, 1] vector to a [no of sample, control_dims, timesteps] tensor
        control_tensor = current_control * torch.ones(
            (self.control_dims, self.timesteps), device=self.device
        )
        pertubed_control = control_tensor + noise_sample
        rolled_out_trajectories = self.rollout(current_state, pertubed_control)
        weights = self.evaluate_trajectories(rolled_out_trajectories, goal_pose)
        control = self.compute_average_control(current_control, weights, noise_sample)

        return self._limit_control(control)

    def _limit_control(self, control):
        return torch.clamp(control, self.min_control, self.max_control)

    @torch.no_grad()
    def __call__(
        self,
        current_state: torch.Tensor,
        current_control: torch.Tensor,
        goal_pose: torch.Tensor,
    ) -> torch.Tensor:
        return self.forward(current_state, current_control, goal_pose)
