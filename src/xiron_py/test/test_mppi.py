from xiron_py.controller.mppi import *
import pytest
import torch

@pytest.fixture
def critics():
    return [
        "PathLengthCritic",
        "GoalReachingCritic",
        "AngularVelocityCritic",
        "AlignToPathCritic",
    ]

@pytest.fixture
def controller(critics):
    
    controller = MPPIController(device="cpu", max_control=[0.5, 1.0], min_control=[0.0, -1.0], critics=critics, timesteps=10, no_of_samples=2, control_std_dev=[0.2, 0.3])

    return controller

def test_initialisation(controller: MPPIController, critics: list):
    assert len(controller.mppi_controller.critics) == len(critics)
    assert controller.mppi_controller.model.__class__.__name__ == DiffDriveModel.__name__

def test_noise_generated(controller: MPPIController):
    noise = controller.mppi_controller.sample()
    control = torch.tensor([0.1, 0.1]).reshape(-1, 1) * torch.ones((2, 2, 10))
    perturbed_control = control + noise
    clamped_perturbed_control = controller.mppi_controller._limit_control(perturbed_control)

    print(clamped_perturbed_control[:, 0, :])
    print(clamped_perturbed_control[:, 1, :])

    assert torch.max(clamped_perturbed_control[:, 0, :]) <= 0.5
    assert torch.max(clamped_perturbed_control[:, 1, :]) <= 1.0
    assert torch.min(clamped_perturbed_control[:, 0, :]) >= 0.0
    assert torch.min(clamped_perturbed_control[:, 1, :]) >= -1.0

def test_rollout(controller: MPPIController):
    current_state = torch.tensor([0.0, 0.0, 0.0]).reshape(-1, 1)
    control = torch.tensor([0.1, 0.0]).reshape(-1, 1) * torch.ones((2, 2, 10))

    rolled_out_traj = controller.mppi_controller.rollout(current_state, control)
    assert abs(rolled_out_traj[0, 0, -1].item() - 0.1) < 1e-5
    assert abs(rolled_out_traj[1, 0, -1].item() - 0.1) < 1e-5

def test_weighted_sum(controller: MPPIController):
    VCONTROL_1 = 0.1
    VCONTROL_2 = -0.2
    WCONTROL_1 = -0.8
    WCONTROL_2 = 0.2
    WEIGHT_1 = 1.6
    WEIGHT_2 = 0.4

    current_control = torch.tensor([0.0, 0.0]).reshape(-1, 1)
    perturbed_control = torch.zeros((2, 2, 10))
    perturbed_control[0, 0, :] = VCONTROL_1
    perturbed_control[1, 0, :] = VCONTROL_2

    perturbed_control[0, 1, :] = WCONTROL_1
    perturbed_control[1, 1, :] = WCONTROL_2

    weights = torch.tensor([WEIGHT_1, WEIGHT_2]).reshape(-1, 1)

    weighted_control = controller.mppi_controller.compute_average_control(current_control, weights, perturbed_control)
    
    weights_sum = WEIGHT_1 + WEIGHT_2
    vsum_to_be_added = (VCONTROL_1 * WEIGHT_1 + VCONTROL_2 * WEIGHT_2)/weights_sum
    wsum_to_be_added = (WCONTROL_1 * WEIGHT_1 + WCONTROL_2 * WEIGHT_2)/weights_sum

    print(weighted_control[0, 0].item())
    print(weighted_control[1, 0].item())


    assert weighted_control[0, 0].item() == pytest.approx(current_control[0][0].item() + vsum_to_be_added, 1e-4)
    assert weighted_control[1, 0].item() == pytest.approx(current_control[1][0].item() + wsum_to_be_added, 1e-4)

def test_path_length_critic(critics):
    controller = MPPIController(device="cpu", max_control=[0.5, 1.0], min_control=[0.0, -1.0], critics=critics, timesteps=10, no_of_samples=3, control_std_dev=[0.2, 0.3])
    current_state = torch.tensor([0.0, 0.0, 0.0]).reshape(-1, 1)
    control = torch.zeros((3, 2, 10))

    control[0, 0, :] = 0.1
    control[1, 0, :] = 0.2
    control[2, 0, :] = 0.3

    control[0, 1, :] = 0
    control[1, 1, :] = 0

    rolled_out_traj = controller.mppi_controller.rollout(current_state, control)
    
    pathLengthCritic = PathLengthCritic()
    cost_vector = pathLengthCritic(rolled_out_traj, current_state, control)

    assert cost_vector[2][0] == pytest.approx(1.0)
    assert cost_vector[0][0] == pytest.approx(0.0)

    pathLengthCritic2 = PathLengthCritic(weight=2.0)
    cost_vector2 = pathLengthCritic2(rolled_out_traj, current_state, control)

    assert cost_vector2[2][0] == pytest.approx(2.0)
    assert cost_vector2[0][0] == pytest.approx(0.0)