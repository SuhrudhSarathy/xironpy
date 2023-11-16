import numpy as np

from xiron_py.env import EnvironmentManager


class Planner:
    def __init__(self, env_manager: EnvironmentManager):
        self.env_manager = env_manager

    def compute_plan(
        self, start_pose: np.ndarray, end_pose: np.ndarray
    ) -> tuple[bool, np.ndarray]:
        raise NotImplementedError


# Some global parameters for the scene
XLIMS = [-15.0, 15.0]
YLIMS = [-15.0, 15.0]
