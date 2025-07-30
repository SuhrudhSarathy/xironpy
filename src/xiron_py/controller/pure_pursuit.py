from typing import override
import numpy as np
from xiron_py.controller import Controller


class PurePursuitConfig:
    def __init__(
        self,
        lookahead_distance: float = 0.5,
        max_linear_vel: float = 0.5,
        wheelbase: float = 0.3,  # Distance between front and rear axle
    ):
        self.lookahead_distance = lookahead_distance
        self.max_linear_vel = max_linear_vel
        self.wheelbase = wheelbase


class PurePursuitController(Controller):
    def __init__(self, config: PurePursuitConfig = PurePursuitConfig()):
        super().__init__()
        self.config = config
        self.plan = None
        self.path = None

    def transform_path(self, current_pose: np.ndarray):
        x, y, theta = list(
            current_pose.reshape(
                -1,
            )
        )

        tf_matrix = np.array(
            [
                [np.cos(theta), -np.sin(theta), x],
                [np.sin(theta), np.cos(theta), y],
                [0, 0, 1],
            ]
        )
        tfed_plan = np.linalg.inv(tf_matrix) @ self.plan

        return tfed_plan

    def get_nearest_point_idx(self, plan: np.ndarray):
        norm_of_path = np.linalg.norm(plan[:2, :])
        return np.argmin(norm_of_path)

    @override
    def set_target_plan(self, plan: np.ndarray | list) -> None:
        self.plan = np.array(plan)[:2, :]  # Extract x, y coordinates
        ones = np.ones((1, self.plan.shape[1]))
        self.plan = np.vstack([self.plan, ones])

    @override
    def get_control_input(
        self, current_state: np.ndarray | list, last_control: np.ndarray | list
    ) -> np.ndarray:
        plan_in_robot_frame = self.transform_path(current_state)
        nearest_pt_in_path = self.get_nearest_point_idx(plan_in_robot_frame)

        # Check if the
