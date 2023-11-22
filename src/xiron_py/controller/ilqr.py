import numpy as np
from xiron_py.controller import Controller, cutils
import scipy
import scipy.linalg


class ILQR(Controller):
    def __init__(
        self,
        Q: np.ndarray,
        R: np.ndarray,
        max_horizon_distance: float = 1.2,
        max_control: list[float] = [0.5, 1.0],
        min_control: list[float] = [-0.5, -1.0],
    ):
        super().__init__()
        self.plan = None
        self.Q = Q
        self.R = R
        self.max_horizon_distance = max_horizon_distance
        self.max_control = max_control
        self.min_control = min_control

    def get_AB_matrices(self, state: np.ndarray, control: np.ndarray):
        A = np.zeros((3, 3))
        B = np.zeros((3, 2))

        A[2, 0] = -control[0][0] * np.sin(state[2][0])
        A[2, 1] = control[0][0] * np.cos(state[2][0])

        B[0, 0] = np.cos(state[2][0])
        B[1, 0] = np.sin(state[2][0])
        B[2, 1] = 1

        return A, B

    def get_k_matrix(self, A, B, Q, R):
        P = scipy.linalg.solve_discrete_are(A, B, Q, R)
        K = np.dot(np.linalg.inv(R), np.dot(B.T, P))

        return K

    def set_plan(self, plan: np.ndarray | list) -> None:
        self.plan = plan
        self.goal_pose = self.plan[:, -1].reshape(-1, 1)

    def compute_contol(
        self, current_state: np.ndarray | list, last_contol: np.ndarray | list
    ) -> np.ndarray:
        # Get lookahead path
        # lookahead_path = cutils.get_receding_horizon_path(
        #     self.plan,
        #     current_pose=current_state,
        #     max_horizon_distance=self.max_horizon_distance,
        # )
        # lookahead_pose = lookahead_path[:, -1].reshape(-1, 1)
        
        # Linearise the model near the current_state and last_control
        A, B = self.get_AB_matrices(self.goal_pose, last_contol)
        try:
            K = self.get_k_matrix(A, B, self.Q, self.R)

            control = last_contol - K @ (current_state - self.goal_pose)
            control[0][0] = np.clip(
                control[0][0], self.min_control[0], self.max_control[0]
            )
            control[1][0] = np.clip(
                control[1][0], self.min_control[1], self.max_control[1]
            )

            return control
        except Exception as e:
            print(e)
            return None
