# This is experimental and subject to change.

import casadi as cs
import numpy as np

from xiron_py.controller import Controller, cutils

"""
Notes
1. Having a atan2 function to guide inplace rotation is generally not working. 
    Keeping A to zero helps in providing a solution.
2. In some cases where the goal pose is behind the robot, the robot is unable to go towards the goal.
3. There are orientations where the robot gets stuck. Happens when the angle between the goal pose and the robots angle is wierdly placed
"""


class ModelPredictiveControllerOM(Controller):
    def __init__(
        self,
        model_type: str = "differential",
        timesteps: int = 10,
        dt: float = 0.1,
        max_linear_speed: float = 0.75,
        max_angular_speed: float = 1.0,
        max_horizon_distance: float = 1.2,
        reversing_allowed: bool = False,
        rotate_to_heading: bool = False,
    ):
        super().__init__()

        # Parameters
        self.model_type = model_type
        self.timesteps = timesteps
        self.dt = dt

        # Speed Limits in the Kwargs
        self.max_linear_speed = max_linear_speed
        self.max_angular_speed = max_angular_speed
        self.max_horizon_distance = max_horizon_distance
        self.reversing_allowed = reversing_allowed
        self.rotate_to_heading = rotate_to_heading

        # Cost Matrices
        self.Q = np.diag([10.0, 10.0])
        self.P = np.diag([25.0, 25.0, 1.0])
        self.R = np.diag([1.0, 2.0])
        self.A = 0.001

        self.initialise_optimisation_problem()

        self.plan = None
        self.lookahead_plan = None
        self.lookahead_index = 0

        print("Initialised Model Prdictive Controller!")

    def Exp(theta):
        return np.array(
            [[cs.cos(theta), -cs.sin(theta)], [cs.sin(theta), cs.cos(theta)]]
        )

    def Log(A):
        return np.arctan2(A[1, 0], A[0, 0])

    def initialise_optimisation_problem(self):
        # define the optimiser
        self.optimiser = cs.Opti()

        self.X = self.optimiser.variable(2, self.timesteps + 1)
        self.R = self.optimiser.variable(2, 2, self.timesteps + 1)
        self.U = self.optimiser.variable(2, self.timesteps)

        # define the parameters that need not be optimised
        self.x0 = self.optimiser.parameter(3, 1)
        self.X_track = self.optimiser.parameter(3, self.timesteps + 1)

        # Define the cost
        cost = 0.0

        # define the alpha function
        alpha_func = lambda X, Xt: X[2][0] - cs.atan2(
            Xt[1][0] - X[1][0], (Xt[0][0] - X[0][0] + 0.00001)
        )

        for i in range(self.timesteps):
            # Add Cost
            dx = self.X[:2, i] - self.X_track[:2, i]
            cost += dx.T @ self.Q @ dx + self.U[:, i].T @ self.R @ self.U[:, i]

        dx = self.X[:2, -1] - self.X_track[:2, -1]
        alpha = alpha_func(self.X[:, 0], self.x0)

        cost += dx.T @ self.Q @ dx
        cost += self.A * alpha

        # Add constraints
        for i in range(self.timesteps):
            # Dynamics constraints
            self.optimiser.subject_to(
                self.X[:, i + 1]
                == self.__differential_model(self.X[:, i], self.U[:, i])
            )

            # Value constraints
            if not self.reversing_allowed:
                self.optimiser.subject_to(0.0 <= self.U[0][0])
                self.optimiser.subject_to(self.U[0][0] <= self.max_linear_speed)
            else:
                self.optimiser.subject_to(-self.max_linear_speed <= self.U[0][0])
                self.optimiser.subject_to(self.U[0][0] <= self.max_linear_speed)

            # Angular Velocity constraints
            self.optimiser.subject_to(-self.max_angular_speed <= self.U[1][0])
            self.optimiser.subject_to(self.U[1][0] <= self.max_angular_speed)

        # Add initial position constraint
        self.optimiser.subject_to(self.X[:, 0] == self.x0)

        # Set optimiser settings and solve to minimse cost
        self.optimiser.minimize(cost)

        opts_setting = {
            "ipopt.max_iter": 200,
            "ipopt.print_level": 0,
            "print_time": 0,
            "ipopt.acceptable_tol": 1e-4,
            "ipopt.acceptable_obj_change_tol": 1e-6,
        }
        self.optimiser.solver("ipopt", opts_setting)

    def __differential_model(self, X: np.ndarray, U: np.ndarray) -> np.ndarray:
        """Computes the next state given the current state and the control vector"""
        # X : [x, y, theta].T
        # U : [v, w].T

        theta_next = X[2][0] + U[1][0] * self.dt
        x_next = X[0][0] + U[0][0] * cs.cos(theta_next)
        y_next = X[1][0] + U[0][0] * cs.sin(theta_next)

        return cs.vertcat(x_next, y_next, theta_next)

    def normalise_angle(self, theta):
        if cs.Gt(theta, cs.pi):
            return theta - 2 * cs.pi
        elif cs.lt(theta, -cs.pi):
            return theta + 2 * cs.pi
        else:
            return theta

    # rotate to heading
    def should_rotate_to_heading(self, current_pose):
        """This decides if we have to rotate in place first"""
        first_pose_in_plan = self.plan[0]
        alpha_val = current_pose[2][0] - cs.atan(
            first_pose_in_plan[1]
            - current_pose[1][0]
            / (first_pose_in_plan[0] - current_pose[0][0] + 0.00001)
        )

        if abs(alpha_val) > 0.4:
            # normalise alpha val
            return True, 0.5

        return False, 0.0

    # Controller interface functions
    def set_plan(self, plan: np.ndarray) -> None:
        self.plan = plan

    def compute_contol(
        self, current_state: np.ndarray, last_control: np.ndarray
    ) -> np.ndarray:
        # Raise exception if plan is not set
        if self.plan is None:
            raise Exception("Set Plan first")

        if self.rotate_to_heading:
            should_rotate, angular_vel = self.should_rotate_to_heading(current_state)
            if should_rotate:
                return np.array([0.0, angular_vel]).reshape(-1, 1)

        # # Reset lookahead index when you reach the end of path to continue on it again
        # if self.lookahead_index == -1:
        #     self.lookahead_index = 0

        # self.lookahead_plan, self.lookahead_index = cutils.get_receding_horizon_path(
        #     self.plan, current_state, self.max_horizon_distance, self.lookahead_index
        # )

        self.lookahead_plan = cutils.get_n_points_from_nearest_path(
            self.plan, current_state, self.timesteps + 1, move_ahead_by=5
        )

        # last_pose = self.lookahead_plan[:, -1].reshape(-1, 1)

        # plan = np.ones((3, self.timesteps + 1)) * last_pose

        # Set parameters and values for the problem
        self.optimiser.set_value(self.X_track, self.lookahead_plan)
        self.optimiser.set_value(self.x0, current_state)

        self.optimiser.set_initial(self.X, self.lookahead_plan)
        self.optimiser.set_initial(self.U, np.ones((2, self.timesteps)) * last_control)

        try:
            sol = self.optimiser.solve()

            x = sol.value(self.X)
            u = sol.value(self.U)

            return u[:, 0].reshape(-1, 1)

        except Exception as e:
            print("Exception: ", e)
            print(self.optimiser.debug.value(self.U))

            return None
