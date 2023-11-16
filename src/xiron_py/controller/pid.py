import numpy as np

from xiron_py.controller import Controller


class PIDConfig:
    def __init__(
        self,
        linear_kp: float = 0.5,
        linear_kd: float = 0.0,
        linear_ki: float = 0.0,
        angular_kp: float = 0.5,
        angular_kd: float = 0.0,
        angular_ki: float = 0.0,
        min_linear_vel: float = 0.0,
        max_linear_vel: float = 0.5,
        min_angular_vel: float = -1.0,
        max_angular_vel: float = 1.0,
    ):
        self.linear_kp = linear_kp
        self.linear_kd = linear_kd
        self.linear_ki = linear_ki

        self.angular_kp = angular_kp
        self.angular_kd = angular_kd
        self.angular_ki = angular_ki

        self.min_linear_vel = min_linear_vel
        self.max_linear_vel = max_linear_vel

        self.min_angular_vel = min_angular_vel
        self.max_angular_vel = max_angular_vel


class PIDController(Controller):
    def __init__(self, config: PIDConfig = PIDConfig()):
        self.config = config

        # Store the errors
        self.linear_error: float = 0.0
        self.linear_error_prev: float = 0.0
        self.linear_error_integral: float = 0.0

        self.angular_error: float = 0.0
        self.angular_error_prev: float = 0.0
        self.angular_error_integral: float = 0.0

        self.plan: list = None
        self.current_goal_index: int = 0

        self.angle_threshold: float = 0.5
        self.distance_threshold: float = 0.2

        print("Initialised PID Controller")

    def distance(self, pose1, pose2):
        return np.sqrt((pose1[0] - pose2[0]) ** 2 + (pose1[1] - pose2[1]) ** 2).item()

    def reset_variables(self):
        self.linear_error: float = 0.0
        self.linear_error_prev: float = 0.0
        self.linear_error_integral: float = 0.0

        self.angular_error: float = 0.0
        self.angular_error_prev: float = 0.0
        self.angular_error_integral: float = 0.0

    def normalise(self, angle):
        if angle > np.pi:
            angle -= 2 * np.pi

            return angle

        elif angle < -np.pi:
            angle += 2 * np.pi

            return angle

        else:
            return angle

    def clamp_vels(self, velocity, vel_name: str = "linear"):
        if vel_name == "linear":
            if velocity > self.config.max_linear_vel:
                velocity = self.config.max_linear_vel
            elif velocity < self.config.min_linear_vel:
                velocity = self.config.min_linear_vel

            return velocity

        elif vel_name == "angular":
            if velocity > self.config.max_angular_vel:
                velocity = self.config.max_angular_vel
            elif velocity < self.config.min_angular_vel:
                velocity = self.config.min_angular_vel

            return velocity

    def set_plan(self, plan: np.ndarray | list) -> None:
        self.plan = plan
        self.current_goal_index = 0
        self.reset_variables()

    def compute_contol(
        self, current_state: np.ndarray | list, last_contol: np.ndarray | list
    ) -> np.ndarray:
        # Angle to pose
        target_pose = self.plan[self.current_goal_index]
        angle = (
            np.arctan2(
                target_pose[1] - current_state[1][0],
                target_pose[0] - current_state[0][0],
            )
            - current_state[2][0]
        )
        distance = self.distance(current_state.reshape(-1, 1), target_pose)

        if abs(angle) > self.angle_threshold:
            # We have rotated towards the goal position now, let's move forward towards the goal
            self.angular_error = angle
            angular_error_diff = self.angular_error - self.angular_error_prev

            ang_vel = self.clamp_vels(
                self.config.angular_kp * self.angular_error
                + self.config.angular_kd * angular_error_diff
                + self.config.angular_ki * self.angular_error_integral,
                "angular",
            )

            # Update the error variables
            self.angular_error_prev = self.angular_error
            self.angular_error_integral += self.angular_error

            return np.array([0.0, ang_vel]).reshape(-1, 1)
        else:
            if distance > self.distance_threshold:
                # We have rotated towards the goal position now, let's move forward towards the goal
                self.linear_error = distance
                linear_error_diff = self.linear_error - self.linear_error_prev

                lin_vel = self.clamp_vels(
                    self.config.linear_kp * self.linear_error
                    + self.config.linear_kd * linear_error_diff
                    + self.config.linear_ki * self.linear_error_integral,
                    "linear",
                )

                if abs(angle) > self.angle_threshold * 0.2:
                    ang_vel = self.clamp_vels(self.config.angular_kp * angle, "angular")

                else:
                    ang_vel = 0.0

                # Update the error variables
                self.linear_error_prev = self.linear_error
                self.linear_error_integral += self.linear_error

                return np.array([lin_vel, ang_vel]).reshape(-1, 1)

            else:
                # Here you pop the goal and go to the next one.
                if self.current_goal_index < len(self.plan) - 1:
                    self.current_goal_index += 1
                    print("Reached Waypoint. Going to the next waypoint")

                    # Reset all the variables
                    self.reset_variables()

                    return np.array([0.0, 0.0]).reshape(-1, 1)

                else:
                    print("Done with control")
                    self.reset_variables()

                    return np.array([0.0, 0.0]).reshape(-1, 1)
