import numpy as np


class Controller:
    def __init__(self, *args, **kwargs):
        pass

    def set_plan(self, plan: np.ndarray | list) -> None:
        raise NotImplementedError

    def compute_contol(
        self, current_state: np.ndarray | list, last_contol: np.ndarray | list
    ) -> np.ndarray:
        raise NotImplementedError


class cutils:
    @staticmethod
    def get_receding_horizon_path(
        path, current_pose, max_horizon_distance, lookahead_index=0
    ):
        # print(max_horizon_distance)
        def first_after_integrated_distance(path, distance):
            # Get the last point that is atleast 1.5 m away
            cum_distance = 0.0
            for i in range(0, path.shape[1] - 2):
                distance = np.linalg.norm(path[:2, i + 1] - path[:2, i])
                cum_distance = distance + cum_distance
                if cum_distance > max_horizon_distance:
                    return i + 1
            return -1

        # first get the nearest pose in the path
        # TODO: Do we handle already travelled poses?
        path = path[:, lookahead_index:]
        nearest_pose_index = np.argmin(
            np.linalg.norm(path[:2, :] - current_pose[:2], axis=0)
        )
        new_plan = path[:, nearest_pose_index:]
        index_after_horizon_dist = first_after_integrated_distance(
            new_plan, max_horizon_distance
        )

        if index_after_horizon_dist == -1:
            return new_plan, -1
        else:
            return new_plan[
                :, : index_after_horizon_dist + 1
            ], nearest_pose_index + index_after_horizon_dist

    @staticmethod
    def get_n_points_from_nearest_path(
        path, current_pose, number_of_points, move_ahead_by=0
    ):
        nearest_pose_index = np.argmin(
            np.linalg.norm(path[:2, :] - current_pose[:2], axis=0)
        )
        return_list = cutils.get_next_elements(
            list(range(path.shape[1])),
            nearest_pose_index + move_ahead_by,
            number_of_points,
        )
        return path[:, return_list]

    @staticmethod
    def normalise_angle(theta):
        if theta > np.pi:
            return theta - 2 * np.pi
        elif theta < -np.pi:
            return theta + 2 * np.pi
        else:
            return theta

    @staticmethod
    def get_next_elements(L, i, n):
        result = []
        length = len(L)

        for j in range(i, i + n):
            if j < length:
                result.append(L[j])
            else:
                # Repeat the last element if not enough elements are left
                result.append(L[-1])

        return result
