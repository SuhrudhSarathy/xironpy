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
    def get_receding_horizon_path(path, current_pose, max_horizon_distance):
        def first_after_integrated_distance(path, distance):
            # Get the last point that is atleast 1.5 m away
            cum_distance = 0.0
            for i in range(0, path.shape[1]-2):
                distance = np.linalg.norm(path[:, i+1] - path[:, i])
                cum_distance = distance + cum_distance
                if cum_distance > max_horizon_distance:
                    return i+1
            return -1
        
        # first get the nearest pose in the path
        # TODO: Do we handle already travelled poses?
        nearest_pose_index = np.argmin(np.linalg.norm(path - current_pose, axis=0))
        new_plan = path[:, nearest_pose_index:]
        index_after_horizon_dist = first_after_integrated_distance(new_plan, max_horizon_distance)

        if index_after_horizon_dist == -1:
            return new_plan
        else:
            return new_plan[:, :index_after_horizon_dist+1]