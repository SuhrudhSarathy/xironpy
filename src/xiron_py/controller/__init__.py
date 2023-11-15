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
