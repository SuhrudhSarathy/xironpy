# Controllers
This contains the base implementation of the Controller class. There are few examples of controllers implemented. Most of them are experimental.

| Controller        | Stable | Differential | Omnidrive | Ackermann | 
| :---------------- | :------: | :----: | :-------: | :--------: |
| PID Controller    |   ✅  | ✅ | ❌ | ❌ |
| MPC               |   ❌   | ❌ | ❌ | ❌ |
| MPPI              |  ❌   | ❌| ❌ | ❌ |

## Writing your own Controller
1. A custom controller should inherit from the base `Controller` class and implement the functions `set_plan` and `compute_control`.

2. The base controller class is implemented as follows and can be imported from `xiron_py.controller`
```python
# Base controller class
class Controller:
    def __init__(self, *args, **kwargs):
        pass

    def set_plan(self, plan: np.ndarray | list) -> None:
        raise NotImplementedError

    def compute_contol(
        self, current_state: np.ndarray | list, last_contol: np.ndarray | list
    ) -> np.ndarray:
        raise NotImplementedError

```

3. A new controller should inherit from the base `Controller` class
```python
## Custom controller
from xiron_py.controller

class MyController(Controller):
    def __init__(self, *args, **kwargs):
        super().__init__(**args, **kwargs)

        # Custom members

    def set_plan(self, plan: np.ndarray | list) -> None:
        self.plan = plan

        # custom stuff

    def compute_contol(
        self, current_state: np.ndarray | list, last_contol: np.ndarray | list
    ) -> np.ndarray:
        
        # Process current state and give out velocity

        return np.array([lin_vel, ang_vel]).reshape(-1, 1)
```

A full example of using the controller for path following is implemented [here](../examples/pid/follow_path.py)