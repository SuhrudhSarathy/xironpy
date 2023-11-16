import matplotlib.pyplot as plt
import numpy as np

from xiron_py.env import EnvironmentManager
from xiron_py.planner.rrt import RRT, RRTConfig

if __name__ == "__main__":
    env = EnvironmentManager("src/xiron_py/examples/rrt/config.yaml")
    fig, ax = plt.subplots()
    planner = RRT(env, RRTConfig(0.1, expand_dist=2.5))
    start = np.array([-5.0, 2.5]).reshape(-1, 1)
    goal = np.array([4.0, 7.0]).reshape(-1, 1)
    path_found, path = planner.compute_plan(start, goal)

    env.plot(ax)
    planner.plot(ax, path)

    plt.show()
