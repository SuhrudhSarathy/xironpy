import matplotlib.pyplot as plt
import numpy as np

from xiron_py.env import EnvironmentManager
from xiron_py.planner.rrt_star import RRTStar, RRTStarConfig

if __name__ == "__main__":
    env = EnvironmentManager("src/xiron_py/examples/rrt_star/config.yaml")
    fig, ax = plt.subplots()
    planner = RRTStar(env, RRTStarConfig(EXPAND_DIST=2.5))
    start = np.array([-5.0, 2.5]).reshape(-1, 1)
    goal = np.array([4.0, 7.0]).reshape(-1, 1)
    path_found, path = planner.compute_plan(start, goal)

    print("Path Found: ", path_found)

    env.plot(ax)
    planner.plot(ax, path)

    plt.show()
