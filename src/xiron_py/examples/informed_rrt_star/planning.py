import matplotlib.pyplot as plt
import numpy as np

from xiron_py.env import EnvironmentManager
from xiron_py.planner.informed_rrt_star import InformedRRTStar, IRRTStarConfig

if __name__ == "__main__":
    env = EnvironmentManager("src/xiron_py/examples/informed_rrt_star/config.yaml")
    fig, ax = plt.subplots()
    planner = InformedRRTStar(
        env,
        IRRTStarConfig(
            EXPAND_DIST=2.5, MAX_ITERS=5000, GOAL_DIST=0.5, BIAS_THRESHOLD=0.25
        ),
    )
    start = np.array([-5.0, 2.5]).reshape(-1, 1)
    goal = np.array([2.0, 6.0]).reshape(-1, 1)
    path_found, path = planner.compute_plan(start, goal)

    print("Path Found: ", path_found)

    env.plot(ax)
    planner.plot(ax, path)

    plt.savefig("media/planner_results/informed_rrt.png")
