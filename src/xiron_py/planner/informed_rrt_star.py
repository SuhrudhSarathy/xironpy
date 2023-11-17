import numpy as np
from time import time
import logging

from xiron_py.planner.utils import Node, plot_path, plot_tree
from xiron_py.env import EnvironmentManager
from xiron_py.planner import Planner, XLIMS, YLIMS

LOGGING = False
DEBUG = True
TIMEIT = True
SAVE = True


class IRRTStarConfig:
    def __init__(
        self,
        NEIGHBOUR_DIST=1.0,
        STEER_DIST=0.5,
        GOAL_DIST=0.2,
        EXPAND_DIST=2.0,
        BIAS_THRESHOLD=0.1,
        MAX_ITERS=1000,
        MAX_PLANNING_TIME=5.0,
        USE_TIME=True,
    ) -> None:
        # Hyperparameters
        self.NEIGHBOUR_DIST = NEIGHBOUR_DIST
        self.STEER_DIST = STEER_DIST
        self.GOAL_DIST = GOAL_DIST
        self.EXPAND_DIST = EXPAND_DIST
        self.BIAS_THRESHOLD = BIAS_THRESHOLD
        self.MAX_ITERS = MAX_ITERS
        self.MAX_PLANNING_TIME = MAX_PLANNING_TIME
        self.USE_TIME = USE_TIME


class InformedRRTStar(Planner):
    """Main RRT* algorithm"""

    def __init__(self, cost_controller: EnvironmentManager, config: IRRTStarConfig):
        super().__init__(env_manager=cost_controller)
        self.cost_controller = cost_controller

        self.lower = np.asarray([XLIMS[0], YLIMS[0]])
        self.upper = np.asarray([XLIMS[1], YLIMS[1]])

        self.ndim = 2
        self.config = config

        # Hyperparameters
        self.NEIGHBOUR_DIST = self.config.NEIGHBOUR_DIST
        self.STEER_DIST = self.config.STEER_DIST
        self.GOAL_DIST = self.config.GOAL_DIST
        self.EXPAND_DIST = self.config.EXPAND_DIST
        self.BIAS_THRESHOLD = self.config.BIAS_THRESHOLD
        self.MAX_PLANNING_TIME = self.config.MAX_PLANNING_TIME
        self.MAX_ITERS = self.config.MAX_ITERS
        self.USE_TIME = self.config.USE_TIME

        self.XSoln = []
        self.tree = []

        self.path = []
        self.cbest = 100.0
        self.planning_complete = False
        self.i = 0

    def sample(self, start, goal, cbest) -> Node:
        self.points_sampled_in_last_iter = []

        # Sample the circle
        r = np.random.uniform(0, 1, size=(1, 1))
        theta = np.random.uniform(-np.pi, np.pi, size=(1, 1))

        x = r * np.cos(theta)
        y = r * np.sin(theta)

        X = np.hstack([x, y])

        # Now convert this to a ellipse
        x_center = (start + goal) / 2

        cmin = np.linalg.norm(start - goal)
        cbest = self.cbest

        L = np.diag([cbest * 0.5, 0.5 * np.sqrt(cbest**2 - cmin**2)])
        a = (goal - start) / cmin
        i = np.array([1, 0]).T

        M = np.outer(a, i)
        U, sigma, V_T = np.linalg.svd(M)

        M = U @ np.diag([1, np.linalg.det(U) * np.linalg.det(V_T)]) @ V_T
        Xe = M @ L @ X.T + x_center

        pt = Xe.T[0]
        sample = np.array([pt[0], pt[1]]).reshape(-1, 1)
        # print(cmin, cbest)
        return Node(sample)

    def get_nearest_node(self, zrand: Node, tree) -> Node:
        cost = np.inf
        nearest = None

        for node in tree:
            if Node.distance(zrand, node) < cost and Node.distance(zrand, node) > 1e-3:
                cost = Node.distance(zrand, node)
                nearest = node

        return nearest

    def get_nodes_in_ball(self, znew: Node, znearest: Node, tree, dist=None):
        nearest = []

        if dist is None:
            dist = self.NEIGHBOUR_DIST
        for node in tree:
            assert type(node) == Node, "Tree does not contain Node class"
            assert type(znew) == Node, "Znew is not Node"
            if (
                Node.distance(node, znew) < self.NEIGHBOUR_DIST
                and Node.distance(node, znew) > 1e-4
            ):
                nearest.append(node)

        return nearest

    def steer(self, zrand: Node, znearest: Node):
        # If the distance between the nodes is greater than DELTA, then move towards the node by delta
        distance = Node.distance(zrand, znearest)

        if distance > self.EXPAND_DIST:
            unit_vector = (zrand.state - znearest.state) / distance
            new_state = znearest.state + self.STEER_DIST * unit_vector

            return Node(new_state)

        return zrand

    def choose_parent(self, ZNear, znearest: Node, znew: Node):
        zmin = znearest
        cmin = znearest.cost + Node.distance(znearest, znew)

        for znear in ZNear:
            if self.cost_controller.isPathValid([znear.state, znew.state]):
                c_hat = znear.cost + Node.distance(znear, znew)
                if c_hat < cmin:
                    zmin = znear
                    cmin = c_hat

        return zmin, cmin

    def rewire(self, tree, ZNear, zmin: Node, znew: Node):
        for znear in ZNear:
            if znear != zmin:
                # If znear and znew are traversable
                if self.cost_controller.isPathValid([znear.state, znew.state]):
                    # If connecting them reduces cost
                    if znew.cost + Node.distance(znew, znear) < znear.cost:
                        # Add znew as the parent to znear and change costs
                        znear.parent = znew
                        znear.cost = znew.cost + Node.distance(znear, znew)

    def get_best_solution(self, Xsoln):
        cbest = np.inf
        zbest = None

        for node in Xsoln:
            if node.cost < cbest:
                cbest = node.cost
                zbest = node

        return zbest, cbest

    def backtrack(self, node: Node):
        """Return a list of states"""

        # Sometimes it is getting stuck in a loop of nodes. It is better to return then
        i = 0
        if LOGGING:
            logging.debug(f"Bactracking started at {time.time()}")
        current = node
        path = [self.goal.state]

        while current is not None and i < 500:
            i += 1
            if LOGGING:
                logging.info(f"Current : {current}; time: {time.time()}")
            path.append(current.state)
            current = current.parent

        return path[::-1]

    """Main Search Algorithm"""

    def plan(self, start: np.ndarray, goal: np.ndarray, n: int):
        # Initialise tree
        self.tree = []
        self.start = Node(start)
        self.goal = Node(goal)

        # Add Start to the tree
        self.tree.append(self.start)

        if self.USE_TIME:
            start_time = time()
        else:
            iteration = 0

        done = False
        # while n iterations
        while not done:
            # Sample a new node
            zrand = self.sample(start, goal, self.cbest)

            # Get the nearest node
            znearest = self.get_nearest_node(zrand, self.tree)

            # Steer. This is usually to move towards nearest node towards the goal. This can be customized as per use
            znew = self.steer(zrand, znearest)

            # Check if you can traverse between znew and znearest
            if self.cost_controller.isPathValid([znew.state, znearest.state]):
                # Here is where RRT* differs
                # Get the nodes within a ball
                ZNear = self.get_nodes_in_ball(znew, znearest, self.tree)

                # Select the parent from the nearest nodes
                zmin, cmin = self.choose_parent(ZNear, znearest, znew)

                # Add parent and insert it into the tree
                znew.parent = zmin
                znew.cost = zmin.cost + Node.distance(zmin, znew)
                self.tree.append(znew)

                # Rewire
                self.rewire(self.tree, ZNear, zmin, znew)

                # Check if goal location is reached
                if Node.distance(
                    znew, self.goal
                ) < self.GOAL_DIST and self.cost_controller.isPathValid(
                    [znew.state, self.goal.state]
                ):
                    # Add the node to XSoln as this is a feasible solution
                    self.path = self.backtrack(znew)
                    self.cbest = znew.cost + Node.distance(znew, self.goal)

                    # bactrack and check if there is start at the end of the this path
                    # if not remove znew from XSoln

                    for point in self.path:
                        if np.linalg.norm(point - self.start.state) < 1e-3:
                            self.XSoln.append(znew)
                            break
                    else:
                        self.tree.remove(znew)

                    # Store self.path as the best solution
                    zbest, cbest = self.get_best_solution(self.XSoln)
                    self.path = self.backtrack(zbest)
                    self.cbest = cbest + Node.distance(zbest, self.goal)

            # Update iteration condition
            if self.USE_TIME:
                if time() - start_time >= self.MAX_PLANNING_TIME:
                    done = True

                # print(
                #     f"Elapsed time: {time() - start_time : 0.4f}",
                # )

            else:
                iteration += 1
                if iteration == self.MAX_ITERS:
                    done = True

                if iteration % 100 == 0:
                    print("Iteration: ", iteration)

        # Now all the time is over. This is not anytime run
        # Check for solutions and backtrack

        # No feasible solution found
        if len(self.XSoln) == 0:
            print("No feasible solution found")
            return False, [], np.inf

        # Atleast one solution found
        else:
            print("Atleast one solution found")
            zbest, cbest = self.get_best_solution(self.XSoln)

            # backtrack from zbest
            path = self.backtrack(zbest)
            self.planning_complete = True
            self.path = path
            self.cbest = cbest
            return True, path, cbest

    def compute_plan(
        self, start_pose: np.ndarray, end_pose: np.ndarray
    ) -> tuple[bool, np.ndarray]:
        path_found, path, self.cbest = self.plan(
            start_pose, end_pose, self.config.MAX_ITERS
        )

        return [path_found, path]

    def plot_path(self, ax, path=None):
        plot_path(path, ax, "IRRT-Path", "green")

    def plot_tree(self, ax):
        plot_tree(self.tree, ax)

    def plot(self, ax, path=[]):
        self.plot_path(ax, path)
        self.plot_tree(ax)

    def get_path_length(self, path):
        dist = 0.0
        for i in range(0, len(path) - 1):
            x1, y1 = path[i]
            x2, y2 = path[i + 1]
            dist += np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

        return dist
