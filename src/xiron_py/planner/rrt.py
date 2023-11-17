import numpy as np

from xiron_py.env import EnvironmentManager
from xiron_py.planner import XLIMS, YLIMS, Planner


class RRTConfig:
    def __init__(
        self,
        goal_bias_threshold: float = 0.1,
        expand_dist: float = 0.5,
        max_iter: int = 1000,
        stay_away_dist: float = 0.2,
    ) -> None:
        self.goal_bias_threshold = goal_bias_threshold
        self.expand_dist = expand_dist
        self.max_iter = max_iter
        self.stay_away_dist = stay_away_dist


class RRT(Planner):
    class Node:
        def __init__(self, x, y):
            self.x = x
            self.y = y
            self.parent = None
            self.distance = 0

        def __call__(self) -> tuple[float, float]:
            return [self.x, self.y]

    def __init__(
        self, env_manager: EnvironmentManager, config: RRTConfig = RRTConfig()
    ):
        super().__init__(env_manager)
        self.config = config
        self.nodes: list[RRT.Node] = []
        self.is_reached = False

    @staticmethod
    def distance(point1: Node, point2: Node):
        distance = np.sqrt((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2)
        return distance

    @staticmethod
    def new_vector(point1: Node, point2: Node, threshold):
        vector = RRT.Node((point2.x - point1.x), (point2.y - point1.x))
        vector.x, vector.y = (
            vector.x / np.sqrt((vector.x**2 + vector.y**2)),
            vector.y / np.sqrt((vector.x**2 + vector.y**2)),
        )
        vector.x, vector.y = (
            point1.x + vector.x * threshold,
            point1.y + vector.y * threshold,
        )
        return vector

    def compute_plan(
        self, start_pose: np.ndarray, end_pose: np.ndarray
    ) -> tuple[bool, np.ndarray]:
        self.start = RRT.Node(start_pose[0][0], start_pose[1][0])
        self.goal = RRT.Node(end_pose[0][0], end_pose[1][0])

        self.nodes.clear()
        self.nodes.append(self.start)

        for i in range(self.config.max_iter):
            new_node = self.generate_random_node()
            nearby_node = sorted(
                self.nodes, key=lambda node: RRT.distance(node, new_node)
            )[0]
            if RRT.distance(nearby_node, new_node) <= self.config.expand_dist:
                if (
                    self.env_manager.isPathValid([nearby_node(), new_node()])
                    and self.env_manager.distance(new_node())
                    > self.config.stay_away_dist
                ):
                    new_node.parent = nearby_node
                    self.nodes.append(new_node)
                else:
                    pass
            else:
                new_node = RRT.new_vector(
                    nearby_node, new_node, self.config.expand_dist
                )
                if (
                    self.env_manager.isPathValid([nearby_node(), new_node()])
                    and self.env_manager.distance(new_node())
                    > self.config.stay_away_dist
                ):
                    new_node.parent = nearby_node
                    self.nodes.append(new_node)
                else:
                    pass
            if new_node.x == self.goal.x and new_node.y == self.goal.y:
                self.is_reached = True
                path = self.backtrack(new_node)
                print(f"Found a valid path in {i+i} iterations")
                return [True, path]

        if not self.is_reached:
            path = []
            print("No valid path found after all iterations")

        return [False, path]

    def generate_random_node(self):
        if np.random.random_sample() > self.config.goal_bias_threshold:
            x = np.random.uniform(XLIMS[0], XLIMS[1])
            y = np.random.uniform(YLIMS[0], YLIMS[1])
            new_node = RRT.Node(x, y)
        else:
            new_node = self.goal
        return new_node

    def backtrack(self, goal: Node) -> list[tuple[float, float]]:
        path = [goal()]
        current = goal.parent

        while current is not None:
            path.append(current())
            current = current.parent

        return path[::-1]

    def plot(self, ax, path):
        for node in self.nodes:
            node1 = node()
            if node.parent is not None:
                node2 = node.parent()
                ax.plot(
                    [node1[0], node2[0]], [node1[1], node2[1]], color="green", alpha=0.5
                )

        ax.plot([p[0] for p in path], [p[1] for p in path], color="red")
