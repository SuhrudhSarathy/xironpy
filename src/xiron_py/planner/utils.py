"""Utilities"""

import numpy as np


class Node:
    def __init__(self, state):
        self.state = state
        self.cost = 0
        self.parent = None

    @classmethod
    def distance(cls, node1, node2):
        assert (
            type(node1.state) == np.ndarray
        ), f"Node1 state is not numpy array. It is {type(node2.state)}"
        assert (
            type(node2.state) == np.ndarray
        ), f"Node2 state is not numpy array. It is {type(node2.state)}"
        return np.linalg.norm(node1.state - node2.state)

    def __str__(self):
        return str(self.state)

    def __repr__(self) -> str:
        return str(self.state)


def plot_tree(tree, ax):
    """Plots a given tree.
    TODO: Try to visualise better with better performance
    """
    nodesx = []
    nodesy = []

    for node in tree:
        state = node.state.reshape(
            node.state.shape[0],
        )
        x, y = state[0], state[1]

        nodesx.append(x)
        nodesy.append(y)

        if node.parent is not None:
            parent_state = node.parent.state.reshape(
                node.state.shape[0],
            )
            x_, y_ = parent_state[0], parent_state[1]

            ax.plot([x, x_], [y, y_], color="green", alpha=0.25)

    ax.scatter(nodesx, nodesy, color="red", alpha=0.5)


def plot_path(path, ax, path_name=None, color=None):
    """Plots the path given. Assumes that the first point is start and the last is goal"""
    if len(path) != 0:
        if type(path[0]) == np.ndarray:
            pointsx = [pt[0][0] for pt in path]
            pointsy = [pt[1][0] for pt in path]
        else:
            pointsx = [pt.state[0][0] for pt in path]
            pointsy = [pt.state[1][0] for pt in path]

        if path_name is not None:
            ax.plot(
                pointsx, pointsy, label=f"{path_name}", color=f"{color}", linewidth=3
            )
            ax.scatter(pointsx, pointsy, alpha=0.75, color="red")
        else:
            ax.plot(pointsx, pointsy, color="blue", linewidth=3)
            ax.scatter(pointsx, pointsy, alpha=0.75, color="red")

        ax.scatter([pointsx[0], pointsx[-1]], [pointsy[0], pointsy[-1]], color="blue")
