from typing import Any
import numpy as np
import yaml
from shapely.geometry import LineString, Point


def read_config_file(config_file_path):
    with open(config_file_path, "r") as file:
        yaml_data = yaml.safe_load(file)

    return yaml_data["walls"], yaml_data["static_objects"]


class Wall:
    def __init__(self, wall_dict: dict):
        self.endpoints = []
        for e in wall_dict["endpoints"]:
            self.endpoints.append([float(e[0]), float(e[1])])

        self.collision_object = LineString(self.endpoints)

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.collision_object

    def __str__(self) -> str:
        string = "----Wall----\n"
        for e in self.endpoints:
            string += f"{e[0], e[1]}\n"

        string += "-----"

        return string


class StaticObject:
    def __init__(self, static_obj_dict: dict) -> None:
        self.endpoints = []

        center = [
            float(static_obj_dict["center"][0]),
            float(static_obj_dict["center"][1]),
        ]
        width = float(static_obj_dict["width"]) * 0.5
        height = float(static_obj_dict["height"]) * 0.5
        _rotation = float(static_obj_dict["rotation"])

        self.endpoints += [
            [
                center[0] + width,
                center[1] + height,
            ],
            [
                center[0] + width,
                center[1] - height,
            ],
            [
                center[0] - width,
                center[1] - height,
            ],
            [
                center[0] - width,
                center[1] + height,
            ],
            [
                center[0] + width,
                center[1] + height,
            ],
        ]

        self.collision_object = LineString(self.endpoints)

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.collision_object

    def __str__(self) -> str:
        string = "----StaticObject----\n"
        for e in self.endpoints:
            string += f"{e[0], e[1]}\n"

        string += "-----"

        return string


class EnvironmentManager:
    def __init__(self, filename) -> None:
        walls, static_objects = read_config_file(filename)

        self.walls = [Wall(w) for w in walls]
        self.static_objects = [StaticObject(obj) for obj in static_objects]

        self.obstacles = self.walls + self.static_objects

    def isPathValid(
        self, path: list, buffer: bool = True, buffer_dist: float = 0.5
    ) -> bool:
        for i in range(0, len(path) - 1):
            next_pose = path[i + 1]
            current_pose = path[i]

            if type(next_pose) == np.ndarray:
                next_pose = [next_pose[0], next_pose[1]]
            if type(current_pose) == np.ndarray:
                current_pose = [current_pose[0], current_pose[1]]

            if buffer:
                line = LineString([current_pose, next_pose]).buffer(buffer_dist)
            else:
                line = LineString([current_pose, next_pose])

            for obj in self.obstacles:
                if line.intersects(obj.collision_object):
                    return False

        return True

    def isPoseWithin(
        self, pose: list, buffer: bool = True, buffer_dist: float = 0.25
    ) -> bool:
        if type(pose) == np.ndarray:
            pose = list(
                pose.reshape(
                    -1,
                )
            )
        if buffer:
            pose_point = Point([pose[0], pose[1]]).buffer(buffer_dist)
        else:
            pose_point = Point([pose[0], pose[1]]).buffer(buffer_dist)

        for obj in self.obstacles:
            if pose_point.intersects(obj.collision_object) or pose_point.within(
                obj.collision_object
            ):
                return False

        return True

    def distance(self, pose):
        pose_point = Point([pose[0], pose[1]])
        distances = [
            obj.collision_object.distance(pose_point) for obj in self.obstacles
        ]
        distance = sorted(distances)[0]

        return distance

    def plot(self, ax):
        for i, obst in enumerate(self.static_objects):
            if i == 0:
                ax.fill(
                    [x[0] for x in obst.endpoints],
                    [x[1] for x in obst.endpoints],
                    color="gray",
                    label="Static Objects",
                    alpha=0.5,
                )
            else:
                ax.fill(
                    [x[0] for x in obst.endpoints],
                    [x[1] for x in obst.endpoints],
                    color="gray",
                    alpha=0.5,
                )

        for i, obst in enumerate(self.walls):
            if i == 0:
                ax.plot(
                    [x[0] for x in obst.endpoints],
                    [x[1] for x in obst.endpoints],
                    color="black",
                    label="Walls",
                )
            else:
                ax.plot(
                    [x[0] for x in obst.endpoints],
                    [x[1] for x in obst.endpoints],
                    color="black",
                )

        ax.legend()
