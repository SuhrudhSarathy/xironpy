from dataclasses import dataclass
from typing import List


@dataclass
class Twist:
    robot_id: str
    linear: tuple[float, float]
    angular: float


@dataclass
class Pose:
    robot_id: str
    position: tuple[float, float]
    orientation: float


@dataclass
class LaserScan:
    robot_id: str
    angle_min: float
    angle_max: float
    num_readings: int
    values: List[float]
