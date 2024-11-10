from dataclasses import dataclass
from typing import List


@dataclass
class Twist:
    timestamp: float
    robot_id: str
    linear: tuple[float, float]
    angular: float


@dataclass
class Pose:
    timestamp: float
    robot_id: str
    position: tuple[float, float]
    orientation: float


@dataclass
class LaserScan:
    timestamp: float
    robot_id: str
    angle_min: float
    angle_max: float
    num_readings: int
    values: List[float]
