from Robot import Robot
from Obstacle import Obstacle
from typing import List
import numpy as np

class Environment:
    """
    A class to represent the environment in which the robot and obstacles move.

    Attributes:
        width (float): The width of the environment in meters.
        height (float): The height of the environment in meters.
        obstacles (list): A list of obstacles in the environment.
        robot (Robot): The robot in the environment.
    """
    def __init__(self, width: float, height: float, obstacles: List[Obstacle], robot: Robot):
        self.width = width
        self.height = height
        self.obstacles = obstacles
        self.robot = robot

    def next_state(self, dt: float):
        self.robot.next_state(dt)
        for obstacle in self.obstacles:
            obstacle.next_state(dt)


def random_environment(width: float, height: float, num_obstacles: int, min_obstacle_radius: float, max_obstacle_radius: float) -> Environment:
    """
    Generate a random environment with a given number of obstacles.
    """
    min_speed = 0.5
    max_speed = 1.5

    obstacles = []
    for _ in range(num_obstacles):
        obstacles.append(Obstacle(
            np.random.uniform(0, width), # x
            np.random.uniform(0, height), # y
            np.random.uniform(min_obstacle_radius, max_obstacle_radius), # radius
            np.random.uniform(min_speed, max_speed), # velocity
            np.random.uniform(0, 2*np.pi), # theta
            0)) # omega

    return Environment(width, height, obstacles, Robot(0, 0, 0, 0, 0))