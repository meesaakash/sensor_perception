import numpy as np

class Obstacle:
    """
    A class to represent an obstacle in the environment.

    Attributes:
        x (float): The x-coordinate of the obstacle.
        y (float): The y-coordinate of the obstacle.
        radius (float): The radius of the obstacle.
        velocity (float): The velocity of the obstacle.
        theta (float): The angle of the obstacle.
        omega (float): The angular velocity of the obstacle.
        acceleration (float): The acceleration of the obstacle.
    """
    def __init__(self, x, y, radius, velocity, theta, omega, acceleration=0):
        self.x = x
        self.y = y
        self.radius = radius
        self.velocity = velocity
        self.theta = theta
        self.omega = omega
        self.acceleration = acceleration  # New attribute

    def next_state(self, dt):
        # Update velocity with acceleration
        self.velocity += self.acceleration * dt
        self.x += self.velocity * dt * np.cos(self.theta)
        self.y += self.velocity * dt * np.sin(self.theta)
        self.theta += self.omega * dt
