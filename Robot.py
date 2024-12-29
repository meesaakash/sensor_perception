import numpy as np

class Robot:
    def __init__(self, x, y, theta, velocity, omega):
        self.x = x
        self.y = y
        self.theta = theta
        self.velocity = velocity
        self.omega = omega
        self.radius = 1

    def next_state(self, dt):
        self.x += self.velocity * dt * np.cos(self.theta)
        self.y += self.velocity * dt * np.sin(self.theta)
        self.theta += self.omega * dt
