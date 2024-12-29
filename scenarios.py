from Robot import Robot
from Obstacle import Obstacle
from Environment import Environment
from Simulation import Simulation
import numpy as np

def scenario_one():
    # Robot moving toward the obstacle
    robot = Robot(x=0, y=0, theta=0, velocity=1, omega=0)  # Moving along the positive X-axis

    # Static obstacle placed ahead of the robot
    obstacle = Obstacle(x=10, y=0, radius=3, velocity=0, theta=0, omega=0)

    # Set up the environment with the robot and the static obstacle
    environment = Environment(width=40, height=40, obstacles=[obstacle], robot=robot)

    # Initialize the simulation with desired LiDAR settings
    simulation = Simulation(
        environment=environment,
        lidar_resolution=5,  # Higher resolution for better visualization
        lidar_range=15,
        noise_type='zero',  # Assuming no noise for initial testing
        noise_std_dev=0,
        fov=(-90, 90)  # Forward-facing LiDAR
    )

    # Run the simulation
    states = simulation.loop(duration=5)
    return states

def scenario_two(noise_type='zero', noise_std_dev=1e2):
    # Stationary robot
    robot = Robot(x=0, y=0, theta=0, velocity=0, omega=0)

    # Obstacle starting right in front of the robot, with initial v=0, accelerating forward
    obstacle = Obstacle(
        x=1, y=0, radius=1, velocity=0, theta=0, omega=0, acceleration=1  # Moving along positive X-axis
    )

    # Set up the environment
    environment = Environment(width=40, height=40, obstacles=[obstacle], robot=robot)

    # Initialize the simulation
    simulation = Simulation(
        environment=environment,
        lidar_resolution=1,
        lidar_range=200,  # Increase the LiDAR range
        noise_type=noise_type,
        noise_std_dev=noise_std_dev,
        fov=(-10, 10)  # Narrow field of view directly ahead
    )

    # Initialize lists to store data
    relative_velocities = []
    obstacle_velocities = []
    l1_errors = []
    times = []

    # Run the simulation and collect data
    t = 0
    dt = 0.1
    duration = 15
    while t < duration:
        simulation.run()
        state = simulation.states[-1]

        # Extract the estimated velocity at angle 0 (directly in front)
        lidar = state['lidar']
        angles = np.rad2deg(lidar['angles'])
        index_front = np.argmin(np.abs(angles))  # Closest to 0 degrees
        v_rel_los_est = lidar['estimated_velocities'][index_front]
        relative_velocities.append(v_rel_los_est)

        # Record the obstacle's velocity
        obstacle_vel = obstacle.velocity
        obstacle_velocities.append(obstacle_vel)

        # Compute L1 error
        l1_error = np.abs(v_rel_los_est - obstacle_vel)
        l1_errors.append(l1_error)

        times.append(t)
        t += dt

    return simulation.states, times, relative_velocities, obstacle_velocities, l1_errors

def scenario_three():
    # Random movement of the robot
    robot = Robot(x=0, y=0, theta=0, velocity=0.5, omega=0)

    # Create moving obstacles with random initial positions and velocities
    obstacles = []
    num_obstacles = 5
    for _ in range(num_obstacles):
        obs = Obstacle(
            x=np.random.uniform(-25, 25),
            y=np.random.uniform(-12.5, 12.5),
            radius=np.random.uniform(0.5, 1.5),
            velocity=np.random.uniform(-0.5, 0.5),
            theta=np.random.uniform(0, 2 * np.pi),
            omega=np.random.uniform(-0.2, 0.2)
        )
        obstacles.append(obs)

    # Set up the environment
    environment = Environment(width=40, height=40, obstacles=obstacles, robot=robot)

    # Initialize the simulation
    simulation = Simulation(
        environment=environment,
        lidar_resolution=2,
        lidar_range=30,
        noise_type='constant',
        noise_std_dev=1e2,
        fov=(0, 360)  # 360-degree LiDAR
    )

    # Run the simulation
    states = simulation.loop(duration=30)
    return states
