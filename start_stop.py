from CustomTypes import RobotMovement, AlgorithmInput
import numpy as np
import math

def start_stop(input: AlgorithmInput) -> RobotMovement:
    """
    An algorithm that starts and stops depending on if it is on a path to hit an obstacle.
    If an obstacle is detected very close via LiDAR, the robot stops until the path is clear.
    Otherwise, it proceeds straight ahead at a constant velocity of 0.1.
    
    This version also computes:
    - The total distance traveled by the robot from the start (0,0) until it reaches (10,0).
    - The total time taken to reach (10,0).
    """

    (experiment_states, destination, obstacles) = input
    dest_x, dest_y = 10.0, 0.0

    # Take the latest experiment state
    # ExperimentState = (LiDAR hit_points, LiDAR velocities/doppler, (robot_x, robot_y, robot_theta))
    latest_state = experiment_states[-1]
    hit_points, estimated_velocities, (robot_x, robot_y, robot_theta) = latest_state

    # Arbitrary radius for robot 
    ROBOT_RADIUS = 1.0
    ROBOT_SPEED = 0.1

    # Check each LiDAR hit point for potential collision
    collision_detected = False
    for (hx, hy) in hit_points:
        distance = math.sqrt((robot_x - hx)**2 + (robot_y - hy)**2)
        # If a hit point is extremely close to the robot, consider it a collision
        if distance < 0.1:
            collision_detected = True
            break

    # If a collision is detected, robot stops
    if collision_detected:
        vx, vy = (-0.1, 0.0)
    else:
        # Otherwise, move forward at constant speed 0.1 along x-axis
        vx, vy = (0.1, 0.0)

    # Calculate total distance traveled so far by summing distances between consecutive states
    # and also estimate the time.
    # Assuming a known dt for each simulation step. If not known, we can guess dt=0.1s.
    dt = 0.1  # Adjust if your simulation uses a different timestep

    total_distance = 0.0
    for i in range(1, len(experiment_states)):
        # Extract robot positions from consecutive states
        _, _, (prev_x, prev_y, _) = experiment_states[i-1]
        _, _, (curr_x, curr_y, _) = experiment_states[i]
        dx = curr_x - prev_x
        dy = curr_y - prev_y
        segment_dist = math.sqrt(dx*dx + dy*dy)
        total_distance += segment_dist

    total_time = len(experiment_states) * dt

    # Check if robot has reached destination (10,0)
    # Consider "reached" if within a small threshold
    reach_threshold = 0
    dist_to_dest = math.sqrt((robot_x - dest_x)**2 + (robot_y - dest_y)**2)
    if dist_to_dest < reach_threshold:
        print("Destination reached!")
        print(f"Total distance traveled: {total_distance:.2f} m")
        print(f"Total time taken: {total_time:.2f} s")

    return (vx, vy)
