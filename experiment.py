from CustomTypes import Algorithm, Destination
from Simulation import Simulation
from typing import Tuple
import numpy as np
from Environment import random_environment
from plotting import animate_simulation

PENALTY_FOR_COLLISION = 10
DESTINATION_DISTANCE_THRESHOLD = 1
MAX_DURATION = 15

def robot_at_destination(robot_position: Tuple[float, float], destination: Tuple[float, float]) -> bool:
    # print("Robot reached destination")
    return np.sqrt((robot_position[0] - destination[0])**2 + (robot_position[1] - destination[1])**2) < DESTINATION_DISTANCE_THRESHOLD

def save_destination(destination) -> Destination:
    print("destination is", destination)
    return destination

def experiment(algorithm: Algorithm, simulation: Simulation, destination: Tuple[float, float]) -> Tuple[float, Simulation]:
    """
    Run an experiment with a given algorithm and simulation.
    """
    print("Running experiment...")
    score = 0
    experiment_states = []
    current_experiment_state = simulation.get_experiment_state()
    collision_index = None
    t = 0
    arrival_time = None
    
    save_destination(destination)
    # Run until the robot is at the destination
    while (not robot_at_destination(current_experiment_state[2], destination)) and t < 10:
        t += simulation.dt
        experiment_states.append(current_experiment_state)

        # Check for new collision, if no collision, reset collision index
        if temp_collision_index := simulation.collision():
            if collision_index != temp_collision_index:
                collision_index = temp_collision_index
        else:
            collision_index = None

        # Run algorithm, feeding it the experiment states and current obstacles (obstacles for debugging)
        current_obstacles = simulation.environment.obstacles
        robot_movement = algorithm((experiment_states, destination, current_obstacles))
        
        # Update robot state
        simulation.update_robot_movement(robot_movement)

        # Run simulation
        simulation.run()

        # Get current experiment state
        current_experiment_state = simulation.get_experiment_state()

    # If the loop ended because the robot reached its destination
    if robot_at_destination(current_experiment_state[2], destination):
        arrival_time = t
        print(f"Robot reached destination at time {arrival_time:.2f}s.")

        # Example scoring scheme:
        # The faster the arrival, the higher the score.
        # If MAX_DURATION=10 and arrives at t, score = 10 - t.
        # If it arrives at t=0, score=10; if at t=10, score=0.
        score = arrival_time
    else:
        # Robot did not reach the destination in the allowed time
        score= 0
        print("Robot did not reach the destination in time.")
        # score remains as is (0 or penalized by collisions)

    # Return the final score, the simulation, and the arrival_time
    return score, simulation, arrival_time

def run_random_experiment(algorithm: Algorithm, num_obstacles: int = 10, plotting: bool = False) -> float:
    """
    Run a random experiment with a given algorithm.
    """
    world_width = 10
    world_height = 10
    min_obstacle_radius = 0.5
    max_obstacle_radius = 1.5

    environment = random_environment(world_width, world_height, num_obstacles, min_obstacle_radius, max_obstacle_radius)
    simulation = Simulation(environment)
    # get rid of manual destination input below
    score, simulation_after, arrival_time = experiment(algorithm, simulation, (10.0, 0.0))
    print("Final score is", score)
    

    if plotting:
        print("Plotting...")
        animate_simulation(simulation_after.states, world_width, world_height)

    return score
