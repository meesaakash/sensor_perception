from typing import Tuple, List, Callable
from Obstacle import Obstacle
# from Simulation import Simulation
# A tuple of (speed change, steering angle change)
RobotMovement = Tuple[float, float]

Destination = Tuple[10.0, 0.0]

# A tuple of (x position, y position, theta, radius(maybe))
RobotState = Tuple[float, float, float]

# A tuple of (LiDAR hit points, LiDAR estimated velocities [line of sight], robot position)
ExperimentState = Tuple[List[float], List[float], RobotState]

# A tuple of (current and previous experiment states, destination, obstacles FOR DEBUGGING)
# AlgorithmInput = Tuple[List[ExperimentState], Tuple[float, float], List[Obstacle]]
AlgorithmInput = Tuple[List[ExperimentState], Tuple[float,float], List[Obstacle]]

# A function that takes an AlgorithmInput and returns a RobotMovement
Algorithm = Callable[[AlgorithmInput], RobotMovement]
