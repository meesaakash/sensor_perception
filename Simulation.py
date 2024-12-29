import numpy as np
from typing import Tuple, Dict, List, Any
from Environment import Environment
from typing import Optional
from CustomTypes import ExperimentState
import math

class Simulation:
    """
    A class to simulate the lidar measurements of a robot in an environment with obstacles.

    Attributes:
        environment (Environment): The environment in which the simulation takes place.
        dt (float): The time step for the simulation in seconds.
        states (list): A list of dictionaries containing the state of the robot and obstacles at each time step.
        lidar_resolution (float): The resolution of the lidar in degrees.
        lidar_range (float): The range of the lidar in meters.
        lidar_frequency (float): The frequency of the lidar in Hz.
        noise_type (str): The type of noise to add to the lidar measurements. "zero", "constant", "proportional"
        noise_std_dev (float): The standard deviation of the noise to add to the lidar measurements.
        fov (tuple): The field of view of the lidar in degrees.
    """
    def __init__(self, environment: Environment, dt: float = 0.1, lidar_resolution: float=5, lidar_range: float=15, lidar_frequency: float=2e14,
                 noise_type: str='constant', noise_std_dev: float=1e2, fov: Tuple[float, float]=(0, 360)):
        self.environment = environment
        self.states = []
        self.lidar_resolution = lidar_resolution  # Resolution of the LiDAR in degrees
        self.lidar_range = lidar_range            # Range of the LiDAR in meters
        self.lidar_frequency = lidar_frequency    # Emitted frequency of the LiDAR
        self.noise_type = noise_type              # 'zero', 'constant', 'proportional'
        self.noise_std_dev = noise_std_dev        # Standard deviation or scaling factor
        self.fov = fov                            # Field of view in degrees
        self.dt = dt                              # Time step for the simulation

    def save_state(self) -> None:
        """
        Save the state of the robot and obstacles at the current time step.
        """
        state = {
            'robot': (self.environment.robot.x, self.environment.robot.y, self.environment.robot.theta),
            'obstacles': [
                (obs.x, obs.y, obs.radius, obs.velocity, obs.theta) for obs in self.environment.obstacles
            ],
            'lidar': self.perform_lidar()
        }
        self.states.append(state)

    def perform_lidar(self) -> Dict[str, Any]:
        """
        Perform the lidar measurement, including the frequency shift and estimated velocity.

        Returns:
            A dictionary containing:
                'angles': The angles of the rays.
                'hit_points': The x, y coordinates of the hit points for each ray.
                'frequency_shifts': The frequency shifts computed for each ray.
                'estimated_velocities': The line-of-sight estimated velocities for each ray.
                'obstacle_clusters': A dictionary keyed by obstacle index, each containing a tuple of
                                     (x_hit, y_hit, estimated_velocity) for each ray, and a final tuple
                                     (angle, speed) appended after motion estimation.
        """
        c = 3e8  # Speed of light in m/s
        robot = self.environment.robot
        robot_x, robot_y, robot_theta = robot.x, robot.y, robot.theta
        angles = np.deg2rad(np.arange(self.fov[0], self.fov[1], self.lidar_resolution))
        rays_dx = np.cos(angles + robot_theta)
        rays_dy = np.sin(angles + robot_theta)
        distances = np.full_like(angles, self.lidar_range)
        obstacle_indices = np.full_like(angles, -1, dtype=int)

        # Loop through each obstacle to find lidar hits
        for idx, obstacle in enumerate(self.environment.obstacles):
            dx = obstacle.x - robot_x
            dy = obstacle.y - robot_y
            a = rays_dx**2 + rays_dy**2
            b = 2 * (rays_dx * (-dx) + rays_dy * (-dy))
            c_quad = dx**2 + dy**2 - obstacle.radius**2
            discriminant = b**2 - 4 * a * c_quad
            valid = discriminant >= 0

            if np.any(valid):
                sqrt_disc = np.sqrt(discriminant[valid])
                a_valid = a[valid]
                b_valid = b[valid]
                t1 = (-b_valid - sqrt_disc) / (2 * a_valid)
                t2 = (-b_valid + sqrt_disc) / (2 * a_valid)

                t_positive = np.minimum(
                    np.where(t1 > 0, t1, np.inf),
                    np.where(t2 > 0, t2, np.inf)
                )

                # Update distances and obstacle indices
                update_mask = t_positive < distances[valid]
                indices_valid = np.where(valid)[0]
                indices_to_update = indices_valid[update_mask]

                distances[indices_to_update] = t_positive[update_mask]
                obstacle_indices[indices_to_update] = idx

        hit_x = robot_x + distances * rays_dx
        hit_y = robot_y + distances * rays_dy

        # Initialize frequency shifts and estimated velocities arrays
        frequency_shifts = np.zeros_like(angles)
        estimated_velocities = np.zeros_like(angles)

        # Compute robot's velocity components
        v_robot_x = robot.velocity * np.cos(robot.theta)
        v_robot_y = robot.velocity * np.sin(robot.theta)

        # Compute frequency shifts and estimated velocities for rays that hit obstacles
        for idx, obstacle in enumerate(self.environment.obstacles):
            ray_indices = np.where(obstacle_indices == idx)[0]
            if len(ray_indices) > 0:
                # Points where rays hit the obstacle
                point_x = hit_x[ray_indices]
                point_y = hit_y[ray_indices]

                # Obstacle's translational velocity components
                v_obs_trans_x = obstacle.velocity * np.cos(obstacle.theta)
                v_obs_trans_y = obstacle.velocity * np.sin(obstacle.theta)

                # Vector from obstacle center to hit points
                dx = point_x - obstacle.x
                dy = point_y - obstacle.y

                # Obstacle's rotational velocity at hit points
                v_rot_x = -obstacle.omega * dy
                v_rot_y = obstacle.omega * dx

                # Total obstacle velocity at hit points
                v_obs_x = v_obs_trans_x + v_rot_x
                v_obs_y = v_obs_trans_y + v_rot_y

                # Relative velocity components
                v_rel_x = v_obs_x - v_robot_x
                v_rel_y = v_obs_y - v_robot_y

                # Line-of-sight unit vectors
                los_dx = point_x - robot_x
                los_dy = point_y - robot_y
                distances_to_points = np.sqrt(los_dx**2 + los_dy**2)
                los_unit_x = los_dx / distances_to_points
                los_unit_y = los_dy / distances_to_points

                # Relative velocity along the line of sight
                v_rel_los = v_rel_x * los_unit_x + v_rel_y * los_unit_y

                # Simulate frequency shift using Doppler effect
                delta_f = (2 * v_rel_los * self.lidar_frequency) / c  # Factor of 2 for round trip

                # Handle noise based on the noise setting
                if self.noise_type == 'zero':
                    noise_std_dev = 0
                elif self.noise_type == 'constant':
                    noise_std_dev = self.noise_std_dev
                elif self.noise_type == 'proportional':
                    noise_std_dev = np.abs(v_rel_los) * self.noise_std_dev
                else:
                    noise_std_dev = 0  # Default to zero noise if unknown type

                # Add Gaussian noise to the frequency shift
                noise = np.random.normal(0, noise_std_dev, size=delta_f.shape)
                delta_f_noisy = delta_f + noise

                # Estimate the relative velocity from noisy frequency shift
                v_rel_los_est = (delta_f_noisy * c) / (2 * self.lidar_frequency)

                # Update frequency_shifts and estimated_velocities arrays
                frequency_shifts[ray_indices] = delta_f_noisy
                estimated_velocities[ray_indices] = v_rel_los_est

        # Build obstacle_clusters structure
        obstacle_clusters = {}
        for idx, obstacle in enumerate(self.environment.obstacles):
            ray_indices = np.where(obstacle_indices == idx)[0]
            # Each entry: (hit_x, hit_y, estimated_velocity)
            obstacle_data = [
                (hit_x[i], hit_y[i], estimated_velocities[i]) for i in ray_indices
            ]
            obstacle_clusters[idx] = tuple(obstacle_data)

        # Estimate obstacle motion and update obstacle_clusters with (angle, speed)
        obstacle_clusters = self.estimate_obstacle_motion(obstacle_clusters)

        return {
            'angles': angles,
            'hit_points': np.vstack((hit_x, hit_y)).T,
            'frequency_shifts': frequency_shifts,
            'estimated_velocities': estimated_velocities,
            'obstacle_clusters': obstacle_clusters
        }

    def estimate_obstacle_motion(self, obstacle_clusters: Dict[int, Tuple[Tuple[float, float, float], ...]]) -> Dict[int, Tuple[Tuple[float, float, float], ...]]:
        """
        Given the obstacle_clusters containing (hit_x, hit_y, v_rel_los_est) for each hit,
        estimate the obstacle's direction of motion and speed.

        Args:
            obstacle_clusters: Dictionary keyed by obstacle index, each value is a tuple of (x_hit, y_hit, v_rel_los_est).

        Returns:
            A dictionary with the same keys as obstacle_clusters, but each value will have two additional 
            fields appended at the end of the tuple: (angle, speed).
        """

        # Get robot state and compute robot velocity components
        robot_x, robot_y, robot_theta = self.environment.robot.x, self.environment.robot.y, self.environment.robot.theta
        v_robot_x = self.environment.robot.velocity * np.cos(robot_theta)
        v_robot_y = self.environment.robot.velocity * np.sin(robot_theta)

        new_clusters = {}
        for obs_idx, hits in obstacle_clusters.items():
            if len(hits) == 0:
                # No hits, no estimation possible
                # Just append (angle=0, speed=0)
                augmented_hits = hits + ((0.0, 0.0),)
                new_clusters[obs_idx] = augmented_hits
                continue

            A = []
            b = []
            for (x_hit, y_hit, v_rel_los_est) in hits:
                dx = x_hit - robot_x
                dy = y_hit - robot_y
                dist = np.sqrt(dx**2 + dy**2)
                if dist == 0:
                    # In the extremely unlikely scenario that the hit point is the robot's location
                    # Skip this measurement
                    continue

                # Line-of-sight unit vector
                los_x = dx / dist
                los_y = dy / dist

                # Equation:
                # v_rel_los_est = (v_obs_x - v_robot_x)*los_x + (v_obs_y - v_robot_y)*los_y
                # v_rel_los_est + (v_robot_x*los_x + v_robot_y*los_y) = v_obs_x*los_x + v_obs_y*los_y
                lhs = v_rel_los_est + (v_robot_x * los_x + v_robot_y * los_y)
                A.append([los_x, los_y])
                b.append(lhs)

            A = np.array(A)
            b = np.array(b)

            if len(A) < 2:
                # Not enough equations to solve properly. Assume no reliable solution.
                v_obs_x, v_obs_y = 0.0, 0.0
            else:
                # Solve least squares to find v_obs_x and v_obs_y
                v_obs, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
                v_obs_x, v_obs_y = v_obs

            # Compute angle and speed of the obstacle
            angle = math.atan2(v_obs_y, v_obs_x)
            speed = math.sqrt(v_obs_x**2 + v_obs_y**2)

            # Append (angle, speed) to the tuple for this obstacle
            augmented_hits = hits + ((angle, speed),)
            new_clusters[obs_idx] = augmented_hits

        return new_clusters

    def run(self) -> None:
        """
        Run the simulation for a single time step.
        """
        self.save_state()
        self.environment.next_state(self.dt)

    def collision(self) -> Optional[int]:
        """
        Check if the robot has collided with an obstacle.
        Returns the index of the obstacle if there is a collision, otherwise None.
        """
        for i, obstacle in enumerate(self.environment.obstacles):
            distance_apart = np.sqrt((self.environment.robot.x - obstacle.x)**2 + (self.environment.robot.y - obstacle.y)**2)
            if distance_apart < self.environment.robot.radius + obstacle.radius:
                return i
        return None

    def get_experiment_state(self) -> ExperimentState:
        """
        Get the state of the robot and obstacles for the experiment.

        Returns:
            A tuple containing:
                obstacle_clusters: Dictionary keyed by obstacle index, each containing a tuple of (x_hit, y_hit, estimated_velocity),
                                   followed by (angle, speed).
                robot_state: A tuple containing (robot_x, robot_y, robot_theta).
        """
        lidar = self.perform_lidar()
        return (lidar['hit_points'],lidar['obstacle_clusters'], (self.environment.robot.x, self.environment.robot.y, self.environment.robot.theta))

    def update_robot_movement(self, robot_movement: Tuple[float, float]):
        """
        Update the robot's state by a given speed change and theta change.
        """
        speed_change, theta_change = robot_movement

        self.environment.robot.theta += theta_change

        x_change = speed_change * np.cos(self.environment.robot.theta)
        y_change = speed_change * np.sin(self.environment.robot.theta)

        self.environment.robot.x += x_change
        self.environment.robot.y += y_change

    def loop(self, duration: float = 15) -> List[Dict[str, Any]]:
        """
        Run the simulation for a given duration.
        """
        t = 0
        while t < duration:
            self.run()
            t += self.dt
        return self.states
