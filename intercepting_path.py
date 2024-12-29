from CustomTypes import RobotMovement, AlgorithmInput
import heapq
import math
from typing import Tuple, List, Any
from debug import debug_visualization

def intercepting_path(input: Any) -> Tuple[float, float]:
    """
    An algorithm that does complex intercepting path planning using A*.
    It uses the LiDAR (doppler) obstacle clusters, each with a final (angle, speed) tuple,
    to build a cost/occupancy map and find a shortest path to the destination (10,0).

    Args:
        input: (experiment_states, destination, obstacles)
               - experiment_states: List of ExperimentState = (hit_points, doppler, (robot_x, robot_y, robot_theta))
                 where doppler = obstacle_clusters dictionary
               - destination: (x, y)
               - obstacles: (Not directly used here, we rely on doppler)

    Returns:
        (vx, vy): The robot's commanded velocity to move toward the found path at speed 0.1.
    """

    (experiment_states, destination, obstacles) = input
    # Latest experiment state
    # According to the user's scenario:
    # latest_state = (hit_points, doppler, (robot_x, robot_y, robot_theta))
    latest_state = experiment_states[-1]
    hit_points, doppler, (robot_x, robot_y, robot_theta) = latest_state
    frame_number = len(experiment_states)
    # Constants and parameters
    ROBOT_RADIUS = 1.0
    OBSTACLE_RADIUS = 1.5
    ROBOT_SPEED = 0.1
    dest_x, dest_y = 10.0, 0.0

    # Build a simple grid for demonstration
    # Assume a region from [-5, 15] in x and [-10, 10] in y for planning
    # Grid resolution: 0.1 m per cell
    x_min, x_max = -5, 15
    y_min, y_max = -10, 10
    resolution = 0.1
    width = int((x_max - x_min) / resolution)
    height = int((y_max - y_min) / resolution)

    obstacles_info = []
    for obs_idx, obs_data in doppler.items():
        if len(obs_data) == 0:
            continue
        angle, speed = obs_data[-1]
        # Compute obstacle center
        hits = obs_data[:-1]
        if len(hits) > 0:
            avg_x = sum(h[0] for h in hits) / len(hits)
            avg_y = sum(h[1] for h in hits) / len(hits)
            obstacles_info.append((avg_x, avg_y, angle, speed))

    def world_to_grid(x, y):
        gx = int((x - x_min) / resolution)
        gy = int((y - y_min) / resolution)
        return gx, gy

    def grid_to_world(gx, gy):
        wx = gx * resolution + x_min
        wy = gy * resolution + y_min
        return wx, wy

    # Check bounds
    def in_bounds(gx, gy):
        return 0 <= gx < width and 0 <= gy < height

    # Initialize occupancy grid: False = free, True = obstacle
    occupancy = [[False for _ in range(height)] for _ in range(width)]

    # Extract obstacle info from doppler clusters
    # doppler is a dict: obs_idx -> ((x_hit, y_hit, v_est), ..., (angle, speed))
    # The last tuple (angle, speed) gives obstacle motion.
    for obs_idx, obs_data in doppler.items():
        if len(obs_data) == 0:
            continue

        # The last tuple is (angle, speed)
        angle, speed = obs_data[-1]

        # Compute obstacle center from hits (average of hit points)
        hits = obs_data[:-1]  # all but last are hits
        if len(hits) > 0:
            avg_x = sum(h[0] for h in hits) / len(hits)
            avg_y = sum(h[1] for h in hits) / len(hits)
        else:
            # If no hits, skip
            continue

        # Predict obstacle position - for simplicity, we consider the obstacle at its current center
        # and also a few steps into the future.
        # Let's say we predict for a few time steps assuming 0.1 second steps (same dt).
        future_positions = []
        for t_step in range(0, 5):
            # Predict position after t_step * dt seconds
            # Assume dt from simulation is known (0.1)
            dt = 0.1
            pred_x = avg_x + speed * math.cos(angle) * t_step * dt
            pred_y = avg_y + speed * math.sin(angle) * t_step * dt
            future_positions.append((pred_x, pred_y))

        # Mark these future positions as blocked in the occupancy grid
        # Considering the obstacle radius
        radius_cells = int(OBSTACLE_RADIUS / resolution)
        for (ox, oy) in future_positions:
            ogx, ogy = world_to_grid(ox, oy)
            for dx in range(-radius_cells, radius_cells+1):
                for dy in range(-radius_cells, radius_cells+1):
                    nx = ogx + dx
                    ny = ogy + dy
                    if in_bounds(nx, ny):
                        cx, cy = grid_to_world(nx, ny)
                        # Check actual distance from center
                        dist = math.sqrt((cx - ox)**2 + (cy - oy)**2)
                        if dist <= OBSTACLE_RADIUS:
                            occupancy[nx][ny] = True

    # Now we have an occupancy grid marking moving obstacles regions

    # A* Search from (robot_x, robot_y) to (10,0)
    start_gx, start_gy = world_to_grid(robot_x, robot_y)
    goal_gx, goal_gy = world_to_grid(dest_x, dest_y)

    # If start or goal are out of bounds or blocked, just return no movement
    if not in_bounds(start_gx, start_gy) or not in_bounds(goal_gx, goal_gy):
        return (0.0, 0.0)

    if occupancy[start_gx][start_gy]:
        return (0.0, 0.0)
    if occupancy[goal_gx][goal_gy]:
        # If goal is blocked, still try to get as close as possible or return no movement.
        return (0.0, 0.0)

    # A* implementation
    def heuristic(a, b):
        # Euclidean distance heuristic
        return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

    neighbors = [(1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]

    open_set = []
    heapq.heappush(open_set, (0, (start_gx, start_gy)))
    came_from = {}
    g_score = {(start_gx, start_gy): 0}
    f_score = {(start_gx, start_gy): heuristic((start_gx, start_gy), (goal_gx, goal_gy))}

    while open_set:
        _, current = heapq.heappop(open_set)

        if current == (goal_gx, goal_gy):
            # Reconstruct path
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append((start_gx, start_gy))
            path.reverse()

            # Convert path to world coordinates
            wpath = [grid_to_world(p[0], p[1]) for p in path]

            # Decide next move:
            # Take the first waypoint after the start (or the goal if no intermediate)
            if len(wpath) > 1:
                next_x, next_y = wpath[1]
                # print('next are', next_x, next_y)
            else:
                # Already at goal
                next_x, next_y = wpath[0]

            # Compute direction to next waypoint
            dx = next_x - robot_x
            dy = next_y - robot_y
            dist = math.sqrt(dx**2 + dy**2)
            if dist > 1e-6:
                vx = next_x
                vy = next_y
            else:
                vx, vy = 0.0, 0.0
                
            debug_visualization(robot_x, robot_y, dest_x, dest_y, occupancy, wpath, obstacles_info, x_min, x_max, y_min, y_max, resolution, frame_number, robot_speed=0.1)
            print('velo is at frame ', frame_number, vx, vy)
            return (vx, vy)

        for nx, ny in neighbors:
            neighbor = (current[0] + nx, current[1] + ny)
            if not in_bounds(neighbor[0], neighbor[1]):
                continue
            if occupancy[neighbor[0]][neighbor[1]]:
                # blocked cell
                continue

            # Tentative g_score
            tentative_g = g_score[current] + math.sqrt(nx*nx + ny*ny)*resolution
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score[neighbor] = tentative_g + heuristic(neighbor, (goal_gx, goal_gy))
                heapq.heappush(open_set, (f_score[neighbor], neighbor))

    # If no path found
    print('returning none for frame', frame_number)
    return (0.0, 0.0)

