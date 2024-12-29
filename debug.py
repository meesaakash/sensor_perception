import matplotlib.pyplot as plt
import numpy as np
import math
import os

def debug_visualization(robot_x, robot_y, dest_x, dest_y, occupancy, path, obstacles_info, 
                        x_min, x_max, y_min, y_max, resolution, frame_number, robot_speed=0.1):
    """
    Visualize the occupancy grid, the robot, the destination, obstacles, and the planned path.
    Also calculates and displays the total path distance and the expected travel time at the given robot speed.
    Saves the figure as an image file in the 'viz' folder.

    Args:
        robot_x, robot_y: Robot's current position
        dest_x, dest_y: Goal position
        occupancy: 2D array of booleans (True = obstacle)
        path: list of (x, y) tuples representing the computed path
        obstacles_info: list of (ox, oy, angle, speed) for obstacles
        x_min, x_max, y_min, y_max: Plot boundaries
        resolution: Grid resolution
        frame_number: Integer frame number for saving the file
        robot_speed: Speed of the robot, used to compute travel time
    """

    # Create viz directory if it doesn't exist
    if not os.path.exists('viz'):
        os.makedirs('viz')

    plt.figure(figsize=(10,6))

    # Convert occupancy to an integer array for visualization
    occ_map = np.array(occupancy, dtype=int).T

    # Show occupancy grid
    plt.imshow(occ_map, origin='lower', extent=[x_min, x_max, y_min, y_max], cmap='gray_r', alpha=0.8)

    # Plot robot and destination
    plt.plot(robot_x, robot_y, 'ro', label='Robot Start', markersize=8)
    plt.plot(dest_x, dest_y, 'g*', label='Destination', markersize=12)

    # Plot path if available
    total_distance = 0.0
    if path and len(path) > 1:
        px, py = zip(*path)
        plt.plot(px, py, 'b-', label='Planned Path', linewidth=2)

        # Calculate the total distance of the path
        for i in range(len(path)-1):
            dx = path[i+1][0] - path[i][0]
            dy = path[i+1][1] - path[i][1]
            segment_dist = math.sqrt(dx**2 + dy**2)
            total_distance += segment_dist

    # Plot obstacles with direction arrows
    for (ox, oy, angle, speed) in obstacles_info:
        plt.plot(ox, oy, 'ko', markersize=5)
        arrow_length = speed * 0.5  # scale for display
        dx = arrow_length * np.cos(angle)
        dy = arrow_length * np.sin(angle)
        plt.arrow(ox, oy, dx, dy, head_width=0.1, head_length=0.1, fc='r', ec='r')

    plt.xlim([x_min, x_max])
    plt.ylim([y_min, y_max])
    plt.grid(True)
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title(f'Path Planning Debug Visualization (Frame {frame_number})')

    # Compute the travel time if we have a non-zero path distance
    if total_distance > 0 and robot_speed > 0:
        travel_time = total_distance / robot_speed
        info_text = f"Path Distance: {total_distance:.2f} m\nTravel Time: {travel_time:.2f} s"
        plt.text(x_min + 1, y_max - 1, info_text, fontsize=10, 
                 bbox=dict(facecolor='white', alpha=0.8, edgecolor='black'))
    else:
        plt.text(x_min + 1, y_max - 1, "No valid path", fontsize=10, 
                 bbox=dict(facecolor='white', alpha=0.8, edgecolor='black'))

    plt.legend()

    # Save the figure instead of showing it
    filename = f"viz/frame_{frame_number:04d}.png"
    plt.savefig(filename, dpi=150)
    plt.close()
