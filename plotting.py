import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
from matplotlib.lines import Line2D
from matplotlib import animation
from typing import List, Dict, Tuple

def animate_simulation(states: List[Dict], environment_width: float, environment_height: float, save: bool = True, filename: str='start_stop3_simulation.gif'):
    """
    Animate the simulation of a robot and obstacles with LiDAR.
    Generates a GIF animation if save=True.
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(-environment_width / 2, environment_width*1.5)
    ax.set_ylim(-environment_height / 2, environment_height*1.5)
    ax.set_aspect('equal')  # Ensure equal scaling on both axes
    ax.set_xlabel('X position')
    ax.set_ylabel('Y position')
    ax.set_title('Robot, Obstacles, and LiDAR Simulation')
    plt.grid(True)

    # Initialize plot elements
    robot_dot, = ax.plot([], [], 'bo', label='Robot')
    obstacle_patches = []
    velocity_arrows = []
    n_obstacles = len(states[0]['obstacles']) if states else 0

    for _ in range(n_obstacles):
        # Create obstacle patches
        patch = Circle((0, 0), 0.5, color='red', alpha=0.5)
        ax.add_patch(patch)
        obstacle_patches.append(patch)

        # Initialize empty line for obstacle velocity
        arrow_line, = ax.plot([], [], color='green')
        velocity_arrows.append(arrow_line)

    lidar_points, = ax.plot([], [], 'k.', markersize=2)
    scan_line, = ax.plot([], [], 'y-', linewidth=0.5, alpha=0.7)

    doppler_arrows = []  # Will be a list of arrow patches
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

    # Create proxy artists for legend
    obstacle_arrow_proxy = Line2D([0], [0], color='green', lw=2, label='Obstacle Velocity')
    doppler_arrow_proxy = FancyArrowPatch((0,0),(1,0), color='purple', label='Estimated Velocity', arrowstyle='->')
    lidar_points_proxy = Line2D([0], [0], linestyle='', marker='.', color='k', label='LiDAR Hits')
    robot_dot_proxy = Line2D([0], [0], linestyle='', marker='o', color='b', label='Robot')
    obstacle_patch_proxy = Circle((0,0), radius=0.5, color='red', alpha=0.5, label='Obstacle')

    # Set up legend with proxy artists
    handles = [robot_dot_proxy, obstacle_patch_proxy, obstacle_arrow_proxy, lidar_points_proxy, doppler_arrow_proxy]
    labels = [h.get_label() for h in handles]
    ax.legend(handles, labels, loc='upper right')

    def init():
        robot_dot.set_data([], [])
        for patch in obstacle_patches:
            patch.center = (-10, -10)
        for arrow in velocity_arrows:
            arrow.set_data([], [])
        lidar_points.set_data([], [])
        scan_line.set_data([], [])
        # Remove any existing Doppler arrows
        for arrow in doppler_arrows:
            arrow.remove()
        doppler_arrows.clear()
        time_text.set_text('')
        return [robot_dot] + obstacle_patches + velocity_arrows + [lidar_points, scan_line] + doppler_arrows + [time_text]

    def update(frame):
        state = states[frame]
        robot_x, robot_y, robot_theta = state['robot']

        # Update robot position
        robot_dot.set_data([robot_x], [robot_y])

        # Update time annotation (assuming dt=0.1)
        current_time = frame * 0.1  
        time_text.set_text(f'Time: {current_time:.1f}s')

        # Update obstacles and their velocity arrows
        for i, obstacle_data in enumerate(state['obstacles']):
            obstacle_x, obstacle_y, obstacle_radius, obstacle_velocity, obstacle_theta = obstacle_data
            obstacle_patches[i].center = (obstacle_x, obstacle_y)
            obstacle_patches[i].radius = obstacle_radius

            # Update velocity arrows
            dx = obstacle_velocity * np.cos(obstacle_theta)
            dy = obstacle_velocity * np.sin(obstacle_theta)
            x_values = [obstacle_x, obstacle_x + dx]
            y_values = [obstacle_y, obstacle_y + dy]
            velocity_arrows[i].set_data(x_values, y_values)

        # Update LiDAR points
        lidar = state['lidar']
        hit_points = lidar['hit_points']
        frequency_shifts = lidar['frequency_shifts']
        estimated_velocities = lidar['estimated_velocities']
        lidar_points.set_data(hit_points[:, 0], hit_points[:, 1])

        # Remove previous Doppler arrows
        for arrow in doppler_arrows:
            arrow.remove()
        doppler_arrows.clear()

        # Compute line-of-sight unit vectors
        los_dx = hit_points[:, 0] - robot_x
        los_dy = hit_points[:, 1] - robot_y
        distances = np.sqrt(los_dx**2 + los_dy**2)
        los_unit_x = los_dx / distances
        los_unit_y = los_dy / distances

        # Mask for points that hit obstacles
        hit_obstacle_mask = frequency_shifts != 0

        if np.any(hit_obstacle_mask):
            hit_points_obstacles = hit_points[hit_obstacle_mask]
            v_rel_los_est = estimated_velocities[hit_obstacle_mask]
            los_unit_x_obst = los_unit_x[hit_obstacle_mask]
            los_unit_y_obst = los_unit_y[hit_obstacle_mask]

            arrow_dx = v_rel_los_est * los_unit_x_obst * 3
            arrow_dy = v_rel_los_est * los_unit_y_obst * 3

            for x, y, dx, dy in zip(hit_points_obstacles[:, 0], hit_points_obstacles[:, 1], arrow_dx, arrow_dy):
                arrow = FancyArrowPatch((x, y), (x + dx, y + dy), color='purple', arrowstyle='->', mutation_scale=5)
                ax.add_patch(arrow)
                doppler_arrows.append(arrow)

        return [robot_dot] + obstacle_patches + velocity_arrows + [lidar_points, scan_line] + doppler_arrows + [time_text]

    ani = animation.FuncAnimation(
        fig, update, frames=len(states),
        init_func=init, blit=False, interval=150, repeat=False
    )

    if save:
        # Save as GIF using pillow writer
        ani.save(filename, writer='pillow', fps=10)
        print(f'Animation saved as {filename}')
    else:
        plt.show()
