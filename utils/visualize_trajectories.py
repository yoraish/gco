import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from matplotlib.patches import Circle, RegularPolygon, Rectangle
import json
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
import sys
import argparse
import os

# Global visualization parameters
BACKGROUND_COLOR = 'white'
TEXT_COLOR = BACKGROUND_COLOR
GRID_COLOR = BACKGROUND_COLOR

def visualize_trajectories(cfgs_start, cfgs_goal, paths, robot_radii=None, obstacle_positions=None, obstacle_radii=None, world_bounds=[-2, -2, 2, 2], smooth_paths=True, num_smooth_points=100, savgol_window_length=3, robot_shape='circle'):
    """
    Visualize the CTSWAP trajectories.
    
    Args:
        cfgs_start: Dictionary of robot start configurations {robot_name: [x, y, theta]}
        cfgs_goal: Dictionary of robot goal configurations {robot_name: [x, y, theta]}
        paths: Dictionary of robot paths {robot_name: [[x, y, theta], ...]}
        robot_radii: Dictionary of robot radii {robot_name: radius}
        obstacle_positions: Dictionary of obstacle positions {obstacle_name: [x, y]}
        obstacle_radii: Dictionary of obstacle radii {obstacle_name: radius}
        world_bounds: [x_min, y_min, x_max, y_max] for plot limits
        smooth_paths: Whether to smooth the trajectories
        num_smooth_points: Number of points to use for smoothing
        robot_shape: Shape to use for robot visualization ('circle' or 'hexagon')
    """

    # Remove the "robot_" prefix from the robot names.
    cfgs_start = {robot_name.replace("robot_", ""): cfgs_start[robot_name] for robot_name in cfgs_start}
    cfgs_goal = {robot_name.replace("robot_", ""): cfgs_goal[robot_name] for robot_name in cfgs_goal}
    paths = {robot_name.replace("robot_", ""): paths[robot_name] for robot_name in paths}
    
    
    # Smooth the trajectories if requested
    if smooth_paths:
        paths_to_plot = smooth_all_trajectories(paths, num_smooth_points, savgol_window_length)
    else:
        paths_to_plot = {name: np.array(path) for name, path in paths.items()}
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Set background color
    fig.patch.set_facecolor(BACKGROUND_COLOR)
    ax.set_facecolor(BACKGROUND_COLOR)
    
    # Set up color map for robots - using Tab20c for muted, professional colors
    num_robots = len(cfgs_start)
    # Use Tab20c colormap which provides muted, professional colors
    colors = plt.cm.Set2(np.linspace(0, 1, num_robots))
    
    # Plot obstacles
    if obstacle_positions and obstacle_radii:
        for obstacle_name, position in obstacle_positions.items():
            # print(f"Obstacle name: {obstacle_name}")
            # print(f"Obstacle radius: {obstacle_radii.get(obstacle_name)}")
            if "circle" in obstacle_name:
                radius = obstacle_radii.get(obstacle_name, 0.3)  # Default radius if not specified
            elif "square" in obstacle_name:
                radius = obstacle_radii.get(obstacle_name, 0.3)  # Default radius if not specified
            elif "rectangle" in obstacle_name:
                radius = obstacle_radii.get(obstacle_name, [0.3, 0.3])  # Default radius if not specified
                # print(f"Rectangle radius: {radius}")
            else:
                radius = 0.3  # Default radius if not specified
            if len(radius) == 1:
                circle = Circle(position, radius, color='#8B0000', fill=True, alpha=0.7, label='Obstacle')  # Dark red instead of bright red
                ax.add_patch(circle)
            elif len(radius) == 2:
                # print(f"Rectangle radius: {radius}")
                rectangle = Rectangle(position, radius[0], radius[1], color='#8B0000', fill=True, alpha=0.7, label='Obstacle')  # Dark red instead of bright red
                ax.add_patch(rectangle)
    
    # Plot start positions
    for i, (robot_name, position) in enumerate(cfgs_start.items()):
        robot_radius = robot_radii.get(robot_name, 0.12) if robot_radii else 0.12  # Default radius if not specified
        robot_radius *= 0.8  # Scale to 80% of prescribed size
        if robot_shape == 'hexagon':
            robot_patch = RegularPolygon(position[:2], 6, radius=robot_radius, color=colors[i], fill=True, alpha=0.6, 
                           label=f'{robot_name} (start)')
        else:  # Default to circle
            robot_patch = Circle(position[:2], robot_radius, color=colors[i], fill=True, alpha=0.6, 
                           label=f'{robot_name} (start)')
        ax.add_patch(robot_patch)
        # Add robot name label
        ax.text(position[0], position[1], robot_name, ha='center', va='center', fontsize=8, 
                weight='bold', color=TEXT_COLOR)
    
    # Plot goal positions
    for i, (robot_name, position) in enumerate(cfgs_goal.items()):
        robot_radius = robot_radii.get(robot_name, 0.12) if robot_radii else 0.12  # Default radius if not specified
        robot_radius *= 0.8  # Scale to 80% of prescribed size
        if robot_shape == 'hexagon':
            goal_patch = RegularPolygon(position[:2], 6, radius=robot_radius, color='gray', fill=True, alpha=0.1, linewidth=0)
        else:  # Default to circle
            goal_patch = Circle(position[:2], robot_radius, color='gray', fill=True, alpha=0.1, linewidth=0)
        ax.add_patch(goal_patch)
        # Add goal label
        # ax.text(position[0], position[1], 'G', ha='center', va='center', fontsize=10, 
        #         weight='bold', color='black')
    
    # Plot trajectories
    for i, (robot_name, path) in enumerate(paths_to_plot.items()):
        if path.size > 0:
            ax.plot(path[:, 0], path[:, 1], color=colors[i], linewidth=2, 
                   label=f'{robot_name} trajectory', alpha=0.8)
            
            # Add arrows to show direction
            for j in range(0, len(path), max(1, len(path)//10)):
                if j < len(path) - 1:
                    dx = path[j+1, 0] - path[j, 0]
                    dy = path[j+1, 1] - path[j, 1]
                    ax.arrow(path[j, 0], path[j, 1], dx*0.3, dy*0.3, 
                            head_width=0.05, head_length=0.05, fc=colors[i], ec=colors[i], alpha=0.7)
    
    # Set plot limits and properties
    ax.set_xlim(world_bounds[0], world_bounds[2])
    ax.set_ylim(world_bounds[1], world_bounds[3])
    ax.set_aspect('equal', adjustable='box')
    # Remove axes, grid, and labels
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    # Add legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper right', bbox_to_anchor=(1.15, 1), 
              facecolor=BACKGROUND_COLOR, edgecolor=TEXT_COLOR, labelcolor=TEXT_COLOR)
    
    plt.tight_layout()
    plt.show()

def create_animation(cfgs_start, cfgs_goal, paths, robot_radii=None, obstacle_positions=None, obstacle_radii=None, 
                    world_bounds=[-3, -3, 3, 3], fps=10, smooth_paths=True, num_points_per_edge=10, show_trajectories=True,
                    savgol_window_length=3, robot_shape='circle'):
    """
    Create an animated visualization of the trajectories.
    """
    
    # Smooth the trajectories if requested
    if smooth_paths:
        num_points_path = max([len(path) for path in paths.values()])
        num_points_smooth_path = 60
        paths_to_animate = smooth_all_trajectories(paths, num_points_smooth_path, savgol_window_length)  # now has num_points_smooth_path points per path.
        # Convert to lists for densification
        paths_to_animate = {k: v.tolist() for k, v in paths_to_animate.items()}
        # Densify with a reasonable number of points
        # densify_factor = max(1, num_points_smooth_path // num_points_per_edge)
        num_points_per_edge_for_smoothed_path = max([2, int(1.0 / num_points_smooth_path * num_points_per_edge * num_points_path)])
        print(f"q: {num_points_per_edge_for_smoothed_path} = {1.0 / num_points_smooth_path} * {num_points_per_edge} * {num_points_path}")
        paths_to_animate = densify_all_trajectories(paths_to_animate, num_points_per_edge_for_smoothed_path)
    else:
        # Get a reasonable number of points for densification
        paths_to_animate = densify_all_trajectories(paths, num_points_per_edge)
        # paths_to_animate = {k : np.array(v)[::4] for(k,v) in paths.items()}

    # Add a few goal configurations to the paths.
    for robot_name in paths_to_animate:
        for _ in range(10):
            paths_to_animate[robot_name] = np.concatenate([paths_to_animate[robot_name], [paths_to_animate[robot_name][-1]]])
    
    # Find the maximum path length
    max_path_length = max(len(path) for path in paths_to_animate.values()) if paths_to_animate else 1
    
    # Create figure and axis.
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Set background color
    fig.patch.set_facecolor(BACKGROUND_COLOR)
    ax.set_facecolor(BACKGROUND_COLOR)
    
    # Set up color map for robots - using Tab20c for muted, professional colors
    num_robots = len(cfgs_start)
    # Use Tab20c colormap which provides muted, professional colors
    colors = plt.cm.tab20c(np.linspace(0, 1, num_robots))
    
    # Plot obstacles
    if obstacle_positions and obstacle_radii:
        for obstacle_name, position in obstacle_positions.items():
            if "circle" in obstacle_name:
                radius = obstacle_radii.get(obstacle_name, 0.3)  # Default radius if not specified
                circle = Circle(position, radius, color='#8B0000', fill=True, alpha=0.7)  # Dark red instead of bright red
                ax.add_patch(circle)
            elif "square" in obstacle_name:
                radius = obstacle_radii.get(obstacle_name, 0.3)  # Default radius if not specified
                square = Rectangle(position, radius, radius, color='#8B0000', fill=True, alpha=0.7)  # Dark red instead of bright red
                ax.add_patch(square)
            elif "rectangle" in obstacle_name:
                position_x = position[0] - obstacle_radii.get(obstacle_name, [0.3, 0.3])[1] / 2
                position_y = position[1] - obstacle_radii.get(obstacle_name, [0.3, 0.3])[0] / 2
                position = (position_x, position_y)
                radius = obstacle_radii.get(obstacle_name, [0.3, 0.3])  # Default radius if not specified
                rectangle = Rectangle(position, radius[1], radius[0], color='#8B0000', fill=True, alpha=0.7)  # Dark red instead of bright red
                ax.add_patch(rectangle)
    
    # Plot goal positions
    for i, (robot_name, position) in enumerate(cfgs_goal.items()):
        robot_radius = robot_radii.get(robot_name, 0.12) if robot_radii else 0.12  # Default radius if not specified
        robot_radius *= 0.8  # Scale to 80% of prescribed size
        if robot_shape == 'hexagon':
            goal_patch = RegularPolygon(position[:2], 6, radius=robot_radius, color='gray', fill=True, alpha=0.1, linewidth=0)
        else:  # Default to circle
            goal_patch = Circle(position[:2], robot_radius, color='gray', fill=True, alpha=0.1, linewidth=0)
        ax.add_patch(goal_patch)
        # ax.text(position[0], position[1], 'G', ha='center', va='center', fontsize=10, 
        #         weight='bold', color='black')
    
    # Initialize robot positions and trajectory lines
    robot_patches = []
    trajectory_lines = []
    robot_texts = []
    
    for i, (robot_name, path) in enumerate(paths_to_animate.items()):
        # Create robot patch
        start_pos = cfgs_start[robot_name]
        robot_radius = robot_radii.get(robot_name, 0.12) if robot_radii else 0.12  # Default radius if not specified
        robot_radius *= 0.8  # Scale to 80% of prescribed size
        if robot_shape == 'hexagon':
            robot_patch = RegularPolygon(start_pos[:2], 6, radius=robot_radius, color=colors[i], fill=True, alpha=0.9)
        else:  # Default to circle
            robot_patch = Circle(start_pos[:2], robot_radius, color=colors[i], fill=True, alpha=0.9)
        ax.add_patch(robot_patch)
        robot_patches.append(robot_patch)
        
        # Create robot name text
        robot_name = robot_name.replace("robot_", "")
        text = ax.text(start_pos[0], start_pos[1], robot_name, ha='center', va='center', 
                      fontsize=8, weight='bold', color=TEXT_COLOR)
        robot_texts.append(text)
        
        # Create trajectory line (optional for performance)
        if show_trajectories and path.size > 0:
            line, = ax.plot([], [], color=colors[i], linewidth=2, alpha=0.6)
            trajectory_lines.append(line)
        else:
            trajectory_lines.append(None)
    
    # Set plot limits and properties
    ax.set_xlim(world_bounds[0], world_bounds[2])
    ax.set_ylim(world_bounds[1], world_bounds[3])
    ax.set_aspect('equal', adjustable='box')
    # Remove axes, grid, and labels
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    # Create title text that will be updated during animation
    title_text = ax.text(0.5, 1.02, 'CTSWAP Multi-Robot Trajectories Animation - Step: 0', 
                        ha='center', va='bottom', transform=ax.transAxes, fontsize=12, weight='bold', color=TEXT_COLOR)
    
    def animate(frame):
        # Calculate actual frame index with skipping
        actual_frame = frame * frame_skip
        
        # Update title with current step
        title_text.set_text(f'CTSWAP Multi-Robot Trajectories Animation - Step: {actual_frame / 6:.2f}')
        
        for i, (robot_name, path) in enumerate(paths_to_animate.items()):
            if path.size > 0 and actual_frame < len(path):
                # Update robot position
                if robot_shape == 'hexagon':
                    robot_patches[i].xy = (path[actual_frame, 0], path[actual_frame, 1])
                else:  # Default to circle
                    robot_patches[i].center = (path[actual_frame, 0], path[actual_frame, 1])
                
                # Update robot name text position
                robot_texts[i].set_position((path[actual_frame, 0], path[actual_frame, 1]))
                
                # Update trajectory line (show full trajectory up to current frame)
                if trajectory_lines[i] and show_trajectories:
                    path_array = path[:actual_frame+1]
                    trajectory_lines[i].set_data(path_array[:, 0], path_array[:, 1])
        
        return robot_patches + robot_texts + [line for line in trajectory_lines if line is not None]
    
    # Create animation with frame skipping to reduce total frames
    # Skip frames to keep animation duration reasonable (target ~10-15 seconds)
    target_duration = 12  # seconds
    frame_skip = max(1, max_path_length // (target_duration * fps))
    actual_frames = max_path_length // frame_skip
    
    print(f"Original frames: {max_path_length}, Skipping every {frame_skip} frames, Final frames: {actual_frames}")
    
    # Create animation
    anim = animation.FuncAnimation(fig, animate, frames=actual_frames, 
                                  interval=1000//fps, blit=False, repeat=True)
    
    plt.tight_layout()
    
    # Save the last frame in high resolution
    save_last_frame_high_res(fig, ax, robot_patches, robot_texts, trajectory_lines, 
                            paths_to_animate, cfgs_start, cfgs_goal, robot_radii, 
                            obstacle_positions, obstacle_radii, world_bounds, colors, 
                            show_trajectories, frame_skip, robot_shape)
    
    plt.show()
    
    return anim

def save_last_frame_high_res(fig, ax, robot_patches, robot_texts, trajectory_lines, 
                            paths_to_animate, cfgs_start, cfgs_goal, robot_radii, 
                            obstacle_positions, obstacle_radii, world_bounds, colors, 
                            show_trajectories, frame_skip, robot_shape='circle'):
    """
    Save the last frame of the animation in high resolution.
    """
    # Find the maximum path length
    max_path_length = max(len(path) for path in paths_to_animate.values()) if paths_to_animate else 1
    
    # Calculate the last frame index
    actual_frames = max_path_length // frame_skip
    last_frame = actual_frames - 1
    actual_frame = last_frame * frame_skip
    
    # Create a new high-resolution figure
    fig_hr, ax_hr = plt.subplots(figsize=(16, 12), dpi=150)  # High resolution: 16x12 inches at 300 DPI
    
    # Set background color
    fig_hr.patch.set_facecolor(BACKGROUND_COLOR)
    ax_hr.set_facecolor(BACKGROUND_COLOR)
    
    # Plot obstacles (same as in animation)
    if obstacle_positions and obstacle_radii:
        for obstacle_name, position in obstacle_positions.items():
            if "circle" in obstacle_name:
                radius = obstacle_radii.get(obstacle_name, 0.3)
                circle = Circle(position, radius, color='#8B0000', fill=True, alpha=0.7)
                ax_hr.add_patch(circle)
            elif "square" in obstacle_name:
                radius = obstacle_radii.get(obstacle_name, 0.3)
                square = Rectangle(position, radius, radius, color='#8B0000', fill=True, alpha=0.7)
                ax_hr.add_patch(square)
            elif "rectangle" in obstacle_name:
                position_x = position[0] - obstacle_radii.get(obstacle_name, [0.3, 0.3])[1] / 2
                position_y = position[1] - obstacle_radii.get(obstacle_name, [0.3, 0.3])[0] / 2
                position = (position_x, position_y)
                radius = obstacle_radii.get(obstacle_name, [0.3, 0.3])
                rectangle = Rectangle(position, radius[1], radius[0], color='#8B0000', fill=True, alpha=0.7)
                ax_hr.add_patch(rectangle)
    
    # Plot goal positions
    for i, (robot_name, position) in enumerate(cfgs_goal.items()):
        robot_radius = robot_radii.get(robot_name, 0.12) if robot_radii else 0.12
        robot_radius *= 0.8  # Scale to 80% of prescribed size
        if robot_shape == 'hexagon':
            goal_patch = RegularPolygon(position[:2], 6, radius=robot_radius, color='gray', fill=True, alpha=0.1, linewidth=0)
        else:  # Default to circle
            goal_patch = Circle(position[:2], robot_radius, color='gray', fill=True, alpha=0.1, linewidth=0)
        ax_hr.add_patch(goal_patch)
    
    # Plot robots at their final positions and trajectories
    for i, (robot_name, path) in enumerate(paths_to_animate.items()):
        if path.size > 0 and actual_frame < len(path):
            # Plot robot at final position
            robot_radius = robot_radii.get(robot_name, 0.12) if robot_radii else 0.12
            robot_radius *= 0.8  # Scale to 80% of prescribed size
            if robot_shape == 'hexagon':
                robot_patch = RegularPolygon(path[actual_frame, :2], 6, radius=robot_radius, 
                                       color=colors[i], fill=True, alpha=0.9)
            else:  # Default to circle
                robot_patch = Circle(path[actual_frame, :2], robot_radius, 
                                   color=colors[i], fill=True, alpha=0.9)
            ax_hr.add_patch(robot_patch)
            
            # Add robot name text
            robot_name_clean = robot_name.replace("robot_", "")
            ax_hr.text(path[actual_frame, 0], path[actual_frame, 1], robot_name_clean, 
                      ha='center', va='center', fontsize=12, weight='bold', color=TEXT_COLOR)
            
            # Plot full trajectory
            if show_trajectories:
                ax_hr.plot(path[:, 0], path[:, 1], color=colors[i], linewidth=3, alpha=0.8)
                
                # Add arrows to show direction (fewer arrows for cleaner look)
                for j in range(0, len(path), max(1, len(path)//15)):
                    if j < len(path) - 1:
                        dx = path[j+1, 0] - path[j, 0]
                        dy = path[j+1, 1] - path[j, 1]
                        ax_hr.arrow(path[j, 0], path[j, 1], dx*0.3, dy*0.3, 
                                  head_width=0.08, head_length=0.08, fc=colors[i], ec=colors[i], alpha=0.7)
    
    # Set plot limits and properties
    ax_hr.set_xlim(world_bounds[0], world_bounds[2])
    ax_hr.set_ylim(world_bounds[1], world_bounds[3])
    ax_hr.set_aspect('equal', adjustable='box')
    
    # Remove axes, grid, and labels
    ax_hr.set_xticks([])
    ax_hr.set_yticks([])
    ax_hr.spines['top'].set_visible(False)
    ax_hr.spines['right'].set_visible(False)
    ax_hr.spines['bottom'].set_visible(False)
    ax_hr.spines['left'].set_visible(False)
    
    # Add title
    ax_hr.text(0.5, 1.02, f'CTSWAP Multi-Robot Trajectories - Final Configuration (Step: {actual_frame / 6:.2f})', 
              ha='center', va='bottom', transform=ax_hr.transAxes, fontsize=16, weight='bold', color=TEXT_COLOR)
    
    plt.tight_layout()
    
    # Create output directory if it doesn't exist
    os.makedirs('output', exist_ok=True)
    
    # Save the high-resolution image
    output_filename = f'output/final_frame_high_res_{len(cfgs_start)}_robots.png'
    plt.savefig(output_filename, dpi=150, bbox_inches='tight', facecolor=BACKGROUND_COLOR, 
                edgecolor='none', pad_inches=0.2)
    print(f"High-resolution final frame saved to: {output_filename}")
    
    plt.close(fig_hr)

def smooth_trajectory(path, num_points=100, savgol_window_length=5):
    """
    Smooth a trajectory by interpolating between waypoints and applying Savitzky-Golay filter.
    
    Args:
        path: List of [x, y, theta] waypoints
        num_points: Number of points to interpolate to
    
    Returns:
        Smoothed trajectory as numpy array
    """
    if len(path) < 2:
        return np.array(path)
    
    # Convert to numpy array
    path_array = np.array(path)
    
    # Create parameter t for interpolation (0 to 1)
    t_original = np.linspace(0, 1, len(path))
    t_new = np.linspace(0, 1, num_points)
    
    # Interpolate each dimension separately
    smoothed_path = np.zeros((num_points, 3))
    
    for i in range(3):  # x, y, theta
        if len(path) > 1:
            # Use cubic interpolation for smoother curves
            interpolator = interp1d(t_original, path_array[:, i], kind='cubic', 
                                   bounds_error=False, fill_value='extrapolate')
            interpolated_data = interpolator(t_new)
            
            # Apply Savitzky-Golay filter for aggressive smoothing
            # Use a window length that's about 1/3 of the data length, but at least 5 and odd
            # window_length = max(5, min(num_points // 3, num_points - 1))
            window_length = savgol_window_length
            if window_length % 2 == 0:  # Ensure odd window length
                window_length += 1
            
            # Use a polynomial order of 2 for smooth curves
            smoothed_path[:, i] = savgol_filter(interpolated_data, window_length, 2)
        else:
            smoothed_path[:, i] = path_array[0, i]
    
    return smoothed_path

def smooth_all_trajectories(paths, num_points=100, savgol_window_length=3):
    """
    Smooth all robot trajectories.
    
    Args:
        paths: Dictionary of robot paths
        num_points: Number of points to interpolate each trajectory to
    
    Returns:
        Dictionary of smoothed trajectories
    """
    smoothed_paths = {}
    for robot_name, path in paths.items():
        if path and len(path) > 0:
            smoothed_paths[robot_name] = smooth_trajectory(path, num_points, savgol_window_length)
        else:
            smoothed_paths[robot_name] = np.array([])
    return smoothed_paths

def densify_trajectory(path, num_points=100):
    """
    Densify a trajectory by adding intermediate points between waypoints (linearly interpolated).
    
    Args:
        path: List of [x, y, theta] waypoints
        num_points: Target number of total points (approximately)
    
    Returns:
        Densified trajectory as numpy array
    """
    if len(path) < 2:
        return np.array(path)
    
    # Convert to numpy array
    path_array = np.array(path)
    
    # Calculate how many intermediate points to add between each pair of waypoints
    num_segments = len(path) - 1
    if num_segments == 0:
        return path_array
    
    # Distribute the target points across segments
    # points_per_segment = max(1, int(num_points / num_segments))
    points_per_segment = num_points
    
    # Create the densified path
    densified_path = []
    
    for i in range(num_segments):
        start_point = path_array[i]
        end_point = path_array[i + 1]
        
        # Add intermediate points for this segment
        for j in range(points_per_segment):
            t = j / points_per_segment
            interpolated_point = start_point + t * (end_point - start_point)
            densified_path.append(interpolated_point)
    
    # Add the final point
    densified_path.append(path_array[-1])
    
    return np.array(densified_path)

def densify_all_trajectories(paths, num_points=100):
    """
    Densify all robot trajectories by adding intermediate points between waypoints.
    
    Args:
        paths: Dictionary of robot paths
        num_points: Target number of total points per trajectory (approximately)
    
    Returns:
        Dictionary of densified trajectories
    """
    densified_paths = {}
    for robot_name, path in paths.items():
        if path and len(path) > 0:
            densified_paths[robot_name] = densify_trajectory(path, num_points)
        else:
            densified_paths[robot_name] = np.array([])
    return densified_paths

# Example usage with sample data
if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Visualize robot trajectories from JSON file')
    parser.add_argument('json_file', nargs='?', default="gco-cpp/build/examples/amrmp_solution.json",
                       help='Path to the JSON file containing trajectory data (default: gco-cpp/build/examples/amrmp_solution.json)')
    parser.add_argument('--fps', type=int, default=20, help='Animation FPS (default: 20)')
    parser.add_argument('--smooth', action='store_true', help='Enable trajectory smoothing')
    parser.add_argument('--no-trajectories', action='store_true', help='Disable trajectory visualization')
    parser.add_argument('--robot-shape', choices=['circle', 'hexagon'], default='circle', 
                       help='Shape to use for robot visualization (default: circle)')
    
    args = parser.parse_args()

    # Load data from JSON file
    try:
        with open(args.json_file, "r") as f:
            data = json.load(f)
        print(f"Loaded trajectory data from: {args.json_file}")
    except FileNotFoundError:
        print(f"Error: File '{args.json_file}' not found.")
        print("Usage: python visualize_trajectories.py [json_file] [options]")
        print("Example: python visualize_trajectories.py benchmark_solution_pibt_0_1_1.json --fps 30")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: File '{args.json_file}' is not a valid JSON file.")
        sys.exit(1)

    
    cfgs_start = data["cfgs_start"]
    cfgs_goal = data["cfgs_goal"]
    paths = data["paths"]

    # Add a few goal configurations to the paths.
    for robot_name in paths:
        paths[robot_name].append(paths[robot_name][-1])
        paths[robot_name].append(paths[robot_name][-1])
        paths[robot_name].append(paths[robot_name][-1])
    
    # Load robot radii and obstacle information from JSON
    robot_radii = data.get("robot_radii", {})
    obstacle_positions = data.get("obstacle_positions", {})
    obstacle_radii = data.get("obstacle_radii", {})
    
    # print(f"Number of robots: {len(cfgs_start)}")
    # print(f"Robot radii: {robot_radii}")
    # print(f"Obstacle positions: {obstacle_positions}")
    # print(f"Obstacle radii: {obstacle_radii}")
    # print(f"Start configurations: {cfgs_start}")
    # print(f"Goal configurations: {cfgs_goal}")

    # Make all paths the same length by padding with the last point.
    T = max([len(path) for path in paths.values()])
    for robot_name in paths:
        paths[robot_name] = paths[robot_name] + [paths[robot_name][-1]] * (T - len(paths[robot_name]))
    
    # If paths are 500 steps or longer, only visualize the first 150 steps
    if T >= 490:
        print(f"Path length is {T} steps (>= 500), limiting visualization to first 150 steps")
        for robot_name in paths:
            paths[robot_name] = paths[robot_name][:800]
        
    # Static visualization
    print("Creating static visualization...")
    # visualize_trajectories(cfgs_start, cfgs_goal, paths, robot_radii, obstacle_positions, obstacle_radii, 
                        #   smooth_paths=True, num_points_per_edge=200)

    # Only use first 1000 frames.
    # for robot_name in paths:
    #     paths[robot_name] = paths[robot_name][:1000]
    
    # Animated visualization
    print("Creating animated visualization...")
    print(f"Animation settings: FPS={args.fps}, Smoothing={args.smooth}, Show trajectories={not args.no_trajectories}")
    
    # anim = create_animation(cfgs_start, cfgs_goal, paths, robot_radii, obstacle_positions, obstacle_radii,
    #                         smooth_paths=False, num_points_per_edge=2, fps=90) 
    args.smooth = True
    anim = create_animation(cfgs_start, cfgs_goal, paths, robot_radii, obstacle_positions, obstacle_radii,
                            smooth_paths=args.smooth, num_points_per_edge=1, fps=args.fps, 
                            show_trajectories=not args.no_trajectories,
                            savgol_window_length=7, robot_shape=args.robot_shape)  # Disabled trajectories for speed
    
    # Save the animation before showing it to avoid the tkinter error
    print("Saving animation...")
    try:
        anim.save(f'output/animation_{len(cfgs_start)}.mp4', writer='ffmpeg', fps=30)  # Reduced from 90 to 30 FPS
        print(f"Animation saved successfully to: output/animation_{len(cfgs_start)}.mp4")
    except Exception as e:
        print(f"Error saving animation: {e}")
        print("Trying with Pillow writer instead...")
        try:
            anim.save(f'output/animation_{len(cfgs_start)}.gif', writer='pillow', fps=30)
            print(f"Animation saved as GIF to: output/animation_{len(cfgs_start)}.gif")
        except Exception as e2:
            print(f"Error saving GIF: {e2}")
            print("Animation will be displayed but not saved.")
