# General imports.
from typing import List, Dict
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle, Polygon

# Project imports.
from gco.utils.transform_utils import Transform
from gco.world.obstacles import ObstacleCircle, ObstacleSquare, Obstacle
from gco.config import Config as cfg
from gco.utils.model_utils import tokens_to_meters_local, tokens_to_pixels

# Global colors.
RED = "\033[91m"
RESET = "\033[0m"

def _visualize_circle(ax, pose, radius, color):
    """Visualize a circle at the given pose."""
    ax.add_patch(plt.Circle((pose.t[0], pose.t[1]), radius, color=color))

def _visualize_rectangle(ax, pose, width, height, color):
    """Visualize a rectangle at the given pose with rotation."""
    rect = plt.Rectangle((pose.t[0], pose.t[1]),
                        width, height,
                        angle=pose.get_theta() * 180 / torch.pi,  # Convert to degrees
                        color=color)
    ax.add_patch(rect)

def _compute_triangle_vertices(width: float, height: float) -> List[List[float]]:
    """Compute the vertices of a triangle given its width and height.
    The triangle is centered at the origin with its base along the x-axis."""
    return [
        [-width/2, -height/2],  # Bottom left
        [width/2, -height/2],   # Bottom right
        [0, height/2]           # Top center
    ]

def _visualize_polygon(ax, pose, vertices, color):
    """Visualize a polygon at the given pose with rotation."""
    # Transform vertices by rotation and translation
    rotated_vertices = []
    for vertex in vertices:
        # Rotate vertex
        x = vertex[0] * torch.cos(pose.get_theta()) - vertex[1] * torch.sin(pose.get_theta())
        y = vertex[0] * torch.sin(pose.get_theta()) + vertex[1] * torch.cos(pose.get_theta())
        # Translate vertex
        x += pose.t[0]
        y += pose.t[1]
        rotated_vertices.append([x, y])
    print(rotated_vertices)
    rotated_vertices = torch.tensor(rotated_vertices)
    ax.add_patch(plt.Polygon(rotated_vertices, color=color))

def _visualize_obstacle(ax, obstacle, obstacle_pose, color):
    """Visualize an obstacle at the given pose."""
    if isinstance(obstacle, ObstacleCircle):
        _visualize_circle(ax, obstacle_pose, obstacle.radius.cpu(), color)
    elif isinstance(obstacle, ObstacleSquare):
        _visualize_rectangle(ax, obstacle_pose, obstacle.width.cpu(), obstacle.width.cpu(), color)  # Square uses width for both dimensions

def _visualize_object(ax, obj, obj_pose, color):
    """Visualize an object at the given pose."""
    if obj.__class__.__name__ == 'ObjectCircle':
        _visualize_circle(ax, obj_pose, obj.radius, color)
    elif obj.__class__.__name__ == 'ObjectRectangle':
        _visualize_rectangle(ax, obj_pose, obj.width, obj.height, color)
    elif obj.__class__.__name__ == 'ObjectSquare':
        _visualize_rectangle(ax, obj_pose, obj.width, obj.width, color)
    elif obj.__class__.__name__ == 'ObjectTriangle':
        vertices = _compute_triangle_vertices(obj.width, obj.height)
        _visualize_polygon(ax, obj_pose, vertices, color)
    elif obj.__class__.__name__ == 'ObjectT':
        # Use the vertices property of the ObjectT class which now correctly matches the T-shape generation
        vertices = obj.vertices.cpu().numpy().tolist()
        _visualize_polygon(ax, obj_pose, vertices, color)
    elif obj.__class__.__name__ == 'ObjectPolygon':
        # Convert vertices tensor to list of lists for visualization
        vertices = obj.vertices.cpu().numpy().tolist()
        _visualize_polygon(ax, obj_pose, vertices, color)


def visualize_world_trajectory(world: 'World',
                               world_states: List[Dict[str, Transform]],
                               save_path: str,
                               tail_length: int = 4):  # Number of previous positions to show
    """
    Visualize the world states with robot motion trails.
    :param world: The world instance containing obstacles, objects, and SDF
    :param world_states: a list of world states
    :param save_path: the path to save the visualization
    :param tail_length: number of previous positions to show in the trail
    """
    # Move all transforms to the CPU
    world_states_cpu = []
    for world_state in world_states:
        world_state_cpu = {}
        world_state_cpu['robots'] = {robot_name: robot_pose.to('cpu') for robot_name, robot_pose in
                                     world_state['robots'].items()}
        world_state_cpu['objects'] = {object_name: object_pose.to('cpu') for object_name, object_pose in
                                      world_state['objects'].items()}
        world_states_cpu.append(world_state_cpu)

    world_states = world_states_cpu

    # Get world dimensions from sdf
    world_height, world_width = world.size
    aspect_ratio = world_width / world_height

    base_size = 10
    if aspect_ratio > 1:
        fig_width = base_size
        fig_height = base_size / aspect_ratio
    else:
        fig_height = base_size
        fig_width = base_size * aspect_ratio

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # Set axis limits
    ax.set_xlim(-world_width / 2, world_width / 2)
    ax.set_ylim(-world_height / 2, world_height / 2)
    ax.set_aspect('equal')

    # Choose colors
    robot_colors = {robot_name: plt.cm.Set3(i / len(world_states[0]["robots"].keys()))
                    for i, robot_name in enumerate(world_states[0]["robots"].keys())}
    object_colors = {object_name: plt.cm.Reds(30 + 70 * (i / len(world.objects_d.keys())))
                     for i, object_name in enumerate(world.objects_d.keys())}
    obstacle_colors = {obstacle_name: plt.cm.Blues(30 + 70 * (i / len(world.obstacles_d.keys())))
                       for i, obstacle_name in enumerate(world.obstacles_d.keys())}

    # Dictionary to store position history for each robot
    robot_history = {robot_name: [] for robot_name in world_states[0]["robots"].keys()}

    def update(frame):
        ax.clear()
        ax.set_xlim(-world_width / 2, world_width / 2)
        ax.set_ylim(-world_height / 2, world_height / 2)
        ax.set_aspect('equal')

        # Add static obstacles
        for obstacle_name, obstacle_pose in world.obstacle_poses_d.items():
            _visualize_obstacle(ax, world.obstacles_d[obstacle_name],
                                obstacle_pose.cpu(), obstacle_colors[obstacle_name])

        world_state = world_states[frame]

        # Update robot positions and draw trails
        for robot_name, robot_pose in world_state["robots"].items():
            robot = world.robots_d[robot_name]
            color = robot_colors[robot_name]

            # Update position history
            robot_history[robot_name].append(robot_pose.t.clone())
            if len(robot_history[robot_name]) > tail_length:
                robot_history[robot_name].pop(0)

            # Draw trail
            for i, past_pos in enumerate(robot_history[robot_name][:-1]):
                # Calculate alpha and radius based on position in trail
                alpha = (i + 1) / len(robot_history[robot_name])
                radius = robot.radius.cpu() * (0.3 + 0.7 * alpha)  # Smaller circles for older positions

                # Create color with alpha
                trail_color = list(color)
                trail_color[3] = alpha * 0.5  # Reduce alpha for trail

                # Draw trail circle
                ax.add_patch(plt.Circle((past_pos[0], past_pos[1]),
                                        radius, color=trail_color))

            # Draw current robot position
            _visualize_circle(ax, robot_pose, robot.radius.cpu(), color)

        # Draw objects
        for object_name, object_pose in world_state["objects"].items():
            _visualize_object(ax, world.get_object(object_name), 
                              object_pose, object_colors[object_name])

    ani = animation.FuncAnimation(fig, update, frames=len(world_states), interval=100)
    ani.save(save_path)
    
def plot_contact_points_meters(contact_points: torch.Tensor):
    """Plot the contact points in meters.
    Note that the origin is in the middle of the image, x points up and y points left.
    """
    # Convert to x-right, y-up.
    contact_points_x_right_y_up = torch.stack([-contact_points[:, 1], contact_points[:, 0]], dim=1)
    plt.figure(figsize=(3,3))
    # Scatter, but make sure to adapt the xy values to x-right, y-up.
    plt.scatter(contact_points_x_right_y_up[:, 0], contact_points_x_right_y_up[:, 1])
    # Add axes. Axes should be at through the metric origin (middle of the image).
    plt.axhline(0, color='black', linewidth=1)
    plt.axvline(0, color='black', linewidth=1)
    plt.xlabel('y (meters)', fontsize=12)
    plt.ylabel('x (meters)', fontsize=12)
    plt.title('Contact Points')
    plt.xlim(-cfg.OBS_M / 2, cfg.OBS_M / 2)
    plt.ylim(-cfg.OBS_M / 2, cfg.OBS_M / 2)
    # Add labels for each point.    
    for i in range(contact_points.shape[0]):
        plt.text(contact_points_x_right_y_up[i, 0], contact_points_x_right_y_up[i, 1], f"({contact_points[i, 0]:.2f}, {contact_points[i, 1]:.2f})", fontsize=12)
    # Add arrows for the axes. Red for x, green for y.
    plt.arrow(0, 0, -0.2, 0, head_width=0.03, head_length=0.1, width=0.02, fc='g', ec='g')
    plt.arrow(0, 0, 0, 0.2, head_width=0.03, head_length=0.1, width=0.02, fc='r', ec='r')
    plt.show()

def plot_contact_points_pixels(contact_points_pixels: torch.Tensor):
    """Plot the contact points in pixels."""
    # Convert to x-right, y-up.
    img = torch.zeros((cfg.OBS_PIX, cfg.OBS_PIX))
    img[contact_points_pixels[:, 0], contact_points_pixels[:, 1]] = 1
    plt.figure(figsize=(3,3))
    plt.imshow(img)
    plt.show()

# =====================
# Originally from CTSWAP.
# =====================

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from matplotlib.patches import Circle, RegularPolygon
import json
from scipy.interpolate import interp1d

def visualize_trajectories(cfgs_start, cfgs_goal, paths, robot_radii=None, obstacle_positions=None, obstacle_radii=None, world_bounds=[-2, -2, 2, 2], smooth_paths=True, num_smooth_points=100):
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
    """

    # Remove the "robot_" prefix from the robot names.
    cfgs_start = {robot_name.replace("robot_", ""): cfgs_start[robot_name] for robot_name in cfgs_start}
    cfgs_goal = {robot_name.replace("robot_", ""): cfgs_goal[robot_name] for robot_name in cfgs_goal}
    paths = {robot_name.replace("robot_", ""): paths[robot_name] for robot_name in paths}
    
    
    # Smooth the trajectories if requested
    if smooth_paths:
        paths_to_plot = smooth_all_trajectories(paths, num_smooth_points)
    else:
        paths_to_plot = {name: np.array(path) for name, path in paths.items()}
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Set up color map for robots - using Tab20c for muted, professional colors
    num_robots = len(cfgs_start)
    # Use Tab20c colormap which provides muted, professional colors
    colors = plt.cm.Set2(np.linspace(0, 1, num_robots))
    
    # Plot obstacles
    if obstacle_positions and obstacle_radii:
        for obstacle_name, position in obstacle_positions.items():
            radius = obstacle_radii.get(obstacle_name, 0.3)  # Default radius if not specified
            circle = Circle(position, radius, color='#8B0000', fill=True, alpha=0.7, label='Obstacle')  # Dark red instead of bright red
            ax.add_patch(circle)
    
    # Plot start positions
    for i, (robot_name, position) in enumerate(cfgs_start.items()):
        robot_radius = robot_radii.get(robot_name, 0.12) if robot_radii else 0.12  # Default radius if not specified
        hexagon = RegularPolygon(position[:2], 6, radius=robot_radius, color=colors[i], fill=True, alpha=0.6, 
                       label=f'{robot_name} (start)')
        ax.add_patch(hexagon)
        # Add robot name label
        ax.text(position[0], position[1], robot_name, ha='center', va='center', fontsize=8, 
                weight='bold', color='black')
    
    # Plot goal positions
    for i, (robot_name, position) in enumerate(cfgs_goal.items()):
        robot_radius = robot_radii.get(robot_name, 0.12) if robot_radii else 0.12  # Default radius if not specified
        hexagon = RegularPolygon(position[:2], 6, radius=robot_radius, color='gray', fill=True, alpha=0.1, linewidth=0)
        ax.add_patch(hexagon)
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
    ax.grid(True, alpha=0.1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('CTSWAP Multi-Robot Trajectories')
    
    # Add legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper right', bbox_to_anchor=(1.15, 1))
    
    plt.tight_layout()
    plt.show()

def draw_rectangle(ax, height, width, cx, cy, theta):
    """
    Draw a rotated rectangle in matplotlib.
    """
    # Half‐dimensions
    l2 = height / 2.0
    w2 = width / 2.0

    corners = np.array([
        [-l2, -w2],
        [ l2, -w2],
        [ l2,  w2],
        [-l2,  w2]
    ])
    
    return draw_polygon(ax, corners, cx, cy, theta)


def draw_polygon(ax, vertices_local, cx, cy, theta):
    if ax is None:
        fig, ax = plt.subplots()

    # Rectangle corners in local frame (centered at origin)
    corners = np.array(vertices_local)

    # Rotation matrix
    R = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)]
    ])

    # Rotate and translate corners into world frame
    rotated = corners @ R.T + np.array([cx, cy])

    # Create a polygon patch
    poly = Polygon(rotated[0], closed=True, fill=True, color='#8B0000')
    ax.add_patch(poly)

    return ax

# def create_mrmp_animation(cfgs_start,
#                           cfgs_goal,
#                           paths,
#                           robot_radii=None,
#                           obstacles: Dict[str, Obstacle] = None,
#                           obstacle_poses: Dict[str, Transform] = None,
#                           world_bounds=[-2, -2, 2, 2], fps=10, smooth_paths=True, num_smooth_points=100):
#     """
#     Create an animated visualization of the trajectories.
#     """
#     if num_smooth_points < max(len(path) for path in paths.values()):
#         num_smooth_points = max(len(path) for path in paths.values())
    
#     # Smooth the trajectories if requested
#     if smooth_paths:
#         paths_to_animate = smooth_all_trajectories(paths, num_smooth_points)
#     else:
#         paths_to_animate = densify_all_trajectories(paths, num_smooth_points)
#         # paths_to_animate = {name: np.array(path) for name, path in paths.items()}
    
#     # Find the maximum path length
#     max_path_length = max(len(path) for path in paths_to_animate.values()) if paths_to_animate else 1
    
#     # Create figure and axis
#     fig, ax = plt.subplots(figsize=(12, 10))
    
#     # Set up color map for robots - using Tab20c for muted, professional colors
#     num_robots = len(cfgs_start)
#     # Use Tab20c colormap which provides muted, professional colors
#     colors = plt.cm.tab20c(np.linspace(0, 1, num_robots))
    
#     # Plot obstacles
#     for obstacle_name, obstacle in obstacles.items():
#         if isinstance(obstacle, ObstacleCircle):
#             position = obstacle_poses[obstacle_name].t.cpu().numpy()
#             radius = obstacle.radius.cpu().numpy()
#             circle = Circle(position, radius, color='#8B0000', fill=True, alpha=0.7)  # Dark red instead of bright red
#             ax.add_patch(circle)
#         elif obstacle.__class__.__name__ == "ObjectCircle":
#             position = obstacle_poses[obstacle_name].t.cpu().numpy()
#             radius = obstacle.radius
#             circle = Circle(position, radius, color='#8B0000', fill=True, alpha=0.7)  # Dark red instead of bright red
#             ax.add_patch(circle)
#         elif obstacle.__class__.__name__ == "ObjectRectangle":
#             position = obstacle_poses[obstacle_name].t.cpu().numpy()
#             theta = obstacle_poses[obstacle_name].get_theta().cpu().numpy()
#             # Use the correct width and height (width along x, height along y)
#             width = obstacle.width
#             height = obstacle.height
#             draw_rectangle(ax, height, width, position[0], position[1], theta)

#         else:
#             raise ValueError(f"Obstacle {obstacle_name} is not a circle. Only supporting circles for now in vis.")
    
#     # Plot goal positions
#     for i, (robot_name, position) in enumerate(cfgs_goal.items()):
#         robot_radius = robot_radii.get(robot_name, 0.12) if robot_radii else 0.12  # Default radius if not specified
#         hexagon = RegularPolygon(position[:2], 6, radius=robot_radius, color='gray', fill=True, alpha=0.1, linewidth=0)
#         ax.add_patch(hexagon)
#         # ax.text(position[0], position[1], 'G', ha='center', va='center', fontsize=10, 
#         #         weight='bold', color='black')
    
#     # Initialize robot positions and trajectory lines
#     robot_hexagons = []
#     trajectory_lines = []
#     robot_texts = []
    
#     for i, (robot_name, path) in enumerate(paths_to_animate.items()):
#         # Create robot hexagon
#         start_pos = cfgs_start[robot_name]
#         robot_radius = robot_radii.get(robot_name, 0.12) if robot_radii else 0.12  # Default radius if not specified
#         hexagon = RegularPolygon(start_pos[:2], 6, radius=robot_radius, color=colors[i], fill=True, alpha=0.9)
#         ax.add_patch(hexagon)
#         robot_hexagons.append(hexagon)
        
#         # Create robot name text
#         robot_name = robot_name.replace("robot_", "")
#         text = ax.text(start_pos[0], start_pos[1], robot_name, ha='center', va='center', 
#                       fontsize=8, weight='bold', color='black')
#         robot_texts.append(text)
        
#         # Create trajectory line
#         if path.size > 0:
#             line, = ax.plot([], [], color=colors[i], linewidth=2, alpha=0.6)
#             trajectory_lines.append(line)
#         else:
#             trajectory_lines.append(None)
    
#     # Set plot limits and properties
#     ax.set_xlim(world_bounds[0], world_bounds[2])
#     ax.set_ylim(world_bounds[1], world_bounds[3])
#     ax.set_aspect('equal', adjustable='box')
#     ax.grid(True, alpha=0.05)
#     ax.set_xlabel('X')
#     ax.set_ylabel('Y')
#     ax.set_title('CTSWAP Multi-Robot Trajectories Animation')
    
#     def animate(frame):
#         for i, (robot_name, path) in enumerate(paths_to_animate.items()):
#             if path.size > 0 and frame < len(path):
#                 # Update robot position
#                 robot_hexagons[i].xy = (path[frame, 0], path[frame, 1])
                
#                 # Update robot name text position
#                 robot_texts[i].set_position((path[frame, 0], path[frame, 1]))
                
#                 # Update trajectory line
#                 if trajectory_lines[i]:
#                     path_array = path[:frame+1]
#                     trajectory_lines[i].set_data(path_array[:, 0], path_array[:, 1])
        
#         return robot_hexagons + robot_texts + [line for line in trajectory_lines if line is not None]
    
#     # Create animation
#     anim = animation.FuncAnimation(fig, animate, frames=max_path_length, 
#                                   interval=1000//fps, blit=True, repeat=True)
    
#     plt.tight_layout()
#     plt.show()
    
#     return anim

def smooth_trajectory(path, num_points=100):
    """
    Smooth a trajectory by interpolating between waypoints.
    
    Args:
        path: List of [x, y, theta] waypoints
        num_points: Number of points to interpolate to
    
    Returns:
        Smoothed trajectory as numpy array
    """
    from scipy.interpolate import interp1d
    from scipy.signal import savgol_filter
    
    path = np.array(path)
    if len(path) < 2:
        return path

    # Step 1: arc-length-based parameterization
    diffs = np.diff(path[:, :2], axis=0)
    dists = np.linalg.norm(diffs, axis=1)
    arc_lengths = np.insert(np.cumsum(dists), 0, 0.0)
    arc_lengths /= arc_lengths[-1]  # Normalize to [0, 1]

    t_new = np.linspace(0, 1, num_points)
    result = np.zeros((num_points, 3))

    # Step 2: interpolate x, y
    for i in range(2):  # x and y
        interp = interp1d(arc_lengths, path[:, i], kind='linear')
        result[:, i] = interp(t_new)

    # Step 3: unwrap and interpolate theta
    theta_unwrapped = np.unwrap(path[:, 2])
    interp_theta = interp1d(arc_lengths, theta_unwrapped, kind='linear')
    result[:, 2] = interp_theta(t_new)

    # Step 4: apply Savitzky–Golay smoothing (requires window length <= num_points)
    window_length = 20
    polyorder = 2
    win = min(window_length, num_points if num_points % 2 == 1 else num_points - 1)
    if win > polyorder:
        for i in range(3):
            result[:, i] = savgol_filter(result[:, i], window_length=win, polyorder=polyorder, mode='nearest')

    smoothed_path = result
    
    return smoothed_path

def smooth_all_trajectories(paths, num_points=100):
    """
    Smooth all robot trajectories.
    
    Args:
        paths: Dictionary of robot paths (each is a list of [x, y, theta] waypoints)
        num_points: Number of points to interpolate each trajectory to
    
    Returns:
        Dictionary of smoothed trajectories
    """
    smoothed_paths = {}
    for robot_name, path in paths.items():
        if path and len(path) > 0:
            smoothed_paths[robot_name] = smooth_trajectory(path, num_points)
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
    points_per_segment = max(1, int(num_points / num_segments))
    # points_per_segment = 10
    total_points = num_segments * points_per_segment + 1
    
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

def densify_all_trajectories(paths, num_points_per_edge=10):
    """
    Densify all robot trajectories by adding intermediate points between waypoints.
    
    Args:
        paths: Dictionary of robot paths
        num_points: Target number of total points per trajectory (approximately)
    
    Returns:
        Dictionary of densified trajectories
    """
    num_points = num_points_per_edge * max(len(path) for path in paths.values())
    densified_paths = {}
    for robot_name, path in paths.items():
        if path and len(path) > 0:
            densified_paths[robot_name] = densify_trajectory(path, num_points)
        else:
            densified_paths[robot_name] = np.array([])
    return densified_paths
