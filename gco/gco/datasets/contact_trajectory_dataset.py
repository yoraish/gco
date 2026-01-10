# General imports.
from torch.utils.data import Dataset
import random
import torch
import matplotlib.pyplot as plt
from scipy import ndimage
import matplotlib.animation as animation
import math
from matplotlib.animation import FuncAnimation
import os
import pickle
import json
from tqdm import tqdm
import numpy as np
from matplotlib.path import Path as MatplotlibPath

# Project imports.
from gco.config import Config as cfg
from gco.utils.transform_utils import *
from gco.utils.model_utils import *
from gco.world.world import World
from gco.world.robot import RobotDisk
from gco.world.objects import ObjectRectangle, ObjectCircle, MovableObject, ObjectT, ObjectPolygon, ObjectTriangle
from gco.utils.viz_utils import densify_all_trajectories, smooth_all_trajectories, RED, RESET
from gco.utils.data_vis_utils import visualize_transformed_mask, visualize_push_trajectory
from gco.observations.observation_generator import ObservationGenerator

class ContactTrajectoryDataset(Dataset):
    
    mask_expand_kernel = None
    
    def __init__(self, num_samples: int = 10000, 
                 primitive_type: str = "primitives-linear", 
                 shape_types: list[str] = None,
                 visualize: bool = False, 
                 save_intermediates = False,
                 dt: float = 0.003,
                 robot_budgets: list[int] = None):
        self.num_samples = num_samples
        self.shape_types = shape_types if shape_types is not None else cfg.shape_types
        self.primitive_type = primitive_type
        self.dt = dt
        self.robot_budgets = robot_budgets if robot_budgets is not None else [3]  # Default to 3 robots for backward compatibility
        if self.primitive_type == "primitives-physics" or self.primitive_type == "random-physics":
            self.datapoints = self.generate_dataset_physics(visualize=visualize, save_intermediates=save_intermediates)
        elif self.primitive_type == "primitives-linear":
            self.datapoints = self.generate_dataset_linear(save_intermediates=save_intermediates)
        else:
            raise ValueError(f"Invalid primitive type: {self.primitive_type}")
                

    def _create_polygon_mask_from_vertices(self, vertices_pixels):
        """
        Create a binary mask from polygon vertices.
        :param vertices_pixels: List of (x, y) tuples in pixel coordinates
        :return: Binary mask tensor
        """
        # Create a grid of points
        yy, xx = np.meshgrid(np.arange(cfg.observation_side_pixels), 
                            np.arange(cfg.observation_side_pixels), 
                            indexing='ij')
        points = np.column_stack((xx.ravel(), yy.ravel()))

        # Mirror over the rows. Each point x should be subtracted from cfg.observation_side_pixels/2 and then mirrored.
        points[:, 0] = cfg.observation_side_pixels - points[:, 0]
        points[:, 1] = cfg.observation_side_pixels - points[:, 1]
        
        # Create path and check which points are inside
        path = MatplotlibPath(vertices_pixels)
        inside = path.contains_points(points).reshape(cfg.observation_side_pixels, cfg.observation_side_pixels)

        # MAY NEED TO ADD A SMALL BUFFER TO THE MASK.
        
        # Create mask
        mask = torch.zeros((cfg.observation_side_pixels, cfg.observation_side_pixels), device=cfg.device)
        mask[inside] = 1
        
        return mask

    def _generate_shape(self, shape_type=None):
        """
        Generate a binary shape, given a shape type.
        :param shape_type: the type of shape to generate. If none, choose randomly.
        :return: a tuple of (mask: torch.Tensor (OBS_PIX, OBS_PIX), 
                 object_details: dict of the object details.
        """
        # Initialize outputs.
        mask = torch.zeros((cfg.observation_side_pixels, cfg.observation_side_pixels), device=cfg.device)
        object_details = {}

        # Choose the shape type.
        if shape_type is None:
            shape_type = self.shape_types[random.randint(0, len(self.shape_types) - 1)]

        # Generate the shape first
        if shape_type == "rectangle":
            # Width and height in pixels.
            w_min_pixels = meters2pixel_dist(cfg.shape_size_range.rectangle.width.min_meters)
            w_max_pixels = meters2pixel_dist(cfg.shape_size_range.rectangle.width.max_meters)
            h_min_pixels = meters2pixel_dist(cfg.shape_size_range.rectangle.height.min_meters)
            h_max_pixels = meters2pixel_dist(cfg.shape_size_range.rectangle.height.max_meters)
            w, h = random.randint(w_min_pixels, w_max_pixels), \
                   random.randint(h_min_pixels, h_max_pixels)
            # Top left corner.
            x0, y0 = (cfg.observation_side_pixels - w) // 2, \
                     (cfg.observation_side_pixels - h) // 2
            mask[y0:y0+h, x0:x0+w] = 1
            object_details["width"] = w * cfg.OBS_M / cfg.OBS_PIX
            object_details["height"] = h * cfg.OBS_M / cfg.OBS_PIX

        # Triangle.
        elif shape_type == "triangle":  
            s_min_meters = cfg.shape_size_range.triangle.side_length.min_meters
            s_max_meters = cfg.shape_size_range.triangle.side_length.max_meters
            height_meters = random.uniform(s_min_meters, s_max_meters)
            width_meters = random.uniform(s_min_meters, s_max_meters)
            object_triangle_meters = ObjectTriangle(name="temp_triangle", width=width_meters, height=height_meters )
            object_details["vertices"] = object_triangle_meters.vertices
            obs_generator = ObservationGenerator()
            mask = obs_generator.get_object_centric_observation_polygon(object_triangle_meters, (cfg.observation_side_pixels, cfg.observation_side_pixels))
            mask = mask.to(cfg.device)
        # Half moon.
        elif shape_type == "half_moon":
            R_min_pixels = meters2pixel_dist(cfg.shape_size_range.half_moon.radius.min_meters)
            R_max_pixels = meters2pixel_dist(cfg.shape_size_range.half_moon.radius.max_meters)
            R = random.randint(R_min_pixels, R_max_pixels)
            cy = cx = cfg.observation_side_pixels // 2
            yy, xx = torch.meshgrid(torch.arange(cfg.observation_side_pixels), 
                                    torch.arange(cfg.observation_side_pixels), 
                                    indexing="ij")
            inside = (xx - cx) ** 2 + (yy - cy) ** 2 <= R ** 2
            mask[inside & (yy >= cy)] = 1
            object_details["radius"] = R * cfg.OBS_M / cfg.OBS_PIX

        # T shape.  
        elif shape_type == "t_shape":
            # First, generate the T-shape parameters in meters
            bar_w_min_meters = cfg.shape_size_range.t_shape.bar_width.min_meters
            bar_w_max_meters = cfg.shape_size_range.t_shape.bar_width.max_meters
            bar_w_meters = random.uniform(bar_w_min_meters, bar_w_max_meters)
            
            bar_h_min_meters = cfg.shape_size_range.t_shape.bar_height.min_meters
            bar_h_max_meters = cfg.shape_size_range.t_shape.bar_height.max_meters
            bar_h_meters = random.uniform(bar_h_min_meters, bar_h_max_meters)
            
            stem_w_min_meters = cfg.shape_size_range.t_shape.stem_width.min_meters
            stem_w_max_meters = cfg.shape_size_range.t_shape.stem_width.max_meters
            stem_w_meters = random.uniform(stem_w_min_meters, stem_w_max_meters)
            
            stem_h_min_meters = cfg.shape_size_range.t_shape.stem_height.min_meters
            stem_h_max_meters = cfg.shape_size_range.t_shape.stem_height.max_meters
            stem_h_meters = random.uniform(stem_h_min_meters, stem_h_max_meters)
            
            # Create the ObjectT object
            object_t = ObjectT("temp_t_shape", 
                               bar_width=bar_w_meters, 
                               bar_height=bar_h_meters, 
                               stem_width=stem_w_meters,
                               stem_height=stem_h_meters)
            
            # Get vertices from the object (in meters, centered at origin)
            vertices_meters = object_t.vertices
            
            # Create mask using the helper method
            obs_generator = ObservationGenerator()
            mask = obs_generator.get_object_centric_observation_polygon(object_t, (cfg.observation_side_pixels, cfg.observation_side_pixels))
            mask = mask.to(cfg.device)
            
            # Store object details
            object_details["bar_width"] = bar_w_meters
            object_details["stem_width"] = stem_w_meters
            object_details["bar_height"] = bar_h_meters
            object_details["stem_height"] = stem_h_meters
        
        elif shape_type == "circle":
            R_min_pixels = meters2pixel_dist(cfg.shape_size_range.circle.radius.min_meters)
            R_max_pixels = meters2pixel_dist(cfg.shape_size_range.circle.radius.max_meters)
            R = random.randint(R_min_pixels, R_max_pixels)
            cy = cx = cfg.observation_side_pixels // 2
            yy, xx = torch.meshgrid(torch.arange(cfg.observation_side_pixels), 
                                    torch.arange(cfg.observation_side_pixels), 
                                    indexing="ij")
            inside = (xx - cx) ** 2 + (yy - cy) ** 2 <= R ** 2
            mask[inside] = 1
            object_details["radius"] = R * cfg.OBS_M / cfg.OBS_PIX

        elif shape_type == "polygon":
            # Generate an arbitrary polygon with random number of vertices
            num_vertices = random.randint(cfg.shape_size_range.polygon.num_vertices.min_vertices, cfg.shape_size_range.polygon.num_vertices.max_vertices)  # Random polygon with 5-8 vertices
            radius_min_meters = cfg.shape_size_range.polygon.radius.min_meters
            radius_max_meters = cfg.shape_size_range.polygon.radius.max_meters
            radius = random.uniform(radius_min_meters, radius_max_meters)
            
            # Generate vertices in a roughly circular pattern with some randomness
            cx = cy = 0.0
            vertices = []
            for i in range(num_vertices):
                angle = 2 * math.pi * i / num_vertices
                # Add some randomness to the radius and angle
                r = (radius_min_meters + (radius_max_meters - radius_min_meters) * random.random())  # Radius varies between 70% and 130%
                angle_offset = (random.random() - 0.5) * 0.5  # Small random angle offset
                x = cx + r * math.cos(angle + angle_offset)
                y = cy + r * math.sin(angle + angle_offset)
                vertex_meters = torch.tensor([x, y])
                vertices.append(vertex_meters)
            vertices = torch.stack(vertices, dim=0)

            # Store polygon details
            object_details["vertices"] = vertices
            object_details["num_vertices"] = num_vertices
            object_details["radius"] = radius

            # Create mask using the helper method.
            object_polygon_meters = ObjectPolygon(name="temp_polygon", vertices=vertices)
            obs_generator = ObservationGenerator()
            mask = obs_generator.get_object_centric_observation_polygon(object_polygon_meters, (cfg.observation_side_pixels, cfg.observation_side_pixels))
            mask = mask.to(cfg.device)
            
        else:
            raise ValueError(f"Invalid shape type: {shape_type}. Must be 'rectangle', 'triangle', 'half_moon', 't_shape', 'circle', or 'polygon'.")
        object_details["shape_type"] = shape_type
        
        return mask, object_details
    
    def generate_shape_with_contacts(self, shape_type=None, num_contacts: int = 3, transform: Transform2 = None):
        """
        Generate a high-resolution binary shape and its contact points.
        :param shape_type: the type of shape to generate. If none, choose randomly.
        :param num_contacts: the number of contact points to generate (1, 2, or 3).
        :param transform: optional Transform2 object to determine contact point placement.
        :return: a tuple of (mask: torch.Tensor (OBS_PIX, OBS_PIX), 
                 contacts: torch.Tensor (num_contacts, 2)
                 object_details: dict of the object details.
        """
        object_details = {}

        if num_contacts not in [1, 2, 3]:  
            raise ValueError(f"Only 1, 2, or 3 contact points are supported. Got {num_contacts}.")
        
        mask, object_details = self._generate_shape(shape_type=shape_type)

        # Now generate contact points based on the transform and robot count
        contacts = self._generate_flexible_contacts_from_transform(mask, transform, num_contacts)
        if contacts is None:
            return None, None, None
        
        return mask, contacts, object_details

    def _generate_contacts_from_transform(self, mask: torch.Tensor, transform: Transform2) -> torch.Tensor:
        """
        Generate contact points based on transform direction.
        Place contacts a robot radius away from the object boundary with randomization.
        :param mask: Binary mask of the object
        :param transform: Transform2 object describing the motion
        :return: Contact points tensor (num_contacts, 2)
        """
        # This method is kept for backward compatibility but now delegates to the flexible version
        return self._generate_flexible_contacts_from_transform(mask, transform, 3)
    
    def _generate_flexible_contacts_from_transform(self, mask: torch.Tensor, transform: Transform2, num_robots: int) -> torch.Tensor:
        """
        Generate contact points based on transform direction and robot count.
        Uses different strategies for 1, 2, and 3 robots.
        :param mask: Binary mask of the object
        :param transform: Transform2 object describing the motion
        :param num_robots: Number of robots to use (1, 2, or 3)
        :return: Contact points tensor (num_robots, 2)
        """
        if transform is None:
            # Default placement: center of the image
            center = cfg.observation_side_pixels // 2
            return torch.tensor([[center, center]] * num_robots, device=cfg.device)
        
        # Get the translation vector
        dx, dy = transform.get_t().squeeze()
        theta = transform.get_theta().squeeze()
        
        # Find object bounds
        object_pixels = torch.where(mask == 1)
        if len(object_pixels[0]) == 0:
            # No object found, return center
            center = cfg.observation_side_pixels // 2
            raise ValueError(f"No object found in the image. Center: {center}, {center}, {center}")
        
        y_coords, x_coords = object_pixels
        min_x, max_x = x_coords.min().item(), x_coords.max().item()
        min_y, max_y = y_coords.min().item(), y_coords.max().item()
        
        # Calculate object width and height
        obj_width = max_x - min_x + 1
        obj_height = max_y - min_y + 1
        
        # Convert robot radius from meters to pixels
        robot_radius_pixels = meters2pixel_dist(cfg.robot_radius)
        
        # Try multiple attempts to find valid contact points
        max_attempts = 5
        for attempt in range(max_attempts):
            contacts = self._generate_flexible_contact_candidates(
                num_robots, dx, dy, theta, min_x, max_x, min_y, max_y, 
                obj_width, obj_height, robot_radius_pixels, mask
            )

            contacts = contacts.to(cfg.device, dtype=torch.float32)
            
            # Check if robots at these contacts would collide
            if not self._check_robot_collisions(contacts, robot_radius_pixels):
                return contacts
        
        # If we couldn't find non-colliding contacts, return the last attempt
        print(f"Warning: Could not find non-colliding contact points after {max_attempts} attempts")
        return None
    
    def _generate_contact_candidates_with_noise(self, strategy: str, min_x: int, max_x: int, 
                                              min_y: int, max_y: int, obj_width: int, obj_height: int,
                                              robot_radius_pixels: float, mask: torch.Tensor) -> torch.Tensor:
        """
        Generate contact point candidates with added noise to the library-based positions.
        :param strategy: Contact placement strategy (e.g., "all_from_front", "one_from_top")
        :param min_x, max_x, min_y, max_y: Object bounds
        :param obj_width, obj_height: Object dimensions
        :param robot_radius_pixels: Robot radius in pixels
        :param mask: Binary mask of the object
        :return: Contact points tensor (num_contacts, 2)
        """
        # Add noise factor (percentage of object size)
        # noise_factor = 0.25  # 15% of object dimension
        noise_factor = 0.0  # 15% of object dimension

        # With some probability, choose points on a circle and move them to the center of the object.
        # if random.random() < 0.99:
        #     # Choose points on a circle and move them to the center of the object.
        #     x_positions = [0.8 * np.cos(0.0), 0.8 * np.cos(math.pi/2), 0.8 * np.cos(math.pi)]  
        #     y_positions = [0.8 * np.sin(0.0), 0.8 * np.sin(math.pi/2), 0.8 * np.sin(math.pi)]
        #     step_vectors = [[-np.sign(x_positions[i]), -np.sign(y_positions[i])] for i in range(3)]
        
        if strategy == "all_from_front":
            # All three contacts from front (top edge) - add horizontal noise
            noise_range = int(obj_width * noise_factor)
            x_positions = [
                min_x + obj_width//4 + random.randint(-noise_range, noise_range),
                min_x + obj_width//2 + random.randint(-noise_range, noise_range),
                min_x + 3*obj_width//4 + random.randint(-noise_range, noise_range)
            ]
            y_positions = [0, 0, 0]  # Top edge of image
            step_vectors = [[0, 1], [0, 1], [0, 1]]  # All move down
            
        elif strategy == "all_from_back":
            # All three contacts from back (bottom edge) - add horizontal noise
            noise_range = int(obj_width * noise_factor)
            x_positions = [
                min_x + obj_width//4 + random.randint(-noise_range, noise_range),
                min_x + obj_width//2 + random.randint(-noise_range, noise_range),
                min_x + 3*obj_width//4 + random.randint(-noise_range, noise_range)
            ]
            y_positions = [cfg.observation_side_pixels - 1, cfg.observation_side_pixels - 1, cfg.observation_side_pixels - 1]
            step_vectors = [[0, -1], [0, -1], [0, -1]]  # All move up
            
        elif strategy == "all_from_left":
            # All three contacts from left (left edge) - add vertical noise
            noise_range = int(obj_height * noise_factor)
            x_positions = [0, 0, 0]  # Left edge of image
            y_positions = [
                min_y + obj_height//4 + random.randint(-noise_range, noise_range),
                min_y + obj_height//2 + random.randint(-noise_range, noise_range),
                min_y + 3*obj_height//4 + random.randint(-noise_range, noise_range)
            ]
            step_vectors = [[1, 0], [1, 0], [1, 0]]  # All move right
            
        elif strategy == "all_from_right":
            # All three contacts from right (right edge) - add vertical noise
            noise_range = int(obj_height * noise_factor)
            x_positions = [cfg.observation_side_pixels - 1, cfg.observation_side_pixels - 1, cfg.observation_side_pixels - 1]
            y_positions = [
                min_y + obj_height//4 + random.randint(-noise_range, noise_range),
                min_y + obj_height//2 + random.randint(-noise_range, noise_range),
                min_y + 3*obj_height//4 + random.randint(-noise_range, noise_range)
            ]
            step_vectors = [[-1, 0], [-1, 0], [-1, 0]]  # All move left
            
        elif strategy == "one_from_front":
            # Two from sides, one from front
            noise_range_x = int(obj_width * noise_factor)
            noise_range_y = int(obj_height * noise_factor)
            x_positions = [
                0 + random.randint(-0, 0),  # Left edge with small noise
                cfg.observation_side_pixels - 1 + random.randint(-0, 0),  # Right edge with small noise
                min_x + obj_width//2 + random.randint(-noise_range_x, noise_range_x)  # Front center with noise
            ]
            y_positions = [
                min_y + obj_height//2 + random.randint(-noise_range_y, noise_range_y),  # Middle height with noise
                min_y + obj_height//2 + random.randint(-noise_range_y, noise_range_y),  # Middle height with noise
                0 + random.randint(-0, 0)  # Front edge with small noise
            ]
            step_vectors = [[1, 0], [-1, 0], [0, 1]]  # Move right, left, down
        
        elif strategy == "one_from_front_rot_left":
            # Two from sides, one from front, rotated left
            noise_range_x = int(obj_width * noise_factor)
            noise_range_y = int(obj_height * noise_factor)
            x_positions = [
                0 + random.randint(-0, 0),  # Left edge with small noise
                cfg.observation_side_pixels - 1 + random.randint(-0, 0),  # Right edge with small noise
                min_x + obj_width//4 + random.randint(-noise_range_x, noise_range_x)  # Front center with noise
            ]
            y_positions = [
                min_y + 3 * obj_height//4 + random.randint(-noise_range_y, noise_range_y),  # Middle height with noise
                min_y + obj_height//4 + random.randint(-noise_range_y, noise_range_y),  # Middle height with noise
                0 + random.randint(-0, 0)  # Front edge with small noise
            ]
            step_vectors = [[1, 0], [-1, 0], [0, 1]]  # Move right, left, down

        elif strategy == "one_from_front_rot_right":
            # Two from sides, one from front, rotated right
            noise_range_x = int(obj_width * noise_factor)
            noise_range_y = int(obj_height * noise_factor)
            x_positions = [
                0 + random.randint(-0, 0),  # Left edge with small noise
                cfg.observation_side_pixels - 1 + random.randint(-0, 0),  # Right edge with small noise
                min_x + 3 * obj_width//4 + random.randint(-noise_range_x, noise_range_x)  # Front center with noise
            ]
            y_positions = [
                min_y + obj_height//4 + random.randint(-noise_range_y, noise_range_y),  # Middle height with noise
                min_y + 3 * obj_height//4 + random.randint(-noise_range_y, noise_range_y),  # Middle height with noise
                0 + random.randint(-0, 0)  # Front edge with small noise
            ]
            step_vectors = [[1, 0], [-1, 0], [0, 1]]  # Move right, left, down


        elif strategy == "one_from_back":
            # Two from sides, one from back
            noise_range_x = int(obj_width * noise_factor)
            noise_range_y = int(obj_height * noise_factor)
            x_positions = [
                0 + random.randint(-0, 0),  # Left edge with small noise
                cfg.observation_side_pixels - 1 + random.randint(-0, 0),  # Right edge with small noise
                min_x + obj_width//2 + random.randint(-noise_range_x, noise_range_x)  # Back center with noise
            ]
            y_positions = [
                min_y + obj_height//2 + random.randint(-noise_range_y, noise_range_y),  # Middle height with noise
                min_y + obj_height//2 + random.randint(-noise_range_y, noise_range_y),  # Middle height with noise
                cfg.observation_side_pixels - 1 + random.randint(-0, 0)  # Back edge with small noise
            ]
            step_vectors = [[1, 0], [-1, 0], [0, -1]]  # Move right, left, up
            
        elif strategy == "one_from_back_rot_left":
            # Two from sides, one from back, rotated left
            noise_range_x = int(obj_width * noise_factor)
            noise_range_y = int(obj_height * noise_factor)
            x_positions = [
                0 + random.randint(-0, 0),  # Left edge with small noise
                cfg.observation_side_pixels - 1 + random.randint(-0, 0),  # Right edge with small noise
                min_x + 3 * obj_width//4 + random.randint(-noise_range_x, noise_range_x)  # Back center with noise
            ]
            y_positions = [
                min_y + 3 * obj_height//4 + random.randint(-noise_range_y, noise_range_y),  # Middle height with noise
                min_y + obj_height//4 + random.randint(-noise_range_y, noise_range_y),  # Middle height with noise
                cfg.observation_side_pixels - 1 + random.randint(-0, 0)  # Back edge with small noise
            ]
            step_vectors = [[1, 0], [-1, 0], [0, -1]]  # Move right, left, up

        elif strategy == "one_from_back_rot_right":
            # Two from sides, one from back, rotated right
            noise_range_x = int(obj_width * noise_factor)
            noise_range_y = int(obj_height * noise_factor)
            x_positions = [
                0 + random.randint(-0, 0),  # Left edge with small noise
                cfg.observation_side_pixels - 1 + random.randint(-0, 0),  # Right edge with small noise
                min_x + obj_width//4 + random.randint(-noise_range_x, noise_range_x)  # Back center with noise
            ]
            y_positions = [
                min_y + obj_height//4 + random.randint(-noise_range_y, noise_range_y),  # Middle height with noise
                min_y + 3 * obj_height//4 + random.randint(-noise_range_y, noise_range_y),  # Middle height with noise
                cfg.observation_side_pixels - 1 + random.randint(-0, 0)  # Back edge with small noise
            ]
            step_vectors = [[1, 0], [-1, 0], [0, -1]]  # Move right, left, up

        elif strategy == "one_from_left":
            # Two from top/bottom, one from left
            noise_range_x = int(obj_width * noise_factor)
            noise_range_y = int(obj_height * noise_factor)
            x_positions = [
                0 + random.randint(-0, 0),  # Left edge with small noise
                min_x + obj_width//2 + random.randint(-noise_range_x, noise_range_x),  # Left center with noise
                min_x + obj_width//2 + random.randint(-noise_range_x, noise_range_x)  # Left center with noise
            ]
            y_positions = [
                min_y + obj_height//2 + random.randint(-noise_range_y, noise_range_y),  # Middle height with noise
                0 + random.randint(-0, 0),  # Top edge with small noise
                cfg.observation_side_pixels - 1 + random.randint(-0, 0)  # Bottom edge with small noise
            ]
            step_vectors = [[1, 0], [0, 1], [0, -1]] 

        elif strategy == "one_from_left_rot_left":
            # Two from top/bottom, one from left, rotated left
            noise_range_x = int(obj_width * noise_factor)
            noise_range_y = int(obj_height * noise_factor)
            x_positions = [
                0 + random.randint(-0, 0),  # Left edge with small noise
                min_x + obj_width//4 + random.randint(-noise_range_x, noise_range_x),  # Left center with noise
                min_x + 3 * obj_width//4 + random.randint(-noise_range_x, noise_range_x)  # Left center with noise
            ]
            y_positions = [
                min_y + 3 * obj_height//4 + random.randint(-noise_range_y, noise_range_y),  # Middle height with noise
                0 + random.randint(-0, 0),  # Top edge with small noise
                cfg.observation_side_pixels - 1 + random.randint(-0, 0)  # Bottom edge with small noise
            ]
            step_vectors = [[1, 0], [0, 1], [0, -1]]  # Move right, left, down

        elif strategy == "one_from_left_rot_right":
            # Two from top/bottom, one from left, rotated right
            noise_range_x = int(obj_width * noise_factor)
            noise_range_y = int(obj_height * noise_factor)
            x_positions = [
                0 + random.randint(-0, 0),  # Left edge with small noise
                min_x + 3 * obj_width//4 + random.randint(-noise_range_x, noise_range_x),  # Left center with noise
                min_x + obj_width//4 + random.randint(-noise_range_x, noise_range_x)  # Left center with noise
            ]
            y_positions = [
                min_y + obj_height//4 + random.randint(-noise_range_y, noise_range_y),  # Middle height with noise
                0 + random.randint(-0, 0),  # Top edge with small noise
                cfg.observation_side_pixels - 1 + random.randint(-0, 0)  # Bottom edge with small noise
            ]
            step_vectors = [[1, 0], [0, 1], [0, -1]]  # Move right, left, down
            
        elif strategy == "one_from_right":
            # Two from top/bottom, one from right
            noise_range_x = int(obj_width * noise_factor)
            noise_range_y = int(obj_height * noise_factor)
            x_positions = [
                cfg.observation_side_pixels - 1 + random.randint(-0, 0),  # Right edge with small noise
                min_x + obj_width//2 + random.randint(-noise_range_x, noise_range_x),  # Right center with noise
                min_x + obj_width//2 + random.randint(-noise_range_x, noise_range_x)  # Right center with noise
            ]
            y_positions = [
                min_y + obj_height//2 + random.randint(-noise_range_y, noise_range_y),  # Middle height with noise
                0 + random.randint(-0, 0),  # Top edge with small noise
                cfg.observation_side_pixels - 1 + random.randint(-0, 0)  # Bottom edge with small noise
            ]
            step_vectors = [[-1, 0], [0, 1], [0, -1]]  # All move left

        elif strategy == "one_from_right_rot_left":
            # Two from top/bottom, one from right, rotated left
            noise_range_x = int(obj_width * noise_factor)
            noise_range_y = int(obj_height * noise_factor)
            x_positions = [
                cfg.observation_side_pixels - 1 + random.randint(-0, 0),  # Right edge
                min_x + obj_width//4 + random.randint(-noise_range_x, noise_range_x),  # Top edge
                min_x + 3 * obj_width//4 + random.randint(-noise_range_x, noise_range_x)  # Bottom edge.
            ]
            y_positions = [
                min_y + obj_height//4 + random.randint(-noise_range_y, noise_range_y),  # Right edge
                0 + random.randint(-0, 0),  # Top edge
                cfg.observation_side_pixels - 1 + random.randint(-0, 0)  # Bottom edge
            ]
            step_vectors = [[-1, 0], [0, 1], [0, -1]]  # Move right, left, down
        
        elif strategy == "one_from_right_rot_right":
            # Two from top/bottom, one from right, rotated right
            noise_range_x = int(obj_width * noise_factor)
            noise_range_y = int(obj_height * noise_factor)
            x_positions = [
                cfg.observation_side_pixels - 1 + random.randint(-0, 0),  # Right edge with small noise
                min_x + 3 * obj_width//4 + random.randint(-noise_range_x, noise_range_x),  # Left center with noise
                min_x + obj_width//4 + random.randint(-noise_range_x, noise_range_x)  # Left center with noise
            ]
            y_positions = [
                min_y + 3 * obj_height//4 + random.randint(-noise_range_y, noise_range_y),  # Middle height with noise
                0 + random.randint(-0, 0),  # Top edge with small noise
                cfg.observation_side_pixels - 1 + random.randint(-0, 0)  # Bottom edge with small noise
            ]
            step_vectors = [[-1, 0], [0, 1], [0, -1]]  # Move right, left, down
            
        # elif strategy == "one_from_top":
        #     # Two from sides, one from top
        #     noise_range_x = int(obj_width * noise_factor)
        #     noise_range_y = int(obj_height * noise_factor)
        #     x_positions = [
        #         0 + random.randint(-0, 0),  # Left edge with small noise
        #         cfg.observation_side_pixels - 1 + random.randint(-0, 0),  # Right edge with small noise
        #         min_x + obj_width//2 + random.randint(-noise_range_x, noise_range_x)  # Top center with noise
        #     ]
        #     y_positions = [
        #         min_y + obj_height//2 + random.randint(-noise_range_y, noise_range_y),  # Middle height with noise
        #         min_y + obj_height//2 + random.randint(-noise_range_y, noise_range_y),  # Middle height with noise
        #         0 + random.randint(-0, 0)  # Top edge with small noise
        #     ]
        #     step_vectors = [[1, 0], [-1, 0], [0, 1]]  # Move right, left, down
            
        # elif strategy == "one_from_bottom":
        #     # Two from sides, one from bottom
        #     noise_range_x = int(obj_width * noise_factor)
        #     noise_range_y = int(obj_height * noise_factor)
        #     x_positions = [
        #         0 + random.randint(-0, 0),  # Left edge with small noise
        #         cfg.observation_side_pixels - 1 + random.randint(-0, 0),  # Right edge with small noise
        #         min_x + obj_width//2 + random.randint(-noise_range_x, noise_range_x)  # Bottom center with noise
        #     ]
        #     y_positions = [
        #         min_y + obj_height//2 + random.randint(-noise_range_y, noise_range_y),  # Middle height with noise
        #         min_y + obj_height//2 + random.randint(-noise_range_y, noise_range_y),  # Middle height with noise
        #         cfg.observation_side_pixels - 1 + random.randint(-0, 0)  # Bottom edge with small noise
        #     ]
        #     step_vectors = [[1, 0], [-1, 0], [0, -1]]  # Move right, left, up
            
        else:
            # # Random choice of strategy
            # center = cfg.observation_side_pixels // 2
            # noise_range = 10
            # return torch.tensor([
            #     [center + random.randint(-noise_range, noise_range), center + random.randint(-noise_range, noise_range)],
            #     [center + random.randint(-noise_range, noise_range), center + random.randint(-noise_range, noise_range)],
            #     [center + random.randint(-noise_range, noise_range), center + random.randint(-noise_range, noise_range)]
            # ], device=cfg.device)
            raise ValueError(f"Invalid strategy: {strategy}")
        
        # Move contacts inward until they are robot_radius away from the object
        contacts = []
        for i in range(3):
            x, y = x_positions[i], y_positions[i]
            step_vector = step_vectors[i]
            
            # Ensure we start within bounds
            x = max(0, min(x, cfg.observation_side_pixels - 1))
            y = max(0, min(y, cfg.observation_side_pixels - 1))
            
            # Sweep to boundary
            contact_x, contact_y = self._sweep_to_boundary(mask, x, y, step_vector, robot_radius_pixels)

            contacts.append([contact_y, contact_x])
        # If any of the contact coordinates is at zero, print the strategy.
        if torch.any(torch.tensor(contacts) == 0):
            print("Strategy:", strategy)
            print("Points start at", y_positions, x_positions)
            print("Points end at", contacts)
            print("Step vectors:", step_vectors)

        
        return torch.tensor(contacts, device=cfg.device)
    
    def _generate_flexible_contact_candidates(self, num_robots: int, dx: float, dy: float, theta: float,
                                            min_x: int, max_x: int, min_y: int, max_y: int,
                                            obj_width: int, obj_height: int,
                                            robot_radius_pixels: float, mask: torch.Tensor) -> torch.Tensor:
        """
        Generate contact point candidates based on robot count and transform.
        :param num_robots: Number of robots (1, 2, or 3)
        :param dx, dy: Translation components
        :param theta: Rotation component
        :param min_x, max_x, min_y, max_y: Object bounds
        :param obj_width, obj_height: Object dimensions
        :param robot_radius_pixels: Robot radius in pixels
        :param mask: Binary mask of the object
        :return: Contact points tensor (num_robots, 2)
        """
        # Determine motion type
        is_rotation = abs(theta) > 0.1 and abs(dx) < 0.05 and abs(dy) < 0.05
        is_translation = abs(theta) < 0.05
        
        if num_robots == 1:
            return self._generate_single_robot_contacts(dx, dy, theta, min_x, max_x, min_y, max_y, 
                                                     obj_width, obj_height, robot_radius_pixels, mask)
        elif num_robots == 2:
            return self._generate_two_robot_contacts(dx, dy, theta, min_x, max_x, min_y, max_y,
                                                   obj_width, obj_height, robot_radius_pixels, mask)
        else:  # num_robots == 3
            return self._generate_three_robot_contacts(dx, dy, theta, min_x, max_x, min_y, max_y,
                                                     obj_width, obj_height, robot_radius_pixels, mask)
    
    def _generate_single_robot_contacts(self, dx: float, dy: float, theta: float,
                                       min_x: int, max_x: int, min_y: int, max_y: int,
                                       obj_width: int, obj_height: int,
                                       robot_radius_pixels: float, mask: torch.Tensor) -> torch.Tensor:
        """
        Generate contact points for single robot pushing.
        Strategy: For pure translation, push from opposite direction. For rotations, use offset contacts
        to create torque while maintaining translation capability.
        """
        # Object center
        obj_center_x = (min_x + max_x) // 2  # Column (x-coordinate)
        obj_center_y = (min_y + max_y) // 2  # Row (y-coordinate)

        # Determine strategy based on motion characteristics
        if abs(theta) > 0.1 and abs(dx) < 0.05 and abs(dy) < 0.05:
            # Pure rotation - use offset contact to create torque
            if theta > 0:  # clockwise rotation
                # Offset contact to create clockwise torque
                if random.random() < 0.5:
                    # Push from left side, offset up to create clockwise torque
                    start_x = 0
                    start_y = obj_center_y - obj_height // 4  # offset up
                    step_vector = [1, 0]  # move right
                else:
                    # Push from right side, offset down to create clockwise torque
                    start_x = cfg.observation_side_pixels - 1
                    start_y = obj_center_y + obj_height // 4  # offset down
                    step_vector = [-1, 0]  # move left
            else:  # counter-clockwise rotation
                # Offset contact to create counter-clockwise torque
                if random.random() < 0.5:
                    # Push from left side, offset down to create counter-clockwise torque
                    start_x = 0
                    start_y = obj_center_y + obj_height // 4  # offset down
                    step_vector = [1, 0]  # move right
                else:
                    # Push from right side, offset up to create counter-clockwise torque
                    start_x = cfg.observation_side_pixels - 1
                    start_y = obj_center_y - obj_height // 4  # offset up
                    step_vector = [-1, 0]  # move left
        elif abs(theta) < 0.05:
            # Pure translation - push from opposite direction
            norm = torch.norm(torch.tensor([dx, dy]))
            if norm < 0.01:  # Very small translation, use default direction
                start_x = max(0, min_x - max(obj_width, obj_height))
                start_y = obj_center_y
                step_vector = [1, 0]  # Move right
            else:
                drow = -dx / norm
                dcol = -dy / norm

                # Get a point on the line from the center against the push direction.
                start_x = obj_center_x - dcol * 100
                start_y = obj_center_y - drow * 100

                # The step direction should be along the push direction.
                step_vector = [dcol, drow]
        else:
            # Mixed translation and rotation - combine both strategies
            norm = torch.norm(torch.tensor([dx, dy]))
            if norm < 0.001:  # Very small translation, use rotation strategy
                if theta > 0:  # clockwise rotation
                    start_x = 0
                    start_y = obj_center_y - obj_height // 4  # offset up
                    step_vector = [1, 0]  # move right
                else:  # counter-clockwise rotation
                    start_x = 0
                    start_y = obj_center_y + obj_height // 4  # offset down
                    step_vector = [1, 0]  # move right
            else:
                # Combine translation direction with rotation offset
                drow = -dx / norm
                dcol = -dy / norm

                # Get base translation direction
                base_x = obj_center_x - dcol * 100
                base_y = obj_center_y - drow * 100

                # Add rotation offset perpendicular to translation direction
                if theta > 0:  # clockwise rotation
                    # Offset perpendicular to push direction to create clockwise torque
                    offset_x = -drow * obj_width // 4  # perpendicular offset
                    offset_y = dcol * obj_height // 4
                else:  # counter-clockwise rotation
                    # Offset perpendicular to push direction to create counter-clockwise torque
                    offset_x = drow * obj_width // 4  # perpendicular offset
                    offset_y = -dcol * obj_height // 4

                start_x = base_x + offset_x
                start_y = base_y + offset_y
                step_vector = [dcol, drow]
        
        contact_x, contact_y = self._sweep_to_boundary(mask, start_x, start_y, step_vector, robot_radius_pixels)
        return torch.tensor([[contact_y, contact_x]], device=cfg.device)
    
    def _generate_two_robot_contacts(self, dx: float, dy: float, theta: float,
                                    min_x: int, max_x: int, min_y: int, max_y: int,
                                    obj_width: int, obj_height: int,
                                    robot_radius_pixels: float, mask: torch.Tensor) -> torch.Tensor:
        """
        Generate contact points for two robots.
        We employ a strategy-based approach akin to the three-robot variant, choosing the
        contact pattern according to the object motion (dx, dy, theta).  This yields a
        richer and more diverse set of placements than the previous simplistic rule.
        """
        # ----------------------------------------------
        # 1) Choose strategy based on transform
        # ----------------------------------------------
        all_strategies = [
            "both_from_front", "both_from_back", "both_from_left", "both_from_right",
            "front_and_back", "left_and_right",
            "front_and_left", "front_and_right", "back_and_left", "back_and_right",
            "back_and_left_rot_left", "back_and_left_rot_right",
            "front_and_right_rot_left", "front_and_right_rot_right",
            "front_and_left_rot_left", "front_and_left_rot_right",
            "back_and_right_rot_left", "back_and_right_rot_right",
            "left_and_right_rot_left", "left_and_right_rot_right",
            "front_and_back_rot_left", "front_and_back_rot_right"
        ]
        # Determine primary motion characteristics
        if abs(theta) > 0.1 and abs(dx) < 0.05 and abs(dy) < 0.05:
            # Predominantly rotational motion – place robots on opposite sides to create torque
            if theta > 0:
                # Clockwise rotation - choose randomly between opposing strategies
                strategy_transform = random.choice(["left_and_right_rot_left", "front_and_back_rot_left"])
            else:
                # Counter-clockwise rotation - choose randomly between opposing strategies
                strategy_transform = random.choice(["left_and_right_rot_right", "front_and_back_rot_right"])
        elif abs(theta) < 0.05:
            # Pure translation – push from the opposite direction of travel
            if abs(dx) > abs(dy):
                if dx > 0.05:  # object translates toward +x (towards front)
                    strategy_transform = random.choice(["both_from_back", "back_and_left", "back_and_right"])
                else:          # object translates toward –x (towards back)
                    strategy_transform = random.choice(["both_from_front", "front_and_left", "front_and_right"])
            else:
                if dy > 0.05:  # object translates toward +y (towards left)
                    strategy_transform = random.choice(["both_from_right", "front_and_right", "back_and_right"])
                else:          # object translates toward –y (towards right)
                    strategy_transform = random.choice(["both_from_left", "front_and_left", "back_and_left"])
        else:
            # Mixed translation & rotation – use rotation-aware strategies to avoid slipping
            strategy_front_back = "back" if dx > 0 else "front"
            strategy_left_right = "right" if dy > 0 else "left"
            strategy_rot = "rot_left" if theta > 0 else "rot_right"
            strategy_transform = f"{strategy_front_back}_and_{strategy_left_right}_{strategy_rot}"
            
        strategy = strategy_transform if random.random() < 1.0 else random.choice(all_strategies)

        # ----------------------------------------------
        # 2) Determine initial seed positions & step vectors
        # ----------------------------------------------
        # Noise is disabled for now (set to 0).  Can be re-enabled later if desired.
        noise_factor = 0.0

        def rand_int(r):
            return random.randint(-r, r) if r > 0 else 0

        positions = []  # (x, y, step_vec)

        if strategy == "both_from_front":
            noise = int(obj_width * noise_factor)
            xs = [min_x + obj_width // 3 + rand_int(noise), min_x + 2 * obj_width // 3 + rand_int(noise)]
            ys = [0, 0]
            steps = [[0, 1], [0, 1]]
            positions = list(zip(xs, ys, steps))

        elif strategy == "both_from_back":
            noise = int(obj_width * noise_factor)
            xs = [min_x + obj_width // 3 + rand_int(noise), min_x + 2 * obj_width // 3 + rand_int(noise)]
            ys = [cfg.observation_side_pixels - 1, cfg.observation_side_pixels - 1]
            steps = [[0, -1], [0, -1]]
            positions = list(zip(xs, ys, steps))

        elif strategy == "both_from_left":
            noise = int(obj_height * noise_factor)
            ys = [min_y + obj_height // 3 + rand_int(noise), min_y + 2 * obj_height // 3 + rand_int(noise)]
            xs = [0, 0]
            steps = [[1, 0], [1, 0]]
            positions = list(zip(xs, ys, steps))

        elif strategy == "both_from_right":
            noise = int(obj_height * noise_factor)
            ys = [min_y + obj_height // 3 + rand_int(noise), min_y + 2 * obj_height // 3 + rand_int(noise)]
            xs = [cfg.observation_side_pixels - 1, cfg.observation_side_pixels - 1]
            steps = [[-1, 0], [-1, 0]]
            positions = list(zip(xs, ys, steps))

        elif strategy == "front_and_back":
            xs = [min_x + obj_width // 2, min_x + obj_width // 2]
            ys = [0, cfg.observation_side_pixels - 1]
            steps = [[0, 1], [0, -1]]
            positions = list(zip(xs, ys, steps))

        elif strategy == "left_and_right":
            ys = [min_y + obj_height // 2, min_y + obj_height // 2]
            xs = [0, cfg.observation_side_pixels - 1]
            steps = [[1, 0], [-1, 0]]
            positions = list(zip(xs, ys, steps))

        elif strategy == "front_and_left":
            xs = [min_x + obj_width // 2, 0]
            ys = [0, min_y + obj_height // 2]
            steps = [[0, 1], [1, 0]]
            positions = list(zip(xs, ys, steps))

        elif strategy == "front_and_right":
            xs = [min_x + obj_width // 2, cfg.observation_side_pixels - 1]
            ys = [0, min_y + obj_height // 2]
            steps = [[0, 1], [-1, 0]]
            positions = list(zip(xs, ys, steps))

        elif strategy == "back_and_left":
            xs = [min_x + obj_width // 2, 0]
            ys = [cfg.observation_side_pixels - 1, min_y + obj_height // 2]
            steps = [[0, -1], [1, 0]]
            positions = list(zip(xs, ys, steps))

        elif strategy == "back_and_right":
            xs = [min_x + obj_width // 2, cfg.observation_side_pixels - 1]
            ys = [cfg.observation_side_pixels - 1, min_y + obj_height // 2]
            steps = [[0, -1], [-1, 0]]
            positions = list(zip(xs, ys, steps))

        # Rotation-aware strategies with offset contacts to create torque while avoiding slipping
        elif strategy == "back_and_left_rot_left":
            # Offset contacts to create clockwise torque while pushing from back-left
            xs = [min_x + 3 * obj_width // 4, 0]  # back contact offset left, left contact at edge
            ys = [cfg.observation_side_pixels - 1, min_y + 3 * obj_height // 4]  # back at edge, left contact offset down
            steps = [[0, -1], [1, 0]]
            positions = list(zip(xs, ys, steps))

        elif strategy == "back_and_left_rot_right":
            # Offset contacts to create counter-clockwise torque while pushing from back-left
            xs = [min_x + obj_width // 4, 0]  # back contact offset right, left contact at edge
            ys = [cfg.observation_side_pixels - 1, min_y + obj_height // 4]  # back at edge, left contact offset up
            steps = [[0, -1], [1, 0]]
            positions = list(zip(xs, ys, steps))

        elif strategy == "front_and_right_rot_left":
            # Offset contacts to create clockwise torque while pushing from front-right
            xs = [min_x + obj_width // 4, cfg.observation_side_pixels - 1]  # front contact offset left, right contact at edge
            ys = [0, min_y + obj_height // 4]  # front at edge, right contact offset up
            steps = [[0, 1], [-1, 0]]
            positions = list(zip(xs, ys, steps))

        elif strategy == "front_and_right_rot_right":
            # Offset contacts to create counter-clockwise torque while pushing from front-right
            xs = [min_x + 3 * obj_width // 4, cfg.observation_side_pixels - 1]  # front contact offset right, right contact at edge
            ys = [0, min_y + 3 * obj_height // 4]  # front at edge, right contact offset down
            steps = [[0, 1], [-1, 0]]
            positions = list(zip(xs, ys, steps))

        elif strategy == "front_and_left_rot_left":
            # Offset contacts to create clockwise torque while pushing from front-left
            xs = [min_x + obj_width // 4, 0]  # front contact offset right, left contact at edge
            ys = [0, min_y + 3 * obj_height // 4]  # front at edge, left contact offset down
            steps = [[0, 1], [1, 0]]
            positions = list(zip(xs, ys, steps))

        elif strategy == "front_and_left_rot_right":
            # Offset contacts to create counter-clockwise torque while pushing from front-left
            xs = [min_x + 3 * obj_width // 4, 0]  # front contact offset left, left contact at edge
            ys = [0, min_y + obj_height // 4]  # front at edge, left contact offset up
            steps = [[0, 1], [1, 0]]
            positions = list(zip(xs, ys, steps))

        elif strategy == "back_and_right_rot_left":
            # Offset contacts to create clockwise torque while pushing from back-right
            xs = [min_x + 3 * obj_width // 4, cfg.observation_side_pixels - 1]  # back contact offset right, right contact at edge
            ys = [cfg.observation_side_pixels - 1, min_y + obj_height // 4]  # back at edge, right contact offset up
            steps = [[0, -1], [-1, 0]]
            positions = list(zip(xs, ys, steps))

        elif strategy == "back_and_right_rot_right":
            # Offset contacts to create counter-clockwise torque while pushing from back-right
            xs = [min_x + obj_width // 4, cfg.observation_side_pixels - 1]  # back contact offset left, right contact at edge
            ys = [cfg.observation_side_pixels - 1, min_y + 3 * obj_height // 4]  # back at edge, right contact offset down
            steps = [[0, -1], [-1, 0]]
            positions = list(zip(xs, ys, steps))

        elif strategy == "left_and_right_rot_left":
            # Offset contacts to create clockwise torque while pushing from left and right
            xs = [0, cfg.observation_side_pixels - 1]  # left and right at edges
            ys = [min_y + 3 * obj_height // 4, min_y + obj_height // 4]  # left offset up, right offset down
            steps = [[1, 0], [-1, 0]]
            positions = list(zip(xs, ys, steps))

        elif strategy == "left_and_right_rot_right":
            # Offset contacts to create counter-clockwise torque while pushing from left and right
            xs = [0, cfg.observation_side_pixels - 1]  # left and right at edges
            ys = [min_y + obj_height // 4, min_y + 3 * obj_height // 4]  # left offset down, right offset up
            steps = [[1, 0], [-1, 0]]
            positions = list(zip(xs, ys, steps))

        elif strategy == "front_and_back_rot_left":
            # Offset contacts to create clockwise torque while pushing from front and back
            xs = [min_x + obj_width // 4, min_x + 3 * obj_width // 4]  # front offset left, back offset right
            ys = [0, cfg.observation_side_pixels - 1]  # front and back at edges
            steps = [[0, 1], [0, -1]]
            positions = list(zip(xs, ys, steps))

        elif strategy == "front_and_back_rot_right":
            # Offset contacts to create counter-clockwise torque while pushing from front and back
            xs = [min_x + 3 * obj_width // 4, min_x + obj_width // 4]  # front offset right, back offset left
            ys = [0, cfg.observation_side_pixels - 1]  # front and back at edges
            steps = [[0, 1], [0, -1]]
            positions = list(zip(xs, ys, steps))

        else:
            # Fallback to center-based opposing sides
            xs = [min_x + obj_width // 2, 0]
            ys = [0, min_y + obj_height // 2]
            steps = [[0, 1], [1, 0]]
            positions = list(zip(xs, ys, steps))

        # ----------------------------------------------
        # 3) Sweep each seed to boundary respecting robot radius
        # ----------------------------------------------
        contacts = []
        for x_seed, y_seed, step_vec in positions:
            contact_x, contact_y = self._sweep_to_boundary(mask, x_seed, y_seed, step_vec, robot_radius_pixels)
            contacts.append([contact_y, contact_x])  # (row, col)

        return torch.tensor(contacts, device=cfg.device)
    
    def _generate_three_robot_contacts(self, dx: float, dy: float, theta: float,
                                      min_x: int, max_x: int, min_y: int, max_y: int,
                                      obj_width: int, obj_height: int,
                                      robot_radius_pixels: float, mask: torch.Tensor) -> torch.Tensor:
        """
        Generate contact points for three robots.
        Strategy: Choose contact placement based on motion direction to optimize pushing effectiveness.
        """
        # Define contact strategies
        all_strategies = [
            "all_from_front", "all_from_back", "all_from_left", "all_from_right",
            "one_from_front", "one_from_back", "one_from_left", "one_from_right",
            "one_from_front", "one_from_back"
        ]

        
        # Determine the best strategy based on transform.
        if abs(theta) > 0.1 and abs(dx) < 0.05 and abs(dy) < 0.05:
            # Pure rotation
            if theta > 0.0:
                strategy_transform = random.choice(["one_from_front_rot_left", "one_from_back_rot_left", "one_from_left_rot_left", "one_from_right_rot_left"])
            else:
                strategy_transform = random.choice(["one_from_front_rot_right", "one_from_back_rot_right", "one_from_left_rot_right", "one_from_right_rot_right"])

        elif abs(theta) < 0.05:
            # Pure translation - choose between all from direction or two sides + one from direction
            if abs(dx) > abs(dy):
                # X motion dominates (back-to-front)
                if dx > 0.05:
                    # Moving front (positive x) - push from back
                    strategy_transform = random.choice(["all_from_back", "one_from_back"])
                else:
                    # Moving back (negative x) - push from front
                    strategy_transform = random.choice(["all_from_front", "one_from_front"])
            else:
                # Y motion dominates (right-to-left)
                if dy > 0.05:
                    # Moving left (positive y) - push from right
                    strategy_transform = random.choice(["all_from_right", "one_from_right"])
                else:
                    # Moving right (negative y) - push from left
                    strategy_transform = random.choice(["all_from_left", "one_from_left"])
        else:
            if theta > 0.0:
                strategy_rot = "rot_left"
            else:
                strategy_rot = "rot_right"
            # Combined motion - determine dominant component.
            if abs(dx) > abs(dy):
                # X translation dominates (back-to-front)
                strategy_transform = "one_from_back" if dx > 0.05 else "one_from_front"
            elif abs(dy) > abs(dx):
                # Y translation dominates (right-to-left)
                if dy > 0.05:
                    strategy_transform = "one_from_right"  # Moving left, push from right
                else:
                    strategy_transform = "one_from_left"   # Moving right, push from left
            else:
                # Rotation dominates
                strategy_transform = random.choice(["one_from_front", "one_from_back", "one_from_left", "one_from_right"])
            
            # Append the rotation strategy.
            strategy_transform += "_" + strategy_rot

        # Use the transform-appropriate strategy with high probability
        if random.random() < 1.0:
            strategy = strategy_transform
        else:
            strategy = random.choice(all_strategies)

        # Use the existing contact generation logic with the chosen strategy
        return self._generate_contact_candidates_with_noise(
            strategy, min_x, max_x, min_y, max_y, obj_width, obj_height, 
            robot_radius_pixels, mask
        )
    
    def _generate_trajectory_from_transform_interpolation(self, contacts: torch.Tensor,
                                                         transform: Transform2, 
                                                         num_steps: int) -> torch.Tensor:
        """
        Generate contact trajectory by interpolating object transformations.
        This creates more physically realistic trajectories where contacts maintain
        their relative positions to the object as it moves.
        
        :param contacts: Initial contact points (num_contacts, 2)
        :param transform: Final object transformation
        :param num_steps: Number of trajectory steps
        :return: Contact trajectory (num_contacts, num_steps, 2)
        """
        num_contacts = contacts.shape[0]
        trajectory = torch.zeros((num_contacts, num_steps, 2), device=cfg.device)
        
        # Get the final transform components
        (dx, dy), theta = transform.get_t(), transform.get_theta()
        
        # Generate intermediate transformations
        for i in range(num_steps):
            # Interpolate the transformation
            t_interp = i / (num_steps - 1) if num_steps > 1 else 1.0
            
            # Interpolate translation and rotation
            dx_interp = dx * t_interp
            dy_interp = dy * t_interp
            theta_interp = theta * t_interp
            
            # Create intermediate transform
            transform_interp = Transform2(
                t=torch.tensor([dx_interp, dy_interp], device=cfg.device),
                theta=torch.tensor([theta_interp], device=cfg.device)
            )
            
            # Apply intermediate transform to contacts
            contacts_interp = apply_transform_to_contacts(contacts.clone(), transform_interp)
            trajectory[:, i, :] = contacts_interp
        
        return trajectory
    
    def _create_circular_kernel(self, radius_pixels):
        """
        Create a circular kernel.
        :param radius_pixels: Radius of the kernel
        :return: Circular kernel
        """
        radius_pixels = int(radius_pixels)
        size = 2 * radius_pixels + 1
        kernel = torch.zeros((size, size))
        center = radius_pixels
        for x in range(size):
            for y in range(size):
                if (x - center)**2 + (y - center)**2 <= radius_pixels**2:
                    kernel[x, y] = 1
        return kernel

    def _expand_binary_image(self, image, radius):
        """
        Expand a binary image by a given radius.
        :param image: Binary image
        :param radius: Radius of the expansion
        :return: Expanded binary image
        """
        # Ensure the image is a binary tensor
        assert torch.unique(image).tolist() == [0, 1], "Input must be a binary image"
        
        # Create the circular kernel, if not already created.
        radius = int(radius)
        if self.mask_expand_kernel is None:
            self.mask_expand_kernel = self._create_circular_kernel(radius)
        
        # Add batch and channel dimensions to the image and kernel
        image = image.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, H, W)
        kernel = self.mask_expand_kernel.unsqueeze(0).unsqueeze(0).to(cfg.device)  # Shape: (1, 1, kernel_size, kernel_size)
        
        # Perform the convolution
        expanded = F.conv2d(image.float(), kernel.float(), padding=radius)
        
        # Threshold the result to get back a binary image
        expanded = (expanded > 0).float()
        
        # Remove the extra dimensions
        expanded = expanded.squeeze(0).squeeze(0)
        
        return expanded

    def _sweep_to_boundary(self, mask: torch.Tensor, start_x: int, start_y: int, 
                          step_vector: list, robot_radius_pixels: float) -> tuple:
        """
        Sweep from start point using step vector until hitting object boundary.
        :param mask: Binary mask of the object
        :param start_x: Starting x coordinate
        :param start_y: Starting y coordinate
        :param step_vector: [dx, dy] step vector
        :param robot_radius_pixels: Robot radius in pixels
        :return: (x, y) coordinates of contact point
        """
        x, y = start_x, start_y

        # this checks if the center hits the object and then backtracks. This is wrong, as we need to check on each point whether the robot intersects with the mask instead.
        # Create a mask where the robot radius pads the object.
        robot_radius_pixels = robot_radius_pixels
        mask_expanded = self._expand_binary_image(mask, robot_radius_pixels)
        
        # Move inward until we hit the object boundary
        boundary_found = False
        max_steps = cfg.observation_side_pixels  # Safety limit
        for step in range(max_steps):
            if 0 <= x < cfg.observation_side_pixels and 0 <= y < cfg.observation_side_pixels:
                if mask_expanded[int(y), int(x)] == 1:
                    # Found the object boundary, now move back by robot_radius
                    boundary_found = True
                    break
            
            # Move in step direction
            x += step_vector[0]
            y += step_vector[1]
        
        # if boundary_found:
        #     # Move back by a bit to stay away from the object boundary.
        #     x -= step_vector[0] * 1.1
        #     y -= step_vector[1] * 1.1
        
        # Ensure we're within bounds
        x = max(0, min(x, cfg.observation_side_pixels - 1))
        y = max(0, min(y, cfg.observation_side_pixels - 1))
        
        return x, y
    
    def _check_robot_collisions(self, contacts: torch.Tensor, robot_radius_pixels: float) -> bool:
        """
        Check if robots at the given contact points would collide with each other.
        :param contacts: Contact points tensor (num_contacts, 2)
        :param robot_radius_pixels: Robot radius in pixels
        :return: True if there are collisions, False otherwise
        """
        
        num_contacts = contacts.shape[0]
        min_distance = robot_radius_pixels * 2.1  # Minimum distance between robot centers
        
        for i in range(num_contacts):
            for j in range(i + 1, num_contacts):
                # Calculate distance between contact points
                dist = torch.norm(contacts[i] - contacts[j])
                if dist < min_distance:
                    return True  # Collision detected
        
        return False  # No collisions

    def generate_push_trajectory(self, mask: torch.Tensor,  # Binary mask (HIGH_RES, HIGH_RES), 1 is object, 0 is background.
                                contacts: torch.Tensor,  # Contact points (num_contacts, 2).
                                method: str,  # The method to choose the transformation. Could be one of "random", "primitives".
                                transform_object: Transform2 = None):  
        """
        Generate a push trajectory for the given mask and contacts.
        * Physically, all trajectories start at the contact points and move in sync. 
        * All returned trajectories are tanslated to start at the origin, and 
            normalized such that the maximal value for each of x and y is 1 (done by dividing by the observation side length).
        :param mask: Binary mask (HIGH_RES, HIGH_RES), 1 is object, 0 is background.
        :param contacts: Contact points (num_contacts, 2).
        :param method: The method to choose the transformation. Could be one of "random", "primitives".
        :param transform_object: Transform2 object describing the object motion.
        :return: A tuple of (trajectory: torch.Tensor (num_contacts, num_steps, 2), 
                 start_contact_points: torch.Tensor (num_steps, num_contacts, 2) contacts are in the image frame,
                 transform: Transform -- the resulting transform for the object, in meters). 
                     Canonically, the x axis for the object points up (negative rows of the mask), and y axis points to the left.
        """
        # Initialize the trajectory.
        num_contacts = contacts.shape[0]
        trajectory = torch.zeros((num_contacts, cfg.H, 2), device=cfg.device)

        if method == "primitives-simulated":
            # Select a primitive from the library.
            # For N trials,
            #    Choose random contact points on the boundary of the mask,
            #    Sweep the object to the transformation.
            #    Record the trajectory of the contact points.
            #    Simulate the trajectory in a physics simulator -- only moving the contacts.
            #    If the object is pushed to the transformation, record the trajectory and return.
            #    If the object is not pushed to the transformation, try again.
            #    If the object is not pushed to the transformation after N trials, give up.
            pass

        elif method == "primitives-linear":
            if transform_object is None:
                raise ValueError("transform_object is None")
            
            # Create a new transform in the image frame (but still in meters).
            (dx, dy), theta = transform_object.get_t(), transform_object.get_theta()
            transform = Transform2(t=torch.tensor([-dx, -dy], device=cfg.device), theta=theta)
            # Apply the transformation to the mask.
            mask_goal = apply_transform_to_mask(mask, transform)
            
            # Generate trajectory by interpolating object transformations
            start_contacts = contacts
            trajectory = self._generate_trajectory_from_transform_interpolation(
                contacts, transform, cfg.H
            )

        elif method == "random":
            if transform_object is None:
                # Choose a random transformation.
                transform_object = Transform2.random(-0.5, 0.5, -0.5, 0.5, -0.5, 0.5)

            # Create a new transform in the image frame (but still in meters).
            (dx, dy), theta = transform_object.get_t(), transform_object.get_theta()
            transform = Transform2(t=torch.tensor([-dx, -dy], device=cfg.device), theta=theta)

            # Apply the transformation to the mask.
            mask_goal = apply_transform_to_mask(mask, transform)
            
            # Generate trajectory by interpolating object transformations
            start_contacts = contacts
            trajectory = self._generate_trajectory_from_transform_interpolation(
                contacts, transform, cfg.H
            )
        
        else:
            raise ValueError(f"Invalid method: {method}")
        
        # Translate the trajectory to start at the origin.
        trajectory = trajectory - start_contacts.unsqueeze(1)
        # # Normalize the trajectory.
        # trajectory = trajectory / cfg.observation_side_pixels * 2 - 1
        # Return the trajectory, contacts, and transform.   

        return mask, mask_goal, trajectory, start_contacts, transform_object

    def __len__(self):
        return self.num_samples
    
    def generate_dataset_linear(self, save_intermediates=False) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Transform2]]:
        """
        Generate a dataset of push trajectories.
        :return: A dict of keys: mask, mask_goal, trajectory, start_contacts, transform_object.
        """
        datapoints = []
        pbar = tqdm(total=self.num_samples, desc="Generating dataset (linear)")
        
        # Create temporary directory for periodic saves
        import tempfile
        import os
        from datetime import datetime
        
        temp_dir = tempfile.mkdtemp(prefix=f"gco_datasets/gco_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}_")
        print(f"Temporary dataset directory: {temp_dir}")
        
        save_interval = max(1, self.num_samples // 10)  # Save every 10% of samples
        
        # Create balanced combinations of robot budgets and shape types
        combinations = []
        for robot_budget in self.robot_budgets:
            for shape_type in self.shape_types:
                combinations.append((robot_budget, shape_type))
        budget_shape_combo_ix_current = 0
        
        while len(datapoints) < self.num_samples:
            pbar.n = len(datapoints)
            pbar.refresh()

            # Use current combination (balanced sampling)
            num_robots, shape_type = combinations[budget_shape_combo_ix_current]
            
            # First, generate the transform
            if self.primitive_type == "primitives-linear":
                object_linear_primitives = cfg.motion_primitives.object_linear
                transforms_l = [(Transform2(t=torch.tensor([prim['dx'], prim['dy']], device=cfg.device), 
                                            theta=torch.tensor([prim['dtheta']], device=cfg.device)), name) 
                                            for name, prim in object_linear_primitives.items()]
                # Choose one randomly.
                transform_object, transform_name = transforms_l[random.randint(0, len(transforms_l) - 1)]
            elif self.primitive_type == "random":
                # Choose a random transformation.
                transform_object = Transform2.random(-0.2, 0.2, -0.2, 0.2, -0.5, 0.5)
            else:
                # Default to random if primitive_type is not recognized
                transform_object = Transform2.random(-0.2, 0.2, -0.2, 0.2, -0.5, 0.5)

            # If very very small in both the x, y, and theta dims, continue.
            if abs(transform_object.t[0]) < 0.02 and abs(transform_object.t[1]) < 0.02 and abs(transform_object.get_theta()) < 0.05:
                continue
            
            # Generate shape with contacts based on the transform and robot budget
            mask, contacts, object_details = self.generate_shape_with_contacts(shape_type=shape_type, num_contacts=num_robots, transform=transform_object)
            if contacts is None:
                continue
            
            # Corrupt the masks. With some small probability, push values of 1 to zero.
            mask_random = torch.rand_like(mask)
            mask = torch.where(mask_random < cfg.mask_corruption_prob, torch.zeros_like(mask), mask)
            
            # Generate the push trajectory
            mask, mask_goal, trajectory, start_contacts, _ = self.generate_push_trajectory(mask, contacts, method=self.primitive_type, transform_object=transform_object)
            
            # If contact points are colliding with each other, continue.
            robot_radius_pixels = meters2pixel_dist(cfg.robot_radius)
            if num_robots > 1 and self._check_robot_collisions(start_contacts, robot_radius_pixels):
                continue

            # Convert transform_object to a tensor.
            transform_object = transform_object.to_tensor()
            datapoint = {"mask": mask, 
                        "mask_goal": mask_goal, 
                        "trajectory": trajectory, 
                        "start_contacts": start_contacts, 
                        "transform_object": transform_object,
                        "num_robots": num_robots,  # Add robot budget information
                        "robot_budget": num_robots}  # For consistency with model interface
            datapoints.append(datapoint)
            budget_shape_combo_ix_current = (budget_shape_combo_ix_current + 1) % len(combinations)
            
            # Periodically save dataset and visualizations
            if len(datapoints) % save_interval == 0 and len(datapoints) > 0 and save_intermediates:
                self._save_intermediate_dataset(datapoints, temp_dir, len(datapoints))
        
        pbar.close()
        print(f"Final dataset saved to: file://{temp_dir}")
        return datapoints
    
    def generate_dataset_physics(self, visualize=False, save_intermediates=False) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Transform2]]:
        """
        Generate a dataset of push trajectories. This dataset is generated by randomly moving three robots in simulation and observing (a) the contact points, (b) the push trajectories, and (c) the end pose of the object.
        :return: A dict of keys: mask, mask_goal, trajectory, start_contacts, transform_object.
        """
        # Keep track of the samples. Datapoints are of form:
        # {
        #     "mask": mask,  # Binary mask (OBS_PIX, OBS_PIX), 1 is object, 0 is background.
        #     "mask_goal": mask_goal,  # Binary mask (OBS_PIX, OBS_PIX), 1 is object, 0 is background.
        #     "trajectory": trajectory,  # (N, H, 2) in pixel scale, all starting from origin and axis aligned.
        #     "start_contacts": start_contacts,  # (N, 2) in pixels.
        #     "transform_object": transform_object  # Transform2 object.
        # }
        datapoints = []
        pbar = tqdm(total=self.num_samples, desc="Generating dataset (physics)")
        
        # Create temporary directory for periodic saves
        import tempfile
        import os
        from datetime import datetime
        
        temp_dir = tempfile.mkdtemp(prefix=f"gco_datasets/gco_dataset_physics_{datetime.now().strftime('%Y%m%d_%H%M%S')}_")
        print(f"Temporary dataset directory: {temp_dir}")
        
        save_interval = max(1, self.num_samples // 10)  # Save every 10% of samples
        
        # Create balanced combinations of robot budgets and shape types
        combinations = []
        for robot_budget in self.robot_budgets:
            for shape_type in self.shape_types:
                combinations.append((robot_budget, shape_type))
        budget_shape_combo_ix_current = 0
        
        while len(datapoints) < self.num_samples:
            pbar.n = len(datapoints)
            pbar.refresh()
            # ====================
            # Generate a transform for the object. This is tentative and will be updated by the motion that the object actually takes.
            # ====================
            # transform_object = Transform2.random(-0.2, 0.2, -0.2, 0.2, -0.5, 0.5)
            # transform_object = Transform2(t=torch.tensor([0.5, 0.0]), theta=torch.tensor([0.0]))
            
            if self.primitive_type == "primitives-physics":
                object_linear_primitives = cfg.motion_primitives.object_linear
                transforms_l = [(Transform2(t=torch.tensor([prim['dx'], prim['dy']], device=cfg.device), 
                                            theta=torch.tensor([prim['dtheta']], device=cfg.device)), name) 
                                            for name, prim in object_linear_primitives.items()]
                # Choose one randomly.
                transform_object, transform_name = transforms_l[random.randint(0, len(transforms_l) - 1)]
            elif self.primitive_type == "random-physics":
                # Choose a random transformation.
                transform_object = Transform2.random(-0.2, 0.2, -0.2, 0.2, -0.5, 0.5)
                # transform_object = Transform2.random(-0.4, 0.4, -0.4, 0.4, -0.5, 0.5)
            else:
                raise ValueError(f"Invalid primitive type: {self.primitive_type}")
            
            # Skip very small transforms
            translation_magnitude = torch.norm(transform_object.get_t()).item()
            rotation_magnitude = abs(transform_object.get_theta().item())
            
            # Skip if translation is too small (< 0.01m) or rotation is too small (< 0.01 rad ≈ 0.57°)
            if translation_magnitude < 0.05 and rotation_magnitude < 0.02:
                print(f"Skipping this sample because the transform is too small. Translation magnitude: {translation_magnitude}, Rotation magnitude: {rotation_magnitude}")
                continue
            
            # ====================
            # Use current combination (balanced sampling)
            # ====================
            num_robots, shape_type = combinations[budget_shape_combo_ix_current]

            # Generate shape with contacts based on the transform and robot budget
            mask, contacts, object_details = self.generate_shape_with_contacts(shape_type=shape_type, num_contacts=num_robots, transform=transform_object)
            if contacts is None:
                continue
            
            # ====================
            # Generate world object for simulation with flexible robot count.
            # ====================
            robots = []
            robot_poses = []
            
            # Create robots based on budget
            for i in range(num_robots):
                robot = RobotDisk(f"robot_{i+1}", radius=cfg.robot_radius)
                robots.append(robot)

            # Create the object in the world.
            if object_details["shape_type"] == "rectangle":
                object_1 = ObjectRectangle("object_1", width=object_details["width"], height=object_details["height"])
            elif object_details["shape_type"] == "circle":
                object_1 = ObjectCircle("object_1", radius=object_details["radius"])
            elif object_details["shape_type"] == "t_shape":
                object_1 = ObjectT("object_1", 
                                   bar_width=object_details["bar_width"], 
                                   bar_height=object_details["bar_height"], 
                                   stem_width=object_details["stem_width"],
                                   stem_height=object_details["stem_height"])
                vertices = object_1.vertices
                object_1 = ObjectPolygon("object_1", vertices=vertices)
            elif object_details["shape_type"] == "polygon" or object_details["shape_type"] == "triangle":
                # Convert vertices from pixel coordinates to meter coordinates
                object_1 = ObjectPolygon("object_1", vertices=object_details["vertices"])
            else:
                raise ValueError(f"Invalid shape type: {object_details['shape_type']}")

            # Add the object to the world. At the origin and zero rotation.
            object_pose_init = Transform2(t=torch.tensor([0.0, 0.0]), theta=torch.tensor([0.0]))

            # ====================
            # Generate a tentative trajectory for the contact points.
            # ====================
            # Get initial trajectories for the robots. Get linear trajectories for the robots.
            # The trajectory from the dataset is in pixels and starting at the origin [0, 0].
            # The start_contacts are (num_robots, 2) in pixels in the image frame.
            mask, mask_goal_linear, trajectory, start_contacts, _ = self.generate_push_trajectory(mask, 
                                                                                   contacts, 
                                                                                   method="primitives-linear", 
                                                                                   transform_object=transform_object)
            

            # If contact points are colliding with each other, continue.
            robot_radius_pixels = meters2pixel_dist(cfg.robot_radius)
            if num_robots > 1 and self._check_robot_collisions(start_contacts, robot_radius_pixels):
                continue

            # ====================
            # Transform contacts to the world frame. Place robots there.
            # ====================
            # Be explicit about the object starting at the origin with zero rotation.
            X_world_object = Transform2(t=torch.tensor([0.0, 0.0]), theta=torch.tensor([0.0]))
            
            # Pad contacts to always have 3 robots for consistency with existing functions
            start_contacts_padded = torch.zeros((3, 2), device=start_contacts.device)
            start_contacts_padded[:num_robots] = start_contacts
            
            contact_points_tokens = start_contacts_padded.reshape(-1, 6) # (B, N, 2) -> (B, 6), B=1 here.
            contact_points_meters_local = tokens_to_meters_local(contact_points_tokens)
            contact_points_meters_world = points_local_to_world(contact_points_meters_local.squeeze(), X_world_object)
            
            # Add the robots to the world based on budget
            robots_with_pose_init = {}
            for i in range(num_robots):
                robot_pose_init = Transform2(t=contact_points_meters_world[i], theta=torch.tensor([0.0]))
                robots_with_pose_init[robots[i]] = robot_pose_init

            # Pad trajectory to always have 3 robots for consistency
            trajectory_padded = torch.zeros((3, trajectory.shape[1], 2), device=trajectory.device)
            trajectory_padded[:num_robots] = trajectory
            
            push_trajectories_pixels_local = trajectory_padded.unsqueeze(0)  # Pixel scale but with origin in center. (B, N, H, 2)
            push_trajectories_meters_local = push_trajectories_pixels_local_to_meters_local(push_trajectories_pixels_local, contact_points_meters_local)
            push_trajectories_meters_world = points_local_to_world(push_trajectories_meters_local, X_world_object)
            # Add zeros as last element (theta) to make it (B, N, H, 3).
            B, N, H, _ = push_trajectories_meters_world.shape
            push_points = push_trajectories_meters_world.reshape(-1, 2)
            push_points = torch.cat((push_points, torch.zeros((push_points.shape[0], 1), device=cfg.device)), dim=-1)
            push_trajectories_meters_world = push_points.reshape(B, N, H, 3)

            # Finalize the world.
            world = World(size=(1.0, 1.0), resolution=0.05, dt=self.dt,
                          robots_with_pose_init=robots_with_pose_init,
                          objects_with_pose_init={object_1: object_pose_init},
                          launch_viewer=visualize)

            # ====================
            # Execute the trajectory in the world.
            # ====================
            traj_dict = {f"robot_{i+1}": push_trajectories_meters_world[:, i].squeeze().tolist() for i in range(num_robots)}

            # Transform and densify the trajectories.
            traj_dict = smooth_all_trajectories(traj_dict, 5)
            traj_dict = {k: v.tolist() for k, v in traj_dict.items()}
            traj_dict = densify_all_trajectories(traj_dict, 50)
                
            traj_dict = {robot_name: torch.tensor(path)[:, :2] for robot_name, path in traj_dict.items()}

            # ====================
            # Apply the trajectories.
            # ====================
            world_states = world.apply_trajectories(traj_dict, visualize=visualize, real_time=visualize)
            
            # Always cleanup the world immediately after use
            world.cleanup_persistent_simulator()
            del world
            import gc
            gc.collect()
            
            # Get the final state of the object.
            transform_object_final = world_states[-1]["objects"]["object_1"]
            # Compare with requested.
            # Check if the final transform is close to the requested transform.
            if torch.norm(transform_object_final.get_t() - transform_object.get_t()) > 0.1 or torch.norm(transform_object_final.get_theta() - transform_object.get_theta()) > 0.5:
                dist_t = torch.norm(transform_object_final.get_t() - transform_object.get_t()).item()
                dist_theta = torch.norm(transform_object_final.get_theta() - transform_object.get_theta()).item()
                # print(RED, f"Skipping this sample because the final transform is not close to the requested transform. Dist in t: {format(dist_t, '.2f')} and dist in theta: {format(dist_theta, '.2f')}, requested transform: {transform_object.get_t()} {transform_object.get_theta()}, final transform: {transform_object_final.get_t()} {transform_object_final.get_theta()}", RESET)
                continue
            
            # ====================
            # Save the datapoint.
            # ====================
            # Down sample the trajectory. This entails keeping H steps, equally spaced, from the world state.
            H = cfg.H
            idx_selected = torch.arange(0, len(world_states), len(world_states)//H)
            trajectory_downsampled = torch.zeros((num_robots, H, 2), device=cfg.device)
            for i in range(num_robots):
                for idx in range(H):
                    trajectory_downsampled[i, idx] = world_states[idx_selected[idx]]["robots"][f"robot_{i+1}"].t
            trajectories_meters_world = trajectory_downsampled
            trajectories_meters_local = points_world_to_local(trajectories_meters_world, X_world_object)
            
            # Fix shape mismatch: contact_points_meters_local already has batch dim [1, N, 2]
            # We need to select only the first num_robots and keep the batch dimension
            contact_points_subset = contact_points_meters_local[0, :num_robots].unsqueeze(0)  # [1, num_robots, 2]
            
            trajectories_pixels_local = push_trajectories_meters_local_to_pixels_local(trajectories_meters_local.unsqueeze(0), contact_points_subset)
            # Add zero theta to make it (N, H, 3).
            B, N_actual, H, _ = trajectories_pixels_local.shape
            trajectories_pixels_local = trajectories_pixels_local.reshape(-1, 2)
            trajectories_pixels_local = trajectories_pixels_local.reshape(B, N_actual, H, 2).squeeze()

            # Get the observed object transformation. Convert it to a tensor.
            transform_object_observed = world_states[-1]["objects"]["object_1"]
            # Convert to object centric frame.
            # Create a new transform in the image frame (but still in meters).
            (dx, dy), theta = transform_object_observed.get_t(), transform_object_observed.get_theta()
            transform_mask = Transform2(t=torch.tensor([-dx, -dy], device=cfg.device), theta=theta)
            # Apply the transformation to the mask.
            mask_goal = apply_transform_to_mask(mask, transform_mask)

            transform_object_observed = transform_object_observed.to_tensor()
            
            # Create datapoint
            datapoint = {
                "mask": mask,
                "mask_goal": mask_goal,
                "trajectory": trajectories_pixels_local,
                "start_contacts": start_contacts,
                "transform_object": transform_object_observed,
                "num_robots": num_robots,  # Add robot budget information
                "robot_budget": num_robots  # For consistency with model interface
            }
            
            datapoints.append(datapoint)
            budget_shape_combo_ix_current = (budget_shape_combo_ix_current + 1) % len(combinations)
            
            # Periodically save dataset and visualizations
            if len(datapoints) % save_interval == 0 and len(datapoints) > 0 and save_intermediates:
                self._save_intermediate_dataset(datapoints, temp_dir, len(datapoints))
            
            # # Corrupt the masks. With some small probability, push values of 1 to zero.
            # mask_random = torch.rand_like(mask)
            # mask = torch.where(mask_random < cfg.mask_corruption_prob, torch.zeros_like(mask), mask)
            
            # # Generate the push trajectory
            # mask, mask_goal, trajectory, start_contacts, _ = self.generate_push_trajectory(mask, contacts, method=self.primitive_type, transform_object=transform_object)
            
            # # Convert transform_object to a tensor.
            # transform_object = transform_object.to_tensor()
            # datapoints.append({"mask": mask, 
            #                    "mask_goal": mask_goal, 
            #                    "trajectory": trajectory, 
            #                    "start_contacts": start_contacts, 
            #                    "transform_object": transform_object})
        
        pbar.close()
        print(f"Final dataset saved to: file://{temp_dir}")
        return datapoints
    
    def _save_intermediate_dataset(self, datapoints: list, temp_dir: str, num_samples: int):
        """
        Save intermediate dataset and visualizations to temporary directory.
        :param datapoints: List of datapoints to save
        :param temp_dir: Temporary directory to save to
        :param num_samples: Number of samples in the dataset
        """
        import os
        import json
        import pickle
        from datetime import datetime
        
        # Create subdirectory for this save
        timestamp = datetime.now().strftime('%H%M%S')
        save_dir = os.path.join(temp_dir, f"checkpoint_{num_samples:06d}_{timestamp}")
        os.makedirs(save_dir, exist_ok=True)
        
        # Save metadata
        metadata = {
            "num_samples": num_samples,
            "primitive_type": self.primitive_type,
            "shape_types": self.shape_types,
            "robot_budgets": self.robot_budgets if hasattr(self, 'robot_budgets') else [cfg.N],
            "device": str(cfg.device),
            "observation_side_pixels": cfg.observation_side_pixels,
            "H": cfg.H,
            "N": cfg.N,
            "timestamp": timestamp,
            "total_samples_requested": self.num_samples
        }
        
        metadata_path = os.path.join(save_dir, "contact_trajectory_dataset_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save datapoints
        datapoints_path = os.path.join(save_dir, "contact_trajectory_dataset_datapoints.pkl")
        
        # Convert tensors to CPU for saving
        cpu_datapoints = []
        for datapoint in datapoints:
            cpu_datapoint = {}
            for key, value in datapoint.items():
                if isinstance(value, torch.Tensor):
                    cpu_datapoint[key] = value.cpu()
                else:
                    cpu_datapoint[key] = value
            cpu_datapoints.append(cpu_datapoint)
        
        with open(datapoints_path, 'wb') as f:
            pickle.dump(cpu_datapoints, f)
        
        # Save sample visualizations
        viz_dir = os.path.join(save_dir, "visualizations")
        os.makedirs(viz_dir, exist_ok=True)
        
        # Create a temporary dataset object for visualization
        temp_dataset = ContactTrajectoryDataset.__new__(ContactTrajectoryDataset)
        temp_dataset.num_samples = num_samples
        temp_dataset.primitive_type = self.primitive_type
        temp_dataset.shape_types = self.shape_types
        temp_dataset.datapoints = datapoints
        
        # Save visualizations of a few samples
        num_viz_samples = min(5, num_samples)
        temp_dataset.save_sample_visualizations(viz_dir, num_viz_samples, "sample")
        
        # Save dataset statistics
        stats = temp_dataset.get_dataset_stats()
        stats_path = os.path.join(save_dir, "stats.json")
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        # print(f"  Saved intermediate dataset: {num_samples} samples to {save_dir}")

    def __getitem__(self, idx):
        return self.datapoints[idx]

    def save_dataset(self, save_dir: str, filename_prefix: str = "contact_trajectory_dataset", include_visualization: bool = False):
        """
        Save the dataset to disk.
        :param save_dir: Directory to save the dataset files
        :param filename_prefix: Prefix for the saved files
        :param include_visualization: Whether to include the visualization of the dataset in the saved files
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # Save metadata
        metadata = {
            "num_samples": self.num_samples,
            "primitive_type": self.primitive_type,
            "shape_types": self.shape_types,
            "robot_budgets": self.robot_budgets if hasattr(self, 'robot_budgets') else [cfg.N],
            "device": str(cfg.device),
            "observation_side_pixels": cfg.observation_side_pixels,
            "H": cfg.H,
            "N": cfg.N
        }
        
        metadata_path = os.path.join(save_dir, f"{filename_prefix}_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save datapoints
        datapoints_path = os.path.join(save_dir, f"{filename_prefix}_datapoints.pkl")
        
        # Convert tensors to CPU for saving
        cpu_datapoints = []
        for datapoint in self.datapoints:
            cpu_datapoint = {}
            for key, value in datapoint.items():
                if isinstance(value, torch.Tensor):
                    cpu_datapoint[key] = value.cpu()
                else:
                    cpu_datapoint[key] = value
            cpu_datapoints.append(cpu_datapoint)
        
        with open(datapoints_path, 'wb') as f:
            pickle.dump(cpu_datapoints, f)
        
        print(f"Dataset saved to {save_dir}")
        print(f"  Metadata: {metadata_path}")
        print(f"  Datapoints: {datapoints_path}")
        print(f"  Total samples: {len(self.datapoints)}")


    @classmethod
    def load_dataset(cls, save_dir: str, filename_prefix: str = "contact_trajectory_dataset", device: str = None):
        """
        Load a dataset from disk.
        :param save_dir: Directory containing the dataset files
        :param filename_prefix: Prefix for the saved files
        :param device: Device to load tensors on (if None, uses cfg.device)
        :return: ContactTrajectoryDataset instance
        """
        # Load metadata
        metadata_path = os.path.join(save_dir, f"{filename_prefix}_metadata.json")
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Load datapoints
        datapoints_path = os.path.join(save_dir, f"{filename_prefix}_datapoints.pkl")
        with open(datapoints_path, 'rb') as f:
            cpu_datapoints = pickle.load(f)
        
        # Convert tensors to the specified device and pad to cfg.N robots
        target_device = device if device is not None else cfg.device
        datapoints = []
        for datapoint in tqdm(cpu_datapoints, desc="Loading datapoints"):
            device_datapoint = {}
            for key, value in datapoint.items():
                if isinstance(value, torch.Tensor):
                    device_datapoint[key] = value.to(target_device)
                else:
                    device_datapoint[key] = value
            
            # Pad datapoint to have cfg.N robots if needed
            device_datapoint = cls._pad_datapoint_to_n_robots(device_datapoint, target_device)
            datapoints.append(device_datapoint)
        
        # Create dataset instance
        dataset = cls.__new__(cls)
        dataset.num_samples = metadata["num_samples"]
        dataset.primitive_type = metadata["primitive_type"]
        dataset.shape_types = metadata["shape_types"]
        dataset.robot_budgets = metadata.get("robot_budgets", [cfg.N])  # Backward compatibility
        dataset.datapoints = datapoints

        # Get dataset stats.
        stats = dataset.get_dataset_stats()
        
        print(f"Dataset loaded from {save_dir}")
        print(f"  Total samples: {len(datapoints)}")
        print(f"  Dataset stats:")
        for key, value in stats.items():
            print(f"    {key}: {value}")
        

        
        return dataset
    
    @classmethod
    def _pad_datapoint_to_n_robots(cls, datapoint: dict, device: str) -> dict:
        """
        Pad a datapoint to have cfg.N robots by adding zero trajectories and mask tokens for extra robots.
        :param datapoint: The datapoint to pad
        :param device: Device to place tensors on
        :return: Padded datapoint
        """
        # Get the number of robots in this datapoint
        num_robots = datapoint.get("num_robots", datapoint.get("robot_budget", cfg.N))
        
        if num_robots >= cfg.N:
            # Already has enough robots, just ensure robot_budget field exists
            if "robot_budget" not in datapoint:
                datapoint["robot_budget"] = num_robots
            return datapoint
        
        # Need to pad to cfg.N robots
        mask_token = cfg.mask_token
        
        # Pad start_contacts
        start_contacts = datapoint["start_contacts"]
        if start_contacts.shape[0] < cfg.N:
            padded_contacts = torch.zeros((cfg.N, 2), device=device)
            padded_contacts[:num_robots] = start_contacts
            # Fill extra contacts with mask token positions (bottom right outside of image)
            padded_contacts[num_robots:] = torch.tensor([[cfg.pixel_value_mask, cfg.pixel_value_mask]] * (cfg.N - num_robots), device=device)
            datapoint["start_contacts"] = padded_contacts
        
        # Pad trajectory
        trajectory = datapoint["trajectory"]
        # Get the N, H, 2 shape. If there is only one robot then it'll be the (H, 2) shape and we'll infer N = 1.
        if len(trajectory.shape) == 2:
            N = 1
            H = trajectory.shape[0]
        else:
            N = trajectory.shape[0]
            H = trajectory.shape[1]

        if N < cfg.N:
            padded_trajectory = torch.zeros((cfg.N, H, 2), device=device)
            padded_trajectory[:num_robots] = trajectory
            # Extra robots have zero trajectories (they don't move)
            datapoint["trajectory"] = padded_trajectory
        
        # Update robot budget information
        datapoint["num_robots"] = num_robots
        datapoint["robot_budget"] = num_robots
        
        return datapoint
    
    def get_contact_tokens_with_mask(self, start_contacts: torch.Tensor, robot_budget: int) -> torch.Tensor:
        """
        Convert contact points to tokens, filling unused robots with mask tokens.
        :param start_contacts: Contact points tensor (N, 2) where N may be less than cfg.N
        :param robot_budget: Number of robots actually used
        :return: Contact tokens tensor (cfg.N * 2) with mask tokens for unused robots
        """
        mask_token = cfg.mask_token
        
        # Pad contacts to cfg.N if needed
        if start_contacts.shape[0] < cfg.N:
            padded_contacts = torch.zeros((cfg.N, 2), device=start_contacts.device)
            padded_contacts[:robot_budget] = start_contacts
            # Fill extra contacts with mask token positions (bottom right outside of image)
            padded_contacts[robot_budget:] = torch.tensor([[cfg.pixel_value_mask, cfg.pixel_value_mask]] * (cfg.N - robot_budget), device=start_contacts.device)
            start_contacts = padded_contacts
        
        # Convert to tokens
        contact_tokens = pixels_to_tokens(start_contacts.unsqueeze(0)).squeeze(0)  # (cfg.N * 2)
        
        # Replace tokens for unused robots with mask tokens
        for i in range(robot_budget, cfg.N):
            contact_tokens[i*2] = mask_token     # x coordinate
            contact_tokens[i*2+1] = mask_token   # y coordinate
        
        return contact_tokens

    def save_sample_visualizations(self, save_dir: str, num_samples: int = 10, filename_prefix: str = "sample"):
        """
        Save visualizations of random samples from the dataset.
        :param save_dir: Directory to save the visualizations
        :param num_samples: Number of samples to visualize
        :param filename_prefix: Prefix for the saved image files
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # Select random samples
        indices = random.sample(range(len(self.datapoints)), min(num_samples, len(self.datapoints)))
        
        for i, idx in enumerate(indices):
            datapoint = self.datapoints[idx]
            
            # Create visualization
            fig, ax = plt.subplots(1, 1, figsize=(8, 8))
            
            # Get data
            mask = datapoint["mask"].cpu()
            mask_goal = datapoint["mask_goal"].cpu()
            trajectory = datapoint["trajectory"].cpu()
            start_contacts = datapoint["start_contacts"].cpu()
            transform_object = datapoint["transform_object"].cpu()
            
            # Convert transform back to Transform2 object
            transform = Transform2(transform_object[:2], transform_object[2])
            
            # Visualize
            visualize_push_trajectory(
                mask, trajectory, start_contacts, transform, mask_goal=mask_goal, 
                save_path=None, ax=ax
            )
            
            # Save
            filename = f"{filename_prefix}_{i:03d}.png"
            filepath = os.path.join(save_dir, filename)
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            plt.close()
                    
        print("\n", f"Saved {len(indices)} sample visualizations to {save_dir}")

    def get_dataset_stats(self):
        """
        Get statistics about the dataset.
        :return: Dictionary containing dataset statistics
        """
        if not self.datapoints:
            return {"error": "No datapoints available"}
        
        # Count shape types
        shape_counts = {}
        transform_types = {}
        robot_budget_counts = {}
        
        for datapoint in self.datapoints:
            # Count shape types (infer from mask)
            mask = datapoint["mask"]
            # This is a simplified approach - in practice you might want to store shape type explicitly
            shape_counts["total"] = shape_counts.get("total", 0) + 1
            
            # Count robot budgets
            robot_budget = datapoint.get("robot_budget", datapoint.get("num_robots", cfg.N))
            robot_budget_counts[robot_budget] = robot_budget_counts.get(robot_budget, 0) + 1
            
            # Count transform types
            transform = datapoint["transform_object"]
            dx, dy, theta = transform[0], transform[1], transform[2]
            
            if abs(theta) > 0.1:
                if theta > 0:
                    transform_type = "rotation_ccw"
                else:
                    transform_type = "rotation_cw"
            else:
                if abs(dx) > abs(dy):
                    if dx > 0:
                        transform_type = "translation_forward"
                    else:
                        transform_type = "translation_backward"
                else:
                    if dy > 0:
                        transform_type = "translation_left"
                    else:
                        transform_type = "translation_right"
            
            transform_types[transform_type] = transform_types.get(transform_type, 0) + 1
        
        stats = {
            "total_samples": len(self.datapoints),
            "shape_counts": shape_counts,
            "transform_types": transform_types,
            "robot_budget_counts": robot_budget_counts,
            "primitive_type": self.primitive_type,
            "shape_types": self.shape_types,
            "robot_budgets": self.robot_budgets if hasattr(self, 'robot_budgets') else [cfg.N]
        }
        
        return stats