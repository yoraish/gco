# Imports for flow-matching.
import time
import torch
import random
from dataclasses import dataclass
from torch import nn, Tensor
from torch.utils.data import DataLoader
import math
from datetime import datetime
import os
import json
import pickle
# flow_matching
from flow_matching.path import MixtureDiscreteProbPath
from flow_matching.path.scheduler import PolynomialConvexScheduler
from flow_matching.solver import MixtureDiscreteEulerSolver
from flow_matching.utils import ModelWrapper, ModelWrapperCoGen
from flow_matching.loss import MixturePathGeneralizedKL
from flow_matching.path.scheduler import CondOTScheduler
from flow_matching.path import AffineProbPath
from flow_matching.solver import Solver, ODESolver, CoGenSolver

# visualization
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from tqdm import tqdm

# Imports for GCo.
from gco.config import Config as cfg
from gco.datasets import ContactTrajectoryDataset
from gco.utils.transform_utils import *
from gco.models.contact_push_model import ContactModel, TrajectoryModel, ContactTrajectoryModel
from gco.utils.model_utils import *
from gco.world.world import World
from gco.world.robot import RobotDisk
from gco.world.objects import ObjectRectangle, ObjectCircle, ObjectT, ObjectPolygon
from gco.utils.viz_utils import smooth_all_trajectories, densify_all_trajectories

# Device is set in cfg.device.
print("Device: ", cfg.device)

# Seed everything.
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

class FlexibleRobotContactTrajectoryDataset(ContactTrajectoryDataset):
    """
    Extended dataset class that supports flexible robot budgets (1, 2, or 3 robots)
    with improved contact point placement strategies for each robot count.
    """
    
    def __init__(self, num_samples: int = 10000, 
                 primitive_type: str = "random-physics", 
                 shape_types: list[str] = None,
                 visualize: bool = False, 
                 save_intermediates = True,
                 dt: float = 0.003,
                 robot_budgets: list[int] = None):
        self.num_samples = num_samples
        self.shape_types = shape_types if shape_types is not None else cfg.shape_types
        self.primitive_type = primitive_type
        self.dt = dt
        self.robot_budgets = robot_budgets if robot_budgets is not None else [1, 2, 3]
        
        # Call our own dataset generation method instead of parent's
        if self.primitive_type == "primitives-physics" or self.primitive_type == "random-physics":
            self.datapoints = self.generate_dataset_physics(visualize=visualize, save_intermediates=save_intermediates)
        elif self.primitive_type == "primitives-linear":
            self.datapoints = self.generate_dataset_linear(save_intermediates=save_intermediates)
        else:
            raise ValueError(f"Invalid primitive type: {self.primitive_type}")
    
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
        Strategy: Draw a line from center to translation required, start sweep from other side of object.
        """
        # Object center
        obj_center_x = (min_x + max_x) // 2  # Column (x-coordinate)
        obj_center_y = (min_y + max_y) // 2  # Row (y-coordinate)

        # The push direction in image space.
        norm = torch.norm(torch.tensor([dx, dy]))
        drow = -dx / norm
        dcol = -dy / norm

        # Get a point on the line from the center against the push direction.
        start_x = obj_center_x - dcol * 100
        start_y = obj_center_y - drow * 100

        # The step direction should be along the push direction.
        step_vector = [dcol, drow]

        # # Determine push strategy based on dominant direction (matching two-robot logic)
        # if abs(dx) < 0.001 and abs(dy) < 0.001:
        #     # No translation, use default direction (push from left)
        #     start_x = max(0, min_x - max(obj_width, obj_height))
        #     start_y = obj_center_y
        #     step_vector = [1, 0]  # Move right

        #     """
        #     dx is the actual translation in the x direction (meters). Object frame.
        #     dy is the actual translation in the y direction (meters). Object frame.

        #     start_x is the starting x coordinate (pixels). Image frame. Object aligned.
        #     start_y is the starting y coordinate (pixels). Image frame. Object aligned.
        #     step_vector is the step vector (pixels). Image frame. Object aligned.
        #     """
        # elif abs(dx) > abs(dy):
        #     # X motion dominates
        #     if dx > 0:  # Moving front, push from back.
        #         print(f"Moving front, push from back.")
        #         start_x = obj_center_x
        #         start_y = cfg.observation_side_pixels - 1
        #         step_vector = [0, -1]  # Sweep up.
        #     else:  # Moving back, push from front.
        #         print(f"Moving back, push from front.")
        #         start_x = obj_center_x
        #         start_y = 0
        #         step_vector = [0, 1]  # back.
        # else:
        #     # Y motion dominates
        #     if dy > 0:  # Moving left, push from right.
        #         print(f"Moving left, push from right.")
        #         start_x = cfg.observation_side_pixels - 1
        #         start_y = obj_center_y
        #         step_vector = [-1, 0]  # Sweep right
        #     else:  # Moving right, push from left.
        #         print(f"Moving right, push from left.")
        #         start_x = 0
        #         start_y = obj_center_y
        #         step_vector = [1, 0]  # Sweep up
        
        contact_x, contact_y = self._sweep_to_boundary(mask, start_x, start_y, step_vector, robot_radius_pixels)
        return torch.tensor([[contact_y, contact_x]], device=cfg.device)
    
    def _generate_two_robot_contacts(self, dx: float, dy: float, theta: float,
                                    min_x: int, max_x: int, min_y: int, max_y: int,
                                    obj_width: int, obj_height: int,
                                    robot_radius_pixels: float, mask: torch.Tensor) -> torch.Tensor:
        """
        Generate contact points for two robots.
        Strategy: If very little rotation, push from opposite direction to translation with separated contacts.
        If some rotation, place contacts on opposite sides determined by rotation direction.
        """
        obj_center_x = (min_x + max_x) // 2
        obj_center_y = (min_y + max_y) // 2
        
        # Simplified strategy: always have one robot from left/right and one from top/bottom
        # This provides balanced coverage and works well for all motion types
        
        # Robot 1: from bottom side if dx > 0, top side if dx < 0
        if dx > 0:
            contact1_x = obj_center_x
            contact1_y = cfg.observation_side_pixels - 1
            step1 = [0, -1]  # Step up toward object center
        else:
            contact1_x = obj_center_x
            contact1_y = 0
            step1 = [0, 1]  # Step down toward object center
        
        # Robot 2: from left side if dy > 0, right side if dy < 0
        if dy < 0:
            contact2_x = 0
            contact2_y = obj_center_y
            step2 = [1, 0]  # Step right toward object center
        else:
            contact2_x = cfg.observation_side_pixels - 1
            contact2_y = obj_center_y
            step2 = [-1, 0]  # Step left toward object center
        
        step_vectors = [step1, step2]
        
        # Sweep contacts to boundary
        contacts = []
        for i, (x, y, step_vec) in enumerate([(contact1_x, contact1_y, step_vectors[0]), 
                                             (contact2_x, contact2_y, step_vectors[1])]):
            contact_x, contact_y = self._sweep_to_boundary(mask, x, y, step_vec, robot_radius_pixels)
            contacts.append([contact_y, contact_x])
        
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
            if abs(theta) > 0.02:
                strategy_transform = random.choice(["one_from_front", "one_from_back", "one_from_left", "one_from_right"])
            else: 
                raise ValueError(f"Invalid transform: {dx}, {dy}, {theta}. Might be too small.")

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

        # Use the transform-appropriate strategy with high probability
        if random.random() < 0.9:
            strategy = strategy_transform
        else:
            strategy = random.choice(all_strategies)

        # Use the existing contact generation logic with the chosen strategy
        return self._generate_contact_candidates_with_noise(
            strategy, min_x, max_x, min_y, max_y, obj_width, obj_height, 
            robot_radius_pixels, mask
        )
    
    def generate_dataset_physics(self, visualize=False, save_intermediates=False) -> list[dict]:
        """
        Generate dataset with flexible robot budgets using physics simulation.
        Adapted from the original physics generation method to support 1, 2, or 3 robots.
        """
        datapoints = []
        pbar = tqdm(total=self.num_samples, desc="Generating flexible robot dataset (physics)")
        
        # Create temporary directory for periodic saves
        import tempfile
        import os
        from datetime import datetime
        
        temp_dir = tempfile.mkdtemp(prefix=f"gco_datasets/gco_dataset_flexible_{datetime.now().strftime('%Y%m%d_%H%M%S')}_")
        print(f"Temporary dataset directory: {temp_dir}")
        
        # Create balanced combinations of robot budgets and shape types
        combinations = []
        for robot_budget in self.robot_budgets:
            for shape_type in self.shape_types:
                combinations.append((robot_budget, shape_type))
        budget_shape_combo_ix_current = 0
        
        save_interval = max(1, self.num_samples // 5)  # Save every 20% of samples
        
        while len(datapoints) < self.num_samples:
            pbar.n = len(datapoints)
            pbar.refresh()
            
            # ====================
            # Generate a transform for the object. This is tentative and will be updated by the motion that the object actually takes.
            # ====================
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
            else:
                raise ValueError(f"Invalid primitive type: {self.primitive_type}")
            
            # ====================
            # Use current combination (balanced sampling)
            # ====================
            num_robots, shape_type = combinations[budget_shape_combo_ix_current]

            # Generate shape with contacts based on the transform
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
                pbar.set_postfix({"Saving checkpoint": f"{len(datapoints)}"})
                self._save_temp_checkpoint(datapoints, temp_dir, len(datapoints))
        
        pbar.close()
        print(f"Final dataset saved to: {temp_dir}")
        
        print(f"Dataset generation complete. Total samples: {len(datapoints)}")
        print(f"Robot budget distribution:")
        for budget in self.robot_budgets:
            count = sum(1 for dp in datapoints if dp["robot_budget"] == budget)
            percentage = (count / len(datapoints)) * 100
            print(f"  {budget} robots: {count} samples ({percentage:.1f}%)")
        
        return datapoints
    
    def _save_temp_checkpoint(self, datapoints, temp_dir, count):
        """Save temporary checkpoint during generation."""
        checkpoint_path = os.path.join(temp_dir, f"checkpoint_{count:06d}_{int(time.time())}")
        os.makedirs(checkpoint_path, exist_ok=True)
        
        # Save metadata
        metadata = {
            "num_samples": count,
            "primitive_type": self.primitive_type,
            "shape_types": self.shape_types,
            "robot_budgets": self.robot_budgets,
            "generation_time": datetime.datetime.now().isoformat()
        }
        
        with open(os.path.join(checkpoint_path, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)
        
        # Save datapoints
        with open(os.path.join(checkpoint_path, "datapoints.pkl"), "wb") as f:
            pickle.dump(datapoints, f)
        
        # Save visualizations
        viz_dir = os.path.join(checkpoint_path, "visualizations")
        os.makedirs(viz_dir, exist_ok=True)
        
        # Temporarily set datapoints for visualization
        temp_datapoints = self.datapoints if hasattr(self, 'datapoints') else []
        self.datapoints = datapoints
        try:
            print(f"  Generating visualizations for checkpoint {count}...")
            self.save_sample_visualizations(viz_dir, num_samples=min(10, len(datapoints)))
        finally:
            # Restore original datapoints or remove if it didn't exist
            if temp_datapoints:
                self.datapoints = temp_datapoints
            elif hasattr(self, 'datapoints'):
                delattr(self, 'datapoints')
        
        print(f"Saved checkpoint {count} with visualizations to file://{checkpoint_path}")

print("Creating flexible robot dataset...")
# Create a dataset with flexible robot budgets
dataset = FlexibleRobotContactTrajectoryDataset(
    num_samples=100,
    primitive_type="random-physics", 
    visualize=False, 
    dt=0.003,
    robot_budgets=[1, 2, 3]  # Support 1, 2, and 3 robots
)

# Save sample visualizations with progress tracking
print("Saving sample visualizations...")
dataset.save_sample_visualizations(cfg.data_dir_contacts_trajs_test / "visualizations", num_samples=20)

print("Dataset creation complete!")
print(f"Total samples: {len(dataset)}")
print(f"Robot budget distribution:")
for budget in [1, 2, 3]:
    count = sum(1 for dp in dataset.datapoints if dp["robot_budget"] == budget)
    percentage = (count / len(dataset.datapoints)) * 100
    print(f"  {budget} robots: {count} samples ({percentage:.1f}%)")
