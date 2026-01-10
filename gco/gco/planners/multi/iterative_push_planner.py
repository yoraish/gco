"""
Iterative Push Planner for multi-robot object manipulation.
This planner observes an object, receives a goal pose, and iteratively pushes the object
until it reaches the goal using the learned contact trajectory model.
"""
import torch
import numpy as np
import time
import matplotlib
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod

# GCO imports
from gco.world.world import World
from gco.world.robot import RobotDisk
from gco.world.obstacles import ObstacleCircle
from gco.world.objects import ObjectRectangle, ObjectCircle
from gco.utils.transform_utils import Translation2, Transform2, Transform, points_local_to_world
from gco.utils.model_utils import *
from gco.config import Config as cfg
from gco.utils.viz_utils import densify_all_trajectories, smooth_all_trajectories
from gcocpp import ObjectCentricAStarPlanner, GSPIPlanner
from gco.utils.data_vis_utils import visualize_transformed_mask, visualize_batch_denoising, visualize_push_trajectory, visualize_batch
from gco.models.contact_push_model import ContactTrajectoryModel
from gco.planners.multi.multi_robot_planner import MultiRobotPlanner
from gco.tasks.tasks import TaskType, ManipulationTask

# Type checking imports
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from gco.tasks.tasks import Task


@dataclass
class IterativePushPlannerParams:
    """Parameters for the iterative push planner."""
    max_iterations: int = 10
    goal_tolerance_position: float = 0.15  # meters
    goal_tolerance_orientation: float = 0.15 # radians
    min_push_distance: float = 0.02  # minimum distance to consider a push worthwhile
    visualization_interval: int = 5  # visualize every N iterations
    visualize_planning: bool = True
    save_visualizations: bool = False
    output_dir: str = "output/iterative_push"
    model_checkpoint: Optional[Path] = None
    model_type: str = "discrete"  # "discrete" or "continuous"
    persistent_simulator: bool = True  # Use a single persistent MuJoCo instance
    planner_type: str = "a_star"  # "interpolate" or "a_star"

class IterativePushPlanner(MultiRobotPlanner):
    """
    Iterative push planner that repeatedly pushes an object until it reaches a goal pose.
    
    This planner:
    1. Observes the object at its current pose
    2. Receives a goal pose for the object
    3. On each iteration, chooses the best motion primitive
    4. Carries out inference on the pushing model
    5. Moves robots to contact points and pushes
    6. Repeats until the object is at or near the goal
    """
    
    def __init__(self, world: World, params: Dict[str, any]):
        super().__init__(world, params)
        self.params = IterativePushPlannerParams(**params)
        
        # Initialize the contact trajectory model
        self._initialize_model()
        
        # Create output directory
        os.makedirs(self.params.output_dir, exist_ok=True)
        
        # Track planning history
        self.planning_history = []
        
        # Track timing statistics
        self.timing_stats = {
            'total_iterations': 0,
            'total_time': 0.0,
            'current_iteration_times': [],  # List of (name, time) tuples for current iteration
            'all_iteration_times': [],       # List of all iteration timing data
            'total_travel_distances': {},    # Dict of robot_name -> total_distance
            'current_iteration_distances': {}, # Dict of robot_name -> distance for current iteration
            'push_accuracy_errors': [],      # List of push accuracy errors (position, orientation)
            'total_push_accuracy_position': 0.0,  # Cumulative position error
            'total_push_accuracy_orientation': 0.0, # Cumulative orientation error
            'current_push_accuracy_position': 0.0,  # Current iteration position error
            'current_push_accuracy_orientation': 0.0 # Current iteration orientation error
        }
    
    # ====================
    # Travel distance functions.
    # ====================
    def _calculate_travel_distance(self, paths: Dict[str, List]) -> Dict[str, float]:
        """Calculate total travel distance for each robot from their paths."""
        distances = {}
        
        for robot_name, path in paths.items():
            if not path or len(path) < 2:
                distances[robot_name] = 0.0
                continue
                
            total_distance = 0.0
            for i in range(1, len(path)):
                # Calculate distance between consecutive poses
                prev_pose = torch.tensor(path[i-1][:2])  # Only x, y coordinates
                curr_pose = torch.tensor(path[i][:2])
                distance = torch.norm(curr_pose - prev_pose).item()
                total_distance += distance
            
            distances[robot_name] = total_distance
            
        return distances
    
    def _calculate_push_distance(self, push_trajectories_world: torch.Tensor) -> Dict[str, float]:
        """Calculate travel distance for push trajectories."""
        distances = {}
        
        # push_trajectories_world shape: (B, N, H, 3) where B=1, N=num_robots, H=horizon, 3=(x,y,theta)
        if len(push_trajectories_world.shape) == 4:
            push_trajectories = push_trajectories_world.squeeze(0)  # Remove batch dimension: (N, H, 3)
        else:
            push_trajectories = push_trajectories_world  # Already (N, H, 3)
        
        robot_names = list(self.world.robots_d.keys())
        
        for i, robot_name in enumerate(robot_names):
            if i < push_trajectories.shape[0]:
                robot_trajectory = push_trajectories[i]  # (H, 3)
                total_distance = 0.0
                
                for j in range(1, robot_trajectory.shape[0]):
                    # Calculate distance between consecutive waypoints
                    prev_point = robot_trajectory[j-1][:2]  # (x, y)
                    curr_point = robot_trajectory[j][:2]     # (x, y)
                    distance = torch.norm(curr_point - prev_point).item()
                    total_distance += distance
                
                distances[robot_name] = total_distance
            else:
                distances[robot_name] = 0.0
        
        return distances
    
    def _update_travel_distances(self, paths: Dict[str, List]):
        """Update travel distance statistics with new robot paths."""
        iteration_distances = self._calculate_travel_distance(paths)
        
        # Update current iteration distances
        self.timing_stats['current_iteration_distances'] = iteration_distances
        
        # Update total travel distances
        for robot_name, distance in iteration_distances.items():
            if robot_name not in self.timing_stats['total_travel_distances']:
                self.timing_stats['total_travel_distances'][robot_name] = 0.0
            self.timing_stats['total_travel_distances'][robot_name] += distance
    
    def _update_push_distances(self, push_trajectories_world: torch.Tensor):
        """Update travel distance statistics with push trajectories."""
        push_distances = self._calculate_push_distance(push_trajectories_world)
        
        # Update current iteration distances (add to existing)
        for robot_name, distance in push_distances.items():
            if robot_name not in self.timing_stats['current_iteration_distances']:
                self.timing_stats['current_iteration_distances'][robot_name] = 0.0
            self.timing_stats['current_iteration_distances'][robot_name] += distance
        
        # Update total travel distances
        for robot_name, distance in push_distances.items():
            if robot_name not in self.timing_stats['total_travel_distances']:
                self.timing_stats['total_travel_distances'][robot_name] = 0.0
            self.timing_stats['total_travel_distances'][robot_name] += distance

    def _calculate_push_accuracy(self, pose_before: Transform2, pose_after: Transform2, requested_transform: Transform2):
        """Calculate push accuracy by comparing requested vs actual transformation."""
        # Calculate the actual transformation achieved
        actual_transform = pose_before.inverse() * pose_after
        
        # Calculate position error
        position_error = torch.norm(actual_transform.get_t() - requested_transform.get_t()).item()
        
        # Calculate orientation error
        orientation_error = torch.abs(actual_transform.get_theta() - requested_transform.get_theta()).item()
        
        # Store the errors
        self.timing_stats['push_accuracy_errors'].append({
            'position_error': position_error,
            'orientation_error': orientation_error,
            'requested_position': requested_transform.get_t().tolist(),
            'requested_orientation': requested_transform.get_theta().item(),
            'actual_position': actual_transform.get_t().tolist(),
            'actual_orientation': actual_transform.get_theta().item()
        })
        
        # Update current iteration errors
        self.timing_stats['current_push_accuracy_position'] = position_error
        self.timing_stats['current_push_accuracy_orientation'] = orientation_error
        
        # Update total errors
        self.timing_stats['total_push_accuracy_position'] += position_error
        self.timing_stats['total_push_accuracy_orientation'] += orientation_error
        
        print(f"🎯 Push Accuracy: Position error: {position_error:.4f}m, Orientation error: {orientation_error:.4f}rad")
        print(f"   Requested: pos={requested_transform.get_t().tolist()}, ori={requested_transform.get_theta().item():.4f}")
        print(f"   Actual:    pos={actual_transform.get_t().tolist()}, ori={actual_transform.get_theta().item():.4f}")

    # ====================
    # Timing functions.
    # ====================
    def _time_function(self, func, name: str, *args, **kwargs):
        """
        Wrapper function that times a function execution and stores the result.
        
        Args:
            func: Function to time
            name: Name/description of the process being timed
            *args, **kwargs: Arguments to pass to the function
            
        Returns:
            Result of the function execution
        """
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        execution_time = end_time - start_time
        self.timing_stats['current_iteration_times'].append((name, execution_time))
        
        return result
    
    def _print_iteration_timing_summary(self, iteration: int):
        """Print a nicely formatted timing summary for the current iteration."""
        if not self.timing_stats['current_iteration_times']:
            return
            
        print("\n" + "="*60)
        print(f"📊 ITERATION {iteration + 1}/{self.params.max_iterations} TIMING SUMMARY")
        print("="*60)
        
        # Calculate total iteration time
        total_time = sum(time for _, time in self.timing_stats['current_iteration_times'])
        
        # Print each component
        for name, component_time in self.timing_stats['current_iteration_times']:
            percentage = (component_time / total_time) * 100 if total_time > 0 else 0
            print(f"⏱️  {name:<20} {component_time:.3f}s ({percentage:.1f}%)")
        
        print(f"📊 Total Iteration Time: {total_time:.3f}s")
        
        # Print travel distances for this iteration
        if self.timing_stats['current_iteration_distances']:
            print("\n🚀 ROBOT TRAVEL DISTANCES (This Iteration):")
            for robot_name, distance in self.timing_stats['current_iteration_distances'].items():
                print(f"   {robot_name}: {distance:.3f}m")
        
        # Print push accuracy for this iteration
        if self.timing_stats['current_push_accuracy_position'] > 0 or self.timing_stats['current_push_accuracy_orientation'] > 0:
            print("\n🎯 PUSH ACCURACY (This Iteration):")
            print(f"   Position Error: {self.timing_stats['current_push_accuracy_position']:.4f}m")
            print(f"   Orientation Error: {self.timing_stats['current_push_accuracy_orientation']:.4f}rad")
        
        print("="*60)
        
        # Store iteration data and update statistics
        self.timing_stats['all_iteration_times'].append({
            'iteration': iteration,
            'total_time': total_time,
            'components': self.timing_stats['current_iteration_times'].copy()
        })
        self.timing_stats['total_iterations'] += 1
        self.timing_stats['total_time'] += total_time
    
    def _print_and_reset_timing(self):
        """Print final timing statistics and reset the timing data."""
        if not self.timing_stats['all_iteration_times']:
            print("No timing data to display.")
            return
            
        print("\n" + "="*80)
        print("🎯 FINAL TIMING SUMMARY")
        print("="*80)
        
        total_iterations = self.timing_stats['total_iterations']
        total_time = self.timing_stats['total_time']
        
        print(f"📈 Total Iterations: {total_iterations}")
        print(f"⏱️  Total Time: {total_time:.3f}s")
        print(f"📊 Average Time per Iteration: {total_time/total_iterations:.3f}s")
        print()
        
        # Aggregate component times across all iterations
        component_totals = {}
        component_counts = {}
        
        for iteration_data in self.timing_stats['all_iteration_times']:
            for name, time in iteration_data['components']:
                if name not in component_totals:
                    component_totals[name] = 0.0
                    component_counts[name] = 0
                component_totals[name] += time
                component_counts[name] += 1
        
        # Print component breakdown
        print("🔍 COMPONENT BREAKDOWN:")
        for name in sorted(component_totals.keys()):
            total_component_time = component_totals[name]
            avg_component_time = total_component_time / component_counts[name]
            percentage = (total_component_time / total_time) * 100 if total_time > 0 else 0
            print(f"   {name:<20} {total_component_time:.3f}s total, {avg_component_time:.3f}s avg ({percentage:.1f}%)")
        
        # Print total travel distances
        if self.timing_stats['total_travel_distances']:
            print("\n🚀 TOTAL ROBOT TRAVEL DISTANCES:")
            total_distance = 0.0
            for robot_name, distance in self.timing_stats['total_travel_distances'].items():
                print(f"   {robot_name}: {distance:.3f}m")
                total_distance += distance
            print(f"   📊 Total Distance: {total_distance:.3f}m")
        
        # Print total push accuracy
        if self.timing_stats['push_accuracy_errors']:
            num_pushes = len(self.timing_stats['push_accuracy_errors'])
            avg_position_error = self.timing_stats['total_push_accuracy_position'] / num_pushes
            avg_orientation_error = self.timing_stats['total_push_accuracy_orientation'] / num_pushes
            print(f"\n🎯 TOTAL PUSH ACCURACY ({num_pushes} pushes):")
            print(f"   Avg Position Error: {avg_position_error:.4f}m")
            print(f"   Avg Orientation Error: {avg_orientation_error:.4f}rad")
            print(f"   Total Position Error: {self.timing_stats['total_push_accuracy_position']:.4f}m")
            print(f"   Total Orientation Error: {self.timing_stats['total_push_accuracy_orientation']:.4f}rad")
        
        print("="*80)
        
        # Reset timing data
        self.timing_stats['current_iteration_times'] = []
        self.timing_stats['all_iteration_times'] = []
        self.timing_stats['total_iterations'] = 0
        self.timing_stats['total_time'] = 0.0
        self.timing_stats['total_travel_distances'] = {}
        self.timing_stats['current_iteration_distances'] = {}
        self.timing_stats['push_accuracy_errors'] = []
        self.timing_stats['total_push_accuracy_position'] = 0.0
        self.timing_stats['total_push_accuracy_orientation'] = 0.0
        self.timing_stats['current_push_accuracy_position'] = 0.0
        self.timing_stats['current_push_accuracy_orientation'] = 0.0
    
    def time_custom_function(self, func, name: str, *args, **kwargs):
        """
        Public method to time any custom function and include it in the timing statistics.
        
        Args:
            func: Function to time
            name: Name/description of the process being timed
            *args, **kwargs: Arguments to pass to the function
            
        Returns:
            Result of the function execution
        """
        return self._time_function(func, name, *args, **kwargs)
    
    def print_current_timing(self):
        """Print the current iteration's timing data without resetting."""
        if not self.timing_stats['current_iteration_times']:
            print("No timing data for current iteration.")
            return
            
        print("\n" + "="*60)
        print("📊 CURRENT ITERATION TIMING")
        print("="*60)
        
        total_time = sum(time for _, time in self.timing_stats['current_iteration_times'])
        
        for name, component_time in self.timing_stats['current_iteration_times']:
            percentage = (component_time / total_time) * 100 if total_time > 0 else 0
            print(f"⏱️  {name:<20} {component_time:.3f}s ({percentage:.1f}%)")
        
        print(f"📊 Total Time: {total_time:.3f}s")
        print("="*60)

    # ====================
    # Model functions.
    # ====================
    def _initialize_model(self):
        """Initialize the contact trajectory model based on model type."""
        # Load the model weights
        if self.params.model_checkpoint is None:
            raise ValueError("Model checkpoint path is required.")
        
        model_checkpoint_path = self.params.model_checkpoint
        
        # Load the model checkpoint
        print(f"Loading model checkpoint from: {model_checkpoint_path}")
        
        if self.params.model_type == "continuous":
            # Initialize continuous model
            self.cogen_model = ContinuousContactTrajectoryModel(hidden_dim=128).to(cfg.device)
            
            # Load checkpoint
            checkpoint = torch.load(model_checkpoint_path, map_location=cfg.device)
            if 'model_state_dict' in checkpoint:
                self.cogen_model.load_state_dict(checkpoint['model_state_dict'])
                # Set trajectory scale if available
                if 'trajectory_scale' in checkpoint:
                    self.cogen_model.trajectory_scale = checkpoint['trajectory_scale']
            else:
                self.cogen_model.load_state_dict(checkpoint)
            
            print(f"Loaded continuous contact trajectory model from: {model_checkpoint_path}")
            
        elif self.params.model_type == "continuous_trajectory":
            # Initialize continuous trajectory model
            self.cogen_model = ContinuousTrajectoryModel(hidden_dim=128).to(cfg.device)
            
            # Load checkpoint
            checkpoint = torch.load(model_checkpoint_path, map_location=cfg.device)
            if 'model_state_dict' in checkpoint:
                self.cogen_model.load_state_dict(checkpoint['model_state_dict'])
                # Set trajectory scale if available
                if 'trajectory_scale' in checkpoint:
                    self.cogen_model.trajectory_scale = checkpoint['trajectory_scale']
            else:
                self.cogen_model.load_state_dict(checkpoint)
            
            print(f"Loaded continuous trajectory model from: {model_checkpoint_path}")
            
        else:  # discrete model (default)
            # Initialize discrete model
            # Note: vocab_size is cfg.V + 1 to match the trained model checkpoint
            # (cfg.V accounts for spatial tokens, +1 for mask token compatibility)
            vocab_size = cfg.V + 1
            self.cogen_model = ContactTrajectoryModel(vocab_size=vocab_size).to(cfg.device)
            
            # Load checkpoint
            self.cogen_model.load_state_dict(torch.load(model_checkpoint_path, map_location=cfg.device))
            
            print(f"Loaded discrete contact trajectory model from: {model_checkpoint_path}")
            
        self.cogen_model.eval()
 
    # ====================
    # Planning functions.
    # ====================
    def _is_goal_reached(self, current_pose: Transform2, goal_pose: Transform2) -> bool:
        """Check if the object has reached the goal pose within tolerance."""
        position_error = torch.norm(current_pose.get_t() - goal_pose.get_t())
        orientation_error = torch.abs(current_pose.get_theta() - goal_pose.get_theta())

        print("================================================")
        GREEN = "\033[92m"
        RESET = "\033[0m"
        print(f"{GREEN}POS Error: {position_error}, Tol: {self.params.goal_tolerance_position}{RESET}")
        print(f"{GREEN}ORI Error: {orientation_error.item()}, Tol: {self.params.goal_tolerance_orientation}{RESET}")
        print(f"Current: {current_pose}")
        print(f"Goal   : {goal_pose}")
        print("================================================")
        
        return (position_error <= self.params.goal_tolerance_position and 
                orientation_error <= self.params.goal_tolerance_orientation)
    
    def _compute_relative_transform(self, current_pose: Transform2, goal_pose: Transform2) -> Transform2:
        """Compute the relative transform from current pose to goal pose."""
        return current_pose.inverse() * goal_pose
    
    def _generate_push_plan(self, observation: torch.Tensor, relative_transform: Transform2, seed: int = 42) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate contact points and push trajectories using the learned model."""
        # Convert to batched tensors
        observation_batched = observation.unsqueeze(0).to(cfg.device)
        relative_transform_batched = relative_transform.to_tensor().unsqueeze(0).to(cfg.device)
        
        # Get the number of available robots as budget
        num_robots = len(self.world.robots_d)
        budget = torch.tensor([num_robots], device=cfg.device)
        
        # Generate using the model with iteration-based seed
        with torch.no_grad():
            sol_l = self.cogen_model.generate(observation_batched, relative_transform_batched, budget=budget, seed=seed, smoothness_weight=cfg.smoothness_weight)
        
        # Extract contact points and trajectories
        sol_d = torch.stack([s[0] for s in sol_l], dim=0)  # contact points tokens
        sol_c = torch.stack([s[1] for s in sol_l], dim=0)  # push trajectories  
        
        return sol_d[-1], sol_c[-1]  # Return the final generation step
    
    def _transform_to_world_frame(self, contact_points_tokens: torch.Tensor, 
                                 push_trajectories_pixels: torch.Tensor,
                                 object_pose: Transform2) -> Tuple[torch.Tensor, torch.Tensor]:
        """Transform contact points and push trajectories to world frame."""
        # Transform contact points
        contact_points_meters_local = tokens_to_meters_local(contact_points_tokens)
        contact_points_meters_world = points_local_to_world(contact_points_meters_local.squeeze(), object_pose)
        if len(contact_points_meters_world.shape) == 1:
            contact_points_meters_world = contact_points_meters_world.unsqueeze(0)
        
        # Transform push trajectories
        push_trajectories_meters_local = push_trajectories_pixels_local_to_meters_local(
            push_trajectories_pixels, contact_points_meters_local)
        push_trajectories_meters_world = points_local_to_world(push_trajectories_meters_local, object_pose)
        
        # Add theta dimension
        B, N, H, _ = push_trajectories_meters_world.shape
        push_points = push_trajectories_meters_world.reshape(-1, 2)
        push_points = torch.cat((push_points, torch.zeros((push_points.shape[0], 1), device=cfg.device)), dim=-1)
        push_trajectories_meters_world = push_points.reshape(B, N, H, 3)
        
        return contact_points_meters_world, push_trajectories_meters_world
    
    # ====================
    # Helper functions.
    # ====================
    def _filter_masked_contacts(self,
                                contact_points_world: torch.Tensor,
                                push_trajectories_world: torch.Tensor,
                                contact_points_tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Remove masked contact points/trajectories that correspond to unused robots.

        A contact is considered *masked* if either the row or column token equals
        the mask token defined in the configuration. For any masked contact we
        drop both the contact points in world frame and the corresponding push trajectory
        along the N dimension.
        
        Args:
            contact_points_world: Contact points in world frame, shape (B, N, 2) or (N, 2)
            push_trajectories_world: Push trajectories in world frame, shape (B, N, H, 3) or (N, H, 3)
            contact_points_tokens: Token pairs for determining which contacts are masked, shape (B, N*2)
            
        Returns:
            Tuple of (filtered_contact_points_world, filtered_push_trajectories_world)
        """
        # contact_points_tokens has shape (B, N*2) where B=1 in our usage.
        # contact_points_world has shape (B, N, 2) or (N, 2)
        # push_trajectories_world has shape (B, N, H, 3) or (N, H, 3)
        
        # Handle different input shapes
        if len(contact_points_world.shape) == 2:
            # (N, 2) -> (1, N, 2)
            contact_points_world = contact_points_world.unsqueeze(0)
        if len(push_trajectories_world.shape) == 3:
            # (N, H, 3) -> (1, N, H, 3)
            push_trajectories_world = push_trajectories_world.unsqueeze(0)
            
        B = contact_points_tokens.shape[0]
        N_total = cfg.N
        tokens = contact_points_tokens.clone()
        # Reshape to (B,N,2) to inspect row/col tokens.
        tokens_pairs = tokens.reshape(B, N_total, 2)

        # Build boolean mask of valid (unmasked) contacts.
        # Check for distance within 5 units from mask token instead of exact equality
        mask_valid = (torch.abs(tokens_pairs[:, :, 0] - cfg.mask_token) > 5) & (torch.abs(tokens_pairs[:, :, 1] - cfg.mask_token) > 5)
        # For B=1 -> (N,) otherwise (B,N)
        if B == 1:
            mask_valid = mask_valid.squeeze(0)
        # Get indices of valid contacts
        valid_indices = torch.nonzero(mask_valid, as_tuple=False).squeeze()
        if valid_indices.numel() == 0:
            # No valid contacts – return empty tensors with correct dims
            if len(contact_points_world.shape) == 3:
                return (contact_points_world[:, :0], push_trajectories_world[:, :0])
            else:
                return (contact_points_world[:0], push_trajectories_world[:0])

        # Ensure valid_indices is 1-D
        valid_indices = valid_indices.flatten()

        # Filter contact points in world frame
        filtered_contact_points_world = contact_points_world[:, valid_indices]
        
        # Filter trajectories along N dimension
        filtered_push_trajectories_world = push_trajectories_world[:, valid_indices]

        return filtered_contact_points_world, filtered_push_trajectories_world

    def _filter_masked_contacts_original(self,
                                        contact_points_tokens: torch.Tensor,
                                        push_trajectories_pixels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Remove masked contact points/trajectories that correspond to unused robots (original version for visualization).

        A contact is considered *masked* if either the row or column token equals
        the mask token defined in the configuration. For any masked contact we
        drop both the contact tokens pair and the corresponding push trajectory
        along the N dimension.
        
        This is the original version that works with tokens and pixel trajectories for visualization.
        """
        # contact_points_tokens has shape (B, N*2) where B=1 in our usage.
        # push_trajectories_pixels has shape (B, N, H, 2)
        B = contact_points_tokens.shape[0]
        N_total = cfg.N
        tokens = contact_points_tokens.clone()
        # Reshape to (B,N,2) to inspect row/col tokens.
        tokens_pairs = tokens.reshape(B, N_total, 2)

        # Build boolean mask of valid (unmasked) contacts.
        # Check for distance within 5 units from mask token instead of exact equality
        mask_valid = (torch.abs(tokens_pairs[:, :, 0] - cfg.mask_token) > 5) & (torch.abs(tokens_pairs[:, :, 1] - cfg.mask_token) > 5)
        # For B=1 -> (N,) otherwise (B,N)
        if B == 1:
            mask_valid = mask_valid.squeeze(0)
        # Get indices of valid contacts
        valid_indices = torch.nonzero(mask_valid, as_tuple=False).squeeze()
        if valid_indices.numel() == 0:
            # No valid contacts – return empty tensors with correct dims
            return (contact_points_tokens[:, :0], push_trajectories_pixels[:, :0])

        # Ensure valid_indices is 1-D
        valid_indices = valid_indices.flatten()

        # Filter tokens: keep rows corresponding to valid contacts
        filtered_tokens_pairs = tokens_pairs[:, valid_indices]
        filtered_tokens = filtered_tokens_pairs.reshape(B, -1)

        # Filter trajectories along N dimension
        filtered_push_trajs = push_trajectories_pixels[:, valid_indices]

        return filtered_tokens, filtered_push_trajs

    def _plan_robot_paths(self, contact_points_world: torch.Tensor) -> Dict[str, List]:
        """Plan paths for robots to reach contact points."""
        # Get current robot poses
        robot_names = list(self.world.robots_d.keys())
        starts = {name: self.world.get_robot_pose(name).to_tensor() for name in robot_names}
        
        # Set goals as contact points
        goals = {}
        for i, name in enumerate(robot_names):
            if i < contact_points_world.shape[1]:
                goals[name] = torch.cat((contact_points_world[:, i, :].squeeze(), 
                                       torch.tensor([0.0], device=cfg.device)), dim=-1)
            else:
                # If fewer contact points than robots, keep some robots in place
                goals[name] = starts[name]

        max_start_goal_distance = 0.0
        for name in robot_names:
            for name_goal in robot_names:
                distance = torch.norm(starts[name] - goals[name_goal])
                if distance > max_start_goal_distance:
                    max_start_goal_distance = distance
        
        # Plan using CTSWAP with deterministic seed
        planner = GSPIPlanner(self.world, goal_tolerance=0.1, seed=self.world.simulator.seed)
        planner.set_grid_resolution_heuristic(0.05)
        planner.set_max_distance_heuristic(max(max_start_goal_distance * 2.0, 3.0))
        planner.set_max_iterations(100)
        planner.set_verbose(False)

        # If the distance between start/goal pairs (not necessarily in the same ordering as the lists) is very small (< 0.1), then skip planning and have each path be just the start and then the goal.
        # Check if there exists any assignment where all distances are small
        import itertools
        
        # Create distance matrix between all starts and goals
        distance_matrix = {}
        for start_name in robot_names:
            distance_matrix[start_name] = {}
            for goal_name in robot_names:
                distance_matrix[start_name][goal_name] = torch.norm(starts[start_name] - goals[goal_name]).item()
        
        # Check all possible permutations to see if any has all distances < 0.1
        exists_small_assignment = False
        best_assignment = None
        min_max_distance = float('inf')
        
        for permutation in itertools.permutations(robot_names):
            max_distance_in_perm = 0
            for i, start_name in enumerate(robot_names):
                goal_name = permutation[i]
                distance = distance_matrix[start_name][goal_name]
                max_distance_in_perm = max(max_distance_in_perm, distance)
            
            if max_distance_in_perm < 0.1:
                exists_small_assignment = True
                best_assignment = permutation
                break
            elif max_distance_in_perm < min_max_distance:
                min_max_distance = max_distance_in_perm
                best_assignment = permutation
        
        if exists_small_assignment:
            # Skip planning and create simple paths using the best assignment
            paths = {}
            for i, start_name in enumerate(robot_names):
                goal_name = best_assignment[i]
                paths[start_name] = [starts[start_name].tolist(), goals[goal_name].tolist()]
        else:
            # Plan using CTSWAP with deterministic seed
            paths = planner.plan(starts, goals)
            
            # Check if planning failed (no paths returned or empty paths)
            if not paths or any(len(path) == 0 for path in paths.values()):
                print("Robot planner failed to find valid paths. Moving robots randomly to break deadlock.")
                paths = self._generate_random_movement_paths(starts)
        
        return paths
    
    def _generate_random_movement_paths(self, starts: Dict[str, torch.Tensor]) -> Dict[str, List]:
        """
        Generate random movement paths for robots when planning fails.
        
        This method moves each robot a small random distance in a random direction
        with intermediate steps to help break deadlocks and potentially find new valid configurations.
        
        Args:
            starts: Dictionary mapping robot names to their start poses
            
        Returns:
            Dictionary mapping robot names to their random movement paths
        """
        paths = {}
        total_step_size = 0.06  # Total distance to move
        intermediate_step_size = 0.01  # Size of each intermediate step
        
        for robot_name, start_pose in starts.items():
            # Generate random direction (angle)
            random_angle = np.random.uniform(0, 2 * np.pi)
            
            # Calculate total displacement
            dx_total = total_step_size * np.cos(random_angle)
            dy_total = total_step_size * np.sin(random_angle)
            
            # Create path with intermediate steps
            current_pos = start_pose[:2]  # x, y coordinates
            path_points = [start_pose.tolist()]
            
            # Calculate number of intermediate steps needed
            num_steps = int(total_step_size / intermediate_step_size)
            
            # Add intermediate steps
            for i in range(1, num_steps + 1):
                # Calculate position for this step
                step_progress = i / num_steps
                step_dx = dx_total * step_progress
                step_dy = dy_total * step_progress
                
                intermediate_pos = current_pos + torch.tensor([step_dx, step_dy], device=cfg.device)
                
                # Keep the same orientation (theta)
                intermediate_pose = torch.cat([intermediate_pos, start_pose[2:3]], dim=0)  # x, y, theta
                path_points.append(intermediate_pose.tolist())
            
            paths[robot_name] = path_points
            
            final_pos = current_pos + torch.tensor([dx_total, dy_total], device=cfg.device)
            print(f"Robot {robot_name}: random movement from {current_pos.tolist()} to {final_pos.tolist()} with {num_steps} intermediate steps")
        
        return paths
    
    def _execute_push(self, paths: Dict[str, List], push_trajectories_world: torch.Tensor, visualize: bool = False) -> List[Dict]:
        """Execute the push by moving robots to contact points and then pushing."""
        # Record object pose before push execution
        object_name = list(self.world.objects_d.keys())[0]  # Assume single object
        pose_before_push = self.world.get_object_pose(object_name).to(cfg.device)
        
        # Add push trajectories to the paths
        ctswap_path_lengths = {k: len(v) for k, v in paths.items()}
        paths_with_push = add_push_trajectories_to_paths(paths, push_trajectories_world)
        
        # Densify and smooth the paths
        paths_prepare = {k: v[:ctswap_path_lengths[k]] for k, v in paths_with_push.items()}
        paths_act = {k: v[ctswap_path_lengths[k]:] for k, v in paths_with_push.items()}
        
        paths_prepare = densify_all_trajectories(paths_prepare, 20)  # Doubled from 10 to make twice as slow
        # paths_act = smooth_all_trajectories(paths_act, 20)
        # paths_act = {k: v.tolist() for k, v in paths_act.items()}
        paths_act = densify_all_trajectories(paths_act, 20)  # Doubled from 10 to make twice as slow

        # Paths act should start from the last configuration of paths prepare.
        for k in paths_act.keys():
            if  len(paths_act[k]) == 0 or len(paths_prepare[k]) == 0:
                print(f"Warning: No valid robot paths found for execution. len(paths_act[k]) = {len(paths_act[k])}, len(paths_prepare[k]) = {len(paths_prepare[k])}")
                return []
            paths_act[k] += paths_prepare[k][-1:] - paths_act[k][0]

        final_paths = {k: paths_prepare[k].tolist() + paths_act[k].tolist() for k in paths_with_push.keys()}
        
        # Track push motion distances (paths_act contains the push trajectories)
        push_distances = self._calculate_push_distance(push_trajectories_world)
        for robot_name, distance in push_distances.items():
            if robot_name not in self.timing_stats['total_travel_distances']:
                self.timing_stats['total_travel_distances'][robot_name] = 0.0
            self.timing_stats['total_travel_distances'][robot_name] += distance
        
        # Convert to tensor format
        final_paths_tensor = {k: torch.tensor(path)[:, :2] for k, path in final_paths.items()}
        
        # Execute trajectories in persistent simulator
        print("\n\n\n Executing trajectories in persistent simulator...")
        world_states = self.world.apply_trajectories(final_paths_tensor, visualize=visualize, real_time=False)
        
        # Record object pose after push execution and calculate push accuracy
        pose_after_push = self.world.get_object_pose(object_name).to(cfg.device)
        
        # Get the requested transformation from the current iteration
        if self.planning_history:
            requested_transform = self.planning_history[-1]['relative_transform']
            self._calculate_push_accuracy(pose_before_push, pose_after_push, requested_transform)
        
        return world_states
    
    def _visualize_iteration(self, iteration: int, observation: torch.Tensor, 
                           relative_transform: Transform2, contact_points_tokens: torch.Tensor,
                           push_trajectories: torch.Tensor, object_name: str):
        """Visualize the current iteration's planning."""
        fig, axs = visualize_batch({
            "mask": observation.unsqueeze(0),
            "transform_object": relative_transform.to_tensor().unsqueeze(0),
            "trajectory": push_trajectories,
            "tokens": contact_points_tokens
        }, save_path=None)
        
        save_path = os.path.join(self.params.output_dir, f"iteration_{iteration:03d}.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")
        
        plt.close(fig)
  
    def _interpolate_push_trajectories(self, push_trajectories: torch.Tensor) -> torch.Tensor:
        """Interpolate the push trajectories to be a linear interpolation between start and end."""
        # Push trajectories is B, N, H, 3.
        push_trajectories = push_trajectories.cpu().numpy()
        push_trajectories_interpolated = np.zeros_like(push_trajectories)
        for i in range(push_trajectories.shape[0]):
            for j in range(push_trajectories.shape[1]):
                for t in range(push_trajectories.shape[2]):
                    t1 = t / push_trajectories.shape[2]
                    t2 = (t + 1) / push_trajectories.shape[2]   
                    push_trajectories_interpolated[i, j, t] = push_trajectories[i, j, -1] * t1 + push_trajectories[i, j, 0] * (1 - t1)
        return torch.from_numpy(push_trajectories_interpolated).to(cfg.device)
    
    def _save_planner_visualization(self, object_centric_a_star_planner, object_name: str, 
                                  object_path: List, start_objects: Dict, goal_objects: Dict):
        """Save planner output for visualization."""
        # Determine object type and size
        object_obj = self.world.objects_d[object_name]
        if hasattr(object_obj, 'radius'):
            object_type = "circle"
            object_size = object_obj.radius
            # Convert path from list format to dict format
            path_list = []
            for waypoint in object_path:
                path_list.append({
                    "x": waypoint[0],
                    "y": waypoint[1], 
                    "theta": waypoint[2]
                })
            object_centric_a_star_planner.save_planner_output(
                path_list, start_objects[object_name], goal_objects[object_name],
                os.path.join(self.params.output_dir, "a_star_path.json"),
                object_type, object_size
            )
        elif hasattr(object_obj, 'width') and hasattr(object_obj, 'height'):
            object_type = "rectangle"
            object_width = object_obj.width
            object_height = object_obj.height
            # Convert the path to the format expected by the C++ method
            path_list = []
            for waypoint in object_path:
                path_list.append({
                    "x": waypoint[0],
                    "y": waypoint[1], 
                    "theta": waypoint[2]
                })
            object_centric_a_star_planner.save_planner_output_rectangle(
                path_list, start_objects[object_name], goal_objects[object_name],
                os.path.join(self.params.output_dir, "a_star_path.json"),
                object_type, object_width, object_height
            )
 

    # ====================
    # Main functions.
    # ==================== 
    def _plan_action_trajs_interpolate(self, task: 'Task', visualize: bool = False) -> Dict[str, torch.Tensor]:
        """
        Plan iterative push actions to achieve the manipulation task.
        
        Args:
            task: ManipulationTask containing object_id and target_pose
            
        Returns:
            Dictionary mapping robot names to their final planned trajectories
        """
        if not isinstance(task, ManipulationTask):
            raise ValueError("IterativePushPlanner only supports ManipulationTask")
        
        object_name = task.object_id
        goal_pose = Transform2(t=torch.tensor(task.target_pose["position"]), 
                             theta=torch.tensor(task.target_pose["orientation"])).to(cfg.device)
        
        print(f"Starting iterative push planning for object '{object_name}'")
        print(f"Goal pose: position={goal_pose.get_t()}, orientation={goal_pose.get_theta()}")
        
        iteration = 0
        try:
            while iteration < self.params.max_iterations:
                # Reset timing for this iteration
                self.timing_stats['current_iteration_times'] = []
                
                print(f"\n=== Iteration {iteration + 1}/{self.params.max_iterations} ===")
                print("--------------------------------")
                # Print all robot and object poses.
                for robot_name in self.world.robots_d.keys():
                    print(f"Robot {robot_name} pose: {self.world.get_robot_pose(robot_name).to_tensor()}")
                for object_name in self.world.objects_d.keys():
                    print(f"Object {object_name} pose: {self.world.get_object_pose(object_name).to_tensor()}")
                print("--------------------------------")
                
                # Get current object pose
                current_pose = self.world.get_object_pose(object_name).to(cfg.device)
                print(f"Current pose: position={current_pose.get_t()}, orientation={current_pose.get_theta()}")
                
                # Check if goal is reached
                if self._is_goal_reached(current_pose, goal_pose):
                    print("Goal reached! Planning complete.")
                    break
                
                # Get observation of the object
                observation = self.world.get_object_centric_observation(object_name, (cfg.OBS_PIX, cfg.OBS_PIX), visualize=False)
                
                # Compute relative transform to goal
                relative_transform = self._compute_relative_transform(current_pose, goal_pose)

                # Clip at 0.2 for the translation and orientation.
                t = torch.clamp(relative_transform.get_t(), min=-0.2, max=0.2)
                theta = torch.clamp(relative_transform.get_theta(), min=-0.3, max=0.3)
                relative_transform = Transform2(t=t, theta=theta)
                
                # Time model generation using wrapper with iteration-based seed
                contact_points_tokens, push_trajectories = self._time_function(
                    self._generate_push_plan, "Model Generation", observation, relative_transform, seed=self.world.simulator.seed + iteration)

                # Transform to world frame first
                contact_points_world, push_trajectories_world = self._transform_to_world_frame(
                    contact_points_tokens, push_trajectories, current_pose)
                
                # Filter out masked contacts (unused robots) using world frame data
                contact_points_world, push_trajectories_world = self._filter_masked_contacts(
                    contact_points_world, push_trajectories_world, contact_points_tokens)
                
                # Filter tokens and trajectories for visualization (using original format)
                filtered_tokens, filtered_trajectories = self._filter_masked_contacts_original(
                    contact_points_tokens, push_trajectories)
                
                # Visualize current iteration. This is only triggered if save_visualizations is True.
                if iteration % self.params.visualization_interval == 0 and self.params.save_visualizations:
                    self._time_function(self._visualize_iteration, "Visualization - visualize_iteration()", iteration, observation, relative_transform, 
                                            filtered_tokens, filtered_trajectories, object_name)
                
                # Time TSWAP planning using wrapper
                robot_paths = self._time_function(
                    self._plan_robot_paths, "TSWAP Planning", contact_points_world)
                
                if not robot_paths:
                    print(f"Warning: No valid robot paths found for iteration {iteration + 1}")
                    iteration += 1
                    continue
                
                # Track travel distances for freespace motion (TSWAP planning)
                self._update_travel_distances(robot_paths)
                
                # Time execution using wrapper
                world_states = self._time_function(
                    self._execute_push, "Execution", robot_paths, push_trajectories_world, visualize)
                
                # Print timing summary for this iteration
                self._print_iteration_timing_summary(iteration)
                
                # Record planning step
                self.planning_history.append({
                    'iteration': iteration,
                    'current_pose': current_pose,
                    'relative_transform': relative_transform,
                    'contact_points_world': contact_points_world,
                    'push_trajectories_world': push_trajectories_world,
                    'world_states': world_states
                })
                
                iteration += 1
        except Exception as e:
            print(f"Error during planning iteration {iteration}: {e}")
            # Clean up persistent simulator on error
            if self.params.persistent_simulator:
                self.world.cleanup_persistent_simulator()
            raise
        
        if iteration >= self.params.max_iterations:
            print(f"Warning: Reached maximum iterations ({self.params.max_iterations}) without reaching goal.")
        
        # Clean up persistent simulator
        if self.params.persistent_simulator:
            self.world.cleanup_persistent_simulator()
        
        # Print final timing summary and reset
        self._print_and_reset_timing()
        
        # Return the final robot trajectories (empty for now since execution is done)
        final_poses = {name: self.world.get_robot_pose(name).to_tensor() for name in self.world.robots_d.keys()}
        return {name: pose.unsqueeze(0).unsqueeze(0) for name, pose in final_poses.items()}
    
    def _plan_action_trajs_a_star(self, task: 'Task', visualize: bool = False) -> Dict[str, torch.Tensor]:
        """
        Plan the next actions for the given task using object-centric A*.
        :param task: The task to plan for
        :return: A dictionary mapping robot names to their planned trajectories (B, H, dim(q) + dim(q_dot))
        """
        if not isinstance(task, ManipulationTask):
            raise ValueError("IterativePushPlanner only supports ManipulationTask")
        
        object_name = task.object_id
        goal_pose = Transform2(t=torch.tensor(task.target_pose["position"]), 
                             theta=torch.tensor(task.target_pose["orientation"])).to(cfg.device)
                
        # Get current object pose.
        current_pose = self.world.get_object_pose(object_name).to(cfg.device)
        iteration = 0
        try:
            while iteration < self.params.max_iterations:
                time_start_iteration = time.time()
                # Reset timing for this iteration
                self.timing_stats['current_iteration_times'] = []
                
                current_pose = self.world.get_object_pose(object_name).to(cfg.device)
                print(f"[A*] Current pose: position={current_pose.get_t()}, orientation={current_pose.get_theta()}")
                print(f"[A*] Goal pose: position={goal_pose.get_t()}, orientation={goal_pose.get_theta()}")

                # Check if goal is reached.
                if self._is_goal_reached(current_pose, goal_pose):
                    print("Goal already reached! No planning needed.")
                    final_poses = {name: self.world.get_robot_pose(name).to_tensor() for name in self.world.robots_d.keys()}
                    return {name: pose.unsqueeze(0).unsqueeze(0) for name, pose in final_poses.items()}

                # Create object-centric A* planner (needed for saving output later)
                object_centric_a_star_planner = ObjectCentricAStarPlanner(
                    self.world, 
                    goal_tolerance=0.15,
                    weight=2.0  # Weight for weighted A*
                )
                
                # Get current robot poses
                robot_names = list(self.world.robots_d.keys())
                starts = {}
                for name in robot_names:
                    pose = self.world.get_robot_pose(name)
                    starts[name] = {
                        "x": pose.get_t()[0].item(),
                        "y": pose.get_t()[1].item(),
                        "theta": pose.get_theta().item()
                    }
                
                # Prepare start and goal object configurations
                start_objects = {
                    object_name: {
                        "x": current_pose.get_t()[0].item(),
                        "y": current_pose.get_t()[1].item(),
                        "theta": current_pose.get_theta().item()
                    }
                }
                
                goal_objects = {
                    object_name: {
                        "x": goal_pose.get_t()[0].item(),
                        "y": goal_pose.get_t()[1].item(),
                        "theta": goal_pose.get_theta().item()
                    }
                }
                
                # Time A* planning using wrapper
                paths = self._time_function(
                    object_centric_a_star_planner.plan, "A* Planning",
                    starts, start_objects, goal_objects, goal_tolerance=0.15
                )

                if not paths:
                    print("No path found with object-centric A*!")
                    final_poses = {name: self.world.get_robot_pose(name).to_tensor() for name in self.world.robots_d.keys()}
                    return {name: pose.unsqueeze(0).unsqueeze(0) for name, pose in final_poses.items()}
                
                # Extract object path if available
                object_path = None
                if object_name in paths:
                    object_path = paths[object_name]
                    print(f"Object path has {len(object_path)} waypoints")
                    
                    # Save planner output for visualization
                    if self.params.save_visualizations:
                        self._time_function(
                            self._save_planner_visualization, "Visualization",
                            object_centric_a_star_planner, object_name, object_path, 
                            start_objects, goal_objects
                        )
                else:
                    print(f"No path found with object-centric A*! For object name: {object_name}")
                    raise ValueError("No path found with object-centric A*!")
                # raise ValueError("Stop here")
                # Compute the relative transform between the current and first waypoint of the object path.
                a_star_idx = min(3, len(object_path) - 1)
                relative_transform = self._compute_relative_transform(current_pose, Transform2(t=torch.tensor([object_path[a_star_idx][0], object_path[a_star_idx][1]], device=cfg.device), theta=torch.tensor([object_path[a_star_idx][2]], device=cfg.device)))

                # ====================
                # Generate push plan.
                # ====================

                # Get observation of the object
                observation = self.world.get_object_centric_observation(object_name, (cfg.OBS_PIX, cfg.OBS_PIX), visualize=False)
                
                # Clip at 0.2 for the translation and orientation.
                t = torch.clamp(relative_transform.get_t(), min=-0.2, max=0.2)
                theta = torch.clamp(relative_transform.get_theta(), min=-0.3, max=0.3)
                relative_transform = Transform2(t=t, theta=theta)
                print(f"Clipped relative transform: position={relative_transform.get_t()}, orientation={relative_transform.get_theta()}")
                
                # Time model generation using wrapper with iteration-based seed
                contact_points_tokens, push_trajectories = self._time_function(
                    self._generate_push_plan, "Model Generation", observation, relative_transform, seed=self.world.simulator.seed + iteration)

                # Transform to world frame
                contact_points_world, push_trajectories_world = self._transform_to_world_frame(
                    contact_points_tokens, push_trajectories, current_pose)

                # Filter out masked contacts (unused robots) using world frame data
                contact_points_world, push_trajectories_world = self._filter_masked_contacts(
                    contact_points_world, push_trajectories_world, contact_points_tokens)
            
                # Visualize current iteration
                if iteration % self.params.visualization_interval == 0 and self.params.save_visualizations:
                    self._visualize_iteration(iteration, observation, relative_transform, 
                                              contact_points_tokens, push_trajectories, object_name)
                
                # Time TSWAP planning using wrapper
                robot_paths = self._time_function(
                    self._plan_robot_paths, "TSWAP Planning", contact_points_world)
                
                if not robot_paths:
                    print(f"Warning: No valid robot paths found for iteration {iteration + 1}")
                    iteration += 1
                    continue
                
                # Track travel distances for freespace motion (TSWAP planning)
                self._update_travel_distances(robot_paths)
                
                # Time execution using wrapper
                world_states = self._time_function(
                    self._execute_push, "Execution", robot_paths, push_trajectories_world, visualize)
                
                # Print timing summary for this iteration
                self._print_iteration_timing_summary(iteration)
                
                # Record planning step
                self.planning_history.append({
                    'iteration': iteration,
                    'current_pose': current_pose,
                    'relative_transform': relative_transform,
                    'contact_points_world': contact_points_world,
                    'push_trajectories_world': push_trajectories_world,
                    'world_states': world_states
                })
                
                iteration += 1
                print(f"A* planning took {time.time() - time_start_iteration} seconds")

        except Exception as e:
            print(f"Error during planning iteration {iteration}: {e}")
            # Clean up persistent simulator on error
            if self.params.persistent_simulator:
                self.world.cleanup_persistent_simulator()
        
        if iteration >= self.params.max_iterations:
            print(f"Warning: Reached maximum iterations ({self.params.max_iterations}) without reaching goal.")
        
        # Clean up persistent simulator
        if self.params.persistent_simulator:
            self.world.cleanup_persistent_simulator()
        
        # Print final timing summary and reset
        self._print_and_reset_timing()
        
        # Return the final robot trajectories (empty for now since execution is done)
        final_poses = {name: self.world.get_robot_pose(name).to_tensor() for name in self.world.robots_d.keys()}
        return {name: pose.unsqueeze(0).unsqueeze(0) for name, pose in final_poses.items()}

    def plan_action_trajs(self, task: 'Task', visualize: bool = False) -> Dict[str, torch.Tensor]:
        if self.params.planner_type == "a_star":
            return self._plan_action_trajs_a_star(task, visualize)
        elif self.params.planner_type == "interpolate":
            return self._plan_action_trajs_interpolate(task, visualize)
        else:
            raise ValueError(f"Invalid planner type: {self.params.planner_type}")
    
    def get_planning_summary(self) -> Dict:
        """Get a summary of the planning process."""
        if not self.planning_history:
            return {"status": "No planning performed"}
        
        initial_pose = self.planning_history[0]['current_pose']
        final_pose = self.planning_history[-1]['current_pose']
        
        # Calculate total travel distance
        total_travel_distance = sum(self.timing_stats['total_travel_distances'].values()) if self.timing_stats['total_travel_distances'] else 0.0
        
        # Calculate push accuracy metrics
        num_pushes = len(self.timing_stats['push_accuracy_errors'])
        avg_push_position_error = self.timing_stats['total_push_accuracy_position'] / num_pushes if num_pushes > 0 else 0.0
        avg_push_orientation_error = self.timing_stats['total_push_accuracy_orientation'] / num_pushes if num_pushes > 0 else 0.0
        
        return {
            "total_iterations": len(self.planning_history),
            "initial_pose": {
                "position": initial_pose.get_t().tolist(),
                "orientation": initial_pose.get_theta().item()
            },
            "final_pose": {
                "position": final_pose.get_t().tolist(),
                "orientation": final_pose.get_theta().item()
            },
            "total_displacement": torch.norm(final_pose.get_t() - initial_pose.get_t()).item(),
            "total_rotation": torch.abs(final_pose.get_theta() - initial_pose.get_theta()).item(),
            "robot_travel_distances": self.timing_stats['total_travel_distances'].copy(),
            "total_travel_distance": total_travel_distance,
            "push_accuracy": {
                "num_pushes": num_pushes,
                "avg_position_error": avg_push_position_error,
                "avg_orientation_error": avg_push_orientation_error,
                "total_position_error": self.timing_stats['total_push_accuracy_position'],
                "total_orientation_error": self.timing_stats['total_push_accuracy_orientation'],
                "push_errors": self.timing_stats['push_accuracy_errors'].copy()
            }
        }


def main():
    """Example usage of the IterativePushPlanner."""
    # ====================
    # Create the world.
    # ====================
    world = World(size=(1.0, 1.0), resolution=0.05, dt=0.1)

    # Create robots
    robot_1 = RobotDisk("robot_1", radius=cfg.robot_radius)
    robot_2 = RobotDisk("robot_2", radius=cfg.robot_radius)
    robot_3 = RobotDisk("robot_3", radius=cfg.robot_radius)

    # Create obstacles
    obstacle_1 = ObstacleCircle("obstacle_1", radius=0.4)

    # Create object
    object_1 = ObjectRectangle("object_1", width=0.2, height=0.4)

    # Add to world
    world.add_robot(robot_1, Transform2(t=torch.tensor([-0.7, 0.4]), theta=torch.tensor([0.0])))
    world.add_robot(robot_2, Transform2(t=torch.tensor([-0.7, 0.0]), theta=torch.tensor([0.0])))
    world.add_robot(robot_3, Transform2(t=torch.tensor([-0.7, -0.4]), theta=torch.tensor([0.0])))
    world.add_obstacle(obstacle_1, Transform2(t=torch.tensor([0.0, 0.9]), theta=torch.tensor([0.0])))
    world.add_object(object_1, Transform2(t=torch.tensor([0.0, 0.0]), theta=torch.tensor([0.0])))

    # ====================
    # Create the planner.
    # ====================
    planner_params = {
        "max_iterations": 15,
        "goal_tolerance_position": 0.05,
        "goal_tolerance_orientation": 0.1,
        "min_push_distance": 0.01,
        "visualization_interval": 2,
        "save_visualizations": True,
        "output_dir": "output/iterative_push_example"
    }
    
    planner = IterativePushPlanner(world, planner_params)

    # ====================
    # Create the task.
    # ====================
    goal_pose = Transform2(t=torch.tensor([0.3, 0.2]), theta=torch.tensor([0.5]))
    task = ManipulationTask()
    task.object_id = "object_1"
    task.target_pose = {
        "position": goal_pose.get_t().tolist(),
        "orientation": goal_pose.get_theta().item()
    }

    # ====================
    # Execute the plan.
    # ====================
    print("Starting iterative push planning...")
    start_time = time.time()
    
    final_trajectories = planner.plan_action_trajs(task)
    
    end_time = time.time()
    print(f"Planning completed in {end_time - start_time:.2f} seconds")
    
    # Print summary
    summary = planner.get_planning_summary()
    print("\nPlanning Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main() 