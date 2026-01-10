"""
Multi-Object Multi-Robot Planner for simultaneous manipulation of multiple objects.

This planner extends the single-object iterative push planner to handle multiple objects
by using object-level GSPI planning followed by learned model-based push execution.
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
from gcocpp import ObjectCentricAStarPlanner, GSPIPlanner, ObjectGSPIPlanner
from gco.utils.data_vis_utils import visualize_transformed_mask, visualize_batch_denoising, visualize_push_trajectory, visualize_batch
from gco.models.contact_push_model import ContactTrajectoryModel
from gco.planners.multi.multi_robot_planner import MultiRobotPlanner
from gco.tasks.tasks import TaskType, ManipulationTask, MultiObjectManipulationTask

# Type checking imports
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from gco.tasks.tasks import Task


@dataclass
class MultiObjectPlannerParams:
    """Parameters for the multi-object planner."""
    max_iterations: int = 10
    goal_tolerance_position: float = 0.15  # meters
    goal_tolerance_orientation: float = 0.5  # radians
    min_push_distance: float = 0.02  # minimum distance to consider a push worthwhile
    visualization_interval: int = 5  # visualize every N iterations
    visualize_planning: bool = False
    save_visualizations: bool = False
    output_dir: str = "output/multi_object_push"
    model_checkpoint: Optional[Path] = None
    model_type: str = "discrete"
    persistent_simulator: bool = True  # Use a single persistent MuJoCo instance
    object_planning_horizon: int = 3  # Number of object moves to plan ahead
    object_radius: float = 0.5  # Radius for objects in object-level planning


class MultiObjectPlanner(MultiRobotPlanner):
    """
    Multi-object multi-robot planner that simultaneously manipulates multiple objects.
    
    This planner:
    1. Uses object-level GSPI to plan the first m moves for all objects
    2. For each object move, uses the learned model to generate push trajectories
    3. Uses robot-level GSPI to plan robot paths to contact points
    4. Executes the combined trajectories
    5. Repeats until all objects reach their goals
    """
    
    def __init__(self, world: World, params: Dict[str, any]):
        super().__init__(world, params)
        self.params = MultiObjectPlannerParams(**params)
        
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
            'current_iteration_times': [],
            'all_iteration_times': [],
            'total_travel_distances': {},
            'current_iteration_distances': {},
            'push_accuracy_errors': [],
            'total_push_accuracy_position': 0.0,
            'total_push_accuracy_orientation': 0.0,
            'current_push_accuracy_position': 0.0,
            'current_push_accuracy_orientation': 0.0
        }
        
        # Initialize persistent ObjectGSPI planner
        self.object_planner = None
        self.object_targets_initialized = False
    
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
    
    def _initialize_model(self):
        """Initialize the contact trajectory model based on model type."""
        # Load the model weights
        if self.params.model_checkpoint is None:
            raise ValueError("Model checkpoint path is required.")
        
        model_checkpoint_path = self.params.model_checkpoint
        
        # Load the model checkpoint
        print(f"Loading model checkpoint from: {model_checkpoint_path}")        
        # Initialize discrete model
        # Note: vocab_size is cfg.V + 1 to match the trained model checkpoint
        # (cfg.V accounts for spatial tokens, +1 for mask token compatibility)
        vocab_size = cfg.V + 1
        self.cogen_model = ContactTrajectoryModel(vocab_size=vocab_size).to(cfg.device)
        
        # Load checkpoint
        self.cogen_model.load_state_dict(torch.load(model_checkpoint_path, map_location=cfg.device))
        
        print(f"Loaded discrete contact trajectory model from: {model_checkpoint_path}")
            
        self.cogen_model.eval()
    
    def _initialize_object_planner(self, task: MultiObjectManipulationTask):
        """Initialize the persistent ObjectGSPI planner with object targets."""
        if self.object_planner is None:
            # Create ObjectGSPI planner
            heuristic_params = {
                "grid_resolution": "0.05",  # Fixed resolution for better pathfinding
                "max_distance_meters": str(10.0),  # Increased to handle larger world distances
                "object_radius": str(self.params.object_radius),
                "heuristic_type": "bwd_dijkstra"  # Use euclidean to avoid precomputation issues
            }
            
            self.object_planner = ObjectGSPIPlanner(
                self.world, 
                goal_tolerance= self.params.goal_tolerance_position,
                heuristic_params=heuristic_params,
                seed=self.world.simulator.seed
            )
            self.object_planner.set_verbose(False)
        
        if not self.object_targets_initialized:
            # Reset planner state for new scenario
            self.object_planner.reset_planner_state()
            
            # Convert task targets to C++ format
            object_targets = {}
            for obj_name in task.get_object_names():
                target_pose = task.get_target_pose(obj_name)
                object_targets[obj_name] = [
                    target_pose["position"][0],
                    target_pose["position"][1], 
                    target_pose["orientation"]
                ]
            
            # Initialize the planner with object targets
            self.object_planner.initialize_object_targets(object_targets)
            self.object_targets_initialized = True
            
            print(f"Initialized ObjectGSPI planner with {len(object_targets)} object targets")
    
    def plan_action_trajs(self, task: 'Task', visualize: bool = False) -> Dict[str, torch.Tensor]:
        """
        Plan actions for multi-object manipulation task.
        
        Args:
            task: MultiObjectManipulationTask containing object targets
            visualize: Whether to visualize the planning process
            
        Returns:
            Dictionary mapping robot names to their final planned trajectories
        """
        if not isinstance(task, MultiObjectManipulationTask):
            raise ValueError("MultiObjectPlanner only supports MultiObjectManipulationTask")
        
        print(f"Starting multi-object manipulation planning for {len(task.get_object_names())} objects")
        
        # Initialize the persistent object planner
        self._initialize_object_planner(task)
        
        iteration = 0
        try:
            while iteration < self.params.max_iterations:
                # Reset timing for this iteration
                self.timing_stats['current_iteration_times'] = []
                
                print(f"\n=== Multi-Object Iteration {iteration + 1}/{self.params.max_iterations} ===")
                translation_error_l = []
                orientation_error_l = []
                for obj_name in task.get_object_names():
                    current_pose = self.world.get_object_pose(obj_name)
                    target_pose = task.get_target_pose(obj_name)
                    dx = current_pose.get_t()[0] - target_pose["position"][0]
                    dy = current_pose.get_t()[1] - target_pose["position"][1]
                    dtheta = current_pose.get_theta() - target_pose["orientation"]
                    translation_error_l.append(torch.norm(torch.tensor([dx, dy])))
                    orientation_error_l.append(torch.abs(dtheta))
                print(f"=== Translation errors: {translation_error_l} ===")
                print(f"=== Orientation errors: {orientation_error_l} ===")
                
                # Check if all objects have reached their goals
                if task.is_complete(self.world):
                    print("All objects reached their goals! Planning complete.")
                    break
                
                # Plan object-level moves using persistent ObjectGSPI
                object_moves = self._plan_object_moves_persistent(task)
                
                if not object_moves:
                    print("No valid object moves found") ####################
                    iteration += 1
                    continue
                
                # Execute object moves using learned model and robot planning
                self._execute_object_moves(object_moves, task, seed=iteration, visualize=visualize)
                
                iteration += 1
                
        except Exception as e:
            print(f"Error during multi-object planning iteration {iteration}: {e}")
            import traceback
            traceback.print_exc()
            # Clean up persistent simulator on error
            if self.params.persistent_simulator:
                self.world.cleanup_persistent_simulator()
            raise
        
        if iteration >= self.params.max_iterations:
            print(f"Warning: Reached maximum iterations ({self.params.max_iterations}) without completing all goals.")
        
        # Clean up persistent simulator
        if self.params.persistent_simulator:
            self.world.cleanup_persistent_simulator()
        
        # Return the final robot trajectories
        final_poses = {name: self.world.get_robot_pose(name).to_tensor() for name in self.world.robots_d.keys()}
        return {name: pose.unsqueeze(0).unsqueeze(0) for name, pose in final_poses.items()}
    
    def _plan_object_moves_persistent(self, task: MultiObjectManipulationTask) -> Dict[str, List]:
        """
        Plan object-level moves using persistent ObjectGSPI planner.
        
        Args:
            task: MultiObjectManipulationTask
            
        Returns:
            Dictionary mapping object names to their planned paths
        """
        print("Planning object-level moves using persistent ObjectGSPI...")
        
        # Get current object poses
        object_names = task.get_object_names()
        current_positions = {}
        
        for obj_name in object_names:
            current_pose = self.world.get_object_pose(obj_name)
            current_positions[obj_name] = [
                current_pose.get_t()[0].item(),
                current_pose.get_t()[1].item(),
                current_pose.get_theta().item()
            ]
        
        # Get next moves from persistent planner
        object_moves_raw = self.object_planner.get_next_object_moves(
            current_positions, 
            horizon=self.params.object_planning_horizon
        )
        
        # Get current assignments from the planner
        current_assignments = self.object_planner.get_current_assignments()
        
        # Update task with current assignments
        task.update_assignments(current_assignments)
                
        # Convert to expected format
        object_moves = {}
        for obj_name, moves_list in object_moves_raw.items():
            if moves_list:  # If there are moves
                # Convert to list of [x, y, theta] lists
                moves = []
                for move in moves_list:
                    moves.append([move[0], move[1], move[2]])
                object_moves[obj_name] = moves
            else:
                print(f"No moves returned for object {obj_name}")
        
        return object_moves

    def _execute_object_moves(self, object_moves: Dict[str, List], task: MultiObjectManipulationTask, seed: int = 42, visualize: bool = False):
        """
        Execute object moves using learned model and robot planning with batch inference.
        
        Args:
            object_moves: Dictionary mapping object names to their planned paths
            task: MultiObjectManipulationTask
            seed: Random seed for generation
            visualize: Whether to visualize the execution
        """
        print("Executing object moves using learned model with batch inference...")
        
        # Collect all object data for batch processing
        object_data = []
        object_names = []
        
        for obj_name, move_path in object_moves.items():
            if len(move_path) < 2:
                continue
                
            # Get current object pose
            current_pose = self.world.get_object_pose(obj_name).to(cfg.device)
            
            # Compute relative transform for the move
            end_pos = torch.tensor([move_path[-1][0], move_path[-1][1]], device=cfg.device)
            relative_transform = Transform2(t=end_pos - current_pose.get_t(), theta=torch.tensor([0.0], device=cfg.device))
            # Clamp x y and theta to 0.2 maximum abs.
            relative_transform.t = relative_transform.get_t().clamp(-0.2, 0.2)
            # relative_transform.theta = relative_transform.get_theta().clamp(-0.2, 0.2)

            # Add a theta motion towards the theta target.
            obj_goal = task.get_target_pose(obj_name)
            theta_target = torch.tensor([obj_goal["orientation"]], device=cfg.device)
            theta_diff = theta_target - current_pose.get_theta()
            if theta_diff > np.pi:
                theta_diff -= 2 * np.pi
            if theta_diff < -np.pi:
                theta_diff += 2 * np.pi
            theta_diff = theta_diff.clamp(-0.3, 0.3)

            relative_transform = Transform2(t=relative_transform.get_t(), theta=theta_diff)
            print("\n", f"Robot {obj_name} relative transform: {relative_transform}","\n")            
            
            # Check if object is at its goal using proper goal tolerance parameters
            current_position = current_pose.get_t()
            current_orientation = current_pose.get_theta()
            target_position = torch.tensor(obj_goal["position"], device=cfg.device)
            target_orientation = torch.tensor([obj_goal["orientation"]], device=cfg.device)
            
            position_error = torch.norm(current_position - target_position).item()
            orientation_error = abs(current_orientation - target_orientation).item()
            
            # Normalize orientation error to [-π, π]
            if orientation_error > np.pi:
                orientation_error = 2 * np.pi - orientation_error
            
            if position_error < self.params.goal_tolerance_position and orientation_error < self.params.goal_tolerance_orientation:
                print(f"Robot {obj_name} is at its goal (pos_error: {position_error:.3f}m < {self.params.goal_tolerance_position}m, "
                      f"orient_error: {orientation_error:.3f}rad < {self.params.goal_tolerance_orientation}rad), releasing it.")
            # if relative_transform.get_t().norm() < 0.05 and abs(relative_transform.get_theta()) < 0.05:  # TODO: this is an arbitrary threshold.
            #     print(f"Robot {obj_name} got small relative transform, not pushing it.")
                continue

            # Get observation of the object
            observation = self.world.get_object_centric_observation(obj_name, (cfg.OBS_PIX, cfg.OBS_PIX), visualize=False)
            
            object_data.append({
                'name': obj_name,
                'observation': observation,
                'relative_transform': relative_transform,
                'current_pose': current_pose
            })
            object_names.append(obj_name)
        
        if not object_data:
            print("No valid object moves to execute")
            return
        
        # Sort objects by distance to their goals (furthest first) - needed for budget strategy
        object_data_with_distances = []
        for original_idx, obj_data in enumerate(object_data):
            obj_name = obj_data['name']
            current_pose = obj_data['current_pose']
            
            # Get target pose from task
            target_pose = task.get_target_pose(obj_name)
            target_position = torch.tensor(target_pose["position"], device=cfg.device)
            current_position = current_pose.get_t()
            
            # Calculate distance to goal
            distance_to_goal = torch.norm(current_position - target_position).item()
            
            object_data_with_distances.append((distance_to_goal, original_idx, obj_data))
        
        # Sort by distance (furthest first)
        object_data_with_distances.sort(key=lambda x: x[0], reverse=True)
        print(f"Object processing order (furthest to closest to goal): {[(obj_data['name'], f'{dist:.3f}m') for dist, _, obj_data in object_data_with_distances]}")

        print("Requesting contacts and trajectories for N =", len(object_data))
        
        # Batch inference: generate push plans for all objects simultaneously with priority strategy
        batch_contact_points_tokens, batch_push_trajectories, generation_steps = self._generate_batch_push_plan(
            object_data, budget_strategy="equal", object_data_with_distances=object_data_with_distances, seed=seed)
        
        print("Received contacts and trajectories for N =", batch_contact_points_tokens.shape[0])

        # Visualize current iteration if verbosity is requested
        if self.params.save_visualizations:
            # Get current iteration from planning history length
            iteration = len(self.planning_history)
            self._visualize_iteration(iteration, object_data, batch_contact_points_tokens, batch_push_trajectories, generation_steps)

            # Also save the generation intermediate steps to files.
            print(f"Saving generation intermediate steps to files: shape= {batch_contact_points_tokens.shape} {os.path.join(self.params.output_dir, f'iteration_{iteration:03d}_contact_points_tokens.pt')}")
            torch.save(batch_contact_points_tokens, os.path.join(self.params.output_dir, f"iteration_{iteration:03d}_contact_points_tokens.pt"))
            print(f"Saving generation intermediate steps to files: shape= {batch_push_trajectories.shape} {os.path.join(self.params.output_dir, f'iteration_{iteration:03d}_push_trajectories.pt')}")
            torch.save(batch_push_trajectories, os.path.join(self.params.output_dir, f"iteration_{iteration:03d}_push_trajectories.pt"))
        
        # Collect all contact points and push trajectories from all objects
        all_contact_points = []
        all_push_trajectories = []
        robot_names = list(self.world.robots_d.keys())
        available_robots = set(robot_names)
        
        for i, (distance_to_goal, original_idx, obj_data) in enumerate(object_data_with_distances):
            obj_name = obj_data['name']
            current_pose = obj_data['current_pose']
            # Extract this object's results from batch using original index
            contact_points_tokens = batch_contact_points_tokens[original_idx:original_idx+1]  # Keep batch dimension
            push_trajectories = batch_push_trajectories[original_idx:original_idx+1]  # Keep batch dimension
            
            # Transform to world frame
            contact_points_world, push_trajectories_world = self._transform_to_world_frame(
                contact_points_tokens, push_trajectories, current_pose)
            
            # Filter out masked contacts
            contact_points_world, push_trajectories_world = self._filter_masked_contacts(
                contact_points_world, push_trajectories_world, contact_points_tokens)
            
            # print(f"Object {obj_name} is at {current_pose.get_t()} and has {contact_points_world.shape[1]} valid contact points: {contact_points_world.reshape(1, -1, 2)}")
            
            # Add all valid contact points and trajectories to the global pool with priority weighting
            if contact_points_world.shape[1] > 0:
                # Calculate priority weight based on distance to goal
                # Objects further from goal get higher priority (more robots assigned)
                priority_weight = max(1.0, distance_to_goal * 2.0)  # Scale factor for priority
                
                for j in range(contact_points_world.shape[1]):
                    # Only add if not already in all_contact_points. This should be equality up to a small epsilon.
                    contact_point = contact_points_world[:, j, :].squeeze()
                    push_trajectory = push_trajectories_world[:, j, :].squeeze(0)
                    
                    # Check for duplicates
                    is_duplicate = False
                    for existing_contact in all_contact_points:
                        if torch.allclose(contact_point, existing_contact, atol=1e-6):
                            is_duplicate = True
                            break
                    
                    if not is_duplicate:
                        # Add contact point and trajectory with priority weighting
                        # For now, we'll add them normally but the assignment algorithm will prioritize
                        all_contact_points.append(contact_point)
                        all_push_trajectories.append(push_trajectory)
                        
        
        # Assign robots to contact points using optimal assignment
        # Since contact points are already generated with priority-based budget, we just need optimal assignment
        robot_contact_assignments = {}
        if available_robots:
            # Get current robot positions for distance calculation
            robot_positions = {}
            for robot_name in available_robots:
                robot_pose = self.world.get_robot_pose(robot_name)
                robot_positions[robot_name] = robot_pose.get_t()
            
            if all_contact_points:
                # Convert to tensors for easier computation
                all_contact_points_tensor = torch.stack(all_contact_points, dim=0)  # (M, 2)
                all_push_trajectories_tensor = torch.stack(all_push_trajectories, dim=0)  # (M, H, 3)
                
                # Use optimal assignment since contact points are already prioritized
                robot_contact_assignments = self._greedy_robot_contact_assignment(
                    robot_positions, all_contact_points_tensor, all_push_trajectories_tensor)
            
            # Create goals for unassigned robots using a simple policy
            unassigned_robots = set(available_robots) - set(robot_contact_assignments.keys())
            if unassigned_robots:
                print(f"Creating goals for {len(unassigned_robots)} unassigned robots")
                self._create_goals_for_unassigned_robots(unassigned_robots, robot_contact_assignments)
        
        # Plan robot paths for all assigned contacts
        if robot_contact_assignments:
            self._plan_and_execute_robot_trajectories(robot_contact_assignments, visualize)
    
    def _plan_and_execute_robot_trajectories(self, robot_contact_assignments: Dict[str, Dict], visualize: bool = False):
        """
        Plan and execute trajectories for robots assigned to contact points.
        
        Args:
            robot_contact_assignments: Dictionary mapping robot names to their contact assignments
            visualize: Whether to visualize the execution
        """
        print("Planning and executing robot trajectories...")
        
        # Collect robot starts and goals
        robot_starts = {}
        robot_goals = {}
        
        for robot_name, assignment in robot_contact_assignments.items():
            robot_starts[robot_name] = self.world.get_robot_pose(robot_name).to_tensor()
            contact_point = assignment['contact_point']  # (2,)
            robot_goals[robot_name] = torch.cat((
                contact_point,  # (x, y)
                torch.tensor([0.0], device=cfg.device)  # theta
            ), dim=-1)
        
        # Plan robot paths using GSPI
        if robot_starts and robot_goals:
            robot_paths = self._plan_robot_paths_with_optimization(robot_starts, robot_goals)
            
            # Execute the trajectories
            if robot_paths:
                # Reassign push trajectories based on where robots actually end up after GSPI
                reassigned_assignments = self._reassign_trajectories_after_planning(
                    robot_paths, robot_contact_assignments)
                self._execute_push_trajectories(robot_paths, reassigned_assignments, visualize)
    
    def _reassign_trajectories_after_planning(self, robot_paths: Dict[str, List], 
                                            original_assignments: Dict[str, Dict]) -> Dict[str, Dict]:
        """
        Reassign push trajectories after GSPI planning based on where robots actually end up.
        
        GSPI may swap robots between contact points, so we need to reassign the push trajectories
        to match the actual end positions of the robots.
        
        Args:
            robot_paths: Robot paths from GSPI planning
            original_assignments: Original robot-to-contact-point assignments
            
        Returns:
            Dictionary mapping robot names to their reassigned contact assignments
        """
        
        # Collect all available contact points and push trajectories
        all_contact_points = []
        all_push_trajectories = []
        
        for robot_name, assignment in original_assignments.items():
            all_contact_points.append(assignment['contact_point'])
            all_push_trajectories.append(assignment['push_trajectory'])
        
        if not all_contact_points:
            return {}
        
        # Convert to tensors for easier computation
        all_contact_points_tensor = torch.stack(all_contact_points, dim=0)  # (M, 2)
        all_push_trajectories_tensor = torch.stack(all_push_trajectories, dim=0)  # (M, H, 3)
        
        # Get actual end positions of robots after GSPI
        robot_end_positions = {}
        for robot_name, robot_path in robot_paths.items():
            robot_end_positions[robot_name] = torch.tensor(robot_path[-1][:2], device=cfg.device)
        
        # Use optimal assignment to avoid conflicts
        reassigned_assignments = self._greedy_robot_contact_assignment(
            robot_end_positions, all_contact_points_tensor, all_push_trajectories_tensor)
        
        # Create goals for unassigned robots after GSPI planning
        all_robot_names = set(robot_end_positions.keys())
        assigned_robot_names = set(reassigned_assignments.keys())
        unassigned_robots = all_robot_names - assigned_robot_names
        
        if unassigned_robots:
            self._create_goals_for_unassigned_robots(unassigned_robots, reassigned_assignments)
        
        # Log the reassignments
        for robot_name, assignment in reassigned_assignments.items():
            contact_point = assignment['contact_point']
            robot_pos = robot_end_positions[robot_name]
            distance = torch.norm(contact_point - robot_pos).item()
        
        return reassigned_assignments
    
    def _greedy_robot_contact_assignment(self, robot_positions: Dict[str, torch.Tensor], 
                                        contact_points: torch.Tensor, 
                                        push_trajectories: torch.Tensor) -> Dict[str, Dict]:
        """
        Perform greedy assignment of robots to contact points to avoid conflicts.
        
        This method ensures that each contact point is assigned to at most one robot,
        and each robot is assigned to at most one contact point.
        
        Args:
            robot_positions: Dictionary mapping robot names to their positions
            contact_points: Tensor of contact points (M, 2)
            push_trajectories: Tensor of push trajectories (M, H, 3)
            
        Returns:
            Dictionary mapping robot names to their assigned contact assignments
        """
        if not robot_positions or contact_points.shape[0] == 0:
            return {}
        
        # Convert robot positions to list for easier processing
        robot_names = list(robot_positions.keys())
        robot_positions_list = [robot_positions[name] for name in robot_names]
        robot_positions_tensor = torch.stack(robot_positions_list, dim=0)  # (N, 2)
        
        # Compute distance matrix between all robots and contact points
        # robot_positions_tensor: (N, 2), contact_points: (M, 2)
        # distances: (N, M)
        distances = torch.cdist(robot_positions_tensor, contact_points, p=2)
        
        # Create assignment using greedy approach with sorting
        # Sort robots by their minimum distance to any contact point
        min_distances = torch.min(distances, dim=1)[0]  # (N,)
        sorted_robot_indices = torch.argsort(min_distances)
        
        assignments = {}
        used_contact_indices = set()
        
        for robot_idx in sorted_robot_indices:
            robot_name = robot_names[robot_idx]
            robot_distances = distances[robot_idx]  # (M,)
            
            # Find the closest unused contact point
            best_contact_idx = None
            best_distance = float('inf')
            
            for contact_idx in range(contact_points.shape[0]):
                if contact_idx not in used_contact_indices:
                    distance = robot_distances[contact_idx].item()
                    if distance < best_distance:
                        best_distance = distance
                        best_contact_idx = contact_idx
            
            if best_contact_idx is not None:
                # Assign robot to contact point
                assignments[robot_name] = {
                    'contact_point': contact_points[best_contact_idx],
                    'push_trajectory': push_trajectories[best_contact_idx]
                }
                used_contact_indices.add(best_contact_idx)
        
        # Validate that no contact point is assigned to multiple robots
        assigned_contact_points = set()
        for assignment in assignments.values():
            contact_point = assignment['contact_point']
            contact_key = (contact_point[0].item(), contact_point[1].item())
            if contact_key in assigned_contact_points:
                print(f"WARNING: Duplicate contact point assignment detected: {contact_key}, skipping assignment.")
            assigned_contact_points.add(contact_key)
        
        return assignments
    
    def _create_goals_for_unassigned_robots(self, unassigned_robots: set, robot_contact_assignments: Dict[str, Dict]):
        """
        Create goals for unassigned robots using a simple policy.
        
        Policy: Place unassigned robots at y=-1, distributed along the x-axis.
        This keeps them out of the way while still giving them a goal to move towards.
        
        Args:
            unassigned_robots: Set of robot names that don't have contact point assignments
            robot_contact_assignments: Dictionary to add the new assignments to
        """
        if not unassigned_robots:
            return
        
        # Choose y position that's within world bounds and away from objects
        y_goal = 0.0
        
        # Distribute robots along x-axis
        unassigned_list = list(unassigned_robots)
        num_unassigned = len(unassigned_list)
        
        if num_unassigned == 1:
            # Single robot goes to center
            x_positions = [0.0]
        else:
            # Multiple robots distributed along x-axis
            x_positions = np.linspace(-1.0, 1.0, num_unassigned).tolist()
        
        # Create assignments for unassigned robots
        for i, robot_name in enumerate(unassigned_list):
            x_goal = x_positions[i]
            goal_position = torch.tensor([x_goal, y_goal], device=cfg.device)
            
            # Create a dummy push trajectory (just stay in place)
            dummy_push_trajectory = torch.zeros((cfg.H, 3), device=cfg.device)  # (H, 3)
            dummy_push_trajectory[:, :2] = goal_position.unsqueeze(0)  # Stay at goal position
            
            robot_contact_assignments[robot_name] = {
                'contact_point': goal_position,
                'push_trajectory': dummy_push_trajectory
            }
                

    def _execute_push_trajectories(self, robot_paths: Dict[str, List], 
                                               robot_contact_assignments: Dict[str, Dict],
                                               visualize: bool = False):
        """
        Execute push trajectories for robots.
        
        Args:
            robot_paths: Robot paths from start to contact points
            robot_contact_assignments: Robot-to-contact-point assignments
            visualize: Whether to visualize the execution
        """
        
        # Combine robot paths with push trajectories for each robot
        paths_with_push = {}
        
        for robot_name, assignment in robot_contact_assignments.items():
            if robot_name not in robot_paths:
                print(f"Warning: Robot {robot_name} not found in robot paths")
                continue
                
            push_trajectory = assignment['push_trajectory']  # (H, 3)
            contact_point = assignment['contact_point']  # (2,)
            
            # Get the robot's path
            robot_path = robot_paths[robot_name]
            
            # Translate the push trajectory to start from the robot's current position
            robot_current_pos = torch.tensor(robot_path[-1], device=cfg.device)  # [x, y, theta]
            push_traj_start = push_trajectory[0]  # First waypoint of push trajectory
            
            # Calculate translation needed to move push trajectory to start from robot position
            translation = robot_current_pos - push_traj_start
            
            # Apply translation to entire push trajectory
            push_traj_translated = push_trajectory + translation.unsqueeze(0)
            
            # Convert to list format and combine with robot path
            push_traj_list = push_traj_translated.tolist()
            paths_with_push[robot_name] = robot_path + push_traj_list
        
        # If no push trajectories were added, use original robot paths
        if not paths_with_push:
            paths_with_push = robot_paths
        
        # Densify and smooth the paths (same as IterativePushPlanner)
        ctswap_path_lengths = {k: len(v) for k, v in robot_paths.items()}
        paths_prepare = {k: v[:ctswap_path_lengths[k]] for k, v in paths_with_push.items() if k in ctswap_path_lengths}
        paths_act = {k: v[ctswap_path_lengths[k]:] for k, v in paths_with_push.items() if k in ctswap_path_lengths}
        
        # Densify trajectories
        paths_prepare = densify_all_trajectories(paths_prepare, 10)
        paths_act = densify_all_trajectories(paths_act, 20)
        
        # Combine prepare and act paths
        final_paths = {k: paths_prepare[k].tolist() + paths_act[k].tolist() for k in paths_with_push.keys() if k in paths_prepare and k in paths_act}
        
        # Convert to tensor format for execution
        final_paths_tensor = {k: torch.tensor(path)[:, :2] for k, path in final_paths.items()}
        
        # Execute trajectories in persistent simulator
        world_states = self.world.apply_trajectories(final_paths_tensor, visualize=visualize, real_time=False)
        
        # Calculate and update travel distances for this iteration
        iteration_distances = self._calculate_travel_distance(final_paths)
        for robot_name, distance in iteration_distances.items():
            if robot_name not in self.timing_stats['total_travel_distances']:
                self.timing_stats['total_travel_distances'][robot_name] = 0.0
            self.timing_stats['total_travel_distances'][robot_name] += distance
        
        # Record planning step
        self.planning_history.append({
            'robot_paths': robot_paths,
            'robot_contact_assignments': robot_contact_assignments,
            'paths_with_push': paths_with_push,
            'final_paths': final_paths,
            'world_states': world_states,
            'iteration_distances': iteration_distances
        })
        
        return world_states
    
    
    def _generate_batch_push_plan(self, object_data: List[Dict], budget_strategy: str = "equal", 
                                 object_data_with_distances: List = None, seed: int = 42) -> Tuple[torch.Tensor, torch.Tensor, List]:
        """
        Generate contact points and push trajectories for multiple objects using batch inference.
        
        Args:
            object_data: List of dictionaries containing object information
            budget_strategy: Strategy for allocating robots to objects:
                - "equal": Equal distribution of robots among objects
                - "priority": Objects further from goals get more robots (up to 3 per object)
                - "proximity": Objects closer to robot swarm get more robots (up to 3 per object)
            object_data_with_distances: List of (distance, original_idx, obj_data) tuples for priority/proximity strategies
            seed: Random seed for generation
            
        Returns:
            Tuple of (batch_contact_points_tokens, batch_push_trajectories)
        """
        batch_size = len(object_data)
        
        # Prepare batch inputs
        observations_batch = torch.stack([obj['observation'] for obj in object_data]).to(cfg.device)  # (B, H, W)
        relative_transforms_batch = torch.stack([obj['relative_transform'].to_tensor() for obj in object_data]).to(cfg.device)  # (B, 3)
        
        num_robots = len(self.world.robots_d)
        num_objects = len(object_data)
        
        # Allocate budget based on strategy
        if budget_strategy == "priority" and object_data_with_distances is not None:
            budget = self._calculate_priority_budget(object_data_with_distances, num_robots)
        elif budget_strategy == "proximity" and object_data_with_distances is not None:
            budget = self._calculate_proximity_budget(object_data_with_distances, num_robots)
        else:  # "equal" strategy (default)
            budget = self._calculate_equal_budget(num_robots, num_objects)
        
        print(f"Budget strategy: {budget_strategy}, Budget per object: {budget}")
                
        # Generate using the model with batch inference
        with torch.no_grad():
            sol_l = self.cogen_model.generate(
                observations_batch, 
                relative_transforms_batch, 
                budget=budget, 
                seed=seed, 
                smoothness_weight=cfg.smoothness_weight
            )
        
        # Extract contact points and trajectories from the final generation step.
        batch_contact_points_tokens = sol_l[-1][0]  # (B, N*2)
        batch_push_trajectories = sol_l[-1][1]      # (B, N, H, 2)
        
        return batch_contact_points_tokens, batch_push_trajectories, sol_l
    
    def _calculate_equal_budget(self, num_robots: int, num_objects: int) -> torch.Tensor:
        """
        Calculate equal budget allocation for all objects.
        
        Args:
            num_robots: Total number of available robots
            num_objects: Number of objects
            
        Returns:
            Tensor of budget per object
        """
        base = num_robots // num_objects
        rem = num_robots % num_objects
        
        budget = torch.full((num_objects,), base, device=cfg.device, dtype=torch.long)
        if rem:
            # idx = torch.randperm(num_objects, device=cfg.device)[:rem]
            idx = torch.arange(num_objects, device=cfg.device)[:rem]
            budget[idx] += 1
        
        # Clip at 3.
        budget = budget.clamp(0, 3)
        return budget
    
    def _calculate_priority_budget(self, object_data_with_distances: List, num_robots: int) -> torch.Tensor:
        """
        Calculate priority-based budget allocation for objects.
        
        Objects further from their goals get more robots (up to max 3 per object).
        This ensures that the most challenging objects get the most help.
        
        Args:
            object_data_with_distances: List of (distance, original_idx, obj_data) tuples
            num_robots: Total number of available robots
            
        Returns:
            Tensor of budget per object in original order
        """
        num_objects = len(object_data_with_distances)
        budget = torch.zeros((num_objects,), device=cfg.device, dtype=torch.long)
        
        # Calculate priority weights for each object based on distance to goal
        object_priorities = {}
        for distance_to_goal, original_idx, obj_data in object_data_with_distances:
            obj_name = obj_data['name']
            # Higher distance = higher priority = more robots assigned
            object_priorities[obj_name] = max(1.0, distance_to_goal * 2.0)
        
        # Distribute robots among objects based on priority
        total_priority = sum(object_priorities.values())
        
        # Calculate initial allocation based on priority
        for distance_to_goal, original_idx, obj_data in object_data_with_distances:
            obj_name = obj_data['name']
            priority = object_priorities[obj_name]
            
            # Allocate robots proportionally to priority, but cap at 3 per object
            allocated_robots = min(3, max(1, int(num_robots * priority / total_priority)))
            budget[original_idx] = allocated_robots
        
        # Adjust if total exceeds available robots
        total_allocated = budget.sum().item()
        if total_allocated > num_robots:
            # Scale down proportionally while maintaining relative priorities
            scale_factor = num_robots / total_allocated
            budget = (budget.float() * scale_factor).long()
            
            # Ensure each object gets at least 1 robot if possible
            remaining_robots = num_robots - budget.sum().item()
            if remaining_robots > 0:
                # Give remaining robots to highest priority objects
                priority_order = sorted(object_data_with_distances, key=lambda x: x[0], reverse=True)
                for distance_to_goal, original_idx, obj_data in priority_order:
                    if remaining_robots <= 0:
                        break
                    if budget[original_idx] < 3:  # Don't exceed max of 3
                        budget[original_idx] += 1
                        remaining_robots -= 1
        
        # Ensure we don't exceed available robots
        while budget.sum().item() > num_robots:
            # Reduce from lowest priority objects first
            min_budget_idx = budget.argmin().item()
            if budget[min_budget_idx] > 1:
                budget[min_budget_idx] -= 1
            else:
                break
        
        print(f"Priority budget allocation: {budget.tolist()}")
        return budget
    
    def _calculate_proximity_budget(self, object_data_with_distances: List, num_robots: int) -> torch.Tensor:
        """
        Calculate proximity-based budget allocation for objects.
        
        Objects that are closer to the robot swarm get more robots assigned to them.
        This strategy prioritizes efficiency by assigning more robots to objects that are
        easier to reach.
        
        Args:
            object_data_with_distances: List of (distance, original_idx, obj_data) tuples
            num_robots: Total number of available robots
            
        Returns:
            Tensor of budget per object in original order
        """
        num_objects = len(object_data_with_distances)
        budget = torch.zeros((num_objects,), device=cfg.device, dtype=torch.long)
        
        # Get robot positions for proximity calculation
        robot_positions = []
        for robot_name in self.world.robots_d.keys():
            robot_pose = self.world.get_robot_pose(robot_name)
            robot_positions.append(robot_pose.get_t())
        
        if not robot_positions:
            # Fallback to equal distribution if no robots
            return self._calculate_equal_budget(num_robots, num_objects)
        
        robot_positions_tensor = torch.stack(robot_positions, dim=0)  # (N, 2)
        
        # Calculate proximity scores for each object
        object_proximities = {}
        for distance_to_goal, original_idx, obj_data in object_data_with_distances:
            obj_name = obj_data['name']
            current_pose = obj_data['current_pose']
            object_position = current_pose.get_t()
            
            # Calculate minimum distance from any robot to this object
            distances_to_robots = torch.norm(robot_positions_tensor - object_position.unsqueeze(0), dim=1)
            min_distance_to_robots = torch.min(distances_to_robots).item()
            
            # Calculate average distance from all robots to this object
            avg_distance_to_robots = torch.mean(distances_to_robots).item()
            
            # Proximity score: closer objects get higher scores
            # Use both min and avg distance for a balanced approach
            proximity_score = 1.0 / (min_distance_to_robots + 0.1) + 0.5 / (avg_distance_to_robots + 0.1)
            object_proximities[obj_name] = proximity_score
            
            # print(f"Object {obj_name}: min_dist={min_distance_to_robots:.3f}m, avg_dist={avg_distance_to_robots:.3f}m, proximity_score={proximity_score:.3f}")
        
        # Distribute robots among objects based on proximity
        total_proximity = sum(object_proximities.values())
        
        # Calculate initial allocation based on proximity
        for distance_to_goal, original_idx, obj_data in object_data_with_distances:
            obj_name = obj_data['name']
            proximity = object_proximities[obj_name]
            
            # Allocate robots proportionally to proximity, but cap at 3 per object
            allocated_robots = min(3, max(1, int(num_robots * proximity / total_proximity)))
            budget[original_idx] = allocated_robots
        
        # Adjust if total exceeds available robots
        total_allocated = budget.sum().item()
        if total_allocated > num_robots:
            # Scale down proportionally while maintaining relative proximities
            scale_factor = num_robots / total_allocated
            budget = (budget.float() * scale_factor).long()
            
            # Ensure each object gets at least 1 robot if possible
            remaining_robots = num_robots - budget.sum().item()
            if remaining_robots > 0:
                # Give remaining robots to closest objects
                proximity_order = sorted(object_data_with_distances, 
                                      key=lambda x: object_proximities[x[2]['name']], reverse=True)
                for distance_to_goal, original_idx, obj_data in proximity_order:
                    if remaining_robots <= 0:
                        break
                    if budget[original_idx] < 3:  # Don't exceed max of 3
                        budget[original_idx] += 1
                        remaining_robots -= 1
        
        # Ensure we don't exceed available robots
        while budget.sum().item() > num_robots:
            # Reduce from least proximate objects first
            min_budget_idx = budget.argmin().item()
            if budget[min_budget_idx] > 1:
                budget[min_budget_idx] -= 1
            else:
                break
        
        print(f"Proximity budget allocation: {budget.tolist()}")
        return budget
    
    def _generate_push_plan(self, observation: torch.Tensor, relative_transform: Transform2, seed: int = 42) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate contact points and push trajectories using the learned model (single object)."""
        # Convert to batched tensors
        observation_batched = observation.unsqueeze(0).to(cfg.device)
        relative_transform_batched = relative_transform.to_tensor().unsqueeze(0).to(cfg.device)
        
        # Get the number of available robots as budget
        num_robots = len(self.world.robots_d)
        budget = torch.tensor([num_robots], device=cfg.device)
        
        # Generate using the model
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
    
    def _filter_masked_contacts(self,
                                contact_points_world: torch.Tensor,
                                push_trajectories_world: torch.Tensor,
                                contact_points_tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Remove masked contact points/trajectories that correspond to unused robots."""
        # Handle different input shapes
        if len(contact_points_world.shape) == 2:
            contact_points_world = contact_points_world.unsqueeze(0)
        if len(push_trajectories_world.shape) == 3:
            push_trajectories_world = push_trajectories_world.unsqueeze(0)
            
        B = contact_points_tokens.shape[0]
        N_total = cfg.N
        tokens = contact_points_tokens.clone()
        tokens_pairs = tokens.reshape(B, N_total, 2)

        # Build boolean mask of valid (unmasked) contacts
        mask_valid = (torch.abs(tokens_pairs[:, :, 0] - cfg.mask_token) > 5) & (torch.abs(tokens_pairs[:, :, 1] - cfg.mask_token) > 5)
        if B == 1:
            mask_valid = mask_valid.squeeze(0)
        
        valid_indices = torch.nonzero(mask_valid, as_tuple=False).squeeze()
        if valid_indices.numel() == 0:
            print("WARNING: No valid contact points found after filtering!")
            if len(contact_points_world.shape) == 3:
                return (contact_points_world[:, :0], push_trajectories_world[:, :0])
            else:
                return (contact_points_world[:0], push_trajectories_world[:0])

        valid_indices = valid_indices.flatten()
        filtered_contact_points_world = contact_points_world[:, valid_indices]
        filtered_push_trajectories_world = push_trajectories_world[:, valid_indices]

        return filtered_contact_points_world, filtered_push_trajectories_world
    
    def _visualize_iteration(self, iteration: int, object_data: List[Dict], 
                           batch_contact_points_tokens: torch.Tensor, 
                           batch_push_trajectories: torch.Tensor,
                           generation_steps: List = None):
        """Visualize the current iteration's planning for multiple objects."""
        batch_size = len(object_data)
        
        # Create batch data for visualization
        batch_dict = {
            "mask": torch.stack([obj['observation'] for obj in object_data]).to(cfg.device),
            "transform_object": torch.stack([obj['relative_transform'].to_tensor() for obj in object_data]).to(cfg.device),
            "trajectory": batch_push_trajectories,
            "tokens": batch_contact_points_tokens
        }
        
        if generation_steps is not None:
            # Create animation of the generation process
            # Extract tokens and trajectories from all generation steps
            tokens_t = torch.stack([step[0] for step in generation_steps], dim=0)  # (K, B, N*2)
            trajectory_t = torch.stack([step[1] for step in generation_steps], dim=0)  # (K, B, N, H, 2)
            
            # Save animation
            animation_path = os.path.join(self.params.output_dir, f"iteration_{iteration:03d}_generation_animation.mp4")
            visualize_batch_denoising(
                batch_dict,
                tokens_t,
                trajectory_t,
                save_path=animation_path,
                interval=500,  # 500ms between frames
                repeat=True
            )
            print(f"Generation animation saved to: {animation_path}")
        else:
            # Fallback to static visualization
            fig, axs = visualize_batch(batch_dict, save_path=None)
            
            # Save visualization
            save_path = os.path.join(self.params.output_dir, f"iteration_{iteration:03d}_multi_object.png")
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Multi-object visualization saved to: {save_path}")
            
            plt.close(fig)
    
    def _plan_robot_paths_with_optimization(self, starts: Dict[str, torch.Tensor], goals: Dict[str, torch.Tensor]) -> Dict[str, List]:
        """
        Plan robot paths with optimization to skip planning when robots are very close to goals.
        
        This implements the same optimization as IterativePushPlanner where if all robots are
        very close to their goals (< 0.1m), we skip expensive GSPI planning and create
        simple straight-line paths.
        
        Args:
            starts: Dictionary mapping robot names to their start poses
            goals: Dictionary mapping robot names to their goal poses
            
        Returns:
            Dictionary mapping robot names to their planned paths
        """
        robot_names = list(starts.keys())
        
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
            print("All robots very close to goals - skipping GSPI planning")
            paths = {}
            for i, start_name in enumerate(robot_names):
                goal_name = best_assignment[i]
                paths[start_name] = [starts[start_name].tolist(), goals[goal_name].tolist()]
        else:
            # Compute the maximum distance between any goal-robot pair.
            max_distance = 0.0
            for start_name in robot_names:
                for goal_name in robot_names:
                    distance = torch.norm(starts[start_name] - goals[goal_name]).item()
                    max_distance = max(max_distance, distance)
            
            # Plan using GSPI
            planner = GSPIPlanner(self.world, goal_tolerance=0.1, seed=self.world.simulator.seed)
            planner.set_grid_resolution_heuristic(0.05)
            planner.set_max_distance_heuristic(max_distance * 1.5 + 1)
            planner.set_max_iterations(300)
            planner.set_verbose(False)
            paths = planner.plan(starts, goals)
            
            # Check if planning failed (no paths returned or empty paths)
            if not paths or any(len(path) == 0 for path in paths.values()):
                print("Robot planner failed to find valid paths. Moving robots randomly to break deadlock.")
                paths = self._generate_random_movement_paths(starts)
            
            # Calculate the maximum trajectory length
            max_trajectory_length = max(len(path) for path in paths.values()) if paths else 0
            # paths = smooth_all_trajectories(paths, max_trajectory_length)
            # paths = {k: v.tolist() for k, v in paths.items()}
        
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
    
    def get_planning_summary(self) -> Dict:
        """Get a summary of the planning process."""
        if not self.planning_history:
            return {"status": "No planning performed"}
        
        # Calculate total travel distance
        total_travel_distance = sum(self.timing_stats['total_travel_distances'].values()) if self.timing_stats['total_travel_distances'] else 0.0
        
        return {
            "total_iterations": len(self.planning_history),
            "status": "Multi-object planning completed",
            "robot_travel_distances": self.timing_stats['total_travel_distances'].copy(),
            "total_travel_distance": total_travel_distance
        }
