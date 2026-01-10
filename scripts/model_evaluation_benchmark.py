"""
Model Evaluation Benchmark for Single-Object Manipulation Tasks

This script evaluates different models on single-object manipulation tasks with varying:
- Obstacle landscapes (empty, simple obstacles, complex mazes)
- Start/goal positions for the object
- Number of robots in the scene

The script records metrics including:
- Success rate (did the object arrive at its goal)
- Path length of the robots
- Number of pushes
- Computation time for generation (only the learning modules)
"""

import torch
import numpy as np
import time
import json
import os
import random
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
import pandas as pd
from datetime import datetime
from tqdm import tqdm

# GCO imports
from gco.world.world import World
from gco.world.robot import RobotDisk
from gco.world.obstacles import ObstacleCircle, ObstacleRectangle
from gco.world.objects import ObjectRectangle, ObjectCircle, ObjectT, ObjectPolygon, ObjectTriangle
from gco.utils.transform_utils import Translation2, Transform2
from gco.config import Config as cfg
from gco.planners.multi.iterative_push_planner import IterativePushPlanner
from gco.planners.multi.multi_object_planner import MultiObjectPlanner
from gco.tasks.tasks import ManipulationTask, MultiObjectManipulationTask
from gco.utils.model_utils import get_recent_model


@dataclass
class TestScenario:
    """Represents a single test scenario configuration."""
    scenario_id: str
    num_robots: int
    robot_start_configurations: List[Tuple[float, float, float]]  # List of (x, y, theta) for each robot
    obstacle_type: str  # "empty", "simple", "maze", "slalom"
    object_type: str    # "rectangle", "circle", "triangle", "t_shape", "multi"
    object_size: Dict[str, float]  # Size parameters for the object (for single object scenarios)
    start_position: Tuple[float, float]  # For single object scenarios
    goal_position: Tuple[float, float]  # For single object scenarios
    start_orientation: float  # For single object scenarios
    goal_orientation: float  # For single object scenarios
    seed: int
    problem_name: str = ""  # Problem specification name (e.g., "up_env_slalom_nobj_1_tobj_rectangle_nrob_1_2_3")
    # Multi-object support
    is_multi_object: bool = False  # Whether this is a multi-object scenario
    object_configurations: Optional[List[Dict]] = None  # List of object configs for multi-object scenarios


@dataclass
class TestResult:
    """Represents the result of a single test."""
    scenario_id: str
    model_name: str
    success: bool
    final_position_error: float
    final_orientation_error: float
    total_path_length: float
    num_pushes: int
    generation_time: float
    total_planning_time: float
    num_iterations: int
    total_travel_distance: float = 0.0  # Total robot travel distance
    robot_travel_distances: Dict[str, float] = None  # Per-robot travel distances
    avg_push_position_error: float = 0.0  # Average push position error
    avg_push_orientation_error: float = 0.0  # Average push orientation error
    error_message: Optional[str] = None
    # Additional fields for CSV output
    problem_name: str = ""  # Specification name (e.g., "empty_single_object_6robots")
    method: str = ""  # Model type (discrete, continuous, continuous_trajectory)
    num_robots: int = 0  # Number of robots in the scenario
    num_objects: int = 0  # Number of objects in the scenario
    cost: float = 0.0  # Overall motion cost (total_travel_distance)


@dataclass
class ModelConfig:
    """Configuration for a model to be tested."""
    name: str
    checkpoint_path: str
    model_type: str = "discrete"  # "discrete" or "continuous"
    smoothness_weight: float = 1.0
    max_iterations: int = 50
    goal_tolerance_position: float = 0.15
    goal_tolerance_orientation: float = 0.15
    visualize: bool = False  # Flag to control visualization


class ModelInterface(ABC):
    """Abstract interface for models to be tested."""
    
    @abstractmethod
    def generate(self, observation_batched: torch.Tensor, 
                relative_transform_batched: torch.Tensor, 
                budget: torch.Tensor, 
                seed: int = 42, 
                smoothness_weight: float = 1.0) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Generate contact points and trajectories."""
        pass


class ContactTrajectoryModelWrapper(ModelInterface):
    """Wrapper for the ContactTrajectoryModel to match the interface."""
    
    def __init__(self, model_path: str):
        from gco.models.contact_push_model import ContactTrajectoryModel
        vocab_size = cfg.V + 1
        self.model = ContactTrajectoryModel(vocab_size=vocab_size).to(cfg.device)
        self.model.load_state_dict(torch.load(model_path, map_location=cfg.device))
        self.model.eval()
    
    def generate(self, observation_batched: torch.Tensor, 
                relative_transform_batched: torch.Tensor, 
                budget: torch.Tensor, 
                seed: int = 42, 
                smoothness_weight: float = 1.0) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        return self.model.generate(
            observation_batched, 
            relative_transform_batched, 
            budget=budget, 
            seed=seed, 
            smoothness_weight=smoothness_weight
        )


class ModelEvaluator:
    """Main class for evaluating models on manipulation tasks."""
    
    def __init__(self, output_dir: str = "output/model_evaluation"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / "results").mkdir(exist_ok=True)
        (self.output_dir / "visualizations").mkdir(exist_ok=True)
        (self.output_dir / "logs").mkdir(exist_ok=True)
        
        self.results = []
        self.scenarios = []
        
    def create_obstacle_line_segment(self, start_x: float, start_y: float, 
                                   end_x: float, end_y: float, 
                                   width: float = 0.1, 
                                   name_prefix: str = "obstacle") -> Dict:
        """Create an obstacle line segment, consisting of circles, between two points."""
        length = np.sqrt((end_x - start_x)**2 + (end_y - start_y)**2)
        radius = width / 2.0
        num_obstacles = max(1, int(length / (2 * radius)))
        obstacles_and_poses = {}
        
        for i in range(num_obstacles):
            t = i / max(1, num_obstacles - 1)
            x = start_x + t * (end_x - start_x)
            y = start_y + t * (end_y - start_y)
            
            obstacle = ObstacleCircle(f"{name_prefix}_{i}", radius=radius)
            pose = Transform2(t=torch.tensor([x, y]), theta=torch.tensor([0.0]))
            obstacles_and_poses[obstacle] = pose
            
        return obstacles_and_poses
    
    def create_obstacle_landscape(self, obstacle_type: str) -> Dict:
        """Create different obstacle landscapes for testing."""
        obstacles_and_poses = {}
        
        # ================================ 
        # EMPTY ENVIRONMENT
        # ================================
        if obstacle_type == "empty":
            return obstacles_and_poses
            
        # ================================ 
        # EASY OBSTACLES
        # ================================
        elif obstacle_type == "easy_up":
            obstacle_1 = ObstacleCircle("obstacle_1", radius=0.1)
            obstacles_and_poses[obstacle_1] = Transform2(t=torch.tensor([0.0, 2.0]), theta=torch.tensor([0.0]))
            return obstacles_and_poses

        elif obstacle_type == "easy_down":
            obstacle_1 = ObstacleCircle("obstacle_1", radius=0.1)
            obstacles_and_poses[obstacle_1] = Transform2(t=torch.tensor([0.0, 0.0]), theta=torch.tensor([0.0]))
            return obstacles_and_poses

        elif obstacle_type == "easy_left":
            obstacle_1 = ObstacleCircle("obstacle_1", radius=0.1)
            obstacles_and_poses[obstacle_1] = Transform2(t=torch.tensor([-1.0, 1.0]), theta=torch.tensor([0.0]))
            return obstacles_and_poses

        elif obstacle_type == "easy_right":
            obstacle_1 = ObstacleCircle("obstacle_1", radius=0.1)
            obstacles_and_poses[obstacle_1] = Transform2(t=torch.tensor([1.0, 1.0]), theta=torch.tensor([0.0]))
            return obstacles_and_poses

        # ================================ 
        # WALL OBSTACLES
        # ================================
        elif obstacle_type == "wall_up":
            obstacle_1 = ObstacleCircle("obstacle_1", radius=0.15)
            obstacles_and_poses[obstacle_1] = Transform2(t=torch.tensor([0.0, 2.0]), theta=torch.tensor([0.0]))
            obstacle_2 = ObstacleCircle("obstacle_2", radius=0.15)
            obstacles_and_poses[obstacle_2] = Transform2(t=torch.tensor([-0.25, 2.0]), theta=torch.tensor([0.0]))
            obstacle_3 = ObstacleCircle("obstacle_3", radius=0.15)
            obstacles_and_poses[obstacle_3] = Transform2(t=torch.tensor([-0.5, 2.0]), theta=torch.tensor([0.0]))
            obstacle_4 = ObstacleCircle("obstacle_4", radius=0.15)
            obstacles_and_poses[obstacle_4] = Transform2(t=torch.tensor([0.5, 2.0]), theta=torch.tensor([0.0]))
            obstacle_5 = ObstacleCircle("obstacle_5", radius=0.15)
            obstacles_and_poses[obstacle_5] = Transform2(t=torch.tensor([0.25, 2.0]), theta=torch.tensor([0.0]))

            obstacle_6 = ObstacleCircle("obstacle_6", radius=0.15)
            obstacles_and_poses[obstacle_6] = Transform2(t=torch.tensor([2.1, 2.0]), theta=torch.tensor([0.0]))
            obstacle_7 = ObstacleCircle("obstacle_7", radius=0.15)
            obstacles_and_poses[obstacle_7] = Transform2(t=torch.tensor([2.35, 2.0]), theta=torch.tensor([0.0]))
            obstacle_8 = ObstacleCircle("obstacle_8", radius=0.15)
            obstacles_and_poses[obstacle_8] = Transform2(t=torch.tensor([2.6, 2.0]), theta=torch.tensor([0.0]))
            obstacle_9 = ObstacleCircle("obstacle_9", radius=0.15)
            obstacles_and_poses[obstacle_9] = Transform2(t=torch.tensor([2.85, 2.0]), theta=torch.tensor([0.0]))

            obstacle_10 = ObstacleCircle("obstacle_10", radius=0.15)
            obstacles_and_poses[obstacle_10] = Transform2(t=torch.tensor([-2.1, 2.0]), theta=torch.tensor([0.0]))
            obstacle_11 = ObstacleCircle("obstacle_11", radius=0.15)
            obstacles_and_poses[obstacle_11] = Transform2(t=torch.tensor([-2.35, 2.0]), theta=torch.tensor([0.0]))
            obstacle_12 = ObstacleCircle("obstacle_12", radius=0.15)
            obstacles_and_poses[obstacle_12] = Transform2(t=torch.tensor([-2.6, 2.0]), theta=torch.tensor([0.0]))
            obstacle_13 = ObstacleCircle("obstacle_13", radius=0.15)
            obstacles_and_poses[obstacle_13] = Transform2(t=torch.tensor([-2.85, 2.0]), theta=torch.tensor([0.0]))
            return obstacles_and_poses


        elif obstacle_type == "wall_down":
            obstacle_1 = ObstacleCircle("obstacle_1", radius=0.15)
            obstacles_and_poses[obstacle_1] = Transform2(t=torch.tensor([0.0, 0.0]), theta=torch.tensor([0.0]))
            obstacle_2 = ObstacleCircle("obstacle_2", radius=0.15)
            obstacles_and_poses[obstacle_2] = Transform2(t=torch.tensor([-0.25, 0.0]), theta=torch.tensor([0.0]))
            obstacle_3 = ObstacleCircle("obstacle_3", radius=0.15)
            obstacles_and_poses[obstacle_3] = Transform2(t=torch.tensor([-0.5, 0.0]), theta=torch.tensor([0.0]))
            obstacle_4 = ObstacleCircle("obstacle_4", radius=0.15)
            obstacles_and_poses[obstacle_4] = Transform2(t=torch.tensor([0.5, 0.0]), theta=torch.tensor([0.0]))
            obstacle_5 = ObstacleCircle("obstacle_5", radius=0.15)
            obstacles_and_poses[obstacle_5] = Transform2(t=torch.tensor([0.25, 0.0]), theta=torch.tensor([0.0]))
            return obstacles_and_poses
        
        
        # ================================ 
        # SIMPLE OBSTACLES
        # ================================
        elif obstacle_type == "simple":
            # Simple obstacles: a few circles
            obstacle_1 = ObstacleCircle("obstacle_1", radius=0.2)
            obstacles_and_poses[obstacle_1] = Transform2(t=torch.tensor([-0.5, 2.0]), theta=torch.tensor([0.0]))
            
            obstacle_2 = ObstacleCircle("obstacle_2", radius=0.2)
            obstacles_and_poses[obstacle_2] = Transform2(t=torch.tensor([0.0, 2.0]), theta=torch.tensor([0.0]))
            
            obstacle_3 = ObstacleCircle("obstacle_3", radius=0.2)
            obstacles_and_poses[obstacle_3] = Transform2(t=torch.tensor([0.5, 2.0]), theta=torch.tensor([0.0]))

        # ================================ 
        # SLALOM OBSTACLES
        # ================================
        elif obstacle_type == "slalom":
            # Slalom course
            y_bottom_wall = -0.2
            y_top_wall = 6.0
            # obstacles_and_poses.update(self.create_obstacle_line_segment(-2.0, -1.0 + 1, 2.0, -1.0 + 1, 0.1, "slalom_1"))
            obstacles_and_poses.update(self.create_obstacle_line_segment(-0.0,  3.5, 2.0,  3.5 , 0.1, "slalom_2"))  # Top (rhs) barrier
            obstacles_and_poses.update(self.create_obstacle_line_segment(-2.0,  2.0, 0.0,  2.0, 0.1, "slalom_5"))  # Bottom (lhs) barrier
            obstacles_and_poses.update(self.create_obstacle_line_segment(-2.0, y_bottom_wall, -2.0, y_top_wall, 0.1, "slalom_3"))  # Left wall.
            obstacles_and_poses.update(self.create_obstacle_line_segment(2.0,  y_bottom_wall, 2.0,  y_top_wall, 0.1, "slalom_4"))  # Right wall.
            obstacles_and_poses.update(self.create_obstacle_line_segment(-2.0,  y_bottom_wall, 2.0,  y_bottom_wall, 0.1, "slalom_6"))  # bottom wall.
            obstacles_and_poses.update(self.create_obstacle_line_segment(-2.0,  y_top_wall, 2.0,  y_top_wall, 0.1, "slalom_7"))  # top wall.

        elif obstacle_type == "empty_multi_object":
            # Empty environment for multi-object scenarios (same as empty)
            pass  # No obstacles
            
        else:
            raise ValueError(f"Unknown obstacle type: {obstacle_type}")
            
        return obstacles_and_poses
    
    def create_object(self, object_type: str, object_size: Dict[str, float], name: str = "object_1") -> Any:
        """Create different object types for testing with given size parameters."""
        if object_type == "rectangle":
            width = object_size.get("width", 0.15)
            height = object_size.get("height", 0.35)
            return ObjectRectangle(name, width=width, height=height)
        if object_type == "rectangle_noise":
            vertices_min = torch.tensor([[-0.15, 0.1],
                                         [-0.15, 0.0],
                                         [-0.15, -0.1],
                                         [0.0, -0.1],
                                         [0.15, -0.1],
                                         [0.15, 0.0],
                                         [0.15, 0.1],
                                         [0.0, 0.1]], dtype=torch.float32)
            vertices_max = 0.3 * torch.tensor([[np.cos(3 * np.pi/4), np.sin( 3 * np.pi/4)],
                                               [np.cos(np.pi), np.sin(np.pi)],
                                               [np.cos(5 * np.pi/4), np.sin(5 * np.pi/4)],
                                               [np.cos(3 * np.pi/2), np.sin(3 * np.pi/2)],
                                               [np.cos(7 * np.pi/4), np.sin(7 * np.pi/4)],
                                               [np.cos(0), np.sin(0)],
                                               [np.cos(np.pi/4), np.sin(np.pi/4)],
                                               [np.cos(np.pi/2), np.sin(np.pi/2)]], dtype=torch.float32)
            expansion_vectors = vertices_max - vertices_min
            noise = object_size.get("noise", 0.02)
            random_offsets = (torch.rand(vertices_min.shape[0]) - 0.5) * 2.0 * noise
            # Clamp at 1.0.
            random_offsets = torch.clamp(random_offsets, 0.0, 1.0)
            vertices = vertices_min + expansion_vectors * random_offsets.unsqueeze(1)



            
            return ObjectPolygon(name, vertices=vertices)
        
        elif object_type == "circle":
            radius = object_size.get("radius", 0.2)
            return ObjectCircle(name, radius=radius)
        elif object_type == "triangle":
            width = object_size.get("width", 0.32)
            height = object_size.get("height", 0.32)
            vertices = ObjectTriangle(name, width=width, height=height).vertices
            return ObjectPolygon(name, vertices=vertices)
            # return ObjectTriangle(name, width=width, height=height)
        elif object_type == "t_shape":
            bar_width = object_size.get("bar_width", 0.35)
            bar_height = object_size.get("bar_height", 0.15)
            stem_width = object_size.get("stem_width", 0.15)
            stem_height = object_size.get("stem_height", 0.35)

            vertices = ObjectT(name, bar_width=bar_width, bar_height=bar_height, 
                         stem_width=stem_width, stem_height=stem_height).vertices
            return ObjectPolygon(name, vertices=vertices)

            # return ObjectT(name, bar_width=bar_width, bar_height=bar_height, 
                        #  stem_width=stem_width, stem_height=stem_height)
        else:
            raise ValueError(f"Unknown object type: {object_type}")
   
    def create_world_for_scenario(self, scenario: TestScenario) -> World:
        """Create a world configuration for a given scenario."""
        # Create robots
        robots = []
        robot_poses = []
        
        # Position robots using the provided start configurations
        for i in range(scenario.num_robots):
            robot = RobotDisk(f"robot_{i+1}", radius=cfg.robot_radius)
            robots.append(robot)
            
            # Use the robot start configuration from the scenario
            robot_config = scenario.robot_start_configurations[i]
            x_pos, y_pos, theta = robot_config
            pose = Transform2(t=torch.tensor([x_pos, y_pos]), theta=torch.tensor([theta]))
            robot_poses.append(pose)
        
        # Create obstacles
        obstacles_and_poses = self.create_obstacle_landscape(scenario.obstacle_type)
        
        # Create objects
        objects_with_pose_init = {}
        
        if scenario.is_multi_object and scenario.object_configurations:
            # Multi-object scenario
            for i, obj_config in enumerate(scenario.object_configurations):
                obj_name = f"object_{i+1}"
                obj = self.create_object(obj_config["type"], obj_config["size"], name=obj_name)
                # Handle both tuple and list formats for start_position
                start_pos = obj_config["start_position"]
                if isinstance(start_pos, tuple) and len(start_pos) == 3:
                    # (x, y, theta) format
                    obj_pose = Transform2(
                        t=torch.tensor([start_pos[0], start_pos[1]]), 
                        theta=torch.tensor([start_pos[2]])
                    )
                else:
                    # (x, y) format with separate orientation
                    obj_pose = Transform2(
                        t=torch.tensor(start_pos), 
                        theta=torch.tensor([obj_config["start_orientation"]])
                    )
                objects_with_pose_init[obj] = obj_pose
        else:
            # Single object scenario
            object_1 = self.create_object(scenario.object_type, scenario.object_size)
            object_pose = Transform2(
                t=torch.tensor(scenario.start_position), 
                theta=torch.tensor([scenario.start_orientation])
            )
            objects_with_pose_init[object_1] = object_pose
        
        # Create world
        world = World(
            size=(1.0, 1.0),
            resolution=0.05,
            robots_with_pose_init={robot: pose for robot, pose in zip(robots, robot_poses)},
            obstacles_with_pose_init=obstacles_and_poses, 
            objects_with_pose_init=objects_with_pose_init,
            seed=scenario.seed
        )
        
        return world
    
    def create_task_for_scenario(self, scenario: TestScenario, model_type: str = "discrete"):
        """Create a manipulation task for a given scenario."""
        if scenario.is_multi_object and scenario.object_configurations:
            # Multi-object scenario
            object_targets = {}
            for i, obj_config in enumerate(scenario.object_configurations):
                obj_name = f"object_{i+1}"
                # Handle both tuple and list formats for goal_position
                goal_pos = obj_config["goal_position"]
                if isinstance(goal_pos, tuple) and len(goal_pos) == 3:
                    # (x, y, theta) format
                    object_targets[obj_name] = {
                        "position": [goal_pos[0], goal_pos[1]],
                        "orientation": goal_pos[2]
                    }
                else:
                    # (x, y) format with separate orientation
                    object_targets[obj_name] = {
                        "position": list(goal_pos),
                        "orientation": obj_config["goal_orientation"]
                    }
            task = MultiObjectManipulationTask(
                object_targets=object_targets,
                position_tolerance=0.15,
                orientation_tolerance=0.5
            )
            return task
        elif model_type == "heuristic":
            # Single object scenario with heuristic model - use MultiObjectManipulationTask
            # since MultiObjectPlanner only supports MultiObjectManipulationTask
            object_targets = {
                "object_1": {
                    "position": list(scenario.goal_position),
                    "orientation": scenario.goal_orientation
                }
            }
            task = MultiObjectManipulationTask(
                object_targets=object_targets,
                position_tolerance=0.15,
                orientation_tolerance=0.5
            )
            return task
        else:
            # Single object scenario with non-heuristic model
            task = ManipulationTask()
            task.object_id = "object_1"
            task.target_pose = {
                "position": list(scenario.goal_position),
                "orientation": scenario.goal_orientation
            }
            return task
    
    def run_single_test(self, scenario: TestScenario, 
                       model_config: ModelConfig,
                       problem_name: str = "",
                       method: str = "") -> TestResult:
        """Run a single test scenario."""
        print(f"Running test: {scenario.scenario_id}")
        print(f"  Robots: {scenario.num_robots}, Obstacles: {scenario.obstacle_type}, Object: {scenario.object_type}")
        if scenario.is_multi_object:
            print(f"  Multi-object scenario with {len(scenario.object_configurations)} objects")
            for i, obj_config in enumerate(scenario.object_configurations):
                print(f"    [run_single_test] Object {i+1}: {obj_config['type']} at {obj_config['start_position']} -> {obj_config['goal_position']}")
        else:
            print(f"  [run_single_test] Start: {scenario.start_position}, Orientation: {scenario.start_orientation}, Goal: {scenario.goal_position}, Orientation: {scenario.goal_orientation}")
        
        try:
            # Set deterministic seeds
            random.seed(scenario.seed)
            np.random.seed(scenario.seed)
            torch.manual_seed(scenario.seed)

            # Create world and task
            world = self.create_world_for_scenario(scenario)
            task = self.create_task_for_scenario(scenario, model_config.model_type)
            
            # Create appropriate planner based on scenario type and model type
            # Use MultiObjectPlanner for heuristic model regardless of scenario type
            # since IterativePushPlanner doesn't support heuristic model
            if scenario.is_multi_object or model_config.model_type == "heuristic":
                # Multi-object scenario or heuristic model - use MultiObjectPlanner
                planner_params = {
                    "max_iterations": model_config.max_iterations,
                    "goal_tolerance_position": model_config.goal_tolerance_position,
                    "goal_tolerance_orientation": model_config.goal_tolerance_orientation,
                    "min_push_distance": 0.02,
                    "visualization_interval": 5,
                    "save_visualizations": False,  ######
                    "visualize_planning": model_config.visualize,
                    "output_dir": str(self.output_dir / "visualizations" / scenario.scenario_id),
                    "model_checkpoint": model_config.checkpoint_path,
                    "model_type": model_config.model_type,
                    "persistent_simulator": True,
                    "object_planning_horizon": 3,
                    "object_radius": 0.45
                }
                planner = MultiObjectPlanner(world, params=planner_params)
            else:
                # Single object scenario with non-heuristic model - use IterativePushPlanner
                planner_params = {
                    "max_iterations": model_config.max_iterations,
                    "goal_tolerance_position": model_config.goal_tolerance_position,
                    "goal_tolerance_orientation": model_config.goal_tolerance_orientation,
                    "min_push_distance": 0.02,
                    "visualization_interval": 1,
                    "save_visualizations": False,
                    "visualize_planning": model_config.visualize,
                    "output_dir": str(self.output_dir / "visualizations" / scenario.scenario_id),
                    "model_checkpoint": model_config.checkpoint_path,
                    "model_type": model_config.model_type,
                    "persistent_simulator": True,
                    "planner_type": "a_star"
                }
                planner = IterativePushPlanner(world, params=planner_params)
            
            # Run planning
            start_time = time.time()
            final_trajectories = planner.plan_action_trajs(task, visualize=model_config.visualize)
            total_planning_time = time.time() - start_time
            
            # Calculate errors based on scenario type
            if scenario.is_multi_object:
                # Multi-object scenario - use task's success criteria
                success = task.is_complete(world)
                
                # Calculate errors for reporting (using specific object assignments)
                position_errors = []
                orientation_errors = []
                for i, obj_config in enumerate(scenario.object_configurations):
                    obj_name = f"object_{i+1}"
                    final_object_pose = world.get_object_pose(obj_name).to(cfg.device)
                    
                    # Handle both tuple and list formats for goal_position
                    goal_pos = obj_config["goal_position"]
                    if isinstance(goal_pos, tuple) and len(goal_pos) == 3:
                        # (x, y, theta) format
                        goal_position = torch.tensor([goal_pos[0], goal_pos[1]], device=cfg.device)
                        goal_orientation = torch.tensor([goal_pos[2]], device=cfg.device)
                    else:
                        # (x, y) format with separate orientation
                        goal_position = torch.tensor(obj_config["goal_position"], device=cfg.device)
                        goal_orientation = torch.tensor([obj_config["goal_orientation"]], device=cfg.device)
                    
                    goal_pose = Transform2(t=goal_position, theta=goal_orientation).to(cfg.device)
                    
                    pos_error = torch.norm(final_object_pose.get_t() - goal_pose.get_t()).item()
                    orient_error = torch.abs(final_object_pose.get_theta() - goal_pose.get_theta()).item()
                    position_errors.append(pos_error)
                    orientation_errors.append(orient_error)
                
                # Use average errors for reporting
                position_error = np.mean(position_errors)
                orientation_error = np.mean(orientation_errors)
            else:
                # Single object scenario - handle both ManipulationTask and MultiObjectManipulationTask
                final_object_pose = world.get_object_pose("object_1").to(cfg.device)
                
                # Check task type to get goal pose correctly
                if hasattr(task, 'target_pose'):
                    # ManipulationTask
                    goal_pose = Transform2(
                        t=torch.tensor(task.target_pose["position"]), 
                        theta=torch.tensor(task.target_pose["orientation"])
                    ).to(cfg.device)
                else:
                    # MultiObjectManipulationTask (for heuristic models)
                    goal_pose = Transform2(
                        t=torch.tensor(task.object_targets["object_1"]["position"]), 
                        theta=torch.tensor(task.object_targets["object_1"]["orientation"])
                    ).to(cfg.device)
                
                position_error = torch.norm(final_object_pose.get_t() - goal_pose.get_t()).item()
                orientation_error = torch.abs(final_object_pose.get_theta() - goal_pose.get_theta()).item()
                success = (position_error <= model_config.goal_tolerance_position and 
                          orientation_error <= model_config.goal_tolerance_orientation)
            
            # Print success information
            print(f"Position error: {position_error:.3f} ?< {model_config.goal_tolerance_position}, Orientation error: {orientation_error:.3f} ?< {model_config.goal_tolerance_orientation}")
            print(f"Success: {success}")
            
            # Get planning statistics
            summary = planner.get_planning_summary()
            num_iterations = summary.get("total_iterations", 0)
            
            # Extract travel distance information
            total_travel_distance = summary.get("total_travel_distance", 0.0)
            robot_travel_distances = summary.get("robot_travel_distances", {})
            
            # Extract push accuracy information
            push_accuracy = summary.get("push_accuracy", {})
            avg_push_position_error = push_accuracy.get("avg_position_error", 0.0)
            avg_push_orientation_error = push_accuracy.get("avg_orientation_error", 0.0)
            
            # Use travel distance as path length (more accurate than the old calculation)
            total_path_length = total_travel_distance
            
            # Get generation time from timing stats
            generation_time = 0.0
            if hasattr(planner, 'timing_stats') and planner.timing_stats['all_iteration_times']:
                for iteration_data in planner.timing_stats['all_iteration_times']:
                    for name, time_val in iteration_data['components']:
                        if 'Model Generation' in name:
                            generation_time += time_val
            
            # Calculate number of objects
            num_objects = len(scenario.object_configurations) if scenario.object_configurations else 1
            
            # Create result
            result = TestResult(
                scenario_id=scenario.scenario_id,
                model_name=model_config.name,
                success=success,
                final_position_error=position_error,
                final_orientation_error=orientation_error,
                total_path_length=total_path_length,
                num_pushes=num_iterations,
                generation_time=generation_time,
                total_planning_time=total_planning_time,
                num_iterations=num_iterations,
                total_travel_distance=total_travel_distance,
                robot_travel_distances=robot_travel_distances,
                avg_push_position_error=avg_push_position_error,
                avg_push_orientation_error=avg_push_orientation_error,
                # Additional fields for CSV output
                problem_name=scenario.problem_name,
                method=method,
                num_robots=scenario.num_robots,
                num_objects=num_objects,
                cost=total_travel_distance  # Use total_travel_distance as cost
            )
            GREEN = "\033[92m"
            RED = "\033[91m"
            RESET = "\033[0m"
            if success:
                print(GREEN, "="*50 )
                print(f"  Result: {'SUCCESS' if success else 'FAILED'}")
                print(f"  Position error: {position_error:.3f}, Orientation error: {orientation_error:.3f}")
                print(f"  Iterations: {num_iterations}, Time: {total_planning_time:.2f}s")
                print("="*50, RESET)
            else:
                print(RED, "="*50 )
                print(f"  Result: {'SUCCESS' if success else 'FAILED'}")
                print(f"  Position error: {position_error:.3f}, Orientation error: {orientation_error:.3f}")
                print(f"  Iterations: {num_iterations}, Time: {total_planning_time:.2f}s")
                print("="*50, RESET)

            return result
            
        except Exception as e:
            print(f"  Error: {e}")
            # Calculate number of objects for error case
            num_objects = len(scenario.object_configurations) if scenario.object_configurations else 1
            
            return TestResult(
                scenario_id=scenario.scenario_id,
                model_name=model_config.name,
                success=False,
                final_position_error=float('inf'),
                final_orientation_error=float('inf'),
                total_path_length=0.0,
                num_pushes=0,
                generation_time=0.0,
                total_planning_time=0.0,
                num_iterations=0,
                total_travel_distance=0.0,
                robot_travel_distances={},
                avg_push_position_error=0.0,
                avg_push_orientation_error=0.0,
                error_message=str(e),
                # Additional fields for CSV output
                problem_name=scenario.problem_name,
                method=method,
                num_robots=scenario.num_robots,
                num_objects=num_objects,
                cost=0.0  # No cost for failed cases
            )
    
    def evaluate_model(self, model_config: ModelConfig, 
                      scenarios: Optional[List[TestScenario]] = None,
                      problem_name: str = "",
                      method: str = "") -> List[TestResult]:
        """Evaluate a model on all scenarios."""
        if scenarios is None:
            scenarios = self.scenarios
        
        print(f"\n{'='*60}")
        print(f"Evaluating model: {model_config.name}")
        print(f"Checkpoint: {model_config.checkpoint_path}")
        print(f"Model Type: {model_config.model_type}")
        print(f"{'='*60}")
        
        # No need to load model separately - planner will handle it
        
        results = []
        for i, scenario in enumerate(tqdm(scenarios, desc=f"Evaluating {model_config.name}", unit="scenario")):
            print(f"\nTest {i+1}/{len(scenarios)}")
            result = self.run_single_test(scenario, model_config, problem_name, method)
            results.append(result)
            
            # Save results after every problem (scenario)
            # self.save_results(results, model_config.name)
            # Save only the new result to CSV to avoid duplication
            self.save_results_csv([result], f"{model_config.name}_incremental.csv", append=True)
        
        self.results.extend(results)
        return results
    
    def save_results(self, results: List[TestResult], model_name: str):
        """Save results to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{model_name}_{timestamp}.json"
        filepath = self.output_dir / "results" / filename
        
        # Ensure the results directory exists
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert results to dict format
        results_dict = [asdict(result) for result in results]
        
        with open(filepath, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        print(f"Results saved to: file://{filepath.absolute()}")
    
    def save_results_csv(self, results: List[TestResult], filename: str = None, append: bool = False):
        """Save results to CSV file with the requested columns."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"results_{timestamp}.csv"
        
        filepath = self.output_dir / "results" / filename
        
        # Ensure the results directory exists
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Define CSV columns
        csv_columns = [
            'problem_name',
            'method', 
            'num_robots',
            'num_objects',
            'success',
            'cost',
            'num_actions'
        ]
        
        # Convert results to CSV format
        csv_data = []
        for result in results:
            csv_data.append({
                'problem_name': result.problem_name,
                'method': result.method,
                'num_robots': result.num_robots,
                'num_objects': result.num_objects,
                'success': 1 if result.success else 0,  # Convert boolean to 1/0
                'cost': result.cost,
                'num_actions': result.num_pushes
            })
        
        # Write CSV file
        df = pd.DataFrame(csv_data)
        
        if append and filepath.exists():
            # Append to existing file
            existing_df = pd.read_csv(filepath)
            df = pd.concat([existing_df, df], ignore_index=True)
        
        df.to_csv(filepath, index=False)
        
        print(f"CSV results saved to: file://{filepath.absolute()}")
        return filepath
    
    def generate_summary_report(self, model_names: List[str]):
        """Generate a comprehensive summary report."""
        # Filter results by model names
        model_results = {}
        for model_name in model_names:
            model_results[model_name] = [r for r in self.results if r.model_name == model_name]
        
        # Calculate statistics
        summary_stats = {}
        for model_name, results in model_results.items():
            if not results:
                continue
                
            successful = [r for r in results if r.success]
            
            summary_stats[model_name] = {
                "total_tests": len(results),
                "successful_tests": len(successful),
                "success_rate": len(successful) / len(results) if results else 0.0,
                "avg_position_error": np.mean([r.final_position_error for r in results]),
                "avg_orientation_error": np.mean([r.final_orientation_error for r in results]),
                "avg_path_length": np.mean([r.total_path_length for r in results]),
                "avg_travel_distance": np.mean([r.total_travel_distance for r in results]),
                "avg_push_position_error": np.mean([r.avg_push_position_error for r in results]),
                "avg_push_orientation_error": np.mean([r.avg_push_orientation_error for r in results]),
                "avg_num_pushes": np.mean([r.num_pushes for r in results]),
                "avg_generation_time": np.mean([r.generation_time / max(r.num_iterations, 1) for r in results]),
                "avg_total_time": np.mean([r.total_planning_time for r in results]),
            }
        
        # Save summary
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_file = self.output_dir / "results" / f"summary_{timestamp}.json"
        
        with open(summary_file, 'w') as f:
            json.dump(summary_stats, f, indent=2)
        
        # Print summary
        print(f"\n{'='*80}")
        print("SUMMARY REPORT")
        print(f"{'='*80}")
        
        for model_name, stats in summary_stats.items():
            print(f"\nModel: {model_name}")
            print(f"  Success Rate: {stats['success_rate']:.2%} ({stats['successful_tests']}/{stats['total_tests']})")
            print(f"  🎯 Avg Push Position Error: {stats['avg_push_position_error']:.4f}m")
            print(f"  🎯 Avg Push Orientation Error: {stats['avg_push_orientation_error']:.4f}rad")
            print(f"  Avg Path Length: {stats['avg_path_length']:.3f}m")
            print(f"  Avg Travel Distance: {stats['avg_travel_distance']:.3f}m")
            print(f"  Avg Num Pushes: {stats['avg_num_pushes']:.1f}")
            print(f"  Avg Generation Time (per iteration): {stats['avg_generation_time']:.3f}s")
            print(f"  Avg Total Time: {stats['avg_total_time']:.3f}s")
            print(f"  Avg Final Position Error: {stats['avg_position_error']:.3f}m")
            print(f"  Avg Final Orientation Error: {stats['avg_orientation_error']:.3f}rad")
        
        return summary_stats
    
    def create_visualization(self, model_names: List[str]):
        """Create visualizations of the results."""
        # Filter results by model names
        model_results = {}
        for model_name in model_names:
            model_results[model_name] = [r for r in self.results if r.model_name == model_name]
        
        # Debug output
        print(f"Creating visualization for models: {model_names}")
        print(f"Total results available: {len(self.results)}")
        for model_name, results in model_results.items():
            print(f"  {model_name}: {len(results)} results")
        
        # Check if we have any results to visualize
        total_results = sum(len(results) for results in model_results.values())
        if total_results == 0:
            print("No results available for visualization. Skipping plot creation.")
            return
        
        # Create comparison plots
        fig, axes = plt.subplots(3, 3, figsize=(15, 15))
        fig.suptitle('Model Comparison Results', fontsize=16)
        
        metrics = ['success_rate', 'avg_push_position_error', 'avg_push_orientation_error', 
                  'avg_path_length', 'avg_travel_distance', 'avg_num_pushes', 'avg_generation_time']
        metric_names = ['Success Rate', 'Push Position Error (m)', 'Push Orientation Error (rad)',
                       'Path Length (m)', 'Travel Distance (m)', 'Num Pushes', 'Generation Time per Iteration (s)']
        
        for i, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
            ax = axes[i // 3, i % 3]
            
            model_names_list = []
            values = []
            
            for model_name, results in model_results.items():
                if not results:
                    continue
                    
                if metric == 'success_rate':
                    successful = [r for r in results if r.success]
                    value = len(successful) / len(results)
                elif metric == 'avg_generation_time':
                    # Special case: generation time per iteration
                    value = np.mean([r.generation_time / max(r.num_iterations, 1) for r in results])
                else:
                    # Map metric names to actual attribute names
                    metric_to_attribute = {
                        'avg_path_length': 'total_path_length',
                        'avg_travel_distance': 'total_travel_distance',
                        'avg_push_position_error': 'avg_push_position_error',
                        'avg_push_orientation_error': 'avg_push_orientation_error',
                        'avg_num_pushes': 'num_pushes'
                    }
                    attribute_name = metric_to_attribute.get(metric, metric.replace('avg_', ''))
                    value = np.mean([getattr(r, attribute_name) for r in results])
                
                model_names_list.append(model_name)
                values.append(value)
            
            if model_names_list:
                bars = ax.bar(model_names_list, values)
                ax.set_title(metric_name)
                ax.set_ylabel(metric_name)
                
                # Add value labels on bars
                for bar, value in zip(bars, values):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_file = self.output_dir / "results" / f"comparison_plot_{timestamp}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Visualization saved to: file://{plot_file.absolute()}")

