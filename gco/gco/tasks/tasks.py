# General imports.
from typing import List, Dict, Any
from dataclasses import dataclass
from enum import Enum
from abc import ABC, abstractmethod
import torch

# Project imports.
from gco.world import World
from gco.utils.transform_utils import Translation2


class TaskType(Enum):
    MANIPULATION = "manipulation"  # Object manipulation tasks
    NAVIGATION = "navigation"     # Movement tasks
    FORMATION = "formation"       # Multi-robot formation tasks
    INTERPOLATION = "interpolation"  # Path/trajectory following tasks

@dataclass
class Task(ABC):
    """Base class for all tasks."""

    def __init__(self):
        self.type: TaskType = None

    @abstractmethod
    def is_complete(self, world: World) -> bool:
        """Check if the task is complete based on current world state."""
        pass

@dataclass
class ManipulationTask(Task):
    """Task for manipulating objects."""
    def __init__(self, object_id: str = None, target_pose: Dict[str, Any] = None):
        super().__init__()
        self.type: TaskType = TaskType.MANIPULATION
        self.object_id: str = object_id
        self.target_pose: Dict[str, Any] = target_pose  # position and orientation
        self.position_tolerance: float = 0.05
        self.orientation_tolerance: float = 0.1
    
    def is_complete(self, world: World) -> bool:
        """Check if object is at target pose within tolerance."""
        if self.object_id is None or self.target_pose is None:
            return False
            
        current_pose = world.get_object_pose(self.object_id)
        target_position = torch.tensor(self.target_pose["position"])
        target_orientation = torch.tensor(self.target_pose["orientation"])
        
        position_error = torch.norm(current_pose.get_t() - target_position)
        orientation_error = torch.abs(current_pose.get_theta() - target_orientation)
        
        return (position_error <= self.position_tolerance and 
                orientation_error <= self.orientation_tolerance)

@dataclass
class NavigationTask(Task):
    """Task for robot navigation."""
    def __init__(self, goal_pos_d: Dict[str, Translation2]):
        super().__init__()
        self.type: TaskType = TaskType.NAVIGATION
        self.goal_pos_d: Dict[str, Translation2] = goal_pos_d
        self.position_tolerance: float = 0.05
    
    def is_complete(self, world: World) -> bool:
        """Check if robot is at goal pose within tolerance."""
        for robot_name, goal_pose in self.goal_pos_d.items():
            current_pose = world.get_robot_pose(robot_name)
            if torch.norm(current_pose.get_t() - goal_pose.get_t()) > self.position_tolerance:
                return False
        return True


@dataclass
class InterpolationTask(Task):
    """Task for following interpolated paths or trajectories.
    
    This task type is used when robots or objects need to follow a smooth path
    between waypoints, with optional timing constraints.
    """
    def __init__(self, 
                 goal_poses: Dict[str, Translation2],
                 position_tolerance: float = 0.05):  # meters
        super().__init__()
        self.type: TaskType = TaskType.INTERPOLATION
        self.goal_poses: Dict[str, Translation2] = goal_poses
        self.position_tolerance: float = position_tolerance
    
    def is_complete(self, world: World) -> bool:
        """Check if the interpolation task is complete.
        
        This checks if all robots have reached their goal poses within
        specified tolerances using vectorized operations.
        """
        # Get all current poses and goal poses as tensors
        current_poses = torch.stack([world.get_robot_pose(name).t for name in self.goal_poses.keys()])
        goal_poses = torch.stack([pose.t for pose in self.goal_poses.values()])
        
        # Compute distances for all robots at once
        distances = torch.norm(current_poses - goal_poses, dim=1)
        
        # Check if all distances are within tolerance
        return torch.all(distances <= self.position_tolerance)

@dataclass
class MultiObjectManipulationTask(Task):
    """Task for manipulating multiple objects simultaneously."""
    def __init__(self, 
                 object_targets: Dict[str, Dict[str, Any]],
                 position_tolerance: float = 0.15,
                 orientation_tolerance: float = 0.5):
        """
        Initialize multi-object manipulation task.
        
        Args:
            object_targets: Dictionary mapping object names to their target poses
                          Each target pose should have "position" and "orientation" keys
            position_tolerance: Position tolerance in meters
            orientation_tolerance: Orientation tolerance in radians
        """
        super().__init__()
        self.type: TaskType = TaskType.MANIPULATION
        self.object_targets: Dict[str, Dict[str, Any]] = object_targets
        self.position_tolerance: float = position_tolerance
        self.orientation_tolerance: float = orientation_tolerance
    
    def is_complete(self, world: World) -> bool:
        """Check if all goal positions have objects at them within tolerance.
        
        This checks if every target position has some object (not necessarily the 
        originally assigned one) within the specified tolerances. Each object can
        satisfy at most one target.
        """
        translation_error_l = []
        orientation_error_l = []
        for obj_name in self.get_object_names():
            current_pose = world.get_object_pose(obj_name)
            target_pose = self.get_target_pose(obj_name)
            dx = current_pose.get_t()[0] - target_pose["position"][0]
            dy = current_pose.get_t()[1] - target_pose["position"][1]
            dtheta = current_pose.get_theta() - target_pose["orientation"]
            translation_error_l.append(torch.norm(torch.tensor([dx, dy])))
            orientation_error_l.append(torch.abs(dtheta))
        print(f"CHECKING IF ALL OBJECTS HAVE REACHED THEIR GOALS:", translation_error_l, orientation_error_l)
        if torch.all(torch.tensor(translation_error_l) <= self.position_tolerance) and torch.all(torch.tensor(orientation_error_l) <= self.orientation_tolerance):
            return True
        return False
    
    def get_object_names(self) -> List[str]:
        """Get list of object IDs in this task."""
        return list(self.object_targets.keys())
    
    def get_target_pose(self, object_id: str) -> Dict[str, Any]:
        """Get target pose for a specific object."""
        return self.object_targets[object_id]
    
    def update_assignments(self, new_assignments: Dict[str, List[float]]):
        """
        Update object-to-goal assignments based on planner decisions.
        
        Args:
            new_assignments: Dictionary mapping object names to [x, y, theta] goal positions
        """
        for object_name, goal_config in new_assignments.items():
            if object_name in self.object_targets:
                self.object_targets[object_name] = {
                    "position": [goal_config[0], goal_config[1]],
                    "orientation": goal_config[2]
                }
        