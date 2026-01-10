# General imports.
from abc import ABC, abstractmethod
from typing import Dict, Any, TYPE_CHECKING
import torch

# Project imports.
from gco.planners.planner import Planner
from gco.world import World

if TYPE_CHECKING:
    from gco.tasks.tasks import Task

####################
# Base classes.
####################
class MultiRobotPlanner(Planner, ABC):
    """Planner for multiple robots."""
    def __init__(self, world: World, params: Dict[str, Any]):
        """ Initialize the MultiRobotPlanner.
        
        Args:
            world (World): The world instance
            params
        """
        super().__init__(world, params)

    @abstractmethod
    def plan_action_trajs(self, task: 'Task') -> Dict[str, torch.Tensor]:
        """
        Plan the next actions for the given task.
        :param task: The task to plan for
        :return: A dictionary mapping robot names to their planned trajectories (B, H, dim(q) + dim(q_dot))
        """
        pass