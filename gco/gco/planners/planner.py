# General imports.
from abc import ABC, abstractmethod
from typing import Dict, Any, TYPE_CHECKING
import torch

# Project imports.
from gco.world import World

if TYPE_CHECKING:
    from gco.tasks.tasks import Task

####################
# Base classes.
####################
class Planner(ABC):
    """Base class for all planners."""
    def __init__(self, world: World, params: Dict[str, Any]):
        """ Initialize the Planner.

        Args:
            world (World): The world instance
            params
        """
        self.world = world
        self.params = params

class SingleRobotPlanner(Planner, ABC):
    """Planner for one robot."""
    def __init__(self, world: World, params: Dict[str, Any]):
        super().__init__(world, params)

    @abstractmethod
    def plan_action_traj(self, start_state: torch.Tensor, 
                         task: 'Task', 
                         constraints: Any,
                         **kwargs) -> torch.Tensor:
        """
        Plan the next action for the given task.
        :param start_state: The start state of the robot
        :param task: The task to plan for
        :param constraints: Any constraints on the action trajectory
        :param kwargs: Additional arguments for the planner
        :return: The planned action trajectory (B, H, dim(q) + dim(q_dot))
        """
        pass
