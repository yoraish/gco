"""
Collaborative Motion Planning with Negotiated Diffusion Models
Author: Yorai Shaoul, 2025.

Robot: in charge of defining their shape and joint limits.
"""

# General imports.
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
from dataclasses import dataclass
import math
from abc import ABC, abstractmethod
from torch import Tensor

# Project imports.
from gco.config import Config as cfg


class Robot(ABC):
    def __init__(self, name: str):
        self.name = name
        self.a_max = None
        self.v_max = None

    @abstractmethod
    def get_collision_circles(self) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Get the collision circles of the robot.
        :return: List of tuples, each containing a tensor of shape (C, 2) and a tensor of shape (C, 1) where C is the number of collision circles. The first tensor is the center of the circle, and the second tensor is the radius.
        """
        pass


class RobotDisk(Robot):
    def __init__(self, name: str, radius: float):
        super().__init__(name)
        self.radius = torch.tensor(radius, device=cfg.device)
        self.a_max = 0.2
        self.v_max = 1.0

    def get_collision_circles(self) -> list[tuple[Tensor, float]]:
        """
        Get the collision circles of the robot.
        :return: List of tuples, each containing a tensor of shape (C, 2) and a tensor of shape (C, 1) where C is the number of collision circles. The first tensor is the center of the circle, and the second tensor is the radius.
        """
        return [
            (torch.tensor([0.0, 0.0]), self.radius)
        ]
