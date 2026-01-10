"""
Collaborative Motion Planning with Negotiated Diffusion Models
Author: Yorai Shaoul, 2025.

Obstacles: define data structures for obstacles.
"""

# General imports.
from abc import ABC
from dataclasses import dataclass
import torch

# Project imports.
from gco.config import Config as cfg

class Obstacle(ABC):
    def __init__(self, name):
        self.name = name
        self.type = "obstacle"

class ObstacleCircle(Obstacle):
    def __init__(self, name: str, radius: float):
        super().__init__(name)
        self.radius = torch.tensor(radius, device=cfg.device)
        self.type = "circle"

class ObstacleRectangle(Obstacle):
    def __init__(self, name: str, width: float, height: float):
        super().__init__(name)
        self.width = torch.tensor(width, device=cfg.device)
        self.height = torch.tensor(height, device=cfg.device)
        self.type = "rectangle"

class ObstacleSquare(Obstacle):
    def __init__(self, name: str, width: float):
        super().__init__(name)
        self.width = torch.tensor(width, device=cfg.device)
        self.type = "square"

class ObstaclePolygon(Obstacle):
    def __init__(self, name: str, vertices: torch.Tensor):
        super().__init__(name)
        self.vertices = vertices
        self.type = "polygon"