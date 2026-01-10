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
from gco.utils.transform_utils import Transform


class MovableObject(ABC):
    def __init__(self, name: str):
        self.name = name
        self.type = ""

class ObjectCircle(MovableObject):
    def __init__(self, name: str, radius: float):
        super().__init__(name)
        self.radius = radius
        self.type = "circle"

class ObjectRectangle(MovableObject):
    def __init__(self, name: str, width: float, height: float):
        super().__init__(name)
        self.width = width
        self.height = height
        self.type = "rectangle"

class ObjectSquare(MovableObject):
    def __init__(self, name: str, width: float):
        super().__init__(name)
        self.width = width
        self.type = "square"

class ObjectTriangle(MovableObject):
    def __init__(self, name: str, width: float, height: float):
        super().__init__(name)
        self.vertices = torch.tensor([
            [-height/2, -width/2],
            [-height/2, width/2],
            [height/2, 0.0]
        ])
        self.type = "triangle"

class ObjectT(MovableObject):
    # T-shape with horizontal bar at top and vertical stem below
    def __init__(self, name: str, bar_width: float, bar_height: float, stem_width: float, stem_height: float = None):
        super().__init__(name)
        self.bar_width = bar_width
        self.bar_height = bar_height
        self.stem_width = stem_width
        # If stem_height is not provided, use bar_height as default
        self.stem_height = stem_height if stem_height is not None else bar_height
        self.type = "polygon"
        
        # For backward compatibility, keep the old attribute names
        self.width = bar_width
        self.height = bar_height
        self.leg_width = stem_width

    # Method to get vertices by just doing .vertices.
    @property
    def vertices(self):
        # Calculate the total height to center the entire shape
        total_height = self.bar_height + self.stem_height
        
        # The vertices are in the object frame (centered at the origin) and in meters. (x, y), with x increasing forward and y increasing to the left.
        
        return torch.tensor([
            [-total_height/2, self.stem_width/2],  # Bottom left (stem).
            [-total_height/2, -self.stem_width/2],  # Bottom right (stem).
            [total_height/2 - self.bar_height, -self.stem_width/2],  # right armpit.
            [total_height/2 - self.bar_height, -self.bar_width/2],  # bottom right (bar).
            [total_height/2 , -self.bar_width/2],  # top right (bar).
            [total_height/2 , self.bar_width/2],  # top left (bar).
            [total_height/2 - self.bar_height, self.bar_width/2],  # bottom left (bar).
            [total_height/2 - self.bar_height, self.stem_width/2],  # left armpit.
        ])

class ObjectPolygon(MovableObject):
    def __init__(self, name: str, vertices: torch.Tensor):
        """
        Initialize a polygon object with arbitrary vertices.
        
        Args:
            name: Name of the object
            vertices: Tensor of shape (N, 2) containing the vertices in counter-clockwise order
                     relative to the object's center. Each vertex is [x, y] in meters.
        """
        super().__init__(name)
        self.vertices = vertices
        self.type = "polygon"



