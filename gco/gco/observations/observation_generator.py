# General imports.
from typing import Tuple
import torch
import matplotlib.pyplot as plt

# Project imports.
from gco.world.objects import MovableObject, ObjectCircle, ObjectRectangle, ObjectPolygon
from gco.config import Config as cfg

class ObservationGenerator:
    def __init__(self):
        pass

    def get_object_centric_observation(self, object: MovableObject, size: Tuple, visualize: bool = False):
        if isinstance(object, ObjectCircle):
            observation = self.get_object_centric_observation_circle(object, size, visualize)
        elif isinstance(object, ObjectRectangle):
            observation = self.get_object_centric_observation_rectangle(object, size, visualize)
        elif isinstance(object, ObjectPolygon):
            observation = self.get_object_centric_observation_polygon(object, size, visualize)
        else:
            raise ValueError(f"Unsupported object type for observation generation: {type(object)}")
        
        if visualize:
            plt.imshow(observation.cpu().numpy())
            plt.show()
        return observation.float()

    def get_object_centric_observation_circle(self, object: ObjectCircle, size: Tuple, visualize: bool = False) -> torch.Tensor:
        radius = object.radius
        # Convert radius in meters to pixels
        radius_pixels = radius / (cfg.OBS_M / cfg.OBS_PIX)
        H, W = size
        # Create a grid of (x, y) coordinates with the center at (H/2, W/2)
        y = torch.arange(H).float() - (H - 1) / 2
        x = torch.arange(W).float() - (W - 1) / 2
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        dist = torch.sqrt(xx**2 + yy**2)
        observation = (dist < radius_pixels)      
        return observation

    def get_object_centric_observation_rectangle(self, object: ObjectRectangle, size: Tuple, visualize: bool = False) -> torch.Tensor:
        width = object.width
        height = object.height
        # Convert width and height in meters to pixels.
        width_pixels = width / (cfg.OBS_M / cfg.OBS_PIX)
        height_pixels = height / (cfg.OBS_M / cfg.OBS_PIX)
        H, W = size
        # Create a grid of (x, y) coordinates with the center at (H/2, W/2)
        y = torch.arange(H).float() - (H - 1) / 2
        x = torch.arange(W).float() - (W - 1) / 2
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        observation = (torch.abs(xx) < width_pixels / 2) & (torch.abs(yy) < height_pixels / 2)
        return observation

    def get_object_centric_observation_polygon(self, object: ObjectPolygon, size: Tuple, visualize: bool = False) -> torch.Tensor:
        """
        Generate observation for a polygon object.
        Uses point-in-polygon test to determine which pixels are inside the polygon.
        """
        # Get the vertices in the object frame and in meters.
        vertices = object.vertices

        # Convert vertices from meters to pixels. This is only the scale for now.
        scale_factor = cfg.OBS_PIX / cfg.OBS_M
        vertices_pixels = vertices * scale_factor

        # Exchange between x and y.
        vertices_pixels_flipped = torch.stack([vertices_pixels[:, 1], vertices_pixels[:, 0]], dim=1)
        # Flip the y-axis.
        vertices_pixels_flipped[:, 1] = -vertices_pixels_flipped[:, 1]
        # Flip the x-axis.
        vertices_pixels_flipped[:, 0] = -vertices_pixels_flipped[:, 0]
        vertices_pixels = vertices_pixels_flipped
        
        H, W = size
        # Create a grid of (x, y) coordinates with the center at (H/2, W/2)
        y = torch.arange(H, device=vertices.device).float() - (H - 1) / 2
        x = torch.arange(W, device=vertices.device).float() - (W - 1) / 2
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        
        # Flatten coordinates for vectorized processing
        points = torch.stack([xx.flatten(), yy.flatten()], dim=1)
        
        # Point-in-polygon test using ray casting algorithm
        observation = self._point_in_polygon(points, vertices_pixels)
        observation = observation.reshape(H, W)
        
        return observation
    
    def _point_in_polygon(self, points: torch.Tensor, vertices: torch.Tensor) -> torch.Tensor:
        """
        Point-in-polygon test using ray casting algorithm.
        
        Args:
            points: Tensor of shape (N, 2) containing query points
            vertices: Tensor of shape (M, 2) containing polygon vertices in counter-clockwise order
            
        Returns:
            Boolean tensor of shape (N,) indicating which points are inside the polygon
        """
        x, y = points[:, 0], points[:, 1]
        n_vertices = vertices.shape[0]
        
        # Initialize result
        inside = torch.zeros(points.shape[0], dtype=torch.bool, device=points.device)
        
        # Ray casting algorithm
        for i in range(n_vertices):
            j = (i + 1) % n_vertices
            xi, yi = vertices[i, 0], vertices[i, 1]
            xj, yj = vertices[j, 0], vertices[j, 1]
            
            # Check if point is on the same side of the edge as the ray
            # and if the ray intersects the edge
            intersect = ((yi > y) != (yj > y)) & (x < (xj - xi) * (y - yi) / (yj - yi) + xi)
            inside = inside ^ intersect
        
        return inside