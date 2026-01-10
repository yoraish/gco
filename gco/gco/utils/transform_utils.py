"""
Collaborative Motion Planning with Negotiated Diffusion Models
Author: Yorai Shaoul, 2025.

Pose Utils: data structures and utilities for poses and transformations.
"""
# General imports.
import torch
import math
from typing import Tuple, Dict, Union
from abc import ABC, abstractmethod
import random
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy import ndimage

# Project imports.
from gco.config import Config as cfg

class Transform(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def get_t(self) -> torch.Tensor:
        """Get the translation component."""
        pass

    @abstractmethod
    def get_R(self) -> torch.Tensor:
        """Get the rotation matrix."""
        pass

    def to(self, device):
        """Move the transform to a device."""
        raise NotImplementedError

    def cpu(self):
        """Move the transform to CPU."""
        raise NotImplementedError
    
    def to_tensor(self) -> torch.Tensor:
        """Convert the transform to a tensor."""
        raise NotImplementedError

class Transform2(Transform):
    def __init__(self, t: torch.Tensor, theta: torch.Tensor):
        """
        Initialize a 2D transform with translation and rotation.
        Args:
            t: Translation vector (2, )
            theta: Rotation angle in radians (1,)
        """
        super().__init__()
        self.t = t if isinstance(t, torch.Tensor) else torch.tensor(t)

        # Ensure correct shapes
        assert self.t.shape[-1] == 2, "Translation vector must be 2D"
        if len(theta.shape) == 0:
            theta = theta.unsqueeze(0)

        # Compute rotation matrix
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)
        self.R = torch.stack([
            torch.stack([cos_theta, -sin_theta], dim=-1),
            torch.stack([sin_theta, cos_theta], dim=-1)
        ], dim=-2)

    def get_t(self) -> torch.Tensor:
        """Get translation component"""
        return self.t

    def get_R(self) -> torch.Tensor:
        """Get rotation matrix"""
        return self.R

    def get_theta(self) -> torch.Tensor:
        """Get rotation angle"""
        return torch.atan2(self.R[..., 1, 0], self.R[..., 0, 0])

    def inverse(self) -> 'Transform2':
        """Compute the inverse transform"""
        # Inverse rotation
        inv_R = self.R.transpose(-2, -1)
        inv_theta = torch.atan2(inv_R[..., 1, 0], inv_R[..., 0, 0])

        # Inverse translation
        inv_t = -torch.matmul(inv_R, self.t.unsqueeze(-1)).squeeze(-1)

        return Transform2(inv_t, inv_theta)
    
    def __mul__(self, other: 'Transform2') -> 'Transform2':
        """Compose two transforms."""
        out = Transform2(
            self.t + torch.matmul(self.R, other.t),
            self.get_theta() + other.get_theta()
        )
        out.t = out.t.squeeze()
        return out

    def to(self, device) -> 'Transform2':
        """Move transform to specified device"""
        return Transform2(
            self.t.to(device),
            self.get_theta().to(device)
        )

    def cpu(self):
        """Move transform to CPU"""
        return Transform2(
            self.t.cpu(),
            self.get_theta().cpu()
        )

    def to_tensor(self) -> torch.Tensor:
        """Convert the transform to a tensor. The tensor is of shape (3,) of [dx, dy, theta (radians)].
        """
        return torch.cat([self.t, self.get_theta()], dim=-1)
    
    @staticmethod
    def random(dx_min: float, # Meters.
               dx_max: float, # Meters.
               dy_min: float, # Meters.
               dy_max: float, # Meters.
               angle_min: float,   # Radians. 
               angle_max: float):  # Radians.
        """Generate a random transformation."""
        angle = random.uniform(angle_min, angle_max)
        dx = random.uniform(dx_min, dx_max)
        dy = random.uniform(dy_min, dy_max)

        return Transform2(
            t=torch.tensor([dx, dy], device=cfg.device),
            theta=torch.tensor([angle], device=cfg.device),
        )

    def __str__(self):
        return f"<{self.t[0].item():.4f}, {self.t[1].item():.4f}, {self.get_theta().item():.4f}>"

    def __repr__(self):
        return self.__str__()
    
class Translation2(Transform2):
    def __init__(self, t: torch.Tensor):
        """
        Initialize a 2D translation. This is a specialization of Transform2 with zero rotation.
        Args:
            t: Translation vector (2,)
        """
        super().__init__(t, torch.zeros(1, device=t.device))

    def get_R(self) -> torch.Tensor:
        """Get rotation matrix (identity for translations)"""
        return torch.eye(2)

    def get_theta(self) -> torch.Tensor:
        """Get rotation angle (zero for translations)"""
        return torch.zeros(1)

    def inverse(self) -> 'Translation2':
        """Compute the inverse translation"""
        return Translation2(-self.t)
    
    def __mul__(self, other: 'Translation2') -> 'Translation2':
        """Compose two translations."""
        return Translation2(self.t + other.t)

    def to(self, device) -> 'Translation2':
        """Move translation to specified device"""
        return Translation2(self.t.to(device))

    def cpu(self):
        """Move translation to CPU"""
        return Translation2(self.t.cpu())
    
    def to_tensor(self) -> torch.Tensor:
        """Convert the translation to a tensor. The tensor is of shape (3,) of [dx, dy, 0]."""
        return torch.cat([self.t, torch.zeros(1, device=self.t.device)], dim=-1)
    
def mujoco_quat_to_z_theta(quat: torch.Tensor) -> torch.Tensor:
    """Convert a quaternion (x, y, z, w) to a theta angle along the z axis (radians, [-pi, pi])."""
    if isinstance(quat, torch.Tensor):  # If quat is a tensor, return a tensor.
        theta = -R.from_quat(quat.cpu().numpy()).as_euler('zyx', degrees=False)[2] - np.pi / 2 + np.pi
        return torch.tensor(theta, device=quat.device, dtype=torch.float32)
    else:
        raise ValueError(f"Quaternion must be a tensor, got {type(quat)}")
    
def z_theta_to_mujoco_theta(theta: torch.Tensor) -> torch.Tensor:
    """Convert a theta angle along the z axis (radians, [-pi, pi]) to a quaternion."""
    theta = theta + np.pi / 2 
    return theta

# ====================
# Mask transformation utilities.
# ====================
def pixel_dist_to_meters(pixels: int) -> float:
    return pixels * cfg.OBS_M / cfg.OBS_PIX
    
def meters2pixel_dist(meters: float, resolution: str = "high") -> int:
    return int(meters * cfg.OBS_PIX / cfg.OBS_M)

def apply_transform_to_mask(mask: torch.Tensor, transform: Transform2):
    """
    Apply a transformation to a mask.
    
    Args:
        mask: The mask to transform
        transform: The transformation to apply (rotation and translation)
    
    Returns:
        The transformed mask
    """
    angle = transform.get_theta()
    mask = ndimage.rotate(mask.cpu().numpy(), angle.cpu().numpy().squeeze() * 180 / 3.14159, reshape=False, order=0)
    dy, dx = transform.get_t().squeeze()
    # Convert from meters to pixels
    dy, dx = meters2pixel_dist(dy), meters2pixel_dist(dx)
    mask = ndimage.shift(mask, (dy, dx), order=0)
    return torch.tensor(mask, device=cfg.device)

def apply_transform_to_contacts(contacts: torch.Tensor, transform: Transform2):
    """
    Apply a transformation to a set of contacts -- those are in the robot/object frame.
    :param contacts: the contacts to apply the transformation to (pixels, high-res).
    :param transform: the transformation to apply.
    :return: the transformed contacts (rounded and clamped to image bounds).
    """
    # Convert pixel coordinates to float for transformation
    contacts = contacts.float()

    # If the transform is a torch tensor, convert it to a Transform2 object.
    assert isinstance(transform, Transform2), "Transform must be a Transform2 object."

    # Get rotation and translation
    theta = transform.get_theta().squeeze()
    dy, dx = transform.get_t().squeeze()

    # Convert translation from meters to pixels
    dy = meters2pixel_dist(dy)
    dx = meters2pixel_dist(dx)

    # Center of the image (to rotate around)
    center = torch.tensor([cfg.observation_side_pixels / 2.0, cfg.observation_side_pixels / 2.0], device=contacts.device)

    # Translate to origin, rotate, translate back
    shifted = contacts - center
    rot_matrix = torch.tensor([[torch.cos(theta), -torch.sin(theta)],
                            [torch.sin(theta),  torch.cos(theta)]], device=contacts.device)
    rotated = (rot_matrix @ shifted.T).T
    transformed = rotated + center + torch.tensor([dy, dx], device=contacts.device)

    # Round to nearest integer pixel and clamp to image bounds
    transformed = transformed.round().long()
    transformed = torch.clamp(transformed, 0, cfg.observation_side_pixels - 1)

    return transformed

def points_local_to_world(X_local_p: torch.Tensor, transform_world_local: Transform2) -> torch.Tensor:
    """
    Convert a point from the local frame to the world frame.
    :param X_local_p: the points in the local frame (..., 2).
    :param transform_world_local: the transform from the local frame to the world frame.
    :return: the points in the world frame (..., 2).
    """
    shape_out = X_local_p.shape
    X_local_p_flat = X_local_p.reshape(-1, 2)  # (M, 2)
    # Points to homogeneous coordinates.
    X_local_p_h = torch.cat([X_local_p_flat, torch.ones((X_local_p_flat.shape[0], 1), device=X_local_p.device)], dim=-1)  # (M, 3)

    # Create a transformation matrix from the transform.
    X_world_local = torch.eye(3, device=X_local_p.device)
    X_world_local[:2, :2] = transform_world_local.get_R()
    X_world_local[:2, 2] = transform_world_local.get_t()

    # Apply transformation
    X_world_p_h = (X_world_local @ X_local_p_h.T).T  # (M, 3)
    X_world_p = X_world_p_h[:, :2] / X_world_p_h[:, 2:3]  # (M, 2)

    return X_world_p.reshape(shape_out)

def points_world_to_local(X_world_p: torch.Tensor, transform_world_local: Transform2) -> torch.Tensor:
    """
    Convert a point from the world frame to the local frame.
    :param X_world_p: the points in the world frame (..., 2).
    :param transform_world_local: the transform from the world frame to the local frame.
    :return: the points in the local frame (..., 2).
    """
    return points_local_to_world(X_world_p, transform_world_local)