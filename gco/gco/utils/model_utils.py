# General imports.
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from pathlib import Path
import os 
import datetime

# Project imports.
from gco.config import Config as cfg
from gco.utils.transform_utils import Transform2

class SinusoidalTimeEmbed(nn.Module):
    """Embed scalar t∈[0,1] into a 1-D vector."""
    def __init__(self, dim: int = 32):
        super().__init__()
        self.dim = dim
        # Pre-compute frequencies during initialization
        self.register_buffer('freqs', 2 * math.pi * torch.arange(dim // 2) / (dim // 2))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        t: shape (B,) or (B,1) with values in [0,1]
        returns: (B, dim)
        """
        # Ensure t is on the same device as freqs
        t = t.to(self.freqs.device)
        # Handle both (B,) and (B,1) inputs
        t = t.reshape(-1, 1)  # (B,1)
        
        # Broadcast multiplication for batched inputs
        # freqs: (dim//2,) -> (1,dim//2) -> (B,dim//2) with t: (B,1)
        freqs_t = self.freqs.unsqueeze(0) * t  # (B,dim//2)
        
        # Compute sin and cos embeddings
        sin_emb = torch.sin(freqs_t)  # (B,dim//2)
        cos_emb = torch.cos(freqs_t)  # (B,dim//2)
        
        # Concatenate along the last dimension
        emb = torch.cat([sin_emb, cos_emb], dim=-1)  # (B,dim)
        return emb

class ResidualBlock(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        self.conv1 = nn.Conv2d(c, c, 3, padding=1)
        self.conv2 = nn.Conv2d(c, c, 3, padding=1)
        self.norm1 = nn.BatchNorm2d(c)
        self.norm2 = nn.BatchNorm2d(c)

    def forward(self, x):
        h = F.relu(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        return F.relu(x + h)
    

def clamp_obs_pix(t: torch.Tensor) -> torch.Tensor:
    """Clamp to [0, PIX-1] after rounding, so we never index out of bounds."""
    return t.round().clamp_(0, cfg.OBS_PIX - 1).to(torch.int64)

def tokens_onehot_to_pixels(tokens: torch.Tensor) -> torch.Tensor:
    """
    tokens : (B, N, V) one-hot/logits
    returns: (B, N, 2)  (row, col) in the image frame.
    """
    return tokens_to_pixels(tokens.argmax(dim=-1))

def tokens_to_pixels(tokens: torch.Tensor) -> torch.Tensor:
    """
    tokens : (B, N * 2) tokens
    returns: (B, N, 2)  (row, col) in the image frame.
    """
    # The odd indices are the rows, the even indices are the cols.
    # Tokens are ordered consecutive pairs. r1, c1, r2, c2, ...
    rows = tokens[..., 0::2].to(torch.float32)                       # (B, N)
    cols = tokens[..., 1::2].to(torch.float32)                       # (B, N)

    return torch.stack((rows, cols), dim=-1).to(torch.int64)

def pixels_to_tokens(pixels: torch.Tensor) -> torch.Tensor:
    """
    pixels : (B, N, 2)  (row, col) in the image frame. This is the high res image.
    returns: (B, N * 2) tokens, where the vocabulary is H*W of the mask image.
    """
    # Tokens are ordered consecutive pairs. r1, c1, r2, c2, ...
    ids = pixels.reshape(pixels.shape[0], -1)  # (B, N * 2)
    return ids.to(dtype=torch.int64)

def pixels_to_meters(pixels: torch.Tensor) -> torch.Tensor:
    """
    pixels : (B, N, 2)  (row, col) in the image frame.
    returns: (B, N, 2)  (x [m], y [m]) in the robot/object frame.
    """
    # Pixels arond the center of the image then scaled to meters.
    meters = (pixels - cfg.OBS_PIX / 2.0) * (cfg.OBS_M / cfg.OBS_PIX)
    x = -meters[:, :, 0]
    y = -meters[:, :, 1]
    meters = torch.stack((x, y), dim=-1)
    return meters

def tokens_to_meters_local(tokens: torch.Tensor) -> torch.Tensor:
    """
    tokens : (B, N * 2) tokens
    returns: (B, N, 2)  (x [m], y [m]) in the robot/object frame.
    """
    pixels = tokens_to_pixels(tokens)   
    meters = pixels_to_meters(pixels)
    return meters

def push_trajectories_pixels_local_to_meters_local(push_trajectories_pixels_local: torch.Tensor,
                                                   contact_points_meters_local: torch.Tensor) -> torch.Tensor:
    """
    push_trajectories_pixels_local : (B, N, H, 2)  (x [px], y [px]) in the robot/object frame.
    contact_points_meters_world : (B, N, 2)  (x [m], y [m]) in the world frame.
    returns: (B, N, H, 2)  (x [m], y [m]) in the world frame.
    """
    push_trajectories_pixels_local = push_trajectories_pixels_local.clone()
    push_trajectories_pixels_local[:, :, :, 1] = -push_trajectories_pixels_local[:, :, :, 1]
    push_trajectories_pixels_local[:, :, :, 0] = -push_trajectories_pixels_local[:, :, :, 0]
    contact_point_shift = contact_points_meters_local.unsqueeze(2)  # (B, N, 1, 2)
    meters = push_trajectories_pixels_local * (cfg.OBS_M / cfg.OBS_PIX) + contact_point_shift  # (B, N, H, 2)
    return meters

def push_trajectories_meters_local_to_pixels_local(push_trajectories_meters_local: torch.Tensor, contact_points_meters_local: torch.Tensor) -> torch.Tensor:
    """
    push_trajectories_meters_local : (B, N, H, 2)  (x [m], y [m]) in the robot/object frame.
    returns: (B, N, H, 2)  (x [px], y [px]) in the robot/object frame.
    """
    contact_point_shift = contact_points_meters_local.unsqueeze(2)  # (B, N, 1, 2)
    pixels = (push_trajectories_meters_local - contact_point_shift) * (cfg.OBS_PIX / cfg.OBS_M)
    # Flip the y-axis. and the x-axis.
    pixels[:, :, :, 1] = -pixels[:, :, :, 1]
    pixels[:, :, :, 0] = -pixels[:, :, :, 0]
    return pixels

def add_push_trajectories_to_paths(paths: dict, push_trajectories_meters_world: torch.Tensor) -> dict:
    """
    Append push trajectories to robot paths using greedy nearest-neighbor assignment.
    
    Args:
        paths: Dictionary mapping robot names to paths (N robots, each with H waypoints of (x, y, theta))
        push_trajectories_meters_world: Tensor of shape (B, N, H, 3) with (x, y, theta) in world frame
    
    Returns:
        Dictionary with extended paths including push trajectories
    
    Note:
        Uses greedy assignment based on distance between path endpoints and trajectory start points.
        For optimal assignment, consider using Hungarian algorithm (scipy.optimize.linear_sum_assignment).
    """
    paths_extended = {}
    available_push_trajectory_idxs = list(range(push_trajectories_meters_world.shape[1]))
    for robot_name, path_robot in paths.items():
        nearest_push_trajectory_index = None
        nearest_push_trajectory_distance = float('inf')
        for i in available_push_trajectory_idxs:
            # Find the nearest last-state in the path to the push trajectory.
            state_last_path = torch.tensor(path_robot[-1], device=cfg.device)
            state_first_push = push_trajectories_meters_world[:, i, 0]
            distance = torch.norm(state_last_path - state_first_push, dim=-1)
            if distance < nearest_push_trajectory_distance:
                nearest_push_trajectory_distance = distance
                nearest_push_trajectory_index = i

        if nearest_push_trajectory_index is None:
            print(f"Warning: No push trajectory found for {robot_name}. The options were: {available_push_trajectory_idxs} which begin at {[push_trajectories_meters_world[:, i, 0].tolist() for i in available_push_trajectory_idxs]}.")
            return paths
        
        if nearest_push_trajectory_distance > 0.2:
            print(f"Warning: Push trajectory {nearest_push_trajectory_index} is too far from the path {robot_name} that starts at {path_robot[0]}. The distance is {nearest_push_trajectory_distance}.")
            return paths
        
        available_push_trajectory_idxs.remove(nearest_push_trajectory_index)
        # Add the push trajectory to the path.
        paths_extended[robot_name] = paths[robot_name] + push_trajectories_meters_world[:, nearest_push_trajectory_index].squeeze().tolist()  # (H, 3)
    # Return the paths.
    return paths_extended

def get_recent_model(chkpt_dir: Path):
    models = os.listdir(chkpt_dir)
    # Add creation dates.
    models_with_dates = []
    for model in models:
        creation_date = datetime.datetime.fromtimestamp(os.path.getctime(chkpt_dir / model)).strftime("%Y-%m-%d %H:%M:%S")
        models_with_dates.append((model, creation_date))

    # Sort by creation date and time.
    models_with_dates.sort(key=lambda x: x[1], reverse=True)

    # Print the models.
    # print("Available models:")
    # print("\n * ".join([f"{model} (created {date})" for model, date in models_with_dates]))

    # Choose the most recent model.
    recent_chkpt_path = chkpt_dir / models_with_dates[0][0]
    print("Using", models_with_dates[0][0])
    return recent_chkpt_path