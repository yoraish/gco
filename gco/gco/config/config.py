"""
Configuration settings for the GCO project.
"""
# General imports.
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import torch

# Automatically detect project root by finding the directory containing this file
# and going up to the gco-dev root directory
_current_file = Path(__file__).resolve()
PROJECT_ROOT = _current_file.parent.parent.parent.parent


class Config:
    ####################
    # Project settings.
    ####################
    # Project root.
    
    ####################
    # Checkpoint paths.
    ####################
    # Contact model checkpoint paths.
    chkpt_dir_cogen: Path = PROJECT_ROOT / "data/checkpoints/cogen_model"

    # Specific checkpoints.
    class checkpoints:
        c_r_flex: Path = PROJECT_ROOT / "data/checkpoints/cogen_model/cogen_model_cogen_model_20250926_125948_epoch_999_r_c_flex.pth"


    ####################
    # Training settings.
    ####################
    # Training settings.
    batch_size: int = 256
    learning_rate: float = 0.001
    num_epochs: int = 1000
    # device: str = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    ####################
    # Data settings.
    ####################
    data_dir_contacts_trajs_test: Path = PROJECT_ROOT / "data/gco_push_dataset_test"

    data_dir_contacts_trajs_r_c_flex: Path = PROJECT_ROOT / "data/datasets/r_c_flex"


    ####################
    # Co-generation model settings.
    ####################
    num_steps_denoise_cogen = 10
    smoothness_weight = 0.5
    source_distribution_discrete = "mask"

    ####################
    # General settings.
    ####################

    seed: int = 42
    # Map/observation size.
    map_side_meters: float = 4.0  # [-2, 2] x [-2, 2] meters.
    # Observation size. In meters.
    observation_side_meters: float = 1.0  # [-0.5, 0.5] x [-0.5, 0.5] meters.
    OBS_M = observation_side_meters

    # Low-resolution grid for model input, visualization, and generation. In pixels.
    observation_side_pixels: int = 128
    OBS_PIX = observation_side_pixels
    V = OBS_PIX + 1 if source_distribution_discrete == "mask" else OBS_PIX # A factorized vocabulary. x, y for each contact point, each getting a token. Adding 1 for the mask token.
    # Mask token for unused robots/contact points
    mask_token: int = OBS_PIX + 1
    # Pixel value for mask positions (center of image)
    pixel_value_mask: int = OBS_PIX + 1
    # Mask corruption probability.
    mask_corruption_prob: float = 0.0
    # Number of steps (horizon) for push trajectories.
    H: int = 20
    # Number of contact points.
    N: int = 3
    # Shape size ranges. In meters.
    class shape_size_range:
        class rectangle:
            class width:
                min_meters: float = 0.15
                max_meters: float = 0.5
            class height:
                min_meters: float = 0.15
                max_meters: float = 0.5
        class triangle:
            class side_length:
                min_meters: float = 0.15
                max_meters: float = 0.5
        class half_moon:
            class radius:
                min_meters: float = 0.4
                max_meters: float = 0.8
        class t_shape:
            class bar_width:
                min_meters: float = 0.15
                max_meters: float = 0.5
            class bar_height:
                min_meters: float = 0.1
                max_meters: float = 0.25
            class stem_width:
                min_meters: float = 0.1
                max_meters: float = 0.25
            class stem_height:
                min_meters: float = 0.15
                max_meters: float = 0.5
        class circle:
            class radius:
                min_meters: float = 0.1
                max_meters: float = 0.3
        class polygon:
            class radius:
                min_meters: float = 0.1
                max_meters: float = 0.4
            class num_vertices:
                min_vertices: int = 3
                max_vertices: int = 9

    class motion_primitives:
        object_linear = {
            "fwd":        {"dx":  0.2,   "dy":  0.0, "dtheta":  0.0},
            "bwd":        {"dx": -0.2,   "dy":  0.0, "dtheta":  0.0},
            "left":       {"dx":  0.0,   "dy":  0.2, "dtheta":  0.0},
            "right":      {"dx":  0.0,   "dy": -0.2, "dtheta":  0.0},
        }
    shape_types = ["circle", "rectangle"]
    # shape_types = ["circle", "rectangle", "t_shape"]

    ##############
    # Planner settings.
    ##############
    # Planner settings.
    class planner_params:
        class iterative_push:
            max_iterations: int = 10
            goal_tolerance_position: float = 0.15
            goal_tolerance_orientation: float = 0.2
            min_push_distance: float = 0.02
            visualization_interval: int = 1
            save_visualizations: bool = False

    ####################
    # Experiment tracking.
    ####################
    experiment_name: str = "default_experiment"
    log_dir: Path = Path("logs")
    wandb_project: Optional[str] = None

    ####################
    # Physics.
    ####################
    dt = 0.003

    ####################
    # Normalization settings.
    ####################
    # Scale factor for trajectory normalization during training.
    # This is a ballpark number to normalize trajectory data to reasonable scale.
    trajectory_scale_factor: float = 1

    ####################
    # Robot settings.
    ####################
    robot_radius = 0.05
    robot_height = 0.03

    ####################
    # Object settings.
    ####################
    object_height = 0.04

# Define the iterative_push configuration after the Config class is fully defined   
Config.planner_params.iterative_push = {
    "max_iterations": 20,
    "goal_tolerance_position": 0.15,
    "goal_tolerance_orientation": 0.2,
    "min_push_distance": 0.02,
    "visualization_interval": 1, # Save an image of the model generation every n iterations.
    "save_visualizations": True, # Whether to save visualizations of the planning process. This is of the planner output and the generation contacts and push trajectories.
    "output_dir": Path("output/iterative_push_example"),
    "persistent_simulator": True, # Use persistent MuJoCo instance.
    "model_checkpoint": Config.checkpoints.c_r_flex, # Use the model checkpoint from the config.
    "model_type": "discrete", # Model type: "discrete" or "continuous"
    "planner_type": "a_star"
}