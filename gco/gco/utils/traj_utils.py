# # General imports.
# import torch

# # Project imports.
# from gco.config import Config as cfg

# def normalize_trajectory(trajectory: torch.Tensor):
#     """
#     Normalize a trajectory to be between -1 and 1.
#     :param trajectory: The trajectory to normalize, in meters.
#     :return: The normalized trajectory.
#     """
#     return trajectory / cfg.observation_side_pixels

# def unnormalize_trajectory(trajectory: torch.Tensor):
#     """
#     Unnormalize a trajectory to be in in the range [-1, 1].
#     :param trajectory: The trajectory to unnormalize, in normalized units.
#     :return: The unnormalized trajectory.
#     """
#     return trajectory * cfg.observation_side_pixels