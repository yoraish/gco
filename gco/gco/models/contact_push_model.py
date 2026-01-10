# A co-generation model for pushing objects. Discrete aspect: choosing contact points. Continuous aspect: generating push trajectories.

# General imports.
import torch
from typing import Dict, Any, List, Optional, Tuple
import torch.nn.functional as F
from torch import nn, Tensor
import math
from flow_matching.path import MixtureDiscreteProbPath
from flow_matching.path.scheduler import PolynomialConvexScheduler
from flow_matching.solver import MixtureDiscreteEulerSolver
from flow_matching.utils import ModelWrapper, ModelWrapperCoGen
from flow_matching.loss import MixturePathGeneralizedKL
from flow_matching.path.scheduler import CondOTScheduler
from flow_matching.path import AffineProbPath
from flow_matching.solver import Solver, ODESolver, CoGenSolver

# Project imports.
from gco.config import Config as cfg
from gco.utils import SinusoidalTimeEmbed, ResidualBlock

# Input: (noisy) trajectory, (noisy) contact points tokens, context (observation mask, goal mask (after the push is applied), robot positions).
# Output (intermediate): velocity field for trajectories, discrete velocity contact points tokens with assignments to robots (matched by index).
# Output (final): (clean) trajectories, (clean) contact points tokens with assignments to robots (matched by index).
# Architecture for co-generation:
#     A separate head for the trajectory and a separate head for the contact points, each taking the other as context, in addition to the context.
#     The output are treated as the trajectory and contact points tokens, respectively.

####################
# Improved embedding networks. #
####################

# Activation class
class Swish(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        return torch.sigmoid(x) * x

class MaskConvEncoder(torch.nn.Module):
    """
    Encodes a binary mask into a 2D field.
    Input: mask (B, H=128, W=128) - Binary mask where B is batch size, H and W are spatial dimensions
    Output: feature_map (B, C, W=32, W=32) - Feature map where C is out_channels
    """
    def __init__(self, out_channels: int = 64, num_res_blocks: int = 3):
        super().__init__()
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        
        # Add channel dimension to input mask
        self.stem = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1, out_channels=self.out_channels, kernel_size=3, padding=1, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=self.out_channels, out_channels=self.out_channels, kernel_size=3, padding=1, stride=2),
            torch.nn.ReLU(),
        )
        self.res_blocks = torch.nn.Sequential(*[ResidualBlock(self.out_channels) for _ in range(self.num_res_blocks)])

    def forward(self, mask: torch.Tensor) -> torch.Tensor:
        # Add channel dimension if not present
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)  # (B, 1, H, W)
        x = self.stem(mask)
        x = self.res_blocks(x)
        return x

class SimpleMaskEncoder(torch.nn.Module):
    """
    Simple but effective mask encoder that converts binary masks to feature vectors.
    """
    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(1, 64, 3, padding=1, stride=2),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 128, 3, padding=1, stride=2),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, hidden_dim, 3, padding=1),
            torch.nn.BatchNorm2d(hidden_dim),
            torch.nn.ReLU(),  # (B, hidden_dim, H', W')
        )

    def forward(self, mask: torch.Tensor) -> torch.Tensor:
        # Add channel dimension if not present
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)  # (B, 1, H, W)
        
        features = self.encoder(mask)  # (B, hidden_dim, H', W')
        # Global average pooling to get feature vector
        features = torch.nn.functional.adaptive_avg_pool2d(features, 1).squeeze(-1).squeeze(-1)  # (B, hidden_dim)
        return features

class ContactEmbedder(torch.nn.Module):
    """
    Embeds contact point tokens into feature vectors.
    """
    def __init__(self, hidden_dim: int = 128, vocab_size: int = None):
        super().__init__()
        self.hidden_dim = hidden_dim
        if vocab_size is None:
            raise ValueError("Vocab size must be provided.")
        else:
            self.vocab_size = vocab_size
            
        self.embedder = torch.nn.Sequential(
            torch.nn.Embedding(self.vocab_size, 64),
            torch.nn.Linear(64 * cfg.N * 2, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N * 2) the contact point tokens
        Returns:
            features: (B, hidden_dim) embedded contact features
        """
        B = x.shape[0]
        
        # Embed contact points
        embedded_contacts = self.embedder[0](x)  # (B, N*2, 64)
        embedded_contacts = embedded_contacts.reshape(B, -1)  # (B, N*2*64)
        contact_features = self.embedder[1:](embedded_contacts)  # (B, hidden_dim)
        
        return contact_features

class TransformEmbedder(torch.nn.Module):
    """
    Embeds object transform information into feature vectors.
    """
    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        self.embedder = torch.nn.Sequential(
            torch.nn.Linear(3, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, hidden_dim),
            torch.nn.ReLU(),
        )

    def forward(self, transform_object: torch.Tensor) -> torch.Tensor:
        """
        Args:
            transform_object: (B, 3) the transform of the object (dx, dy, dtheta)
        Returns:
            features: (B, hidden_dim) embedded transform features
        """
        return self.embedder(transform_object)

class BudgetEmbedder(torch.nn.Module):
    """
    Embeds robot budget information into feature vectors.
    """
    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        self.embedder = torch.nn.Sequential(
            torch.nn.Linear(1, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, hidden_dim),
            torch.nn.ReLU(),
        )

    def forward(self, budget: torch.Tensor) -> torch.Tensor:
        """
        Args:
            budget: (B,) the robot budget for each batch item
        Returns:
            features: (B, hidden_dim) embedded budget features
        """
        # Convert to float and add dimension for linear layer
        budget_float = budget.float().unsqueeze(-1)  # (B, 1)
        return self.embedder(budget_float)

class TimeEmbedder(torch.nn.Module):
    """
    Embeds time information using sinusoidal embeddings.
    """
    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.time_embedder = SinusoidalTimeEmbed(dim=hidden_dim)

    def forward(self, t: float) -> torch.Tensor:
        """
        Args:
            t: float: the current time step in the flow-matching process
        Returns:
            features: (B, hidden_dim) embedded time features
        """
        return self.time_embedder(t)

class TrajectoriesEncoder(torch.nn.Module):
    """
    Encodes trajectories into feature vectors.
    """
    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(cfg.N * cfg.H * 2, self.hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_dim, self.hidden_dim),
            torch.nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, H, 2) the trajectories. Batch, num robots, horizon, 2. All trajectories begin at the origin. They will later be transformed to the actual starting position.
        Returns:
            features: (B, hidden_dim) embedded trajectory features
        """
        B = x.shape[0]
        x = x.reshape(B, -1)  # (B, N*H*2)
        # Encode the trajectories.
        x_features = self.encoder(x)  # (B, hidden_dim)
        
        return x_features

# Model for contacts only.
class ContactModel(torch.nn.Module):
    """
    Lightweight contact model that better learns to place contact points given object observations.
    Uses efficient tensor operations without any for loops or complex attention mechanisms.
    """
    def __init__(self, hidden_dim: int = 128, vocab_size: int = None):
        super().__init__()
        self.hidden_dim = hidden_dim
        if vocab_size is None:
            raise ValueError("Vocab size must be provided.")
        self.vocab_size = vocab_size
        
        # Encoders and embedders
        self.mask_encoder = SimpleMaskEncoder(hidden_dim)
        self.contact_embedder = ContactEmbedder(hidden_dim, vocab_size)
        self.transform_embedder = TransformEmbedder(hidden_dim)
        self.time_embedder = TimeEmbedder(hidden_dim)
        
        # Feature fusion
        self.fusion = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim * 4, hidden_dim * 2),  # mask + contact + transform + time
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(hidden_dim * 2, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, self.vocab_size * cfg.N * 2)
        )

    def forward(self, 
                x: torch.Tensor, 
                t: float, # (B,)
                mask: torch.Tensor, 
                transform_object: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N * 2) the contact point tokens. Batch, num robots/contacts * 2 tokens per contact point. Ordered consecutive pairs.
            mask: (B, OBS_PIX, OBS_PIX)
            transform_object: (B, 3) the transform of the object (dx, dy, dtheta) from its current position (axis aligned at the origin). In meters and radians.
            t: float: the current time step in the flow-matching process.
        Returns:
            x: (B, N * 2, V) logits over the vocabulary. Batch, num robots/contacts * 2, logits over the V tokens.
        """
        B = x.shape[0]
        
        # Encode all inputs using separate embedders
        mask_features = self.mask_encoder(mask)  # (B, hidden_dim)
        contact_features = self.contact_embedder(x)  # (B, hidden_dim)
        transform_features = self.transform_embedder(transform_object)  # (B, hidden_dim)
        time_features = self.time_embedder(t)  # (B, hidden_dim)
        
        # Combine all features
        combined_features = torch.cat([mask_features, contact_features, transform_features, time_features], dim=1)  # (B, 4*hidden_dim)
        
        # Decode
        all_logits = self.fusion(combined_features)  # (B, V * N * 2)
        
        # Reshape to expected output format
        output = all_logits.reshape(B, cfg.N * 2, self.vocab_size)  # (B, N*2, V)
        
        return output
    

####################
# Model for trajectories, given contact points.
####################

class TrajectoryModel(torch.nn.Module):
    """
    Model for trajectories, given contact points.
    """
    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Encoders and embedders
        self.trajectory_encoder = TrajectoriesEncoder(hidden_dim).to(cfg.device)
        self.mask_encoder = SimpleMaskEncoder(hidden_dim)
        # Calculate vocabulary size based on distribution type for contact embedder
        contact_vocab_size = cfg.V
        self.contact_embedder = ContactEmbedder(hidden_dim, contact_vocab_size)
        self.transform_embedder = TransformEmbedder(hidden_dim)
        self.time_embedder = TimeEmbedder(hidden_dim)
        
        # Feature fusion
        self.fusion = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim * 5, hidden_dim),  # mask + contact + transform + time + x
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, cfg.N * cfg.H * 2)
        )

    def forward(self, 
                x: torch.Tensor, 
                t: float,
                mask: torch.Tensor, 
                contact_tokens: torch.Tensor,
                transform_object: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, H, 2) the trajectories. Batch, num robots, horizon, 2. All trajectories begin at the origin. They will later be transformed to the actual starting position.
            t: float: the current time step in the flow-matching process.
            mask: (B, OBS_PIX, OBS_PIX)
            contact_tokens: (B, N * 2) the contact point tokens. Batch, num robots/contacts * 2 tokens per contact point. Ordered consecutive pairs. 
                This is the output of the contact model.
            transform_object: (B, 3) the transform of the object (dx, dy, dtheta) from its current position (axis aligned at the origin). In meters and radians.
        Returns:
            x: (B, N, H, 2) The flow-matching velocity for the batch. The velocity is over the space of N-robot trajectories.
        """
        B = x.shape[0]
        N = x.shape[1]
        H = x.shape[2]

        # Check if there is a need to repeat t.
        if t.dim() == 0:
            t = t.unsqueeze(0).repeat(B)  # (B,)

        # Encode all inputs using separate embedders
        mask_features = self.mask_encoder(mask)  # (B, hidden_dim)
        contact_features = self.contact_embedder(contact_tokens)  # (B, hidden_dim)
        transform_features = self.transform_embedder(transform_object)  # (B, hidden_dim)
        time_features = self.time_embedder(t)  # (B, hidden_dim)

        # Encode the trajectories.
        x_features = self.trajectory_encoder(x)  # (B, hidden_dim)
        
        # Combine all features
        combined_features = torch.cat([x_features, mask_features, contact_features, transform_features, time_features], dim=1)  # (B, 5*hidden_dim)
        
        # Decode
        all_trajectories = self.fusion(combined_features)  # (B, N * H * 2)
        
        # Reshape to expected output format
        return all_trajectories.reshape(B, N, H, 2)  # (B, N, H, 2)
    
####################
# Co-Generation Model.
####################

class ContactTrajectoryModel(torch.nn.Module):
    """
    Co-generation model for contact points and trajectories.
    This model is similar in structure to the ContactModel and TrajectoryModel, with a shared embedding space. This model has two heads, one for the contact points and one for the trajectories. The outputs of the model are, like before, just contact points logits and the trajectories.
    """

    def __init__(self, hidden_dim: int = 128, vocab_size: int = None):
        super().__init__()
        
        # Store trajectory scale for inference (will be set during training)
        self.trajectory_scale = cfg.trajectory_scale_factor
        
        # Set vocab size
        if vocab_size is None:
            raise ValueError("Vocab size must be provided.")
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        
        # Encoders and embedders.
        self.trajectory_encoder = TrajectoriesEncoder(hidden_dim).to(cfg.device)
        self.mask_encoder = SimpleMaskEncoder(hidden_dim)
        self.contact_embedder = ContactEmbedder(hidden_dim, self.vocab_size)
        self.transform_embedder = TransformEmbedder(hidden_dim)
        self.budget_embedder = BudgetEmbedder(hidden_dim)
        self.time_embedder = TimeEmbedder(hidden_dim)
        
        # Feature fusion
        self.head_contact = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim * 6, hidden_dim),  # mask + contact + transform + budget + time + x
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, cfg.N * 2 * self.vocab_size)
        )
        self.head_trajectory = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim * 6, hidden_dim),  # mask + contact + transform + budget + time + x
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, cfg.N * cfg.H * 2)
        )

    def forward(self, 
                x_d: torch.Tensor, 
                x_c: torch.Tensor,
                t: float,
                mask: torch.Tensor, 
                transform_object: torch.Tensor,
                budget: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x_d: (B, N * 2) the contact point tokens (discrete). Batch, num robots/contacts * 2 tokens per contact point. Ordered consecutive pairs.
            x_c: (B, N, H, 2) the trajectories (continuous). Batch, num robots, horizon, 2. All trajectories begin at the origin. They will later be transformed to the actual starting position.
            t: float: the current time step in the flow-matching process.
            mask: (B, OBS_PIX, OBS_PIX)
            transform_object: (B, 3) the transform of the object (dx, dy, dtheta) from its current position (axis aligned at the origin). In meters and radians.
            budget: (B,) optional tensor specifying the number of robots available for each batch item.
        Returns:
            x_d: (B, N * 2, V) logits over the vocabulary. Batch, num robots/contacts * 2, logits over the V tokens.
            x_c: (B, N, H, 2) The flow-matching velocity for the batch. The velocity is over the space of N-robot trajectories.
        """
        B = x_d.shape[0]

        # Check if there is a need to repeat t.
        if t.dim() == 0:
            t = t.unsqueeze(0).repeat(B)  # (B,)

        # Encode all inputs using separate embedders
        mask_features = self.mask_encoder(mask)  # (B, hidden_dim)
        contact_features = self.contact_embedder(x_d)  # (B, hidden_dim)
        transform_features = self.transform_embedder(transform_object)  # (B, hidden_dim)
        time_features = self.time_embedder(t)  # (B, hidden_dim)
        
        # Encode budget if provided, otherwise use zeros
        if budget is not None:
            budget_features = self.budget_embedder(budget)  # (B, hidden_dim)
        else:
            budget_features = torch.zeros(B, self.hidden_dim, device=mask.device)

        # Encode the trajectories.
        x_c_features = self.trajectory_encoder(x_c)  # (B, hidden_dim)
        
        # Combine all features
        combined_features = torch.cat([x_c_features, mask_features, contact_features, transform_features, budget_features, time_features], dim=1)  # (B, 6*hidden_dim)
        
        # Decode
        all_logits = self.head_contact(combined_features)  # (B, N * 2 * V)
        all_trajectories = self.head_trajectory(combined_features)  # (B, N * H * 2)
        
        # Reshape to expected output format
        v_d = all_logits.reshape(B, cfg.N * 2, self.vocab_size)  # (B, N*2, V)
        v_c = all_trajectories.reshape(B, cfg.N, cfg.H, 2)  # (B, N, H, 2)
        return v_d, v_c
    
    def generate(self, mask: torch.Tensor, transform_object: torch.Tensor, budget: torch.Tensor = None, seed: int = 42, smoothness_weight: float = 1.2) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Generate a contact point and trajectory.
        :param mask: (B, OBS_PIX, OBS_PIX)
        :param transform_object: (B, 3) the transform of the object (dx, dy, dtheta) from its current position (axis aligned at the origin). In meters and radians.
        :param budget: (B,) tensor specifying the number of robots available for each batch item. If None, uses all available robots.
        :param seed: Random seed for deterministic generation
        :param smoothness_weight: Weight for the smoothness guidance term (default 1.0, set to 0.0 to disable)
        :return: List of tuples, one for each generation step. Each tuple contains the contact points tokens and the push trajectories (torch.Tensor (B, N * 2, V), torch.Tensor (B, N, H, 2)).
        """
        
        # Set deterministic seed for generation
        torch.manual_seed(seed)
        
        smoothness_guide = TrajectorySmoothnessLoss(weight=smoothness_weight)
        
        class WrappedCoGenModel(ModelWrapperCoGen):
            def forward(self, x_d: Tensor, x_c: Tensor, t: Tensor, **extras) -> Tensor:
                vd, vc = self.model(x_d=x_d, x_c=x_c, t=t, **extras)

                vd = torch.softmax(vd, dim=-1)
                # Guidance: add grad of smoothness loss to continuous velocity
                if smoothness_weight > 0:
                    with torch.enable_grad():
                        x_c_grad = x_c.clone().detach().requires_grad_(True)
                        smooth_loss = smoothness_guide(x_c_grad)
                        grad = torch.autograd.grad(smooth_loss, x_c_grad, retain_graph=True, create_graph=True)[0]
                        vc = vc - grad  # Subtract grad to descend loss (make smoother).
                return vd, vc
        
        wrapped_cogen_model = WrappedCoGenModel(self)
        
        # Generation parameters.
        num_steps = cfg.num_steps_denoise_cogen
        step_size = 1 / num_steps
        # Visualization parameters.
        B = mask.shape[0]
        num_intermediates = num_steps #10

        # Probability paths. This is for sampling xd_t and xc_t.
        prob_path_continuous = AffineProbPath(scheduler=CondOTScheduler())
        scheduler = PolynomialConvexScheduler(n=2.0)
        prob_path_discrete = MixtureDiscreteProbPath(scheduler=scheduler)

        # Get initial samples; xd_0 and xc_0.
        # Starting with the discrete part, we have two options: random or mask.
        token_seq_len = cfg.N * 2  # Number of robots * 2 (two coords each), i.e., number of entries in the token sequence.

        if cfg.source_distribution_discrete == "uniform":
            xd_0 = torch.randint(size=(B, token_seq_len), high=self.vocab_size, device=cfg.device)
        elif cfg.source_distribution_discrete == "mask":
            mask_token = self.vocab_size - 1  # Last token is the mask token
            xd_0 = (torch.zeros(size=(B, token_seq_len), device=cfg.device) + mask_token).long()
        else:
            raise NotImplementedError

        # Get initial samples; xc_0.
        xc_0 = torch.randn(size=(B, cfg.N, cfg.H, 2), device=cfg.device)
        xc_0[:, :, 0, :] *= 0.0

        # Create a co-generation solver.
        solver = CoGenSolver(model=wrapped_cogen_model,
                            num_steps=num_steps,
                            prob_path_discrete=prob_path_discrete,
                            vocabulary_size=self.vocab_size)
        import time
        gentime_start = time.time()
        sol_l, time_grid = solver.sample(
            x_d_init=xd_0,
            x_c_init=xc_0,
            step_size=step_size,
            time_grid=torch.linspace(0, 1 - 1e-6, num_intermediates),
            return_intermediates=True,
            mask=mask,
            transform_object=transform_object,
            budget=budget,  # Pass budget to solver
            verbose=False,
        )

        print(f"Co-generation time: {time.time() - gentime_start}")
        
        # Scale trajectories back to original scale for inference
        # The model was trained on scaled data, so we need to scale the output back
        if self.trajectory_scale != 1.0:
            # Scale each trajectory in the solution list
            scaled_sol_l = []
            for sol in sol_l:
                contact_tokens, trajectory = sol
                scaled_trajectory = trajectory / self.trajectory_scale
                scaled_sol_l.append((contact_tokens, scaled_trajectory))
            sol_l = scaled_sol_l
        
        return sol_l

class TrajectorySmoothnessLoss(torch.nn.Module):
    """
    Computes a smoothness loss for a batch of trajectories.
    The loss is the sum of squared second differences (discrete acceleration) along the trajectory.
    Args:
        weight: scaling factor for the loss (default: 1.0)
    """
    def __init__(self, weight: float = 1.0):
        super().__init__()
        self.weight = weight

    def forward(self, traj: torch.Tensor) -> torch.Tensor:
        # traj: (B, N, H, 2)
        # Compute second differences along H (horizon)
        d2 = traj[:, :, 2:, :] - 2 * traj[:, :, 1:-1, :] + traj[:, :, :-2, :]
        # Loss is the total laplacian of the trajectory.
        loss = (d2 ** 2).sum(dim=(-1, -2, -3))  # sum over N, H-2, 2
        return self.weight * loss.mean()
