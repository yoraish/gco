

# Set matplotlib to use non-interactive backend to avoid threading issues
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# Add signal handling for graceful shutdown
import signal
import sys

def signal_handler(sig, frame):
    print('\nReceived interrupt signal. Cleaning up...')
    try:
        wandb.finish()
    except:
        pass
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

from matplotlib import rc
rc("animation", html="jshtml")     # or "html5" if you prefer an <video> tag

# Imports for flow-matching.
import time
import torch
import random
from dataclasses import dataclass
from torch import nn, Tensor
from torch.utils.data import DataLoader
import math
import datetime
# flow_matching
from flow_matching.path import MixtureDiscreteProbPath
from flow_matching.path.scheduler import PolynomialConvexScheduler
from flow_matching.solver import MixtureDiscreteEulerSolver
from flow_matching.utils import ModelWrapper, ModelWrapperCoGen
from flow_matching.loss import MixturePathGeneralizedKL
from flow_matching.path.scheduler import CondOTScheduler
from flow_matching.path import AffineProbPath
from flow_matching.solver import Solver, ODESolver, CoGenSolver

# visualization
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt

# Imports for GCo.
from gco.config import Config as cfg
from gco.datasets import ContactTrajectoryDataset
from gco.utils.transform_utils import Transform2
from gco.models.contact_push_model import ContactModel, TrajectoryModel, ContactTrajectoryModel
from gco.utils.model_utils import *

# Device is set in cfg.device.
print("Device: ", cfg.device)

# Remove the artificial scaling - let's work with the real data scale
# TRAJECTORY_SCALE = cfg.trajectory_scale

# Seed everything.
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

# ================================
is_training = True
# ================================

# Load a dataset.
dataset_path = cfg.data_dir_contacts_trajs_r_c_flex
dataset = ContactTrajectoryDataset.load_dataset(dataset_path)
len(dataset)

from gco.utils.data_vis_utils import visualize_batch
# Create a dataloader and load a batch.
dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)
batch = next(iter(dataloader))
print(batch.keys())
batch_vis = {k: v[:10] for k, v in batch.items()}
try:
    fig, axs = visualize_batch(batch_vis, save_path=None)
    plt.close(fig)  # Close the figure to free memory
except Exception as viz_error:
    print(f"Initial visualization error (non-critical): {viz_error}")
# plt.show()

# Create a co-generation model.
vocab_size = cfg.mask_token + 1
cogen_model = ContactTrajectoryModel(vocab_size=vocab_size).to(cfg.device)

# Initialize model weights properly for better training
def init_weights(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)

cogen_model.apply(init_weights)
print(f"Model initialized with Xavier uniform initialization")

# Test the model internal dimensions.
dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)
batch = next(iter(dataloader))
print(batch.keys())

# Try to run a batch through the model. The output is v_d, v_c, i.e., logits on the target contact points and the flow-matching velocity for the batch.
x_c = batch["trajectory"].to(cfg.device)
contact_points = batch["start_contacts"]
mask = batch["mask"]
transform_object = batch["transform_object"]
# Check if budget is available in the dataset
budget = batch.get("robot_budget", torch.ones(contact_points.shape[0], dtype=torch.long) * cfg.N).to(cfg.device)
B = contact_points.shape[0]

# Contact points to tokens.
x_d = pixels_to_tokens(contact_points)

# Time [0, 1].
t = torch.randn(B, device=cfg.device, dtype=torch.float32)  # (B,)

# Print number of parameters.
print(f"Number of parameters: {sum(p.numel() for p in cogen_model.parameters()):,}")

# Print shapes of inputs.
print(f"x_d shape: {x_d.shape}", x_d.dtype)
print(f"x_c shape: {x_c.shape}", x_c.dtype)
print(f"t shape: {t.shape}", t.dtype)
print(f"mask shape: {mask.shape}", mask.dtype)
print(f"transform_object shape: {transform_object.shape}", transform_object.dtype)
print(f"budget shape: {budget.shape}", budget.dtype)

# Run the model.
v_d, v_c = cogen_model(x_d, x_c, t, mask, transform_object, budget)

cogen_model = ContactTrajectoryModel(vocab_size=vocab_size).to(cfg.device)

# Enhanced loss functions
class BudgetConstraintLoss(torch.nn.Module):
    """
    Loss function to ensure the number of robots used is less than or equal to the budget.
    """
    def __init__(self, weight: float = 1.0):
        super().__init__()
        self.weight = weight
    
    def forward(self, contact_tokens_logits: torch.Tensor, budget: torch.Tensor) -> torch.Tensor:
        """
        :param contact_tokens_logits: (B, N*2, V) contact point token logits
        :param budget: (B,) available robot budget for each batch item
        :return: Budget constraint loss
        """
        B = contact_tokens_logits.shape[0]
        N = cfg.N
        
        # Convert logits to tokens by taking argmax
        contact_tokens = contact_tokens_logits.argmax(dim=-1)  # (B, N*2)
        
        # Reshape to (B, N, 2) for easier processing
        contact_tokens_reshaped = contact_tokens.reshape(B, N, 2)
        
        # Count non-mask tokens for each robot (assuming mask token is vocab_size - 1)
        mask_token = cfg.mask_token
        non_mask_mask = (contact_tokens_reshaped != mask_token).any(dim=-1)  # (B, N)
        robots_used = non_mask_mask.sum(dim=1)  # (B,)
        
        # Calculate excess robots (should be non-negative)
        excess_robots = torch.relu(robots_used - budget)
        
        return self.weight * excess_robots.float().mean()

from tqdm import tqdm
import matplotlib.pyplot as plt
from IPython.display import clear_output, display
import datetime
import time
import numpy as np
import wandb
import torch

# Initialize experiment name
experiment_name = "cogen_model_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
recent_chkpt_path = cfg.chkpt_dir_cogen / f"cogen_model_{experiment_name}_epoch_{0}.pth"
# Loss weights
budget_constraint_weight = 1.0
# Initialize wandb
wandb.init(
    project="cogen-training",
    name=experiment_name,
    config={
        "learning_rate": cfg.learning_rate,
        "epochs": cfg.num_epochs,
        "optimizer": "AdamW",
        "weight_decay": 1e-6,
        "epsilon": 0.01,
        "vocab_size": cfg.V,
        "source_distribution_discrete": cfg.source_distribution_discrete,
        "budget_constraint_weight": budget_constraint_weight
    }
)

# Learning setup
lr = cfg.learning_rate
epochs = cfg.num_epochs
save_every_n_epochs = 10
epsilon = 0.001

# Probabilistic path samplers
prob_path_continuous = AffineProbPath(scheduler=CondOTScheduler())
# Alternative: Use a scheduler that produces smaller velocities
# prob_path_continuous = AffineProbPath(scheduler=PolynomialConvexScheduler(n=2.0))
scheduler = PolynomialConvexScheduler(n=2.0)
prob_path_discrete = MixtureDiscreteProbPath(scheduler=scheduler)
loss_fn_discrete = MixturePathGeneralizedKL(path=prob_path_discrete)

# Enhanced loss functions
budget_loss_fn = BudgetConstraintLoss(weight=budget_constraint_weight)

# Optimizer
optim = torch.optim.AdamW(
    cogen_model.parameters(), 
    lr=lr, 
    betas=(0.9, 0.999),
    eps=1e-8
)

# # Simple ADAM.
# optim = torch.optim.Adam(
#     cogen_model.parameters(), 
#     lr=lr, 
#     betas=(0.9, 0.999),
#     eps=1e-8
# )

# Vocab setup
vocab_size = cfg.V
mask_token = cfg.mask_token

# Training loop setup
losses = []
pbar = tqdm(range(epochs), desc="Training")

for epoch in pbar:
    for batch_idx, batch in enumerate(dataloader):
        optim.zero_grad()

        mask = batch["mask"]
        contact_points = batch["start_contacts"]
        contact_points_tokens = pixels_to_tokens(contact_points)
        transform_object = batch["transform_object"]
        # Get budget from batch, default to cfg.N if not available
        budget = batch.get("robot_budget", torch.ones(contact_points.shape[0], dtype=torch.long) * cfg.N).to(cfg.device)
        xc_1 = batch["trajectory"].to(cfg.device) # / 4.0  #####
        xc_0 = torch.randn_like(xc_1).to(cfg.device)

        # Use the real data scale - no artificial normalization
        # xc_1 = xc_1 * TRAJECTORY_SCALE
        # xc_0 = xc_0 * TRAJECTORY_SCALE

        xd_1 = pixels_to_tokens(contact_points)
        if cfg.source_distribution_discrete == "mask":
            xd_0 = torch.zeros_like(xd_1).to(cfg.device) + mask_token
        elif cfg.source_distribution_discrete == "uniform":
            xd_0 = torch.randint_like(xd_1, high=vocab_size)
        else:
            raise ValueError(f"Invalid source distribution: {cfg.source_distribution_discrete}")

        t = torch.rand(xd_1.shape[0]).to(cfg.device) * (1 - epsilon)

        path_sample_continuous = prob_path_continuous.sample(t=t, x_0=xc_0, x_1=xc_1)
        path_sample_discrete = prob_path_discrete.sample(t=t, x_0=xd_0, x_1=xd_1)

        vd_pred, vc_pred = cogen_model(
            path_sample_discrete.x_t, 
            path_sample_continuous.x_t, 
            path_sample_continuous.t, 
            mask, 
            transform_object,
            budget
        )

        # Use the target velocity directly - this is the correct flow matching loss
        loss_continuous = torch.pow(vc_pred - path_sample_continuous.dx_t, 2).mean() # (B, N, H, 2)

        loss_discrete = loss_fn_discrete(
            logits=vd_pred, 
            x_1=xd_1, 
            x_t=path_sample_discrete.x_t, 
            t=path_sample_discrete.t
        )

        # Enhanced loss components
        # Budget constraint loss
        loss_budget = budget_loss_fn(vd_pred, budget)

        # Combined loss
        loss = loss_continuous + loss_discrete + loss_budget
        
        loss.backward()
        optim.step()

        # Logging
        wandb.log({
            "loss_total": loss.item(),
            "loss_continuous": loss_continuous.item(),
            "loss_discrete": loss_discrete.item(),
            "loss_budget": loss_budget.item(),
            "epoch": epoch
        })

        pbar.set_postfix({
            'Epoch': f'{epoch+1}/{epochs}', 
            'l_d': f'{loss_discrete.item():.6f}',
            'l_c': f'{loss_continuous.item():.6f}',
            'l_b': f'{loss_budget.item():.6f}',
            'loss': f'{loss.item():.6f} | recent avg: {np.mean(losses[-100:]) if len(losses) > 0 else 0:.6f}'
        })
        losses.append(loss.item())

    if (epoch % save_every_n_epochs == 0 and save_every_n_epochs != -1) or epoch == epochs - 1:
        if not cfg.chkpt_dir_cogen.exists():
            cfg.chkpt_dir_cogen.mkdir(parents=True, exist_ok=False)
        recent_chkpt_path = cfg.chkpt_dir_cogen / f"cogen_model_{experiment_name}_epoch_{epoch}.pth"
        torch.save(cogen_model.state_dict(), recent_chkpt_path)
        print(f"Saved model to {recent_chkpt_path}")
        
        # Log model to wandb
        artifact = wandb.Artifact(f"cogen-model-{epoch}", type="model")
        artifact.add_file(str(recent_chkpt_path))
        wandb.log_artifact(artifact)

        # ================================
        # Log generation visualization.
        # ================================
        try:
            cogen_model.eval()
            with torch.no_grad():
                # Fetch a random batch each time for diverse visualizations
                sample_batch = next(iter(DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)))
                mask_sample = sample_batch["mask"].to(cfg.device)
                transform_sample = sample_batch["transform_object"].to(cfg.device)
                # Get budget from sample batch, default to cfg.N if not available
                budget_sample = sample_batch.get("robot_budget", torch.ones(mask_sample.shape[0], dtype=torch.long) * cfg.N).to(cfg.device)

                # Visualize at most 4 samples to keep the figure readable
                num_vis = min(4, mask_sample.shape[0])

                # Run the co-generation model with budget constraints
                sol_l = cogen_model.generate(
                    mask_sample[:num_vis],
                    transform_sample[:num_vis],
                    budget_sample[:num_vis],
                    seed=random.randint(0, 1_000_000),
                    smoothness_weight=cfg.smoothness_weight,
                )

                # Extract the final generation outputs
                pred_contact_tokens = sol_l[-1][0]              # (B, N*2)
                pred_trajectory_pixels = sol_l[-1][1]

                # Convert contact tokens to pixel coordinates
                pred_contacts_pixels = tokens_to_pixels(pred_contact_tokens)

                # Build a lightweight ground-truth batch for visualization
                batch_vis = {k: v[:num_vis].cpu() for k, v in sample_batch.items()}

                # Create and log the figure with error handling
                try:
                    fig, _ = visualize_batch(batch_vis, pred_contacts_pixels.cpu(), pred_trajectory_pixels.cpu())
                    wandb.log({"generation_example": wandb.Image(fig)})
                    plt.close(fig)
                except Exception as viz_error:
                    print(f"Visualization error (non-critical): {viz_error}")
                    # Continue training even if visualization fails
        except Exception as gen_error:
            print(f"Generation error (non-critical): {gen_error}")
            # Continue training even if generation fails
        finally:
            cogen_model.train()

wandb.finish()

