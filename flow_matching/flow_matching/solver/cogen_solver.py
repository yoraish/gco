# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, Optional, Sequence, Tuple, Union
from contextlib import nullcontext
from math import ceil

import torch
from torch import Tensor
from torch.nn import functional as F

from flow_matching.solver.ode_solver import ODESolver
from flow_matching.solver.discrete_solver import MixtureDiscreteEulerSolver
from flow_matching.path import MixtureDiscreteProbPath

from flow_matching.solver.solver import Solver
from flow_matching.utils import gradient, categorical
from flow_matching.utils.model_wrapper import ModelWrapperCoGen
from .utils import get_nearest_times

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False


class CoGenSolver(Solver):
    """A class to solve ordinary differential equations (ODEs) using a specified velocity model.

    This class utilizes a velocity field model to solve ODEs over a given time grid using numerical ode solvers.

    Args:
        velocity_logits_model (Union[ModelWrapper, Callable]): a two headed velocity field model receiving :math:`(x_c, x_d, t)` and returning :math:`u_t(x_c)` and :math:`\logits_t(x_d)`
    """

    def __init__(self,        
                 model: ModelWrapperCoGen,
                 prob_path_discrete: MixtureDiscreteProbPath,
                 vocabulary_size: int,
                 num_steps: int,
                 source_distribution_p: Optional[Tensor] = None,
                 ):
        super().__init__()
        self.model = model
        self.prob_path_discrete = prob_path_discrete
        self.vocabulary_size = vocabulary_size
        self.num_steps = num_steps
        
        if source_distribution_p is not None:
            assert source_distribution_p.shape == torch.Size(
                [vocabulary_size]
            ), f"Source distribution p dimension must match the vocabulary size {vocabulary_size}. Got {source_distribution_p.shape}."
        
        self.source_distribution_p = source_distribution_p

    def sample(
        self,
        x_d_init: Tensor,
        x_c_init: Tensor,
        step_size: Optional[float],
        time_grid: Tensor,  # The times at which to record intermediates.
        return_intermediates: bool = False,
        verbose: bool = False,
        div_free: Union[float, Callable[[float], float]] = 0.0,
        dtype_categorical: torch.dtype = torch.float32,
        **model_extras,
    ) -> Union[Tuple[Tensor, Tensor], Tuple[Sequence[Tuple[Tensor, Tensor]], Tensor]]:
        """
        Co-generation solver. Given initial continuous and discrete states, step the process forward to denoise them.
        Args:
            x_d_init (Tensor): The initial discrete state.
            x_c_init (Tensor): The initial continuous state.
            step_size (Optional[float]): If float then time discretization is uniform with the given step size. 
                        If None then time discretization is set to be time_grid. Must be None for adaptive step solvers (currently not supported).
            time_grid (Tensor): The times at which to record intermediates.
            return_intermediates (bool): Whether to return intermediates.
            verbose (bool): Whether to print progress bars.
            div_free (Union[float, Callable[[float], float]]): The coefficient of the divergence-free term in the probability velocity. Can be either a float or a time dependent function. Defaults to 0.0.
            dtype_categorical (torch.dtype): Precision to use for categorical sampler. Defaults to torch.float32.
            **model_extras: Additional input for the model.

        Returns:
            Union[Tuple[Tensor, Tensor], Tuple[Sequence[Tuple[Tensor, Tensor]], Tensor]]: The sampled sequence of discrete and continuous states.

        Raises:
            ImportError: To run in verbose mode, tqdm must be installed.
        """
        if step_size is None:
            raise NotImplementedError("Step size must not be None for co-generation solver.")
        
        # Make sure that the time grid has at most as many points as the number of steps.
        assert len(time_grid) <= self.num_steps, f"Time grid must have at most {self.num_steps} points (num steps)."
        
        # Check divergence-free requirements
        if not div_free == 0.0:
            assert (
                hasattr(self, 'source_distribution_p') and self.source_distribution_p is not None
            ), "Source distribution p must be specified in order to add a divergence-free term to the probability velocity."
        
        # Initialize the current states
        time_grid = time_grid.to(device=x_d_init.device)
        
        # Set up time discretization
        t_init = time_grid[0].item()
        t_final = time_grid[-1].item()
        assert (t_final - t_init) > step_size, f"Time interval [{t_init}, {t_final}] must be larger than step_size {step_size}."
        
        n_steps = ceil((t_final - t_init) / step_size)
        t_discretization = torch.tensor(
            [t_init + step_size * i for i in range(n_steps)] + [t_final],
            device=x_d_init.device,
        )
        
        if return_intermediates:
            # Get order of intermediate steps
            order = torch.argsort(time_grid)
            # Compute intermediate steps to return via nearest points in t_discretization to time_grid
            time_grid = get_nearest_times(
                time_grid=time_grid, t_discretization=t_discretization
            )
        
        # Initialize current states
        x_d_t = x_d_init.clone()
        x_c_t = x_c_init.clone()
        
        steps_counter = 0
        res = []
        
        if return_intermediates:
            res = [(x_d_init.clone(), x_c_init.clone())]
        
        if verbose:
            if not TQDM_AVAILABLE:
                raise ImportError("tqdm is required for verbose mode. Please install it.")
            ctx = tqdm(total=t_final, desc=f"NFE: {steps_counter}")
        else:
            ctx = nullcontext()
        
        with ctx:
            for i in range(n_steps):

                # Zero out the start position in the first step.
                # x_c_t[:, :, 0, :] *= 0.0

                t = t_discretization[i : i + 1]
                h = t_discretization[i + 1 : i + 2] - t_discretization[i : i + 1]

                # Get model outputs for both discrete and continuous parts
                discrete_logits, continuous_velocity = self.model(
                    x_d=x_d_t, 
                    x_c=x_c_t, 
                    t=t.repeat(x_d_t.shape[0]), 
                    **model_extras
                )
                
                # Handle discrete part (similar to MixtureDiscreteEulerSolver)
                discrete_probs = discrete_logits.to(dtype=dtype_categorical)
                x_d_1 = categorical(discrete_probs)

                # Check if final step for discrete part
                if i == n_steps - 1:
                    x_d_t = x_d_1
                else:
                    # Compute discrete velocity using the probability path
                    scheduler_output = self.prob_path_discrete.scheduler(t=t)
                    # The scheduler provides k_t and d_k_t, which determine the discrete velocity.
                    k_t = scheduler_output.alpha_t
                    d_k_t = scheduler_output.d_alpha_t

                    delta_1 = F.one_hot(x_d_1, num_classes=self.vocabulary_size).to(
                        k_t.dtype
                    )
                    u_d = d_k_t / (1 - k_t) * delta_1

                    # Add divergence-free part
                    div_free_t = div_free(t) if callable(div_free) else div_free

                    if div_free_t > 0:
                        p_0 = self.source_distribution_p[(None,) * x_d_t.dim()]
                        u_d = u_d + div_free_t * d_k_t / (k_t * (1 - k_t)) * (
                            (1 - k_t) * p_0 + k_t * delta_1
                        )

                    # Set u_t(x_t|x_t,x_1) = 0
                    delta_t = F.one_hot(x_d_t, num_classes=self.vocabulary_size)
                    u_d = torch.where(
                        delta_t.to(dtype=torch.bool), torch.zeros_like(u_d), u_d
                    )

                    # Sample discrete jumps
                    intensity = u_d.sum(dim=-1)  # Assuming u_t(xt|xt,x1) := 0
                    mask_jump = torch.rand(
                        size=x_d_t.shape, device=x_d_t.device
                    ) < 1 - torch.exp(-h * intensity)

                    if mask_jump.sum() > 0:
                        x_d_t[mask_jump] = categorical(
                            u_d[mask_jump].to(dtype=dtype_categorical)
                        )
                
                # Handle continuous part (Euler step)
                x_c_t = x_c_t + h * continuous_velocity
                
                steps_counter += 1
                t = t + h
                
                if return_intermediates and (t in time_grid):
                    res.append((x_d_t.clone(), x_c_t.clone()))
                
                if verbose:
                    ctx.n = t.item()
                    ctx.refresh()
                    ctx.set_description(f"NFE: {steps_counter}")
        
        if return_intermediates:
            if step_size is None:
                return res, time_grid
            else:
                return [res[i] for i in order], time_grid
        else:
            return x_d_t, x_c_t
        
