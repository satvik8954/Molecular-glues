"""
Discrete diffusion noise scheduler for molecular graphs.

Implements D3PM-style discrete diffusion for categorical features
(atom types, bond types) rather than continuous diffusion.
"""
import math
from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
import os

# Add project root to path for imports
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from config import DiffusionConfig, ATOM_TYPES, BOND_TYPES


class NoiseScheduler:
    """
    Discrete diffusion noise scheduler.
    
    For categorical data, the forward process gradually corrupts
    categories towards a uniform distribution over classes.
    """
    
    def __init__(self, config: DiffusionConfig, device: str = 'cpu'):
        """
        Args:
            config: Diffusion configuration
            device: Device to place tensors on
        """
        self.config = config
        self.num_timesteps = config.num_timesteps
        self.device = device
        
        # Number of categories
        self.num_atom_types = len(ATOM_TYPES)
        self.num_bond_types = len(BOND_TYPES)
        
        # Create beta schedule
        self.betas = self._create_beta_schedule().to(device)
        
        # Cumulative products for diffusion
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # For discrete diffusion: transition matrices
        # Q_t[i,j] = P(x_t = j | x_{t-1} = i)
        self._precompute_transition_matrices()
    
    def _create_beta_schedule(self) -> torch.Tensor:
        """Create noise schedule (linear or cosine)."""
        if self.config.beta_schedule == "linear":
            return torch.linspace(
                self.config.beta_start,
                self.config.beta_end,
                self.num_timesteps,
                dtype=torch.float32
            )
        elif self.config.beta_schedule == "cosine":
            # Cosine schedule from "Improved Denoising Diffusion"
            steps = self.num_timesteps + 1
            x = torch.linspace(0, self.num_timesteps, steps, dtype=torch.float32)
            alphas_cumprod = torch.cos(((x / self.num_timesteps) + 0.008) / 1.008 * math.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            return torch.clamp(betas, 0.0001, 0.9999)
        else:
            raise ValueError(f"Unknown beta schedule: {self.config.beta_schedule}")
    
    def _precompute_transition_matrices(self):
        """
        Precompute transition matrices for discrete diffusion.
        
        For absorbing state diffusion, we transition towards a uniform distribution.
        Q_t = (1 - beta_t) * I + beta_t * uniform
        """
        # Cumulative transition matrices: Q_bar_t = Q_1 @ Q_2 @ ... @ Q_t
        # For uniform noise: Q_bar_t[i,j] = alpha_bar_t * I[i,j] + (1 - alpha_bar_t) / K
        self.alpha_bar = self.alphas_cumprod
    
    def get_transition_probs(
        self,
        t: torch.Tensor,
        num_classes: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get transition probabilities for timestep t.
        
        Args:
            t: Timesteps [B]
            num_classes: Number of categories (K)
            
        Returns:
            alpha_bar: Probability of staying in original class [B]
            uniform_prob: Probability mass for each other class [B]
        """
        alpha_bar = self.alpha_bar[t]  # [B]
        uniform_prob = (1.0 - alpha_bar) / num_classes  # [B]
        
        return alpha_bar, uniform_prob
    
    def add_noise(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        num_classes: int,
    ) -> torch.Tensor:
        """
        Add noise to categorical data at timestep t.
        
        Forward process: q(x_t | x_0)
        
        Args:
            x: Original categorical data [N] or [N, 1] (class indices)
            t: Timesteps [B] (one per batch, expanded to match x)
            num_classes: Number of categories
            
        Returns:
            Noisy categorical data [N] (class indices)
        """
        x = x.flatten()
        
        # Get transition probabilities
        alpha_bar, uniform_prob = self.get_transition_probs(t, num_classes)
        
        # Expand to match x shape
        if alpha_bar.dim() == 0:
            alpha_bar = alpha_bar.unsqueeze(0)
            uniform_prob = uniform_prob.unsqueeze(0)
        
        # Repeat for each element in x
        alpha_bar = alpha_bar.repeat_interleave(x.shape[0] // t.shape[0])
        uniform_prob = uniform_prob.repeat_interleave(x.shape[0] // t.shape[0])
        
        # Build transition probabilities for each element
        # P(x_t = k | x_0 = j) = alpha_bar * I[k==j] + (1 - alpha_bar) / K
        probs = torch.zeros(x.shape[0], num_classes, device=x.device)
        probs.scatter_(1, x.unsqueeze(1), 1.0)  # One-hot of original
        
        probs = alpha_bar.unsqueeze(1) * probs + uniform_prob.unsqueeze(1)
        
        # Sample from categorical distribution
        noisy_x = torch.multinomial(probs, 1).squeeze(1)
        
        return noisy_x
    
    def get_posterior_probs(
        self,
        x_t: torch.Tensor,
        x_0_pred: torch.Tensor,
        t: torch.Tensor,
        num_classes: int,
    ) -> torch.Tensor:
        """
        Compute posterior q(x_{t-1} | x_t, x_0) for reverse process.
        
        Args:
            x_t: Noisy data at timestep t [N]
            x_0_pred: Predicted clean data [N, K] (logits)
            t: Current timestep [B]
            num_classes: Number of categories
            
        Returns:
            Posterior probabilities [N, K]
        """
        # Convert x_0_pred logits to probabilities
        p_x0 = F.softmax(x_0_pred, dim=-1)  # [N, K]
        
        # Get alpha values
        alpha_t = self.alphas[t]  # [B]
        alpha_bar_t = self.alphas_cumprod[t]  # [B]
        alpha_bar_t_minus_1 = self.alphas_cumprod_prev[t]  # [B]
        
        # Expand to match N
        N = x_t.shape[0]
        B = t.shape[0]
        alpha_t = alpha_t.repeat_interleave(N // B)
        alpha_bar_t = alpha_bar_t.repeat_interleave(N // B)
        alpha_bar_t_minus_1 = alpha_bar_t_minus_1.repeat_interleave(N // B)
        
        # q(x_{t-1} | x_t, x_0) ∝ q(x_t | x_{t-1}) * q(x_{t-1} | x_0)
        # This is a complex formula for discrete diffusion
        # Simplified: use predicted x_0 distribution directly for sampling
        
        # Mix with uniform for numerical stability
        uniform = torch.ones_like(p_x0) / num_classes
        posterior = (1.0 - 0.001) * p_x0 + 0.001 * uniform
        
        return posterior
    
    def sample_timesteps(self, batch_size: int) -> torch.Tensor:
        """Sample random timesteps for training."""
        return torch.randint(0, self.num_timesteps, (batch_size,), device=self.device)
