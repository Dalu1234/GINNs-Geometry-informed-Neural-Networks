"""
Conditional latent prior p(z_base | c). Maps conditioning c to a diagonal Gaussian
over the shape latent z_base. Used to (1) regularize z_base toward condition-dependent
regions and (2) optionally sample z_base from the prior during training.
"""

import math
import torch
from torch import nn


class ConditionalPrior(nn.Module):
    """
    MLP that maps c (B, c_dim) to parameters of a diagonal Gaussian over z_base:
    mu (B, z_base_dim), log_sigma (B, z_base_dim) with sigma = min_sigma + softplus(log_sigma).
    """

    def __init__(
        self,
        c_dim: int,
        z_base_dim: int = 2,
        hidden_dims=(32, 32),
        min_sigma: float = 1e-3,
    ):
        super().__init__()
        self.c_dim = c_dim
        self.z_base_dim = z_base_dim
        self.min_sigma = min_sigma

        layers = []
        prev = c_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU(inplace=True))
            prev = h
        self.mlp = nn.Sequential(*layers)
        self.mu_head = nn.Linear(prev, z_base_dim)
        self.log_sigma_head = nn.Linear(prev, z_base_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.mu_head.weight, gain=0.1)
        nn.init.zeros_(self.mu_head.bias)
        nn.init.xavier_uniform_(self.log_sigma_head.weight, gain=0.1)
        # Start with small sigma (negative log_sigma after softplus)
        nn.init.constant_(self.log_sigma_head.bias, -2.0)

    def get_params(self, c: torch.Tensor):
        """Return mu (B, z_base_dim), sigma (B, z_base_dim)."""
        h = self.mlp(c)
        mu = self.mu_head(h)
        log_sigma = self.log_sigma_head(h)
        sigma = self.min_sigma + torch.nn.functional.softplus(log_sigma)
        return mu, sigma

    def sample(self, c: torch.Tensor):
        """Sample z_base ~ N(mu(c), diag(sigma(c)^2)). Returns (B, z_base_dim)."""
        mu, sigma = self.get_params(c)
        eps = torch.randn_like(mu, device=mu.device, dtype=mu.dtype)
        return mu + sigma * eps

    def log_prob(self, z_base: torch.Tensor, c: torch.Tensor):
        """Log probability of z_base under N(mu(c), diag(sigma(c)^2)). Returns (B,) or scalar."""
        mu, sigma = self.get_params(c)
        var = sigma ** 2
        log_var = torch.log(var.clamp(min=1e-8))
        diff = z_base - mu
        # Gaussian log prob: -0.5 * (log(2*pi) + log_var + (x-mu)^2/var)
        log_p = -0.5 * (math.log(2 * math.pi) + log_var + (diff ** 2) / var)
        return log_p.sum(dim=-1)
