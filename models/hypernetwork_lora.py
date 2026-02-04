"""
Minimal hypernetwork that outputs LoRA factors (U, V) for one or more Linear layers.
Conditioned on continuous c = (Nx_norm, Ny_norm, height_norm) for generalization to unseen bricks.
Supports single-layer (original) or multi-layer LoRA with larger/deeper MLP.
"""

import torch
from torch import nn


class HypernetworkLoRA(nn.Module):
    """
    Small MLP that maps conditioning c (B, c_dim) to LoRA factors U (B, out_f, r), V (B, r, in_f).
    For the output layer: out_f=1, in_f=H (last hidden size). So U (B, 1, r), V (B, r, H).
    Init scale keeps U @ V small so the base INR dominates (stable SDF / eikonal).
    """

    def __init__(
        self,
        c_dim: int,
        in_features: int,
        out_features: int,
        rank: int,
        hidden_dims=(32, 32),
        init_scale: float = 0.01,
    ):
        super().__init__()
        self.c_dim = c_dim
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.init_scale = init_scale

        # c -> embed
        layers = []
        prev = c_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU(inplace=True))
            prev = h
        self.mlp = nn.Sequential(*layers)

        # embed -> flat vector that we split into U and V
        # U: (out_f, r), V: (r, in_f) -> flat length out_f*r + r*in_f
        self.flat_dim = out_features * rank + rank * in_features
        self.head = nn.Linear(prev, self.flat_dim)

        self._init_weights()

    def _init_weights(self):
        # Small init so U @ V is small at start
        nn.init.normal_(self.head.weight, mean=0.0, std=self.init_scale)
        nn.init.zeros_(self.head.bias)
        with torch.no_grad():
            self.head.weight.data.mul_(1.0 / (self.rank ** 0.5))
            self.head.bias.data.zero_()

    def forward(self, c: torch.Tensor):
        """
        c: (B, c_dim), e.g. (Nx_norm, Ny_norm, height_norm) in [0, 1].
        Returns:
            U: (B, out_features, rank)
            V: (B, rank, in_features)
        """
        B = c.shape[0]
        embed = self.mlp(c)
        flat = self.head(embed)
        u_flat = flat[:, : self.out_features * self.rank]
        v_flat = flat[:, self.out_features * self.rank :]
        U = u_flat.view(B, self.out_features, self.rank)
        V = v_flat.view(B, self.rank, self.in_features)
        return U, V


class MultiLayerHypernetworkLoRA(nn.Module):
    """
    MLP that maps c (B, c_dim) to LoRA factors for multiple layers.
    layer_specs: list of (out_features, in_features) for each layer, e.g. [(512, 256), (1, 256)]
    for penult Gabor Linear(256, 512) and final Linear(256, 1).
    Returns list of (U, V) per layer; U (B, out_f, r), V (B, r, in_f).
    """

    def __init__(
        self,
        c_dim: int,
        layer_specs: list,
        rank: int,
        hidden_dims=(64, 64, 32),
        init_scale: float = 0.01,
    ):
        super().__init__()
        self.c_dim = c_dim
        self.layer_specs = list(layer_specs)
        self.rank = rank
        self.init_scale = init_scale

        # c -> embed (larger/deeper)
        layers = []
        prev = c_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU(inplace=True))
            prev = h
        self.mlp = nn.Sequential(*layers)

        # Total flat dim: for each (out_f, in_f) we need out_f*r + r*in_f
        self.flat_dims = []
        for out_f, in_f in self.layer_specs:
            self.flat_dims.append(out_f * rank + rank * in_f)
        self.flat_dim = sum(self.flat_dims)
        self.head = nn.Linear(prev, self.flat_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.head.weight, mean=0.0, std=self.init_scale)
        nn.init.zeros_(self.head.bias)
        with torch.no_grad():
            self.head.weight.data.mul_(1.0 / (self.rank ** 0.5))
            self.head.bias.data.zero_()

    def forward(self, c: torch.Tensor):
        """
        Returns list of (U, V), each U (B, out_f, r), V (B, r, in_f).
        """
        B = c.shape[0]
        embed = self.mlp(c)
        flat = self.head(embed)
        results = []
        offset = 0
        for (out_f, in_f), fd in zip(self.layer_specs, self.flat_dims):
            chunk = flat[:, offset : offset + fd]
            offset += fd
            u_len = out_f * self.rank
            u_flat = chunk[:, :u_len]
            v_flat = chunk[:, u_len:]
            U = u_flat.view(B, out_f, self.rank)
            V = v_flat.view(B, self.rank, in_f)
            results.append((U, V))
        return results
