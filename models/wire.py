#!/usr/bin/env python

import os
import sys
from typing import List
import tqdm
import pdb

import numpy as np
import torch
from torch import nn

import torch.nn.functional as F

class RealGaborLayerLegacy(nn.Module):
    '''
        Implicit representation with Gabor nonlinearity
        
        Inputs;
            in_features: Input features
            out_features; Output features
            bias: if True, enable bias for the linear operation
            is_first: Legacy SIREN parameter
            omega_0: Legacy SIREN parameter
            omega: Frequency of Gabor sinusoid term
            scale: Scaling of Gabor Gaussian term
    '''
    
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega0=10.0, sigma0=10.0,
                 trainable=False):
        super().__init__()
        self.omega_0 = omega0
        self.scale_0 = sigma0
        self.is_first = is_first
        
        self.in_features = in_features
        
        self.freqs = nn.Linear(in_features, out_features, bias=bias)
        self.scale = nn.Linear(in_features, out_features, bias=bias)
        
    def forward(self, input):
        omega = self.omega_0 * self.freqs(input)
        scale = self.scale(input) * self.scale_0
        return torch.cos(omega)*torch.exp(-(scale**2))
        

class RealGaborLayer(nn.Module):
    '''
        Implicit representation with Gabor nonlinearity
        
        Inputs;
            in_features: Input features
            out_features; Output features
            bias: if True, enable bias for the linear operation
            is_first: Legacy SIREN parameter
            omega_0: Legacy SIREN parameter
            omega: Frequency of Gabor sinusoid term
            scale: Scaling of Gabor Gaussian term
    '''
    
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega0=10.0, sigma0=10.0,
                 trainable=False):
        super().__init__()
        self.omega_0 = omega0
        self.scale_0 = sigma0
        self.is_first = is_first
        
        self.in_features = in_features
        
        # self.freqs = nn.Linear(in_features, out_features, bias=bias)
        # self.scale = nn.Linear(in_features, out_features, bias=bias)
        self.freq_scale = nn.Linear(in_features, 2*out_features, bias=bias)
        
    def forward(self, input):
        # omega = self.omega_0 * self.freqs(input)
        # scale = self.scale(input) * self.scale_0
        # return torch.cos(omega)*torch.exp(-(scale**2))
        omega, scale = torch.chunk(self.freq_scale(input), 2, dim=-1)
        return torch.cos(self.omega_0 * omega)*torch.exp(-(self.scale_0 * scale**2))

class ComplexGaborLayer(nn.Module):
    '''
        Implicit representation with complex Gabor nonlinearity
        
        Inputs;
            in_features: Input features
            out_features; Output features
            bias: if True, enable bias for the linear operation
            is_first: Legacy SIREN parameter
            omega_0: Legacy SIREN parameter
            omega0: Frequency of Gabor sinusoid term
            sigma0: Scaling of Gabor Gaussian term
            trainable: If True, omega and sigma are trainable parameters
    '''
    
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega0=10.0, sigma0=40.0,
                 trainable=False):
        super().__init__()
        self.omega_0 = omega0
        self.scale_0 = sigma0
        self.is_first = is_first
        
        self.in_features = in_features
        
        if self.is_first:
            dtype = torch.float
        else:
            dtype = torch.cfloat
            
        # Set trainable parameters if they are to be simultaneously optimized
        self.omega_0 = nn.Parameter(self.omega_0*torch.ones(1), trainable)
        self.scale_0 = nn.Parameter(self.scale_0*torch.ones(1), trainable)
        
        self.linear = nn.Linear(in_features,
                                out_features,
                                bias=bias,
                                dtype=dtype)
    
    def forward(self, input):
        lin = self.linear(input)
        omega = self.omega_0 * lin
        scale = self.scale_0 * lin
        
        return torch.exp(1j*omega - scale.abs().square())

class WIRE_original(nn.Module):
    def __init__(self, in_features, hidden_features, 
                 hidden_layers, 
                 out_features, outermost_linear=True,
                 first_omega_0=30, hidden_omega_0=30., scale=10.0,
                 pos_encode=False, sidelength=512, fn_samples=None,
                 use_nyquist=True):
        super().__init__()
        
        # All results in the paper were with the default complex 'gabor' nonlinearity
        self.nonlin = ComplexGaborLayer
        
        # Since complex numbers are two real numbers, reduce the number of 
        # hidden parameters by 2
        hidden_features = int(hidden_features/np.sqrt(2))
        dtype = torch.cfloat
        self.complex = True
        self.wavelet = 'gabor'
        
        # Legacy parameter
        self.pos_encode = False
            
        self.net = []
        self.net.append(self.nonlin(in_features,
                                    hidden_features, 
                                    omega0=first_omega_0,
                                    sigma0=scale,
                                    is_first=True,
                                    trainable=False))

        for i in range(hidden_layers):
            self.net.append(self.nonlin(hidden_features,
                                        hidden_features, 
                                        omega0=hidden_omega_0,
                                        sigma0=scale))

        final_linear = nn.Linear(hidden_features,
                                 out_features,
                                 dtype=dtype)            
        self.net.append(final_linear)
        
        self.net = nn.Sequential(*self.net)
    
    def forward(self, coords):
        output = self.net(coords)
        
        if self.wavelet == 'gabor':
            return output.real
         
        return output


class FiLMBlock(nn.Module):
    """Feature-wise Linear Modulation: gamma(z) * h + beta(z). Init so gamma≈1, beta≈0 (identity at start)."""
    def __init__(self, z_dim: int, out_features: int, film_hidden: int = 64):
        super().__init__()
        self.gamma_net = nn.Sequential(
            nn.Linear(z_dim, film_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(film_hidden, out_features),
        )
        self.beta_net = nn.Sequential(
            nn.Linear(z_dim, film_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(film_hidden, out_features),
        )
        nn.init.zeros_(self.gamma_net[-1].weight)
        nn.init.zeros_(self.gamma_net[-1].bias)
        nn.init.zeros_(self.beta_net[-1].weight)
        nn.init.zeros_(self.beta_net[-1].bias)

    def forward(self, z):
        # (1 + gamma) * h + beta so at init (gamma=0, beta=0) we get identity
        gamma = self.gamma_net(z)
        beta = self.beta_net(z)
        return gamma, beta


class ConditionalWIRE(nn.Module):
    '''
    Since complex numbers don't work with jacrev, we need to use the real number version of the WIRE.
    Optional tiled coordinates: append (x_tile, y_tile) modulo stud spacing so the network
    sees position within each tile and can learn one cylindrical feature per tile.
    Optional dist_to_edge_x: append signed distance to nearest x-boundary (positive inside brick)
    so the network can ignore phantom studs outside the brick (e.g. dist_to_edge_x < 0).
    Optional hypernetwork: LoRA-style weight deltas on the last linear, conditioned on c = (Nx_norm, Ny_norm, height_norm)
    to reduce gradient interference across brick types and support unseen bricks.
    '''
    def __init__(self, 
                 layers: List[int],
                 return_density,
                 first_omega_0=30, 
                 hidden_omega_0=30., 
                 scale=10.0, 
                 use_legacy_gabor=False,
                 use_tiled_coords=False,
                 stud_spacing=1.0,
                 stud_spacing_y=None,
                 use_dist_to_edge=False,
                 n_studs_values=None,
                 n_norm_col=2,
                 use_smooth_n_encoding=False,
                 use_hypernet=False,
                 lora_rank=4,
                 c_dim=3,
                 hypernet_hidden=(32, 32),
                 lora_init_scale=0.01,
                 dropout_p=0.0,
                 use_film=False,
                 film_after_layers=(0, 2),
                 film_hidden=64,
                 nz=None,
                 **kwargs):
        super().__init__()
        self.layers = layers
        self.return_density = return_density
        self.dropout_p = float(dropout_p) if dropout_p else 0.0
        self.first_omega_0 = first_omega_0
        self.hidden_omega_0 = hidden_omega_0
        self.scale = scale
        self.use_legacy_gabor = use_legacy_gabor
        self.use_tiled_coords = use_tiled_coords
        self.stud_spacing_x = stud_spacing
        self.stud_spacing_y = stud_spacing if stud_spacing_y is None else stud_spacing_y
        self.use_dist_to_edge = use_dist_to_edge and (n_studs_values is not None and len(n_studs_values) > 0)
        self.n_studs_values = list(n_studs_values) if n_studs_values is not None else []
        self.n_norm_col = n_norm_col
        self.use_smooth_n_encoding = use_smooth_n_encoding and len(self.n_studs_values) > 0
        self.n_norm_col_end = (n_norm_col + len(self.n_studs_values)) if self.use_smooth_n_encoding else None
        self.use_hypernet = use_hypernet
        self.c_dim = c_dim
        self.use_film = bool(use_film)
        self._film_after_indices = tuple(film_after_layers) if use_film else ()

        # All results in the paper were with the default complex 'gabor' nonlinearity
        # NOTE: I used partial(RelGaborLayer, omega0=first_omega_0, sigma0=scale) to set the default values, but there was some weird behavior. 
        if use_legacy_gabor:
            self.nonlin = RealGaborLayerLegacy
        else:
            self.nonlin = RealGaborLayer
        
        self.net = []
        self.net.append(self.nonlin(layers[0],
                                    # int(layers[1]/np.sqrt(2)), 
                                    layers[1], 
                                    omega0=first_omega_0,
                                    sigma0=scale,
                                    is_first=True,
                                    trainable=False))
        if self.dropout_p > 0:
            self.net.append(nn.Dropout(self.dropout_p))

        for i in range(1, len(layers) - 2):
            self.net.append(self.nonlin(layers[i],
                                        layers[i+1], 
                                        omega0=hidden_omega_0,
                                        sigma0=scale))
            if self.dropout_p > 0:
                self.net.append(nn.Dropout(self.dropout_p))

        final_linear = nn.Linear(layers[-2],
                                 layers[-1])
        self.net.append(final_linear)
        if self.return_density:
            self.net.append(nn.Sigmoid())
        
        self.net = nn.Sequential(*self.net)

        # Optional: FiLM re-injection of z at selected layers so conditioning is not washed out
        if self.use_film:
            z_dim = int(nz) if nz is not None else layers[0]
            hidden_dim = layers[1]
            self.film_blocks = nn.ModuleList([
                FiLMBlock(z_dim, hidden_dim, film_hidden)
                for _ in self._film_after_indices
            ])
            # Indices in _backbone (for hypernet path): same logical positions (after 1st and 3rd Gabor)
            self._film_backbone_indices = self._film_after_indices
        else:
            self.film_blocks = nn.ModuleList()
            self._film_backbone_indices = ()

        # Optional: LoRA on last two layers (penult Gabor's Linear + final Linear) for more expressivity
        if use_hypernet:
            from models.hypernetwork_lora import MultiLayerHypernetworkLoRA
            net_list = list(self.net)
            # Backbone: all but last two (penult Gabor + final Linear); if return_density, net ends with Sigmoid so drop 3
            n_tail = 3 if self.return_density else 2
            self._backbone = nn.Sequential(*(net_list[:-n_tail]))
            self._penult_gabor = self.net[-3] if self.return_density else self.net[-2]
            self._final_linear = self.net[-2] if self.return_density else self.net[-1]
            # Penult Gabor has freq_scale: Linear(layers[-2], 2*layers[-2]); final is Linear(layers[-2], layers[-1])
            penult_out = 2 * layers[-2]  # RealGaborLayer uses 2*out_features in freq_scale
            layer_specs = [
                (penult_out, layers[-2]),   # LoRA for Gabor's freq_scale (256 -> 512)
                (layers[-1], layers[-2]),   # LoRA for final linear (256 -> 1)
            ]
            self.hypernet = MultiLayerHypernetworkLoRA(
                c_dim=c_dim,
                layer_specs=layer_specs,
                rank=lora_rank,
                hidden_dims=hypernet_hidden,
                init_scale=lora_init_scale,
            )

    # Chunk size for LoRA bmm to avoid OOM when batch is large (e.g. tensor_product_xz with 70k+ rows)
    LORA_BMM_CHUNK = 4096

    def _chunked_lora_bmm(self, h, U, V):
        """Compute h @ (U @ V).T over batch in chunks to avoid materializing full (B, out, in) on GPU."""
        B = h.shape[0]
        if B <= self.LORA_BMM_CHUNK:
            uv = U @ V  # (B, out, in)
            return torch.bmm(h, uv.transpose(1, 2))
        out = []
        for start in range(0, B, self.LORA_BMM_CHUNK):
            end = min(start + self.LORA_BMM_CHUNK, B)
            h_ch = h[start:end]
            uv = U[start:end] @ V[start:end]
            out.append(torch.bmm(h_ch, uv.transpose(1, 2)))
        return torch.cat(out, dim=0)

    def forward(self, x, z):
        # vmap (e.g. in recalc_output/get_mesh) calls with unbatched x (3,) and z (nz,); hypernet path needs batched z
        unbatched = x.dim() == 1
        if unbatched:
            x = x.unsqueeze(0)
            z = z.unsqueeze(0)
        if self.use_dist_to_edge:
            from util.model_utils import dist_to_edge_x_from_z
            dist_x = dist_to_edge_x_from_z(x, z, self.n_studs_values, self.n_norm_col, n_norm_col_end=self.n_norm_col_end)
            x = torch.cat([x, dist_x.unsqueeze(-1)], dim=-1)
        if self.use_tiled_coords:
            from util.model_utils import tile_coords_xy
            x = tile_coords_xy(x, period_x=self.stud_spacing_x, period_y=self.stud_spacing_y, center=True)
        xz = torch.cat([x, z], dim=-1)

        if self.use_hypernet:
            # Contract: last c_dim columns of z must be (Nx_norm, Ny_norm, height_norm)
            assert z.shape[-1] >= self.c_dim, (
                f"use_hypernet with c_dim={self.c_dim} requires z to have at least {self.c_dim} columns; got z.shape[-1]={z.shape[-1]}"
            )
            c = z[:, -self.c_dim:].clamp(0.0, 1.0)
            lora_pairs = self.hypernet(c)  # list of (U, V) for penult and final
            U1, V1 = lora_pairs[0]
            U2, V2 = lora_pairs[1]

            if self.use_film:
                h = xz
                film_idx = 0
                for i, m in enumerate(self._backbone):
                    h = m(h)
                    if i in self._film_backbone_indices:
                        gamma, beta = self.film_blocks[film_idx](z)
                        film_idx += 1
                        h = (1 + gamma) * h + beta
            else:
                h = self._backbone(xz)  # (B, 256)
            # Penult Gabor with LoRA: base_freq + (U1@V1)@h per sample -> (B, 512)
            base_freq = self._penult_gabor.freq_scale(h)
            # Chunked to avoid OOM: (U1@V1) is (B, 512, 256); full batch can be 70k+ on 8GB GPU
            lora1 = self._chunked_lora_bmm(h.unsqueeze(1), U1, V1).squeeze(1)
            freq_out = base_freq + lora1
            omega, scale = torch.chunk(freq_out, 2, dim=-1)
            h2 = torch.cos(self._penult_gabor.omega_0 * omega) * torch.exp(-(self._penult_gabor.scale_0 * scale ** 2))

            y_base = self._final_linear(h2)
            y_lora = self._chunked_lora_bmm(h2.unsqueeze(1), U2, V2).squeeze(-1)
            output = y_base + y_lora
            if self.return_density:
                output = torch.sigmoid(output)
            if unbatched:
                output = output.squeeze(0)
            return output

        if self.use_film:
            h = xz
            film_idx = 0
            for i, m in enumerate(self.net):
                h = m(h)
                if i in self._film_after_indices:
                    gamma, beta = self.film_blocks[film_idx](z)
                    film_idx += 1
                    h = (1 + gamma) * h + beta
            output = h
        else:
            output = self.net(xz)
        if unbatched:
            output = output.squeeze(0)
        return output