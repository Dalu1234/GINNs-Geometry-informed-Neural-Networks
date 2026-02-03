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
    
class ConditionalWIRE(nn.Module):
    '''
    Since complex numbers don't work with jacrev, we need to use the real number version of the WIRE.
    Optional tiled coordinates: append (x_tile, y_tile) modulo stud spacing so the network
    sees position within each tile and can learn one cylindrical feature per tile.
    Optional dist_to_edge_x: append signed distance to nearest x-boundary (positive inside brick)
    so the network can ignore phantom studs outside the brick (e.g. dist_to_edge_x < 0).
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
                 **kwargs):
        super().__init__()
        self.layers = layers
        self.return_density = return_density
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

        for i in range(1, len(layers) - 2):
            self.net.append(self.nonlin(layers[i],
                                        layers[i+1], 
                                        omega0=hidden_omega_0,
                                        sigma0=scale))

        final_linear = nn.Linear(layers[-2],
                                 layers[-1])
        self.net.append(final_linear)
        if self.return_density:
            self.net.append(nn.Sigmoid())
        
        self.net = nn.Sequential(*self.net)
    
    def forward(self, x, z):
        if self.use_dist_to_edge:
            from util.model_utils import dist_to_edge_x_from_z
            dist_x = dist_to_edge_x_from_z(x, z, self.n_studs_values, self.n_norm_col)
            x = torch.cat([x, dist_x.unsqueeze(-1)], dim=-1)
        if self.use_tiled_coords:
            from util.model_utils import tile_coords_xy
            x = tile_coords_xy(x, period_x=self.stud_spacing_x, period_y=self.stud_spacing_y, center=True)
        xz = torch.cat([x, z], dim=-1)
        output = self.net(xz)
        return output