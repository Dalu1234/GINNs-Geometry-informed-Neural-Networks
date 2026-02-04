"""
Structured LEGO Decoder

A geometry-aware neural network that predicts geometric parameters,
then uses analytic SDF formulas to compute the shape.

Key difference from GINN: the network outputs parameters (L, H, D, stud_positions),
NOT raw SDF values. The SDF is computed by fixed analytic formulas.

This guarantees:
1. Interpolated latents produce valid LEGO shapes (parameters interpolate)
2. Output is always a valid box + cylinders
3. No blurry SDF blending

V2 Changes:
- Continuous N input (not one-hot) for proper generalization
- Real latent z for variation
- Test generalization to unseen N values
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional, Dict, List


# =============================================================================
# ANALYTIC SDF PRIMITIVES
# =============================================================================

def sdf_box(p: torch.Tensor, half_extents: torch.Tensor) -> torch.Tensor:
    """
    Signed distance to an axis-aligned box centered at origin.
    
    Args:
        p: (B, N, 3) query points
        half_extents: (B, 3) half-sizes (half_L, half_H, half_D)
    
    Returns:
        (B, N) signed distances (negative inside, positive outside)
    """
    # Expand half_extents for broadcasting: (B, 1, 3)
    h = half_extents.unsqueeze(1)
    
    # Distance to each face
    q = torch.abs(p) - h  # (B, N, 3)
    
    # Outside distance: length of positive part of q
    outside = torch.norm(torch.clamp(q, min=0.0), dim=-1)
    
    # Inside distance: negative of distance to nearest face
    inside = torch.clamp(q.max(dim=-1).values, max=0.0)
    
    return outside + inside


def sdf_cylinder_z(p: torch.Tensor, center_xy: torch.Tensor, 
                   radius: torch.Tensor, z_min: torch.Tensor, 
                   z_max: torch.Tensor) -> torch.Tensor:
    """
    Signed distance to a Z-aligned cylinder (vertical cylinder).
    
    Args:
        p: (B, N, 3) query points
        center_xy: (B, 2) center in XY plane
        radius: (B,) cylinder radius
        z_min: (B,) bottom z coordinate
        z_max: (B,) top z coordinate
    
    Returns:
        (B, N) signed distances
    """
    # Shift to cylinder's XY center
    px = p[..., 0] - center_xy[:, 0:1]  # (B, N)
    py = p[..., 1] - center_xy[:, 1:2]  # (B, N)
    pz = p[..., 2]  # (B, N)
    
    # Radial distance in XY
    d_radial = torch.sqrt(px**2 + py**2 + 1e-8) - radius.unsqueeze(1)  # (B, N)
    
    # Z distance (capped cylinder)
    half_h = (z_max - z_min).unsqueeze(1) / 2  # (B, 1)
    center_z = ((z_max + z_min) / 2).unsqueeze(1)  # (B, 1)
    d_z = torch.abs(pz - center_z) - half_h  # (B, N)
    
    # Combine: 2D SDF for capped cylinder
    outside = torch.norm(torch.stack([
        torch.clamp(d_radial, min=0.0),
        torch.clamp(d_z, min=0.0)
    ], dim=-1), dim=-1)
    
    inside = torch.clamp(torch.maximum(d_radial, d_z), max=0.0)
    
    return outside + inside


def sdf_cylinder_y(p: torch.Tensor, center_xz: torch.Tensor,
                   radius: torch.Tensor, y_min: torch.Tensor,
                   y_max: torch.Tensor) -> torch.Tensor:
    """
    Signed distance to a Y-aligned cylinder (studs pointing up in Y).
    
    Args:
        p: (B, N, 3) query points (x, y, z)
        center_xz: (B, 2) center in XZ plane
        radius: (B,) cylinder radius
        y_min: (B,) bottom y coordinate
        y_max: (B,) top y coordinate
    
    Returns:
        (B, N) signed distances
    """
    # Shift to cylinder's XZ center
    px = p[..., 0] - center_xz[:, 0:1]  # (B, N)
    py = p[..., 1]  # (B, N)
    pz = p[..., 2] - center_xz[:, 1:2]  # (B, N)
    
    # Radial distance in XZ plane
    d_radial = torch.sqrt(px**2 + pz**2 + 1e-8) - radius.unsqueeze(1)  # (B, N)
    
    # Y distance (capped cylinder)
    half_h = (y_max - y_min).unsqueeze(1) / 2  # (B, 1)
    center_y = ((y_max + y_min) / 2).unsqueeze(1)  # (B, 1)
    d_y = torch.abs(py - center_y) - half_h  # (B, N)
    
    # Combine: 2D SDF for capped cylinder
    outside = torch.norm(torch.stack([
        torch.clamp(d_radial, min=0.0),
        torch.clamp(d_y, min=0.0)
    ], dim=-1), dim=-1)
    
    inside = torch.clamp(torch.maximum(d_radial, d_y), max=0.0)
    
    return outside + inside


# =============================================================================
# SMOOTH CSG OPERATIONS
# =============================================================================

def smooth_union(d1: torch.Tensor, d2: torch.Tensor, k: float = 0.05) -> torch.Tensor:
    """
    Smooth union (blend) of two SDFs.
    
    Args:
        d1, d2: SDF values (same shape)
        k: smoothing radius (larger = more blending)
    
    Returns:
        Blended SDF (inside both or either)
    """
    h = torch.clamp(0.5 + 0.5 * (d2 - d1) / (k + 1e-8), 0.0, 1.0)
    return d2 * (1 - h) + d1 * h - k * h * (1 - h)


def smooth_subtraction(d1: torch.Tensor, d2: torch.Tensor, k: float = 0.05) -> torch.Tensor:
    """
    Smooth subtraction: d1 minus d2.
    
    Args:
        d1: SDF of base shape
        d2: SDF of shape to subtract
        k: smoothing radius
    
    Returns:
        d1 with d2 carved out
    """
    h = torch.clamp(0.5 - 0.5 * (d1 + d2) / (k + 1e-8), 0.0, 1.0)
    return d1 * (1 - h) + (-d2) * h + k * h * (1 - h)


def smooth_intersection(d1: torch.Tensor, d2: torch.Tensor, k: float = 0.05) -> torch.Tensor:
    """
    Smooth intersection of two SDFs.
    
    Args:
        d1, d2: SDF values
        k: smoothing radius
    
    Returns:
        Blended SDF (inside both)
    """
    h = torch.clamp(0.5 - 0.5 * (d2 - d1) / (k + 1e-8), 0.0, 1.0)
    return d2 * (1 - h) + d1 * h + k * h * (1 - h)


def hard_union(d1: torch.Tensor, d2: torch.Tensor) -> torch.Tensor:
    """Hard (non-smooth) union: min(d1, d2)."""
    return torch.minimum(d1, d2)


# =============================================================================
# PARAMETER HEAD V2 - Continuous N input, LEARNS the relationship
# =============================================================================

class ParameterHeadV2(nn.Module):
    """
    MLP that predicts geometric parameters from CONTINUOUS N input.
    
    Key: Network must LEARN that half_L = N * spacing / 2.
    We do NOT hardcode this — the network discovers it from supervision.
    
    Inputs:
    - n_studs: (B, 1) continuous stud count (e.g., 2.0, 3.5, 4.0)
    - z: (B, z_dim) optional latent for variation
    
    Outputs:
    - half_extents: (B, 3) box dimensions — ALL LEARNED
    - stud_radius: (B, 1) 
    - stud_height: (B, 1)
    """
    
    def __init__(self, 
                 z_dim: int = 4,
                 hidden_dims: List[int] = [64, 64],
                 base_half_H: float = 0.15,
                 base_half_D: float = 0.125,
                 stud_spacing: float = 0.25,  # only used for stud positions, not L
                 base_stud_radius: float = 0.10,
                 base_stud_height: float = 0.08):
        super().__init__()
        
        self.z_dim = z_dim
        self.stud_spacing = stud_spacing  # for stud position computation only
        
        # Store base values for reference (but network learns to predict)
        self.base_half_H = base_half_H
        self.base_half_D = base_half_D
        self.base_stud_radius = base_stud_radius
        self.base_stud_height = base_stud_height
        
        # Input: n_studs (1) + z (z_dim)
        input_dim = 1 + z_dim
        
        # MLP predicts ALL parameters from N
        layers = []
        in_d = input_dim
        for h_d in hidden_dims:
            layers.extend([nn.Linear(in_d, h_d), nn.ReLU()])
            in_d = h_d
        self.backbone = nn.Sequential(*layers)
        
        # Output heads - network learns everything
        # half_L: must learn half_L = N * spacing / 2
        # half_H, half_D: should stay ~constant
        self.box_head = nn.Sequential(
            nn.Linear(hidden_dims[-1], 3),
            nn.Softplus()  # ensure positive
        )
        
        # Stud params
        self.stud_head = nn.Sequential(
            nn.Linear(hidden_dims[-1], 2),
            nn.Softplus()
        )
    
    def forward(self, n_studs: torch.Tensor, z: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Args:
            n_studs: (B, 1) continuous stud count
            z: (B, z_dim) latent vector for variation
        
        Returns:
            dict with geometric parameters — ALL LEARNED FROM DATA
        """
        B = n_studs.shape[0]
        device = n_studs.device
        
        # Concatenate inputs
        if z is not None and self.z_dim > 0:
            x = torch.cat([n_studs, z], dim=-1)
        else:
            z = torch.zeros(B, self.z_dim, device=device)
            x = torch.cat([n_studs, z], dim=-1)
        
        # Get features
        features = self.backbone(x)
        
        # Predict box dimensions — LEARNED, not hardcoded
        # Network must discover: half_L = N * 0.125 (for spacing=0.25)
        half_extents = self.box_head(features)  # (B, 3)
        
        # Predict stud params
        stud_params = self.stud_head(features)  # (B, 2)
        stud_radius = stud_params[:, 0:1]
        stud_height = stud_params[:, 1:2]
        
        return {
            'half_extents': half_extents,
            'n_studs': n_studs,  # pass through for SDF layer
            'stud_radius': stud_radius,
            'stud_height': stud_height
        }


# =============================================================================
# ANALYTIC LEGO SDF V2 - Vectorized studs
# =============================================================================

class AnalyticLegoSDFV2(nn.Module):
    """
    Computes LEGO brick SDF from geometric parameters.
    V2: Vectorized stud computation, handles fractional N gracefully.
    """
    
    def __init__(self, 
                 smooth_k: float = 0.02,
                 max_studs: int = 12,
                 stud_spacing: float = 0.25):
        super().__init__()
        self.smooth_k = smooth_k
        self.max_studs = max_studs
        self.stud_spacing = stud_spacing
    
    def forward(self, 
                p: torch.Tensor,
                half_extents: torch.Tensor,
                n_studs: torch.Tensor,
                stud_radius: torch.Tensor,
                stud_height: torch.Tensor,
                use_studs: bool = True) -> torch.Tensor:
        """
        Compute SDF for LEGO brick at query points.
        
        Args:
            p: (B, N_pts, 3) query points
            half_extents: (B, 3) box half-sizes
            n_studs: (B, 1) stud count (can be fractional)
            stud_radius: (B, 1) stud radius
            stud_height: (B, 1) stud height
            use_studs: whether to add studs
        
        Returns:
            (B, N_pts) signed distances
        """
        B, N_pts, _ = p.shape
        device = p.device
        
        half_L = half_extents[:, 0]  # (B,)
        half_H = half_extents[:, 1]  # (B,)
        half_D = half_extents[:, 2]  # (B,)
        
        # Base box SDF
        sdf = sdf_box(p, half_extents)  # (B, N_pts)
        
        if not use_studs:
            return sdf
        
        # Add studs
        n_studs_int = torch.round(n_studs.squeeze(-1)).long()  # (B,)
        n_studs_float = n_studs.squeeze(-1)  # (B,) for soft masking
        
        # For each sample in batch, compute studs
        for b in range(B):
            n = min(n_studs_int[b].item(), self.max_studs)
            if n <= 0:
                continue
            
            # Stud X positions: evenly spaced
            half_L_b = half_L[b].item()
            stud_xs = torch.linspace(-half_L_b + self.stud_spacing/2, 
                                     half_L_b - self.stud_spacing/2, 
                                     n, device=device)
            
            # Stud params for this sample
            r = stud_radius[b, 0]
            h = stud_height[b, 0]
            y_min = half_H[b]
            y_max = half_H[b] + h
            
            # Points for this sample
            p_b = p[b:b+1]  # (1, N_pts, 3)
            sdf_b = sdf[b:b+1]  # (1, N_pts)
            
            for i, x_pos in enumerate(stud_xs):
                # Cylinder center in XZ plane
                center_xz = torch.tensor([[x_pos.item(), 0.0]], device=device)  # (1, 2)
                
                # Cylinder SDF
                cyl_sdf = sdf_cylinder_y(p_b, center_xz, r.unsqueeze(0), 
                                        y_min.unsqueeze(0), y_max.unsqueeze(0))  # (1, N_pts)
                
                # Soft mask for fractional studs
                # Stud i contributes fully if i < floor(n), partially if i is fractional
                stud_weight = torch.clamp(n_studs_float[b] - i, 0.0, 1.0)
                
                # Blend
                blended = smooth_union(sdf_b, cyl_sdf, k=self.smooth_k)
                sdf_b = stud_weight * blended + (1 - stud_weight) * sdf_b
            
            sdf[b:b+1] = sdf_b
        
        return sdf


# =============================================================================
# FULL STRUCTURED MODEL V2
# =============================================================================

class StructuredLegoModelV2(nn.Module):
    """
    Structured LEGO model V2 with continuous N input.
    
    Key differences from V1:
    1. N is continuous, not one-hot → can generalize to unseen N
    2. Latent z allows variation in stud dimensions
    3. Box length is analytically derived from N (not learned)
    """
    
    def __init__(self,
                 z_dim: int = 4,
                 hidden_dims: List[int] = [64, 64],
                 max_studs: int = 12,
                 smooth_k: float = 0.02,
                 use_studs: bool = True,
                 # Base dimensions
                 base_half_H: float = 0.15,
                 base_half_D: float = 0.125,
                 stud_spacing: float = 0.25,
                 base_stud_radius: float = 0.10,
                 base_stud_height: float = 0.08):
        super().__init__()
        
        self.z_dim = z_dim
        self.max_studs = max_studs
        self.use_studs = use_studs
        self.stud_spacing = stud_spacing
        
        self.param_head = ParameterHeadV2(
            z_dim=z_dim,
            hidden_dims=hidden_dims,
            base_half_H=base_half_H,
            base_half_D=base_half_D,
            stud_spacing=stud_spacing,
            base_stud_radius=base_stud_radius,
            base_stud_height=base_stud_height
        )
        
        self.sdf_layer = AnalyticLegoSDFV2(
            smooth_k=smooth_k,
            max_studs=max_studs,
            stud_spacing=stud_spacing
        )
    
    def get_params(self, n_studs: torch.Tensor, z: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Get geometric parameters without computing SDF."""
        return self.param_head(n_studs, z)
    
    def forward(self, 
                p: torch.Tensor, 
                n_studs: torch.Tensor,
                z: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass: (n_studs, z) -> params -> SDF.
        
        Args:
            p: (B, N_pts, 3) query points
            n_studs: (B, 1) stud count (continuous)
            z: (B, z_dim) optional latent
        
        Returns:
            (B, N_pts) signed distances
        """
        params = self.param_head(n_studs, z)
        
        sdf = self.sdf_layer(
            p,
            params['half_extents'],
            params['n_studs'],
            params['stud_radius'],
            params['stud_height'],
            use_studs=self.use_studs
        )
        
        return sdf
    
    def forward_with_params(self, 
                            p: torch.Tensor,
                            n_studs: torch.Tensor,
                            z: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass returning both SDF and parameters."""
        params = self.param_head(n_studs, z)
        
        sdf = self.sdf_layer(
            p,
            params['half_extents'],
            params['n_studs'],
            params['stud_radius'],
            params['stud_height'],
            use_studs=self.use_studs
        )
        
        return sdf, params


# =============================================================================
# KEEP ORIGINAL CLASSES FOR BACKWARD COMPATIBILITY
# =============================================================================

class ParameterHead(nn.Module):
    """
    MLP that predicts geometric parameters from condition vector.
    
    For LEGO 1xN bricks:
    - Box: half_L, half_H, half_D (3 params)
    - Studs: n_studs + per-stud params (positions derived from half_L)
    """
    
    def __init__(self, 
                 condition_dim: int = 2,
                 z_dim: int = 0,  # optional latent input
                 hidden_dims: list = [64, 64],
                 max_studs: int = 8,
                 predict_stud_params: bool = True):
        super().__init__()
        
        self.condition_dim = condition_dim
        self.z_dim = z_dim
        self.max_studs = max_studs
        self.predict_stud_params = predict_stud_params
        
        input_dim = condition_dim + z_dim
        
        # Shared backbone
        layers = []
        in_d = input_dim
        for h_d in hidden_dims:
            layers.extend([nn.Linear(in_d, h_d), nn.ReLU()])
            in_d = h_d
        self.backbone = nn.Sequential(*layers)
        
        # Box parameters head: (half_L, half_H, half_D)
        self.box_head = nn.Sequential(
            nn.Linear(hidden_dims[-1], 3),
            nn.Softplus()  # ensure positive
        )
        
        # Stud count head: predicts soft n_studs
        self.n_studs_head = nn.Linear(hidden_dims[-1], 1)
        
        # Optional: per-stud radius/height offsets (usually fixed)
        if predict_stud_params:
            # (radius_offset, height_offset) per stud — small adjustments
            self.stud_head = nn.Linear(hidden_dims[-1], max_studs * 2)
        else:
            self.stud_head = None
        
        # Default stud dimensions (normalized, approximately LEGO proportions)
        # Real LEGO: stud_radius ≈ 2.4mm, stud_height ≈ 1.8mm, brick_height ≈ 9.6mm
        self.register_buffer('default_stud_radius', torch.tensor(0.15))
        self.register_buffer('default_stud_height', torch.tensor(0.12))
    
    def forward(self, condition: torch.Tensor, z: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Args:
            condition: (B, condition_dim) one-hot or embedding
            z: (B, z_dim) optional latent vector
        
        Returns:
            dict with:
                - half_extents: (B, 3)
                - n_studs_soft: (B, 1) soft stud count
                - stud_radii: (B, max_studs)
                - stud_heights: (B, max_studs)
        """
        B = condition.shape[0]
        
        # Concatenate inputs
        if z is not None and self.z_dim > 0:
            x = torch.cat([condition, z], dim=-1)
        else:
            x = condition
        
        # Shared features
        features = self.backbone(x)
        
        # Box parameters
        half_extents = self.box_head(features)  # (B, 3)
        
        # Stud count (soft, 0 to max_studs)
        n_studs_soft = torch.sigmoid(self.n_studs_head(features)) * self.max_studs  # (B, 1)
        
        # Stud parameters
        if self.stud_head is not None:
            stud_offsets = self.stud_head(features).view(B, self.max_studs, 2)
            stud_radii = self.default_stud_radius + 0.05 * torch.tanh(stud_offsets[..., 0])
            stud_heights = self.default_stud_height + 0.05 * torch.tanh(stud_offsets[..., 1])
        else:
            stud_radii = self.default_stud_radius.expand(B, self.max_studs)
            stud_heights = self.default_stud_height.expand(B, self.max_studs)
        
        return {
            'half_extents': half_extents,
            'n_studs_soft': n_studs_soft,
            'stud_radii': stud_radii,
            'stud_heights': stud_heights
        }


class AnalyticLegoSDF(nn.Module):
    """
    Computes LEGO brick SDF from geometric parameters.
    
    No learnable parameters — pure geometry.
    Differentiable w.r.t. input params for end-to-end training.
    """
    
    def __init__(self, 
                 smooth_k: float = 0.02,
                 stud_y_up: bool = True):  # Y is up (LEGO convention)
        super().__init__()
        self.smooth_k = smooth_k
        self.stud_y_up = stud_y_up
    
    def compute_stud_positions(self, half_L: torch.Tensor, n_studs: int) -> torch.Tensor:
        """
        Compute stud X positions based on brick length.
        Studs are evenly spaced along X axis.
        
        Args:
            half_L: (B,) half-length of brick
            n_studs: integer number of studs
        
        Returns:
            (B, n_studs) X positions for each stud
        """
        if n_studs == 0:
            return None
        
        B = half_L.shape[0]
        device = half_L.device
        
        # Studs evenly distributed from -half_L to +half_L
        # For N studs: positions at -half_L + spacing/2 + i*spacing
        # where spacing = 2*half_L / N
        
        L = 2 * half_L  # full length
        spacing = L / n_studs  # (B,)
        
        # Stud indices: 0, 1, ..., n_studs-1
        indices = torch.arange(n_studs, device=device).float()  # (n_studs,)
        
        # Positions: -half_L + spacing/2 + i * spacing
        positions = -half_L.unsqueeze(1) + spacing.unsqueeze(1) / 2 + \
                    indices.unsqueeze(0) * spacing.unsqueeze(1)  # (B, n_studs)
        
        return positions
    
    def forward(self, 
                p: torch.Tensor,
                half_extents: torch.Tensor,
                n_studs_soft: torch.Tensor,
                stud_radii: torch.Tensor,
                stud_heights: torch.Tensor,
                max_studs: int = 8) -> torch.Tensor:
        """
        Compute SDF for LEGO brick at query points.
        
        Args:
            p: (B, N, 3) query points
            half_extents: (B, 3) box half-sizes (half_L, half_H, half_D)
            n_studs_soft: (B, 1) soft stud count
            stud_radii: (B, max_studs) per-stud radii
            stud_heights: (B, max_studs) per-stud heights
            max_studs: maximum number of studs to consider
        
        Returns:
            (B, N) signed distances
        """
        B, N, _ = p.shape
        device = p.device
        
        half_L = half_extents[:, 0]  # (B,)
        half_H = half_extents[:, 1]  # (B,)
        half_D = half_extents[:, 2]  # (B,)
        
        # Base box SDF
        sdf = sdf_box(p, half_extents)  # (B, N)
        
        # Add studs (cylinders on top face)
        # Use integer n_studs for positions, soft mask for blending
        n_studs_int = torch.round(n_studs_soft).long().squeeze(-1)  # (B,)
        n_studs_max = min(n_studs_int.max().item(), max_studs)
        
        if n_studs_max > 0:
            for b in range(B):
                n = min(n_studs_int[b].item(), max_studs)
                if n > 0:
                    # Compute stud positions for this sample
                    stud_x = self.compute_stud_positions(half_L[b:b+1], n)  # (1, n)
                    
                    for i in range(n):
                        # Stud center in XZ plane (Y is up)
                        center_xz = torch.stack([stud_x[0, i], torch.zeros_like(stud_x[0, i])], dim=0).unsqueeze(0)  # (1, 2)
                        
                        radius = stud_radii[b, i].unsqueeze(0)  # (1,)
                        height = stud_heights[b, i]
                        
                        # Y bounds: top of box to top of box + stud height
                        y_min = half_H[b:b+1]  # (1,)
                        y_max = half_H[b:b+1] + height.unsqueeze(0)  # (1,)
                        
                        # Cylinder SDF for this stud
                        p_b = p[b:b+1]  # (1, N, 3)
                        stud_sdf = sdf_cylinder_y(p_b, center_xz, radius, y_min, y_max)  # (1, N)
                        
                        # Soft mask: stud contributes if i < n_studs_soft
                        mask = torch.sigmoid(10 * (n_studs_soft[b] - i - 0.5))  # scalar
                        
                        # Blend with smooth union
                        sdf_b = sdf[b:b+1]
                        blended = smooth_union(sdf_b, stud_sdf, k=self.smooth_k)
                        sdf[b:b+1] = mask * blended + (1 - mask) * sdf_b
        
        return sdf


class StructuredLegoModel(nn.Module):
    """
    Complete structured LEGO model.
    
    condition -> ParameterHead -> (half_extents, stud_params)
                                         |
                                         v
    query_points -----------------> AnalyticLegoSDF -> SDF values
    
    For interpolation: interpolate conditions or latents,
    get interpolated parameters, compute valid SDF.
    """
    
    def __init__(self,
                 condition_dim: int = 2,
                 z_dim: int = 0,
                 hidden_dims: list = [64, 64],
                 max_studs: int = 8,
                 predict_stud_params: bool = True,
                 smooth_k: float = 0.02,
                 use_studs: bool = True):
        super().__init__()
        
        self.condition_dim = condition_dim
        self.z_dim = z_dim
        self.max_studs = max_studs
        self.use_studs = use_studs
        
        self.param_head = ParameterHead(
            condition_dim=condition_dim,
            z_dim=z_dim,
            hidden_dims=hidden_dims,
            max_studs=max_studs,
            predict_stud_params=predict_stud_params
        )
        
        self.sdf_layer = AnalyticLegoSDF(smooth_k=smooth_k)
    
    def get_params(self, condition: torch.Tensor, z: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Get geometric parameters without computing SDF."""
        return self.param_head(condition, z)
    
    def forward(self, 
                p: torch.Tensor, 
                condition: torch.Tensor,
                z: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass: condition -> params -> SDF.
        
        Args:
            p: (B, N, 3) query points
            condition: (B, condition_dim) brick type encoding
            z: (B, z_dim) optional latent
        
        Returns:
            (B, N) signed distances
        """
        params = self.param_head(condition, z)
        
        if self.use_studs:
            sdf = self.sdf_layer(
                p,
                params['half_extents'],
                params['n_studs_soft'],
                params['stud_radii'],
                params['stud_heights'],
                max_studs=self.max_studs
            )
        else:
            # Box only
            sdf = sdf_box(p, params['half_extents'])
        
        return sdf
    
    def forward_with_params(self, 
                            p: torch.Tensor,
                            condition: torch.Tensor,
                            z: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass returning both SDF and parameters."""
        params = self.param_head(condition, z)
        
        if self.use_studs:
            sdf = self.sdf_layer(
                p,
                params['half_extents'],
                params['n_studs_soft'],
                params['stud_radii'],
                params['stud_heights'],
                max_studs=self.max_studs
            )
        else:
            sdf = sdf_box(p, params['half_extents'])
        
        return sdf, params


# =============================================================================
# TARGET PARAMETER COMPUTATION (for supervision)
# =============================================================================

def compute_target_params(condition: torch.Tensor, 
                          brick_specs: Dict[str, Dict]) -> Dict[str, torch.Tensor]:
    """
    Compute target geometric parameters from brick specifications.
    
    Args:
        condition: (B, condition_dim) one-hot encoding
        brick_specs: dict mapping condition index to brick params
                     e.g., {0: {'n': 2, 'half_L': 0.25, 'half_H': 0.15, 'half_D': 0.125},
                            1: {'n': 4, 'half_L': 0.5, ...}}
    
    Returns:
        dict with target parameters
    """
    B = condition.shape[0]
    device = condition.device
    
    # Get condition indices
    cond_idx = condition.argmax(dim=-1)  # (B,)
    
    half_extents = torch.zeros(B, 3, device=device)
    n_studs = torch.zeros(B, 1, device=device)
    
    for b in range(B):
        idx = cond_idx[b].item()
        spec = brick_specs[idx]
        half_extents[b, 0] = spec['half_L']
        half_extents[b, 1] = spec['half_H']
        half_extents[b, 2] = spec['half_D']
        n_studs[b, 0] = spec['n']
    
    return {
        'half_extents': half_extents,
        'n_studs': n_studs
    }

