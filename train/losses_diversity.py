"""
Diversity Losses for Generative GINN Training

This module implements losses to maximize dissimilarity between generated shapes,
encouraging the model to explore the full space of valid structures.

Three complementary approaches:
1. Chamfer Distance: Geometric dissimilarity on surface points
2. Contrastive Loss: Latent-space regularization
3. Volume Symmetric Difference: Voxel-based dissimilarity
"""

import torch
import torch.nn.functional as F
import einops
from typing import Optional, Tuple, Literal

from models.point_wrapper import PointWrapper


# =============================================================================
# Option A: Chamfer Distance on Surface Points
# =============================================================================

def pairwise_chamfer_distance(
    pts_a: torch.Tensor,  # [N, 3]
    pts_b: torch.Tensor,  # [M, 3]
    p: int = 2,
    eps: float = 1e-8
) -> torch.Tensor:
    """
    Compute asymmetric Chamfer distance from pts_a to pts_b.
    CD(A→B) = (1/N) * sum_{a in A} min_{b in B} ||a - b||^p
    
    Returns scalar distance.
    """
    if len(pts_a) == 0 or len(pts_b) == 0:
        return torch.tensor(0.0, device=pts_a.device)
    
    # Pairwise distances: [N, M]
    diff = pts_a[:, None, :] - pts_b[None, :, :]
    dists = torch.norm(diff, dim=-1, p=2)  # Euclidean
    
    # For each point in A, find min distance to B
    min_dists, _ = dists.min(dim=1)
    
    # Aggregate with p-norm
    cd = (min_dists ** p).mean() ** (1/p)
    return cd


def symmetric_chamfer_distance(
    pts_a: torch.Tensor,
    pts_b: torch.Tensor,
    p: int = 2,
    eps: float = 1e-8
) -> torch.Tensor:
    """Symmetric Chamfer distance: CD(A,B) = CD(A→B) + CD(B→A)"""
    cd_ab = pairwise_chamfer_distance(pts_a, pts_b, p=p, eps=eps)
    cd_ba = pairwise_chamfer_distance(pts_b, pts_a, p=p, eps=eps)
    return cd_ab + cd_ba


def diversity_loss_chamfer(
    surface_pts_batch: PointWrapper,
    aggregation: Literal['min', 'mean'] = 'min',
    p: int = 2,
    eps: float = 1e-8,
    max_diversity: Optional[float] = None,
) -> torch.Tensor:
    """
    Encourage shapes to be different from each other using Chamfer distance.
    
    For each shape i, compute min_j CD(i, j) for all j != i.
    Then maximize this minimum (push apart the most similar pairs).
    
    Args:
        surface_pts_batch: PointWrapper containing surface points for all shapes
        aggregation: 'min' maximizes minimum pairwise distance (robust to outliers)
                    'mean' maximizes average pairwise distance
        p: Chamfer distance power (1 or 2)
        eps: Numerical stability
        max_diversity: If set, loss is 0 when diversity exceeds this threshold
        
    Returns:
        Negative of aggregated Chamfer distance (to minimize = maximize diversity)
    """
    n_shapes = surface_pts_batch.bz
    
    if n_shapes < 2:
        return torch.tensor(0.0, device=surface_pts_batch.data.device)
    
    # Compute pairwise Chamfer distances
    cd_matrix = torch.zeros(n_shapes, n_shapes, device=surface_pts_batch.data.device)
    
    for i in range(n_shapes):
        pts_i = surface_pts_batch.pts_of_shape(i)
        for j in range(i + 1, n_shapes):
            pts_j = surface_pts_batch.pts_of_shape(j)
            cd_ij = symmetric_chamfer_distance(pts_i, pts_j, p=p, eps=eps)
            cd_matrix[i, j] = cd_ij
            cd_matrix[j, i] = cd_ij  # Symmetric
    
    # For each shape, find distance to its nearest neighbor
    # Set diagonal to inf so we don't pick same shape
    cd_matrix_no_diag = cd_matrix + torch.eye(n_shapes, device=cd_matrix.device) * 1e10
    min_distances = cd_matrix_no_diag.min(dim=1).values
    
    if aggregation == 'min':
        # Maximize the minimum of minimum distances (most similar pair)
        diversity = min_distances.min()
    else:  # 'mean'
        # Maximize average of minimum distances
        diversity = min_distances.mean()
    
    # Optional threshold
    if max_diversity is not None and diversity > max_diversity:
        return torch.tensor(0.0, device=surface_pts_batch.data.device)
    
    # Return negative because we want to maximize diversity
    # (optimizer minimizes, so minimize negative = maximize)
    return -diversity


# =============================================================================
# Option B: Latent-Space Contrastive Loss  
# =============================================================================

def diversity_loss_contrastive(
    z_batch: torch.Tensor,       # [B, nz] latent codes
    f_batch: torch.Tensor,       # [B, D] shape features (e.g., grid of SDF values)
    temperature: float = 0.1,
    similarity_threshold: float = 0.9,
) -> torch.Tensor:
    """
    Pull apart latent codes that produce similar shapes.
    
    This is a contrastive loss that penalizes when:
    - Two shapes are similar (high f correlation) 
    - But their latents are close (should be far apart for different shapes)
    
    The idea: if z_i ≈ z_j, the shapes should be similar. 
    Conversely, if shapes are similar but z_i != z_j, push z apart.
    
    Args:
        z_batch: Latent vectors [B, nz]
        f_batch: Shape features [B, D] - can be flattened SDF grid, 
                 surface point statistics, or learned embedding
        temperature: Scaling for similarity (lower = sharper)
        similarity_threshold: Shape similarity above which to apply penalty
        
    Returns:
        Contrastive diversity loss
    """
    B = z_batch.shape[0]
    if B < 2:
        return torch.tensor(0.0, device=z_batch.device)
    
    # Normalize for cosine similarity
    z_norm = F.normalize(z_batch, dim=-1)
    f_norm = F.normalize(f_batch, dim=-1)
    
    # Pairwise similarities [B, B]
    z_sim = torch.mm(z_norm, z_norm.t())  # Latent similarity
    f_sim = torch.mm(f_norm, f_norm.t())  # Shape similarity
    
    # Mask diagonal
    mask = 1 - torch.eye(B, device=z_batch.device)
    
    # Find pairs where shapes are similar but should have distinct latents
    # Penalty: shapes are similar (f_sim high) → latents should be different (z_sim low)
    # If f_sim is high and z_sim is also high → that's bad (same z producing same shape is fine,
    # but we want DIFFERENT z to produce DIFFERENT shapes)
    
    # For diversity: penalize when two RANDOM z produce SIMILAR shapes
    # This encourages the network to use the latent space efficiently
    shape_similarity_penalty = f_sim * mask  # How similar are shape pairs
    
    # Softmax over shape similarities to focus on most similar pairs
    weights = F.softmax(shape_similarity_penalty / temperature, dim=-1)
    
    # Loss: weighted average of shape similarities
    # Higher loss = shapes are too similar → want to push apart
    loss = (weights * shape_similarity_penalty).sum()
    
    return loss


def diversity_loss_latent_coverage(
    z_batch: torch.Tensor,  # [B, nz]
    target_std: float = 0.5,
) -> torch.Tensor:
    """
    Encourage latent codes to spread across the latent space.
    
    Simple regularizer that penalizes if z vectors are too clustered.
    
    Args:
        z_batch: Latent vectors [B, nz]
        target_std: Target standard deviation for latent distribution
        
    Returns:
        Loss penalizing low spread in latent space
    """
    if z_batch.shape[0] < 2:
        return torch.tensor(0.0, device=z_batch.device)
    
    # Compute variance across batch
    z_std = z_batch.std(dim=0)
    
    # Penalize if std is below target
    loss = F.relu(target_std - z_std).mean()
    
    return loss


# =============================================================================
# Option C: Volume Symmetric Difference
# =============================================================================

def diversity_loss_volume_symmetric_difference(
    sdf_batch: torch.Tensor,  # [B, X, Y, Z] or [B, X, Y] SDF grid
    level_set: float = 0.0,
    aggregation: Literal['min', 'mean'] = 'min',
    max_diversity: Optional[float] = None,
) -> torch.Tensor:
    """
    Encourage shapes to be different using volume symmetric difference.
    
    Volume Symmetric Difference: |A Δ B| = |A ∪ B| - |A ∩ B|
                                        = |A| + |B| - 2|A ∩ B|
    
    This is the voxel-based analog of Chamfer distance.
    
    Args:
        sdf_batch: SDF values on a grid [B, *spatial_dims]
        level_set: SDF value defining the surface (usually 0)
        aggregation: 'min' or 'mean' over pairwise differences
        max_diversity: Optional threshold above which loss is 0
        
    Returns:
        Negative of aggregated symmetric difference (to maximize diversity)
    """
    B = sdf_batch.shape[0]
    if B < 2:
        return torch.tensor(0.0, device=sdf_batch.device)
    
    # Binarize: 1 if inside shape (SDF < level_set), 0 if outside
    # Using sigmoid for differentiability
    beta = 50.0  # Sharpness of Heaviside approximation
    occupancy = torch.sigmoid(-beta * (sdf_batch - level_set))  # [B, *spatial]
    
    # Flatten spatial dims
    occupancy_flat = occupancy.view(B, -1)  # [B, N_voxels]
    
    # Compute pairwise symmetric difference
    # |A Δ B| = |A| + |B| - 2|A ∩ B|
    # |A ∩ B| ≈ sum(occ_A * occ_B)
    
    sym_diff_matrix = torch.zeros(B, B, device=sdf_batch.device)
    
    for i in range(B):
        vol_i = occupancy_flat[i].sum()
        for j in range(i + 1, B):
            vol_j = occupancy_flat[j].sum()
            intersection = (occupancy_flat[i] * occupancy_flat[j]).sum()
            sym_diff = vol_i + vol_j - 2 * intersection
            sym_diff_matrix[i, j] = sym_diff
            sym_diff_matrix[j, i] = sym_diff
    
    # Normalize by max possible volume
    max_vol = occupancy_flat.shape[1]
    sym_diff_matrix = sym_diff_matrix / max_vol
    
    # For each shape, find its most similar neighbor
    sym_diff_no_diag = sym_diff_matrix + torch.eye(B, device=sdf_batch.device) * 1e10
    min_differences = sym_diff_no_diag.min(dim=1).values
    
    if aggregation == 'min':
        diversity = min_differences.min()
    else:
        diversity = min_differences.mean()
    
    # Optional threshold
    if max_diversity is not None and diversity > max_diversity:
        return torch.tensor(0.0, device=sdf_batch.device)
    
    return -diversity


# =============================================================================
# Combined Diversity Loss
# =============================================================================

def diversity_loss_combined(
    surface_pts_batch: Optional[PointWrapper] = None,
    sdf_batch: Optional[torch.Tensor] = None,
    z_batch: Optional[torch.Tensor] = None,
    f_batch: Optional[torch.Tensor] = None,
    weights: dict = None,
    **kwargs,
) -> Tuple[torch.Tensor, dict]:
    """
    Combined diversity loss using multiple metrics.
    
    Args:
        surface_pts_batch: For Chamfer diversity
        sdf_batch: For volume symmetric difference
        z_batch, f_batch: For contrastive loss
        weights: Dict with keys 'chamfer', 'volume', 'contrastive', 'coverage'
        
    Returns:
        total_loss: Combined weighted loss
        loss_dict: Individual loss components for logging
    """
    if weights is None:
        weights = {'chamfer': 1.0, 'volume': 0.0, 'contrastive': 0.0, 'coverage': 0.0}
    
    loss_dict = {}
    total_loss = torch.tensor(0.0)
    
    # Get device from any available input
    device = None
    if surface_pts_batch is not None:
        device = surface_pts_batch.data.device
    elif sdf_batch is not None:
        device = sdf_batch.device
    elif z_batch is not None:
        device = z_batch.device
    
    if device is not None:
        total_loss = total_loss.to(device)
    
    # Chamfer diversity
    if weights.get('chamfer', 0) > 0 and surface_pts_batch is not None:
        loss_chamfer = diversity_loss_chamfer(surface_pts_batch, **kwargs)
        loss_dict['div_chamfer'] = loss_chamfer
        total_loss = total_loss + weights['chamfer'] * loss_chamfer
    
    # Volume symmetric difference
    if weights.get('volume', 0) > 0 and sdf_batch is not None:
        loss_volume = diversity_loss_volume_symmetric_difference(sdf_batch, **kwargs)
        loss_dict['div_volume'] = loss_volume
        total_loss = total_loss + weights['volume'] * loss_volume
    
    # Contrastive loss
    if weights.get('contrastive', 0) > 0 and z_batch is not None and f_batch is not None:
        loss_contrastive = diversity_loss_contrastive(z_batch, f_batch, **kwargs)
        loss_dict['div_contrastive'] = loss_contrastive
        total_loss = total_loss + weights['contrastive'] * loss_contrastive
    
    # Latent coverage
    if weights.get('coverage', 0) > 0 and z_batch is not None:
        loss_coverage = diversity_loss_latent_coverage(z_batch, **kwargs)
        loss_dict['div_coverage'] = loss_coverage
        total_loss = total_loss + weights['coverage'] * loss_coverage
    
    loss_dict['div_total'] = total_loss
    return total_loss, loss_dict
