"""
Losses for Structured LEGO Decoder

Supervises geometric parameters directly, not SDF values.
"""

import torch
import torch.nn.functional as F
from typing import Dict, Optional


def loss_param_supervision(pred_params: Dict[str, torch.Tensor],
                           target_params: Dict[str, torch.Tensor],
                           weights: Optional[Dict[str, float]] = None) -> Dict[str, torch.Tensor]:
    """
    Supervise predicted parameters against target values.
    
    Args:
        pred_params: dict with 'half_extents', 'n_studs_soft', etc.
        target_params: dict with target values
        weights: optional per-component weights
    
    Returns:
        dict with individual losses and total
    """
    if weights is None:
        weights = {
            'half_extents': 1.0,
            'n_studs': 1.0
        }
    
    losses = {}
    
    # Box dimensions loss
    if 'half_extents' in target_params:
        losses['box'] = F.mse_loss(
            pred_params['half_extents'], 
            target_params['half_extents']
        )
    
    # Stud count loss
    if 'n_studs' in target_params:
        losses['n_studs'] = F.mse_loss(
            pred_params['n_studs_soft'],
            target_params['n_studs'].float()
        )
    
    # Weighted total
    total = 0.0
    for k, v in losses.items():
        w = weights.get(k, 1.0)
        total = total + w * v
    
    losses['total'] = total
    return losses


def loss_sdf_supervision(pred_sdf: torch.Tensor,
                         target_sdf: torch.Tensor,
                         mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Optional SDF supervision (in addition to parameter supervision).
    
    Useful for fine-tuning or when you have ground-truth SDF values.
    
    Args:
        pred_sdf: (B, N) predicted SDF
        target_sdf: (B, N) target SDF
        mask: (B, N) optional mask for which points to supervise
    
    Returns:
        scalar loss
    """
    if mask is not None:
        diff = (pred_sdf - target_sdf) ** 2
        return (diff * mask).sum() / (mask.sum() + 1e-8)
    else:
        return F.mse_loss(pred_sdf, target_sdf)


def loss_eikonal_structured(model, p: torch.Tensor, 
                            condition: torch.Tensor,
                            z: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Eikonal loss for structured model.
    
    Even though the analytic SDF should already satisfy |∇f|=1,
    this can help with smooth blending regions.
    
    Args:
        model: StructuredLegoModel
        p: (B, N, 3) query points (requires_grad=True)
        condition: (B, condition_dim)
        z: optional latent
    
    Returns:
        scalar loss
    """
    p = p.requires_grad_(True)
    sdf = model(p, condition, z)
    
    grad = torch.autograd.grad(
        outputs=sdf,
        inputs=p,
        grad_outputs=torch.ones_like(sdf),
        create_graph=True,
        retain_graph=True
    )[0]
    
    grad_norm = torch.norm(grad, dim=-1)
    return F.mse_loss(grad_norm, torch.ones_like(grad_norm))


def loss_interpolation_validity(model, 
                                condition_1: torch.Tensor,
                                condition_2: torch.Tensor,
                                p: torch.Tensor,
                                n_interp: int = 5) -> torch.Tensor:
    """
    Encourage valid shapes along interpolation path.
    
    For structured decoder, this should be nearly free (params interpolate).
    This loss is mainly for diagnostics/debugging.
    
    Args:
        model: StructuredLegoModel
        condition_1, condition_2: endpoint conditions (B, condition_dim)
        p: (B, N, 3) query points
        n_interp: number of interpolation steps
    
    Returns:
        scalar loss (variance of SDF values along path — should be smooth)
    """
    alphas = torch.linspace(0, 1, n_interp, device=condition_1.device)
    
    sdfs = []
    for alpha in alphas:
        cond_interp = (1 - alpha) * condition_1 + alpha * condition_2
        sdf = model(p, cond_interp)
        sdfs.append(sdf)
    
    sdfs = torch.stack(sdfs, dim=0)  # (n_interp, B, N)
    
    # Check smoothness: SDF should change smoothly along interpolation
    # Variance of differences should be low
    diffs = sdfs[1:] - sdfs[:-1]  # (n_interp-1, B, N)
    
    # Penalize non-uniform changes (jerky interpolation)
    diff_variance = diffs.var(dim=0).mean()
    
    return diff_variance
