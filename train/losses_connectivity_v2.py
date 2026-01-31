"""
Phase 3.5: Improved Connectivity Constraint

Improvements over v1:
1. Filter saddles to near-surface ones (only penalize |f| < threshold)
2. Clamp Newton steps to prevent wild jumps
3. Better saddle targeting (attract to surface vs penalize being off)
4. Eikonal regularization to keep f well-behaved as SDF

The key insight from Phase 3 testing:
- Component count went from 6.9 → 83.9 (worse!)
- This means the connectivity loss signal was either too weak or misdirected
- We need to: (a) strengthen signal, (b) filter to relevant saddles
"""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple, Callable, List
from tqdm import trange


# =============================================================================
# Improved Gradient/Hessian with Stability
# =============================================================================

def compute_gradient_stable(f: Callable, x: torch.Tensor, create_graph: bool = True) -> torch.Tensor:
    """Compute gradient with NaN protection."""
    x = x.clone().requires_grad_(True)
    y = f(x)
    if y.dim() > 1:
        y = y.squeeze(-1)
    
    grad = torch.autograd.grad(
        outputs=y,
        inputs=x,
        grad_outputs=torch.ones_like(y),
        create_graph=create_graph,
        retain_graph=True
    )[0]
    
    # Replace NaNs with zeros
    grad = torch.where(torch.isnan(grad), torch.zeros_like(grad), grad)
    
    return grad


def compute_hessian_stable(f: Callable, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute gradient and Hessian with NaN protection."""
    x = x.clone().requires_grad_(True)
    D = x.shape[1]
    
    y = f(x)
    if y.dim() > 1:
        y = y.squeeze(-1)
    
    grad = torch.autograd.grad(
        outputs=y,
        inputs=x,
        grad_outputs=torch.ones_like(y),
        create_graph=True,
        retain_graph=True
    )[0]
    
    hessian_rows = []
    for i in range(D):
        grad_i = grad[:, i]
        hess_row = torch.autograd.grad(
            outputs=grad_i,
            inputs=x,
            grad_outputs=torch.ones_like(grad_i),
            create_graph=True,
            retain_graph=True
        )[0]
        hessian_rows.append(hess_row)
    
    hessian = torch.stack(hessian_rows, dim=1)
    
    # Replace NaNs
    grad = torch.where(torch.isnan(grad), torch.zeros_like(grad), grad)
    hessian = torch.where(torch.isnan(hessian), torch.zeros_like(hessian), hessian)
    
    return grad, hessian


# =============================================================================
# Improved Critical Point Detection
# =============================================================================

def find_critical_points_newton_v2(
    f: Callable,
    bounds: torch.Tensor,
    n_starting_points: int = 1000,
    n_iter: int = 50,
    lr: float = 0.1,
    grad_tol: float = 1e-3,
    regularization: float = 1e-2,  # Increased regularization
    max_step_size: float = 0.5,     # NEW: Clamp Newton steps
    near_surface_threshold: float = 0.1,  # NEW: Filter to near-surface
    verbose: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Find critical points with improved stability.
    
    NEW features:
    - max_step_size: Clamp Newton steps to prevent wild jumps
    - near_surface_threshold: Only return critical points with |f| < threshold
    - Better regularization
    
    Returns:
        critical_pts: [M, D] converged critical points near surface
        grad_norms: [M] gradient norm at each point
        f_values: [M] function value at each critical point
        all_critical_pts: [K, D] ALL critical points (before surface filtering)
    """
    device = bounds.device
    D = bounds.shape[0]
    
    # Initialize random starting points
    x = torch.rand(n_starting_points, D, device=device)
    x = x * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]
    
    # Also add some points biased toward where surfaces are likely
    # (near middle of bounds, since shapes tend to be there)
    center = (bounds[:, 0] + bounds[:, 1]) / 2
    spread = (bounds[:, 1] - bounds[:, 0]) / 4
    n_center_pts = n_starting_points // 4
    x_center = center + torch.randn(n_center_pts, D, device=device) * spread
    x_center = torch.clamp(x_center, bounds[:, 0], bounds[:, 1])
    x = torch.cat([x, x_center], dim=0)
    
    iterator = trange(n_iter, disable=not verbose)
    for i in iterator:
        try:
            grad, hessian = compute_hessian_stable(f, x)
        except RuntimeError:
            # Autograd can fail; just skip this iteration
            continue
        
        # Regularize Hessian
        eye = torch.eye(D, device=device).unsqueeze(0).expand(len(x), -1, -1)
        hessian_reg = hessian + regularization * eye
        
        # Newton step
        try:
            delta = torch.linalg.solve(hessian_reg, grad.unsqueeze(-1)).squeeze(-1)
        except:
            delta = torch.bmm(torch.linalg.pinv(hessian_reg), grad.unsqueeze(-1)).squeeze(-1)
        
        # NEW: Clamp step size
        step_norm = delta.norm(dim=1, keepdim=True)
        scale = torch.clamp(max_step_size / (step_norm + 1e-8), max=1.0)
        delta = delta * scale
        
        delta = delta.detach()
        x = x.detach()
        
        # Damped update
        x = x - lr * delta
        
        # Clamp to bounds
        x = torch.clamp(x, bounds[:, 0], bounds[:, 1])
        
        if verbose:
            grad_norm = grad.detach().norm(dim=1)
            iterator.set_description(f"|∇f| mean={grad_norm.mean():.2e}")
    
    # Final evaluation
    grad_final = compute_gradient_stable(f, x, create_graph=False)
    grad_norms = grad_final.norm(dim=1)
    
    with torch.no_grad():
        f_values = f(x)
        if f_values.dim() > 1:
            f_values = f_values.squeeze(-1)
    
    # Filter to converged points
    converged_mask = grad_norms < grad_tol
    all_critical_pts = x[converged_mask]
    all_grad_norms = grad_norms[converged_mask]
    all_f_values = f_values[converged_mask]
    
    # NEW: Filter to near-surface only
    near_surface_mask = all_f_values.abs() < near_surface_threshold
    
    return (
        all_critical_pts[near_surface_mask],
        all_grad_norms[near_surface_mask],
        all_f_values[near_surface_mask],
        all_critical_pts,  # Return all for diagnostics
    )


def classify_critical_points(
    f: Callable,
    critical_pts: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Classify critical points by Morse index."""
    if len(critical_pts) == 0:
        return torch.zeros(0, dtype=torch.long), torch.zeros(0, critical_pts.shape[1])
    
    _, hessian = compute_hessian_stable(f, critical_pts)
    eigenvalues = torch.linalg.eigvalsh(hessian)
    morse_index = (eigenvalues < 0).sum(dim=1)
    
    return morse_index, eigenvalues


# =============================================================================
# Improved Connectivity Loss
# =============================================================================

def connectivity_loss_saddle_v2(
    f: Callable,
    bounds: torch.Tensor,
    n_samples: int = 1000,
    n_iter: int = 30,
    grad_tol: float = 1e-2,
    near_surface_threshold: float = 0.2,  # Only consider saddles with |f| < this
    max_step_size: float = 0.3,
    saddle_index: int = 1,
    loss_type: str = 'attract',  # 'attract' or 'penalize'
    verbose: bool = False,
) -> Tuple[torch.Tensor, dict]:
    """
    Improved connectivity loss.
    
    NEW: loss_type options:
    - 'penalize': L = f(saddle)^2 (original - push saddles to surface)
    - 'attract': L = (|f(saddle)| - 0)^2 same but only for near-surface
    - 'hinge': L = max(0, |f(saddle)| - threshold)^2 (only penalize if too far)
    
    The key improvement is near_surface_threshold:
    - Only penalize saddles that are "close enough" to matter
    - Saddles deep inside or outside don't affect topology
    """
    device = bounds.device
    
    # Find critical points (already filtered to near-surface)
    near_critical_pts, grad_norms, f_values, all_critical_pts = find_critical_points_newton_v2(
        f, bounds,
        n_starting_points=n_samples,
        n_iter=n_iter,
        grad_tol=grad_tol,
        near_surface_threshold=near_surface_threshold,
        max_step_size=max_step_size,
        verbose=verbose,
    )
    
    info = {
        'n_critical_pts_total': len(all_critical_pts),
        'n_critical_pts_near_surface': len(near_critical_pts),
        'n_saddles': 0,
        'saddle_f_values': None,
        'saddle_pts': None,
    }
    
    if len(near_critical_pts) == 0:
        return torch.tensor(0.0, device=device, requires_grad=True), info
    
    # Classify only near-surface critical points
    morse_index, _ = classify_critical_points(f, near_critical_pts)
    saddle_mask = morse_index == saddle_index
    info['n_saddles'] = saddle_mask.sum().item()
    
    if saddle_mask.sum() == 0:
        return torch.tensor(0.0, device=device, requires_grad=True), info
    
    # Get saddle points
    saddle_pts = near_critical_pts[saddle_mask]
    info['saddle_pts'] = saddle_pts.detach()
    
    # Recompute f with gradient tracking
    f_at_saddles = f(saddle_pts)
    if f_at_saddles.dim() > 1:
        f_at_saddles = f_at_saddles.squeeze(-1)
    
    info['saddle_f_values'] = f_at_saddles.detach()
    
    # Compute loss based on type
    if loss_type == 'penalize':
        # Original: penalize any f(saddle) ≠ 0
        loss = f_at_saddles.square().mean()
    elif loss_type == 'attract':
        # Same as penalize but we already filtered to near-surface
        loss = f_at_saddles.square().mean()
    elif loss_type == 'hinge':
        # Only penalize if |f| > small threshold
        margin = 0.01
        loss = F.relu(f_at_saddles.abs() - margin).square().mean()
    else:
        loss = f_at_saddles.square().mean()
    
    return loss, info


# =============================================================================
# Eikonal Loss (keep f well-behaved as SDF)
# =============================================================================

def eikonal_loss(
    f: Callable,
    pts: torch.Tensor,
    target_grad_norm: float = 1.0,
) -> torch.Tensor:
    """
    Eikonal loss: ||∇f|| should be close to 1 for a proper SDF.
    
    This helps the connectivity constraint work better because:
    1. f-values become meaningful distances
    2. Critical points are more stable
    3. Newton iteration behaves better
    
    Args:
        f: SDF function
        pts: [N, D] sample points
        target_grad_norm: Target gradient magnitude (1 for SDF)
        
    Returns:
        loss: Mean squared deviation from target norm
    """
    grad = compute_gradient_stable(f, pts, create_graph=True)
    grad_norm = grad.norm(dim=1)
    loss = (grad_norm - target_grad_norm).square().mean()
    return loss


# =============================================================================
# Improved Cached Connectivity Loss
# =============================================================================

class CachedConnectivityLossV2:
    """
    Improved cached connectivity loss with Phase 3.5 features.
    
    Key improvements:
    1. Near-surface filtering
    2. Clamped Newton steps
    3. Optional eikonal regularization
    4. Better diagnostics
    """
    
    def __init__(
        self,
        bounds: torch.Tensor,
        n_samples: int = 1000,
        n_iter: int = 30,
        grad_tol: float = 1e-2,
        update_every_n_epochs: int = 50,
        saddle_index: int = 1,
        near_surface_threshold: float = 0.2,
        max_step_size: float = 0.3,
        loss_type: str = 'attract',
    ):
        self.bounds = bounds
        self.n_samples = n_samples
        self.n_iter = n_iter
        self.grad_tol = grad_tol
        self.update_every_n_epochs = update_every_n_epochs
        self.saddle_index = saddle_index
        self.near_surface_threshold = near_surface_threshold
        self.max_step_size = max_step_size
        self.loss_type = loss_type
        
        # Cache
        self.cached_saddle_pts = None
        self.last_update_epoch = -1
        
        # History for diagnostics
        self.history = {
            'n_saddles': [],
            'mean_f_saddle': [],
            'update_epochs': [],
        }
    
    def __call__(
        self,
        f: Callable,
        epoch: int,
        force_update: bool = False,
    ) -> Tuple[torch.Tensor, dict]:
        device = self.bounds.device
        
        should_update = (
            force_update or
            self.cached_saddle_pts is None or
            (epoch - self.last_update_epoch) >= self.update_every_n_epochs
        )
        
        if should_update:
            # Full search
            loss, info = connectivity_loss_saddle_v2(
                f, self.bounds,
                n_samples=self.n_samples,
                n_iter=self.n_iter,
                grad_tol=self.grad_tol,
                near_surface_threshold=self.near_surface_threshold,
                max_step_size=self.max_step_size,
                saddle_index=self.saddle_index,
                loss_type=self.loss_type,
            )
            
            # Cache saddle locations
            if info.get('saddle_pts') is not None and len(info['saddle_pts']) > 0:
                self.cached_saddle_pts = info['saddle_pts'].detach().clone()
            else:
                self.cached_saddle_pts = None
            
            self.last_update_epoch = epoch
            info['cache_updated'] = True
            
            # Record history
            self.history['n_saddles'].append(info['n_saddles'])
            self.history['update_epochs'].append(epoch)
            if info['saddle_f_values'] is not None:
                self.history['mean_f_saddle'].append(info['saddle_f_values'].abs().mean().item())
            
        else:
            # Use cached points
            info = {
                'n_saddles': 0 if self.cached_saddle_pts is None else len(self.cached_saddle_pts),
                'saddle_f_values': None,
                'cache_updated': False,
            }
            
            if self.cached_saddle_pts is None or len(self.cached_saddle_pts) == 0:
                loss = torch.tensor(0.0, device=device, requires_grad=True)
            else:
                f_at_saddles = f(self.cached_saddle_pts)
                if f_at_saddles.dim() > 1:
                    f_at_saddles = f_at_saddles.squeeze(-1)
                info['saddle_f_values'] = f_at_saddles.detach()
                
                if self.loss_type == 'hinge':
                    margin = 0.01
                    loss = F.relu(f_at_saddles.abs() - margin).square().mean()
                else:
                    loss = f_at_saddles.square().mean()
        
        return loss, info
    
    def get_diagnostics(self) -> dict:
        """Return diagnostic history."""
        return self.history.copy()


# =============================================================================
# Combined Loss for Phase 3.5
# =============================================================================

def phase35_connectivity_loss(
    f: Callable,
    bounds: torch.Tensor,
    pts_for_eikonal: torch.Tensor,
    cached_loss: CachedConnectivityLossV2,
    epoch: int,
    lambda_connectivity: float = 0.01,
    lambda_eikonal: float = 0.1,
) -> Tuple[torch.Tensor, dict]:
    """
    Combined Phase 3.5 loss: connectivity + eikonal.
    
    The eikonal term helps keep f well-behaved as an SDF,
    which makes the connectivity constraint more effective.
    """
    # Connectivity loss (cached)
    loss_conn, info_conn = cached_loss(f, epoch)
    
    # Eikonal loss (cheap, every step)
    loss_eik = eikonal_loss(f, pts_for_eikonal)
    
    # Combined
    total = lambda_connectivity * loss_conn + lambda_eikonal * loss_eik
    
    info = {
        **info_conn,
        'loss_connectivity': loss_conn.item() if isinstance(loss_conn, torch.Tensor) else loss_conn,
        'loss_eikonal': loss_eik.item(),
        'loss_total': total.item(),
    }
    
    return total, info
