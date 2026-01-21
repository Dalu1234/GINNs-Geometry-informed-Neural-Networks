"""
Connectivity Constraint via Morse Theory

This module implements losses to ensure generated shapes form a single connected body
using Morse theory concepts.

**Key Insight**: For a signed distance function f(x), the topology of the level set {f=0} 
is determined by critical points where ∇f = 0:
- Minima (index 0): Create new components
- Saddle-1 (index 1): Connect/disconnect components  
- Saddle-2 (index 2): Create handles/tunnels
- Maxima (index 3): Create voids

**Strategy**: Find index-1 saddle points and penalize those with f(x) ≠ 0.
If all saddles lie ON the surface, the shape is more likely to be connected.

**Efficiency Note**: This is computationally expensive. Use caching and sparse sampling.
"""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple, Callable
from tqdm import trange


# =============================================================================
# Gradient and Hessian Computation
# =============================================================================

def compute_gradient(f: Callable, x: torch.Tensor, create_graph: bool = True) -> torch.Tensor:
    """
    Compute gradient of scalar function f with respect to input x.
    
    Args:
        f: Callable that takes x [N, D] and returns [N, 1] or [N]
        x: Input points [N, D]
        create_graph: Whether to create graph for higher-order derivatives
        
    Returns:
        grad: [N, D] gradient vectors
    """
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
    
    return grad


def compute_hessian(f: Callable, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute gradient and Hessian of scalar function f at points x.
    
    Args:
        f: Callable that takes x [N, D] and returns [N, 1] or [N]
        x: Input points [N, D]
        
    Returns:
        grad: [N, D] gradient vectors
        hessian: [N, D, D] Hessian matrices
    """
    x = x.clone().requires_grad_(True)
    D = x.shape[1]
    
    y = f(x)
    if y.dim() > 1:
        y = y.squeeze(-1)
    
    # First derivatives (gradient)
    grad = torch.autograd.grad(
        outputs=y,
        inputs=x,
        grad_outputs=torch.ones_like(y),
        create_graph=True,
        retain_graph=True
    )[0]
    
    # Second derivatives (Hessian) - compute row by row
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
    
    hessian = torch.stack(hessian_rows, dim=1)  # [N, D, D]
    
    return grad, hessian


# =============================================================================
# Critical Point Detection
# =============================================================================

def find_critical_points_newton(
    f: Callable,
    bounds: torch.Tensor,
    n_starting_points: int = 1000,
    n_iter: int = 50,
    lr: float = 0.1,
    grad_tol: float = 1e-3,
    regularization: float = 1e-3,
    verbose: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Find critical points where ∇f = 0 using Newton's method on the gradient.
    
    Newton update: x <- x - H^{-1} @ ∇f
    Regularized: H_reg = H + λI to avoid singularity.
    
    Args:
        f: Callable taking x [N, D] returning [N, 1] or [N]
        bounds: [D, 2] tensor with (min, max) for each dimension
        n_starting_points: Number of random starting points
        n_iter: Maximum Newton iterations
        lr: Learning rate (damping factor for Newton step)
        grad_tol: Gradient norm threshold for convergence
        regularization: Regularization strength for Hessian
        verbose: Whether to print progress
        
    Returns:
        critical_pts: [M, D] converged critical points
        grad_norms: [M] gradient norm at each point
        f_values: [M] function value at each critical point
    """
    device = bounds.device
    D = bounds.shape[0]
    
    # Initialize random starting points within bounds
    x = torch.rand(n_starting_points, D, device=device)
    x = x * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]
    
    iterator = trange(n_iter, disable=not verbose)
    for i in iterator:
        # Compute gradient and Hessian
        grad, hessian = compute_hessian(f, x)
        
        # Regularize Hessian for numerical stability
        eye = torch.eye(D, device=device).unsqueeze(0).expand(len(x), -1, -1)
        hessian_reg = hessian + regularization * eye
        
        # Newton step: solve H @ delta = grad
        try:
            delta = torch.linalg.solve(hessian_reg, grad.unsqueeze(-1)).squeeze(-1)
        except:
            # Fallback to pseudoinverse if solve fails
            delta = torch.bmm(torch.linalg.pinv(hessian_reg), grad.unsqueeze(-1)).squeeze(-1)
        
        # Detach for update (no gradient through Newton iteration itself)
        delta = delta.detach()
        x = x.detach()
        
        # Damped Newton update
        x = x - lr * delta
        
        # Clamp to bounds
        x = torch.clamp(x, bounds[:, 0], bounds[:, 1])
        
        # Progress tracking
        if verbose:
            grad_norm = grad.detach().norm(dim=1)
            iterator.set_description(f"|∇f| mean={grad_norm.mean():.2e}, min={grad_norm.min():.2e}")
    
    # Final gradient computation (no graph needed)
    grad_final = compute_gradient(f, x, create_graph=False)
    grad_norms = grad_final.norm(dim=1)
    
    with torch.no_grad():
        f_values = f(x)
        if f_values.dim() > 1:
            f_values = f_values.squeeze(-1)
    
    # Filter to converged points
    converged_mask = grad_norms < grad_tol
    
    return x[converged_mask], grad_norms[converged_mask], f_values[converged_mask]


# =============================================================================
# Critical Point Classification
# =============================================================================

def classify_critical_points(
    f: Callable,
    critical_pts: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Classify critical points by Morse index (number of negative Hessian eigenvalues).
    
    Args:
        f: Callable for computing Hessian
        critical_pts: [N, D] critical point locations
        
    Returns:
        morse_index: [N] Morse index (0=min, 1=saddle1, 2=saddle2, 3=max for D=3)
        eigenvalues: [N, D] Hessian eigenvalues at each point
    """
    if len(critical_pts) == 0:
        return torch.zeros(0, dtype=torch.long), torch.zeros(0, critical_pts.shape[1])
    
    _, hessian = compute_hessian(f, critical_pts)
    
    # Compute eigenvalues (sorted ascending)
    eigenvalues = torch.linalg.eigvalsh(hessian)  # [N, D]
    
    # Morse index = number of negative eigenvalues
    morse_index = (eigenvalues < 0).sum(dim=1)
    
    return morse_index, eigenvalues


# =============================================================================
# Connectivity Loss Functions
# =============================================================================

def connectivity_loss_saddle(
    f: Callable,
    bounds: torch.Tensor,
    n_samples: int = 1000,
    n_iter: int = 30,
    grad_tol: float = 1e-2,
    regularization: float = 1e-3,
    saddle_index: int = 1,
    verbose: bool = False,
) -> Tuple[torch.Tensor, dict]:
    """
    Connectivity loss based on penalizing index-1 saddle points with f ≠ 0.
    
    For a connected level set {f = 0}, index-1 saddles should lie ON the surface.
    If f(saddle) ≠ 0, there's a potential disconnection.
    
    Args:
        f: Callable SDF function
        bounds: [D, 2] bounding box
        n_samples: Starting points for critical point search
        n_iter: Newton iterations
        grad_tol: Convergence tolerance
        regularization: Hessian regularization
        saddle_index: Which saddle type to penalize (1 for connectivity)
        verbose: Print progress
        
    Returns:
        loss: Scalar connectivity loss
        info: Dict with diagnostic information
    """
    device = bounds.device
    
    # Find critical points
    critical_pts, grad_norms, f_values = find_critical_points_newton(
        f, bounds,
        n_starting_points=n_samples,
        n_iter=n_iter,
        grad_tol=grad_tol,
        regularization=regularization,
        verbose=verbose,
    )
    
    info = {
        'n_critical_pts': len(critical_pts),
        'n_saddles': 0,
        'saddle_f_values': None,
    }
    
    if len(critical_pts) == 0:
        return torch.tensor(0.0, device=device), info
    
    # Classify by Morse index
    morse_index, eigenvalues = classify_critical_points(f, critical_pts)
    
    # Get target saddles (index-1 by default)
    saddle_mask = morse_index == saddle_index
    info['n_saddles'] = saddle_mask.sum().item()
    
    if saddle_mask.sum() == 0:
        return torch.tensor(0.0, device=device), info
    
    # Recompute f with gradient tracking for backprop
    saddle_pts = critical_pts[saddle_mask]
    f_at_saddles = f(saddle_pts)
    if f_at_saddles.dim() > 1:
        f_at_saddles = f_at_saddles.squeeze(-1)
    
    info['saddle_f_values'] = f_at_saddles.detach()
    
    # Loss: penalize f(saddle) ≠ 0
    loss = f_at_saddles.square().mean()
    
    return loss, info


def connectivity_loss_component_count(
    f: Callable,
    bounds: torch.Tensor,
    resolution: int = 64,
    level: float = 0.0,
    target_components: int = 1,
) -> Tuple[torch.Tensor, int]:
    """
    Connectivity metric based on discrete connected component count.
    
    NOTE: This is NOT differentiable! Use for evaluation only.
    
    Args:
        f: Callable SDF function
        bounds: [D, 2] bounding box
        resolution: Grid resolution per dimension
        level: Level set value (usually 0)
        target_components: Desired number of components
        
    Returns:
        metric: |n_components - target| (non-differentiable)
        n_components: Actual component count
    """
    from scipy import ndimage
    
    device = bounds.device
    D = bounds.shape[0]
    
    # Create grid
    grids = [torch.linspace(bounds[i, 0], bounds[i, 1], resolution, device=device) for i in range(D)]
    mesh = torch.meshgrid(*grids, indexing='ij')
    pts = torch.stack([g.flatten() for g in mesh], dim=1)
    
    with torch.no_grad():
        values = f(pts)
        if values.dim() > 1:
            values = values.squeeze(-1)
    
    # Reshape to grid
    shape = [resolution] * D
    values_grid = values.reshape(*shape)
    
    # Binary interior mask
    interior = (values_grid < level).cpu().numpy()
    
    # Count connected components
    _, n_components = ndimage.label(interior)
    
    metric = torch.tensor(abs(n_components - target_components), device=device, dtype=torch.float32)
    
    return metric, n_components


# =============================================================================
# Efficient Cached Connectivity Loss
# =============================================================================

class CachedConnectivityLoss:
    """
    Connectivity loss with caching for efficiency.
    
    Critical point detection is expensive, so we:
    1. Cache critical points across epochs
    2. Update cache periodically (every N epochs)
    3. Only recompute f-values at cached locations
    
    This reduces cost from O(expensive) to O(forward pass) most epochs.
    """
    
    def __init__(
        self,
        bounds: torch.Tensor,
        n_samples: int = 1000,
        n_iter: int = 30,
        grad_tol: float = 1e-2,
        update_every_n_epochs: int = 50,
        saddle_index: int = 1,
    ):
        self.bounds = bounds
        self.n_samples = n_samples
        self.n_iter = n_iter
        self.grad_tol = grad_tol
        self.update_every_n_epochs = update_every_n_epochs
        self.saddle_index = saddle_index
        
        # Cache
        self.cached_saddle_pts = None
        self.last_update_epoch = -1
    
    def __call__(
        self,
        f: Callable,
        epoch: int,
        force_update: bool = False,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute connectivity loss with caching.
        
        Args:
            f: Callable SDF function
            epoch: Current training epoch
            force_update: Force recomputation of critical points
            
        Returns:
            loss: Connectivity loss
            info: Diagnostic dict
        """
        device = self.bounds.device
        
        # Decide whether to update cache
        should_update = (
            force_update or
            self.cached_saddle_pts is None or
            (epoch - self.last_update_epoch) >= self.update_every_n_epochs
        )
        
        if should_update:
            # Full critical point search
            loss, info = connectivity_loss_saddle(
                f, self.bounds,
                n_samples=self.n_samples,
                n_iter=self.n_iter,
                grad_tol=self.grad_tol,
                saddle_index=self.saddle_index,
            )
            
            # Cache saddle locations
            if info['n_saddles'] > 0:
                # Find saddle points again for caching
                critical_pts, _, _ = find_critical_points_newton(
                    f, self.bounds,
                    n_starting_points=self.n_samples,
                    n_iter=self.n_iter,
                    grad_tol=self.grad_tol,
                )
                if len(critical_pts) > 0:
                    morse_index, _ = classify_critical_points(f, critical_pts)
                    self.cached_saddle_pts = critical_pts[morse_index == self.saddle_index].detach()
            else:
                self.cached_saddle_pts = None
            
            self.last_update_epoch = epoch
            info['cache_updated'] = True
            
        else:
            # Use cached saddle locations, just recompute f-values
            info = {
                'n_critical_pts': 0,
                'n_saddles': 0 if self.cached_saddle_pts is None else len(self.cached_saddle_pts),
                'saddle_f_values': None,
                'cache_updated': False,
            }
            
            if self.cached_saddle_pts is None or len(self.cached_saddle_pts) == 0:
                loss = torch.tensor(0.0, device=device)
            else:
                f_at_saddles = f(self.cached_saddle_pts)
                if f_at_saddles.dim() > 1:
                    f_at_saddles = f_at_saddles.squeeze(-1)
                info['saddle_f_values'] = f_at_saddles.detach()
                loss = f_at_saddles.square().mean()
        
        return loss, info


# =============================================================================
# Alternative: Soft Connectivity via Min-Cut Penalty
# =============================================================================

def soft_connectivity_loss(
    sdf_values: torch.Tensor,
    spatial_coords: torch.Tensor,
    temperature: float = 10.0,
) -> torch.Tensor:
    """
    Soft differentiable connectivity proxy.
    
    Idea: Penalize "thin necks" in the shape where SDF values are close to 0
    over a large region, indicating potential disconnection.
    
    This is a cheap approximation - not true connectivity but helps.
    
    Args:
        sdf_values: [N] SDF values at grid points
        spatial_coords: [N, D] spatial coordinates
        temperature: Sharpness of the surface indicator
        
    Returns:
        loss: Penalty for thin connections
    """
    # Soft indicator of being near surface
    near_surface = torch.sigmoid(-temperature * sdf_values.abs())
    
    # Penalty: high when many points are near surface (thin necks)
    # Normalized by total "interior" volume
    interior = torch.sigmoid(-temperature * sdf_values)
    
    surface_ratio = near_surface.sum() / (interior.sum() + 1e-8)
    
    # We want low surface-to-volume ratio for connected shapes
    # Disconnected shapes have high surface-to-volume
    return surface_ratio
