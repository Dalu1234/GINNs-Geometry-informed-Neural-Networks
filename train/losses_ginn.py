import time
from cycler import V
import einops
import torch
import typing

from torch._functorch.eager_transforms import jacrev
from torch import vmap
from torch.nn import functional as F


from GINN.fenitop.heaviside_filter import heaviside
from models.net_w_partials import NetWithPartials
from models.point_wrapper import PointWrapper
from util.model_utils import tensor_product_xz, dist_to_edge_x_from_z
from train.losses import CD_dCDdy_loss, chamfer_diversity_loss, cuboid_rule_loss_sdf, diversity_loss, dirichlet_loss, envelope_loss_density, expression_curvature_loss, interface_loss, eikonal_loss, envelope_loss_sdf, l1_loss, mse_loss, normal_loss_euclidean, wasserstein_diversity_loss

Scalar: typing.TypeAlias = torch.Tensor #TODO: need to implement this more rigorously using torchtypeing https://github.com/patrick-kidger/torchtyping
Grad_Field: typing.TypeAlias = torch.Tensor


# =============================================================================
# EIKONAL LOSS
# Purpose: Make the network output a valid SDF (signed distance function)
# How: The gradient of a true SDF has magnitude 1 everywhere (||∇f|| = 1)
# Penalizes: (||∇f|| - 1)² at random points in the domain
# =============================================================================
def loss_eikonal(z, p_sampler, netp, scale_eikonal, **kwargs) -> Scalar:
    loss_eikonal = torch.tensor(0.0, device=z.device, dtype=z.dtype)
    xs_domain = p_sampler.sample_from_domain()
    ## Eikonal loss: NN should have gradient norm 1 everywhere
    x_tp, z_tp = tensor_product_xz(xs_domain, z)
    y_x_eikonal = netp.grouped_fwd('vf_x', x_tp, z_tp)
    loss_eikonal = eikonal_loss(y_x_eikonal)
    loss_eikonal = scale_eikonal * loss_eikonal
    return loss_eikonal


# =============================================================================
# OBSTACLE LOSS
# Purpose: Keep shape OUTSIDE of forbidden obstacle regions
# How: At points inside obstacles, push f > 0 (exterior)
# Penalizes: ReLU(-f) — only activates when f < 0 (incorrectly inside)
# =============================================================================
def loss_obst(z, netp, p_sampler, level_set, nf_is_density, **kwargs) -> Scalar:
    loss_obst = torch.tensor(0.0, device=z.device, dtype=z.dtype)
    ys_obst = netp(*tensor_product_xz(p_sampler.sample_from_obstacles(), z))
    if nf_is_density:
        loss_obst = envelope_loss_density(ys_obst, level_set=level_set)
    else:
        loss_obst = envelope_loss_sdf(ys_obst, level_set=level_set)
    return loss_obst



#DALU SAY GET RID OF 
# =============================================================================
# ENVELOPE LOSS
# Purpose: Keep shape INSIDE the bounding box / allowed region
# How: At points outside the envelope, push f > 0 (exterior)
# Penalizes: ReLU(-f) — only activates when f < 0 (shape leaks outside bounds)
# =============================================================================
def loss_env(z, netp, p_sampler, level_set, nf_is_density, **kwargs) -> Scalar:
    loss_env = torch.tensor(0.0, device=z.device, dtype=z.dtype)
    ys_env = netp(*tensor_product_xz(p_sampler.sample_from_envelope(), z)).squeeze(1)
    if nf_is_density:
        loss_env = envelope_loss_density(ys_env, level_set=level_set)
    else:
        loss_env = envelope_loss_sdf(ys_env)
    return loss_env

#DALU SAY GET RID OF 
# =============================================================================
# INTERFACE LOSS
# Purpose: Place the zero-level set at the target surface
# How: At points sampled from the target mesh surface, push f = 0
# Penalizes: MSE(f, 0) — surface should be exactly at these points
# Note: This is SHAPE-encoding, not RULE-encoding (memorizes pointcloud)
# =============================================================================
def loss_if(z, netp, p_sampler, level_set, **kwargs) -> Scalar:
    ys_BC = netp(*tensor_product_xz(p_sampler.sample_from_interface()[0], z)).squeeze(1)
    loss_if = interface_loss(ys_BC, level_set=level_set)
    return loss_if


# =============================================================================
# UNSEEN-N VIOLATION LOSS (GENERALIZATION)
# Purpose: Penalize poor constraint satisfaction on unseen N (e.g. 3, 6, 7) so the
#          model generalizes beyond training n_studs_values.
# How: For each unseen N, build z with conditioning for that N, compute interface +
#      envelope loss (same as violation metric), mean over unseen N and z samples.
# Note: Differentiable w.r.t. model parameters; use lambda_unseen_violation to enable.
# =============================================================================
def loss_unseen_violation(z, netp, unseen_problems, _build_z_for_n, n_z_samples, level_set, nf_is_density, **kwargs) -> Scalar:
    """
    Mean interface + envelope loss over unseen N (and over n_z_samples per N).
    If unseen_problems is empty, returns 0.0. Uses _build_z_for_n(N, n_samples, device) to get z.
    """
    if not unseen_problems:
        return torch.tensor(0.0, device=z.device, dtype=z.dtype)
    device = z.device
    total = torch.tensor(0.0, device=device, dtype=z.dtype)
    count = 0
    for N, problem in unseen_problems.items():
        z_n = _build_z_for_n(N, n_z_samples, device)
        for i in range(z_n.shape[0]):
            z_single = z_n[i : i + 1]
            l_if = loss_if(z=z_single, netp=netp, p_sampler=problem, level_set=level_set)
            l_env = loss_env(z=z_single, netp=netp, p_sampler=problem, level_set=level_set, nf_is_density=nf_is_density)
            total = total + l_if + l_env
            count += 1
    if count == 0:
        return torch.tensor(0.0, device=device, dtype=z.dtype)
    return total / count


# =============================================================================
# LATENT INTERPOLATION MIDPOINT LOSS
# Purpose: Encourage valid, smooth decoding at the midpoint between two brick types
#          (e.g. z_mid between 1x2 and 1x4 → 1x3-like). Different brick types only.
# How: Build z1, z2 for two different N; z_mid = (z1+z2)/2. Sample x in domain,
#      keep points near zero-level set of z_mid (|f - level_set| < threshold).
#      At those x: (A) eikonal (||∇_x f|| - 1)², (B) penalize ||∇_z f|| (smooth in z).
# Use: lambda_interp_midpoint > 0; requires condition_on_n_studs and ≥2 n_studs_values.
# =============================================================================
def loss_interp_midpoint(z, netp, p_sampler, _build_z_for_n, n_studs_values, level_set, nf_is_density,
                        interp_near_threshold, interp_n_domain_calls, interp_min_near_points,
                        scale_interp_eikonal, scale_interp_smooth_z, **kwargs) -> Scalar:
    """
    Eikonal + smooth-z at midpoint between two different brick-type latents.
    Samples x near zero-level set of f(·, z_mid), then applies eikonal and ∇_z penalty.
    """
    if not callable(_build_z_for_n) or not n_studs_values or len(n_studs_values) < 2:
        return torch.tensor(0.0, device=z.device, dtype=z.dtype)
    device = z.device
    dtype = z.dtype
    N1, N2 = n_studs_values[0], n_studs_values[-1]
    if N1 == N2:
        return torch.tensor(0.0, device=device, dtype=dtype)

    # Different brick types: build z1, z2 and midpoint
    z1 = _build_z_for_n(N1, 1, device)   # (1, nz)
    z2 = _build_z_for_n(N2, 1, device)   # (1, nz)
    z_mid = (z1 + z2) / 2.0              # (1, nz)

    # Sample domain points (multiple calls for more points)
    xs = []
    for _ in range(max(1, int(interp_n_domain_calls))):
        xs.append(p_sampler.sample_from_domain())
    x = torch.cat(xs, dim=0)   # (M, 3)

    # Forward at z_mid to find near zero-level set
    x_tp, z_tp = tensor_product_xz(x, z_mid)
    y = netp.grouped_no_grad_fwd('vf', x_tp, z_tp).squeeze(-1)   # (M,)
    near = (y - level_set).abs() < interp_near_threshold

    n_near = near.sum().item()
    if n_near < interp_min_near_points:
        x_use = x
        z_use = z_mid.expand(x.shape[0], -1)
    else:
        x_use = x[near]
        z_use = z_mid.expand(x_use.shape[0], -1)

    if x_use.shape[0] == 0:
        return torch.tensor(0.0, device=device, dtype=dtype)

    # (A) Eikonal at x_use, z_use
    y_x = netp.grouped_fwd('vf_x', x_use, z_use)
    loss_eik = eikonal_loss(y_x)

    # (B) Smooth w.r.t. z: penalize ||∇_z f||
    y_z = netp.vf_z(x_use, z_use).squeeze(-1)
    loss_z = y_z.square().mean()

    loss = scale_interp_eikonal * loss_eik + scale_interp_smooth_z * loss_z
    return loss


# =============================================================================
# CUBOID PRIMITIVE LOSS (RULE-BASED)
# Purpose: Teach the network what a box IS, not memorize specific surfaces
# How: 
#   1. Inside box → f < -margin (safely inside)
#   2. Near faces → f > +margin (safely outside)  
#   3. Corners → f > +margin (safely outside)
#   4. Far field → f > +margin (safely outside)
#   5. Boundary sharpening → f ≈ true_distance near faces
# Note: This IS rule-encoding — learns "box = negative inside, positive outside"
# =============================================================================
def loss_cuboid_primitive(z, netp, bounds=None, bounds_body=None, n_inside=2000, n_near_faces=1000,
                          n_corners=500, n_far=500, n_boundary=1000,
                          safety_margin=0.1, boundary_weight=0.5, **kwargs) -> Scalar:
    """
    Teach: A cuboid is a region where f < 0 inside a box, f > 0 outside.
    Uses stratified sampling and boundary sharpening for better learning.
    Bounds can be passed at call time (e.g. for conditional LEGO per-batch problem).
    For LEGO: pass bounds_body (body-only, z max = stud_z_bottom) so the cuboid
    does not sample inside the stud volume; stud_grid handles the studs.
    """
    # Prefer body-only bounds when given (LEGO) so cuboid does not intersect stud
    bounds = bounds_body if bounds_body is not None else (bounds if bounds is not None else kwargs.get('bounds'))
    if bounds is None:
        return torch.tensor(0.0, device=z.device, dtype=z.dtype)
    from GINN.problems.sampling_primitives import (
        sample_inside_box, sample_near_box_faces, sample_near_box_corners,
        sample_far_from_box, sample_on_box_faces_with_sdf
    )
    
    device = z.device
    dtype = z.dtype
    
    # =========================================================================
    # 1. INSIDE: should be f < -safety_margin (safely inside)
    # =========================================================================
    x_inside = sample_inside_box(bounds, n_inside, margin=0.0)
    x_in_tp, z_in_tp = tensor_product_xz(x_inside, z)
    f_inside = netp.grouped_fwd('vf', x_in_tp, z_in_tp).squeeze(-1)
    # Penalize when f > -safety_margin (should be < -margin, i.e. safely negative)
    loss_inside = torch.relu(f_inside + safety_margin).mean()
    
    # =========================================================================
    # 2. NEAR FACES: should be f > +safety_margin (safely outside)
    # =========================================================================
    x_near_faces = sample_near_box_faces(bounds, margin=0.2, n=n_near_faces)
    x_faces_tp, z_faces_tp = tensor_product_xz(x_near_faces, z)
    f_near_faces = netp.grouped_fwd('vf', x_faces_tp, z_faces_tp).squeeze(-1)
    # Penalize when f < +safety_margin (should be > +margin, i.e. safely positive)
    loss_near_faces = torch.relu(safety_margin - f_near_faces).mean()
    
    # =========================================================================
    # 3. CORNERS: should be f > +safety_margin (safely outside)
    # =========================================================================
    x_corners = sample_near_box_corners(bounds, margin=0.3, n=n_corners)
    x_corners_tp, z_corners_tp = tensor_product_xz(x_corners, z)
    f_corners = netp.grouped_fwd('vf', x_corners_tp, z_corners_tp).squeeze(-1)
    loss_corners = torch.relu(safety_margin - f_corners).mean()
    
    # =========================================================================
    # 4. FAR FIELD: should be f > +safety_margin (safely outside)
    # =========================================================================
    x_far = sample_far_from_box(bounds, margin_range=(1.0, 2.0), n=n_far)
    if x_far.shape[0] > 0:
        x_far_tp, z_far_tp = tensor_product_xz(x_far, z)
        f_far = netp.grouped_fwd('vf', x_far_tp, z_far_tp).squeeze(-1)
        loss_far = torch.relu(safety_margin - f_far).mean()
    else:
        loss_far = torch.tensor(0.0, device=device, dtype=dtype)
    
    # =========================================================================
    # 5. BOUNDARY SHARPENING: f ≈ true_distance near faces
    # This teaches the network that SDF equals actual distance to surface
    # =========================================================================
    x_boundary, true_sdf = sample_on_box_faces_with_sdf(bounds, n_boundary)
    x_bound_tp, z_bound_tp = tensor_product_xz(x_boundary, z)
    f_boundary = netp.grouped_fwd('vf', x_bound_tp, z_bound_tp).squeeze(-1)
    # Supervised loss: f should match true SDF
    loss_boundary = (f_boundary - true_sdf.repeat(z.shape[0])).pow(2).mean()
    
    # =========================================================================
    # COMBINE
    # =========================================================================
    loss_outside = loss_near_faces + loss_corners + loss_far
    total_loss = loss_inside + loss_outside + boundary_weight * loss_boundary
    
    return total_loss


# =============================================================================
# PARAMETRIC STUD CENTERS (rule-based, no precomputed point clouds)
# Matches ProblemLego1xN convention: brick_length = n_studs * spacing - inset_mm/scale
# =============================================================================
def _stud_centers_parametric(n_studs: int, n_studs_y: int, stud_spacing: float = 1.0,
                             lego_inset_norm: float = 0.025) -> typing.List[typing.Tuple[float, float]]:
    """
    Compute stud center (x, y) in normalized space from parameters only.
    Same rule as ProblemLego1xN: first stud at -half_length + spacing/2, then regular grid.
    lego_inset_norm = 0.2mm / 8mm = 0.025 so brick_length_norm = n_studs - 0.025.
    """
    half_length_x = (n_studs * stud_spacing - lego_inset_norm) * 0.5
    first_x = -half_length_x + stud_spacing * 0.5
    if n_studs_y is None or n_studs_y <= 1:
        return [(first_x + i * stud_spacing, 0.0) for i in range(n_studs)]
    half_length_y = (n_studs_y * stud_spacing - lego_inset_norm) * 0.5
    first_y = -half_length_y + stud_spacing * 0.5
    return [
        (first_x + i * stud_spacing, first_y + j * stud_spacing)
        for i in range(n_studs) for j in range(n_studs_y)
    ]


# =============================================================================
# CYLINDER PRIMITIVE LOSS (for studs: SDF < 0 inside each cylinder)
# Used by loss_stud_grid to teach "cylinder at every grid position" → generalizes to any N
# =============================================================================
def cylinder_primitive_loss(z, netp, center_xy, radius, z_bottom, z_top, n_samples=200, safety_margin=0.05,
                           n_outside=0, shell_thickness=0.05, **kwargs) -> Scalar:
    """
    Penalize SDF > -safety_margin inside the cylinder; optionally penalize SDF < safety_margin
    in a thin shell outside so the zero-level set sits at the cylinder boundary.
    """
    from GINN.problems.sampling_primitives import sample_inside_cylinder, sample_outside_cylinder_shell
    device = z.device
    dtype = z.dtype
    center_xy = torch.as_tensor(center_xy, device=device, dtype=dtype)
    if center_xy.dim() == 1 and center_xy.shape[0] == 2:
        pass
    else:
        center_xy = center_xy.reshape(2)
    x_inside = sample_inside_cylinder(center_xy, radius, z_bottom, z_top, n_samples, device=device)
    x_in_tp, z_in_tp = tensor_product_xz(x_inside, z)
    f_inside = netp.grouped_fwd('vf', x_in_tp, z_in_tp).squeeze(-1)
    loss = torch.relu(f_inside + safety_margin).mean()
    if n_outside > 0 and shell_thickness > 0:
        ro = float(radius) + float(shell_thickness)
        x_outside = sample_outside_cylinder_shell(
            center_xy, radius, ro, z_bottom, z_top, n_outside, device=device
        )
        x_out_tp, z_out_tp = tensor_product_xz(x_outside, z)
        f_outside = netp.grouped_fwd('vf', x_out_tp, z_out_tp).squeeze(-1)
        loss = loss + torch.relu(safety_margin - f_outside).mean()
    return loss


def loss_stud_grid(z, netp, problem=None, n_samples_per_stud=200, safety_margin=0.05, stud_spacing=1.0, **kwargs) -> Scalar:
    """
    Parametric stud grid loss: cylinder at every stud position.
    Stud centers are computed from (N, N_y, stud_spacing) at call time—no precomputed
    point clouds. Supports both problem-based calls (LEGO) and direct (n_studs, n_studs_y)
    for unseen N. Geometry (radius, z extents) from problem when present, else kwargs.
    """
    problem = problem if problem is not None else kwargs.get('problem')
    device = z.device
    dtype = z.dtype

    # Resolve N, N_y: from problem or explicit kwargs (for parametric / unseen N)
    n_studs = kwargs.get('n_studs')
    n_studs_y = kwargs.get('n_studs_y')
    if problem is not None:
        if getattr(problem, 'n_studs', None) is None and n_studs is None:
            return torch.tensor(0.0, device=device, dtype=dtype)
        if n_studs is None:
            n_studs = problem.n_studs
        if n_studs_y is None:
            n_studs_y = getattr(problem, 'n_studs_y', None) or 1
    if n_studs is None or n_studs < 1:
        return torch.tensor(0.0, device=device, dtype=dtype)
    if n_studs_y is None:
        n_studs_y = 1

    # Geometry: from problem when available, else kwargs (allows calling without problem)
    stud_radius = kwargs.get('stud_radius')
    stud_z_bottom = kwargs.get('stud_z_bottom')
    stud_z_top = kwargs.get('stud_z_top')
    if problem is not None and getattr(problem, 'stud_radius', None) is not None:
        if stud_radius is None:
            stud_radius = problem.stud_radius
        if stud_z_bottom is None:
            stud_z_bottom = problem.stud_z_bottom
        if stud_z_top is None:
            stud_z_top = problem.stud_z_top
    if stud_radius is None or stud_z_bottom is None or stud_z_top is None:
        return torch.tensor(0.0, device=device, dtype=dtype)
    stud_radius = float(stud_radius.item() if hasattr(stud_radius, 'item') else stud_radius)
    stud_z_bottom = float(stud_z_bottom.item() if hasattr(stud_z_bottom, 'item') else stud_z_bottom)
    stud_z_top = float(stud_z_top.item() if hasattr(stud_z_top, 'item') else stud_z_top)

    # Parametric rule: centers from N, N_y, spacing (matches ProblemLego1xN convention)
    lego_inset_norm = kwargs.get('lego_inset_norm', 0.025)
    centers_list = _stud_centers_parametric(
        n_studs, n_studs_y, stud_spacing, lego_inset_norm=lego_inset_norm
    )
    if not centers_list:
        return torch.tensor(0.0, device=device, dtype=dtype)

    n_outside = kwargs.get('n_outside_per_stud', 0)
    shell_thickness = kwargs.get('stud_shell_thickness', 0.05)
    loss_total = torch.tensor(0.0, device=device, dtype=dtype)
    for center_xy in centers_list:
        loss_stud = cylinder_primitive_loss(
            z, netp,
            center_xy=center_xy,
            radius=stud_radius,
            z_bottom=stud_z_bottom,
            z_top=stud_z_top,
            n_samples=n_samples_per_stud,
            safety_margin=safety_margin,
            n_outside=n_outside,
            shell_thickness=shell_thickness,
        )
        loss_total = loss_total + loss_stud
    return loss_total / len(centers_list)


def loss_phantom_stud(z, netp, problem=None, margin=0.1, n_phantom_per_side=2, phantom_offsets=None, **kwargs) -> Scalar:
    """
    Rule-enforcement (phantom-stud) loss: penalize negative SDF at positions that have
    the same tiled-coord pattern as studs but are outside the brick (e.g. just past the
    last stud). Encourages using dist_to_edge and the "stud only if inside" rule.
    Phantom points: same stud height, x just outside half_length (centered coords).
    """
    problem = problem if problem is not None else kwargs.get('problem')
    if problem is None:
        return torch.tensor(0.0, device=z.device, dtype=z.dtype)
    bounds = getattr(problem, 'bounds', None)
    if bounds is None or bounds.dim() < 2:
        return torch.tensor(0.0, device=z.device, dtype=z.dtype)
    device = z.device
    dtype = z.dtype
    stud_z_bottom = getattr(problem, 'stud_z_bottom', None)
    stud_z_top = getattr(problem, 'stud_z_top', None)
    if stud_z_bottom is None or stud_z_top is None:
        return torch.tensor(0.0, device=device, dtype=dtype)
    stud_z_bottom = float(stud_z_bottom.item() if hasattr(stud_z_bottom, 'item') else stud_z_bottom)
    stud_z_top = float(stud_z_top.item() if hasattr(stud_z_top, 'item') else stud_z_top)
    stud_z_mid = (stud_z_bottom + stud_z_top) * 0.5
    half_x = float(bounds[0, 1].item() if hasattr(bounds[0, 1], 'item') else bounds[0, 1])
    if phantom_offsets is not None:
        offsets = list(phantom_offsets)
    else:
        offsets = [0.5 + i * 1.0 for i in range(n_phantom_per_side)]
    # Phantom points: x = half_x + off, x = -half_x - off, y = 0, z = stud_z_mid (centered coords)
    y_mid = 0.0
    if bounds.shape[0] >= 2:
        y_mid = float((bounds[1, 0] + bounds[1, 1]).item() * 0.5) if hasattr(bounds[1, 0], 'item') else (bounds[1, 0] + bounds[1, 1]) * 0.5
    pts_list = []
    for off in offsets:
        pts_list.append([half_x + off, y_mid, stud_z_mid])
        pts_list.append([-half_x - off, y_mid, stud_z_mid])
    x_phantom = torch.tensor(pts_list, device=device, dtype=dtype)
    x_tp, z_tp = tensor_product_xz(x_phantom, z)
    sdf = netp.grouped_fwd('vf', x_tp, z_tp).squeeze(-1)
    loss = torch.relu(-sdf - margin).mean()
    return loss


def loss_outside_brick_sdf(z, netp, p_sampler, n_studs_values, n_norm_col=2, n_norm_col_end=None,
                           margin=0.1, n_samples=500, **kwargs) -> Scalar:
    """
    Stronger "outside brick" signal: at points where dist_to_edge < -margin, penalize f < 0
    (push SDF above a small positive value). Reinforces "outside brick → positive SDF" and
    reduces phantom studs. Reuses existing dist_to_edge_x_from_z.
    """
    if not n_studs_values or len(n_studs_values) == 0:
        return torch.tensor(0.0, device=z.device, dtype=z.dtype)
    sampler = kwargs.get('problem', p_sampler)
    xs = sampler.sample_from_envelope()
    if xs.shape[0] > n_samples:
        perm = torch.randperm(xs.shape[0], device=xs.device)[:n_samples]
        xs = xs[perm]
    x_tp, z_tp = tensor_product_xz(xs, z)
    dist = dist_to_edge_x_from_z(x_tp, z_tp, n_studs_values, n_norm_col, n_norm_col_end)
    sdf = netp.grouped_fwd('vf', x_tp, z_tp).squeeze(-1)
    outside = dist < -margin
    if outside.sum() == 0:
        return torch.tensor(0.0, device=z.device, dtype=z.dtype)
    # Push SDF above margin where clearly outside brick
    loss = torch.relu(margin - sdf[outside]).mean()
    return loss


# Keep old loss_cuboid_rule for backward compatibility (deprecated)
def loss_cuboid_rule(z, netp, p_sampler, level_set, nf_is_density, **kwargs) -> Scalar:
    """
    DEPRECATED: Use loss_cuboid_primitive instead.
    This version uses simple inside/outside sampling without stratification or boundary sharpening.
    """
    if nf_is_density:
        return torch.tensor(0.0, device=z.device, dtype=z.dtype)
    x_inside = p_sampler.sample_from_inside_envelope()
    x_outside = p_sampler.sample_from_envelope()
    x_in_tp, z_in_tp = tensor_product_xz(x_inside, z)
    x_out_tp, z_out_tp = tensor_product_xz(x_outside, z)
    ys_inside = netp.grouped_fwd('vf', x_in_tp, z_in_tp).squeeze(1)
    ys_outside = netp.grouped_fwd('vf', x_out_tp, z_out_tp).squeeze(1)
    return cuboid_rule_loss_sdf(ys_inside, ys_outside, level_set=level_set)


#DALU SAY GET RID OF 
# =============================================================================
# INTERFACE NORMAL LOSS
# Purpose: Make surface normals match target geometry
# How: At surface points, gradient ∇f should equal the mesh normal
# Penalizes: ||∇f - target_normal||² (Euclidean distance)
# Note: This is SHAPE-encoding — normals come from mesh, not from rules
# =============================================================================
def loss_if_normal(z, netp, p_sampler, nf_is_density, loss_scale, ginn_bsize=None, **kwargs) -> Scalar:
    pts_normal, target_normal = p_sampler.sample_from_interface()
    x_tp, z_tp = tensor_product_xz(pts_normal, z)
    ys_normal = netp.grouped_fwd('vf_x', x_tp, z_tp).squeeze(1)
    
    if nf_is_density:
        # if the neural field is a density field, make normal vectors unit vectors
        ys_normal = F.normalize(ys_normal, p=2, dim=-1)
    
    loss_if_normal = normal_loss_euclidean(ys_normal, target_normal=torch.cat([target_normal for _ in range(ginn_bsize)]))
    loss_if_normal = loss_scale * loss_if_normal
    return loss_if_normal


# =============================================================================
# LATENT STRUCTURE LOSS
# Purpose: Keep latent space healthy — not collapsed, not exploded, meaningful
# How: Three components:
#   1. Variance: z should have variance (prevent collapse to same point)
#   2. Magnitude: z should be bounded (prevent explosion)
#   3. Diversity: different z should give different SDFs (optional)
# Note: Cleaner alternative to separate prior/diversity losses
# =============================================================================
def loss_latent_structure(z, netp, p_sampler, 
                          min_variance=0.1, max_magnitude=3.0,
                          diversity_weight=1.0, n_diversity_pts=100,
                          **kwargs) -> Scalar:
    """
    Encourage latent space to have meaningful structure.
    
    NOT a VAE prior - just basic regularization to keep z healthy.
    
    Args:
        z: latent codes (B, nz) - includes z_base and conditioning cols
        netp: network with partials
        p_sampler: problem for getting bounds
        min_variance: minimum variance per z dimension (prevent collapse)
        max_magnitude: maximum |z| per dimension (prevent explosion)
        diversity_weight: weight for SDF diversity term
        n_diversity_pts: number of points for SDF comparison
    """
    device = z.device
    dtype = z.dtype
    B = z.shape[0]
    
    # =========================================================================
    # 1. VARIANCE: Prevent collapse — z should spread out, not cluster
    # =========================================================================
    # Penalize if variance is below threshold
    z_var = z.var(dim=0)  # Variance per dimension
    loss_variance = torch.relu(min_variance - z_var).mean()
    
    # =========================================================================
    # 2. MAGNITUDE: Prevent explosion — z should stay bounded
    # =========================================================================
    # Penalize if |z| exceeds threshold
    loss_magnitude = torch.relu(z.abs() - max_magnitude).mean()
    
    # =========================================================================
    # 3. DIVERSITY: Different z should produce different SDFs
    # =========================================================================
    loss_diversity = torch.tensor(0.0, device=device, dtype=dtype)
    
    if B > 1 and diversity_weight > 0:
        # Sample points in domain
        bounds = p_sampler.bounds
        x_sample = bounds[:, 0] + (bounds[:, 1] - bounds[:, 0]) * torch.rand(n_diversity_pts, 3, device=device)
        
        # Evaluate SDF at sample points for each z
        sdfs = []
        for i in range(B):
            z_i = z[i:i+1].expand(n_diversity_pts, -1)
            with torch.no_grad():
                sdf_i = netp.grouped_fwd('vf', x_sample, z_i).squeeze(-1)
            sdfs.append(sdf_i)
        
        sdfs = torch.stack(sdfs)  # [B, n_diversity_pts]
        
        # Different z should give different SDF values at same points
        # Variance across batch dimension at each point
        sdf_variance = sdfs.var(dim=0).mean()  # Average variance across points
        
        # Penalize if SDF variance is too low (all z give same shape)
        loss_diversity = torch.relu(0.01 - sdf_variance) * diversity_weight
    
    # =========================================================================
    # COMBINE
    # =========================================================================
    total_loss = loss_variance + loss_magnitude + loss_diversity
    
    return total_loss


# =============================================================================
# PRIOR LOSS
# Purpose: Learn a conditional distribution over latent codes p(z | c)
# How: Fit a small network to predict mean/std of z given conditioning c
# Penalizes: -log p(z_base | c) — negative log likelihood
# Use: Enables sampling z from learned prior at inference time
# =============================================================================
def loss_prior(z, conditional_prior, nz_base, c_dim, **kwargs) -> Scalar:
    """
    Negative log probability of z_base under the conditional prior p(z_base | c).
    Fits the prior to assign high density to the z_base we use (learns a latent prior per condition).
    """
    if z.shape[1] < nz_base + c_dim:
        return torch.tensor(0.0, device=z.device, dtype=z.dtype)
    z_base = z[:, :nz_base]
    c = z[:, -c_dim:].clamp(0.0, 1.0)
    log_p = conditional_prior.log_prob(z_base, c)
    return -log_p.mean()


# =============================================================================
# SCC LOSS (Single Connected Component)
# Purpose: Ensure shape is ONE piece, not fragmented blobs
# How: Uses persistent homology (PH) to count connected components
# Penalizes: Having more than 1 connected component (Betti_0 > 1)
# Result: Pushes network to output a single connected mesh
# =============================================================================
def loss_scc(z, ph_manager, **kwargs) -> Scalar:
    loss_scc = torch.tensor(0.0, device=z.device, dtype=z.dtype)
    loss_sub0 = torch.tensor(0.0, device=z.device, dtype=z.dtype)
    loss_super0 = torch.tensor(0.0, device=z.device, dtype=z.dtype)
    success, loss_scc, loss_super0, loss_sub0 = ph_manager.calc_ph_loss_cripser(z)
    return loss_scc


# =============================================================================
# DATA LOSS
# Purpose: Fit network to ground-truth SDF samples (supervised)
# How: For (x, y_gt) pairs from dataset, minimize MSE(f(x), y_gt)
# Use: Only when lambda_data > 0 and you have GT SDF samples
# Note: NOT used in pure GINN (constraint-only training)
# =============================================================================
def loss_data(z, netp, batch, z_corners, **kwargs) -> Scalar:
    loss_data = torch.tensor(0.0, device=z_corners.device, dtype=z_corners.dtype)
    x, y, idcs = batch
    # x, y, idcs = x.to(self.device), y.to(self.device), idcs.to(self.device)
    z_data = z_corners[idcs]
    y_pred = netp(x, z_data).squeeze(1)
    loss_data = mse_loss(y_pred, y)

    return loss_data


# =============================================================================
# DIRICHLET LOSS
# Purpose: Penalize how much the SDF changes with respect to z (latent)
# How: Compute ∂f/∂z and minimize its magnitude
# Effect: Different z values give similar shapes (reduces diversity)
# Use: Rarely used; conflicts with diversity loss
# =============================================================================
def loss_dirichlet(z, netp, p_surface, p_sampler, batch, **kwargs) -> Scalar:
    '''Uses surface points to enforce Dirichlet energy'''
    l_dirich =  torch.tensor(0.0, device=z.device, dtype=z.dtype)
    if p_surface is None:
        xs_domain = p_sampler.sample_from_domain()
        y_z = netp.vf_z(*tensor_product_xz(xs_domain, z)).squeeze(-1)
    else:
        y_z = netp.vf_z(p_surface.data, p_surface.z_in(z)).squeeze(1)
    l_dirich = dirichlet_loss(y_z)
    return l_dirich


# =============================================================================
# LIPSCHITZ LOSS
# Purpose: Regularize network to be Lipschitz continuous
# How: Penalize large weight norms or spectral norms
# Effect: Smoother SDF, more stable gradients
# Use: For Lipschitz-constrained architectures (lip_mlp, lip_siren)
# =============================================================================
def loss_lip(z, netp, **kwargs) -> Scalar:
    loss_lip = torch.tensor(0.0, device=z.device, dtype=z.dtype)
    loss_lip = netp.get_lipschitz_loss()

    return loss_lip


# =============================================================================
# VOLUME LOSS
# Purpose: Constrain total volume of shape to target fraction
# How: Integrate density field, penalize deviation from vol_frac
# Penalizes: (actual_vol / target_vol - 1)²
# Use: Topology optimization (TO) problems
# =============================================================================
def loss_vol2(rho_batch, vol_frac, nf_is_density, beta, **kwargs) -> Scalar:
    print(f'WARNING: vol2 heaviside is hardcoded to 128')
    rho_batch_vol = heaviside(rho_batch, beta, nf_is_density)
    vol_batch = rho_batch_vol.mean(dim=1)
    vol_loss_2 = (vol_batch / vol_frac - 1) ** 2
    vol_loss_2 = torch.clip(vol_loss_2, min=0.).mean()
    return vol_loss_2


# =============================================================================
# CHAMFER DIVERSITY LOSS
# Purpose: Encourage different z → different shapes
# How: Compute Chamfer distance between surface points of different shapes
# Penalizes: Small distance between shapes (pushes them apart)
# Use: Generative mode — want variety in output shapes
# =============================================================================
def loss_chamfer_div(z, netp: NetWithPartials, p_surface: PointWrapper, subsample, chamfer_p, max_div, chamfer_div_eps, loss_scale, **kwargs) -> Scalar:
    
    print(f'Total len of surface points: {len(p_surface)}')
    
    if subsample > 0:
        # subsample points from the surface
        list_of_pts = []
        for i_shape in range(p_surface.bz):
            # get at most 5000 points
            x_i = p_surface.pts_of_shape(i_shape)
            if len(x_i) > subsample:
                idx = torch.randperm(len(x_i))[:subsample]
                x_i = x_i[idx]
            list_of_pts.append(x_i)
        p_surface = PointWrapper.create_from_pts_per_shape_list(list_of_pts)
    
    y_x = netp.grouped_no_grad_fwd('vf_x', p_surface.data, p_surface.z_in(z)).squeeze(1)
    py_x = PointWrapper(data=y_x, map=p_surface._map) 

    start_t = time.time()
    CD, pdCDdy = chamfer_diversity_loss(p_surface, py_x, p=chamfer_p, eps=chamfer_div_eps)
    print(f'CD time: {time.time() - start_t}')
    
    if max_div is not None:
        print(f'CD before max_div: {CD}')
        CD = CD - max_div
        if CD < 0:
            return (torch.tensor(0.0, device=z.device, dtype=z.dtype),
                    p_surface,
                    torch.tensor(0.0, device=z.device, dtype=z.dtype),
                    torch.tensor(0.0, device=z.device, dtype=z.dtype))
    
    CD = CD * loss_scale
    pdCDdy.data = pdCDdy.data * loss_scale
    
    # TODO: is this superfluous? and we can return y_x?
    y_surface = netp(p_surface.data, p_surface.z_in(z)).squeeze(1)
    return y_surface, p_surface, CD, pdCDdy

def loss_wasserstein_div(geomloss_wasserstein_dist_fn, z, netp: NetWithPartials, p_sampler, beta, n_subsample, max_div, loss_scale, **kwargs) -> Scalar:
    
    x_domain_full = p_sampler.sample_from_inside_envelope()
    p_domain_full = PointWrapper.create_from_equal_bx(einops.repeat(x_domain_full, 'n d -> bz n d', bz=z.shape[0]))

    y_domain_full = netp.grouped_no_grad_fwd('vf', p_domain_full.data, p_domain_full.z_in(z)).squeeze(1)
    # y_domain_full = heaviside(y_domain_full, beta, nf_is_density=nf_is_density)
    y_domain_full = y_domain_full ** 2
    print(f'WARNING: Wasserstein loss is hardcoded to power 2')

    y_batch = einops.rearrange(y_domain_full, '(bz n) -> bz n', bz=z.shape[0])
    # sample along dim 1; y_batch does not need to be normalized for multinomial
    idx = torch.multinomial(y_batch, num_samples=n_subsample, replacement=False)
    x_domain = x_domain_full[idx]
    p_domain = PointWrapper.create_from_equal_bx(x_domain)

    y_x = netp.grouped_no_grad_fwd('vf_x', p_domain.data, p_domain.z_in(z)).squeeze(1)
    py_x = PointWrapper(data=y_x, map=p_domain._map)

    start_t = time.time()
    W, p_dWdy = wasserstein_diversity_loss(geomloss_wasserstein_dist_fn, p_domain, py_x)
    print(f'Wasserstein time: {time.time() - start_t}')
    
    W = W - max_div
    if W < 0:
        return (torch.tensor(0.0, device=z.device, dtype=z.dtype),
                torch.tensor(0.0, device=z.device, dtype=z.dtype),
                torch.tensor(0.0, device=z.device, dtype=z.dtype))
        
    W = W * loss_scale
    p_dWdy.data = p_dWdy.data * loss_scale
    
    # to pass gradients later through the network
    y_domain = netp(p_domain.data, p_domain.z_in(z)).squeeze(1)
    
    return p_domain, y_domain, W, p_dWdy
    
    

def loss_div(z, netp, p_sampler, p_surface, weights_surf_pts, logger, max_div, ginn_bsize, loss_scale,
             diversity_pts_source, div_norm_order, div_neighbor_agg_fn, leinster_temp, leinster_q, **kwargs) -> Scalar:
    assert diversity_pts_source in ['domain', 'surface'], f"diversity_pts_source ({diversity_pts_source}) must be 'domain' or 'surface'"
    loss_div = torch.tensor(0.0, device=z.device, dtype=z.dtype)
    
    if diversity_pts_source == 'domain':
        y_div = netp(*tensor_product_xz(p_sampler.sample_from_inside_envelope(), z)).squeeze(1)
        weights_surf_pts = None
    elif diversity_pts_source == 'surface':
        if p_surface is None:
            logger.info('No surface points found - skipping diversity loss')
            return loss_div
        else:    
            y_div = netp(*tensor_product_xz(p_surface.data, z)).squeeze(1)  # [(bz k)] whereas k is n_surface_points; evaluate netp at all surface points for each shape
    loss_div = diversity_loss(einops.rearrange(y_div, '(bz k)-> bz k', bz=ginn_bsize), 
                                                                weights=weights_surf_pts,
                                                                norm_order=div_norm_order,
                                                                neighbor_agg_fn=div_neighbor_agg_fn,
                                                                leinster_temp=leinster_temp,
                                                                leinster_q=leinster_q)
    if torch.isnan(loss_div) or torch.isinf(loss_div):
        logger.warning(f'NaN or Inf loss_div: {loss_div}')
        loss_div = torch.tensor(0.0, device=z.device, dtype=z.dtype)

    loss_div = torch.clamp(loss_div - max_div, min=0)
    loss_div = loss_scale * loss_div
    return loss_div

# CURVATURE LOSS

def loss_curv(z, epoch, netp, p_surface, weights_surf_pts, logger, max_curv, loss_scale, device, curvature_pts_source, \
              curvature_expression, strain_curvature_clip_max, curvature_use_gradnorm_weights, \
              curvature_after_5000_epochs, **kwargs) -> Scalar:
    loss_curv = torch.tensor(0.0, device=device, dtype=z.dtype)
    k_theta_gradnorm_fn = get_k_theta_gradnorm_func(netp, curvature_expression, strain_curvature_clip_max, max_curv)
    if p_surface is None:
        logger.debug('No surface points found - skipping curvature loss')
    else:
        # check this here, as for vmap-ed curvature it can't be checked there
        assert weights_surf_pts is None or torch.allclose(weights_surf_pts.sum(), torch.tensor(1.0, device=weights_surf_pts.device, dtype=weights_surf_pts.dtype)), f"weights must sum to 1"
        
        weights = weights_surf_pts
        if curvature_use_gradnorm_weights:
            # weights = self.k_theta_gradnorm_fn(self.netp.params_, self.p_surface.data, self.p_surface.z_in(z), torch.ones(size=(len(self.p_surface.data), 1), device=self.config['device']) / len(self.p_surface.data)) # gradnorm - unsqueeze is needed for vmap
            gn = k_theta_gradnorm_fn(netp.params_, p_surface.data, p_surface.z_in(z), 
                                     torch.ones(size=(len(p_surface.data), 1), device=device) / len(p_surface.data))
            weights = gn.sum() / gn
            weights = weights / weights.sum()
        
        y_x_surf = netp.vf_x(p_surface.data, p_surface.z_in(z)).squeeze(1)
        y_xx_surf = netp.vf_xx(p_surface.data, p_surface.z_in(z)).squeeze(1)
        loss_curv, loss_curv_unweighted = expression_curvature_loss(y_x_surf, y_xx_surf, 
                                                expression=curvature_expression,
                                                clip_max_value=strain_curvature_clip_max,
                                                weights=weights)
        
        loss_curv = max(torch.tensor(0.0, device=device, dtype=z.dtype), loss_curv - max_curv)

    if curvature_after_5000_epochs and epoch < 5000:
        loss_curv = torch.tensor(0.0, device=device)

    loss_curv = loss_scale * loss_curv

    return loss_curv


def get_k_theta_gradnorm_func(netp, curvature_expression, strain_curvature_clip_max, max_curv): 

    # non-vectorized loss function
    def curvature_loss_wrapper(params, x, z, weights):
        # for netp calls, use the properties f_x_ and f_xx_ instead of the methods f_x and f_xx
        y_x = netp.vf_x_(params, x, z).squeeze(1)
        y_xx = netp.vf_xx_(params, x, z).squeeze(1)
        loss_curv, loss_curv_unweighted = expression_curvature_loss(y_x, y_xx, 
                                            expression=curvature_expression,
                                            clip_max_value=strain_curvature_clip_max,
                                            weights=weights)
        loss_curv = torch.clamp(loss_curv - max_curv, min=0)
        return loss_curv

    # compute gradient wrt to first argument, which is theta
    k_theta = jacrev(curvature_loss_wrapper, argnums=0) # params, nx, nz, nx -> [ny, params]

    # vectorize
    vk_theta = vmap(k_theta, in_dims=(None, 0, 0, 0), out_dims=(0))  ## params, [bxz, nx], [bxz, nz] [bxz, nx] -> [bxz, params]

    def final_func(params_, x, z, weights):
        params = {key: tensor.detach() for key, tensor in params_.items()}
        res = vk_theta(params, x, z, weights)
        grads = torch.hstack([g.flatten(start_dim=1) for g in res.values()]) ## flatten batched grads per parameter
        # grads = [param.grad.detach().flatten() for param in params if param.grad is not None ]
        grad_norm = grads.norm(dim=1)
        return grad_norm
        
    return final_func

def loss_rotation_symmetric(z, netp, p_sampler, n_cycles, **kwargs) -> Scalar:
    loss_rot = torch.tensor(0.0, device=z.device, dtype=z.dtype)
    pts_rot_sym = p_sampler.sample_rotation_symmetric()
    y_rot_sym = netp(*tensor_product_xz(pts_rot_sym, z)).squeeze(1)
    y_reshaped = einops.rearrange(y_rot_sym, '(z k n) -> n k z', k=n_cycles, z=len(z))
    y_var = y_reshaped.var(dim=1)
    # replace NaNs or Infs with 0
    if torch.isnan(y_var).any() or torch.isinf(y_var).any():
        print(f'NaN or Inf in y_var')
    y_var[torch.isnan(y_var) | torch.isinf(y_var)] = 0.0
    loss_rot = y_var.mean()  # omitting the square as it's in the var
    return loss_rot

def loss_null(z, **kwargs) -> Scalar:
    return torch.tensor(0.0, device=z.device)


def loss_volume(z, netp: NetWithPartials, p_sampler, vol_frac, x_fem, beta, p, p_const, vol_loss, loss_scale, nf_is_density, **kwargs) -> Grad_Field:
    
    x = x_fem
    # print(f'WARNING: hard coded taking from envelop')
    # x = None
    if x is None:
        x = p_sampler.sample_from_inside_envelope()
        assert len(x) > 15000, f'x_domain should be at least as big as the mesh'
    
    y = netp(*tensor_product_xz(x, z)).squeeze(1)
    y = einops.rearrange(y, '(bz n) -> bz n', bz=z.shape[0])
    
    y = heaviside(y, beta=beta, nf_is_density=nf_is_density)
    exponent = p if p is not None else p_const
    y = y ** exponent
    
    mass_frac = y.mean(dim=1)

    if vol_loss == 'mse':
        V_loss = (mass_frac / vol_frac - 1) ** 2
    elif vol_loss == 'mae':
        V_loss = torch.abs(mass_frac / vol_frac - 1)
    elif vol_loss == 'mse_pushdown':
        V_loss = (mass_frac / vol_frac - 1) ** 2
        V_loss = torch.where(mass_frac > vol_frac, V_loss, torch.tensor(0.0, device=z.device, dtype=z.dtype))
    elif vol_loss == 'mse_unscaled_pushdown':
        V_loss = (mass_frac - vol_frac) ** 2
        V_loss = torch.where(mass_frac > vol_frac, V_loss, torch.tensor(0.0, device=z.device, dtype=z.dtype))
    elif vol_loss == 'mae_unscaled_pushdown':
        V_loss = torch.abs(mass_frac - vol_frac)
        V_loss = torch.where(mass_frac > vol_frac, V_loss, torch.tensor(0.0, device=z.device, dtype=z.dtype))
    else:
        raise ValueError(f'Unknown volume loss: {vol_loss}')
    
    V_loss = loss_scale * V_loss.mean()
    V_loss = V_loss ** 2 # square the loss to compensate for the square root in the ALM
    return V_loss


# =============================================================================
# Shape Diversity Loss (for generative training)
# =============================================================================

def loss_shape_diversity(z, p_surface, diversity_type, aggregation, max_diversity, loss_scale, **kwargs) -> Scalar:
    """
    Wrapper for diversity losses that matches the standard loss signature.
    
    Encourages generated shapes to be different from each other.
    
    Args:
        z: Latent vectors [B, nz]
        p_surface: PointWrapper with surface points for all shapes
        diversity_type: 'chamfer', 'volume', or 'contrastive'
        aggregation: 'min' or 'mean'
        max_diversity: Optional threshold above which loss is 0
        loss_scale: Scaling factor for the loss
        
    Returns:
        Scalar diversity loss
    """
    from train.losses_diversity import (
        diversity_loss_chamfer,
        diversity_loss_volume_symmetric_difference,
        diversity_loss_contrastive,
    )
    
    loss = torch.tensor(0.0, device=z.device, dtype=z.dtype)
    
    if p_surface is None:
        return loss
    
    if diversity_type == 'chamfer':
        loss = diversity_loss_chamfer(
            surface_pts_batch=p_surface,
            aggregation=aggregation,
            max_diversity=max_diversity,
        )
    elif diversity_type == 'volume':
        # Volume-based diversity requires SDF grid - not available here
        # This would be computed from the FEM grid if needed
        pass
    elif diversity_type == 'contrastive':
        # Contrastive loss requires shape features
        # Could use flattened surface point statistics
        pass
    
    loss = loss_scale * loss
    return loss


# =============================================================================
# Connectivity Loss (Morse Theory - for generative training)
# =============================================================================

def loss_connectivity(
    z,
    netp,
    bounds=None,
    cached_connectivity_loss=None,
    epoch=None,
    loss_scale=1.0,
    **kwargs
) -> Scalar:
    """
    Wrapper for connectivity loss. Bounds can be passed at call time for conditional LEGO.
    Uses cached_connectivity_loss (whose .bounds is updated in _sync_problem_dependents).
    """
    from train.losses_connectivity_v2 import CachedConnectivityLossV2 as CachedConnectivityLoss

    if cached_connectivity_loss is None:
        return torch.tensor(0.0, device=z.device, dtype=z.dtype)
    epoch = epoch if epoch is not None else kwargs.get('epoch', 0)
    loss_total = torch.tensor(0.0, device=z.device, dtype=z.dtype)
    n_shapes = z.shape[0]

    # Process each shape in batch
    # Note: This is expensive - consider only computing for subset of batch
    for i in range(min(n_shapes, 4)):  # Limit to 4 shapes for efficiency
        z_i = z[i:i+1]  # [1, nz]

        # Create callable for this shape
        def f_i(x):
            z_expanded = z_i.expand(len(x), -1)
            return netp(x, z_expanded)

        loss_i, info = cached_connectivity_loss(f_i, epoch)
        loss_total = loss_total + loss_i
    
    # Average over shapes processed
    loss_total = loss_total / min(n_shapes, 4)
    loss_total = loss_scale * loss_total
    
    return loss_total