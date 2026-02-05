import torch
import numpy as np

def sample_bbox(bnds, N=1000):
    """
    Sample N points in nx dimenions from a rectangular domain given by bnds.
    bnds[i] gives min, max along dimension i
    """
    return bnds[:,0] + (bnds[:,-1] - bnds[:,0])*torch.rand(N, bnds.shape[0]) ## Sample bx number of points within the specified domain

class BBLineSampler():

    def __init__(self, nx, bnds) -> None:
        self.nx = nx
        self.lines_list, self.ratio_per_line_list = self.get_lines_and_n_points_to_sample_uniformly(bnds, self.nx)

    def sample_on_bb(self, N):
        """
        Sample N points in nx dimenions on a rectangular domain given by bnds.
        bnds[i] gives min, max along dimension i
        """
        samples = []
        for i, (p1, p2) in enumerate(self.lines_list):
            # line_samples = p1 + (p2 - p1) * sample_unit_interval(n_points_list[i], method='random')
            samples.append(sample_line_segment(p1, p2, N=int(N * self.ratio_per_line_list[i])))
            
        samples = torch.concatenate(samples, dim=1).T
        return samples

    def get_lines_and_n_points_to_sample_uniformly(self, bnds, nx):
        bbox = bnds.cpu().numpy()
        if nx != 2:
            raise NotImplementedError('Sample on Bounding box not implemented for nx != 2')

        #        0       1
        # 0: [[x_min, x_max], 
        # 1:  [y_min, y_max]]
        #               x           y
        lower_left = torch.tensor([bbox[0, 0], bbox[1, 0]])
        upper_left = torch.tensor([bbox[0, 0], bbox[1, 1]])
        lower_right = torch.tensor([bbox[0, 1], bbox[1, 0]])
        upper_right = torch.tensor([bbox[0, 1], bbox[1, 1]])
            
        lines_list = [ 
                    (lower_left, lower_right),
                    (lower_right, upper_right),
                    (upper_right, upper_left),
                    (upper_left, lower_left),
                    ]
        line_lengths = ([torch.linalg.vector_norm(p1 - p2).item() for p1, p2 in lines_list])
        ratio_per_line_list = [line_len / np.sum(line_lengths) for line_len in line_lengths]
        return lines_list, ratio_per_line_list
        

def sample_unit_interval(N, method='random'):
    """
    Sample N points from the unit interval [0, 1].
    """
    if method=='random': ## sample random uniform distribution
        return torch.rand(N)
    if method=='grid': ## sample equidistant points
        return torch.linspace(0, 1, N)

def sample_line_segment(x_start, x_end, N=50):
    """
    Sample N points from a line segment from x_a to x_b.
    """
    x_start = x_start[:,None]
    x_end = x_end[:,None]
    ts = sample_unit_interval(N, method='grid')
    return x_start + (x_end-x_start)*ts ## linear interpolation


def inside_disk(xs, x, R, strict=True):
    """Returns mask describing if points xs are strictly inside the disk."""
    compare = torch.less if strict else torch.less_equal
    return compare(torch.norm(xs - x, dim=1), R)


def sample_disk(center, R, N=50):
    """Sample (approximately) N points from a 2D disk with radius R centered at x."""
    bbox = torch.vstack([center - R, center + R]).T
    xs_bbox = sample_bbox(bbox, N=int(N * 4 / torch.pi))

    # Reject points outside the disk
    inside_mask = inside_disk(xs_bbox, center, R)
    xs_disk = xs_bbox[inside_mask]
    return xs_disk

def sample_ring(x, r1, r2, N=50):
    """Sample (approximately) N points from a 2D ring with inner radius r1 and outer radius r2 centered at x."""
    bbox = torch.vstack([x - r2, x + r2]).T
    xs_bbox = sample_bbox(bbox, N=int(N * 4 * r2**2 / (r2**2 - r1**2) / torch.pi))

    # Reject points outside the ring
    inside_mask = inside_disk(xs_bbox, x, r2)
    outside_mask = ~inside_disk(xs_bbox, x, r1)
    ring_mask = inside_mask & outside_mask
    xs_ring = xs_bbox[ring_mask]
    return xs_ring

def sample_circle(x, r, N=50):
    """Sample (approximately) N points from a 2D circle with radius r centered at x."""
    # sample between 0 and 1 and multiply by 2pi to get the angle
    theta = 2 * np.pi * torch.rand(N)
    xs = x + r * torch.vstack([torch.cos(theta), torch.sin(theta)]).T
    return xs

def sample_axis_parallel_rectangle_in_3d(start_xyz, end_xyz, N=50):
    """Sample (approximately) N points from a 3D rectangle with corners at start_xyz and end_xyz."""
    bbox = torch.vstack([start_xyz, end_xyz]).T
    xs_bbox = sample_bbox(bbox, N=N)
    return xs_bbox


# =============================================================================
# CYLINDER PRIMITIVE SAMPLING (axis-aligned along z; for stud-grid loss)
# =============================================================================

def sample_inside_cylinder(center_xy, radius, z_bottom, z_top, n, device=None):
    """
    Sample n points uniformly inside a vertical cylinder (axis along z).
    
    Args:
        center_xy: (2,) or (x, y) center in the xy-plane
        radius: scalar radius
        z_bottom, z_top: scalar z extent
        n: number of points
        device: optional torch device
    
    Returns:
        (n, 3) tensor of points inside the cylinder
    """
    if device is None:
        device = center_xy.device if hasattr(center_xy, 'device') else torch.device('cpu')
    center_xy = torch.as_tensor(center_xy, device=device, dtype=torch.float32)
    if center_xy.dim() == 0:
        center_xy = center_xy.unsqueeze(0).expand(2)
    radius = float(radius) if hasattr(radius, 'item') else float(radius)
    z_bottom = float(z_bottom) if hasattr(z_bottom, 'item') else float(z_bottom)
    z_top = float(z_top) if hasattr(z_top, 'item') else float(z_top)
    # Oversample in bbox then reject outside disk
    n_try = max(n * 4, 500)
    xy = center_xy + (2 * radius) * (torch.rand(n_try, 2, device=device) - 0.5)
    z_pts = z_bottom + (z_top - z_bottom) * torch.rand(n_try, 1, device=device)
    pts = torch.cat([xy, z_pts], dim=1)
    dist_xy = torch.norm(pts[:, :2] - center_xy.unsqueeze(0), dim=1)
    inside = dist_xy < radius
    pts = pts[inside]
    if pts.shape[0] >= n:
        return pts[:n]
    # If not enough after reject, sample again to fill
    while pts.shape[0] < n:
        extra = sample_inside_cylinder(center_xy, radius, z_bottom, z_top, n - pts.shape[0], device)
        pts = torch.cat([pts, extra], dim=0)
    return pts[:n]


def sample_outside_cylinder_shell(center_xy, radius_inner, radius_outer, z_bottom, z_top, n, device=None):
    """
    Sample n points in a thin shell just outside a vertical cylinder (annulus in xy).
    Used to push SDF > 0 outside the stud so the zero-level set is at the cylinder boundary.
    """
    if device is None:
        device = center_xy.device if hasattr(center_xy, 'device') else torch.device('cpu')
    center_xy = torch.as_tensor(center_xy, device=device, dtype=torch.float32)
    if center_xy.dim() == 0:
        center_xy = center_xy.unsqueeze(0).expand(2)
    ri, ro = float(radius_inner), float(radius_outer)
    z_bottom, z_top = float(z_bottom), float(z_top)
    n_try = max(n * 4, 500)
    # sample in box [center - ro, center + ro] x [z_bottom, z_top], keep annulus ri < r < ro
    xy = center_xy + (2 * ro) * (torch.rand(n_try, 2, device=device) - 0.5)
    z_pts = z_bottom + (z_top - z_bottom) * torch.rand(n_try, 1, device=device)
    pts = torch.cat([xy, z_pts], dim=1)
    dist_xy = torch.norm(pts[:, :2] - center_xy.unsqueeze(0), dim=1)
    in_shell = (dist_xy >= ri) & (dist_xy <= ro)
    pts = pts[in_shell]
    while pts.shape[0] < n:
        xy = center_xy + (2 * ro) * (torch.rand(n_try, 2, device=device) - 0.5)
        z_pts = z_bottom + (z_top - z_bottom) * torch.rand(n_try, 1, device=device)
        pts2 = torch.cat([xy, z_pts], dim=1)
        dist_xy = torch.norm(pts2[:, :2] - center_xy.unsqueeze(0), dim=1)
        in_shell = (dist_xy >= ri) & (dist_xy <= ro)
        pts = torch.cat([pts, pts2[in_shell]], dim=0)
    return pts[:n]


# =============================================================================
# BOX PRIMITIVE SAMPLING (for rule-based cuboid loss)
# =============================================================================

def sample_inside_box(bounds, n, margin=0.0):
    """
    Sample n points uniformly inside a 3D box.
    
    Args:
        bounds: (3, 2) tensor [[x_min, x_max], [y_min, y_max], [z_min, z_max]]
        n: number of points
        margin: shrink box by this amount on each side (for "safely inside")
    
    Returns:
        (n, 3) tensor of points
    """
    device = bounds.device
    mins = bounds[:, 0] + margin
    maxs = bounds[:, 1] - margin
    pts = mins + (maxs - mins) * torch.rand(n, 3, device=device)
    return pts


def sample_near_box_faces(bounds, margin, n):
    """
    Sample n points in a shell just outside each face of the box.
    
    Args:
        bounds: (3, 2) tensor
        margin: distance from face (points are between face and face+margin)
        n: total number of points (distributed across 6 faces)
    """
    device = bounds.device
    pts_per_face = n // 6
    pts_list = []
    
    for axis in range(3):
        for side in [0, 1]:  # min face, max face
            face_pts = torch.rand(pts_per_face, 3, device=device)
            # Scale to box dimensions for non-axis dims
            for d in range(3):
                if d == axis:
                    if side == 0:
                        # Just outside min face
                        face_pts[:, d] = bounds[d, 0] - margin * torch.rand(pts_per_face, device=device)
                    else:
                        # Just outside max face
                        face_pts[:, d] = bounds[d, 1] + margin * torch.rand(pts_per_face, device=device)
                else:
                    face_pts[:, d] = bounds[d, 0] + (bounds[d, 1] - bounds[d, 0]) * face_pts[:, d]
            pts_list.append(face_pts)
    
    return torch.cat(pts_list, dim=0)


def sample_near_box_corners(bounds, margin, n):
    """
    Sample n points near the 8 corners of the box (outside).
    
    Args:
        bounds: (3, 2) tensor
        margin: distance from corner
        n: total number of points
    """
    device = bounds.device
    pts_per_corner = n // 8
    pts_list = []
    
    for x_side in [0, 1]:
        for y_side in [0, 1]:
            for z_side in [0, 1]:
                # Corner position
                corner = torch.tensor([
                    bounds[0, x_side],
                    bounds[1, y_side],
                    bounds[2, z_side]
                ], device=device)
                
                # Direction outward from box
                direction = torch.tensor([
                    -1.0 if x_side == 0 else 1.0,
                    -1.0 if y_side == 0 else 1.0,
                    -1.0 if z_side == 0 else 1.0
                ], device=device)
                
                # Sample in outward cone from corner
                offsets = torch.rand(pts_per_corner, 3, device=device) * margin
                offsets = offsets * direction.abs()  # Make positive
                offsets = offsets * direction.sign()  # Apply direction
                pts_list.append(corner + offsets)
    
    return torch.cat(pts_list, dim=0)


def sample_far_from_box(bounds, margin_range, n):
    """
    Sample n points far outside the box (between margin_range[0] and margin_range[1]).
    
    Args:
        bounds: (3, 2) tensor
        margin_range: (min_dist, max_dist) from box surface
        n: number of points
    """
    device = bounds.device
    min_dist, max_dist = margin_range
    
    # Expand box by max_dist, then reject points too close
    expanded_bounds = bounds.clone()
    expanded_bounds[:, 0] -= max_dist
    expanded_bounds[:, 1] += max_dist
    
    inner_bounds = bounds.clone()
    inner_bounds[:, 0] -= min_dist
    inner_bounds[:, 1] += min_dist
    
    # Oversample and reject
    pts = sample_bbox(expanded_bounds, N=n * 3)
    
    # Keep only points outside inner_bounds (far enough from box)
    inside_inner = (
        (pts[:, 0] >= inner_bounds[0, 0]) & (pts[:, 0] <= inner_bounds[0, 1]) &
        (pts[:, 1] >= inner_bounds[1, 0]) & (pts[:, 1] <= inner_bounds[1, 1]) &
        (pts[:, 2] >= inner_bounds[2, 0]) & (pts[:, 2] <= inner_bounds[2, 1])
    )
    pts = pts[~inside_inner][:n]
    
    return pts


def sample_on_box_faces_with_sdf(bounds, n):
    """
    Sample n points ON the 6 faces of the box, with their true SDF values (0 on surface).
    Also returns points slightly inside and outside with their true SDF.
    
    Args:
        bounds: (3, 2) tensor
        n: total number of points
    
    Returns:
        pts: (n, 3) points on and near faces
        true_sdf: (n,) true signed distance (negative inside, positive outside)
    """
    device = bounds.device
    pts_per_face = n // 6
    pts_list = []
    sdf_list = []
    
    for axis in range(3):
        for side in [0, 1]:  # min face, max face
            # Points on the face (SDF = 0)
            n_on = pts_per_face // 3
            n_inside = pts_per_face // 3
            n_outside = pts_per_face - n_on - n_inside
            
            for offset, count, sdf_sign in [(0.0, n_on, 0.0), 
                                             (-0.05, n_inside, -0.05),  # inside
                                             (0.05, n_outside, 0.05)]:  # outside
                face_pts = torch.rand(count, 3, device=device)
                sdf_vals = torch.full((count,), abs(offset), device=device)
                
                for d in range(3):
                    if d == axis:
                        if side == 0:
                            face_pts[:, d] = bounds[d, 0] - offset  # offset inward is negative
                            sdf_vals = sdf_vals * (-1 if offset < 0 else 1)
                        else:
                            face_pts[:, d] = bounds[d, 1] + offset
                            sdf_vals = sdf_vals * (-1 if offset < 0 else 1)
                    else:
                        face_pts[:, d] = bounds[d, 0] + (bounds[d, 1] - bounds[d, 0]) * face_pts[:, d]
                
                # True SDF for a box: distance to nearest face
                # For points exactly on a face, SDF = 0
                # For points offset from face, SDF = offset (with sign)
                pts_list.append(face_pts)
                if offset == 0:
                    sdf_list.append(torch.zeros(count, device=device))
                elif offset < 0:
                    sdf_list.append(torch.full((count,), offset, device=device))
                else:
                    sdf_list.append(torch.full((count,), offset, device=device))
    
    return torch.cat(pts_list, dim=0), torch.cat(sdf_list, dim=0)