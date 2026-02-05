"""
LEGO 1xN / NxN Brick Problem for GINN

A parametric LEGO brick with N studs in a row (1xN) or a grid of studs (NxN).
Standard LEGO dimensions (in mm, then normalized):
- Stud diameter: 4.8mm
- Stud height: 1.7mm (commonly cited; was 1.8)
- Stud spacing: 8.0mm (center to center)
- Brick height: 9.6mm (without stud)
- Brick width: 7.8mm (for 1-wide)
- Brick length: N * 8.0mm - 0.2mm

The envelope is the outer bounding box.
Interfaces are the tops of the studs (where bricks connect).
Walls are the six faces of the brick body (SDF=0 with outward normals).
"""

import os
import torch
import numpy as np

from GINN.problems.constraints import (
    BoundingBox2DConstraint, 
    BoundingBox3DConstraint,
    CircleConstraint,
    CompositeInterface, 
    CompositeConstraint, 
    SampleConstraint, 
    SampleConstraintWithNormals,
    CuboidEnvelope,
    RectangleInterface3D
)
from GINN.problems.problem_base import ProblemBase
from GINN.problems.sampling_primitives import sample_bbox, sample_axis_parallel_rectangle_in_3d


def t_(x):
    return torch.tensor(x, dtype=torch.float32)


class CylinderTopInterface(SampleConstraintWithNormals):
    """Sample points on top of a cylinder with upward normals."""
    
    def __init__(self, center_xy, z_top, radius, n_precompute=10000):
        """
        Args:
            center_xy: (x, y) center of cylinder
            z_top: z-coordinate of cylinder top
            radius: cylinder radius
        """
        device = torch.get_default_device()
        
        # Sample points on circular disk at z_top
        angles = torch.rand(n_precompute) * 2 * np.pi
        radii = torch.sqrt(torch.rand(n_precompute)) * radius  # sqrt for uniform area sampling
        
        pts = torch.zeros(n_precompute, 3, device=device)
        pts[:, 0] = center_xy[0] + radii * torch.cos(angles)
        pts[:, 1] = center_xy[1] + radii * torch.sin(angles)
        pts[:, 2] = z_top
        
        # Normals point upward (outward from shape for SDF)
        normals = torch.zeros(n_precompute, 3, device=device)
        normals[:, 2] = 1.0  # All point up
        
        super().__init__(sample_pts=pts, normals=normals)


class CylinderSideInterface(SampleConstraintWithNormals):
    """Sample points on the side of a cylinder with outward normals."""
    
    def __init__(self, center_xy, z_bottom, z_top, radius, n_precompute=10000):
        device = torch.get_default_device()
        
        # Sample points on cylinder surface
        angles = torch.rand(n_precompute) * 2 * np.pi
        heights = torch.rand(n_precompute) * (z_top - z_bottom) + z_bottom
        
        pts = torch.zeros(n_precompute, 3, device=device)
        pts[:, 0] = center_xy[0] + radius * torch.cos(angles)
        pts[:, 1] = center_xy[1] + radius * torch.sin(angles)
        pts[:, 2] = heights
        
        # Normals point outward radially
        normals = torch.zeros(n_precompute, 3, device=device)
        normals[:, 0] = torch.cos(angles)
        normals[:, 1] = torch.sin(angles)
        
        super().__init__(sample_pts=pts, normals=normals)


class ProblemLego1xN(ProblemBase):
    """
    LEGO 1xN brick problem.
    
    The neural field learns an SDF for a LEGO brick with N studs.
    Constraints enforce:
    - Envelope: Shape stays within the brick bounding box
    - Interfaces: Stud tops and sides at SDF=0 with outward normals
    - Walls: Six faces of the brick body at SDF=0 with outward normals
    - Domain: Interior can have material
    """
    
    # Standard LEGO dimensions in mm (per common specs: Brick Owl, Orion, LUGNET)
    STUD_DIAMETER = 4.8
    STUD_HEIGHT = 1.7   # was 1.8; 1.7 mm is the commonly cited value
    STUD_SPACING = 8.0  # center-to-center pitch (8 mm standard; ~7.986 at 25°C)
    BRICK_HEIGHT = 9.6  # body height without stud
    BRICK_WIDTH = 7.8   # 1-wide brick (8 - 0.2)
    
    def __init__(self, 
                 nx,
                 n_studs: int,  # N in "1xN" or first dim in "NxN"
                 n_points_envelope: int,
                 n_points_interfaces: int,
                 n_points_domain: int,
                 n_points_normals: int = None,
                 normalize_coords: bool = True,
                 nf_is_density: bool = False,
                 height_scale: float = 1.0,  # scale brick + stud height (1.0 = nominal LEGO)
                 n_studs_y: int = None,  # if set, brick is n_studs x n_studs_y (NxN when equal)
                 no_studs: bool = True,  # if True, treat as plain cuboid (no studs) — temporary
                 **kwargs) -> None:
        super().__init__(nx=nx)
        
        assert nx == 3, "LEGO problem is 3D only"
        assert n_studs >= 1, "Need at least 1 stud"
        assert height_scale > 0, "height_scale must be positive"
        if n_studs_y is not None:
            assert n_studs_y >= 1, "n_studs_y must be >= 1"
        
        self.n_studs = n_studs
        self.n_studs_y = n_studs_y  # None = 1xN
        self.height_scale = float(height_scale)
        self.no_studs = bool(no_studs)
        self.n_points_envelope = n_points_envelope
        self.n_points_interfaces = n_points_interfaces
        self.n_points_domain = n_points_domain
        self.n_points_normals = n_points_normals or n_points_interfaces
        self.nf_is_density = nf_is_density
        
        device = torch.get_default_device()
        
        # Compute brick dimensions (only brick body height scaled; stud height fixed)
        brick_length = n_studs * self.STUD_SPACING - 0.2  # mm
        brick_width = (self.n_studs_y * self.STUD_SPACING - 0.2) if self.n_studs_y is not None else self.BRICK_WIDTH
        brick_height = self.BRICK_HEIGHT * self.height_scale
        stud_radius = self.STUD_DIAMETER / 2
        total_height = brick_height if self.no_studs else (brick_height + self.STUD_HEIGHT)
        total_studs = 0 if self.no_studs else (n_studs * (self.n_studs_y if self.n_studs_y is not None else 1))
        
        # Bounding box (before normalization)
        # Center at origin, z=0 is bottom of brick
        bounds_mm = t_([
            [-brick_length/2, brick_length/2],  # x
            [-brick_width/2, brick_width/2],     # y  
            [0, total_height]                     # z (cuboid only when no_studs)
        ])
        
        # Envelope is slightly inside bounds (the brick body without studs)
        envelope_mm = t_([
            [-brick_length/2, brick_length/2],
            [-brick_width/2, brick_width/2],
            [0, brick_height]  # Just the body, not studs
        ])
        
        # Stud centers: 1xN = one row along x at y=0; NxM = grid (empty when no_studs)
        stud_centers_mm = []
        if not self.no_studs:
            if self.n_studs_y is None:
                for i in range(n_studs):
                    x = -brick_length/2 + self.STUD_SPACING/2 + i * self.STUD_SPACING
                    stud_centers_mm.append((x, 0.0))
            else:
                for i in range(n_studs):
                    for j in range(self.n_studs_y):
                        x = -brick_length/2 + self.STUD_SPACING/2 + i * self.STUD_SPACING
                        y = -brick_width/2 + self.STUD_SPACING/2 + j * self.STUD_SPACING
                        stud_centers_mm.append((x, y))
        
        # Normalization
        if normalize_coords:
            # Use FIXED scale factor based on 1x1 brick reference
            # This ensures studs are always the same size in normalized space
            # regardless of how many studs the brick has
            reference_dim = self.STUD_SPACING  # 8.0mm = one stud unit
            self.scale_factor = 1.0 / reference_dim  # 1 stud spacing = 1 unit
            self.center = t_([0, 0, total_height/2])
            
            def normalize(pts):
                return (pts - self.center) * self.scale_factor
            
            def normalize_bounds(b):
                b_centered = b.clone()
                b_centered[:, 0] = (b[:, 0] - self.center) * self.scale_factor
                b_centered[:, 1] = (b[:, 1] - self.center) * self.scale_factor
                return b_centered
            
            self.bounds = t_([
                [-brick_length/2 * self.scale_factor, brick_length/2 * self.scale_factor],
                [-brick_width/2 * self.scale_factor, brick_width/2 * self.scale_factor],
                [(0 - total_height/2) * self.scale_factor, (total_height - total_height/2) * self.scale_factor]
            ])
            
            stud_radius_norm = stud_radius * self.scale_factor
            stud_z_top = (brick_height + self.STUD_HEIGHT - total_height/2) * self.scale_factor
            stud_z_bottom = (brick_height - total_height/2) * self.scale_factor
            stud_centers_norm = [(x * self.scale_factor, y * self.scale_factor) for (x, y) in stud_centers_mm]
        else:
            self.scale_factor = 1.0
            self.center = t_([0, 0, 0])
            self.bounds = bounds_mm
            stud_radius_norm = stud_radius
            stud_z_top = brick_height + self.STUD_HEIGHT
            stud_z_bottom = brick_height
            stud_centers_norm = list(stud_centers_mm)
        
        # ============================================
        # Create constraint point clouds (SimJEB-style regions)
        # ============================================
        
        # 1. FAR OUTSIDE: Points far from brick (strongly enforce SDF > 0)
        pts_far_outside = self._sample_far_outside_brick(
            self.bounds, expansion_factor=2.0, n_samples=n_points_envelope
        )
        
        # 2. OUTSIDE ENVELOPE: Points just outside the brick surface
        pts_outside = self._sample_outside_brick(
            self.bounds, n_samples=n_points_envelope * 2
        )
        
        # 3. AROUND INTERFACE: Buffer zone around studs (skip when no_studs)
        if self.no_studs:
            pts_around_interface = pts_outside[:n_points_envelope]  # reuse outside points
        else:
            pts_around_interface = self._sample_around_studs(
                stud_centers_norm, stud_radius_norm, stud_z_bottom, stud_z_top,
                buffer_distance=stud_radius_norm * 0.5,  # 50% of stud radius
                n_samples=n_points_envelope
            )
        
        # 4. INSIDE ENVELOPE: Where material CAN exist (cuboid only when no_studs)
        pts_inside = self._sample_inside_brick(
            self.bounds, stud_centers_norm, stud_radius_norm, 
            stud_z_bottom, stud_z_top, n_samples=n_points_domain * 2
        )
        
        # INTERFACE POINTS: Stud cylinders when not no_studs; else none (walls only below)
        interface_pts_list = []
        interface_normals_list = []
        if not self.no_studs and total_studs > 0:
            pts_per_stud = n_points_interfaces // total_studs
            pts_per_stud_top = max(1, pts_per_stud // 3)
            pts_per_stud_side = max(1, pts_per_stud * 2 // 3)
            for center in stud_centers_norm:
                stud_top = CylinderTopInterface(
                    center_xy=t_(center), 
                    z_top=stud_z_top,
                    radius=stud_radius_norm,
                    n_precompute=pts_per_stud_top * 10
                )
                pts_top, normals_top = stud_top.get_sampled_points(pts_per_stud_top)
                interface_pts_list.append(pts_top)
                interface_normals_list.append(normals_top)
                stud_side = CylinderSideInterface(
                    center_xy=t_(center),
                    z_bottom=stud_z_bottom,
                    z_top=stud_z_top,
                    radius=stud_radius_norm,
                    n_precompute=pts_per_stud_side * 10
                )
                pts_side, normals_side = stud_side.get_sampled_points(pts_per_stud_side)
                interface_pts_list.append(pts_side)
                interface_normals_list.append(normals_side)
        if interface_pts_list:
            interface_pts = torch.cat(interface_pts_list, dim=0)
            interface_normals = torch.cat(interface_normals_list, dim=0)
            if nf_is_density:
                interface_normals = -interface_normals
        else:
            interface_pts = torch.zeros(0, 3, device=device)
            interface_normals = torch.zeros(0, 3, device=device)
        
        # ============================================
        # Wall constraints: six faces of the brick body (SDF=0, outward normals)
        # ============================================
        b = self.bounds  # (3, 2): [x_min,x_max], [y_min,y_max], [z_min,z_max]
        z_bottom = b[2, 0].item() if b[2, 0].numel() == 1 else float(b[2, 0])
        z_top_body = float(stud_z_bottom)
        # Bottom face (z = z_bottom), normal -z
        wall_bottom = RectangleInterface3D(
            start=t_([b[0, 0].item(), b[1, 0].item(), z_bottom]),
            end=t_([b[0, 1].item(), b[1, 1].item(), z_bottom]),
            target_normal=t_([0.0, 0.0, -1.0])
        )
        # Top face: full rectangle when no_studs, else exclude stud footprints
        if self.no_studs:
            wall_top_body = RectangleInterface3D(
                start=t_([b[0, 0].item(), b[1, 0].item(), z_top_body]),
                end=t_([b[0, 1].item(), b[1, 1].item(), z_top_body]),
                target_normal=t_([0.0, 0.0, 1.0])
            )
        else:
            pts_top_wall, normals_top_wall = self._sample_top_wall_excluding_studs(
                b, z_top_body, stud_centers_norm, stud_radius_norm, n_precompute=8000
            )
            wall_top_body = SampleConstraintWithNormals(sample_pts=pts_top_wall, normals=normals_top_wall)
        # x_min face, normal -x
        wall_x_min = RectangleInterface3D(
            start=t_([b[0, 0].item(), b[1, 0].item(), z_bottom]),
            end=t_([b[0, 0].item(), b[1, 1].item(), z_top_body]),
            target_normal=t_([-1.0, 0.0, 0.0])
        )
        # x_max face, normal +x
        wall_x_max = RectangleInterface3D(
            start=t_([b[0, 1].item(), b[1, 0].item(), z_bottom]),
            end=t_([b[0, 1].item(), b[1, 1].item(), z_top_body]),
            target_normal=t_([1.0, 0.0, 0.0])
        )
        # y_min face, normal -y
        wall_y_min = RectangleInterface3D(
            start=t_([b[0, 0].item(), b[1, 0].item(), z_bottom]),
            end=t_([b[0, 1].item(), b[1, 0].item(), z_top_body]),
            target_normal=t_([0.0, -1.0, 0.0])
        )
        # y_max face, normal +y
        wall_y_max = RectangleInterface3D(
            start=t_([b[0, 0].item(), b[1, 1].item(), z_bottom]),
            end=t_([b[0, 1].item(), b[1, 1].item(), z_top_body]),
            target_normal=t_([0.0, 1.0, 0.0])
        )
        wall_constraints = [wall_bottom, wall_top_body, wall_x_min, wall_x_max, wall_y_min, wall_y_max]
        
        # ============================================
        # Create constraint objects (SimJEB-style)
        # ============================================
        
        pts_far_outside_constraint = SampleConstraint(sample_pts=pts_far_outside)
        pts_outside_constraint = SampleConstraint(sample_pts=pts_outside)
        pts_around_interface_constraint = SampleConstraint(sample_pts=pts_around_interface)
        pts_inside_constraint = SampleConstraint(sample_pts=pts_inside)
        if interface_pts.shape[0] > 0:
            interface_constraint = SampleConstraintWithNormals(
                sample_pts=interface_pts, normals=interface_normals
            )
        else:
            interface_constraint = None
        
        inside_envelope = CompositeConstraint([pts_inside_constraint])
        domain = CompositeConstraint([pts_inside_constraint, pts_outside_constraint])
        
        # Number of points per wall for visualization (flat array of all wall points)
        n_pts_wall = max(1, n_points_interfaces // (len(wall_constraints) + (1 if interface_constraint is not None else 0)))
        wall_pts_list = [wc.get_sampled_points(n_pts_wall)[0].cpu() for wc in wall_constraints]
        wall_pts_np = torch.cat(wall_pts_list, dim=0).numpy()
        
        n_env_third = max(1, n_points_envelope // 3)
        self.constr_pts_dict = {
            'far_outside_envelope': pts_far_outside_constraint.get_sampled_points(N=n_env_third).cpu().numpy(),
            'outside_envelope': pts_outside_constraint.get_sampled_points(N=n_env_third).cpu().numpy(),
            'envelope_around_interface': pts_around_interface_constraint.get_sampled_points(N=n_env_third).cpu().numpy(),
            'inside_envelope': inside_envelope.get_sampled_points(N=n_points_domain).cpu().numpy(),
            'interface': interface_constraint.get_sampled_points(N=n_points_interfaces)[0].cpu().numpy() if interface_constraint is not None else np.zeros((0, 3)),
            'domain': domain.get_sampled_points(N=n_points_domain).cpu().numpy(),
            'walls': wall_pts_np,
        }
        
        self._envelope_constr = [pts_outside_constraint, pts_around_interface_constraint, pts_far_outside_constraint]
        self._interface_constraints = ([interface_constraint] + wall_constraints) if interface_constraint is not None else wall_constraints
        self._obstacle_constraints = None
        self._inside_envelope = inside_envelope
        self._domain = domain
        
        # Store geometry info for visualization
        self.stud_centers = stud_centers_norm
        self.stud_radius = stud_radius_norm
        self.stud_z_bottom = stud_z_bottom
        self.stud_z_top = stud_z_top
        
        # Body-only bounds (z max = stud_z_bottom) for cuboid_primitive so it does not sample inside stud volume
        self._bounds_body = self.bounds.clone()
        if not self.no_studs:
            sb = stud_z_bottom
            if hasattr(sb, 'to'):
                sb = sb.to(self.bounds.device, self.bounds.dtype)
            else:
                sb = torch.tensor(sb, device=self.bounds.device, dtype=self.bounds.dtype)
            self._bounds_body[2, 1] = sb
        
        label = f"{n_studs}x{self.n_studs_y}" if self.n_studs_y is not None else f"1x{n_studs}"
        print(f"Created LEGO {label} problem (height_scale={self.height_scale:.2f})" + (" [no_studs=cuboid only]" if self.no_studs else "") + ":")
        print(f"  Bounds: {self.bounds.tolist()}")
        print(f"  Stud centers: {len(stud_centers_norm)} studs")
        print(f"  Points: {len(pts_far_outside)} far_outside, {len(pts_outside)} outside, {len(pts_around_interface)} around_if, {len(pts_inside)} inside, {len(interface_pts)} interface, 6 walls")
    
    @property
    def bounds_body(self):
        """Bounds of the brick body only (z max = stud_z_bottom). Use for cuboid_primitive so it does not sample inside stud volume."""
        return self._bounds_body
    
    def sample_from_interface(self):
        """When no_studs: equal split across 6 walls. Else: 50% studs, 50% walls."""
        if self.no_studs:
            return super().sample_from_interface()
        n_stud = self.n_points_interfaces // 2  # half to studs (tops + sides)
        n_wall_total = self.n_points_interfaces - n_stud
        wall_constraints = self._interface_constraints[1:]
        pts_per_wall = max(1, n_wall_total // len(wall_constraints))
        pts = []
        normals = []
        pts_stud, normals_stud = self._interface_constraints[0].get_sampled_points(n_stud)
        pts.append(pts_stud)
        normals.append(normals_stud)
        for c in wall_constraints:
            pts_i, normals_i = c.get_sampled_points(pts_per_wall)
            pts.append(pts_i)
            normals.append(normals_i)
        return torch.cat(pts, dim=0), torch.cat(normals, dim=0)
    
    def _sample_far_outside_brick(self, bounds, expansion_factor, n_samples):
        """Sample points FAR outside the brick (like SimJEB's pts_far_outside)."""
        device = torch.get_default_device()
        
        # Expand bounds significantly for "far outside" region
        expanded = bounds.clone()
        expansion = (expansion_factor - 1.0) * (bounds[:, 1] - bounds[:, 0])
        expanded[:, 0] = bounds[:, 0] - expansion
        expanded[:, 1] = bounds[:, 1] + expansion
        
        # Sample in far expanded region, reject points inside near-envelope region
        near_expanded = bounds.clone()
        near_expansion = 0.3 * (bounds[:, 1] - bounds[:, 0])
        near_expanded[:, 0] = bounds[:, 0] - near_expansion
        near_expanded[:, 1] = bounds[:, 1] + near_expansion
        
        pts = sample_bbox(expanded, N=n_samples * 3)
        
        # Keep only points outside the near-envelope region
        inside_near_mask = (
            (pts[:, 0] >= near_expanded[0, 0]) & (pts[:, 0] <= near_expanded[0, 1]) &
            (pts[:, 1] >= near_expanded[1, 0]) & (pts[:, 1] <= near_expanded[1, 1]) &
            (pts[:, 2] >= near_expanded[2, 0]) & (pts[:, 2] <= near_expanded[2, 1])
        )
        pts_far = pts[~inside_near_mask]
        
        return pts_far[:n_samples]
    
    def _sample_outside_brick(self, bounds, n_samples):
        """Sample points outside the brick bounding box."""
        device = torch.get_default_device()
        
        # Expand bounds slightly for "outside" region
        expanded = bounds.clone()
        expansion = 0.3 * (bounds[:, 1] - bounds[:, 0])
        expanded[:, 0] = bounds[:, 0] - expansion
        expanded[:, 1] = bounds[:, 1] + expansion
        
        # Sample in expanded region, reject points inside original bounds
        pts = sample_bbox(expanded, N=n_samples * 2)
        
        # Keep only points outside original bounds
        inside_mask = (
            (pts[:, 0] >= bounds[0, 0]) & (pts[:, 0] <= bounds[0, 1]) &
            (pts[:, 1] >= bounds[1, 0]) & (pts[:, 1] <= bounds[1, 1]) &
            (pts[:, 2] >= bounds[2, 0]) & (pts[:, 2] <= bounds[2, 1])
        )
        pts_outside = pts[~inside_mask]
        
        return pts_outside[:n_samples]
    
    def _sample_top_wall_excluding_studs(self, b, z_top_body, stud_centers_norm, stud_radius_norm, n_precompute=8000):
        """Sample points on the top face of the brick body (z = stud_z_bottom), excluding the stud footprints so the studs are not interrupted."""
        start = t_([b[0, 0].item(), b[1, 0].item(), z_top_body])
        end = t_([b[0, 1].item(), b[1, 1].item(), z_top_body])
        pts = sample_axis_parallel_rectangle_in_3d(start, end, N=n_precompute)
        r = float(stud_radius_norm) if hasattr(stud_radius_norm, 'item') else stud_radius_norm
        outside_all_studs = torch.ones(pts.shape[0], dtype=torch.bool, device=pts.device)
        for (cx, cy) in stud_centers_norm:
            cx_val = float(cx) if hasattr(cx, 'item') else cx
            cy_val = float(cy) if hasattr(cy, 'item') else cy
            dist_xy = torch.sqrt((pts[:, 0] - cx_val)**2 + (pts[:, 1] - cy_val)**2)
            outside_all_studs = outside_all_studs & (dist_xy > r)
        pts = pts[outside_all_studs]
        normals = torch.zeros_like(pts)
        normals[:, 2] = 1.0
        return pts, normals
    
    def _sample_around_studs(self, stud_centers, stud_radius, stud_z_bottom, stud_z_top, buffer_distance, n_samples):
        """Sample points in a buffer zone around the stud cylinders (like SimJEB's near-interface buffer)."""
        device = torch.get_default_device()
        
        n_per_stud = n_samples // len(stud_centers) + 1
        all_pts = []
        
        for cx, cy in stud_centers:
            # Sample in a box around this stud
            outer_radius = stud_radius + buffer_distance
            z_margin = buffer_distance
            
            box_min = torch.tensor([cx - outer_radius, cy - outer_radius, stud_z_bottom - z_margin], device=device)
            box_max = torch.tensor([cx + outer_radius, cy + outer_radius, stud_z_top + z_margin], device=device)
            box_bounds = torch.stack([box_min, box_max], dim=1)
            
            pts = sample_bbox(box_bounds, N=n_per_stud * 3)
            
            # Keep only points in the buffer zone: outside stud but within buffer distance
            dist_xy = torch.sqrt((pts[:, 0] - cx)**2 + (pts[:, 1] - cy)**2)
            
            # Points in the buffer zone around the cylinder side
            in_side_buffer = (
                (dist_xy >= stud_radius) & (dist_xy <= outer_radius) &
                (pts[:, 2] >= stud_z_bottom) & (pts[:, 2] <= stud_z_top)
            )
            
            # Points in buffer zone above the stud top
            in_top_buffer = (
                (dist_xy <= outer_radius) &
                (pts[:, 2] > stud_z_top) & (pts[:, 2] <= stud_z_top + z_margin)
            )
            
            in_buffer = in_side_buffer | in_top_buffer
            all_pts.append(pts[in_buffer])
        
        pts_around = torch.cat(all_pts, dim=0)
        
        # Shuffle and return requested number
        perm = torch.randperm(len(pts_around), device=device)
        return pts_around[perm[:n_samples]]
    
    def _sample_inside_brick(self, bounds, stud_centers, stud_radius, stud_z_bottom, stud_z_top, n_samples):
        """Sample points inside the brick (body + studs)."""
        device = torch.get_default_device()
        
        # Sample from bounding box
        pts = sample_bbox(bounds, N=n_samples * 3)
        
        # A point is inside if:
        # 1. It's in the main body (below stud_z_bottom), OR
        # 2. It's in a stud cylinder (above stud_z_bottom and within stud radius)
        
        in_body = pts[:, 2] <= stud_z_bottom
        
        in_any_stud = torch.zeros(len(pts), dtype=torch.bool, device=device)
        for cx, cy in stud_centers:
            dist_xy = torch.sqrt((pts[:, 0] - cx)**2 + (pts[:, 1] - cy)**2)
            in_stud = (dist_xy <= stud_radius) & (pts[:, 2] >= stud_z_bottom) & (pts[:, 2] <= stud_z_top)
            in_any_stud = in_any_stud | in_stud
        
        inside_mask = in_body | in_any_stud
        pts_inside = pts[inside_mask]
        
        return pts_inside[:n_samples]
    
    def is_inside_envelope(self, pts):
        """Check if points are inside the LEGO brick envelope.
        
        Returns a boolean tensor (not numpy) to match ph_manager expectations.
        """
        if isinstance(pts, np.ndarray):
            pts = torch.from_numpy(pts).float()
        
        # Check body
        in_body = (
            (pts[:, 0] >= self.bounds[0, 0]) & (pts[:, 0] <= self.bounds[0, 1]) &
            (pts[:, 1] >= self.bounds[1, 0]) & (pts[:, 1] <= self.bounds[1, 1]) &
            (pts[:, 2] >= self.bounds[2, 0]) & (pts[:, 2] <= self.stud_z_bottom)
        )
        
        # Check studs
        in_any_stud = torch.zeros(len(pts), dtype=torch.bool, device=pts.device)
        for cx, cy in self.stud_centers:
            dist_xy = torch.sqrt((pts[:, 0] - cx)**2 + (pts[:, 1] - cy)**2)
            in_stud = (
                (dist_xy <= self.stud_radius) & 
                (pts[:, 2] >= self.stud_z_bottom) & 
                (pts[:, 2] <= self.stud_z_top)
            )
            in_any_stud = in_any_stud | in_stud
        
        return in_body | in_any_stud
