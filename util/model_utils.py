import numpy as np
import torch


def tile_coords_xy(x, period_x=1.0, period_y=1.0, center=True):
    """
    Append tiled (periodic) x and y coordinates to support a stud-spacing inductive bias.
    In normalized LEGO space, one stud spacing = 1.0, so the network sees "position within
    the current tile" and can learn one cylindrical feature per tile.

    x: [..., nx] with nx >= 2 (x, y are first two dims).
    period_x, period_y: period in the same units as x (e.g. 1.0 for normalized LEGO).
    center: if True, tile coords are in [-0.5, 0.5] (cylinder centered at 0); else [0, 1).

    Returns:
        x_aug: [..., nx+2] with x_tile and y_tile appended.
    """
    if isinstance(x, torch.Tensor):
        # x, y are first two columns
        x_raw = x[..., 0]
        y_raw = x[..., 1]
        # Position within tile: modulo then optionally center
        x_tile = ((x_raw % period_x) + period_x) % period_x / period_x
        y_tile = ((y_raw % period_y) + period_y) % period_y / period_y
        if center:
            x_tile = x_tile - 0.5
            y_tile = y_tile - 0.5
        x_tile = x_tile.unsqueeze(-1)
        y_tile = y_tile.unsqueeze(-1)
        return torch.cat([x, x_tile, y_tile], dim=-1)
    else:
        x_raw = x[..., 0]
        y_raw = x[..., 1]
        x_tile = ((x_raw % period_x) + period_x) % period_x / period_x
        y_tile = ((y_raw % period_y) + period_y) % period_y / period_y
        if center:
            x_tile = x_tile - 0.5
            y_tile = y_tile - 0.5
        return np.concatenate([x, x_tile[..., None], y_tile[..., None]], axis=-1)


def dist_to_edge_x_from_z(x, z, n_studs_values, n_norm_col=2, n_norm_col_end=None, n_norm_override=None):
    """
    Signed distance to nearest x-boundary of the brick (positive inside, negative outside).
    Decodes brick type from z and uses normalized LEGO formula: half_length_x = (n_studs - 0.025) / 2.

    x: [B, 3] or [3] (x, y, z in normalized space; 1D when called under vmap).
    z: [B, nz] or [nz]. N conditioning: either single column n_norm in [0,1], or Gaussian weights in z[:, n_norm_col:n_norm_col_end].
    n_studs_values: list of N values (e.g. [2, 3, 4, 6]).
    n_norm_col_end: if set and > n_norm_col+1, z[:, n_norm_col:n_norm_col_end] is weight vector over n_studs_values; n_eff = sum(weights * support).
    n_norm_override: optional [B] or scalar; when set (e.g. encoder 9th dim), use instead of z[:, n_norm_col] so decoder does not see raw n_norm.

    Returns:
        dist_to_edge_x: [B] or scalar signed distance (positive inside brick).
    """
    support = torch.tensor(n_studs_values, device=z.device, dtype=z.dtype)
    n_min = min(n_studs_values)
    n_max = max(n_studs_values)
    nz = z.shape[-1]
    use_smooth = (
        n_norm_col_end is not None
        and (n_norm_col_end - n_norm_col) >= len(n_studs_values)
        and nz >= n_norm_col_end
    )
    if z.dim() == 1:
        if n_norm_override is not None:
            n_norm_val = n_norm_override.flatten()[0].clamp(0.0, 1.0).item() if isinstance(n_norm_override, torch.Tensor) else max(0.0, min(1.0, float(n_norm_override)))
            n_eff = n_min + n_norm_val * (n_max - n_min)
        elif use_smooth:
            w = z[n_norm_col:n_norm_col_end]
            n_eff = (w * support).sum()
        else:
            n_norm = z[n_norm_col].clamp(0.0, 1.0)
            n_eff = n_min + n_norm * (n_max - n_min)
        half_length_x = (n_eff - 0.025) / 2.0
        x_coord = x[0]
        dist_to_edge_x = torch.minimum(
            x_coord + half_length_x,
            half_length_x - x_coord
        )
    else:
        if n_norm_override is not None:
            n_norm = n_norm_override.flatten() if n_norm_override.dim() > 1 else n_norm_override
            n_norm = n_norm.to(z.device).to(z.dtype).clamp(0.0, 1.0)
            n_eff = n_min + n_norm * (n_max - n_min)
        elif use_smooth:
            w = z[:, n_norm_col:n_norm_col_end]  # (B, len(support))
            n_eff = (w * support.unsqueeze(0)).sum(dim=1)  # (B,)
        else:
            n_norm = z[:, n_norm_col].clamp(0.0, 1.0)
            n_eff = n_min + n_norm * (n_max - n_min)
        half_length_x = (n_eff - 0.025) / 2.0
        dist_to_edge_x = torch.minimum(
            x[:, 0] + half_length_x,
            half_length_x - x[:, 0]
        )
    return dist_to_edge_x


def tensor_product_xz(x, z):
    """
    Generate correct inputs for bx different xs and bz different zs.
    For each z we want to evaluate it at all xs and vice versa, so we need a tensor product between the rows of the two vectors.
    x: [bx, nx]
    z: [bz, nz]
    returns: ([bx*bz, nx], [bz*bx, nz])
    e.g.:
    x = [[1, 2], [3, 4]]
    z = [[5, 6], [7, 8]]
    x_tp = [[1, 2], [3, 4], [1, 2], [3, 4]]
    z_tp = [[5, 6], [5, 6], [7, 8], [7, 8]]
    """
    z_tp = z.repeat_interleave(repeats=len(x), dim=0)
    x_tp = x.repeat(len(z), 1)
    return x_tp, z_tp


def tensor_product_xz_np(x, z):
    """
    Generate correct inputs for bx different xs and bz different zs.
    For each z we want to evaluate it at all xs and vice versa, so we need a tensor product between the rows of the two vectors.
    x: [bx, nx]
    z: [bz, nz]
    returns: ([bx*bz, nx], [bz*bx, nz])
    e.g.:
    x = [[1, 2], [3, 4]]
    z = [[5, 6], [7, 8]]
    x_tp = [[1, 2], [3, 4], [1, 2], [3, 4]]
    z_tp = [[5, 6], [5, 6], [7, 8], [7, 8]]
    """
    bx, nx = x.shape
    bz, nz = z.shape

    # Repeat each row of z for bx times
    z_tp = np.repeat(z, bx, axis=0)

    # Tile x for bz times
    x_tp = np.tile(x, (bz, 1))

    return x_tp, z_tp