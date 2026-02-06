# use_n_encoder: Critical Audit (Training Disruption Prevention)

When **use_n_encoder: true**, the latent has **17** columns (6 base + **9** encoder + 1 n_y + 1 height). The model expects this; any path that builds or passes **z** with the wrong length or wrong `n_norm_col` can crash or train incorrectly.

---

## Fixes applied

### 1. **z_val (validation/plotting)** — FIXED
- **Issue:** z_val was built with scalar n_norm (1 col) → 9 cols total. Model expected 17; `dist_to_edge` indexed column 14 → IndexError.
- **Fix:** In `train/ginn_trainer.py`, when building z_val, added `elif getattr(self, 'use_n_encoder', False):` branch that appends `self.model.encode_n(n_norms)` (9 dims) instead of a single n_norm column.

### 2. **outside_brick_sdf loss n_norm_col** — FIXED
- **Issue:** init_loss_dispatcher bound `n_norm_col=2` and `n_norm_col_end=...` in the partial. With encoder, the 9th dim is at column 14; reading column 2 would be wrong.
- **Fix:** Use `getattr(self.model, 'n_norm_col', 2)` and `getattr(self.model, 'n_norm_col_end', None)` in the partial so the loss uses the model’s layout (column 14 when encoder on).

### 3. **z_corners vs z shape mismatch** — FIXED
- **Issue:** `z_corners` from LatentSampler has `nz=6` (base only). Combining or passing it where full z has 17 cols would crash or corrupt.
- **Fix (latent_sampler):** In `combine_z_with_z_corners`, only concatenate when `z_corners.shape[1] == z.shape[1]`.
- **Fix (trainer):** When computing metrics with `z_corners`, only call `get_mesh_or_contour(..., z_corners)` when `z_corners.shape[1] == z.shape[1]`.

### 4. **Conditional prior c_dim** — VERIFIED
- **Comment:** Updated to “9 encoded dims (8 backbone + 1 for dist_to_edge)”. c_dim_prior = 11 (9+1+1) is correct; prior sees last 11 cols of z.

---

## Paths verified (no change needed)

| Path | Status |
|------|--------|
| **Violation buffer z** | Builds z with `encode_n` (9 dims); correct. |
| **Non-VB per-epoch z** | Appends `encode_n(n_norm_t)` when use_n_encoder; correct. |
| **\_build_z_for_n** | Returns z with 9 encoder dims; used by unseen_violation, interp_midpoint; correct. |
| **step_kw n_norm_col** | Passes `getattr(self.model, 'n_norm_col', 2)` to losses; correct. |
| **loss_prior** | Slices z_base and c from z; last 11 cols = c when encoder; correct. |
| **conditional_prior.sample** | Replaces z_base, keeps z[:, nz_base:]; correct. |
| **PH calc_ph(z)** | z is training batch (17 cols); correct. |
| **recalc_output / get_mesh** | All call sites now receive full-length z (z_val fixed; z_plot from combine only when dims match; z and z_row from z_val). |

---

## Summary

- **z_val** and **outside_brick_sdf** dispatcher are fixed so encoder training and validation use the right z layout and column index.
- **z_corners** is guarded so it is never mixed or passed to the model when its latent size differs from full z (conditioning).
- No other paths build or pass z with the wrong length for use_n_encoder; training should run without disruption from latent layout.
