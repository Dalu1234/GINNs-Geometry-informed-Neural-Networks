# Critical issues: stud formation not showing at all

This document lists the root causes that prevent studs from appearing in LEGO 1xN training and visualization.

---

## 1. **Stale problem in loss dispatcher (main bug)**

**What:** `init_loss_dispatcher()` binds `self.problem` (and `self.problem.bounds`) when the trainer is created. With conditional LEGO we have one problem per brick type `(n_studs, n_studs_y, height)` and we set `self.problem = self.problems[key]` **per batch**. The partials created at init still hold a reference to the **first** problem (e.g. key0 = 1x2), not the current batch’s problem.

**Effect:**

- **stud_grid:** Cylinder loss is always computed for key0’s `n_studs` (e.g. 2). For batches with N=4, 5, 8, 12 we only get gradient for 2 cylinders at 1x2 positions; the other stud positions get **no** stud_grid gradient, so studs never form there.
- **cuboid_primitive:** Always uses key0’s bounds. For a 1x8 batch we enforce “inside box” for the 1x2 box, so the shape is pushed toward the wrong (smaller) cuboid.
- **connectivity:** Uses key0’s bounds for the same reason.
- **eikonal, if, env, if_normal, latent_structure, dirichlet, etc.:** Same pattern: they use the initial `p_sampler`/problem, so all are effectively evaluated for key0 only.

**Fix:** Do **not** bind `problem` / `bounds` in the partials for conditional LEGO. Pass `problem=self.problem` and `bounds=self.problem.bounds` at **call time** in the training step so the current batch’s problem is used. Apply this for at least: `stud_grid`, `cuboid_primitive`, `connectivity`, and any other loss that uses `self.problem` or `self.problem.bounds`.

---

## 2. **Stud center coordinate mismatch**

**What:** In `loss_stud_grid`, cylinder centers are computed as:

- `cx = (i - (N_studs - 1) / 2.0) * stud_spacing` (symmetric grid around 0).

In `problem_lego_1xN`, stud centers use:

- `x_mm = -brick_length/2 + STUD_SPACING/2 + i * STUD_SPACING` with `brick_length = n_studs * 8 - 0.2`.

After normalization this gives a small constant offset (e.g. ~0.0125 in normalized space) relative to the grid used in the loss. So the “inside cylinder” samples are slightly shifted relative to the true stud positions.

**Effect:** Even for the correct N, the stud_grid loss pushes SDF &lt; 0 in cylinders that are slightly offset from the actual stud geometry, so the zero-level set may not align well with the intended stud surface.

**Fix:** When the problem has `stud_centers` (e.g. LEGO with `no_studs=False`), use `problem.stud_centers` (and `problem.stud_radius`, `problem.stud_z_bottom`, `problem.stud_z_top`) in `loss_stud_grid` instead of recomputing centers from the grid formula. That way cylinder positions and mesh/interface definitions match exactly.

---

## 3. **Only “inside negative” enforced (no sharp boundary)**

**What:** `cylinder_primitive_loss` only penalizes `SDF > -safety_margin` **inside** the cylinder (ReLU(f + margin)). There is no term that explicitly enforces “outside the cylinder → SDF &gt; 0” or “on the cylinder surface → SDF = 0”.

**Effect:** The network can satisfy the loss by making a flat negative region (blob) inside the cylinder instead of a sharp cylindrical boundary. Without an “outside positive” or surface term, the isosurface can be blurry or wrong.

**Mitigation (optional):** Add an “outside cylinder” term (sample points just outside the cylinder and penalize `f < margin`) and/or tie in interface loss on stud tops/sides when `lambda_if` is non-zero.

---

## 4. **Interface loss disabled for studs**

**What:** In the LEGO config, `lambda_if: 0`. So we do not directly supervise SDF = 0 (and normals) on stud tops and sides.

**Effect:** The only stud-related signal is stud_grid (inside negative). There is no direct “surface here” signal for the stud boundary, which makes it harder to get sharp, correctly placed studs.

**Mitigation:** Consider turning on `lambda_if` (or a stud-only interface term) with a small weight once stud_grid + correct problem per batch are in place.

---

## 5. **Pretrained “cuboid-only” prior**

**What:** If training is resumed from a checkpoint that was trained with `no_studs: true`, the network has never seen negative SDF in the stud region. The cuboid-only prior strongly prefers “flat top” at `stud_z_bottom`.

**Effect:** Even with stud_grid and correct problem, the optimizer must overcome that prior; studs can take many epochs to appear.

**Mitigation:** Train from scratch (`load_model: false`, etc.) when enabling stud formation, or train longer and/or increase `lambda_stud_grid` relative to other losses.

---

## 6. **SCC (single connected component) vs studs**

**What:** `lambda_scc` penalizes multiple connected components. Early in training, studs can appear as separate blobs before they connect to the body, so SCC can push to remove them.

**Effect:** Already mitigated in the current config by setting `lambda_scc: 0`. If re-enabled, use a delayed start or lower weight so stud formation is not suppressed.

---

## Summary of fixes (in order of impact)

1. **Pass current problem/bounds at call time** for conditional LEGO so `stud_grid`, `cuboid_primitive`, and related losses use the batch’s brick type (critical).
2. **Use `problem.stud_centers`** (and problem’s stud geometry) in `loss_stud_grid` when available so cylinder positions match the problem (removes formula mismatch).
3. Optionally add “outside cylinder” or interface terms for sharper stud boundaries and consider enabling a small `lambda_if` or stud-only interface loss.
4. When enabling studs, prefer training from scratch or long runs so the cuboid-only prior is overcome.
