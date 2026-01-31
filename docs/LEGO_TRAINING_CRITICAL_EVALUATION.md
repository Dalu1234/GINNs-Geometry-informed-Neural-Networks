# Critical evaluation: will LEGO GINN training succeed?

## Summary

**Verdict: Training can succeed for a subset of brick types, but several design choices and one validation bug create real risk.** Fix the validation bug first; then consider the recommendations below to improve robustness and observability.

---

## 1. Validation bug (critical)

**Issue:** During validation we set `self.problem = self.problems[key0]` (e.g. 1x2, height 0.75) and then call `recalc_output(netp, z_val, ...)` and `get_mesh_or_contour(..., z_val)`. `z_val` has **multiple rows** with **different** conditioning (n, ny, h) cycled per row, but **marching cubes uses a single problem’s bounds** (`self.problem.bounds`). So rows 1–8 are meshed in the **wrong** bounds (those of the first brick type). Validation plots and metrics are therefore wrong for all but the first shape.

**Fix (implemented):** When conditioning is on, validation now uses `z_val_single = z_val[:1]` for `recalc_output` and `get_mesh_or_contour`, so the single validated shape uses the same problem (and bounds) as `key0`. Validation therefore reports one shape for the first brick type (e.g. 1x2, ny=1, h=0.75) correctly.

**Remaining limitation:** Validation still only evaluates that one brick type; other (N, ny, h) are not validated. For full coverage you could later add a loop over a few keys and log/plot per key.

---

## 2. Problem count and update dilution

**Setup:** With `n_studs_values: [2,3,4,6]`, `n_studs_y_values: [1,2,4]`, `height_values: [0.75,1.0,1.25]` you have **4×3×3 = 36** problem configs. Each step samples **one** (N, ny, h); so each problem gets about **1/36** of the updates.

**Risk:** With 5000 epochs and e.g. 9 shapes per batch, that’s on the order of **~1250** steps per problem. For a single-shape GINN that’s often enough; for 36 different shapes sharing one network it may be thin, especially for rare combinations (e.g. 6×4, h=0.75).

**Recommendations:**
- Reduce the number of conditioning combinations for a first run (e.g. N in [2,4], ny in [1,2], h in [1.0] → 2×2×1 = 4 problems).
- Or increase `max_epochs` / total steps so each problem still gets enough updates.
- Monitor per-problem loss or sample quality (e.g. by logging which (N, ny, h) was used and aggregating loss by key).

---

## 3. Latent z range

**Setup:** `z_sample_interval: [0, 0.1]` — very narrow.

**Risk:** With strong conditioning (n, ny, h), the model may rely almost entirely on the conditioning and use z only weakly. Diversity (div) is then pushing on a small range; latent collapse or negligible z effect is possible.

**Recommendations:**
- Widen to e.g. `[0, 1]` or `[-0.5, 0.5]` so z has room to matter.
- Compare runs with/without div and with different z intervals to see if z is used.

---

## 4. Validation only on one brick type

**Setup:** Validation uses `key0 = (n0, ny0, h0)` (e.g. 1x2, ny=1, h=0.75). All validation plots and val metrics use that single problem.

**Risk:** You get no direct signal on 2×2, 4×4, or other heights. A failure mode is: the first brick type converges, others stay bad, and you don’t see it in validation.

**Recommendations:**
- Fix the validation bug (above), then add validation for a few representative keys (e.g. 1x4, 2x2, 4x4, one height each) and log or plot them separately.
- Optionally log training loss **by (N, ny, h)** so you can see which configs are improving.

---

## 5. Loss balance and schedule

**Current:** `lambda_if=10`, `lambda_scc=10`, `lambda_env=1`, `lambda_eikonal=1`, `lambda_if_normal=1`, then `curv` and `div` from 500/600.

**Assessment:** Interface and SCC dominate; that’s reasonable for fitting geometry and topology. Eikonal and normals are important for a good SDF; keeping them non-zero is good. Curv/div delayed until after geometry is fitting is sensible.

**Risks:**
- SCC (persistent homology) can be noisy or expensive; if it dominates gradients, training can be unstable.
- If curvature kicks in too early (before interface is close to zero), it can smooth away detail (e.g. studs).

**Recommendations:**
- Keep `start_curv` at 500 (or 300 if shapes are already good).
- If training is noisy, try lowering `lambda_scc` slightly (e.g. 5) or checking PH grid resolution.
- Monitor interface vs envelope vs eikonal vs scc loss magnitudes; if one is orders of magnitude larger, consider rebalancing.

---

## 6. Constraint coverage (studs and walls)

**Setup:** Interface points on stud tops and sides; wall constraints on the six faces; envelope inside/outside.

**Assessment:** The set of constraints is in principle sufficient: SDF=0 and normals on interface + walls, SDF>0 outside, eikonal in domain. LEGO geometry is well-defined.

**Risks:**
- Studs are small; if `n_points_interfaces` is spread over many studs, each stud gets few points and the network might underfit there.
- Wall “top” excludes stud footprints; that’s correct but requires enough samples on the remaining top face.

**Recommendations:**
- Ensure `n_points_interfaces` (and normals) are high enough that each stud gets a reasonable number of points (e.g. hundreds per stud).
- In the notebook, inspect constraint point density on a few bricks; if some regions are sparse, increase sampling or add targeted constraints.

---

## 7. Bounds and helpers when switching problem

**Setup:** `_sync_problem_dependents()` updates plotter, shape_boundary_helper, ph_manager, and cached_connectivity_loss with `self.problem.bounds` (and problem) when switching (N, ny, h).

**Assessment:** Training step uses one problem per batch and syncs before the step; surface points, PH grid, and connectivity loss all use the current problem’s bounds. So **training** is consistent.

**Caveat:** Validation still uses one problem’s bounds for all z_val rows (see §1).

---

## 8. What “success” looks like

- **Realistic success:** The network learns an SDF that, for each (N, ny, h), is close to 0 on interface/walls, positive outside, with roughly unit gradient. Meshes extracted at the right bounds look like the corresponding LEGO bricks (1x2, 2x2, 4x4, different heights). Some studs or edges may be slightly off; connectivity (one component) holds.
- **Failure modes:** (a) Only the “first” brick type (used in validation) converges. (b) Studs disappear or merge (underfitting / over-smoothing). (c) Multiple components or holes (SCC/connectivity not satisfied). (d) Validation looks good but other brick types are wrong (validation bug + single-key validation).

---

## 9. Recommended order of actions

1. **Fix validation:** Ensure validation uses the correct bounds (and optionally problem) for each row of `z_val`, or validate only one conditioning and document it.
2. **Run a smaller setup first:** e.g. 2 brick types (1x2 and 2x2), one height; confirm training and validation both improve and meshes look correct.
3. **Widen z interval** to e.g. `[0, 1]` and optionally add lightweight logging of loss by (N, ny, h).
4. **Add validation for 2–3 other brick types** (after fixing the bug) so you can see if 2×2 and 4×4 also converge.
5. **Scale up** to full N, ny, h once the reduced setup is stable.

---

## 10. Conclusion

Training **can** succeed: the constraint set and loss schedule are plausible, and the trainer correctly switches problem and syncs bounds for **training**. The main blocker to **trustworthy** success is the **validation bug** (wrong bounds for most z_val rows) and the **lack of visibility** into other brick types. Fix validation and add a small, observable setup first; then expand.
