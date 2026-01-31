# What Messed Up LEGO GINN Training

Evaluation of what went wrong with the original run (blobby meshes, multiple components, no clear studs/walls) and what was changed.

---

## Note for future readers

**Why was `mesh_reduction: 0.9` there in the first place?**  
It is a **project-wide default** from `configs/base_config.yml` and `configs/generative_base.yml`, inherited by LEGO and other configs (SimJEB, wheel, TOM, etc.). The intent was **smaller meshes** for faster validation/plotting and smaller files (e.g. for wandb or saved outputs). For other problems (smoother geometry, 2D beams, wheel), reducing to ~10% of vertices often still gives a recognizable shape. For **LEGO**, which has **fine detail** (studs, sharp edges, flat walls), that simplification wipes out the very features we care about—so the default was wrong for this use case. **Takeaway:** For any problem with fine geometric detail, set `mesh_reduction: 0` (and use adequate `mc_resolution`) so extraction does not destroy small features.

---

## 1. **Mesh extraction (major)**

**Problem:** Validation/export used **mesh_reduction: 0.9** (keep 10% of vertices). That aggressively simplified the mesh and wiped out small features (studs, sharp edges), so shapes looked like blobs even when the SDF was partly correct.

**Evidence:** After switching to mc_resolution 256 and mesh_reduction 0, re-extraction from the same checkpoint gave more structured shapes (four brick-like blocks, vertical segmentation) but still granular—so extraction was part of the issue, not the whole story.

**Fix:** `meshing.mesh_reduction: 0`, `meshing.mc_resolution: 256` in `configs/GINN/lego_1xN_wire.yml`. Use these for any validation/export.

---

## 2. **Diversity too early (major)**

**Problem:** **start_div: 600** meant diversity loss turned on early. Geometry losses (if, env, if_normal, eikonal, scc) saturated to ~0 within ~50–100 steps, so after that diversity was the main active term. The model then optimized “diversity” inside a shallow geometry basin and drifted toward smooth, blob-like shapes instead of sharp LEGO geometry.

**Evidence:** Loss curves: mu_if, mu_env, mu_scc, etc. flat near 0 from early on; neg_loss_div the only thing moving after ~600. Meshes: amorphous, no clear studs/walls.

**Fix:** `start_div: 2000` so geometry (and curvature from 500) dominates for the first 2000 epochs; diversity only shapes the latent space after the shape is already correct.

---

## 3. **Interface loss trade-off (minor)**

**Problem:** **mu_if** slowly increased from ~0.05 to ~0.1–0.15 by epoch 1400 while other geometry terms stayed near 0. So the zero-level set (interface) was being slightly degraded as curvature and (when active) diversity pulled the SDF, and interface didn’t have enough weight to hold the surface.

**Evidence:** Dashboard at 1400: mu_if only metric that drifts up; others flat near 0.

**Fix (optional for next run):** If meshes are still too smooth after the other changes, try increasing `lambda_if` (e.g. 12–15) or `n_points_interfaces` so the interface (studs + walls) gets a stronger gradient.

---

## 4. **Connectivity (multiple components)**

**Problem:** Meshes had **multiple connected components** (e.g. 2–6 per shape). SCC (lambda_scc) was at 10 and may have been too weak relative to other terms, or diversity/geometry balance allowed disconnected blobs.

**Fix:** `lambda_scc: 20` (was 10). Rely only on SCC (PH, target Betti [1,0,0]) for connectivity; lambda_connectivity removed per your preference.

---

## 5. **Constraint saturation ≠ good geometry**

**Problem:** Low geometry losses (mu_if, mu_env, mu_scc, etc. ≈ 0) do **not** guarantee correct LEGO geometry. The network can satisfy the losses with a smooth, blob-like zero-level set that passes through the constraint points instead of forming sharp studs and walls. So “constraints saturated” was misleading: they were satisfied in a degenerate way.

**Implication:** Judging training by “all geometry losses near 0” is not enough. You also need to inspect meshes (and optionally SDF slices) to confirm studs and walls.

---

## Summary: what we changed

| Item | Before | After |
|------|--------|--------|
| mesh_reduction | 0.9 | 0 |
| mc_resolution | 128 | 256 |
| start_div | 600 | 2000 |
| lambda_scc | 10 | 20 |
| lambda_connectivity | (added then removed) | not used; SCC only |

---

## What to do next

1. **New run** with the current config (extraction, start_div, lambda_scc as above). Train at least 2k–3k epochs before judging.
2. **Validate by meshes** at 2k, 3k, 5k: do studs and walls appear? Is each mesh one connected component?
3. **If still blob-like:** increase `lambda_if` (and/or `n_points_interfaces`) in a follow-up run; optionally inspect SDF slices to confirm the zero-level set is wrong before changing training again.
