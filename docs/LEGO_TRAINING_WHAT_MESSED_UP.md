# What Messed Up LEGO GINN Training

Evaluation of what went wrong with the original run (blobby meshes, multiple components, no clear studs/walls) and what was changed.

---

## Note for future readers

**Why was `mesh_reduction: 0.9` there in the first place?**  
It is the **default** in `configs/base_config.yml` (the base used when loading LEGO). Other configs (SimJEB, wheel, TOM) also use 0.9 in their meshing section. The intent was **smaller meshes** for faster validation/plotting and smaller files (e.g. for wandb or saved outputs). For other problems (smoother geometry, 2D beams, wheel), reducing to ~10% of vertices often still gives a recognizable shape. For **LEGO**, which has **fine detail** (studs, sharp edges, flat walls), that simplification wipes out the very features we care about—so the default is wrong for this use case. **Current LEGO config:** `configs/GINN/lego_1xN_wire.yml` **already overrides** to `meshing.mesh_reduction: 0`, so LEGO does **not** use 0.9. The “before” in the summary below refers to the base default or to earlier runs that relied on it. **Takeaway:** For any problem with fine geometric detail, set `mesh_reduction: 0` (and use adequate `mc_resolution`) so extraction does not destroy small features.

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

**Why disconnected blobs?**  
(1) **Geometry satisfied in a degenerate way:** The model can make interface/env losses low by having SDF ≈ 0 at the constraint points while the full zero-level set is **several separate pieces** (each piece passes through some constraints but the pieces are not connected). So loss is low but topology is wrong.  
(2) **Diversity dominating early:** With diversity on at 600, the main pressure was “be different from other latents.” In a shallow geometry basin, that can favor **disconnected** solutions (e.g. one blob per “region”) that are diverse and still satisfy interface/env on the sampled points.  
(3) **SCC too weak:** The connectivity term (SCC, target one component) had weight 10. If diversity or the degenerate geometry gradient was stronger, the net could satisfy interface/env with multiple components and SCC did not pull hard enough to merge them.  
(4) **Extraction:** Aggressive `mesh_reduction: 0.9` can **cut thin connections** (e.g. between stud and body), so a single connected SDF can sometimes be exported as several blobs. So some of the “multiple components” could have been from simplification; the rest from the SDF actually having multiple zero-level components.

---

## 5. **Constraint saturation ≠ good geometry**

**Problem:** Low geometry losses (mu_if, mu_env, mu_scc, etc. ≈ 0) do **not** guarantee correct LEGO geometry. The network can satisfy the losses with a smooth, blob-like zero-level set that passes through the constraint points instead of forming sharp studs and walls. So “constraints saturated” was misleading: they were satisfied in a degenerate way.

**Implication:** Judging training by “all geometry losses near 0” is not enough. You also need to inspect meshes (and optionally SDF slices) to confirm studs and walls.

---

## Summary: what we changed

| Item | Before | After |
|------|--------|--------|
| mesh_reduction | base default 0.9 (LEGO now overrides to 0) | 0 |
| mc_resolution | 128 | 256 |
| start_div | 600 | 2000 |
| lambda_scc | 10 | 20 → **35** (disjoint at 900) |
| lambda_connectivity | (added then removed) | not used; SCC only |
| lambda_if | 10 | 15 → **22** (mu_if drift, rough surface) |
| lambda_if_normal | 1.0 | 1.5 → **2.0** (flatter walls/studs) |
| n_points_interfaces | 4096 | 8192 |
| n_points_normals | 4096 | 8192 |
| max_points_per_batch | (none) | 32768 (for 8GB GPU) |

**CUDA OOM during validation:** With `mc_resolution: 256`, mesh extraction evaluates 256³ points at once, which can exceed 8GB GPU memory. The code now supports `meshing.max_points_per_batch` (e.g. 32768 ≈ 32³): the grid is evaluated in sub-batches so validation meshing stays within GPU memory. Set it in `lego_1xN_wire.yml`; omit or set to `null` if you have enough VRAM.

---

## Other changes to consider (optional)

- **LR scheduler:** Set `use_scheduler: True` so learning rate decays in later epochs (e.g. after 2k); can help fine-tune geometry without big jumps. Config already has `scheduler_gamma: 0.5`, `decay_steps: 5000`.
- **start_curv:** Try `start_curv: 300` or `400` so curvature regularizes a bit earlier; only if you want curvature to kick in before 500.
- **Validation coverage:** Currently validation uses one brick type (key0). A future code change could validate on multiple (n, ny, h) so all brick types are checked; not a config tweak.

---

## What to do next

1. **New run** with the current config (extraction, start_div, lambda_scc as above). Train at least 2k–3k epochs before judging.
2. **Validate by meshes** at 2k, 3k, 5k: do studs and walls appear? Is each mesh one connected component?
3. **If still blob-like:** increase `lambda_if` (and/or `n_points_interfaces`) in a follow-up run; optionally inspect SDF slices to confirm the zero-level set is wrong before changing training again.

**Follow-up at ~900 epochs:** If shapes are rectangular-ish but **disconnected** (multiple components) and **rough / not flat**, bump `lambda_scc` (e.g. 20 → 35) for connectivity and `lambda_if` / `lambda_if_normal` (e.g. 15 → 22, 1.5 → 2.0) for flatter surfaces. Then start a new run or fine-tune from checkpoint.

---

## Why studs weren’t forming by epoch 1000 (past 3 runs)

**Cause:** The base `sample_from_interface()` splits `n_points_interfaces` **evenly** across all interface constraints. In LEGO, `_interface_constraints = [interface_constraint] + wall_constraints` → **1 stud constraint + 6 wall constraints = 7**. So only **1/7** of interface points were on studs (tops + sides); **6/7** were on walls. The interface loss gradient was dominated by walls, so the model fit walls well and studs got too little signal to form.

**Fix:** In `GINN/problems/problem_lego_1xN.py`, **override `sample_from_interface()`** so that **50%** of points go to studs (tops + sides) and **50%** to the 6 walls (divided equally). With 8192 interface points, studs now get 4096 points instead of ~1170, so the interface loss has a strong enough signal on studs to form them.

**Next run:** Use the updated LEGO problem (no config change needed). Train at least 1000–1500 epochs; studs should start appearing earlier. If they still don’t, try increasing the stud share (e.g. 60% studs) or raising `lambda_if` further.

---

## Why geometry regressed (good ~200, then degraded ~700–1000, mu_if/mu_env rising, field collapse)

This section explains **why** the interface and envelope constraint violations rose during training (visible as rising **loss_unweighted_if** / **loss_unweighted_env** in Wandb, and sometimes as rising **mu_if** / **mu_env** depending on ALM dynamics), leading to geometry degrading and eventually “No points found below the level set” / empty mesh.

**What the metrics mean:** In `train/train_utils/loss_calculator_alm.py`, **loss_unweighted_*** is the raw constraint violation (e.g. interface MSE, envelope violation). **mu_*** is the ALM penalty multiplier: `mu = gamma / (sqrt(nu) + epsilon)` where `nu` is a running average of the violation. So when **violation goes up**, `nu` goes up and **mu goes down**; when violation goes down, mu goes up. If you saw “mu_if” or “mu_env” **rising**, that can mean either (a) the **violation** (loss_unweighted) was rising and you were looking at a different panel, or (b) in a later phase violation dropped and mu increased. The important signal for regression is **loss_unweighted_if** and **loss_unweighted_env** rising: that means the zero-level set and envelope fit are getting worse.

### Root causes (from the code)

1. **Curvature loss competing with interface (start_curv: 500)**  
   Once curvature turns on (`train/losses_ginn.py`: `loss_curv`), the loss penalizes high curvature (e.g. `4*H² - 2*K`) at **surface points** (`p_surface` from `shape_boundary_helper.get_surface_pts`). The gradient of this term w.r.t. network parameters moves the SDF, hence the zero-level set—e.g. smoothing bumps—which can **pull the surface away from the interface constraint points**. So the same network step that reduces curvature can increase interface (and envelope) violation. If the effective weight on curvature is large enough, interface fit degrades and violation rises. **LEGO:** `configs/GINN/lego_1xN_wire.yml` has **start_curv: 500** (line 100).

2. **Objective was curvature (`objective: 'curv'`) in earlier runs**  
   With `objective: 'curv'`, the total loss is structured so curvature is the “objective” and the rest are constraints. In `train/train_utils/loss_calculator_alm.py`, the term whose key equals `obj_key` gets **is_objective=True** (line 104: `is_objective = key==self.obj_key`). For that term, **mu_loss = 0** (scalar: lines 51–54; field: 72–74, 91–94)—i.e. the objective does **not** get the extra ALM penalty (0.5 * mu * sub_loss). In **adaptive_update()** (lines 117–120), the objective key is **skipped** (`if key == self.obj_key: continue`), so its mu/lambda are never updated. So the optimizer was effectively “minimize curvature” while other terms got adaptive penalty. If the curvature gradient is strong, steps can prioritize curvature over interface/env, so interface violation can rise. **Fix already applied:** **objective: 'null'** in `lego_1xN_wire.yml` (line 73). With `objective: 'null'`, `obj_key` is `LossKey('null')`; there is no `lambda_null`, so **no** loss term has `is_objective=True`, and **all** terms (if, env, curv, …) get the mu penalty and are updated in adaptive_update.

3. **Diversity turning on too early (start_div: 600 in earlier runs)**  
   If diversity was active from epoch 600, the diversity loss (e.g. “be different from other latents”) becomes a major driver. In a shallow geometry basin, the model can satisfy interface/env on **sampled points** while the full zero-level set drifts (e.g. blobby, disconnected). Diversity then pushes shapes to be more different, which can **increase** interface/envelope violation. **Fix already applied:** `start_div: 2000` so geometry (and curvature) dominate for the first 2k epochs.

4. **Resampling z every epoch (`reset_zlatents_every_n_epochs: 1`)**  
   In `ginn_trainer.py`, when this is set, **z** (and for LEGO, the conditioning N, N_y, height) is resampled every epoch. So each epoch the 9 shapes in the batch are **different**. The network never gets many consecutive steps on the **same** (z, conditioning). Curvature and other losses act on different latent/conditioning combinations each time, so the optimizer can’t stably improve interface fit for a fixed set of shapes. That **instability** can show up as interface/env violation drifting up over time, especially after curvature and (if early) diversity are active.

5. **Per-batch problem switching (LEGO conditioning)**  
   Each training step, a **single** (N, N_y, height) is drawn and `self.problem = self.problems[key]` is set (`ginn_trainer.py`). So each step only gets gradients for **one** brick type. The shared network must satisfy all brick types; the constantly switching constraint set (different interfaces/envelopes per type) can make the combined loss noisier and less stable, and can contribute to interface drift when combined with curvature and resampled z.

6. **Augmented Lagrangian (ALM) dynamics**  
   In `loss_calculator_alm.py`, after each step `adaptive_update()` does:  
   `nu = alpha*nu + (1-alpha)*loss_unweighted`,  
   `mu = gamma / (sqrt(nu) + epsilon)`,  
   `lambda += mu * sqrt(loss_unweighted)`.  
   So when **violation rises**, `lambda_if` / `lambda_env` increase next epoch. In principle that should push the optimizer to fix the constraint. But if the **curvature** (or diversity) gradient is large, the next step can still increase violation; then lambda and nu rise again. So you can get a **runaway**: violation up → lambda up → but curvature/diversity still dominate the step → violation stays high or keeps rising, leading to field collapse. Stronger fixed weights (`lambda_if: 48`, `lambda_env: 2`) and `objective: 'null'` reduce this by making interface/env matter more from the start.

7. **Interface weight was too low in earlier runs**  
   With `lambda_if` in the 10–22 range, the interface term could be dominated by curvature (and diversity if early). So even with `objective: 'null'`, the **effective** weight on “keep surface at constraint” was insufficient and the surface drifted. **Fix:** Increase `lambda_if` (e.g. 48) and `lambda_env` (e.g. 2) so interface and envelope hold the geometry even when curvature and diversity are active.

### Summary: why regression happened

| Cause | Effect | Fix (already or in config) |
|-------|--------|----------------------------|
| Curvature (start_curv 500) | Gradient can move zero-level set away from interface points | Keep curvature; balance with higher lambda_if / lambda_env |
| objective: 'curv' | Curvature explicitly favored over constraints | **objective: 'null'** |
| start_div: 600 | Diversity dominated early; geometry drifted | **start_div: 2000** |
| Resample z every epoch | No stable target shapes; unstable interface fit | Consider reset_zlatents_every_n_epochs > 1 for a phase |
| Per-batch (N, N_y, h) switch | Noisier gradients across brick types | Accept; counter with higher lambda_if/lambda_env |
| ALM + strong curvature/diversity | Violation up → lambda up but step still worsens violation | **lambda_if: 48**, **lambda_env: 2**, objective 'null' |
| Low lambda_if / lambda_env | Interface/envelope too weak vs curvature | **lambda_if: 48**, **lambda_env: 2** (and optionally ManualWeightedLoss if ALM still unstable) |

**What to monitor:** In Wandb, watch **loss_unweighted_if** and **loss_unweighted_env**. If they rise and stay high, interface/envelope are degrading. If they still rise with current fixes, try: (1) higher `lambda_if` / `lambda_env`, (2) `use_augmented_lagrangian: False` with fixed lambdas, or (3) `reset_zlatents_every_n_epochs` > 1 for part of training so the same shapes get more consecutive steps.
