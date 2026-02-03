# Inherited GINN / base_config: What to Be Wary Of

LEGO config (`configs/GINN/lego_1xN_wire.yml`) is merged **over** `configs/base_config.yml`: anything LEGO doesn’t set is inherited from base. Base was written for SimJEB / topology / smooth shapes, not LEGO. Below: inherited settings that can affect LEGO, **for** / **against** / **evaluation**.

---

## 1. **adaptive_penalty** (ALM)

**Inherited:** `gamma: 0.01`, `epsilon: 1.0e-8`, `alpha: 0.9`  
**Used when:** `use_augmented_lagrangian: True` (inherited).

| For | Against |
|-----|--------|
| ALM adapts constraint weights; can help satisfy interface/env/eikonal/SCC without hand-tuning lambdas. | Tuned for other problems; gamma/alpha may not suit LEGO (e.g. interface drift). |
| Same setup as SimJEB; proven in this codebase. | If interface keeps drifting, manual weights (`use_augmented_lagrangian: False`) plus higher `lambda_if` may be more predictable. |

**Evaluation:** Keep as-is for now. If interface (mu_if) keeps drifting after `objective: 'null'` and higher `lambda_if`, try **ManualWeightedLoss** (`use_augmented_lagrangian: False`) and fix lambdas by hand.

---

## 2. **use_augmented_lagrangian: True**

**Inherited:** from base.

| For | Against |
|-----|--------|
| Adapts penalty (lambda/mu) per constraint; less need to tune each lambda. | “mu_if” in logs is the ALM multiplier, not the raw interface violation; can be confusing. |
| Works for SimJEB and others. | Interface can still drift if effective weight on interface is too low. |

**Evaluation:** Keep. Already addressed interface drift with `objective: 'null'` and higher `lambda_if`. If drift persists, consider switching to manual weights.

---

## 3. **curvature_expression: '4*H**2 - 2*K'**

**Inherited:** from base (E_strain from elasticity paper).

| For | Against |
|-----|--------|
| Standard curvature regularizer (mean/double-mean curvature); smooths surface. | Formula comes from elasticity, not LEGO; LEGO just needs “smooth where appropriate.” |
| Reduces noisy zero-level sets. | Studs and edges are curved; strong curvature penalty can still smooth away detail if over-weighted. |

**Evaluation:** Keep. LEGO already uses `objective: 'null'` and curvature only as regularizer (`lambda_curv`, `start_curv: 500`). No need to change unless studs get over-smoothed.

---

## 4. **curvature_pts_source: surface**

**Inherited:** from base.

| For | Against |
|-----|--------|
| Curvature evaluated on surface (zero-level set) is the right place for a surface regularizer. | Surface points are computed numerically; if surface is wrong early on, curvature is wrong too. |
| Matches SimJEB / GINN usage. | Slight extra cost (need surface points). |

**Evaluation:** Keep. Correct choice for LEGO.

---

## 5. **exclude_surface_points_close_to_interface_cutoff**

**LEGO overrides:** `0.04` (base: `0.025`).

| For (0.04) | Against (0.04) |
|------------|----------------|
| Excludes points near interface from curvature; avoids penalizing stud/wall curvature. | Larger cutoff = fewer points used for curvature; slightly weaker smoothness. |

**Evaluation:** LEGO’s 0.04 is good: more interface margin so curvature doesn’t fight studs. No change needed.

---

## 6. **surface.** (numerical surface points)

**Inherited:** base has `do_numerical_surface_points: True`, many `surf_pts_*`, `bin_search_steps`, etc.  
**LEGO overrides:** `do_numerical_surface_points: False`, plus `nx`, `n_points_surface`, `level_set`, `equidistant_init_grid`, `do_uniform_resampling`, `recompute_every_n_epochs`.

| For (LEGO: False) | Against |
|-------------------|--------|
| LEGO uses analytic interface (studs/walls); no need for numerical surface flow. | Any inherited `surf_pts_*` from base are unused when False but still in config (no harm). |
| Faster and more stable for LEGO. | — |

**Evaluation:** Correct. LEGO should keep `do_numerical_surface_points: False`.

---

## 7. **diversity_pts_source: surface**

**Inherited:** base has `surface`. LEGO doesn’t override.

| For | Against |
|-----|--------|
| Diversity on surface points encourages different shapes in latent space. | Surface points depend on current SDF; early on they can be noisy. |
| Same as other GINN configs. | With `start_div: 2000`, diversity only matters later when surface is already reasonable. |

**Evaluation:** Keep. Fine for LEGO.

---

## 8. **ph.** (SCC / persistence homology)

**Inherited:** base has `ph_loss_sub0_points: False`, `ph_loss_super0_points: False`, `ph_max_num_workers: 16`, `simjeb_root_dir`.  
**LEGO overrides:** full `ph:` block (nx, problem_str, ginn_bsize, is_density, iso_level, ph_1_hole_level, ph_loss_maxdim, ph_loss_target_betti, scc_penalty_norm_eps, scc_n_grid_points, evaluate_ph_grid_per_shape).

| For | Against |
|-----|--------|
| LEGO’s ph block is tailored (target Betti [1,0,0], etc.). | Merge still pulls in base keys not in LEGO block (e.g. simjeb_root_dir); unused for LEGO but harmless. |
| SCC is needed for connectivity. | — |

**Evaluation:** LEGO overrides are correct. No change.

---

## 9. **use_scheduler: False**

**LEGO sets:** explicitly False (base: False).

| For | Against |
|-----|--------|
| Constant LR is simple; no risk of decaying too early. | Late in training, smaller LR could help refine interface without drifting. |

**Evaluation:** Optional improvement: set **`use_scheduler: True`** in LEGO so LR decays (e.g. `scheduler_gamma: 0.5`, `decay_steps: 5000`) and late training is more stable. Not required if training is already stable.

---

## 10. **training_mode: 'single'**

**Inherited:** from base.

| For | Against |
|-----|--------|
| Fixed z per epoch; good for multi-shape conditioning (one z per brick type). | If you wanted “sample random z every batch” you’d use `generative`; LEGO is conditioned, so single is correct. |
| Matches LEGO design (condition on N, M, H). | — |

**Evaluation:** Keep. Correct for LEGO.

---

## 11. **field_losses**

**Inherited:** base has `field_losses: ['comp', 'vol', 'chamfer_div', 'wasserstein_div']`. LEGO doesn’t override.

| For | Against |
|-----|--------|
| LEGO sets `lambda_comp: 0`, `lambda_vol: 0`, etc., so these terms are inactive. | Unused keys in config; slightly confusing. |
| No effect on LEGO loss. | — |

**Evaluation:** Harmless. Optional: in LEGO set `field_losses: []` to make it explicit that no field losses are used.

---

## 12. **weight_rescale_on: False**

**Inherited:** from base.

| For | Against |
|-----|--------|
| No automatic rescaling of lambdas; you have full control. | If losses are very imbalanced, rescaling could help; usually not needed. |

**Evaluation:** Keep False for LEGO.

---

## 13. **grad_clipping** (auto_clip_*)

**Inherited:** base has `auto_clip_percentile: 0.8`, `auto_clip_hist_len: 50`, `auto_clip_min_len: 50`. LEGO overrides only `grad_clipping_on`, `grad_clip`, `auto_clip_on`.

| For | Against |
|-----|--------|
| Auto-clip limits gradient spikes; good for stability. | Percentile/hist_len are generic; could be tuned for LEGO if you see instability. |
| Same as rest of GINN. | — |

**Evaluation:** Keep. No need to change unless you see training instability.

---

## 14. **seed: 11**

**Inherited:** from base. LEGO doesn’t override.

| For | Against |
|-----|--------|
| Reproducibility. | You might want a different seed per run. |
| Fine for LEGO. | — |

**Evaluation:** Keep or override per run via CLI. No LEGO-specific issue.

---

## 15. **model.group_size_fwd_no_grad**

**Inherited:** base has `${eval:2*1024*1024}`. LEGO doesn’t set it.

| For | Against |
|-----|--------|
| Used for batched no-grad forward (e.g. validation); 2M is a safe batch size. | LEGO uses `max_points_per_batch: 32768` for meshing; group_size is separate. |
| No harm. | — |

**Evaluation:** Keep. No change.

---

## 16. **metrics.** (shape_metrics_every_n_epochs, etc.)

**Inherited:** base has very large values (e.g. 1000000) so metrics are effectively off by default.

| For | Against |
|-----|--------|
| Avoids expensive validation metrics every epoch. | If you want validation metrics (e.g. Chamfer) periodically, override in LEGO. |
| Fine for LEGO. | — |

**Evaluation:** Keep unless you explicitly want more frequent validation metrics.

---

## 17. **plot.fig_wandb, plot_2d_resolution**

**Inherited:** base has `fig_wandb: True`, `plot_2d_resolution: 100`. LEGO overrides part of `plot` but not these.

| For | Against |
|-----|--------|
| Wandb logging of figures; 2D resolution for any 2D plots. | LEGO is 3D; plot_2d_resolution rarely matters. |
| Harmless. | — |

**Evaluation:** Keep. No LEGO-specific issue.

---

## 18. **FEM, data, n_points_data**

**Inherited:** base has FEM block, data dir, n_points_data: -1.

| For | Against |
|-----|--------|
| LEGO doesn’t use FEM or data loss; lambda_data is 0. | Dead config for LEGO; could be confusing. |
| No effect on LEGO. | — |

**Evaluation:** Ignore. No need to remove.

---

## Summary: What to change vs leave as-is

| Item | Action |
|------|--------|
| **objective** | Already set to `'null'` in LEGO (was curvature). |
| **adaptive_penalty, use_augmented_lagrangian** | Keep; revisit only if interface keeps drifting. |
| **curvature_expression, curvature_pts_source** | Keep. |
| **exclude_surface_points_close_to_interface_cutoff** | LEGO already 0.04; keep. |
| **surface.do_numerical_surface_points** | LEGO False; keep. |
| **diversity_pts_source, ph, training_mode** | Keep. |
| **use_scheduler** | Optional: set True in LEGO for late-training stability. |
| **field_losses** | Optional: set `[]` in LEGO for clarity. |
| **weight_rescale, grad_clipping, seed, model.group_size, metrics, plot** | Keep as-is. |
| **FEM, data** | Ignore for LEGO. |

**Main takeaway:** LEGO already overrides the worst fits (mesh_reduction, mc_resolution, objective, start_div, lambda_if, lambda_scc, meshing, ph, surface). The remaining inherited settings are either benign or worth keeping. The only optional tweaks are **use_scheduler: True** and, if you want clarity, **field_losses: []**.
