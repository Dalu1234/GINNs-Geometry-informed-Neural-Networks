# GINN Training Process (LEGO 1xN / cond_wire)

This document describes how training is set up and executed for the LEGO 1xN conditioned wire model.

---

## 1. Entry point and config

- **Entry:** `run.py` → `main()`  
  - Parses CLI: `key=value` pairs and **required** `yml=...`, `gpu_list=...`
  - Loads config: `configs/base_config.yml` + `configs/<yml>` (e.g. `configs/GINN/lego_1xN_wire.yml`), with CLI overwrites
  - Sets `CUDA_VISIBLE_DEVICES`, logging, seeds, wandb, then creates `Trainer(config, ...)` and calls `trainer.train()`

- **Config (LEGO):**  
  - Model: `cond_wire`, `nx=3`, `nz` = 2 + (condition_on_n_studs + condition_on_n_studs_y + condition_on_height)  
  - Problem: `problem_str: lego_1xN`, with `n_studs`, `n_studs_y`, `height_scale`  
  - Conditioning: `n_studs_values`, `n_studs_y_values`, `height_values` define brick types; each type gets a `ProblemLego1xN` instance in `trainer.problems[key]`

---

## 2. Trainer initialization (`ginn_trainer.py`)

- **Model:** `get_model(**model_kw)` → `NetWithPartials`; `nz` is expanded by +1 for each of condition_on_n_studs, condition_on_n_studs_y, condition_on_height.
- **Problems:** One `ProblemLego1xN` per `(n_studs, n_studs_y, height)`; each provides:
  - `sample_from_domain()`, `sample_from_envelope()`, `sample_from_interface()`, etc.
  - `bounds`, `is_inside_envelope`, envelope/interface/wall constraints
- **Helpers:**  
  - `ShapeBoundaryHelper` / `NumericalBoundaryHelper` for surface points (curvature, diversity).  
  - `PHManager` (persistent homology) for SCC loss when `lambda_scc > 0`.  
  - `ViolationBuffer` when `violation_buffer.use` and conditioning are on: stores (z_base, c_key, violation) and samples (z, c) by violation for prioritized training.
- **Loss:**  
  - `get_loss_keys_and_lambdas(config)` builds scalar/field loss keys and lambdas from config (`lambda_*`).  
  - `ManualWeightedLoss` or `AdaptiveAugmentedLagrangianLoss` (if `use_augmented_lagrangian`).  
  - `init_loss_dispatcher()`: map each loss name to a partial (e.g. `loss_if`, `loss_env`, `loss_eikonal`, `loss_scc`, `loss_if_normal`, …) with `netp`, `problem`, config args fixed.
- **Optimizer:** `get_opt(config['opt'], config, model.parameters())` (e.g. Adam). Optional scheduler. Optional load from checkpoint.
- **Latents:** `z = z_sampler.train_z()` (shape `[ginn_bsize, nz]`); `z_val` for validation plots; `z_corners` only if not generative.

---

## 3. Per-epoch loop (`train()`)

Each **epoch** does the following in order.

### 3.1 Validation (every `valid_every_n_epochs`)

- Set `problem` to the first brick type (or iterate types for 3D).
- `model.eval()`; for 3D LEGO, build one mesh per brick type from `z_val` rows, pass to plotter.
- Optionally run shape metrics (`meter.get_average_metrics_as_dict`) on validation meshes.

### 3.2 Training batch: (z, c) and problem

- **With violation buffer:**  
  - Pick a random `c_key` (brick type) each step (or by `block_epochs_per_type`).  
  - `violation_buffer.sample_batch(c_key, ginn_bsize, temperature, random_fraction)` → `z_bases`, `indices`.  
  - Append conditioning columns (n_studs, n_studs_y, height normalized) to get `z`; set `self.problem = self.problems[c_key]` and `_sync_problem_dependents()`.
- **Without violation buffer:**  
  - Optionally resample `z` (e.g. every epoch if `reset_zlatents_every_n_epochs`).  
  - Sample brick type (N, ny, h), set `problem` and append conditioning columns to `z`.

So each step has a single **active problem** (one brick type) and a batch of **z** (shape latents + conditioning).

### 3.3 Plots (every `plot_every_n_epochs`)

- Train boundary plot (mesh per shape from `z_plot`), PH diagram if `lambda_scc > 0`, etc.

### 3.4 Loss computation and optimizer step

- **`opt.zero_grad()`** then **`opt_step(opt, epoch, model, loss_and_backward_fn=self.losses_and_backward, z=z, ...)`**.
- Inside `opt_step`, **`closure()`** is called:
  1. `losses_and_backward(z, epoch, batch, z_corners)`:
     - **Surface points:** If curvature/diversity/chamfer_div need them and `recompute_every_n_epochs` → `shape_boundary_helper.get_surface_pts(z, ...)` → `p_surface`, `weights_surf_pts`.
     - **Chamfer div** (if active): uses `p_surface`, dispatcher `chamfer_div` → scalar + field for backward.
     - **TO/FEM** (if `lambda_comp` / `lambda_vol`): evaluate net on FEM grid, heaviside, get compliance/volume and gradients.
     - **Scalar losses:** For each key in `scalar_loss_keys`, call `loss_dispatcher[key.base_key](z=z, ...)`:
       - **if:** interface loss (SDF at interface pts → level_set).  
       - **env:** envelope loss (SDF at envelope pts).  
       - **if_normal:** normal loss at interface (gradient vs target normals).  
       - **eikonal:** gradient norm 1 in domain.  
       - **scc:** `ph_manager.calc_ph_loss_cripser(z)` → persistent homology loss (single connected component).  
       - **curv / div:** only if `epoch >= start_curv` / `start_div`; curv uses surface pts, div uses diversity over shapes.
     - **Combine:** `loss_calculator.compute_loss_and_save_sublosses(loss_dict)` → weighted total (or ALM); then **`loss_calculator.backward(field_dict)`** (or `backward_config`).
  2. **Gradient clipping** (e.g. AutoClip) on `model.parameters()`.
  3. **`opt.step(closure)`** (or LBFGS/NNCG step if switched).
- If **diversity** is on but **surface points failed** (`p_surface is None`), backward is skipped for that step and `adaptive_update` is skipped.

### 3.5 Post-step updates

- **Violation buffer:** For each (z_i, c_key) used in the batch, compute `_compute_violation_for_z_c(z_i, c_key)` = interface + envelope loss (no_grad); update buffer entries at the returned indices with these violation values.
- **ALM:** If using `AdaptiveAugmentedLagrangianLoss` and step was not skipped, `loss_calculator.adaptive_update()` (update μ, λ from current constraint violations).
- **Logging:** Merge loss dict, grad norm, lr, etc. into `cur_log_dict` → `log_history_dict[epoch]`, `log_to_wandb(epoch)`, progress bar update.
- **Checkpointing:** `save_model_every_n_epochs(model, opt, sched, config, epoch)`.

---

## 4. Loss roles (LEGO)

| Loss       | Role |
|-----------|------|
| **if**    | Fit SDF to interface (stud tops, walls) at `level_set`. |
| **env**   | Keep SDF positive outside shape (points on envelope). |
| **if_normal** | Align SDF gradient with target normals at interface (sharp edges, flat studs). |
| **eikonal**   | Enforce ‖∇f‖ = 1 in domain (valid SDF). |
| **scc**   | Persistent homology penalty so level set has one connected component (single mesh). |
| **curv**  | Optional; regularize curvature from `start_curv`. |
| **div**   | Optional; encourage latent diversity from `start_div`. |

Lambdas and start epochs come from config (e.g. `lambda_if`, `lambda_env`, `start_curv`, `start_div`).

---

## 5. Data flow summary

```
run.py
  → config (YAML + CLI)
  → Trainer(config, mp_manager, mp_pool_scc, mp_top)
  → train()
       for epoch in max_epochs:
         [validation + plots if due]
         z, problem = sample (violation buffer or random z + random brick type)
         opt.zero_grad()
         closure:
           loss, dicts = losses_and_backward(z, epoch, ...)
             → problem.sample_*() for points
             → netp(x, z) and netp.vf_x(x, z) for SDF and gradients
             → loss_if, loss_env, loss_if_normal, loss_eikonal, loss_scc, ...
             → loss_calculator.compute_loss_and_save_sublosses(loss_dict)
             → loss_calculator.backward(field_dict)
           clip grad, opt.step(closure)
         violation_buffer.update_violations(...)
         loss_calculator.adaptive_update()  # if ALM
         log_to_wandb(); save_model_every_n_epochs()
```

---

## 6. Key files

| File | Role |
|------|------|
| `run.py` | Entry; config, device, wandb, Trainer, train(). |
| `util/get_config.py` | CLI and YAML merge. |
| `train/ginn_trainer.py` | Trainer init, train loop, loss dispatcher, violation buffer, ALM, logging. |
| `train/opt/opt_util.py` | opt_step, closure, grad clip, get_opt. |
| `train/losses_ginn.py` | loss_if, loss_env, loss_eikonal, loss_scc, loss_if_normal, … |
| `train/train_utils/loss_calculator_alm.py` | AdaptiveAugmentedLagrangianLoss (weighted + μ, λ updates). |
| `train/train_utils/loss_keys.py` | LossKey, get_loss_keys_and_lambdas. |
| `train/train_utils/violation_buffer.py` | ViolationBuffer: (z_base, c_key, violation), sample_batch, update_violations. |
| `GINN/problems/problem_lego_1xN.py` | LEGO problem: envelope, interfaces, walls, sampling. |
| `GINN/ph/ph_manager.py` | calc_ph, calc_ph_loss_cripser (SCC). |

This is the full training process from `run.py` to a single optimizer step and logging.
