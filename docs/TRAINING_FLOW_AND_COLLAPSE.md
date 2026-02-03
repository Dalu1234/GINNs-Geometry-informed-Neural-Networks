# Training flow and “surface points failed” collapse

This doc traces how training works and why you see **“No points found below the level set”** followed by **“Surface points failed — skipping backward”** and all printed losses going to 0.

---

## 1. Per-step training flow

### 1.1 `opt_step` (train/opt/opt_util.py)

- Calls `closure()` and then `opt.step(closure)`.
- `closure()`:
  1. `opt.zero_grad()`
  2. `loss, loss_dicts = loss_and_backward_fn(z, epoch, batch, z_corners)` → **`losses_and_backward`**
  3. Updates `log_dict` from `loss_dicts` (used for progress bar and wandb).
  4. For Adam: backward is **not** done inside `opt_step`; it is done inside `losses_and_backward`.

So the progress bar shows whatever `losses_and_backward` returns via `get_dicts()`.

### 1.2 `losses_and_backward` (train/ginn_trainer.py)

1. **Surface points**
   - If div/curv/chamfer_div need surface points and `surf_needs_update`:
     - `self.p_surface, self.weights_surf_pts = self.shape_boundary_helper.get_surface_pts(...)`  
       (or `NumericalBoundaryHelper.get_surface_pts` when `do_numerical_surface_points: True`).
   - If that fails → `get_surface_pts` returns `(None, None)` → **`self.p_surface = None`**.

2. **Losses**
   - Chamfer div: uses `p_surface` if not None.
   - Scalar losses (if, env, if_normal, eikonal, scc, div, …):  
     `loss_dict[key] = self.loss_dispatcher[key.base_key](..., p_surface=self.p_surface, ...)`.
   - **Div loss** (train/losses_ginn.py): if `p_surface is None` and `diversity_pts_source == 'surface'`, it logs **“No surface points found - skipping diversity loss”** and returns a **zero tensor** (so `loss_dict['div'] = 0`).

3. **Total loss**
   - `loss = self.loss_calculator.compute_loss_and_save_sublosses(loss_dict)`  
     → fills `loss_weighted_dict` / `loss_unweighted_dict` (used by `get_dicts()` and thus the progress bar).

4. **Skip step**
   - `skip_step = (epoch >= start_div) and (self.p_surface is None) and (LossKey('div') in self.all_loss_keys)`.
   - If `skip_step`:
     - Logs **“Surface points failed (epoch >= start_div) — skipping backward”**.
     - **Does not** call `self.loss_calculator.backward(field_dict)`.
   - So when surface points fail and div is active, **no gradients are computed and the optimizer step adds nothing** (model stays frozen).

5. **Return**
   - `return loss, self.loss_calculator.get_dicts()`  
     → progress bar uses these dicts; they are the **current** weighted/unweighted losses for this step.

---

## 2. Where “No points found below the level set” comes from

- **Only** from **numerical** surface points:
  - `GINN/numerical_boundary.py`: `find_boundary_points_numerically_with_binsearch`:
    - Builds a grid, evaluates the network (SDF/density) on it.
    - Looks for grid cells where the field **crosses** the level set (one side &gt; level set, one side &lt;).
    - If **no such crossing** → `inside_boundary_mask.any()` is False → prints **“No points found below the level set.”** and returns `False, (None, None)`.
  - `GINN/numerical_boundary_helper.py`: `get_surface_pts` calls that function; on `False` it returns `(None, None)` → trainer sets `self.p_surface = None`.

So that message means: **on the numerical grid used for surface points, the zero level set does not cross any cell** (entire grid is on one side of the level set). So the “surface” is missing on that grid (collapse or drift).

- If you use **flow-based** surface points (`do_numerical_surface_points: False`), you use `ShapeBoundaryHelper`; it does **not** call `find_boundary_points_numerically_with_binsearch`, so you would **not** see “No points found below the level set” from that path (you could still get “No surface points found” from the flow failing).

---

## 3. Why printed losses go to 0 after collapse

- **When `skip_step` is True:**  
  Backward is skipped but **losses are still computed** and `compute_loss_and_save_sublosses(loss_dict)` still runs, so `get_dicts()` still returns the **current** weighted losses for this step.

- So the progress bar reflects **this step’s** loss values. If they are 0.0 for if/env/if_normal/eikonal/scc, then for **this batch of z** the current (frozen) model is actually giving:
  - Interface value = level_set at sampled interface points → **if** = 0.
  - Envelope and normal constraints satisfied at sampled points → **env**, **if_normal** = 0.
  - Eikonal and SCC satisfied as computed → **eikonal**, **scc** = 0.

- **Div** is 0 by design when `p_surface is None` (diversity loss skipped).

So the 0.0 losses are **consistent with**:  
- Model has **collapsed** (e.g. constant output = level_set, or SDF such that the **numerical** grid sees no crossing).  
- At the **sampled** interface/envelope points used for if/env/if_normal, the model still matches the constraints (so those losses are 0).  
- On the **different** numerical surface-point grid, there is no level-set crossing → “No points found below the level set” and `p_surface = None` → skip backward → model never gets a gradient to fix the collapse.

So: **same model, new z each step; geometry losses can be 0 at the constraint samples while the level set has disappeared on the surface-point grid.**

---

## 4. End-to-end “what’s going on”

1. **Before collapse (e.g. step 2000)**  
   - Surface points are found (grid has a level-set crossing).  
   - All losses (if, env, if_normal, eikonal, scc, div) are computed and backward runs.  
   - Model updates.

2. **First step when numerical surface fails (e.g. 2001)**  
   - New z (and possibly new problem batch) are used.  
   - `get_surface_pts` uses **numerical** method → `find_boundary_points_numerically_with_binsearch` finds **no** crossing → “No points found below the level set.” → `get_surface_pts` returns `(None, None)` → `self.p_surface = None`.  
   - Div loss is skipped (log: “No surface points found - skipping diversity loss”).  
   - Other losses are still computed; for this z and the **current** model they can be 0 (as above).  
   - `skip_step` is True → “Surface points failed (epoch >= start_div) — skipping backward” → **no backward** → **no parameter update**.

3. **Every following step**  
   - Model stays **frozen** (no gradients, no update).  
   - Each step: new z, surface points still fail on the numerical grid, backward still skipped, printed losses can stay 0 (same collapsed model, constraint samples still satisfied).

So training is **stuck**: the level set has collapsed or drifted on the surface-point grid, and the trainer deliberately skips the backward pass, so the model never receives a gradient to recover.

---

## 5. Config and code reference

- **Skip condition** (train/ginn_trainer.py, ~645–651):
  - `skip_step = (epoch >= start_div) and (self.p_surface is None) and (LossKey('div') in self.all_loss_keys)`.
- **start_div** (e.g. configs/GINN/lego_1xN_wire.yml): `start_div: 2000` → from epoch 2000 onward, if surface points fail, backward is skipped.
- **Surface point method**: `do_numerical_surface_points` (base vs LEGO) decides NumericalBoundaryHelper vs ShapeBoundaryHelper; only the numerical path produces “No points found below the level set.”

---

## 6. What to change (options)

1. **Avoid collapse earlier**
   - Increase **lambda_if** / **lambda_env** so the interface and envelope are pinned more strongly.
   - Delay turning on div: increase **start_div** so geometry (if, env, if_normal, eikonal) is stable before surface points are required.

2. **Recover after collapse**
   - Resume from a checkpoint **before** the first “No points found below the level set” (e.g. before 2000), with stronger if/env or later start_div.

3. **Allow recovery without div**
   - In `ginn_trainer.py`: when surface points fail, still run **backward** for the losses that don’t need `p_surface` (if, env, if_normal, eikonal, scc), and only **zero out or omit** the div term in the backward. That way the model can get gradients from geometry and can potentially recover the level set on the grid. (Right now, skipping the whole backward prevents any recovery.)

4. **Numerical vs flow**
   - If you see “No points found below the level set”, you are using **numerical** surface points. You can try **flow-based** surface points (`do_numerical_surface_points: False`) so surface points don’t depend on grid crossings; different failure mode (flow can fail to converge) but no “below the level set” message from the numerical grid.

This file is the single place that documents this flow and the cause of the collapse + zero losses.
