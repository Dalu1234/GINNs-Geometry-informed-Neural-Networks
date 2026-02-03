# What Happens at Epoch 2000 (LEGO Training Collapse)

## Summary

At epoch 2000 the run effectively **collapses**: all losses go to 0, "No points found below the level set" every step, and training does nothing useful afterward. The trigger is **diversity turning on** at `start_div: 2000`, which causes the surface-point finder to run for the first time. For the current batch the SDF is already degenerate (no grid straddles the level set), so we get no surface points. We still take an optimizer step with the other losses; that step appears to push the model into a state where it outputs NaN or a globally degenerate SDF, so from epoch 2001 onward all losses are clamped to 0 and the model never recovers.

---

## Timeline

1. **Epoch 1999 and before**  
   Diversity is off (`epoch < start_div`). We never call `get_surface_pts`. Losses (if, env, if_normal, eikonal, scc) are computed and trained on. Things look fine (e.g. if: 0.13, env: 11, …).

2. **Epoch 2000**  
   - `start_div: 2000` → diversity is turned on.  
   - To compute diversity we need surface points, so we call `shape_boundary_helper.get_surface_pts(z, ...)`.  
   - That calls `find_boundary_points_numerically_with_binsearch`. For the **current batch** (some brick type, some z’s from the violation buffer), the SDF is such that **no grid point is “below” the level set** — i.e. the zero-level set doesn’t cross the grid (all positive or all negative in the domain).  
   - So we get: **"No points found below the level set"** → `get_surface_pts` returns `(None, None)`.  
   - Diversity is skipped (`p_surface is None` → loss_div stays 0).  
   - We still compute if, env, if_normal, eikonal, scc and get **real** values (e.g. if: 1.3e-01, env: 1.1e+01, …).  
   - We do **one optimizer step** with those losses (and div=0).  
   - That step can push the model into a bad state (e.g. weights that later produce NaN, or SDF that is degenerate everywhere).

3. **Epoch 2001 onward**  
   - We again call `get_surface_pts` → still no points (SDF still degenerate).  
   - We compute if, env, etc. → the model now outputs **NaN** (or the losses are NaN).  
   - In the ALM we **clamp** NaN/negative scalar losses to 0, so we don’t crash.  
   - The **logged** losses are the clamped values → **all 0.0e+00** in the progress bar.  
   - Total loss is 0 (or negligible), so gradients are 0 and the optimizer doesn’t fix the model.  
   - We stay in this state: every epoch we try surface points → fail; we compute losses → NaN → clamp to 0 → no useful training.

So **epoch 2000** is when we first ask for surface points (because of `start_div`). For that batch the SDF is already bad enough that no surface points are found; we still take a step with the other losses, and that step collapses the model. After that, everything is NaN/0 and we don’t recover.

---

## Why the SDF is degenerate for this batch

- The **batch at 2000** is one brick type and 9 z’s from the violation buffer (prioritized by violation).  
- For that (type, z) combination the network may already be putting the zero-level set outside the grid or on one side only (e.g. all SDF > 0 in the bounding box). That can happen if:
  - That type has had few steps so far (block schedule),
  - Or those z’s are “hard” and the model hasn’t learned them yet,
  - Or the model is close to a bad basin and the next step pushes it over.

Once we request surface points and fail, we still step. That step can make the SDF (or weights) worse so that **all** subsequent batches see NaN or degenerate SDFs.

---

## Implemented fix: skip backward when surface points fail

When `epoch >= start_div` and `get_surface_pts` returns `(None, None)` (no surface points), we **do not call** `loss_calculator.backward(field_dict)`. So no gradients are computed; `opt.step()` then adds zero to the parameters and the model is unchanged. Next epoch we get a new batch; if surface points succeed then we step as usual.

**Important:** Simply zeroing the *returned* loss would **not** work: backward is run inside `loss_calculator.backward()`, which uses `sum(self.loss_weighted_dict.values())`, not the returned scalar. So we must skip the **backward call** itself.

**Caveat:** If surface points fail for many consecutive epochs (e.g. same brick type and bad z’s), we skip many steps and training stalls for that period. That’s acceptable compared to one bad step that collapses the model. If it becomes a problem, consider increasing `random_fraction` in the violation buffer or delaying `start_div` further.
