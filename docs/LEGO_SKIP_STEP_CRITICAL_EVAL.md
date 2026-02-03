# Critical Evaluation: Skip Step When Surface Points Fail

Does the "skip backward when get_surface_pts fails" fix actually work?

---

## Flow (when surface points fail)

1. **Closure runs:** `opt.zero_grad()`, then `losses_and_backward(z, epoch, ...)`.
2. **Inside losses_and_backward:**  
   - `get_surface_pts(z, ...)` is called (epoch ≥ start_div) → returns `(None, None)`.  
   - `p_surface = None`, `loss_dict` is built (div skipped; if, env, eikonal, scc, etc. still computed).  
   - `compute_loss_and_save_sublosses(loss_dict)` fills `loss_weighted_dict` and returns total loss.  
   - `skip_step = True` (epoch ≥ start_div, p_surface is None, div in loss keys).  
   - We **do not** call `loss_calculator.backward(field_dict)`.  
   - We set `self._last_step_skipped = True`.  
   - We return `(loss, get_dicts())`.
3. **Closure returns** the real `loss` (for logging). Gradients were zeroed at the start and never set (no backward).  
4. **opt.step(closure):** Optimizer runs the closure once (Adam). No second backward on the return value. Parameters are updated with current gradients → which are zero → **no parameter change**.  
5. **After opt_step:** Violation buffer is updated (same model, new violation scores for this batch).  
6. **adaptive_update():** We **skip** it when `_last_step_skipped` is True, so ALM multipliers (μ, ν, λ) are **not** updated on skip epochs. So they don’t grow while we’re not stepping.

**Verdict:** For **Adam**, the fix is correct: no backward → no gradients → step adds zero → model unchanged. ALM not updated on skip → no multiplier blow-up.

---

## What could still go wrong

1. **LBFGS:** If the optimizer were LBFGS, the closure can be called multiple times per step and is expected to perform backward. Skipping backward might leave the optimizer in an inconsistent state. **LEGO uses Adam**, so this is not an issue for the current config.

2. **Many consecutive skips:** If surface points fail for many epochs in a row (e.g. same brick type and hard z’s), we skip many steps and training stalls. The violation buffer still gets updated each epoch, so we keep re-scoring the same (z, c). Eventually a new batch (different type or z’s) may get surface points and we step again. So we don’t collapse, but we can have long “idle” stretches. Mitigation: higher `random_fraction` or delayed `start_div` if this happens often.

3. **First step after long skip:** When we finally take a step after many skips, the model and batch are unchanged from the last *actual* step. ALM multipliers were not updated during skips, so they’re the same as before the skip run. So the first step back is a normal step, not an oversized one. (We fixed the “ALM grows during skips” by skipping adaptive_update.)

4. **grad_norm_post_clip:** When we skip backward, gradients are zero, so `compute_grad_norm(model.parameters())` in the closure will be 0. That’s correct and logged as such.

5. **Violation buffer:** We still call `_compute_violation_for_z_c` and `update_violations` after the step. So the batch we just “skipped” gets new violation scores under the **same** model (we didn’t update). That’s correct: we’re recording how bad those (z, c) still are. Next time we sample for that type, we may pick different z’s.

---

## Edge cases

- **epoch &lt; start_div:** We don’t need surface points; `get_surface_pts` may not be called. `p_surface` can stay `None` from init. Then `skip_step = (epoch >= start_div) and ...` is False (epoch &lt; start_div). So we don’t skip. Correct.

- **p_surface None but epoch &lt; start_div:** Same as above; we don’t skip. Correct.

- **div not in loss_keys:** If diversity were disabled, `skip_step` would be False (third condition fails). We’d never skip. Correct.

- **ManualWeightedLoss:** We still set `_last_step_skipped` and skip `adaptive_update()` when we skipped the step. For ManualWeightedLoss, `adaptive_update()` is a no-op anyway, so skipping it is harmless.

---

## Summary

- **Backward:** Correctly skipped when surface points fail → no gradients → no parameter update.  
- **ALM:** Correctly not updated on skip epochs → no multiplier growth during idle periods.  
- **Logging / violation buffer:** Unchanged; still use real losses and update violations.  
- **Optimizer:** Safe for Adam; LBFGS not used for LEGO.

The fix should work for the current LEGO setup. Remaining risk is long stretches of skips (training stall) rather than collapse; that can be addressed with scheduling or buffer settings if it shows up.
