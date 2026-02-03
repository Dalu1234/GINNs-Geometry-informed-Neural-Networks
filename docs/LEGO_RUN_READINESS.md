# Critical Evaluation: Are We Good to Run?

Final checklist before running LEGO GINN training (block training, violation buffer, start_div 2000).

---

## Fixes in place (no more crashes from these)

| Issue | Fix | Where |
|-------|-----|--------|
| Violation buffer NaN probs | Sanitize violations (nan/inf → finite), fallback to uniform probs if wsum bad | `violation_buffer.py` sample_batch, update_violations |
| ALM "Scalar loss should be non-negative" | Clamp scalar losses (nan/neg → 0) before use | `loss_calculator_alm.py` compute_loss_and_save_sublosses |
| One bad step when surface points fail | Skip backward when epoch ≥ start_div and p_surface is None | `ginn_trainer.py` losses_and_backward |
| ALM growing during skip epochs | Skip adaptive_update() when _last_step_skipped | `ginn_trainer.py` after opt_step |
| compute_grad_norm on empty grads | Return 0 when no param has .grad (e.g. after skip backward) | `util/misc.py` compute_grad_norm |

---

## Flow when surface points fail (epoch ≥ 2000)

1. get_surface_pts → (None, None) → "No surface points found - skipping diversity loss".
2. skip_step = True → we do **not** call loss_calculator.backward(field_dict).
3. _last_step_skipped = True.
4. Closure returns; opt.step(closure) runs → gradients are zero → no parameter update.
5. compute_grad_norm sees no grads → returns 0 (no crash).
6. Violation buffer still updated (same model).
7. adaptive_update() **skipped** (so μ/λ don’t grow).

So we don’t crash and we don’t take a bad step; next epoch we try a new batch.

---

## What could still go wrong (non-crash)

1. **Long skip runs**  
   If surface points fail for many epochs in a row (same type, hard z’s), we skip many steps and training stalls. Mitigation: higher `random_fraction` or later `start_div` if it happens a lot.

2. **Wandb/plotting noise**  
   If the model outputs NaN or empty meshes (e.g. at validation), you can get "level not within volume data range", "cannot convert float NaN to integer", wandb.log failed. Training continues; logs/plots may be wrong for those steps. No code change required; just expect some noisy logs.

3. **Dead state**  
   If at some point the model starts outputting NaN everywhere (e.g. from a bad step we didn’t fully prevent), losses get clamped to 0 and we might step with zero gradient or skip often. Run would complete but not learn. Monitor loss curves; if everything goes to 0 and stays there, something else is wrong.

4. **LBFGS**  
   If you ever switch to LBFGS, skipping backward inside the closure may be wrong (LBFGS expects closure to backward). LEGO uses Adam; no change needed for current config.

---

## Config snapshot (relevant bits)

- block_epochs_per_type: 25  
- start_div: 2000  
- violation_buffer: use, initial_size_per_c 50, random_fraction 0.1  
- ALM on (base_config), scalar losses clamped in ALM calculator  
- compute_grad_norm: handles empty grads  

---

## Verdict

**Good to run** for “complete 5000 epochs without crashing.”  

- Violation buffer, ALM, skip-backward, skip-adaptive_update, and compute_grad_norm are all aligned so the previous failure modes (NaN probs, negative loss assert, bad step, ALM blow-up, empty grads) are handled.  
- You may see: "No surface points found", "Surface points failed — skipping backward", and occasional wandb/plot warnings; training should continue.  
- If you see long stretches of skipped steps or all losses stuck at 0, treat that as a training/configuration issue (e.g. start_div, random_fraction, or data) rather than a missing crash fix.

Run, and watch the first time you pass epoch 2000: you should get either a normal step or a skip (with the skip message), and no crash.
