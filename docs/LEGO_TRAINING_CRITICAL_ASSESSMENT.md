# Critical Assessment: LEGO GINN Training Setup

A frank evaluation of the current LEGO 1xN training configuration and what could still go wrong.

---

## 1. **ALM + Block Training: Shared Multipliers Across Brick Types**

**Setup:** `use_augmented_lagrangian: True` (inherited from base). Adaptive penalties (μ_if, μ_env, μ_scc, …) are **global**—one set of multipliers for the whole run. Block training switches brick type every 50 epochs.

**Risk:** When you switch from type A to type B at epoch 50, the optimizer state (momentum, etc.) and the ALM multipliers were tuned for type A’s constraint set. Type B has different interface/envelope geometry. The first steps on type B use:
- Momentum from type A (may point in unhelpful directions).
- Multipliers that may be too high or too low for type B.

**Possible outcomes:** (a) Slow or noisy convergence at the start of each block. (b) If multipliers had grown large on type A, type B can get oversized constraint gradients and instability. (c) When you cycle back to type A at epoch 1800, multipliers and optimizer state have been shaped by all other types—possible **forgetting** or oscillation.

**Mitigation:** Monitor `loss_unweighted_if` / `loss_unweighted_env` and μ_if / μ_env at block boundaries (e.g. epochs 50, 100, …). If you see spikes or slow recovery, consider: (1) `use_augmented_lagrangian: False` with fixed lambdas, or (2) per-type multipliers (would require code change), or (3) short “warm-up” steps at the start of each block with reduced LR or frozen ALM update.

---

## 2. **Violation Buffer: Fixed Size, No Growth, Narrow z**

**Setup:** 50 entries per brick type (36 types → 1800 entries total). Buffer is **never extended**—the 10% random z per batch are not added. Base latent is sampled in `z_sample_interval: [0, 0.1]` (very narrow).

**Risks:**
- **Limited diversity:** 50 z’s per type in [0, 0.1]² can collapse to a small effective set. Prioritized sampling pulls high-violation z’s; if a few z’s dominate violations, the same ones are trained repeatedly. The 10% random helps but doesn’t expand the buffer.
- **Narrow latent range:** [0, 0.1] means the model sees only a tiny slice of latent space. Diversity loss (after epoch 2000) encourages differences within that slice; you may get limited shape variety.
- **No mechanism to add “good” z’s:** Once a (z, c) pair gets low violation, it’s sampled less. There’s no explicit retention of “solved” pairs for occasional replay, so the buffer is purely “prioritized failure” and may under-represent already-good regions.

**Mitigation:** (1) Widen `z_sample_interval` (e.g. [0, 0.2] or [0, 0.5]) if you want more shape diversity. (2) Optionally add a buffer policy that occasionally appends random (z, c) or high-violation (z, c) so the buffer grows slowly. (3) Log how many unique buffer indices are updated per type per 100 epochs to detect collapse.

---

## 3. **Block Length vs Total Epochs: Coverage and Forgetting**

**Setup:** 36 types × 50 epochs/block = 1800 epochs per full cycle. `max_epochs: 5000` → ~2.78 cycles. So each type gets 50 + 50 + 50 = **150 epochs** total, with long gaps (e.g. type 0 not trained again until epoch 1800).

**Risks:**
- **Forgetting:** After 1750 epochs on other types, the model may have drifted so that type 0’s constraints are no longer well satisfied. You rely on a shared backbone; fine details per type could degrade.
- **Under-training per type:** 150 epochs per type might be enough for “reasonable” geometry but marginal for sharp studs and walls. If interface/envelope loss per type is still high at the end of each block, you’re underfitting that type.
- **Cycle boundary effects:** Epochs 1799→1800 and 3599→3600 switch from the last type back to type 0. Validation at 1800 may show type 0 **worse** than at epoch 49 if forgetting dominates.

**Mitigation:** (1) At each type switch (e.g. 50, 100, …), evaluate **previous** type’s loss on a fixed set of (z, c) and log it (e.g. `loss_if_type_prev`). If it rises sharply after a long gap, you have forgetting. (2) Consider shorter blocks (e.g. 25) so each type is revisited more often, or longer total training. (3) Optionally add a small “replay” fraction: each step, 1 slot could be from a random other type to reduce forgetting (at the cost of noisier gradients).

---

## 4. **Order of c_keys: Determinism vs Dict Order**

**Setup:** Block index is `(epoch // block_epochs) % len(c_keys_list)` with `c_keys_list = list(self.problems.keys())`. Buffer is built with `c_keys = list(self.problems.keys())` at init. In Python 3.7+, dict order is insertion order; `self.problems` is built with nested loops `for n, for ny, for h`, so order is deterministic.

**Risk:** If anywhere the code (or config) ever builds a different dict or iterates in a different order, block index and buffer could get out of sync (e.g. “block 0” meaning different types in buffer vs trainer). Currently the code is consistent; the risk is **future** changes (e.g. sorting keys, or loading problems from a different source).

**Mitigation:** Either (1) explicitly sort `c_keys` and use the same sorted list in both buffer init and training (e.g. `c_keys = sorted(self.problems.keys())`), or (2) add a one-line check at startup that buffer’s `c_keys` equals trainer’s `list(self.problems.keys())`.

---

## 5. **Surface Points and Diversity: Cost and Timing**

**Setup:** `n_points_surface: 20000`, `recompute_every_n_epochs: 1`. From epoch 2000, diversity loss is active and needs surface points every epoch. Curvature is off (`lambda_curv: 0`), so surface points are only for diversity.

**Risks:**
- **Compute cost:** 20k points × 9 shapes, with numerical binsearch and possibly chunked distance to interface, every epoch from 2000 onward can be noticeable. Not necessarily wrong, but worth profiling if training is slow.
- **Diversity vs geometry:** If geometry (interface/envelope) is not fully converged for some types by epoch 2000, diversity can still pull the SDF in directions that hurt studs/walls. You already delayed diversity to 2000; if blocks are only 50 epochs, some types have had at most 100 epochs by 2000 (e.g. types 0 and 1 twice). So a few types may still be underfit when diversity kicks in.

**Mitigation:** (1) Consider `start_div: 2500` or 3000 to give more geometry-only epochs per type across cycles. (2) If training time is an issue, try `recompute_every_n_epochs: 2` for surface points (slightly staler but cheaper).

---

## 6. **Validation and Logging: What You Won’t See**

**Setup:** Validation every 200 epochs; plots every 100. Validation loops over all types and uses fixed `z_val` (one row per type in the grid). Losses logged are for the **current batch** (one type per step).

**Risks:**
- **No per-type loss in logs:** You don’t log `loss_if` / `loss_env` **per c_key**. So you can’t see “type (2,1,0.75) has high interface loss” from the main loss curve; you only see the type that was trained that step. Harder to spot which brick types are failing.
- **Validation mesh quality:** Every 200 epochs you get meshes for all types at fixed z. If a type is forgotten or underfit, you’ll see it in the mesh plot, but you won’t have a scalar “per-type validation loss” to track over time.

**Mitigation:** (1) Optionally log a scalar like `loss_if_c0`, `loss_if_c1`, … for a few representative types every N epochs (e.g. compute once per 100 epochs on one fixed z per type). (2) Or add a “validation loss” pass every 200 epochs that computes mean interface/env loss over all types (e.g. one z per type) and logs `val_loss_if`, `val_loss_env` (and per-type if you want).

---

## 7. **PH / SCC: Memory and Workers**

**Setup:** `scc_n_grid_points: 32`, `ph_max_num_workers: 4`, `evaluate_ph_grid_per_shape: True`. SCC runs for connectivity (target one component).

**Risks:**
- **RAM in workers:** Persistent homology on 32³ points per shape, 9 shapes, with 4 workers can spike RAM. You already reduced from 64 and 16 workers; if you hit MemoryError in a worker, reduce further (e.g. 24 grid, 2 workers) or batch shapes.
- **Staleness:** PH is computed with current `self.problem` (and `set_problem` is called when switching type). So within a block, PH is correct for that type. No bug here; just a note that PH cost is per-step and can dominate step time.

**Mitigation:** Monitor RAM during first few epochs of a block. If OOM in worker, lower `scc_n_grid_points` or `ph_max_num_workers`.

---

## 8. **Grad Clipping and LR**

**Setup:** `grad_clipping_on: True`, `grad_clip: 0.5`, `auto_clip_on: True`. `lr: 0.001`, no scheduler (`use_scheduler: False`).

**Risks:**
- **Constant LR for 5000 epochs:** If loss plateaus early, you might want decay; if loss is still decreasing at 5000, you might be under-training. No built-in schedule.
- **Clipping can hide instability:** If gradients are frequently clipped, effective steps are smaller than LR suggests. That can stabilize but also mask too-large lambdas or ALM multipliers (you might not see NaNs, but progress could be slow).

**Mitigation:** (1) Log `grad_norm_post_clip` and how often it equals `grad_clip`; if it’s at the clip value most of the time, consider lowering lambdas or LR. (2) Optionally try a simple decay (e.g. gamma=0.5 every 2000 epochs) in a later run.

---

## 9. **Summary: Highest-Impact Risks**

| Priority | Issue | What could go wrong | Quick check |
|----------|--------|---------------------|-------------|
| **High** | ALM + block switch | Unstable or slow at block boundaries; forgetting when cycling back | Log μ_if, μ_env at epochs 50, 100, …; compare type 0 loss at 49 vs 1800 |
| **High** | Violation buffer narrow + fixed | Collapse to few z’s per type; limited shape diversity | Widen z interval; log unique buffer indices sampled per 100 epochs |
| **Medium** | Forgetting across long gaps | Type 0 worse at 1800 than at 49 | Log previous-type loss at each block boundary |
| **Medium** | No per-type loss in wandb | Can’t see which types are bad | Add optional per-type or val loss every N epochs |
| **Lower** | Surface point cost, PH RAM, constant LR | Slower training; OOM; plateau | Profile; monitor RAM; optional LR decay |

---

## 10. **Recommended Next Steps (in order)**

1. **Run one full block (e.g. 1800 epochs)** with current config and log at each type switch:
   - Current type’s `loss_unweighted_if` / `loss_unweighted_env` at first and last epoch of block.
   - Optional: previous type’s loss on 5 fixed (z, c) at switch (e.g. epoch 50: evaluate type 0 on 5 fixed z’s).
2. **Inspect buffer usage:** Every 200 epochs, record how many distinct buffer indices were used for the current type (e.g. over the last 200 steps). If it’s always the same ~10–20, consider widening z or adding exploration.
3. **If block boundaries show instability or forgetting:** Try (a) `use_augmented_lagrangian: False` with current lambdas, or (b) shorter blocks (25 epochs) so each type is revisited more often.
4. **If geometry is good but diversity is weak:** Widen `z_sample_interval` and/or delay `start_div` further.

This assessment is deliberately critical: the setup is reasonable given past failures (mesh reduction, diversity timing, interface weight, connectivity, block training), but ALM + blocks, buffer design, and long gaps between revisits are the main remaining failure modes to watch and mitigate.
