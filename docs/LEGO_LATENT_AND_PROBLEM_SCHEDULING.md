# LEGO: Latent (z) and Problem (c) Scheduling

This note explains what we fixed for **latent resampling** and what remains as an open **per-batch problem switch** for LEGO GINN training.

---

## What we fixed: Resampling z every epoch

**Problem (before):** We **resampled the batch of latent vectors z every epoch** (`reset_zlatents_every_n_epochs: 1`). So each epoch the 9 shapes in the batch were **new** (z, conditioning) pairs. The same (z, brick type) almost never got multiple consecutive steps, so the optimizer couldn’t lock in interface fit for any fixed set of shapes. That **instability** contributed to interface/envelope violation drifting up and geometry regressing (good ~200, then degraded ~700–1000).

**How we fixed it:** We introduced a **violation buffer** and use it when `violation_buffer.use: true` (LEGO config):

1. **Buffer of (z_base, c) pairs**  
   We maintain a buffer of (z_base, conditioning key) pairs with a **violation score** (interface + envelope loss) per pair. The buffer is **prefilled** at init with `initial_size_per_c` (e.g. 50) random (z_base, c) per brick type so we cover the (z, c) space.

2. **Sample proportional to violation**  
   Each step we pick one brick type **c** (uniformly at random), then sample a batch of (z_base, c) from the buffer **with probability proportional to violation** (prioritized replay). So we train more on (z, c) pairs where the model currently fails.

3. **Refresh when used**  
   After each optimizer step we **re-evaluate** interface + envelope loss (no grad) for the (z, c) pairs we just used and **update** their violation in the buffer. So scores are refreshed when a pair is used; others stay stale until sampled.

4. **10% random**  
   We mix in a **random fraction** (e.g. 10%) of fresh random z_base per batch so we don’t collapse onto a tiny corner of z space. Random slots are not added to the buffer; we only update violations for buffer entries we used.

5. **Coverage per brick type**  
   The buffer has equal entries per brick type (50 per c_key), and we pick c_key uniformly at random each step, so no brick type dominates the buffer or selection.

**Config:** `configs/GINN/lego_1xN_wire.yml` has `violation_buffer.use: true`, `initial_size_per_c: 50`, `sample_temperature: 1.0`, `random_fraction: 0.1`, `eps: 1e-8`. Code: `train/train_utils/violation_buffer.py`, and trainer logic in `train/ginn_trainer.py` (sample from buffer, build z with conditioning, train, then update violations for the batch we used).

**Result:** We no longer “resample z every epoch” in the old sense. The same (z, c) can appear again (from the buffer), and we focus on high-violation pairs. That addresses the **z resampling** instability.

---

## Follow-up open problem: Per-batch problem switch

**What’s still true:** Each training step we use **one brick type** (one conditioning key c). We set `self.problem = self.problems[c_key]` and the **whole batch** (e.g. 9 shapes) that step is for that one type. The next step we pick a (possibly different) c_key → different interface/envelope → different gradients. So the **shared** network sees a **different constraint set every step** (which brick type we’re on). Gradients are only for that brick type that step.

**Why it’s a problem:** The optimizer gets a **changing** objective each step (which type’s constraints we’re optimizing). That can add **noise** and contribute to interface drift or slower convergence, especially when combined with other factors. We have **not** changed this design: we still do one brick type per step.

**Implemented: Block training**

We implement **block training**: use one brick type for K consecutive epochs, then switch (cycle through all types).

- **Config:** `violation_buffer.block_epochs_per_type: 50` (or top-level `block_epochs_per_type`). Set to 0 for random c per step (previous behavior).
- **Behavior:** Epochs 0–49: only brick type 0. Epochs 50–99: only brick type 1. etc., cycle through all types.

**Key point:** The violation buffer **already** does prioritized replay **within** each brick type (50 (z, c) per type, sample ∝ violation, refresh when used). So block training’s benefit is **not** “see the same (z, c) more”—you already get that from the buffer. Block training only changes **which brick type** we use: stable for 50 epochs instead of random each step.

**The actual trade-off**

| | Random type each step | Block training (50 epochs per type) |
|---|------------------------|-------------------------------------|
| **Pros** | All brick types get roughly equal training time over N steps. Violation buffer focuses on hard (z, c) within current type. | Optimizer sees **consistent constraints** for 50 epochs. **Lagrangian multipliers** (λ_if, λ_env) can adapt to one type without being jerked to another type next step. **Adam momentum** points in a consistent direction for 50 epochs. **Loss landscape** is stationary for 50 epochs, not changing every step. Violation buffer has 50 epochs to drive violations down for the current type. |
| **Cons** | Optimizer sees different constraint sets each step. Lagrangian multipliers chase a moving target. | **Risk:** Does type A regress when we switch to type B at epoch 50? |

**What block training actually buys you**

Given the violation buffer, block training’s benefit is:

1. **Lagrangian multiplier stability:** λ_interface and λ_envelope can adapt to type A’s constraints without being jerked to type B’s constraints next step.
2. **Optimizer momentum alignment:** Adam’s momentum buffers point in a consistent direction for 50 epochs.
3. **Loss landscape consistency:** The loss surface is stationary for 50 epochs, not changing every step.

**Hypothesis to test**

*“The instability is primarily from Lagrangian multipliers and optimizer momentum being disrupted by switching brick types every step, NOT from insufficient exposure to individual (z, c) pairs.”*

**What to measure:** Does interface/envelope loss stay converged when you switch types? At epoch 50 when you switch from type 0 → type 1, does type 0’s loss regress?

**How types are categorized (block index → brick type):**  
`self.problems` is built in `train/ginn_trainer.py` with a **nested loop** over `_n_list`, `_ny_list`, `_h_list` (from config `n_studs_values`, `n_studs_y_values`, `height_values`). Order: **outer** loop = **n** (stud count), then **ny** (width: 1 = 1×N, 2 = 2×N, 4 = 4×N), then **h** (height scale). So:

- **Loop:** `for n in n_studs_values: for ny in n_studs_y_values: for h in height_values: key = (n, ny, h)` (when all three conditionings are on).
- **LEGO config:** `n_studs_values: [2, 3, 4, 6]`, `n_studs_y_values: [1, 2, 4]`, `height_values: [0.75, 1.0, 1.25]` → **36 brick types**.
- **Type 0** = (2, 1, 0.75) = 1×2, 1-wide, short. **Type 1** = (2, 1, 1.0). **Type 2** = (2, 1, 1.25). **Type 3** = (2, 2, 0.75). … **Type 35** = (6, 4, 1.25).

So “type index” is the **index in the list of keys** in this fixed order (n varies slowest, then ny, then h).

Other directions (not implemented): **Round-robin c** (one type per epoch in fixed order); **Multiple c per batch** (different brick types per row in the batch).

---

## Critical evaluation: training on only 2 block types

Current config uses **2 brick types**: `n_studs_values: [2, 4]`, `n_studs_y_values: [1]`, `height_values: [1.0]` → **1×2** and **1×4** only. Block training is **off** (`block_epochs_per_type: 0`), so type is chosen **at random** each step.

### Why 2 types can make sense

- **Simpler experiment:** Fewer variables, easier to debug conditioning and the violation buffer. You can check that (z, c) sampling, problem switching, and validation plots all behave.
- **Random switching is fine:** With 2 types, each gets ~50% of steps in expectation. No need for block training to “balance” many types; you can measure whether **forgetting** happens even with just 2 (e.g. does 1×2 regress when 1×4 is trained).
- **Removes one variable:** Turning off block training removes schedule design from the experiment and isolates “does the model hold both 1×2 and 1×4 with random type per step?”
- **Validation:** `z_val` has one row per type, so you get **2 validation shapes** (1×2 and 1×4) every validation epoch. Easy to inspect both.

### Risks and limitations

- **No generalization to other N:** The model only ever sees conditioning **n_norm ∈ {0, 1}** (for 1×2 and 1×4). It does **not** see 1×3, 1×6, etc. So:
  - **Extrapolation** to unseen N (e.g. 1×3) will be untested and likely poor.
  - The network may **memorize two patterns** rather than learn a smooth “length vs n” mapping. Do not claim generalization to new brick lengths without training on more N (and ideally interpolating).
- **Conditioning span:** With `n_studs_values: [2, 4]`, the normalized input is 0 and 1 only. The model has no incentive to produce sensible outputs for n_norm = 0.5 (1×3). If you later add 1×3, you may need to retrain or at least validate carefully.
- **Violation buffer size:** `initial_size_per_c: 50` with 2 types → **100 entries** total. Enough for 2 types, but buffer diversity is limited; the 10% random fraction helps avoid collapse.
- **Statistical noise:** Random type per step means over short windows (e.g. 100 steps) one type might get 40 updates and the other 60. Over long runs it balances; for quick runs, consider logging **per-type loss** (e.g. loss_if for 1×2 vs 1×4) to see if one type is under-trained or forgotten.
- **SCC / topology:** 1×2 has 2 studs (simpler connectivity) and 1×4 has 4. If one type converges and the other doesn’t, it could be capacity, data balance, or difficulty—hard to tell with only 2 types.

### Recommendations

- **For debugging the pipeline and conditioning:** 2 types is **reasonable**. Use it to confirm violation buffer, random type choice, and validation plots; then consider adding types.
- **For claiming “one model for any 1×N”:** You need **more N** (e.g. 2, 3, 4, 6) and possibly block training or careful logging to avoid forgetting. 2 types is a stepping stone, not the end state.
- **What to log:** If possible, log or periodically evaluate **interface/envelope loss per type** (e.g. average violation over buffer entries for 1×2 vs 1×4). That tells you whether one type is regressing or under-trained.
- **Block training:** With 2 types, `block_epochs_per_type: 0` is a **valid choice** to remove schedule as a variable. If you later scale to many types and see regression at type switches, re-enable block training and tune block length.

**Bottom line:** Training on 2 block types (1×2, 1×4) is **good for experimentation and debugging**, but it does **not** by itself show that the model generalizes to other brick lengths. Use it to validate the setup; then add more types and/or block training if the goal is multi-type or general 1×N.

---

## Critical evaluation: what we should do

**Hypothesis we’re testing:**  
*“The instability (interface/envelope drift, geometry regression) is primarily from Lagrangian multipliers and optimizer momentum being disrupted by switching brick types every step, NOT from insufficient exposure to individual (z, c) pairs.”*

**What we already have:** Violation buffer (prioritized replay within each type, refresh when used, 10% random). Curvature off. Objective null. Block training available (`block_epochs_per_type: 50`). So we’ve addressed z resampling and curvature; block training is the lever for “type switching.”

**What we should do (in order):**

1. **Run with block training on** (`block_epochs_per_type: 50`) and **measure**:
   - Does interface/envelope loss (e.g. loss_unweighted_if, loss_unweighted_env) stay converged **within** each block (e.g. epochs 0–49 for type 0)?
   - **At the switch** (e.g. epoch 50): does type 0’s loss **regress** when we switch to type 1? (Evaluate type 0’s violation on a fixed set of (z, type 0) at epochs 49, 50, 51, … and see if it jumps.)
   - Do meshes stay good across blocks and after full cycles (e.g. after 36 × 50 = 1800 epochs)?

2. **Interpret the result:**
   - **If block training helps** (loss stable within blocks, little or no regression at switches, meshes good): the hypothesis is supported—type switching was a major source of instability; block training is a valid mitigation. Keep block training; optionally tune `block_epochs_per_type` (e.g. 30 vs 50 vs 100) if you want to trade off “stability per type” vs “how often we revisit each type.”
   - **If type 0 regresses badly at epoch 50:** either (a) 50 epochs per type is too short for the Lagrangian to settle, or (b) the shared network forgets type 0 when we train on type 1. Then try: longer blocks (e.g. 100), or periodic “revisit” of previous types (e.g. every 200 epochs evaluate all types and log violations), or accept some regression and rely on later blocks to fix it.
   - **If block training doesn’t help** (loss still drifts, meshes still regress): the instability may be from something else (e.g. ALM dynamics, learning rate, or capacity). Then re-check loss curves and meshes; consider `use_augmented_lagrangian: False` with fixed lambdas, or lower LR, or other architectural/optimization changes.

3. **Baseline (optional but useful):** If you have not already, run **one** comparable run with **block training off** (`block_epochs_per_type: 0`, random type each step) and the same violation buffer / curvature off / objective null. Compare: does block training give clearly better stability (loss and meshes) than the baseline? That tells you whether “type switching every step” was actually the main problem.

**What we don’t know yet:** Whether 50 epochs per type is the right length; whether some brick types are much harder and need more blocks; whether Lagrangian multipliers should be reset or kept when switching types. Treat the first block-training run as an experiment: measure, then decide.

**Short recommendation:** Turn block training on, run at least one full cycle (e.g. 1800+ epochs for 36 types × 50), and measure within-block stability and regression at type switches. Use that to confirm or reject the hypothesis and to decide whether to keep, tune, or drop block training.

---

## Summary

| Issue | Status | What we did |
|-------|--------|-------------|
| **Resampling z every epoch** | Fixed | Violation buffer: prefill (z, c), sample ∝ violation, refresh when used, 10% random, equal coverage per type. |
| **Per-batch problem switch** | Mitigated (block training) | Block training: `block_epochs_per_type: 50` → one brick type for 50 epochs, then cycle. Measure: does type 0’s loss regress when we switch to type 1 at epoch 50? |
