# Plan: Comparing LEGO Model vs Regular GINN Architecture

## 1. Definitions

| Term | What it is in this repo |
|------|--------------------------|
| **LEGO model** | `cond_wire` trained on `lego_1xN` with LEGO-specific inputs: **tiled coords** (`use_tiled_coords: true`), **dist_to_edge** (`use_dist_to_edge: true`), conditioning on N (and optionally N_y, height). Typically larger capacity (e.g. `layers: [256,256,256,256]`). Config: `configs/GINN/lego_1xN_wire.yml`. |
| **Regular GINN** | Same backbone (`cond_wire`) but **without** LEGO-specific features: no tiled coords, no dist_to_edge — only `(x,y,z)` + conditioning vector. Same style as SimJEB / TOM configs (e.g. `cond_wire` in `simjeb_finetune_scc.yml`). |

So the comparison is: **cond_wire with LEGO-specific geometry inputs vs cond_wire with standard (x,y,z)+z only**, both on the **same LEGO 1xN problem and data**.

---

## 2. What to Compare

- **Constraint satisfaction**  
  Interface, envelope, wall, (and if used) PH/SCC metrics — same constraints and evaluation code for both.

- **SDF / data fidelity**  
  Error vs ground-truth SDF or proxy (e.g. mesh-to-SDF, or supervised points) if you have it; otherwise use constraint violation as the main proxy.

- **Training efficiency**  
  Loss curves, wall-clock time, and iterations to reach a target violation level.

- **Generalization**  
  If you train on N ∈ {2,4,5,8,12}, test on held-out N (e.g. 3, 6, 7) and compare metrics.

- **Inference / robustness**  
  Same resolution and marching-cubes (or eval grid) for both; compare sharpness, stud placement, and any visual artifacts.

---

## 3. Setup

### 3.1 Regular GINN config on LEGO (for fair comparison)

Create a config that:

- Uses **same problem**: `lego_1xN`, same `n_studs_values`, same conditioning (N, N_y, height) so the **output** and losses are comparable.
- Uses **same** (or similar) training setup: same losses (GINN, data, PH if any), same epochs, same batch size, same data sampling.
- **Only** changes the model section:
  - `use_tiled_coords: false`
  - `use_dist_to_edge: false`
  - Optionally use the same `layers` as LEGO or a smaller “regular” size (e.g. `[128,128,128]` or `[256,256,256,256]`) to separate “architecture” from “capacity”.

Suggested name: e.g. `configs/GINN/lego_1xN_wire_regular.yml` (inherits or copies from `lego_1xN_wire.yml` and overrides the model flags above).

### 3.2 Evaluation environment

- **Option A – Notebook**  
  One notebook (e.g. `notebooks/lego_vs_regular_ginn_comparison.ipynb`) that:
  1. Loads two checkpoints: LEGO model vs Regular GINN (same problem, same conditioning).
  2. Builds the same problems and runs the same constraint/eval code (and PH if applicable).
  3. Plots side-by-side: loss curves (if logged), violation stats, and optionally meshes or slices.

- **Option B – Script**  
  A small script that takes two configs (or two checkpoint dirs), runs the same evaluation pipeline, and writes a summary (e.g. CSV + short report).

- Reuse existing evaluation wherever possible: e.g. `GINN/evaluation/evaluator.py`, constraint violation aggregation from the trainer, and any existing LEGO visualization from `lego_model_test.ipynb` or `lego_constraint_visualization.ipynb`.

---

## 4. Experiments to Run

1. **Train Regular GINN on LEGO**  
   Train from scratch with `lego_1xN_wire_regular.yml` (same steps/epochs as current LEGO runs). Optionally also train a LEGO model from scratch with the same seed/steps so both are comparable.

2. **Evaluate both on same test set**  
   Same (z, c) conditions and same N (and N_y, height). Report:
   - Per-constraint violation (e.g. interface, envelope, walls).
   - If you use PH: Betti numbers or SCC loss.
   - Optional: SDF error on a grid or on surface points.

3. **Generalization (optional)**  
   If you have held-out N (e.g. 3, 6, 7): evaluate both models on those N and compare violation and visual quality.

4. **Ablation (optional)**  
   - LEGO model vs LEGO model without tiled coords only.  
   - LEGO model vs LEGO model without dist_to_edge only.  
   This shows which geometric input helps more.

---

## 5. How to Analyze

- **Tables**: For each (model, metric), report mean (and std if you have multiple runs or test shapes).  
- **Plots**: Loss vs epoch; violation vs epoch; bar chart of final violations (LEGO vs Regular).  
- **Conclusion**: e.g. “LEGO-specific inputs improve interface/envelope satisfaction by X% and reduce training time to target by Y%.”

---

## 6. Files to Add/Change

| Action | File |
|--------|------|
| Add | `configs/GINN/lego_1xN_wire_regular.yml` — copy of LEGO config with `use_tiled_coords: false`, `use_dist_to_edge: false`. |
| Add | `notebooks/lego_vs_regular_ginn_comparison.ipynb` (or script) — load both checkpoints, run same eval, compare. |
| Reuse | `lego_1xN_wire.yml`, `problem_lego_1xN.py`, trainer loss/constraint code, `evaluator.py`, existing LEGO notebooks. |

---

## 7. Quick start

1. Create `lego_1xN_wire_regular.yml` from `lego_1xN_wire.yml` and set `use_tiled_coords: false`, `use_dist_to_edge: false` (and optionally `layers: [128,128,128]` for a true “regular” size).
2. Train: e.g. `python train/train_ginn.py --config configs/GINN/lego_1xN_wire_regular.yml`.
3. In the comparison notebook/script: load LEGO checkpoint and Regular GINN checkpoint, run the same evaluation pipeline, and compare metrics and plots.

This gives a clear, reproducible comparison of the LEGO model vs the regular GINN architecture on the same task.
