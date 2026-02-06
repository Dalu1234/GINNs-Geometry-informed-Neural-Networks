# Current Methodology: LEGO 1xN Conditional WIRE (GINN)

This document describes the **current implementation and methodology** for training a single conditional implicit model that generates LEGO 1xN brick SDFs across multiple brick types (e.g. 1x1, 1x2, 1x4, 1x5) without shape data—using only geometry-informed rules, constraints, and losses.

---

## 1. Goal and Scope

- **Goal:** Train one neural field \( f(x; z) \) that outputs a **signed distance function (SDF)** for a LEGO brick. The brick type is encoded in the latent \( z \) via conditioning (stud count N, optionally N_y, height). The model should:
  - Satisfy **interface** (stud tops/sides, walls at SDF = 0) and **envelope** (shape inside bounding box) constraints.
  - Produce **valid SDFs** (eikonal: \( \|\nabla_x f\| = 1 \)).
  - Generalize to **unseen** stud counts (e.g. 1x3 when training on 1, 2, 4, 5) when an encoder or smooth encoding is used.
- **No shape data:** All supervision is from **rules** (cuboid, stud cylinders, outside-brick, phantom stud) and **constraint losses** (interface, envelope). No point clouds or meshes are used as regression targets for the SDF.

---

## 2. Problem Definition: LEGO 1xN

- **Problem class:** `ProblemLego1xN` (`GINN/problems/problem_lego_1xN.py`). Parametric LEGO brick with:
  - **N** studs in a row (1xN); optionally **N_y** for NxM grids (config: 1 = 1xN only).
  - **Height scale** (e.g. 1.0 = nominal LEGO).
- **Dimensions (normalized):** Stud spacing = 1.0, brick length = N × spacing − inset (0.025), stud radius/height from standard LEGO mm values.
- **Constraints:** Envelope (3D bbox), interfaces (stud tops and sides, six wall faces). Domain, envelope, and interface samplers are provided by the problem per (N, N_y, height).
- **One problem instance per type:** For each (N, N_y, height) in the config, a separate `ProblemLego1xN` is created. The trainer switches `self.problem` per batch (or uses a violation buffer to sample which type to train on).

---

## 3. Model Architecture: Conditional WIRE

- **Model:** `ConditionalWIRE` (`models/wire.py`). WIRE = Wavelet Implicit Representation; backbone is a stack of **RealGabor** layers (cosine-modulated Gaussian) plus final linear → scalar SDF.
- **Inputs to the network (per forward):**
  1. **x:** 3D coordinates (batch, 3).
  2. **z:** Full latent vector (batch, nz). Layout: `[z_base, conditioning..., n_y_norm, height_norm]`. Conditioning depends on config (see §4).
  3. **Geometry-informed features (appended to x before concat with z):**
     - **Tiled coordinates** (`use_tiled_coords`): \( (x \bmod T_x, y \bmod T_y) \) in [-0.5, 0.5] so the network sees “position within the current stud tile.” One stud per tile; period = stud spacing (1.0 in normalized space). Implemented in `util/model_utils.tile_coords_xy`.
     - **Distance to edge** (`use_dist_to_edge`): Signed distance to the nearest x-boundary of the brick (positive inside, negative outside). Lets the network “ignore” phantom studs outside the brick. Formula uses effective stud count from z (or override). Implemented in `util/model_utils.dist_to_edge_x_from_z`.

- **Backbone input:** `xz = concat(x_augmented, z_backbone)` where `x_augmented` = x + tiled coords + dist_to_edge. **Important:** When the N-encoder is used, `z_backbone` is **not** the full z; the 9th encoded dimension is stripped and used only for `dist_to_edge` (see §4), so the decoder cannot shortcut on raw stud count.

- **Conditioning mechanisms:**
  - **FiLM** (`use_film`): Feature-wise Linear Modulation at selected layers: \( h \mapsto (1 + \gamma(z)) \, h + \beta(z) \). Gamma/beta are MLPs from **z_backbone**. Applied after layers 0 and 2 (configurable).
  - **Hypernetwork LoRA** (`use_hypernet`, optional): Last two layers (penult Gabor + final linear) get low-rank deltas \( U V^T \) from a hypernetwork that takes the last `c_dim` columns of **z_backbone** (e.g. Nx_norm, Ny_norm, height_norm). Not used in the default LEGO 1xN config.

- **Output:** Single scalar SDF (or density if `return_density`); level set 0 defines the surface.

---

## 4. Conditioning and Latent Vector Layout

- **Base latent:** `z_base` has dimension `latent_sampling.nz` (e.g. 6), sampled in `[0, 0.1]` (configurable).
- **Conditioning columns** (order must match trainer and notebook):
  1. **N (stud count):** One of:
     - **Scalar** (default): one column `n_norm = (N - n_min) / (n_max - n_min)` in [0, 1].
     - **Smooth encoding** (`use_smooth_n_encoding`): Gaussian weights over `n_studs_values`; multiple columns.
     - **N-encoder** (`use_n_encoder`): MLP maps `n_norm` (1D) → **9D**: first 8 go to backbone; 9th is used **only** for `dist_to_edge` (sigmoid in forward). No raw `n_norm` is appended, so the decoder cannot bypass the encoder.
  2. **N_y** (if `condition_on_n_studs_y`): one column, normalized.
  3. **Height** (if `condition_on_height`): one column, normalized.

- **N-encoder (CondEncoder):**
  - Architecture: Linear(1 → 32) → ReLU → Linear(32 → 9). Output 9 = 8 (backbone) + 1 (dist_to_edge only).
  - In `forward`: if `use_n_encoder` and z has the extra column at `n_norm_col`, then `n_norm_for_dist = sigmoid(z[:, n_norm_col])`, and `z_backbone = concat(z[:, :n_norm_col], z[:, n_norm_col+1:])`. All backbone and FiLM use `z_backbone`; `dist_to_edge_x_from_z` is called with `n_norm_override=n_norm_for_dist`.
  - Trainer builds z with **only** `model.encode_n(n_norm_t)` (9 dims); no raw `n_norm` column. Model `nz` for the backbone is `nz_base + 8 + n_y + height`; total z length is `nz_base + 9 + n_y + height`.

- **dist_to_edge:** When `n_norm_override` is provided (encoder path), it is used instead of reading from z, so the geometry gets a sensible “effective n” without exposing raw n to the decoder.

---

## 5. Training Pipeline (GINN Trainer)

- **Entry:** `run.py` → `Trainer` in `train/ginn_trainer.py`. Config: `configs/GINN/lego_1xN_wire.yml` (with base merge).
- **Per epoch:**
  - **z:** Either from a **violation buffer** (prioritized replay by interface+envelope violation) or resampled. When conditioning is on, z is built as `[z_base, conditioning columns]`. For N, either scalar, smooth Gaussian, or `model.encode_n(n_norm_t)` (9 dims, no raw).
  - **Problem:** One problem per (N, N_y, height). Trainer sets `self.problem` per batch (or by buffer). All loss calls receive `problem=self.problem` and optional `bounds` / `bounds_body` so cuboid, stud_grid, phantom_stud, outside_brick_sdf use the correct brick type.
  - **Losses:** Dispatcher calls each active loss (keys from `get_loss_keys_and_lambdas`: any `lambda_* > 0`). Scalar losses get the same `step_kw` (z, problem, bounds, n_norm_col, etc.). Field losses (e.g. chamfer_div) use surface points when available.
  - **Backward:** One total loss (weighted sum or ALM); backward on the model. If diversity is on and `get_surface_pts` failed (epoch ≥ start_div), backward is **skipped** and ALM adaptive_update is skipped to avoid bad steps and multiplier growth.
  - **Violation buffer:** After the step, violation (interface + envelope) is recomputed for the (z, c) pairs used and the buffer is updated. Sampling is temperature-scaled; a random fraction of batches use fresh z to avoid collapse.

- **Loss balancing:** Either **ManualWeightedLoss** (fixed lambdas) or **AdaptiveAugmentedLagrangianLoss** (ALM). Scalar losses are clamped (e.g. non-negative) in the calculator to avoid ALM assert failures. `compute_grad_norm` returns 0 when no parameter has gradients (e.g. after a skipped backward).

---

## 6. Loss Methodology: Rule-Based and Constraint-Based

Losses are **rule-encoding** (teach geometry) or **constraint** (enforce feasibility). No direct SDF regression to meshes.

### 6.1 Core geometry (always on in LEGO 1xN)

| Loss | Role | How |
|------|------|-----|
| **eikonal** | Valid SDF | Penalize \( (\|\nabla_x f\| - 1)^2 \) on domain samples. |
| **cuboid_primitive** | Box body | Inside box → f &lt; −margin; near faces/corners/far → f &gt; +margin; boundary sharpening near faces. Uses `bounds_body` when given (LEGO: body only, no stud volume). |
| **stud_grid** | Studs | Cylinder primitive at **every** stud position from parametric rule (N, N_y, spacing). Inside cylinder → f &lt; −margin; optional shell outside. Stud centers from same formula as ProblemLego1xN. |
| **phantom_stud** | Rule enforcement | Penalize negative SDF at points just outside the brick at stud height (same tile pattern as studs but past the edge). Encourages use of dist_to_edge. |
| **outside_brick_sdf** | Outside = positive SDF | Where `dist_to_edge < -margin`, penalize f &lt; margin so “clearly outside” stays positive. Uses `dist_to_edge_x_from_z` (with z / n_norm_override). |

### 6.2 Optional / regularization

| Loss | Role | When |
|------|------|------|
| **prior** | Conditional prior p(z_base \| c) | When `lambda_prior > 0`; supports `sample_z_from_prior_prob`. |
| **latent_structure** | Healthy latent space | Variance, magnitude cap, optional SDF-diversity. |
| **curv** | Surface smoothness | From `start_curv`; curvature expression at surface points. |
| **div** / **chamfer_div** | Latent diversity | From `start_div`; requires surface points. If surface points fail, step is skipped. |

### 6.3 Disabled in current config

| Loss | Purpose when enabled |
|------|----------------------|
| **lambda_unseen_violation: 0** | Would penalize interface+envelope on unseen N (e.g. 1x3) for generalization. |
| **lambda_interp_midpoint: 0** | Would add eikonal + smooth-z at midpoint between two brick-type z’s. |
| **lambda_if / lambda_env / lambda_if_normal: 0** | Interface/envelope/normal from mesh; we use rule-based geometry instead. |

### 6.4 Field losses (from base_config)

- `field_losses`: comp, vol, chamfer_div, wasserstein_div. LEGO config sets lambda_comp/lambda_vol to 0; chamfer_div and diversity-related field terms are used when their lambdas &gt; 0 and surface points exist.

---

## 7. Sampling and Primitives

- **Domain / envelope / interface:** Provided by `ProblemLego1xN` (sample_from_domain, sample_from_envelope, sample_from_interface, etc.).
- **Cuboid and studs:** `GINN/problems/sampling_primitives.py`: box interior, near faces, corners, far field, boundary SDF; cylinder interior and outside shell. Stud centers from parametric formula in `losses_ginn._stud_centers_parametric` (same convention as problem).

---

## 8. Evaluation and Run Readiness

- **Unseen N:** Optional evaluation every K epochs: mean violation (or other metric) over fixed unseen N and a few z samples; logged to wandb.
- **Readiness:** See `docs/LEGO_RUN_READINESS.md`. Fixes in place: violation buffer NaN handling, ALM scalar clamping, skip backward when surface points fail (epoch ≥ start_div), skip ALM update on skip step, compute_grad_norm for empty grads. Verdict: **good to run** for “complete 5000 epochs without crashing.”

---

## 9. Summary Table: Key Design Choices

| Aspect | Current choice |
|--------|----------------|
| **Conditioning** | N (stud count), N_y (1xN only), height; optional N-encoder (9D, no raw n in decoder). |
| **Decoder shortcut** | Prevented: 9th encoder dim only for dist_to_edge; backbone sees 8D from encoder. |
| **Geometry inputs** | Tiled coords (stud-periodic); dist_to_edge (with n_norm_override when encoder on). |
| **Supervision** | Rules only: cuboid, stud grid, phantom_stud, outside_brick_sdf, eikonal; no mesh/point regression. |
| **Unseen / interpolation** | Unseen violation and interp_midpoint losses disabled (lambdas 0). |
| **z sampling** | Violation buffer + random fraction; optional conditional prior. |
| **Diversity** | From start_div; backward skipped when surface points fail. |

This methodology reflects the codebase as of the latest edits: encoder shortcut fix, 9-dim encoder (8+1), no raw n_norm in z, and interpolation/unseen losses turned off.
