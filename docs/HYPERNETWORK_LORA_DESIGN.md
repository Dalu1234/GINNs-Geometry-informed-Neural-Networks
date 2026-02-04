# Minimal Hypernetwork + LoRA for LEGO INR (SDF)

## 1. Module design

- **Base INR**: Your existing `ConditionalWIRE` (or any `nn.Module` that takes `(x, z)` and returns SDF). It stays as-is; its parameters are the shared “backbone.”
- **Hypernetwork**: Small MLP that takes conditioning **c** `(Nx_norm, Ny_norm, height_norm)` and outputs **LoRA factors (U, V)** for a **single** modulated layer (the output linear). No one-hot; continuous c so unseen bricks (e.g. 5×6) generalize.
- **Connection**: Add a **residual LoRA term** on top of the base output; do **not** replace weights. So:
  - `h = last_hidden_activation(x, z)` (from base INR),
  - `y_base = base_linear(h)`,
  - `y_lora = h @ (U @ V).T` with `U, V = hypernet(c)`,
  - `y = y_base + y_lora`.

So the base INR is unchanged; you only add one small hypernet and one additive path in the forward.

---

## 2. Tensor shapes (LoRA for one Linear)

- **Base Linear**: `out_features = 1`, `in_features = H` (e.g. 128). So `W_base`: `(1, H)`, bias: `(1,)`.
- **LoRA**: `W_delta = U @ V` with rank `r` (e.g. 4 or 8).
  - **U**: `(out_features, r)` = `(1, r)`.
  - **V**: `(r, in_features)` = `(r, H)`.
  - **U @ V**: `(1, H)`. So `(U @ V).T` is `(H, 1)`.
- **Forward**: `h`: `(B, H)`.  
  - `y_base = h @ W_base.T + b` → `(B, 1)`.  
  - `y_lora = h @ (U @ V).T = h @ V.T @ U.T` → `(B, H) @ (H, r) @ (r, 1)` = `(B, 1)`.  
  So **U**: `(1, r)`, **V**: `(r, H)`.

Summary:

- **U**: `(1, r)` (or more generally `(out_f, r)`).
- **V**: `(r, H)` (or `(r, in_f)`).
- **Application**: `y_lora = (h @ V.T) @ U.T` or `h @ (U @ V).T`; both give `(B, 1)`.

---

## 3. Conditioning embedding

- **Input**: Numeric conditioning **c** ∈ ℝ³, e.g. `(Nx_norm, Ny_norm, height_norm)` in `[0, 1]³`.
  - Same normalization as in your trainer:  
    `Nx_norm = (Nx - Nx_min) / (Nx_max - Nx_min)`, and similarly for Ny and height.
- **No one-hot**: Use **c** as a 3‑dim vector so the hypernet can interpolate/generalize to unseen (e.g. 5×6).
- **Embedding**: Use a tiny MLP: `c_embed = MLP(c)` with 1–2 hidden layers (e.g. 3 → 32 → 32). Then from `c_embed` predict the flat vectors that you reshape into **U** and **V**.
- **Decoding c from z**: In your setup **z** already contains `[z_latent, n_norm, ny_norm, h_norm]`. So **c** = `z[:, -3:]` (last three dims). If you use a different z layout, take the slice that corresponds to `(Nx_norm, Ny_norm, height_norm)`.
- **Critical contract**: When `use_hypernet=True` and `c_dim=3`, the trainer must append conditioning columns in order **n_norm, ny_norm, h_norm** (i.e. all of `condition_on_n_studs`, `condition_on_n_studs_y`, `condition_on_height` must be used). Otherwise `z[:, -3:]` would not be `(Nx_norm, Ny_norm, height_norm)` and the hypernet would be conditioned on the wrong inputs. The code asserts `z.shape[-1] >= c_dim`.

---

## 4. Initialization and regularization

- **Hypernet output (U, V)**:
  - Initialize so that **U @ V** is small at the start: e.g. draw U, V from **N(0, 0.01)** or use a small scale (e.g. 1e-2) so `y_lora` does not dominate `y_base`.
  - Optionally scale the hypernet’s final layer by **1/sqrt(r)** so that the Frobenius norm of **U @ V** is O(1) in expectation.
- **Regularization**:
  - **Option A**: L2 on hypernet parameters (standard weight decay).
  - **Option B**: Penalize **||U @ V||_F** (e.g. add `lambda_lora * ||U@V||_F^2`) so the delta stays small and the base INR remains the main predictor.
- **Stability**: Clip **c** to `[0, 1]` before feeding the hypernet so out-of-range conditioning doesn’t produce huge U, V. Use a small learning rate for the hypernet at first (e.g. 1× or 0.5× the base INR LR).

---

## 5. Minimal forward-pass changes

- **Single place to change**: After you have the **last hidden activation h** and the **base output y_base**:
  1. Compute **c** from **z** (e.g. `c = z[:, -3:]`).
  2. Run **U, V = hypernet(c)**.
  3. Compute **y_lora = h @ (U @ V).T** (or `(h @ V.T) @ U.T`).
  4. Return **y_base + y_lora**.

So the base INR’s internal forward is unchanged; you only need to **expose the last hidden state h** and then add one residual term. That implies either:

- A thin wrapper that runs the base INR in two steps (run up to last hidden, then last linear + LoRA), or  
- A single “output layer” that wraps “base_linear(h) + lora(h; U, V)” and is fed **h** and **c**.

---

## 6. Pitfalls for SDFs / implicit geometry

- **Sign of SDF**: The LoRA term is additive. If the hypernet outputs are large or poorly initialized, you can get **y_base + y_lora** flipping sign and breaking the zero-level set. Hence: **small init** and optional **regularization on ||U@V||**.
- **Eikonal (|∇f| = 1)**: LoRA only changes the **last** linear; gradients of **f** w.r.t. **x** still go through the base INR and the added term. The extra term is linear in **h**, so it can slightly distort the gradient norm. Keeping the LoRA delta small (init + reg) keeps the eikonal loss from being too disturbed.
- **Multi-brick training**: Each batch has one brick type, so **c** is constant per batch. The hypernet learns to output different (U, V) per **c**; gradient interference is reduced because the **additive** part is conditioned on (Nx, Ny, height).
- **Unseen bricks**: Because **c** is continuous, **c** for (5, 6, 1.0) can be passed in and the hypernet will interpolate; no need to train on 5×6 explicitly for the LoRA to produce a plausible delta.

---

## 7. Optional: two modulated layers

If you later want to modulate **two** layers (e.g. last hidden and output):

- **Layer 1** (e.g. last Gabor → hidden): `in_f = H1`, `out_f = H2`. LoRA: **U1** `(H2, r)`, **V1** `(r, H1)`. Apply: `h2 = base_layer1(h1) + (h1 @ V1.T) @ U1.T`.
- **Layer 2** (output): as above, **U2** `(1, r)`, **V2** `(r, H2)`.

The hypernet then outputs **four** matrices (U1, V1, U2, V2). Same init and regularization ideas; just more parameters and a second injection point in the forward (run base in three segments and add two LoRA residuals).

---

## Summary

- **Base INR**: Unchanged; shared backbone.
- **Hypernet**: **c** (3,) → small MLP → flat → reshape to **U** (1, r), **V** (r, H).
- **Forward**: `y = y_base + h @ (U @ V).T` with **c** from **z**.
- **Shapes**: U `(1, r)`, V `(r, H)`; apply as `h @ V.T @ U.T`.
- **Conditioning**: Continuous (Nx_norm, Ny_norm, height_norm); no one-hot.
- **Stability**: Small init for U/V, optional L2 or ||U@V||_F² reg, clip **c** to [0,1].
- **SDF**: Keep LoRA small so zero-level set and eikonal are not broken.

---

## Implementation

- **HypernetworkLoRA**: `models/hypernetwork_lora.py` — MLP(c) → flat → U (B, out_f, r), V (B, r, in_f); small init and optional 1/sqrt(r) scaling.
- **ConditionalWIRE**: `models/wire.py` — When `use_hypernet=True`, `_backbone` and `_final_linear` reuse the same modules as `self.net`; forward computes `h = _backbone(xz)`, `y_base = _final_linear(h)`, `c = z[:, -c_dim:].clamp(0, 1)`, `U, V = hypernet(c)`, `y = y_base + (h @ V^T @ U^T)` (batched), then sigmoid if `return_density`.
- **Config**: `configs/GINN/lego_1xN_wire.yml` — `use_hypernet: false` by default; set to `true` and set `lora_rank`, `c_dim`, `hypernet_hidden`, `lora_init_scale` as needed. No changes to `get_model` or trainer; all options are passed via `config['model']`.
