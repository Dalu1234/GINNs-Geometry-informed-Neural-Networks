# Hypernetwork + LoRA — Critical Audit

Double-check of the implementation as of the last review.

---

## 1. z layout vs c

- **Trainer** builds z by appending columns in this order: `n_norm` (condition_on_n_studs), `ny_norm` (condition_on_n_studs_y), `h_norm` (condition_on_height). So `z = [z_latent..., n_norm, ny_norm, h_norm]`.
- **Model** uses `c = z[:, -3:]` with `c_dim=3`, so `c = (n_norm, ny_norm, h_norm)` = `(Nx_norm, Ny_norm, height_norm)`. **Correct.**
- **Contract**: With `use_hypernet=True` and `c_dim=3`, all three conditioning options must be on so the last 3 columns are always (Nx_norm, Ny_norm, height_norm). If e.g. only `condition_on_n_studs` is on, z has 3 cols and `z[:, -3:]` would be the full z (latent + n_norm), which is wrong. **Assertion added**: `z.shape[-1] >= c_dim` in `wire.py` forward.

---

## 2. Tensor shapes (LoRA application)

- **h**: `(B, H)` after backbone.
- **U**: `(B, 1, r)`, **V**: `(B, r, H)` from hypernet(c).
- **y_lora** = `(h.unsqueeze(1) @ V.transpose(1,2) @ U.transpose(1,2)).squeeze(-1)` → `(B, 1, 1)` → squeeze(-1) → `(B, 1)`. **Correct.**
- Per-sample: `y_lora[i] = h[i] @ (U[i] @ V[i]).T`; `(U@V).T` is `(H, 1)`, `h[i]` is `(H,)`, result `(1,)`. **Correct.**

---

## 3. vmap / unbatched path (recalc_output, get_mesh)

- **recalc_output** calls `netp.vf(*tensor_product_xz(x, z_.unsqueeze(0))).squeeze(0)`. So x `(N, 3)`, z `(N, 5)`; vmap calls the model N times with x `(3,)`, z `(5,)`.
- **Model** detects `x.dim() == 1`, does `x = x.unsqueeze(0)`, `z = z.unsqueeze(0)`, runs forward, then `output.squeeze(0)`. So each call returns `(1,)` for ny=1.
- **vmap** with `out_dims=(0)` stacks N × `(1,)` → `(N,)`.
- **problem_base** does `.squeeze(0)` on that; for 1D `(N,)`, squeeze(0) leaves `(N,)`.
- **get_mesh** uses the callback so that `vs = model(pts)` has shape `(N,)` (or `(N, 1)`); `vs.reshape(X.shape)` works for the 3D grid. **Correct.**

---

## 4. Batched training path

- Training uses `netp(*tensor_product_xz(x, z))` with x `(B, 3)`, z `(B, 5)`. So model gets x `(B, 3)`, z `(B, 5)`; `unbatched` is False. Forward returns `(B, 1)`. **Correct.**

---

## 5. Hypernetwork init

- **HypernetworkLoRA**: head Linear init with `N(0, init_scale)` then `* (1/sqrt(rank))`, bias zeros. So initial U@V is small. **Correct.**

---

## 6. Backbone / final linear reuse

- `_backbone` = `Sequential(*list(self.net)[:-2])` (with return_density) or `[:-1]` (without). `_final_linear` = `self.net[-2]` or `self.net[-1]`. Same module references as in `self.net`, so parameters are shared. **Correct.**

---

## 7. Summary

| Check | Status |
|-------|--------|
| c = last 3 cols of z = (Nx_norm, Ny_norm, height_norm) | OK (trainer appends in that order) |
| c_dim=3 requires all three conditioning on | Documented + assert in forward |
| LoRA shapes and application | OK |
| Unbatched (vmap) path return shape | OK; (1,) → vmap → (N,) → get_mesh |
| Batched path return shape | OK; (B, 1) |
| Hypernet init | OK |

**Risks if config changes**: Turning off `condition_on_n_studs_y` or `condition_on_height` while keeping `use_hypernet=True` and `c_dim=3` would make `z[:, -3:]` no longer (Nx_norm, Ny_norm, height_norm). The new assert will catch `z.shape[-1] < c_dim`; if z has exactly 3 or 4 columns but in a different order, the hypernet would still get the wrong semantics. So: **keep all three conditioning on when using the hypernetwork**, or change c_dim and the slice to match the actual z layout.
