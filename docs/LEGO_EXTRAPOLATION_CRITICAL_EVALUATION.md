# Why the LEGO GINN Model Is Poor at Extrapolation

This document critically evaluates why a model trained on a **single brick length** (e.g. 1×4 only) extrapolates poorly to unseen lengths (e.g. 1×8, 1×12).

---

## 1. All Training Signal Is Single-Length

**What happens in training**

- Each step, `self.problem` is one brick type (e.g. 1×4). Every sampler is tied to that problem:
  - **Interface:** `p_sampler.sample_from_interface()` → points on **1×4** studs and walls only.
  - **Envelope:** `sample_from_envelope()` / outside / far_outside → points around the **1×4** box.
  - **Domain / inside:** `sample_from_domain()`, `sample_from_inside_envelope()` → inside the **1×4** region only.

So every loss term (interface, envelope, eikonal, SCC, curvature, etc.) sees **only 1×4 geometry**. The network never gets gradients for “what the SDF should be at 1×8” or “what the surface should look like there.”

**Implication:** The learned weights are optimized to satisfy the PDE and constraints **only** on the 1×4 domain and its boundary. There is no incentive for the field to be correct at 1×8.

---

## 2. Zero-Level Set Is Only Supervised at Trained Length

**Interface loss** explicitly trains “SDF = 0” (and normals) at sampled interface points. Those points are **only** on the 1×4 surface (stud tops, stud sides, six walls).

- So the **zero-level set** is only correct where it was supervised: on 1×4.
- For 1×8, the **formulas** (e.g. `dist_to_edge_x`, tiled coords) give the right length and tile structure, but the **network** was never trained to output 0 at the 1×8 boundary.
- Result: at 1×8 you get wrong SDF values (wrong sign, wrong magnitude) near the true boundary → broken or noisy isosurfaces, extra components, holes.

The same holds for **normals**: normal loss is only applied on 1×4 interface points, so gradient direction is not corrected at 1×8.

---

## 3. Eikonal and Regularity Only in the Trained Domain

**Eikonal loss** enforces |∇f| = 1 and is computed on `p_sampler.sample_from_domain()`, which is the **current problem’s domain** (1×4 when you only train on 1×4).

- So the “unit gradient” constraint is only enforced **inside the 1×4 region**.
- In the **extended** 1×8 region (the extra length), the network was never asked to have |∇f| = 1. Gradient norm there can be arbitrary → distorted level sets when marching cubes, stretched or compressed isosurfaces, and unstable normals.

Same idea for any loss that uses domain / inside-envelope sampling: they only regularize the 1×4 region.

---

## 4. SCC and Connectivity Are Length-Specific

**Single-connected-component (SCC)** and related losses are computed using the **current** problem’s geometry (surface points, persistence, etc.). When training on 1×4 only:

- Connectivity is encouraged only for the **1×4** shape.
- There is no supervision that the **1×8** mesh should be one connected component or that phantom studs outside the brick should be suppressed.

So extrapolated 1×8 meshes can easily be disconnected or have extra blobs where the network was never trained to be “empty.”

---

## 5. Inductive Biases Exist but Aren’t Trained for Extrapolation

**dist_to_edge** and **tiled coords** are good inductive biases: they extend to any N (length and tile structure are formula-driven). So in principle the network *could* generalize:

- “Inside brick” (dist_to_edge > 0) → use tiled pattern, SDF = 0 on studs.
- “Outside brick” (dist_to_edge < 0) → ignore tiled pattern, SDF > 0.

But:

- The network **learns** how to use these inputs from the **only** data it sees: 1×4.
- It can satisfy the 1×4 losses by **memorizing** “output 0 here, positive there” for 1×4 positions, without ever relying on dist_to_edge for **novel** regions (e.g. the extra length of 1×8).
- So the model may **underuse** or **ignore** dist_to_edge / tiled coords in regions it never saw during training. Extrapolation then fails because the learned policy is “do what works on 1×4” rather than “follow dist_to_edge and tiles everywhere.”

---

## 6. Shape Latent z Is Optimized for One Length

The **shape latent** (first dimensions of z) is optimized only with 1×4 batches. So it tends to encode 1×4-specific information (stud sharpness, wall flatness, etc.) that minimizes 1×4 loss.

When you **condition on 1×8** you keep the same (or similar) base latent and only change the “N” conditioning. That latent was never trained to be “a good shape code for 1×8,” so the combination (z_base, n_norm=1×8) is out-of-distribution and can produce inconsistent or poor geometry.

---

## 7. High Capacity and Single-Length Overfitting

**WIRE / Gabor** is very expressive. With only 1×4 in the loss:

- The network can fit 1×4 very well without needing to use the conditioning and inductive biases in a **general** way.
- So it can overfit to “this exact 1×4 surface” rather than learning “LEGO brick for any N.” There is no architectural or training pressure that forces the same rule to apply for N = 8 or 12.

---

## 8. Summary: Why Extrapolation Fails

| Cause | Effect |
|-------|--------|
| All sampling from 1×4 problem | No gradient signal at 1×8; weights never updated for extended length. |
| Interface loss only on 1×4 | Zero-level set (and normals) wrong at 1×8 boundary → bad mesh. |
| Eikonal only in 1×4 domain | \|∇f\| wrong in 1×8 region → distorted isosurfaces. |
| SCC/connectivity for 1×4 only | No incentive for 1×8 to be one component or to suppress phantoms. |
| Inductive biases underused | Network may not “trust” dist_to_edge/tiles in unseen regions. |
| Latent z tuned for 1×4 | (z_base, 1×8) is OOD; shape style doesn’t transfer. |
| High capacity + single length | Overfit to 1×4 geometry instead of a general brick rule. |

---

## 9. Recommendations to Improve Extrapolation

1. **Train on multiple N**  
   Include several lengths in `n_studs_values` (e.g. 2, 4, 6, 8). Then:
   - Interface, envelope, eikonal, and SCC all see multiple lengths.
   - “Extrapolation” to 8 or 12 becomes **interpolation** in conditioning space (or mild extrapolation), which is much more reliable.

2. **Optional: light extrapolation loss**  
   For a few points in the 1×8 (or 1×12) region, add a small loss that encourages:
   - SDF sign consistent with dist_to_edge (e.g. positive outside brick), and/or
   - Rough SDF = 0 at the 1×8 boundary (if you can sample it).
   This gives a direct signal that “the rule should extend to 1×8.”

3. **Regularization / capacity**  
   If you still see single-length overfitting when adding more N, consider slightly smaller width/depth or mild weight decay so the model is encouraged to use the shared structure (tiles, dist_to_edge) rather than memorizing per-length detail.

4. **Latent sampling**  
   When visualizing or evaluating 1×8, try several base latents (e.g. from 1×4 runs or resampled). Some may extrapolate better than others; the “best” 1×8 shape might correspond to a latent that generalizes more.

5. **Validate on held-out N**  
   Train on e.g. N ∈ {2, 4, 6} and hold out N = 8 (or 5, 7). Monitor mesh quality and connectivity on the held-out N to measure extrapolation and tune the above.

---

*Summary: The model is bad at extrapolation because every loss is computed only on the geometry it was trained on (e.g. 1×4). The zero-level set, eikonal, and connectivity are never supervised at 1×8, and the network can satisfy 1×4 without learning a general “any N” rule. Training on multiple N is the main lever to fix this.*
