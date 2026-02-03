# LEGO GINN: Losses and Constraints

This document explains each **constraint** (where we sample points and what we want the neural field to satisfy) and each **loss** (what we minimize during training) used in LEGO 1xN/NxN training. Config references are for `configs/GINN/lego_1xN_wire.yml` and base config.

---

## 1. Constraints (point regions)

Constraints define **where** we evaluate the field and **what** we want there. The LEGO problem samples from these regions each step.

### 1.1 Interface (surface where SDF = 0)

- **What:** Points on the true boundary of the brick: SDF should be **0** and the gradient **∇f** should match the **outward normal**.
- **LEGO regions:**
  - **Stud tops:** Circular disks on top of each stud (outward normal = +z).
  - **Stud sides:** Cylindrical surfaces of each stud (radial normals).
  - **Walls:** Six faces of the brick body (bottom, top of body, ±x, ±y) with normals ±x, ±y, ±z.
- **Sampling:** `sample_from_interface()` returns points and target normals. LEGO overrides this to give **50% of points to studs** (tops + sides) and **50% to walls** so studs get enough gradient signal.
- **Config:** `n_points_interfaces`, `n_points_normals` (often same as interfaces). Lambdas: **`lambda_if`**, **`lambda_if_normal`**.

### 1.2 Envelope (shape must stay inside)

- **What:** Points **outside** the desired shape. For SDF: we want **f ≥ 0** there (no material). For density: we want value **≤ level_set**.
- **LEGO regions (SimJEB-style):**
  - **Far outside:** Points far from the brick (strong “no material”).
  - **Outside envelope:** Just outside the brick surface.
  - **Around interface:** Near the boundary.
- **Sampling:** `sample_from_envelope()` (combines these regions).
- **Config:** `lambda_env`. Obstacles (`lambda_obst`) are **0** for LEGO.

### 1.3 Domain (for Eikonal)

- **What:** A set of points **inside and just outside** the brick (inside envelope + outside envelope). We do **not** enforce f = 0 at these points—they are not on the surface. We only enforce **Eikonal**: gradient norm **|∇f| = 1** at each of these points.
- **Why Eikonal?** For a true signed distance function (SDF), the value f(x) is the distance to the nearest surface, so the gradient ∇f points toward the nearest surface and has **length 1** almost everywhere. Enforcing |∇f| = 1 in the domain makes the learned field behave like a distance function: stable gradients, a clean zero-level set, and f(x) roughly meaning “distance to boundary.”
- **Sampling:** `sample_from_domain()` returns points from the composite: **inside** the brick + **outside** (but near) the brick.
- **Config:** `n_points_domain`, **`lambda_eikonal`**.

### 1.4 Inside envelope (volume / diversity)

- **What:** Points inside the brick. Used for volume-style losses or diversity (e.g. SDF/density values over this set).
- **Sampling:** `sample_from_inside_envelope()`.
- **Config:** Used by diversity (and optionally volume) losses; LEGO uses diversity on surface, not inside envelope.

---

## 2. Losses (what we minimize)

Total loss is a weighted sum. Weights are either **fixed** (ManualWeightedLoss) or **adapted** (Augmented Lagrangian: lambdas + multipliers μ). Below: **name**, **role**, **formula/idea**, **LEGO config**.

### 2.1 Interface loss (`if`)

- **Role:** Fit the zero-level set to the target boundary (studs + walls).
- **Formula:** At interface points **x**, minimize **MSE(f(x), level_set)**. Usually `level_set = 0`, so we push **f(x) → 0** on the boundary.
- **Config:** **`lambda_if: 22`** (strong so surfaces stay flat and interface doesn’t drift).

### 2.2 Interface normal loss (`if_normal`)

- **Role:** Match the **outward normal** on the boundary so the surface orientation is correct (flat walls, correct stud direction).
- **Formula:** At the same interface points, minimize **MSE(∇f(x) / |∇f|, n_target)** (or MSE(∇f, n_target) with scaling). For density, gradients are normalized to unit vectors first.
- **Config:** **`lambda_if_normal: 2.0`** (stronger normals for flatter walls/studs).

### 2.3 Envelope loss (`env`)

- **Role:** Keep the shape **inside** the envelope: no material outside (SDF ≥ 0 outside, or density ≤ level_set).
- **Formula:**
  - **SDF:** Sum of **(f(x) − level_set)²** only for points where **f(x) < level_set** (penalize “inside” outside the envelope).
  - **Density:** Same idea, penalize density above level_set outside.
- **Config:** **`lambda_env: 1`**. Obstacle loss **`lambda_obst: 0`** (unused for LEGO).

### 2.4 Eikonal loss (`eikonal`)

- **Role:** Make the field a **valid SDF**: gradient norm **|∇f| = 1** almost everywhere in the domain.
- **Formula:** At domain points, **mean((|∇f| − 1)²)**.
- **Config:** **`lambda_eikonal: 1.0`**, **`scale_eikonal: 0.01`**. Domain points: **`n_points_domain: 2048`**.

### 2.5 SCC loss – Single Connected Component (`scc`)

- **Role:** Encourage **one connected component** and **no holes**: topology of a solid brick (Betti: β₀=1, β₁=0).
- **Mechanism:** Persistent homology (PH) on a 3D grid of field values (inside envelope). PH finds “birth”/“death” of components and holes. The loss penalizes:
  - **Dim 0 (components):** Extra components. At grid cells where a 2nd, 3rd, … component dies, penalize **clamp(iso − f(x), 0)²** so f is pushed above the iso level there (merge components).
  - **Dim 1 (holes):** Unwanted holes. Penalize birth/death of 1-cycles so their birth/death values move toward ±hole_level and disappear.
- **Config:** **`lambda_scc: 35`**. PH: **`ph_loss_target_betti: [1, 0, 0]`**, **`iso_level`**, **`ph_1_hole_level`**, **`scc_n_grid_points: 32`**, **`ph_max_num_workers: 4`**.

### 2.6 Curvature loss (`curv`)

- **Role:** **Regularize** the surface: prefer smoother curvature (avoid noisy or spiky zero-level sets).
- **Formula:** On **surface points** (zero-level set), compute gradient **F = ∇f** and Hessian **H**, then mean curvature **H** and (optionally) Gauss curvature **K**. Loss = weighted sum of an expression such as **4*H² − 2*K** (elasticity-style), clipped and optionally weighted by gradient norm. Only active when **curvature > max_curv** (e.g. **max_curv: 0** so we always penalize positive curvature above 0).
- **Config:** **`lambda_curv: 1`**, **`scale_curv: 5e-4`**, **`start_curv: 500`**, **`max_curv: 0`**. **`curvature_expression`**, **`curvature_pts_source: surface`**. **`exclude_surface_points_close_to_interface_cutoff: 0.04`** so we don’t over-penalize stud/wall curvature.

### 2.7 Diversity loss (`div`)

- **Role:** In a batch of different shapes (different z or conditions), encourage **different** SDF/density outputs so latents don’t collapse to the same shape.
- **Formula:** Evaluate the field at many points (e.g. **surface** or **inside envelope**) for each shape—**the same point locations** for all shapes. You get one vector of values per shape: (f(x₁), f(x₂), …). Compute pairwise **L2 distance between these vectors** (shape “signatures” in value space). Aggregate per shape (e.g. **min** distance to another shape), then **− (aggregate)²** so we maximize minimum distance. Often **clamped**: loss = 0 if diversity already **≥ max_div** (e.g. **max_div: −0.1**).
- **Config:** **`lambda_div: 1`**, **`scale_div: 0.03`**, **`start_div: 2000`**, **`max_div: −0.1`**. **`diversity_pts_source`** (LEGO: surface). **`div_norm_order: 2`**.

### 2.8 Chamfer diversity (`chamfer_div`)

- **Role:** Encourage shapes in a batch to be **geometrically different** by maximizing Chamfer distance between surface point clouds.
- **Formula:** For each shape, take **surface points** (different positions per shape—each shape’s own zero-level set). Optionally move them along ∇f. Then compute **Chamfer distance** between pairs of point clouds: average nearest-neighbor distance in **3D space**. So you compare *where* the surfaces are, not the field values at fixed x.
- **Config:** LEGO inherits **`lambda_chamfer_div: 0`** from base (not used).

**Difference between `div` and `chamfer_div`:**  
- **`div`** compares **field values** at a **fixed** set of points: “Do different z give different f(x) at the same x?” (value-space diversity).  
- **`chamfer_div`** compares **where the surface is** in 3D: “Are the zero-level sets of different shapes far apart in space?” (geometry-space diversity). So `div` = same x, different f; `chamfer_div` = different point clouds (different x per shape), distance in ℝ³.

### 2.9 Compliance / Volume (`comp`, `vol`)

- **Role:** Topology optimization (TO): minimize compliance subject to volume. Used in 2D/3D TO problems.
- **Config:** LEGO inherits **`lambda_comp: 0`**, **`lambda_vol: 0`** from base (not used).

### 2.10 Objective

- **Role:** One loss can be designated the main “objective” (e.g. curvature or compliance). Others then act as constraints (in ALM) or regularizers.
- **LEGO:** **`objective: 'null'`** so there is **no** single objective; all of the above are just weighted terms. This avoids curvature dominating and drifting the interface fit.

---

## 3. Summary table (LEGO-relevant)

| Loss / Constraint | Purpose | LEGO weight / note |
|-------------------|--------|---------------------|
| **Interface (if)** | SDF = 0 on boundary (studs + walls) | λ 22 |
| **Interface normal (if_normal)** | ∇f = outward normal on boundary | λ 2 |
| **Envelope (env)** | f ≥ 0 outside (shape inside envelope) | λ 1 |
| **Eikonal** | \|∇f\| = 1 in domain | λ 1, scale 0.01 |
| **SCC** | One component, no holes (PH) | λ 35 |
| **Curvature (curv)** | Smooth surface (4H²−2K) | λ 1, from epoch 500 |
| **Diversity (div)** | Different z → different shapes | λ 1, from epoch 2000 |
| **Chamfer div** | Geometric diversity | 0 (off) |
| **Comp / Vol** | TO compliance/volume | 0 (off) |
| **Obstacle** | Forbidden regions | 0 (off) |

---

## 4. Constraint regions (LEGO) – quick reference

| Region | Used by | Goal |
|--------|--------|------|
| Interface (stud tops/sides + 6 walls) | `if`, `if_normal` | f = 0, ∇f = n |
| Far outside / outside / around interface | `env` | f ≥ 0 (or ρ ≤ level_set) |
| Domain (inside + outside) | `eikonal` | \|∇f\| = 1 |
| Inside envelope | optional for div/vol | diversity or volume |
| Surface (numerical) | `curv`, `div` (if surface) | curvature reg., diversity |

All of the above together define "what the network should satisfy" (constraints) and "what we minimize" (losses) for LEGO GINN training.

---

## 5. How the model learns N×N and different heights

**Yes**—the model is trained to produce the right brick type (N×N and height)—but **no single loss** is named "learn N×N" or "learn height." It happens through **conditioning in z** and the **same geometry losses** applied per brick type.

1. **Conditioning in z**  
   Each training batch, one brick type (N, N_y, height) is chosen at random. The latent **z** is then extended with **normalized** (N, N_y, height) and fed to the network: the model sees **(x, z)** where **z = [z_base | n_norm | ny_norm | h_norm]** (e.g. base nz=2 plus 3 conditioning dimensions). So the network input explicitly encodes "which brick type."

2. **One problem per batch**  
   For that batch, **self.problem** is set to the problem for that (N, N_y, height): its interface (studs + walls), envelope, bounds, and domain sampling. All geometry losses (interface, envelope, eikonal, if_normal, SCC, curvature) use **this** problem's constraints and sampling. So the model is trained to satisfy the **current** brick type's geometry.

3. **What actually teaches N×N and height**  
   The **interface loss**, **envelope loss**, **eikonal**, **if_normal**, and **SCC** (and curvature) are what train the geometry. They are applied with the **current** problem's interface/envelope/bounds. Over many batches the model sees 1×2, 1×4, 2×4, different heights, etc., each with the correct target surface and envelope, and **z** tells it which type. So the model learns: "when z has this (N, N_y, h) suffix, produce an SDF that matches that brick type's boundary and interior."

4. **Diversity loss**  
   **Diversity (div)** encourages **different z_base** (the first nz dimensions) to give **different** shapes **within the same brick type**. It does **not** teach N×N or height; those are fixed in the appended part of z for the batch. So: conditioning in z = "which brick type"; diversity = "different shapes for the same type."

