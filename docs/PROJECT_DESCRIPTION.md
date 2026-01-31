# Geometry-Informed Neural Networks (GINNs)

## What is GINN?

**Geometry-Informed Neural Networks (GINNs)** is a deep learning framework for generating 3D shapes **without training data**. Instead of learning from examples, GINNs learn to satisfy geometric constraints and physical objectives directly—producing diverse, valid designs that meet engineering requirements.

The key insight is that many design problems can be expressed as constraints:
- "The shape must connect these mounting points"
- "The shape must fit inside this bounding box"
- "The shape must avoid these obstacles"
- "The shape should be structurally stiff"
- "The shape should be a single connected piece"

GINNs encode these requirements as differentiable loss functions and train a neural network to satisfy all of them simultaneously.

---

## How It Works

### Implicit Shape Representation

GINNs represent shapes as **implicit functions** using a neural network $f(x, z)$:

- **Input $x$**: A 3D coordinate $(x, y, z)$
- **Input $z$**: A latent vector controlling shape variation
- **Output**: A scalar value where $f(x,z) = 0$ defines the surface

Points where $f < 0$ are inside the shape, $f > 0$ are outside. This representation is:
- Continuous and differentiable (enables gradient-based optimization)
- Resolution-independent (can extract meshes at any detail level)
- Naturally handles complex topology

### The Training Loop

```
1. Sample latent vectors z₁, z₂, ..., zₙ
2. Sample spatial points from domain, boundaries, obstacles
3. Evaluate neural network f(x, zᵢ) at these points
4. Compute constraint losses:
   - Interface loss: f(boundary_points) = 0
   - Envelope loss: shape stays inside design region
   - Eikonal loss: |∇f| = 1 (valid signed distance field)
   - Connectivity loss: single connected component
   - Compliance loss: structural stiffness (via FEM)
5. Backpropagate and update network weights
6. Repeat until convergence
```

### Extracting Meshes

Once trained, 3D meshes are extracted using **marching cubes**:
1. Evaluate $f(x, z)$ on a dense 3D grid
2. Find the zero-level isosurface
3. Output triangle mesh (vertices + faces)

---

## Key Components

### Neural Network Architectures

| Model | Activation | Best For |
|-------|------------|----------|
| **WIRE** | Gabor wavelets: $\cos(\omega x) \cdot e^{-\sigma^2 x^2}$ | High-frequency details, sharp features |
| **SIREN** | Sinusoidal: $\sin(\omega x)$ | Smooth surfaces, derivatives |
| **FFN** | ReLU/Tanh | General purpose |

### Loss Functions

| Loss | Mathematical Form | Purpose |
|------|-------------------|---------|
| **Interface** | $\|f(x_{boundary})\|^2$ | Surface passes through specified points |
| **Envelope** | $\max(0, -f(x_{outside}))^2$ | Shape stays within design domain |
| **Obstacle** | $\max(0, f(x_{obstacle}))^2$ | Shape avoids forbidden regions |
| **Eikonal** | $(\|\nabla_x f\| - 1)^2$ | Valid signed distance field |
| **Connectivity** | Persistent homology | Single connected component |
| **Compliance** | FEM simulation | Structural stiffness |
| **Volume** | $(\text{vol} - \text{target})^2$ | Material usage constraint |

### Latent Space

The latent vector $z$ enables **generative** capabilities:
- Different $z$ values produce different valid shapes
- Interpolating between $z$ values morphs between designs
- The latent space is organized—similar $z$ → similar shapes

---

## Problem Types

### SimJEB (Simulation-ready Joint Element Brackets)
3D structural brackets with:
- Fixed interface points (mounting holes)
- Complex envelope constraints
- Structural compliance objectives

### Topology Optimization (TOM)
Classic engineering problems:
- **Cantilever beam**: Fixed on one end, loaded on the other
- **MBB beam**: Half-symmetric beam with distributed load
- **Wheel**: Rotational symmetry constraints

### Minimal Surfaces
Mathematical problems:
- Find surfaces with zero mean curvature
- Soap-film-like shapes spanning boundaries

---

## Project Structure

```
GINNs/
├── run.py                    # Entry point
├── configs/                  # YAML experiment configurations
│   ├── base_config.yml       # Default settings
│   ├── GINN/                 # Shape generation configs
│   └── TOM/                  # Topology optimization configs
├── GINN/
│   ├── problems/             # Problem definitions (SimJEB, cantilever, etc.)
│   ├── ph/                   # Persistent homology (topology)
│   ├── fenitop/              # FEM for structural analysis
│   └── evaluation/           # Metrics and evaluation
├── models/
│   ├── wire.py               # WIRE network (Gabor activations)
│   ├── siren.py              # SIREN network
│   └── net_w_partials.py     # Derivative computation wrapper
├── train/
│   ├── ginn_trainer.py       # Main training loop
│   ├── losses_ginn.py        # Constraint loss functions
│   └── losses_diversity.py   # Diversity losses for generative training
├── notebooks/
│   ├── GINN_from_checkpoint.ipynb    # Load and visualize trained models
│   └── minimal_surface.ipynb         # Standalone minimal surface example
└── util/                     # Utilities (checkpointing, visualization, etc.)
```

---

## Usage

### Training a Model

```bash
# Single shape optimization
python run.py --config configs/GINN/simjeb_wire_singleshape.yml

# Multi-shape generative model
python run.py --config configs/GINN/simjeb_wire_multishape.yml

# 2D topology optimization
python run.py --config configs/TOM/cantilever_2d_wire_single.yml
```

### Configuration

Experiments are configured via YAML files:

```yaml
model:
  model_str: 'cond_wire'      # Network architecture
  nx: 3                        # Spatial dimensions
  nz: 2                        # Latent dimensions
  layers: [128, 128, 128]      # Hidden layer sizes

latent_sampling:
  ginn_bsize: 9                # Number of shapes per batch
  z_sample_method: 'equidistant'

# Loss weights
lambda_if: 1.0                 # Interface constraint
lambda_env: 1.0                # Envelope constraint
lambda_eikonal: 0.1            # SDF regularity
lambda_scc: 10.0               # Connectivity
lambda_comp: 1.0               # Structural compliance
```

### Loading a Trained Model

```python
from util.checkpointing import load_yaml_and_drop_keys
from util.misc import get_model
from models.net_w_partials import NetWithPartials

# Load config and model
config = load_yaml_and_drop_keys('checkpoints/GINN-config.yml')
model = get_model(**config['model'])
model.load_state_dict(torch.load('checkpoints/GINN-model.pt'))

# Create derivative-enabled wrapper
netp = NetWithPartials.create_from_model(model, config['nz'], config['nx'])

# Generate shape for latent z
z = torch.tensor([0.5, 0.5])
mesh = get_watertight_mesh_for_latent(netp.f_, netp.params, z, bounds, resolution=128)
```

---

## Key Innovations

1. **Data-free learning**: No dataset required—constraints define valid shapes
2. **Differentiable constraints**: All requirements are encoded as smooth loss functions
3. **Topological control**: Persistent homology ensures connected shapes
4. **Physics integration**: FEM-based compliance for structural optimization
5. **Generative capability**: Latent space enables diverse solution exploration
6. **Augmented Lagrangian**: Adaptive constraint handling for robust optimization

---

## Dependencies

- **PyTorch**: Neural network framework
- **NumPy/SciPy**: Numerical computing
- **Trimesh**: Mesh processing
- **K3D/PyVista**: 3D visualization
- **cripser** (optional): Persistent homology computation
- **FEniCS** (optional): Finite element analysis

---

## References

Based on the paper: [Geometry-Informed Neural Networks](https://arxiv.org/abs/2402.14009)

> GINNs enable the generation of diverse solutions while learning an organized latent space, all without requiring training data—only geometric and physical constraints.
