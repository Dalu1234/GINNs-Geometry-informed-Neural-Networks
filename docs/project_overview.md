# Project Overview: Geometry-Informed Neural Networks (GINNs)

## High-Level Structure

The project is organized into the following key components:

- **Core Modules**: Implementation of GINNs, including data handling, models, training, and evaluation.
- **Configurations**: YAML files for experiment setups.
- **Checkpoints**: Pre-trained models and configurations for resuming experiments.
- **External Libraries**: Third-party tools for geometry and optimization tasks.
- **Notebooks**: Jupyter notebooks for experimentation and visualization.
- **Utilities**: Helper functions for visualization, checkpointing, and configuration management.

---

## Directory Breakdown

### `GINN/`
- **Purpose**: Core implementation of the GINNs framework.
- **Submodules**:
  - `data/`: Data loading and dataset management.
  - `evaluation/`: Tools for evaluating models, including metrics and LaTeX table generation.
  - `models/`: Neural network architectures (e.g., feedforward networks, SIRENs).
  - `train/`: Training logic, loss functions, and utilities.
  - `util/`: General-purpose utilities for configuration, checkpointing, and visualization.

### `configs/`
- **Purpose**: Stores YAML configuration files for experiments.
- **Structure**:
  - `GINN/`: Configurations for GINN experiments.
  - `TOM/`: Configurations for TOM-related experiments.

### `checkpoints/`
- **Purpose**: Stores pre-trained models and configurations.
- **Examples**:
  - `GINN-model.pt`: Pre-trained PyTorch model.
  - `GINN-config.yml`: Configuration file for the model.

### `external/`
- **Purpose**: Third-party libraries integrated into the project.
- **Examples**:
  - `deepsdf/`: Tools for Signed Distance Functions (SDFs).
  - `deflatedbarrier_repo/`: Optimization tools.
  - `mesh_to_sdf/`: Utilities for working with meshes and SDFs.

### `notebooks/`
- **Purpose**: Jupyter notebooks for experimentation and visualization.
- **Examples**:
  - `GINN_from_checkpoint.ipynb`: Demonstrates loading and using pre-trained models.
  - `minimal_surface.ipynb`: Explores minimal surface problems.

### `train/`
- **Purpose**: Training pipelines and loss functions.
- **Examples**:
  - `ginn_trainer.py`: Manages the training process.

### `util/`
- **Purpose**: General-purpose utilities.
- **Examples**:
  - `checkpointing.py`: Manages saving and loading model checkpoints.
  - `vis_utils.py`: Visualization utilities.

---

## Key Files

- **`run.py`**: Likely the entry point for running the project.
- **`README.md`**: Provides an overview of the project.
- **`LICENSE`**: Specifies the licensing terms.

---

## Dependencies

- **Environment Files**:
  - `requirements.txt`: Lists Python dependencies.
  - `requirements-windows.txt`: Windows-specific dependencies.
  - `environment-TOM.yml`: Conda environment setup.

---

## Use Cases

- Geometry-Informed Learning: Solving problems involving geometric constraints.
- Physics-Informed Neural Networks (PINNs): Extending neural networks to solve PDEs.
- Optimization: Using neural networks for constrained optimization problems.