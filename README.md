# Geometry-Informed Neural Networks (GINNs)

## Overview

Geometry-Informed Neural Networks (GINNs) is a framework for training shape generative models without data by satisfying design requirements given as constraints and objectives. GINNs enable the generation of diverse solutions while learning an organized latent space. This project also integrates with the **Topology Optimization Modulated Neural Fields (TOM)** framework for solving topology optimization problems.

### Key Features
- Train generative models without data.
- Support for diverse topology optimization tasks.
- Modular and reusable components for training, evaluation, and visualization.
- Integration with external libraries for geometry and optimization.

---

## Project Structure

```
/
├── run.py                          # Entry point for the program
├── train/                          # Functionality for training
│   └── ginn_trainer.py             # Handles the training loop of the network
├── configs/                        # Contains YAML files to configure experiments
├── GINN/                           # Core implementation of GINNs
│   ├── data/                       # Data loading and dataset management
│   ├── evaluation/                 # Tools for evaluating models
│   ├── models/                     # Neural network architectures
│   ├── util/                       # Utilities for configuration, checkpointing, etc.
├── external/                       # Third-party libraries
├── notebooks/                      # Jupyter notebooks for experimentation
├── requirements.txt                # Python dependencies
├── requirements-windows.txt        # Windows-specific dependencies
├── README.md                       # Project documentation
```

---

## Installation

### Prerequisites
- Python 3.8+
- Conda (recommended) or virtualenv

### Setting Up the Environment

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/GINNs.git
   cd GINNs
   ```

2. Create a virtual environment:
   ```bash
   conda env create -f environment-TOM.yml
   conda activate ginn-env
   ```
   Alternatively, use `requirements.txt` or `requirements-windows.txt` to install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Install additional dependencies for persistent homology (optional):
   ```bash
   pip install cripser==0.0.13
   ```

---

## Usage

### Running the Project
The main entry point for the project is `run.py`. It handles global configurations, initializes the model, and starts the training process.

```bash
python run.py --config configs/GINN/simjeb_wire.yml
```

### Training
- Modify the YAML configuration files in the `configs/` directory to set up your experiment.
- Use `train/ginn_trainer.py` to manage the training loop.

### Evaluation
- Use the `evaluation/` module to compute metrics and visualize results.
- Example notebooks in the `notebooks/` directory demonstrate evaluation workflows.

---

## Dependencies

### Core Libraries
- **PyTorch**: Deep learning framework.
- **NumPy**: Scientific computing.
- **Matplotlib**: Visualization.
- **PyYAML**: Configuration management.

### Geometry and Visualization
- **Trimesh**: Mesh processing.
- **PyVista**: 3D visualization.
- **K3D**: Interactive 3D plotting.

### Persistent Homology (Optional)
- **Cripser**: Library for persistent homology computations.

---

## Contributing

Contributions are welcome! Please follow these steps:
1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Submit a pull request with a detailed description of your changes.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## References
- [Project Page](https://arturs-berzins.github.io/GINN/)
- [arXiv Paper](https://arxiv.org/abs/2402.14009)
- [TOM GitHub](https://github.com/ml-jku/Topology-Optimization-Modulated-Neural-Fields)
