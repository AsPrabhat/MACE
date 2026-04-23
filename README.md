# MACE: Multilayer Atomic Cluster Expansion

A modular repository for implementing the Multilayer Atomic Cluster Expansion (MACE) potential, built in PyTorch. The project is driven by interactive Jupyter notebooks, with the core mathematical engine residing in the `src/` directory. It focuses on pedagogical clarity, emphasizing explicit equivariant message passing and low body-order symmetric contractions.

## 🚀 Quick Start

This project uses `uv` for lightning-fast dependency management. We provide a PowerShell script to automatically install dependencies, set up the virtual environment, and register the Jupyter kernel.

1. **Clone the repository:**
   ```powershell
   git clone https://github.com/AsPrabhat/MACE
   cd MACE
   ```

2. **Install the package and dependencies:**
   You can either run the provided PowerShell script which also installs the Jupyter kernel:
   ```powershell
   powershell -ExecutionPolicy Bypass -File .\setup_environment.ps1
   ```
   *Or* install it manually using `uv` (Mac/Linux/Windows):
   ```bash
   uv venv
   source .venv/bin/activate  # On Windows use: .\.venv\Scripts\Activate.ps1
   uv pip install -e .
   ```

3. **Verify the mathematical engine:**
   ```powershell
   # Asserts strict O(3) rotation equivariance and exact autograd forces
   pytest tests/
   ```

## 📂 Repository Structure

```text
mace-project/
├── pyproject.toml              # Dependencies (PyTorch, PyG, e3nn, ASE)
├── setup_environment.ps1       # Environment bootstrapping script
│
├── src/                        # The MACE Mathematical Engine
│   ├── data.py                 # Graph construction (ASE -> PyG)
│   ├── basis.py                # Radial (Bessel) & Angular (Spherical Harmonics) embeddings
│   ├── blocks.py               # Equivariant message passing and symmetric contractions
│   ├── model.py                # MACE assembly and force computation via autograd
│   ├── training.py             # Energy & Force loss functions
│   └── utils.py                # Logging and device helpers
│
├── notebooks/                  # Interactive Workflow Driver
│   ├── 01_data_and_graphs.ipynb
│   ├── 02_basis_functions.ipynb 
│   ├── 03_message_passing.ipynb 
│   ├── 04_mace_architecture.ipynb
│   ├── 05_energy_force_training.ipynb
│   └── 06_evaluation.ipynb      
│
└── tests/                      # Mathematical Verification
    ├── test_equivariance.py    # Asserts E(Rx) = E(x) and F(Rx) = R F(x)
    └── test_autograd.py        # Asserts F = -\nabla E via gradcheck
```

## Architecture

- **Dependency Minimization:** Unlike heavy-duty MACE implementations, this repository implements the neighbor search (`radius_graph`) in pure PyTorch. This entirely removes the notoriously difficult `torch_cluster` C++ dependency, ensuring that the package installs flawlessly on standard student hardware (Windows/Mac/Linux) without complex CUDA wheel resolution.
- **Equivariant Message Passing:** Uses `e3nn` to mix invariant radial features with equivariant angular edge features (Spherical Harmonics) and node features.
- **Symmetric Contractions:** To maintain pedagogical clarity, the implementation limits the contraction to a low body-order (e.g. $v=2$) to keep the theoretical complexity accessible.
- **Optimization:** Jointly optimizes Total Energy regression and Force-matching. Forces are explicitly computed as the negative gradient of the energy with respect to positions using PyTorch's `autograd.grad`.
