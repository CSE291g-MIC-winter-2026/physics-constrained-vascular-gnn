# gROM — Physics-Informed Graph Neural Network for 1D Blood Flow Simulation

A MeshGraphNet-based surrogate model for cardiovascular hemodynamics, extended with physics-informed constraints (Poiseuille law + continuity) on top of the original [Pegolotti et al. (2024)](https://doi.org/10.1016/j.compbiomed.2023.107676) implementation.

---

## Repository Structure

```
├── graph/
│   ├── generate_dataset.py              # Train/test dataset splitting
│   ├── generate_graphs.py               # Raw graph generation from simulation data
│   └── generate_normalized_graphs.py    # Feature normalization
├── network1d/
│   ├── meshgraphnet.py                  # MeshGraphNet encoder-processor-decoder
│   ├── rollout.py                       # Autoregressive rollout at test time
│   ├── training.py                      # Baseline training (data loss only)
│   ├── training_physical.py             # Physics-constrained: Poiseuille loss
│   ├── training_physical2_normal.py     # Physics-constrained: normalized version
│   ├── training_physical3.py            # Physics-constrained: continuity loss
│   └── training_physical4_inout.py      # Physics-constrained: inlet/outlet weighting
├── test/
│   ├── test_data/                       # Place downloaded .grph files here
│   ├── huatu.py                         # Pressure distribution visualization
│   ├── test_for_time.py                 # Rollout timing and prediction export
│   ├── test_rollout.py                  # Rollout error evaluation
│   └── test_training.py                 # Training sanity check
└── tools/
    ├── io_utils.py                      # File I/O helpers
    └── plot_tools.py                    # Plotting utilities
```

---

## Installation

Tested on **Windows 11** with **CUDA 12.8** and **RTX 4050 Laptop GPU**.

```bash
conda create -n grom python=3.9
conda activate grom

pip install torch==2.3.0+cu121 torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu121

pip install dgl==2.2.1+cu121 \
    -f https://data.dgl.ai/wheels/repo.html

pip install numpy==1.26.4 torchdata==0.7.0
```

> **Note for Windows users:** Set `num_workers=0` in all DataLoader calls to avoid multiprocessing errors.

---

## Data

Download the preprocessed `.grph` graph files from [Google Drive](https://drive.google.com/drive/folders/1IByz6kyouNtNgnOxKrFK4DnAVu2yh6S1).

Place the downloaded files in:

```
test/test_data/graph/
```

The dataset contains ~600 graphs derived from patient-specific vascular geometries (VMR), covering three geometry types:

- `synthetic_aorta_coarctation`
- `synthetic_pulmonary`
- `synthetic_aortofemoral`

Each `.grph` file is a DGL graph with node features including pressure, flowrate, cross-sectional area, tangent vectors, and boundary condition masks (inlet / outlet / branch / junction).

---

## Training

### Baseline (data loss only)

```bash
python network1d/training.py
```

### Physics-Constrained (Poiseuille + Continuity)

```bash
python network1d/training_physical.py
```

Key hyperparameters (editable at the top of each training script):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `epochs` | 15 | Number of training epochs |
| `batch_size` | 150 | Graphs per batch |
| `stride` | 3 | Multi-step training steps |
| `lambda_physics` | 0.01 | Poiseuille loss weight |
| `lambda_continuity` | 0.01 | Continuity loss weight |

Trained models are saved automatically to `models/<timestamp>/trained_gnn.pms`.

---

## Evaluation

### Compute rollout error on test set

```bash
python test/test_rollout.py
```

### Export predictions for visualization

```bash
python test/test_for_time.py
```

This saves the following files for a selected test graph:

```
true_p.npy        # Ground truth pressure  (N, T)
pred_p.npy        # Predicted pressure     (N, T)
inlet_mask.npy    # Inlet node mask        (N,)
outlet_mask.npy   # Outlet node mask       (N,)
node_coords.npy   # Node 3D coordinates   (N, 3)
```

### Visualize pressure distribution

```bash
python test/huatu.py
```

Produces a side-by-side comparison of CFD ground truth, model prediction, and absolute error with inlet (cyan ▲) and outlet (orange ▽) markers.

---

## Results

| Method | Test Rollout Error (Relative L2) |
|--------|----------------------------------|
| Baseline (15 epochs) | 0.231 |
| Physics-Constrained λ=0.01 (15 epochs) | 0.275 |

Physics constraints accelerate early convergence (epochs 0–3) but lead to a rebound at epoch 7 under a fixed λ schedule. An **adaptive λ annealing** strategy is recommended for longer training runs.

---
