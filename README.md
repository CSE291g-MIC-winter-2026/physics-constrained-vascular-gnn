# Physics-Constrained Graph Neural Network for Coronary Blood Flow Prediction

A GNN-based surrogate model for hemodynamic prediction in coronary artery networks, trained on SimVascular patient data with Hagen-Poiseuille and mass conservation physics constraints.

**Key results:**
- Velocity R² = 0.9993 on held-out time steps (single patient, time-split validation)
- Physiologically valid pressure maps (99.6% of nodes in 5.8–138 mmHg) — *with no pressure ground-truth labels*
- Inference: < 1 ms per patient graph vs. 2–8 hours for 3D CFD

---

## Overview

Traditional 3D CFD simulations of coronary blood flow require hours of compute per patient, making real-time clinical assessment impractical. This project trains a Graph Neural Network (GNN) that takes a patient's coronary artery geometry as input and predicts nodal velocity and pressure at each time step of the cardiac cycle.

The key innovation is the **physics constraint**: instead of requiring pressure ground-truth labels (which are unavailable non-invasively), we enforce the Hagen-Poiseuille law and mass conservation as soft losses during training. This allows the model to produce physically consistent pressure predictions from geometry and flow boundary conditions alone.

---

## Pipeline

```
SimVascular raw data          PyG graph dataset         Trained GNN model
(.pth, .ctgr, .flow, .dat)  ──────────────────────►  ──────────────────►  v(cm/s), P(mmHg)
                              convert_all.py            train_baseline.py
                                                        train_stage4.py
```

---

## Repository Structure

```
.
├── convert_all.py          # Stage 1: Convert SimVascular projects → PyG graphs
├── train_baseline.py       # Stage 2/3: GNN training (supports physics constraints)
├── train_stage4.py         # Stage 4: Time-split and cross-patient generalization
├── gen_1d_sim.py           # (Experimental) Generate svOneDSolver 1D input files
├── processed.zip           # Pre-converted PyG graph dataset (4 CORO_KD patients)
└── runs.zip                # Training outputs: best models, loss curves, summaries
```

**Included data** (`processed.zip`): Pre-processed graphs for patients `0068_H_CORO_KD`, `0069_H_CORO_KD`, `0070_H_CORO_KD`, `0074_H_CORO_KD` — 50 time snapshots each.

**Included runs** (`runs.zip`): Best checkpoints and training history for Stage 3 v4 (physics-constrained) and Stage 4 v13 (time-split, R²=0.9993).

---

## Graph Representation

Each patient's coronary artery network is represented as a directed graph:

| Component | Features | Description |
|-----------|----------|-------------|
| **Nodes** | `[x, y, z, r, is_junction, t_norm]` | Centerline points; 6D feature vector |
| **Edges** | `[avg_radius, length]` | Vessel segments; 2D feature vector |
| **Labels** | `[P (dyne/cm²), v (cm/s)]` | Pressure and velocity per node |

- `is_junction`: 1 if node has ≥ 2 outgoing edges (bifurcation), else 0
- `t_norm`: normalized cardiac cycle time ∈ [0, 1]

---

## Model: MiniMeshGraphNet

A 12-layer message-passing GNN with 2.4M parameters:

```
Input node/edge features
    → Node Encoder (MLP)
    → 12 × GraphNet Block
          h_i^(l+1) = MLP( h_i^l,  Σ_{j∈N(i)} MLP(h_j^l || e_ij) )
    → Node Decoder (MLP)
Output: [pressure, velocity] per node
```

---

## Physics Constraints

Three soft losses are added to the velocity MSE:

**1. Hagen-Poiseuille (HP)**

The pressure drop along each vessel segment must satisfy:

```
ΔP_ij = 8 μ L v_avg / r²
```

where μ = 0.04 dyne·s/cm² (blood viscosity), L = segment length, r = radius.

**2. Mass Conservation**

At each bifurcation node:

```
Σ Q_in = Σ Q_out,    Q = v · π r²
```

**3. Pressure Positivity**

```
L_pos = mean( relu(-P)² )
```

Lambda warm-up: physics weights ramp linearly from 0 (epoch 200) to target (epoch 800), allowing the data loss to stabilize first.

---

## Installation

```bash
pip install torch torch-geometric numpy scipy matplotlib
git clone https://github.com/Shengxian2003/physics-constrained-vascular-gnn.git
cd physics-constrained-vascular-gnn
unzip processed.zip
unzip runs.zip
```

**To use your own data**: Download CORO_KD patient projects from the [Vascular Model Repository](https://vascularmodel.org) and place under `data/raw/`.

---

## Usage

### Step 1 — Convert raw SimVascular data to graphs (skip if using processed.zip)

```bash
python convert_all.py \
  --batch \
  --raw_dir data/raw \
  --output data/processed \
  --filter CORO_KD \
  -n 50
```

### Step 2 — Train with physics constraints

```bash
python train_baseline.py \
  --data data/processed/0074_H_CORO_KD/0074_H_CORO_KD_all.pt \
  --output runs/stage3 \
  --epochs 2000 \
  --lambda_hp 0.05 \
  --lambda_cont 0.05 \
  --warmup_start 200 \
  --warmup_end 800
```

### Step 3 — Time-split generalization (Stage 4)

```bash
python train_stage4.py \
  --processed_dir data/processed \
  --train 0074_H_CORO_KD \
  --val_split 0.2 \
  --output runs/stage4 \
  --epochs 5000 \
  --patience 1000 \
  --lr 1e-4
```

---

## Results

### Stage 3 — Physics-Constrained Training

| Metric | Value |
|--------|-------|
| Velocity R² | 0.954 |
| HP residual R² | 0.998 |
| Physiological pressure (5.8–138 mmHg) | 99.6% |
| Training time | ~14 min (RTX 5090) |

> Pressure is predicted **without any pressure ground-truth labels** — driven entirely by Hagen-Poiseuille physics loss.

### Stage 4 — Time-Split Generalization

| Metric | Value |
|--------|-------|
| Velocity R² | **0.9993** |
| Best epoch | 4906 / 5000 |
| Validation | Last 20% of cardiac cycle snapshots |
| Training time | ~69 min |

---

## Limitations

- Cross-patient generalization remains an open challenge — different coronary geometries require more training patients or geometry-agnostic representations
- Pressure labels are synthetic (1D linear approximation); real 1D CFD labels (svOneDSolver) would improve physics constraint effectiveness
- `gen_1d_sim.py` is experimental — solver diverges for complex coronary topologies

---

## Data Source

Patient geometry from the [Vascular Model Repository](https://vascularmodel.org) (SimVascular project, CORO_KD collection).

---

## Course

BENG 280B / CSE 291 — University of California San Diego
