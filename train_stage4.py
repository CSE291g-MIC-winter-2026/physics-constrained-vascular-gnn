#!/usr/bin/env python3
"""
train_stage4.py — Stage 4 v8: Multi-Patient GPU Training with Dimensionless Physics
============================================================================
Key changes from train_physics.py:
  1. Multi-patient dataset: load from multiple _all.pt files
  2. Global normalization stats computed from TRAINING patients only
  3. CosineAnnealingLR (replaces ReduceLROnPlateau)
  4. Per-epoch validation on held-out patient(s)
  5. Best model saved by val_loss (cross-patient generalization)

Default split:
  Train: 0074_H_CORO_KD  (20 snapshots, N=233)
  Val:   0070_H_CORO_KD  (20 snapshots, N=150)

Usage:
  python3 train_stage4.py \
    --processed_dir /projects/data/processed \
    --train 0074_H_CORO_KD \
    --val   0070_H_CORO_KD \
    --output /projects/runs/stage4 \
    --epochs 3000
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ======================================================================
# Physics constants (same as train_physics.py)
# ======================================================================
MU  = 0.035
PI  = math.pi
COL_P, COL_V         = 0, 1
COL_RADIUS_NODE      = 3
COL_IS_JUNCTION      = 4
COL_EDGE_RADIUS      = 0
COL_EDGE_LEN         = 1
LAMBDA_HP_TARGET     = 0.05
LAMBDA_CONT_TARGET   = 0.05
LAMBDA_POS_FIXED     = 0.1


# ======================================================================
# Multi-patient dataset
# ======================================================================
class MultiPatientDataset:
    """
    Loads graphs from multiple _all.pt files.

    Two normalization modes:
      per_patient_norm=False (default, legacy):
        Stats computed from all training graphs combined; applied uniformly.
        Fails when train/val patients have very different velocity scales.

      per_patient_norm=True (recommended for cross-patient):
        Each patient is normalized with its own stats (mean/std).
        Stats are stored on each graph as g.y_mean / g.y_std etc.
        Physics losses and evaluate() automatically use per-graph stats.

    val_split > 0 (e.g. 0.2):
        Instead of a separate val patient, split each train patient's graphs
        by time index: first (1-val_split) fraction → train, remainder → val.
        val_names is ignored when val_split > 0.
    """

    def __init__(self, train_paths: list[Path], val_paths: list[Path], device,
                 per_patient_norm: bool = False, val_split: float = 0.0):
        self.device = device
        self.per_patient_norm = per_patient_norm

        if per_patient_norm:
            # Per-patient normalization: load + normalize in one pass so self.train_graphs
            # / self.val_graphs point directly to the normalized graphs on device.
            logger.info("Normalization: per-patient (stats stored on each graph)")
            self.train_graphs, self.val_graphs = [], []
            if val_split > 0.0:
                logger.info("Loading patients (time-split val_split=%.2f):", val_split)
                self._normalize_per_patient(
                    train_paths, val_split, is_train=True,  out_list=self.train_graphs)
                self._normalize_per_patient(
                    train_paths, val_split, is_train=False, out_list=self.val_graphs)
            else:
                logger.info("Loading TRAIN patients:")
                self._normalize_per_patient(
                    train_paths, 0.0, is_train=True,  out_list=self.train_graphs)
                logger.info("Loading VAL patients:")
                self._normalize_per_patient(
                    val_paths,   0.0, is_train=True,  out_list=self.val_graphs)
            self._compute_global_stats(self.train_graphs)
        elif val_split > 0.0:
            # Time-split, legacy normalization
            logger.info("Loading patients (time-split val_split=%.2f):", val_split)
            self.train_graphs, self.val_graphs = [], []
            for p in train_paths:
                if not p.exists():
                    logger.error("  ❌ Not found: %s", p)
                    continue
                patient = torch.load(p, weights_only=False)
                n = len(patient)
                split = max(1, int(n * (1.0 - val_split)))
                logger.info("  %-30s  %d graphs → %d train / %d val  N=%d  E=%d",
                            p.parent.name, n, split, n - split,
                            patient[0].num_nodes, patient[0].edge_index.shape[1])
                self.train_graphs.extend(patient[:split])
                self.val_graphs.extend(patient[split:])
            self._compute_stats(self.train_graphs)
            self._normalize_and_cache(self.train_graphs)
            self._normalize_and_cache(self.val_graphs)
        else:
            # Legacy: separate train/val patients, global normalization
            logger.info("Loading TRAIN patients:")
            self.train_graphs = self._load_paths(train_paths)
            logger.info("Loading VAL patients:")
            self.val_graphs   = self._load_paths(val_paths)
            self._compute_stats(self.train_graphs)
            self._normalize_and_cache(self.train_graphs)
            self._normalize_and_cache(self.val_graphs)

        logger.info("Train graphs: %d  |  Val graphs: %d",
                    len(self.train_graphs), len(self.val_graphs))

    def _load_paths(self, paths: list[Path]) -> list:
        graphs = []
        for p in paths:
            if not p.exists():
                logger.error("  ❌ Not found: %s", p)
                continue
            patient = torch.load(p, weights_only=False)
            logger.info("  %-30s  %d graphs  N=%d  E=%d",
                        p.parent.name, len(patient),
                        patient[0].num_nodes,
                        patient[0].edge_index.shape[1])
            graphs.extend(patient)
        return graphs

    # ------------------------------------------------------------------
    # Per-patient normalization helpers
    # ------------------------------------------------------------------
    def _normalize_per_patient(self, paths: list[Path], val_split: float,
                               is_train: bool, out_list: list):
        """Load each patient, compute its own stats, normalize, append to out_list."""
        for p in paths:
            if not p.exists():
                logger.error("  ❌ Not found: %s", p)
                continue
            all_graphs = torch.load(p, weights_only=False)
            n = len(all_graphs)
            if val_split > 0.0 and is_train:
                split = max(1, int(n * (1.0 - val_split)))
                graphs = all_graphs[:split]
                logger.info("  %-30s  %d graphs → %d train  N=%d  E=%d",
                            p.parent.name, n, split,
                            all_graphs[0].num_nodes, all_graphs[0].edge_index.shape[1])
            elif val_split > 0.0 and not is_train:
                split = max(1, int(n * (1.0 - val_split)))
                graphs = all_graphs[split:]
                logger.info("  %-30s  %d graphs → %d val  N=%d  E=%d",
                            p.parent.name, n, n - split,
                            all_graphs[0].num_nodes, all_graphs[0].edge_index.shape[1])
            else:
                graphs = all_graphs
                logger.info("  %-30s  %d graphs  N=%d  E=%d",
                            p.parent.name, n,
                            all_graphs[0].num_nodes, all_graphs[0].edge_index.shape[1])

            # Compute stats from this patient's graphs only.
            all_x    = torch.cat([g.x        for g in graphs], dim=0)
            all_y    = torch.cat([g.y        for g in graphs], dim=0)
            x_mean = all_x.mean(0);  x_std = all_x.std(0).clamp(1e-8)
            y_mean = all_y.mean(0);  y_std = all_y.std(0).clamp(1e-8)
            all_edge = torch.cat([g.edge_attr for g in graphs], dim=0)
            em = all_edge.mean(0); es = all_edge.std(0).clamp(1e-8)
            logger.info("  %-30s  y_mean=%s  y_std=%s",
                        p.parent.name,
                        y_mean.numpy().round(1), y_std.numpy().round(1))

            for g in graphs:
                g.edge_attr_phys = g.edge_attr.clone()
                g.x          = ((g.x        - x_mean) / x_std).to(self.device)
                g.edge_attr  = ((g.edge_attr - em)     / es   ).to(self.device)
                g.edge_index = g.edge_index.to(self.device)
                g.edge_attr_phys = g.edge_attr_phys.to(self.device)
                g.y          = ((g.y        - y_mean)  / y_std).to(self.device)
                # Store patient-local stats as Python lists (bypasses PyG tensor storage
                # which does not reliably preserve device for non-node/edge tensors)
                g._norm_y_mean = y_mean.tolist()
                g._norm_y_std  = y_std.tolist()
                g._norm_x_mean = x_mean.tolist()
                g._norm_x_std  = x_std.tolist()
            out_list.extend(graphs)

    def _compute_global_stats(self, graphs: list):
        """Set self.x_mean etc. to CPU zero/one placeholders.
        In per_patient_norm mode these are not used for normalization — each graph
        carries its own _norm_* lists. These placeholders exist so train() can call
        ds.x_mean.to(device) without AttributeError."""
        feat_dim = graphs[0].x.shape[1]
        edge_dim = graphs[0].edge_attr.shape[1]
        y_dim    = graphs[0].y.shape[1]
        self.x_mean    = torch.zeros(feat_dim); self.x_std    = torch.ones(feat_dim)
        self.edge_mean = torch.zeros(edge_dim); self.edge_std = torch.ones(edge_dim)
        self.y_mean    = torch.zeros(y_dim);    self.y_std    = torch.ones(y_dim)

    def _compute_stats(self, graphs: list):
        all_x    = torch.cat([g.x        for g in graphs], dim=0)
        all_y    = torch.cat([g.y        for g in graphs], dim=0)
        all_edge = torch.cat([g.edge_attr for g in graphs], dim=0)

        self.x_mean    = all_x.mean(0);    self.x_std    = all_x.std(0).clamp(1e-8)
        self.edge_mean = all_edge.mean(0); self.edge_std = all_edge.std(0).clamp(1e-8)
        self.y_mean    = all_y.mean(0);    self.y_std    = all_y.std(0).clamp(1e-8)

        logger.info("Normalization stats (from train):")
        logger.info("  x    mean=%s  std=%s",
                    self.x_mean.numpy().round(3), self.x_std.numpy().round(3))
        logger.info("  edge mean=%s  std=%s",
                    self.edge_mean.numpy().round(3), self.edge_std.numpy().round(3))
        logger.info("  y    mean=%s  std=%s",
                    self.y_mean.numpy().round(3), self.y_std.numpy().round(3))

    def _normalize_and_cache(self, graphs: list):
        for g in graphs:
            g.edge_attr_phys = g.edge_attr.clone()
            g.x          = ((g.x        - self.x_mean)    / self.x_std).to(self.device)
            g.edge_attr  = ((g.edge_attr - self.edge_mean) / self.edge_std).to(self.device)
            g.edge_index = g.edge_index.to(self.device)
            g.edge_attr_phys = g.edge_attr_phys.to(self.device)
            g.y          = ((g.y        - self.y_mean)    / self.y_std).to(self.device)
            # Legacy mode: no per-graph stats (use dataset-level fallback in _resolve_*)


# ======================================================================
# Model (identical to train_physics.py)
# ======================================================================
class MLPBlock(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, out_dim)
        )
        self.norm     = nn.LayerNorm(out_dim)
        self.residual = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def forward(self, x):
        return self.norm(self.net(x) + self.residual(x))


class GraphNetBlock(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim):
        super().__init__()
        self.edge_mlp = MLPBlock(edge_dim + 2 * node_dim, hidden_dim, edge_dim)
        self.node_mlp = MLPBlock(node_dim + edge_dim,     hidden_dim, node_dim)

    def forward(self, x, edge_attr, edge_index):
        src, dst = edge_index[0], edge_index[1]
        edge_attr = self.edge_mlp(torch.cat([edge_attr, x[src], x[dst]], dim=-1))
        agg = torch.zeros(x.size(0), edge_attr.size(1), device=x.device, dtype=x.dtype)
        agg.scatter_add_(0, dst.unsqueeze(-1).expand_as(edge_attr), edge_attr)
        x = self.node_mlp(torch.cat([x, agg], dim=-1))
        return x, edge_attr


class MiniMeshGraphNet(nn.Module):
    def __init__(self, node_in=6, edge_in=2, node_out=2, hidden_dim=128, n_layers=12):
        super().__init__()
        self.node_encoder = MLPBlock(node_in, hidden_dim, hidden_dim)
        self.edge_encoder = MLPBlock(edge_in, hidden_dim, hidden_dim)
        self.processor    = nn.ModuleList([
            GraphNetBlock(hidden_dim, hidden_dim, hidden_dim) for _ in range(n_layers)
        ])
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, node_out)
        )

    def forward(self, node_features, edge_features, edge_index):
        x, ea = self.node_encoder(node_features), self.edge_encoder(edge_features)
        for block in self.processor:
            x, ea = block(x, ea, edge_index)
        return self.decoder(x)


# ======================================================================
# Physics losses (identical to train_physics.py)
# ======================================================================
def get_lambda(epoch, warmup_start=200, warmup_end=800, target=1.0):
    if epoch < warmup_start:  return 0.0
    if epoch >= warmup_end:   return target
    return target * (epoch - warmup_start) / (warmup_end - warmup_start)


def _resolve_ym_ys(data, y_mean_fallback, y_std_fallback):
    """Return per-graph stats if stored, else fall back to dataset-level stats.
    Stats stored as Python lists (_norm_y_mean) to bypass PyG's tensor device handling."""
    if hasattr(data, "_norm_y_mean"):
        dev = data.y.device
        return (torch.tensor(data._norm_y_mean, dtype=torch.float32, device=dev),
                torch.tensor(data._norm_y_std,  dtype=torch.float32, device=dev))
    return y_mean_fallback, y_std_fallback


def _resolve_xm_xs(data, x_mean_fallback, x_std_fallback):
    if hasattr(data, "_norm_x_mean"):
        dev = data.x.device
        return (torch.tensor(data._norm_x_mean, dtype=torch.float32, device=dev),
                torch.tensor(data._norm_x_std,  dtype=torch.float32, device=dev))
    return x_mean_fallback, x_std_fallback


def _denorm_pv(pred_norm, y_mean, y_std):
    return (pred_norm[:, COL_P] * y_std[COL_P] + y_mean[COL_P],
            pred_norm[:, COL_V] * y_std[COL_V] + y_mean[COL_V])


def hagen_poiseuille_loss(pred_norm, data, y_mean, y_std):
    """Dimensionless HP residual: per-edge relative error.
    (dP_pred - dP_HP) / scale  — O(1) regardless of patient scale.

    Numerical stability fix (v7→v8):
      Use per-edge scale floored at 1% of the global mean |dP_HP|.
      This prevents near-zero denominators when velocity ≈ 0 at init.
      Floor of 1.0 dyne/cm² ensures minimum meaningful pressure drop scale.
    """
    ym, ys = _resolve_ym_ys(data, y_mean, y_std)
    P, v = _denorm_pv(pred_norm, ym, ys)
    src, dst = data.edge_index
    mask = src < dst
    s, d = src[mask], dst[mask]
    r = data.edge_attr_phys[mask, COL_EDGE_RADIUS]
    L = data.edge_attr_phys[mask, COL_EDGE_LEN]
    dP_pred = P[s] - P[d]
    v_avg   = 0.5 * (v[s] + v[d])
    # Clamp radius to ≥0.01 cm (vessel diameter ≥0.2 mm) — avoids r²→0
    dP_HP   = (8.0 * MU * L / r.pow(2).clamp(min=1e-4)) * v_avg
    # Robust scale: per-edge |dP_HP|, floored at 1% of global mean
    # Prevents explosion when |dP_HP| ≈ 0 at initialization (v≈0)
    global_scale  = dP_HP.abs().mean().clamp(min=1.0)   # ≥1 dyne/cm²
    per_edge_scale = dP_HP.abs().clamp(min=global_scale * 0.01)
    rel_err = (dP_pred - dP_HP) / per_edge_scale
    return rel_err.pow(2).mean()


def mass_conservation_loss(pred_norm, data, y_mean, y_std, r_node_phys):
    """Dimensionless mass conservation: per-junction relative flow imbalance.
    residual / |Q_self|  — O(1) regardless of patient flow magnitudes."""
    ym, ys = _resolve_ym_ys(data, y_mean, y_std)
    _, v = _denorm_pv(pred_norm, ym, ys)
    Q    = v * PI * r_node_phys.pow(2)
    N    = data.x.shape[0]
    src, dst = data.edge_index
    degree = torch.zeros(N, device=data.x.device)
    degree.scatter_add_(0, src, torch.ones(src.shape[0], device=data.x.device))
    sum_Q  = torch.zeros(N, device=data.x.device)
    sum_Q.scatter_add_(0, dst, Q[src])
    residual = sum_Q - degree * Q
    mask = data.x[:, COL_IS_JUNCTION].bool() | (degree >= 3)
    if mask.sum() == 0:
        return torch.tensor(0.0, device=data.x.device)
    # Robust scale: per-junction |Q|, floored at 1% of global mean
    # Prevents explosion when Q ≈ 0 at initialization
    Q_junc = Q[mask]
    global_Q_scale   = Q_junc.abs().mean().clamp(min=1e-4)
    per_node_scale   = Q_junc.abs().clamp(min=global_Q_scale * 0.01)
    rel_err = residual[mask] / per_node_scale
    return rel_err.pow(2).mean()


def pressure_positivity_loss(pred_norm, data, y_mean, y_std):
    ym, ys = _resolve_ym_ys(data, y_mean, y_std)
    P, _ = _denorm_pv(pred_norm, ym, ys)
    return F.relu(-P).pow(2).mean()


def compute_total_loss(pred_norm, data, y_mean, y_std, r_node_phys,
                       lam_hp, lam_cont, lam_pos=LAMBDA_POS_FIXED):
    L_data = F.mse_loss(pred_norm[:, COL_V], data.y[:, COL_V])
    L_hp   = hagen_poiseuille_loss(pred_norm, data, y_mean, y_std)
    L_cont = mass_conservation_loss(pred_norm, data, y_mean, y_std, r_node_phys)
    L_pos  = pressure_positivity_loss(pred_norm, data, y_mean, y_std)
    L_tot  = L_data + lam_hp * L_hp + lam_cont * L_cont + lam_pos * L_pos
    return L_tot, L_data, L_hp, L_cont, L_pos


# ======================================================================
# Evaluation helper
# ======================================================================
@torch.no_grad()
def evaluate(model, graphs, y_mean, y_std, x_mean, x_std, device):
    """Returns (val_total_loss, val_data_loss, v_r2, P_phys_frac).
    val_data_loss = pure velocity MSE (patient-agnostic, reliable for model selection).
    val_total_loss includes physics terms (can diverge due to patient-specific physics).
    Per-graph stats (g._norm_*) take precedence over dataset-level fallbacks."""
    model.eval()
    y_mean_d = y_mean.to(device); y_std_d = y_std.to(device)
    x_mean_d = x_mean.to(device); x_std_d = x_std.to(device)

    losses_tot, losses_dat, vp_all, vt_all, Pp_all = [], [], [], [], []
    for g in graphs:
        pred = model(g.x, g.edge_attr, g.edge_index)
        xm, xs = _resolve_xm_xs(g, x_mean_d, x_std_d)
        ym, ys = _resolve_ym_ys(g, y_mean_d, y_std_d)
        r_phys = g.x[:, COL_RADIUS_NODE] * xs[COL_RADIUS_NODE] + xm[COL_RADIUS_NODE]
        lt, ld, lh, lc, lpos = compute_total_loss(
            pred, g, y_mean_d, y_std_d, r_phys, LAMBDA_HP_TARGET, LAMBDA_CONT_TARGET
        )
        losses_tot.append(lt.item())
        losses_dat.append(ld.item())
        vp_all.append((pred[:, COL_V] * ys[COL_V] + ym[COL_V]).cpu())
        vt_all.append((g.y[:, COL_V]  * ys[COL_V] + ym[COL_V]).cpu())
        Pp_all.append((pred[:, COL_P] * ys[COL_P] + ym[COL_P]).cpu())

    vp = torch.cat(vp_all).numpy()
    vt = torch.cat(vt_all).numpy()
    Pp = torch.cat(Pp_all).numpy() / 1333.22  # → mmHg

    r2 = float(1.0 - np.sum((vp - vt)**2) / (np.sum((vt - vt.mean())**2) + 1e-8))
    frac_phys = float(((Pp >= 20) & (Pp <= 200)).mean())
    return float(np.mean(losses_tot)), float(np.mean(losses_dat)), r2, frac_phys


# ======================================================================
# Training
# ======================================================================
def train(
    processed_dir:     Path,
    train_names:       list[str],
    val_names:         list[str],
    output_dir:        Path,
    hidden_dim:        int   = 128,
    n_layers:          int   = 12,
    lr:                float = 1e-3,
    n_epochs:          int   = 3000,
    log_every:         int   = 50,
    warmup_start:      int   = 200,
    warmup_end:        int   = 800,
    lambda_hp:         float = LAMBDA_HP_TARGET,
    lambda_cont:       float = LAMBDA_CONT_TARGET,
    cosine_t_mult:     int   = 1,   # 1 = single cycle, 2 = warm restarts
    per_patient_norm:  bool  = False,
    val_split:         float = 0.0,
    weight_decay:      float = 0.0,
    patience:          int   = 500,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)
    if device.type == "cuda":
        logger.info("GPU: %s (%.1f GB)", torch.cuda.get_device_name(0),
                    torch.cuda.get_device_properties(0).total_memory / 1e9)

    # --- Resolve paths ---
    def resolve(names):
        return [processed_dir / n / f"{n}_all.pt" for n in names]

    # --- Dataset ---
    ds = MultiPatientDataset(resolve(train_names), resolve(val_names), device,
                             per_patient_norm=per_patient_norm, val_split=val_split)

    node_in  = ds.train_graphs[0].x.shape[1]   # 6
    edge_in  = ds.train_graphs[0].edge_attr.shape[1]  # 2
    node_out = 2

    y_mean_d = ds.y_mean.to(device); y_std_d = ds.y_std.to(device)
    x_mean_d = ds.x_mean.to(device); x_std_d = ds.x_std.to(device)

    # --- Model ---
    model = MiniMeshGraphNet(node_in, edge_in, node_out, hidden_dim, n_layers).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("Model: %dK params | hidden=%d layers=%d", n_params//1000, hidden_dim, n_layers)

    # --- Cosine Annealing ---
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if cosine_t_mult == 1:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=n_epochs, eta_min=1e-6
        )
        sched_label = f"CosineAnnealing(T={n_epochs})"
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=n_epochs // 4, T_mult=cosine_t_mult, eta_min=1e-6
        )
        sched_label = f"CosineWarmRestarts(T0={n_epochs//4})"
    logger.info("Scheduler: %s", sched_label)

    history = {k: [] for k in
               ["epoch", "train_loss", "val_loss", "val_data_loss", "L_data", "L_hp",
                "L_cont", "val_v_r2", "val_P_phys", "lr"]}

    best_val_loss  = float("inf")
    best_val_r2    = float("-inf")
    best_r2_epoch  = 0
    t_start = time.time()

    logger.info("Train: %d graphs | Val: %d graphs | %d epochs",
                len(ds.train_graphs), len(ds.val_graphs), n_epochs)
    logger.info("Physics warm-up: %d → %d | λ_HP=%.3f λ_cont=%.3f",
                warmup_start, warmup_end, lambda_hp, lambda_cont)

    for epoch in range(1, n_epochs + 1):
        model.train()
        lam_hp   = get_lambda(epoch, warmup_start, warmup_end, lambda_hp)
        lam_cont = get_lambda(epoch, warmup_start, warmup_end, lambda_cont)

        ep_tot = ep_data = ep_hp = ep_cont = ep_pos = 0.0
        n_train = len(ds.train_graphs)
        # Batch gradient descent: accumulate gradients over ALL graphs, then single step.
        # Per-graph steps cause 50 conflicting updates/epoch → model collapses to mean predictor.
        optimizer.zero_grad()
        for g in ds.train_graphs:
            xm, xs = _resolve_xm_xs(g, x_mean_d, x_std_d)
            r_phys = g.x[:, COL_RADIUS_NODE] * xs[COL_RADIUS_NODE] + xm[COL_RADIUS_NODE]
            pred = model(g.x, g.edge_attr, g.edge_index)
            L_tot, L_data, L_hp, L_cont, L_pos = compute_total_loss(
                pred, g, y_mean_d, y_std_d, r_phys, lam_hp, lam_cont
            )
            (L_tot / n_train).backward()   # divide before accumulate → correct average grad
            ep_tot += L_tot.item(); ep_data += L_data.item()
            ep_hp  += L_hp.item();  ep_cont += L_cont.item(); ep_pos += L_pos.item()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        ep_tot /= n_train; ep_data /= n_train; ep_hp /= n_train
        ep_cont /= n_train; ep_pos /= n_train

        # Always step cosine scheduler (no freeze needed — it starts at lr and decays smoothly)
        scheduler.step()
        cur_lr = optimizer.param_groups[0]["lr"]

        # Validation
        val_loss, val_data_loss, val_r2, val_phys = evaluate(
            model, ds.val_graphs, ds.y_mean, ds.y_std, ds.x_mean, ds.x_std, device
        )
        model.train()

        # Track best by val_loss (total, for reference)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
        # Save best by val_R² (primary metric for cross-patient generalization).
        # Do NOT use val_loss for selection: physics terms are patient-specific and
        # grow monotonically for unseen patients as model converges on training patient.
        if val_r2 > best_val_r2:
            best_val_r2   = val_r2
            best_r2_epoch = epoch
            torch.save({
                "epoch": epoch, "model_state_dict": model.state_dict(),
                "val_r2": best_val_r2, "val_data_loss": val_data_loss,
                "x_mean": ds.x_mean, "x_std": ds.x_std,
                "edge_mean": ds.edge_mean, "edge_std": ds.edge_std,
                "y_mean": ds.y_mean, "y_std": ds.y_std,
                "train_patients": train_names, "val_patients": val_names,
                "hidden_dim": hidden_dim, "n_layers": n_layers,
            }, output_dir / "best_model.pt")

        # History
        for k, v in zip(
            ["epoch", "train_loss", "val_loss", "val_data_loss", "L_data", "L_hp",
             "L_cont", "val_v_r2", "val_P_phys", "lr"],
            [epoch, ep_tot, val_loss, val_data_loss, ep_data, ep_hp,
             ep_cont, val_r2, val_phys, cur_lr]
        ):
            history[k].append(v)

        if epoch % log_every == 0 or epoch == 1 or epoch <= 5:
            logger.info(
                "  Ep %4d | trn=%.4f dat=%.4f hp=%.4f c=%.5f"
                " | vDat=%.4f vR²=%.4f P%%=%.0f"
                " | LR=%.1e λ=%.3f | %.0fs",
                epoch, ep_tot, ep_data, ep_hp, ep_cont,
                val_data_loss, val_r2, val_phys * 100,
                cur_lr, lam_hp, time.time() - t_start,
            )

        # Early stopping: halt if val R² hasn't improved for `patience` epochs
        if patience > 0 and (epoch - best_r2_epoch) >= patience:
            logger.info(
                "Early stop at epoch %d — val R² hasn't improved for %d epochs "
                "(best=%.4f at epoch %d)",
                epoch, patience, best_val_r2, best_r2_epoch,
            )
            break

    total_time = time.time() - t_start
    logger.info("Done in %.0fs | best_val_R²=%.4f  best_val_loss=%.5f",
                total_time, best_val_r2, best_val_loss)
    with open(output_dir / "history.json", "w") as f:
        json.dump(history, f)

    # ================================================================
    # Final evaluation on val set using best model
    # ================================================================
    ckpt = torch.load(output_dir / "best_model.pt", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    val_loss_f, val_data_loss_f, val_r2_f, val_phys_f = evaluate(
        model, ds.val_graphs, ds.y_mean, ds.y_std, ds.x_mean, ds.x_std, device
    )

    logger.info("")
    logger.info("=" * 60)
    logger.info("FINAL (best model, epoch %d)", ckpt["epoch"])
    logger.info("  Val  v R²         = %.4f", val_r2_f)
    logger.info("  Val  P_phys       = %.1f%%", val_phys_f * 100)
    logger.info("  Val  data loss    = %.5f  (velocity MSE, cross-patient comparable)",
                val_data_loss_f)
    logger.info("  Val  total loss   = %.5f  (includes physics, patient-specific)", val_loss_f)

    if val_r2_f >= 0.85:
        logger.info("✅ STAGE 4 PASSED — cross-patient generalization demonstrated!")
    elif val_r2_f >= 0.70:
        logger.info("⚠️  PARTIAL — val R²=%.3f. Consider more training patients.", val_r2_f)
    else:
        logger.info("❌ val R²=%.3f — model not generalizing yet.", val_r2_f)
    logger.info("=" * 60)

    # ================================================================
    # Plots
    # ================================================================
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # 1. Train vs val loss
        ep = history["epoch"]
        axes[0].semilogy(ep, history["train_loss"],    "b-",  lw=0.8, label="train (total)")
        axes[0].semilogy(ep, history["val_data_loss"], "r-",  lw=0.8, label="val (data only)")
        axes[0].semilogy(ep, history["val_loss"],      "r--", lw=0.5, alpha=0.4,
                         label="val (total+phys)")
        axes[0].axvline(warmup_start, color="gray", ls=":", alpha=0.5)
        axes[0].axvline(warmup_end,   color="gray", ls=":", alpha=0.5)
        axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
        axes[0].set_title("Train vs Val Loss"); axes[0].legend(); axes[0].grid(alpha=0.3)

        # 2. Val velocity R²
        axes[1].plot(ep, history["val_v_r2"], "g-", lw=0.8)
        axes[1].axhline(0.85, color="orange", ls="--", alpha=0.6, label="target 0.85")
        axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("R²")
        axes[1].set_title("Val Velocity R²"); axes[1].legend(); axes[1].grid(alpha=0.3)
        axes[1].set_ylim(-0.5, 1.05)

        # 3. LR schedule
        axes[2].semilogy(ep, history["lr"], "m-", lw=0.8)
        axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("LR")
        axes[2].set_title(f"LR — {sched_label}"); axes[2].grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / "stage4_results.png", dpi=150)
        logger.info("Plot → %s/stage4_results.png", output_dir)
        plt.close()
    except ImportError:
        pass

    json.dump({
        "val_r2": val_r2_f, "val_loss": val_loss_f,
        "val_P_phys": val_phys_f, "best_epoch": int(ckpt["epoch"]),
        "train_patients": train_names, "val_patients": val_names,
        "n_params": n_params, "total_time_sec": round(total_time),
    }, open(output_dir / "summary.json", "w"), indent=2)


# ======================================================================
# CLI
# ======================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Stage 4: Multi-patient physics-constrained GNN",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default: 0074 train, 0070 val
  python3 train_stage4.py \\
    --processed_dir /projects/data/processed \\
    --output /projects/runs/stage4 --epochs 3000

  # Custom split
  python3 train_stage4.py \\
    --processed_dir /projects/data/processed \\
    --train 0074_H_CORO_KD 0136_H_CORO_KD \\
    --val   0070_H_CORO_KD \\
    --output /projects/runs/stage4b --epochs 3000

  # Warm restarts (CosineAnnealingWarmRestarts, T_mult=2)
  python3 train_stage4.py ... --cosine_t_mult 2
        """,
    )
    parser.add_argument("--processed_dir", type=Path,
                        default=Path("./data/processed"))
    parser.add_argument("--train", nargs="+",
                        default=["0074_H_CORO_KD"],
                        metavar="PATIENT")
    parser.add_argument("--val",   nargs="+",
                        default=["0070_H_CORO_KD"],
                        metavar="PATIENT")
    parser.add_argument("--output",      type=Path,  default=Path("./runs/stage4"))
    parser.add_argument("--hidden_dim",  type=int,   default=128)
    parser.add_argument("--n_layers",    type=int,   default=12)
    parser.add_argument("--lr",          type=float, default=1e-3)
    parser.add_argument("--epochs",      type=int,   default=3000)
    parser.add_argument("--log_every",   type=int,   default=50)
    parser.add_argument("--warmup_start",type=int,   default=200)
    parser.add_argument("--warmup_end",  type=int,   default=800)
    parser.add_argument("--lambda_hp",   type=float, default=LAMBDA_HP_TARGET)
    parser.add_argument("--lambda_cont", type=float, default=LAMBDA_CONT_TARGET)
    parser.add_argument("--cosine_t_mult", type=int, default=1,
                        help="1=single cycle, 2=warm restarts")
    parser.add_argument("--per_patient_norm", action="store_true",
                        help="Normalize each patient with its own stats (recommended "
                             "for cross-patient generalization)")
    parser.add_argument("--val_split", type=float, default=0.0,
                        help="Time-split fraction for val (e.g. 0.2 = last 20%% of "
                             "snapshots from each train patient). Ignores --val.")
    parser.add_argument("--weight_decay", type=float, default=0.0,
                        help="Adam weight decay. Default 0.")
    parser.add_argument("--patience", type=int, default=500,
                        help="Early stopping: stop if val R² hasn't improved for this "
                             "many epochs. Set 0 to disable. Default 500.")
    args = parser.parse_args()

    train(
        processed_dir    = args.processed_dir,
        train_names      = args.train,
        val_names        = args.val,
        output_dir       = args.output,
        hidden_dim       = args.hidden_dim,
        n_layers         = args.n_layers,
        lr               = args.lr,
        n_epochs         = args.epochs,
        log_every        = args.log_every,
        warmup_start     = args.warmup_start,
        warmup_end       = args.warmup_end,
        lambda_hp        = args.lambda_hp,
        lambda_cont      = args.lambda_cont,
        cosine_t_mult    = args.cosine_t_mult,
        per_patient_norm = args.per_patient_norm,
        val_split        = args.val_split,
        weight_decay     = args.weight_decay,
        patience         = args.patience,
    )


if __name__ == "__main__":
    main()
