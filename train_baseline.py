#!/usr/bin/env python3
"""
train_baseline.py — Stage 2 Step A: Single-Patient Overfit Test
Goal: Prove the pipeline works by overfitting on 20 temporal snapshots.
Success criteria: Training loss < 0.01
"""
from __future__ import annotations
import json, logging, time, argparse
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)


class VesselDataset:
    def __init__(self, data_path, device):
        self.device = device
        logger.info("Loading dataset from %s", data_path)
        self.dataset = torch.load(data_path, weights_only=False)
        logger.info("Loaded %d graphs", len(self.dataset))
        self._compute_stats()
        self._normalize()

    def _compute_stats(self):
        all_x = torch.cat([d.x for d in self.dataset], dim=0)
        all_edge = torch.cat([d.edge_attr for d in self.dataset], dim=0)
        all_y = torch.cat([d.y for d in self.dataset], dim=0)
        self.x_mean = all_x.mean(dim=0)
        self.x_std = all_x.std(dim=0).clamp(min=1e-8)
        self.edge_mean = all_edge.mean(dim=0)
        self.edge_std = all_edge.std(dim=0).clamp(min=1e-8)
        self.y_mean = all_y.mean(dim=0)
        self.y_std = all_y.std(dim=0).clamp(min=1e-8)
        logger.info("  x    mean=%s std=%s", self.x_mean.numpy().round(3), self.x_std.numpy().round(3))
        logger.info("  edge mean=%s std=%s", self.edge_mean.numpy().round(3), self.edge_std.numpy().round(3))
        logger.info("  y    mean=%s std=%s", self.y_mean.numpy().round(3), self.y_std.numpy().round(3))

    def _normalize(self):
        for data in self.dataset:
            data.x = ((data.x - self.x_mean) / self.x_std).to(self.device)
            data.edge_attr = ((data.edge_attr - self.edge_mean) / self.edge_std).to(self.device)
            data.edge_index = data.edge_index.to(self.device)
            data.y = ((data.y - self.y_mean) / self.y_std).to(self.device)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


class MLPBlock(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, out_dim))
        self.norm = nn.LayerNorm(out_dim)
        self.residual = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def forward(self, x):
        return self.norm(self.net(x) + self.residual(x))


class GraphNetBlock(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim):
        super().__init__()
        self.edge_mlp = MLPBlock(edge_dim + 2 * node_dim, hidden_dim, edge_dim)
        self.node_mlp = MLPBlock(node_dim + edge_dim, hidden_dim, node_dim)

    def forward(self, x, edge_attr, edge_index):
        src, dst = edge_index[0], edge_index[1]
        edge_input = torch.cat([edge_attr, x[src], x[dst]], dim=-1)
        edge_attr = self.edge_mlp(edge_input)
        agg = torch.zeros(x.size(0), edge_attr.size(1), device=x.device, dtype=x.dtype)
        agg.scatter_add_(0, dst.unsqueeze(-1).expand_as(edge_attr), edge_attr)
        node_input = torch.cat([x, agg], dim=-1)
        x = self.node_mlp(node_input)
        return x, edge_attr


class MiniMeshGraphNet(nn.Module):
    def __init__(self, node_in=5, edge_in=2, node_out=2, hidden_dim=64, n_layers=8):
        super().__init__()
        self.node_encoder = MLPBlock(node_in, hidden_dim, hidden_dim)
        self.edge_encoder = MLPBlock(edge_in, hidden_dim, hidden_dim)
        self.processor = nn.ModuleList([GraphNetBlock(hidden_dim, hidden_dim, hidden_dim) for _ in range(n_layers)])
        self.decoder = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, node_out))

    def forward(self, node_features, edge_features, edge_index):
        x = self.node_encoder(node_features)
        edge_attr = self.edge_encoder(edge_features)
        for block in self.processor:
            x, edge_attr = block(x, edge_attr, edge_index)
        return self.decoder(x)


def train(data_path=Path("./data/processed/0074_H_CORO_KD_all.pt"), output_dir=Path("./runs/stepA"),
          hidden_dim=64, n_layers=8, lr=1e-3, n_epochs=500, log_every=25, target="both"):

    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)
    if device.type == "cuda":
        logger.info("GPU: %s (%.1f GB)", torch.cuda.get_device_name(0),
                     torch.cuda.get_device_properties(0).total_memory / 1e9)

    ds = VesselDataset(data_path, device)
    target_cols = {"pressure": [0], "velocity": [1], "both": [0, 1]}[target]
    node_out = len(target_cols)
    node_in = ds.dataset[0].x.shape[1]
    edge_in = ds.dataset[0].edge_attr.shape[1]

    model = MiniMeshGraphNet(node_in=node_in, edge_in=edge_in, node_out=node_out,
                              hidden_dim=hidden_dim, n_layers=n_layers).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("Model: %d params (%.1fK), hidden=%d, layers=%d", n_params, n_params/1000, hidden_dim, n_layers)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    criterion = nn.MSELoss()

    history = {"epoch": [], "loss": [], "lr": []}
    best_loss = float("inf")
    t_start = time.time()

    logger.info("Training: %d epochs, %d graphs, target=%s", n_epochs, len(ds), target)

    for epoch in range(1, n_epochs + 1):
        model.train()
        epoch_loss = 0.0
        for data in ds.dataset:
            optimizer.zero_grad()
            pred = model(data.x, data.edge_attr, data.edge_index)
            loss = criterion(pred, data.y[:, target_cols])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()

        epoch_loss /= len(ds)
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        history["epoch"].append(epoch)
        history["loss"].append(epoch_loss)
        history["lr"].append(current_lr)

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                         "loss": best_loss, "x_mean": ds.x_mean, "x_std": ds.x_std,
                         "edge_mean": ds.edge_mean, "edge_std": ds.edge_std,
                         "y_mean": ds.y_mean, "y_std": ds.y_std}, output_dir / "best_model.pt")

        if epoch % log_every == 0 or epoch == 1:
            logger.info("  Epoch %4d/%d | Loss: %.6f | Best: %.6f | LR: %.2e | %.1fs",
                         epoch, n_epochs, epoch_loss, best_loss, current_lr, time.time() - t_start)

    total_time = time.time() - t_start
    logger.info("Training complete in %.1fs, best loss: %.6f", total_time, best_loss)

    with open(output_dir / "history.json", "w") as f:
        json.dump(history, f)

    # === Evaluation ===
    model.eval()
    all_pred, all_true = [], []
    with torch.no_grad():
        for data in ds.dataset:
            pred = model(data.x, data.edge_attr, data.edge_index)
            all_pred.append(pred.cpu())
            all_true.append(data.y[:, target_cols].cpu())

    all_pred = torch.cat(all_pred, dim=0)
    all_true = torch.cat(all_true, dim=0)
    pred_phys = all_pred * ds.y_std[target_cols] + ds.y_mean[target_cols]
    true_phys = all_true * ds.y_std[target_cols] + ds.y_mean[target_cols]

    col_names = [["Pressure", "Velocity"][c] for c in target_cols]
    for i, name in enumerate(col_names):
        p, t = pred_phys[:, i].numpy(), true_phys[:, i].numpy()
        rmse = np.sqrt(np.mean((p - t) ** 2))
        mask = np.abs(t) > 1e-6
        mape = np.mean(np.abs((p[mask] - t[mask]) / t[mask])) * 100 if mask.any() else 0.0
        r2 = 1.0 - np.sum((p - t) ** 2) / (np.sum((t - t.mean()) ** 2) + 1e-8)
        logger.info("  %s: RMSE=%.2f, MAPE=%.2f%%, R2=%.4f", name, rmse, mape, r2)

    # === Plots ===
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, len(target_cols) + 1, figsize=(6 * (len(target_cols) + 1), 5))
        if not isinstance(axes, np.ndarray):
            axes = np.array([axes])

        for i, name in enumerate(col_names):
            p, t = pred_phys[:, i].numpy(), true_phys[:, i].numpy()
            axes[i].scatter(t, p, alpha=0.3, s=5, c="steelblue")
            lims = [min(t.min(), p.min()), max(t.max(), p.max())]
            axes[i].plot(lims, lims, "r--", linewidth=1)
            axes[i].set_xlabel(f"True {name}")
            axes[i].set_ylabel(f"Predicted {name}")
            axes[i].set_title(f"{name}: Pred vs True")

        axes[-1].semilogy(history["epoch"], history["loss"], "b-")
        axes[-1].set_xlabel("Epoch")
        axes[-1].set_ylabel("Loss")
        axes[-1].set_title("Training Loss")
        axes[-1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / "stepA_results.png", dpi=150)
        logger.info("Plot saved -> %s", output_dir / "stepA_results.png")
        plt.close()
    except ImportError:
        logger.warning("matplotlib not available, skipping plots")

    # === Verdict ===
    if best_loss < 0.01:
        logger.info("🎉 OVERFIT TEST PASSED! (loss=%.6f < 0.01) -> Ready for Step B", best_loss)
    elif best_loss < 0.1:
        logger.info("⚠️  PARTIAL (loss=%.6f < 0.1) -> Try more epochs or higher LR", best_loss)
    else:
        logger.info("❌ FAILED (loss=%.6f) -> Check data/graph", best_loss)

    json.dump({"best_loss": best_loss, "n_params": n_params, "hidden_dim": hidden_dim,
               "n_layers": n_layers, "n_graphs": len(ds), "time_sec": round(total_time, 1)},
              open(output_dir / "summary.json", "w"), indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("./data/processed/0074_H_CORO_KD_all.pt"))
    parser.add_argument("--output", type=Path, default=Path("./runs/stepA"))
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--n_layers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--target", type=str, default="both", choices=["pressure", "velocity", "both"])
    args = parser.parse_args()
    train(data_path=args.data, output_dir=args.output, hidden_dim=args.hidden_dim,
          n_layers=args.n_layers, lr=args.lr, n_epochs=args.epochs, target=args.target)