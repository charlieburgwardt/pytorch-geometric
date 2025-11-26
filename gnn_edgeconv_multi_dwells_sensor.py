
# gnn_edgeconv_multi_dwells_sensor_fixed.py
import os
import argparse
import math
import random
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import EdgeConv
from torch_geometric.nn import knn_graph

# -------------------- Utilities --------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_pos_weight(graphs: List[Data]) -> float:
    """Compute positive class weight for BCEWithLogitsLoss from training graphs."""
    pos = 0
    neg = 0
    for g in graphs:
        y = g.y
        # Only count valid (non-sensor) nodes
        mask = g.valid_mask if hasattr(g, 'valid_mask') else torch.ones_like(y, dtype=torch.bool)
        y_valid = y[mask]
        pos += int((y_valid == 1).sum().item())
        neg += int((y_valid == 0).sum().item())
    if pos == 0:
        print("Warning: No positive samples in training set; setting pos_weight=1.0")
        return 1.0
    return neg / max(1, pos)


def stratified_graph_split(graphs: List[Data],
                           train_ratio=0.7, val_ratio=0.15, test_ratio=0.15,
                           seed: int = 42) -> Tuple[List[Data], List[Data], List[Data]]:
    """Stratified split by dwell-wise majority class (on valid nodes)."""
    assert math.isclose(train_ratio + val_ratio + test_ratio, 1.0, abs_tol=1e-6)
    set_seed(seed)

    pos_majority, neg_majority = [], []
    for g in graphs:
        y = g.y
        mask = g.valid_mask if hasattr(g, 'valid_mask') else torch.ones_like(y, dtype=torch.bool)
        y_valid = y[mask]
        pos_count = int((y_valid == 1).sum().item())
        neg_count = int((y_valid == 0).sum().item())
        if pos_count >= neg_count:
            pos_majority.append(g)
        else:
            neg_majority.append(g)

    random.shuffle(pos_majority)
    random.shuffle(neg_majority)

    def split_list(lst):
        n = len(lst)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        n_test = n - n_train - n_val
        return lst[:n_train], lst[n_train:n_train+n_val], lst[n_train+n_val:]

    tr_p, va_p, te_p = split_list(pos_majority)
    tr_n, va_n, te_n = split_list(neg_majority)

    train_graphs = tr_p + tr_n
    val_graphs = va_p + va_n
    test_graphs = te_p + te_n

    random.shuffle(train_graphs)
    random.shuffle(val_graphs)
    random.shuffle(test_graphs)
    return train_graphs, val_graphs, test_graphs


def standardize_from_training(train_graphs: List[Data]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute mean/std across all nodes of training graphs (valid nodes only)."""
    xs = []
    for g in train_graphs:
        # Use detection nodes only (exclude sensor node if present)
        if hasattr(g, 'valid_mask'):
            xs.append(g.x[g.valid_mask])
        else:
            xs.append(g.x)
    x_train = torch.cat(xs, dim=0)
    mean = x_train.mean(dim=0, keepdim=True)
    std = x_train.std(dim=0, keepdim=True)
    std = torch.where(std < 1e-6, torch.ones_like(std), std)
    return mean, std


def build_knn_edges_from_matrix(x_matrix: torch.Tensor, feature_cols: List[str], knn_on_cols: List[str], k: int) -> torch.Tensor:
    """Build kNN edges using selected columns from an already standardized feature matrix."""
    knn_idx = [feature_cols.index(c) for c in knn_on_cols]
    x_for_knn = x_matrix[:, knn_idx]
    N = x_matrix.size(0)
    k_eff = max(1, min(k, max(1, N - 1)))
    if N <= 1:
        return torch.empty((2, 0), dtype=torch.long)
    return knn_graph(x_for_knn, k=k_eff, loop=False)


def add_sensor_node_and_star_edges(g: Data,
                                   feature_cols: List[str],
                                   mean: torch.Tensor,
                                   std: torch.Tensor,
                                   x_std_detection: torch.Tensor,
                                   edge_index_detection: torch.Tensor) -> None:
    """
    Prepend a sensor node to g.x and connect it to all detection nodes (star edges).
    - Sensor node features: zeros except sensor coords set to per-dwell average, then standardized.
    - g.y is extended with a dummy label for sensor and a valid_mask excludes sensor from loss/metrics.
    - Existing detection edges are shifted by +1 and concatenated with star edges.
    """
    # Build a sensor feature vector in raw space (same dimensionality as g.x)
    # Current g.x is raw (before standardization) when this is called
    sensor_idx = [feature_cols.index('x_sensor'), feature_cols.index('y_sensor'), feature_cols.index('z_sensor')]

    # Compute per-dwell average sensor coordinates from raw detection nodes
    # We need access to raw detection features; since we passed x_std_detection (standardized),
    # we can reconstruct raw via mean/std if needed, but simpler: compute from g.x (raw) now.
    x_raw_detection = g.x  # raw features (before standardization)
    sensor_raw = torch.zeros((1, x_raw_detection.size(1)), dtype=x_raw_detection.dtype)
    sensor_raw[0, sensor_idx] = x_raw_detection[:, sensor_idx].mean(dim=0)
    # Standardize sensor features consistently
    sensor_std = (sensor_raw - mean) / std

    # Prepend sensor node features to standardized detection features
    g.x = torch.cat([sensor_std, x_std_detection], dim=0)  # [1 + N, F]

    # Extend y with a dummy label for sensor and create valid mask
    N = x_std_detection.size(0)
    y_sensor = torch.tensor([-1], dtype=g.y.dtype)  # sentinel
    g.y = torch.cat([y_sensor, g.y], dim=0)
    valid_mask = torch.zeros(N + 1, dtype=torch.bool)
    valid_mask[1:] = True  # exclude sensor node at index 0
    g.valid_mask = valid_mask

    # Build star edges sensor<->detections
    src = torch.zeros(N, dtype=torch.long)  # sensor node id 0
    dst = torch.arange(1, N + 1, dtype=torch.long)
    star_edges = torch.vstack([torch.cat([src, dst]), torch.cat([dst, src])])  # bidirectional

    # Shift detection edges by +1 and combine
    if edge_index_detection is not None and edge_index_detection.numel() > 0:
        edge_index_detection = edge_index_detection + 1  # shift node ids
        g.edge_index = torch.cat([edge_index_detection, star_edges], dim=1)
    else:
        g.edge_index = star_edges

# -------------------- Model --------------------

class EdgeConvNet(nn.Module):
    """Node classification with successive EdgeConv layers and a node-wise MLP head."""
    def __init__(self,
                 in_channels: int,
                 hidden_channels: int = 64,
                 num_layers: int = 2,
                 mlp_hidden: int = 64,
                 dropout: float = 0.2):
        super().__init__()
        self.edgeconvs = nn.ModuleList()
        self.bns = nn.ModuleList()
        ch_in = in_channels
        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(2 * ch_in, hidden_channels),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_channels, hidden_channels),
                nn.ReLU(inplace=True),
            )
            self.edgeconvs.append(EdgeConv(nn=mlp, aggr='max'))
            self.bns.append(nn.BatchNorm1d(hidden_channels))
            ch_in = hidden_channels
        self.dropout = nn.Dropout(p=dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_channels, mlp_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(mlp_hidden, 1)  # output logit per node
        )

    def forward(self, x, edge_index):
        for conv, bn in zip(self.edgeconvs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = self.dropout(x)
        logit = self.head(x).squeeze(-1)
        return logit

# -------------------- Training / Evaluation --------------------

@torch.no_grad()
def evaluate_loader(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total = 0
    total_correct = 0
    tp = fp = fn = 0
    for batch in loader:
        x = batch.x.to(device)
        edge_index = batch.edge_index.to(device)
        y = batch.y.to(device)
        mask = batch.valid_mask.to(device) if hasattr(batch, 'valid_mask') else torch.ones_like(y, dtype=torch.bool)
        logits = model(x, edge_index)
        logits_m = logits[mask]
        y_m = y[mask]
        loss = criterion(logits_m.float(), y_m.float())
        total_loss += float(loss.item()) * y_m.numel()
        probs = torch.sigmoid(logits_m)
        preds = (probs >= 0.5).long()
        y_true = y_m.long()
        total_correct += int((preds == y_true).sum().item())
        total += int(y_true.numel())
        tp += int(((preds == 1) & (y_true == 1)).sum().item())
        fp += int(((preds == 1) & (y_true == 0)).sum().item())
        fn += int(((preds == 0) & (y_true == 1)).sum().item())
    avg_loss = total_loss / max(1, total)
    acc = total_correct / max(1, total)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"loss": avg_loss, "acc": acc, "precision": precision, "recall": recall, "f1": f1, "num": total}


def train_with_dataloaders(model, train_loader, val_loader, test_loader, device,
                           epochs=200, lr=1e-3, weight_decay=1e-4,
                           early_stop_patience=25, pos_weight: float = 1.0):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=8, verbose=True)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))

    best_val = float('inf')
    best_state = None
    patience = 0

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_nodes = 0
        for batch in train_loader:
            optimizer.zero_grad()
            x = batch.x.to(device)
            edge_index = batch.edge_index.to(device)
            y = batch.y.to(device)
            mask = batch.valid_mask.to(device) if hasattr(batch, 'valid_mask') else torch.ones_like(y, dtype=torch.bool)
            logits = model(x, edge_index)
            logits_m = logits[mask]
            y_m = y[mask]
            loss = criterion(logits_m.float(), y_m.float())
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            epoch_loss += float(loss.item()) * y_m.numel()
            epoch_nodes += int(y_m.numel())
        avg_train_loss = epoch_loss / max(1, epoch_nodes)

        val_stats = evaluate_loader(model, val_loader, criterion, device)
        train_stats = evaluate_loader(model, train_loader, criterion, device)
        scheduler.step(val_stats["loss"])

        improved = val_stats["loss"] < best_val - 1e-6
        if improved:
            best_val = val_stats["loss"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:03d} "
                f"Train Loss {train_stats['loss']:.4f} Acc {train_stats['acc']:.3f} "
                f"Val Loss {val_stats['loss']:.4f} Acc {val_stats['acc']:.3f} "
                f"F1 {val_stats['f1']:.3f}"
            )

        if patience >= early_stop_patience:
            print(f"Early stopping at epoch {epoch}. Best val loss: {best_val:.4f}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_stats = evaluate_loader(model, test_loader, criterion, device)
    print("Test:",
          f"Loss {test_stats['loss']:.4f}",
          f"Acc {test_stats['acc']:.3f}",
          f"Prec {test_stats['precision']:.3f}",
          f"Rec {test_stats['recall']:.3f}",
          f"F1 {test_stats['f1']:.3f}")
    return model, test_stats

# -------------------- Data Loading --------------------

def load_multi_dwell_graphs(csv_path: str,
                             base_cols: List[str],
                             label_col: str = "label",
                             dwell_col: str = "dwell_id",
                             extra_cols: List[str] = None) -> List[Data]:
    """
    Load all dwells from a single CSV into a list of PyG Data graphs.
    - Validates only base_cols + label_col + dwell_col are present in CSV.
    - Computes x_rel, y_rel, z_rel internally and appends them to features.
    - extra_cols: optional list of additional numeric columns to include as features.
    """
    assert os.path.exists(csv_path), f"CSV not found: {csv_path}"
    df = pd.read_csv(csv_path)

    extra_cols = extra_cols or []

    missing = [c for c in base_cols + [label_col, dwell_col] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}. Found: {list(df.columns)}")

    graphs = []
    for dwell_id, gdf in df.groupby(dwell_col):
        # Base features
        x_sensor = gdf[["x_sensor", "y_sensor", "z_sensor"]].values.astype(np.float32)
        x_target = gdf[["x_target", "y_target", "z_target"]].values.astype(np.float32)
        x_rel = x_target - x_sensor  # [N, 3]

        base_features = gdf[base_cols].values.astype(np.float32)
        extras = gdf[extra_cols].values.astype(np.float32) if extra_cols else np.empty((len(gdf), 0), dtype=np.float32)

        # Concatenate base + extras + rel
        x_np = np.concatenate([base_features, extras, x_rel], axis=1)
        y_np = gdf[label_col].values.astype(np.int64).reshape(-1)

        data = Data(x=torch.from_numpy(x_np), y=torch.from_numpy(y_np))
        data.dwell_id = dwell_id
        graphs.append(data)
    return graphs

# -------------------- Main --------------------

def main():
    parser = argparse.ArgumentParser(description="EdgeConv GNN for multiple dwells with sensor node, star edges, and relative features (fixed).")
    parser.add_argument("--csv", type=str, default=r"all_dwells.csv", help="Path to the CSV containing all dwells.")
    parser.add_argument("--dwell-col", type=str, default="dwell_id")
    parser.add_argument("--label-col", type=str, default="label")
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--mlp-hidden", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--knn-on", type=str, nargs="+", default=["x_rel", "y_rel", "z_rel"],
                        help="Columns used to build kNN graph (default: x_rel y_rel z_rel).")
    parser.add_argument("--include-extras", type=str, nargs="*", default=[],
                        help="Optional extra numeric columns to include as features, e.g., heading_deg velocity")

    args = parser.parse_args()
    set_seed(args.seed)

    # Columns expected in CSV (do NOT include rel cols here)
    base_cols = [
        "x_sensor", "y_sensor", "z_sensor",
        "x_target", "y_target", "z_target",
        "LOS_Velocity",
    ]

    # Full feature set AFTER we add rel (and extras)
    feature_cols = base_cols + args.include_extras + ["x_rel", "y_rel", "z_rel"]

    # 1) Load graphs (raw features: base + extras + rel computed inside)
    graphs = load_multi_dwell_graphs(csv_path=args.csv,
                                     base_cols=base_cols,
                                     label_col=args.label_col,
                                     dwell_col=args.dwell_col,
                                     extra_cols=args.include_extras)

    if len(graphs) == 0:
        raise RuntimeError("No dwells found in the CSV. Check dwell_id column and data.")

    # 2) For now, create valid_mask placeholders (all True) before standardization; will be set after sensor addition
    for g in graphs:
        g.valid_mask = torch.ones(g.x.size(0), dtype=torch.bool)

    # 3) Split by dwell (graph-level splits)
    train_graphs, val_graphs, test_graphs = stratified_graph_split(
        graphs, train_ratio=args.train_ratio, val_ratio=args.val_ratio, test_ratio=args.test_ratio, seed=args.seed
    )

    print(f"Splits — train: {len(train_graphs)}, val: {len(val_graphs)}, test: {len(test_graphs)}")

    # 4) Standardize based on training graphs only (using valid nodes only)
    mean, std = standardize_from_training(train_graphs)

    # 5) For each split: standardize detection nodes, build kNN edges among detections, then add sensor node + star edges
    def process_graphs(gs: List[Data]):
        for g in gs:
            # Standardize detection features
            x_std_det = (g.x - mean) / std
            # Build kNN edges among detections (no sensor yet)
            edge_det = build_knn_edges_from_matrix(x_std_det, feature_cols, args.knn_on, args.k)
            # Add sensor node, extend y/valid_mask, and add star edges
            add_sensor_node_and_star_edges(g, feature_cols, mean, std, x_std_det, edge_det)

    process_graphs(train_graphs)
    process_graphs(val_graphs)
    process_graphs(test_graphs)

    # 6) Compute pos_weight from training graphs (valid nodes only)
    pos_weight = compute_pos_weight(train_graphs)
    print(f"Train class balance — pos_weight: {pos_weight:.3f}")

    # 7) DataLoaders
    train_loader = DataLoader(train_graphs, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_graphs, batch_size=args.batch_size, shuffle=False)

    # 8) Model
    model = EdgeConvNet(
        in_channels=len(feature_cols),
        hidden_channels=args.hidden,
        num_layers=args.layers,
        mlp_hidden=args.mlp_hidden,
        dropout=args.dropout,
    )

    # 9) Train
    model, test_stats = train_with_dataloaders(
        model, train_loader, val_loader, test_loader, device=args.device,
        epochs=args.epochs, lr=args.lr, weight_decay=args.wd,
        early_stop_patience=25, pos_weight=pos_weight
    )

    # 10) Save best model and preprocessing metadata
    ckpt_path = "edgeconv_multi_dwells_sensor_fixed_best.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "feature_cols": feature_cols,
        "base_cols": base_cols,
        "extra_cols": args.include_extras,
        "label_col": args.label_col,
        "dwell_col": args.dwell_col,
        "feat_mean": mean,
        "feat_std": std,
        "k": args.k,
        "knn_on_cols": args.knn_on,
        "args": vars(args),
        "test_stats": test_stats,
    }, ckpt_path)
    print(f"Saved best model checkpoint to: {os.path.abspath(ckpt_path)}")


if __name__ == "__main__":
    main()
