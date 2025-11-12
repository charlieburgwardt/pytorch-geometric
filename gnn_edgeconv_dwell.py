# gnn_edgeconv_dwell.py
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
from torch_geometric.nn import EdgeConv
from torch_geometric.nn import knn_graph


# --------------- Utilities ---------------

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def stratified_masks(y: torch.Tensor,
                     train_ratio=0.7, val_ratio=0.15, test_ratio=0.15,
                     seed=42) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create boolean masks for train/val/test with approximate stratification on y (binary labels).
    """
    assert math.isclose(train_ratio + val_ratio + test_ratio, 1.0, abs_tol=1e-6)
    set_seed(seed)

    y_np = y.cpu().numpy().astype(int)
    idx0 = np.where(y_np == 0)[0].tolist()
    idx1 = np.where(y_np == 1)[0].tolist()
    random.shuffle(idx0)
    random.shuffle(idx1)

    def split_indices(idxs):
        n = len(idxs)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        n_test = n - n_train - n_val
        return idxs[:n_train], idxs[n_train:n_train+n_val], idxs[n_train+n_val:]

    tr0, va0, te0 = split_indices(idx0)
    tr1, va1, te1 = split_indices(idx1)

    train_idx = np.array(tr0 + tr1, dtype=int)
    val_idx   = np.array(va0 + va1, dtype=int)
    test_idx  = np.array(te0 + te1, dtype=int)

    N = len(y_np)
    train_mask = torch.zeros(N, dtype=torch.bool)
    val_mask   = torch.zeros(N, dtype=torch.bool)
    test_mask  = torch.zeros(N, dtype=torch.bool)
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True
    return train_mask, val_mask, test_mask


def standardize_train(x: torch.Tensor, mask: torch.Tensor):
    """
    Standardize features using only the train split statistics.
    Returns standardized x, and (mean, std) tensors for later use/inference.
    """
    x_train = x[mask]
    mean = x_train.mean(dim=0, keepdim=True)
    std = x_train.std(dim=0, keepdim=True)
    std = torch.where(std < 1e-6, torch.ones_like(std), std)
    x_std = (x - mean) / std
    return x_std, mean, std


def build_knn_edges(x_for_knn: torch.Tensor, k: int, loop: bool = False):
    """
    Build kNN graph edges from coordinates or selected feature dimensions.
    x_for_knn: [N, D_knn]
    """
    # knn_graph returns edge_index shape [2, E]
    edge_index = knn_graph(x_for_knn, k=k, loop=loop)
    return edge_index


# --------------- Model ---------------

class EdgeConvNet(nn.Module):
    """
    Node classification with successive EdgeConv layers and a node-wise MLP head.
    """
    def __init__(self,
                 in_channels: int,
                 hidden_channels: int = 64,
                 num_layers: int = 2,
                 mlp_hidden: int = 64,
                 dropout: float = 0.2):
        super().__init__()

        # Each EdgeConv has an MLP that acts on [x_i || x_j - x_i]
        # so the input dim to that MLP is 2 * channel_dim of the layer input.
        self.edgeconvs = nn.ModuleList()
        self.bns = nn.ModuleList()
        ch_in = in_channels

        for layer in range(num_layers):
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

        # Node-wise classifier head (logit)
        self.head = nn.Sequential(
            nn.Linear(hidden_channels, mlp_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(mlp_hidden, 1)  # output logit
        )

    def forward(self, x, edge_index):
        for conv, bn in zip(self.edgeconvs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = self.dropout(x)
        logit = self.head(x).squeeze(-1)  # [N]
        return logit


# --------------- Training / Evaluation ---------------

@torch.no_grad()
def evaluate(model, data, criterion, device, split: str = "val"):
    model.eval()
    x, edge_index, y = data.x.to(device), data.edge_index.to(device), data.y.to(device)
    if split == "train":
        mask = data.train_mask
    elif split == "val":
        mask = data.val_mask
    else:
        mask = data.test_mask

    mask = mask.to(device)
    logits = model(x, edge_index)
    loss = criterion(logits[mask], y[mask].float())

    probs = torch.sigmoid(logits[mask])
    preds = (probs >= 0.5).long()
    y_true = y[mask].long()

    correct = (preds == y_true).sum().item()
    total = mask.sum().item()
    acc = correct / total if total > 0 else 0.0

    # Precision, Recall, F1 (safe for edge cases)
    tp = ((preds == 1) & (y_true == 1)).sum().item()
    fp = ((preds == 1) & (y_true == 0)).sum().item()
    fn = ((preds == 0) & (y_true == 1)).sum().item()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2*precision*recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "loss": float(loss.item()),
        "acc": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "num": int(total)
    }


def train(model, data, device, epochs=200, lr=1e-3, weight_decay=1e-4,
          early_stop_patience=25, pos_weight: float = 1.0):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                           factor=0.5, patience=8,
                                                           verbose=True)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))

    best_val = float('inf')
    best_state = None
    patience = 0

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        x, edge_index, y = data.x.to(device), data.edge_index.to(device), data.y.to(device)

        logits = model(x, edge_index)
        loss = criterion(logits[data.train_mask].float(), y[data.train_mask].float())
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        # Evaluate
        val_stats = evaluate(model, data, criterion, device, split="val")
        train_stats = evaluate(model, data, criterion, device, split="train")
        scheduler.step(val_stats["loss"])

        improved = val_stats["loss"] < best_val - 1e-6
        if improved:
            best_val = val_stats["loss"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d} | "
                  f"Train Loss {train_stats['loss']:.4f} Acc {train_stats['acc']:.3f} | "
                  f"Val Loss {val_stats['loss']:.4f} Acc {val_stats['acc']:.3f} "
                  f"F1 {val_stats['f1']:.3f}")

        if patience >= early_stop_patience:
            print(f"Early stopping at epoch {epoch}. Best val loss: {best_val:.4f}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_stats = evaluate(model, data, criterion, device, split="test")
    print("Test:",
          f"Loss {test_stats['loss']:.4f}",
          f"Acc {test_stats['acc']:.3f}",
          f"Prec {test_stats['precision']:.3f}",
          f"Rec {test_stats['recall']:.3f}",
          f"F1 {test_stats['f1']:.3f}")
    return model, test_stats


# --------------- Data Loading ---------------

def load_single_dwell_as_graph(csv_path: str,
                               feature_cols: List[str],
                               label_col: str = "label",
                               k: int = 8,
                               knn_on_cols: List[str] = ("x_target", "y_target"),
                               seed: int = 42,
                               train_ratio: float = 0.7,
                               val_ratio: float = 0.15,
                               test_ratio: float = 0.15) -> Data:
    """
    Loads a single dwell of detections into a PyG Data graph, creates kNN edges,
    standardizes features using train split stats, and returns Data with masks.
    """
    assert os.path.exists(csv_path), f"CSV not found: {csv_path}"
    df = pd.read_csv(csv_path)

    missing = [c for c in feature_cols + [label_col] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}. "
                         f"Found: {list(df.columns)}")

    # Features and labels
    x_np = df[feature_cols].values.astype(np.float32)
    y_np = df[label_col].values.astype(np.int64).reshape(-1)

    x = torch.from_numpy(x_np)        # [N, F]
    y = torch.from_numpy(y_np)        # [N]
    N, F = x.shape

    # Split masks (node-wise)
    train_mask, val_mask, test_mask = stratified_masks(y, train_ratio, val_ratio, test_ratio, seed=seed)

    # Standardize features w.r.t training portion
    x_std, mean, std = standardize_train(x, train_mask)

    # Build edges using selected columns (e.g., x_target, y_target)
    knn_idx = [feature_cols.index(c) for c in knn_on_cols]
    x_for_knn = x_std[:, knn_idx]  # standardized coords/features for kNN
    edge_index = build_knn_edges(x_for_knn, k=k, loop=False)  # [2, E]

    data = Data(
        x=x_std,
        edge_index=edge_index,
        y=y,
    )
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

    # Save mean/std for potential later inference (attached to Data object)
    data.feat_mean = mean
    data.feat_std = std
    data.feature_cols = feature_cols
    data.label_col = label_col
    data.knn_on_cols = knn_on_cols
    data.k = k

    return data


# --------------- Main ---------------

def main():
    parser = argparse.ArgumentParser(description="EdgeConv GNN for single STANAG 4607 dwell (node classification).")
    parser.add_argument("--csv",
                        type=str,
                        default=r"C:/Users/charlie.burgwardt/OneDrive - NV5/GMTI/Data/simulation_1.csv",
                        help="Path to the dwell CSV.")
    parser.add_argument("--label-col", type=str, default="label",
                        help="Name of the label column (0/1).")
    parser.add_argument("--k", type=int, default=8,
                        help="k for kNN graph construction.")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--hidden", type=int, default=64, help="Hidden channels in EdgeConv.")
    parser.add_argument("--layers", type=int, default=2, help="Number of EdgeConv layers.")
    parser.add_argument("--mlp-hidden", type=int, default=64, help="Hidden size in classifier MLP.")
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=1e-4, help="Weight decay.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--knn-on", type=str, nargs="+", default=["x_target", "y_target"],
                        help="Columns used to build kNN graph (default: x_target y_target).")
    args = parser.parse_args()

    set_seed(args.seed)

    feature_cols = [
        "x_sensor", "y_sensor", "z_sensor",
        "x_target", "y_target", "LOS_Velocity"
    ]

    # Load data, build graph
    data = load_single_dwell_as_graph(csv_path=args.csv,
                                      feature_cols=feature_cols,
                                      label_col=args.label_col,
                                      k=args.k,
                                      knn_on_cols=args.knn_on,
                                      seed=args.seed,
                                      train_ratio=args.train_ratio,
                                      val_ratio=args.val_ratio,
                                      test_ratio=args.test_ratio)

    # Compute pos_weight for BCE (handles imbalance)
    y_train = data.y[data.train_mask]
    num_pos = int((y_train == 1).sum().item())
    num_neg = int((y_train == 0).sum().item())
    if num_pos == 0:
        print("Warning: No positive samples in train split; setting pos_weight=1.0")
        pos_weight = 1.0
    else:
        pos_weight = num_neg / max(1, num_pos)
    print(f"Train class balance — pos: {num_pos}, neg: {num_neg}, pos_weight: {pos_weight:.3f}")

    # Model
    model = EdgeConvNet(
        in_channels=len(feature_cols),
        hidden_channels=args.hidden,
        num_layers=args.layers,
        mlp_hidden=args.mlp_hidden,
        dropout=args.dropout
    )

    # Train
    model, test_stats = train(model, data, device=args.device,
                              epochs=args.epochs, lr=args.lr, weight_decay=args.wd,
                              early_stop_patience=25, pos_weight=pos_weight)

    # Save best model
    ckpt_path = "edgeconv_dwell_best.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "feature_cols": feature_cols,
        "label_col": args.label_col,
        "feat_mean": data.feat_mean,
        "feat_std": data.feat_std,
        "k": data.k,
        "knn_on_cols": data.knn_on_cols,
        "args": vars(args),
        "test_stats": test_stats
    }, ckpt_path)
    print(f"Saved best model checkpoint to: {os.path.abspath(ckpt_path)}")


if __name__ == "__main__":
    main()