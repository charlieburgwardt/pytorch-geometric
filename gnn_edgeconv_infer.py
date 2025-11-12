# gnn_edgeconv_infer.py
import os
import argparse
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.nn import EdgeConv, knn_graph


# ----------------- Model (same as training) -----------------

class EdgeConvNet(nn.Module):
    """
    Node classification with successive EdgeConv layers and a node-wise MLP head.
    Must match the architecture used in training.
    """
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
            nn.Linear(mlp_hidden, 1)
        )

    def forward(self, x, edge_index):
        for conv, bn in zip(self.edgeconvs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = self.dropout(x)
        logit = self.head(x).squeeze(-1)
        return logit


# ----------------- Helpers -----------------

def load_checkpoint(ckpt_path: str, map_location: str = "cpu"):
    assert os.path.exists(ckpt_path), f"Checkpoint not found: {ckpt_path}"
    ckpt = torch.load(ckpt_path, map_location=map_location)
    # Required fields
    feature_cols = ckpt["feature_cols"]
    label_col = ckpt["label_col"]
    feat_mean = ckpt["feat_mean"].float()
    feat_std = ckpt["feat_std"].float()
    k = ckpt.get("k", 8)
    knn_on_cols = ckpt.get("knn_on_cols", ["x_target", "y_target"])
    args = ckpt.get("args", {})

    # Rebuild model with identical hyperparameters
    in_channels = len(feature_cols)
    hidden = int(args.get("hidden", 64))
    layers = int(args.get("layers", 2))
    mlp_hidden = int(args.get("mlp_hidden", 64))
    dropout = float(args.get("dropout", 0.2))

    model = EdgeConvNet(
        in_channels=in_channels,
        hidden_channels=hidden,
        num_layers=layers,
        mlp_hidden=mlp_hidden,
        dropout=dropout
    )
    model.load_state_dict(ckpt["model_state_dict"], strict=True)

    return {
        "model": model,
        "feature_cols": feature_cols,
        "label_col": label_col,
        "feat_mean": feat_mean,
        "feat_std": feat_std,
        "k": k,
        "knn_on_cols": knn_on_cols,
        "args": args,
        "ckpt": ckpt,
    }


def standardize_with_stats(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    std = torch.where(std < 1e-6, torch.ones_like(std), std)
    return (x - mean) / std


def build_knn_edges_from_cols(x_std: torch.Tensor,
                              feature_cols,
                              knn_on_cols,
                              k: int,
                              loop: bool = False):
    idxs = [feature_cols.index(c) for c in knn_on_cols]
    x_for_knn = x_std[:, idxs]
    edge_index = knn_graph(x_for_knn, k=k, loop=loop)
    return edge_index


def evaluate_predictions(logits: torch.Tensor, y_true: torch.Tensor):
    """
    Compute metrics on the full set. Returns dict of metrics.
    """
    with torch.no_grad():
        criterion = nn.BCEWithLogitsLoss()
        loss = criterion(logits.float(), y_true.float()).item()
        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).long()

        y = y_true.long()
        correct = (preds == y).sum().item()
        total = y.numel()
        acc = correct / total if total > 0 else 0.0

        tp = ((preds == 1) & (y == 1)).sum().item()
        tn = ((preds == 0) & (y == 0)).sum().item()
        fp = ((preds == 1) & (y == 0)).sum().item()
        fn = ((preds == 0) & (y == 1)).sum().item()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        roc_auc = None
        try:
            from sklearn.metrics import roc_auc_score
            roc_auc = float(roc_auc_score(y.cpu().numpy(), probs.cpu().numpy()))
        except Exception:
            pass

        return {
            "loss": float(loss),
            "acc": float(acc),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "tp": int(tp),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "roc_auc": roc_auc
        }


def run_inference(csv_path: str, ckpt_path: str, out_csv: str = None, device: str = None,
                  override_k: int = None, override_knn_on=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    payload = load_checkpoint(ckpt_path, map_location=device)
    model = payload["model"].to(device).eval()
    feature_cols = payload["feature_cols"]
    label_col = payload["label_col"]
    mean = payload["feat_mean"].to(device)
    std = payload["feat_std"].to(device)
    k = override_k if override_k is not None else payload["k"]
    knn_on_cols = override_knn_on if override_knn_on is not None else payload["knn_on_cols"]

    assert os.path.exists(csv_path), f"CSV not found: {csv_path}"
    df = pd.read_csv(csv_path)

    # Validate columns
    missing = [c for c in feature_cols + [label_col] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}. Found: {list(df.columns)}")

    # Build features and labels
    x_np = df[feature_cols].values.astype(np.float32)
    y_np = df[label_col].values.astype(np.int64).reshape(-1)
    x = torch.from_numpy(x_np).to(device)
    y = torch.from_numpy(y_np).to(device)

    # Standardize with training stats
    x_std = standardize_with_stats(x, mean, std)

    # kNN edges using the same columns used in training
    edge_index = build_knn_edges_from_cols(x_std, feature_cols, knn_on_cols, k=k, loop=False)

    # Create Data and run forward
    data = Data(x=x_std, edge_index=edge_index, y=y)
    with torch.no_grad():
        logits = model(data.x, data.edge_index)

    # Metrics
    metrics = evaluate_predictions(logits, data.y)
    probs = torch.sigmoid(logits).detach().cpu().numpy()
    preds = (probs >= 0.5).astype(np.int64)

    # Save predictions
    if out_csv is None:
        base, ext = os.path.splitext(csv_path)
        out_csv = base + "_predictions.csv"

    df_out = df.copy()
    df_out["prob_1"] = probs
    df_out["pred"] = preds
    df_out.to_csv(out_csv, index=False)

    # Pretty print
    print("\n=== Inference & Evaluation ===")
    print(f"CSV:    {os.path.abspath(csv_path)}")
    print(f"CKPT:   {os.path.abspath(ckpt_path)}")
    print(f"kNN k:  {k} | kNN on: {knn_on_cols}")
    print(f"Saved predictions to: {os.path.abspath(out_csv)}")
    print("\nMetrics (full file):")
    print(f"  Loss     : {metrics['loss']:.4f}")
    print(f"  Accuracy : {metrics['acc']:.3f}")
    print(f"  Precision: {metrics['precision']:.3f}")
    print(f"  Recall   : {metrics['recall']:.3f}")
    print(f"  F1       : {metrics['f1']:.3f}")
    if metrics["roc_auc"] is not None:
        print(f"  ROC-AUC  : {metrics['roc_auc']:.3f}")
    print("  Confusion Matrix:")
    print(f"      TP: {metrics['tp']}  FP: {metrics['fp']}")
    print(f"      FN: {metrics['fn']}  TN: {metrics['tn']}")

    return metrics, out_csv


def main():
    parser = argparse.ArgumentParser(description="Inference & evaluation for EdgeConv GNN on a new STANAG 4607 dwell CSV.")
    parser.add_argument("--csv", type=str, required=True,
                        help="Path to the new simulation CSV to score.")
    parser.add_argument("--ckpt", type=str, required=True,
                        help="Path to the saved checkpoint (edgeconv_dwell_best.pt).")
    parser.add_argument("--out-csv", type=str, default=None,
                        help="Optional path for CSV with predictions appended.")
    parser.add_argument("--device", type=str, default=None,
                        help="cuda or cpu (default: auto)")
    parser.add_argument("--k", type=int, default=None,
                        help="Optional override for kNN k (defaults to training value).")
    parser.add_argument("--knn-on", type=str, nargs="+", default=None,
                        help="Optional override for kNN columns (defaults to training value).")
    args = parser.parse_args()

    run_inference(csv_path=args.csv,
                  ckpt_path=args.ckpt,
                  out_csv=args.out_csv,
                  device=args.device,
                  override_k=args.k,
                  override_knn_on=args.knn_on)


if __name__ == "__main__":
    main()
