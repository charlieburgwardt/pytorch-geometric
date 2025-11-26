
import os
import argparse
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.nn import knn_graph

# Import model class and helper from training script
from gnn_edgeconv_multi_dwells_sensor import EdgeConvNet, add_sensor_node_and_star_edges

def build_graphs_for_inference(csv_path, base_cols, extra_cols, feature_cols):
    df = pd.read_csv(csv_path)
    graphs = []
    for dwell_id, gdf in df.groupby('dwell_id'):
        x_sensor = gdf[["x_sensor", "y_sensor", "z_sensor"]].values.astype(np.float32)
        x_target = gdf[["x_target", "y_target", "z_target"]].values.astype(np.float32)
        x_rel = x_target - x_sensor
        base_features = gdf[base_cols].values.astype(np.float32)
        extras = gdf[extra_cols].values.astype(np.float32) if extra_cols else np.empty((len(gdf), 0), dtype=np.float32)
        x_np = np.concatenate([base_features, extras, x_rel], axis=1)
        # Use actual labels if present, else zeros
        y_np = gdf['label'].values.astype(np.int64) if 'label' in gdf.columns else np.zeros((x_np.shape[0],), dtype=np.int64)
        data = Data(x=torch.from_numpy(x_np), y=torch.from_numpy(y_np))
        data.dwell_id = dwell_id
        graphs.append(data)
    return graphs

def prepare_for_inference(graphs, feature_cols, mean, std, knn_on, k):
    def build_knn_edges(x_std):
        idxs = [feature_cols.index(c) for c in knn_on]
        x_knn = x_std[:, idxs]
        N = x_std.size(0)
        k_eff = max(1, min(k, max(1, N-1)))
        return knn_graph(x_knn, k=k_eff, loop=False) if N > 1 else torch.empty((2,0), dtype=torch.long)

    for g in graphs:
        x_std = (g.x - mean) / std
        edge_det = build_knn_edges(x_std)
        add_sensor_node_and_star_edges(g, feature_cols, mean, std, x_std, edge_det)

def run_inference(model, graphs, out_csv="predictions.csv", metrics_csv="per_dwell_metrics.csv", device="cpu"):
    model.eval()
    rows = []
    metrics_rows = []
    for g in graphs:
        x = g.x.to(device)
        edge_index = g.edge_index.to(device)
        mask = g.valid_mask
        y_true = g.y[mask].cpu().numpy()
        with torch.no_grad():
            logits = model(x, edge_index)
            probs = torch.sigmoid(logits[mask]).cpu().numpy()
        preds = (probs >= 0.5).astype(int)

        # Save node-level predictions
        for i, p in enumerate(probs):
            rows.append({"dwell_id": getattr(g, "dwell_id", "unknown"),
                         "node_idx": i,
                         "prob": float(p),
                         "pred": int(preds[i])})

        # Compute per-dwell metrics if labels exist
        if y_true.sum() + (y_true == 0).sum() > 0:  # labels present
            tp = ((preds == 1) & (y_true == 1)).sum()
            fp = ((preds == 1) & (y_true == 0)).sum()
            fn = ((preds == 0) & (y_true == 1)).sum()
            tn = ((preds == 0) & (y_true == 0)).sum()
            prec = tp / (tp + fp) if (tp + fp) else 0.0
            rec = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
            acc = (tp + tn) / max(1, len(y_true))
            metrics_rows.append({"dwell_id": getattr(g, "dwell_id", "unknown"),
                                 "n_nodes": len(y_true),
                                 "acc": acc,
                                 "precision": prec,
                                 "recall": rec,
                                 "f1": f1,
                                 "tp": tp,
                                 "fp": fp,
                                 "fn": fn,
                                 "tn": tn})

    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"Saved predictions to {out_csv}")
    if metrics_rows:
        pd.DataFrame(metrics_rows).to_csv(metrics_csv, index=False)
        print(f"Saved per-dwell metrics to {metrics_csv}")

def main():
    parser = argparse.ArgumentParser(description="Inference for EdgeConv GNN on multiple dwells with per-dwell metrics.")
    parser.add_argument("--csv", type=str, required=True, help="Path to CSV for inference.")
    parser.add_argument("--checkpoint", type=str, default="edgeconv_multi_dwells_sensor_fixed_best.pt", help="Path to model checkpoint.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", type=str, default="predictions.csv", help="Output CSV for predictions.")
    parser.add_argument("--metrics", type=str, default="per_dwell_metrics.csv", help="Output CSV for per-dwell metrics.")
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    feature_cols = ckpt["feature_cols"]
    base_cols = ckpt["base_cols"]
    extra_cols = ckpt.get("extra_cols", [])
    mean, std = ckpt["feat_mean"], ckpt["feat_std"]
    knn_on = ckpt["knn_on_cols"]
    k = ckpt["k"]

    model = EdgeConvNet(in_channels=len(feature_cols), hidden_channels=64, num_layers=2, mlp_hidden=64, dropout=0.2)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(args.device)

    graphs = build_graphs_for_inference(args.csv, base_cols, extra_cols, feature_cols)
    prepare_for_inference(graphs, feature_cols, mean, std, knn_on, k)
    run_inference(model, graphs, out_csv=args.out, metrics_csv=args.metrics, device=args.device)

if __name__ == "__main__":
    main()
