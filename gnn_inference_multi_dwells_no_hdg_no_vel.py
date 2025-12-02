import argparse
import torch
import pandas as pd
import numpy as np
from torch_geometric.data import Data
from torch_geometric.nn import knn_graph
from gnn_multi_dwells_no_hdg_no_vel import EdgeConvNet, add_sensor_node_and_star_edges


def build_graphs(csv_path, feature_cols, has_labels: bool):
    df = pd.read_csv(csv_path)
    graphs = []
    for dwell_id, gdf in df.groupby("dwell_id"):
        x_sensor = gdf[["x_sensor","y_sensor","z_sensor"]].values.astype(np.float32)
        x_target = gdf[["x_target","y_target","z_target"]].values.astype(np.float32)
        x_rel = x_target - x_sensor
        base_features = gdf[feature_cols[:7]].values.astype(np.float32)
        x_np = np.concatenate([base_features, x_rel], axis=1)
        if has_labels:
            y_np = gdf["label"].values.astype(np.int64)
        else:
            # No labels in operations; create a placeholder vector (zeros) for shape
            y_np = np.zeros(x_np.shape[0], dtype=np.int64)
        data = Data(x=torch.from_numpy(x_np), y=torch.from_numpy(y_np))
        data.dwell_id = dwell_id
        graphs.append(data)
    return graphs


def prepare_graphs(graphs, feature_cols, mean, std, k):
    for g in graphs:
        x_std_det = (g.x - mean) / std
        knn_idx = [feature_cols.index(c) for c in ["x_rel","y_rel","z_rel"]]
        x_knn = x_std_det[:, knn_idx]
        N = x_std_det.size(0)
        k_eff = max(1, min(k, max(1, N-1)))
        edge_det = knn_graph(x_knn, k=k_eff, loop=False) if N > 1 else torch.empty((2,0), dtype=torch.long)
        add_sensor_node_and_star_edges(g, feature_cols, mean, std, x_std_det, edge_det)


def run_inference(model, graphs, device, out_csv, compute_metrics: bool = False, metrics_path: str = None):
    model.eval()
    rows = []
    all_preds, all_true = [], []
    with torch.no_grad():
        for g in graphs:
            x, edge_index = g.x.to(device), g.edge_index.to(device)
            mask = g.valid_mask
            logits = model(x, edge_index)[mask]
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            preds = np.argmax(probs, axis=1)
            rows.extend([
                {
                    "dwell_id": g.dwell_id,
                    "node_idx": i,
                    "pred_class": int(preds[i]),
                    "prob_background": float(probs[i][0]),
                    "prob_column": float(probs[i][1]),
                    "prob_line_abreast": float(probs[i][2])
                }
                for i in range(len(preds))
            ])
            if compute_metrics:
                y_true = g.y[mask].cpu().numpy()
                all_preds.extend(preds)
                all_true.extend(y_true)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"Saved predictions to {out_csv}")
    if compute_metrics and metrics_path is not None:
        from sklearn.metrics import classification_report
        report = classification_report(all_true, all_preds, labels=[0,1,2], target_names=["background","column","line_abreast"], digits=3)
        with open(metrics_path, "w") as f:
            f.write(report)
        print(f"Saved metrics to {metrics_path}\n{report}")


def main():
    parser = argparse.ArgumentParser(description="Inference for multi-class EdgeConv GNN without heading/velocity (labels optional)")
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", type=str, default="predictions.csv")
    parser.add_argument("--metrics", type=str, default=None, help="Path to write classification report if labels are present")
    parser.add_argument("--weights-only", action="store_true", help="Use safe torch.load(weights_only=True) if supported")
    args = parser.parse_args()

    # Safe load suggestion; fallback if not supported in current PyTorch version
    try:
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=args.weights_only)
    except TypeError:
        ckpt = torch.load(args.checkpoint, map_location="cpu")

    feature_cols = ckpt["feature_cols"]
    mean, std, k = ckpt["mean"], ckpt["std"], ckpt["k"]

    # Detect if CSV has labels
    df_head = pd.read_csv(args.csv, nrows=1)
    has_labels = "label" in df_head.columns
    if not has_labels and args.metrics is not None:
        print("[Info] 'label' column not found. Metrics will be skipped.")

    graphs = build_graphs(args.csv, feature_cols, has_labels)
    prepare_graphs(graphs, feature_cols, mean, std, k)

    model = EdgeConvNet(in_channels=len(feature_cols))
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(args.device)

    run_inference(model, graphs, args.device, args.out, compute_metrics=has_labels and args.metrics is not None, metrics_path=args.metrics)

if __name__ == "__main__":
    main()

# How to run
# python gnn_inference_multi_dwells_no_hdg_no_vel.py `
#   --csv "C:/Users/charlie.burgwardt/OneDrive - NV5/GMTI/Data/three_class_dwells.csv" `
#   --checkpoint "edgeconv_multi_dwells_multiclass_no_hdg_no_vel.pt" `
#   --device cpu `
#   --out "predictions.csv"
