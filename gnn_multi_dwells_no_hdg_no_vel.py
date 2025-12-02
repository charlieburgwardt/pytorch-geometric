import os
import argparse
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import EdgeConv, knn_graph
from sklearn.metrics import classification_report, confusion_matrix

# -------------------- Utilities --------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def stratified_graph_split(graphs, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    set_seed(seed)
    random.shuffle(graphs)
    n = len(graphs)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    return graphs[:n_train], graphs[n_train:n_train+n_val], graphs[n_train+n_val:]

def standardize_from_training(train_graphs):
    xs = []
    for g in train_graphs:
        xs.append(g.x)
    x_train = torch.cat(xs, dim=0)
    mean = x_train.mean(dim=0, keepdim=True)
    std = x_train.std(dim=0, keepdim=True)
    std = torch.where(std < 1e-6, torch.ones_like(std), std)
    return mean, std

def build_knn_edges(x_matrix, feature_cols, knn_on_cols, k):
    knn_idx = [feature_cols.index(c) for c in knn_on_cols]
    x_for_knn = x_matrix[:, knn_idx]
    N = x_matrix.size(0)
    k_eff = max(1, min(k, max(1, N - 1)))
    if N <= 1:
        return torch.empty((2, 0), dtype=torch.long)
    return knn_graph(x_for_knn, k=k_eff, loop=False)

def add_sensor_node_and_star_edges(g, feature_cols, mean, std, x_std_detection, edge_index_detection):
    sensor_idx = [feature_cols.index('x_sensor'), feature_cols.index('y_sensor'), feature_cols.index('z_sensor')]
    x_raw_detection = g.x
    sensor_raw = torch.zeros((1, x_raw_detection.size(1)), dtype=x_raw_detection.dtype)
    sensor_raw[0, sensor_idx] = x_raw_detection[:, sensor_idx].mean(dim=0)
    sensor_std = (sensor_raw - mean) / std
    g.x = torch.cat([sensor_std, x_std_detection], dim=0)
    N = x_std_detection.size(0)
    y_sensor = torch.tensor([-1], dtype=g.y.dtype)
    g.y = torch.cat([y_sensor, g.y], dim=0)
    valid_mask = torch.zeros(N + 1, dtype=torch.bool)
    valid_mask[1:] = True
    g.valid_mask = valid_mask
    src = torch.zeros(N, dtype=torch.long)
    dst = torch.arange(1, N + 1, dtype=torch.long)
    star_edges = torch.vstack([torch.cat([src, dst]), torch.cat([dst, src])])
    if edge_index_detection.numel() > 0:
        edge_index_detection = edge_index_detection + 1
        g.edge_index = torch.cat([edge_index_detection, star_edges], dim=1)
    else:
        g.edge_index = star_edges

# -------------------- Model --------------------
class EdgeConvNet(nn.Module):
    def __init__(self, in_channels, hidden_channels=64, num_layers=2, mlp_hidden=64, dropout=0.2, num_classes=3):
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
            nn.Linear(mlp_hidden, num_classes)
        )

    def forward(self, x, edge_index):
        for conv, bn in zip(self.edgeconvs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = self.dropout(x)
        return self.head(x)

# -------------------- Training --------------------
@torch.no_grad()
def evaluate_loader(model, loader, criterion, device):
    model.eval()
    total_loss, total_correct, total = 0, 0, 0
    all_preds, all_true = [], []
    for batch in loader:
        x, edge_index = batch.x.to(device), batch.edge_index.to(device)
        y = batch.y.to(device)
        mask = batch.valid_mask.to(device)
        logits = model(x, edge_index)[mask]
        y_true = y[mask]
        loss = criterion(logits, y_true)
        total_loss += loss.item() * y_true.size(0)
        preds = torch.argmax(logits, dim=-1)
        total_correct += (preds == y_true).sum().item()
        total += y_true.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_true.extend(y_true.cpu().numpy())
    acc = total_correct / max(1, total)
    return {"loss": total_loss / max(1, total), "acc": acc, "report": classification_report(all_true, all_preds, labels=[0,1,2], digits=3)}

def train(model, train_loader, val_loader, test_loader, device, epochs=200, lr=1e-3):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    best_val, best_state = float('inf'), None
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss, epoch_nodes = 0, 0
        for batch in train_loader:
            optimizer.zero_grad()
            x, edge_index = batch.x.to(device), batch.edge_index.to(device)
            y = batch.y.to(device)
            mask = batch.valid_mask.to(device)
            logits = model(x, edge_index)[mask]
            y_true = y[mask]
            loss = criterion(logits, y_true)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * y_true.size(0)
            epoch_nodes += y_true.size(0)
        val_stats = evaluate_loader(model, val_loader, criterion, device)
        if val_stats["loss"] < best_val:
            best_val = val_stats["loss"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Train Loss {epoch_loss/epoch_nodes:.4f}, Val Acc {val_stats['acc']:.3f}")
    if best_state:
        model.load_state_dict(best_state)
    test_stats = evaluate_loader(model, test_loader, criterion, device)
    print("Test:", test_stats["report"])
    return model

def load_graphs(csv_path, base_cols, label_col="label", dwell_col="dwell_id"):
    df = pd.read_csv(csv_path)
    graphs = []
    for dwell_id, gdf in df.groupby(dwell_col):
        x_sensor = gdf[["x_sensor", "y_sensor", "z_sensor"]].values.astype(np.float32)
        x_target = gdf[["x_target", "y_target", "z_target"]].values.astype(np.float32)
        x_rel = x_target - x_sensor
        base_features = gdf[base_cols].values.astype(np.float32)
        x_np = np.concatenate([base_features, x_rel], axis=1)
        y_np = gdf[label_col].values.astype(np.int64)
        data = Data(x=torch.from_numpy(x_np), y=torch.from_numpy(y_np))
        data.dwell_id = dwell_id
        graphs.append(data)
    return graphs

def main():
    parser = argparse.ArgumentParser(description="Multi-class EdgeConv GNN without heading/velocity")
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    base_cols = ["x_sensor","y_sensor","z_sensor","x_target","y_target","z_target","LOS_Velocity"]
    feature_cols = base_cols + ["x_rel","y_rel","z_rel"]

    graphs = load_graphs(args.csv, base_cols)
    for g in graphs:
        g.valid_mask = torch.ones(g.x.size(0), dtype=torch.bool)

    train_graphs, val_graphs, test_graphs = stratified_graph_split(graphs)
    print(f"Splits — train: {len(train_graphs)}, val: {len(val_graphs)}, test: {len(test_graphs)}")

    mean, std = standardize_from_training(train_graphs)
    def process(gs):
        for g in gs:
            x_std_det = (g.x - mean) / std
            edge_det = build_knn_edges(x_std_det, feature_cols, ["x_rel","y_rel","z_rel"], args.k)
            add_sensor_node_and_star_edges(g, feature_cols, mean, std, x_std_det, edge_det)
    process(train_graphs)
    process(val_graphs)
    process(test_graphs)

    train_loader = DataLoader(train_graphs, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=args.batch_size)
    test_loader = DataLoader(test_graphs, batch_size=args.batch_size)

    model = EdgeConvNet(in_channels=len(feature_cols))
    model = train(model, train_loader, val_loader, test_loader, args.device, epochs=args.epochs, lr=args.lr)

    ckpt_path = "edgeconv_multi_dwells_multiclass_no_hdg_no_vel.pt"
    torch.save({"model_state_dict": model.state_dict(), "feature_cols": feature_cols, "mean": mean, "std": std, "k": args.k}, ckpt_path)
    print(f"Saved checkpoint to {ckpt_path}")

if __name__ == "__main__":
    main()

# How to run
#python gnn_multi_dwells_no_hdg_no_vel.py `
#--csv "C:/Users/charlie.burgwardt/OneDrive - NV5/GMTI/Data/three_class_dwells.csv" `
#--device cpu `
#--epochs 200 `
#--batch-size 32