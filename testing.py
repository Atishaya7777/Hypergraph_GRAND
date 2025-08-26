import os
import time
import argparse
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# import your model factory (unchanged)
from models.hypergrand import HypergraphGRAND, create_hypergrand_model
from models.layers import IntegrationScheme

# --------------------------
# Utilities (hyperedge builders, dropout etc.)
# --------------------------
def hyperedge_dropout(hyperedge_index: torch.Tensor,
                      hyperedge_weight: Optional[torch.Tensor] = None,
                      p: float = 0.1,
                      training: bool = True) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    if not training or p == 0.0 or hyperedge_index.size(1) == 0:
        return hyperedge_index, hyperedge_weight
    num_connections = hyperedge_index.size(1)
    keep_mask = torch.rand(num_connections, device=hyperedge_index.device) >= p
    if keep_mask.sum() == 0:
        keep_mask[0] = True
    dropped_hyperedge_index = hyperedge_index[:, keep_mask]
    if hyperedge_weight is not None:
        remaining_hyperedges = torch.unique(dropped_hyperedge_index[0])
        hyperedge_mapping = torch.zeros(hyperedge_weight.size(0), dtype=torch.long, device=hyperedge_index.device)
        hyperedge_mapping[remaining_hyperedges] = torch.arange(len(remaining_hyperedges), device=hyperedge_index.device)
        dropped_hyperedge_index[0] = hyperedge_mapping[dropped_hyperedge_index[0]]
        dropped_hyperedge_weight = hyperedge_weight[remaining_hyperedges]
    else:
        dropped_hyperedge_weight = None
    return dropped_hyperedge_index, dropped_hyperedge_weight

def structural_hyperedge_dropout(hyperedge_index: torch.Tensor,
                                 hyperedge_weight: Optional[torch.Tensor] = None,
                                 p: float = 0.1,
                                 training: bool = True) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    if not training or p == 0.0 or hyperedge_index.size(1) == 0:
        return hyperedge_index, hyperedge_weight
    unique_hyperedges = torch.unique(hyperedge_index[0])
    num_hyperedges = len(unique_hyperedges)
    keep_mask = torch.rand(num_hyperedges, device=hyperedge_index.device) >= p
    if keep_mask.sum() == 0:
        keep_mask[0] = True
    kept_hyperedges = unique_hyperedges[keep_mask]
    connection_mask = torch.isin(hyperedge_index[0], kept_hyperedges)
    dropped_hyperedge_index = hyperedge_index[:, connection_mask]
    hyperedge_mapping = torch.zeros(int(hyperedge_index[0].max().item()) + 1, dtype=torch.long,
                                   device=hyperedge_index.device)
    hyperedge_mapping[kept_hyperedges] = torch.arange(len(kept_hyperedges), device=hyperedge_index.device)
    dropped_hyperedge_index[0] = hyperedge_mapping[dropped_hyperedge_index[0]]
    if hyperedge_weight is not None:
        dropped_hyperedge_weight = hyperedge_weight[kept_hyperedges]
    else:
        dropped_hyperedge_weight = None
    return dropped_hyperedge_index, dropped_hyperedge_weight

def graph_to_hypergraph(edge_index: torch.Tensor, num_nodes: int,
                        hyperedge_type: str = "co_citation"):
    if hyperedge_type == "edge":
        num_edges = edge_index.size(1)
        hyperedge_index = torch.zeros(2, num_edges * 2, dtype=torch.long)
        for i in range(num_edges):
            hyperedge_index[0, 2 * i] = i
            hyperedge_index[1, 2 * i] = edge_index[0, i]
            hyperedge_index[0, 2 * i + 1] = i
            hyperedge_index[1, 2 * i + 1] = edge_index[1, i]
        return hyperedge_index, None
    elif hyperedge_type == "co_citation":
        cited_by = {i: [] for i in range(num_nodes)}
        for i in range(edge_index.size(1)):
            citing = edge_index[0, i].item()
            cited = edge_index[1, i].item()
            cited_by[cited].append(citing)
        hyperedge_connections = []
        hyperedge_weights = []
        hid = 0
        for cited_paper in range(num_nodes):
            citing = cited_by[cited_paper]
            if len(citing) >= 2:
                nodes = [cited_paper] + citing
                nodes = sorted(list(set(nodes)))
                for n in nodes:
                    hyperedge_connections.append([hid, n])
                hyperedge_weights.append(len(citing))
                hid += 1
        if hyperedge_connections:
            return torch.tensor(hyperedge_connections, dtype=torch.long).t(), torch.tensor(hyperedge_weights, dtype=torch.float)
        else:
            return torch.zeros(2, 0, dtype=torch.long), None
    elif hyperedge_type == "citation":
        cites = {i: [] for i in range(num_nodes)}
        for i in range(edge_index.size(1)):
            a = edge_index[0, i].item()
            b = edge_index[1, i].item()
            cites[a].append(b)
        hyperedge_connections = []
        hyperedge_weights = []
        hid = 0
        for a in range(num_nodes):
            cited = cites[a]
            if len(cited) >= 1:
                nodes = [a] + cited
                nodes = sorted(list(set(nodes)))
                for n in nodes:
                    hyperedge_connections.append([hid, n])
                hyperedge_weights.append(len(cited))
                hid += 1
        if hyperedge_connections:
            return torch.tensor(hyperedge_connections, dtype=torch.long).t(), torch.tensor(hyperedge_weights, dtype=torch.float)
        else:
            return torch.zeros(2, 0, dtype=torch.long), None
    else:
        raise ValueError(f"Unknown hyperedge_type {hyperedge_type}")

def create_train_only_hypergraph(edge_index, train_mask, num_nodes, hyperedge_type):
    train_nodes = set(torch.where(train_mask)[0].tolist())
    train_edges = []
    for i in range(edge_index.size(1)):
        s = edge_index[0, i].item()
        t = edge_index[1, i].item()
        if s in train_nodes and t in train_nodes:
            train_edges.append([s, t])
    if not train_edges:
        return torch.zeros(2, 0, dtype=torch.long), None
    train_ei = torch.tensor(train_edges, dtype=torch.long).t()
    return graph_to_hypergraph(train_ei, num_nodes, hyperedge_type)

def create_full_split(data, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    num_nodes = data.x.size(0)
    indices = torch.randperm(num_nodes)
    tr = int(train_ratio * num_nodes)
    vr = tr + int(val_ratio * num_nodes)
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[indices[:tr]] = True
    val_mask[indices[tr:vr]] = True
    test_mask[indices[vr:]] = True
    return train_mask, val_mask, test_mask

def load_planetoid_data(name: str, root: str = "./data"):
    dataset = Planetoid(root=root, name=name)
    data = dataset[0]
    print(f"Dataset: {name}")
    print(f"Nodes: {data.x.size(0)}, Edges: {data.edge_index.size(1)}, Feat dim: {data.x.size(1)}, Classes: {dataset.num_classes}")
    print(f"Train nodes: {int(data.train_mask.sum())}, Val nodes: {int(data.val_mask.sum())}, Test nodes: {int(data.test_mask.sum())}")
    return data, dataset.num_classes

# --------------------------
# Classifier wrapper that keeps encoder independent but adds a small residual skip and head
# --------------------------
class HypergraphClassifier(nn.Module):
    def __init__(self, encoder: HypergraphGRAND, num_classes: int, dropout: float = 0.0):
        super().__init__()
        self.encoder = encoder
        enc_h = encoder.hidden_dim
        self.hidden_dim = enc_h
        # classifier MLP with batchnorm
        self.head = nn.Sequential(
            nn.Linear(enc_h, enc_h),
            nn.BatchNorm1d(enc_h),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(enc_h, num_classes)
        )
        # small projection from input features for skip/residual (keeps encoder usable as encoder)
        self.input_proj = nn.Identity()
        if hasattr(encoder, "input_transform"):
            # reuse encoder input transform if available (keeps shapes aligned)
            self.input_proj = encoder.input_transform
        else:
            self.input_proj = nn.Linear(encoder.input_dim, enc_h)
        # init head
        for m in self.head:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, hyperedge_index, hyperedge_weight=None, membership=None, return_embedding=False):
        h_enc = self.encoder(x, hyperedge_index, hyperedge_weight, membership)  # [N, enc_h]
        # residual: combine with simple projection of inputs (helps classifier discriminate)
        h_skip = self.input_proj(x)
        # simple sum with scaling to avoid blowing magnitude
        h = 0.6 * h_enc + 0.4 * h_skip
        logits = self.head(h)
        if return_embedding:
            return logits, h
        return logits

# Simple GCN baseline kept for comparison
class SimpleGCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.5):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)
        self.dropout = dropout
    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.conv2(x, edge_index)

# --------------------------
# Training / Eval helpers
# --------------------------
def model_forward(model, data, hyperedge_index=None, hyperedge_weight=None, mode='hypergrand'):
    if mode == 'gcn':
        return model(data.x, data.edge_index)
    else:
        logits = model(data.x, hyperedge_index, hyperedge_weight)
        return logits

def evaluate_model(model, data, hyperedge_index, hyperedge_weight, mask, mode):
    model.eval()
    with torch.no_grad():
        out = model_forward(model, data, hyperedge_index, hyperedge_weight, mode)
        logits = out[mask]
        pred = logits.argmax(dim=1)
        y_true = data.y[mask]
        acc = accuracy_score(y_true.cpu(), pred.cpu())
        f1 = f1_score(y_true.cpu(), pred.cpu(), average='macro', zero_division=0)
        precision = precision_score(y_true.cpu(), pred.cpu(), average='macro', zero_division=0)
        recall = recall_score(y_true.cpu(), pred.cpu(), average='macro', zero_division=0)
    return acc, f1, precision, recall

def train_epoch(model, data, hyperedge_index, hyperedge_weight, optimizer, criterion,
                edge_dropout_p=0.0, edge_dropout_type="connection", mode='hypergrand'):
    model.train()
    optimizer.zero_grad()
    if mode == 'gcn':
        out = model_forward(model, data, None, None, mode='gcn')
    else:
        if edge_dropout_type == "connection":
            d_hidx, d_hw = hyperedge_dropout(hyperedge_index, hyperedge_weight, p=edge_dropout_p, training=True)
        elif edge_dropout_type == "hyperedge":
            d_hidx, d_hw = structural_hyperedge_dropout(hyperedge_index, hyperedge_weight, p=edge_dropout_p, training=True)
        else:
            d_hidx, d_hw = hyperedge_index, hyperedge_weight
        out = model_forward(model, data, d_hidx, d_hw, mode='hypergrand')
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

# --------------------------
# Main (argparse) + training loop with early stopping & scheduler
# --------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['hypergrand', 'gcn', 'linear'], default='hypergrand')
    parser.add_argument('--dataset', choices=['Cora', 'CiteSeer', 'PubMed'], default='Cora')
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--lr', type=float, default=5e-3)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--integration_scheme', choices=['explicit','implicit','multistep','adaptive'], default='explicit')
    parser.add_argument('--hyperedge_type', choices=['edge','co_citation','citation'], default='co_citation')
    parser.add_argument('--edge_dropout', type=float, default=0.1)
    parser.add_argument('--edge_dropout_type', choices=['connection','hyperedge','none'], default='connection')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--use_full_split', action='store_true')
    parser.add_argument('--use_full_hypergraph', action='store_true',
                        help='Build hyperedges from full graph rather than train-only edges (recommended for encoder).')
    parser.add_argument('--patience', type=int, default=50)
    parser.add_argument('--save_path', type=str, default='best_hypergrand.pt')
    args = parser.parse_args()

    # seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device)
    print("Using device:", device)

    data, num_classes = load_planetoid_data(args.dataset)
    data = data.to(device)

    if args.use_full_split:
        print("Using full split 50/25/25")
        tr, va, te = create_full_split(data)
        data.train_mask = tr
        data.val_mask = va
        data.test_mask = te

    # create hypergraph (optionally full graph)
    print(f"Converting to hypergraph using '{args.hyperedge_type}' (use_full_hypergraph={args.use_full_hypergraph})...")
    if args.use_full_hypergraph:
        hyperedge_index, hyperedge_weight = graph_to_hypergraph(data.edge_index, data.x.size(0), args.hyperedge_type)
    else:
        hyperedge_index, hyperedge_weight = create_train_only_hypergraph(data.edge_index, data.train_mask, data.x.size(0), args.hyperedge_type)

    hyperedge_index = hyperedge_index.to(device)
    if hyperedge_weight is not None:
        hyperedge_weight = hyperedge_weight.to(device)
    num_hyperedges = (int(hyperedge_index[0].max().item()) + 1) if hyperedge_index.size(1) > 0 else 0
    print(f"Created {num_hyperedges} hyperedges with {hyperedge_index.size(1)} connections")

    print("\nCreating model...")
    print("Integration scheme:", args.integration_scheme)
    print("Architecture:", f"{data.x.size(1)} -> {args.hidden_dim} -> {num_classes}")

    mode = None
    if args.model == 'gcn':
        model = SimpleGCN(input_dim=data.x.size(1), hidden_dim=args.hidden_dim, output_dim=num_classes, dropout=args.dropout).to(device)
        mode = 'gcn'
    elif args.model == 'linear':
        # trivial baseline
        from types import SimpleNamespace
        class LinEnc(nn.Module):
            def __init__(self, in_dim, h):
                super().__init__()
                self.input_transform = nn.Linear(in_dim, h)
                self.hidden_dim = h
                self.input_dim = in_dim
            def forward(self, x, *args, **kwargs):
                return F.relu(self.input_transform(x))
        enc = LinEnc(data.x.size(1), args.hidden_dim).to(device)
        model = HypergraphClassifier(enc, num_classes, dropout=args.dropout).to(device)
        mode = 'hypergrand'
    else:
        enc = create_hypergrand_model(input_dim=data.x.size(1),
                                      hidden_dim=args.hidden_dim,
                                      scheme=args.integration_scheme,
                                      num_layers=args.num_layers,
                                      alpha=args.alpha,
                                      dropout=args.dropout).to(device)
        model = HypergraphClassifier(enc, num_classes, dropout=args.dropout).to(device)
        mode = 'hypergrand'

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params:,}, Trainable: {trainable_params:,}")

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10)
    criterion = nn.CrossEntropyLoss()

    best_val = 0.0
    best_test = 0.0
    best_epoch = 0
    start_time = time.time()
    bad_epochs = 0

    print("\n{:<6} {:<8} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<8}".format(
        'Epoch','Loss','TrainAcc','TrainF1','ValAcc','ValF1','TestAcc','TestF1','Time'
    ))
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        loss = train_epoch(model, data, hyperedge_index, hyperedge_weight, optimizer, criterion,
                           edge_dropout_p=args.edge_dropout, edge_dropout_type=args.edge_dropout_type, mode=mode)
        train_acc, train_f1, _, _ = evaluate_model(model, data, hyperedge_index, hyperedge_weight, data.train_mask, mode=mode)
        val_acc, val_f1, _, _ = evaluate_model(model, data, hyperedge_index, hyperedge_weight, data.val_mask, mode=mode)
        test_acc, test_f1, _, _ = evaluate_model(model, data, hyperedge_index, hyperedge_weight, data.test_mask, mode=mode)
        epoch_time = time.time() - epoch_start

        # scheduler step on val
        scheduler.step(val_acc)

        if val_acc > best_val + 1e-6:
            best_val = val_acc
            best_test = test_acc
            best_epoch = epoch
            bad_epochs = 0
            torch.save({'model_state': model.state_dict(),
                        'optimizer_state': optimizer.state_dict(),
                        'epoch': epoch,
                        'val_acc': val_acc,
                        'test_acc': test_acc}, args.save_path)
        else:
            bad_epochs += 1

        print(f"{epoch:<6} {loss:<8.4f} {train_acc:<10.4f} {train_f1:<10.4f} {val_acc:<10.4f} {val_f1:<10.4f} {test_acc:<10.4f} {test_f1:<10.4f} {epoch_time:<8.2f}s")

        if bad_epochs >= args.patience:
            print(f"Early stopping triggered after {bad_epochs} bad epochs. Best val {best_val:.4f} at epoch {best_epoch}.")
            break

    total_time = time.time() - start_time
    print("Training finished. Best val acc:", best_val, "Test at best val:", best_test)
    print("Saved best model to", args.save_path)
    print("Total time: %.2fs | Avg/epoch: %.2fs" % (total_time, total_time / max(1, epoch)))

if __name__ == "__main__":
    main()

