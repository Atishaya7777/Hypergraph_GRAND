import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.datasets import Planetoid
import numpy as np
import time
from typing import Tuple, Optional
import argparse
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Import your model
from models.hypergrand import HypergraphGRAND, create_hypergrand_model
from models.layers import IntegrationScheme


def hyperedge_dropout(hyperedge_index: torch.Tensor, 
                     hyperedge_weight: Optional[torch.Tensor] = None,
                     p: float = 0.1, 
                     training: bool = True) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Apply dropout to hyperedge connections.
    
    Args:
        hyperedge_index: [2, num_connections] hyperedge connectivity
        hyperedge_weight: optional hyperedge weights
        p: dropout probability
        training: whether in training mode
        
    Returns:
        Tuple of (dropped_hyperedge_index, dropped_hyperedge_weight)
    """
    if not training or p == 0.0 or hyperedge_index.size(1) == 0:
        return hyperedge_index, hyperedge_weight
    
    # Create dropout mask for hyperedge connections
    num_connections = hyperedge_index.size(1)
    keep_mask = torch.rand(num_connections, device=hyperedge_index.device) >= p
    
    if keep_mask.sum() == 0:  # Ensure we keep at least one connection
        keep_mask[0] = True
    
    # Apply dropout mask
    dropped_hyperedge_index = hyperedge_index[:, keep_mask]
    
    if hyperedge_weight is not None:
        # For hyperedge weights, we need to handle this more carefully
        # since multiple connections might belong to the same hyperedge
        
        # Get unique hyperedge IDs that remain after connection dropout
        remaining_hyperedges = torch.unique(dropped_hyperedge_index[0])
        
        # Create mapping from old to new hyperedge indices
        hyperedge_mapping = torch.zeros(hyperedge_weight.size(0), dtype=torch.long, device=hyperedge_index.device)
        hyperedge_mapping[remaining_hyperedges] = torch.arange(len(remaining_hyperedges), device=hyperedge_index.device)
        
        # Update hyperedge indices in the dropped index tensor
        dropped_hyperedge_index[0] = hyperedge_mapping[dropped_hyperedge_index[0]]
        
        # Keep only weights for remaining hyperedges
        dropped_hyperedge_weight = hyperedge_weight[remaining_hyperedges]
    else:
        dropped_hyperedge_weight = None
    
    return dropped_hyperedge_index, dropped_hyperedge_weight


def structural_hyperedge_dropout(hyperedge_index: torch.Tensor,
                                hyperedge_weight: Optional[torch.Tensor] = None,
                                p: float = 0.1,
                                training: bool = True) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Apply dropout at the hyperedge level (drop entire hyperedges).
    
    Args:
        hyperedge_index: [2, num_connections] hyperedge connectivity
        hyperedge_weight: optional hyperedge weights
        p: dropout probability for entire hyperedges
        training: whether in training mode
        
    Returns:
        Tuple of (dropped_hyperedge_index, dropped_hyperedge_weight)
    """
    if not training or p == 0.0 or hyperedge_index.size(1) == 0:
        return hyperedge_index, hyperedge_weight
    
    # Get unique hyperedge IDs
    unique_hyperedges = torch.unique(hyperedge_index[0])
    num_hyperedges = len(unique_hyperedges)
    
    # Create dropout mask for entire hyperedges
    keep_mask = torch.rand(num_hyperedges, device=hyperedge_index.device) >= p
    
    if keep_mask.sum() == 0:  # Ensure we keep at least one hyperedge
        keep_mask[0] = True
    
    # Get hyperedges to keep
    kept_hyperedges = unique_hyperedges[keep_mask]
    
    # Create mask for connections belonging to kept hyperedges
    connection_mask = torch.isin(hyperedge_index[0], kept_hyperedges)
    
    # Apply mask to connections
    dropped_hyperedge_index = hyperedge_index[:, connection_mask]
    
    # Remap hyperedge indices to be contiguous
    hyperedge_mapping = torch.zeros(hyperedge_index[0].max().item() + 1, 
                                   dtype=torch.long, device=hyperedge_index.device)
    hyperedge_mapping[kept_hyperedges] = torch.arange(len(kept_hyperedges), device=hyperedge_index.device)
    dropped_hyperedge_index[0] = hyperedge_mapping[dropped_hyperedge_index[0]]
    
    # Handle hyperedge weights
    if hyperedge_weight is not None:
        dropped_hyperedge_weight = hyperedge_weight[kept_hyperedges]
    else:
        dropped_hyperedge_weight = None
    
    return dropped_hyperedge_index, dropped_hyperedge_weight

def graph_to_hypergraph(edge_index: torch.Tensor, num_nodes: int, 
                       hyperedge_type: str = "co_citation") -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Convert a regular graph to a hypergraph using different strategies.
    
    Args:
        edge_index: [2, num_edges] - standard graph edges (assumed to be citation links)
        num_nodes: number of nodes
        hyperedge_type: "co_citation", "citation", or "edge"
    
    Returns:
        hyperedge_index: [2, num_hyperedge_connections] 
        hyperedge_weight: optional weights for hyperedges
    """
    
    if hyperedge_type == "edge":
        # Each edge becomes a hyperedge with 2 nodes
        num_edges = edge_index.size(1)
        hyperedge_index = torch.zeros(2, num_edges * 2, dtype=torch.long)
        
        for i in range(num_edges):
            # Hyperedge i connects nodes edge_index[0, i] and edge_index[1, i]
            hyperedge_index[0, 2*i] = i  # hyperedge id
            hyperedge_index[1, 2*i] = edge_index[0, i]  # first node
            hyperedge_index[0, 2*i + 1] = i  # hyperedge id
            hyperedge_index[1, 2*i + 1] = edge_index[1, i]  # second node
            
        return hyperedge_index, None
    
    elif hyperedge_type == "co_citation":
        # Co-citation: papers that are cited together form a hyperedge
        # If papers A and B both cite paper C, then {A, B, C} form a hyperedge
        
        print("Building co-citation hypergraph...")
        
        # Build citation structure: cited_by[paper] = [papers that cite it]
        cited_by = {i: [] for i in range(num_nodes)}
        
        # In citation networks, edge_index[0] -> edge_index[1] means edge_index[0] cites edge_index[1]
        for i in range(edge_index.size(1)):
            citing_paper = edge_index[0, i].item()
            cited_paper = edge_index[1, i].item()
            cited_by[cited_paper].append(citing_paper)
        
        hyperedge_connections = []
        hyperedge_weights = []
        hyperedge_id = 0
        
        for cited_paper in range(num_nodes):
            citing_papers = cited_by[cited_paper]
            
            # Only create hyperedge if at least 2 papers cite this paper
            if len(citing_papers) >= 2:
                # Create hyperedge with the cited paper and all papers that cite it
                hyperedge_nodes = [cited_paper] + citing_papers
                
                # Remove duplicates and sort
                hyperedge_nodes = sorted(list(set(hyperedge_nodes)))
                
                # Add connections to hyperedge
                for node in hyperedge_nodes:
                    hyperedge_connections.append([hyperedge_id, node])
                
                # Weight based on number of co-citing papers
                weight = len(citing_papers)
                hyperedge_weights.append(weight)
                hyperedge_id += 1
        
        if hyperedge_connections:
            hyperedge_index = torch.tensor(hyperedge_connections, dtype=torch.long).t()
            hyperedge_weight = torch.tensor(hyperedge_weights, dtype=torch.float)
        else:
            hyperedge_index = torch.zeros(2, 0, dtype=torch.long)
            hyperedge_weight = None
            
        return hyperedge_index, hyperedge_weight
    
    elif hyperedge_type == "citation":
        # Citation-based: each paper and all papers it cites form a hyperedge
        # If paper A cites papers {B, C, D}, then {A, B, C, D} form a hyperedge
        
        print("Building citation hypergraph...")
        
        # Build outgoing citations: cites[paper] = [papers it cites]
        cites = {i: [] for i in range(num_nodes)}
        
        for i in range(edge_index.size(1)):
            citing_paper = edge_index[0, i].item()
            cited_paper = edge_index[1, i].item()
            cites[citing_paper].append(cited_paper)
        
        hyperedge_connections = []
        hyperedge_weights = []
        hyperedge_id = 0
        
        for citing_paper in range(num_nodes):
            cited_papers = cites[citing_paper]
            
            # Only create hyperedge if this paper cites at least 1 other paper
            if len(cited_papers) >= 1:
                # Create hyperedge with the citing paper and all papers it cites
                hyperedge_nodes = [citing_paper] + cited_papers
                
                # Remove duplicates and sort
                hyperedge_nodes = sorted(list(set(hyperedge_nodes)))
                
                # Add connections to hyperedge
                for node in hyperedge_nodes:
                    hyperedge_connections.append([hyperedge_id, node])
                
                # Weight based on number of citations made
                weight = len(cited_papers)
                hyperedge_weights.append(weight)
                hyperedge_id += 1
        
        if hyperedge_connections:
            hyperedge_index = torch.tensor(hyperedge_connections, dtype=torch.long).t()
            hyperedge_weight = torch.tensor(hyperedge_weights, dtype=torch.float)
        else:
            hyperedge_index = torch.zeros(2, 0, dtype=torch.long)
            hyperedge_weight = None
            
        return hyperedge_index, hyperedge_weight
    
    else:
        raise ValueError(f"Unknown hyperedge_type: {hyperedge_type}")

def create_full_split(data, train_ratio=0.5, val_ratio=0.25, test_ratio=0.25):
    """Create a more balanced split using all nodes"""
    num_nodes = data.x.size(0)
    indices = torch.randperm(num_nodes)
    
    train_end = int(train_ratio * num_nodes)
    val_end = train_end + int(val_ratio * num_nodes)
    
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    
    train_mask[indices[:train_end]] = True
    val_mask[indices[train_end:val_end]] = True
    test_mask[indices[val_end:]] = True
    
    return train_mask, val_mask, test_mask


def load_planetoid_data(dataset_name: str, root: str = "./data"):
    """Load and preprocess Planetoid dataset"""
    dataset = Planetoid(root=root, name=dataset_name)
    data = dataset[0]
    
    print(f"Dataset: {dataset_name}")
    print(f"Number of nodes: {data.x.size(0)}")
    print(f"Number of edges: {data.edge_index.size(1)}")
    print(f"Number of features: {data.x.size(1)}")
    print(f"Number of classes: {dataset.num_classes}")
    print(f"Training nodes: {data.train_mask.sum().item()}")
    print(f"Validation nodes: {data.val_mask.sum().item()}")
    print(f"Test nodes: {data.test_mask.sum().item()}")
    
    return data, dataset.num_classes


def evaluate_model(model, data, hyperedge_index, hyperedge_weight, mask):
    """Evaluate model performance"""
    model.eval()
    with torch.no_grad():
        out = model(data.x, hyperedge_index, hyperedge_weight)
        logits = out[mask]
        pred = logits.argmax(dim=1)
        y_true = data.y[mask]
        
        acc = accuracy_score(y_true.cpu(), pred.cpu())
        f1 = f1_score(y_true.cpu(), pred.cpu(), average='macro')
        precision = precision_score(y_true.cpu(), pred.cpu(), average='macro')
        recall = recall_score(y_true.cpu(), pred.cpu(), average='macro')
        
        return acc, f1, precision, recall


def train_epoch(model, data, hyperedge_index, hyperedge_weight, optimizer, criterion, edge_dropout_p=0.0, edge_dropout_type="connection"):
    """Train for one epoch with edge dropout"""
    model.train()
    optimizer.zero_grad()
    
    # Apply edge dropout during training
    if edge_dropout_type == "connection":
        dropped_hyperedge_index, dropped_hyperedge_weight = hyperedge_dropout(
            hyperedge_index, hyperedge_weight, p=edge_dropout_p, training=True
        )
    elif edge_dropout_type == "hyperedge":
        dropped_hyperedge_index, dropped_hyperedge_weight = structural_hyperedge_dropout(
            hyperedge_index, hyperedge_weight, p=edge_dropout_p, training=True
        )
    else:
        dropped_hyperedge_index, dropped_hyperedge_weight = hyperedge_index, hyperedge_weight
    
    out = model(data.x, dropped_hyperedge_index, dropped_hyperedge_weight)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    
    return loss.item()


def main():
    parser = argparse.ArgumentParser(description='HyperGRAND on Planetoid datasets')
    parser.add_argument('--dataset', type=str, default='Cora', 
                       choices=['Cora', 'CiteSeer', 'PubMed'],
                       help='Dataset name')
    parser.add_argument('--hidden_dim', type=int, default=32, help='Hidden dimension')
    parser.add_argument('--num_layers', type=int, default=3, help='Number of layers')
    parser.add_argument('--alpha', type=float, default=0.1, help='Diffusion alpha')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs')
    parser.add_argument('--integration_scheme', type=str, default='explicit',
                       choices=['explicit', 'implicit', 'multistep', 'adaptive'],
                       help='Integration scheme')
    parser.add_argument('--hyperedge_type', type=str, default='citation',
                       choices=['edge', 'co_citation', 'citation'],
                       help='How to convert graph to hypergraph')
    parser.add_argument('--edge_dropout', type=float, default=0.1, help='Edge dropout probability')
    parser.add_argument('--edge_dropout_type', type=str, default='connection',
                       choices=['connection', 'hyperedge', 'none'],
                       help='Type of edge dropout: connection-level or hyperedge-level')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--use_full_split', action='store_true', 
                   help='Use 50/25/25 split instead of standard Planetoid split')
    
    args = parser.parse_args()

    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Load data
    data, num_classes = load_planetoid_data(args.dataset)
    data = data.to(device)

    if args.use_full_split:
        print("Using full dataset split (50/25/25)...")
        train_mask, val_mask, test_mask = create_full_split(data)
        data.train_mask = train_mask
        data.val_mask = val_mask  
        data.test_mask = test_mask
    
    # Convert to hypergraph
    print(f"\nConverting to hypergraph using '{args.hyperedge_type}' method...")
    hyperedge_index, hyperedge_weight = graph_to_hypergraph(
        data.edge_index, data.x.size(0), args.hyperedge_type
    )
    hyperedge_index = hyperedge_index.to(device)
    if hyperedge_weight is not None:
        hyperedge_weight = hyperedge_weight.to(device)
    
    num_hyperedges = hyperedge_index[0].max().item() + 1 if hyperedge_index.size(1) > 0 else 0
    print(f"Created {num_hyperedges} hyperedges with {hyperedge_index.size(1)} connections")
    
    # Create model
    print(f"\nCreating HyperGRAND model...")
    print(f"Integration scheme: {args.integration_scheme}")
    print(f"Architecture: {data.x.size(1)} -> {args.hidden_dim} -> {num_classes}")
    
    model = create_hypergrand_model(
        input_dim=data.x.size(1),
        hidden_dim=args.hidden_dim,
        scheme=args.integration_scheme,
        num_layers=args.num_layers,
        alpha=args.alpha,
        dropout=args.dropout
    ).to(device)
    
    # Add output layer for classification
    model.classifier = nn.Linear(args.hidden_dim, num_classes).to(device)
    
    # Update forward method to include classification
    original_forward = model.forward
    def new_forward(x, hyperedge_index, hyperedge_weight=None, membership=None):
        h = original_forward(x, hyperedge_index, hyperedge_weight, membership)
        return model.classifier(h)
    model.forward = new_forward
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Setup training
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    
    print(f"\n{'='*80}")
    print(f"TRAINING HYPERPARAMETERS")
    print(f"{'='*80}")
    print(f"Dataset: {args.dataset}")
    print(f"Integration scheme: {args.integration_scheme}")
    print(f"Hyperedge type: {args.hyperedge_type}")
    print(f"Hidden dimension: {args.hidden_dim}")
    print(f"Number of layers: {args.num_layers}")
    print(f"Alpha (diffusion): {args.alpha}")
    print(f"Dropout: {args.dropout}")
    print(f"Learning rate: {args.lr}")
    print(f"Weight decay: {args.weight_decay}")
    print(f"Edge dropout: {args.edge_dropout}")
    print(f"Edge dropout type: {args.edge_dropout_type}")
    print(f"Epochs: {args.epochs}")
    print(f"Device: {device}")
    print(f"Seed: {args.seed}")
    print(f"{'='*80}")
    
    # Training loop
    best_val_acc = 0
    best_test_acc = 0
    start_time = time.time()
    
    print(f"\n{'Epoch':<6} {'Loss':<8} {'Train Acc':<10} {'Train F1':<10} {'Val Acc':<10} {'Val F1':<10} {'Test Acc':<10} {'Test F1':<10} {'Time':<8}")
    print(f"{'-'*90}")
    
    for epoch in range(args.epochs):
        epoch_start = time.time()
        
        # Train
        train_loss = train_epoch(model, data, hyperedge_index, hyperedge_weight, optimizer, criterion, 
                               args.edge_dropout, args.edge_dropout_type)
        
        # Evaluate (no dropout during evaluation)
        train_acc, train_f1, train_prec, train_rec = evaluate_model(model, data, hyperedge_index, hyperedge_weight, data.train_mask)
        val_acc, val_f1, val_prec, val_rec = evaluate_model(model, data, hyperedge_index, hyperedge_weight, data.val_mask)
        test_acc, test_f1, test_prec, test_rec = evaluate_model(model, data, hyperedge_index, hyperedge_weight, data.test_mask)
        
        epoch_time = time.time() - epoch_start
        
        # Track best performance
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc
        
        # Print epoch results
        print(f"{epoch+1:<6} {train_loss:<8.4f} {train_acc:<10.4f} {train_f1:<10.4f} "
              f"{val_acc:<10.4f} {val_f1:<10.4f} {test_acc:<10.4f} {test_f1:<10.4f} {epoch_time:<8.2f}s")
        
        # Detailed metrics every 50 epochs
        if (epoch + 1) % 50 == 0:
            print(f"    Detailed metrics at epoch {epoch+1}:")
            print(f"    Train - Acc: {train_acc:.4f}, F1: {train_f1:.4f}, Prec: {train_prec:.4f}, Rec: {train_rec:.4f}")
            print(f"    Val   - Acc: {val_acc:.4f}, F1: {val_f1:.4f}, Prec: {val_prec:.4f}, Rec: {val_rec:.4f}")
            print(f"    Test  - Acc: {test_acc:.4f}, F1: {test_f1:.4f}, Prec: {test_prec:.4f}, Rec: {test_rec:.4f}")
    
    total_time = time.time() - start_time
    
    print(f"\n{'='*80}")
    print(f"FINAL RESULTS")
    print(f"{'='*80}")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Test accuracy at best val: {best_test_acc:.4f}")
    print(f"Total training time: {total_time:.2f}s")
    print(f"Average time per epoch: {total_time/args.epochs:.2f}s")
    
    # Final detailed evaluation
    print(f"\nFinal detailed evaluation:")
    final_train_acc, final_train_f1, final_train_prec, final_train_rec = evaluate_model(model, data, hyperedge_index, hyperedge_weight, data.train_mask)
    final_val_acc, final_val_f1, final_val_prec, final_val_rec = evaluate_model(model, data, hyperedge_index, hyperedge_weight, data.val_mask)
    final_test_acc, final_test_f1, final_test_prec, final_test_rec = evaluate_model(model, data, hyperedge_index, hyperedge_weight, data.test_mask)
    
    print(f"Final Train - Acc: {final_train_acc:.4f}, F1: {final_train_f1:.4f}, Prec: {final_train_prec:.4f}, Rec: {final_train_rec:.4f}")
    print(f"Final Val   - Acc: {final_val_acc:.4f}, F1: {final_val_f1:.4f}, Prec: {final_val_prec:.4f}, Rec: {final_val_rec:.4f}")
    print(f"Final Test  - Acc: {final_test_acc:.4f}, F1: {final_test_f1:.4f}, Prec: {final_test_prec:.4f}, Rec: {final_test_rec:.4f}")


if __name__ == "__main__":
    main()
