import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, accuracy_score, adjusted_rand_score
from sklearn.metrics import normalized_mutual_info_score, silhouette_score
from sklearn.cluster import KMeans
import numpy as np
import os
import mlflow
import matplotlib.pyplot as plt
import seaborn as sns
from model import HypergraphGRAND


class HypergraphGRANDWithEmbeddings(torch.nn.Module):
    """Hypergraph GRAND model with learnable node embeddings"""

    def __init__(self, hidden_dim, num_layers=3, alpha=0.1, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers  # Store num_layers
        self.alpha = alpha  # Store alpha
        self.dropout = dropout  # Store dropout
        self.embeddings = None
        self.grand_layers = HypergraphGRAND(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            alpha=alpha,
            dropout=dropout
        )
        self.initialized_num_nodes = 0

    def initialize_for_dataset(self, num_nodes):
        if self.initialized_num_nodes != num_nodes:
            self.embeddings = torch.nn.Embedding(num_nodes, self.hidden_dim)
            self.initialized_num_nodes = num_nodes
            if next(self.parameters(), None) is not None:
                self.embeddings = self.embeddings.to(
                    next(self.parameters()).device)

    def forward(self, num_nodes, hyperedge_index, hyperedge_weight=None, membership=None):
        self.initialize_for_dataset(num_nodes)
        node_indices = torch.arange(num_nodes, device=hyperedge_index.device)
        x = self.embeddings(node_indices)
        return self.grand_layers(x, hyperedge_index, hyperedge_weight, membership)


def load_hypergraph(dataset_name, base_path="./datasets/"):
    """Load hypergraph dataset with MLflow logging"""
    label_file = os.path.join(
        base_path, f"{dataset_name}/node-labels-{dataset_name}.txt")
    edge_file = os.path.join(
        base_path, f"{dataset_name}/hyperedges-{dataset_name}.txt")

    with open(label_file) as f:
        labels_list = [int(line.strip()) for line in f]
    with open(edge_file) as f:
        hyperedges = [list(map(int, line.strip().split(','))) for line in f]

    edge_idx = []
    max_node = 0
    for e_id, edge in enumerate(hyperedges):
        for node in edge:
            max_node = max(max_node, node)
            edge_idx.append((e_id, node))

    num_nodes = max(max_node + 1, len(labels_list))
    if len(labels_list) < num_nodes:
        print(f"Warning: Padding labels from {
              len(labels_list)} to {num_nodes}")
        labels_list.extend([0] * (num_nodes - len(labels_list)))

    labels = torch.tensor(labels_list, dtype=torch.long)
    hyperedge_index = torch.tensor(edge_idx, dtype=torch.long).T

    mlflow.log_params({
        f"{dataset_name}_num_nodes": num_nodes,
        f"{dataset_name}_num_edges": len(hyperedges),
        f"{dataset_name}_num_classes": len(torch.unique(labels)),
        f"{dataset_name}_avg_edge_size": np.mean([len(edge) for edge in hyperedges])
    })

    return hyperedge_index, labels, num_nodes


def contrastive_clustering_loss(model_output, node_labels, temperature=0.1):
    """Contrastive loss for clustering"""
    model_output_norm = F.normalize(model_output, p=2, dim=1)
    similarity_matrix = torch.mm(
        model_output_norm, model_output_norm.t()) / temperature
    label_mask = (node_labels.unsqueeze(0) == node_labels.unsqueeze(1)).float()
    n_nodes = model_output.size(0)

    label_mask = label_mask - torch.eye(n_nodes, device=model_output.device)
    pos_mask = label_mask
    neg_mask = (1 - label_mask) - \
        torch.eye(n_nodes, device=model_output.device)

    pos_loss = -torch.log(torch.exp(similarity_matrix) * pos_mask + 1e-8).sum()
    neg_loss = torch.log(torch.exp(similarity_matrix) * neg_mask + 1e-8).sum()

    n_pos, n_neg = pos_mask.sum(), neg_mask.sum()
    pos_loss = pos_loss / n_pos if n_pos > 0 else 0
    neg_loss = neg_loss / n_neg if n_neg > 0 else 0

    return pos_loss + neg_loss


def triplet_clustering_loss(model_output, node_labels, margin=1.0):
    """Triplet loss for clustering"""
    embeddings = F.normalize(model_output, p=2, dim=1)
    total_loss, n_triplets = 0, 0

    for label in torch.unique(node_labels):
        label_mask = (node_labels == label)
        label_nodes = torch.where(label_mask)[0]
        if len(label_nodes) < 2:
            continue

        other_nodes = torch.where(~label_mask)[0]
        if len(other_nodes) == 0:
            continue

        for anchor_idx in label_nodes:
            anchor = embeddings[anchor_idx]
            pos_candidates = label_nodes[label_nodes != anchor_idx]
            if len(pos_candidates) == 0:
                continue

            pos_idx = pos_candidates[torch.randint(len(pos_candidates), (1,))]
            neg_idx = other_nodes[torch.randint(len(other_nodes), (1,))]

            pos_dist = torch.dist(anchor, embeddings[pos_idx], p=2)
            neg_dist = torch.dist(anchor, embeddings[neg_idx], p=2)

            total_loss += F.relu(pos_dist - neg_dist + margin)
            n_triplets += 1

    return total_loss / max(n_triplets, 1)


def clustering_loss_function(model_output, node_labels):
    """Basic intra-cluster loss"""
    total_loss = 0.0
    for cluster_id in torch.unique(node_labels):
        mask = (node_labels == cluster_id)
        cluster_nodes = model_output[mask]
        if cluster_nodes.size(0) == 0:
            continue
        centroid = cluster_nodes.mean(dim=0)
        errors = torch.norm(cluster_nodes - centroid, dim=1)
        total_loss += errors.sum() / cluster_nodes.size(0)
    return total_loss / len(torch.unique(node_labels))


def combined_clustering_loss(model_output, node_labels, alpha=0.5, beta=0.3):
    """Combined loss function"""
    intra_loss = clustering_loss_function(model_output, node_labels)
    contrastive_loss = contrastive_clustering_loss(model_output, node_labels)
    triplet_loss = triplet_clustering_loss(model_output, node_labels)

    total_loss = alpha * intra_loss + beta * \
        contrastive_loss + (1 - alpha - beta) * triplet_loss

    return total_loss, {
        'intra_loss': intra_loss.item(),
        'contrastive_loss': contrastive_loss.item(),
        'triplet_loss': triplet_loss.item(),
        'total_loss': total_loss.item()
    }


def enhanced_clustering_evaluation(model_output, node_labels, return_details=True):
    """Comprehensive clustering evaluation"""
    embeddings = model_output.detach().cpu().numpy()
    true_labels = node_labels.cpu().numpy()
    n_clusters = len(np.unique(true_labels))

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    pred_labels = kmeans.fit_predict(embeddings)

    silhouette_avg = silhouette_score(embeddings, pred_labels) if len(
        np.unique(pred_labels)) > 1 else -1
    mapping = {
        pred_cluster: np.bincount(
            true_labels[pred_labels == pred_cluster]).argmax()
        if np.sum(pred_labels == pred_cluster) > 0 else 0
        for pred_cluster in range(n_clusters)
    }

    mapped_preds = np.array([mapping[p] for p in pred_labels])
    cm = confusion_matrix(true_labels, mapped_preds)
    metrics = {
        'confusion_matrix': cm,
        'accuracy': accuracy_score(true_labels, mapped_preds),
        'adjusted_rand_index': adjusted_rand_score(true_labels, mapped_preds),
        'normalized_mutual_info': normalized_mutual_info_score(true_labels, mapped_preds),
        'silhouette_score': silhouette_avg,
        'inertia': kmeans.inertia_,
        'per_class_accuracy': cm.diagonal() / cm.sum(axis=1)
    }

    return metrics if return_details else (cm, metrics['accuracy'], metrics['per_class_accuracy'])


def create_confusion_matrix_plot(cm, dataset_name):
    """Create and save confusion matrix plot"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {dataset_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plot_path = f"confusion_matrix_{dataset_name}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    return plot_path


def train_and_evaluate(dataset_name, model, epochs=100, lr=0.001, loss_type='combined'):
    """Combined training and evaluation with comprehensive monitoring"""
    print(f"Training on {dataset_name} with {loss_type} loss")
    mlflow.set_experiment("hypergraph_clustering")

    with mlflow.start_run(run_name=f"hypergraph_clustering_{dataset_name}_{loss_type}"):
        hyperedge_index, labels, num_nodes = load_hypergraph(dataset_name)
        model.initialize_for_dataset(num_nodes)

        optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=30, gamma=0.5)

        mlflow.log_params({
            "dataset": dataset_name,
            "epochs": epochs,
            "learning_rate": lr,
            "loss_type": loss_type,
            "model_type": "HypergraphGRANDWithEmbeddings",
            "hidden_dim": model.hidden_dim,
            "num_layers": model.num_layers,
            "alpha": model.alpha,
            "dropout": model.dropout
        })

        train_metrics = []
        for epoch in range(epochs):
            model.train()
            out = model(num_nodes, hyperedge_index)

            if loss_type == 'combined':
                loss, components = combined_clustering_loss(out, labels)
            elif loss_type == 'contrastive':
                loss = contrastive_clustering_loss(out, labels)
                components = {'contrastive_loss': loss.item()}
            elif loss_type == 'triplet':
                loss = triplet_clustering_loss(out, labels)
                components = {'triplet_loss': loss.item()}
            else:
                loss = clustering_loss_function(out, labels)
                components = {'intra_loss': loss.item()}

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            model.eval()
            with torch.no_grad():
                out_eval = model(num_nodes, hyperedge_index)
                metrics = enhanced_clustering_evaluation(out_eval, labels)

            train_metrics.append(metrics)

            if epoch % 5 == 0:
                log_metrics = {
                    "train_loss": loss.item(),
                    "train_accuracy": metrics['accuracy'],
                    "adjusted_rand_index": metrics['adjusted_rand_index'],
                    "normalized_mutual_info": metrics['normalized_mutual_info'],
                    "silhouette_score": metrics['silhouette_score'],
                    "learning_rate": optimizer.param_groups[0]['lr']
                }
                log_metrics.update(components)
                mlflow.log_metrics(log_metrics, step=epoch)

                print(f"[{dataset_name}] Epoch {epoch}: Loss={loss.item():.4f}, "
                      f"Acc={metrics['accuracy']:.4f}, ARI={metrics['adjusted_rand_index']:.4f}")

        # Create comprehensive training plots
        plt.figure(figsize=(15, 8))
        metrics_to_plot = [
            ('train_loss', train_metrics, lambda x: [m['total_loss'] if 'total_loss' in m else m[list(
                m.keys())[0]] for m in train_metrics], 'Training Loss', 'Loss'),
            ('accuracy', train_metrics, lambda x: [
             m['accuracy'] for m in x], 'Training Accuracy', 'Accuracy'),
            ('ari', train_metrics, lambda x: [
             m['adjusted_rand_index'] for m in x], 'Adjusted Rand Index', 'ARI'),
            ('nmi', train_metrics, lambda x: [
             m['normalized_mutual_info'] for m in x], 'Normalized Mutual Info', 'NMI'),
            ('silhouette', train_metrics, lambda x: [
             m['silhouette_score'] for m in x], 'Silhouette Score', 'Score'),
            ('inertia', train_metrics, lambda x: [
             m['inertia'] for m in x], 'K-means Inertia', 'Inertia')
        ]

        for i, (name, data, fn, title, ylabel) in enumerate(metrics_to_plot, 1):
            plt.subplot(2, 3, i)
            plt.plot(fn(data))
            plt.title(title)
            plt.xlabel('Epoch')
            plt.ylabel(ylabel)

        plot_path = f"training_metrics_{dataset_name}_{loss_type}.png"
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        mlflow.log_artifact(plot_path)

        # Log confusion matrix
        cm_plot_path = create_confusion_matrix_plot(
            metrics['confusion_matrix'], dataset_name)
        mlflow.log_artifact(cm_plot_path)

        # Log final metrics
        mlflow.log_metrics({
            f"{dataset_name}_test_accuracy": metrics['accuracy'],
            f"{dataset_name}_test_error": 1 - metrics['accuracy'],
            f"{dataset_name}_avg_per_class_accuracy": np.mean(metrics['per_class_accuracy'])
        })

        mlflow.set_tags({
            "experiment_type": "single_run",
            "dataset": dataset_name,
            "model_architecture": "HypergraphGRANDWithEmbeddings",
            "status": "completed"
        })

        return model, metrics


def main():
    """Main execution function"""
    try:
        model = HypergraphGRANDWithEmbeddings(
            hidden_dim=64, num_layers=10, alpha=0.01, dropout=0.1)
        train_dataset = "contact-high-school"
        test_dataset = "contact-primary-school"

        trained_model, train_metrics = train_and_evaluate(
            train_dataset, model, loss_type='combined')

        # Evaluate on test dataset
        hyperedge_index, labels, num_nodes = load_hypergraph(test_dataset)
        trained_model.eval()
        with torch.no_grad():
            out = trained_model(num_nodes, hyperedge_index)
            test_metrics = enhanced_clustering_evaluation(out, labels)

        print("\n" + "="*50)
        print(f"Final Training Results ({train_dataset}):")
        print(f"Accuracy: {train_metrics['accuracy']:.4f}")
        print(
            f"Average Per-class Accuracy: {np.mean(train_metrics['per_class_accuracy']):.4f}")
        print(f"\nFinal Test Results ({test_dataset}):")
        print(f"Accuracy: {test_metrics['accuracy']:.4f}")
        print(
            f"Average Per-class Accuracy: {np.mean(test_metrics['per_class_accuracy']):.4f}")
        print("="*50)

    except Exception as e:
        print(f"Error during experiment: {e}")
        mlflow.set_tag("status", "failed")
        mlflow.log_param("error_message", str(e))
        raise


if __name__ == "__main__":
    main()
