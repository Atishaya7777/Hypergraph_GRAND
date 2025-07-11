import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, accuracy_score, adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.cluster import KMeans
import numpy as np
import os
import mlflow
import mlflow.pytorch
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import linear_sum_assignment
from model import HypergraphGRAND


def setup_mlflow():
    """Setup MLflow tracking"""
    mlflow.set_experiment("hypergraph_clustering")
    return mlflow


def load_hypergraph(dataset_name, base_path="./datasets/", max_input_dim=None):
    """Load hypergraph dataset with enhanced logging and valid node tracking"""
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
    # Create valid_mask for labels (length num_nodes)
    valid_mask = torch.arange(num_nodes) < len(
        labels_list)  # True for valid nodes
    # Create input_mask for padded input/output (length max_input_dim)
    input_mask = torch.zeros(
        max_input_dim if max_input_dim is not None else num_nodes, dtype=torch.bool)
    input_mask[:num_nodes] = valid_mask  # Match valid nodes up to num_nodes

    dataset_info = {
        f"{dataset_name}_num_nodes": num_nodes,
        f"{dataset_name}_num_edges": len(hyperedges),
        f"{dataset_name}_num_classes": len(torch.unique(labels[valid_mask])),
        f"{dataset_name}_avg_edge_size": np.mean([len(edge) for edge in hyperedges])
    }
    mlflow.log_params(dataset_info)

    return hyperedge_index, labels, num_nodes, valid_mask, input_mask


def get_dataset_info(dataset_names, base_path="./datasets/"):
    """Get information about all datasets to determine max dimensions"""
    max_nodes = 0
    dataset_infos = {}
    for dataset_name in dataset_names:
        _, _, num_nodes, _, _ = load_hypergraph(dataset_name, base_path)
        dataset_infos[dataset_name] = num_nodes
        max_nodes = max(max_nodes, num_nodes)
    return max_nodes, dataset_infos


def prepare_input_features(num_nodes, target_dim):
    """Prepare input features with consistent dimensionality"""
    if num_nodes <= target_dim:
        x = torch.eye(target_dim)
        x = x[:num_nodes, :]
        if num_nodes < target_dim:
            padding = torch.zeros(target_dim - num_nodes, target_dim)
            x = torch.cat([x, padding], dim=0)
    else:
        x = torch.eye(target_dim)
    return x


def clustering_loss_function(model_output, node_labels, input_mask=None, valid_mask=None):
    """Intra-cluster loss with masking for padded nodes"""
    unique_clusters = torch.unique(node_labels[valid_mask])
    total_loss = 0.0
    for cluster_id in unique_clusters:
        if cluster_id == 0:  # Skip padded label
            continue
        cluster_mask = (node_labels == cluster_id) & valid_mask
        cluster_indices = torch.where(cluster_mask)[0]
        cluster_nodes = model_output[input_mask][cluster_indices]
        if cluster_nodes.size(0) == 0:
            continue
        centroid = cluster_nodes.mean(dim=0)
        errors = torch.norm(cluster_nodes - centroid, dim=1)
        total_loss += errors.sum() / cluster_nodes.size(0)
    return total_loss / max(1, len(unique_clusters) - 1)


def clustering_error_function(model_output, node_labels, input_mask=None, valid_mask=None):
    """Enhanced clustering evaluation with Hungarian algorithm and additional metrics"""
    valid_labels = node_labels[valid_mask]
    valid_output = model_output[input_mask][valid_mask]
    # Exclude label 0
    n_clusters = len(torch.unique(valid_labels[valid_labels != 0]))
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=20)
    preds = kmeans.fit_predict(valid_output.detach().cpu().numpy())
    true_labels = valid_labels.cpu().numpy()

    # Use Hungarian algorithm for optimal label mapping
    cm = confusion_matrix(true_labels, preds, labels=np.arange(n_clusters))
    row_ind, col_ind = linear_sum_assignment(-cm)
    mapped_preds = np.zeros_like(preds)
    for pred_cluster, true_cluster in zip(row_ind, col_ind):
        mapped_preds[preds == pred_cluster] = true_cluster

    cm = confusion_matrix(true_labels, mapped_preds,
                          labels=np.arange(n_clusters))
    acc = accuracy_score(true_labels, mapped_preds)
    per_class_acc = cm.diagonal() / (cm.sum(axis=1) + 1e-10)
    metrics = {
        'confusion_matrix': cm,
        'accuracy': acc,
        'per_class_accuracy': per_class_acc,
        'adjusted_rand_index': adjusted_rand_score(true_labels, mapped_preds),
        'normalized_mutual_info': normalized_mutual_info_score(true_labels, mapped_preds),
        'silhouette_score': silhouette_score(valid_output.detach().cpu().numpy(), mapped_preds) if len(np.unique(mapped_preds)) > 1 else -1
    }
    return metrics


def create_enhanced_confusion_matrix_plot(cm, dataset_name, metrics):
    """Create enhanced confusion matrix plot"""
    plt.figure(figsize=(10, 8))
    cmap = plt.cm.Blues
    ax = sns.heatmap(
        cm, annot=True, fmt='d', cmap=cmap, square=True, linewidths=0.5,
        cbar_kws={"shrink": .8}, annot_kws={"size": 12, "weight": "bold"}
    )
    plt.title(f'Confusion Matrix - {dataset_name}\nAccuracy: {metrics["accuracy"]:.3f}, ARI: {
              metrics["adjusted_rand_index"]:.3f}', fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('True Label', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=14, fontweight='bold')
    num_classes = len(metrics['per_class_accuracy'])
    for i in range(num_classes):
        plt.text(num_classes + 0.5, i + 0.5, f'Acc: {
                 metrics["per_class_accuracy"][i]:.3f}', ha='center', va='center', fontsize=10)
    plt.tight_layout()
    plot_path = f"confusion_matrix_{dataset_name}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    return plot_path


def log_confusion_matrix_to_mlflow(metrics, dataset_name):
    """Log confusion matrix and metrics to MLflow"""
    plot_path = create_enhanced_confusion_matrix_plot(
        metrics['confusion_matrix'], dataset_name, metrics)
    mlflow.log_artifact(plot_path, artifact_path="confusion_matrices")
    mlflow.log_metrics({
        f"{dataset_name}_accuracy": metrics['accuracy'],
        f"{dataset_name}_adjusted_rand_index": metrics['adjusted_rand_index'],
        f"{dataset_name}_normalized_mutual_info": metrics['normalized_mutual_info'],
        f"{dataset_name}_silhouette_score": metrics['silhouette_score']
    })
    for i, class_acc in enumerate(metrics['per_class_accuracy']):
        mlflow.log_metric(f"{dataset_name}_class_{i}_accuracy", class_acc)
    if os.path.exists(plot_path):
        os.remove(plot_path)
    print(f"Confusion matrix for {dataset_name} logged to MLflow")


def visualize_embeddings(model_output, node_labels, dataset_name, epoch=None, input_mask=None, valid_mask=None):
    """Visualize embeddings using t-SNE"""
    valid_output = model_output[input_mask][valid_mask]
    valid_labels = node_labels[valid_mask]
    from sklearn.manifold import TSNE
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(valid_output.detach().cpu().numpy())
    plt.figure(figsize=(8, 6))
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                c=valid_labels.cpu().numpy(), cmap='tab10')
    title = f'Embeddings - {dataset_name}' + \
        (f' (Epoch {epoch})' if epoch is not None else '')
    plt.title(title)
    plot_path = f"embeddings_{dataset_name}{
        '_epoch' + str(epoch) if epoch is not None else ''}.png"
    plt.savefig(plot_path)
    mlflow.log_artifact(plot_path)
    plt.close()


def train_on_dataset(datasets, model, max_input_dim, epochs=100, lr=0.01):
    """Train on multiple datasets with regularization and visualization"""
    print(f"Starting training on {datasets}")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=30, gamma=0.5)
    train_losses = []
    train_accuracies = {}

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for dataset_name in datasets:
            hyperedge_index, labels, num_nodes, valid_mask, input_mask = load_hypergraph(
                dataset_name, max_input_dim=max_input_dim)
            x = prepare_input_features(num_nodes, max_input_dim)
            out = model(x, hyperedge_index)
            loss = clustering_loss_function(
                out, labels, input_mask, valid_mask)
            total_loss += loss

            # Evaluate training metrics
            model.eval()
            with torch.no_grad():
                out_eval = model(x, hyperedge_index)
                metrics = clustering_error_function(
                    out_eval, labels, input_mask, valid_mask)
                train_accuracies.setdefault(
                    dataset_name, []).append(metrics['accuracy'])
                if epoch % 10 == 0:
                    visualize_embeddings(
                        out_eval, labels, dataset_name, epoch, input_mask, valid_mask)

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        scheduler.step()
        train_losses.append(total_loss.item())

        if epoch % 5 == 0:
            mlflow.log_metrics({
                "train_loss": total_loss.item(),
                **{f"{dataset_name}_train_accuracy": train_accuracies[dataset_name][-1] for dataset_name in datasets}
            }, step=epoch)
            print(f"[Epoch {epoch}] Loss: {total_loss.item():.4f}, " +
                  ", ".join(f"{dataset_name} Accuracy: {train_accuracies[dataset_name][-1]:.4f}" for dataset_name in datasets))

    # Log final training metrics and confusion matrices
    for dataset_name in datasets:
        hyperedge_index, labels, num_nodes, valid_mask, input_mask = load_hypergraph(
            dataset_name, max_input_dim=max_input_dim)
        x = prepare_input_features(num_nodes, max_input_dim)
        model.eval()
        with torch.no_grad():
            out_eval = model(x, hyperedge_index)
            metrics = clustering_error_function(
                out_eval, labels, input_mask, valid_mask)
            log_confusion_matrix_to_mlflow(metrics, f"{dataset_name}_training")
            mlflow.log_metrics({
                f"{dataset_name}_final_train_accuracy": metrics['accuracy'],
                f"{dataset_name}_final_train_ari": metrics['adjusted_rand_index'],
                f"{dataset_name}_final_train_nmi": metrics['normalized_mutual_info'],
                f"{dataset_name}_final_train_silhouette": metrics['silhouette_score']
            })

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.subplot(1, 2, 2)
    for dataset_name in datasets:
        plt.plot(train_accuracies[dataset_name], label=dataset_name)
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    training_plot_path = f"training_history.png"
    plt.savefig(training_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    mlflow.log_artifact(training_plot_path)
    return model


def evaluate_on_dataset(dataset_name, model, max_input_dim):
    """Evaluate with enhanced metrics and visualization"""
    print(f"Starting evaluation on {dataset_name}")
    hyperedge_index, labels, num_nodes, valid_mask, input_mask = load_hypergraph(
        dataset_name, max_input_dim=max_input_dim)
    x = prepare_input_features(num_nodes, max_input_dim)
    model.eval()
    with torch.no_grad():
        out = model(x, hyperedge_index)
        loss = clustering_loss_function(out, labels, input_mask, valid_mask)
        metrics = clustering_error_function(
            out, labels, input_mask, valid_mask)

    mlflow.log_metrics({
        f"{dataset_name}_test_loss": loss.item(),
        **{f"{dataset_name}_{k}": v for k, v in metrics.items() if k != 'confusion_matrix' and k != 'per_class_accuracy'}
    })
    log_confusion_matrix_to_mlflow(metrics, dataset_name)
    visualize_embeddings(out, labels, dataset_name,
                         input_mask=input_mask, valid_mask=valid_mask)
    print(f"\nEvaluation on {dataset_name}:")
    print("Confusion Matrix:\n", metrics['confusion_matrix'])
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Per-class accuracies: {metrics['per_class_accuracy']}")
    print(f"ARI: {metrics['adjusted_rand_index']:.4f}, NMI: {
          metrics['normalized_mutual_info']:.4f}, Silhouette: {metrics['silhouette_score']:.4f}")
    return metrics


def main():
    """Main pipeline for Identity Matrix approach"""
    mlflow_manager = setup_mlflow()
    approach_name = "identity_matrix"
    run_name = f"hypergraph_clustering_{approach_name}"

    with mlflow_manager.start_run(run_name=run_name):
        datasets = ["contact-high-school", "contact-primary-school"]
        hidden_dim = 32
        max_nodes, dataset_infos = get_dataset_info(datasets)
        print(f"Dataset information:")
        for dataset, nodes in dataset_infos.items():
            print(f"  {dataset}: {nodes} nodes")
        print(f"Using max input dimension: {max_nodes}")

        model = HypergraphGRAND(input_dim=max_nodes,
                                hidden_dim=hidden_dim, dropout=0.1)
        model_params = {
            "model_type": "HypergraphGRAND",
            "input_dim": max_nodes,
            "hidden_dim": hidden_dim,
            "dropout": 0.1,
            "total_parameters": sum(p.numel() for p in model.parameters()),
            "approach": "identity_matrix"
        }
        mlflow.log_params(model_params)

        try:
            trained_model = train_on_dataset(
                datasets,
                model,
                max_input_dim=max_nodes,
                epochs=100,
                lr=0.01
            )

            for dataset_name in datasets:
                metrics = evaluate_on_dataset(
                    dataset_name, trained_model, max_nodes)

            mlflow.pytorch.log_model(trained_model, "model")
            print("\n" + "="*50)
            print("EXPERIMENT COMPLETED SUCCESSFULLY")
            print("="*50)
            print(f"Approach: {approach_name}")
            print(
                f"Final test accuracy (contact-primary-school): {metrics['accuracy']:.4f}")
            print(f"Average per-class accuracy (contact-primary-school): {
                  np.mean(metrics['per_class_accuracy']):.4f}")
            print("="*50)

            mlflow.set_tags({
                "experiment_type": "multi_dataset",
                "train_datasets": ",".join(datasets),
                "test_dataset": "contact-primary-school",
                "model_architecture": model_params["model_type"],
                "approach": approach_name,
                "status": "completed"
            })

        except Exception as e:
            print(f"Error during experiment: {e}")
            mlflow.set_tag("status", "failed")
            mlflow.log_param("error_message", str(e))
            raise

    print("\nCheck MLflow UI for detailed results and visualizations.")


if __name__ == "__main__":
    print("Running with identity matrix approach")
    main()
