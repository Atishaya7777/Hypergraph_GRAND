import torch
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.cluster import KMeans
import numpy as np
import os
import mlflow
import mlflow.pytorch
import matplotlib.pyplot as plt
import seaborn as sns
from model import HypergraphGRAND


def setup_mlflow():
    """Setup MLflow tracking"""
    mlflow.set_experiment("hypergraph_clustering")
    return mlflow


def load_hypergraph(dataset_name, base_path="./datasets/"):
    """Load hypergraph dataset with enhanced logging"""
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

    # Pad labels if needed
    if len(labels_list) < num_nodes:
        print(f"Warning: Padding labels from {
              len(labels_list)} to {num_nodes}")
        labels_list.extend([0] * (num_nodes - len(labels_list)))

    labels = torch.tensor(labels_list, dtype=torch.long)
    hyperedge_index = torch.tensor(edge_idx, dtype=torch.long).T

    dataset_info = {
        f"{dataset_name}_num_nodes": num_nodes,
        f"{dataset_name}_num_edges": len(hyperedges),
        f"{dataset_name}_num_classes": len(torch.unique(labels)),
        f"{dataset_name}_avg_edge_size": np.mean([len(edge) for edge in hyperedges])
    }
    mlflow.log_params(dataset_info)

    return hyperedge_index, labels, num_nodes


def get_dataset_info(dataset_names, base_path="./datasets/"):
    """Get information about all datasets to determine max dimensions"""
    max_nodes = 0
    dataset_infos = {}

    for dataset_name in dataset_names:
        _, _, num_nodes = load_hypergraph(dataset_name, base_path)
        dataset_infos[dataset_name] = num_nodes
        max_nodes = max(max_nodes, num_nodes)

    return max_nodes, dataset_infos


def prepare_input_features(num_nodes, target_dim):
    """Prepare input features with consistent dimensionality"""
    if num_nodes <= target_dim:
        # Pad with zeros if needed
        x = torch.eye(target_dim)
        x = x[:num_nodes, :]  # Take only the rows we need
        # Pad with zeros for remaining nodes
        if num_nodes < target_dim:
            padding = torch.zeros(target_dim - num_nodes, target_dim)
            x = torch.cat([x, padding], dim=0)
    else:
        # Truncate if larger (shouldn't happen with our max_nodes approach)
        x = torch.eye(target_dim)

    return x


def clustering_loss_function(model_output, node_labels):
    unique_clusters = torch.unique(node_labels)
    total_loss = 0.0

    for cluster_id in unique_clusters:
        mask = (node_labels == cluster_id)
        cluster_nodes = model_output[mask]

        if cluster_nodes.size(0) == 0:
            continue

        centroid = cluster_nodes.mean(dim=0)
        errors = torch.norm(cluster_nodes - centroid, dim=1)
        total_loss += errors.sum() / cluster_nodes.size(0)

    return total_loss / len(unique_clusters)


def clustering_error_function(model_output, node_labels):
    n_clusters = len(torch.unique(node_labels))
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
    preds = kmeans.fit_predict(model_output.detach().cpu().numpy())
    true_labels = node_labels.cpu().numpy()

    # Create mapping from predicted to true clusters
    mapping = {}
    for pred_cluster in range(n_clusters):
        mask = (preds == pred_cluster)
        if np.sum(mask) > 0:
            mapping[pred_cluster] = np.bincount(true_labels[mask]).argmax()
        else:
            mapping[pred_cluster] = 0

    mapped_preds = np.array([mapping[p] for p in preds])
    cm = confusion_matrix(true_labels, mapped_preds)
    acc = accuracy_score(true_labels, mapped_preds)

    # Calculate per-class accuracy
    per_class_acc = cm.diagonal() / cm.sum(axis=1)

    return cm, acc, per_class_acc


def create_enhanced_confusion_matrix_plot(cm, dataset_name, accuracy, per_class_acc):
    """
    Create an enhanced confusion matrix plot with better styling and information
    """
    # Set up the plot with a larger figure size
    plt.figure(figsize=(10, 8))

    # Create a more sophisticated color scheme
    # Use a blue-white-red colormap for better contrast
    cmap = plt.cm.Blues

    # Create the heatmap with enhanced styling
    ax = sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap=cmap,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": .8},
        annot_kws={"size": 12, "weight": "bold"}
    )

    # Customize the plot
    plt.title(f'Confusion Matrix - {dataset_name}\nOverall Accuracy: {accuracy:.3f}',
              fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('True Label', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=14, fontweight='bold')

    # Add per-class accuracy as text annotations
    num_classes = len(per_class_acc)
    for i in range(num_classes):
        plt.text(num_classes + 0.5, i + 0.5, f'Acc: {per_class_acc[i]:.3f}',
                 ha='center', va='center', fontsize=10, fontweight='bold')

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    # Save with high quality
    plot_path = f"confusion_matrix_{dataset_name}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    return plot_path


def log_confusion_matrix_to_mlflow(cm, dataset_name, accuracy, per_class_acc):
    """
    Log confusion matrix to MLflow with enhanced visualization and metrics
    """
    # Create the enhanced plot
    plot_path = create_enhanced_confusion_matrix_plot(
        cm, dataset_name, accuracy, per_class_acc)

    # Log the plot as an artifact
    mlflow.log_artifact(plot_path, artifact_path="confusion_matrices")

    # Log confusion matrix as a figure (for better MLflow UI integration)
    mlflow.log_figure(plt.gcf(), f"confusion_matrix_{dataset_name}_figure.png")

    # Log detailed confusion matrix metrics
    num_classes = cm.shape[0]

    # # Log the raw confusion matrix values
    # for i in range(num_classes):
    #     for j in range(num_classes):
    #         mlflow.log_metric(f"{dataset_name}_cm_{i}_{j}", cm[i, j])

    # Log precision, recall, and F1 score for each class
    for i in range(num_classes):
        # Precision = TP / (TP + FP)
        precision = cm[i, i] / (cm[:, i].sum() + 1e-10)
        # Recall = TP / (TP + FN)
        recall = cm[i, i] / (cm[i, :].sum() + 1e-10)
        # F1 score
        f1 = 2 * (precision * recall) / (precision + recall + 1e-10)

        mlflow.log_metric(f"{dataset_name}_precision_class_{i}", precision)
        mlflow.log_metric(f"{dataset_name}_recall_class_{i}", recall)
        mlflow.log_metric(f"{dataset_name}_f1_class_{i}", f1)

    # Log macro averages
    precisions = [cm[i, i] / (cm[:, i].sum() + 1e-10)
                  for i in range(num_classes)]
    recalls = [cm[i, i] / (cm[i, :].sum() + 1e-10) for i in range(num_classes)]
    f1_scores = [2 * (p * r) / (p + r + 1e-10)
                 for p, r in zip(precisions, recalls)]

    mlflow.log_metric(
        f"{dataset_name}_macro_avg_precision", np.mean(precisions))
    mlflow.log_metric(f"{dataset_name}_macro_avg_recall", np.mean(recalls))
    mlflow.log_metric(f"{dataset_name}_macro_avg_f1", np.mean(f1_scores))

    # Clean up the temporary file
    if os.path.exists(plot_path):
        os.remove(plot_path)

    print(f"Confusion matrix for {
          dataset_name} logged to MLflow with enhanced visualization")


def train_on_dataset(dataset_name, model, max_input_dim=None, epochs=100, lr=0.01):
    print(f"Starting training on {dataset_name}")
    hyperedge_index, labels, num_nodes = load_hypergraph(dataset_name)
    x = prepare_input_features(num_nodes, max_input_dim)

    if num_nodes < max_input_dim:
        labels = torch.cat([labels, torch.zeros(
            max_input_dim - num_nodes, dtype=torch.long)])

    training_params = {
        "epochs": epochs,
        "learning_rate": lr,
        "optimizer": "Adam",
        "train_dataset": dataset_name,
        "input_dim": max_input_dim,
        "actual_nodes": num_nodes,
        "hidden_dim": model.hidden_dim if hasattr(model, 'hidden_dim') else 'unknown',
        "approach": "identity_matrix"
    }
    mlflow.log_params(training_params)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        out = model(x, hyperedge_index)
        out_relevant = out[:num_nodes]
        labels_relevant = labels[:num_nodes]
        loss = clustering_loss_function(out_relevant, labels_relevant)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            out_eval = model(x, hyperedge_index)
            out_eval_relevant = out_eval[:num_nodes]
            cm_train, train_acc, per_class_acc_train = clustering_error_function(
                out_eval_relevant, labels_relevant)

        if epoch % 5 == 0:
            mlflow.log_metrics(
                {"train_loss": loss.item(), "train_accuracy": train_acc}, step=epoch)
            print(f"[{dataset_name}] Epoch {epoch}: Loss = {
                  loss.item():.4f}, Accuracy = {train_acc:.4f}")

    log_confusion_matrix_to_mlflow(
        cm_train, f"{dataset_name}_training", train_acc, per_class_acc_train)

    mlflow.log_metrics({
        "final_train_loss": loss.item(),
        "final_train_accuracy": train_acc
    })

    return model


def evaluate_on_dataset(dataset_name, model, max_input_dim=None):
    print(f"Starting evaluation on {dataset_name}")
    hyperedge_index, labels, num_nodes = load_hypergraph(dataset_name)
    x = prepare_input_features(num_nodes, max_input_dim)

    model.eval()
    with torch.no_grad():
        out = model(x, hyperedge_index)
        out_relevant = out[:num_nodes]
        labels_relevant = labels[:num_nodes]
        eval_loss = clustering_loss_function(out_relevant, labels_relevant)

    cm, acc, per_class_acc = clustering_error_function(
        out_relevant, labels_relevant)

    mlflow.log_metrics({
        f"{dataset_name}_test_loss": eval_loss.item(),
        f"{dataset_name}_test_accuracy": acc,
        f"{dataset_name}_test_error": 1 - acc,
        f"{dataset_name}_avg_per_class_accuracy": np.mean(per_class_acc)
    })

    for i, class_acc in enumerate(per_class_acc):
        mlflow.log_metric(f"{dataset_name}_class_{i}_accuracy", class_acc)

    log_confusion_matrix_to_mlflow(cm, dataset_name, acc, per_class_acc)

    print(f"\nEvaluation on {dataset_name}:")
    print("Confusion Matrix:")
    print(cm)
    print(f"Accuracy: {acc:.4f}")
    print(f"Per-class accuracies: {per_class_acc}")

    return cm, acc, per_class_acc


def main():
    mlflow_manager = setup_mlflow()
    run_name = "hypergraph_clustering_identity_matrix"

    with mlflow_manager.start_run(run_name=run_name):
        datasets = ["contact-primary-school", "contact-high-school"]
        hidden_dim = 4
        max_nodes, dataset_infos = get_dataset_info(datasets)

        print(f"Dataset information:")
        for dataset, nodes in dataset_infos.items():
            print(f"  {dataset}: {nodes} nodes")
        print(f"Using max input dimension: {max_nodes}")

        model = HypergraphGRAND(input_dim=max_nodes, hidden_dim=hidden_dim)

        mlflow.log_params({
            "model_type": "HypergraphGRAND",
            "input_dim": max_nodes,
            "hidden_dim": hidden_dim,
            "total_parameters": sum(p.numel() for p in model.parameters()),
            "approach": "identity_matrix"
        })

        try:
            trained_model = train_on_dataset(
                "contact-high-school", model, max_input_dim=max_nodes, epochs=100, lr=0.001)
            cm, acc, per_class_acc = evaluate_on_dataset(
                "contact-primary-school", trained_model, max_input_dim=max_nodes)
            mlflow.pytorch.log_model(trained_model, "model")

            mlflow.set_tags({
                "experiment_type": "single_run",
                "train_dataset": "contact-high-school",
                "test_dataset": "contact-primary-school",
                "model_architecture": "HypergraphGRAND",
                "approach": "identity_matrix",
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
