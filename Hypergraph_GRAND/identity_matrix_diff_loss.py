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
import time


def setup_mlflow():
    """Setup MLflow tracking"""
    mlflow.set_experiment("hypergraph_clustering")
    return mlflow


def load_hypergraph(dataset_name, base_path="./datasets/", max_input_dim=None):
    """Load hypergraph dataset with enhanced logging and valid node tracking"""
    start_time = time.time()
    print(f"[DEBUG] Loading dataset: {dataset_name}")

    label_file = os.path.join(
        base_path, f"{dataset_name}/node-labels-{dataset_name}.txt")
    edge_file = os.path.join(
        base_path, f"{dataset_name}/hyperedges-{dataset_name}.txt")

    with open(label_file) as f:
        labels_list = [int(line.strip()) for line in f]
    print(f"[DEBUG] Loaded {len(labels_list)} labels from {label_file}")

    with open(edge_file) as f:
        hyperedges = [list(map(int, line.strip().split(','))) for line in f]
    print(f"[DEBUG] Loaded {len(hyperedges)} hyperedges from {edge_file}")

    edge_idx = []
    max_node = 0
    for e_id, edge in enumerate(hyperedges):
        for node in edge:
            if node < 0 or (max_input_dim is not None and node >= max_input_dim):
                print(f"[WARNING] Invalid node index {
                      node} in hyperedge {e_id} for {dataset_name}")
                continue
            max_node = max(max_node, node)
            edge_idx.append((e_id, node))
    print(f"[DEBUG] Processed {
          len(edge_idx)} node-hyperedge pairs, max node index: {max_node}")

    num_nodes = max(max_node + 1, len(labels_list))
    if len(labels_list) < num_nodes:
        print(f"Warning: Padding labels from {
              len(labels_list)} to {num_nodes}")
        labels_list.extend([0] * (num_nodes - len(labels_list)))

    labels = torch.tensor(labels_list, dtype=torch.long)
    hyperedge_index = torch.tensor(edge_idx, dtype=torch.long).T
    valid_mask = torch.arange(num_nodes) < len(
        labels_list)  # Shape: [num_nodes]
    input_mask = torch.zeros(
        max_input_dim if max_input_dim is not None else num_nodes, dtype=torch.bool)
    input_mask[:num_nodes] = valid_mask

    dataset_info = {
        f"{dataset_name}_num_nodes": num_nodes,
        f"{dataset_name}_num_edges": len(hyperedges),
        f"{dataset_name}_num_classes": len(torch.unique(labels[valid_mask])),
        f"{dataset_name}_avg_edge_size": np.mean([len(edge) for edge in hyperedges]) if hyperedges else 0
    }
    mlflow.log_params(dataset_info)
    print(f"[DEBUG] Dataset info: {dataset_info}")
    print(f"[DEBUG] load_hypergraph took {
          time.time() - start_time:.2f} seconds")

    return hyperedge_index, labels, num_nodes, valid_mask, input_mask


def get_dataset_info(dataset_names, base_path="./datasets/"):
    """Get information about all datasets to determine max dimensions"""
    max_nodes = 0
    dataset_infos = {}
    for dataset_name in dataset_names:
        _, _, num_nodes, _, _ = load_hypergraph(dataset_name, base_path)
        dataset_infos[dataset_name] = num_nodes
        max_nodes = max(max_nodes, num_nodes)
    print(f"[DEBUG] Max nodes across datasets: {max_nodes}")
    return max_nodes, dataset_infos


def prepare_input_features(num_nodes, target_dim):
    """Prepare input features with consistent dimensionality"""
    start_time = time.time()
    print(f"[DEBUG] Preparing input features: num_nodes={
          num_nodes}, target_dim={target_dim}")
    if num_nodes <= target_dim:
        x = torch.eye(target_dim)
        x = x[:num_nodes, :]
        if num_nodes < target_dim:
            padding = torch.zeros(target_dim - num_nodes, target_dim)
            x = torch.cat([x, padding], dim=0)
    else:
        x = torch.eye(target_dim)
    print(f"[DEBUG] x shape: {x.shape}, took {
          time.time() - start_time:.2f} seconds")
    return x


def structure_preserved_loss(model_output, hyperedge_index, input_mask=None, num_nodes=None):
    """Label-agnostic loss based on hyperedge structure with regularization"""
    start_time = time.time()
    if input_mask is not None:
        # Shape: [num_nodes, hidden_dim]
        model_output = model_output[input_mask]

    edge_ids = hyperedge_index[0]
    node_ids = hyperedge_index[1]
    num_edges = edge_ids.max().item() + 1

    edge_losses = []
    for e_id in range(num_edges):
        mask = (edge_ids == e_id)
        edge_nodes = node_ids[mask]
        if len(edge_nodes) < 2:  # Skip single-node hyperedges
            continue
        # Ensure node indices are valid
        edge_nodes = edge_nodes[edge_nodes < num_nodes]
        if len(edge_nodes) < 2:
            continue
        edge_embeddings = model_output[edge_nodes]
        centroid = edge_embeddings.mean(dim=0)
        distances = torch.norm(edge_embeddings - centroid, dim=1)
        edge_loss = distances.mean()
        edge_losses.append(edge_loss)

    if len(edge_losses) > 0:
        structure_loss = torch.stack(edge_losses).mean()
    else:
        structure_loss = torch.tensor(
            0., device=model_output.device, dtype=model_output.dtype, requires_grad=True)

    # Add variance regularization to prevent embedding collapse
    valid_embeddings = model_output[:num_nodes]  # Exclude padded nodes
    if valid_embeddings.size(0) > 1:
        variance = torch.var(valid_embeddings, dim=0).mean()
        reg_loss = -0.01 * variance  # Encourage higher variance
    else:
        reg_loss = torch.tensor(
            0., device=model_output.device, dtype=model_output.dtype, requires_grad=True)

    total_loss = structure_loss + reg_loss
    print(f"[DEBUG] structure_preserved_loss: structure={structure_loss.item():.4f}, reg={
          reg_loss.item():.4f}, total={total_loss.item():.4f}, took {time.time() - start_time:.2f} seconds")
    return total_loss


def clustering_error_function(model_output, node_labels, input_mask=None, valid_mask=None):
    """Enhanced clustering evaluation with Hungarian algorithm and additional metrics"""
    start_time = time.time()
    valid_labels = node_labels[valid_mask]
    valid_output = model_output[input_mask][valid_mask]
    n_clusters = len(torch.unique(valid_labels[valid_labels != 0]))
    print(f"[DEBUG] Clustering with n_clusters={n_clusters}")
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=5)
    preds = kmeans.fit_predict(valid_output.detach().cpu().numpy())
    true_labels = valid_labels.cpu().numpy()

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
    print(f"[DEBUG] clustering_error_function took {
          time.time() - start_time:.2f} seconds")
    return metrics


def create_enhanced_confusion_matrix_plot(cm, dataset_name, metrics):
    """Create enhanced confusion matrix plot"""
    start_time = time.time()
    plt.figure(figsize=(10, 8))
    cmap = plt.cm.Blues
    sns.heatmap(
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
    print(f"[DEBUG] create_enhanced_confusion_matrix_plot took {
          time.time() - start_time:.2f} seconds")
    return plot_path


def log_confusion_matrix_to_mlflow(metrics, dataset_name):
    """Log confusion matrix and metrics to MLflow"""
    start_time = time.time()
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
    print(f"[DEBUG] log_confusion_matrix_to_mlflow took {
          time.time() - start_time:.2f} seconds")


def visualize_embeddings(model_output, node_labels, dataset_name, epoch=None, input_mask=None, valid_mask=None):
    """Visualize embeddings using t-SNE"""
    start_time = time.time()
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
    print(f"[DEBUG] visualize_embeddings took {
          time.time() - start_time:.2f} seconds")


def train_on_dataset(datasets, model, max_input_dim, epochs=100, lr=0.001):
    """Train on multiple datasets with label-agnostic loss"""
    start_time = time.time()
    print(f"[DEBUG] Starting training on {datasets}")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=30, gamma=0.5)
    train_losses = []
    train_accuracies = {}

    for epoch in range(epochs):
        epoch_start_time = time.time()
        model.train()
        total_loss = 0.0
        for dataset_name in datasets:
            print(f"[DEBUG] Processing dataset: {dataset_name}")
            hyperedge_index, labels, num_nodes, valid_mask, input_mask = load_hypergraph(
                dataset_name, max_input_dim=max_input_dim)
            print(f"[DEBUG] hyperedge_index shape: {hyperedge_index.shape}, labels shape: {
                  labels.shape}, valid_mask shape: {valid_mask.shape}, input_mask shape: {input_mask.shape}")
            x = prepare_input_features(num_nodes, max_input_dim)
            print(f"[DEBUG] Input features shape: {x.shape}")
            out = model(x, hyperedge_index)
            print(f"[DEBUG] Model output shape: {out.shape}")
            loss = structure_preserved_loss(
                out, hyperedge_index, input_mask, num_nodes)
            total_loss += loss

            # Evaluate training metrics every 10 epochs
            if epoch % 10 == 0:
                model.eval()
                with torch.no_grad():
                    out_eval = model(x, hyperedge_index)
                    print(f"[DEBUG] Evaluation output shape: {out_eval.shape}")
                    metrics = clustering_error_function(
                        out_eval, labels, input_mask, valid_mask)
                    train_accuracies.setdefault(
                        dataset_name, []).append(metrics['accuracy'])
                    if epoch % 20 == 0:  # Visualize less frequently
                        visualize_embeddings(
                            out_eval, labels, dataset_name, epoch, input_mask, valid_mask)

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        scheduler.step()
        train_losses.append(total_loss.item())

        if epoch % 10 == 0:
            mlflow.log_metrics({
                "train_loss": total_loss.item(),
                **{f"{dataset_name}_train_accuracy": train_accuracies.get(dataset_name, [0])[-1] for dataset_name in datasets}
            }, step=epoch)
            print(f"[DEBUG] Epoch {epoch} took {
                  time.time() - epoch_start_time:.2f} seconds")
            print(f"[Epoch {epoch}] Loss: {total_loss.item():.4f}, " +
                  ", ".join(f"{dataset_name} Accuracy: {train_accuracies.get(dataset_name, [0])[-1]:.4f}" for dataset_name in datasets))

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
        plt.plot(train_accuracies.get(dataset_name, []), label=dataset_name)
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    training_plot_path = f"training_history.png"
    plt.savefig(training_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    mlflow.log_artifact(training_plot_path)
    print(f"[DEBUG] Total training took {
          time.time() - start_time:.2f} seconds")
    return model


def evaluate_on_dataset(dataset_name, model, max_input_dim):
    """Evaluate with enhanced metrics and visualization"""
    start_time = time.time()
    print(f"[DEBUG] Starting evaluation on {dataset_name}")
    hyperedge_index, labels, num_nodes, valid_mask, input_mask = load_hypergraph(
        dataset_name, max_input_dim=max_input_dim)
    x = prepare_input_features(num_nodes, max_input_dim)
    model.eval()
    with torch.no_grad():
        out = model(x, hyperedge_index)
        loss = structure_preserved_loss(
            out, hyperedge_index, input_mask, num_nodes)
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
    print(f"[DEBUG] Evaluation took {time.time() - start_time:.2f} seconds")
    return metrics


def main():
    """Main pipeline for Identity Matrix approach"""
    start_time = time.time()
    mlflow_manager = setup_mlflow()
    approach_name = "identity_matrix_label_agnostic"
    run_name = f"hypergraph_clustering_{approach_name}"

    with mlflow_manager.start_run(run_name=run_name):
        datasets = ["contact-high-school", "contact-primary-school"]
        hidden_dim = 4  # Kept from your code
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
            "approach": "identity_matrix",
            "loss_type": "structure_preserved"
        }
        mlflow.log_params(model_params)

        try:
            trained_model = train_on_dataset(
                datasets,
                model,
                max_input_dim=max_nodes,
                epochs=100,
                lr=0.001
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

    print(f"[DEBUG] Total execution took {
          time.time() - start_time:.2f} seconds")
    print("\nCheck MLflow UI for detailed results and visualizations.")


if __name__ == "__main__":
    print("Running with identity matrix approach (label-agnostic loss)")
    main()
