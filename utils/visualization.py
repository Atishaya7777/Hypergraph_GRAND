import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.manifold import TSNE
from typing import List


def visualize_embeddings_tsne(
    embeddings: torch.Tensor,
    true_labels: torch.Tensor,
    predicted_labels: np.ndarray,
    dataset_name: str,
    epoch: int = None
):
    """
    Creates a side-by-side t-SNE visualization of learned embeddings, comparing
    true class labels and predicted cluster assignments.

    Args:
        embeddings (torch.Tensor): Node embeddings of shape [num_nodes, hidden_dim].
        true_labels (torch.Tensor): Ground truth labels for each node.
        predicted_labels (np.ndarray): Cluster labels assigned by a clustering algorithm (e.g., KMeans).
        dataset_name (str): Name of the dataset (used in title and file naming).
        epoch (int, optional): Epoch number to include in the output filename and title.
    """
    # Convert tensors to NumPy arrays if necessary
    embeddings_np = embeddings.detach().cpu().numpy() if isinstance(
        embeddings, torch.Tensor) else embeddings
    true_labels_np = true_labels.cpu().numpy() if isinstance(
        true_labels, torch.Tensor) else true_labels

    print("Applying t-SNE to embeddings...")
    tsne = TSNE(n_components=2, random_state=42,
                perplexity=min(30, len(embeddings_np) - 1))
    embeddings_2d = tsne.fit_transform(embeddings_np)

    plt.figure(figsize=(15, 6))
    colors = ['red', 'blue', 'green', 'orange', 'purple',
              'brown', 'pink', 'gray', 'olive', 'cyan']

    # Plot true labels
    plt.subplot(1, 2, 1)
    for i in np.unique(true_labels_np):
        mask = true_labels_np == i
        plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                    c=colors[i % len(colors)], label=f'Class {i}',
                    alpha=0.7, s=30)
    plt.title(f'True Labels - {dataset_name}')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)

    # Plot predicted clusters
    plt.subplot(1, 2, 2)
    for i in np.unique(predicted_labels):
        mask = predicted_labels == i
        plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                    c=colors[i % len(colors)], label=f'Cluster {i}',
                    alpha=0.7, s=30)

    title = f'Predicted Clusters - {dataset_name}'
    if epoch is not None:
        title += f' (Epoch {epoch})'
    plt.title(title)
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    filename = f"hypergrand_tsne_{dataset_name.replace('-', '_')}"
    if epoch is not None:
        filename += f"_epoch_{epoch}"
    filename += ".png"

    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.show()

    print(f"t-SNE visualization saved as '{filename}'")


def plot_metric_over_epochs(
    values: List[float],
    name: str,
    output_dir: str = "mlflow_plots",
    filename_prefix: str = "metric"
) -> str:
    """
    Plots a single metric over epochs and saves it as a PNG.

    Args:
        values (List[float]): List of metric values (e.g., loss or accuracy) over epochs.
        name (str): Metric name (e.g., "Validation Accuracy") — used for title and axis.
        output_dir (str): Directory to save the plot image.
        filename_prefix (str): Prefix for the filename.

    Returns:
        str: Full path to the saved PNG file.
    """
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(values, marker='o')
    plt.title(f"{name} over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel(name)
    plt.grid(True, alpha=0.3)

    filename = f"{filename_prefix}_{name.replace(' ', '_').lower()}.png"
    full_path = os.path.join(output_dir, filename)

    plt.savefig(full_path, dpi=150, bbox_inches='tight')
    plt.close()

    return full_path
