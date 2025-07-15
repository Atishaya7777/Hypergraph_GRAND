from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
import torch


def visualize_embeddings_tsne(embeddings, true_labels, predicted_labels, dataset_name, epoch=None):
    """
    Create t-SNE visualization of embeddings with true and predicted labels

    Args:
        embeddings: torch.Tensor of shape [num_nodes, hidden_dim]
        true_labels: torch.Tensor of true class labels
        predicted_labels: numpy array of predicted cluster labels
        dataset_name: str, name of the dataset
        epoch: int, current epoch (optional)
    """
    # Convert to numpy if needed
    if isinstance(embeddings, torch.Tensor):
        embeddings_np = embeddings.detach().cpu().numpy()
    else:
        embeddings_np = embeddings

    if isinstance(true_labels, torch.Tensor):
        true_labels_np = true_labels.cpu().numpy()
    else:
        true_labels_np = true_labels

    # Apply t-SNE
    print(f"Applying t-SNE to embeddings...")
    tsne = TSNE(n_components=2, random_state=42,
                perplexity=min(30, len(embeddings_np)-1))
    embeddings_2d = tsne.fit_transform(embeddings_np)

    # Create visualization
    plt.figure(figsize=(15, 6))

    # Define colors for classes
    colors = ['red', 'blue', 'green', 'orange', 'purple',
              'brown', 'pink', 'gray', 'olive', 'cyan']
    num_classes = len(np.unique(true_labels_np))

    # Plot 1: True labels
    plt.subplot(1, 2, 1)
    for i in range(num_classes):
        mask = true_labels_np == i
        if mask.sum() > 0:
            plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                        c=colors[i % len(colors)], label=f'Class {i}',
                        alpha=0.7, s=30)

    plt.title(f'True Labels - {dataset_name}')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)

    # Plot 2: Predicted clusters
    plt.subplot(1, 2, 2)
    num_predicted_classes = len(np.unique(predicted_labels))
    for i in range(num_predicted_classes):
        mask = predicted_labels == i
        if mask.sum() > 0:
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

    # Save with epoch info if provided
    filename = f'hypergrand_tsne_{dataset_name.replace("-", "_")}'
    if epoch is not None:
        filename += f'_epoch_{epoch}'
    filename += '.png'

    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.show()

    print(f"t-SNE visualization saved as '{filename}'")
