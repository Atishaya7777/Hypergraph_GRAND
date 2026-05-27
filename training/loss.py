import torch
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, accuracy_score, adjusted_rand_score
from scipy.optimize import linear_sum_assignment


def clustering_loss_function(
    embeddings,
    true_labels,
    lambda_sep=1.0,
    lambda_reg=0.01
):
    """
    Clustering loss with separation and regularization included

    Args:
        embeddings: The latent representation represented as embeddings
        true_labels: A list of true labels (The ground truth)
        lambda_sep: The factor of which to scale the separate of the centroids by
        lambda_reg: The regularization factor
    """
    device = embeddings.device
    unique_labels = torch.unique(true_labels)
    num_clusters = len(unique_labels)

    if num_clusters <= 1:
        return torch.tensor(0.0, device=device)

    # Compute centroids for each cluster
    centroids = []
    intra_cluster_loss = torch.tensor(0.0, device=device)
    total_points = 0

    for label in unique_labels:
        mask = (true_labels == label)
        cluster_size = mask.sum()

        if cluster_size <= 1:
            continue

        cluster_points = embeddings[mask]
        centroid = cluster_points.mean(dim=0)
        centroids.append(centroid)

        # Intra-cluster loss: minimize distances within clusters, just L^2
        distances_sq = torch.sum((cluster_points - centroid) ** 2, dim=1)
        intra_cluster_loss += distances_sq.sum()
        total_points += cluster_size

    # Normalize intra-cluster loss
    if total_points > 0:
        intra_cluster_loss = intra_cluster_loss / total_points

    # Inter-cluster loss: maximize distances between centroids
    inter_cluster_loss = torch.tensor(0.0, device=device)
    if len(centroids) > 1:
        centroids = torch.stack(centroids)
        # Pairwise L2 without torch.cdist (MPS lacks cdist backward)
        sq = (centroids ** 2).sum(dim=1, keepdim=True)
        dist_sq = sq + sq.T - 2.0 * (centroids @ centroids.T)
        centroid_distances = torch.sqrt(dist_sq.clamp(min=1e-8))
        # Extract upper triangular part (avoid double counting)
        mask = torch.triu(torch.ones_like(
            centroid_distances, dtype=torch.bool), diagonal=1)
        pairwise_distances = centroid_distances[mask]

        # We want to maximize inter-cluster distances, so minimize negative distances
        # Add small epsilon to avoid numerical issues
        inter_cluster_loss = -torch.mean(pairwise_distances) + 1e-6

    # Regularization term to prevent collapse
    embedding_norm = torch.mean(torch.norm(embeddings, dim=1))
    regularization = lambda_reg * (1.0 / (embedding_norm + 1e-8))

    total_loss = intra_cluster_loss + lambda_sep * \
        inter_cluster_loss + regularization

    return total_loss


def clustering_error_function(model_embeddings, true_labels, n_init=20):
    """
    Evaluate the predicted clusters with the actual true labels

    Args:
        model_embeddings: The latent representation represented as embeddings
        true_labels: A list of true labels (The ground truth)
        n_init: The number of times the KMeans algorithm is run with different centroid seeds 
    """
    embeddings_np = model_embeddings.detach().cpu().numpy()
    true_labels_np = true_labels.cpu().numpy()

    n_clusters = len(np.unique(true_labels_np))

    if n_clusters <= 1:
        return np.eye(1), 0.0, 0.0

    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=n_init,
        max_iter=300
    )

    try:
        predicted_clusters = kmeans.fit_predict(embeddings_np)
    except Exception:
        predicted_clusters = np.zeros(len(true_labels_np))

    # Compute ARI (this doesn't require label mapping)
    ari = adjusted_rand_score(true_labels_np, predicted_clusters)

    # For accuracy, we need to find optimal mapping
    cm = confusion_matrix(true_labels_np, predicted_clusters)

    # Find optimal assignment using Hungarian algorithm
    try:
        # Convert to maximization problem
        cost_matrix = -cm
        row_indices, col_indices = linear_sum_assignment(cost_matrix)

        # Create mapping
        cluster_mapping = {col_indices[i]: row_indices[i]
                           for i in range(len(col_indices))}

        # Apply mapping
        mapped_predictions = np.array(
            [cluster_mapping.get(pred, pred) for pred in predicted_clusters])

        # Compute accuracy
        accuracy = accuracy_score(true_labels_np, mapped_predictions)

        # Final confusion matrix
        final_cm = confusion_matrix(true_labels_np, mapped_predictions)

    except ImportError:
        print("Warning: scipy not available, using suboptimal cluster assignment")
        accuracy = accuracy_score(true_labels_np, predicted_clusters)
        final_cm = cm

    return final_cm, accuracy, ari
