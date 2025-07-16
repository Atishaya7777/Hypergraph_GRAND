import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, accuracy_score


class HypergraphGRAND(nn.Module):
    """
    Hypergraph Graph Neural Diffusion (HyperGRAND) implementation
    Optimized for clustering-based learning with latent representations
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 3,
        alpha: float = 0.1,
        dropout: float = 0.1
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.alpha = alpha
        self.dropout = dropout

        self.input_transform = nn.Linear(input_dim, hidden_dim)

        self.diffusion_layers = nn.ModuleList([
            HypergraphDiffusionLayer(hidden_dim, alpha, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor, hyperedge_index: torch.Tensor,
                hyperedge_weight: Optional[torch.Tensor] = None,
                membership: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through HyperGRAND

        Args:
            x: Node features [num_nodes, input_dim]
            hyperedge_index: Hyperedge connectivity [2, num_edges]
            hyperedge_weight: Optional hyperedge weights
            membership: Optional membership matrix [num_hyperedges, num_nodes]

        Returns:
            Latent representations [num_nodes, hidden_dim]
        """
        h = self.input_transform(x)
        h_init = h.clone()

        for layer in self.diffusion_layers:
            h = layer(h, h_init, hyperedge_index, hyperedge_weight, membership)

        # Return latent representations for clustering
        return h


class HypergraphDiffusionLayer(nn.Module):
    """
    Single diffusion layer for hypergraphs implementing the diffusion equation
    """

    def __init__(self, hidden_dim: int, alpha: float = 0.1, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.alpha = alpha
        self.dropout = dropout

        self.W_K = nn.Linear(hidden_dim, hidden_dim)
        self.W_Q = nn.Linear(hidden_dim, hidden_dim)

        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, h, h_init, hyperedge_index, hyperedge_weight=None, membership=None):
        """
        Apply one step of hypergraph diffusion
        """
        num_nodes = h.size(0)

        hyperedges = self.get_hyperedge_structure(hyperedge_index)
        degrees = self.compute_node_degrees(hyperedge_index, num_nodes)

        grad = self.compute_hypergraph_gradient(
            h, hyperedges, degrees, hyperedge_weight, membership)

        # Compute attention-based diffusion tensor G
        G = self.compute_diffusion_tensor(h, hyperedges, membership)

        # Compute divergence operator
        divergence = self.compute_divergence(
            grad, G, hyperedges, degrees, hyperedge_weight, membership, num_nodes)

        # Time discretization with residual connection
        # Explicit Euler step. TODO: In the future, compare and contrast the
        # explicit, implicit Euler methods and the Runge-Kutta methods.
        h_new = h_init + self.alpha * divergence

        # Apply layer normalization and dropout
        h_new = self.layer_norm(h_new)
        h_new = F.dropout(h_new, p=self.dropout, training=self.training)

        return h_new

    def compute_hypergraph_gradient(self, psi, hyperedges, degrees, hyperedge_weight=None, membership=None):
        """
        Compute hypergraph gradient operator

        Args:
            psi: Current feature representation of the nodes - A function from 
                 nodes to R^d
            hyperedges: A list of node sets for each hyperedge
            degrees: A list of degrees for each node
            hyperedge_weight: Optional Hyperedge weights
            membership: Optional membership matrix [num_hyperedges, num_nodes]
        """
        if not hyperedges:
            return torch.zeros(0, self.hidden_dim, device=psi.device)

        gradients = []

        for e_idx, nodes_in_edge in enumerate(hyperedges):
            if len(nodes_in_edge) < 2:
                gradients.append(torch.zeros(
                    self.hidden_dim, device=psi.device))
                continue

            edge_weight = 1.0 if hyperedge_weight is None else hyperedge_weight[e_idx]
            delta_e = len(nodes_in_edge)

            ref_node = nodes_in_edge[0]
            grad_sum = torch.zeros(self.hidden_dim, device=psi.device)

            # Precompute reference node terms
            mu_ref = 1.0 if membership is None else membership[e_idx, ref_node].item(
            )
            ref_term = (psi[ref_node] * mu_ref) / \
                torch.sqrt(
                    # Adding a small epsilon for numerical stability
                    degrees[ref_node] + 1e-8)

            for node in nodes_in_edge[1:]:  # Skip reference node
                mu_node = 1.0 if membership is None else membership[e_idx, node].item(
                )
                node_term = (psi[node] * mu_node) / \
                    torch.sqrt(degrees[node] + 1e-8)
                grad_sum += node_term - ref_term

            # Normalize gradient
            scale = torch.sqrt(torch.tensor(edge_weight, device=psi.device)) / \
                torch.sqrt(torch.tensor(
                    delta_e - 1, dtype=torch.float, device=psi.device))
            gradients.append(scale * grad_sum)

        return torch.stack(gradients)

    def compute_diffusion_tensor(self, psi, hyperedges, membership=None):
        """
        Compute attention-based diffusion tensor G
            psi: Current feature representation of the nodes - A function from 
                 nodes to R^d
            hyperedges: A list of node sets for each hyperedge
            membership: Optional membership matrix [num_hyperedges, num_nodes]
        """
        if not hyperedges:
            return torch.ones(0, device=psi.device)

        G_diag = []

        for e_idx, nodes_in_edge in enumerate(hyperedges):
            if len(nodes_in_edge) < 2:
                G_diag.append(torch.tensor(1.0, device=psi.device))
                continue

            attention_sum = 0.0
            n_nodes = len(nodes_in_edge)

            # Precompute keys and queries for all nodes in this edge
            keys = [self.W_K(psi[v]) for v in nodes_in_edge]
            queries = [self.W_Q(psi[v]) for v in nodes_in_edge]

            for i in range(n_nodes):
                for j in range(n_nodes):
                    if i != j:
                        v1, v2 = nodes_in_edge[i], nodes_in_edge[j]
                        mu_v1 = 1.0 if membership is None else membership[e_idx, v1].item(
                        )
                        mu_v2 = 1.0 if membership is None else membership[e_idx, v2].item(
                        )

                        attention = torch.dot(keys[i], queries[j])
                        attention_sum += mu_v1 * mu_v2 * attention

            G_diag.append(torch.clamp(attention_sum, min=1e-8))

        return torch.stack(G_diag)

    def compute_divergence(self, grad, G, hyperedges, degrees, hyperedge_weight=None, membership=None, num_nodes=None):
        """
        Compute divergence operator 

        Args:
            grad: The gradient tensor of the hypergraph.
            G: The diffusion tensor based on the attention mechanism. Controls how much to diffuse.
            degrees: A list of degrees for each node.
            hyperedge_weight: Optional Hyperedge weights.
            membership: Optional membership matrix [num_hyperedges, num_nodes]
            num_nodes: The number of nodes in this hypergraph.
        """
        divergence = torch.zeros(
            num_nodes, self.hidden_dim, device=grad.device)

        # Create node-to-edges mapping for efficient lookup
        node_to_edges = {}
        for e_idx, nodes_in_edge in enumerate(hyperedges):
            for node in nodes_in_edge:
                if node not in node_to_edges:
                    node_to_edges[node] = []
                node_to_edges[node].append(e_idx)

        for node in range(num_nodes):
            if node not in node_to_edges:
                continue

            div_sum = torch.zeros(self.hidden_dim, device=grad.device)

            for e_idx in node_to_edges[node]:
                if e_idx >= len(G) or e_idx >= len(grad):
                    continue

                edge_weight = 1.0 if hyperedge_weight is None else hyperedge_weight[e_idx]
                delta_e = len(hyperedges[e_idx])
                mu_node = 1.0 if membership is None else membership[e_idx, node].item(
                )

                # Simplified computation
                weight_factor = torch.sqrt(torch.tensor(
                    edge_weight, device=grad.device)) * mu_node
                degree_term = max(1.0, (delta_e - 1) *
                                  (degrees[node].item() + 1e-8))
                norm_factor = torch.sqrt(torch.tensor(
                    degree_term, device=grad.device))

                contribution = (weight_factor / norm_factor) * \
                    G[e_idx] * grad[e_idx]
                div_sum += contribution

            divergence[node] = div_sum

        return divergence

    def get_hyperedge_structure(self, hyperedge_index):
        """
        Convert hyperedge_index to list of node sets for each hyperedge
        """
        if hyperedge_index.size(1) == 0:
            return []

        num_hyperedges = hyperedge_index[0].max().item() + 1
        hyperedges = [[] for _ in range(num_hyperedges)]

        for i in range(hyperedge_index.size(1)):
            edge_idx = hyperedge_index[0, i].item()
            node_idx = hyperedge_index[1, i].item()
            hyperedges[edge_idx].append(node_idx)

        return hyperedges

    def compute_node_degrees(self, hyperedge_index, num_nodes):
        """
        Compute node degrees in hypergraph
        """
        degrees = torch.zeros(num_nodes, device=hyperedge_index.device)

        for i in range(hyperedge_index.size(1)):
            node_idx = hyperedge_index[1, i]
            degrees[node_idx] += 1

        return degrees


def clustering_loss_function(model_output, node_labels):
    """
    Clustering-based loss function

    Args:
        model_output: Latent representations from model [num_nodes, hidden_dim]
        node_labels: Ground truth cluster labels [num_nodes]

    Returns:
        loss: Clustering loss value
    """
    device = model_output.device
    unique_clusters = torch.unique(node_labels)
    total_loss = 0.0

    for cluster_id in unique_clusters:
        # Get nodes in this cluster
        cluster_mask = (node_labels == cluster_id)
        cluster_nodes = model_output[cluster_mask]

        if cluster_nodes.size(0) == 0:
            continue

        # Compute cluster centroid: c_i = sum(v for v in cluster) / |cluster|
        cluster_centroid = cluster_nodes.mean(dim=0)

        # Compute error for this cluster: err_i = sum(||v - c_i|| for v in cluster)
        cluster_errors = torch.norm(
            cluster_nodes - cluster_centroid.unsqueeze(0), dim=1)

        # Normalize by cluster size
        cluster_loss = cluster_errors.sum() / cluster_nodes.size(0)
        total_loss += cluster_loss

    # Average over all clusters
    final_loss = total_loss / len(unique_clusters)
    return final_loss


def clustering_error_function(model_output, correct_node_labeling):
    """
    Clustering evaluation function as per pseudocode

    Args:
        model_output: Latent representations from model [num_nodes, hidden_dim]
        correct_node_labeling: Ground truth cluster labels [num_nodes]

    Returns:
        confusion_matrix: Confusion matrix
        accuracy: Classification accuracy after majority vote mapping
    """
    # Apply K-means to get predicted clusters
    n_clusters = len(torch.unique(correct_node_labeling))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    predicted_clusters = kmeans.fit_predict(
        model_output.detach().cpu().numpy())

    # Convert to numpy for sklearn
    true_labels = correct_node_labeling.cpu().numpy()

    # Create mapping using majority vote
    label_mapping = {}
    for pred_cluster in range(n_clusters):
        mask = predicted_clusters == pred_cluster
        if mask.sum() > 0:
            # Find most common true label in this predicted cluster
            true_labels_in_cluster = true_labels[mask]
            most_common = np.bincount(true_labels_in_cluster).argmax()
            label_mapping[pred_cluster] = most_common
        else:
            label_mapping[pred_cluster] = 0  # Default mapping

    # Apply mapping to predicted clusters
    mapped_predictions = np.array([label_mapping[pred]
                                  for pred in predicted_clusters])

    # Compute confusion matrix and accuracy
    cm = confusion_matrix(true_labels, mapped_predictions)
    accuracy = accuracy_score(true_labels, mapped_predictions)

    return cm, accuracy
