import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.cluster import SpectralClustering, KMeans


class HypergraphGRAND(nn.Module):
    """
    Hypergraph Graph Neural Diffusion (HyperGRAND) implementation
    Modified to learn distances to cluster centers instead of node classification
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int = 1,  # 1 for distance regression
        num_layers: int = 3,
        time_steps: int = 10,
        alpha: float = 0.1,
        dropout: float = 0.1
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.time_steps = time_steps
        self.alpha = alpha
        self.dropout = dropout

        self.input_transform = nn.Linear(input_dim, hidden_dim)

        self.diffusion_layers = nn.ModuleList([
            HypergraphDiffusionLayer(hidden_dim, alpha, dropout)
            for _ in range(num_layers)
        ])

        # Output layer for distance regression
        self.output_transform = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim),
            nn.Softplus()  # Ensure positive distances
        )

    def forward(self, x, hyperedge_index, hyperedge_weight=None, membership=None):
        """
        Forward pass through HyperGRAND
        Now outputs distances instead of log probabilities to measure
        distances to cluster centers
        """
        h = self.input_transform(x)
        h_init = h.clone()

        for layer in self.diffusion_layers:
            h = layer(h, h_init, hyperedge_index, hyperedge_weight, membership)

        # Output distances
        out = self.output_transform(h)
        return out.squeeze(-1) if self.output_dim == 1 else out


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
        h_new = h_init + self.alpha * divergence

        # Apply layer normalization and dropout
        h_new = self.layer_norm(h_new)
        h_new = F.dropout(h_new, p=self.dropout, training=self.training)

        return h_new

    def compute_hypergraph_gradient(self, psi, hyperedges, degrees, hyperedge_weight=None, membership=None):
        """
        Compute hypergraph gradient operator ∇ψ
        """
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

            for node in nodes_in_edge:
                if node != ref_node:
                    # Get membership values (default to 1 if not provided)
                    mu_node = 1.0 if membership is None else membership[e_idx, node].item(
                    )
                    mu_ref = 1.0 if membership is None else membership[e_idx, ref_node].item(
                    )

                    # Gradient calculation
                    term1 = (psi[node] * mu_node) / \
                        torch.sqrt(degrees[node] + 1e-8)
                    term2 = (psi[ref_node] * mu_ref) / \
                        torch.sqrt(degrees[ref_node] + 1e-8)
                    grad_sum += term1 - term2

            # Normalize gradient by edge weight and delta_e
            scale = (torch.sqrt(torch.tensor(edge_weight, device=psi.device)) /
                     torch.sqrt(torch.tensor(delta_e - 1, dtype=torch.float, device=psi.device)))
            grad_val = scale * grad_sum
            gradients.append(grad_val)

        return torch.stack(gradients) if gradients else torch.zeros(0, self.hidden_dim, device=psi.device)

    def compute_diffusion_tensor(self, psi, hyperedges, membership=None):
        """
        Compute attention-based diffusion tensor G
        """
        G_diag = []

        for e_idx, nodes_in_edge in enumerate(hyperedges):
            if len(nodes_in_edge) < 2:
                G_diag.append(torch.tensor(1.0, device=psi.device))
                continue

            attention_sum = 0.0

            for i, v1 in enumerate(nodes_in_edge):
                for j, v2 in enumerate(nodes_in_edge):
                    if i != j:
                        mu_v1 = 1.0 if membership is None else membership[e_idx, v1].item(
                        )
                        mu_v2 = 1.0 if membership is None else membership[e_idx, v2].item(
                        )

                        k_v1 = self.W_K(psi[v1])
                        q_v2 = self.W_Q(psi[v2])
                        attention = torch.dot(k_v1, q_v2)

                        attention_sum += mu_v1 * mu_v2 * attention

            # Add small epsilon for numerical stability
            G_diag.append(torch.clamp(attention_sum, min=1e-8))

        return torch.stack(G_diag) if G_diag else torch.ones(0, device=psi.device)

    def compute_divergence(self, grad, G, hyperedges, degrees, hyperedge_weight=None, membership=None, num_nodes=None):
        """
        Compute divergence operator div[G∇ψ]
        """
        divergence = torch.zeros(
            num_nodes, self.hidden_dim, device=grad.device)

        for node in range(num_nodes):
            div_sum = torch.zeros(self.hidden_dim, device=grad.device)

            # Sum over all hyperedges containing this node
            for e_idx, nodes_in_edge in enumerate(hyperedges):
                if node in nodes_in_edge:
                    edge_weight = 1.0 if hyperedge_weight is None else hyperedge_weight[e_idx]
                    delta_e = len(nodes_in_edge)

                    mu_node = 1.0 if membership is None else membership[e_idx, node].item(
                    )

                    # Compute divergence contribution with numerical stability
                    weight_factor = torch.sqrt(torch.tensor(
                        edge_weight, device=grad.device)) * mu_node
                    degree_term = max(1.0, (delta_e - 1) *
                                      (degrees[node].item() + 1e-8))

                    # Simplified normalization to avoid factorial overflow
                    norm_factor = torch.sqrt(torch.tensor(
                        degree_term, device=grad.device))

                    if e_idx < len(G) and e_idx < len(grad):
                        contribution = (
                            weight_factor / norm_factor) * G[e_idx] * grad[e_idx]
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


class HypergraphClusterAnalyzer:
    """
    Utility class for analyzing hypergraph clusters and computing distances to cluster centers
    """

    def __init__(self, method='spectral', n_clusters=None):
        self.method = method
        self.n_clusters = n_clusters
        self.cluster_centers_ = None
        self.labels_ = None
        self.adjacency_matrix_ = None

    def detect_clusters(self, hyperedge_index, num_nodes, node_features=None):
        """
        Detect clusters in the hypergraph structure

        Args:
            hyperedge_index: Hyperedge connectivity [2, num_edges]
            num_nodes: Number of nodes
            node_features: Optional node features for clustering
        """
        # Convert hypergraph to node-node adjacency for clustering
        self.adjacency_matrix_ = self._hypergraph_to_adjacency(
            hyperedge_index, num_nodes)

        # Determine number of clusters if not specified
        if self.n_clusters is None:
            # Heuristic default
            self.n_clusters = max(2, min(10, num_nodes // 20))

        # Use node features if available, otherwise use adjacency
        if node_features is not None:
            clustering_features = node_features.numpy()
        else:
            clustering_features = self.adjacency_matrix_.numpy()

        if self.method == 'spectral':
            # Use adjacency matrix for spectral clustering
            clustering = SpectralClustering(
                n_clusters=self.n_clusters,
                random_state=42,
                affinity='precomputed'
            )
            self.labels_ = clustering.fit_predict(
                self.adjacency_matrix_.numpy())
        else:  # kmeans
            clustering = KMeans(n_clusters=self.n_clusters,
                                random_state=42, n_init=10)
            self.labels_ = clustering.fit_predict(clustering_features)

        # Compute cluster centers in adjacency space
        self.cluster_centers_ = []
        for cluster_id in range(self.n_clusters):
            cluster_mask = self.labels_ == cluster_id
            if cluster_mask.sum() > 0:
                center = self.adjacency_matrix_[cluster_mask].mean(dim=0)
                self.cluster_centers_.append(center)
            else:
                # Empty cluster, use random center
                self.cluster_centers_.append(torch.randn(num_nodes))

        self.cluster_centers_ = torch.stack(self.cluster_centers_)

        print(f"Detected {self.n_clusters} clusters with sizes: {
              np.bincount(self.labels_)}")
        return self.labels_

    def compute_distances_to_centers(self, hyperedge_index, num_nodes, normalize=True):
        """
        Compute distances from each node to its cluster center
        """
        if self.cluster_centers_ is None:
            raise ValueError("Must call detect_clusters first")

        if self.adjacency_matrix_ is None:
            self.adjacency_matrix_ = self._hypergraph_to_adjacency(
                hyperedge_index, num_nodes)

        distances = torch.zeros(num_nodes)

        for node in range(num_nodes):
            cluster_id = self.labels_[node]
            center = self.cluster_centers_[cluster_id]
            node_features = self.adjacency_matrix_[node]

            distance = torch.norm(node_features - center)
            distances[node] = distance

        if normalize and distances.max() > 0:
            distances = distances / distances.max()

        return distances

    def _hypergraph_to_adjacency(self, hyperedge_index, num_nodes):
        """
        Convert hypergraph to node-node adjacency matrix
        """
        adjacency = torch.zeros(num_nodes, num_nodes)

        hyperedges = {}
        for i in range(hyperedge_index.size(1)):
            edge_idx = hyperedge_index[0, i].item()
            node_idx = hyperedge_index[1, i].item()
            if edge_idx not in hyperedges:
                hyperedges[edge_idx] = []
            hyperedges[edge_idx].append(node_idx)

        for edge_idx, nodes in hyperedges.items():
            for i in range(len(nodes)):
                for j in range(i + 1, len(nodes)):
                    node1, node2 = nodes[i], nodes[j]
                    adjacency[node1, node2] += 1
                    adjacency[node2, node1] += 1

        return adjacency

    def get_cluster_info(self):
        """
        Get information about detected clusters
        """
        if self.labels_ is None:
            return None

        cluster_info = {
            'n_clusters': self.n_clusters,
            'cluster_sizes': np.bincount(self.labels_),
            'labels': self.labels_
        }
        return cluster_info


def create_membership_function(hyperedge_index, num_nodes, sparsity=0.1):
    """
    Create a membership function for hypergraph nodes
    """
    num_hyperedges = hyperedge_index[0].max().item() + 1
    membership = torch.zeros(num_hyperedges, num_nodes)

    for i in range(hyperedge_index.size(1)):
        e_idx = hyperedge_index[0, i]
        n_idx = hyperedge_index[1, i]
        membership[e_idx, n_idx] = 1.0

    # Add some random membership values for nodes not directly in hyperedges
    if sparsity > 0:
        mask = torch.rand_like(membership) < sparsity
        membership[mask] = torch.rand(mask.sum()) * 0.5  # Partial membership

    return membership
