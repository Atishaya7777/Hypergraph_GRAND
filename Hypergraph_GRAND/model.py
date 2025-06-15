import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class HypergraphGRAND(nn.Module):
    """
    Hypergraph Graph Neural Diffusion (HyperGRAND) implementation
    Based on extending GRAND to hypergraphs using diffusion theory
    """

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 num_layers: int = 3,
                 time_steps: int = 10,
                 alpha: float = 0.1,
                 dropout: float = 0.1):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.time_steps = time_steps
        self.alpha = alpha
        self.dropout = dropout

        # Input transformation
        self.input_transform = nn.Linear(input_dim, hidden_dim)

        # Attention weights for membership function
        self.W_K = nn.Linear(hidden_dim, hidden_dim)
        self.W_Q = nn.Linear(hidden_dim, hidden_dim)

        # Diffusion layers
        self.diffusion_layers = nn.ModuleList([
            HypergraphDiffusionLayer(hidden_dim, alpha, dropout)
            for _ in range(num_layers)
        ])

        # Output transformation
        self.output_transform = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, hyperedge_index, hyperedge_weight=None, membership=None):
        """
        Forward pass through HyperGRAND

        Args:
            x: Node features [num_nodes, input_dim]
            hyperedge_index: Hyperedge connectivity [2, num_edges] where
                           hyperedge_index[0] = hyperedge indices
                           hyperedge_index[1] = node indices
            hyperedge_weight: Optional edge weights [num_hyperedges]
            membership: Optional membership function values [num_hyperedges, num_nodes]
        """
        # Initial transformation
        h = self.input_transform(x)
        h_init = h.clone()

        # Apply diffusion layers
        for layer in self.diffusion_layers:
            h = layer(h, h_init, hyperedge_index, hyperedge_weight, membership)

        # Output transformation
        out = self.output_transform(h)
        return F.log_softmax(out, dim=1)


class HypergraphDiffusionLayer(nn.Module):
    """
    Single diffusion layer for hypergraphs implementing the diffusion equation:
    ∂φ/∂t = div[G(ψ)∇ψ]
    """

    def __init__(self, hidden_dim: int, alpha: float = 0.1, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.alpha = alpha
        self.dropout = dropout

        self.W_K = nn.Linear(hidden_dim, hidden_dim)
        self.W_Q = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, h, h_init, hyperedge_index, hyperedge_weight=None, membership=None):
        """
        Apply one step of hypergraph diffusion
        """
        # Compute gradient
        grad = self.compute_hypergraph_gradient(
            h, hyperedge_index, hyperedge_weight, membership)

        # Compute attention-based diffusion tensor G
        G = self.compute_diffusion_tensor(h, hyperedge_index, membership)

        # Compute divergence operator div[G∇ψ]
        divergence = self.compute_divergence(
            grad, G, hyperedge_index, hyperedge_weight, membership, h.size(0))

        # Time discretization with residual connection
        h_new = h_init + self.alpha * divergence

        # Apply dropout
        h_new = F.dropout(h_new, p=self.dropout, training=self.training)

        return h_new

    def compute_hypergraph_gradient(self, psi, hyperedge_index, hyperedge_weight=None, membership=None):
        """
        Compute hypergraph gradient operator ∇ψ
        Based on the formula in your theory
        """
        num_nodes = psi.size(0)
        num_hyperedges = hyperedge_index[0].max().item() + 1

        # Get hyperedge structure
        hyperedges = self.get_hyperedge_structure(hyperedge_index)

        # Initialize gradient for each hyperedge
        gradients = []

        for e_idx, nodes_in_edge in enumerate(hyperedges):
            if len(nodes_in_edge) < 2:
                gradients.append(torch.zeros(1, device=psi.device))
                continue

            edge_weight = torch.tensor(1.0, device=psi.device) if hyperedge_weight is None else torch.tensor(
                hyperedge_weight[e_idx], device=psi.device)
            delta_e = len(nodes_in_edge)

            # Compute degrees
            degrees = self.compute_node_degrees(hyperedge_index, num_nodes)

            # Find reference node (assuming negative membership indicates reference)
            # Simplified - in practice use membership function
            ref_node = nodes_in_edge[0]

            grad_sum = 0.0
            for node in nodes_in_edge:
                if node != ref_node:
                    # Get membership values (default to 1 if not provided)
                    mu_node = 1.0 if membership is None else membership[e_idx, node]
                    mu_ref = - \
                        1.0 if membership is None else membership[e_idx, ref_node]

                    # Compute gradient component
                    term1 = (psi[node] * mu_node) / \
                        torch.sqrt(degrees[node] + 1e-8)
                    term2 = (psi[ref_node] * mu_ref) / \
                        torch.sqrt(degrees[ref_node] + 1e-8)
                    grad_sum += term1 - term2

            # Scale by edge weight and normalization
            grad_val = (torch.sqrt(edge_weight) / torch.sqrt(torch.tensor(delta_e -
                        1, dtype=torch.float, device=psi.device))) * grad_sum
            gradients.append(grad_val)

        return torch.stack(gradients)

    def compute_diffusion_tensor(self, psi, hyperedge_index, membership=None):
        """
        Compute attention-based diffusion tensor G
        G_ii = Σ_{v1,v2 ∈ e_i, v1≠v2} μ(e_i,v1)μ(e_i,v2)(W_K X_{v1})^T W_Q X_{v2}
        """
        num_hyperedges = hyperedge_index[0].max().item() + 1
        hyperedges = self.get_hyperedge_structure(hyperedge_index)

        G_diag = []

        for e_idx, nodes_in_edge in enumerate(hyperedges):
            if len(nodes_in_edge) < 2:
                G_diag.append(torch.tensor(0.0, device=psi.device))
                continue

            attention_sum = 0.0

            # Compute attention for all pairs in hyperedge
            for i, v1 in enumerate(nodes_in_edge):
                for j, v2 in enumerate(nodes_in_edge):
                    if i != j:
                        # Get membership values
                        mu_v1 = 1.0 if membership is None else membership[e_idx, v1]
                        mu_v2 = 1.0 if membership is None else membership[e_idx, v2]

                        # Compute attention
                        k_v1 = self.W_K(psi[v1])
                        q_v2 = self.W_Q(psi[v2])
                        attention = torch.dot(k_v1, q_v2)

                        attention_sum += mu_v1 * mu_v2 * attention

            G_diag.append(attention_sum)

        return torch.stack(G_diag)

    def compute_divergence(self, grad, G, hyperedge_index, hyperedge_weight=None, membership=None, num_nodes=None):
        """
        Compute divergence operator div[G∇ψ]
        """
        if num_nodes is None:
            num_nodes = hyperedge_index[1].max().item() + 1
        num_hyperedges = hyperedge_index[0].max().item() + 1

        # Initialize divergence for each node
        divergence = torch.zeros(
            num_nodes, self.hidden_dim, device=grad.device)

        # Get hyperedge structure
        hyperedges = self.get_hyperedge_structure(hyperedge_index)
        degrees = self.compute_node_degrees(hyperedge_index, num_nodes)

        for node in range(num_nodes):
            div_sum = 0.0

            # Sum over all hyperedges containing this node
            for e_idx, nodes_in_edge in enumerate(hyperedges):
                if node in nodes_in_edge:
                    edge_weight = torch.tensor(1.0, device=grad.device) if hyperedge_weight is None else torch.tensor(
                        hyperedge_weight[e_idx], device=grad.device)
                    delta_e = len(nodes_in_edge)

                    # Get membership
                    mu_node = 1.0 if membership is None else membership[e_idx, node]

                    # Compute divergence contribution
                    weight_factor = torch.sqrt(edge_weight) * mu_node
                    degree_term = (delta_e - 1) * (degrees[node] + 1e-8)
                    norm_factor = math.factorial(
                        delta_e) * torch.sqrt(degree_term)

                    contribution = (weight_factor / norm_factor) * \
                        G[e_idx] * grad[e_idx]
                    div_sum += contribution

            # Set divergence (expand to hidden_dim)
            divergence[node] = div_sum * \
                torch.ones(self.hidden_dim, device=grad.device)

        return divergence

    def get_hyperedge_structure(self, hyperedge_index):
        """
        Convert hyperedge_index to list of node sets for each hyperedge
        """
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

# Utility functions for creating hypergraph data


def create_synthetic_hypergraph(num_nodes=100, num_hyperedges=50, avg_edge_size=3):
    """
    Create synthetic hypergraph data for testing
    """
    # Generate random node features
    x = torch.randn(num_nodes, 16)

    # Generate random hyperedges
    hyperedge_list = []
    for e in range(num_hyperedges):
        edge_size = max(2, int(np.random.poisson(avg_edge_size)))
        nodes = np.random.choice(num_nodes, size=min(
            edge_size, num_nodes), replace=False)
        for node in nodes:
            hyperedge_list.append([e, node])

    hyperedge_index = torch.tensor(
        hyperedge_list, dtype=torch.long).t().contiguous()

    # Generate random labels
    y = torch.randint(0, 3, (num_nodes,))

    return x, hyperedge_index, y


def create_membership_function(hyperedge_index, num_nodes, sparsity=0.1):
    """
    Create a synthetic membership function with some negative values
    """
    num_hyperedges = hyperedge_index[0].max().item() + 1
    membership = torch.zeros(num_hyperedges, num_nodes)

    # Set membership for nodes in hyperedges
    for i in range(hyperedge_index.size(1)):
        e_idx = hyperedge_index[0, i]
        n_idx = hyperedge_index[1, i]
        membership[e_idx, n_idx] = 1.0

    # Add some negative memberships (reference nodes)
    mask = torch.rand_like(membership) < sparsity
    membership[mask] = -1.0

    return membership


# Example usage and testing
if __name__ == "__main__":
    # Create synthetic data
    x, hyperedge_index, y = create_synthetic_hypergraph()
    membership = create_membership_function(hyperedge_index, x.size(0))

    # Initialize model
    model = HypergraphGRAND(
        input_dim=16,
        hidden_dim=32,
        output_dim=3,
        num_layers=3,
        alpha=0.1
    )

    # Forward pass
    out = model(x, hyperedge_index, membership=membership)

    print(f"Input shape: {x.shape}")
    print(f"Hyperedge index shape: {hyperedge_index.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Number of hyperedges: {hyperedge_index[0].max().item() + 1}")

    # Compute loss
    loss = F.nll_loss(out, y)
    print(f"Loss: {loss.item():.4f}")
