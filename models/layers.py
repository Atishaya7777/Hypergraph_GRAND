import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, Callable
from abc import ABC, abstractmethod
from enum import Enum

class IntegrationScheme(Enum):
    EXPLICIT = "explicit"
    IMPLICIT = "implicit"
    MULTISTEP = "multistep"
    ADAPTIVE = "adaptive"


class BaseIntegrator(ABC):
    """Base class for numerical integration schemes"""
    
    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
    
    @abstractmethod
    def step(self, h_current: torch.Tensor, h_init: torch.Tensor, 
             diffusion_fn: Callable, *args, **kwargs) -> torch.Tensor:
        """Perform one integration step"""
        pass


class ExplicitEulerIntegrator(BaseIntegrator):
    """Explicit Euler integration: h_{t+1} = h_0 + α * f(h_t)"""
    
    def step(self, h_current: torch.Tensor, h_init: torch.Tensor, 
             diffusion_fn: Callable, *args, **kwargs) -> torch.Tensor:
        divergence = diffusion_fn(h_current, *args, **kwargs)
        return h_init + self.alpha * divergence


class ImplicitEulerIntegrator(BaseIntegrator):
    """Implicit Euler integration: h_{t+1} = h_0 + α * f(h_{t+1})"""
    
    def __init__(self, alpha: float = 0.1, max_iter: int = 10, tol: float = 1e-6):
        super().__init__(alpha)
        self.max_iter = max_iter
        self.tol = tol
    
    def step(self, h_current: torch.Tensor, h_init: torch.Tensor, 
             diffusion_fn: Callable, *args, **kwargs) -> torch.Tensor:
        # Fixed point iteration for implicit method
        h_new = h_current.clone()
        
        for _ in range(self.max_iter):
            h_prev = h_new.clone()
            divergence = diffusion_fn(h_new, *args, **kwargs)
            h_new = h_init + self.alpha * divergence
            
            # Check convergence
            if torch.norm(h_new - h_prev) < self.tol:
                break
                
        return h_new


class MultistepIntegrator(BaseIntegrator):
    """Adams-Bashforth 2-step method"""
    
    def __init__(self, alpha: float = 0.1):
        super().__init__(alpha)
        self.prev_divergence = None
    
    def step(self, h_current: torch.Tensor, h_init: torch.Tensor, 
             diffusion_fn: Callable, *args, **kwargs) -> torch.Tensor:
        current_divergence = diffusion_fn(h_current, *args, **kwargs)
        
        if self.prev_divergence is None:
            # First step: use explicit Euler
            h_new = h_init + self.alpha * current_divergence
        else:
            # Adams-Bashforth 2-step: h_{t+1} = h_0 + α * (1.5 * f_t - 0.5 * f_{t-1})
            h_new = h_init + self.alpha * (1.5 * current_divergence - 0.5 * self.prev_divergence)
        
        self.prev_divergence = current_divergence.clone()
        return h_new
    
    def reset(self):
        """Reset the integrator state"""
        self.prev_divergence = None


class AdaptiveIntegrator(BaseIntegrator):
    """Adaptive step size using RK45 with error estimation"""
    
    def __init__(self, alpha: float = 0.1, min_alpha: float = 0.01, 
                 max_alpha: float = 0.5, tol: float = 1e-4):
        super().__init__(alpha)
        self.min_alpha = min_alpha
        self.max_alpha = max_alpha
        self.tol = tol
        self.current_alpha = alpha
    
    def step(self, h_current: torch.Tensor, h_init: torch.Tensor, 
             diffusion_fn: Callable, *args, **kwargs) -> torch.Tensor:
        # Runge-Kutta 4th order with embedded 5th order for error estimation
        k1 = diffusion_fn(h_current, *args, **kwargs)
        k2 = diffusion_fn(h_init + self.current_alpha * 0.25 * k1, *args, **kwargs)
        k3 = diffusion_fn(h_init + self.current_alpha * (3/32 * k1 + 9/32 * k2), *args, **kwargs)
        k4 = diffusion_fn(h_init + self.current_alpha * (1932/2197 * k1 - 7200/2197 * k2 + 7296/2197 * k3), *args, **kwargs)
        k5 = diffusion_fn(h_init + self.current_alpha * (439/216 * k1 - 8 * k2 + 3680/513 * k3 - 845/4104 * k4), *args, **kwargs)
        k6 = diffusion_fn(h_init + self.current_alpha * (-8/27 * k1 + 2 * k2 - 3544/2565 * k3 + 1859/4104 * k4 - 11/40 * k5), *args, **kwargs)
        
        # 4th order solution
        h_4th = h_init + self.current_alpha * (25/216 * k1 + 1408/2565 * k3 + 2197/4104 * k4 - 1/5 * k5)
        
        # 5th order solution
        h_5th = h_init + self.current_alpha * (16/135 * k1 + 6656/12825 * k3 + 28561/56430 * k4 - 9/50 * k5 + 2/55 * k6)
        
        # Estimate error
        error = torch.norm(h_5th - h_4th)
        
        # Adapt step size
        if error < self.tol:
            # Accept step and possibly increase step size
            self.current_alpha = min(self.max_alpha, self.current_alpha * 1.2)
            return h_4th
        else:
            # Reject step and decrease step size
            self.current_alpha = max(self.min_alpha, self.current_alpha * 0.5)
            # Retry with smaller step
            return self.step(h_current, h_init, diffusion_fn, *args, **kwargs)


class HypergraphDiffusionLayer(nn.Module):
    """
    Single diffusion layer for hypergraphs implementing the diffusion equation
    """

    def __init__(self, hidden_dim: int, alpha: float = 0.1, dropout: float = 0.1,
                 integration_scheme: IntegrationScheme = IntegrationScheme.EXPLICIT,
                 integrator_kwargs: Optional[dict] = None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.alpha = alpha
        self.dropout = dropout

        self.W_K = nn.Linear(hidden_dim, hidden_dim)
        self.W_Q = nn.Linear(hidden_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)

        # Initialize integrator
        integrator_kwargs = integrator_kwargs or {}
        self.integrator = self._create_integrator(integration_scheme, alpha, **integrator_kwargs)

    def _create_integrator(self, scheme: IntegrationScheme, alpha: float, **kwargs) -> BaseIntegrator:
        """Factory method to create integrators"""
        if scheme == IntegrationScheme.EXPLICIT:
            return ExplicitEulerIntegrator(alpha)
        elif scheme == IntegrationScheme.IMPLICIT:
            return ImplicitEulerIntegrator(alpha, **kwargs)
        elif scheme == IntegrationScheme.MULTISTEP:
            return MultistepIntegrator(alpha)
        elif scheme == IntegrationScheme.ADAPTIVE:
            return AdaptiveIntegrator(alpha, **kwargs)
        else:
            raise ValueError(f"Unknown integration scheme: {scheme}")

    def forward(self, h, h_init, hyperedge_index, hyperedge_weight=None, membership=None):
        """Apply one step of hypergraph diffusion"""
        # Define the diffusion function that will be used by the integrator
        def diffusion_function(h_input):
            return self._compute_diffusion_step(
                h_input, hyperedge_index, hyperedge_weight, membership
            )
        
        # Use the integrator to compute the next step
        h_new = self.integrator.step(h, h_init, diffusion_function)
        
        # Apply layer normalization and dropout
        h_new = self.layer_norm(h_new)
        h_new = F.dropout(h_new, p=self.dropout, training=self.training)

        return h_new

    def _compute_diffusion_step(self, h, hyperedge_index, hyperedge_weight=None, membership=None):
        """Compute the diffusion step (divergence) for a given h"""
        num_nodes = h.size(0)
        hyperedges = self.get_hyperedge_structure(hyperedge_index)
        degrees = self.compute_node_degrees(hyperedge_index, num_nodes)

        grad = self.compute_hypergraph_gradient(
            h, hyperedges, degrees, hyperedge_weight, membership)

        G = self.compute_diffusion_tensor(h, hyperedges, membership)

        divergence = self.compute_divergence(
            grad, G, hyperedges, degrees, hyperedge_weight, membership, num_nodes)

        return divergence

    def reset_integrator(self):
        """Reset integrator state (useful for multistep methods)"""
        if hasattr(self.integrator, 'reset'):
            self.integrator.reset()

    # [Include all the existing methods from your original implementation]
    def compute_hypergraph_gradient(self, psi, hyperedges, degrees, hyperedge_weight=None, membership=None):
        """Compute hypergraph gradient operator"""
        if not hyperedges:
            return torch.zeros(0, self.hidden_dim, device=psi.device)

        gradients = []

        for e_idx, nodes_in_edge in enumerate(hyperedges):
            if len(nodes_in_edge) < 2:
                gradients.append(torch.zeros(self.hidden_dim, device=psi.device))
                continue

            edge_weight = 1.0 if hyperedge_weight is None else hyperedge_weight[e_idx]
            delta_e = len(nodes_in_edge)

            ref_node = nodes_in_edge[0]
            grad_sum = torch.zeros(self.hidden_dim, device=psi.device)

            mu_ref = 1.0 if membership is None else membership[e_idx, ref_node].item()
            ref_term = (psi[ref_node] * mu_ref) / torch.sqrt(degrees[ref_node] + 1e-8)

            for node in nodes_in_edge[1:]:
                mu_node = 1.0 if membership is None else membership[e_idx, node].item()
                node_term = (psi[node] * mu_node) / torch.sqrt(degrees[node] + 1e-8)
                grad_sum += node_term - ref_term

            scale = torch.sqrt(torch.tensor(edge_weight, device=psi.device)) / \
                torch.sqrt(torch.tensor(delta_e - 1, dtype=torch.float, device=psi.device))
            gradients.append(scale * grad_sum)

        return torch.stack(gradients)

    def compute_diffusion_tensor(self, psi, hyperedges, membership=None):
        """Compute attention-based diffusion tensor G"""
        if not hyperedges:
            return torch.ones(0, device=psi.device)

        G_diag = []

        for e_idx, nodes_in_edge in enumerate(hyperedges):
            if len(nodes_in_edge) < 2:
                G_diag.append(torch.tensor(1.0, device=psi.device))
                continue

            attention_sum = 0.0
            n_nodes = len(nodes_in_edge)

            keys = [self.W_K(psi[v]) for v in nodes_in_edge]
            queries = [self.W_Q(psi[v]) for v in nodes_in_edge]

            for i in range(n_nodes):
                for j in range(n_nodes):
                    if i != j:
                        v1, v2 = nodes_in_edge[i], nodes_in_edge[j]
                        mu_v1 = 1.0 if membership is None else membership[e_idx, v1].item()
                        mu_v2 = 1.0 if membership is None else membership[e_idx, v2].item()

                        attention = torch.dot(keys[i], queries[j])
                        attention_sum += mu_v1 * mu_v2 * attention

            G_diag.append(torch.clamp(attention_sum, min=1e-8))

        return torch.stack(G_diag)

    def compute_divergence(self, grad, G, hyperedges, degrees, hyperedge_weight=None, membership=None, num_nodes=None):
        """Compute divergence operator"""
        divergence = torch.zeros(num_nodes, self.hidden_dim, device=grad.device)

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
                mu_node = 1.0 if membership is None else membership[e_idx, node].item()

                weight_factor = torch.sqrt(torch.tensor(edge_weight, device=grad.device)) * mu_node
                degree_term = max(1.0, (delta_e - 1) * (degrees[node].item() + 1e-8))
                norm_factor = torch.sqrt(torch.tensor(degree_term, device=grad.device))

                contribution = (weight_factor / norm_factor) * G[e_idx] * grad[e_idx]
                div_sum += contribution

            divergence[node] = div_sum

        return divergence

    def get_hyperedge_structure(self, hyperedge_index):
        """Convert hyperedge_index to list of node sets for each hyperedge"""
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
        """Compute node degrees in hypergraph"""
        degrees = torch.zeros(num_nodes, device=hyperedge_index.device)

        for i in range(hyperedge_index.size(1)):
            node_idx = hyperedge_index[1, i]
            degrees[node_idx] += 1

        return degrees
