import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, Callable
from abc import ABC, abstractmethod
from enum import Enum
import math

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
    Single diffusion layer for hypergraphs implementing the diffusion equation.
    Fixes: attention normalization, avoids tensor allocations in loops.
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

        integrator_kwargs = integrator_kwargs or {}
        self.integrator = self._create_integrator(integration_scheme, alpha, **integrator_kwargs)

    def _create_integrator(self, scheme: IntegrationScheme, alpha: float, **kwargs):
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
        device = h.device

        def diffusion_function(h_input):
            return self._compute_diffusion_step(
                h_input, hyperedge_index, hyperedge_weight, membership
            )

        # Run integrator step (this returns the "divergence" integrated into an update)
        h_new = self.integrator.step(h, h_init, diffusion_function)

        # Normalize + dropout
        h_new = self.layer_norm(h_new)
        h_new = F.dropout(h_new, p=self.dropout, training=self.training)

        return h_new

    def _compute_diffusion_step(self, h, hyperedge_index, hyperedge_weight=None, membership=None):
        """Compute divergence = aggregated contributions from hyperedges"""
        num_nodes = h.size(0)
        device = h.device

        # Prepare hyperedge lists
        hyperedges = self.get_hyperedge_structure(hyperedge_index)
        degrees = self.compute_node_degrees(hyperedge_index, num_nodes)  # [num_nodes]

        # If no hyperedges, return zero divergence
        if len(hyperedges) == 0:
            return torch.zeros_like(h)

        # Compute gradient per hyperedge (keeps original semantics, but avoids creating scalars)
        grad = self.compute_hypergraph_gradient(h, hyperedges, degrees, hyperedge_weight, membership)
        # Compute diffusion tensor (attention-based per-hyperedge scalar)
        G = self.compute_diffusion_tensor(h, hyperedges, membership)  # [num_hyperedges]

        # Divergence: distribute hyperedge contributions back to nodes
        divergence = self.compute_divergence(grad, G, hyperedges, degrees, hyperedge_weight, membership, num_nodes)
        return divergence

    def reset_integrator(self):
        if hasattr(self.integrator, 'reset'):
            self.integrator.reset()

    def compute_hypergraph_gradient(self, psi, hyperedges, degrees, hyperedge_weight=None, membership=None):
        """Compute hypergraph gradient operator (per-hyperedge vector)."""
        device = psi.device
        grads = []
        deg_eps = degrees + 1e-8  # [num_nodes]
        for e_idx, nodes_in_edge in enumerate(hyperedges):
            m = len(nodes_in_edge)
            if m < 2:
                grads.append(torch.zeros(self.hidden_dim, device=device))
                continue

            edge_w = 1.0 if hyperedge_weight is None else float(hyperedge_weight[e_idx].item())
            delta_e = m

            ref_node = nodes_in_edge[0]
            mu_ref = 1.0 if membership is None else float(membership[e_idx, ref_node].item())
            # compute ref term once
            ref_term = (psi[ref_node] * mu_ref) / torch.sqrt(deg_eps[ref_node])

            grad_sum = torch.zeros(self.hidden_dim, device=device)
            for node in nodes_in_edge[1:]:
                mu_node = 1.0 if membership is None else float(membership[e_idx, node].item())
                node_term = (psi[node] * mu_node) / torch.sqrt(deg_eps[node])
                grad_sum += (node_term - ref_term)

            scale = math.sqrt(edge_w) / math.sqrt(float(delta_e - 1))
            grads.append(scale * grad_sum)
        return torch.stack(grads, dim=0)  # [num_hyperedges, hidden_dim]

    def compute_diffusion_tensor(self, psi, hyperedges, membership=None):
        """
        Compute attention-based diffusion scalar per hyperedge.
        Normalizes attention by sqrt(d_k) and applies softmax over pair interactions to stabilize scale.
        """
        device = psi.device
        G_list = []
        d_k = float(self.hidden_dim)
        scale = 1.0 / math.sqrt(d_k)

        for e_idx, nodes_in_edge in enumerate(hyperedges):
            m = len(nodes_in_edge)
            if m < 2:
                G_list.append(torch.tensor(1.0, device=device))
                continue

            # stack keys/queries -> [m, hidden_dim]
            keys = torch.stack([self.W_K(psi[v]) for v in nodes_in_edge], dim=0)  # [m, d]
            queries = torch.stack([self.W_Q(psi[v]) for v in nodes_in_edge], dim=0)  # [m, d]

            # attention matrix: [m, m] = keys @ queries.T
            # scale and zero diagonal
            attn = (keys @ queries.t()) * scale
            attn = attn - torch.diag(torch.diag(attn))  # zero-out diagonal

            # flatten pairwise scores then softmax to get stable normalized pair importance
            # we compute pairwise contributions per node by summing over columns after softmax
            # but to stabilize magnitudes we normalize with softmax over the matrix rows
            attn_row_soft = F.softmax(attn, dim=1)  # each row sums to 1
            # per-hyperedge scalar: average of row sums (should be 1, but weighted by interactions)
            # better: sum of off-diagonal attention magnitudes divided by number of pair interactions
            pairwise_sum = attn_row_soft.sum()
            G_val = torch.clamp(pairwise_sum / float(m), min=1e-8)
            G_list.append(G_val)

        return torch.stack(G_list, dim=0)  # [num_hyperedges]

    def compute_divergence(self, grad, G, hyperedges, degrees, hyperedge_weight=None, membership=None, num_nodes=None):
        """
        Distribute hyperedge contributions back to nodes to compute node-level divergence.
        Vectorized-ish but loops per hyperedge; avoids creating CPU tensors in loops.
        """
        device = grad.device
        divergence = torch.zeros(num_nodes, self.hidden_dim, device=device)
        deg_eps = degrees + 1e-8

        for e_idx, nodes_in_edge in enumerate(hyperedges):
            m = len(nodes_in_edge)
            if m == 0:
                continue

            edge_w = 1.0 if hyperedge_weight is None else float(hyperedge_weight[e_idx].item())
            mu_vals = None
            if membership is None:
                mu_vals = [1.0] * m
            else:
                mu_vals = [float(membership[e_idx, n].item()) for n in nodes_in_edge]

            # precompute constants
            sqrt_edge_w = math.sqrt(edge_w)
            denom = max(1.0, (m - 1) * float(deg_eps[nodes_in_edge[0]].item()))
            norm_term = math.sqrt(denom)

            # grad[e_idx]: [hidden_dim], G[e_idx]: scalar
            g_e = grad[e_idx]  # vector
            G_e = float(G[e_idx].item())

            for idx, node in enumerate(nodes_in_edge):
                mu_node = mu_vals[idx]
                # compute contribution
                contrib = (sqrt_edge_w * mu_node / norm_term) * (G_e * g_e)
                divergence[node] += contrib

        return divergence

    def get_hyperedge_structure(self, hyperedge_index):
        """Convert hyperedge_index to list of node sets for each hyperedge"""
        if hyperedge_index.size(1) == 0:
            return []

        num_hyperedges = int(hyperedge_index[0].max().item()) + 1
        hyperedges = [[] for _ in range(num_hyperedges)]
        for i in range(hyperedge_index.size(1)):
            e = int(hyperedge_index[0, i].item())
            v = int(hyperedge_index[1, i].item())
            hyperedges[e].append(v)
        return hyperedges

    def compute_node_degrees(self, hyperedge_index, num_nodes):
        """Compute node degrees in hypergraph (vectorized)"""
        device = hyperedge_index.device
        degrees = torch.zeros(num_nodes, device=device)
        if hyperedge_index.size(1) == 0:
            return degrees
        nodes = hyperedge_index[1]  # [num_connections]
        # nodes might be long tensor, use scatter_add
        ones = torch.ones_like(nodes, dtype=degrees.dtype, device=device)
        degrees = degrees.scatter_add(0, nodes, ones)
        return degrees
