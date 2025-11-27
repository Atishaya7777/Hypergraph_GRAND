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

# ----------------- Integrators -----------------
class BaseIntegrator(ABC):
    """Base class for numerical integration schemes"""
    
    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
    
    @abstractmethod
    def step(self, h_current: torch.Tensor, diffusion_fn: Callable, *args, **kwargs) -> torch.Tensor:
        """Perform one integration step"""
        pass

class ExplicitEulerIntegrator(BaseIntegrator):
    """Explicit Euler integration: h_{t+1} = h_t + \alpha * f(h_t)"""
    
    def step(self, h_current: torch.Tensor, diffusion_fn: Callable, *args, **kwargs) -> torch.Tensor:
        divergence = diffusion_fn(h_current, *args, **kwargs)
        return h_current + self.alpha * divergence

class ImplicitEulerIntegrator(BaseIntegrator):
    """Implicit Euler with fixed-point iteration (practical version)."""

    def __init__(self, alpha: float = 0.1, max_iter: int = 10, tol: float = 1e-6, relaxation: float = 1.0, verbose: bool = False):
        super().__init__(alpha)
        self.max_iter = max_iter
        self.tol = tol
        self.relaxation = relaxation
        self.verbose = verbose

    def step(self, h_current: torch.Tensor, diffusion_fn: Callable, *args, **kwargs) -> torch.Tensor:
        device = h_current.device
        h_new = h_current.detach().clone().to(device)

        converged = False
        for it in range(self.max_iter):
            h_prev = h_new
            with torch.no_grad():
                divergence = diffusion_fn(h_new, *args, **kwargs)
                candidate = h_new + self.alpha * divergence
                h_new = h_prev + self.relaxation * (candidate - h_prev)
            if torch.norm(h_new - h_prev) < self.tol:
                converged = True
                break

        if not converged and self.verbose:
            print(f"[ImplicitIntegrator] fixed-point did NOT converge (iter={it+1}/{self.max_iter}) final_norm={torch.norm(h_new - h_prev):.4e}")

        # final differentiable evaluation
        h_final = h_new.clone().detach().requires_grad_(True)
        divergence_final = diffusion_fn(h_final, *args, **kwargs)
        return h_current + self.alpha * divergence_final

class MultistepIntegrator(BaseIntegrator):
    """Adams-Bashforth 2-step method"""
    
    def __init__(self, alpha: float = 0.1):
        super().__init__(alpha)
        self.prev_divergence = None
    
    def step(self, h_current: torch.Tensor, diffusion_fn: Callable, *args, **kwargs) -> torch.Tensor:
        current_divergence = diffusion_fn(h_current, *args, **kwargs)
        if self.prev_divergence is None:
            h_new = h_current + self.alpha * current_divergence
        else:
            h_new = h_current + self.alpha * (1.5 * current_divergence - 0.5 * self.prev_divergence)
        self.prev_divergence = current_divergence.clone()
        return h_new
    
    def reset(self):
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
    
    def step(self, h_current: torch.Tensor, diffusion_fn: Callable, *args, **kwargs) -> torch.Tensor:
        k1 = diffusion_fn(h_current, *args, **kwargs)
        k2 = diffusion_fn(h_current + self.current_alpha * 0.25 * k1, *args, **kwargs)
        k3 = diffusion_fn(h_current + self.current_alpha * (3/32 * k1 + 9/32 * k2), *args, **kwargs)
        k4 = diffusion_fn(h_current + self.current_alpha * (1932/2197 * k1 - 7200/2197 * k2 + 7296/2197 * k3), *args, **kwargs)
        k5 = diffusion_fn(h_current + self.current_alpha * (439/216 * k1 - 8 * k2 + 3680/513 * k3 - 845/4104 * k4), *args, **kwargs)
        k6 = diffusion_fn(h_current + self.current_alpha * (-8/27 * k1 + 2 * k2 - 3544/2565 * k3 + 1859/4104 * k4 - 11/40 * k5), *args, **kwargs)
        
        h_4th = h_current + self.current_alpha * (25/216 * k1 + 1408/2565 * k3 + 2197/4104 * k4 - 1/5 * k5)
        h_5th = h_current + self.current_alpha * (16/135 * k1 + 6656/12825 * k3 + 28561/56430 * k4 - 9/50 * k5 + 2/55 * k6)
        
        error = torch.norm(h_5th - h_4th)
        
        if error < self.tol:
            self.current_alpha = min(self.max_alpha, self.current_alpha * 1.2)
            return h_4th
        else:
            self.current_alpha = max(self.min_alpha, self.current_alpha * 0.5)
            return self.step(h_current, diffusion_fn, *args, **kwargs)

# ----------------- Hypergraph Diffusion Layer -----------------
class HypergraphDiffusionLayer(nn.Module):
    """Single hypergraph diffusion layer"""
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

    def forward(self, h, hyperedge_index, hyperedge_weight=None, membership=None):
        device = h.device
        hyperedges = self.get_hyperedge_structure(hyperedge_index)
        degrees = self.compute_node_degrees(hyperedge_index, h.size(0))

        def diffusion_fn(h_input):
            return self._compute_diffusion_step(h_input, hyperedges, degrees, hyperedge_weight, membership)

        h_new = self.integrator.step(h, diffusion_fn)
        h_new = self.layer_norm(h_new)
        h_new = F.dropout(h_new, p=self.dropout, training=self.training)
        return h_new

    def reset_integrator(self):
        if hasattr(self.integrator, 'reset'):
            self.integrator.reset()

    # --------------- Helper methods ----------------
    def _compute_diffusion_step(self, h, hyperedges, degrees, hyperedge_weight=None, membership=None):
        if len(hyperedges) == 0:
            return torch.zeros_like(h)
        grad = self.compute_hypergraph_gradient(h, hyperedges, degrees, hyperedge_weight, membership)
        G = self.compute_diffusion_tensor(h, hyperedges, membership)
        divergence = self.compute_divergence(grad, G, hyperedges, degrees, hyperedge_weight, membership, h.size(0))
        return divergence

    def compute_hypergraph_gradient(self, psi, hyperedges, degrees, hyperedge_weight=None, membership=None):
        device = psi.device
        deg_eps = degrees + 1e-8
        grads = []
        for e_idx, nodes in enumerate(hyperedges):
            m = len(nodes)
            if m < 2:
                grads.append(torch.zeros(self.hidden_dim, device=device))
                continue
            edge_w = 1.0 if hyperedge_weight is None else float(hyperedge_weight[e_idx].item())
            mu_vals = [1.0] * m if membership is None else [float(membership[e_idx, n].item()) for n in nodes]
            scaled_feats = torch.stack([(psi[n] * mu_vals[idx]) / torch.sqrt(deg_eps[n]) for idx, n in enumerate(nodes)], dim=0)
            mean_scaled = scaled_feats.mean(dim=0)
            grad_e = (scaled_feats - mean_scaled).sum(dim=0)
            grads.append(math.sqrt(edge_w)/math.sqrt(float(m)) * grad_e)
        return torch.stack(grads, dim=0)

    def compute_diffusion_tensor(self, psi, hyperedges, membership=None):
        device = psi.device
        G_list = []
        d_k = float(self.hidden_dim)
        scale = 1.0 / math.sqrt(d_k)
        for e_idx, nodes in enumerate(hyperedges):
            m = len(nodes)
            if m < 2:
                G_list.append(torch.tensor(1.0, device=device))
                continue
            keys = torch.stack([self.W_K(psi[v]) for v in nodes], dim=0)
            queries = torch.stack([self.W_Q(psi[v]) for v in nodes], dim=0)
            attn = (keys @ queries.t()) * scale
            attn = attn - torch.diag(torch.diag(attn))
            attn_row_soft = F.softmax(attn, dim=1)
            pairwise_sum = attn_row_soft.sum()
            G_val = torch.clamp(pairwise_sum / float(m), min=1e-8)
            G_list.append(G_val)
        return torch.stack(G_list, dim=0)

    def compute_divergence(self, grad, G, hyperedges, degrees, hyperedge_weight=None, membership=None, num_nodes=None):
        device = grad.device
        deg_eps = degrees + 1e-8
        divergence = torch.zeros(num_nodes, self.hidden_dim, device=device)
        for e_idx, nodes in enumerate(hyperedges):
            m = len(nodes)
            if m == 0:
                continue
            edge_w = 1.0 if hyperedge_weight is None else float(hyperedge_weight[e_idx].item())
            sqrt_edge_w = math.sqrt(edge_w)
            mu_vals = [1.0]*m if membership is None else [float(membership[e_idx, n].item()) for n in nodes]
            g_e = grad[e_idx]
            G_e = float(G[e_idx].item())
            for idx, node in enumerate(nodes):
                node_deg = float(deg_eps[node].item())
                denom = max(1.0, (m - 1) * node_deg)
                norm_term = math.sqrt(denom)
                mu_node = mu_vals[idx]
                contrib = (sqrt_edge_w * mu_node / norm_term) * (G_e * g_e)
                divergence[node] += contrib
        return divergence

    def get_hyperedge_structure(self, hyperedge_index):
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
        device = hyperedge_index.device
        degrees = torch.zeros(num_nodes, device=device)
        if hyperedge_index.size(1) == 0:
            return degrees
        nodes = hyperedge_index[1]
        ones = torch.ones_like(nodes, dtype=degrees.dtype, device=device)
        degrees = degrees.scatter_add(0, nodes, ones)
        return degrees
