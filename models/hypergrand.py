"""
hypergrand.py — HyperGRAND-v2
==============================
Simplified HypergraphGRAND model.

Changes from v1:
  - Removed: integration_scheme, integrator_kwargs, learnable_T,
              track_energy, size_enc_dim, topk, attention_mode
  - Added: learnable_mu=True by default (joint E(psi,mu) framework)
  - Residual skip (beta) and per-layer learnable alpha retained
  - Public API (forward signature) unchanged from v1
"""

import torch
import torch.nn as nn
from typing import Optional

from .layers import HypergraphDiffusionLayer


class HypergraphGRAND(nn.Module):
    """
    Hypergraph Graph Neural Diffusion (HyperGRAND) — v2.

    Implements the PDE:
        dpsi/dt = -div[ G_theta(psi) * grad(psi) ]

    discretised as L explicit Euler steps, one per diffusion layer, with a
    residual skip connection to prevent over-smoothing:

        h_out = h_L + beta * h_0

    The joint variational energy E(psi, mu) is minimised implicitly during
    training via alternating updates of psi (diffusion layers) and mu
    (the learnable membership W_mu inside each layer).

    Args:
        input_dim   : dimension of raw input features
        hidden_dim  : internal feature dimension d
        num_layers  : number of diffusion steps L
        alpha       : initial step size (learnable log_alpha per layer)
        dropout     : dropout probability after each layer norm
        num_heads   : attention heads for G_theta
        beta_init   : initial value for residual skip scalar
        learnable_mu: if True, jointly optimise mu (default True)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 3,
        alpha: float = 0.1,
        dropout: float = 0.1,
        num_heads: int = 1,
        beta_init: float = 0.1,
        learnable_mu: bool = True,
    ) -> None:
        super().__init__()
        self.input_dim   = input_dim
        self.hidden_dim  = hidden_dim
        self.out_dim     = hidden_dim
        self.num_layers  = num_layers

        self.input_transform = nn.Linear(input_dim, hidden_dim)

        self.diffusion_layers = nn.ModuleList([
            HypergraphDiffusionLayer(
                hidden_dim=hidden_dim,
                alpha=alpha,
                dropout=dropout,
                num_heads=num_heads,
                learnable_mu=learnable_mu,
            )
            for _ in range(num_layers)
        ])

        # Learnable residual skip scalar — prevents over-smoothing
        self.beta = nn.Parameter(torch.tensor(beta_init))

    def forward(
        self,
        x: torch.Tensor,
        hyperedge_index: torch.Tensor,
        hyperedge_weight: Optional[torch.Tensor] = None,
        membership: Optional[torch.Tensor] = None,   # kept for API compat, unused in v2
    ) -> torch.Tensor:
        """
        Args:
            x               : [N, input_dim]  raw node features
            hyperedge_index : [2, M]           COO incidence: row=edge, col=node
            hyperedge_weight: [E]              optional per-edge weights
            membership      : ignored (kept for API compatibility with v1 callers)

        Returns:
            h : [N, hidden_dim]  node embeddings
        """
        h      = self.input_transform(x)
        h_init = h.clone()

        for layer in self.diffusion_layers:
            h = layer(h, hyperedge_index, hyperedge_weight)

        h = h + self.beta * h_init
        return h


def create_hypergrand_model(
    input_dim: int,
    hidden_dim: int = 32,
    num_layers: int = 3,
    alpha: float = 0.1,
    dropout: float = 0.1,
    num_heads: int = 1,
    beta_init: float = 0.1,
    learnable_mu: bool = True,
    # Legacy kwargs accepted but ignored (for backward compat with exp002 MODEL_CONFIG)
    scheme: Optional[str] = None,
    **kwargs,
) -> HypergraphGRAND:
    """
    Factory function for HyperGRAND-v2.

    The `scheme` and any other v1-only kwargs are accepted silently and ignored,
    so existing callers (exp002_clustering_benchmark.py) need no changes beyond
    updating CODEBASE_DIR.
    """
    return HypergraphGRAND(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        alpha=alpha,
        dropout=dropout,
        num_heads=num_heads,
        beta_init=beta_init,
        learnable_mu=learnable_mu,
    )
