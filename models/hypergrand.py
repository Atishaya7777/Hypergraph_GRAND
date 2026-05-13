import torch
import torch.nn as nn
from typing import Optional, Union
from .layers import HypergraphDiffusionLayer, IntegrationScheme

class HypergraphGRAND(nn.Module):
    """
    Hypergraph Graph Neural Diffusion (HyperGRAND) implementation
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 3,
        alpha: float = 0.1,
        dropout: float = 0.1,
        integration_scheme: Union[IntegrationScheme, str] = IntegrationScheme.EXPLICIT,
        integrator_kwargs: Optional[dict] = None,
        track_energy: bool = False,
        num_heads: int = 1,
        beta_init: float = 0.1,
        size_enc_dim: int = 0,
        topk: Optional[int] = None,
        attention_mode: str = 'pairwise',
        learnable_T: bool = False,
        learnable_mu: bool = False,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.out_dim = hidden_dim 
        self.num_layers = num_layers
        self.alpha = alpha
        self.dropout = dropout
        self.track_energy = track_energy
        self.num_heads = num_heads
        self.learnable_mu = learnable_mu
        self.log_T = nn.Parameter(torch.tensor(0.0)) if learnable_T else None

        if isinstance(integration_scheme, str):
            integration_scheme = IntegrationScheme(integration_scheme)

        self.integration_scheme = integration_scheme
        self.integrator_kwargs = integrator_kwargs
        
        self.input_transform = nn.Linear(input_dim, hidden_dim)
        self.diffusion_layers = nn.ModuleList([
            HypergraphDiffusionLayer(
                hidden_dim,
                alpha,
                dropout,
                integration_scheme,
                integrator_kwargs,
                track_energy=track_energy,
                num_heads=num_heads,
                size_enc_dim=size_enc_dim,
                topk=topk,
                attention_mode=attention_mode,
                learnable_mu=learnable_mu,
            )
            for _ in range(num_layers)
        ])
        self.beta = nn.Parameter(torch.tensor(beta_init))
    
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
            layer.reset_integrator()

        if self.log_T is not None:
            T_effective = torch.exp(self.log_T)
            alpha_effective = (T_effective / self.num_layers).item()
            for layer in self.diffusion_layers:
                layer.integrator.alpha = alpha_effective
        
        for layer in self.diffusion_layers:
            h = layer(h, hyperedge_index, hyperedge_weight, membership)
        
        # Residual skip connection from initial embedding (prevents oversmoothing).
        # beta is a learnable scalar, initialised to beta_init (default 0.1).
        h = h + self.beta * h_init
        return h

    def set_integration_scheme(
        self,
        scheme: Union[IntegrationScheme, str], **kwargs
        ):
        """Change integration scheme for all layers"""
        if isinstance(scheme, str):
            scheme = IntegrationScheme(scheme)

        for layer in self.diffusion_layers:
            layer.integrator = layer._create_integrator(scheme, self.alpha, **kwargs)

# factory function for easy creation
def create_hypergrand_model(
    input_dim: int,
    hidden_dim: int,
    scheme: str = "explicit",
    num_layers: int = 3,
    alpha: float = 0.1,
    dropout: float = 0.5,
    track_energy: bool = False,
    num_heads: int = 1,
    beta_init: float = 0.1,
    size_enc_dim: int = 0,
    topk: Optional[int] = None,
    attention_mode: str = 'pairwise',
    learnable_T: bool = False,
    learnable_mu: bool = False,
    **kwargs
) -> HypergraphGRAND:
    """
    Factory function to create HyperGRAND models with different integration schemes
    
    Args:
        scheme: One of ["explicit", "implicit", "multistep", "adaptive", "imex", "verlet", "neural_ode"]
        track_energy: Enable Dirichlet energy tracking in diffusion layers
        num_heads: Number of attention heads for multi-head diffusion tensor (§A)
        beta_init: Initial value for learnable residual skip scalar (§C)
        size_enc_dim: Dimension of hyperedge size encoding, 0 = disabled (§E)
        topk: Top-k membership masking for large hyperedges, None = disabled (§F)
        attention_mode: 'pairwise' or 'set' for SetTransformer-style attention (§G)
        learnable_T: If True, add a learnable log-time-horizon parameter (§C)
        **kwargs: Additional arguments for specific integrators
    """
    
    scheme_defaults = {
        "implicit": {"max_iter": 10, "tol": 1e-6},
        "adaptive": {"min_alpha": 0.01, "max_alpha": 0.5, "tol": 1e-4},
        "imex": {"max_iter": 20, "tol": 1e-6},
    }
    
    integrator_kwargs = scheme_defaults.get(scheme, {})
    integrator_kwargs.update(kwargs)
    
    return HypergraphGRAND(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        alpha=alpha,
        integration_scheme=scheme,
        integrator_kwargs=integrator_kwargs,
        dropout=dropout,
        track_energy=track_energy,
        num_heads=num_heads,
        beta_init=beta_init,
        size_enc_dim=size_enc_dim,
        topk=topk,
        attention_mode=attention_mode,
        learnable_T=learnable_T,
        learnable_mu=learnable_mu,
    )
