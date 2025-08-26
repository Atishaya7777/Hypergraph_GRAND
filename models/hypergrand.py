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
        integrator_kwargs: Optional[dict] = None
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.out_dim = hidden_dim 
        self.num_layers = num_layers
        self.alpha = alpha
        self.dropout = dropout

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
                integrator_kwargs
            )
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
            layer.reset_integrator()
        
        for layer in self.diffusion_layers:
            h = layer(h, hyperedge_index, hyperedge_weight, membership)
        
        # Return latent representations
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
    **kwargs
) -> HypergraphGRAND:
    """
    Factory function to create HyperGRAND models with different integration schemes
    
    Args:
        scheme: One of ["explicit", "implicit", "multistep", "adaptive"]
        **kwargs: Additional arguments for specific integrators
    """
    
    scheme_defaults = {
        "implicit": {"max_iter": 10, "tol": 1e-6},
        "adaptive": {"min_alpha": 0.01, "max_alpha": 0.5, "tol": 1e-4}
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
        dropout=dropout
    )
