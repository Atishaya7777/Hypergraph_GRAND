import torch
import torch.nn as nn
from typing import Optional
from .layers import HypergraphDiffusionLayer

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
        dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.out_dim = hidden_dim 
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
        
        # Return latent representations
        return h
