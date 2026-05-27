"""
layers.py — HyperGRAND-v2
==========================
Clean rewrite of the hypergraph diffusion layer.

Design principles:
  - All core ops (gradient, diffusion tensor, divergence) are fully vectorized
    over the hyperedge_index COO tensor — no Python for-loops over hyperedges.
  - mu(e,v) is computed once per forward pass and shared by both gradient and
    divergence, making the joint E(psi, mu) claim correct in implementation.
  - learnable_mu=True by default (the central theoretical contribution).
  - Single integration scheme: explicit Euler with learnable per-layer log_alpha.
  - Runs on CPU, CUDA, or MPS — device-agnostic via standard scatter ops.

Notation (matches contexts/notation.md):
  psi       : node features [N, d]
  mu(e,v)   : soft membership of node v in hyperedge e  [num_entries]
  G_theta   : learned diffusion scalar per hyperedge    [E]
  L_theta   : grad* G_theta grad — state-dependent Laplacian
  grad_e    : hyperedge gradient (reference-node formula)
  div       : hypergraph divergence (adjoint of grad)
  d(v)      : weighted degree of node v
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _segment_softmax(scores: torch.Tensor, segment_ids: torch.Tensor,
                     num_segments: int) -> torch.Tensor:
    """
    Softmax within each segment (hyperedge).

    Args:
        scores      : [num_entries]  raw logits
        segment_ids : [num_entries]  edge index for each entry (= hyperedge_index[0])
        num_segments: int            number of hyperedges E

    Returns:
        mu : [num_entries]  softmax-normalised per edge, sums to 1 within each edge
    """
    # Subtract per-segment max for numerical stability
    seg_max = torch.zeros(num_segments, dtype=scores.dtype, device=scores.device)
    seg_max.scatter_reduce_(0, segment_ids, scores, reduce="amax", include_self=True)
    scores_shifted = scores - seg_max[segment_ids]

    exp_scores = scores_shifted.exp()

    # Sum of exp per segment
    seg_sum = torch.zeros(num_segments, dtype=scores.dtype, device=scores.device)
    seg_sum.scatter_add_(0, segment_ids, exp_scores)
    seg_sum = seg_sum.clamp(min=1e-8)

    return exp_scores / seg_sum[segment_ids]


# ---------------------------------------------------------------------------
# HypergraphDiffusionLayer
# ---------------------------------------------------------------------------

class HypergraphDiffusionLayer(nn.Module):
    """
    One explicit-Euler step of the hypergraph diffusion PDE:

        psi_{t+1} = psi_t + alpha * div( G_theta(psi_t) * grad(psi_t) )

    where:
        grad_e psi  = sum_{v in e, v != v0} [mu(e,v)/sqrt(d(v))] * psi_v
                      - [mu(e,v0)/sqrt(d(v0))] * psi_v0   (reference-node formula)
        div_v       = sum_{e: v in e} w(e) * mu(e,v) / sqrt((|e|-1)*d(v))
                      * G_e * grad_e psi
        G_e         = clamp( sum_{i,j in e, i!=j} K_i . Q_j / sqrt(d) / |e|, min=1e-8 )
        mu(e,v)     = softmax_e( h_e_mean @ W_mu @ h_v )   if learnable_mu
                    = 1/|e|                                  otherwise

    All three operations are fully vectorized over the COO hyperedge_index tensor.

    Args:
        hidden_dim   : feature dimension d
        alpha        : initial step size (log-parameterised, per-layer learnable)
        dropout      : dropout rate applied after layer norm
        num_heads    : attention heads for G_theta computation
        learnable_mu : if True, compute mu via bilinear form W_mu (default True)
    """

    def __init__(
        self,
        hidden_dim: int,
        alpha: float = 0.1,
        dropout: float = 0.1,
        num_heads: int = 1,
        learnable_mu: bool = True,
    ):
        super().__init__()
        self.hidden_dim  = hidden_dim
        self.num_heads   = num_heads
        self.learnable_mu = learnable_mu

        # Learnable per-layer step size: always positive via exp
        self.log_alpha = nn.Parameter(torch.tensor(math.log(alpha)))

        # Multi-head K/Q projections for G_theta
        self.W_K = nn.Linear(hidden_dim, hidden_dim * num_heads, bias=False)
        self.W_Q = nn.Linear(hidden_dim, hidden_dim * num_heads, bias=False)
        # Scalar mix across heads
        self.head_mix = nn.Parameter(torch.ones(num_heads) / num_heads)

        # Bilinear membership: mu(e,v) = softmax_e( h_e @ W_mu @ h_v )
        if learnable_mu:
            self.W_mu = nn.Linear(hidden_dim, hidden_dim, bias=False)
        else:
            self.W_mu = None

        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout_p  = dropout

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        psi: torch.Tensor,
        hyperedge_index: torch.Tensor,
        hyperedge_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            psi             : [N, d]   node features
            hyperedge_index : [2, M]   COO: row=edge id, col=node id
            hyperedge_weight: [E]      optional per-edge weights (default 1.0)
        Returns:
            psi_new         : [N, d]
        """
        N, d  = psi.shape
        e_idx = hyperedge_index[0]   # [M]  edge id per entry
        v_idx = hyperedge_index[1]   # [M]  node id per entry
        M     = e_idx.shape[0]
        E     = int(e_idx.max().item()) + 1

        if hyperedge_weight is None:
            edge_w = psi.new_ones(E)
        else:
            edge_w = hyperedge_weight.to(psi.device)

        # --- 1. Compute mu(e,v) -------------------------------------------
        mu = self._compute_mu(psi, e_idx, v_idx, E)   # [M]

        # --- 2. Compute node degrees d(v) = sum_e mu(e,v) -----------------
        #    (hypergraph weighted degree under current mu)
        deg = psi.new_zeros(N)
        deg.scatter_add_(0, v_idx, mu)
        deg = deg.clamp(min=1e-8)                      # [N]

        # --- 3. Compute hyperedge gradient grad_e -------------------------
        grad = self._compute_gradient(psi, e_idx, v_idx, mu, deg, edge_w, E, M)
        # grad: [E, d]

        # --- 4. Compute diffusion tensor G_theta --------------------------
        G = self._compute_diffusion_tensor(psi, e_idx, v_idx, E)
        # G: [E]

        # --- 5. Compute divergence div_v ----------------------------------
        div = self._compute_divergence(grad, G, e_idx, v_idx, mu, deg, edge_w, N, E)
        # div: [N, d]

        # --- 6. Explicit Euler step ---------------------------------------
        alpha    = self.log_alpha.exp()
        psi_new  = psi + alpha * div
        psi_new  = self.layer_norm(psi_new)
        psi_new  = F.dropout(psi_new, p=self.dropout_p, training=self.training)
        return psi_new

    # ------------------------------------------------------------------
    # Core vectorized operations
    # ------------------------------------------------------------------

    def _compute_mu(
        self,
        psi: torch.Tensor,
        e_idx: torch.Tensor,
        v_idx: torch.Tensor,
        E: int,
    ) -> torch.Tensor:
        """
        Compute soft membership mu(e,v) for all (e,v) entries.

        If learnable_mu:
            h_e = mean of member features for each edge   [E, d]
            score(e,v) = h_e[e] @ W_mu @ psi[v]
            mu = softmax within each edge (sum_v mu(e,v) = 1)

        Else:
            mu = uniform = 1 / |e|  for each entry

        Returns:
            mu : [M]  (M = total entries in hyperedge_index)
        """
        M = e_idx.shape[0]

        if not self.learnable_mu or self.W_mu is None:
            # Uniform: count members per edge
            counts = psi.new_zeros(E)
            counts.scatter_add_(0, e_idx, psi.new_ones(M))
            counts = counts.clamp(min=1.0)
            mu = 1.0 / counts[e_idx]          # [M]
            return mu

        # Learnable bilinear membership
        # h_e = mean pooling of member features per edge
        node_feats = psi[v_idx]               # [M, d]
        h_e_sum = psi.new_zeros(E, psi.shape[1])
        h_e_sum.scatter_add_(0, e_idx.unsqueeze(-1).expand_as(node_feats), node_feats)
        counts = psi.new_zeros(E)
        counts.scatter_add_(0, e_idx, psi.new_ones(M))
        counts = counts.clamp(min=1.0)
        h_e = h_e_sum / counts.unsqueeze(-1)  # [E, d]

        # Bilinear score: score(e,v) = (W_mu h_e[e]) . psi[v]
        h_e_proj   = self.W_mu(h_e)           # [E, d]
        scores     = (h_e_proj[e_idx] * psi[v_idx]).sum(dim=-1)  # [M]

        # Softmax within each edge
        mu = _segment_softmax(scores, e_idx, E)   # [M], sum per edge = 1
        return mu

    def _compute_gradient(
        self,
        psi: torch.Tensor,
        e_idx: torch.Tensor,
        v_idx: torch.Tensor,
        mu: torch.Tensor,
        deg: torch.Tensor,
        edge_w: torch.Tensor,
        E: int,
        M: int,
    ) -> torch.Tensor:
        """
        Compute hyperedge gradients grad_e for all edges simultaneously.

        Reference-node formula (fixes the always-zero bug of deviation-from-mean):
            scaled_v = mu(e,v) / sqrt(d(v)) * psi_v    for each entry (e,v)
            grad_e   = sum_{v in e} scaled_v
                       - |e| * scaled_{v_ref}            (subtract ref node scaled |e| times)

        Equivalent to sum of pairwise differences anchored to the first listed node,
        but computed entirely with scatter ops.

        Normalised by sqrt(w(e)) / sqrt(|e|) following the standard hypergraph
        gradient normalisation.

        Returns:
            grad : [E, d]
        """
        d = psi.shape[1]

        scaled = mu / deg[v_idx].sqrt()                    # [M]  scalar weight per entry
        scaled_feats = psi[v_idx] * scaled.unsqueeze(-1)  # [M, d]

        # Sum scaled features per edge
        grad_sum = psi.new_zeros(E, d)
        grad_sum.scatter_add_(
            0,
            e_idx.unsqueeze(-1).expand(M, d),
            scaled_feats,
        )                                                  # [E, d]

        # Reference-node correction:
        # For each edge e, subtract the contribution of the reference node |e| times.
        # We use the first occurrence of each edge in the COO as the reference node.
        # Build a mask: is this entry the FIRST occurrence of its edge id?
        # Equivalent: entry i is reference iff e_idx[i] != e_idx[i-1] (or i==0)
        is_ref = psi.new_zeros(M, dtype=torch.bool)
        is_ref[0] = True
        if M > 1:
            is_ref[1:] = e_idx[1:] != e_idx[:-1]

        # Count members per edge
        counts = psi.new_zeros(E)
        counts.scatter_add_(0, e_idx, psi.new_ones(M))    # [E]

        # Subtract ref_scaled * counts[e]  from grad_sum
        ref_scaled_feats = scaled_feats[is_ref]            # [E, d]  (one per edge)
        correction = ref_scaled_feats * counts.unsqueeze(-1)  # [E, d]
        grad = grad_sum - correction                        # [E, d]

        # Normalise by sqrt(w(e)) / sqrt(|e|)
        norm = (edge_w.sqrt() / counts.sqrt()).unsqueeze(-1)   # [E, 1]
        grad = grad * norm

        return grad   # [E, d]

    def _compute_diffusion_tensor(
        self,
        psi: torch.Tensor,
        e_idx: torch.Tensor,
        v_idx: torch.Tensor,
        E: int,
    ) -> torch.Tensor:
        """
        Compute per-edge diffusion scalar G_e using multi-head dot-product attention.

        For each edge e:
            keys_i    = W_K psi_i   for i in e
            queries_j = W_Q psi_j   for j in e
            attn_ij   = keys_i . queries_j / sqrt(d)  (i != j)
            G_h       = sum_{i!=j} attn_ij^h / |e|   per head h
            G_e       = clamp( sum_h mix_h * G_h, min=1e-8 )

        Fully vectorized: all node projections in one batched matmul;
        per-edge sums via scatter_add.

        Returns:
            G : [E]
        """
        M      = e_idx.shape[0]
        H      = self.num_heads
        d      = self.hidden_dim
        scale  = 1.0 / math.sqrt(d)

        node_feats = psi[v_idx]                # [M, d]

        # Project to K/Q spaces — single batched matmul each
        keys_all    = self.W_K(node_feats)     # [M, H*d]
        queries_all = self.W_Q(node_feats)     # [M, H*d]

        # Reshape to [M, H, d]
        keys    = keys_all.view(M, H, d)
        queries = queries_all.view(M, H, d)

        head_weights = torch.softmax(self.head_mix, dim=0)   # [H]

        # For each edge we need sum_{i,j in e, i!=j} K_i . Q_j
        # = (sum_i K_i) . (sum_j Q_j) - sum_i K_i . Q_i  (subtract diagonal)
        # This is the standard O(|e|) trick.

        # Sum K and Q per edge: [E, H, d]
        K_sum = psi.new_zeros(E, H, d)
        K_sum.scatter_add_(
            0,
            e_idx.unsqueeze(-1).unsqueeze(-1).expand(M, H, d),
            keys,
        )
        Q_sum = psi.new_zeros(E, H, d)
        Q_sum.scatter_add_(
            0,
            e_idx.unsqueeze(-1).unsqueeze(-1).expand(M, H, d),
            queries,
        )

        # (sum K) . (sum Q) per edge per head: [E, H]
        KQ_total = (K_sum * Q_sum).sum(dim=-1)   # [E, H]

        # Diagonal: sum_i K_i . Q_i per edge per head: [E, H]
        KQ_diag_entries = (keys * queries).sum(dim=-1)   # [M, H]
        KQ_diag = psi.new_zeros(E, H)
        KQ_diag.scatter_add_(
            0,
            e_idx.unsqueeze(-1).expand(M, H),
            KQ_diag_entries,
        )

        attn_offdiag = (KQ_total - KQ_diag) * scale   # [E, H]

        # Count members per edge for normalisation
        counts = psi.new_zeros(E)
        counts.scatter_add_(0, e_idx, psi.new_ones(M))
        counts = counts.clamp(min=1.0)

        G_per_head = attn_offdiag / counts.unsqueeze(-1)   # [E, H]

        # Mix heads: [E]
        G = (G_per_head * head_weights.unsqueeze(0)).sum(dim=-1)
        G = G.clamp(min=1e-8)
        return G   # [E]

    def _compute_divergence(
        self,
        grad: torch.Tensor,
        G: torch.Tensor,
        e_idx: torch.Tensor,
        v_idx: torch.Tensor,
        mu: torch.Tensor,
        deg: torch.Tensor,
        edge_w: torch.Tensor,
        N: int,
        E: int,
    ) -> torch.Tensor:
        """
        Compute hypergraph divergence div_v for all nodes.

        div_v = sum_{e: v in e} w(e) * mu(e,v) / sqrt((|e|-1)*d(v))
                * G_e * grad_e psi

        Uses the SAME mu as compute_gradient — this is the fix for the joint
        E(psi, mu) claim (previously divergence used uniform mu regardless).

        Returns:
            divergence : [N, d]
        """
        M = e_idx.shape[0]
        d = grad.shape[1]

        # Count members per edge
        counts = grad.new_zeros(E)
        counts.scatter_add_(0, e_idx, grad.new_ones(M))
        counts = counts.clamp(min=1.0)

        # Scalar coefficient per entry: w(e) * mu(e,v) / sqrt((|e|-1) * d(v))
        # |e| - 1 clamped to >= 1 to handle singleton edges gracefully
        m_minus_1 = (counts[e_idx] - 1.0).clamp(min=1.0)
        coeff = (
            edge_w[e_idx].sqrt()
            * mu
            / (m_minus_1 * deg[v_idx]).sqrt()
        )   # [M]

        # G_e * grad_e: broadcast G over entries of each edge
        Gg = G[e_idx].unsqueeze(-1) * grad[e_idx]   # [M, d]

        # Weighted contribution per entry
        contrib = coeff.unsqueeze(-1) * Gg           # [M, d]

        # Scatter-add to nodes
        divergence = grad.new_zeros(N, d)
        divergence.scatter_add_(
            0,
            v_idx.unsqueeze(-1).expand(M, d),
            contrib,
        )
        return divergence   # [N, d]
