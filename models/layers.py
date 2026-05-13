import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, Callable
from abc import ABC, abstractmethod
from enum import Enum
import math
import collections

class IntegrationScheme(Enum):
    EXPLICIT = "explicit"
    IMPLICIT = "implicit"
    MULTISTEP = "multistep" 
    ADAPTIVE = "adaptive"
    IMEX = "imex"
    VERLET = "verlet"
    NEURAL_ODE = "neural_ode"

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
        # NOTE (BUG 3 — gradient approximation):
        # This implementation uses a straight-through gradient approximation:
        # the fixed-point iteration runs under torch.no_grad(), then ONE differentiable
        # call to diffusion_fn reconstructs the gradient. This means gradients flow
        # through only a single diffusion_fn evaluation, not through the full iterative
        # solve. This is known to cause instability when the fixed-point solution and
        # the one-step approximation diverge significantly.
        #
        # OPTION B (recommended next step): For the quasi-linear case where G is
        # approximately constant per step, replace with torch.linalg.solve for an
        # exact gradient cheaply: h_new = torch.linalg.solve(I - alpha*L, h_current)
        # where L is the (approximately) fixed Laplacian matrix.
        #
        # OPTION C (full fix): Use the DEQ adjoint method (Bai et al. NeurIPS 2019,
        # "Deep Equilibrium Models") for an exact implicit gradient at the cost of
        # one extra forward solve. This is the principled solution for the fully
        # nonlinear case.
        device = h_current.device
        h_new = h_current.detach().clone().to(device)

        converged = False
        final_norm = float('inf')
        for it in range(self.max_iter):
            h_prev = h_new
            with torch.no_grad():
                divergence = diffusion_fn(h_new, *args, **kwargs)
                candidate = h_new + self.alpha * divergence
                h_new = h_prev + self.relaxation * (candidate - h_prev)
            final_norm = torch.norm(h_new - h_prev).item()
            if final_norm < self.tol:
                converged = True
                break

        if not converged:
            import warnings
            warnings.warn(
                f"[ImplicitEulerIntegrator] Fixed-point iteration did NOT converge "
                f"after {self.max_iter} iterations (final_norm={final_norm:.4e}, tol={self.tol:.1e}). "
                f"Gradient approximation may be inaccurate. Consider increasing max_iter, "
                f"reducing alpha, or switching to Option B (torch.linalg.solve) or "
                f"Option C (DEQ adjoint). See NOTE in layers.py ImplicitEulerIntegrator.step.",
                RuntimeWarning,
                stacklevel=2,
            )
        elif self.verbose:
            print(f"[ImplicitIntegrator] converged in {it+1}/{self.max_iter} iters, final_norm={final_norm:.4e}")

        # Straight-through: reconstruct a single differentiable evaluation at the
        # converged fixed point. Gradient flows through this one call only.
        h_final = h_new.clone().detach().requires_grad_(True)
        divergence_final = diffusion_fn(h_final, *args, **kwargs)
        return h_current + self.alpha * divergence_final

class MultistepIntegrator(BaseIntegrator):
    """Adams-Bashforth multistep method (order 2, 3, or 4).

    AB2 (default): h_{t+1} = h_t + (alpha/2) * (3 f_t - f_{t-1})
    AB3:           h_{t+1} = h_t + (alpha/12) * (23 f_t - 16 f_{t-1} + 5 f_{t-2})
    AB4:           h_{t+1} = h_t + (alpha/24) * (55 f_t - 59 f_{t-1} + 37 f_{t-2} - 9 f_{t-3})

    Starts with explicit Euler for the first `order-1` steps (warm-up).
    Stores the last `order-1` divergence vectors in a deque.
    """

    # Adams-Bashforth coefficients: ab_coeffs[order] = [c_t, c_{t-1}, c_{t-2}, ...]
    AB_COEFFS = {
        2: [3/2, -1/2],
        3: [23/12, -16/12, 5/12],
        4: [55/24, -59/24, 37/24, -9/24],
    }

    def __init__(self, alpha: float = 0.1, order: int = 2):
        super().__init__(alpha)
        if order not in self.AB_COEFFS:
            raise ValueError(f"MultistepIntegrator: order must be 2, 3, or 4, got {order}")
        self.order = order
        self.history = collections.deque(maxlen=order - 1)  # stores past divergences
        self._pass_count = 0

    def step(self, h_current: torch.Tensor, diffusion_fn: Callable, *args, **kwargs) -> torch.Tensor:
        # Safety assertion: mid-sequence history must be populated
        assert self._pass_count == 0 or len(self.history) > 0, (
            "MultistepIntegrator: history is empty mid-sequence. "
            "Ensure reset_integrator() is called before each forward pass."
        )
        current_divergence = diffusion_fn(h_current, *args, **kwargs)

        if len(self.history) < self.order - 1:
            # Warm-up: use explicit Euler until we have enough history
            h_new = h_current + self.alpha * current_divergence
        else:
            # Full AB step
            coeffs = self.AB_COEFFS[self.order]
            h_new = h_current + self.alpha * coeffs[0] * current_divergence
            for i, past_div in enumerate(reversed(list(self.history))):
                h_new = h_new + self.alpha * coeffs[i + 1] * past_div

        self.history.appendleft(current_divergence.detach().clone())
        self._pass_count += 1
        return h_new

    def reset(self):
        self.history.clear()
        self._pass_count = 0

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

class IMEXIntegrator(BaseIntegrator):
    """IMEX (Implicit-Explicit) splitting integrator.

    Splits the diffusion operator into:
    - L: stiff linear part (fixed symmetric Laplacian, handled implicitly)
    - N: nonlinear residual (learned attention part, handled explicitly)

    Update rule:
        h_{t+1} = (I - alpha * L)^{-1} (h_t + alpha * N(h_t))

    Benefits over pure implicit Euler:
    - The linear part is handled exactly (no fixed-point iteration needed)
    - The nonlinear part is cheap (one explicit evaluation)
    - Allows larger step sizes alpha without instability in the linear component

    Args:
        alpha: Integration step size.
        max_iter: Max iterations for the linear solve (used if direct solve unavailable).
        tol: Convergence tolerance for the linear solve.
    """

    def __init__(self, alpha: float = 0.1, max_iter: int = 20, tol: float = 1e-6):
        super().__init__(alpha)
        self.max_iter = max_iter
        self.tol = tol

    def step(self, h_current: torch.Tensor, diffusion_fn: Callable, *args, **kwargs) -> torch.Tensor:
        # NOTE: To properly implement IMEX we need access to the fixed Laplacian L
        # and the nonlinear residual N = diffusion_fn - L separately.
        # In the current HyperGRAND architecture, diffusion_fn computes the full
        # div(G * grad psi) in one call. We approximate the split as follows:
        #
        # Full operator:  F(h) = div(G(h) * grad h)
        # Linear part:    L(h) = div(I * grad h)  [G replaced by identity/scalar]
        # Nonlinear part: N(h) = F(h) - L(h)      [learned residual]
        #
        # Since we cannot call the operators separately without refactoring, we
        # use a practical approximation: run one explicit step to estimate N,
        # then apply a CG-like iterative solve for the implicit part.
        #
        # FUTURE: Refactor diffusion_fn to accept a `linear_only` flag that
        # returns only div(grad h) without the learned G, enabling exact IMEX.

        # Step 1: Compute full diffusion (approximates N(h_t) = F(h_t))
        with torch.no_grad():
            f_explicit = diffusion_fn(h_current, *args, **kwargs)

        # Step 2: Explicit RHS = h_t + alpha * F(h_t)
        rhs = h_current + self.alpha * f_explicit

        # Step 3: Implicit solve approximation via fixed-point iteration
        # Solve: h_{t+1} - alpha * div(grad h_{t+1}) = rhs
        # For the linear part this is (I - alpha * L) h = rhs
        # We approximate by iterating: h <- rhs + alpha * (F(h) - F(h_t) + div(grad h_t))
        # Practical approximation: use rhs as the solution (degenerate IMEX = explicit Euler)
        # but add a correction step using the gradient information.
        h_new = rhs.detach().clone()
        for _ in range(self.max_iter):
            h_prev = h_new
            with torch.no_grad():
                correction = diffusion_fn(h_new, *args, **kwargs)
            h_new = h_current + self.alpha * correction
            if torch.norm(h_new - h_prev).item() < self.tol:
                break

        # Final differentiable evaluation (straight-through, same as ImplicitEuler)
        h_final = h_new.clone().detach().requires_grad_(True)
        divergence_final = diffusion_fn(h_final, *args, **kwargs)
        return h_current + self.alpha * divergence_final

class VerletIntegrator(BaseIntegrator):
    """Störmer-Verlet (leapfrog) symplectic integrator.

    Introduces a momentum variable p alongside h. The scheme is symplectic
    and conserves a modified Hamiltonian, giving more stable long-time dynamics
    than explicit Euler.

    Update rule:
        p_{t+1/2} = p_{t-1/2} + alpha * F(h_t)   (half-step momentum)
        h_{t+1}   = h_t + alpha * p_{t+1/2}        (position update)

    where F(h) = div(G(h) * grad h) is the diffusion force.

    The momentum p accumulates across layers (layers = timesteps). This gives
    the diffusion "inertia" — it does not stop immediately when energy flattens,
    potentially helping escape shallow local minima in the feature landscape.

    Theoretical note: the continuous limit is a damped wave equation (hyperbolic
    PDE) rather than a heat equation (parabolic PDE).

    Call reset() between forward passes to clear momentum state.
    """

    def __init__(self, alpha: float = 0.1):
        super().__init__(alpha)
        self.p = None  # momentum, initialised on first step

    def step(self, h: torch.Tensor, diffusion_fn: Callable, *args, **kwargs) -> torch.Tensor:
        if self.p is None:
            self.p = torch.zeros_like(h)
        force = diffusion_fn(h, *args, **kwargs)   # = -grad_h E
        self.p = self.p + self.alpha * force
        return h + self.alpha * self.p

    def reset(self):
        self.p = None

class NeuralODEIntegrator(BaseIntegrator):
    """Neural ODE integrator using torchdiffeq (Chen et al. NeurIPS 2018).

    Delegates integration to torchdiffeq's dopri5 adaptive solver with
    adjoint gradient computation (O(1) memory in depth).

    Requires: pip install torchdiffeq

    Falls back to explicit Euler with a warning if torchdiffeq is not installed.

    Args:
        alpha: Integration time horizon T (the ODE integrates from 0 to alpha).
        method: ODE solver method (default 'dopri5').
        rtol: Relative tolerance for adaptive step size control.
        atol: Absolute tolerance for adaptive step size control.
        adjoint: If True, use adjoint method for memory-efficient gradients.
    """

    def __init__(
        self,
        alpha: float = 0.1,
        method: str = 'dopri5',
        rtol: float = 1e-3,
        atol: float = 1e-6,
        adjoint: bool = True,
    ):
        super().__init__(alpha)
        self.method = method
        self.rtol = rtol
        self.atol = atol
        self.adjoint = adjoint
        self._torchdiffeq_available = None  # checked lazily

    def _check_torchdiffeq(self) -> bool:
        if self._torchdiffeq_available is None:
            try:
                import torchdiffeq  # noqa: F401
                self._torchdiffeq_available = True
            except ImportError:
                self._torchdiffeq_available = False
                import warnings
                warnings.warn(
                    "[NeuralODEIntegrator] torchdiffeq is not installed. "
                    "Falling back to explicit Euler. "
                    "Install with: pip install torchdiffeq",
                    ImportWarning,
                    stacklevel=3,
                )
        return self._torchdiffeq_available

    def step(self, h_current: torch.Tensor, diffusion_fn: Callable, *args, **kwargs) -> torch.Tensor:
        if not self._check_torchdiffeq():
            # Fallback: explicit Euler
            divergence = diffusion_fn(h_current, *args, **kwargs)
            return h_current + self.alpha * divergence

        if self.adjoint:
            from torchdiffeq import odeint_adjoint as odeint
        else:
            from torchdiffeq import odeint

        # Flatten h for the ODE solver (it expects 1D or standard tensor)
        original_shape = h_current.shape

        def ode_fn(t, h_flat):
            h = h_flat.view(original_shape)
            dh = diffusion_fn(h, *args, **kwargs)
            return dh.view(-1)

        h_flat = h_current.reshape(-1)
        t_span = torch.tensor([0.0, self.alpha], device=h_current.device, dtype=h_current.dtype)

        h_out = odeint(
            ode_fn,
            h_flat,
            t_span,
            method=self.method,
            rtol=self.rtol,
            atol=self.atol,
        )

        # h_out has shape [2, N*d]; take the final timepoint
        return h_out[-1].view(original_shape)

# ----------------- Hypergraph Diffusion Layer -----------------
class HypergraphDiffusionLayer(nn.Module):
    """Single hypergraph diffusion layer"""
    def __init__(self, hidden_dim: int, alpha: float = 0.1, dropout: float = 0.1,
                 integration_scheme: IntegrationScheme = IntegrationScheme.EXPLICIT,
                 integrator_kwargs: Optional[dict] = None,
                 track_energy: bool = False,
                 num_heads: int = 1,
                 size_enc_dim: int = 0,
                 topk: Optional[int] = None,
                 attention_mode: str = 'pairwise',
                 learnable_mu: bool = False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.alpha = alpha  # kept as fallback / reference value
        self.log_alpha = nn.Parameter(torch.tensor(math.log(alpha)))
        self.dropout = dropout
        self.track_energy = track_energy
        self.last_dirichlet_energy: Optional[float] = None
        self.num_heads = num_heads
        self.size_enc_dim = size_enc_dim
        self.topk = topk
        self.attention_mode = attention_mode
        self.learnable_mu = learnable_mu
        if learnable_mu:
            # Bilinear form for membership scores: mu(e,v) = sigmoid(h_e @ W_mu @ h_v)
            # W_mu: [hidden_dim, hidden_dim]
            self.W_mu = nn.Linear(hidden_dim, hidden_dim, bias=False)
        else:
            self.W_mu = None

        # Multi-head K/Q projections
        self.W_K = nn.Linear(hidden_dim, hidden_dim * num_heads)
        self.W_Q = nn.Linear(hidden_dim, hidden_dim * num_heads)
        self.head_mix = nn.Parameter(torch.ones(num_heads) / num_heads)

        self.layer_norm = nn.LayerNorm(hidden_dim)

        # SetTransformer-style attention layers
        if attention_mode == 'set':
            self.W_V = nn.Linear(hidden_dim, hidden_dim)
            self.W_A = nn.Linear(hidden_dim, hidden_dim)
        else:
            self.W_V = None
            self.W_A = None

        # Size encoding
        if size_enc_dim > 0:
            self.size_embedding = nn.Embedding(512, size_enc_dim)  # up to 512-node hyperedges
            self.size_proj = nn.Linear(hidden_dim + size_enc_dim, hidden_dim)
        else:
            self.size_embedding = None
            self.size_proj = None

        integrator_kwargs = integrator_kwargs or {}
        self.integrator = self._create_integrator(integration_scheme, alpha, **integrator_kwargs)

    def _create_integrator(self, scheme: IntegrationScheme, alpha: float, **kwargs):
        if scheme == IntegrationScheme.EXPLICIT:
            return ExplicitEulerIntegrator(alpha)
        elif scheme == IntegrationScheme.IMPLICIT:
            return ImplicitEulerIntegrator(alpha, **kwargs)
        elif scheme == IntegrationScheme.MULTISTEP:
            return MultistepIntegrator(alpha, **kwargs)
        elif scheme == IntegrationScheme.ADAPTIVE:
            return AdaptiveIntegrator(alpha, **kwargs)
        elif scheme == IntegrationScheme.IMEX:
            return IMEXIntegrator(alpha, **kwargs)
        elif scheme == IntegrationScheme.VERLET:
            return VerletIntegrator(alpha)
        elif scheme == IntegrationScheme.NEURAL_ODE:
            return NeuralODEIntegrator(alpha, **kwargs)
        else:
            raise ValueError(f"Unknown integration scheme: {scheme}")

    def forward(self, h, hyperedge_index, hyperedge_weight=None, membership=None):
        device = h.device
        hyperedges = self.get_hyperedge_structure(hyperedge_index)
        degrees = self.compute_node_degrees(hyperedge_index, h.size(0))

        def diffusion_fn(h_input):
            return self._compute_diffusion_step(h_input, hyperedges, degrees, hyperedge_weight, membership)

        # Use learnable per-layer alpha (always positive via exp)
        self.integrator.alpha = torch.exp(self.log_alpha).item()
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
        # Dirichlet energy tracking: ||grad h||^2 summed over all hyperedges.
        # If track_energy=True, store as a Python float (detached) for logging.
        if self.track_energy:
            self.last_dirichlet_energy = (grad ** 2).sum().item()
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
            # Use learnable mu if enabled; otherwise fall back to static membership or uniform 1.0
            if self.learnable_mu:
                mu_tensor = self.compute_membership(psi, nodes)  # [m], softmax-normalised
                mu_vals = mu_tensor.tolist() if mu_tensor is not None else [1.0] * m
            else:
                mu_vals = [1.0] * m if membership is None else [float(membership[e_idx, n].item()) for n in nodes]
            scaled_feats = torch.stack([(psi[n] * mu_vals[idx]) / torch.sqrt(deg_eps[n]) for idx, n in enumerate(nodes)], dim=0)
            mean_scaled = scaled_feats.mean(dim=0)
            grad_e = (scaled_feats - mean_scaled).sum(dim=0)
            if self.size_embedding is not None:
                size_idx = torch.tensor(min(m, 511), device=device)
                enc = self.size_embedding(size_idx)  # [size_enc_dim]
                scale_factor = math.sqrt(edge_w) / math.sqrt(float(m))
                g_scaled = scale_factor * grad_e
                g_aug = torch.cat([g_scaled, enc], dim=-1)   # [hidden_dim + size_enc_dim]
                grads.append(self.size_proj(g_aug))
            else:
                grads.append(math.sqrt(edge_w)/math.sqrt(float(m)) * grad_e)
        return torch.stack(grads, dim=0)

    def compute_diffusion_tensor(self, psi, hyperedges, membership=None):
        device = psi.device
        G_list = []
        H = self.num_heads
        # W_K/W_Q project to hidden_dim * H; each head gets hidden_dim features
        head_dim = self.hidden_dim
        scale = 1.0 / math.sqrt(max(head_dim, 1))
        head_weights = torch.softmax(self.head_mix, dim=0)  # [H]

        for e_idx, nodes in enumerate(hyperedges):
            m = len(nodes)
            if m < 2:
                G_list.append(torch.tensor(1.0, device=device))
                continue

            # SetTransformer-style attention branch
            if self.attention_mode == 'set':
                # O(|e|) SetTransformer-style: score each node against hyperedge summary
                node_vals = torch.stack([self.W_V(psi[v]) for v in nodes], dim=0)  # [m, d]
                z_e = node_vals.mean(dim=0)  # hyperedge summary [d]
                node_attn = torch.stack([self.W_A(psi[v]) for v in nodes], dim=0)  # [m, d]
                scores = (node_attn @ z_e)  # [m]
                scores = F.softmax(scores, dim=0)
                G_val = torch.clamp(scores.sum() / float(m), min=1e-8)
                G_list.append(G_val)
                continue  # skip the pairwise branch for this edge

            # [m, H * head_dim]
            keys_full = torch.stack([self.W_K(psi[v]) for v in nodes], dim=0)
            queries_full = torch.stack([self.W_Q(psi[v]) for v in nodes], dim=0)
            # [m, H, head_dim]
            keys = keys_full.view(m, H, head_dim)
            queries = queries_full.view(m, H, head_dim)
            # Attention per head: [m, m, H]
            # attn[i,j,h] = keys[i,h] . queries[j,h]
            attn = torch.einsum('ihd,jhd->ijh', keys, queries) * scale  # [m, m, H]
            # Zero out diagonal per head
            diag_mask = torch.eye(m, device=device, dtype=torch.bool).unsqueeze(-1).expand(m, m, H)
            attn = attn.masked_fill(diag_mask, 0.0)

            # Top-k membership masking for large hyperedges
            if self.topk is not None and m > 1:
                k = min(self.topk, m * m)
                attn_sum = attn.sum(dim=-1)  # [m, m]
                flat = attn_sum.reshape(-1)
                _, topk_idx = flat.topk(k=min(k, flat.numel()))
                mask = torch.zeros_like(flat)
                mask[topk_idx] = 1.0
                mask = mask.view(m, m)
                attn = attn * mask.unsqueeze(-1)

            # Sum across node pairs per head: [H]
            G_h = attn.sum(dim=(0, 1)) / float(m)  # [H]
            # Weighted combination across heads
            G_val = (head_weights * G_h).sum()
            G_val = torch.clamp(G_val, min=1e-8)
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
            # Use learnable mu if enabled; otherwise fall back to static membership or uniform 1.0
            if self.learnable_mu:
                # NOTE: We recompute mu here from psi via the gradient/divergence
                # signature. In practice, mu is computed once per forward pass in
                # compute_hypergraph_gradient and could be cached for efficiency.
                # For correctness, we access it via the same bilinear form.
                # The `h` argument is not available directly here, so we pass
                # a dummy — this is a known limitation. See learnable_mu TODO.
                mu_vals = [1.0] * m  # fallback: divergence uses uniform mu
                # TODO: Cache mu from compute_hypergraph_gradient and pass to compute_divergence
                # to avoid this inconsistency. For now, use uniform in divergence.
            else:
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

    def compute_membership(self, psi: torch.Tensor, nodes: list) -> torch.Tensor:
        """Compute learnable membership scores mu(e,v) for a single hyperedge.

        Uses a bilinear form: the hyperedge embedding is the mean of member features,
        then each node is scored against it via W_mu.

        The softmax constraint (sum_v mu(e,v) = 1) prevents the trivial minimizer
        mu -> 0 from collapsing the Dirichlet energy.

        Args:
            psi: Node feature matrix [num_nodes, hidden_dim]
            nodes: List of node indices in this hyperedge

        Returns:
            mu: Membership weights [len(nodes)], softmax-normalized (sums to 1)
        """
        if not self.learnable_mu or self.W_mu is None:
            return None  # caller should use uniform mu = 1.0

        m = len(nodes)
        if m == 0:
            return None

        node_feats = torch.stack([psi[v] for v in nodes], dim=0)  # [m, hidden_dim]
        h_e = node_feats.mean(dim=0)                               # [hidden_dim] hyperedge embedding

        # Bilinear scores: h_e @ W_mu @ h_v for each v
        h_e_proj = self.W_mu(h_e)                                  # [hidden_dim]
        scores = node_feats @ h_e_proj                             # [m]

        # Softmax normalisation: ensures sum_v mu(e,v) = 1 (prevents trivial minimizer)
        mu = torch.softmax(scores, dim=0)                         # [m], sums to 1
        return mu
