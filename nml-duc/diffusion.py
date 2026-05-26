"""Diffusion trajectory model: GNN context encoder + transformer denoiser.

Architecture
------------
* `GNNContextEncoder` runs the same heterogeneous RGAT-GRU stack used by
  `model.NBAModel`, but only over the 8 context frames — no autoregressive
  rollout, no delta head. Returns per-entity context embeddings `c[B, N, D]`.

* `const_vel_anchor` builds a constant-velocity baseline `μ[B, H, N, 2]` from
  the last context frame and its velocity. The diffusion model is trained to
  denoise *residuals to this anchor* (`y - μ`), not raw positions, so the
  network only has to model deviations from physics — much faster convergence.

* `TrajectoryDenoiser` is a small transformer over `H*N` tokens (one per
  (horizon_step, entity)). At each diffusion timestep it takes a noisy residual
  trajectory + per-entity context + sinusoidal time embedding and predicts the
  clean residual (x0-prediction). Output layer is zero-initialized so the
  initial prediction is "stay on the constant-velocity anchor".

* `DiffusionNBA` composes the above with a cosine variance schedule. Training:
  sample t ~ Uniform{0, T-1}, noise the residual, MSE on x0_hat. Inference:
  DDIM sampling (η=0) for K trajectories, averaged for the Kaggle submission.

The model exposes a `forward(X, y=None, tf_prob=0.0)` that returns either the
training loss (when y is given) or a sampled `[H, B*N, 2]` trajectory (when y
is None). This matches `NBATrainer._predict_positions` which calls
`model(self._normalize_xy(X))`.
"""

import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from model import (HeteroSTGCNCell, build_hetero_edges, compute_edge_attr,
                  compute_possession)


# ---------------------------------------------------------------------------
# Context encoder
# ---------------------------------------------------------------------------


class GNNContextEncoder(nn.Module):
    """RGAT-GRU encoder over the C context frames; outputs per-entity context.

    Feature flags mirror the subset of `NBAModel`'s that are most useful for
    encoding the context (kinematics + possession + edge_attr). The heavier
    flags from NBAModel (handler-relations, ball-centric, basket-dist,
    rel-vel-edge-attr) are deliberately omitted for this initial diffusion
    cut — they're orthogonal improvements that can be ported in later.
    """

    def __init__(self, input_dim: int = 4, state_dim: int = 128,
                 context_size: int = 8, num_layers: int = 1,
                 use_velocity: bool = True, use_kinematics: bool = True,
                 use_edge_attr: bool = True, use_possession: bool = True,
                 coord_scale: float = 1.0):
        super().__init__()
        self.context_size = context_size
        self.state_dim = int(state_dim)
        self.num_layers = max(1, int(num_layers))
        self.use_velocity = bool(use_velocity)
        self.use_kinematics = bool(use_kinematics)
        self.use_edge_attr = bool(use_edge_attr)
        self.use_possession = bool(use_possession)
        self.coord_scale = float(coord_scale)

        extra = 0
        if use_velocity:    extra += 2
        if use_kinematics:  extra += 5  # acc(2) + speed(1) + heading(2)
        if use_possession:  extra += 2
        cell_input_dim = input_dim + extra
        edge_dim = 3 if use_edge_attr else None
        self.cells = nn.ModuleList([
            HeteroSTGCNCell(cell_input_dim if i == 0 else state_dim, state_dim,
                            num_relations=5, edge_dim=edge_dim)
            for i in range(self.num_layers)
        ])

    def forward(self, X: Tensor) -> Tensor:
        """X: [B, C, N, F] in the model's normalized space. Returns [B, N, D]."""
        B, C, N, _ = X.shape
        device = X.device
        team = X[:, 0, :, 3]
        edge_index, edge_type = build_hetero_edges(team, device)
        hs = [torch.zeros(B * N, self.state_dim, device=device)
              for _ in range(self.num_layers)]
        static = X[:, 0, :, 2:]
        prev_pos = X[:, 0, :, :2]
        prev_vel = torch.zeros(B, N, 2, device=device)
        need_vel = self.use_velocity or self.use_kinematics

        for t in range(C):
            pos = X[:, t, :, :2]
            parts = [pos]
            vel = pos - prev_pos if (need_vel and t > 0) else torch.zeros_like(pos)
            if self.use_velocity:
                parts.append(vel)
            if self.use_kinematics:
                acc = vel - prev_vel if t > 1 else torch.zeros_like(pos)
                speed = torch.linalg.vector_norm(vel, dim=-1, keepdim=True)
                heading = vel / (speed + 1e-6)
                parts += [acc, speed, heading]
            parts.append(static)
            if self.use_possession:
                parts.append(compute_possession(pos, team,
                                                tau=5.0 / self.coord_scale))
            feat = torch.cat(parts, dim=-1)
            x = feat.reshape(B * N, -1)
            edge_attr = (compute_edge_attr(pos.reshape(B * N, 2), edge_index)
                         if self.use_edge_attr else None)
            for li, cell in enumerate(self.cells):
                new_h = cell(x, hs[li], edge_index, edge_type,
                             edge_attr=edge_attr)
                hs[li] = new_h
                x = new_h if li == 0 else x + new_h
            prev_pos = pos
            prev_vel = vel
        return hs[-1].reshape(B, N, self.state_dim)


# ---------------------------------------------------------------------------
# Anchor: constant-velocity baseline trajectory
# ---------------------------------------------------------------------------


def const_vel_anchor(X: Tensor, horizon: int) -> Tensor:
    """Linear extrapolation from the last context frame using its 1-step velocity.

    X: [B, C, N, F]. Returns `[B, H, N, 2]` predicted positions in the same
    (normalized) units as the input.
    """
    last_pos = X[:, -1, :, :2]                                # [B, N, 2]
    last_vel = X[:, -1, :, :2] - X[:, -2, :, :2]              # [B, N, 2]
    steps = torch.arange(1, horizon + 1, device=X.device,
                         dtype=X.dtype).view(1, horizon, 1, 1)
    return last_pos.unsqueeze(1) + steps * last_vel.unsqueeze(1)


# ---------------------------------------------------------------------------
# Denoiser
# ---------------------------------------------------------------------------


class SinusoidalTimeEmbed(nn.Module):
    """Standard sinusoidal embedding of the diffusion timestep (DDPM-style)."""

    def __init__(self, dim: int):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f'SinusoidalTimeEmbed dim must be even, got {dim}')
        self.dim = dim

    def forward(self, t: Tensor) -> Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, device=t.device, dtype=torch.float32) / half)
        a = t.float()[:, None] * freqs[None]
        return torch.cat([a.sin(), a.cos()], dim=-1)


class TrajectoryDenoiser(nn.Module):
    """Transformer over (H, N) tokens that predicts the clean residual.

    Tokens at position (h, n) are the noisy residual position `xt[:, h, n, :]`
    projected to D dims, plus the per-entity context broadcast across H,
    plus a horizon-step positional embedding, plus a per-batch diffusion-time
    embedding (broadcast across H*N). A stack of standard transformer encoder
    blocks does joint spatiotemporal self-attention. The output linear layer
    is zero-initialized so the initial prediction is x0_hat ≈ 0 — i.e. the
    sampler initially decodes back to the constant-velocity anchor.
    """

    def __init__(self, state_dim: int = 128, horizon: int = 12,
                 num_entities: int = 11, num_layers: int = 4,
                 num_heads: int = 4, ff_mult: int = 4, dropout: float = 0.0):
        super().__init__()
        if state_dim % num_heads != 0:
            raise ValueError(f'state_dim {state_dim} not divisible by num_heads {num_heads}')
        D = state_dim
        self.state_dim = D
        self.horizon = horizon
        self.num_entities = num_entities
        self.in_proj = nn.Linear(2, D)
        self.ctx_proj = nn.Linear(D, D)
        self.h_emb = nn.Embedding(horizon, D)
        self.t_embed = SinusoidalTimeEmbed(D)
        self.t_proj = nn.Sequential(nn.Linear(D, D), nn.SiLU(), nn.Linear(D, D))
        layer = nn.TransformerEncoderLayer(
            d_model=D, nhead=num_heads, dim_feedforward=ff_mult * D,
            dropout=dropout, batch_first=True, norm_first=True,
            activation='gelu')
        self.blocks = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.out = nn.Linear(D, 2)
        # x0-prediction: zero-init so denoiser output starts at 0; samples
        # initially decode to the anchor trajectory.
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x_noisy: Tensor, t: Tensor, context: Tensor) -> Tensor:
        """x_noisy: [B, H, N, 2]   t: [B] long   context: [B, N, D]."""
        B, H, N, _ = x_noisy.shape
        D = self.state_dim
        device = x_noisy.device
        h_idx = torch.arange(H, device=device).view(1, H, 1).expand(B, H, N)
        tok = (self.in_proj(x_noisy)
               + self.ctx_proj(context).unsqueeze(1)     # [B,1,N,D] → broadcast
               + self.h_emb(h_idx))
        tok = tok + self.t_proj(self.t_embed(t)).view(B, 1, 1, D)
        tok = self.blocks(tok.reshape(B, H * N, D))
        return self.out(tok).reshape(B, H, N, 2)


# ---------------------------------------------------------------------------
# Variance schedule
# ---------------------------------------------------------------------------


class CosineSchedule:
    """DDPM cosine ᾱ schedule (Nichol & Dhariwal 2021). Plain dataclass — no
    nn.Module so the buffer isn't accidentally swept into a state_dict."""

    def __init__(self, T: int = 1000, s: float = 0.008):
        t = torch.arange(T + 1) / T
        f = torch.cos((t + s) / (1.0 + s) * math.pi / 2) ** 2
        ab = (f / f[0]).clamp(min=1e-5)
        self.alpha_bar = ab  # CPU tensor; callers .to(device) on lookup
        self.T = int(T)

    def q_sample(self, x0: Tensor, t: Tensor, eps: Tensor) -> Tensor:
        ab = self.alpha_bar.to(x0.device)[t].view(-1, 1, 1, 1)
        return ab.sqrt() * x0 + (1.0 - ab).sqrt() * eps


# ---------------------------------------------------------------------------
# Top-level model
# ---------------------------------------------------------------------------


class DiffusionNBA(nn.Module):
    """Diffusion trajectory model = GNN context encoder + transformer denoiser.

    Train (`y` provided): sample t, noise the residual y-anchor, return MSE
    between predicted and true residual. Inference (`y=None`): DDIM-sample
    `num_samples` trajectories and return their mean as a `[H, B*N, 2]` tensor.

    Output / submission contract `[H, B*N, 2]` is shared with `NBAModel` and
    `NBATrainer.get_kaggle_submission` — do not change without updating both.
    """

    def __init__(self, encoder: GNNContextEncoder, denoiser: TrajectoryDenoiser,
                 horizon: int, num_entities: int = 11, T: int = 1000,
                 ddim_steps: int = 20, num_samples: int = 8):
        super().__init__()
        if encoder.state_dim != denoiser.state_dim:
            raise ValueError(
                f'encoder.state_dim ({encoder.state_dim}) must match '
                f'denoiser.state_dim ({denoiser.state_dim})')
        if denoiser.horizon != horizon:
            raise ValueError(
                f'denoiser.horizon ({denoiser.horizon}) must match '
                f'horizon ({horizon})')
        self.encoder = encoder
        self.denoiser = denoiser
        self.sched = CosineSchedule(T)
        self.horizon_size = int(horizon)
        self.num_entities = int(num_entities)
        # Expose a few fields the trainer / inference code reads directly.
        self.context_size = encoder.context_size
        self.coord_scale = encoder.coord_scale
        self.ddim_steps = int(ddim_steps)
        self.num_samples = int(num_samples)

    # ---- training ----------------------------------------------------------

    def loss(self, X: Tensor, y: Tensor) -> Tensor:
        """X: [B, C, N, F]  y: [B, H, N, F] (or [B, H, N, 2]). All normalized.
        Returns a scalar MSE on the predicted x0 residual."""
        B = X.shape[0]
        device = X.device
        ctx = self.encoder(X)                                  # [B, N, D]
        anchor = const_vel_anchor(X, self.horizon_size)        # [B, H, N, 2]
        x0 = y[..., :2] - anchor                               # residual target
        t = torch.randint(0, self.sched.T, (B,), device=device)
        eps = torch.randn_like(x0)
        xt = self.sched.q_sample(x0, t, eps)
        x0_hat = self.denoiser(xt, t, ctx)
        return F.mse_loss(x0_hat, x0)

    # ---- inference ---------------------------------------------------------

    @torch.no_grad()
    def sample(self, X: Tensor, ddim_steps: Optional[int] = None,
               num_samples: Optional[int] = None) -> Tensor:
        """DDIM sampling (η=0). Returns `[H, B*N, 2]` mean trajectory in
        normalized space — caller multiplies by `coord_scale` if needed."""
        ddim_steps = self.ddim_steps if ddim_steps is None else int(ddim_steps)
        num_samples = self.num_samples if num_samples is None else int(num_samples)
        B, _, N, _ = X.shape
        H = self.horizon_size
        device = X.device
        ctx = self.encoder(X)
        anchor = const_vel_anchor(X, H)
        ab = self.sched.alpha_bar.to(device)
        # DDIM step indices: decreasing from T-1 → 0 in `ddim_steps` jumps.
        ts = torch.linspace(self.sched.T - 1, 0, ddim_steps + 1,
                            device=device).long()

        traj_sum = torch.zeros(B, H, N, 2, device=device)
        for _ in range(num_samples):
            xt = torch.randn(B, H, N, 2, device=device)
            for i in range(ddim_steps):
                t = ts[i]
                t_next = ts[i + 1]
                a_t = ab[t]
                a_n = ab[t_next]
                x0_hat = self.denoiser(xt, t.expand(B), ctx)
                eps_hat = ((xt - a_t.sqrt() * x0_hat)
                           / (1.0 - a_t).clamp(min=1e-8).sqrt())
                xt = a_n.sqrt() * x0_hat + (1.0 - a_n).sqrt() * eps_hat
            traj_sum = traj_sum + (xt + anchor)
        traj = traj_sum / num_samples                          # [B, H, N, 2]
        # Match existing [H, B*N, 2] contract used by MultiStepMSE and CSV writer.
        return traj.permute(1, 0, 2, 3).reshape(H, B * N, 2)

    # ---- trainer-facing entry point ---------------------------------------

    def forward(self, X: Tensor, y: Optional[Tensor] = None,
                tf_prob: float = 0.0) -> Tensor:
        """Dispatch: training loss (y given) vs. sampling (y=None).

        `tf_prob` is accepted only to match `NBAModel.forward`'s signature so
        the trainer can call both models uniformly — it is unused here
        (diffusion has no teacher-forcing schedule)."""
        del tf_prob  # noqa: unused
        if y is not None:
            return self.loss(X, y)
        return self.sample(X)
