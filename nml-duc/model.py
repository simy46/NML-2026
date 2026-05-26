"""Heterogeneous spatial-temporal GNN for NBA trajectory prediction.

`NBAModel` rolls out a per-timestep heterogeneous relational graph attention
(`RGATConv`) followed by a `GRUCell`, predicting coordinate deltas. The output
contract `[H, B*N, 2]` matches `MultiStepMSE` and the submission code.
"""

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch_geometric.nn import RGATConv


NUM_RELATIONS_BASE = 5
NUM_RELATIONS_HANDLER = 11


def build_hetero_edges(team_batch: Tensor, device):
    """Vectorized construction of heterogeneous edges for a whole batch.

    `team_batch` has shape `(B, N)` with team IDs in {-1, 0, 1}.
    Returns `edge_index [2, B*N*N]` and `edge_type [B*N*N]` with five relation types:
    0 teammate, 1 opponent, 2 player→ball, 3 ball→player, 4 self-loop.

    For the handler-aware 11-relation typing (re-derived per timestep), use
    `compute_handler_edge_type` to overwrite `edge_type` while keeping the
    same `edge_index`.
    """
    B, N = team_batch.shape
    t_i = team_batch.unsqueeze(2)  # source nodes
    t_j = team_batch.unsqueeze(1)  # destination nodes

    rel = torch.zeros((B, N, N), dtype=torch.long, device=device)
    rel[(t_i != 0) & (t_i == t_j)] = 0  # teammate (same team, not ball)
    rel[(t_i != 0) & (t_j != 0) & (t_i != t_j)] = 1  # opponent
    rel[(t_i != 0) & (t_j == 0)] = 2  # player → ball
    rel[(t_i == 0) & (t_j != 0)] = 3  # ball → player
    i_idx = torch.arange(N, device=device)
    rel[:, i_idx, i_idx] = 4  # self-loop
    edge_type = rel.view(-1)

    src = torch.arange(N, device=device).view(1, N, 1).expand(B, N, N)
    dst = torch.arange(N, device=device).view(1, 1, N).expand(B, N, N)
    batch_offset = (torch.arange(B, device=device) * N).view(B, 1, 1)
    src = (src + batch_offset).reshape(-1)
    dst = (dst + batch_offset).reshape(-1)
    edge_index = torch.stack([src, dst], dim=0)
    return edge_index, edge_type


def compute_handler_mask(pos: Tensor, team_batch: Tensor) -> Tensor:
    """Boolean `[B, N]` mask of the ball-handler in each sample.

    Defined as the player (team_id != 0) closest to the ball in Euclidean
    distance. The ball itself is always False.
    """
    ball_mask = (team_batch == 0)                                # [B, N]
    ball_pos = (pos * ball_mask.unsqueeze(-1)).sum(dim=1, keepdim=True)  # [B, 1, 2]
    dist = torch.linalg.vector_norm(pos - ball_pos, dim=-1)      # [B, N]
    dist = dist.masked_fill(ball_mask, float('inf'))             # exclude ball
    handler_idx = dist.argmin(dim=1, keepdim=True)               # [B, 1]
    return torch.zeros_like(ball_mask).scatter_(1, handler_idx, True)


def compute_handler_edge_type(team_batch: Tensor, handler_mask: Tensor,
                              device) -> Tensor:
    """Handler-aware 11-relation `edge_type` (re-derived per timestep).

    Same edge ordering as `build_hetero_edges` (src→dst), but each
    player↔ball, teammate, and opponent relation is split by whether the
    src or dst is the current ball-handler:
       0 teammate                          (neither endpoint is handler)
       1 teammate, dst is handler          (off-ball teammate → handler)
       2 teammate, src is handler          (handler → off-ball teammate)
       3 opponent                          (neither endpoint is handler)
       4 opponent, dst is handler          (defender → handler, "on-ball defense")
       5 opponent, src is handler          (handler → defender)
       6 player → ball                     (off-ball player → ball)
       7 player → ball, src is handler     (handler → ball)
       8 ball → player                     (ball → off-ball player)
       9 ball → player, dst is handler     (ball → handler)
      10 self-loop
    """
    B, N = team_batch.shape
    t_i = team_batch.unsqueeze(2)
    t_j = team_batch.unsqueeze(1)
    h_i = handler_mask.unsqueeze(2)
    h_j = handler_mask.unsqueeze(1)

    same_team = (t_i != 0) & (t_i == t_j)
    opp_team  = (t_i != 0) & (t_j != 0) & (t_i != t_j)
    p_to_b    = (t_i != 0) & (t_j == 0)
    b_to_p    = (t_i == 0) & (t_j != 0)

    rel = torch.zeros((B, N, N), dtype=torch.long, device=device)
    # Order matters: assign the generic bucket first, then overwrite with
    # handler-aware subtypes.
    rel[same_team]        = 0
    rel[same_team & h_j]  = 1
    rel[same_team & h_i]  = 2
    rel[opp_team]         = 3
    rel[opp_team & h_j]   = 4
    rel[opp_team & h_i]   = 5
    rel[p_to_b]           = 6
    rel[p_to_b & h_i]     = 7
    rel[b_to_p]           = 8
    rel[b_to_p & h_j]     = 9
    i_idx = torch.arange(N, device=device)
    rel[:, i_idx, i_idx] = 10
    return rel.view(-1)


def compute_player_player_mask(team_batch: Tensor, edge_index: Tensor) -> Tensor:
    """Bool `[E]` mask: True iff both endpoints are players (not the ball)
    and the edge is not a self-loop. Used by the ball-centric topology to
    select which edges are subject to the distance gate.

    The mask depends only on team IDs (static across time), so callers can
    cache it once per batch rather than recomputing per step.
    """
    team_flat = team_batch.reshape(-1)
    src, dst = edge_index[0], edge_index[1]
    return (team_flat[src] != 0) & (team_flat[dst] != 0) & (src != dst)


def compute_edge_attr(pos_flat: Tensor, edge_index: Tensor,
                      vel_flat: Tensor = None) -> Tensor:
    """Per-edge geometric features.

    Default (`vel_flat=None`): `[Δx, Δy, ‖Δ‖]` of shape `[E, 3]`.

    With `vel_flat` (per-node velocity `[B*N, 2]`): also appends
    `[Δvx, Δvy, closing_speed]` for shape `[E, 6]`, where
    `closing_speed = (v_dst - v_src) · Δp̂` is the signed approach rate
    along the edge direction (positive = moving apart, since `Δp = dst - src`).
    """
    src, dst = edge_index[0], edge_index[1]
    delta = pos_flat[dst] - pos_flat[src]
    dist = torch.linalg.vector_norm(delta, dim=-1, keepdim=True)
    if vel_flat is None:
        return torch.cat([delta, dist], dim=-1)
    dvel = vel_flat[dst] - vel_flat[src]
    unit = delta / (dist + 1e-6)
    closing = (dvel * unit).sum(dim=-1, keepdim=True)
    return torch.cat([delta, dist, dvel, closing], dim=-1)


def compute_possession(pos: Tensor, team_batch: Tensor, tau: float = 5.0) -> Tensor:
    """Per-node `[dist_to_ball, soft_possession]` of shape `[B, N, 2]`.

    `soft_possession` is a softmin over players (ball masked to -inf so it
    contributes 0 mass), giving the ball-handler a value close to 1.
    """
    ball_mask = (team_batch == 0)                              # [B, N]
    ball_pos = (pos * ball_mask.unsqueeze(-1)).sum(dim=1, keepdim=True)  # [B, 1, 2]
    dist = torch.linalg.vector_norm(pos - ball_pos, dim=-1)    # [B, N]
    logits = (-dist / tau).masked_fill(ball_mask, float('-inf'))
    soft_poss = torch.softmax(logits, dim=-1)
    return torch.stack([dist, soft_poss], dim=-1)


class HeteroSTGCNCell(nn.Module):
    """One spatial-temporal step: relational GAT over entities, then a GRU update."""

    def __init__(self, input_dim: int, state_dim: int, num_relations: int = 5,
                 edge_dim: int = None):
        super().__init__()
        self.rgcn = RGATConv(input_dim, state_dim, num_relations=num_relations,
                             edge_dim=edge_dim)
        self.gru = nn.GRUCell(state_dim, state_dim)
        self.ln = nn.LayerNorm(state_dim)

    def forward(self, x: Tensor, h_prev: Tensor, edge_index: Tensor, edge_type: Tensor,
                edge_attr: Tensor = None) -> Tensor:
        s = F.relu(self.rgcn(x, edge_index, edge_type, edge_attr=edge_attr))
        h = self.gru(s, h_prev)
        return self.ln(h)


class HeteroGraphTransformerCell(nn.Module):
    """One spatial-temporal step: relation-aware multi-head self-attention over
    the per-sample entity graph, then a GRU update.

    Drop-in for `HeteroSTGCNCell`: same `(x, h_prev, edge_index, edge_type,
    edge_attr)` signature and same output shape `[B*N, state_dim]`. The
    "graph element" is preserved by:
      * per-(relation, head) additive bias on attention logits (the same
        five or eleven relation types as the RGAT path), and
      * hard-masking attention to -inf on (dst, src) pairs that are absent
        from `edge_index` — so ball-centric pruning carries through to the
        attention mask unchanged.

    Optional `edge_attr` is projected to a per-head bias and added on top of
    the relation bias, mirroring the RGAT cell's edge-feature handling.
    """

    def __init__(self, input_dim: int, state_dim: int, num_relations: int = 5,
                 edge_dim: int = None, num_heads: int = 4,
                 num_entities: int = 11, dropout: float = 0.0):
        super().__init__()
        if state_dim % num_heads != 0:
            raise ValueError(
                f'state_dim ({state_dim}) must be divisible by num_heads ({num_heads})')
        self.state_dim = state_dim
        self.num_heads = num_heads
        self.head_dim = state_dim // num_heads
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.edge_dim = edge_dim

        self.input_proj = nn.Linear(input_dim, state_dim)
        self.qkv = nn.Linear(state_dim, 3 * state_dim, bias=True)
        self.out_proj = nn.Linear(state_dim, state_dim)
        # Per-(relation, head) additive bias on attention logits — the
        # "edge encoding" that makes attention heterogeneous.
        self.rel_bias = nn.Embedding(num_relations, num_heads)
        nn.init.zeros_(self.rel_bias.weight)
        if edge_dim is not None:
            self.edge_proj = nn.Linear(edge_dim, num_heads)
            nn.init.zeros_(self.edge_proj.weight)
            nn.init.zeros_(self.edge_proj.bias)
        self.attn_drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.gru = nn.GRUCell(state_dim, state_dim)
        self.ln = nn.LayerNorm(state_dim)

    def forward(self, x: Tensor, h_prev: Tensor, edge_index: Tensor, edge_type: Tensor,
                edge_attr: Tensor = None) -> Tensor:
        N = self.num_entities
        B = x.shape[0] // N
        H = self.num_heads
        D = self.head_dim

        h = self.input_proj(x).view(B, N, self.state_dim)
        qkv = self.qkv(h).view(B, N, 3, H, D)
        q, k, v = qkv.unbind(dim=2)                         # each [B, N, H, D]
        q = q.transpose(1, 2)                               # [B, H, N, D]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # scores[b, head, i, j] = attention from query i (dst) to key j (src).
        scores = torch.einsum('bhid,bhjd->bhij', q, k) / (D ** 0.5)

        # Decompose flat edge endpoints back to (batch, local src/dst). The
        # edge convention in `build_hetero_edges` is `edge_index[0]=src`,
        # `edge_index[1]=dst` (meaning u → v), so v attends to u — bias goes
        # at position (dst=v, src=u).
        src = edge_index[0]
        dst = edge_index[1]
        b_idx = src // N
        src_local = src % N
        dst_local = dst % N
        edge_bias = self.rel_bias(edge_type)                # [E, H]
        if edge_attr is not None and self.edge_dim is not None:
            edge_bias = edge_bias + self.edge_proj(edge_attr)
        bias = torch.full((B, H, N, N), float('-inf'),
                          device=x.device, dtype=scores.dtype)
        # Advanced indexing assignment: the [:, ...] axis broadcasts edge_bias
        # of shape [E, H] across heads. Missing (dst, src) pairs stay -inf and
        # contribute zero post-softmax.
        bias[b_idx, :, dst_local, src_local] = edge_bias.to(scores.dtype)

        attn = torch.softmax(scores + bias, dim=-1)
        attn = self.attn_drop(attn)
        out = torch.einsum('bhij,bhjd->bhid', attn, v)      # [B, H, N, D]
        out = out.transpose(1, 2).contiguous().view(B, N, self.state_dim)
        out = F.relu(self.out_proj(out)).view(B * N, self.state_dim)

        h_new = self.gru(out, h_prev)
        return self.ln(h_new)


class NBAModel(nn.Module):
    """Heterogeneous spatial-temporal GNN; predicts per-step coordinate deltas.

    Per-step node features stacked into the RGAT input, controlled by flags:
      * `pos` (always, 2)
      * `vel`        if `use_velocity`     (+2)
      * `acc, |vel|, (cosθ, sinθ)` if `use_kinematics` (+5; vel is computed
        regardless so this flag is independent of `use_velocity`)
      * `static`     (input_dim − 2)
      * `dist_to_ball, soft_possession` if `use_possession` (+2)
      * `dist_to_basket_left, dist_to_basket_right` if `use_basket_dist` (+2)

    With `use_edge_attr`, each RGAT message also sees `[Δx, Δy, ‖Δ‖]` for its
    edge, recomputed every step from the rolled-out positions. When
    `use_rel_vel_edge_attr` is also on, three more dims `[Δvx, Δvy,
    closing_speed]` are appended (requires `use_edge_attr=True`).

    `use_handler_relations` expands the heterogeneous edge typing from 5 to
    11 relations by splitting each player↔ball / teammate / opponent edge
    by whether the src or dst is the current ball-handler (closest player
    to the ball; re-derived per timestep). See `compute_handler_edge_type`
    for the relation table.

    `cell_type='transformer'` swaps the spatial layer of every cell from
    `HeteroSTGCNCell` (RGAT) to `HeteroGraphTransformerCell` (relation-aware
    multi-head self-attention with per-(relation, head) bias). The temporal
    GRU + LayerNorm are unchanged, all five/eleven relation types and edge
    attributes still apply, and ball-centric pruning still works via the
    attention mask. `num_heads` controls head count for the transformer cell
    and must divide `state_dim`; it is ignored when `cell_type='gnn'`.

    `use_ball_centric` switches the topology from fully-connected to
    ball-centric: every player↔ball edge and self-loop is always kept,
    while player↔player edges (teammate + opponent) are pruned per step
    if their endpoints are farther apart than `player_edge_radius` feet.
    The mask is re-derived from the current (rolled-out) positions, so
    the player-subgraph evolves over time. Set `player_edge_radius` very
    large (e.g. 100.0) to recover the fully-connected behavior for
    ablation while still going through the masking code path.
    """

    def __init__(self, input_dim: int = 4, output_dim: int = 2, state_dim: int = 64,
                 context_size: int = 8, horizon_size: int = 12,
                 num_entities: int = 11, hidden_dim: int = 64,
                 use_velocity: bool = True, use_kinematics: bool = True,
                 use_edge_attr: bool = True, use_possession: bool = True,
                 use_basket_dist: bool = False,
                 use_ball_head: bool = False,
                 use_handler_relative_ball: bool = False,
                 use_na_ball: bool = False,
                 use_handler_relations: bool = False,
                 use_rel_vel_edge_attr: bool = False,
                 use_ball_centric: bool = False,
                 player_edge_radius: float = 20.0,
                 cell_type: str = 'gnn', num_heads: int = 4,
                 num_layers: int = 2, coord_scale: float = 1.0):
        super().__init__()
        self.context_size = context_size
        self.horizon_size = horizon_size
        self.num_entities = num_entities
        self.state_dim = state_dim
        self.use_velocity = use_velocity
        self.use_kinematics = use_kinematics
        self.use_edge_attr = use_edge_attr
        self.use_possession = use_possession
        self.use_basket_dist = bool(use_basket_dist)
        self.use_ball_head = bool(use_ball_head)
        self.use_handler_relative_ball = bool(use_handler_relative_ball)
        self.use_na_ball = bool(use_na_ball)
        self.use_handler_relations = bool(use_handler_relations)
        self.use_rel_vel_edge_attr = bool(use_rel_vel_edge_attr)
        self.use_ball_centric = bool(use_ball_centric)
        # Store the radius in feet (user-facing units); convert to the
        # model's internal coord units lazily in `_ball_centric_keep`.
        self.player_edge_radius = float(player_edge_radius)
        if self.use_ball_centric and self.player_edge_radius <= 0:
            raise ValueError(
                f'player_edge_radius must be > 0, got {player_edge_radius}')
        if self.use_na_ball and self.use_handler_relative_ball:
            raise ValueError('use_na_ball and use_handler_relative_ball are mutually exclusive')
        if self.use_rel_vel_edge_attr and not self.use_edge_attr:
            raise ValueError('use_rel_vel_edge_attr requires use_edge_attr=True')
        if cell_type not in ('gnn', 'transformer'):
            raise ValueError(f"cell_type must be 'gnn' or 'transformer', got {cell_type!r}")
        self.cell_type = cell_type
        self.num_heads = int(num_heads)
        self.num_layers = max(1, int(num_layers))
        # Inputs and outputs are assumed already divided by `coord_scale` (the
        # trainer does this at the boundary). Only used to convert the 5-foot
        # possession softmin temperature into whatever units we're now in.
        self.coord_scale = float(coord_scale)

        extra = 0
        if use_velocity:    extra += 2
        if use_kinematics:  extra += 5  # acc(2) + speed(1) + heading(2)
        if use_possession:  extra += 2
        if use_basket_dist: extra += 2  # distance to each basket
        cell_input_dim = input_dim + extra
        # Edge attr is 3 dims by default ([Δx, Δy, ‖Δ‖]); with rel-vel it adds
        # [Δvx, Δvy, closing_speed] for a total of 6.
        edge_dim = (6 if self.use_rel_vel_edge_attr else 3) if use_edge_attr else None
        num_relations = (NUM_RELATIONS_HANDLER if self.use_handler_relations
                         else NUM_RELATIONS_BASE)

        # Basket centers in raw court units, normalized by coord_scale so they
        # live in the same space as the (already-normalized) positions the
        # model sees. NBA half-court is 47 ft; rim is 5.25 ft from baseline.
        if self.use_basket_dist:
            basket_xy = torch.tensor([[-41.75, 0.0], [41.75, 0.0]]) / self.coord_scale
            self.register_buffer('basket_xy', basket_xy)

        # Stack of HeteroSTGCNCells. Layer 0 ingests the raw per-step features;
        # deeper layers consume the state-dim output of the layer below, so the
        # graph receptive field grows by one hop per layer. Each layer keeps its
        # own recurrent hidden state.
        def _make_cell(in_dim: int):
            if self.cell_type == 'transformer':
                return HeteroGraphTransformerCell(
                    in_dim, state_dim, num_relations=num_relations,
                    edge_dim=edge_dim, num_heads=self.num_heads,
                    num_entities=self.num_entities)
            return HeteroSTGCNCell(in_dim, state_dim,
                                   num_relations=num_relations,
                                   edge_dim=edge_dim)
        self.cells = nn.ModuleList([
            _make_cell(cell_input_dim if i == 0 else state_dim)
            for i in range(self.num_layers)
        ])
        self.proj = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        # Optional ball-specific delta head. The shared RGAT stack still does
        # the spatial reasoning; only the final state→Δxy projection branches
        # by entity type, so the ball can specialize its motion model (high-
        # frequency, impulsive) without affecting the player head.
        if self.use_ball_head:
            self.proj_ball = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim),
            )
        # Handler-relative ball reparameterization. When on, the ball's
        # predicted position is `sum_i w_i p_hat_i + offset`, where `w_i` is
        # the soft-possession over players (zero on the ball) and the offset
        # is a small learned residual from this dedicated head. During
        # possession the offset is near zero, so the ball inherits the
        # much-better player-position predictions.
        if self.use_handler_relative_ball:
            self.proj_ball_offset = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim),
            )
            # Zero-init the final layer so initial offsets are 0 and training
            # starts at "ball is exactly at the handler-weighted center".
            nn.init.zeros_(self.proj_ball_offset[-1].weight)
            nn.init.zeros_(self.proj_ball_offset[-1].bias)
        # Non-autoregressive ball head. At the last context step we decode all
        # H horizon ball positions in one shot from the ball-node hidden state,
        # as deltas from the last context ball position. Output is independent
        # per horizon step → no rollout-drift compounding for the ball. Players
        # still roll out autoregressively; their graph features see the NA ball
        # so possession / player→ball edges remain consistent.
        if self.use_na_ball:
            self.proj_ball_na = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, horizon_size * output_dim),
            )
            # Zero-init final layer → initial prediction = "ball stays put".
            nn.init.zeros_(self.proj_ball_na[-1].weight)
            nn.init.zeros_(self.proj_ball_na[-1].bias)

    def forward(self, X: Tensor, y: Tensor = None, tf_prob: float = 0.0) -> Tensor:
        """X: [B, C, N, F]. Returns predicted positions [H, B*N, 2].

        Scheduled sampling (training-only): if `y` is provided, `tf_prob > 0`, and
        `self.training`, then at each horizon step a per-sample Bernoulli decides
        whether the *input* position is the previous-step prediction (default) or
        the ground-truth position drawn from `y[:, t - context_size, :, :2]`. The
        appended predictions are still the model's own outputs, so the loss is
        unchanged — only what the rollout conditions on changes.
        """
        B, _, N, _ = X.shape
        T = self.context_size + self.horizon_size
        device = X.device
        use_tf = self.training and y is not None and tf_prob > 0.0

        team_batch = X[:, 0, :, 3]  # static team IDs
        edge_index, edge_type = build_hetero_edges(team_batch, device)
        # Player-player mask is static for the whole sequence (depends only
        # on team IDs and node identity), so cache it once. The distance
        # gate that uses it IS time-varying — see the loop below.
        pp_mask = (compute_player_player_mask(team_batch, edge_index)
                   if self.use_ball_centric else None)
        radius_scaled = (self.player_edge_radius / self.coord_scale
                         if self.use_ball_centric else None)
        ball_mask_flat = ((team_batch == 0).reshape(B * N, 1)
                          if self.use_ball_head else None)
        ball_mask3 = ((team_batch == 0).unsqueeze(-1)
                      if self.use_na_ball else None)  # [B, N, 1]
        ball_traj = None  # populated at t == context_size - 1 if use_na_ball

        hs = [torch.zeros(B * N, self.state_dim, device=device) for _ in range(self.num_layers)]
        static = X[:, 0, :, 2:]  # [B, N, F-2], reused during the horizon
        preds = []
        curr_coords = X[:, 0, :, :2]  # overwritten before the horizon starts
        prev_pos = X[:, 0, :, :2]
        prev_vel = torch.zeros(B, N, 2, device=device)
        need_vel = (self.use_velocity or self.use_kinematics
                    or self.use_rel_vel_edge_attr)

        for t in range(T):
            if t < self.context_size:
                pos = X[:, t, :, :2]
            else:
                pos = curr_coords
                if use_tf:
                    gt_pos = y[:, t - self.context_size, :, :2]
                    tf_mask = torch.rand(B, 1, 1, device=device) < tf_prob
                    pos = torch.where(tf_mask, gt_pos, pos)
                if self.use_na_ball:
                    # Inject the NA ball position so player rollout features
                    # (graph edges, possession, velocity for the ball row) see
                    # the right ball location.
                    na_pos_h = ball_traj[:, t - self.context_size]  # [B, 2]
                    pos = torch.where(ball_mask3,
                                      na_pos_h.unsqueeze(1).expand(-1, N, -1),
                                      pos)
            parts = [pos]

            vel = pos - prev_pos if (need_vel and t > 0) else torch.zeros_like(pos)
            if self.use_velocity:
                parts.append(vel)
            if self.use_kinematics:
                acc = vel - prev_vel if t > 1 else torch.zeros_like(pos)
                speed = torch.linalg.vector_norm(vel, dim=-1, keepdim=True)
                heading = vel / (speed + 1e-6)  # (cos θ, sin θ); zero when stationary
                parts += [acc, speed, heading]
            parts.append(static)
            if self.use_possession:
                # 5.0 is "5 feet" in court units; divide by coord_scale so the
                # softmin keeps the same physical sharpness when positions are
                # pre-normalized by the trainer.
                parts.append(compute_possession(pos, team_batch,
                                                tau=5.0 / self.coord_scale))
            if self.use_basket_dist:
                # [B, N, 2 baskets]: L2 distance from each node to each basket.
                diff = pos.unsqueeze(2) - self.basket_xy.view(1, 1, 2, 2)
                parts.append(torch.linalg.vector_norm(diff, dim=-1))

            feat = torch.cat(parts, dim=-1)
            x = feat.reshape(B * N, -1)
            if self.use_edge_attr:
                vel_flat = (vel.reshape(B * N, 2)
                            if self.use_rel_vel_edge_attr else None)
                edge_attr = compute_edge_attr(pos.reshape(B * N, 2), edge_index,
                                              vel_flat=vel_flat)
            else:
                edge_attr = None
            if self.use_handler_relations:
                # Handler is re-derived per step so handoffs are picked up
                # during the context and predicted-position handler shifts
                # during the horizon also re-route edge types.
                handler_mask = compute_handler_mask(pos, team_batch)
                edge_type_t = compute_handler_edge_type(team_batch, handler_mask,
                                                        device)
            else:
                edge_type_t = edge_type

            if self.use_ball_centric:
                # Re-derive the player-player distance gate each step from
                # current positions; ball edges + self-loops always survive.
                if edge_attr is not None:
                    dist = edge_attr[:, 2]
                else:
                    src, dst = edge_index[0], edge_index[1]
                    pos_flat = pos.reshape(B * N, 2)
                    dist = torch.linalg.vector_norm(
                        pos_flat[dst] - pos_flat[src], dim=-1)
                keep = (~pp_mask) | (dist <= radius_scaled)
                edge_index_t = edge_index[:, keep]
                edge_type_t = edge_type_t[keep]
                if edge_attr is not None:
                    edge_attr = edge_attr[keep]
            else:
                edge_index_t = edge_index

            # Layer 0 has no residual (input_dim ≠ state_dim); from layer 1
            # onwards every cell adds its output to the previous layer's.
            for li, cell in enumerate(self.cells):
                new_h = cell(x, hs[li], edge_index_t, edge_type_t, edge_attr=edge_attr)
                hs[li] = new_h
                x = new_h if li == 0 else x + new_h

            if self.use_na_ball and t == self.context_size - 1:
                # Decode H ball positions in one shot from the ball-node's
                # top-layer hidden state (one ball per sample → masked sum).
                h_top = hs[-1].reshape(B, N, self.state_dim)
                ball_h = (h_top * ball_mask3).sum(dim=1)              # [B, state_dim]
                last_ctx_ball = (pos * ball_mask3).sum(dim=1)          # [B, 2]
                deltas = self.proj_ball_na(ball_h).reshape(B, self.horizon_size, 2)
                ball_traj = last_ctx_ball.unsqueeze(1) + deltas        # [B, H, 2]

            if self.context_size - 1 <= t < T - 1:
                delta = self.proj(x)  # [B*N, 2]
                if self.use_ball_head:
                    delta = torch.where(ball_mask_flat, self.proj_ball(x), delta)
                curr_coords = pos + delta.reshape(B, N, 2)
                if self.use_handler_relative_ball:
                    # w_i is soft-possession over players (zero on the ball, sums to
                    # 1 over players). We use it to anchor the predicted ball to a
                    # weighted average of the *predicted* player positions, plus a
                    # small learned offset. Held frames → near-zero offset, ball
                    # inherits the player error; passes/shots → offset compensates.
                    w = compute_possession(pos, team_batch,
                                           tau=5.0 / self.coord_scale)[..., 1]  # [B, N]
                    anchor = (curr_coords * w.unsqueeze(-1)).sum(dim=1, keepdim=True)  # [B, 1, 2]
                    offset = self.proj_ball_offset(x).reshape(B, N, 2)
                    ball_pos_new = anchor + offset
                    hr_mask3 = (team_batch == 0).unsqueeze(-1)  # [B, N, 1]
                    curr_coords = torch.where(hr_mask3, ball_pos_new, curr_coords)
                if self.use_na_ball:
                    # Replace the autoregressively-predicted ball entries with
                    # the NA prediction for this horizon step.
                    h_idx = t - (self.context_size - 1)  # 0..H-1
                    na_pos_h = ball_traj[:, h_idx]       # [B, 2]
                    curr_coords = torch.where(ball_mask3,
                                              na_pos_h.unsqueeze(1).expand(-1, N, -1),
                                              curr_coords)
                preds.append(curr_coords)
            prev_pos = pos
            prev_vel = vel

        return torch.stack(preds, dim=0).reshape(self.horizon_size, B * N, 2)


@torch.no_grad()
def ensemble_rollout(models, X: Tensor) -> Tensor:
    """Step-wise ensemble rollout over a list of `NBAModel`s.

    At each timestep every model produces its own delta from the *shared*
    current position; the deltas are averaged and the averaged position is
    fed back to all models as the input for the next step. Per-model hidden
    states are kept independent.

    All members must share `context_size`, `horizon_size`, and `coord_scale`
    (so they agree on the input normalization). `state_dim`, `num_layers`,
    feature flags, and head sizes may differ.

    X: [B, C, N, F] already normalized to the models' internal scale.
    Returns predicted positions [H, B*N, 2] in that same normalized scale.
    """
    if not models:
        raise ValueError('ensemble_rollout requires at least one model')
    ref = models[0]
    for m in models[1:]:
        if (m.context_size, m.horizon_size) != (ref.context_size, ref.horizon_size):
            raise ValueError('ensemble members must share context_size and horizon_size')
        if m.coord_scale != ref.coord_scale:
            raise ValueError('ensemble members must share coord_scale')

    B, _, N, _ = X.shape
    device = X.device
    T = ref.context_size + ref.horizon_size
    team_batch = X[:, 0, :, 3]
    edge_index, edge_type = build_hetero_edges(team_batch, device)
    static = X[:, 0, :, 2:]
    ball_mask_flat = ((team_batch == 0).reshape(B * N, 1)
                      if any(m.use_ball_head for m in models) else None)
    # Player-player static mask, shared across any ball-centric members.
    pp_mask = (compute_player_player_mask(team_batch, edge_index)
               if any(m.use_ball_centric for m in models) else None)

    per_model_hs = [
        [torch.zeros(B * N, m.state_dim, device=device) for _ in range(m.num_layers)]
        for m in models
    ]
    prev_pos = X[:, 0, :, :2]
    prev_vel = torch.zeros(B, N, 2, device=device)
    curr_coords = prev_pos
    preds = []

    for t in range(T):
        pos = X[:, t, :, :2] if t < ref.context_size else curr_coords
        vel = pos - prev_pos if t > 0 else torch.zeros_like(pos)
        acc = vel - prev_vel if t > 1 else torch.zeros_like(pos)
        speed = torch.linalg.vector_norm(vel, dim=-1, keepdim=True)
        heading = vel / (speed + 1e-6)
        # Per-timestep caches; members may need different (edge_attr, edge_type)
        # variants depending on their flags.
        edge_attr_pos_cache = None    # 3-dim (no rel-vel)
        edge_attr_relvel_cache = None # 6-dim (rel-vel)
        edge_type_handler_cache = None
        # Distance-along-edges cache; reused by every ball-centric member
        # regardless of edge_attr variant.
        dist_cache = None

        deltas = []
        # Per-member ball-position overrides for members using the
        # handler-relative reparametrization. We collect a *predicted ball
        # position* (anchor + offset) from each such member; the final ball
        # row is the mean of those overrides plus the (non-overriding)
        # members' implicit ball deltas — see below.
        ball_overrides = []
        for mi, m in enumerate(models):
            parts = [pos]
            if m.use_velocity:
                parts.append(vel)
            if m.use_kinematics:
                parts += [acc, speed, heading]
            parts.append(static)
            if m.use_possession:
                parts.append(compute_possession(pos, team_batch,
                                                tau=5.0 / m.coord_scale))
            if m.use_basket_dist:
                diff = pos.unsqueeze(2) - m.basket_xy.view(1, 1, 2, 2)
                parts.append(torch.linalg.vector_norm(diff, dim=-1))
            x = torch.cat(parts, dim=-1).reshape(B * N, -1)
            if m.use_edge_attr:
                if m.use_rel_vel_edge_attr:
                    if edge_attr_relvel_cache is None:
                        edge_attr_relvel_cache = compute_edge_attr(
                            pos.reshape(B * N, 2), edge_index,
                            vel_flat=vel.reshape(B * N, 2))
                    edge_attr = edge_attr_relvel_cache
                else:
                    if edge_attr_pos_cache is None:
                        edge_attr_pos_cache = compute_edge_attr(
                            pos.reshape(B * N, 2), edge_index)
                    edge_attr = edge_attr_pos_cache
            else:
                edge_attr = None
            if m.use_handler_relations:
                if edge_type_handler_cache is None:
                    handler_mask = compute_handler_mask(pos, team_batch)
                    edge_type_handler_cache = compute_handler_edge_type(
                        team_batch, handler_mask, device)
                edge_type_m = edge_type_handler_cache
            else:
                edge_type_m = edge_type
            if m.use_ball_centric:
                if dist_cache is None:
                    src, dst = edge_index[0], edge_index[1]
                    pos_flat = pos.reshape(B * N, 2)
                    dist_cache = torch.linalg.vector_norm(
                        pos_flat[dst] - pos_flat[src], dim=-1)
                radius_scaled = m.player_edge_radius / m.coord_scale
                keep = (~pp_mask) | (dist_cache <= radius_scaled)
                edge_index_m = edge_index[:, keep]
                edge_type_m = edge_type_m[keep]
                edge_attr_m = edge_attr[keep] if edge_attr is not None else None
            else:
                edge_index_m = edge_index
                edge_attr_m = edge_attr
            for li, cell in enumerate(m.cells):
                new_h = cell(x, per_model_hs[mi][li], edge_index_m, edge_type_m,
                             edge_attr=edge_attr_m)
                per_model_hs[mi][li] = new_h
                x = new_h if li == 0 else x + new_h
            if ref.context_size - 1 <= t < T - 1:
                delta = m.proj(x)
                if m.use_ball_head:
                    delta = torch.where(ball_mask_flat, m.proj_ball(x), delta)
                deltas.append(delta)
                if m.use_handler_relative_ball:
                    # Compute this member's predicted ball position: anchor
                    # (from the *predicted* player positions of THIS member)
                    # + its learned offset. We need the per-member predicted
                    # player positions, which depend on the per-member delta.
                    member_pos = pos + delta.reshape(B, N, 2)
                    w_m = compute_possession(pos, team_batch,
                                             tau=5.0 / m.coord_scale)[..., 1]
                    anchor = (member_pos * w_m.unsqueeze(-1)).sum(dim=1, keepdim=True)
                    offset = m.proj_ball_offset(x).reshape(B, N, 2)
                    # Pick the ball row's offset; member_pos already has shape
                    # [B, N, 2], we just need ball row of offset.
                    ball_mask3 = (team_batch == 0).unsqueeze(-1)
                    ball_offset = (offset * ball_mask3).sum(dim=1, keepdim=True)
                    ball_overrides.append(anchor + ball_offset)  # [B, 1, 2]

        if ref.context_size - 1 <= t < T - 1:
            mean_delta = torch.stack(deltas).mean(dim=0)
            curr_coords = pos + mean_delta.reshape(B, N, 2)
            if ball_overrides:
                # Replace the ball row with the average of every
                # handler-relative member's prediction. Members without the
                # flag still contributed to `mean_delta`, but their ball
                # prediction was the standard pos + delta — for a clean
                # comparison we just override.
                ball_pos_new = torch.stack(ball_overrides).mean(dim=0)
                ball_mask3 = (team_batch == 0).unsqueeze(-1)
                curr_coords = torch.where(ball_mask3, ball_pos_new, curr_coords)
            preds.append(curr_coords)
        prev_pos = pos
        prev_vel = vel

    return torch.stack(preds, dim=0).reshape(ref.horizon_size, B * N, 2)


class MultiStepMSE:
    """Mean per-entity-weighted loss over the prediction horizon.

    By default (`ball_loss='mse'`, `ball_weight=1.0`) this is plain MSE.

    `ball_weight > 1` upweights the ball relative to each player, addressing
    the 1:10 ball/player imbalance that lets the unweighted loss be
    dominated by player error.

    `ball_loss='huber'` swaps the ball's per-sample squared error for
    Huber/smooth-L1 with transition point `huber_delta` (in the *training*
    units the model sees, i.e. positions / coord_scale). Players stay on
    MSE either way. Huber clips the gradient magnitude for residuals
    larger than `huber_delta`, which attacks the MSE-mode-averaging that
    makes the ball under-shoot its true motion magnitude — at the cost of
    less sensitivity to large rare errors. `huber_delta` matters: too small
    collapses to L1 (slower to converge near zero); too large reverts to
    MSE.

    Reported loss values are not comparable across `ball_weight` or
    `ball_loss` settings; use per-entity ADE from
    `NBATrainer.evaluate_breakdown` for cross-run comparisons.
    """

    def __init__(self, ball_weight: float = 1.0,
                 ball_loss: str = 'mse', huber_delta: float = 1.0):
        if ball_weight <= 0:
            raise ValueError(f'ball_weight must be > 0, got {ball_weight}')
        if ball_loss not in ('mse', 'huber'):
            raise ValueError(f"ball_loss must be 'mse' or 'huber', got {ball_loss!r}")
        if huber_delta <= 0:
            raise ValueError(f'huber_delta must be > 0, got {huber_delta}')
        self.ball_weight = float(ball_weight)
        self.ball_loss = ball_loss
        self.huber_delta = float(huber_delta)

    def compute(self, pred: Tensor, target: Tensor) -> Tensor:
        # target: [B, H, N, F] with F >= 4 (team_id at index 3); pred: [H, B*N, 2]
        B, H, N, _ = target.shape
        pos = target[..., :2].permute(1, 0, 2, 3).reshape(H, B * N, 2)
        if pred.shape != pos.shape:
            raise ValueError(f'Shape mismatch: pred {tuple(pred.shape)} vs target {tuple(pos.shape)}')

        # Fast path: plain MSE, no per-entity reweighting.
        if self.ball_weight == 1.0 and self.ball_loss == 'mse':
            return F.mse_loss(pred, pos)

        ball_mask = (target[:, 0, :, 3] == 0).to(pred.dtype)  # [B, N]; 1.0 at ball
        ball_mask_flat = ball_mask.reshape(1, B * N, 1)  # broadcastable over [H, B*N, 2]

        # Per-element loss: MSE for players, MSE-or-Huber for ball.
        diff = pred - pos
        mse = diff * diff
        if self.ball_loss == 'mse':
            elem = mse
        else:
            # Smooth-L1 (β=huber_delta). Matches F.smooth_l1_loss elementwise.
            sl1 = F.smooth_l1_loss(pred, pos, reduction='none', beta=self.huber_delta)
            elem = mse * (1.0 - ball_mask_flat) + sl1 * ball_mask_flat

        # Per-entity weight: 1.0 for players, `ball_weight` for ball.
        w = (1.0 + (self.ball_weight - 1.0) * ball_mask).reshape(1, B * N, 1)
        return (elem * w).sum() / (w.sum() * H * 2)
