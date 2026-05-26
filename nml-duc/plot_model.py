"""Diagrams describing the NBAModel architecture and its inputs/outputs.

Writes to `plots/model/`:

    hetero_graph.png           5-relation heterogeneous graph on a real frame
    edge_type_legend.png       table-style legend with relation-type counts
    architecture.png           per-step pipeline (features → RGAT stack → Δxy)
    stgcn_cell.png             internals of a single HeteroSTGCNCell
    feature_stack.png          per-node feature vector composition
    possession_softmin.png     real-frame example of the soft-possession field
    rollout_timeline.png       context vs horizon, autoregressive feedback

All figures are pure matplotlib (no graphviz dep) so the script is self-contained.
"""

import argparse
import os
from typing import List, Optional, Tuple

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle


COURT_X, COURT_Y = 47.0, 25.0

# Five relation types from build_hetero_edges in model.py
RELATIONS = [
    (0, 'teammate', 'tab:green'),
    (1, 'opponent', 'tab:purple'),
    (2, 'player → ball', 'tab:orange'),
    (3, 'ball → player', 'tab:brown'),
    (4, 'self-loop', 'tab:gray'),
]
TEAM_COLOR = {-1.0: 'tab:blue', 0.0: 'gold', 1.0: 'tab:red'}
TEAM_NAME = {-1.0: 'Team_A', 0.0: 'Ball', 1.0: 'Team_B'}


def pick_frame(train_path: str, seq_idx: Optional[int] = None, t: int = 0
               ) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Pick a visually clear frame: ten players spread enough that node overlap
    is small. Defaults to scanning the first ~200 files and choosing the frame
    with the widest player bounding box."""
    files = sorted(f for f in os.listdir(train_path) if f.endswith('.pt'))
    if seq_idx is None:
        best_score, best = -1.0, (0, 0)
        for i, fn in enumerate(files[:200]):
            s = torch.load(os.path.join(train_path, fn), weights_only=False)
            if s.shape[0] <= t:
                continue
            pos = s[t, :, :2]
            x_range = (pos[:, 0].max() - pos[:, 0].min()).item()
            y_range = (pos[:, 1].max() - pos[:, 1].min()).item()
            score = x_range * y_range
            if score > best_score:
                best_score = score
                best = (i, fn)
        seq_idx, fn = best
    else:
        fn = files[seq_idx]
    seq = torch.load(os.path.join(train_path, fn), weights_only=False)
    pos = seq[t, :, :2]
    team = seq[t, :, 3]
    return pos, team, seq_idx


# ---------- 1. heterogeneous graph diagram ----------

def plot_hetero_graph(out_dir, pos, team):
    """Draw all 11 nodes and color edges by relation type. Bidirectional pairs are
    drawn as a single line for legibility; self-loops as small circles."""
    fig, ax = plt.subplots(figsize=(10.5, 7))

    # Court rectangle for spatial context.
    ax.add_patch(Rectangle((-COURT_X, -COURT_Y), 2 * COURT_X, 2 * COURT_Y,
                           fill=False, edgecolor='black', lw=1.0))
    ax.axvline(0, color='black', lw=0.5)

    N = pos.shape[0]
    pts = pos.numpy()
    tm = team.numpy()

    # Edges by relation. We mirror what build_hetero_edges does.
    rel_pairs = {0: [], 1: [], 2: [], 3: [], 4: []}
    for i in range(N):
        for j in range(N):
            ti, tj = tm[i], tm[j]
            if i == j:
                rel_pairs[4].append((i, j))
            elif ti != 0 and ti == tj:
                rel_pairs[0].append((i, j))
            elif ti != 0 and tj != 0 and ti != tj:
                rel_pairs[1].append((i, j))
            elif ti != 0 and tj == 0:
                rel_pairs[2].append((i, j))
            elif ti == 0 and tj != 0:
                rel_pairs[3].append((i, j))

    # For visual clarity, draw bidirectional symmetric relations (0, 1) as a
    # single undirected line and player-ball (2, 3) as a single line annotated
    # with both colors via dashed/solid styling.
    def _draw_undirected(pairs, color, lw, alpha):
        seen = set()
        for i, j in pairs:
            key = (min(i, j), max(i, j))
            if key in seen:
                continue
            seen.add(key)
            ax.plot([pts[i, 0], pts[j, 0]], [pts[i, 1], pts[j, 1]],
                    color=color, lw=lw, alpha=alpha, zorder=1)

    _draw_undirected(rel_pairs[0], 'tab:green', 1.6, 0.7)
    _draw_undirected(rel_pairs[1], 'tab:purple', 1.0, 0.45)
    # Player↔ball: draw once per player; orange (player→ball) is more salient.
    _draw_undirected(rel_pairs[2], 'tab:orange', 1.8, 0.85)

    # Self-loops: small circle around each node.
    for i in range(N):
        ax.add_patch(Circle((pts[i, 0], pts[i, 1]), 1.8, fill=False,
                            edgecolor='gray', lw=0.7, ls=':', zorder=2))

    # Nodes on top.
    for i in range(N):
        color = TEAM_COLOR[float(tm[i])]
        size = 360 if tm[i] == 0 else 240
        marker = 'o'
        ax.scatter(pts[i, 0], pts[i, 1], c=color, s=size, marker=marker,
                   edgecolor='black', linewidth=1.0, zorder=3)
        ax.text(pts[i, 0], pts[i, 1], f'{i}', ha='center', va='center',
                fontsize=9, fontweight='bold', zorder=4)

    # Legend
    handles = [
        mpatches.Patch(color='tab:green', label=f'teammate ({len(rel_pairs[0])} dir. edges, 5×4 per team)'),
        mpatches.Patch(color='tab:purple', label=f'opponent ({len(rel_pairs[1])} dir. edges, 5×5×2)'),
        mpatches.Patch(color='tab:orange', label=f'player ↔ ball ({len(rel_pairs[2]) + len(rel_pairs[3])} dir. edges, 10×2)'),
        mpatches.Patch(color='gray', label=f'self-loop (11)'),
    ]
    team_handles = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=TEAM_COLOR[-1.0],
                   markeredgecolor='black', markersize=10, label='Team_A (team_id = -1)'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=TEAM_COLOR[0.0],
                   markeredgecolor='black', markersize=12, label='Ball (team_id = 0)'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=TEAM_COLOR[1.0],
                   markeredgecolor='black', markersize=10, label='Team_B (team_id = +1)'),
    ]
    leg1 = ax.legend(handles=handles, loc='upper left', fontsize=9, title='edge relation')
    ax.add_artist(leg1)
    ax.legend(handles=team_handles, loc='upper right', fontsize=9, title='node type')

    total_edges = sum(len(v) for v in rel_pairs.values())
    ax.set_title(f'Per-frame heterogeneous graph — N=11 nodes, '
                 f'{total_edges} directed edges (= N²) over 5 relation types\n'
                 f'(complete graph; relation type chosen per ordered pair from static team IDs)')
    ax.set_aspect('equal')
    ax.set_xlim(-COURT_X - 5, COURT_X + 5)
    ax.set_ylim(-COURT_Y - 8, COURT_Y + 8)
    ax.set_xlabel('x (ft)')
    ax.set_ylabel('y (ft)')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'hetero_graph.png'), dpi=150)
    plt.close(fig)


# ---------- 2. edge-type legend / count table ----------

def plot_edge_type_table(out_dir):
    """Adjacency-matrix view of relation types over the 11×11 ordered pair space."""
    # Labels: B = ball (1), A = team A (5), B' = team B (5). Order: A, B, A_team.
    N = 11
    team = np.array([0] + [-1] * 5 + [1] * 5, dtype=float)  # ball at index 0, then 5×A, 5×B
    rel = np.zeros((N, N), dtype=int)
    for i in range(N):
        for j in range(N):
            ti, tj = team[i], team[j]
            if i == j:
                rel[i, j] = 4
            elif ti != 0 and ti == tj:
                rel[i, j] = 0
            elif ti != 0 and tj != 0 and ti != tj:
                rel[i, j] = 1
            elif ti != 0 and tj == 0:
                rel[i, j] = 2
            else:
                rel[i, j] = 3

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 5.5),
                                  gridspec_kw={'width_ratios': [1.2, 1]})
    rel_colors = ['tab:green', 'tab:purple', 'tab:orange', 'tab:brown', 'tab:gray']
    cmap = plt.matplotlib.colors.ListedColormap(rel_colors)
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
    norm = plt.matplotlib.colors.BoundaryNorm(bounds, cmap.N)
    im = ax.imshow(rel, cmap=cmap, norm=norm)

    labels = ['Ball'] + [f'A{i+1}' for i in range(5)] + [f'B{i+1}' for i in range(5)]
    ax.set_xticks(range(N))
    ax.set_yticks(range(N))
    ax.set_xticklabels(labels, rotation=45, fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel('destination node j')
    ax.set_ylabel('source node i')
    ax.set_title('Relation type for each ordered pair (i, j)')
    for i in range(N):
        for j in range(N):
            ax.text(j, i, str(rel[i, j]), ha='center', va='center',
                    fontsize=8, color='white')

    # Bar chart of counts.
    counts = np.bincount(rel.flatten(), minlength=5)
    names = ['0 teammate', '1 opponent', '2 player→ball', '3 ball→player', '4 self-loop']
    ax2.barh(names, counts, color=rel_colors, edgecolor='black')
    ax2.invert_yaxis()
    for i, c in enumerate(counts):
        ax2.text(c + 1.2, i, f'{c}', va='center', fontsize=10)
    ax2.set_xlim(0, max(counts) * 1.18)
    ax2.set_xlabel('# directed edges per sample')
    ax2.set_title(f'Edge counts (total = N² = {N*N})')

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'edge_type_legend.png'), dpi=150)
    plt.close(fig)


# ---------- 3. per-step architecture diagram ----------

def _box(ax, xy, w, h, text, fc='#eef4ff', ec='black', fs=10, lw=1.0):
    box = FancyBboxPatch(xy, w, h, boxstyle='round,pad=0.04,rounding_size=0.08',
                         fc=fc, ec=ec, lw=lw)
    ax.add_patch(box)
    cx, cy = xy[0] + w / 2, xy[1] + h / 2
    ax.text(cx, cy, text, ha='center', va='center', fontsize=fs)
    return (cx, cy), (xy[0], cy), (xy[0] + w, cy), (cx, xy[1]), (cx, xy[1] + h)


def _arrow(ax, p1, p2, color='black', lw=1.2, ls='-'):
    arr = FancyArrowPatch(p1, p2, arrowstyle='-|>', mutation_scale=12,
                          color=color, lw=lw, ls=ls)
    ax.add_patch(arr)


def plot_architecture(out_dir):
    fig, ax = plt.subplots(figsize=(14, 8.5))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 9)
    ax.axis('off')
    ax.set_title('NBAModel — one rollout step (autoregressive over T = C + H = 20 timesteps)',
                 fontsize=13)

    # Left column: feature builders.
    feat_boxes = [
        (0.2, 7.5, 'pos\n[B, N, 2]', '#fef3e4'),
        (0.2, 6.4, 'velocity\npos_t − pos_{t-1}', '#fef3e4'),
        (0.2, 5.3, 'kinematics\nacc, |v|, (cosθ, sinθ)', '#fef3e4'),
        (0.2, 4.2, 'static\nteam_id, entity_id', '#fef3e4'),
        (0.2, 3.1, 'possession\n[dist_to_ball, soft_poss]', '#fef3e4'),
    ]
    feat_anchors = []
    for x, y, txt, fc in feat_boxes:
        c, l, r, b, top = _box(ax, (x, y), 2.6, 0.9, txt, fc=fc, fs=9)
        feat_anchors.append(r)

    # Concat block.
    c, cl, cr, cb, ct = _box(ax, (3.5, 5.2), 1.6, 1.4,
                              'concat\n[B·N, F_in]', fc='#dcecff', fs=10)
    for a in feat_anchors:
        _arrow(ax, a, cl)

    # Edge attr branch (top): pos → compute_edge_attr → cell.
    eab_c, eab_l, eab_r, eab_b, eab_t = _box(ax, (3.5, 7.6), 1.6, 0.7,
                                              'edge_attr\n[Δx, Δy, ‖Δ‖]',
                                              fc='#ffefef', fs=9)
    _arrow(ax, feat_anchors[0], eab_l)

    # Stack of HeteroSTGCNCells.
    cell_x = 6.0
    cell_w, cell_h = 2.8, 1.0
    cell_anchors = []
    titles = ['HeteroSTGCNCell ℓ=0\n(no residual)',
              'HeteroSTGCNCell ℓ=1\nresidual: x + h',
              '…']
    for i in range(3):
        y0 = 6.4 - i * 1.3
        c, l, r, b, t = _box(ax, (cell_x, y0), cell_w, cell_h, titles[i],
                             fc='#e7f6ec', fs=9.5, ec='tab:green')
        cell_anchors.append((l, r, c))

    # Connect concat → cell0; cell0 → cell1 → …
    _arrow(ax, cr, cell_anchors[0][0])
    for i in range(2):
        _arrow(ax, cell_anchors[i][1], cell_anchors[i + 1][0])

    # edge_attr feeds every cell.
    for la, ra, _ in cell_anchors:
        _arrow(ax, eab_r, (la[0], la[1] + 0.25), color='tab:red', lw=0.9, ls='--')

    # graph (edge_index, edge_type) box.
    g_c, g_l, g_r, g_b, g_t = _box(ax, (3.5, 1.0), 1.6, 1.2,
                                    'edge_index,\nedge_type\n(static per batch)',
                                    fc='#f1e6ff', fs=9)
    for la, ra, _ in cell_anchors:
        _arrow(ax, g_r, (la[0], la[1] - 0.25), color='tab:purple', lw=0.9, ls=':')

    # team_id → graph builder.
    team_c, team_l, team_r, team_b, team_t = _box(ax, (0.2, 1.0), 2.6, 1.2,
                                                   'team_id at t=0\n[B, N]\n→ build_hetero_edges',
                                                   fc='#f1e6ff', fs=9)
    _arrow(ax, team_r, g_l)

    # Output projection.
    proj_c, proj_l, proj_r, proj_b, proj_t = _box(
        ax, (9.4, 5.5), 2.4, 1.1,
        'proj\nLinear → ReLU → Linear\n[B·N, 2] (Δxy)',
        fc='#fff7d6', fs=9.5)
    _arrow(ax, cell_anchors[-1][1], proj_l)

    # Position update box.
    add_c, add_l, add_r, add_b, add_t = _box(
        ax, (12.0, 5.5), 1.7, 1.1, 'pos_{t+1}\n= pos_t + Δxy', fc='#fff7d6', fs=10)
    _arrow(ax, proj_r, add_l)

    # Feedback arrow (rollout): pos_{t+1} → pos input next step.
    ax.annotate('', xy=(0.2, 7.95), xycoords='data',
                xytext=(12.85, 5.0), textcoords='data',
                arrowprops=dict(arrowstyle='-|>', color='tab:blue', lw=1.3,
                                connectionstyle='arc3,rad=-0.35'))
    ax.text(8.0, 4.3, 'autoregressive feedback during horizon (t ≥ C-1)',
            color='tab:blue', fontsize=9, ha='center')

    # Side note about hidden state recurrence.
    ax.text(7.4, 0.2,
            'each HeteroSTGCNCell keeps its own hidden state across all 20 timesteps '
            '(GRUCell recurrence)',
            fontsize=9, ha='center', color='tab:green')

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'architecture.png'), dpi=150)
    plt.close(fig)


# ---------- 4. single STGCN cell ----------

def plot_stgcn_cell(out_dir):
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4.5)
    ax.axis('off')
    ax.set_title('HeteroSTGCNCell — one spatial-temporal step', fontsize=13)

    # Inputs
    _box(ax, (0.1, 2.6), 1.8, 0.9, 'x_t\n[B·N, F_in or D]', fc='#dcecff')
    _box(ax, (0.1, 1.4), 1.8, 0.9, 'edge_index,\nedge_type', fc='#f1e6ff', fs=9)
    _box(ax, (0.1, 0.2), 1.8, 0.9, 'edge_attr\n[Δx, Δy, ‖Δ‖]', fc='#ffefef', fs=9)
    _box(ax, (0.1, 3.8), 1.8, 0.6, 'h_{t-1}', fc='#e7f6ec')

    rgat = _box(ax, (2.4, 1.6), 1.9, 1.7, 'RGATConv\nnum_relations=5\nedge_dim=3',
                fc='#e7f6ec', ec='tab:green')
    relu = _box(ax, (4.8, 2.1), 1.2, 0.8, 'ReLU', fc='#e7f6ec')
    gru  = _box(ax, (6.3, 1.7), 1.7, 1.6, 'GRUCell\nstate_dim → state_dim',
                fc='#e7f6ec', ec='tab:green')
    ln   = _box(ax, (8.5, 2.1), 1.4, 0.8, 'LayerNorm', fc='#e7f6ec')
    out  = _box(ax, (10.3, 2.1), 1.5, 0.8, 'h_t\n[B·N, D]', fc='#dcecff')

    _arrow(ax, (1.9, 3.05), rgat[1])      # x_t → rgat
    _arrow(ax, (1.9, 1.85), rgat[1])      # edges → rgat
    _arrow(ax, (1.9, 0.65), rgat[3])      # edge_attr → rgat (bottom)
    _arrow(ax, rgat[2], relu[1])
    _arrow(ax, relu[2], gru[1])
    _arrow(ax, (1.9, 4.1), gru[4])        # h_{t-1} → gru top
    _arrow(ax, gru[2], ln[1])
    _arrow(ax, ln[2], out[1])

    ax.text(6.0, 0.3,
            'Layer ℓ ≥ 1 adds a residual: input to next cell = x_in + h_t  (skip lets the '
            'receptive field grow without losing earlier features)',
            fontsize=10, ha='center', color='tab:gray')

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'stgcn_cell.png'), dpi=150)
    plt.close(fig)


# ---------- 5. feature stack composition ----------

def plot_feature_stack(out_dir):
    """Visualize the per-node feature vector as concatenated blocks. Defaults
    assume all flags on: F_in = 2 + 2 + 5 + (F-2) + 2 = 13 with F=4."""
    blocks = [
        ('pos (x, y)', 2, '#fbb4ae'),
        ('velocity (vx, vy)', 2, '#b3cde3'),
        ('kinematics:\nacc(2) + |v|(1) + heading(2)', 5, '#ccebc5'),
        ('static (entity_id, team_id)', 2, '#decbe4'),
        ('possession:\ndist_to_ball + soft_possession', 2, '#fed9a6'),
    ]
    total = sum(b[1] for b in blocks)

    fig, ax = plt.subplots(figsize=(13, 3.2))
    ax.set_xlim(0, total + 1)
    ax.set_ylim(0, 2.8)
    ax.axis('off')
    x = 0.5
    for name, k, color in blocks:
        rect = Rectangle((x, 0.8), k, 1.0, facecolor=color,
                         edgecolor='black', lw=1.2)
        ax.add_patch(rect)
        for d in range(k):
            ax.add_patch(Rectangle((x + d, 0.8), 1, 1.0, facecolor=color,
                                   edgecolor='black', lw=0.5))
        ax.text(x + k / 2, 0.45, name, ha='center', va='top', fontsize=10)
        ax.text(x + k / 2, 1.3, f'{k} dim', ha='center', va='center',
                fontsize=10, fontweight='bold')
        x += k

    ax.text((total + 1) / 2, 2.4,
            f'Per-node feature vector (per timestep) — F_in = {total} dims when all flags on',
            ha='center', fontsize=13)
    ax.text((total + 1) / 2, 2.05,
            f'Velocity is computed regardless if either use_velocity or use_kinematics is set; '
            f'positions feed back autoregressively, so all of these are recomputed every step.',
            ha='center', fontsize=9, color='tab:gray')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'feature_stack.png'), dpi=150)
    plt.close(fig)


# ---------- 6. possession softmin example ----------

def plot_possession_example(out_dir, pos, team, tau=5.0):
    """Draw a single frame and overlay soft-possession weights as node-size."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Compute possession exactly like model.compute_possession.
    pos_b = pos.unsqueeze(0)             # [1, N, 2]
    team_b = team.unsqueeze(0)           # [1, N]
    ball_mask = (team_b == 0)
    ball_pos = (pos_b * ball_mask.unsqueeze(-1)).sum(dim=1, keepdim=True)
    dist = torch.linalg.vector_norm(pos_b - ball_pos, dim=-1)  # [1, N]
    logits = (-dist / tau).masked_fill(ball_mask, float('-inf'))
    soft_poss = torch.softmax(logits, dim=-1).squeeze(0).numpy()  # [N]
    dist_np = dist.squeeze(0).numpy()
    pts = pos.numpy()
    tm = team.numpy()

    # Left: court with weighted markers.
    ax = axes[0]
    ax.add_patch(Rectangle((-COURT_X, -COURT_Y), 2 * COURT_X, 2 * COURT_Y,
                           fill=False, edgecolor='black', lw=1.0))
    ax.axvline(0, color='black', lw=0.4)
    for i in range(pts.shape[0]):
        s = 100 + soft_poss[i] * 1100
        color = TEAM_COLOR[float(tm[i])]
        ax.scatter(pts[i, 0], pts[i, 1], s=s, color=color,
                   edgecolor='black', linewidth=1.0, zorder=3)
        ax.text(pts[i, 0], pts[i, 1] - 2.5, f'p={soft_poss[i]:.2f}',
                ha='center', va='top', fontsize=8)
        # Draw line ball-to-player.
        if tm[i] != 0:
            ax.plot([pts[i, 0], ball_pos[0, 0, 0]], [pts[i, 1], ball_pos[0, 0, 1]],
                    color='gray', lw=0.4, alpha=0.4, zorder=1)
    ax.set_aspect('equal')
    ax.set_xlim(-COURT_X - 5, COURT_X + 5)
    ax.set_ylim(-COURT_Y - 8, COURT_Y + 8)
    ax.set_title(f'Soft-possession on a real frame (τ={tau:g} ft)\n'
                 'node size ∝ softmin weight; ball is masked out (p=0)')
    ax.set_xlabel('x (ft)'); ax.set_ylabel('y (ft)')

    # Right: 1-D softmin curve over distance.
    ax = axes[1]
    d_grid = np.linspace(0, 40, 400)
    # Single-player softmin against a fixed competitor set is hard to vis in 1-D,
    # so we show -d/τ vs d, plus the actual normalized weights of this frame.
    ax.plot(d_grid, np.exp(-d_grid / tau), color='tab:blue', lw=2,
            label=f'exp(-d / τ),  τ = {tau} ft')
    ax.set_xlabel('distance to ball (ft)')
    ax.set_ylabel('exp(-d/τ)  (unnormalized softmin weight)')
    ax.set_title('Softmin weight vs distance')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')

    # Show this frame's players as dots on the curve.
    for i in range(pts.shape[0]):
        if tm[i] == 0:
            continue
        ax.scatter(dist_np[i], np.exp(-dist_np[i] / tau),
                   color=TEAM_COLOR[float(tm[i])], s=80, edgecolor='black',
                   zorder=3)
        ax.annotate(f'p={soft_poss[i]:.2f}',
                    (dist_np[i], np.exp(-dist_np[i] / tau)),
                    textcoords='offset points', xytext=(6, 6), fontsize=8)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'possession_softmin.png'), dpi=150)
    plt.close(fig)


# ---------- 7. rollout timeline ----------

def plot_rollout_timeline(out_dir, context=8, horizon=12):
    T = context + horizon
    fig, ax = plt.subplots(figsize=(15.5, 4.8))
    ax.set_xlim(-4.5, T - 0.4)
    ax.set_ylim(-1.5, 4.0)
    ax.axis('off')
    ax.set_title('Per-step rollout — input source, hidden state, output (context=8, horizon=12)',
                 fontsize=13)

    # Horizontal "rails" labels.
    ax.text(-4.3, 3.35, 'input pos at step t',     fontsize=10, ha='left')
    ax.text(-4.3, 2.10, 'graph cells (shared)',    fontsize=10, ha='left')
    ax.text(-4.3, 0.80, 'output Δxy → pos_{t+1}',  fontsize=10, ha='left')

    for t in range(T):
        x = t
        # Input source: ground-truth pos for t<C, predicted pos otherwise.
        if t < context:
            ic = '#dcecff'
            label = f'X[:, {t}]'
        else:
            ic = '#fff7d6'
            label = f'̂p_{{{t}}}'
        ax.add_patch(Rectangle((x - 0.35, 3.0), 0.7, 0.7, fc=ic,
                               ec='black', lw=0.8))
        ax.text(x, 3.35, label, ha='center', va='center', fontsize=9)

        # Cell.
        ax.add_patch(Rectangle((x - 0.35, 1.75), 0.7, 0.7, fc='#e7f6ec',
                               ec='tab:green', lw=0.8))
        ax.text(x, 2.1, 'cell', ha='center', va='center', fontsize=9)

        # Output: only when context-1 ≤ t < T-1.
        produces_pred = (context - 1) <= t < (T - 1)
        if produces_pred:
            ax.add_patch(Rectangle((x - 0.35, 0.45), 0.7, 0.7, fc='#ffe2c8',
                                   ec='black', lw=0.8))
            ax.text(x, 0.80, f'̂p_{{{t+1}}}', ha='center', va='center', fontsize=9)
            # Arrow from out_t to in_{t+1}.
            if t + 1 < T:
                _arrow(ax, (x + 0.05, 0.80), (x + 1 - 0.4, 3.35),
                       color='tab:orange', lw=1.0)

        # Arrows input→cell, cell→output.
        _arrow(ax, (x, 3.0), (x, 2.45), color='black')
        if produces_pred:
            _arrow(ax, (x, 1.75), (x, 1.15), color='black')

        # Hidden state recurrence horizontal arrow.
        if t > 0:
            _arrow(ax, (x - 1 + 0.35, 2.1), (x - 0.35, 2.1),
                   color='tab:green', lw=1.0, ls='--')

    # Context vs horizon shading.
    ax.add_patch(Rectangle((-0.5, -0.2), context, 0.4, fc='#dcecff', alpha=0.6))
    ax.text(context / 2 - 0.5, 0.0, 'context (teacher inputs)',
            ha='center', va='center', fontsize=10)
    ax.add_patch(Rectangle((context - 0.5, -0.2), horizon, 0.4, fc='#fff7d6', alpha=0.6))
    ax.text(context + horizon / 2 - 0.5, 0.0, 'horizon (autoregressive)',
            ha='center', va='center', fontsize=10)

    ax.text(context - 0.5, -1.1,
            'first prediction is produced at t = C - 1 = 7 (= pos at frame 8); '
            'last at t = T - 2 = 18 (= pos at frame 19). H = 12 predictions total.',
            fontsize=9, ha='center', color='tab:gray')

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'rollout_timeline.png'), dpi=120)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train-path', default='data/train/train')
    ap.add_argument('--out-dir', default='plots/model')
    ap.add_argument('--seq-idx', type=int, default=None,
                    help='Pick a specific train sequence index for the graph/possession plots.')
    ap.add_argument('--frame', type=int, default=4,
                    help='Frame within the picked sequence to visualize.')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    pos, team, seq_idx = pick_frame(args.train_path, args.seq_idx, args.frame)
    print(f'using train sequence #{seq_idx}, frame {args.frame}')

    plot_hetero_graph(args.out_dir, pos, team)
    plot_edge_type_table(args.out_dir)
    plot_architecture(args.out_dir)
    plot_stgcn_cell(args.out_dir)
    plot_feature_stack(args.out_dir)
    plot_possession_example(args.out_dir, pos, team)
    plot_rollout_timeline(args.out_dir)
    print(f'wrote plots to {args.out_dir}/')


if __name__ == '__main__':
    main()
