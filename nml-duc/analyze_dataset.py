"""Dataset analysis plots for the NBA trajectory-prediction task.

Loads every `.pt` sequence in `data/train/train` and `data/test/test`, then
writes a battery of summary plots to `plots/dataset/`. Pure consumer of the
raw `.pt` files — no model code is invoked.

Feature layout per row: `(x, y, entity_id, team_id)` with
`team_id in {-1 (Team_A), 0 (Ball), +1 (Team_B)}` and the ball as row 0.
Court coordinates are in feet, centered at midcourt (extent ~ ±47, ±25).
"""

import argparse
import os
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm


COURT_X, COURT_Y = 47.0, 25.0  # NBA half-court extents in feet (full court 94 x 50)
TEAM_LABELS = {-1.0: 'Team_A', 0.0: 'Ball', 1.0: 'Team_B'}
TEAM_COLORS = {-1.0: 'tab:blue', 0.0: 'tab:orange', 1.0: 'tab:red'}


def load_sequences(path: str) -> List[torch.Tensor]:
    files = sorted(f for f in os.listdir(path) if f.endswith('.pt'))
    seqs = []
    for f in tqdm(files, desc=f'load {os.path.basename(path.rstrip("/"))}'):
        seqs.append(torch.load(os.path.join(path, f), weights_only=False))
    return seqs


def draw_court(ax):
    """Rough rectangle indicating the NBA full-court boundary."""
    ax.add_patch(plt.Rectangle((-COURT_X, -COURT_Y), 2 * COURT_X, 2 * COURT_Y,
                               fill=False, edgecolor='black', linewidth=1.2))
    ax.axvline(0, color='black', lw=0.6)
    # Hoops are about 4 ft from each baseline.
    for x in (-COURT_X + 4, COURT_X - 4):
        ax.add_patch(plt.Circle((x, 0), 0.75, fill=False, edgecolor='black', lw=0.6))


def plot_sequence_lengths(train_seqs, test_seqs, out_dir):
    train_T = np.array([s.shape[0] for s in train_seqs])
    test_T = np.array([s.shape[0] for s in test_seqs])

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    ax = axes[0]
    bins = np.arange(train_T.min(), train_T.max() + 2) - 0.5
    ax.hist(train_T, bins=bins, color='tab:blue', alpha=0.85, edgecolor='white')
    ax.axvline(train_T.mean(), color='red', ls='--',
               label=f'mean={train_T.mean():.1f}')
    ax.axvline(np.median(train_T), color='black', ls=':',
               label=f'median={int(np.median(train_T))}')
    ax.set_title(f'Train sequence lengths (n={len(train_T)})')
    ax.set_xlabel('frames per sequence')
    ax.set_ylabel('count')
    ax.legend()

    ax = axes[1]
    if test_T.min() == test_T.max():
        ax.bar([int(test_T[0])], [len(test_T)], color='tab:orange',
               edgecolor='white', width=0.8)
        ax.set_xticks([int(test_T[0])])
        ax.text(int(test_T[0]), len(test_T) * 1.01,
                f'all {len(test_T)} test files = {int(test_T[0])} frames',
                ha='center', va='bottom', fontsize=10)
    else:
        bins = np.arange(test_T.min(), test_T.max() + 2) - 0.5
        ax.hist(test_T, bins=bins, color='tab:orange', edgecolor='white')
    ax.set_title(f'Test sequence lengths (n={len(test_T)})')
    ax.set_xlabel('frames per sequence')
    ax.set_ylabel('count')

    fig.suptitle('Sequence length distribution — train vs test')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'sequence_lengths.png'), dpi=140)
    plt.close(fig)

    # Number of usable windows at default context=8, horizon=12 (window=20).
    window = 20
    n_windows = np.maximum(0, train_T - window + 1)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(n_windows, bins=40, color='tab:green', edgecolor='white')
    ax.set_title(f'Number of (context=8 + horizon=12) windows per train sequence\n'
                 f'mean={n_windows.mean():.1f}, total={int(n_windows.sum())}')
    ax.set_xlabel('# valid start offsets')
    ax.set_ylabel('count of sequences')
    ax.axvline(n_windows.mean(), color='red', ls='--')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'sequence_windows.png'), dpi=140)
    plt.close(fig)


def plot_position_heatmaps(train_seqs, out_dir):
    """Court-occupancy heatmaps by entity type (ball vs players, optionally per team)."""
    # Gather x/y per team_id across all frames.
    xs = {-1.0: [], 0.0: [], 1.0: []}
    ys = {-1.0: [], 0.0: [], 1.0: []}
    for seq in train_seqs:
        team = seq[0, :, 3]  # team labels are constant in time
        for label in (-1.0, 0.0, 1.0):
            mask = (team == label)
            if not mask.any():
                continue
            sub = seq[:, mask, :2]  # [T, k, 2]
            xs[label].append(sub[..., 0].flatten().numpy())
            ys[label].append(sub[..., 1].flatten().numpy())

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharex=True, sharey=True)
    extent = [-COURT_X - 5, COURT_X + 5, -COURT_Y - 5, COURT_Y + 5]
    for ax, label in zip(axes, (-1.0, 0.0, 1.0)):
        x = np.concatenate(xs[label]) if xs[label] else np.array([])
        y = np.concatenate(ys[label]) if ys[label] else np.array([])
        h, xed, yed = np.histogram2d(x, y, bins=[80, 50], range=[[-COURT_X - 5, COURT_X + 5],
                                                                  [-COURT_Y - 5, COURT_Y + 5]])
        ax.imshow(np.log1p(h.T), origin='lower', extent=extent, cmap='magma', aspect='equal')
        draw_court(ax)
        ax.set_title(f'{TEAM_LABELS[label]} (n_samples={len(x):,})')
        ax.set_xlabel('x (ft)')
        ax.set_ylabel('y (ft)')
    fig.suptitle('Court occupancy heatmap (log scale) — all train frames')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'court_heatmap.png'), dpi=140)
    plt.close(fig)


def plot_coordinate_ranges(train_seqs, out_dir):
    """Per-frame x and y marginal distributions, by entity type."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for label in (-1.0, 0.0, 1.0):
        xs, ys = [], []
        for seq in train_seqs:
            team = seq[0, :, 3]
            mask = (team == label)
            if not mask.any():
                continue
            sub = seq[:, mask, :2]
            xs.append(sub[..., 0].flatten().numpy())
            ys.append(sub[..., 1].flatten().numpy())
        x = np.concatenate(xs) if xs else np.array([])
        y = np.concatenate(ys) if ys else np.array([])
        axes[0].hist(x, bins=120, histtype='step', density=True,
                     color=TEAM_COLORS[label], label=TEAM_LABELS[label], lw=1.6)
        axes[1].hist(y, bins=80, histtype='step', density=True,
                     color=TEAM_COLORS[label], label=TEAM_LABELS[label], lw=1.6)
    for ax, name, lim in zip(axes, ('x', 'y'), (COURT_X, COURT_Y)):
        ax.axvline(-lim, color='gray', ls=':', lw=0.8)
        ax.axvline(+lim, color='gray', ls=':', lw=0.8)
        ax.set_xlabel(f'{name} (ft)')
        ax.set_ylabel('density')
        ax.legend()
        ax.set_title(f'{name}-coordinate distribution')
    fig.suptitle('Marginal coordinate distributions across all train frames')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'coord_marginals.png'), dpi=140)
    plt.close(fig)


def _compute_speeds(seqs, fps=25.0):
    """Frame-to-frame speeds by team label. Returns dict label -> 1-D numpy array (ft/s)."""
    out = {-1.0: [], 0.0: [], 1.0: []}
    for seq in seqs:
        team = seq[0, :, 3].numpy()
        d = (seq[1:, :, :2] - seq[:-1, :, :2]).numpy()  # [T-1, N, 2]
        sp = np.linalg.norm(d, axis=-1) * fps  # ft / s
        for label in (-1.0, 0.0, 1.0):
            mask = team == label
            if mask.any():
                out[label].append(sp[:, mask].flatten())
    return {k: np.concatenate(v) if v else np.array([]) for k, v in out.items()}


def plot_speed_distribution(train_seqs, out_dir, fps=25.0):
    speeds = _compute_speeds(train_seqs, fps=fps)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    ax = axes[0]
    for label, arr in speeds.items():
        if arr.size:
            ax.hist(np.clip(arr, 0, 40), bins=120, histtype='step',
                    density=True, color=TEAM_COLORS[label],
                    label=f'{TEAM_LABELS[label]} (med={np.median(arr):.1f})', lw=1.6)
    ax.set_xlabel('speed (ft/s, capped at 40)')
    ax.set_ylabel('density')
    ax.set_title(f'Per-frame speed distribution (fps={fps:g})')
    ax.legend()

    ax = axes[1]
    data = [speeds[l] for l in (-1.0, 0.0, 1.0)]
    bp = ax.boxplot([d if d.size else [0] for d in data], showfliers=False,
                    labels=[TEAM_LABELS[l] for l in (-1.0, 0.0, 1.0)], patch_artist=True)
    for patch, label in zip(bp['boxes'], (-1.0, 0.0, 1.0)):
        patch.set_facecolor(TEAM_COLORS[label])
        patch.set_alpha(0.6)
    ax.set_ylabel('speed (ft/s)')
    ax.set_title('Speed boxplot (no outliers)')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'speed_distribution.png'), dpi=140)
    plt.close(fig)


def plot_step_displacement(train_seqs, out_dir, horizon=12, context=8):
    """Per-frame displacement magnitude as a function of frame index within sequences
    long enough to span context + horizon. Indicates how 'fast' the average horizon
    frame is relative to context frames."""
    window = context + horizon
    accum = np.zeros(window - 1)
    accum_ball = np.zeros(window - 1)
    counts = 0
    for seq in train_seqs:
        if seq.shape[0] < window:
            continue
        # Use first window for a stable per-frame mean.
        w = seq[:window].numpy()
        d = np.linalg.norm(w[1:, :, :2] - w[:-1, :, :2], axis=-1)  # [window-1, N]
        team = seq[0, :, 3].numpy()
        ball_mask = team == 0.0
        player_mask = ~ball_mask
        accum += d[:, player_mask].mean(axis=1)
        accum_ball += d[:, ball_mask].mean(axis=1)
        counts += 1
    accum /= max(counts, 1)
    accum_ball /= max(counts, 1)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    frames = np.arange(1, window)
    ax.plot(frames, accum, '-o', ms=4, label='players (mean)', color='tab:red')
    ax.plot(frames, accum_ball, '-o', ms=4, label='ball', color='tab:orange')
    ax.axvline(context, color='gray', ls='--', label=f'context boundary (t={context})')
    ax.set_xlabel('frame transition index')
    ax.set_ylabel('mean displacement per frame (ft)')
    ax.set_title('Average step size over the prediction window')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'step_displacement.png'), dpi=140)
    plt.close(fig)


def plot_distance_to_ball(train_seqs, out_dir):
    """Histogram of player↔ball distance and time-fraction each player holds 'possession'
    (= argmin distance over the 10 players in a frame)."""
    dists_team = {-1.0: [], 1.0: []}
    poss_team = {-1.0: 0, 1.0: 0}
    total_frames = 0
    for seq in train_seqs:
        team = seq[0, :, 3].numpy()
        ball_mask = team == 0.0
        if not ball_mask.any():
            continue
        ball = seq[:, ball_mask, :2].squeeze(1).numpy()  # [T, 2]
        for label in (-1.0, 1.0):
            mask = team == label
            if not mask.any():
                continue
            pl = seq[:, mask, :2].numpy()  # [T, k, 2]
            d = np.linalg.norm(pl - ball[:, None, :], axis=-1)  # [T, k]
            dists_team[label].append(d.flatten())

        # Possession = argmin distance among the 10 players per frame.
        player_mask = team != 0.0
        pl_all = seq[:, player_mask, :2].numpy()  # [T, 10, 2]
        d_all = np.linalg.norm(pl_all - ball[:, None, :], axis=-1)
        owner_idx = d_all.argmin(axis=1)  # [T]
        owner_team = team[player_mask][owner_idx]
        for label in (-1.0, 1.0):
            poss_team[label] += int(np.sum(owner_team == label))
        total_frames += d_all.shape[0]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    ax = axes[0]
    for label in (-1.0, 1.0):
        arr = np.concatenate(dists_team[label])
        ax.hist(np.clip(arr, 0, 80), bins=80, histtype='step', density=True,
                color=TEAM_COLORS[label], label=f'{TEAM_LABELS[label]} (med={np.median(arr):.1f} ft)',
                lw=1.6)
    ax.set_xlabel('player → ball distance (ft, capped at 80)')
    ax.set_ylabel('density')
    ax.set_title('Distance-to-ball distribution')
    ax.legend()

    ax = axes[1]
    labels = [TEAM_LABELS[-1.0], TEAM_LABELS[1.0]]
    vals = [poss_team[-1.0] / max(total_frames, 1), poss_team[1.0] / max(total_frames, 1)]
    bars = ax.bar(labels, vals, color=[TEAM_COLORS[-1.0], TEAM_COLORS[1.0]])
    ax.set_ylim(0, max(vals) * 1.2)
    ax.set_ylabel('fraction of frames')
    ax.set_title(f'Possession (closest player to ball) — total frames = {total_frames:,}')
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f'{v:.3f}', ha='center', va='bottom')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'distance_to_ball.png'), dpi=140)
    plt.close(fig)


def plot_inter_player_distance(train_seqs, out_dir):
    """Mean teammate-pair vs opponent-pair distance per frame, aggregated as a histogram."""
    teammate, opponent = [], []
    for seq in train_seqs:
        team = seq[0, :, 3].numpy()
        player_mask = team != 0.0
        pl = seq[:, player_mask, :2].numpy()  # [T, 10, 2]
        labels = team[player_mask]
        T, N, _ = pl.shape
        # pairwise [T, N, N]
        diff = pl[:, :, None, :] - pl[:, None, :, :]
        d = np.linalg.norm(diff, axis=-1)
        same = labels[:, None] == labels[None, :]
        iu = np.triu_indices(N, k=1)
        same_pairs = same[iu]
        d_pairs = d[:, iu[0], iu[1]]  # [T, n_pairs]
        teammate.append(d_pairs[:, same_pairs].flatten())
        opponent.append(d_pairs[:, ~same_pairs].flatten())
    teammate = np.concatenate(teammate)
    opponent = np.concatenate(opponent)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bins = np.linspace(0, 100, 100)
    ax.hist(teammate, bins=bins, histtype='step', density=True, lw=1.6,
            color='tab:green', label=f'teammate (med={np.median(teammate):.1f})')
    ax.hist(opponent, bins=bins, histtype='step', density=True, lw=1.6,
            color='tab:purple', label=f'opponent (med={np.median(opponent):.1f})')
    ax.set_xlabel('pairwise player distance (ft)')
    ax.set_ylabel('density')
    ax.set_title('Pairwise player–player distances')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'pairwise_player_distance.png'), dpi=140)
    plt.close(fig)


def plot_sample_trajectories(train_seqs, out_dir, n=6, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(train_seqs), size=n, replace=False)

    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4.5 * rows), squeeze=False)
    for ax, i in zip(axes.flat, idx):
        seq = train_seqs[i].numpy()
        T = seq.shape[0]
        team = seq[0, :, 3]
        for k in range(seq.shape[1]):
            label = team[k]
            ax.plot(seq[:, k, 0], seq[:, k, 1], color=TEAM_COLORS[label],
                    lw=1.2 if label == 0.0 else 0.8,
                    alpha=0.9 if label == 0.0 else 0.65)
            ax.scatter(seq[0, k, 0], seq[0, k, 1], color=TEAM_COLORS[label],
                       s=20, marker='o', edgecolor='black', linewidth=0.4)
            ax.scatter(seq[-1, k, 0], seq[-1, k, 1], color=TEAM_COLORS[label],
                       s=25, marker='s', edgecolor='black', linewidth=0.4)
        draw_court(ax)
        ax.set_xlim(-COURT_X - 5, COURT_X + 5)
        ax.set_ylim(-COURT_Y - 5, COURT_Y + 5)
        ax.set_aspect('equal')
        ax.set_title(f'seq #{i} — T={T} frames')
    fig.suptitle('Sample trajectories (○ start, □ end) — orange=ball, blue=A, red=B')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'sample_trajectories.png'), dpi=140)
    plt.close(fig)


def write_summary(train_seqs, test_seqs, out_dir):
    train_T = np.array([s.shape[0] for s in train_seqs])
    test_T = np.array([s.shape[0] for s in test_seqs])
    lines = [
        '# Dataset summary',
        '',
        f'- train sequences: {len(train_seqs)}',
        f'  - length min/median/mean/max: {train_T.min()} / {int(np.median(train_T))} / {train_T.mean():.2f} / {train_T.max()}',
        f'  - total frames: {int(train_T.sum())}',
        f'- test sequences: {len(test_seqs)}',
        f'  - length min/median/mean/max: {test_T.min()} / {int(np.median(test_T))} / {test_T.mean():.2f} / {test_T.max()}',
        f'  - total frames: {int(test_T.sum())}',
        '',
        '## per-window count (context=8 + horizon=12 = 20)',
        f'- usable train windows: {int(np.maximum(0, train_T - 20 + 1).sum())}',
        f'- train sequences too short (<20 frames): {int((train_T < 20).sum())}',
    ]
    with open(os.path.join(out_dir, 'summary.md'), 'w') as f:
        f.write('\n'.join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train-path', default='data/train/train')
    ap.add_argument('--test-path', default='data/test/test')
    ap.add_argument('--out-dir', default='plots/dataset')
    ap.add_argument('--fps', type=float, default=25.0,
                    help='assumed sampling rate for speed conversion (ft/s)')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    train_seqs = load_sequences(args.train_path)
    test_seqs = load_sequences(args.test_path)

    plot_sequence_lengths(train_seqs, test_seqs, args.out_dir)
    plot_position_heatmaps(train_seqs, args.out_dir)
    plot_coordinate_ranges(train_seqs, args.out_dir)
    plot_speed_distribution(train_seqs, args.out_dir, fps=args.fps)
    plot_step_displacement(train_seqs, args.out_dir)
    plot_distance_to_ball(train_seqs, args.out_dir)
    plot_inter_player_distance(train_seqs, args.out_dir)
    plot_sample_trajectories(train_seqs, args.out_dir)
    write_summary(train_seqs, test_seqs, args.out_dir)
    print(f'wrote plots to {args.out_dir}/')


if __name__ == '__main__':
    main()