"""Aggregate per-epoch and end-of-training metrics across multi-seed runs.

Reads `<metrics_dir>/seed_*/{metrics.csv, val_per_horizon.csv, val_per_entity.csv}`
and writes plots with mean ± 95% confidence interval (Student's t):

    loss_curves_seeds.png      train/val loss vs epoch — mean ± 95% CI band
    per_horizon_ade_seeds.png  per-horizon-step ADE — mean ± 95% CI band
    per_entity_ade_seeds.png   per-entity ADE — bar with 95% CI error bars + per-seed dots
    best_val_loss_seeds.png    best val loss per seed + mean line + 95% CI band

CI = mean ± t_{.025, n-1} * std / sqrt(n). Bands degrade to mean-only when n<2.

Usage:
    python plot_seeds.py                                 # defaults to metrics/ and plots/seeds/
    python plot_seeds.py --metrics-dir metrics/ablation_x --out-dir plots/ablation_x_seeds
"""

import argparse
import csv
import glob
import json
import os
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats  # type: ignore


ENTITY_ORDER = ['Team_A', 'Ball', 'Team_B']
ENTITY_COLORS = {'Team_A': 'tab:blue', 'Ball': 'tab:orange', 'Team_B': 'tab:red'}


def _read_csv(path: str) -> List[dict]:
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def _to_float(s) -> float:
    if s in (None, '', 'None'):
        return float('nan')
    return float(s)


def discover_seeds(metrics_dir: str) -> List[Tuple[int, str]]:
    """Return sorted list of (seed_id, path) tuples for `seed_*` subdirs."""
    out = []
    for d in sorted(glob.glob(os.path.join(metrics_dir, 'seed_*'))):
        if not os.path.isdir(d):
            continue
        try:
            seed_id = int(os.path.basename(d).split('_')[-1])
        except ValueError:
            continue
        out.append((seed_id, d))
    return sorted(out)


def mean_ci(arr: np.ndarray, axis: int = 0, conf: float = 0.95
            ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (mean, lower, upper) along axis using Student's-t CI. NaN-aware."""
    mean = np.nanmean(arr, axis=axis)
    n = np.sum(~np.isnan(arr), axis=axis)
    std = np.nanstd(arr, axis=axis, ddof=1)
    with np.errstate(invalid='ignore', divide='ignore'):
        sem = std / np.sqrt(n)
        # n>=2 → real CI; n<2 → 0 half-width so plotting still works.
        t = np.where(n >= 2, stats.t.ppf(0.5 + conf / 2, np.maximum(n - 1, 1)), 0.0)
        half = t * sem
    return mean, mean - half, mean + half


# ---------- loaders ----------

def load_loss_curves(seed_dirs: List[Tuple[int, str]]
                     ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Stack per-epoch train/val loss and lr across seeds. Returns (epoch, train, val, lr)
    where train/val/lr are [n_seeds, n_epochs] aligned on the shortest run."""
    train_per_seed, val_per_seed, lr_per_seed = [], [], []
    for _, d in seed_dirs:
        rows = _read_csv(os.path.join(d, 'metrics.csv'))
        if not rows:
            continue
        train_per_seed.append(np.array([_to_float(r['train_loss']) for r in rows]))
        val_per_seed.append(np.array([_to_float(r['val_loss']) for r in rows]))
        lr_per_seed.append(np.array([_to_float(r['lr']) for r in rows]))

    if not train_per_seed:
        raise SystemExit('no metrics.csv files found')
    n_min = min(len(a) for a in train_per_seed)
    train = np.stack([a[:n_min] for a in train_per_seed])
    val = np.stack([a[:n_min] for a in val_per_seed])
    lr = np.stack([a[:n_min] for a in lr_per_seed])
    epoch = np.arange(1, n_min + 1)
    return epoch, train, val, lr


def load_per_horizon(seed_dirs: List[Tuple[int, str]]) -> Optional[np.ndarray]:
    rows_per_seed = []
    for _, d in seed_dirs:
        rows = _read_csv(os.path.join(d, 'val_per_horizon.csv'))
        if not rows:
            continue
        rows_per_seed.append(np.array([float(r['ade']) for r in rows]))
    if not rows_per_seed:
        return None
    n_min = min(len(a) for a in rows_per_seed)
    return np.stack([a[:n_min] for a in rows_per_seed])


def load_per_entity(seed_dirs: List[Tuple[int, str]]) -> Optional[Dict[str, np.ndarray]]:
    per_entity: Dict[str, List[float]] = {e: [] for e in ENTITY_ORDER}
    found = False
    for _, d in seed_dirs:
        rows = _read_csv(os.path.join(d, 'val_per_entity.csv'))
        if not rows:
            continue
        found = True
        m = {r['entity']: float(r['ade']) for r in rows}
        for e in ENTITY_ORDER:
            per_entity[e].append(m.get(e, np.nan))
    if not found:
        return None
    return {e: np.asarray(v) for e, v in per_entity.items()}


# ---------- plots ----------

def plot_loss_curves(epoch, train, val, lr, n_seeds, out_path):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    tm, tl, tu = mean_ci(train)
    vm, vl, vu = mean_ci(val)

    ax.plot(epoch, tm, color='tab:blue', label=f'train (mean of {n_seeds} seeds)', lw=1.8)
    ax.fill_between(epoch, tl, tu, color='tab:blue', alpha=0.20, label='train 95% CI')
    ax.plot(epoch, vm, color='tab:red', ls='--', label=f'val (mean of {n_seeds} seeds)', lw=1.8)
    ax.fill_between(epoch, vl, vu, color='tab:red', alpha=0.20, label='val 95% CI')

    ax.set_xlabel('epoch')
    ax.set_ylabel('loss (MSE, ft²)')
    ax.set_yscale('log')
    ax.set_title(f'Train / val loss across {n_seeds} seeds (mean ± 95% CI)')
    ax.grid(True, which='both', alpha=0.25)

    twin = ax.twinx()
    twin.plot(epoch, np.nanmean(lr, axis=0), color='tab:gray', alpha=0.6, lw=1.0,
              label='lr')
    twin.set_ylabel('lr (mean across seeds)', color='tab:gray')
    twin.set_yscale('log')
    twin.tick_params(axis='y', colors='tab:gray')

    lines, labels = ax.get_legend_handles_labels()
    tlines, tlabels = twin.get_legend_handles_labels()
    ax.legend(lines + tlines, labels + tlabels, loc='upper right', fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_per_horizon(arr, n_seeds, out_path):
    steps = np.arange(1, arr.shape[1] + 1)
    m, lo, hi = mean_ci(arr)
    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.plot(steps, m, '-o', color='tab:blue', label=f'mean of {n_seeds} seeds')
    ax.fill_between(steps, lo, hi, color='tab:blue', alpha=0.22, label='95% CI')
    # Show per-seed light curves too.
    for i in range(arr.shape[0]):
        ax.plot(steps, arr[i], color='tab:blue', alpha=0.25, lw=0.8)
    ax.set_xticks(steps)
    ax.set_xlabel('horizon step (1 = first predicted frame)')
    ax.set_ylabel('ADE (ft)')
    ax.set_title('Per-horizon-step ADE — mean ± 95% CI across seeds')
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_per_entity(per_entity, n_seeds, out_path):
    arr = np.stack([per_entity[e] for e in ENTITY_ORDER])  # [3, n_seeds]
    means = np.nanmean(arr, axis=1)
    n = np.sum(~np.isnan(arr), axis=1)
    std = np.nanstd(arr, axis=1, ddof=1)
    with np.errstate(invalid='ignore', divide='ignore'):
        t = np.where(n >= 2, stats.t.ppf(0.975, np.maximum(n - 1, 1)), 0.0)
        half = t * std / np.sqrt(n)

    fig, ax = plt.subplots(figsize=(7.5, 5))
    x = np.arange(len(ENTITY_ORDER))
    colors = [ENTITY_COLORS[e] for e in ENTITY_ORDER]
    bars = ax.bar(x, means, yerr=half, capsize=6, color=colors, alpha=0.7,
                  edgecolor='black', linewidth=0.8)
    # Per-seed scatter on top of each bar.
    rng = np.random.default_rng(0)
    for i, e in enumerate(ENTITY_ORDER):
        jitter = rng.uniform(-0.12, 0.12, size=arr.shape[1])
        ax.scatter(x[i] + jitter, arr[i], color='black', s=30, zorder=3,
                   alpha=0.8, edgecolor='white', linewidth=0.5)
    for b, m, h in zip(bars, means, half):
        ax.text(b.get_x() + b.get_width() / 2, m + h + 0.05,
                f'{m:.2f}\n±{h:.2f}', ha='center', va='bottom', fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(ENTITY_ORDER)
    ax.set_ylabel('ADE (ft)')
    ax.set_title(f'Per-entity ADE — bar = mean across {n_seeds} seeds, error = 95% CI')
    ax.grid(True, axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_best_val_loss(val, seed_ids, out_path):
    best = np.nanmin(val, axis=1)  # [n_seeds]
    mean = float(np.nanmean(best))
    n = int(np.sum(~np.isnan(best)))
    std = float(np.nanstd(best, ddof=1)) if n >= 2 else 0.0
    half = float(stats.t.ppf(0.975, n - 1) * std / np.sqrt(n)) if n >= 2 else 0.0

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(seed_ids))
    ax.bar(x, best, color='tab:purple', alpha=0.75, edgecolor='black')
    ax.axhline(mean, color='red', ls='--', label=f'mean = {mean:.4f}')
    ax.fill_between([-0.5, len(seed_ids) - 0.5], mean - half, mean + half,
                    color='red', alpha=0.15, label=f'95% CI (±{half:.4f})')
    ax.set_xlim(-0.5, len(seed_ids) - 0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f'seed {s}' for s in seed_ids])
    ax.set_ylabel('best val loss (MSE, ft²)')
    ax.set_title('Best val loss per seed')
    for i, v in enumerate(best):
        ax.text(i, v, f'{v:.3f}', ha='center', va='bottom', fontsize=9)
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def write_summary(epoch, train, val, per_horizon, per_entity, seed_ids, out_path):
    best = np.nanmin(val, axis=1)
    final_train = train[:, -1]
    final_val = val[:, -1]

    def fmt_ci(arr):
        a = np.asarray(arr, dtype=float).ravel()
        m = float(np.nanmean(a))
        n = int(np.sum(~np.isnan(a)))
        if n >= 2:
            std = float(np.nanstd(a, ddof=1))
            half = float(stats.t.ppf(0.975, n - 1) * std / np.sqrt(n))
        else:
            half = 0.0
        return f'{m:.4f}  (95% CI [{m - half:.4f}, {m + half:.4f}])'

    lines = [
        f'# Multi-seed aggregate summary (n_seeds = {len(seed_ids)})',
        '',
        f'- seeds: {seed_ids}',
        f'- epochs aggregated: {epoch[-1]}',
        '',
        '## final-epoch loss',
        f'- train: {fmt_ci(final_train)}',
        f'- val:   {fmt_ci(final_val)}',
        '',
        '## best val loss',
        f'- per seed: {[round(float(b), 4) for b in best]}',
        f'- aggregate: {fmt_ci(best)}',
    ]
    if per_horizon is not None:
        m, lo, hi = mean_ci(per_horizon)
        lines += [
            '',
            '## per-horizon ADE (val, end of training)',
            *[f'- step {i+1:2d}: {m[i]:.3f}  [95% CI {lo[i]:.3f}, {hi[i]:.3f}]'
              for i in range(len(m))],
        ]
    if per_entity is not None:
        lines += ['', '## per-entity ADE (val, end of training)']
        for e in ENTITY_ORDER:
            lines.append(f'- {e}: {fmt_ci(per_entity[e])}')
    with open(out_path, 'w') as f:
        f.write('\n'.join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--metrics-dir', default='metrics',
                    help='Directory containing seed_*/ subdirectories.')
    ap.add_argument('--out-dir', default='plots/seeds',
                    help='Output directory for the aggregate plots.')
    args = ap.parse_args()

    seed_dirs = discover_seeds(args.metrics_dir)
    if not seed_dirs:
        raise SystemExit(f'no seed_*/ subdirectories found under {args.metrics_dir}/')
    seed_ids = [s for s, _ in seed_dirs]
    print(f'aggregating {len(seed_dirs)} seeds: {seed_ids}')

    os.makedirs(args.out_dir, exist_ok=True)
    epoch, train, val, lr = load_loss_curves(seed_dirs)
    plot_loss_curves(epoch, train, val, lr, len(seed_dirs),
                     os.path.join(args.out_dir, 'loss_curves_seeds.png'))
    plot_best_val_loss(val, seed_ids,
                       os.path.join(args.out_dir, 'best_val_loss_seeds.png'))

    per_horizon = load_per_horizon(seed_dirs)
    if per_horizon is not None:
        plot_per_horizon(per_horizon, len(seed_dirs),
                         os.path.join(args.out_dir, 'per_horizon_ade_seeds.png'))

    per_entity = load_per_entity(seed_dirs)
    if per_entity is not None:
        plot_per_entity(per_entity, len(seed_dirs),
                        os.path.join(args.out_dir, 'per_entity_ade_seeds.png'))

    write_summary(epoch, train, val, per_horizon, per_entity, seed_ids,
                  os.path.join(args.out_dir, 'summary.md'))
    print(f'wrote plots to {args.out_dir}/')


if __name__ == '__main__':
    main()