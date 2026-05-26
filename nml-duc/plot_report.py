"""Generate report plots from one or more --metrics-dir directories.

Each input directory should contain (some subset of):
    metrics.csv            per-epoch curves written by NBATrainer
    val_per_horizon.csv    end-of-training per-horizon-step ADE on val
    val_per_entity.csv     end-of-training per-entity-class ADE on val
    config.json            run config

Single-run mode: writes
    loss_curves.png        train/val loss vs epoch (+ LR on secondary axis)
    per_horizon_ade.png    ADE vs horizon step
    per_entity_ade.png     ADE per entity class (Team_A / Ball / Team_B)

Multi-run mode (>=2 dirs) additionally writes
    summary_bar.png        best val loss per run, for ablation comparison
and overlays the per-run curves on loss_curves / per_horizon / per_entity.

Usage:
    python plot_report.py metrics/baseline
    python plot_report.py metrics/baseline metrics/no_aug metrics/no_tf
    python plot_report.py metrics/* --out-dir report_plots
"""

import argparse
import csv
import json
import os
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np


# ---------- data loading ----------

def _read_csv(path: str) -> List[dict]:
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def _to_float(s) -> float:
    if s in (None, '', 'None'):
        return float('nan')
    return float(s)


def load_run(run_dir: str) -> dict:
    cfg: Dict = {}
    cfg_p = os.path.join(run_dir, 'config.json')
    if os.path.exists(cfg_p):
        with open(cfg_p) as f:
            cfg = json.load(f)
    return {
        'name':         os.path.basename(os.path.normpath(run_dir)),
        'config':       cfg,
        'metrics':      _read_csv(os.path.join(run_dir, 'metrics.csv')),
        'per_horizon':  _read_csv(os.path.join(run_dir, 'val_per_horizon.csv')),
        'per_entity':   _read_csv(os.path.join(run_dir, 'val_per_entity.csv')),
    }


# ---------- plots ----------

def plot_loss_curves(runs: List[dict], out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    twin = ax.twinx() if len(runs) == 1 else None
    for run in runs:
        if not run['metrics']:
            continue
        ep = [int(r['epoch'])             for r in run['metrics']]
        tr = [_to_float(r['train_loss'])  for r in run['metrics']]
        vl = [_to_float(r['val_loss'])    for r in run['metrics']]
        prefix = '' if len(runs) == 1 else f'{run["name"]} '
        ax.plot(ep, tr, label=f'{prefix}train')
        if not all(np.isnan(vl)):
            ax.plot(ep, vl, linestyle='--', label=f'{prefix}val')
        if twin is not None:
            lr = [_to_float(r['lr']) for r in run['metrics']]
            twin.plot(ep, lr, color='tab:gray', alpha=0.6, label='lr')
            twin.set_yscale('log')
            twin.set_ylabel('lr')
    ax.set_xlabel('epoch')
    ax.set_ylabel('loss (MSE)')
    ax.set_yscale('log')
    ax.set_title('Training and validation loss')
    ax.legend(loc='upper right', fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_per_horizon(runs: List[dict], out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    plotted = False
    for run in runs:
        if not run['per_horizon']:
            continue
        steps = [int(r['horizon_step']) for r in run['per_horizon']]
        ade   = [float(r['ade'])         for r in run['per_horizon']]
        ax.plot(steps, ade, marker='o', label=run['name'])
        plotted = True
    if not plotted:
        plt.close(fig)
        return
    ax.set_xlabel('horizon step')
    ax.set_ylabel('average displacement error (val)')
    ax.set_title('Per-horizon-step ADE')
    ax.set_xticks(steps)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_per_entity(runs: List[dict], out_path: str) -> None:
    entities = ['Team_A', 'Ball', 'Team_B']
    runs_with_data = [r for r in runs if r['per_entity']]
    if not runs_with_data:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    width = 0.8 / len(runs_with_data)
    x = np.arange(len(entities))
    for i, run in enumerate(runs_with_data):
        pe = {r['entity']: float(r['ade']) for r in run['per_entity']}
        bars = [pe.get(e, 0.0) for e in entities]
        ax.bar(x + i * width - 0.4 + width / 2, bars, width, label=run['name'])
    ax.set_xticks(x)
    ax.set_xticklabels(entities)
    ax.set_ylabel('average displacement error (val)')
    ax.set_title('Per-entity-class ADE')
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_summary_bar(runs: List[dict], out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    names = [r['name'] for r in runs]
    best: List[float] = []
    for r in runs:
        vals = [_to_float(m['val_loss']) for m in r['metrics']]
        vals = [v for v in vals if not np.isnan(v)]
        best.append(min(vals) if vals else float('nan'))
    ax.bar(names, best)
    ax.set_ylabel('best val loss (MSE)')
    ax.set_title('Best val loss by run')
    for label in ax.get_xticklabels():
        label.set_rotation(30)
        label.set_ha('right')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------- entry point ----------

def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument('run_dirs', nargs='+',
                   help='One or more --metrics-dir directories produced by run.py.')
    p.add_argument('--out-dir', default='report_plots',
                   help='Directory to write the PNGs into.')
    args = p.parse_args()

    runs = [load_run(d) for d in args.run_dirs]
    os.makedirs(args.out_dir, exist_ok=True)

    plot_loss_curves(runs, os.path.join(args.out_dir, 'loss_curves.png'))
    plot_per_horizon(runs, os.path.join(args.out_dir, 'per_horizon_ade.png'))
    plot_per_entity(runs, os.path.join(args.out_dir, 'per_entity_ade.png'))
    if len(runs) > 1:
        plot_summary_bar(runs, os.path.join(args.out_dir, 'summary_bar.png'))

    written = sorted(os.listdir(args.out_dir))
    print(f'[plot] wrote {len(written)} file(s) to {args.out_dir}/:')
    for f in written:
        print(f'         {f}')


if __name__ == '__main__':
    main()
