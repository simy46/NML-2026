"""Generate viz/data.json for the interactive prediction viewer.

Loads the seed-0 checkpoint, samples N random plays from the deterministic val
split, and dumps context + prediction + ground truth (plus a few summary
metrics) as a single JSON file consumed by viz/index.html.

Usage (run from nml-claude/):
    python viz/make_data.py
    python viz/make_data.py --num 50 --model-path model_seed0.pth \
                            --config metrics/seed_0/config.json
"""

import argparse
import json
import os
import random
import sys
from typing import List

# Allow `python viz/make_data.py` from the project root.
_THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_THIS))

import torch

from data import split_train_val
from inference import _ARCH_KEYS
from model import NBAModel


def _load_config(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def _build_model(cfg: dict, device: str) -> NBAModel:
    arch = {k: cfg[k] for k in _ARCH_KEYS if k in cfg}
    model = NBAModel(
        input_dim=arch.get('input_dim', 4),
        output_dim=arch.get('output_dim', 2),
        state_dim=arch['state_dim'],
        context_size=arch['context_size'],
        horizon_size=arch['horizon_size'],
        num_layers=arch['num_layers'],
        coord_scale=arch.get('coord_scale', 1.0),
        use_kinematics=arch.get('use_kinematics', True),
        use_possession=arch.get('use_possession', True),
        use_basket_dist=arch.get('use_basket_dist', False),
        use_ball_head=arch.get('use_ball_head', False),
        use_handler_relative_ball=arch.get('use_handler_relative_ball', False),
        use_na_ball=arch.get('use_na_ball', False),
    ).to(device)
    return model


def _ade_fde(pred: torch.Tensor, gt: torch.Tensor) -> tuple[float, float]:
    """pred, gt: [H, N, 2] in feet. Returns (ADE, FDE) averaged over entities."""
    diff = (pred - gt).norm(dim=-1)  # [H, N]
    ade = diff.mean().item()
    fde = diff[-1].mean().item()
    return ade, fde


@torch.no_grad()
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train-path', default='data/train/train',
                   help='Directory of training .pt files (used for val split).')
    p.add_argument('--model-path', default='checkpoints/model_seed0.pth',
                   help='Checkpoint to load.')
    p.add_argument('--config', default='metrics/seed_0/config.json',
                   help='Run config that produced the checkpoint.')
    p.add_argument('--num', type=int, default=100,
                   help='Number of val plays to dump.')
    p.add_argument('--out', default='viz/data.js',
                   help='Output JS path (sets window.VIZ_DATA so the viewer '
                        'works without a local server).')
    p.add_argument('--pick-seed', type=int, default=0,
                   help='Seed for selecting which val plays to dump '
                        '(independent of training seed).')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = p.parse_args()

    cfg = _load_config(args.config)
    val_split = cfg.get('val_split', 0.1)
    train_seed = cfg.get('seed', 0)
    context_size = cfg['context_size']
    horizon_size = cfg['horizon_size']
    coord_scale = cfg.get('coord_scale', 1.0)
    window = context_size + horizon_size

    _, val_files = split_train_val(args.train_path, val_split, seed=train_seed)
    if not val_files:
        raise SystemExit(f'no val files (val_split={val_split}, seed={train_seed})')

    # Each .pt is [T, N, F]; only keep ones long enough to give us GT for the
    # whole horizon. Almost all train sequences are 22 frames so this is a
    # no-op in practice, but it keeps the script robust.
    eligible = []
    for f in val_files:
        seq = torch.load(f, weights_only=False)
        if seq.size(0) >= window:
            eligible.append((f, seq))
    print(f'[data] {len(eligible)} eligible val plays of length >= {window}')

    rng = random.Random(args.pick_seed)
    picked = rng.sample(eligible, k=min(args.num, len(eligible)))

    model = _build_model(cfg, args.device)
    state = torch.load(args.model_path, weights_only=True, map_location=args.device)
    model.load_state_dict(state)
    model.eval()
    print(f'[model] loaded {args.model_path}  device={args.device}')

    samples: List[dict] = []
    for f, seq in picked:
        sid = int(os.path.basename(f).removesuffix('.pt'))
        seq = seq[:window]  # [T, N, F]
        N = seq.size(1)
        ctx = seq[:context_size]                          # [C, N, F]
        gt_future = seq[context_size:window, :, :2]       # [H, N, 2] feet
        X = ctx.unsqueeze(0).to(args.device).clone()
        X_in = X.clone()
        X_in[..., :2] = X_in[..., :2] / coord_scale       # normalize positions
        pred = model(X_in) * coord_scale                   # [H, 1*N, 2]
        pred = pred.detach().cpu().reshape(horizon_size, N, 2)

        ade, fde = _ade_fde(pred, gt_future)
        # Per-horizon-step error averaged over entities.
        per_step = (pred - gt_future).norm(dim=-1).mean(dim=-1).tolist()

        teams = ctx[0, :, 3].tolist()  # static team_id per node (-1, 0, +1)

        samples.append({
            'id': sid,
            'teams': [int(t) for t in teams],
            'context': ctx[:, :, :2].tolist(),      # [C][N][2]
            'gt_future': gt_future.tolist(),         # [H][N][2]
            'pred_future': pred.tolist(),            # [H][N][2]
            'ade': ade,
            'fde': fde,
            'per_step_error': per_step,
        })

    # Order samples by ADE so the viewer's "next/prev" walks from best to worst
    # (a useful failure-mode browser by default).
    samples.sort(key=lambda s: s['ade'])

    aggregate = {
        'mean_ade': sum(s['ade'] for s in samples) / max(len(samples), 1),
        'mean_fde': sum(s['fde'] for s in samples) / max(len(samples), 1),
        'count': len(samples),
    }

    payload = {
        'meta': {
            'context_size': context_size,
            'horizon_size': horizon_size,
            'coord_scale': coord_scale,
            'court_x_half': 47.0,   # storage convention (centered at 0)
            'court_y_half': 25.0,
            'model_path': os.path.basename(args.model_path),
            'val_split': val_split,
            'train_seed': train_seed,
            'pick_seed': args.pick_seed,
            'aggregate': aggregate,
        },
        'samples': samples,
    }

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w') as f:
        if args.out.endswith('.js'):
            f.write('window.VIZ_DATA = ')
            json.dump(payload, f, separators=(',', ':'))
            f.write(';\n')
        else:
            json.dump(payload, f, separators=(',', ':'))
    size_kb = os.path.getsize(args.out) / 1024
    print(f'[write] {args.out}  ({size_kb:.1f} KB, {len(samples)} samples)')
    print(f'[stats] mean ADE = {aggregate["mean_ade"]:.3f} ft, '
          f'mean FDE = {aggregate["mean_fde"]:.3f} ft')


if __name__ == '__main__':
    main()
