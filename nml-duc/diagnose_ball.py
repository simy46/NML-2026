"""Diagnose where the ball-prediction error concentrates.

For every (sample, horizon-step) of the validation set, classify the ground-
truth ball state as one of:
    held       — nearest player < HELD_TH ft   (ball-handler in possession)
    contested  — HELD_TH ≤ d < FLIGHT_TH       (pass/catch transition)
    flight     — d ≥ FLIGHT_TH                 (mid-flight pass or shot)
and report ball ADE separately in each bucket. The diagnostic uses
*ground-truth* nearest-player distance so the bucketing is independent of
model quality. We also break the result down by horizon step so we can see
whether early horizon errors look the same as late ones.

Usage:
    python diagnose_ball.py                          # uses seed 0 by default
    python diagnose_ball.py --seeds 0,1,2,3,4 --out-dir plots/diagnostic
"""

import argparse
import json
import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from data import NBADataset, NBASampler, split_train_val
from trainer import NBATrainer


HELD_TH = 3.0     # ft
FLIGHT_TH = 8.0   # ft
BUCKETS = ('held', 'contested', 'flight')
BUCKET_COLORS = {'held': 'tab:green', 'contested': 'tab:orange', 'flight': 'tab:red'}


def _bucketize(min_d: np.ndarray) -> np.ndarray:
    """Return an int array of bucket indices: 0 held, 1 contested, 2 flight."""
    out = np.full(min_d.shape, 2, dtype=np.int64)
    out[min_d < FLIGHT_TH] = 1
    out[min_d < HELD_TH] = 0
    return out


def trainer_for(seed: int, config_dir: str, model_path: str,
                train_path: str, device: str) -> NBATrainer:
    with open(os.path.join(config_dir, 'config.json')) as f:
        cfg = json.load(f)
    return NBATrainer(
        batch_size=cfg['batch_size'], epochs=1,
        train_path=train_path, val_split=cfg['val_split'],
        device=device,
        context_size=cfg['context_size'], horizon_size=cfg['horizon_size'],
        input_dim=cfg['input_dim'], output_dim=cfg['output_dim'],
        state_dim=cfg['state_dim'], num_layers=cfg['num_layers'],
        coord_scale=cfg['coord_scale'],
        use_ball_head=cfg.get('use_ball_head', False),
        use_handler_relative_ball=cfg.get('use_handler_relative_ball', False),
        teacher_forcing=False, tf_k=cfg['tf_k'],
        ball_weight=cfg.get('ball_weight', 1.0),
        grad_clip=cfg['grad_clip'],
        model_path=model_path,
        windows_per_sequence=1,
        augment=False, seed=seed, tta=False,
    )


def run_seed(seed: int, train_path: str, metrics_root: str,
             ckpt_root: str, device: str) -> Dict:
    config_dir = os.path.join(metrics_root, f'seed_{seed}')
    model_path = os.path.join(ckpt_root, f'model_seed{seed}.pth')
    trainer = trainer_for(seed, config_dir, model_path, train_path, device)
    H = trainer.horizon_size
    val_files = sorted(trainer.val_dataset.sequences[:0] or [])  # placeholder
    val_dataset = trainer.val_dataset
    val_sampler = NBASampler(trainer.batch_size, val_dataset.max_start,
                             seed=seed, shuffle=False, windows_per_sequence=1)
    val_loader = DataLoader(val_dataset, batch_size=trainer.batch_size,
                            sampler=val_sampler)

    # Accumulators: ball errors per (bucket, horizon-step) and per bucket overall.
    err_sq_sum = {b: np.zeros(H, dtype=np.float64) for b in BUCKETS}
    err_sq_n = {b: np.zeros(H, dtype=np.int64) for b in BUCKETS}
    err_ade_sum = {b: np.zeros(H, dtype=np.float64) for b in BUCKETS}
    bucket_total = {b: 0 for b in BUCKETS}
    overall_ball_ade_sum = 0.0
    overall_ball_ade_n = 0
    bucket_by_step = np.zeros((H, 3), dtype=np.int64)

    trainer._load_model()
    with torch.no_grad():
        for X, y in val_loader:
            X = X.to(device)
            y = y.to(device)
            pred = trainer._predict_positions(X)  # [H, B*N, 2] in feet
            B, N = X.size(0), X.size(2)
            target = y[..., :2].permute(1, 0, 2, 3).reshape(H, B * N, 2)
            dist_h = (pred - target).norm(dim=-1).cpu().numpy()  # [H, B*N]
            team_ids = y[:, 0, :, 3].reshape(-1).cpu().numpy()   # [B*N]
            ball_mask_flat = (team_ids == 0.0)                   # [B*N]

            # Ground-truth nearest-player distance per (h, b) on the ball entity.
            # y is [B, H, N, F]; ball at the row where team_id == 0.
            y_np = y[..., :2].cpu().numpy()                      # [B, H, N, 2]
            team_b = y[:, 0, :, 3].cpu().numpy()                 # [B, N]
            for b_idx in range(B):
                team_row = team_b[b_idx]
                ball_idx = int(np.where(team_row == 0.0)[0][0])
                player_idx = np.where(team_row != 0.0)[0]
                ball_pos = y_np[b_idx, :, ball_idx, :]                # [H, 2]
                # Fancy indexing pulls player axis to the front → [10, H, 2];
                # transpose so the broadcast against ball_pos[:, None, :] works.
                player_pos = y_np[b_idx, :, player_idx, :].transpose(1, 0, 2)  # [H, 10, 2]
                d = np.linalg.norm(player_pos - ball_pos[:, None, :], axis=-1)
                min_d = d.min(axis=1)                            # [H]
                buckets = _bucketize(min_d)                      # [H]
                # Ball ADE for this sample (per horizon step).
                row_in_flat = b_idx * N + ball_idx
                ball_err = dist_h[:, row_in_flat]                # [H]
                for h in range(H):
                    bk = BUCKETS[buckets[h]]
                    err_sq_sum[bk][h] += float(ball_err[h] ** 2)
                    err_sq_n[bk][h] += 1
                    err_ade_sum[bk][h] += float(ball_err[h])
                    bucket_total[bk] += 1
                    bucket_by_step[h, buckets[h]] += 1
                overall_ball_ade_sum += float(ball_err.sum())
                overall_ball_ade_n += H

    # Aggregate.
    out = {
        'seed': seed,
        'overall_ball_ade_ft': overall_ball_ade_sum / max(overall_ball_ade_n, 1),
        'bucket_total': bucket_total,
        'per_step_count': bucket_by_step.tolist(),
        'per_step_ade': {},
        'per_step_rmse': {},
        'overall_ade': {},
        'overall_rmse': {},
    }
    for b in BUCKETS:
        n = err_sq_n[b]
        ade = np.where(n > 0, err_ade_sum[b] / np.maximum(n, 1), np.nan)
        rmse = np.where(n > 0, np.sqrt(err_sq_sum[b] / np.maximum(n, 1)), np.nan)
        out['per_step_ade'][b] = ade.tolist()
        out['per_step_rmse'][b] = rmse.tolist()
        n_total = int(n.sum())
        out['overall_ade'][b] = float(err_ade_sum[b].sum() / max(n_total, 1))
        out['overall_rmse'][b] = float(np.sqrt(err_sq_sum[b].sum() / max(n_total, 1)))
    return out


def aggregate_seeds(results: List[Dict]) -> Dict:
    H = len(results[0]['per_step_ade']['held'])
    agg = {'overall_ade': {}, 'overall_rmse': {}, 'per_step_ade_mean': {},
           'per_step_ade_std': {}, 'bucket_share': {}}
    total_counts = {b: 0 for b in BUCKETS}
    total_all = 0
    for r in results:
        for b in BUCKETS:
            total_counts[b] += r['bucket_total'][b]
            total_all += r['bucket_total'][b]
    for b in BUCKETS:
        ades = np.array([r['overall_ade'][b] for r in results])
        rmses = np.array([r['overall_rmse'][b] for r in results])
        agg['overall_ade'][b] = (float(ades.mean()),
                                 float(ades.std(ddof=1)) if len(ades) > 1 else 0.0)
        agg['overall_rmse'][b] = (float(rmses.mean()),
                                  float(rmses.std(ddof=1)) if len(rmses) > 1 else 0.0)
        step_arr = np.array([r['per_step_ade'][b] for r in results])  # [n_seeds, H]
        agg['per_step_ade_mean'][b] = step_arr.mean(axis=0).tolist()
        agg['per_step_ade_std'][b] = (step_arr.std(axis=0, ddof=1).tolist()
                                       if step_arr.shape[0] > 1
                                       else [0.0] * H)
        agg['bucket_share'][b] = total_counts[b] / max(total_all, 1)
    overall_ball_ades = np.array([r['overall_ball_ade_ft'] for r in results])
    agg['overall_ball_ade'] = (float(overall_ball_ades.mean()),
                                float(overall_ball_ades.std(ddof=1))
                                if len(overall_ball_ades) > 1 else 0.0)
    agg['n_seeds'] = len(results)
    agg['H'] = H
    return agg


def plot_results(agg: Dict, out_path: str):
    H = agg['H']
    steps = np.arange(1, H + 1)

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    # Panel 1: ball ADE per bucket (bar with std error and bucket share annotation).
    ax = axes[0]
    bs = [agg['overall_ade'][b][0] for b in BUCKETS]
    es = [agg['overall_ade'][b][1] for b in BUCKETS]
    shares = [agg['bucket_share'][b] for b in BUCKETS]
    bars = ax.bar(BUCKETS, bs, yerr=es, capsize=6,
                  color=[BUCKET_COLORS[b] for b in BUCKETS],
                  edgecolor='black', alpha=0.85)
    for bar, mean, share in zip(bars, bs, shares):
        ax.text(bar.get_x() + bar.get_width() / 2, mean,
                f'{mean:.2f} ft\n({share*100:.0f}% of frames)',
                ha='center', va='bottom', fontsize=10)
    ax.set_ylabel('ball ADE (ft)')
    ax.set_title(f'Ball ADE by ground-truth ball state\n'
                 f'(overall ball ADE = {agg["overall_ball_ade"][0]:.2f} ft)')
    ax.grid(True, axis='y', alpha=0.3)
    ax.set_ylim(0, max(bs) * 1.35)

    # Panel 2: per-horizon-step ADE, one line per bucket.
    ax = axes[1]
    for b in BUCKETS:
        m = np.array(agg['per_step_ade_mean'][b])
        s = np.array(agg['per_step_ade_std'][b])
        ax.plot(steps, m, marker='o', color=BUCKET_COLORS[b],
                label=f'{b}  ({agg["bucket_share"][b]*100:.0f}%)')
        if agg['n_seeds'] > 1:
            ax.fill_between(steps, m - s, m + s, color=BUCKET_COLORS[b], alpha=0.18)
    ax.set_xticks(steps)
    ax.set_xlabel('horizon step')
    ax.set_ylabel('ball ADE (ft)')
    ax.set_title('Per-horizon-step ball ADE, split by bucket')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Panel 3: error budget contribution (share × ADE = expected ft contribution).
    ax = axes[2]
    contrib = [agg['overall_ade'][b][0] * agg['bucket_share'][b] for b in BUCKETS]
    total = sum(contrib)
    bars = ax.bar(BUCKETS, contrib, color=[BUCKET_COLORS[b] for b in BUCKETS],
                  edgecolor='black', alpha=0.85)
    for bar, c in zip(bars, contrib):
        ax.text(bar.get_x() + bar.get_width() / 2, c,
                f'{c:.2f} ft\n({c/total*100:.0f}% of ball error)',
                ha='center', va='bottom', fontsize=10)
    ax.set_ylabel('contribution to overall ball ADE (ft)')
    ax.set_title('Where the ball error budget is spent\n(= bucket share × bucket ADE)')
    ax.grid(True, axis='y', alpha=0.3)
    ax.set_ylim(0, max(contrib) * 1.4)

    fig.suptitle(f'Ball-state diagnostic — HELD < {HELD_TH} ft ≤ CONTESTED < {FLIGHT_TH} ft ≤ FLIGHT '
                 f'(n_seeds = {agg["n_seeds"]})', y=1.00)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', default='0', help='Comma-separated seeds to evaluate.')
    ap.add_argument('--metrics-root', default='metrics')
    ap.add_argument('--ckpt-root', default='checkpoints')
    ap.add_argument('--train-path', default='data/train/train')
    ap.add_argument('--out-dir', default='plots/diagnostic')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(',') if s.strip()]
    os.makedirs(args.out_dir, exist_ok=True)
    results = []
    for s in seeds:
        print(f'-- seed {s} --')
        r = run_seed(s, args.train_path, args.metrics_root, args.ckpt_root, args.device)
        print(f'   overall ball ADE = {r["overall_ball_ade_ft"]:.3f} ft')
        for b in BUCKETS:
            print(f'   {b:>9s}: {r["overall_ade"][b]:.3f} ft   (n={r["bucket_total"][b]:,})')
        results.append(r)

    agg = aggregate_seeds(results)
    out_path = os.path.join(args.out_dir, 'ball_state.png')
    plot_results(agg, out_path)
    with open(os.path.join(args.out_dir, 'ball_state.json'), 'w') as f:
        json.dump(agg, f, indent=2)

    print()
    print(f'== aggregate across {agg["n_seeds"]} seed(s) ==')
    print(f'overall ball ADE: {agg["overall_ball_ade"][0]:.3f} '
          f'± {agg["overall_ball_ade"][1]:.3f} ft')
    for b in BUCKETS:
        m, s = agg['overall_ade'][b]
        share = agg['bucket_share'][b]
        contrib = m * share
        print(f'  {b:>9s}: ADE = {m:.3f} ± {s:.3f} ft   '
              f'share = {share*100:5.1f}%   '
              f'contribution = {contrib:.3f} ft '
              f'({contrib / agg["overall_ball_ade"][0] * 100:4.1f}%)')
    print(f'\nwrote {out_path}')


if __name__ == '__main__':
    main()
