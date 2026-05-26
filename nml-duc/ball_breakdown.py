"""Per-regime breakdown of validation ball error for a trained checkpoint.

Splits ball L2 error three ways to localize where the model fails:

  1. Per-horizon ADE, ball vs players (does ball error grow faster?).
  2. Ball ADE bucketed by GT step-speed (held / dribble / pass / flight).
  3. Window-level: ball ADE for windows where the ball was held at the last
     context frame vs already in flight (does it fail at the handoff?).

The val set is the same one used during training (deterministic, keyed on
the checkpoint's --seed and --val-split via the config).

Usage:
    python ball_breakdown.py --config metrics/seed_0/config.json \
                             --model-path model.pth
"""

import argparse
import json

import torch
from torch.utils.data import DataLoader

from data import NBASampler
from trainer import NBATrainer


def _load_trainer(cfg_path: str, model_path: str, train_path: str,
                  device: str) -> NBATrainer:
    with open(cfg_path) as f:
        cfg = json.load(f)
    return NBATrainer(
        batch_size=cfg.get('batch_size', 64),
        epochs=0,
        train_path=train_path,
        val_split=cfg.get('val_split', 0.1),
        context_size=cfg.get('context_size', 8),
        horizon_size=cfg.get('horizon_size', 12),
        input_dim=cfg.get('input_dim', 4),
        output_dim=cfg.get('output_dim', 2),
        state_dim=cfg['state_dim'],
        num_layers=cfg['num_layers'],
        coord_scale=cfg['coord_scale'],
        use_kinematics=cfg.get('use_kinematics', True),
        use_possession=cfg.get('use_possession', True),
        use_ball_head=cfg.get('use_ball_head', False),
        use_handler_relative_ball=cfg.get('use_handler_relative_ball', False),
        windows_per_sequence=1,
        augment=False,
        seed=cfg.get('seed', 0),
        model_path=model_path,
        device=device,
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', required=True,
                   help='Path to config.json from the training run.')
    p.add_argument('--model-path', default='model.pth')
    p.add_argument('--train-path', default='data/train/train')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--speed-buckets', default='0.5,2.0,5.0',
                   help='Comma-separated ft/frame thresholds defining ball '
                        'speed buckets (held / dribble / pass / flight).')
    args = p.parse_args()

    thresh = [float(x) for x in args.speed_buckets.split(',')]
    bucket_labels = [f'held    (< {thresh[0]})']
    for i in range(len(thresh) - 1):
        bucket_labels.append(f'mid {i+1}  ({thresh[i]} - {thresh[i+1]})')
    bucket_labels.append(f'flight  (>= {thresh[-1]})')
    n_buckets = len(thresh) + 1

    trainer = _load_trainer(args.config, args.model_path, args.train_path, args.device)
    trainer._load_model()
    val_ds = trainer.val_dataset
    if len(val_ds) == 0:
        raise SystemExit('val set is empty — re-check --config val_split / --train-path')
    sampler = NBASampler(trainer.batch_size, val_ds.max_start,
                         seed=trainer.seed, shuffle=False, windows_per_sequence=1)
    loader = DataLoader(val_ds, batch_size=trainer.batch_size, sampler=sampler)

    H = trainer.horizon_size
    ball_h_sum, ball_h_cnt = torch.zeros(H), 0
    plyr_h_sum, plyr_h_cnt = torch.zeros(H), 0
    bkt_sum, bkt_cnt = torch.zeros(n_buckets), torch.zeros(n_buckets)
    held_sum, held_cnt = 0.0, 0
    move_sum, move_cnt = 0.0, 0

    boundaries = torch.tensor(thresh)
    with torch.no_grad():
        for X, y in loader:
            X = X.to(args.device); y = y.to(args.device)
            pred = trainer._predict_positions(X)                    # [H, B*N, 2] ft
            B, N = X.size(0), X.size(2)
            target = y[..., :2].permute(1, 0, 2, 3).reshape(H, B * N, 2)
            dist = (pred - target).norm(dim=-1).cpu()               # [H, B*N]

            team_ids = y[:, 0, :, 3].reshape(-1).cpu()              # [B*N]
            ball_rows = (team_ids == 0)
            plyr_rows = ~ball_rows
            ball_dist = dist[:, ball_rows]                          # [H, B]
            plyr_dist = dist[:, plyr_rows]                          # [H, B*10]
            ball_h_sum += ball_dist.sum(dim=1); ball_h_cnt += ball_dist.size(1)
            plyr_h_sum += plyr_dist.sum(dim=1); plyr_h_cnt += plyr_dist.size(1)

            # Per-sample ball trajectory (exactly one ball per sample) via masked sum.
            ball_mask = (X[:, 0, :, 3] == 0).unsqueeze(1).unsqueeze(-1)  # [B,1,N,1]
            ctx_ball = (X[..., :2] * ball_mask).sum(dim=2).cpu()        # [B, C, 2]
            gt_ball  = (y[..., :2] * ball_mask).sum(dim=2).cpu()        # [B, H, 2]
            # Speed at horizon step h = ||gt[h] - prev||; prev = last ctx ball at h=0.
            prev = torch.cat([ctx_ball[:, -1:], gt_ball[:, :-1]], dim=1)  # [B, H, 2]
            step_speed = (gt_ball - prev).norm(dim=-1)                   # [B, H]

            speed_flat = step_speed.reshape(-1)
            err_flat = ball_dist.transpose(0, 1).reshape(-1)             # same ordering
            bkt = torch.bucketize(speed_flat, boundaries)                # 0..n_buckets-1
            for k in range(n_buckets):
                m = (bkt == k)
                bkt_sum[k] += err_flat[m].sum()
                bkt_cnt[k] += m.sum()

            # Window class from ball speed at the last context frame.
            handoff_speed = (ctx_ball[:, -1] - ctx_ball[:, -2]).norm(dim=-1)  # [B]
            is_held = handoff_speed < thresh[0]
            mean_win_err = ball_dist.mean(dim=0)                              # [B]
            held_sum += mean_win_err[is_held].sum().item();  held_cnt += int(is_held.sum())
            move_sum += mean_win_err[~is_held].sum().item(); move_cnt += int((~is_held).sum())

    print(f'\nVal windows: {ball_h_cnt}\n')
    print('Per-horizon ADE (ft):')
    print('  step | ball   | players | ratio')
    for h in range(H):
        b = (ball_h_sum[h] / ball_h_cnt).item()
        q = (plyr_h_sum[h] / plyr_h_cnt).item()
        print(f'   {h+1:>3} | {b:>6.3f} | {q:>7.3f} | {b/q:>5.2f}x')

    total = int(bkt_cnt.sum().item())
    print('\nBall ADE by GT step-speed bucket (ft):')
    print('  bucket                          |  count (share) | ade')
    for k, label in enumerate(bucket_labels):
        n = int(bkt_cnt[k].item())
        ade = (bkt_sum[k] / max(n, 1)).item()
        share = 100 * n / max(total, 1)
        print(f'  {label:<32}|{n:>7} ({share:4.1f}%) | {ade:.3f}')

    print('\nWindow class by handoff speed (ball motion at last context frame):')
    if held_cnt:
        print(f'  held    (< {thresh[0]} ft/frame): n={held_cnt:>5}  ball ADE = {held_sum/held_cnt:.3f}')
    if move_cnt:
        print(f'  moving  (>= {thresh[0]} ft/frame): n={move_cnt:>5}  ball ADE = {move_sum/move_cnt:.3f}')


if __name__ == '__main__':
    main()
