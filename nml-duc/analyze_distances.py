"""Analyze player-player distances in the training set.

Goal: pick a sensible cutoff for distance-gated player-player edges in the
ball-centric topology. We report the distribution of per-frame, per-(i,j)
distances among the 10 players (ball excluded) over a sample of training
sequences, plus the distribution of the k-th nearest neighbor distance for
each player (helps decide whether k-NN or a hard radius is more appropriate).

Run:
    python analyze_distances.py [--limit 500] [--data data/train/train]
"""
import argparse
import glob
import os
import random

import numpy as np
import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='data/train/train')
    p.add_argument('--limit', type=int, default=500,
                   help='number of randomly sampled sequences')
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()

    files = sorted(glob.glob(os.path.join(args.data, '*.pt')))
    if not files:
        raise SystemExit(f'no .pt files in {args.data}')
    rng = random.Random(args.seed)
    if len(files) > args.limit:
        files = rng.sample(files, args.limit)
    print(f'sampling {len(files)} sequences')

    # Buckets we will fill.
    all_player_player = []         # every pair-frame distance among players
    knn_distances = {k: [] for k in (1, 2, 3, 4, 5)}  # per-player k-th NN

    total_frames = 0
    for path in files:
        seq = torch.load(path, weights_only=False)  # [T, N, F]
        if seq.dim() != 3 or seq.shape[1] != 11:
            continue
        team = seq[0, :, 3]                          # [N]
        player_idx = (team != 0).nonzero(as_tuple=True)[0]  # 10 entries
        if player_idx.numel() != 10:
            continue
        pos = seq[:, player_idx, :2].numpy()         # [T, 10, 2]
        T = pos.shape[0]
        total_frames += T

        # Pairwise distances per frame: [T, 10, 10]
        diff = pos[:, :, None, :] - pos[:, None, :, :]
        d = np.linalg.norm(diff, axis=-1)
        iu = np.triu_indices(10, k=1)
        all_player_player.append(d[:, iu[0], iu[1]].reshape(-1))

        # k-th NN per player: sort each row, skip self (column 0 after sort).
        d_sorted = np.sort(d, axis=-1)               # [T, 10, 10]
        for k in knn_distances:
            knn_distances[k].append(d_sorted[:, :, k].reshape(-1))

    if not all_player_player:
        raise SystemExit('no valid sequences found')

    pp = np.concatenate(all_player_player)
    print(f'\ntotal frames considered: {total_frames:,}')
    print(f'total player-player pair-frame samples: {pp.size:,}')

    pcts = (5, 10, 25, 50, 75, 90, 95, 99)
    print('\n=== player-player pairwise distance (feet) ===')
    print(f'  mean   {pp.mean():7.2f}')
    print(f'  std    {pp.std():7.2f}')
    print(f'  min    {pp.min():7.2f}')
    print(f'  max    {pp.max():7.2f}')
    for q in pcts:
        print(f'  p{q:<3d}   {np.percentile(pp, q):7.2f}')

    print('\n=== k-th nearest-neighbor distance per player (feet) ===')
    print(f'{"k":>3} | {"mean":>6} | ' + ' | '.join(f'p{q:>2}' for q in pcts))
    for k, vals in knn_distances.items():
        v = np.concatenate(vals)
        line = f'{k:>3} | {v.mean():>6.2f} | ' + ' | '.join(
            f'{np.percentile(v, q):>5.2f}' for q in pcts
        )
        print(line)

    print('\n=== fraction of player-player pairs within radius R ===')
    for R in (5, 8, 10, 12, 15, 18, 20, 25, 30, 35, 40):
        frac = (pp <= R).mean()
        avg_deg = frac * 9  # each player has 9 potential player-player neighbors
        print(f'  R={R:>2} ft → keep {frac*100:>5.2f}% of pairs '
              f'(avg player-player degree ≈ {avg_deg:.2f})')


if __name__ == '__main__':
    main()
