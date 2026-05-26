"""Quick error-pattern analysis from viz/data.js.

Loads the 100 val samples produced by make_data.py and reports several
breakdowns we can act on: per-horizon-step error, ball-vs-player split,
under/over-shoot of motion magnitude, error vs context speed, etc.
"""

import json
import math
import os
import re
import statistics
from collections import defaultdict


def load_data(path='viz/data.js'):
    with open(path) as f:
        s = f.read()
    m = re.match(r'^window\.VIZ_DATA\s*=\s*(.*);\s*$', s, re.S)
    return json.loads(m.group(1))


def vsub(a, b):
    return [a[0] - b[0], a[1] - b[1]]


def vnorm(v):
    return math.hypot(v[0], v[1])


def main():
    d = load_data()
    samples = d['samples']
    H = d['meta']['horizon_size']
    C = d['meta']['context_size']
    print(f'{len(samples)} samples, context={C}, horizon={H}')
    print(f'mean ADE = {d["meta"]["aggregate"]["mean_ade"]:.3f} ft, '
          f'mean FDE = {d["meta"]["aggregate"]["mean_fde"]:.3f} ft')
    print()

    # ---------- 1. Per-step error (mean + ball vs player split) ----------
    step_err_all = [[] for _ in range(H)]
    step_err_ball = [[] for _ in range(H)]
    step_err_player = [[] for _ in range(H)]
    for s in samples:
        teams = s['teams']
        pred = s['pred_future']
        gt = s['gt_future']
        for t in range(H):
            for n, team in enumerate(teams):
                d_ = vnorm(vsub(pred[t][n], gt[t][n]))
                step_err_all[t].append(d_)
                (step_err_ball if team == 0 else step_err_player)[t].append(d_)

    print('per-step error (ft):')
    print(f'  {"step":>4}  {"all":>6}  {"ball":>6}  {"player":>7}')
    for t in range(H):
        print(f'  {t+1:>4}  {statistics.mean(step_err_all[t]):>6.3f}  '
              f'{statistics.mean(step_err_ball[t]):>6.3f}  '
              f'{statistics.mean(step_err_player[t]):>7.3f}')

    # ---------- 2. Per-entity (ball vs player) overall ADE/FDE ----------
    ball_ade, ball_fde = [], []
    pl_ade, pl_fde = [], []
    for s in samples:
        teams = s['teams']
        pred, gt = s['pred_future'], s['gt_future']
        for n, team in enumerate(teams):
            errs = [vnorm(vsub(pred[t][n], gt[t][n])) for t in range(H)]
            if team == 0:
                ball_ade.append(sum(errs) / H); ball_fde.append(errs[-1])
            else:
                pl_ade.append(sum(errs) / H); pl_fde.append(errs[-1])
    print('\nball  ADE={:.3f}  FDE={:.3f}   (n={})'.format(
        statistics.mean(ball_ade), statistics.mean(ball_fde), len(ball_ade)))
    print('player ADE={:.3f}  FDE={:.3f}   (n={})'.format(
        statistics.mean(pl_ade), statistics.mean(pl_fde), len(pl_ade)))

    # ---------- 3. Distance traveled by pred vs GT (under/over-shoot?) ----------
    pred_path_len = defaultdict(list)  # 'ball' / 'player' -> [path_len]
    gt_path_len = defaultdict(list)
    pred_disp = defaultdict(list)
    gt_disp = defaultdict(list)
    for s in samples:
        teams = s['teams']
        pred, gt = s['pred_future'], s['gt_future']
        last_ctx = s['context'][-1]
        for n, team in enumerate(teams):
            key = 'ball' if team == 0 else 'player'
            pl = 0.0
            gl = 0.0
            prev_p, prev_g = last_ctx[n], last_ctx[n]
            for t in range(H):
                pl += vnorm(vsub(pred[t][n], prev_p))
                gl += vnorm(vsub(gt[t][n], prev_g))
                prev_p = pred[t][n]
                prev_g = gt[t][n]
            pred_path_len[key].append(pl)
            gt_path_len[key].append(gl)
            pred_disp[key].append(vnorm(vsub(pred[-1][n], last_ctx[n])))
            gt_disp[key].append(vnorm(vsub(gt[-1][n], last_ctx[n])))

    print('\nmotion magnitude over horizon (ft):')
    for key in ['ball', 'player']:
        pp = statistics.mean(pred_path_len[key])
        gp = statistics.mean(gt_path_len[key])
        pd = statistics.mean(pred_disp[key])
        gd = statistics.mean(gt_disp[key])
        print(f'  {key:>6}  path_len pred={pp:.2f}  gt={gp:.2f}  ratio={pp/gp:.3f}  '
              f'|  net-disp pred={pd:.2f}  gt={gd:.2f}  ratio={pd/gd:.3f}')

    # ---------- 4. Error vs context speed ----------
    print('\nerror vs context-frame motion magnitude (per-entity FDE):')
    bins = [(0, 1), (1, 2), (2, 3), (3, 5), (5, 99)]
    binned = {b: [] for b in bins}
    for s in samples:
        teams = s['teams']
        ctx = s['context']
        pred, gt = s['pred_future'], s['gt_future']
        for n, team in enumerate(teams):
            # average speed in context = total ctx path / (C-1) frames
            pl = sum(vnorm(vsub(ctx[k+1][n], ctx[k][n])) for k in range(C - 1)) / (C - 1)
            fde = vnorm(vsub(pred[-1][n], gt[-1][n]))
            for lo, hi in bins:
                if lo <= pl < hi:
                    binned[(lo, hi)].append(fde); break
    print(f'  ctx-speed bin (ft/frame)   N      FDE')
    for b in bins:
        n = len(binned[b])
        if n:
            print(f'  [{b[0]:>3}, {b[1]:>3})              {n:>4}   {statistics.mean(binned[b]):.3f}')

    # ---------- 5. Heading drift: does pred curve away from GT heading? ----------
    # At each horizon step compute angle between pred displacement and gt
    # displacement; report mean |angle| in degrees.
    ang_step = [[] for _ in range(H)]
    for s in samples:
        teams = s['teams']
        pred, gt = s['pred_future'], s['gt_future']
        last_ctx = s['context'][-1]
        prev_p = [last_ctx[n] for n in range(len(teams))]
        prev_g = [last_ctx[n] for n in range(len(teams))]
        for t in range(H):
            for n in range(len(teams)):
                dp = vsub(pred[t][n], prev_p[n])
                dg = vsub(gt[t][n], prev_g[n])
                np_, ng = vnorm(dp), vnorm(dg)
                if np_ > 0.05 and ng > 0.05:
                    cos = (dp[0]*dg[0] + dp[1]*dg[1]) / (np_ * ng)
                    cos = max(-1, min(1, cos))
                    ang_step[t].append(math.degrees(math.acos(cos)))
                prev_p[n] = pred[t][n]
                prev_g[n] = gt[t][n]
    print('\nmean |heading mismatch| per step (deg):')
    for t in range(H):
        print(f'  step {t+1:>2}  N={len(ang_step[t]):>4}  '
              f'{statistics.mean(ang_step[t]):.1f}')

    # ---------- 6. Sample distribution of fde ----------
    fdes = [s['fde'] for s in samples]
    print(f'\nsample FDE percentiles (ft): p10={sorted(fdes)[10]:.2f}  '
          f'p50={sorted(fdes)[50]:.2f}  p90={sorted(fdes)[90]:.2f}  '
          f'max={max(fdes):.2f}')


if __name__ == '__main__':
    main()
