"""Train the model and write a Kaggle submission.

Usage:
    python run.py
"""

import argparse
import os

import torch

from trainer import NBATrainer


def _parse_seeds(s: str):
    """Parse a comma-separated seed spec, e.g. "0" or "0,1,2,3,4"."""
    out = [int(x.strip()) for x in s.split(',') if x.strip()]
    if not out:
        raise argparse.ArgumentTypeError('--seeds must contain at least one integer')
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    data = p.add_argument_group('data')
    data.add_argument('--train-path', default='data/train/train',
                      help='Directory of training .pt sequences.')
    data.add_argument('--test-path', default='data/test/test',
                      help='Directory of Kaggle test .pt sequences (8-frame context-only files).')
    data.add_argument('--submission-dir', default='submission',
                      help='Output directory for solution.csv (created if missing).')
    data.add_argument('--val-split', type=float, default=0.1,
                      help='Fraction of --train-path carved off as a held-out validation set, '
                           'deterministically split by --seed. Set 0 to train on everything.')
    data.add_argument('--windows-per-sequence', type=int, default=5,
                      help='Number of random (context+horizon) windows drawn per training '
                           'sequence per epoch. Higher = more SGD steps per epoch and better '
                           'coverage of each clip; linearly increases per-epoch wall time.')
    data.add_argument('--augment', dest='augment', action='store_true', default=True,
                      help='Apply symmetry-preserving data augmentation on training '
                           'windows: random x/y court flips, team-label swap, and within-team '
                           'player permutation. Validation/test windows are never augmented.')
    data.add_argument('--no-augment', dest='augment', action='store_false',
                      help='Disable training-time data augmentation.')

    model = p.add_argument_group('model')
    model.add_argument('--model', dest='model_type', choices=('ar', 'diffusion'),
                       default='ar',
                       help='Top-level model family. "ar" (default) is the '
                            'autoregressive RGAT-GRU rollout. "diffusion" '
                            'replaces the rollout with a transformer denoiser '
                            'on top of the same GNN context encoder, trained '
                            'to denoise residuals to a constant-velocity '
                            'anchor; predictions are DDIM-sampled and averaged '
                            'over --num-samples trajectories. Different '
                            'objective — train/val losses are NOT comparable '
                            'across model types; use val_per_entity.csv for '
                            'cross-model evaluation.')
    model.add_argument('--state-dim', type=int, default=128,
                       help='Hidden-state width of the RGAT/GRU cell. With '
                            '--model diffusion this is also the denoiser '
                            'token width (must be divisible by '
                            '--denoiser-heads).')
    model.add_argument('--num-layers', type=int, default=1,
                       help='Number of stacked HeteroSTGCNCells per timestep; '
                            'layers >= 2 add a residual connection from the previous '
                            'layer. Each extra layer widens the graph receptive field '
                            'by one hop.')
    model.add_argument('--ball-head', dest='use_ball_head', action='store_true',
                       default=False,
                       help='Use a separate projection head for the ball at the final '
                            'state→Δxy step. The shared RGAT stack still does spatial '
                            'reasoning; only the per-step delta projection branches by '
                            'entity type, letting the ball specialize on its high-frequency '
                            'motion. Off by default (preserves checkpoint compatibility).')
    model.add_argument('--no-ball-head', dest='use_ball_head', action='store_false',
                       help='Disable the ball-specific projection head (default).')
    model.add_argument('--handler-relative-ball', dest='use_handler_relative_ball',
                       action='store_true', default=False,
                       help='Reparametrize the ball as anchor + offset, where anchor '
                            'is the soft-possession-weighted average of the predicted '
                            'player positions and offset is a small learned residual '
                            'from a dedicated head (zero-init). Decisively ties the '
                            'ball to the handler during possession; offset compensates '
                            'during flight. Off by default. Use --no-handler-relative-ball '
                            'to make the off-state explicit (e.g. for ablation scripts).')
    model.add_argument('--no-handler-relative-ball', dest='use_handler_relative_ball',
                       action='store_false',
                       help='Disable handler-relative ball reparametrization (default).')
    model.add_argument('--na-ball', dest='use_na_ball', action='store_true',
                       default=False,
                       help='Predict all H horizon ball positions in one shot '
                            'from the ball-node hidden state at the last context '
                            'step, as deltas from the last context ball position. '
                            'Players still roll out autoregressively but see the '
                            'NA ball through the graph. Mutually exclusive with '
                            '--handler-relative-ball. Off by default.')
    model.add_argument('--no-na-ball', dest='use_na_ball', action='store_false',
                       help='Disable non-autoregressive ball head (default).')
    model.add_argument('--kinematics', dest='use_kinematics', action='store_true',
                       default=True,
                       help='Append per-node kinematic features (acceleration, '
                            'speed, heading) to the RGAT input. Default on; '
                            'use --no-kinematics to ablate.')
    model.add_argument('--no-kinematics', dest='use_kinematics', action='store_false',
                       help='Ablate the kinematic features (acc, speed, heading). '
                            'Velocity is still included if --velocity is on.')
    model.add_argument('--possession', dest='use_possession', action='store_true',
                       default=True,
                       help='Append [dist_to_ball, soft_possession] per node to '
                            'the RGAT input. Default on; use --no-possession to '
                            'ablate.')
    model.add_argument('--no-possession', dest='use_possession', action='store_false',
                       help='Ablate the possession features.')
    model.add_argument('--basket-dist', dest='use_basket_dist',
                       action='store_true', default=False,
                       help='Append per-node L2 distance to each basket '
                            '(rim centers at (±41.75, 0) ft) to the RGAT input. '
                            'Cheap spatial prior; off by default.')
    model.add_argument('--no-basket-dist', dest='use_basket_dist',
                       action='store_false',
                       help='Disable basket-distance features (default).')
    model.add_argument('--handler-relations', dest='use_handler_relations',
                       action='store_true', default=False,
                       help='Expand the heterogeneous edge typing from 5 to 11 '
                            'relations by splitting each player↔ball / teammate / '
                            'opponent edge by whether the src or dst is the ball-'
                            'handler (closest player to the ball, re-derived per '
                            'timestep). Lets RGAT learn distinct weights for on-ball '
                            'vs off-ball interactions. Off by default. '
                            'Adds (num_handler_relations - 5) × state_dim² params '
                            'per layer; checkpoints from runs with this flag differ '
                            'shape and cannot be loaded into a non-handler model.')
    model.add_argument('--no-handler-relations', dest='use_handler_relations',
                       action='store_false',
                       help='Use the legacy 5-relation typing (default).')
    model.add_argument('--rel-vel-edge-attr', dest='use_rel_vel_edge_attr',
                       action='store_true', default=False,
                       help='Append per-edge relative velocity features '
                            '[Δvx, Δvy, closing_speed] to the existing [Δx, Δy, '
                            '‖Δ‖] edge attrs (edge_dim 3 → 6). Closing speed is '
                            'the signed approach rate along the edge direction. '
                            'Requires --no-no-edge-attr (the default). Off by default; '
                            'increases the RGAT edge MLP slightly.')
    model.add_argument('--no-rel-vel-edge-attr', dest='use_rel_vel_edge_attr',
                       action='store_false',
                       help='Use the legacy 3-dim edge attrs (default).')
    model.add_argument('--ball-centric', dest='use_ball_centric',
                       action='store_true', default=False,
                       help='Switch the graph topology from fully-connected to '
                            'ball-centric: every player↔ball edge and self-loop '
                            'is kept, while player↔player edges are pruned per '
                            'step if the endpoints are farther apart than '
                            '--player-edge-radius feet. The mask is re-derived '
                            'from current (rolled-out) positions, so the player '
                            'subgraph is time-varying. Off by default.')
    model.add_argument('--no-ball-centric', dest='use_ball_centric',
                       action='store_false',
                       help='Use the fully-connected topology (default).')
    model.add_argument('--player-edge-radius', dest='player_edge_radius',
                       type=float, default=20.0,
                       help='Distance gate (feet) for player↔player edges when '
                            '--ball-centric is on. Dataset-derived defaults: 15 ft '
                            '≈ direct matchup + 3-NN (avg degree ~3), 20 ft ≈ '
                            'tactical neighborhood (avg degree ~4.6, ~51%% pairs '
                            'kept), 25 ft ≈ extended help defense (~69%% pairs). '
                            'No effect when --ball-centric is off.')
    model.add_argument('--cell-type', dest='cell_type',
                       choices=('gnn', 'transformer'), default='gnn',
                       help='Spatial layer inside each HeteroSTGCN cell. '
                            "'gnn' = RGAT (default); 'transformer' = "
                            'relation-aware multi-head self-attention with '
                            'per-(relation, head) bias. The temporal GRU and '
                            'all feature flags work the same in both.')
    model.add_argument('--num-heads', dest='num_heads', type=int, default=4,
                       help='Number of attention heads when --cell-type=transformer. '
                            'Must divide --state-dim. Ignored for --cell-type=gnn.')
    model.add_argument('--coord-scale', type=float, default=1.0,
                       help='Position normalization scale (court half-width, feet). '
                            'The trainer divides input (x,y) by this before the model '
                            'sees them and rescales predictions back on the way out, so '
                            'the network sees O(1) features but reported losses/ADE and '
                            'the Kaggle submission stay in court units (feet, feet²). '
                            'Set to 1.0 to disable normalization.')

    diff = p.add_argument_group('diffusion (only used when --model diffusion)')
    diff.add_argument('--diffusion-T', type=int, default=1000,
                      help='Number of noise levels in the training schedule '
                           '(DDPM cosine ᾱ).')
    diff.add_argument('--ddim-steps', type=int, default=20,
                      help='Number of DDIM sampling steps at inference. '
                           'Lower = faster but less accurate; 20 is a '
                           'reasonable default.')
    diff.add_argument('--num-samples', type=int, default=8,
                      help='Number of DDIM sample trajectories averaged into '
                           'the final prediction. 1 = single sample (high '
                           'variance, captures one mode); 8 is a good default '
                           'for the deterministic Kaggle ADE metric.')
    diff.add_argument('--denoiser-layers', type=int, default=4,
                      help='Number of transformer-encoder layers in the '
                           'denoiser.')
    diff.add_argument('--denoiser-heads', type=int, default=4,
                      help='Attention heads in the denoiser; must divide '
                           '--state-dim.')

    opt = p.add_argument_group('optimization')
    opt.add_argument('--batch-size', type=int, default=64,
                     help='Mini-batch size in (clip, window) units.')
    opt.add_argument('--epochs', type=int, default=50,
                     help='Maximum number of training epochs (no early stopping; best '
                          'checkpoint is kept regardless).')
    opt.add_argument('--lr', type=float, default=1e-3,
                     help='Initial Adam learning rate.')
    opt.add_argument('--weight-decay', type=float, default=5e-4,
                     help='Adam L2 weight decay.')
    opt.add_argument('--scheduler', choices=('plateau', 'cosine'), default='cosine',
                     help='LR schedule. "plateau" uses ReduceLROnPlateau on val_loss '
                          '(or train_loss if no val). "cosine" uses CosineAnnealingLR '
                          'with T_max=--epochs.')
    opt.add_argument('--scheduler-patience', type=int, default=5,
                     help='[plateau only] Epochs of no val-loss improvement to wait '
                          'before ReduceLROnPlateau drops the LR.')
    opt.add_argument('--scheduler-factor', type=float, default=0.5,
                     help='[plateau only] Multiplicative factor applied to the LR '
                          'when the plateau scheduler fires (e.g. 0.5 halves it).')
    opt.add_argument('--eta-min', type=float, default=1e-5,
                     help='[cosine only] Minimum LR reached at the end of the '
                          'CosineAnnealingLR schedule.')
    opt.add_argument('--teacher-forcing', dest='teacher_forcing', action='store_true',
                     default=False,
                     help='Use scheduled sampling during training: at each horizon '
                          'step, with probability ε(epoch) feed the ground-truth '
                          'previous position to the rollout instead of the model '
                          'prediction. Inverse-sigmoid decay (Bengio 2015): '
                          'ε(epoch) = tf_k / (tf_k + exp(epoch / tf_k)).')
    opt.add_argument('--no-teacher-forcing', dest='teacher_forcing', action='store_false',
                     help='Disable scheduled sampling (pure autoregressive training).')
    opt.add_argument('--tf-k', type=float, default=10.0,
                     help='Decay-rate parameter for the inverse-sigmoid teacher-forcing '
                          'schedule. Larger = slower decay. Scale with --epochs; tf_k≈10 '
                          'fits a 50-epoch run (ε≈0.91→0.06).')
    opt.add_argument('--ball-weight', type=float, default=1.0,
                     help='Per-entity weight on the ball in the training loss. 1.0 (default) '
                          'is plain MSE — each of the 11 entities contributes equally, so the '
                          'ball is 1/11 of the loss and the optimizer is dominated by player '
                          'error. Values > 1 upweight the ball; e.g. 10.0 makes the ball count '
                          'roughly as much as all 10 players combined. Reported train/val '
                          'losses are no longer comparable to runs at a different setting; use '
                          'per-entity ADE (val_per_entity.csv) for cross-run comparison.')
    opt.add_argument('--ball-loss', choices=('mse', 'huber'), default='mse',
                     help='Per-element loss applied to the ball channel. Default mse keeps '
                          'the original behavior; huber swaps the ball to smooth-L1 (clips '
                          'gradient magnitude beyond --ball-huber-delta) to fight '
                          'mode-averaging that shrinks predicted ball motion to ~70%% of '
                          'true magnitude. Players stay on MSE either way.')
    opt.add_argument('--ball-huber-delta', type=float, default=1.0,
                     help='Transition point (β) for Huber/smooth-L1 on the ball, in the '
                          'training units the loss sees (positions / coord_scale; with the '
                          'default coord_scale=1.0 this is feet). Quadratic for |Δ| < β, '
                          'linear beyond. 1.0 ft is a reasonable starting point given '
                          'observed ball errors in the 0.5–6 ft range. Ignored unless '
                          '--ball-loss huber.')
    infer = p.add_argument_group('inference')
    infer.add_argument('--tta', dest='tta', action='store_true', default=False,
                       help='Enable test-time augmentation: at inference, run each test '
                            'sequence through all 8 court-symmetry transforms (x-flip, '
                            'y-flip, team-swap) and average the inverse-transformed '
                            'predictions. Affects the Kaggle submission and the val '
                            'breakdown CSVs, but NOT the per-epoch train/val passes.')
    infer.add_argument('--no-tta', dest='tta', action='store_false',
                       help='Disable test-time augmentation (default).')

    report = p.add_argument_group('report')
    report.add_argument('--metrics-dir', dest='metrics_dir', default='metrics',
                        help='Directory to write per-epoch metrics.csv, end-of-training '
                             'val_per_horizon.csv / val_per_entity.csv, and config.json. '
                             'Use --no-metrics to disable.')
    report.add_argument('--no-metrics', dest='metrics_dir', action='store_const', const=None,
                        help='Disable metrics dump (overrides --metrics-dir).')

    misc = p.add_argument_group('misc')
    misc.add_argument('--seeds', type=_parse_seeds, default=[0],
                      help='Comma-separated master seed(s); each seed drives the train/val '
                           'split, sampler, weight init, and CUDA RNG for that run. '
                           'Single seed (default "0") = one model, identical to the legacy '
                           '--seed behavior. Multiple seeds (e.g. "0,1,2,3,4") = train one '
                           'model per seed and average their test predictions for the Kaggle '
                           'submission. Per-seed checkpoints are saved as <model_path stem>'
                           '_seed{N}.<ext> and metrics under <metrics_dir>/seed_{N}/.')
    misc.add_argument('--model-path', default='model.pth',
                      help='Path where the best-by-val-loss checkpoint is saved (and '
                           'reloaded before the Kaggle submission).')
    misc.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu',
                      help='Torch device string (e.g. cuda, cuda:0, cpu).')
    return p.parse_args()


def _build_trainer(args, seed: int, model_path: str, metrics_dir):
    return NBATrainer(
        batch_size=args.batch_size,
        epochs=args.epochs,
        train_path=args.train_path,
        val_split=args.val_split,
        state_dim=args.state_dim,
        num_layers=args.num_layers,
        coord_scale=args.coord_scale,
        use_kinematics=args.use_kinematics,
        use_possession=args.use_possession,
        use_basket_dist=args.use_basket_dist,
        use_ball_head=args.use_ball_head,
        use_handler_relative_ball=args.use_handler_relative_ball,
        use_na_ball=args.use_na_ball,
        use_handler_relations=args.use_handler_relations,
        use_rel_vel_edge_attr=args.use_rel_vel_edge_attr,
        use_ball_centric=args.use_ball_centric,
        player_edge_radius=args.player_edge_radius,
        cell_type=args.cell_type,
        num_heads=args.num_heads,
        lr=args.lr,
        weight_decay=args.weight_decay,
        windows_per_sequence=args.windows_per_sequence,
        augment=args.augment,
        scheduler=args.scheduler,
        scheduler_patience=args.scheduler_patience,
        scheduler_factor=args.scheduler_factor,
        eta_min=args.eta_min,
        teacher_forcing=args.teacher_forcing,
        tf_k=args.tf_k,
        ball_weight=args.ball_weight,
        ball_loss=args.ball_loss,
        huber_delta=args.ball_huber_delta,
        tta=args.tta,
        metrics_dir=metrics_dir,
        model_path=model_path,
        device=args.device,
        seed=seed,
        model_type=args.model_type,
        diffusion_T=args.diffusion_T,
        ddim_steps=args.ddim_steps,
        num_samples=args.num_samples,
        denoiser_layers=args.denoiser_layers,
        denoiser_heads=args.denoiser_heads,
    )


def main() -> None:
    args = parse_args()
    os.makedirs(args.submission_dir, exist_ok=True)
    out_csv = os.path.join(args.submission_dir, 'solution.csv')
    seeds = args.seeds

    if len(seeds) == 1:
        trainer = _build_trainer(args, seed=seeds[0],
                                 model_path=args.model_path,
                                 metrics_dir=args.metrics_dir)
        print(f'[data] train={len(trainer.train_dataset)} val={len(trainer.val_dataset)} '
              f'(val_split={args.val_split})', flush=True)
        trainer.train()
        trainer.get_kaggle_submission(test_dir=args.test_path,
                                      target_dir=args.submission_dir)
        print(f'[submit] wrote {out_csv}', flush=True)
        return

    # Multi-seed ensemble: train each independently, then average test predictions.
    stem, ext = os.path.splitext(args.model_path)
    ids, seed_preds = None, []
    for k, s in enumerate(seeds):
        mpath = f'{stem}_seed{s}{ext}'
        mdir = (os.path.join(args.metrics_dir, f'seed_{s}')
                if args.metrics_dir else None)
        print(f'\n=== seed {s} ({k+1}/{len(seeds)}) → {mpath} ===', flush=True)
        trainer = _build_trainer(args, seed=s, model_path=mpath, metrics_dir=mdir)
        if k == 0:
            print(f'[data] train={len(trainer.train_dataset)} val={len(trainer.val_dataset)} '
                  f'(val_split={args.val_split})', flush=True)
        trainer.train()
        sid, spreds = trainer.predict_test_raw(args.test_path)
        if ids is None:
            ids = sid
        elif sid != ids:
            raise RuntimeError('test-file ordering changed across seeds — refusing to average')
        seed_preds.append(spreds)

    mean_preds = torch.stack(seed_preds).mean(dim=0)
    NBATrainer.write_kaggle_csv(ids, mean_preds, args.submission_dir)
    print(f'\n[ensemble] averaged {len(seeds)} seeds → {out_csv}', flush=True)


if __name__ == '__main__':
    main()
