"""Load one or more checkpoints and write the Kaggle submission for the test set.

With a single checkpoint, behaves like the original single-model inference.
With multiple checkpoints, runs a *step-wise ensemble*: at each rollout step
every model predicts a delta from the shared current position, the deltas are
averaged, and the averaged position is fed back to every model as the input
for the next step (see `model.ensemble_rollout`). This differs from the
seed-averaging in `run.py`, which averages each member's fully-rolled-out
horizon at the very end.

All ensemble members must share `context_size`, `horizon_size`, and
`coord_scale`; other architecture knobs (`state_dim`, `num_layers`, feature
flags, head sizes) can differ.

Usage:
    python inference.py                                         # defaults
    python inference.py --tta                                   # 8x test-time aug
    python inference.py --model-paths m_seed0.pth m_seed1.pth m_seed2.pth \
                        --configs metrics/seed_0/config.json    # one shared config
    python inference.py --model-paths a.pth b.pth \
                        --configs cfg_a.json cfg_b.json         # per-model configs

The architecture flags (--state-dim, --num-layers, --coord-scale, --input-dim,
--output-dim, --context-size, --horizon-size, --use-ball-head) MUST match the
values used when each checkpoint was trained, otherwise load_state_dict raises
a shape mismatch. Pass --configs with one config.json per model (or a single
shared one) to populate them automatically.
"""

import argparse
import json
import os
from typing import List, Optional

import torch

from diffusion import DiffusionNBA, GNNContextEncoder, TrajectoryDenoiser
from model import NBAModel, ensemble_rollout
from trainer import NBATrainer

# Architecture keys whose values affect the model's state_dict shapes
# OR its forward-pass topology / normalization. Must round-trip with
# NBATrainer._build_model / _config_dict.
_ARCH_KEYS = ('state_dim', 'num_layers', 'coord_scale', 'context_size',
              'horizon_size', 'input_dim', 'output_dim',
              'use_kinematics', 'use_possession', 'use_basket_dist',
              'use_ball_head', 'use_handler_relative_ball', 'use_na_ball',
              'use_handler_relations', 'use_rel_vel_edge_attr',
              'use_ball_centric', 'player_edge_radius',
              'cell_type', 'num_heads',
              # Diffusion-specific keys (ignored for --model-type ar):
              'model_type', 'diffusion_T', 'ddim_steps', 'num_samples',
              'denoiser_layers', 'denoiser_heads')

# Mirrors NBATrainer._TTA_TRANSFORMS: independent x-flip x y-flip x team-swap.
_TTA_TRANSFORMS = tuple((fx, fy, ft)
                        for fx in (False, True)
                        for fy in (False, True)
                        for ft in (False, True))


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--model-paths', nargs='+', default=['model.pth'],
                   help='One or more checkpoints (state_dict). Multiple paths '
                        'trigger a step-wise ensemble rollout.')
    p.add_argument('--configs', nargs='+', default=None,
                   help='One config.json per checkpoint, or a single config '
                        'shared across all. If omitted, architecture defaults '
                        'come from the CLI flags below.')
    p.add_argument('--test-path', default='data/test/test',
                   help='Directory of Kaggle test .pt sequences (8-frame context-only).')
    p.add_argument('--submission-dir', default='submission',
                   help='Output directory for solution.csv (created if missing).')

    arch = p.add_argument_group('default architecture (overridden by --configs)')
    arch.add_argument('--model-type', dest='model_type',
                      choices=('ar', 'diffusion'), default='ar',
                      help='Checkpoint family. "ar" uses ensemble_rollout '
                           '(step-wise mean of per-model deltas). "diffusion" '
                           'builds DiffusionNBA per model and averages '
                           'per-model DDIM-sampled trajectories.')
    arch.add_argument('--state-dim', type=int, default=128)
    arch.add_argument('--num-layers', type=int, default=1)
    arch.add_argument('--coord-scale', type=float, default=1.0)
    arch.add_argument('--context-size', type=int, default=8)
    arch.add_argument('--horizon-size', type=int, default=12)
    arch.add_argument('--input-dim', type=int, default=4)
    arch.add_argument('--output-dim', type=int, default=2)
    arch.add_argument('--no-kinematics', dest='use_kinematics',
                      action='store_false', default=True,
                      help='Checkpoint was trained without kinematic features '
                           '(acc, speed, heading).')
    arch.add_argument('--no-possession', dest='use_possession',
                      action='store_false', default=True,
                      help='Checkpoint was trained without possession features.')
    arch.add_argument('--use-basket-dist', dest='use_basket_dist',
                      action='store_true', default=False,
                      help='Checkpoint was trained with per-node basket-distance '
                           'features.')
    arch.add_argument('--use-ball-head', dest='use_ball_head', action='store_true',
                      default=False,
                      help='Checkpoint was trained with a separate ball projection head.')
    arch.add_argument('--use-handler-relative-ball', dest='use_handler_relative_ball',
                      action='store_true', default=False,
                      help='Checkpoint was trained with the handler-relative ball '
                           'reparametrization (anchor + offset).')
    arch.add_argument('--use-na-ball', dest='use_na_ball',
                      action='store_true', default=False,
                      help='Checkpoint was trained with the non-autoregressive '
                           'ball head.')
    arch.add_argument('--use-handler-relations', dest='use_handler_relations',
                      action='store_true', default=False,
                      help='Checkpoint was trained with the handler-aware '
                           '11-relation edge typing.')
    arch.add_argument('--use-rel-vel-edge-attr', dest='use_rel_vel_edge_attr',
                      action='store_true', default=False,
                      help='Checkpoint was trained with relative-velocity edge '
                           'attributes (edge_dim 6 instead of 3).')
    arch.add_argument('--use-ball-centric', dest='use_ball_centric',
                      action='store_true', default=False,
                      help='Checkpoint was trained with ball-centric topology '
                           '(dense player↔ball edges, distance-gated player↔player).')
    arch.add_argument('--player-edge-radius', dest='player_edge_radius',
                      type=float, default=20.0,
                      help='Distance gate (feet) for player↔player edges under '
                           '--use-ball-centric. Must match the trained checkpoint.')
    arch.add_argument('--cell-type', dest='cell_type',
                      choices=('gnn', 'transformer'), default='gnn',
                      help='Spatial layer type. Must match the trained checkpoint.')
    arch.add_argument('--num-heads', dest='num_heads', type=int, default=4,
                      help='Attention heads when --cell-type=transformer. '
                           'Must match the trained checkpoint.')
    arch.add_argument('--diffusion-T', dest='diffusion_T', type=int, default=1000,
                      help='[diffusion only] Number of training noise levels '
                           '(cosine ᾱ schedule). Must match the trained '
                           'checkpoint when the schedule choice matters; in '
                           'practice the schedule is deterministic in T.')
    arch.add_argument('--ddim-steps', dest='ddim_steps', type=int, default=20,
                      help='[diffusion only] DDIM sampling steps at inference.')
    arch.add_argument('--num-samples', dest='num_samples', type=int, default=8,
                      help='[diffusion only] Number of DDIM trajectories '
                           'averaged per model at inference.')
    arch.add_argument('--denoiser-layers', dest='denoiser_layers', type=int,
                      default=4,
                      help='[diffusion only] Transformer-encoder layers; must '
                           'match the trained checkpoint.')
    arch.add_argument('--denoiser-heads', dest='denoiser_heads', type=int,
                      default=4,
                      help='[diffusion only] Attention heads; must match the '
                           'trained checkpoint and divide --state-dim.')

    p.add_argument('--tta', action='store_true',
                   help='Test-time augmentation: average the ensemble over the '
                        '8 court-symmetry transforms (x-flip x y-flip x team-swap).')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu',
                   help='Torch device string (e.g. cuda, cuda:0, cpu).')
    return p


def _resolve_arch(args: argparse.Namespace, parser: argparse.ArgumentParser,
                  cfg_path: Optional[str]) -> dict:
    """Per-model arch dict: start from CLI flags, overlay values from `cfg_path`
    for any flag the user didn't explicitly set (detected vs parser defaults)."""
    arch = {k: getattr(args, k) for k in _ARCH_KEYS}
    if cfg_path is None:
        return arch
    with open(cfg_path) as f:
        cfg = json.load(f)
    defaults = {a.dest: a.default for a in parser._actions}
    for k in _ARCH_KEYS:
        if k in cfg and getattr(args, k) == defaults.get(k):
            arch[k] = cfg[k]
    return arch


def _build_one(arch: dict, device: str) -> torch.nn.Module:
    if arch.get('model_type', 'ar') == 'diffusion':
        encoder = GNNContextEncoder(
            input_dim=arch['input_dim'], state_dim=arch['state_dim'],
            context_size=arch['context_size'], num_layers=arch['num_layers'],
            use_kinematics=arch['use_kinematics'],
            use_possession=arch['use_possession'],
            coord_scale=arch['coord_scale'])
        denoiser = TrajectoryDenoiser(
            state_dim=arch['state_dim'], horizon=arch['horizon_size'],
            num_entities=11,
            num_layers=arch['denoiser_layers'],
            num_heads=arch['denoiser_heads'])
        return DiffusionNBA(
            encoder=encoder, denoiser=denoiser,
            horizon=arch['horizon_size'],
            T=arch['diffusion_T'],
            ddim_steps=arch['ddim_steps'],
            num_samples=arch['num_samples']).to(device)
    return NBAModel(
        input_dim=arch['input_dim'], output_dim=arch['output_dim'],
        state_dim=arch['state_dim'], context_size=arch['context_size'],
        horizon_size=arch['horizon_size'], num_layers=arch['num_layers'],
        coord_scale=arch['coord_scale'],
        use_kinematics=arch['use_kinematics'],
        use_possession=arch['use_possession'],
        use_basket_dist=arch['use_basket_dist'],
        use_ball_head=arch['use_ball_head'],
        use_handler_relative_ball=arch['use_handler_relative_ball'],
        use_na_ball=arch['use_na_ball'],
        use_handler_relations=arch['use_handler_relations'],
        use_rel_vel_edge_attr=arch['use_rel_vel_edge_attr'],
        use_ball_centric=arch['use_ball_centric'],
        player_edge_radius=arch['player_edge_radius'],
        cell_type=arch.get('cell_type', 'gnn'),
        num_heads=arch.get('num_heads', 4),
    ).to(device)


def _load_models(args: argparse.Namespace,
                 parser: argparse.ArgumentParser) -> tuple:
    paths = args.model_paths
    configs = args.configs
    if configs is not None and len(configs) not in (1, len(paths)):
        raise ValueError(f'--configs must have 1 or {len(paths)} entries '
                         f'to match --model-paths, got {len(configs)}')
    models: List[torch.nn.Module] = []
    coord_scales = []
    model_types = []
    for i, path in enumerate(paths):
        if not os.path.isfile(path):
            raise FileNotFoundError(f'checkpoint not found: {path}')
        cfg_path = None
        if configs:
            cfg_path = configs[i] if len(configs) == len(paths) else configs[0]
        arch = _resolve_arch(args, parser, cfg_path)
        m = _build_one(arch, args.device)
        state = torch.load(path, weights_only=True, map_location=args.device)
        m.load_state_dict(state)
        m.eval()
        models.append(m)
        coord_scales.append(arch['coord_scale'])
        model_types.append(arch.get('model_type', 'ar'))
    if len(set(coord_scales)) != 1:
        raise ValueError(f'ensemble members must share coord_scale, got {coord_scales}')
    if len(set(model_types)) != 1:
        raise ValueError(f'ensemble members must share model_type, got {model_types}')
    return models, coord_scales[0], model_types[0]


@torch.no_grad()
def _predict(models: List[torch.nn.Module], X: torch.Tensor,
             coord_scale: float, model_type: str,
             tta: bool) -> torch.Tensor:
    """X: [B, T, N, F] in raw court coordinates. Returns [H, B*N, 2] in feet.

    AR ensemble: step-wise mean of per-model deltas (`ensemble_rollout`).
    Diffusion ensemble: mean of per-model DDIM-sampled trajectories
    (each model samples independently with its own context encoder)."""
    def _normalize(x: torch.Tensor) -> torch.Tensor:
        out = x.clone()
        out[..., :2] = out[..., :2] / coord_scale
        return out

    def _once(X_in: torch.Tensor) -> torch.Tensor:
        Xn = _normalize(X_in)
        if model_type == 'diffusion':
            sampled = torch.stack([m.sample(Xn) for m in models]).mean(dim=0)
            return sampled * coord_scale
        return ensemble_rollout(models, Xn) * coord_scale

    if not tta:
        return _once(X)
    preds = []
    for fx, fy, ft in _TTA_TRANSFORMS:
        X_aug = X.clone()
        if fx: X_aug[..., 0] = -X_aug[..., 0]
        if fy: X_aug[..., 1] = -X_aug[..., 1]
        if ft: X_aug[..., 3] = -X_aug[..., 3]  # team-swap; positions unaffected
        p = _once(X_aug)
        if fx: p[..., 0] = -p[..., 0]
        if fy: p[..., 1] = -p[..., 1]
        preds.append(p)
    return torch.stack(preds).mean(dim=0)


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    if not os.path.isdir(args.test_path):
        raise NotADirectoryError(f'--test-path is not a directory: {args.test_path}')
    os.makedirs(args.submission_dir, exist_ok=True)

    models, coord_scale, model_type = _load_models(args, parser)
    context_size = models[0].context_size
    print(f'[load] ensemble of {len(models)} model(s) ({model_type})  '
          f'device={args.device}' + ('  tta=on' if args.tta else ''),
          flush=True)

    ids, all_preds = [], []
    for fname in sorted(os.listdir(args.test_path)):
        if not fname.endswith('.pt'):
            continue
        seq = torch.load(os.path.join(args.test_path, fname), weights_only=False)
        X_ctx = seq[:context_size].unsqueeze(0).to(args.device)
        p = _predict(models, X_ctx, coord_scale, model_type, args.tta).cpu()  # [H, 1*N, 2]
        H, N = p.size(0), X_ctx.size(2)
        ids.append(int(fname.removesuffix('.pt')))
        all_preds.append(p.reshape(H, N, 2))

    NBATrainer.write_kaggle_csv(ids, torch.stack(all_preds), args.submission_dir)
    print(f'[submit] wrote {os.path.join(args.submission_dir, "solution.csv")}',
          flush=True)


if __name__ == '__main__':
    main()
