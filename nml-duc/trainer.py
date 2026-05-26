"""Training, evaluation, and submission utilities for NBA trajectory prediction."""

import csv
import json
import math
import os
import random
import time
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import torch
from IPython.display import HTML
from matplotlib.animation import FuncAnimation
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from data import NBADataset, NBASampler, split_train_val
from diffusion import DiffusionNBA, GNNContextEncoder, TrajectoryDenoiser
from model import MultiStepMSE, NBAModel


class NBATrainer:
    ENTITY_MAPPING = {-1: 'Team_A', 0: 'Ball', 1: 'Team_B'}

    def __init__(self, batch_size: int, epochs: int, train_path: str,
                 val_path: Optional[str] = None, val_split: float = 0.0,
                 device: str = 'cuda', context_size: int = 8, horizon_size: int = 12,
                 input_dim: int = 4, output_dim: int = 2, state_dim: int = 64,
                 num_layers: int = 2, coord_scale: float = 47.5,
                 use_kinematics: bool = True,
                 use_possession: bool = True,
                 use_basket_dist: bool = False,
                 use_ball_head: bool = False,
                 use_handler_relative_ball: bool = False,
                 use_na_ball: bool = False,
                 use_handler_relations: bool = False,
                 use_rel_vel_edge_attr: bool = False,
                 use_ball_centric: bool = False,
                 player_edge_radius: float = 20.0,
                 cell_type: str = 'gnn', num_heads: int = 4,
                 lr: float = 1e-3, weight_decay: float = 5e-4,
                 scheduler: str = 'plateau',
                 scheduler_patience: int = 5, scheduler_factor: float = 0.5,
                 eta_min: float = 0.0,
                 teacher_forcing: bool = True, tf_k: float = 10.0,
                 ball_weight: float = 1.0,
                 ball_loss: str = 'mse', huber_delta: float = 1.0,
                 grad_clip: float = 10.0, model_path: str = 'model.pth',
                 metrics_dir: Optional[str] = None,
                 windows_per_sequence: int = 5, augment: bool = True, seed: int = 0,
                 tta: bool = False,
                 model_type: str = 'ar',
                 diffusion_T: int = 1000, ddim_steps: int = 20,
                 num_samples: int = 8, denoiser_layers: int = 4,
                 denoiser_heads: int = 4):
        """If `val_split > 0`, carve that fraction off `train_path` (deterministically,
        keyed on `seed`) and ignore `val_path`. With `scheduler='plateau'` the LR is
        reduced on plateau of val_loss (or train_loss when val is empty); with
        `scheduler='cosine'` a `CosineAnnealingLR(T_max=epochs, eta_min=eta_min)`
        steps the LR every epoch regardless of loss. The best checkpoint (by val_loss
        or train_loss) is the one written to `model_path` in either case.

        `tta` enables test-time augmentation at inference (8 court-symmetry
        transforms averaged): affects `_predict_positions`, the Kaggle
        submission and `evaluate_breakdown`, but not the per-epoch
        training/val passes."""
        self.batch_size = batch_size
        self.epochs = epochs
        self.train_path = train_path
        self.val_path = val_path
        self.val_split = val_split
        self.device = device
        self.context_size = context_size
        self.horizon_size = horizon_size
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.state_dim = state_dim
        self.num_layers = num_layers
        if coord_scale <= 0:
            raise ValueError(f'coord_scale must be > 0, got {coord_scale}')
        self.coord_scale = float(coord_scale)
        self.use_kinematics = bool(use_kinematics)
        self.use_possession = bool(use_possession)
        self.use_basket_dist = bool(use_basket_dist)
        self.use_ball_head = bool(use_ball_head)
        self.use_handler_relative_ball = bool(use_handler_relative_ball)
        self.use_na_ball = bool(use_na_ball)
        self.use_handler_relations = bool(use_handler_relations)
        self.use_rel_vel_edge_attr = bool(use_rel_vel_edge_attr)
        self.use_ball_centric = bool(use_ball_centric)
        self.player_edge_radius = float(player_edge_radius)
        if cell_type not in ('gnn', 'transformer'):
            raise ValueError(f"cell_type must be 'gnn' or 'transformer', got {cell_type!r}")
        self.cell_type = cell_type
        self.num_heads = int(num_heads)
        self.lr = lr
        self.weight_decay = weight_decay
        if scheduler not in ('plateau', 'cosine'):
            raise ValueError(f"scheduler must be 'plateau' or 'cosine', got {scheduler!r}")
        self.scheduler = scheduler
        self.scheduler_patience = scheduler_patience
        self.scheduler_factor = scheduler_factor
        self.eta_min = eta_min
        self.teacher_forcing = teacher_forcing
        if tf_k <= 0:
            raise ValueError(f'tf_k must be > 0, got {tf_k}')
        self.tf_k = float(tf_k)
        self.ball_weight = float(ball_weight)
        self.ball_loss = ball_loss
        self.huber_delta = float(huber_delta)
        self.grad_clip = grad_clip
        self.model_path = model_path
        self.windows_per_sequence = windows_per_sequence
        self.augment = augment
        self.seed = seed
        self.tta = bool(tta)
        if model_type not in ('ar', 'diffusion'):
            raise ValueError(f"model_type must be 'ar' or 'diffusion', got {model_type!r}")
        self.model_type = model_type
        self.diffusion_T = int(diffusion_T)
        self.ddim_steps = int(ddim_steps)
        self.num_samples = int(num_samples)
        self.denoiser_layers = int(denoiser_layers)
        self.denoiser_heads = int(denoiser_heads)
        # Train uses the configured loss (Huber-for-ball, ball reweighting, …);
        # val is always plain MSE so val_loss, the best-checkpoint monitor, and
        # ReduceLROnPlateau remain comparable across ball_loss/ball_weight runs.
        # Per-entity ADE (val_per_entity.csv) is loss-independent in either case.
        self.train_loss_fn = MultiStepMSE(ball_weight=self.ball_weight,
                                          ball_loss=self.ball_loss,
                                          huber_delta=self.huber_delta)
        self.val_loss_fn = MultiStepMSE()  # plain MSE, ball_weight=1.0
        self.metrics_dir = metrics_dir
        self._metrics_csv = None
        if metrics_dir:
            os.makedirs(metrics_dir, exist_ok=True)
            self._metrics_csv = os.path.join(metrics_dir, 'metrics.csv')
            with open(self._metrics_csv, 'w', newline='') as f:
                csv.writer(f).writerow(
                    ['epoch', 'train_loss', 'val_loss', 'lr', 'tf_prob', 'time_s'])
            with open(os.path.join(metrics_dir, 'config.json'), 'w') as f:
                json.dump(self._config_dict(), f, indent=2)

        if val_split > 0:
            train_files, val_files = split_train_val(train_path, val_split, seed=seed)
            self.train_dataset = NBADataset(None, context_size, horizon_size,
                                            files=train_files, augment=augment)
            self.val_dataset = NBADataset(None, context_size, horizon_size,
                                          files=val_files, augment=False)
        else:
            self.train_dataset = NBADataset(train_path, context_size, horizon_size,
                                            augment=augment)
            self.val_dataset = (NBADataset(val_path, context_size, horizon_size, augment=False)
                                if val_path else
                                NBADataset(None, context_size, horizon_size, files=[]))
        self.model: Optional[torch.nn.Module] = None

    # ---------- setup ----------

    def _set_seed(self) -> torch.Generator:
        torch.manual_seed(self.seed)
        random.seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        # DataLoader requires a CPU generator even when training on cuda.
        return torch.Generator().manual_seed(self.seed)

    def _build_model(self) -> torch.nn.Module:
        if self.model_type == 'diffusion':
            encoder = GNNContextEncoder(
                input_dim=self.input_dim, state_dim=self.state_dim,
                context_size=self.context_size, num_layers=self.num_layers,
                use_kinematics=self.use_kinematics,
                use_possession=self.use_possession,
                coord_scale=self.coord_scale)
            denoiser = TrajectoryDenoiser(
                state_dim=self.state_dim, horizon=self.horizon_size,
                num_entities=11,
                num_layers=self.denoiser_layers,
                num_heads=self.denoiser_heads)
            return DiffusionNBA(encoder=encoder, denoiser=denoiser,
                                horizon=self.horizon_size,
                                T=self.diffusion_T,
                                ddim_steps=self.ddim_steps,
                                num_samples=self.num_samples).to(self.device)
        return NBAModel(input_dim=self.input_dim, output_dim=self.output_dim,
                        state_dim=self.state_dim, context_size=self.context_size,
                        horizon_size=self.horizon_size, num_layers=self.num_layers,
                        coord_scale=self.coord_scale,
                        use_kinematics=self.use_kinematics,
                        use_possession=self.use_possession,
                        use_basket_dist=self.use_basket_dist,
                        use_ball_head=self.use_ball_head,
                        use_handler_relative_ball=self.use_handler_relative_ball,
                        use_na_ball=self.use_na_ball,
                        use_handler_relations=self.use_handler_relations,
                        use_rel_vel_edge_attr=self.use_rel_vel_edge_attr,
                        use_ball_centric=self.use_ball_centric,
                        player_edge_radius=self.player_edge_radius,
                        cell_type=self.cell_type,
                        num_heads=self.num_heads,
                        ).to(self.device)

    def _normalize_xy(self, t: Tensor) -> Tensor:
        """Return a copy of `t` with the first two feature channels divided by
        `coord_scale`. Team-ID and other static channels are left untouched."""
        out = t.clone()
        out[..., :2] = t[..., :2] / self.coord_scale
        return out

    def _load_model(self) -> torch.nn.Module:
        """Cache and return the trained model for inference."""
        if self.model is None:
            self.model = self._build_model()
            state = torch.load(self.model_path, weights_only=True, map_location=self.device)
            self.model.load_state_dict(state)
        self.model.eval()
        return self.model

    # ---------- metrics dump ----------

    def _config_dict(self) -> dict:
        """JSON-serializable snapshot of the run config; written to config.json."""
        return {
            'batch_size': self.batch_size, 'epochs': self.epochs,
            'context_size': self.context_size, 'horizon_size': self.horizon_size,
            'input_dim': self.input_dim, 'output_dim': self.output_dim,
            'state_dim': self.state_dim, 'num_layers': self.num_layers,
            'coord_scale': self.coord_scale,
            'use_kinematics': self.use_kinematics,
            'use_possession': self.use_possession,
            'use_basket_dist': self.use_basket_dist,
            'use_ball_head': self.use_ball_head,
            'use_handler_relative_ball': self.use_handler_relative_ball,
            'use_na_ball': self.use_na_ball,
            'use_handler_relations': self.use_handler_relations,
            'use_rel_vel_edge_attr': self.use_rel_vel_edge_attr,
            'use_ball_centric': self.use_ball_centric,
            'player_edge_radius': self.player_edge_radius,
            'cell_type': self.cell_type,
            'num_heads': self.num_heads,
            'lr': self.lr, 'weight_decay': self.weight_decay,
            'scheduler': self.scheduler, 'scheduler_patience': self.scheduler_patience,
            'scheduler_factor': self.scheduler_factor, 'eta_min': self.eta_min,
            'teacher_forcing': self.teacher_forcing, 'tf_k': self.tf_k,
            'ball_weight': self.ball_weight,
            'ball_loss': self.ball_loss, 'huber_delta': self.huber_delta,
            'grad_clip': self.grad_clip,
            'windows_per_sequence': self.windows_per_sequence,
            'augment': self.augment, 'seed': self.seed,
            'val_split': self.val_split, 'model_path': self.model_path,
            'tta': self.tta,
            'model_type': self.model_type,
            'diffusion_T': self.diffusion_T, 'ddim_steps': self.ddim_steps,
            'num_samples': self.num_samples,
            'denoiser_layers': self.denoiser_layers,
            'denoiser_heads': self.denoiser_heads,
        }

    def _append_metrics(self, epoch: int, train_loss: Optional[float],
                        val_loss: Optional[float], lr: float, tf_prob: float,
                        t_sec: float) -> None:
        if not self._metrics_csv:
            return
        with open(self._metrics_csv, 'a', newline='') as f:
            csv.writer(f).writerow([
                epoch + 1,
                f'{train_loss:.6f}' if train_loss is not None else '',
                f'{val_loss:.6f}'   if val_loss   is not None else '',
                f'{lr:.6e}', f'{tf_prob:.6f}', f'{t_sec:.2f}',
            ])

    def evaluate_breakdown(self, loader: DataLoader) -> tuple:
        """Return `(per_horizon_ade, per_entity_ade)` on `loader`.

        per_horizon_ade: list[float] of length H, average L2 displacement at each
            horizon step, micro-averaged over (sample, entity).
        per_entity_ade: dict[label, float] for labels in {'Team_A','Ball','Team_B'},
            average L2 displacement over (sample, horizon step, entity-of-class).

        Uses the cached model if available, otherwise loads from `model_path`.
        Honors `self.tta` — when on, predictions are TTA-averaged before metrics.
        """
        self._load_model()
        H = self.horizon_size
        h_sum = torch.zeros(H)
        h_count = 0
        ent_sum = {-1.0: 0.0, 0.0: 0.0, 1.0: 0.0}
        ent_count = {-1.0: 0, 0.0: 0, 1.0: 0}
        with torch.no_grad():
            for X, y in loader:
                X = X.to(self.device)
                y = y.to(self.device)
                pred = self._predict_positions(X)                # [H, B*N, 2], feet
                B, N = X.size(0), X.size(2)
                target = y[..., :2].permute(1, 0, 2, 3).reshape(H, B * N, 2)
                dist = (pred - target).norm(dim=-1).cpu()        # [H, B*N], feet
                h_sum += dist.sum(dim=1)
                h_count += dist.size(1)
                team_ids = y[:, 0, :, 3].reshape(-1).cpu()       # [B*N]
                for ent in ent_sum:
                    mask = team_ids == ent
                    if mask.any():
                        ent_sum[ent] += dist[:, mask].sum().item()
                        ent_count[ent] += int(mask.sum().item()) * H
        per_h = (h_sum / max(h_count, 1)).tolist()
        labels = {-1.0: 'Team_A', 0.0: 'Ball', 1.0: 'Team_B'}
        per_ent = {labels[k]: ent_sum[k] / ent_count[k]
                   for k in ent_sum if ent_count[k]}
        return per_h, per_ent

    def _save_val_breakdown(self, val_loader: DataLoader) -> None:
        if not self.metrics_dir or len(self.val_dataset) == 0:
            return
        per_h, per_ent = self.evaluate_breakdown(val_loader)
        with open(os.path.join(self.metrics_dir, 'val_per_horizon.csv'),
                  'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['horizon_step', 'ade'])
            for i, v in enumerate(per_h):
                w.writerow([i + 1, f'{v:.6f}'])
        with open(os.path.join(self.metrics_dir, 'val_per_entity.csv'),
                  'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['entity', 'ade'])
            for label in ('Team_A', 'Ball', 'Team_B'):
                if label in per_ent:
                    w.writerow([label, f'{per_ent[label]:.6f}'])

    # ---------- training ----------

    def train(self):
        generator = self._set_seed()
        train_sampler = NBASampler(self.batch_size, self.train_dataset.max_start,
                                   seed=self.seed, shuffle=True,
                                   windows_per_sequence=self.windows_per_sequence)
        val_sampler = NBASampler(self.batch_size, self.val_dataset.max_start,
                                 seed=self.seed, shuffle=False, windows_per_sequence=1)
        train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size,
                                  sampler=train_sampler, generator=generator)
        val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size,
                                sampler=val_sampler, generator=generator)

        model = self._build_model()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        if self.scheduler == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.epochs, eta_min=self.eta_min)
        else:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=self.scheduler_factor,
                patience=self.scheduler_patience)

        n_params = sum(p.numel() for p in model.parameters())
        print(f'[setup] {model.__class__.__name__}: {n_params:,} params  '
              f'train_batches={len(train_loader)}  val_batches={len(val_loader)}  '
              f'windows_per_seq={self.windows_per_sequence}  device={self.device}',
              flush=True)

        best_loss = float('inf')
        best_epoch = -1
        global_t0 = time.time()
        for epoch in range(self.epochs):
            train_sampler.set_epoch(epoch)
            val_sampler.set_epoch(0)
            ep_t0 = time.time()
            tf_prob = self._tf_prob(epoch)
            train_loss = self._run_epoch(train_loader, model, optimizer,
                                         split='train', epoch=epoch, tf_prob=tf_prob)
            with torch.no_grad():
                val_loss = self._run_epoch(val_loader, model, optimizer,
                                           split='val', epoch=epoch)
            monitor = val_loss if val_loss is not None else train_loss
            improved = False
            if self.scheduler == 'cosine':
                scheduler.step()
            elif monitor is not None:
                scheduler.step(monitor)
            if monitor is not None and monitor < best_loss:
                best_loss = monitor
                best_epoch = epoch
                torch.save(model.state_dict(), self.model_path)
                improved = True
            lr = optimizer.param_groups[0]['lr']

            parts = [f'[epoch {epoch+1:>3}/{self.epochs}]',
                     f'train={train_loss:.4f}' if train_loss is not None else 'train=  n/a ']
            if val_loss is not None:
                parts.append(f'val={val_loss:.4f}')
            ep_dt = time.time() - ep_t0
            parts += [f'lr={lr:.2e}', f'time={ep_dt:.1f}s']
            if self.teacher_forcing:
                parts.append(f'tf={tf_prob:.2f}')
            if improved:
                parts.append('*best*')
            print('  '.join(parts), flush=True)
            self._append_metrics(epoch, train_loss, val_loss, lr, tf_prob, ep_dt)

        # Restore best weights into the cached model so inference uses them.
        if best_epoch >= 0:
            model.load_state_dict(torch.load(self.model_path, weights_only=True,
                                             map_location=self.device))
        else:
            torch.save(model.state_dict(), self.model_path)
        self.model = model
        print(f'[done] total={time.time()-global_t0:.1f}s  '
              f'best=epoch {best_epoch+1} loss={best_loss:.4f}  '
              f'checkpoint={self.model_path}', flush=True)
        self._save_val_breakdown(val_loader)

    def _tf_prob(self, epoch: int) -> float:
        """Bengio 2015 inverse-sigmoid schedule: ε(i) = k / (k + exp(i / k)).
        ε(0) ≈ 1 (mostly teacher-forced), ε → 0 as training proceeds. The
        `min(..., 30)` clamp avoids overflow when epoch >> tf_k."""
        if not self.teacher_forcing:
            return 0.0
        k = self.tf_k
        return k / (k + math.exp(min(epoch / k, 30.0)))

    def _run_epoch(self, dataloader: DataLoader,
                   model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                   split: str, epoch: int, tf_prob: float = 0.0) -> Optional[float]:
        if split == 'train':
            model.train()
        else:
            model.eval()
        total_loss = 0.0
        batches = 0
        bar = tqdm(dataloader, desc=f'  ep {epoch+1:>2} {split:5s}', leave=False,
                   dynamic_ncols=True, mininterval=0.3)
        # Internally the model trains in normalized space (positions ÷ coord_scale)
        # so gradient magnitudes are decoupled from court-unit scale. Reported
        # losses are rescaled back to court units (feet²) so they're comparable
        # to un-normalized runs — with one exception: the diffusion *train* loss
        # is denoising MSE on normalized residuals (no physical unit, not
        # comparable to AR). The diffusion *val* loss runs full DDIM sampling
        # and reports trajectory MSE in feet², so val IS comparable to AR and
        # to Kaggle.
        scale_sq = self.coord_scale ** 2
        is_diffusion = self.model_type == 'diffusion'
        # Seed val sampling so the diffusion val loss is deterministic across
        # epochs given fixed weights — any change in val_loss then reflects
        # model improvement, not sampling noise.
        if is_diffusion and split == 'val':
            torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.seed)
        for X, y in bar:
            X = X.to(self.device)
            y = y.to(self.device)
            X_n = self._normalize_xy(X)
            y_n = self._normalize_xy(y)
            if is_diffusion and split == 'train':
                # Cheap denoising-MSE objective at a random noise level — this
                # is what provides the gradient signal. Not in feet².
                loss = model.loss(X_n, y_n)
                rescale = 1.0
            elif is_diffusion:
                # Val: full DDIM sampling, then trajectory MSE in feet².
                pred = model(X_n)                                  # [H, B*N, 2]
                loss = self.val_loss_fn.compute(pred, y_n)
                rescale = scale_sq
            else:
                pred = model(X_n, y=y_n, tf_prob=tf_prob)
                loss_fn = self.train_loss_fn if split == 'train' else self.val_loss_fn
                loss = loss_fn.compute(pred, y_n)
                rescale = scale_sq
            if split == 'train':
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip)
                optimizer.step()
            total_loss += loss.item() * rescale
            batches += 1
            bar.set_postfix(loss=f'{total_loss / batches:.4f}')
        bar.close()
        if not batches:
            return None
        avg = total_loss / batches
        return avg

    # ---------- inference ----------

    # The 8 court-symmetry transforms used for TTA: independent x-flip, y-flip,
    # and team-swap. The training augmenter exploits exactly these symmetries
    # ([[data.py]] `_augment`); applying them at inference and averaging the
    # inverse-transformed predictions cancels asymmetric biases the model
    # picks up from the static team-ID feature.
    _TTA_TRANSFORMS = tuple((fx, fy, ft)
                            for fx in (False, True)
                            for fy in (False, True)
                            for ft in (False, True))

    def _predict_positions(self, X: Tensor) -> Tensor:
        """X: [B, T, N, F] in raw court coordinates. Returns [H, B*N, 2] in feet.

        If `self.tta` is True, runs the model under each of the 8 symmetry
        transforms (x-flip × y-flip × team-swap), inverse-transforms the
        spatial predictions, and averages. Otherwise a single forward pass.
        Assumes the model is in eval mode (set by `_load_model`).
        """
        model = self._load_model()
        if not self.tta:
            return model(self._normalize_xy(X)) * self.coord_scale
        preds = []
        for fx, fy, ft in self._TTA_TRANSFORMS:
            X_aug = X.clone()
            if fx: X_aug[..., 0] = -X_aug[..., 0]
            if fy: X_aug[..., 1] = -X_aug[..., 1]
            if ft: X_aug[..., 3] = -X_aug[..., 3]  # team-swap; positions unaffected
            p = model(self._normalize_xy(X_aug)) * self.coord_scale  # [H, B*N, 2]
            if fx: p[..., 0] = -p[..., 0]
            if fy: p[..., 1] = -p[..., 1]
            preds.append(p)
        return torch.stack(preds).mean(dim=0)

    def get_trajectory(self, X: Tensor) -> Tensor:
        """Predict the horizon for a single context sequence and stitch onto it.

        X: [T, N, F] in raw court coordinates. Returns [T + H, N, F] (predicted
        frames carry the static features of the last input frame). The model
        runs in normalized space; predictions are denormalized back to court
        units before stitching.
        """
        X_ctx = X[:self.context_size].unsqueeze(0).to(self.device)
        with torch.no_grad():
            pred = self._predict_positions(X_ctx).cpu()  # [H, N, 2] (B=1 folded)
        H, N = pred.size(0), X.size(1)
        pred = pred.reshape(H, N, 2)
        static = X[-1, :, 2:].unsqueeze(0).expand(H, -1, -1)
        pred = torch.cat([pred, static], dim=-1)
        return torch.cat([X, pred], dim=0).detach()

    def predict_test_raw(self, test_dir: str):
        """Run inference over every `.pt` file in `test_dir`.

        Returns `(ids, preds)` where `ids` is a `list[int]` of test sequence IDs
        in the order they were read and `preds` is a `[n_files, H, N, 2]` tensor
        of predicted positions in court units. Used by `get_kaggle_submission`
        for the single-model case and by `run.py` to average across an ensemble
        of seeds before writing the CSV.
        """
        self._load_model()
        ids, all_preds = [], []
        with torch.no_grad():
            for fname in sorted(os.listdir(test_dir)):
                if not fname.endswith('.pt'):
                    continue
                seq = torch.load(os.path.join(test_dir, fname), weights_only=False)
                X_ctx = seq[:self.context_size].unsqueeze(0).to(self.device)
                p = self._predict_positions(X_ctx).cpu()  # [H, 1*N, 2]
                H, N = p.size(0), X_ctx.size(2)
                ids.append(int(fname.removesuffix('.pt')))
                all_preds.append(p.reshape(H, N, 2))
        return ids, torch.stack(all_preds)

    @staticmethod
    def write_kaggle_csv(ids, preds: Tensor, target_dir: str) -> pd.DataFrame:
        """Write `solution.csv` from raw predictions.

        `preds`: `[n_files, H, N, 2]` tensor of court-unit positions, in the same
        order as `ids`. Column layout (t outer, entity middle, axis inner) matches
        what the Kaggle grader expects — do not reorder.
        """
        _, H, N, _ = preds.shape
        rows = [[i, *p.reshape(-1).tolist()] for i, p in zip(ids, preds)]
        cols = ['id'] + [f'entity_{i}_time_{t}_{axis}'
                         for t in range(H)
                         for i in range(N)
                         for axis in ('x', 'y')]
        df = pd.DataFrame(rows, columns=cols).set_index('id').sort_index()
        df.to_csv(os.path.join(target_dir, 'solution.csv'))
        return df

    def get_kaggle_submission(self, test_dir: str, target_dir: str) -> pd.DataFrame:
        """Single-model submission: predict, then write `solution.csv`."""
        ids, preds = self.predict_test_raw(test_dir)
        return self.write_kaggle_csv(ids, preds, target_dir)

    # ---------- visualization ----------

    def animate_sequence(self, sequence: Tensor, interval: int = 50,
                         pred_seq: Optional[Tensor] = None):
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.set_xlim(0, 95)
        ax.set_ylim(0, 50)

        entities = sequence[:, :, -1].unique()
        gt_palette = dict(zip(['GT_' + self.ENTITY_MAPPING[e.item()] for e in entities],
                              ['#BBDEFB', '#000000', '#1E88E5']))
        palette = dict(gt_palette)
        if pred_seq is not None:
            palette.update(zip(['Pred_' + self.ENTITY_MAPPING[e.item()] for e in entities],
                               ['#FFCDD2', '#FFFFFF', '#D32F2F']))

        scatters = {name: ax.scatter([], [], s=40, color=color, label=name)
                    for name, color in palette.items()}
        ax.legend()

        def update(frame: int):
            fig.suptitle(f'Frame {frame + 1}')
            for name, scatter in scatters.items():
                entity_id = next(eid for eid, ename in self.ENTITY_MAPPING.items() if ename in name)
                source = sequence if name.startswith('GT') else pred_seq
                if source is None or (name.startswith('Pred') and frame <= self.context_size):
                    continue
                frame_data = source[frame].numpy()
                mask = frame_data[:, -1] == entity_id
                xy = frame_data[mask][:, :2].copy()
                xy[:, 0] += 47.5
                xy[:, 1] += 25
                scatter.set_offsets(xy)
            return scatters.values()

        ani = FuncAnimation(fig, update, frames=len(sequence), interval=interval, blit=True)
        plt.close(fig)
        return HTML(ani.to_jshtml())