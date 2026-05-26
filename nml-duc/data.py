"""Dataset and sampler utilities for NBA trajectory prediction."""

import os
from typing import List, Optional, Sequence, Tuple

import torch
from torch import Tensor
from torch.utils.data import Dataset, Sampler


def list_pt_files(data_path: str) -> List[str]:
    """Sorted list of absolute `.pt` paths in `data_path` (sorted for determinism)."""
    return sorted(os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith('.pt'))


def split_train_val(data_path: str, val_split: float, seed: int = 0
                    ) -> Tuple[List[str], List[str]]:
    """Deterministically partition `.pt` files in `data_path` into (train, val) lists."""
    files = list_pt_files(data_path)
    n = len(files)
    n_val = int(round(n * val_split))
    if n_val <= 0 or n_val >= n:
        return files, []
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed)).tolist()
    val_idx = set(perm[:n_val])
    train_files = [f for i, f in enumerate(files) if i not in val_idx]
    val_files = [f for i, f in enumerate(files) if i in val_idx]
    return train_files, val_files


class NBADataset(Dataset):
    """Loads raw `.pt` sequences of shape [T, N, F]; positions are kept in court coordinates.

    Each item is a (context, horizon) window addressed by `(seq_idx, start)`.
    Either `data_path` (every `.pt` in a directory) or `files` (an explicit list) must be given.

    When `augment=True`, every fetched window is independently transformed by a
    composition of symmetry-preserving operations (see `_augment`). The transforms
    are valid because the court is symmetric about both axes, the heterogeneous
    edge construction depends only on team-equality (not on which team is +1 vs
    -1), and the five players on each team are interchangeable graph nodes.
    """

    def __init__(self, data_path: Optional[str], context_size: int, horizon_size: int,
                 files: Optional[Sequence[str]] = None, augment: bool = False):
        super().__init__()
        self.context_size = context_size
        self.horizon_size = horizon_size
        self.window_size = context_size + horizon_size
        self.augment = augment
        if files is None:
            if data_path is None:
                raise ValueError('NBADataset: provide data_path or files')
            files = list_pt_files(data_path)
        self.sequences: List[Tensor] = []
        self.max_start: List[int] = []
        self._load_data(files)

    def _load_data(self, files: Sequence[str]):
        for f in files:
            seq = torch.load(f, weights_only=False)  # [T, N, F]
            if seq.shape[0] < self.window_size:
                continue
            self.sequences.append(seq)
            self.max_start.append(max(0, seq.shape[0] - self.window_size))

    def __getitem__(self, index: Tuple[int, int]) -> Tuple[Tensor, Tensor]:
        seq_idx, start = index
        seq = self.sequences[seq_idx]
        X = seq[start:start + self.context_size]
        y = seq[start + self.context_size:start + self.window_size]
        if self.augment:
            X, y = self._augment(X, y)
        return X, y

    def _augment(self, X: Tensor, y: Tensor) -> Tuple[Tensor, Tensor]:
        """Symmetry-preserving augmentations on a single (context, horizon) window.

        Applies x-flip, y-flip, team-label swap, and within-team player
        permutation. Velocity / acceleration / edge geometry / possession are
        all recomputed from positions inside the model, so this is the only
        place that needs to know about the transforms.
        """
        full = torch.cat([X, y], dim=0).clone()  # [T, N, F]; clone so we don't touch the cached tensor

        if torch.rand(()) < 0.5:
            full[..., 0] = -full[..., 0]
        if torch.rand(()) < 0.5:
            full[..., 1] = -full[..., 1]
        if torch.rand(()) < 0.5:
            full[..., 3] = -full[..., 3]  # team swap (-1 <-> +1; ball is 0)

        team = full[0, :, 3]  # [N], post-swap labels
        perm = torch.arange(full.shape[1])
        for label in (-1.0, 1.0):
            idx = (team == label).nonzero(as_tuple=True)[0]
            if idx.numel() > 1:
                perm[idx] = idx[torch.randperm(idx.numel())]
        full = full[:, perm]

        C = self.context_size
        return full[:C], full[C:]

    def __len__(self) -> int:
        return len(self.sequences)


class NBASampler(Sampler):
    """Yields `(seq_idx, start)` index pairs.

    Each sequence contributes `windows_per_sequence` independent random start
    offsets per epoch, so longer training schedules see far more of the available
    `(seq, start)` window space than the original one-per-sequence sampler.
    """

    def __init__(self, batch_size: int, max_start: List[int], seed: int = 0,
                 shuffle: bool = True, windows_per_sequence: int = 1):
        self.batch_size = batch_size
        self.max_start = max_start
        self.seed = seed
        self.shuffle = shuffle
        self.windows_per_sequence = max(1, int(windows_per_sequence))
        self.epoch = 0
        self.generator = torch.Generator().manual_seed(seed)

    def set_epoch(self, epoch: int):
        self.epoch = epoch
        self.generator.manual_seed(self.seed + epoch)

    def __iter__(self):
        n_seq = len(self.max_start)
        order = list(range(n_seq)) * self.windows_per_sequence
        if self.shuffle:
            perm = torch.randperm(len(order), generator=self.generator).tolist()
            order = [order[i] for i in perm]
        for i in order:
            start = int(torch.randint(0, self.max_start[i] + 1, size=(), generator=self.generator))
            yield i, start

    def __len__(self) -> int:
        return len(self.max_start) * self.windows_per_sequence