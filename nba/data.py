import os
import random

import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, Sampler

from consts.consts import TRAIN_DIR, VAL_DIR


class NBADataset(Dataset):
    """Dataset to load NBA highlights tensor data."""

    def __init__(
        self,
        data_path: str,
        context_size: int,
        horizon_size: int,
        mu: Tensor,
        sigma: Tensor,
    ):
        super().__init__()
        self.context_size = context_size
        self.horizon_size = horizon_size
        self.window_size = context_size + horizon_size
        self.load_data(data_path, mu, sigma)

    def load_data(self, data_path: str, mu: Tensor, sigma: Tensor):
        """Load all sequences and normalize positions."""
        self.sequences = []
        self.max_start = []

        for f in [
            os.path.join(data_path, f)
            for f in os.listdir(data_path)
            if f.endswith(".pt")
        ]:
            seq = torch.load(f, weights_only=False)  # [T, N, F]
            seq[:, :, [0, 1]] = (seq[:, :, [0, 1]].clone() - mu) / sigma

            if seq.shape[0] >= self.window_size:
                self.sequences.append(seq)
                self.max_start.append(max(0, len(seq) - self.window_size))

    def __getitem__(self, index) -> tuple:
        """Retrieve sequence given an index in format: (seq_idx:int, start_point:int)."""
        seq_idx, start = index
        T = self.window_size

        X = self.sequences[seq_idx][start : start + self.context_size]
        y = self.sequences[seq_idx][start + self.context_size : start + T]

        return X, y

    def __len__(self):
        return len(self.sequences)


class NBASampler(Sampler):
    def __init__(self, batch_size: int, max_start: list, seed=0, shuffle=True):
        self.batch_size = batch_size
        self.max_start = max_start
        self.epoch = 0
        self.generator = torch.Generator().manual_seed(seed)
        self.seed = seed
        self.shuffle = shuffle

    def set_epoch(self, epoch):
        self.epoch = epoch
        self.generator.manual_seed(self.seed + epoch)

    def __iter__(self):
        n = len(self)

        if self.shuffle:
            perm = torch.randperm(n, generator=self.generator).tolist()
        else:
            perm = list(range(n))

        perm_start = [(i, self.max_start[i]) for i in perm]

        for k in range(0, n, self.batch_size):
            batch = perm_start[k : k + self.batch_size]

            for idx, max_start in batch:
                start = torch.randint(
                    0,
                    max_start + 1,
                    size=(),
                    generator=self.generator,
                )
                yield idx, start

    def __len__(self):
        return len(self.max_start)


class NBADataModule:
    def __init__(self, cfg: dict):
        self.cfg = cfg

        self.train_path = TRAIN_DIR
        self.val_path = VAL_DIR

        self.batch_size = cfg["training"]["batch_size"]
        self.context_size = cfg["data"]["context_size"]
        self.horizon_size = cfg["data"]["horizon_size"]
        self.seed = cfg["training"].get("seed", 0)

        self.mu = None
        self.sigma = None

        self.train_dataset = None
        self.val_dataset = None

        self.train_dataloader = None
        self.val_dataloader = None

    def setup(self):
        self.mu, self.sigma = self.compute_normalization_statistics(self.train_path)

        self.train_dataset = NBADataset(
            self.train_path,
            self.context_size,
            self.horizon_size,
            self.mu,
            self.sigma,
        )

        self.val_dataset = NBADataset(
            self.val_path,
            self.context_size,
            self.horizon_size,
            self.mu,
            self.sigma,
        )

        generator = self.set_seed()

        train_sampler = NBASampler(
            batch_size=self.batch_size,
            max_start=self.train_dataset.max_start,
            seed=self.seed,
            shuffle=True,
        )

        val_sampler = NBASampler(
            batch_size=self.batch_size,
            max_start=self.val_dataset.max_start,
            seed=self.seed,
            shuffle=False,
        )

        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=train_sampler,
            generator=generator,
        )

        self.val_dataloader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            sampler=val_sampler,
            generator=generator,
        )

    def compute_normalization_statistics(self, data_path: str) -> tuple:
        all_pos = []

        for f in [
            os.path.join(data_path, f)
            for f in os.listdir(data_path)
            if f.endswith(".pt")
        ]:
            seq = torch.load(f, weights_only=False)
            pos = seq[:, :, [0, 1]]
            all_pos.append(pos)

        all_pos = torch.cat(all_pos, dim=0)
        mu = all_pos.mean(dim=(0, 1))
        sigma = all_pos.std(dim=(0, 1))

        return mu, sigma

    def set_seed(self) -> torch.Generator:
        torch.manual_seed(self.seed)
        random.seed(self.seed)

        generator = torch.Generator("cpu").manual_seed(self.seed)

        return generator

    def set_epoch(self, epoch: int):
        self.train_dataloader.sampler.set_epoch(epoch)
        self.val_dataloader.sampler.set_epoch(epoch)