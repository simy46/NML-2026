import os

import torch
from torch import Tensor

from consts.consts import ENTITY_MAPPING, MODEL_PATH
from nba.data import NBADataModule
from nba.models import build_model
from nba.wandb_logger import WandBLogger


class MultiStepMSE:
    def __init__(self):
        self.loss_fn = torch.nn.MSELoss()

    def compute(self, pred, target) -> Tensor:
        """Compute the average MSE over the horizon."""
        B, T, N, F = target.shape

        target = target[:, :, :, :2].permute(1, 0, 2, 3).reshape(T, B * N, 2)

        if pred.size(1) != target.size(1):
            raise ValueError(
                f"Dimension mismatch: pred {pred.size(1)} vs target {target.size(1)}"
            )

        loss = 0

        for t in range(T):
            loss += self.loss_fn(pred[t, :, :], target[t, :, :])

        return loss / T


class NBATrainer:
    ENTITY_MAPPING = ENTITY_MAPPING

    def __init__(self, cfg: dict):
        self.cfg = cfg

        self.batch_size = cfg["training"]["batch_size"]
        self.context_size = cfg["data"]["context_size"]
        self.horizon_size = cfg["data"]["horizon_size"]

        requested_device = cfg["training"].get("device", "cuda")
        self.device = torch.device(
            requested_device
            if torch.cuda.is_available() or requested_device == "cpu"
            else "cpu"
        )

        self.epochs = cfg["training"]["epochs"]
        self.seed = cfg["training"].get("seed", 0)
        self.model_path = MODEL_PATH

        self.data = NBADataModule(cfg)
        self.data.setup()

        self.mu = self.data.mu.to(self.device)
        self.sigma = self.data.sigma.to(self.device)

        self.model = build_model(cfg).to(self.device)

        self.loss_fn = MultiStepMSE()

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=cfg["training"].get("lr", 1e-3),
            weight_decay=cfg["training"].get("weight_decay", 5e-4),
        )

        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=cfg["training"].get("scheduler_step_size", 300),
            gamma=cfg["training"].get("scheduler_gamma", 0.5),
        )

        self.logger = WandBLogger(cfg)

    def train(self):
        run = self.logger.start()

        try:
            dataloader_dict = {
                "train": self.data.train_dataloader,
                "val": self.data.val_dataloader,
            }

            for epoch in range(self.epochs):
                print(f"Epoch {epoch + 1}/{self.epochs}")

                for split, dataloader in dataloader_dict.items():
                    if split == "val":
                        self.model.eval()

                        with torch.no_grad():
                            self.train_one_epoch(dataloader, split)
                    else:
                        self.model.train()
                        self.train_one_epoch(dataloader, split)

                    sampler = dataloader.sampler
                    sampler.set_epoch(epoch)
                    self.scheduler.step()

            torch.save(self.model.state_dict(), os.path.join(self.model_path))
            print(f"Training Complete. Model saved as '{self.model_path}'.")

        finally:
            if run is not None:
                run.finish()

    def train_one_epoch(self, dataloader, split):
        avg_loss = 0
        batches_processed = 0

        for batch in dataloader:
            X, y = batch
            X, y = X.to(self.device), y.to(self.device)

            pred = self.model(X)
            loss = self.loss_fn.compute(pred, y)

            if split == "train":
                self.optimizer.zero_grad()
                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.cfg["training"].get("grad_clip", 10.0),
                )

                self.optimizer.step()

            avg_loss += loss.item()
            batches_processed += 1

        if batches_processed > 0:
            avg_loss = avg_loss / batches_processed
            self.logger.log({f"{split}_loss": avg_loss})
            print(f"  [{split.upper()}] Loss: {avg_loss:.4f}")
        else:
            print(f"  Warning: No batches processed for {split} split.")

    def evaluate(self):
        self.model.eval()

        with torch.no_grad():
            self.train_one_epoch(self.data.val_dataloader, "val")

    def load_model(self):
        self.model.load_state_dict(
            torch.load(
                self.model_path,
                weights_only=True,
                map_location=self.device,
            )
        )
        self.model.eval()

    def get_trajectory(self, X: Tensor) -> Tensor:
        model = build_model(self.cfg).to(self.device)

        model.load_state_dict(
            torch.load(
                self.model_path,
                weights_only=True,
                map_location=self.device,
            )
        )

        model.eval()

        with torch.no_grad():
            pred = model(X.unsqueeze(0).to(self.device)).cpu()

        # Denormalize
        X = X.cpu()
        mu = self.mu.cpu()
        sigma = self.sigma.cpu()

        X[:, :, :2] = X[:, :, :2] * sigma + mu
        pred = pred * sigma + mu

        # Format
        static = X[-1, :, 2:].unsqueeze(0).repeat(pred.size(0), 1, 1)
        pred = torch.cat([pred, static], dim=-1)
        pred = torch.cat([X, pred], dim=0).detach()

        return pred