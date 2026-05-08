import os
import warnings

import torch
from torch import Tensor

from consts.consts import ENTITY_MAPPING
from nba.data import NBADataModule, FoldSampler
from nba.models import build_model
from nba.wandb_logger import WandBLogger
from nba.localLogger import LocalLogger
from sklearn.model_selection import KFold, train_test_split
from torch.utils.data import DataLoader, Subset
import wandb
import copy
import numpy as np


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

        if requested_device == "cuda" and torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif requested_device == "mps" and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
            warnings.warn('Training on cpu, this might take ages.')
        print(f"Training on {self.device}")

        # Important: keep cfg consistent with actual selected device
        self.cfg["training"]["device"] = str(self.device)

        self.epochs = cfg["training"]["epochs"]
        self.seed = cfg["training"].get("seed", 0)
        self.model_path = f'saved_models/{cfg["MODEL_NAME"]}/checkpoint.pth'

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
        self.local_logger = LocalLogger(log_dir="saved_models", model_name=cfg["MODEL_NAME"])


    def train(self):
        """Train model on all the data"""
        run = self.logger.start()

        try:
            dataloader_dict = {
                "train": self.data.train_dataloader,
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

    def train_one_epoch(self, dataloader, split, silent=False):
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
            if not silent:
                self.logger.log({f"{split}_loss": avg_loss})
                print(f"  [{split.upper()}] Loss: {avg_loss:.4f}")
            return avg_loss
        return 0.0

    def train_k_fold(self, k: int = 2):
        num_sequences = len(self.data.train_dataset)
        indices = list(range(num_sequences))
        
        kf = KFold(n_splits=k, shuffle=True, random_state=self.seed)
        global_best_loss = float('inf')
        
        patience = self.cfg["training"].get("early_stopping_patience", 5)
        min_delta = self.cfg["training"].get("early_stopping_delta", 1e-4)

        test_losses = []

        for fold, (train_idx, temp_val_idx) in enumerate(kf.split(indices)):
            val_idx, test_idx = train_test_split(
                temp_val_idx, test_size=0.5, random_state=self.seed
            )

            self.model = build_model(self.cfg).to(self.device)
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), 
                lr=self.cfg["training"].get("lr", 1e-3)
            )
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=300, gamma=0.5)
            early_stopping = EarlyStopping(patience=patience, min_delta=min_delta)

            train_sampler = FoldSampler(train_idx.tolist(), self.batch_size, self.data.train_dataset.max_start, seed=self.seed)
            val_sampler = FoldSampler(val_idx.tolist(), self.batch_size, self.data.train_dataset.max_start, seed=self.seed, shuffle=False)
            test_sampler = FoldSampler(test_idx.tolist(), self.batch_size, self.data.train_dataset.max_start, seed=self.seed, shuffle=False)

            train_loader = DataLoader(self.data.train_dataset, batch_size=self.batch_size, sampler=train_sampler)
            val_loader = DataLoader(self.data.train_dataset, batch_size=self.batch_size, sampler=val_sampler)
            test_loader = DataLoader(self.data.train_dataset, batch_size=self.batch_size, sampler=test_sampler)

            # with wandb.init(project="nba-kfold", group="kfold-exp", name=f"fold_{fold}") as run:
            for epoch in range(self.epochs):
                train_sampler.set_epoch(epoch)
                
                self.model.train()
                train_loss = self.train_one_epoch(train_loader, "train", silent=True)

                self.model.eval()
                with torch.no_grad():
                    val_loss = self.train_one_epoch(val_loader, "val", silent=True)

                self.scheduler.step()
                # run.log({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

                self.local_logger.log_epoch(fold, epoch, {
                    "train_loss": train_loss,
                    "val_loss": val_loss
                })

                early_stopping(val_loss, self.model)
                if early_stopping.early_stop:
                    print(f"  Fold {fold} stopped early at epoch {epoch}")
                    break

            if early_stopping.best_model_state:
                self.model.load_state_dict(early_stopping.best_model_state)

            self.model.eval()
            with torch.no_grad():
                test_loss = self.train_one_epoch(test_loader, "test", silent=True)

            test_losses.append(test_loss)
            self.local_logger.log_test(fold, test_loss)

            if test_loss < global_best_loss:
                    global_best_loss = test_loss
                    torch.save(self.model.state_dict(), self.model_path)

            # run.log({"final_test_loss": test_loss})
            print(f"Fold {fold} | Test Loss: {test_loss:.4f}")

        self.local_logger.save("kfold_results.json")
        print(f"Training finished\nAvg. test loss over folds: {np.mean(test_losses)}")
        # TODO: still train the model on all the data for the predictions

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
    
class EarlyStopping:
    def __init__(self, patience: int = 5, min_delta: float = 0, verbose: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False
        self.best_model_state = None

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.best_model_state = copy.deepcopy(model.state_dict())
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f"  [EarlyStopping] Counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True