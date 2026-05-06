import json
import csv
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

class LocalLogger:
    def __init__(self, model_name:str, log_dir="logs"):
        self.log_dir = os.path.join(log_dir, model_name)
        os.makedirs(self.log_dir, exist_ok=True)
        # Structure: { fold_idx: { "epochs": [ {metrics} ], "test_loss": float } }
        self.data = {}

    def log_epoch(self, fold: int, epoch: int, metrics: dict):
        """Logs training and validation metrics for a specific epoch."""
        if fold not in self.data:
            self.data[fold] = {"epochs": [], "test_loss": None}
        
        metrics["epoch"] = epoch
        self.data[fold]["epochs"].append(metrics)

    def log_test(self, fold: int, test_loss: float):
        if fold not in self.data:
            self.data[fold] = {"epochs": [], "test_loss": None}
        self.data[fold]["test_loss"] = test_loss

    def save(self, filename="kfold_results.json"):
        path = os.path.join(self.log_dir, filename)
        with open(path, "w") as f:
            json.dump(self.data, f, indent=4)
        print(f"Logs saved to {path}")

        plot_path = os.path.join(self.log_dir,"kfold_plot.png")
        self.plot_kfold_with_variance(json_path=path, save_path=plot_path)
        print(f"Training stats plot saved to {plot_path}")

    def plot(self, metric_name="val_loss", save_path="kfold_plot.png"):
        plt.figure(figsize=(10, 6))
        
        for fold, content in self.data.items():
            epochs = [e["epoch"] for e in content["epochs"]]
            values = [e[metric_name] for e in content["epochs"]]
            plt.plot(epochs, values, label=f"Fold {fold}")

        plt.title(f"{metric_name} across K-Folds")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.log_dir, save_path))
        plt.show

    @staticmethod
    def plot_kfold_with_variance(json_path: str, save_path="kfold_variance_plot.png"):
        with open(json_path, "r") as f:
            data = json.load(f)

        rows = []
        for fold, content in data.items():
            for epoch_data in content["epochs"]:
                rows.append({
                    "Fold": fold,
                    "Epoch": epoch_data["epoch"],
                    "Loss": epoch_data["train_loss"],
                    "Split": "train"
                })
                rows.append({
                    "Fold": fold,
                    "Epoch": epoch_data["epoch"],
                    "Loss": epoch_data["val_loss"],
                    "Split": "val"
                })
        
        df = pd.DataFrame(rows)

        sns.set_theme(style="whitegrid")
        plt.figure(figsize=(12, 6))

        plot = sns.lineplot(
            data=df, 
            x="Epoch", 
            y="Loss", 
            hue="Split", 
            style="Split", 
            markers=True, 
            dashes=False,
            errorbar="sd" 
        )

        plot.set_yscale("log")
        plt.title(f"Training and Validation Loss across {len(data)} Folds")
        plt.grid(True, which="both", linestyle='--', alpha=0.5)
        
        plt.savefig(save_path, dpi=300)
        plt.show()