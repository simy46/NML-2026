import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import networkx as nx
import torch


class NBAVisualizer:
    def __init__(self, cfg: dict, trainer):
        self.cfg = cfg
        self.trainer = trainer

    def plot_example(self, sequence_idx: int = 0, start: int = 0):
        dataset = self.trainer.data.val_dataset

        X, y = dataset[(sequence_idx, start)]

        full_target = torch.cat([X, y], dim=0)
        full_pred = self.trainer.get_trajectory(X)

        plt.figure(figsize=(8, 6))

        for entity_idx in range(full_target.shape[1]):
            true_xy = full_target[:, entity_idx, :2].cpu()
            pred_xy = full_pred[:, entity_idx, :2].cpu()

            plt.plot(true_xy[:, 0], true_xy[:, 1], linestyle="-", alpha=0.7)
            plt.plot(pred_xy[:, 0], pred_xy[:, 1], linestyle="--", alpha=0.7)

        plt.title("Ground-truth and predicted trajectories")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.grid(True)
        plt.show()

    def animate_example(self, sequence_idx: int = 0, start: int = 0):
        dataset = self.trainer.data.val_dataset

        X, y = dataset[(sequence_idx, start)]
        full_pred = self.trainer.get_trajectory(X)

        fig, ax = plt.subplots(figsize=(8, 6))
        scat = ax.scatter([], [])

        ax.set_title("Predicted trajectory animation")
        ax.set_xlabel("x")
        ax.set_ylabel("y")

        coords = full_pred[:, :, :2].cpu()

        x_min, x_max = coords[:, :, 0].min(), coords[:, :, 0].max()
        y_min, y_max = coords[:, :, 1].min(), coords[:, :, 1].max()

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

        def update(frame):
            scat.set_offsets(coords[frame])
            return (scat,)

        anim = FuncAnimation(
            fig,
            update,
            frames=coords.shape[0],
            interval=250,
            blit=True,
        )

        plt.show()

        return anim