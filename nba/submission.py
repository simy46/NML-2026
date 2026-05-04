import os

import pandas as pd
import torch

from consts.consts import TEST_DIR, SUBMISSION_DIR


class NBASubmission:
    def __init__(self, cfg: dict, trainer):
        self.cfg = cfg
        self.trainer = trainer

        self.test_dir = TEST_DIR
        self.target_dir = SUBMISSION_DIR

        self.context_size = cfg["data"]["context_size"]
        self.horizon_size = cfg["data"]["horizon_size"]
        self.num_entities = cfg["data"]["num_entities"]

    def create(self) -> pd.DataFrame:
        all_traj = []

        for f, f_path in [
            (f, os.path.join(self.test_dir, f))
            for f in os.listdir(self.test_dir)
            if f.endswith(".pt")
        ]:
            seq = torch.load(f_path, weights_only=False)

            seq[:, :, [0, 1]] = (
                seq[:, :, [0, 1]].clone() - self.trainer.mu.cpu()
            ) / self.trainer.sigma.cpu()

            traj = self.trainer.get_trajectory(seq)
            traj = traj[self.context_size :, :, :2]
            traj = traj.reshape(-1)

            all_traj.append([int(f.removesuffix(".pt"))] + traj.tolist())

        columns = ["id"] + [
            f"entity_{i}_time_{t}_{axis}"
            for t in range(self.horizon_size)
            for i in range(self.num_entities)
            for axis in ["x", "y"]
        ]

        all_traj = pd.DataFrame(all_traj, columns=columns).set_index("id")
        all_traj = all_traj.sort_index()

        os.makedirs(self.target_dir, exist_ok=True)

        output_path = os.path.join(self.target_dir, "solution.csv")
        all_traj.to_csv(output_path)

        print(f"Submission saved to {output_path}")

        return all_traj