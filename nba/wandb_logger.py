import wandb

class WandBLogger:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.enabled = cfg.get("wandb", {}).get("enabled", True)

    def start(self):
        if not self.enabled:
            return None

        mode = self.cfg.get("wandb", {}).get("mode", "offline")

        return wandb.init(
            mode=mode,
            config=self.cfg,
            settings=wandb.Settings(
                _disable_stats=True,
                _disable_meta=True,
            ),
        )

    def log(self, metrics: dict):
        if self.enabled:
            wandb.log(metrics)