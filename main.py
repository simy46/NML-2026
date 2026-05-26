from config.config import Config
from nba import NBATrainer, NBASubmission

cfg = Config(path="config/heteroSTGCN.yaml").load()
trainer = NBATrainer(cfg)

trainer.train_with_val(100)