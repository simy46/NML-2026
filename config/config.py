import yaml
from consts.consts import CONFIG_PATH

class Config:
    def __init__(self, path: str = CONFIG_PATH):
        self.path = path

    def load(self) -> dict:
        with open(self.path, "r") as f:
            return yaml.safe_load(f)