import yaml

class Config:
    def __init__(self, path: str):
        self.path = path

    def load(self) -> dict:
        with open(self.path, "r") as f:
            return yaml.safe_load(f)