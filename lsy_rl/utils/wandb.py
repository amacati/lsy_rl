from typing import Any
from pathlib import Path

import yaml
from types import SimpleNamespace


def load_config(path: Path):
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return WandBConfig(config)


class WandBConfig(SimpleNamespace):

    def __init__(self, config: dict[str, Any]):
        self.__dict__.update(self.parse(config).__dict__)
        self.__original_dict = config

    @staticmethod
    def parse(data: dict) -> SimpleNamespace | dict:
        if not data:
            return dict()
        x = SimpleNamespace()
        for k, v in data.items():
            if v is None:
                continue
            if isinstance(v["value"], dict):
                if all(isinstance(v, dict) for v in v["value"].values()):
                    if all(k in ("value", "desc") for keys in v["value"].values() for k in keys):
                        setattr(x, k, WandBConfig.parse(v["value"]))
                else:
                    setattr(x, k, v["value"])
            else:
                setattr(x, k, v["value"])
        return x

    def asdict(self):  # TODO: Fix changes not being reflected
        return self.__original_dict.copy()
