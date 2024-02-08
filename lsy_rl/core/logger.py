from abc import ABC, abstractmethod
from pathlib import Path
import json
from typing import Mapping
import numpy as np
import torch
import copy

from lsy_rl.utils.wandb import WandBConfig, load_config


class Logger(ABC):

    def __init__(self):
        super().__init__()

    @abstractmethod
    def log(self, data, step: int, flush: bool = False):
        ...

    @abstractmethod
    def flush(self):
        ...

    def stop(self):
        ...


class EmptyLogger(Logger):

    def __init__(self):
        super().__init__()

    def log(self, data, step: int, flush: bool = False):
        ...

    def flush(self):
        ...

    def stop(self):
        ...


class LoggerList(Logger):

    def __init__(self, loggers: list):
        super().__init__()
        assert all([isinstance(logger, Logger) for logger in loggers])
        self.loggers = loggers

    def log(self, data: dict, step: int, flush: bool = False):
        for logger in self.loggers:
            logger.log(data, step, flush)

    def flush(self):
        for logger in self.loggers:
            logger.flush()

    def stop(self):
        for logger in self.loggers:
            logger.stop()

    def append(self, logger: Logger):
        assert isinstance(logger, Logger)
        self.loggers.append(logger)


class ConsoleLogger(Logger):

    def __init__(self, filter: str | None = None):
        super().__init__()
        self._log = dict()
        self._current_step = None
        self.filter = filter

    def log(self, data: dict, step: int, flush: bool = False):
        self._log[step] = self._log.get(step, dict()) | data
        if flush or self._current_step is not None and step > self._current_step:
            self.flush()
        self._current_step = step

    def flush(self):
        log = self._log[self._current_step]
        if self.filter is not None:
            log = {k: v for k, v in self._log[self._current_step].items() if self.filter in k}
        if log:
            print(f"\nStep {self._current_step}:")
            for key, value in log.items():
                match value:
                    case float():
                        print(f"\t{key}: {value:.2f}")
                    case _:
                        print(f"\t{key}: {value}")

    def stop(self):
        ...


class FileLogger(Logger):

    def __init__(self, path: Path):
        super().__init__()
        if not path.parent.exists():
            path.parent.mkdir(parents=True)
        self.path = path
        self._log = dict()

    def log(self, data: dict, step: int, flush: bool = False):
        self._log[step] = self._log.get(step, dict()) | data
        if flush:
            self.flush()

    def flush(self):
        log = self.jsonify(copy.deepcopy(self._log))
        with open(self.path, 'w') as f:
            json.dump(log, f)

    def stop(self):
        self.flush()

    def jsonify(self, data: dict):
        for key, value in data.items():
            if isinstance(value, Mapping):
                data[key] = self.jsonify(value)
            elif isinstance(value, (np.ndarray, torch.Tensor)):
                data[key] = value.tolist()


class WandBLogger(Logger):

    def __init__(self,
                 wandb_api_key: str,
                 save_path: Path,
                 config: WandBConfig | None = None,
                 config_path: Path | None = None):
        assert config is not None or config_path is not None, "Must provide config or config_path."
        super().__init__()
        import wandb  # Import on class init to avoid unnecessary non-optional dependencies

        # Load config from file if not provided directly
        config = config or load_config(config_path)
        self._check_wandb_config(config)
        wandb.login(key=wandb_api_key)
        self.run = wandb.init(project=config.wandb.project,
                              entity=config.wandb.entity,
                              group=config.wandb.group,
                              config=config.asdict(),
                              dir=save_path)

    def log(self, data, step: int, flush: bool = False):
        self.run.log(data, step=step, commit=flush)

    def flush(self):
        self.run.log({}, commit=True)

    def stop(self):
        self.run.finish()

    def _check_wandb_config(self, config: WandBConfig):
        if not hasattr(config, "wandb"):
            raise AttributeError("WandB config missing 'wandb' namespace.")
        for attr in ["project", "entity", "group"]:
            if not hasattr(config.wandb, attr):
                raise AttributeError(f"WandB config missing attribute '{attr}' in wandb namespace.")
