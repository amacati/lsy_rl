import copy
import json
from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path
from types import NoneType
from typing import Mapping

import numpy as np
import torch
from array_api_compat import array_namespace


class Logger(ABC):
    def __init__(self, filter: str | list[str] | None = None, rate_limit: int | None = None):
        # Enable filtering out based on topic or list of topics
        assert isinstance(filter, (str | list | NoneType))
        if isinstance(filter, str):
            filter = [filter]
        self._filters = filter
        # Enable rate limiting logging
        self._rate_limit = rate_limit
        self._last_log = defaultdict(lambda: -float("inf"))

    @abstractmethod
    def log(self, data: dict, step: int, flush: bool = False): ...

    @abstractmethod
    def flush(self): ...

    def stop(self): ...

    def filter(self, data: dict) -> dict:
        if self._filters is None:
            return data
        return {k: v for k, v in data.items() if any(f in k for f in self._filters)}

    def rate_limit(self, data: dict, step: int) -> dict:
        if self._rate_limit is None:
            return data
        data = {k: v for k, v in data.items() if step - self._last_log[k] >= self._rate_limit}
        for k in data:
            self._last_log[k] = step
        return data


class EmptyLogger(Logger):
    def __init__(self):
        super().__init__()

    def log(self, data: dict, step: int, flush: bool = False): ...

    def flush(self): ...


class LoggerList(Logger):
    def __init__(self, loggers: list[Logger]):
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
    def __init__(self, filter: str | list[str] | None = None, rate_limit: float | None = None):
        super().__init__(filter=filter, rate_limit=rate_limit)
        self._log = dict()
        self._current_step = None

    def log(self, data: dict, step: int, flush: bool = False):
        data = self.filter(data)
        data = self.rate_limit(data, step)
        self._log[step] = self._log.get(step, dict()) | data
        if flush or self._current_step is not None and step > self._current_step:
            self.flush()
        self._current_step = step

    def flush(self):
        log = self._log[self._current_step]
        if not log:
            return
        print(f"\nStep {self._current_step}:")
        for key, value in log.items():
            match value:
                case float():
                    print(f"\t{key}: {value:.2f}")
                case _:
                    print(f"\t{key}: {value}")


class MemLogger(Logger):
    def __init__(self, filter: str | None = None):
        super().__init__()
        self._log = dict()
        self._filter = filter

    @property
    def data(self):
        return self._log

    def log(self, data: dict, step: int, flush: bool = False):
        data = {k: v for k, v in data.items() if self._filter is None or self._filter in k}
        if not data:
            return
        self._log[step] = self._log.get(step, dict()) | data

    def flush(self):
        pass

    def stop(self):
        pass


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
        with open(self.path, "w") as f:
            json.dump(log, f)

    def stop(self):
        self.flush()

    def jsonify(self, data: dict):
        for key, value in data.items():
            if isinstance(value, Mapping):
                data[key] = self.jsonify(value)
            elif isinstance(value, (np.ndarray, torch.Tensor)):
                data[key] = value.tolist()
        return data


class WandBLogger(Logger):
    def __init__(self, filter: str | list[str] | None = None, rate_limit: float | None = None):
        super().__init__(filter=filter, rate_limit=rate_limit)
        import wandb  # Import on class init to avoid unnecessary non-optional dependencies

        assert wandb.run is not None, "WandB must be initialized before creating a WandBLogger"
        self.run = wandb.run

    def log(self, data, step: int, flush: bool = False):
        data = self.filter(data)
        data = self.rate_limit(data, step)
        self.run.log(data, step=step, commit=flush)

    def flush(self):
        self.run.log({}, commit=True)


class Collector:
    def collect(self, obs, actions, next_obs, rewards, terminated, truncated, info, autoreset): ...

    def log(self, mask):
        return {}

    def clear(self, mask): ...


class LogCollector(Collector):
    """Collect and aggregate statistics for logging."""

    valid_targets = ["rewards", "steps"]

    def __init__(self, target: str, log_key: str):
        self._target = target
        self._log_key = log_key
        self._log = {}
        self._xp = None

    def collect(self, obs, actions, next_obs, rewards, terminated, truncated, info, autoreset):
        if self._xp is None:
            self._xp = array_namespace(rewards)
        if self._target == "reward":
            self._collect_reward(rewards)
        elif self._target == "step":
            if self._log_key not in self._log:
                self._log[self._log_key] = self._xp.zeros_like(rewards)
            self._log[self._log_key] += 1
        else:
            raise ValueError(f"Invalid target {self._target}")

    def log(self, mask):
        if self._log is None:
            return {}
        return {k: float(self._xp.mean(v[mask])) for k, v in self._log.items()}

    def clear(self, mask):
        for k in self._log:
            self._log[k][mask] = 0

    def _collect_reward(self, rewards):
        if self._log_key not in self._log:
            self._log[self._log_key] = rewards
        else:
            self._log[self._log_key] += rewards


class LogCollectorList(Collector):
    """A list of LogCollectors that can be called together."""

    def __init__(self, collectors: list[LogCollector] | None = None):
        self._collectors = [] if collectors is None else collectors
        assert all(isinstance(c, Collector) for c in self._collectors)

    def append(self, collector: Collector):
        assert isinstance(collector, Collector)
        self._collectors.append(collector)

    def collect(self, obs, actions, next_obs, rewards, terminated, truncated, info, autoreset):
        for collector in self._collectors:
            collector.collect(
                obs, actions, next_obs, rewards, terminated, truncated, info, autoreset
            )

    def log(self, mask):
        logs = {}
        for collector in self._collectors:
            logs.update(collector.log(mask))
        return logs

    def clear(self, mask):
        for collector in self._collectors:
            collector.clear(mask)
