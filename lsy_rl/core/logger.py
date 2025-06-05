import copy
import json
from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path
from types import NoneType
from typing import Any, Mapping

import numpy as np
import torch
from array_api_compat import array_namespace
from numpy.typing import ArrayLike


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
        if self._current_step is not None and step > self._current_step:
            self.flush()
        # Handle the case that we are at a new step (which does not get flushed by the previous one)
        # and flush was explicitly called
        flush_current = flush and self._current_step != step
        self._current_step = step
        if flush_current:
            self.flush()

    def flush(self):
        if self._current_step is None:
            return
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
    def __init__(self, filter: str | None = None, rate_limit: float | None = None):
        super().__init__(filter=filter, rate_limit=rate_limit)
        self._log = dict()

    @property
    def data(self) -> dict:
        return self._log

    def log(self, data: dict, step: int, flush: bool = False):
        data = self.filter(data)
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

    def jsonify(self, data: dict) -> dict:
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

    def log(self, data: dict, step: int, flush: bool = False):
        data = self.filter(data)
        data = self.rate_limit(data, step)
        self.run.log(data, step=step, commit=flush)

    def flush(self):
        self.run.log({}, commit=True)


class Collector:
    def collect(self, **kwargs: Any): ...

    def log(self, mask: ArrayLike | None = None) -> dict[str, float]:
        return {}

    def clear(self, mask: ArrayLike | None = None): ...


class LogCollector(Collector):
    """Collect and aggregate statistics for logging.

    Args:
        target: The target value to collect from the environment step
        log_key: The key to use when logging the collected values
        reduce: The reduction method to use. One of:
            - "cnt": Count occurrences
            - "sum": Sum values
            - "mean": Average values
    """

    def __init__(self, target: str, log_key: str, reduce: str = "mean"):
        self._target = target
        self._log_key = log_key
        if reduce not in ["cnt", "sum", "mean"]:
            raise ValueError(f"Invalid reduce method {reduce}")
        self._reduce = reduce
        self._cnt = None
        self._log = None
        self._xp = None

    def collect(self, **kwargs: Any) -> None:
        if self._target not in kwargs:
            return
        if self._xp is None:
            self._xp = array_namespace(kwargs[self._target])
        target_val = kwargs[self._target]

        # Initialize or update log based on reduction method
        if self._log is None:
            return self._init_log(target_val)
        if self._reduce == "cnt":
            self._log += 1
        elif self._reduce == "mean":
            self._log += target_val
            self._cnt += 1
        else:  # sum
            self._log += target_val

    def _init_log(self, target_val: ArrayLike):
        if self._reduce == "cnt":
            device = target_val.device
            self._log = self._xp.zeros(len(target_val), device=device)
        elif self._reduce == "mean":
            n = 1 if target_val.ndim == 0 else len(target_val)
            self._cnt = self._xp.zeros(n, device=target_val.device)
            self._log = target_val
        else:  # sum
            self._log = target_val

    def log(self, mask: ArrayLike | None = None) -> dict[str, float]:
        if self._log is None:
            return {}
        mask = mask if mask is not None else ...
        if self._reduce == "cnt":
            return {self._log_key: float(self._xp.mean(self._log[mask]))}
        if self._reduce == "mean":
            return {self._log_key: float(self._xp.mean(self._log[mask] / self._cnt[mask]))}
        else:  # sum
            return {self._log_key: float(self._xp.mean(self._log[mask]))}

    def clear(self, mask: ArrayLike | None = None):
        if self._log is None:
            return
        mask = mask if mask is not None else ...
        if self._reduce == "cnt":
            self._log[mask] = 0
        elif self._reduce == "mean":
            self._log[mask] = 0
            self._cnt[mask] = 0
        else:  # sum
            self._log[mask] = 0


class CollectorList(Collector):
    """A list of LogCollectors that can be called together."""

    def __init__(self, collectors: list[LogCollector] | None = None):
        self._collectors = [] if collectors is None else collectors
        assert all(isinstance(c, Collector) for c in self._collectors)

    def append(self, collector: Collector):
        assert isinstance(collector, Collector)
        self._collectors.append(collector)

    def collect(self, **kwargs: Any):
        for collector in self._collectors:
            collector.collect(**kwargs)

    def log(self, mask: ArrayLike | None = None) -> dict[str, float]:
        logs = {}
        for collector in self._collectors:
            logs.update(collector.log(mask))
        return logs

    def clear(self, mask: ArrayLike | None = None):
        for collector in self._collectors:
            collector.clear(mask)

    def __getitem__(self, idx: int) -> Collector:
        return self._collectors[idx]
