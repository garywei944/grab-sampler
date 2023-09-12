# -*- coding: utf-8 -*-
from collections import defaultdict
from contextlib import contextmanager

import torch
import numpy as np
import pandas as pd


def pretty_time(s: float) -> str:
    """
    Pretty print time in a human-readable fashion, unit: h, min, s, ms, us
    """
    if s > 3600:
        h = int(s) // 3600
        return f"{h}h {(s - h * 3600) / 60:.0f}min"
    elif s > 60:
        m = int(s) // 60
        return f"{m}min {(s - m * 60):.0f}s"
    elif s > 1:
        return f"{s:.2f}s"
    elif s > 1e-3:
        return f"{s * 1e3:.2f}ms"
    else:
        return f"{s * 1e6:.2f}us"


class EventTimer:
    """
    Timer for PyTorch code, measured in milliseconds
    Comes in the form of a contextmanager:

    Example:
    >>> timer = EventTimer()
    ... for i in range(10):
    ...     with timer("expensive operation"):
    ...         x = torch.randn(100)
    ... print(timer.summary())
    """

    def __init__(
        self,
        device: (str | torch.device) = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
        disable: bool = False,
    ):
        self.disable = disable

        # Warm-up GPU
        torch.randn(3, 3, device=device) @ torch.randn(3, 3, device=device)
        self.records = defaultdict(list)

    def reset(self):
        """Reset the timer"""
        # the time for each occurrence of each event
        self.records.clear()

    @contextmanager
    def __call__(self, name):
        if self.disable:
            yield
            return

        # Wait for everything before me to finish
        torch.cuda.synchronize()

        # Measure the time
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        yield
        # Wait for operations that happen during yield to finish
        torch.cuda.synchronize()
        end.record()

        # Need to wait once more for operations to finish
        torch.cuda.synchronize()

        # Update first and last occurrence of this label
        self.records[name].append(start.elapsed_time(end) / 1000)

    def __getitem__(self, item) -> list[float]:
        return self.records[item]

    def save(self, path):
        torch.save(self.records, path)

    @classmethod
    def load(
        cls,
        path,
        device: str
        | torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ) -> "EventTimer":
        timer = cls(device=device)
        timer.records = torch.load(path)
        return timer

    def summary(self) -> pd.DataFrame:
        return pd.DataFrame.from_dict(
            {
                k: {
                    "mean": pretty_time(np.mean(v).item()),
                    "std": pretty_time(np.std(v).item()),
                    "min": pretty_time(np.min(v)),
                    "max": pretty_time(np.max(v)),
                    "count": len(v),
                    "total": pretty_time(np.sum(v).item()),
                }
                for k, v in self.records.items()
            },
            orient="index",
        )
