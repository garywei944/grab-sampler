from typing import Sequence
from abc import ABC, abstractmethod
from overrides import overrides

import torch


class EstimatorBase(ABC):
    value: torch.Tensor

    @abstractmethod
    def update(self, value, *args, **kwargs):
        ...

    def __call__(self, *args, **kwargs):
        try:
            return self.update(*args, **kwargs)
        except TypeError:
            return self.value


class StaleMeanEstimator(EstimatorBase):
    def __init__(
        self,
        shape: (int | Sequence[int] | torch.Size),
        dtype=torch.float32,
        device: (str | torch.device) = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
        *args,
        **kwargs,
    ):
        self.value = torch.zeros(shape, dtype=dtype, device=device)
        self.run_sum = torch.zeros_like(self.value)
        self.n = 0

    @torch.no_grad()
    def update(self, value) -> torch.Tensor:
        self.run_sum.add_(value)
        self.n += 1

        return self.value

    def reset(self):
        # Don't reset if we haven't seen any data
        if self.n == 0:
            return
        self.value = self.run_sum / self.n
        self.run_sum.zero_()
        self.n = 0


class EMAEstimator(EstimatorBase):
    def __init__(
        self,
        shape: (int | Sequence[int] | torch.Size),
        beta: float = 0.9,  # 1 - decay
        dtype=torch.float32,
        device: (str | torch.device) = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
    ):
        self._value = torch.zeros(shape, dtype=dtype, device=device)
        self.beta = beta
        self.t = 0

    @torch.no_grad()
    def update(self, value, test_and_set: bool = False) -> torch.Tensor:
        r = self.value if test_and_set else None

        self._value.mul_(self.beta).add_(value, alpha=1 - self.beta)
        self.t += 1

        return r if test_and_set else self.value

    @property
    def value(self) -> torch.Tensor:
        return (
            self._value / (1 - self.beta**self.t)
            if self.t != 0
            else torch.zeros_like(self._value)
        )
