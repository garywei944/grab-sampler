from abc import ABC, abstractmethod
from enum import Enum

import torch

from sklearn.random_projection import johnson_lindenstrauss_min_dim

from absl import logging


def get_rng(
    seed_rng: int | torch.Generator | None,
    device: (str | torch.device) = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    ),
) -> torch.Generator:
    if isinstance(seed_rng, int):
        rng = torch.Generator(device=device)
        rng.manual_seed(seed_rng)
    elif isinstance(seed_rng, torch.Generator):
        rng = seed_rng
    elif seed_rng is None:
        rng = torch.Generator(device=device)
    else:
        raise ValueError(f"seed_rng must be int or torch.Generator, got {seed_rng}")

    return rng


class RandomProjectionType(Enum):
    NONE = "none"
    JL_GAUSSIAN = "jl"
    JL_VERY_SPARSE = "jl_sparse"
    KRONECKER = "kron"


class RandomProjection(ABC):
    def __init__(
        self,
        n: int,
        d: int,
        dd: int = 0,
        eps: float = 0.1,
        seed_rng: int | torch.Generator | None = None,
        dtype: torch.dtype = torch.float32,
        device: (str | torch.device) = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
        *args,
        **kwargs,
    ):
        self.n = n
        self.d = d
        self.dtype = dtype
        self.device = device

        if dd > 0:
            self.dd = dd
        else:
            self.dd = johnson_lindenstrauss_min_dim(n, eps=eps)
            logging.info(f"JL Random Projection minimum dimension: {self.dd}")
        self.shape = torch.tensor([self.dd, self.d], dtype=torch.int64)

        self.rng = get_rng(seed_rng, device=device)

    @abstractmethod
    def project(self, x: torch.Tensor) -> torch.Tensor:
        ...
