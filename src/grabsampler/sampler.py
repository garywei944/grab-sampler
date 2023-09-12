from enum import Enum
from typing import Sized

import torch
from torch import nn, Tensor
from torch.utils.data import Sampler, IterableDataset
from absl import logging

from .sorter import SorterBase
from .utils import EventTimer


class BalanceType(Enum):
    RANDOM_RESHUFFLING = "rr"
    MEAN_BALANCE = "mean"
    PAIR_BALANCE = "pair"
    BATCH_BALANCE = "batch"
    RECURSIVE_BALANCE = "recursive"
    RECURSIVE_PAIR_BALANCE = "recursive-pair"
    EMA_BALANCE = "ema"
    FIXED_ORDERING = "fixed"
    SHUFFLE_ONCE = "so"
    SAMPLE_W_REPLACEMENT = "swr"
    NTK_BALANCE = "ntk"
    NTK_FIXED_BALANCE = "ntk-fixed"
    NTK_EIGEN = "ntk-eigen"
    NTK_FIXED_EIGEN = "ntk-fixed-eigen"
    ORDER_BY_NORM = "norm"
    ADAPTIVE_MEAN_BALANCE = "adamean"
    ADAPTIVE_EMA_BALANCE = "adaema"
    SIGN_BALANCE = "sign"


class GraBSampler(Sampler[list[int]]):
    sorter: SorterBase

    orders_history: list[list[int]]

    def __init__(
        self,
        data_source: Sized,
        trainable_params: dict[str, Tensor],
        balance_type: str | BalanceType = BalanceType.MEAN_BALANCE,
        device: (str | torch.device) = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
        timer: EventTimer | None = None,
        record_orders: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__(data_source)

        # Disable timer by default in production
        self.timer = timer or EventTimer(device=device, disable=True)
        self.record_orders = record_orders
        if record_orders:
            self.orders_history = []

        self.n = len(data_source)
        self.d = sum(p.numel() for p in trainable_params.values())

        balance_type = BalanceType(balance_type)

        with self.timer("init_sorter"):
            match balance_type:
                # Well studied orderings
                case BalanceType.RANDOM_RESHUFFLING:
                    from .sorter import RandomReshuffling as Sorter
                case BalanceType.SHUFFLE_ONCE:
                    from .sorter import ShuffleOnce as Sorter
                case BalanceType.SAMPLE_W_REPLACEMENT:
                    from .sorter import SampleWReplacement as Sorter
                case BalanceType.FIXED_ORDERING:
                    from .sorter import FixedOrdering as Sorter
                case BalanceType.MEAN_BALANCE:
                    from .sorter import MeanBalance as Sorter
                case BalanceType.PAIR_BALANCE:
                    from .sorter import PairBalance as Sorter

                # Beta ordering for research purposes
                case BalanceType.BATCH_BALANCE:
                    from .sorter.beta import BatchBalance as Sorter
                case BalanceType.RECURSIVE_BALANCE:
                    from .sorter.beta import RecursiveBalance as Sorter
                case BalanceType.RECURSIVE_PAIR_BALANCE:
                    from .sorter.beta import RecursivePairBalance as Sorter
                case BalanceType.EMA_BALANCE:
                    from .sorter.beta import EMABalance as Sorter
                case BalanceType.NTK_BALANCE:
                    from .sorter.beta import NTKBalance as Sorter
                case BalanceType.NTK_FIXED_BALANCE:
                    from .sorter.beta import NTKFixedBalance as Sorter
                case BalanceType.NTK_EIGEN:
                    from .sorter.beta import NTKEigen as Sorter
                case BalanceType.NTK_FIXED_EIGEN:
                    from .sorter.beta import NTKFixedEigen as Sorter
                case BalanceType.ORDER_BY_NORM:
                    from .sorter.beta import OrderByNorm as Sorter
                case BalanceType.ADAPTIVE_MEAN_BALANCE:
                    from .sorter.beta import AdaMeanBalance as Sorter
                case BalanceType.ADAPTIVE_EMA_BALANCE:
                    from .sorter.beta import AdaEMABalance as Sorter
                case BalanceType.SIGN_BALANCE:
                    from .sorter.beta import SignBalance as Sorter
                case _:
                    raise ValueError("Unsupported balance type!")

            self.sorter = Sorter(
                self.n, self.d, device=device, timer=timer, *args, **kwargs
            )

    def __iter__(self):
        # Record orders
        if self.record_orders:
            with self.timer("record_orders"):
                self.orders_history.append(self.sorter.orders.tolist())

        # Reset epoch
        with self.timer("reset_epoch"):
            self.sorter.reset_epoch()

        # Return the iterator
        with self.timer("sorter_iter"):
            return iter(self.sorter.orders.tolist())

    def __len__(self):
        return self.n

    def step(self, per_sample_grads: dict[str, Tensor], *args, **kwargs):
        with self.timer("sorter_step"):
            self.sorter.step(per_sample_grads, *args, **kwargs)
