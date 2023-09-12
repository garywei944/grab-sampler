import torch
from torch import Tensor
from overrides import overrides

from .SorterBase import SorterBase


class ShuffleOnce(SorterBase):
    def __init__(
        self, n: int, d: int, random_first_epoch: bool = True, *args, **kwargs
    ):
        super().__init__(n, d, *args, **kwargs)

        self.orders = (
            torch.randperm(self.n, dtype=torch.int64)
            if random_first_epoch
            else torch.arange(self.n, dtype=torch.int64)
        )

        del self.next_orders

    @overrides
    def _reset_epoch(self):
        pass

    @overrides
    def _step(self, grads: Tensor, b: int, *args, **kwargs):
        pass
