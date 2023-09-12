import torch
from overrides import overrides

from .SorterBase import SorterBase


class RandomReshuffling(SorterBase):
    def __init__(
        self, n: int, d: int, random_first_epoch: bool = True, *args, **kwargs
    ):
        super().__init__(n, d, *args, **kwargs)

        self.inited = random_first_epoch

        del self.next_orders

    @overrides()
    def _reset_epoch(self):
        if not self.inited:
            self.inited = True
            self.orders = torch.arange(self.n, dtype=torch.int64)
        else:
            self.orders = torch.randperm(self.n, dtype=torch.int64)

    @overrides()
    def _step(self, *args, **kwargs):
        pass
