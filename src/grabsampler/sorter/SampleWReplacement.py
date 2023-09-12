import torch
from overrides import overrides

from .SorterBase import SorterBase


class SampleWReplacement(SorterBase):
    def __init__(self, n: int, d: int, *args, **kwargs):
        super().__init__(n, d, *args, **kwargs)

        del self.next_orders

    @overrides()
    def _reset_epoch(self):
        self.orders = torch.randint(self.n, (self.n,), dtype=torch.int64)

    @overrides()
    def _step(self, *args, **kwargs):
        pass
