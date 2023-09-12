import torch
from torch import Tensor
from overrides import overrides

from .SorterBase import SorterBase

from absl import logging


class FixedOrdering(SorterBase):
    def __init__(self, n: int, d: int, orders: list[int], *args, **kwargs):
        super().__init__(n, d, *args, **kwargs)

        assert len(orders) == n, "The length of orders must be equal to n!"

        if orders is None:
            logging.warning("No orders are provided, using fixed natural ordering.")
            self.orders = torch.arange(self.n, dtype=torch.int64)
        else:
            self.orders = torch.tensor(orders, dtype=torch.int64)

        del self.next_orders

    @overrides
    def _reset_epoch(self):
        pass

    @overrides
    def _step(self, grads: Tensor, b: int, *args, **kwargs):
        pass
