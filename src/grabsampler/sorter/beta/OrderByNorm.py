import torch
from torch import Tensor
from overrides import overrides

from absl import logging

from .. import SorterBase, OfflineMeanBalance


class OrderByNorm(SorterBase):
    # m: stale mean of shape (d,)
    run_sum: Tensor
    run_sum_next: Tensor

    def __init__(self, n: int, d: int, descending_norm: bool = True, *args, **kwargs):
        super().__init__(n, d, *args, **kwargs)

        self.idx = 0
        self.norms = torch.zeros(n, dtype=self.dtype, device=self.device)
        self.descending = descending_norm

        self.run_sum = torch.zeros(self.d, dtype=self.dtype, device=self.device)
        self.run_sum_next = torch.zeros_like(self.run_sum)

        self.inited = False

        self.sorter = OfflineMeanBalance(n, 1)

    @overrides
    def _reset_epoch(self):
        if not self.inited:
            self.orders.copy_(self.next_orders)
            self.inited = True
            del self.next_orders
        else:
            logging.info(f"norms: {self.norms}")
            self.orders = self.orders[self.sorter.offline_balance(self.norms, 16).cpu()]
            logging.info(f"orders: {self.orders}")
        self.idx = 0
        self.run_sum = self.run_sum_next / self.n
        self.run_sum_next.zero_()

    @torch.no_grad()
    @overrides
    def _step(self, grads: Tensor, b: int, *args, **kwargs):
        self.run_sum_next += grads.sum(dim=0)
        self.norms[self.idx : self.idx + b] = grads.norm(dim=1)
        self.idx += b
