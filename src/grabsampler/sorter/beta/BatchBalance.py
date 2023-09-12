import torch
from torch import Tensor
from overrides import overrides

from grabsampler.sorter.SorterBase import SorterBase


class BatchBalance(SorterBase):
    idx: int

    # acc_next: running partial sum of shape (d,)
    acc_next: Tensor

    # m: stale mean of shape (d,)
    run_sum: Tensor
    run_sum_next: Tensor

    left: int
    right: int

    def __init__(self, n: int, d: int, batch_size: int, *args, **kwargs):
        super().__init__(n, d, *args, **kwargs)

        self.batch_size = batch_size

        # Init in a way that self.reset_epoch is always called at the start of
        # each epoch
        self.idx = self.n
        self.left = self.n
        self.right = self.n - 1

        self.acc_next = torch.zeros(self.d, dtype=self.dtype, device=self.device)

        self.run_sum = torch.zeros_like(self.acc_next)
        self.run_sum_next = torch.zeros_like(self.acc_next)

    @overrides
    def _reset_epoch(self):
        # TODO: no guarantee that the entire dataset is looped during an epoch
        # let's assert it is for now.
        assert self.left > self.right
        assert self.idx == self.n

        self.idx = 0
        self.orders.copy_(self.next_orders)
        self.next_orders.zero_()

        self.left = 0
        self.right = self.n - 1

        self.acc_next.zero_()

        # We need to special care about the last batch
        self.run_sum = self.run_sum_next / self.n
        self.run_sum_next.zero_()

    @torch.no_grad()
    @overrides
    def _step(self, grads: Tensor, b: int, *args, **kwargs):
        self.run_sum_next += grads.sum(dim=0)

        acc = self.acc_next / self.idx if self.idx != 0 else self.acc_next
        signs = self.balance(grads - self.run_sum, acc)

        # Batch efficiently update the acc
        self.acc_next += torch.einsum("i,ij->j", (signs.int() - 0.5) * 2, grads)

        for s in signs:
            if s:
                self.next_orders[self.left] = self.orders[self.idx]
                self.left += 1
            else:
                self.next_orders[self.right] = self.orders[self.idx]
                self.right -= 1
            self.idx += 1
