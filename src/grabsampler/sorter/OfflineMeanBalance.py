import torch
from torch import Tensor
from overrides import overrides

from .SorterBase import SorterBase


class OfflineMeanBalance(SorterBase):
    def __init__(self, n: int, d: int, *args, **kwargs):
        super().__init__(n, d, *args, **kwargs)

        # Init in a way that self.reset_epoch is always called at the start of
        # each epoch
        self.idx = self.n
        self.left = self.n
        self.right = self.n - 1

        self.acc = torch.zeros(self.d, dtype=self.dtype, device=self.device)

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

        self.acc.zero_()

    @torch.no_grad()
    @overrides
    def single_step(self, grad: Tensor, *args, **kwargs):
        """

        :param grad: per sample gradient
        :return:
        """
        # if epsilon_{k,t} == +1
        if self.balance(grad, self.acc):
            self.next_orders[self.left] = self.orders[self.idx]
            self.acc.add_(grad)
            self.left += 1
        else:
            self.next_orders[self.right] = self.orders[self.idx]
            self.acc.sub_(grad)
            self.right -= 1

        self.idx += 1

    @torch.no_grad()
    @overrides
    def _step(self, grads: Tensor, b: int, centered: bool = False, *args, **kwargs):
        assert (
            b == self.n
        ), "OfflineMeanBalance requires all per-sample gradients to be passed in at once"

        if not centered:
            grads = grads - grads.mean(dim=0)

        for i in range(b):
            self.single_step(grads[i])

    @torch.no_grad()
    def offline_balance(self, grads: Tensor, rounds: int):
        """Directly balance the entire vector set for r rounds"""
        n = grads.shape[0]

        assert n == self.n, "number of examples doesn't match"

        for _ in range(rounds):
            self.reset_epoch()
            self.step({"": grads}, centered=True)

        return self.next_orders
