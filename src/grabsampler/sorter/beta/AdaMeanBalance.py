import torch
from torch import Tensor
from overrides import overrides
from torchopt.typing import OptState

from .. import SorterBase
from ...utils import StaleMeanEstimator


class AdaMeanBalance(SorterBase):
    idx: int

    # acc: running partial sum of shape (d,)
    acc: Tensor

    m: StaleMeanEstimator
    v: Tensor

    left: int
    right: int

    def __init__(self, n: int, d: int, eps: float = 1e-8, *args, **kwargs):
        super().__init__(n, d, *args, **kwargs)

        # Init in a way that self.reset_epoch is always called at the start of
        # each epoch
        self.idx = self.n
        self.left = self.n
        self.right = self.n - 1

        self.eps = eps

        self.acc = torch.zeros(self.d, dtype=self.dtype, device=self.device)

        self.m = StaleMeanEstimator(self.d, dtype=self.dtype, device=self.device)
        self.v = torch.ones_like(self.acc)

    @torch.no_grad()
    def update_v(self, v: Tensor):
        self.v = v.sqrt() + self.eps

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
        self.m.reset()

    @torch.no_grad()
    @overrides
    def single_step(self, grad: Tensor, *args, **kwargs):
        """

        :param grad: per sample gradient
        :return:
        """
        # It's crucial here to work on a clone of grad.
        # Don't modify grad in-place

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
    @overrides()
    def _step(self, grads: Tensor, b: int, *args, **kwargs):
        grads = (grads - self.m()) / self.v
        for i in range(b):
            self.single_step(grads[i], *args, **kwargs)
