import torch
from torch import Tensor
from overrides import overrides

from grabsampler.sorter.SorterBase import SorterBase
from grabsampler.utils import EMAEstimator


class EMABalance(SorterBase):
    idx: int

    # acc: running partial sum of shape (d,)
    acc: Tensor
    # m: stale mean of shape (d,)
    run_sum: Tensor

    ema: EMAEstimator

    left: int
    right: int

    def __init__(
        self, n: int, d: int, ema_decay: float = 0.1, *args, **kwargs  # EMA decay rate
    ):
        super().__init__(n, d, *args, **kwargs)

        # Init in a way that self.reset_epoch is always called at the start of
        # each epoch
        self.idx = self.n
        self.left = self.n
        self.right = self.n - 1

        self.acc = torch.zeros(self.d, dtype=self.dtype, device=self.device)

        self.ema = EMAEstimator(
            shape=self.d, beta=1 - ema_decay, dtype=self.dtype, device=self.device
        )

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
        # Note that ema() return the ema mean of last observation, which is 0 for the first call
        grad = grad - self.ema()

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
    def _step(self, grads: Tensor, b: int, *args, **kwargs):
        self.ema.update(grads.mean(dim=0))
        for i in range(b):
            self.single_step(grads[i])
