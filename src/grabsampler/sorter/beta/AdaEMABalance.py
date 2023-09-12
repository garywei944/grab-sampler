import torch
from torch import Tensor
from overrides import overrides

from .. import SorterBase
from grabsampler.utils import EMAEstimator


class AdaEMABalance(SorterBase):
    idx: int

    # acc: running partial sum of shape (d,)
    acc: Tensor

    ema_m: EMAEstimator
    ema_v: EMAEstimator

    left: int
    right: int

    def __init__(
        self,
        n: int,
        d: int,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        *args,
        **kwargs
    ):
        super().__init__(n, d, *args, **kwargs)

        # Init in a way that self.reset_epoch is always called at the start of
        # each epoch
        self.idx = self.n
        self.left = self.n
        self.right = self.n - 1

        self.acc = torch.zeros(self.d, dtype=self.dtype, device=self.device)

        beta1, beta2 = betas
        self.ema_m = EMAEstimator(
            shape=self.d, beta=beta1, dtype=self.dtype, device=self.device
        )
        # TODO: ema_v might diverge from the variance used by adam, alternative is to use
        # the nu variable from adam optimizer
        self.ema_v = EMAEstimator(
            shape=self.d, beta=beta2, dtype=self.dtype, device=self.device
        )
        self.eps = eps

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
        # It's crucial here to work on a clone of grad.
        # Don't modify grad in-place
        grad = (grad - self.ema_m()) / (self.ema_v().sqrt() + self.eps)

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
        # Gary: this update assume that the entire batch is passed in, but if we are
        # using gradient accumulate, only a subset of the mini-batch is passed
        # then this is incorrect.
        self.ema_m.update(grads.mean(dim=0))
        self.ema_v.update(grads.mean(dim=0) ** 2)
        for i in range(b):
            self.single_step(grads[i])
