import torch
from torch import Tensor
from overrides import overrides

from .SorterBase import SorterBase


class PairBalance(SorterBase):
    def __init__(self, n: int, d: int, *args, **kwargs):
        super().__init__(n, d, *args, **kwargs)

        assert n % 2 == 0, "Only support even number of examples in Pair balance"

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
        self.pair_cache = None

    @torch.no_grad()
    @overrides
    def single_step(self, grad: Tensor, *args, **kwargs):
        """

        :param grad: per sample gradient
        :return:
        """
        if self.pair_cache is None:
            self.pair_cache = grad
        else:
            # pair_g and grad should already be converted to dtype and device
            pair_g = self.pair_cache - grad

            # if epsilon_{k,t} == +1
            # equivalent to
            # |accumulator + grad_diff|_2 <= |accumulator 2 grad_diff|
            # where grad_diff = grad_1 - grad_2
            if self.balance(pair_g, self.acc):
                self.next_orders[self.left] = self.orders[self.idx]
                self.idx += 1
                self.next_orders[self.right] = self.orders[self.idx]
                self.acc += pair_g
            else:
                self.next_orders[self.right] = self.orders[self.idx]
                self.idx += 1
                self.next_orders[self.left] = self.orders[self.idx]
                self.acc -= pair_g

            self.idx += 1
            self.left += 1
            self.right -= 1

            self.pair_cache = None
