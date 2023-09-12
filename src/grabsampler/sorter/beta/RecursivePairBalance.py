import torch
from overrides import overrides
from torch import Tensor
import torch.nn.functional as F

from absl import logging
from overrides import overrides

from grabsampler.sorter.SorterBase import SorterBase


class RecursivePairBalance(SorterBase):
    idx: int
    batch_size: int

    # Actually an array implementation of a full binary tree, each acc is of
    # size (d,), so the full binary tree is of size (b-1, d)
    acc_bt: Tensor

    batch_orders: list[list[int]]

    def __init__(self, n: int, d: int, batch_size: int, *args, **kwargs):
        # https://stackoverflow.com/a/57025941
        assert (
            batch_size & (batch_size - 1) == 0
        ) and batch_size >= 2, "Only support batch size power of 2!"
        assert n % batch_size == 0, "Only support n % batch_size == 0!"

        super().__init__(n, d, *args, **kwargs)

        self.batch_size = batch_size

        # Init in a way that self.reset_epoch is always called at the start of
        # each epoch
        self.idx = -1

        self.acc_bt = torch.zeros(
            self.batch_size - 1, self.d, dtype=self.dtype, device=self.device
        )

        self.batch_orders = []

        # used to compute pair_grads efficiently
        self.filter = torch.tensor([1, -1], dtype=self.dtype, device=self.device).view(
            1, 1, 2, 1
        )

        # used to efficiently divide the grads of size (b, d) into left and
        # right sub grads of size (b/2, d) according to the signs of herding
        self.pick_grad_by_signs = torch.tensor([[False, True], [True, False]])

        del self.next_orders

    def get_orders(self) -> list:
        # Now we have self.batch_order to be size (B, b), where B is number of
        # batches and b is the batch size
        # the odd entry of each batch belongs to a positive leaf, while the
        # even ones belong to a negative leaf
        # Now we need to reverse the negative leaves
        orders = []
        B = len(self.batch_orders)
        b = len(self.batch_orders[0])

        for i in range(b):
            for j in range(B):
                jj = j if i % 2 == 0 else B - 1 - j
                orders.append(self.batch_orders[jj][i])

        return orders

    @overrides
    def _reset_epoch(self):
        # TODO: no guarantee that the entire dataset is looped during an epoch
        # let's assert it is for now.

        # For recursive balance, the orders are not updated on the fly, we have
        # to update it at the beginning of reset_epoch() in the root

        if self.orders is not None:
            # hack: it's not before the start of 1st epoch
            if self.idx != -1:
                self.orders = torch.tensor(self.get_orders(), dtype=torch.int64)

            logging.debug(len(self.orders))
            logging.debug(self.orders)

            assert len(self.orders) == self.n, (
                "The sampler is not updated with all examples during last "
                "epoch. Aborted."
            )

        self.idx = 0

        self.acc_bt.zero_()

        self.batch_orders.clear()

    def recursive_balance(
        self, grads: Tensor, indices: list[int], tree_idx: int = 0
    ) -> list[int]:
        b, d = grads.shape
        assert b == len(indices)

        # Base case of recursion
        if b == 1:
            return indices

        # To perform pair batch balance
        # 1. the odd rows minus the even rows, ending in b/2 pairs. An
        #   efficient implementation is by conv2d
        # 2. inner product to compute the herding
        # 3. construct 2 new grads of size (b/2, d), and call this function
        #   recursively.

        # step 1: compute pairs
        pair_grad = F.conv2d(grads.view(1, 1, b, d), self.filter, stride=(2, 1)).view(
            -1, d
        )  # (b/2, d)

        # step 2: get the signs
        signs = self.balance(pair_grad, self.acc_bt[tree_idx]).cpu()  # (d/2,)

        # step 3: construct left and right grads for recursive computation
        left_mask = self.pick_grad_by_signs[signs.int()].flatten()  # (d,)
        right_mask = self.pick_grad_by_signs[~signs.int()].flatten()  # (d,)
        left_grads = grads[left_mask]  # (b/2, d)
        right_grads = grads[right_mask]  # (b/2, d)
        left_indices = torch.tensor(indices)[left_mask].tolist()  # (b/2,)
        right_indices = torch.tensor(indices)[right_mask].tolist()  # (b/2,)

        # step 4: update the corresponding acc
        self.acc_bt[tree_idx] += left_grads.sum(axis=0) - right_grads.sum(axis=0)

        # Gary: don't manually free GPU tensors!
        # For cifar-10 experiments, batch size = 16 only saves ~14 MB RAM
        # but slows down 1/3 speed, from 90it/s -> 60it/s

        return self.recursive_balance(
            left_grads, left_indices, 2 * tree_idx + 1
        ) + self.recursive_balance(right_grads, right_indices, 2 * tree_idx + 2)

    @torch.no_grad()
    @overrides
    def _step(self, grads: Tensor, b: int, *args, **kwargs):
        self.batch_orders.append(
            self.recursive_balance(grads, self.orders[self.idx : self.idx + b].tolist())
        )
        self.idx += b
