import torch
import torch.nn as nn
from torch import Tensor

from overrides import overrides
from absl import logging

from grabsampler.sorter.beta.functional import compute_kernel
from grabsampler.sorter.SorterBase import SorterBase


class NTKBalance(SorterBase):
    def __init__(
        self,
        n: int,
        d: int,
        dataset: torch.utils.data.Dataset,
        largest_eig: bool = False,
        descending_eig: bool = True,
        abs_eig: bool = False,
        *args,
        **kwargs,
    ):
        raise NotImplementedError
        assert n == len(dataset), "The dataset size does not match n"

        super().__init__(n, d, *args, **kwargs)

        self.loader = torch.utils.data.DataLoader(dataset, batch_size=n)

        self.largest_eig = largest_eig
        self.descending_eig = descending_eig
        self.abs_eig = abs_eig
        self.args, self.kwargs = args, kwargs

        self.kwargs.pop("model", None)
        self.kwargs.pop("params", None)
        self.kwargs.pop("buffers", None)
        self.kwargs.pop("loss_fn", None)

        # The kernel matrix need to be computed before each epoch, but it is
        # not supported by current framework, so hack it here by setting
        # self.orders to None and asks users to call self.compute_order manually
        self.orders = None

        del self.next_orders

    def compute_order(
        self,
        # model and dataset are required to obtain the kernel matrix
        model: nn.Module,
        params: dict[str, Tensor],
        buffers: dict[str, Tensor],
        loss_fn: callable,
    ):
        data, targets = next(iter(self.loader))
        K = compute_kernel(
            model=model,
            params=params,
            buffers=buffers,
            loss_fn=loss_fn,
            data=data,
            targets=targets,
            *self.args,
            **self.kwargs,
        )  # (n, n)

        _, eigvec = torch.lobpcg(K, k=1, largest=self.largest_eig)

        logging.info(f"Computed largest/Smallest eigenvector of K(shape: {K.shape})")

        if self.abs_eig:
            eigvec = eigvec.abs()

        self.orders = torch.argsort(eigvec.squeeze(), descending=self.descending_eig)

        logging.info(f"ordering: {self.orders}")

    @overrides
    def _reset_epoch(self):
        if self.orders is None:
            raise RuntimeError(
                "self.orders is None, please call self.compute_order manually before each epoch"
            )

    @overrides
    def _step(self, grads: Tensor, b: int, *args, **kwargs):
        pass
