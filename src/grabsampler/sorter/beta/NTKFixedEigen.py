import torch
import torch.nn as nn
from torch import Tensor

from overrides import overrides
from absl import logging

from grabsampler.sorter.beta.functional import compute_kernel
from grabsampler.sorter.SorterBase import SorterBase


class NTKFixedEigen(SorterBase):
    def __init__(
        self,
        n: int,
        d: int,
        # model and dataset are required to obtain the kernel matrix
        model: nn.Module,
        params: dict[str, Tensor],
        buffers: dict[str, Tensor],
        loss_fn: callable,
        dataset: torch.utils.data.Dataset,
        largest_eig: bool = False,
        descending_eig: bool = True,
        abs_eig: bool = False,
        save_k: bool = False,
        *args,
        **kwargs,
    ):
        assert n == len(dataset), "The dataset size does not match n"

        super().__init__(n, d, *args, **kwargs)

        with self.timer("Compute kernel matrix"):
            loader = torch.utils.data.DataLoader(dataset, batch_size=n)

            data, targets = next(iter(loader))
            del loader

            K = compute_kernel(
                model=model,
                params=params,
                buffers=buffers,
                loss_fn=loss_fn,
                data=data,
                targets=targets,
                *args,
                **kwargs,
            )  # (n, n)

            _, eigvec = torch.lobpcg(K, k=1, largest=largest_eig)

            logging.info(
                f"Computed largest/Smallest eigenvector of K(shape: {K.shape})"
            )

            if abs_eig:
                eigvec = eigvec.abs()

            self.orders = torch.argsort(eigvec.squeeze(), descending=descending_eig)

        logging.info(f"ordering: {self.orders}")

        del self.next_orders

        if save_k:
            self.K = K
        else:
            del K
            torch.cuda.empty_cache()

    @overrides
    def _reset_epoch(self):
        pass

    @overrides
    def _step(self, grads: Tensor, b: int, *args, **kwargs):
        pass
