import torch
import torch.nn as nn
from torch import Tensor

from overrides import overrides
from absl import logging
from typing import Callable

from grabsampler.sorter.beta.functional import compute_kernel
from grabsampler.sorter import SorterBase, OfflineMeanBalance


class NTKFixedBalance(SorterBase):
    def __init__(
        self,
        n: int,
        d: int,
        # model and dataset are required to obtain the kernel matrix
        model: nn.Module,
        params: dict[str, Tensor],
        buffers: dict[str, Tensor],
        loss_fn: Callable,
        dataset: torch.utils.data.Dataset,
        num_rounds: int = 16,
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

        with self.timer("Balancing kernel matrix"):
            sorter = OfflineMeanBalance(
                n,
                n,
                random_first_epoch=False,
                **{k: kwargs[k] for k in kwargs.keys() & {"dtype", "device"}},
            )

            logging.info(f"Balancing ({n}, {n}) kernel matrix for {num_rounds} rounds")
            for round in range(num_rounds):
                logging.info(f"Round {round} of balancing kernel matrix")
                sorter.reset_epoch()
                sorter.step({"": K[sorter.orders]})

            self.orders = sorter.next_orders

        del sorter
        del self.next_orders

    @overrides
    def _reset_epoch(self):
        pass

    @overrides
    def _step(self, grads: Tensor, b: int, *args, **kwargs):
        pass
