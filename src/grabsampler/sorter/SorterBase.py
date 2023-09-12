import random
from abc import ABC, abstractmethod
from typing import Callable

import torch
from torch import Tensor

from overrides import final

from grabsampler.utils import EventTimer
from grabsampler.utils.random_projection import RandomProjectionType


def deterministic_balance() -> Callable[[Tensor, Tensor], bool | Tensor]:
    # The following is equivalent to
    # return 1 if torch.norm(aggregator + vec) <= torch.norm(aggregator - vec)
    #   else -1

    # Gary: Note that if we use <= instead of <, the first vector is always
    # assigned with "+1"
    return lambda s, z: torch.inner(s, z) < 0


def probabilistic_balance(c: float = 30) -> Callable[[Tensor, Tensor], bool | Tensor]:
    # p = 0.5 - torch.dot(vec, aggregator) / 60
    # if random.random() <= p:
    #     return 1
    # else:
    #     return -1
    return lambda s, z: random.random() < 0.5 - torch.inner(s, z) / (2 * c)


class SorterBase(ABC):
    def __init__(
        self,
        n: int,
        d: int,
        random_first_epoch: bool = True,
        normalize_grads: bool = False,
        random_projection: RandomProjectionType = RandomProjectionType.NONE,
        pi_eps: float = 0.1,
        seed: int = 42,  # Only used for generating random projection
        prob_balance: bool = False,
        prob_balance_c: float = 30,
        dtype: torch.dtype = torch.float32,
        device: (str | torch.device) = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
        timer: EventTimer | None = None,
        record_herding: bool = False,
        stale_mean_herding: bool = False,
        cuda_herding: bool = False,
        record_norm: bool = False,
        *args,
        **kwargs,
    ):
        self.n: int = n
        self.d: int = d
        self.dtype: torch.dtype = dtype
        self.device: str | torch.device = device

        # Disable timer by default in production
        self.timer: EventTimer = timer or EventTimer(device=device, disable=True)

        self.normalize: bool = normalize_grads
        self.balance: Callable[[Tensor, Tensor], bool | Tensor] = (
            probabilistic_balance(prob_balance_c)
            if prob_balance
            else deterministic_balance()
        )

        self.orders: Tensor = (
            torch.randperm(self.n, dtype=torch.int64)
            if random_first_epoch
            else torch.arange(self.n, dtype=torch.int64)
        )
        self.next_orders: Tensor = self.orders.clone()

        # Random projection
        with self.timer("init_random_projection"):
            random_projection = RandomProjectionType(random_projection)
            self.random_projection = True

            match random_projection:
                case RandomProjectionType.NONE:
                    self.random_projection = False
                    RP = lambda *_args, **_kwargs: None
                case RandomProjectionType.JL_GAUSSIAN:
                    from grabsampler.utils.random_projection import JLGaussianRP as RP
                case RandomProjectionType.JL_VERY_SPARSE:
                    from grabsampler.utils.random_projection import JLVerySparseRP as RP
                case RandomProjectionType.KRONECKER:
                    from grabsampler.utils.random_projection import KroneckerRP as RP
                case _:
                    raise ValueError(
                        f"Unknown random projection type {random_projection}"
                    )
            if self.random_projection:
                self.PI = RP(
                    n=self.n,
                    d=self.d,
                    eps=pi_eps,
                    seed_rng=seed,
                    dtype=dtype,
                    device=device,
                    order=kwargs.get("kron_order", 2),
                )
                self.d = self.PI.dd

        # Only for research purpose
        self.record_herding = record_herding

        if self.record_herding:
            self.stale_mean_herding = stale_mean_herding
            # use cuda to store herding if feasible
            self.herding_device = "cuda" if cuda_herding else "cpu"

            if self.stale_mean_herding:
                # this is equivalent to the self.run_sum for some balancing,
                # but like pair balance doesn't have it
                # Gary: use d instead of self.d b/c self.d may be indeed dd
                self.herding_prefix_sum = torch.zeros(d, dtype=dtype, device=device)
                self.herding_stale_mean = torch.zeros_like(self.herding_prefix_sum)
                self.herding_run_sum = torch.zeros_like(self.herding_prefix_sum)
                self.herding = -torch.inf  # inf norm
                self.avg_grad_error = -torch.inf  # 2-norm
            else:
                # Always use CPU to store all the gradients
                self.herding_idx = 0
                self.herding_grads = torch.zeros(
                    n, d, dtype=dtype, device=self.herding_device
                )

        # only for research purpose
        self.record_norm = record_norm
        if self.record_norm:
            self.grad_norms = []

    @abstractmethod
    def _reset_epoch(self):
        ...

    @final
    def reset_epoch(self):
        self._reset_epoch()

        # Reset the prefix sum of herding
        if self.record_herding:
            self.herding = -torch.inf
            self.avg_grad_error = -torch.inf

            if self.stale_mean_herding:
                self.herding_stale_mean = self.herding_run_sum / self.n
                self.herding_run_sum.zero_()
                self.herding_prefix_sum.zero_()
            else:
                self.herding_grads.zero_()
                self.herding_idx = 0

        # Reset the norm records
        if self.record_norm:
            self.grad_norms.clear()

    def single_step(self, grad: Tensor, *args, **kwargs):
        raise NotImplementedError

    @torch.no_grad()
    def _step(self, grads: Tensor, b: int, *args, **kwargs):
        for i in range(b):
            self.single_step(grads[i], *args, **kwargs)

    @torch.no_grad()
    @final
    def step(
        self, per_sample_grads: dict[str, Tensor], batch: bool = True, *args, **kwargs
    ):
        # Always assume the first axis is batch size
        b = next(iter(per_sample_grads.values())).shape[0] if batch else 1

        # To make everything simpler, always assume the order of the current
        # epoch is 1,2,3,...
        grads = torch.cat(
            [g.reshape(b, -1) for g in per_sample_grads.values()], dim=1
        ).to(
            dtype=self.dtype, device=self.device
        )  # (B, d)

        # record herding
        if self.record_herding:
            with self.timer("record_herding"):
                self.herding_step(grads)

        # record norm
        if self.record_norm:
            with self.timer("record_norm"):
                self.grad_norms.extend(grads.norm(dim=1).tolist())

        if self.random_projection:
            with self.timer("random_projection"):
                grads = self.PI.project(grads)

        if self.normalize:
            grads = torch.nn.functional.normalize(grads, dim=1)

        assert grads.shape[1] == self.d

        with self.timer("balance_step"):
            self._step(grads, b, *args, **kwargs)

    @torch.no_grad()
    @final
    def herding_step(self, grads: Tensor):
        if self.stale_mean_herding:
            # explicitly make a new copy of grads
            # grads_hat is centered at 0 by stale mean
            grads_hat = grads - self.herding_stale_mean

            self.herding_run_sum += grads.sum(dim=0)
            self.herding_prefix_sum += grads_hat.sum(dim=0)

            torch.cumsum(grads_hat, dim=0, out=grads_hat)
            grads_hat += self.herding_run_sum

            self.herding = max(self.herding, grads_hat.abs().max().item())

            self.avg_grad_error = max(
                self.avg_grad_error, grads_hat.norm(dim=1).max().item()
            )
        else:
            b = grads.shape[0]
            self.herding_grads[self.herding_idx : self.herding_idx + b] = (
                grads.detach().clone().to(device=self.herding_device)
            )
            self.herding_idx += b

            # Compute the herding
            if self.herding_idx == self.n:
                self.herding_grads -= self.herding_grads.mean(dim=0)
                torch.cumsum(self.herding_grads, dim=0, out=self.herding_grads)
                self.herding = self.herding_grads.abs().max().item()
                self.avg_grad_error = self.herding_grads.norm(dim=1).max().item()
