import torch
import numpy as np
from torch_sparse import SparseTensor

from sklearn.random_projection import _check_density

from . import RandomProjection, get_rng


class JLVerySparseRP(RandomProjection):
    def __init__(self, density: float | str = "auto", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.PI = self.sparse_random_matrix(
            self.d,
            self.dd,
            density=density,
            seed_rng=self.rng,
            dtype=self.dtype,
            device=self.device,
            return_pytorch_sparse=False,
        )

    def project(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project a vector or a batch of vectors

        x need to be either in shape (d,) or (..., d)
        :param x:
        :return:
        """
        if len(x.shape) == 1:
            return self.PI.matmul(x.reshape(-1, 1)).squeeze(-1)
        elif x.shape[-1] == self.d:
            if len(x.shape) == 2:
                return self.PI.matmul(x.T).T
            shape = x.shape[:-1] + (self.dd,)
            x = x.reshape(-1, self.d).T.contiguous()
            x = self.PI.matmul(x)
            return x.T.reshape(shape)
        else:
            raise ValueError(
                f"Input shape {x.shape} is not compatible with projection matrix "
                f"{self.PI.shape}"
            )

    @staticmethod
    def sparse_random_matrix(
        d: int,
        dd: int,
        density: float | str = "auto",
        seed_rng: int | torch.Generator | None = None,
        dtype: torch.dtype = torch.float32,
        device: (str | torch.device) = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
        return_pytorch_sparse: bool = False,
    ) -> torch.Tensor | torch.sparse.Tensor | SparseTensor:
        # https://github.com/scikit-learn/scikit-learn/blob/7f9bad99d/sklearn/random_projection.py#L206

        # Note that this is not exactly "orthogonal", but "good enough"
        # https://stats.stackexchange.com/a/383443

        rng = get_rng(seed_rng, device=device)

        density = _check_density(density, d)

        if density == 1:
            return (
                torch.bernoulli(
                    torch.full((dd, d), 0.5, dtype=dtype, device=device),
                    generator=rng,
                )
                .mul_(2)
                .sub_(1)
                .sqrt_()
                .pow_(-1)
            )

        offset = 0
        crow_indices = [offset]
        col_indices = []
        for i in range(dd):
            columns = torch.bernoulli(
                torch.full((d,), density, dtype=dtype, device=device),
                generator=rng,
            )
            col_indices.append(columns.nonzero().squeeze(-1))
            offset += columns.count_nonzero().item()
            crow_indices.append(offset)
        col_indices = torch.cat(col_indices)

        data = (
            torch.bernoulli(torch.full((offset,), 0.5, dtype=dtype, device=device))
            .mul_(2)
            .sub_(1)
            .div_(np.sqrt(dd * density))
        )

        return (
            torch.sparse_csr_tensor(
                crow_indices,
                col_indices,
                data,
                size=(dd, d),
                dtype=dtype,
                device=device,
            )
            if return_pytorch_sparse
            else SparseTensor(
                rowptr=torch.tensor(crow_indices, dtype=torch.int64, device=device),
                col=col_indices,
                value=data,
                sparse_sizes=(dd, d),
                is_sorted=True,
                trust_data=True,
            )
        )
