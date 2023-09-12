import torch
import numpy as np

from . import RandomProjection, get_rng


class JLGaussianRP(RandomProjection):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.PI = self.gaussian_random_matrix(
            self.d,
            self.dd,
            seed_rng=self.rng,
            dtype=self.dtype,
            device=self.device,
        )

    def project(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project a vector or a batch of vectors

        x need to be either in shape (d,) or (..., d)
        :param x:
        :return:
        """
        return torch.einsum("ik,...k->...i", self.PI, x)

    @staticmethod
    def gaussian_random_matrix(
        d: int,
        dd: int,
        seed_rng: int | torch.Generator | None = None,
        dtype: torch.dtype = torch.float32,
        device: (str | torch.device) = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
    ) -> torch.Tensor:
        # https://github.com/scikit-learn/scikit-learn/blob/7f9bad99d/sklearn/random_projection.py#L166

        # Note that this is not exactly "orthogonal", but "good enough"
        # https://stats.stackexchange.com/a/383443

        rng = get_rng(seed_rng, device=device)

        return torch.normal(
            0,
            1 / np.sqrt(dd),
            size=(dd, d),
            dtype=dtype,
            device=device,
            generator=rng,
        )
