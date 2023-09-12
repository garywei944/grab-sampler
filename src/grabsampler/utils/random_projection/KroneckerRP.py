import torch
import numpy as np

from absl import logging

from sklearn.random_projection import _check_density

from . import RandomProjection, JLVerySparseRP, JLGaussianRP


class KroneckerRP(RandomProjection):
    def __init__(self, order: int = 2, sparse: bool = False, *args, **kwargs):
        """

        :param order: number of element random projection matrices used to construct the final projection matrix by kronecker product
        :param args:
        :param kwargs:
        """
        super().__init__(*args, **kwargs)

        assert not sparse, "Sparse Kronecker random projection is not supported yet"

        self.order = order

        self.ele_d = int(np.ceil(self.d ** (1 / self.order)))
        self.ele_dd = int(np.ceil(self.dd ** (1 / self.order)))

        logging.info(
            f"Constructing {self.order} random projection matrices of dimension "
            f"{self.ele_d} -> {self.ele_dd}"
        )

        self.dd = self.ele_dd**self.order

        self.real_d = self.d
        logging.info(f"Real incoming dimension: {self.real_d}")
        self.d = self.ele_d**self.order
        logging.info(f"Random projection dimension: {self.d} -> {self.dd}")

        self.padding = torch.nn.ConstantPad1d((0, self.d - self.real_d), 0)

        # We might want to use element rp matrices in different sizes in the future
        # Gary: Note that this should be reversed than the shapes of self.rp
        self.d_shapes = [self.ele_d] * self.order
        self.rp = []
        self.sparse = sparse
        if self.sparse:
            logging.info("Using sparse random projection matrices")

        # Make sure they use the same generator, otherwise, all random matrices are
        # equivalent
        kwargs["seed_rng"] = self.rng
        kwargs.pop("d", None)
        kwargs.pop("dd", None)
        for i in range(self.order):
            if self.sparse:
                # Gary: it's important that the density is correctly handled
                density = _check_density(kwargs.get("density", "auto"), self.ele_d)
                kwargs["density"] = density

                self.rp.append(
                    JLVerySparseRP(d=self.ele_d, dd=self.ele_dd, *args, **kwargs)
                )
            else:
                self.rp.append(
                    JLGaussianRP(d=self.ele_d, dd=self.ele_dd, *args, **kwargs)
                )

    def project(self, x: torch.Tensor) -> torch.Tensor:
        assert len(x.shape) <= 2, f"Input shape {x.shape} is not supported"

        # reshape x to fit kronecker product computation
        b = x.shape[0] if len(x.shape) > 1 else 1
        x = self.padding(x).reshape(b, *self.d_shapes)

        # for i, ele_d in zip(range(self.order), self.d_shapes):
        for i in range(self.order):
            # KP: kronecker product
            # MatMul(KP(A, B), C)
            # = MatMul(MatMul(B, C.reshape()), A.T)
            # = MatMul(A, MatMul(B, C.reshape()).T).T
            x = self.rp[i].project(x)
            x = x.permute(-1, *range(self.order))

        # move b to the first dimension
        return x.permute(-1, *range(self.order)).reshape(b, -1).squeeze()
