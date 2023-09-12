import torch
import torch.nn as nn
from torch import Tensor
from torch.func import functional_call, vmap, grad

from absl import logging
from tqdm.auto import tqdm


def compute_kernel(
    model: nn.Module,
    params: dict[str, Tensor],
    buffers: dict[str, Tensor],
    loss_fn: callable,
    data: Tensor,
    targets: Tensor,
    batch_size: int = 128,
    # used by model and data
    dtype: torch.dtype = torch.float32,
    device: (str | torch.device) = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    ),
    # used to store kernel matrix
    kernel_dtype: torch.dtype = torch.float32,
    kernel_device: str
    | torch.device = ("cuda" if torch.cuda.is_available() else "cpu"),
    centered_feature_map: bool = True,
    *args,
    **kwargs,
):
    logging.info("Computing the kernel matrix, this may take a while...")

    n = data.shape[0]
    d = sum(p.numel() for p in params.values())

    def opt(_params, _buffers, _model, _x, _y):
        return loss_fn(functional_call(_model, (_params, _buffers), (_x,)), _y)

    with torch.no_grad():
        PHI = torch.zeros((n, d), dtype=kernel_dtype, device=kernel_device)

        for i in tqdm(range(0, n, batch_size), leave=False):
            j = min(i + batch_size, n)

            X = data[i:j].to(dtype=dtype, device=device)
            Y = targets[i:j].to(device=device)

            grads = vmap(grad(opt), (None, None, None, 0, 0))(
                params, buffers, model, X, Y
            )
            PHI[i:j] = torch.cat(
                [g.reshape(j - i, -1) for g in grads.values()], dim=1
            ).to(
                dtype=kernel_dtype, device=kernel_device
            )  # (batch_size, d)

        # We might need to center the feature map, so we need to compute the mean
        # of the feature map after getting all gradients
        if centered_feature_map:
            PHI -= PHI.mean(dim=0)

        logging.info(
            "Finish computing the feature map, now computing the kernel matrix..."
        )

        return PHI @ PHI.T  # (n, n)
