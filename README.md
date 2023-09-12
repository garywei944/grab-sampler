`grab-sampler` is an efficient PyTorch-based sampler that supports GraB-style
example ordering by Online Gradient Balancing.
GraB algorithm takes O(d) extra memory and O(1) extra time compared with Random
Reshuffling.

Proposed in the
paper [GraB: Finding Provably Better Data Permutations than Random Reshuffling](https://arxiv.org/abs/2205.10733),
GraB (Gradient Balancing) is a data permutation algorithm that greedily choose
data orderings depending on per-sample gradients to further speed up
convergence of neural network training empirically.
Recent paper [Tighter Lower Bounds for Shuffling SGD: Random Permutations and Beyond
](https://arxiv.org/abs/2303.07160) shows that GraB provably achieves optimal
convergence
rate among arbitrary data permutations on SGD.
Observation shows that not only does GraB allow fast minimization of the
empirical risk, but also lets the model generalize better on multiple deep
learning tasks.

![](https://s3.amazonaws.com/ariseus.net/grab-sampler/grab-performance.png)

# Supported GraB Algorithms

- Mean Balance (Vanilla GraB, default)
- Pair Balance
- Recursive Balance
- Recursive Pair Balance
- Random Reshuffling (RR)
- Various experimental balance algorithms that doesn't provably outperform Mean Balance

In terms of balancing, all of the above algorithm supports

- Deterministic Balancing (default)
- Probabilistic Balancing

# Per-sample gradients, PyTorch 2, and Functional programming

GraB algorithm requires per-sample gradients while solving the *herding*
problem.
In general, it's hard to implement it in the vanilla PyTorch Automatic
Differentiation (AD) framework because the C++ kernel average the per-sample
gradients within a batch before it is passed to the next layer.

PyTorch 2 integrates Functorch that supports [efficient computation of
Per-sample Gradients](https://pytorch.org/tutorials/intermediate/per_sample_grads.html).
Alas, it requires
a [Functional programming](https://en.wikipedia.org/wiki/Functional_programming) style
of coding and requires the model to be pure functions, disallowing layers
including randomness (Dropout) or storing inter-batch statistics (BathNorm).

# Example Usage

To train a PyTorch model in a functional programming style using per-sample
gradients, one is likely to write a script like

```python
import torch
import torchopt
from torch.func import (
    grad, grad_and_value, vmap, functional_call
)
from functools import partial

from grabsampler import GraBSampler

# Initiate model, loss function, and dataset
model = ...
loss_fn = ...
dataset = ...

# Transform model into functional programming
# https://pytorch.org/docs/master/func.migrating.html#functorch-make-functional
# https://pytorch.org/docs/stable/generated/torch.func.functional_call.html
params = dict(model.named_parameters())
buffers = dict(model.named_buffers())

# initiate optimizer, using torchopt package
optimizer = torchopt.sgd(...)
opt_state = optimizer.init(params)  # init optimizer

###############################################################################
# Initiate GraB sampler and dataloader
sampler = GraBSampler(dataset, params)  # <- add this init of GraB sampler
dataloader = torch.utils.data.DataLoader(dataset, sampler=sampler)


###############################################################################


# pure function
def compute_loss(model, loss_fn, params, buffers, inputs, targets):
    prediction = functional_call(model, (params, buffers), (inputs,))

    return loss_fn(prediction, targets)


# Compute per sample gradients and loss
ft_compute_sample_grad_and_loss = vmap(
    grad_and_value(partial(compute_loss, model, loss_fn)),
    in_dims=(None, None, 0, 0)
)  # the only argument of compute_loss is batched along the first axis

for epoch in range(...):
    for _, (x, y) in enumerate(dataloader):
        ft_per_sample_grads, batch_loss = ft_compute_sample_grad_and_loss(
            params, buffers, x, y
        )

        #######################################################################
        sampler.step(ft_per_sample_grads)  # <- step compute GraB algorithm
        #######################################################################

        # The following is equivalent to
        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()
        grads = {k: g.mean(dim=0) for k, g in ft_per_sample_grads.items()}
        updates, opt_state = optimizer.update(
            grads, opt_state, params=params
        )  # get updates
        params = torchopt.apply_updates(
            params, updates
        )  # update model parameters
```

# Experiment Training Scripts

* [Image Classification](https://github.com/GarlGuo/GraB-lib/blob/main/experiments/cv/main.py) (
  CIFAR-10, MNIST, etc)
* [Causal Language Modeling](https://github.com/GarlGuo/GraB-lib/blob/main/experiments/nlp/clm/main.py) (
  Wikitext-103, OpenWebText, etc)

# How does `grabngo` work?

The reordering of data permutation happens at the beginning of each training
epoch, whenever an iterator of the dataloader is created,
e.g. `for _ in enumerate(dataloader):` internally calls `__iter__()` of the
`sampler` and updates the data ordering.
