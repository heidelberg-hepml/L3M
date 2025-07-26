from torch.optim import Optimizer
from functools import partial
from torch.optim.lr_scheduler import LambdaLR

import math

def get_wsd_schedule(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_stable_steps: int,
    num_decay_steps: int,
):
    """
    Returns warmup-stable-decay learning rate scheduler, with a linear warmup and a cosine decay.

    Arguments:
        optimizer (torch.optim.lr_scheduler.Optimizer):
            The optimizer to which the scheduler is applied.
        num_warmup_steps (int):
            Number of warmup steps
        num_stable_steps (int):
            Number of stable steps
        num_decay_steps (int):
            Number of decay steps.
    """
    def _lambda(
        current_step: int,
        *,
        num_warmup_steps: int,
        num_stable_steps: int,
        num_decay_steps: int,
    ):
        _current_step = current_step % (num_warmup_steps + num_stable_steps + num_decay_steps)

        if _current_step < num_warmup_steps:
            return float(_current_step) / float(max(1, num_warmup_steps))
        if _current_step < num_warmup_steps + num_stable_steps:
            return 1.0
        progress = float(_current_step - num_warmup_steps - num_stable_steps) / float(max(1, num_decay_steps))
        value = max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
        return value

    lr_lambda = partial(
        _lambda,
        num_warmup_steps=num_warmup_steps,
        num_stable_steps=num_stable_steps,
        num_decay_steps=num_decay_steps
    )
    return LambdaLR(optimizer, lr_lambda)
