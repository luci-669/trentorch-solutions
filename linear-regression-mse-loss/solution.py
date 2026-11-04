import numpy as np


def mse_loss(input: np.ndarray, target: np.ndarray, reduction: str = "mean") -> float | np.ndarray:
    """
    input:     shape matching target, any shape
    target:    shape matching input, any shape
    reduction: 'mean' | 'sum' | 'none'

    Returns:
        a Python float when reduction is 'mean' or 'sum';
        an array shaped like input when reduction is 'none'
    """
    # TODO: Implement mean squared error from Theory.
    # Branch on `reduction` explicitly. Raise ValueError for anything else.
def mse_loss(input, target, reduction="mean"):
    loss = (input - target) ** 2

    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    elif reduction == "none":
        return loss
    else:
        raise ValueError(f"Invalid reduction: {reduction}")
    pass
