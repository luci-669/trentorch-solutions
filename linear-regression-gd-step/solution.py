import numpy as np


def gd_step(
    weight: np.ndarray,
    bias: np.ndarray | None,
    grad_weight: np.ndarray,
    grad_bias: np.ndarray | None,
    lr: float,
) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Apply one gradient-descent update.

    Returns:
        updated_weight, updated_bias
    """
    # TODO: Implement the update rule from Theory.
    # Return new values -- do not mutate weight or bias in place.
    # bias=None means there's no bias parameter -- updated_bias stays None.
    weight = np.asarray(weight, dtype=float)
    new_weight = weight - lr * np.asarray(grad_weight, dtype=float)

    if bias is None:
        return new_weight, None

    bias = np.asarray(bias, dtype=float)
    if grad_bias is None:
        return new_weight, bias.copy()

    return new_weight, bias - lr * np.asarray(grad_bias, dtype=float)





    
    pass
