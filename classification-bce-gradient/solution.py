import numpy as np


def bce_gradient(
    input: np.ndarray,
    p: np.ndarray,
    target: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    input:  shape (batch_size, in_features)
    p:      shape (batch_size, 1), sigmoid(linear(input, weight, bias))
    target: shape (batch_size, 1)

    Returns:
        grad_weight: shape (1, in_features)
        grad_bias: shape (1,)
    """
    # TODO: Implement the gradient formulas derived in Theory.
    # Do not use an autograd library -- these are the manual derivatives.
    n = input.shape[0]
    eps = 1e-12
    p = np.clip(p, eps, 1 - eps)
    error = (p - target)
    w =  error.T @ input / n
    b = error.mean(axis = 0)
    return w,b
    
    
    pass
