import numpy as np


def mse_gradient(
    input: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray | None,
    target: np.ndarray,
) -> tuple[np.ndarray, np.ndarray | None]:
    """
    input:  shape (batch_size, in_features)
    weight: shape (out_features, in_features)
    bias:   shape (out_features,), or None
    target: shape (batch_size, out_features)

    Returns:
        grad_weight: same shape as weight
        grad_bias: same shape as bias, or None if bias is None
    """
    # TODO: Implement the gradient formulas derived in Theory.
    # Do not use an autograd library -- these are the manual derivatives.
    
    y_pred = input @ weight.T 
    if bias is not None:
         y_pred = input @ weight.T+bias
    n = y_pred.size
        
    g = (2/n)*(y_pred - target)

    dw = g.T @ input
    db = g.sum(axis = 0) if bias is not None else None

    return dw,db
    
    


    
    pass
