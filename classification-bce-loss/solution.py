import numpy as np


def bce_loss(p: np.ndarray, y: np.ndarray) -> float:
    """
    p: predicted probabilities, shape (n_samples,)
    y: true labels (0 or 1), shape (n_samples,)
    Returns a single scalar.
    """
    # TODO: Implement Binary Cross-Entropy, as derived in Theory.
    # Clip p away from exactly 0 or 1 before taking a log, and return
    # a plain Python float, not a NumPy scalar.
    eps = 1e-12
    p = np.clip(p, eps, 1 - eps)
    p1 = np.log(p)
    p2 = np.log1p(-p)
    lossi = y*p1 + (1-y)*p2

    loss = -np.mean(lossi)

    return loss

    
    



    
    pass
