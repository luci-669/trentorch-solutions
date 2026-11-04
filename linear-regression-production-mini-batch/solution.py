import numpy as np


def train_linear_regression_production(
    input: np.ndarray,
    target: np.ndarray,
    lr: float,
    epochs: int,
    batch_size: int,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Mini-batch gradient descent, reusing mse_gradient / gd_step per batch
    instead of once per epoch over the full dataset.

    seed: pass a fixed integer to make training reproducible run to run.
    Leave as None for a genuinely random shuffle order.
    """
    # TODO: Each epoch, shuffle the sample order, split into batches of
    # batch_size, and take one gradient step per batch (not per epoch).
    # The last batch may be smaller -- handle that without crashing.
    # Build the RNG (np.random.default_rng(seed)) ONCE, before the epoch
    # loop -- not inside it. Re-creating it every epoch would give every
    # epoch the identical shuffle order.
    n_samples, in_features = input.shape
    target = target.reshape(-1, 1)                 

    weight = np.zeros((1, in_features))
    bias = np.zeros(1)
    rng = np.random.default_rng(seed)              

    for _ in range(epochs):
        order = rng.permutation(n_samples)         # every epoch
        for start in range(0, n_samples, batch_size):
            batch = order[start:start + batch_size]
            xb = np.asarray(input[batch], dtype=float)    # only this batch
            yb = np.asarray(target[batch], dtype=float)
            dw, db = mse_gradient(xb, weight, bias, yb)
            weight, bias = gd_step(weight, bias, dw, db, lr)

    return weight, bias
    pass
