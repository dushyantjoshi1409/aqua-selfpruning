"""Dataset loading utilities.

Scikit-learn is used exclusively for loading standard datasets — not for
training, models, or gradient computation.
"""

import numpy as np


def load_digits_data(test_fraction=0.2, seed=42):
    """Load sklearn's 8x8 digits dataset (1797 samples, 10 classes).

    Returns (X_train, y_train, X_test, y_test) with X normalized to [0, 1].
    """
    from sklearn.datasets import load_digits

    digits = load_digits()
    X = digits.data.astype(np.float64) / 16.0
    y = digits.target.astype(np.int64)

    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(X))
    split = int(len(X) * (1.0 - test_fraction))

    return X[idx[:split]], y[idx[:split]], X[idx[split:]], y[idx[split:]]


def load_mnist_data(n_train=10000, n_test=2000, seed=42):
    """Load MNIST via sklearn's fetch_openml (subsampled for speed).

    Returns (X_train, y_train, X_test, y_test) with X normalized to [0, 1].
    """
    from sklearn.datasets import fetch_openml

    mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
    X = mnist.data.astype(np.float64) / 255.0
    y = mnist.target.astype(np.int64)

    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(X))
    X, y = X[idx], y[idx]

    return X[:n_train], y[:n_train], X[n_train:n_train + n_test], y[n_train:n_train + n_test]


def iterate_batches(X, y, batch_size, shuffle=True, rng=None):
    """Yield (X_batch, y_batch) mini-batches."""
    n = len(X)
    if shuffle:
        if rng is None:
            rng = np.random.RandomState()
        idx = rng.permutation(n)
    else:
        idx = np.arange(n)

    for start in range(0, n, batch_size):
        batch_idx = idx[start:start + batch_size]
        yield X[batch_idx], y[batch_idx]
