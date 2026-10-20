"""Neural network layers built on the autodiff engine.

Each layer stores its parameters as Tensor objects and supports a forward
pass via __call__.  Linear layers carry a binary mask for pruning support:
pruned (masked-out) weights are zeroed before the forward matmul so they
contribute exactly zero to the output.  Because the matmul gradient
d(x@W)/dW = x^T @ grad_output does not depend on W, the autodiff engine
computes a meaningful gradient even for zeroed positions — this is
essential for importance scoring and regrowth decisions.
"""

import numpy as np
from engine.tensor import Tensor


class Linear:
    """Fully-connected layer with optional binary mask for pruning.

    Weight initialization follows He/Kaiming (scale = sqrt(2/fan_in)),
    suitable for networks using ReLU activations.
    """

    def __init__(self, in_features: int, out_features: int):
        scale = np.sqrt(2.0 / in_features)
        self.weight = Tensor(
            np.random.randn(in_features, out_features) * scale,
            requires_grad=True,
        )
        self.bias = Tensor(np.zeros(out_features), requires_grad=True)
        self.mask = np.ones((in_features, out_features), dtype=np.float64)

    def __call__(self, x: Tensor) -> Tensor:
        """Forward pass: x @ (weight * mask) + bias."""
        self.weight.data *= self.mask
        return x @ self.weight + self.bias

    def parameters(self):
        """Return trainable parameters."""
        return [self.weight, self.bias]

    @property
    def prunable_params(self):
        """Parameters that can be pruned (biases are excluded)."""
        return [self.weight]

    def active_count(self) -> int:
        """Number of non-masked weight entries."""
        return int(self.mask.sum())

    def total_count(self) -> int:
        """Total number of weight entries."""
        return self.mask.size

    def sparsity(self) -> float:
        """Fraction of weights that are masked out."""
        return 1.0 - self.active_count() / self.total_count()


class ReLU:
    """Rectified Linear Unit activation."""

    def __call__(self, x: Tensor) -> Tensor:
        return x.relu()

    def parameters(self):
        return []


class Sigmoid:
    """Logistic sigmoid activation."""

    def __call__(self, x: Tensor) -> Tensor:
        return x.sigmoid()

    def parameters(self):
        return []


class GELU:
    """Gaussian Error Linear Unit activation."""

    def __call__(self, x: Tensor) -> Tensor:
        return x.gelu()

    def parameters(self):
        return []


class Sequential:
    """Runs layers in sequence."""

    def __init__(self, *layers):
        self.layers = list(layers)

    def __call__(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        """Collect parameters from all layers."""
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params

    def linear_layers(self):
        """Return only the Linear layers (useful for pruning)."""
        return [l for l in self.layers if isinstance(l, Linear)]


def make_mlp(layer_sizes, activation="relu"):
    """Build an MLP from a list of layer widths.

    Inserts an activation after every hidden layer but not after the output.

    Args:
        layer_sizes: e.g. [784, 256, 128, 10] for a 3-layer MLP.
        activation: one of 'relu', 'sigmoid', 'gelu'.
    """
    act_cls = {"relu": ReLU, "sigmoid": Sigmoid, "gelu": GELU}[activation]
    layers = []
    for i in range(len(layer_sizes) - 1):
        layers.append(Linear(layer_sizes[i], layer_sizes[i + 1]))
        if i < len(layer_sizes) - 2:
            layers.append(act_cls())
    return Sequential(*layers)
