"""Reverse-mode automatic differentiation engine.

Provides a Tensor type wrapping NumPy arrays that tracks computation history
and supports gradient computation via reverse-mode autodiff (backpropagation).
All operations build a DAG; calling backward() on a scalar output traverses it
in reverse topological order to accumulate gradients.

The engine correctly handles broadcasting in gradient accumulation and remains
correct when weight masks change between training steps (see cross_entropy_loss
and the masked-weight tests for details).
"""

import numpy as np
from typing import Tuple, Optional


def _reduce_grad(grad: np.ndarray, shape: Tuple[int, ...]) -> np.ndarray:
    """Sum-reduce a gradient array to match the original tensor's shape.

    When NumPy broadcasts (a, b) -> c during a forward op, the backward
    gradient has shape c but must be accumulated into shapes a and b.
    This function sums over the axes that were broadcast-expanded.
    """
    # Sum over extra leading dimensions added by broadcasting
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    # Sum over dimensions that were size-1 in the original
    for axis, s in enumerate(shape):
        if s == 1 and grad.shape[axis] != 1:
            grad = grad.sum(axis=axis, keepdims=True)
    return grad


class Tensor:
    """Wraps a NumPy array with computation-graph tracking for autodiff.

    Attributes:
        data: The underlying NumPy array (float64).
        grad: Accumulated gradient after backward(), or None.
        requires_grad: Whether this tensor participates in gradient computation.
    """

    __slots__ = ("data", "grad", "requires_grad", "_parents", "_backward_fn", "_op")

    def __init__(self, data, requires_grad=False, _parents=(), _op=""):
        if isinstance(data, Tensor):
            data = data.data
        self.data = np.asarray(data, dtype=np.float64)
        self.requires_grad = requires_grad
        self.grad: Optional[np.ndarray] = None
        self._parents = _parents
        self._backward_fn = lambda: None
        self._op = _op

    # ---- properties -----------------------------------------------------------

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def size(self):
        return self.data.size

    def item(self):
        """Return the scalar value (only valid for size-1 tensors)."""
        return self.data.item()

    def numpy(self):
        """Return a copy of the underlying array."""
        return self.data.copy()

    def detach(self):
        """Return a new Tensor sharing data but detached from the graph."""
        return Tensor(self.data.copy())

    def zero_grad(self):
        """Reset the gradient to None."""
        self.grad = None

    # ---- gradient accumulation ------------------------------------------------

    def _acc(self, g: np.ndarray):
        """Accumulate gradient *g* with correct broadcast reduction."""
        g = _reduce_grad(g, self.shape)
        if self.grad is None:
            self.grad = g.copy()
        else:
            self.grad = self.grad + g

    # ---- backward pass --------------------------------------------------------

    def backward(self):
        """Run reverse-mode autodiff from this (scalar) tensor.

        Builds a topological ordering of the computation graph and propagates
        gradients from output to inputs.  Gradients are *accumulated*, so
        calling backward() twice without zeroing grad doubles the values.
        """
        topo = []
        visited = set()

        def _build(t):
            tid = id(t)
            if tid not in visited:
                visited.add(tid)
                for p in t._parents:
                    _build(p)
                topo.append(t)

        _build(self)

        self.grad = np.ones_like(self.data)
        for t in reversed(topo):
            t._backward_fn()

    # ---- element-wise arithmetic ----------------------------------------------

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(
            self.data + other.data,
            self.requires_grad or other.requires_grad,
            (self, other),
            "+",
        )

        def _bw():
            if self.requires_grad:
                self._acc(out.grad)
            if other.requires_grad:
                other._acc(out.grad)

        out._backward_fn = _bw
        return out

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(
            self.data - other.data,
            self.requires_grad or other.requires_grad,
            (self, other),
            "-",
        )

        def _bw():
            if self.requires_grad:
                self._acc(out.grad)
            if other.requires_grad:
                other._acc(-out.grad)

        out._backward_fn = _bw
        return out

    def __rsub__(self, other):
        return Tensor(other).__sub__(self) if not isinstance(other, Tensor) else other.__sub__(self)

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(
            self.data * other.data,
            self.requires_grad or other.requires_grad,
            (self, other),
            "*",
        )

        def _bw():
            if self.requires_grad:
                self._acc(out.grad * other.data)
            if other.requires_grad:
                other._acc(out.grad * self.data)

        out._backward_fn = _bw
        return out

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(
            self.data / other.data,
            self.requires_grad or other.requires_grad,
            (self, other),
            "/",
        )

        def _bw():
            if self.requires_grad:
                self._acc(out.grad / other.data)
            if other.requires_grad:
                other._acc(-out.grad * self.data / (other.data ** 2))

        out._backward_fn = _bw
        return out

    def __rtruediv__(self, other):
        return Tensor(other).__truediv__(self) if not isinstance(other, Tensor) else other.__truediv__(self)

    def __neg__(self):
        return Tensor(0.0) - self

    def __pow__(self, exponent):
        """Raise to a constant (int/float) power."""
        assert isinstance(exponent, (int, float))
        out = Tensor(
            self.data ** exponent,
            self.requires_grad,
            (self,),
            f"**{exponent}",
        )

        def _bw():
            if self.requires_grad:
                self._acc(out.grad * exponent * self.data ** (exponent - 1))

        out._backward_fn = _bw
        return out

    # ---- matrix operations ----------------------------------------------------

    def matmul(self, other):
        """Matrix multiplication (2-D only)."""
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(
            self.data @ other.data,
            self.requires_grad or other.requires_grad,
            (self, other),
            "@",
        )

        def _bw():
            if self.requires_grad:
                self._acc(out.grad @ other.data.T)
            if other.requires_grad:
                other._acc(self.data.T @ out.grad)

        out._backward_fn = _bw
        return out

    def __matmul__(self, other):
        return self.matmul(other)

    # ---- reductions -----------------------------------------------------------

    def sum(self, axis=None, keepdims=False):
        """Sum reduction."""
        out = Tensor(
            self.data.sum(axis=axis, keepdims=keepdims),
            self.requires_grad,
            (self,),
            "sum",
        )

        def _bw():
            if self.requires_grad:
                g = out.grad
                if axis is not None and not keepdims:
                    ax = (axis,) if isinstance(axis, int) else tuple(axis)
                    for a in sorted(ax):
                        g = np.expand_dims(g, a)
                self._acc(np.broadcast_to(g, self.shape) * np.ones_like(self.data))

        out._backward_fn = _bw
        return out

    def mean(self, axis=None, keepdims=False):
        """Mean reduction."""
        if axis is None:
            n = self.data.size
        elif isinstance(axis, int):
            n = self.data.shape[axis]
        else:
            n = int(np.prod([self.data.shape[a] for a in axis]))

        out = Tensor(
            self.data.mean(axis=axis, keepdims=keepdims),
            self.requires_grad,
            (self,),
            "mean",
        )

        def _bw():
            if self.requires_grad:
                g = out.grad / n
                if axis is not None and not keepdims:
                    ax = (axis,) if isinstance(axis, int) else tuple(axis)
                    for a in sorted(ax):
                        g = np.expand_dims(g, a)
                self._acc(np.broadcast_to(g, self.shape) * np.ones_like(self.data))

        out._backward_fn = _bw
        return out

    # ---- activations ----------------------------------------------------------

    def relu(self):
        """Rectified Linear Unit."""
        mask = (self.data > 0).astype(np.float64)
        out = Tensor(self.data * mask, self.requires_grad, (self,), "relu")

        def _bw():
            if self.requires_grad:
                self._acc(out.grad * mask)

        out._backward_fn = _bw
        return out

    def sigmoid(self):
        """Logistic sigmoid with clipping for numerical safety."""
        s = 1.0 / (1.0 + np.exp(-np.clip(self.data, -500, 500)))
        out = Tensor(s, self.requires_grad, (self,), "sigmoid")

        def _bw():
            if self.requires_grad:
                self._acc(out.grad * s * (1.0 - s))

        out._backward_fn = _bw
        return out

    def tanh_act(self):
        """Hyperbolic tangent activation."""
        t = np.tanh(self.data)
        out = Tensor(t, self.requires_grad, (self,), "tanh")

        def _bw():
            if self.requires_grad:
                self._acc(out.grad * (1.0 - t ** 2))

        out._backward_fn = _bw
        return out

    def gelu(self):
        """Gaussian Error Linear Unit (tanh approximation)."""
        x = self.data
        k = np.sqrt(2.0 / np.pi)
        inner = k * (x + 0.044715 * x ** 3)
        t = np.tanh(inner)
        y = 0.5 * x * (1.0 + t)
        out = Tensor(y, self.requires_grad, (self,), "gelu")

        def _bw():
            if self.requires_grad:
                sech2 = 1.0 - t ** 2
                d_inner = k * (1.0 + 3.0 * 0.044715 * x ** 2)
                local_grad = 0.5 * (1.0 + t) + 0.5 * x * sech2 * d_inner
                self._acc(out.grad * local_grad)

        out._backward_fn = _bw
        return out

    # ---- element-wise math ----------------------------------------------------

    def log(self):
        """Natural logarithm (clamped input for safety)."""
        safe = np.clip(self.data, 1e-12, None)
        out = Tensor(np.log(safe), self.requires_grad, (self,), "log")

        def _bw():
            if self.requires_grad:
                self._acc(out.grad / safe)

        out._backward_fn = _bw
        return out

    def exp(self):
        """Exponential (clamped input for safety)."""
        e = np.exp(np.clip(self.data, -500, 500))
        out = Tensor(e, self.requires_grad, (self,), "exp")

        def _bw():
            if self.requires_grad:
                self._acc(out.grad * e)

        out._backward_fn = _bw
        return out

    # ---- shape manipulation ---------------------------------------------------

    def reshape(self, *shape):
        """Reshape tensor."""
        out = Tensor(self.data.reshape(*shape), self.requires_grad, (self,), "reshape")

        def _bw():
            if self.requires_grad:
                self._acc(out.grad.reshape(self.shape))

        out._backward_fn = _bw
        return out

    def transpose(self):
        """2-D transpose."""
        out = Tensor(self.data.T, self.requires_grad, (self,), "T")

        def _bw():
            if self.requires_grad:
                self._acc(out.grad.T)

        out._backward_fn = _bw
        return out

    @property
    def T(self):
        """Convenience property for transpose."""
        return self.transpose()

    # ---- dunder helpers -------------------------------------------------------

    def __repr__(self):
        return f"Tensor(shape={self.shape}, grad={self.requires_grad})"

    def __len__(self):
        return len(self.data)


# ---------------------------------------------------------------------------
# Numerically stable softmax + cross-entropy (combined for stability)
# ---------------------------------------------------------------------------

def cross_entropy_loss(logits, targets):
    """Numerically stable softmax cross-entropy loss.

    Combines log-softmax and negative log-likelihood into a single operation
    to avoid computing exp(x) explicitly, which would overflow for large logits.

    The gradient is the simple, well-known expression:
        dL/d(logits) = (softmax(logits) - one_hot(targets)) / N

    Args:
        logits: Tensor of shape (N, C) — raw (pre-softmax) class scores.
        targets: NumPy int array of shape (N,) — ground-truth class indices.

    Returns:
        Scalar Tensor — mean cross-entropy loss over the batch.
    """
    N = logits.shape[0]
    # log-softmax via the log-sum-exp trick
    max_logits = logits.data.max(axis=1, keepdims=True)
    shifted = logits.data - max_logits
    log_sum_exp = np.log(np.sum(np.exp(shifted), axis=1, keepdims=True))
    log_probs = shifted - log_sum_exp                   # (N, C)

    nll = -log_probs[np.arange(N), targets].mean()      # scalar

    out = Tensor(nll, logits.requires_grad, (logits,), "ce")
    probs = np.exp(log_probs)                            # softmax probs

    def _bw():
        if logits.requires_grad:
            g = probs.copy()
            g[np.arange(N), targets] -= 1.0
            g /= N
            logits._acc(out.grad * g)

    out._backward_fn = _bw
    return out
