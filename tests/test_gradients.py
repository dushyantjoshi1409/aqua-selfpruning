"""Gradient-checking suite for every differentiable operation.

For each operation, computes the analytical gradient via backward() and
compares it against a finite-difference numerical gradient.  The relative
error must be below a tolerance (default 1e-5) for the test to pass.

Usage:
    python -m tests.test_gradients
"""

import numpy as np
from engine.tensor import Tensor, cross_entropy_loss


def numerical_gradient(make_loss_fn, x, eps=1e-5):
    """Central-difference numerical gradient of a scalar loss w.r.t. x.data.

    Args:
        make_loss_fn: callable() -> scalar Tensor loss (must rebuild graph).
        x: Tensor whose .data will be perturbed.
        eps: finite-difference step size.

    Returns:
        NumPy array of the same shape as x.data.
    """
    grad = np.zeros_like(x.data)
    it = np.nditer(x.data, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        old = x.data[idx]

        x.data[idx] = old + eps
        loss_plus = make_loss_fn().data.copy().sum()

        x.data[idx] = old - eps
        loss_minus = make_loss_fn().data.copy().sum()

        x.data[idx] = old
        grad[idx] = (loss_plus - loss_minus) / (2.0 * eps)
        it.iternext()
    return grad


def check(name, make_loss_fn, tensor, tol=1e-5):
    """Check analytical vs numerical gradient and assert they match."""
    tensor.zero_grad()
    loss = make_loss_fn()
    loss.backward()
    analytical = tensor.grad.copy()
    num = numerical_gradient(make_loss_fn, tensor)

    denom = np.maximum(np.abs(analytical), np.abs(num)) + 1e-8
    rel_err = np.max(np.abs(analytical - num) / denom)

    status = "PASS" if rel_err < tol else "FAIL"
    print(f"  [{status}] {name:30s}  rel_err = {rel_err:.2e}")
    assert rel_err < tol, (
        f"{name}: max relative error {rel_err:.2e} exceeds tolerance {tol}"
    )


def test_addition():
    """Test gradient of element-wise addition with broadcasting."""
    a = Tensor(np.random.randn(3, 4), requires_grad=True)
    b = Tensor(np.random.randn(4), requires_grad=True)  # broadcasts (4,) -> (3,4)

    check("add a", lambda: (a + b).sum(), a)
    check("add b (broadcast)", lambda: (a + b).sum(), b)


def test_subtraction():
    """Test gradient of element-wise subtraction."""
    a = Tensor(np.random.randn(3, 4), requires_grad=True)
    b = Tensor(np.random.randn(3, 4), requires_grad=True)

    check("sub a", lambda: (a - b).sum(), a)
    check("sub b", lambda: (a - b).sum(), b)


def test_multiplication():
    """Test gradient of element-wise multiplication."""
    a = Tensor(np.random.randn(3, 4), requires_grad=True)
    b = Tensor(np.random.randn(3, 4), requires_grad=True)

    check("mul a", lambda: (a * b).sum(), a)
    check("mul b", lambda: (a * b).sum(), b)


def test_division():
    """Test gradient of element-wise division."""
    a = Tensor(np.random.randn(3, 4), requires_grad=True)
    b = Tensor(np.random.randn(3, 4) * 0.5 + 2.0, requires_grad=True)  # avoid near-zero

    check("div a", lambda: (a / b).sum(), a)
    check("div b", lambda: (a / b).sum(), b)


def test_matmul():
    """Test gradient of matrix multiplication."""
    a = Tensor(np.random.randn(5, 4), requires_grad=True)
    b = Tensor(np.random.randn(4, 3), requires_grad=True)

    check("matmul a", lambda: (a @ b).sum(), a)
    check("matmul b", lambda: (a @ b).sum(), b)


def test_sum():
    """Test gradient of sum reduction along an axis."""
    a = Tensor(np.random.randn(3, 4), requires_grad=True)

    check("sum all", lambda: a.sum(), a)
    check("sum axis=0", lambda: a.sum(axis=0).sum(), a)
    check("sum axis=1", lambda: a.sum(axis=1).sum(), a)


def test_mean():
    """Test gradient of mean reduction."""
    a = Tensor(np.random.randn(3, 4), requires_grad=True)

    check("mean all", lambda: a.mean(), a)
    check("mean axis=0", lambda: a.mean(axis=0).sum(), a)
    check("mean axis=1", lambda: a.mean(axis=1).sum(), a)


def test_relu():
    """Test gradient of ReLU activation."""
    a = Tensor(np.random.randn(4, 5), requires_grad=True)
    check("relu", lambda: a.relu().sum(), a)


def test_sigmoid():
    """Test gradient of sigmoid activation."""
    a = Tensor(np.random.randn(4, 5) * 0.5, requires_grad=True)
    check("sigmoid", lambda: a.sigmoid().sum(), a)


def test_tanh():
    """Test gradient of tanh activation."""
    a = Tensor(np.random.randn(4, 5) * 0.5, requires_grad=True)
    check("tanh", lambda: a.tanh_act().sum(), a)


def test_gelu():
    """Test gradient of GELU activation (tanh approximation)."""
    a = Tensor(np.random.randn(4, 5) * 0.5, requires_grad=True)
    check("gelu", lambda: a.gelu().sum(), a)


def test_log():
    """Test gradient of natural logarithm."""
    a = Tensor(np.abs(np.random.randn(3, 4)) + 0.1, requires_grad=True)
    check("log", lambda: a.log().sum(), a)


def test_exp():
    """Test gradient of exponential."""
    a = Tensor(np.random.randn(3, 4) * 0.5, requires_grad=True)
    check("exp", lambda: a.exp().sum(), a)


def test_power():
    """Test gradient of power operation."""
    a = Tensor(np.abs(np.random.randn(3, 4)) + 0.5, requires_grad=True)
    check("pow(2)", lambda: (a ** 2).sum(), a)
    check("pow(0.5)", lambda: (a ** 0.5).sum(), a)


def test_cross_entropy():
    """Test gradient of the combined softmax-cross-entropy loss."""
    logits = Tensor(np.random.randn(8, 5), requires_grad=True)
    targets = np.array([0, 1, 2, 3, 4, 0, 1, 2])

    check("cross_entropy", lambda: cross_entropy_loss(logits, targets), logits)


def test_broadcast_bias():
    """Test gradient through a typical layer: (N, D) @ (D, C) + (C,).

    This is the exact broadcasting pattern used in a Linear layer and
    must backpropagate correctly to both the weight and the bias.
    """
    x = Tensor(np.random.randn(8, 4), requires_grad=True)
    w = Tensor(np.random.randn(4, 3), requires_grad=True)
    b = Tensor(np.random.randn(3), requires_grad=True)

    check("layer -> x", lambda: ((x @ w) + b).sum(), x)
    check("layer -> w", lambda: ((x @ w) + b).sum(), w)
    check("layer -> b", lambda: ((x @ w) + b).sum(), b)


def test_chain():
    """Test gradient through a multi-layer computation."""
    x = Tensor(np.random.randn(4, 3), requires_grad=True)
    w1 = Tensor(np.random.randn(3, 5), requires_grad=True)
    b1 = Tensor(np.random.randn(5), requires_grad=True)
    w2 = Tensor(np.random.randn(5, 2), requires_grad=True)
    b2 = Tensor(np.random.randn(2), requires_grad=True)

    def forward():
        h = (x @ w1 + b1).relu()
        return (h @ w2 + b2).sum()

    check("chain -> x", forward, x)
    check("chain -> w1", forward, w1)
    check("chain -> b1", forward, b1)
    check("chain -> w2", forward, w2)
    check("chain -> b2", forward, b2)


def test_gradient_accumulation():
    """Test that gradients accumulate when backward is called twice."""
    a = Tensor(np.random.randn(3, 4), requires_grad=True)
    loss1 = a.sum()
    loss1.backward()
    grad1 = a.grad.copy()

    loss2 = (a * 2).sum()
    loss2.backward()
    # grad should be grad1 + 2*ones
    expected = grad1 + 2.0 * np.ones_like(a.data)
    assert np.allclose(a.grad, expected), "Gradient accumulation failed"
    print("  [PASS] gradient accumulation")


if __name__ == "__main__":
    np.random.seed(42)
    print("Gradient checks (analytical vs numerical finite-difference):\n")

    test_addition()
    test_subtraction()
    test_multiplication()
    test_division()
    test_matmul()
    test_sum()
    test_mean()
    test_relu()
    test_sigmoid()
    test_tanh()
    test_gelu()
    test_log()
    test_exp()
    test_power()
    test_cross_entropy()
    test_broadcast_bias()
    test_chain()
    test_gradient_accumulation()

    print("\nAll gradient checks passed!")
