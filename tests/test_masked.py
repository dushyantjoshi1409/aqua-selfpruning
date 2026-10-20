"""Tests for masked-weight correctness and optimizer state management.

These tests verify the critical correctness requirement: the autodiff engine
and optimizer remain correct when parts of the parameter tensors are masked
out between training steps.

Key properties tested:

1. Masked weights contribute exactly zero to the forward pass.
2. The autodiff engine computes meaningful gradients even for zeroed
   (masked-out) weight positions, because the matmul gradient d(x@W)/dW
   does not depend on W.
3. The optimizer does not update masked weights.
4. When a weight's mask transitions (prune or revive), the optimizer resets
   that weight's moment estimates (m, v) so stale state does not corrupt
   the learning dynamics of revived connections.

Usage:
    python -m tests.test_masked
"""

import numpy as np
from engine.tensor import Tensor, cross_entropy_loss
from nn.layers import Linear
from nn.optim import Adam


def test_masked_forward_is_zero():
    """Masked weights produce exactly zero contribution to the output."""
    np.random.seed(42)
    layer = Linear(4, 3)

    # Record output with all weights active
    x = Tensor(np.random.randn(2, 4))
    out_full = layer(x).data.copy()

    # Mask out first row of weights
    layer.mask[0, :] = 0.0
    out_masked = layer(x).data.copy()

    # The outputs must differ (unless weights were coincidentally zero)
    assert not np.allclose(out_full, out_masked), \
        "Masking should change the output"

    # The masked weight entries must be exactly zero
    assert np.all(layer.weight.data[0, :] == 0.0), \
        "Masked weight positions must be exactly zero"

    print("[PASS] Masked weights contribute zero to forward pass")


def test_gradient_nonzero_for_masked():
    """Gradients are computed for masked (zeroed) weight positions.

    Because d(x @ W)/dW = x^T @ grad_output, the gradient does not depend
    on the value of W.  Even when W[i,j] = 0 (masked), the gradient at
    that position is generally non-zero.  This is essential for importance
    scoring and regrowth: we need to know how much the loss would benefit
    from reactivating a pruned connection.
    """
    np.random.seed(42)
    layer = Linear(4, 3)

    # Mask some weights and zero them
    layer.mask[0, :] = 0.0
    layer.mask[1, 1] = 0.0
    layer.weight.data *= layer.mask

    x = Tensor(np.random.randn(5, 4))
    out = layer(x)
    loss = out.sum()
    loss.backward()

    # Check that masked positions have non-zero gradients
    grad_at_masked = layer.weight.grad[layer.mask == 0]
    assert np.any(grad_at_masked != 0), \
        "Masked positions should have non-zero gradients (for importance scoring)"

    print("[PASS] Gradients available for masked positions (importance scoring)")


def test_optimizer_skips_masked():
    """The optimizer does not update masked (pruned) weight positions."""
    np.random.seed(42)
    layer = Linear(4, 3)

    # Prune some weights
    layer.mask[0, :] = 0.0
    layer.weight.data *= layer.mask
    pruned_before = layer.weight.data[0, :].copy()

    optimizer = Adam([layer.weight, layer.bias], lr=0.01)

    # Do a forward-backward-step cycle
    x = Tensor(np.random.randn(3, 4))
    out = layer(x)
    loss = out.sum()
    optimizer.zero_grad()
    loss.backward()

    masks = {0: layer.mask}  # param index 0 = weight
    optimizer.step(masks)

    pruned_after = layer.weight.data[0, :]
    assert np.all(pruned_after == 0.0), \
        "Masked weights must remain exactly zero after optimizer step"

    print("[PASS] Optimizer does not update masked weights")


def test_optimizer_state_reset_on_mask_change():
    """Optimizer resets moment estimates (m, v) when the mask changes.

    This is the critical test that catches the most common implementation
    bug in pruning + Adam.

    The failure mode without the fix:
    - When a weight is pruned, its gradient becomes 0 (via mask zeroing).
    - Adam's first moment m decays toward 0 (fast, beta1=0.9).
    - Adam's second moment v decays toward 0 (slow, beta2=0.999).
    - After 100 steps of pruning, v retains ~90% of its pre-pruning value.
    - When the weight is revived, the stale v creates a denominator that is
      far too large relative to the fresh gradient, suppressing the effective
      learning rate by ~10x.  The revived weight cannot learn.

    The fix: reset m and v to zero when the mask transitions.
    """
    np.random.seed(42)

    # Set up a simple model with one weight
    w = Tensor(np.array([[0.5, -0.3], [0.2, 0.8]]), requires_grad=True)
    optimizer = Adam([w], lr=0.01, betas=(0.9, 0.999))

    # Phase 1: Train normally for 50 steps to build up optimizer state
    mask = np.ones_like(w.data)
    for _ in range(50):
        w.zero_grad()
        loss = (w ** 2).sum()
        loss.backward()
        optimizer.step(masks={0: mask})

    m_before_prune = optimizer.m[0].copy()
    v_before_prune = optimizer.v[0].copy()

    # Phase 2: Prune position [0,0] for 100 steps
    mask[0, 0] = 0.0
    w.data[0, 0] = 0.0

    for _ in range(100):
        w.zero_grad()
        loss = (w ** 2).sum()
        loss.backward()
        optimizer.step(masks={0: mask})

    # After pruning, m and v for [0,0] should be near zero
    # because the optimizer resets state on mask change and then
    # only sees zero gradients
    assert abs(optimizer.m[0][0, 0]) < 1e-10, \
        f"m should be ~0 for pruned position, got {optimizer.m[0][0, 0]}"
    assert abs(optimizer.v[0][0, 0]) < 1e-10, \
        f"v should be ~0 for pruned position, got {optimizer.v[0][0, 0]}"

    # Phase 3: Revive position [0,0]
    mask[0, 0] = 1.0
    w.data[0, 0] = 0.1  # small initial value

    # Record v at the revived position right after mask change
    w.zero_grad()
    loss = (w ** 2).sum()
    loss.backward()
    optimizer.step(masks={0: mask})

    m_after_revive = optimizer.m[0][0, 0]
    v_after_revive = optimizer.v[0][0, 0]

    # v should be fresh (based only on the current gradient), not stale
    # With state reset: v ≈ (1-beta2) * g^2 = 0.001 * g^2 (fresh)
    # Without reset:    v ≈ beta2^100 * v_old ≈ 0.90 * v_old (stale!)
    #
    # Check that v is small (fresh), not ~90% of its pre-pruning value
    assert v_after_revive < 0.1 * v_before_prune[0, 0], \
        (f"Stale v detected! v_revived={v_after_revive:.6f} should be much "
         f"smaller than v_pre_prune={v_before_prune[0, 0]:.6f}. "
         f"Did you reset optimizer state on mask change?")

    print("[PASS] Optimizer state correctly reset on mask change")


def test_bug_demonstration():
    """Demonstrate what goes WRONG without optimizer state reset.

    This test constructs the exact failure scenario and computes the
    step-size ratio between a correctly-reset and an incorrectly-stale
    optimizer.  It is informational (always passes) and prints the evidence.
    """
    np.random.seed(42)

    # Simulate the bug: manually track what happens without reset
    beta1, beta2, lr, eps = 0.9, 0.999, 0.01, 1e-8

    # Pre-pruning state after 50 steps
    m_pre, v_pre = 0.02, 0.001  # typical accumulated state

    # Decay during 100 steps of pruning (gradient = 0)
    m_stale = m_pre * beta1 ** 100   # ≈ 5.3e-7  (decays fast)
    v_stale = v_pre * beta2 ** 100   # ≈ 9.05e-4 (decays slowly!)

    # First gradient after revival
    g = 0.1

    # --- Without reset (the bug) ---
    m_bug = beta1 * m_stale + (1 - beta1) * g
    v_bug = beta2 * v_stale + (1 - beta2) * g ** 2
    t = 151
    m_hat_bug = m_bug / (1 - beta1 ** t)
    v_hat_bug = v_bug / (1 - beta2 ** t)
    step_bug = lr * m_hat_bug / (np.sqrt(v_hat_bug) + eps)

    # --- With reset (correct) ---
    m_fix = (1 - beta1) * g  # m was reset to 0
    v_fix = (1 - beta2) * g ** 2  # v was reset to 0
    m_hat_fix = m_fix / (1 - beta1 ** t)
    v_hat_fix = v_fix / (1 - beta2 ** t)
    step_fix = lr * m_hat_fix / (np.sqrt(v_hat_fix) + eps)

    ratio = step_fix / step_bug

    print(f"\n  Bug demonstration (informational):")
    print(f"    v_stale after 100 pruned steps: {v_stale:.6f}")
    print(f"    v_pre (before pruning):         {v_pre:.6f}")
    print(f"    v retained after pruning:       {v_stale/v_pre:.1%}")
    print(f"    Step size WITHOUT reset (buggy): {step_bug:.6f}")
    print(f"    Step size WITH reset (correct):  {step_fix:.6f}")
    print(f"    Correct/Buggy ratio:             {ratio:.1f}x")
    print(f"    Without reset, revived weights learn ~{ratio:.0f}x slower!")
    print(f"  [INFO] Bug demo complete — see DESIGN.md for full analysis")


def test_full_training_loop_with_mask_changes():
    """Integration test: train, prune, continue training, verify correctness."""
    np.random.seed(42)

    layer = Linear(4, 3)
    optimizer = Adam(layer.parameters(), lr=0.01)

    X = np.random.randn(20, 4)
    y = np.random.randint(0, 3, size=20)

    # Phase 1: Train normally for 10 steps
    for _ in range(10):
        optimizer.zero_grad()
        logits = layer(Tensor(X))
        loss = cross_entropy_loss(logits, y)
        loss.backward()
        masks = {0: layer.mask}
        optimizer.step(masks)

    acc_before = (layer(Tensor(X)).data.argmax(axis=1) == y).mean()

    # Phase 2: Prune 50% of weights
    flat_w = np.abs(layer.weight.data.ravel())
    threshold = np.median(flat_w[flat_w > 0])
    layer.mask = (np.abs(layer.weight.data) >= threshold).astype(np.float64)
    layer.weight.data *= layer.mask

    # Phase 3: Continue training for 20 steps
    for _ in range(20):
        optimizer.zero_grad()
        logits = layer(Tensor(X))
        loss = cross_entropy_loss(logits, y)
        loss.backward()
        masks = {0: layer.mask}
        optimizer.step(masks)

    # Verify masked weights are still zero
    assert np.all(layer.weight.data[layer.mask == 0] == 0.0), \
        "Masked weights must stay zero throughout training"

    # Verify active weights have been updated (not frozen)
    active_grads = layer.weight.grad[layer.mask == 1]
    # At least some active weights should have non-zero gradients
    assert np.any(active_grads != 0), \
        "Active weights should have non-zero gradients"

    print("[PASS] Full training loop with mask changes maintains correctness")


if __name__ == "__main__":
    print("Masked-weight correctness tests:\n")

    test_masked_forward_is_zero()
    test_gradient_nonzero_for_masked()
    test_optimizer_skips_masked()
    test_optimizer_state_reset_on_mask_change()
    test_bug_demonstration()
    test_full_training_loop_with_mask_changes()

    print("\nAll masked-weight tests passed!")
