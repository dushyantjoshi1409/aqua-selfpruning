# Design Document

## 1. Importance Criterion — Derivation

### Why magnitude-only pruning is insufficient

Magnitude pruning removes weights with the smallest |w|.  The implicit assumption is that small weights contribute little to the output.  This ignores gradient information: a small weight with a large gradient is being actively used by the optimizer and removing it hurts, while a large weight with a near-zero gradient is a dead connection that could be safely pruned.

### First-order Taylor expansion of the loss change

To estimate how much the loss changes when we remove connection (i, j) — that is, set w_ij to zero — we use a first-order Taylor expansion around the current parameter values:

    ΔL = L(W with w_ij = 0) − L(W)
       ≈ −w_ij · ∂L/∂w_ij

This follows from treating "removal" as a perturbation δ = −w_ij and applying:

    L(W + δe_ij) ≈ L(W) + δ · ∂L/∂w_ij = L(W) − w_ij · ∂L/∂w_ij

The importance score is the absolute loss change:

    I(w_ij) = |w_ij · ∂L/∂w_ij|

This is the *saliency* criterion.  It subsumes magnitude (|w|) by also accounting for gradient magnitude (|∂L/∂w|), and it costs nothing extra because the gradient is already computed during backpropagation.

### Stabilizing the estimate

A single mini-batch gradient is noisy.  We use an exponential moving average of gradient magnitudes across batches (EMA with β = 0.9), which smooths the importance estimate without adding memory overhead beyond one array per layer.

### Why the criterion works

A connection scores high when *both* conditions hold:
1. The weight is large (|w| is big) — the connection has capacity.
2. The gradient is large (|∂L/∂w| is big) — the capacity is being used.

Connections where either factor is small can be removed with minimal loss increase.  This is provably a better heuristic than magnitude alone because it captures the curvature of the loss landscape near the current point, not just the position.

## 2. The Gradient of a Masked Weight

### Setup

In our implementation, when a weight w_ij is pruned (mask_ij = 0), the forward pass enforces `weight.data *= mask` so the weight is exactly zero.  The question is: what does `backward()` compute as the gradient at that position?

### Answer

For a linear layer `output = input @ W`, the gradient is:

    ∂L/∂W = input^T @ ∂L/∂output

This expression does **not depend on W's value**.  Even when W[i,j] = 0 (masked), the gradient ∂L/∂W[i,j] is generally non-zero.  It represents "how much the loss would decrease per unit increase in this weight," which is exactly the information we need for:

1. **Importance scoring**: The saliency score |w · ∂L/∂w| uses the gradient directly.
2. **Regrowth decisions**: A pruned weight with a large |∂L/∂w| is a strong candidate for reactivation.

### Why this is the right choice

Some implementations zero out the gradient for masked weights during backward.  This is mathematically wrong for a matrix multiplication (the derivative does not depend on the weight value) and practically harmful because it destroys the information needed for regrowth.

We zero the gradient **before the optimizer step**, not during backward.  This means:
- The full gradient is available for importance scoring and regrowth.
- The optimizer does not accumulate moments for pruned weights.
- Masked weights stay at exactly zero.

## 3. Optimizer State Corruption — The Bug Most Implementations Ship

### The failure mode

When a weight is pruned, its gradient becomes zero (we zero it before the optimizer).  Adam then decays the moment estimates:

    m ← β₁ · m + (1 − β₁) · 0 = β₁ · m     (decays fast, β₁ = 0.9)
    v ← β₂ · v + (1 − β₂) · 0 = β₂ · v     (decays slow, β₂ = 0.999)

After 100 steps of pruning:
- m ≈ 0.9^100 · m₀ ≈ 2.66 × 10⁻⁵ · m₀  (essentially zero)
- v ≈ 0.999^100 · v₀ ≈ 0.905 · v₀  (**retains 90.5% of its pre-pruning value!**)

When the weight is revived, the first real gradient g arrives:
- m_new ≈ (1 − β₁) · g = 0.1 · g  (small numerator)
- v_new ≈ 0.999 · 0.905 · v₀ + 0.001 · g²  (large denominator, dominated by stale v₀)

The effective step is suppressed by ~10× compared to a fresh optimizer state:

    step_buggy / step_correct ≈ 0.10   (see test_masked.py for the exact numbers)

The revived weight effectively cannot learn, defeating the purpose of regrowth.

### The fix

On every optimizer step, we compare the current mask to the previous mask.  Wherever the mask changed (prune or revive), we zero out both m and v:

```python
changed = (prev_mask != mask)
if np.any(changed):
    self.m[i][changed] = 0.0
    self.v[i][changed] = 0.0
```

This ensures revived weights start with fresh optimizer state, and the bias correction terms handle the initial transient correctly.

### Test evidence

`tests/test_masked.py::test_optimizer_state_reset_on_mask_change` verifies:
- After pruning + 100 steps + revival, v is fresh (not stale).
- `test_bug_demonstration` computes the exact 9.6× step-size ratio.

## 4. Autodiff Engine Bottleneck and Optimization

### Current bottleneck

The engine creates a new Tensor object (and Python closure) for every operation, and traverses the full computation graph on every `backward()` call.  For a typical forward pass of an MLP with 3 layers:
- ~12 Tensor allocations (matmul, add, relu × 3 + loss)
- ~12 closure allocations
- Topological sort via recursive DFS (Python function call overhead)

The actual NumPy computations (matmul, additions) are O(n²) or O(n³) and dominate for large tensors, but for small networks the Python overhead of graph construction and traversal is significant.

### Optimization opportunities

1. **Fused operations**: Combine `linear + bias + relu` into a single graph node, reducing Tensor allocations from 3 to 1 per layer.  This is what PyTorch's `torch.nn.functional` layer does.

2. **Graph caching**: If the network architecture does not change between iterations, reuse the topological order instead of recomputing it.

3. **In-place accumulation**: Use `np.add(target, source, out=target)` instead of `target = target + source` to avoid temporary array allocations during gradient accumulation.

4. **Compiled backward**: Generate a sequence of NumPy calls from the graph at trace time and replay it, eliminating the per-node Python dispatch overhead.  This is analogous to PyTorch's `torch.compile`.

5. **Sparse backward**: Skip gradient computation for pruned nodes entirely, which would make backward cost proportional to the active parameter count rather than the total.

## 5. Serving a Self-Pruned Model at Scale

### Multi-tenant inference service architecture

For a real multi-tenant inference service at thousands of requests per second:

**Model format**: Convert the pruned model to a sparse format (CSR/CSC for weight matrices) at export time.  The training-time binary mask and dense weights become sparse weight tensors with only the non-zero entries stored.

**Inference runtime**: Use a runtime that supports sparse matrix operations natively:
- NVIDIA sparse tensor cores (2:4 structured sparsity on A100/H100)
- cuSPARSE for unstructured sparsity
- ONNX Runtime with sparse operator support
- For CPU: Intel MKL sparse BLAS, or hand-vectorized CSR-times-dense

**Batching**: Dynamic batching (as in Triton Inference Server) collects requests from multiple tenants into a single batch, amortizing the kernel launch overhead.  The sparse model benefits proportionally more from batching because sparse operations have higher per-element overhead that is amortized.

**Tenant isolation**: Each tenant may have a differently-pruned version of the base model (different sparsity patterns).  Store per-tenant sparse overlays: a shared dense base plus tenant-specific sparse delta masks.  Inference computes `base + delta` using only the delta's non-zero entries.

**Memory**: A 90% sparse model stores ~10% of the parameters, enabling more model replicas per GPU.  For a model that was 2GB dense, the sparse version is ~200MB, fitting 10 replicas where 2 would have fit before.

**Latency**: The theoretical 10× FLOP reduction translates to 3–5× wall-clock speedup on current hardware (limited by memory bandwidth, not compute, for typical MLP sizes).  For transformer-style models, the attention mechanism dominates, so weight pruning benefits the FFN layers disproportionately.

**Caching**: Sparsity patterns are static after training, so the sparse matrix layout can be precomputed and pinned in GPU memory.  No runtime reindexing is needed.

**Monitoring**: Track per-request latency and per-tenant accuracy.  A pruned model that degrades for a specific tenant's input distribution should be detected by an online A/B test, not by a user complaint.
