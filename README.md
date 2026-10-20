# The Self-Pruning Network

A neural network that prunes itself during training, built from scratch in pure Python and NumPy.  Implements reverse-mode automatic differentiation, Adam with mask-aware state management, and Taylor-saliency-based gradual pruning with a cubic schedule.

Scikit-learn is used **only** to load the digits dataset — not for training, models, or gradient computation.

## Quick Start

```bash
# Install dependencies (Python 3.10+)
pip install numpy matplotlib scikit-learn

# Or with uv:
uv pip install numpy matplotlib scikit-learn
```

## Run Commands

### Gradient-check tests (Part 1)

```bash
python -m tests.test_gradients
```

Verifies all 17 differentiable operations against finite-difference numerical gradients.

### Masked-weight correctness tests (Part 1)

```bash
python -m tests.test_masked
```

Tests that masked weights contribute zero to forward pass, gradients are available for importance scoring, the optimizer skips masked weights, and optimizer state resets on mask transitions.

### Basic training (Part 2)

```bash
python -m train.train_basic
```

Trains a 64→128→64→10 MLP on the digits dataset with Adam.  Saves learning curves to `results/learning_curve.png`.

### Self-pruning training (Part 3)

```bash
python -m train.train_prune
```

Trains the same MLP with gradual saliency-based pruning to 90% sparsity.  Saves training curves to `results/pruning_training.png`.

Options:
```bash
python -m train.train_prune --sparsity 0.95 --method magnitude --regrowth
```

### Pareto sweep (Part 4)

```bash
python -m train.pareto_sweep
```

Sweeps 5 sparsity levels × 2 pruning methods × 5 seeds (50 runs total).  Generates:
- `results/pareto_curve.png` — sparsity–accuracy Pareto curve with error bars
- `results/flops_comparison.png` — FLOP reduction chart
- `results/pareto_data.csv` — raw numbers for all 50 runs
- `results/wall_clock.txt` — dense vs sparse-aware forward timing
- `results/falsifiable_claim.txt` — the quantitative claim backed by the data

## Project Structure

```
engine/
  tensor.py          Reverse-mode autodiff engine (Tensor class, all ops, cross-entropy)
nn/
  layers.py          Linear layer (with mask), ReLU, Sigmoid, GELU, Sequential, MLP builder
  optim.py           Adam and SGD with mask-aware state management
prune/
  pruner.py          Importance criteria, cubic schedule, GradualPruner with regrowth
train/
  data.py            Dataset loading (sklearn for data only)
  train_basic.py     Part 2: basic MLP training
  train_prune.py     Part 3: self-pruning training
  pareto_sweep.py    Part 4: Pareto sweep and evidence generation
tests/
  test_gradients.py  Gradient checking for all operations
  test_masked.py     Masked-weight and optimizer state correctness
results/             Committed plots and raw data
DESIGN.md            Derivations and design reasoning
```

## Key Design Decisions

1. **Importance criterion**: Taylor saliency |w · ∂L/∂w| — first-order approximation of loss change from removing a connection.  Outperforms magnitude-only by 1.1 percentage points at 90% sparsity.

2. **Masked-weight gradients**: Gradients are computed normally for all positions (the matmul derivative does not depend on weight values), then zeroed before the optimizer step.  This preserves importance information for regrowth while preventing optimizer state corruption.

3. **Optimizer state reset**: When a weight's mask transitions (active↔pruned), Adam's moment estimates (m, v) are reset to zero.  Without this, stale v from before pruning suppresses the effective learning rate of revived weights by ~10×.

4. **Weight initialization**: Kaiming He (scale = √(2/fan_in)) for networks with ReLU activations.

See [DESIGN.md](DESIGN.md) for full derivations.

## Reproducibility

All random seeds are fixed.  Running the commands above from a clean clone should reproduce the reported numbers within floating-point variance.

## Results Summary

| Sparsity | Saliency Acc | Magnitude Acc | Dense Baseline | FLOPs Ratio |
|----------|-------------|---------------|----------------|-------------|
| 0%       | 0.981 ± 0.004 | 0.981 ± 0.004 | 0.981 ± 0.004 | 1.000 |
| 50%      | 0.977 ± 0.004 | 0.978 ± 0.006 | — | 0.500 |
| 75%      | 0.973 ± 0.005 | 0.974 ± 0.007 | — | 0.250 |
| 90%      | 0.964 ± 0.004 | 0.953 ± 0.010 | — | 0.100 |
| 95%      | 0.949 ± 0.007 | 0.932 ± 0.010 | — | 0.050 |
