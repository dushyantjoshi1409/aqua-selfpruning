# The Self-Pruning Network

A neural network that prunes itself during training, built from scratch in pure Python and NumPy.  Implements reverse-mode automatic differentiation, Adam with mask-aware state management, and Taylor-saliency-based gradual pruning with a cubic schedule.

Scikit-learn is used **only** to load the digits dataset — not for training, models, or gradient computation.

## Quick Start

```bash
pip install -r requirements.txt

# Run everything (tests + training + sweep):
python run_all.py

# Or run just the tests (< 10 seconds):
python run_all.py --quick
```

## Run Commands (individual parts)

### Part 1: Gradient Checks

```bash
python -m tests.test_gradients    # 34 numerical gradient checks across 17 ops
python -m tests.test_masked       # masked-weight correctness + optimizer state tests
```

### Part 2: Basic MLP Training

```bash
python -m train.train_basic       # 64→128→64→10 MLP on digits, Adam
```

Saves learning curves to `results/learning_curve.png`.

### Part 3: Self-Pruning Training

```bash
python -m train.train_prune                              # saliency, 90% sparsity
python -m train.train_prune --regrowth --sparsity 0.9    # with dynamic regrowth
python -m train.train_prune --sparsity 0.95 --method magnitude
```

Saves training curves to `results/pruning_training.png`.

### Part 4: Pareto Sweep

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
  layers.py          Linear layer (with mask), activations, Sequential, MLP builder
  optim.py           Adam and SGD with mask-aware state management
prune/
  pruner.py          Importance criteria, cubic schedule, GradualPruner with regrowth
train/
  data.py            Dataset loading (sklearn for data only)
  train_basic.py     Part 2: basic MLP training
  train_prune.py     Part 3: self-pruning training with optional regrowth
  pareto_sweep.py    Part 4: Pareto sweep and evidence generation
tests/
  test_gradients.py  Numerical gradient checking for all operations
  test_masked.py     Masked-weight, optimizer state, and stale-v bug tests
results/             Committed plots and raw data
DESIGN.md            Derivations, theory, and design reasoning
run_all.py           One-command full reproducibility
```

## Key Design Decisions

1. **Importance criterion**: Taylor saliency `|w · ∂L/∂w|` — first-order approximation of loss change from removing a connection, with connections to Optimal Brain Damage (LeCun et al., 1989) and the Fisher Information Matrix.  Outperforms magnitude-only by 1.1pp at 90% sparsity and 1.7pp at 95%.

2. **Masked-weight gradients**: Gradients flow normally through masked positions because `∂(x @ W)/∂W = x^T @ grad_output` does not depend on `W`'s values.  Gradients are zeroed before the optimizer step, not during backward — preserving the signal needed for importance scoring and regrowth.

3. **Optimizer state reset on mask transition**: When a weight's mask changes (active↔pruned), Adam's moment estimates `(m, v)` are reset to zero.  Without this, stale `v` from before pruning suppresses the learning rate of revived weights by ~10× (see `test_bug_demonstration` for exact numbers).

4. **Dynamic regrowth**: Pruned connections with high gradient magnitude can be reactivated, allowing the network to explore alternative sparse topologies during training.  Related to RigL (Evci et al., 2020) and Sparse Evolutionary Training (Mocanu et al., 2018).

5. **Cubic sparsity schedule** (Zhu & Gupta, 2017): Prunes aggressively early when redundancy is highest, then slows near the target.  Better than linear or step schedules because the network has more training budget to recover from early pruning.

See [DESIGN.md](DESIGN.md) for full derivations, second-order analysis, and the multi-tenant serving architecture.

## Reproducibility

All random seeds are fixed.  Running the commands above from a clean clone reproduces the reported numbers within floating-point variance.  `run_all.py` exercises the full pipeline end-to-end.

## Results Summary

| Sparsity | Saliency Acc | Magnitude Acc | Dense Baseline | FLOPs Ratio |
|----------|-------------|---------------|----------------|-------------|
| 0%       | 0.981 ± 0.004 | 0.981 ± 0.004 | 0.981 ± 0.004 | 1.000 |
| 50%      | 0.977 ± 0.004 | 0.978 ± 0.006 | — | 0.500 |
| 75%      | 0.973 ± 0.005 | 0.974 ± 0.007 | — | 0.250 |
| 90%      | 0.964 ± 0.004 | 0.953 ± 0.010 | — | 0.100 |
| 95%      | 0.949 ± 0.007 | 0.932 ± 0.010 | — | 0.050 |

At 90% sparsity: **96.4% accuracy retained, 10× FLOP reduction, saliency beats magnitude by 1.1pp.**
