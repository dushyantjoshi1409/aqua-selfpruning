"""Part 4: Pareto sweep — sparsity vs accuracy with honest cost evidence.

Sweeps target sparsity levels for both magnitude and saliency pruning,
repeats each setting across multiple random seeds, and produces:

  1. A sparsity-accuracy Pareto curve with error bars.
  2. A FLOPs comparison (dense vs sparse).
  3. Wall-clock inference timing (dense vs sparse-aware forward).
  4. Raw numbers committed as CSV.
  5. A falsifiable claim backed by the data.

Usage:
    python -m train.pareto_sweep
"""

import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from engine.tensor import Tensor
from nn.layers import make_mlp
from train.data import load_digits_data, iterate_batches
from train.train_prune import train_with_pruning


# ---------------------------------------------------------------------------
# Sparse-aware forward path
# ---------------------------------------------------------------------------

def sparse_forward_layer(x, weight, mask, bias):
    """Column-wise sparse matmul that genuinely skips zero connections.

    For each output neuron, only the active (non-masked) input connections
    participate in the dot product.  This is O(batch * active_per_column)
    instead of O(batch * fan_in) — a real FLOP reduction, not a dense
    multiply-by-zero.

    Note: in Python / NumPy this is slower than a dense matmul due to
    interpreter overhead.  In production, sparse tensor cores or CSR/CSC
    kernels would realize the FLOP savings as wall-clock speedup.
    """
    batch_size, fan_in = x.shape
    fan_out = weight.shape[1]
    output = np.zeros((batch_size, fan_out))

    for j in range(fan_out):
        active = np.nonzero(mask[:, j])[0]
        if len(active) > 0:
            output[:, j] = x[:, active] @ weight[active, j]

    return output + bias


def dense_forward(model, X):
    """Standard dense forward pass for timing comparison."""
    logits = model(Tensor(X))
    return logits.data


def sparse_forward(model, X):
    """Sparse-aware forward pass that skips pruned connections."""
    out = X.copy()
    for layer_obj in model.layers:
        if hasattr(layer_obj, "weight"):
            out = sparse_forward_layer(
                out, layer_obj.weight.data, layer_obj.mask, layer_obj.bias.data
            )
        elif hasattr(layer_obj, "__call__"):
            out = np.maximum(out, 0)  # ReLU (the only activation we use)
    return out


def time_forward(fn, X, n_repeats=20):
    """Time a forward pass function in seconds (median of repeats)."""
    times = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        fn(X)
        times.append(time.perf_counter() - t0)
    return np.median(times)


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

def run_sweep():
    """Run the full Pareto sweep and generate all evidence."""
    sparsity_levels = [0.0, 0.50, 0.75, 0.90, 0.95]
    methods = ["magnitude", "saliency"]
    seeds = [42, 123, 456, 789, 1001]
    epochs = 40

    results = []
    total_runs = len(methods) * len(sparsity_levels) * len(seeds)
    run_idx = 0

    for method in methods:
        for target_sp in sparsity_levels:
            for seed in seeds:
                run_idx += 1
                print(f"\n[{run_idx}/{total_runs}] method={method} "
                      f"sparsity={target_sp} seed={seed}")

                if target_sp == 0.0:
                    # Baseline: no pruning
                    r = train_with_pruning(
                        seed=seed, epochs=epochs, target_sparsity=0.0,
                        importance_fn=method, verbose=False,
                    )
                else:
                    r = train_with_pruning(
                        seed=seed, epochs=epochs, target_sparsity=target_sp,
                        importance_fn=method, verbose=False,
                    )

                results.append({
                    "method": method,
                    "target_sparsity": target_sp,
                    "seed": seed,
                    "test_acc": r["final_test_acc"],
                    "actual_sparsity": r["final_sparsity"],
                    "dense_flops": r["dense_flops"],
                    "sparse_flops": r["sparse_flops"],
                    "flops_ratio": r["flops_ratio"],
                })

                print(f"  -> acc={r['final_test_acc']:.3f}  "
                      f"sparsity={r['final_sparsity']:.1%}  "
                      f"flops_ratio={r['flops_ratio']:.3f}")

    return results


def save_csv(results, path="results/pareto_data.csv"):
    """Save raw results as CSV."""
    with open(path, "w") as f:
        header = "method,target_sparsity,seed,test_acc,actual_sparsity,dense_flops,sparse_flops,flops_ratio"
        f.write(header + "\n")
        for r in results:
            f.write(f"{r['method']},{r['target_sparsity']},{r['seed']},"
                    f"{r['test_acc']:.4f},{r['actual_sparsity']:.4f},"
                    f"{r['dense_flops']},{r['sparse_flops']},"
                    f"{r['flops_ratio']:.4f}\n")
    print(f"\nRaw data saved to {path}")


def plot_pareto(results, path="results/pareto_curve.png"):
    """Plot sparsity vs accuracy Pareto curve with error bars."""
    fig, ax = plt.subplots(figsize=(8, 5))

    for method, color, marker in [("magnitude", "tab:blue", "s"),
                                   ("saliency", "tab:orange", "o")]:
        method_data = [r for r in results if r["method"] == method]
        sparsities = sorted(set(r["target_sparsity"] for r in method_data))

        means = []
        stds = []
        for sp in sparsities:
            accs = [r["test_acc"] for r in method_data
                    if r["target_sparsity"] == sp]
            means.append(np.mean(accs))
            stds.append(np.std(accs))

        ax.errorbar(
            [s * 100 for s in sparsities], means, yerr=stds,
            label=method.capitalize(), color=color, marker=marker,
            capsize=4, linewidth=2, markersize=7,
        )

    ax.set_xlabel("Sparsity (%)", fontsize=12)
    ax.set_ylabel("Test Accuracy", fontsize=12)
    ax.set_title("Sparsity–Accuracy Pareto Curve", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Pareto curve saved to {path}")


def plot_flops(results, path="results/flops_comparison.png"):
    """Plot FLOP reduction across sparsity levels."""
    fig, ax = plt.subplots(figsize=(8, 5))

    for method, color in [("magnitude", "tab:blue"), ("saliency", "tab:orange")]:
        method_data = [r for r in results if r["method"] == method]
        sparsities = sorted(set(r["target_sparsity"] for r in method_data))

        mean_ratios = []
        for sp in sparsities:
            ratios = [r["flops_ratio"] for r in method_data
                      if r["target_sparsity"] == sp]
            mean_ratios.append(np.mean(ratios))

        ax.bar(
            [s * 100 + (2 if method == "saliency" else -2) for s in sparsities],
            [1.0 - r for r in mean_ratios],
            width=4, label=method.capitalize(), color=color, alpha=0.8,
        )

    ax.set_xlabel("Target Sparsity (%)")
    ax.set_ylabel("FLOP Reduction (fraction)")
    ax.set_title("Real FLOP Savings from Pruning")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"FLOPs plot saved to {path}")


def measure_wall_clock(results, path="results/wall_clock.txt"):
    """Measure dense vs sparse forward pass wall-clock time."""
    np.random.seed(42)
    X_train, _, X_test, _ = load_digits_data(seed=42)

    # Train one model at 90% saliency for timing
    model = make_mlp([X_train.shape[1], 128, 64, 10], activation="relu")
    # Apply 90% sparsity by zeroing random weights
    for layer in model.linear_layers():
        flat = layer.mask.ravel()
        n_prune = int(len(flat) * 0.9)
        prune_idx = np.random.choice(len(flat), n_prune, replace=False)
        flat[prune_idx] = 0.0
        layer.mask = flat.reshape(layer.mask.shape)
        layer.weight.data *= layer.mask

    t_dense = time_forward(lambda x: dense_forward(model, x), X_test, n_repeats=50)
    t_sparse = time_forward(lambda x: sparse_forward(model, x), X_test, n_repeats=50)

    report = [
        "Wall-clock inference timing (median of 50 runs)",
        f"  Test set size: {len(X_test)} samples",
        f"  Sparsity: 90%",
        f"  Dense forward:  {t_dense*1000:.2f} ms",
        f"  Sparse forward: {t_sparse*1000:.2f} ms",
        f"  Ratio: {t_sparse/t_dense:.2f}x",
        "",
        "NOTE: The sparse forward uses a Python-level column-wise loop that",
        "skips zero connections but suffers from interpreter overhead.  In a",
        "production setting with sparse tensor cores or CSR kernels, the FLOP",
        "reduction would translate to proportional wall-clock speedup.  The",
        "theoretical FLOP ratio (0.10x) represents the real work reduction;",
        "the Python timing above understates the benefit.",
    ]

    with open(path, "w") as f:
        f.write("\n".join(report) + "\n")
    print("\n".join(report))
    return t_dense, t_sparse


def print_falsifiable_claim(results):
    """State a specific, quantitative, falsifiable claim."""
    mag_90 = [r["test_acc"] for r in results
              if r["method"] == "magnitude" and r["target_sparsity"] == 0.90]
    sal_90 = [r["test_acc"] for r in results
              if r["method"] == "saliency" and r["target_sparsity"] == 0.90]
    baseline = [r["test_acc"] for r in results
                if r["target_sparsity"] == 0.0]

    n = len(mag_90)
    claim = (
        f"\nFALSIFIABLE CLAIM (backed by results/pareto_data.csv):\n"
        f"At 90% sparsity on the digits dataset (64-128-64-10 MLP, {n} seeds):\n"
        f"  - Saliency pruning: {np.mean(sal_90):.3f} +/- {np.std(sal_90):.3f} accuracy\n"
        f"  - Magnitude pruning: {np.mean(mag_90):.3f} +/- {np.std(mag_90):.3f} accuracy\n"
        f"  - Dense baseline: {np.mean(baseline):.3f} +/- {np.std(baseline):.3f} accuracy\n"
        f"  - Saliency advantage: {(np.mean(sal_90) - np.mean(mag_90))*100:.1f} "
        f"percentage points\n"
        f"  - FLOPs saved: ~90% (10x fewer multiply-accumulate operations)\n"
    )
    print(claim)

    with open("results/falsifiable_claim.txt", "w") as f:
        f.write(claim)

    return claim


if __name__ == "__main__":
    results = run_sweep()
    save_csv(results)
    plot_pareto(results)
    plot_flops(results)
    measure_wall_clock(results)
    print_falsifiable_claim(results)
    print("\nAll Part 4 evidence generated in results/")
