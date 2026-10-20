"""Part 3: Train a network that prunes itself.

Trains an MLP with gradual self-pruning using the Taylor saliency criterion
and a cubic sparsity schedule.  The network progressively removes its least
important connections during training until it reaches the target sparsity.

Usage:
    python -m train.train_prune [--sparsity 0.9] [--method saliency]
"""

import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from engine.tensor import Tensor, cross_entropy_loss
from nn.layers import make_mlp
from nn.optim import Adam
from prune.pruner import GradualPruner
from train.data import load_digits_data, iterate_batches


def evaluate(model, X, y, batch_size=256):
    """Compute accuracy and average loss."""
    correct = 0
    total_loss = 0.0
    n = len(X)
    for Xb, yb in iterate_batches(X, y, batch_size, shuffle=False):
        logits = model(Tensor(Xb))
        loss = cross_entropy_loss(logits, yb)
        total_loss += loss.item() * len(yb)
        preds = logits.data.argmax(axis=1)
        correct += (preds == yb).sum()
    return correct / n, total_loss / n


def sparse_forward_flops(model, input_dim, batch_size=1):
    """Count actual multiply-accumulate ops for one forward pass.

    Dense FLOPs:  2 * batch * fan_in * fan_out  per layer.
    Sparse FLOPs: 2 * batch * active_weights    per layer.
    """
    dense = 0
    sparse = 0
    for layer in model.linear_layers():
        fan_in, fan_out = layer.weight.shape
        dense += 2 * batch_size * fan_in * fan_out
        sparse += 2 * batch_size * layer.active_count()
    return dense, sparse


def train_with_pruning(
    seed=42, epochs=40, batch_size=64, lr=1e-3,
    target_sparsity=0.9, importance_fn="saliency",
    allow_regrowth=False, verbose=True,
):
    """Train an MLP with gradual self-pruning.

    Returns a dict with training history and final metrics.
    """
    np.random.seed(seed)
    X_train, y_train, X_test, y_test = load_digits_data(seed=seed)

    model = make_mlp([X_train.shape[1], 128, 64, 10], activation="relu")
    params = model.parameters()
    optimizer = Adam(params, lr=lr)
    rng = np.random.RandomState(seed)

    n_batches = int(np.ceil(len(X_train) / batch_size))
    total_steps = epochs * n_batches

    # Start pruning at 20% of training, end at 80%
    pruner = GradualPruner(
        model.linear_layers(),
        final_sparsity=target_sparsity,
        start_step=int(total_steps * 0.2),
        end_step=int(total_steps * 0.8),
        prune_every=max(1, n_batches),  # prune once per epoch
        importance_fn=importance_fn,
        allow_regrowth=allow_regrowth,
        regrowth_fraction=0.1,
    )

    history = {"epoch": [], "train_loss": [], "train_acc": [],
               "test_loss": [], "test_acc": [], "sparsity": []}

    step = 0
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0

        for Xb, yb in iterate_batches(X_train, y_train, batch_size, rng=rng):
            logits = model(Tensor(Xb))
            loss = cross_entropy_loss(logits, yb)

            optimizer.zero_grad()
            loss.backward()

            # Accumulate gradient info for importance scoring
            pruner.accumulate_gradients()

            # Maybe prune at this step
            pruner.maybe_prune(step)

            # Optimizer step with current masks
            masks = pruner.get_masks_dict(params)
            optimizer.step(masks)

            epoch_loss += loss.item() * len(yb)
            preds = logits.data.argmax(axis=1)
            epoch_correct += (preds == yb).sum()
            epoch_total += len(yb)
            step += 1

        train_acc = epoch_correct / epoch_total
        train_loss = epoch_loss / epoch_total
        test_acc, test_loss = evaluate(model, X_test, y_test)
        sparsity = pruner.current_sparsity()

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)
        history["sparsity"].append(sparsity)

        if verbose and (epoch % 5 == 0 or epoch == 1):
            print(f"  Epoch {epoch:3d}  loss={train_loss:.4f}  "
                  f"acc={test_acc:.3f}  sparsity={sparsity:.1%}")

    dense_flops, sparse_flops = sparse_forward_flops(
        model, X_train.shape[1], batch_size=1
    )

    return {
        "history": history,
        "final_test_acc": history["test_acc"][-1],
        "final_sparsity": pruner.current_sparsity(),
        "dense_flops": dense_flops,
        "sparse_flops": sparse_flops,
        "flops_ratio": pruner.flops_ratio(),
        "pruner_report": pruner.report(),
    }


def plot_pruning_curves(history, path="results/pruning_training.png"):
    """Save a three-panel plot: loss, accuracy, sparsity over epochs."""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))

    ax1.plot(history["epoch"], history["train_loss"], label="Train")
    ax1.plot(history["epoch"], history["test_loss"], label="Test")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(history["epoch"], history["train_acc"], label="Train")
    ax2.plot(history["epoch"], history["test_acc"], label="Test")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    ax3.plot(history["epoch"], [s * 100 for s in history["sparsity"]],
             color="red")
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Sparsity (%)")
    ax3.set_title("Sparsity Schedule")
    ax3.grid(True, alpha=0.3)

    fig.suptitle("Part 3: Self-Pruning Training (Saliency, 90% Target)",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Pruning curves saved to {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sparsity", type=float, default=0.9)
    parser.add_argument("--method", choices=["magnitude", "saliency"],
                        default="saliency")
    parser.add_argument("--regrowth", action="store_true")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"Training with {args.method} pruning, target sparsity={args.sparsity}")
    result = train_with_pruning(
        seed=args.seed,
        epochs=args.epochs,
        target_sparsity=args.sparsity,
        importance_fn=args.method,
        allow_regrowth=args.regrowth,
    )

    print(f"\n{result['pruner_report']}")
    print(f"\nFinal test accuracy: {result['final_test_acc']:.3f}")
    print(f"Dense FLOPs:  {result['dense_flops']:,}")
    print(f"Sparse FLOPs: {result['sparse_flops']:,}")
    print(f"FLOPs ratio:  {result['flops_ratio']:.3f}")

    plot_pruning_curves(result["history"])
