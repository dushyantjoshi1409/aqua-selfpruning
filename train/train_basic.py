"""Part 2: Train a network with a real optimizer.

Trains a small MLP on the digits dataset using our custom autodiff engine
and Adam optimizer.  Reports a learning curve (loss and accuracy over epochs).

Usage:
    python -m train.train_basic
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from engine.tensor import Tensor, cross_entropy_loss
from nn.layers import make_mlp
from nn.optim import Adam
from train.data import load_digits_data, iterate_batches


def evaluate(model, X, y, batch_size=256):
    """Compute accuracy and average loss on a dataset."""
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


def train(seed=42, epochs=30, batch_size=64, lr=1e-3):
    """Train an MLP and return the learning curves."""
    np.random.seed(seed)

    X_train, y_train, X_test, y_test = load_digits_data(seed=seed)
    print(f"Dataset: {len(X_train)} train, {len(X_test)} test, "
          f"{X_train.shape[1]} features, {len(np.unique(y_train))} classes")

    # 64 -> 128 -> 64 -> 10  (He initialization, ReLU activations)
    model = make_mlp([X_train.shape[1], 128, 64, 10], activation="relu")
    optimizer = Adam(model.parameters(), lr=lr)
    rng = np.random.RandomState(seed)

    history = {"epoch": [], "train_loss": [], "train_acc": [],
               "test_loss": [], "test_acc": []}

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0

        for Xb, yb in iterate_batches(X_train, y_train, batch_size, rng=rng):
            logits = model(Tensor(Xb))
            loss = cross_entropy_loss(logits, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * len(yb)
            preds = logits.data.argmax(axis=1)
            epoch_correct += (preds == yb).sum()
            epoch_total += len(yb)

        train_acc = epoch_correct / epoch_total
        train_loss = epoch_loss / epoch_total
        test_acc, test_loss = evaluate(model, X_test, y_test)

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}  loss={train_loss:.4f}  "
                  f"train_acc={train_acc:.3f}  test_acc={test_acc:.3f}")

    return history


def plot_learning_curves(history, path="results/learning_curve.png"):
    """Save a two-panel learning curve plot."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(history["epoch"], history["train_loss"], label="Train")
    ax1.plot(history["epoch"], history["test_loss"], label="Test")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cross-Entropy Loss")
    ax1.set_title("Loss Curve")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(history["epoch"], history["train_acc"], label="Train")
    ax2.plot(history["epoch"], history["test_acc"], label="Test")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Accuracy Curve")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.suptitle("Part 2: Basic MLP Training on Digits Dataset", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Learning curve saved to {path}")


if __name__ == "__main__":
    history = train()
    plot_learning_curves(history)
    print(f"\nFinal test accuracy: {history['test_acc'][-1]:.3f}")
