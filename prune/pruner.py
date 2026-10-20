"""Self-pruning mechanism: importance criteria, sparsity schedules, and pruner.

The importance criterion approximates the loss change from removing a
connection using a first-order Taylor expansion:

    DeltaL approx -w_ij * (dL/dw_ij)

so the importance score is |w_ij * dL/dw_ij|.  This is strictly better
than magnitude-only pruning (|w_ij|) because a large weight whose gradient
is near zero contributes little to current training dynamics and can safely
be removed, while a small weight with a large gradient is actively being
used and should be kept.

Gradients are accumulated (exponential moving average) across mini-batches
for stable estimates rather than relying on a single noisy batch.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Importance criteria
# ---------------------------------------------------------------------------

def magnitude_importance(weight_data, grad_data, mask):
    """Importance = |weight|.  Simple baseline, ignores gradients."""
    return np.abs(weight_data)


def saliency_importance(weight_data, grad_data, mask):
    """Importance = |weight * gradient|.

    First-order Taylor-expansion approximation of the loss change from
    removing each connection.  Active weights use their actual gradient;
    pruned weights (weight=0) score zero, which is correct because they
    are already removed.
    """
    return np.abs(weight_data * grad_data)


IMPORTANCE_FNS = {
    "magnitude": magnitude_importance,
    "saliency": saliency_importance,
}


# ---------------------------------------------------------------------------
# Sparsity schedule
# ---------------------------------------------------------------------------

def cubic_schedule(step, start_step, end_step, initial_sparsity, final_sparsity):
    """Cubic sparsity ramp (Zhu & Gupta, 2017).

    Prunes aggressively early — when the network has the most redundancy —
    and slows down near the target, where each surviving connection matters
    more.  This outperforms linear and step schedules in practice because
    the network has more training budget to recover from early pruning.

    Returns the target sparsity at the given step.
    """
    if step < start_step:
        return initial_sparsity
    if step >= end_step:
        return final_sparsity
    frac = (step - start_step) / (end_step - start_step)
    return final_sparsity + (initial_sparsity - final_sparsity) * (1.0 - frac) ** 3


# ---------------------------------------------------------------------------
# Gradual pruner
# ---------------------------------------------------------------------------

class GradualPruner:
    """Orchestrates gradual pruning during training.

    Maintains per-layer binary masks, accumulates gradient-based importance
    scores, and at each pruning step removes the globally least-important
    connections across all layers to meet the scheduled sparsity target.

    Optionally supports regrowth: pruned connections whose raw gradient
    magnitude exceeds a threshold may be reactivated with small random
    weights, allowing the network to explore alternative sparse structures.
    """

    def __init__(
        self,
        linear_layers,
        final_sparsity=0.9,
        start_step=0,
        end_step=1000,
        prune_every=100,
        importance_fn="saliency",
        allow_regrowth=False,
        regrowth_fraction=0.1,
    ):
        self.layers = list(linear_layers)
        self.final_sparsity = final_sparsity
        self.start_step = start_step
        self.end_step = end_step
        self.prune_every = prune_every
        self.importance_fn = IMPORTANCE_FNS[importance_fn]
        self.allow_regrowth = allow_regrowth
        self.regrowth_fraction = regrowth_fraction

        # Exponential moving average of gradient magnitudes
        self._ema_grad = [np.zeros_like(l.weight.data) for l in self.layers]
        self._ema_beta = 0.9
        self._steps_since_prune = 0

    def accumulate_gradients(self):
        """Update the gradient EMA from the current weight gradients.

        Call this after backward() and before the optimizer zeros gradients.
        The EMA smooths noisy per-batch gradient estimates for more stable
        importance scores.
        """
        for i, layer in enumerate(self.layers):
            if layer.weight.grad is not None:
                self._ema_grad[i] = (
                    self._ema_beta * self._ema_grad[i]
                    + (1.0 - self._ema_beta) * np.abs(layer.weight.grad)
                )
        self._steps_since_prune += 1

    def maybe_prune(self, step):
        """Prune if this is a scheduled pruning step.

        Returns True if pruning was applied.
        """
        if step < self.start_step:
            return False
        if (step - self.start_step) % self.prune_every != 0:
            return False

        target = cubic_schedule(
            step, self.start_step, self.end_step, 0.0, self.final_sparsity
        )
        if target <= 0:
            return False

        self._apply_pruning(target)
        self._steps_since_prune = 0
        return True

    def _apply_pruning(self, target_sparsity):
        """Global unstructured pruning to the target sparsity level."""
        # Compute per-weight importance scores
        all_scores = []
        for i, layer in enumerate(self.layers):
            scores = self.importance_fn(
                layer.weight.data, self._ema_grad[i], layer.mask
            )
            all_scores.append(scores.ravel())

        flat_scores = np.concatenate(all_scores)
        total = len(flat_scores)
        n_keep = max(1, int(total * (1.0 - target_sparsity)))

        # Global threshold: keep the n_keep highest-scoring weights
        if n_keep < total:
            # partition is O(n) vs O(n log n) for full sort
            threshold = np.partition(flat_scores, total - n_keep)[total - n_keep]
        else:
            threshold = -1.0

        # Apply new masks
        offset = 0
        for i, layer in enumerate(self.layers):
            size = layer.mask.size
            layer_scores = all_scores[i]
            new_mask = (layer_scores >= threshold).astype(np.float64)
            new_mask = new_mask.reshape(layer.mask.shape)

            if self.allow_regrowth:
                self._regrowth(layer, new_mask, i, target_sparsity)
            else:
                layer.mask = new_mask

            layer.weight.data *= layer.mask
            offset += size

    def _regrowth(self, layer, new_prune_mask, layer_idx, target_sparsity):
        """Allow pruned connections to regrow if their gradient is high.

        After applying the new pruning mask, we identify the top-scoring
        currently-pruned connections (by raw gradient magnitude) and
        reactivate a fraction of them with small random initial weights.
        """
        old_mask = layer.mask
        combined = new_prune_mask.copy()

        # Identify newly pruned positions
        newly_pruned = (old_mask == 1) & (new_prune_mask == 0)

        # Identify regrowth candidates: currently pruned with high gradients
        grad_mag = self._ema_grad[layer_idx]
        candidate_mask = (combined == 0).astype(np.float64)
        candidate_scores = grad_mag * candidate_mask

        n_regrow = int(layer.mask.size * self.regrowth_fraction * target_sparsity)
        n_regrow = min(n_regrow, int(candidate_mask.sum()))

        if n_regrow > 0:
            flat_cand = candidate_scores.ravel()
            topk = np.argpartition(-flat_cand, n_regrow)[:n_regrow]
            flat_combined = combined.ravel()
            flat_combined[topk] = 1.0
            combined = flat_combined.reshape(layer.mask.shape)

            # Initialize regrown weights to small random values
            regrown = (combined == 1) & (old_mask == 0)
            if np.any(regrown):
                active_std = np.std(layer.weight.data[layer.mask == 1])
                scale = max(active_std * 0.01, 1e-4)
                layer.weight.data[regrown] = np.random.randn(int(regrown.sum())) * scale

        layer.mask = combined

    def get_masks_dict(self, all_params):
        """Map parameter indices to masks for the optimizer."""
        masks = {}
        param_to_idx = {id(p): i for i, p in enumerate(all_params)}
        for layer in self.layers:
            idx = param_to_idx.get(id(layer.weight))
            if idx is not None:
                masks[idx] = layer.mask
        return masks

    def current_sparsity(self):
        """Compute the current overall sparsity."""
        total = sum(l.total_count() for l in self.layers)
        active = sum(l.active_count() for l in self.layers)
        return 1.0 - active / total

    def report(self):
        """Human-readable sparsity report."""
        lines = []
        total_params = 0
        total_active = 0
        for i, layer in enumerate(self.layers):
            a = layer.active_count()
            t = layer.total_count()
            total_params += t
            total_active += a
            lines.append(
                f"  Layer {i}: {a}/{t} active ({layer.sparsity() * 100:.1f}% sparse)"
            )
        overall = 1.0 - total_active / total_params
        lines.insert(0, f"Overall: {total_active}/{total_params} active "
                        f"({overall * 100:.1f}% sparse)")
        return "\n".join(lines)

    def flops_ratio(self):
        """Ratio of sparse-to-dense FLOPs (lower = cheaper)."""
        total_active = sum(l.active_count() for l in self.layers)
        total_params = sum(l.total_count() for l in self.layers)
        return total_active / total_params if total_params > 0 else 1.0
