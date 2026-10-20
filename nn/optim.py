"""Optimizers with mask-aware state management.

The critical correctness requirement: when a weight is pruned (mask flips
from 1 to 0) or revived (0 to 1), the optimizer must reset that weight's
moment estimates (m and v in Adam) to zero.

Without this reset, the second moment v decays slowly (beta2=0.999) during
the pruning period but retains ~90% of its pre-pruning value after 100 steps.
When the weight is revived, this stale v creates a denominator that is far
too large relative to the fresh numerator, suppressing the effective learning
rate for the revived weight by roughly an order of magnitude.  The revived
connection effectively cannot learn, defeating the purpose of regrowth.

The fix: detect mask transitions via a per-parameter previous-mask snapshot,
and zero out m[changed] and v[changed] whenever the mask changes.
"""

import numpy as np


class Adam:
    """Adam optimizer (Kingma & Ba, 2015) with mask-aware state management.

    Tracks the previous pruning mask for each parameter.  When a mask
    element flips (active->pruned or pruned->active), the corresponding
    first-moment (m) and second-moment (v) estimates are reset to zero so
    that the revived weight starts with fresh optimizer state.
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.0):
        self.params = list(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0
        self.m = [np.zeros_like(p.data) for p in self.params]
        self.v = [np.zeros_like(p.data) for p in self.params]
        self._prev_masks: dict[int, np.ndarray] = {}

    def step(self, masks=None):
        """Perform one Adam update.

        Args:
            masks: optional dict {param_index: binary_mask_array}.  When
                provided, gradients for masked-out positions are zeroed
                before the update and weights are re-masked afterward.
                Mask transitions trigger optimizer-state resets.
        """
        self.t += 1

        for i, p in enumerate(self.params):
            if p.grad is None:
                continue

            g = p.grad.copy()

            if self.weight_decay > 0:
                p.data -= self.lr * self.weight_decay * p.data

            if masks is not None and i in masks:
                mask = masks[i]

                # Detect mask transitions and reset stale state
                if i in self._prev_masks:
                    changed = self._prev_masks[i] != mask
                    if np.any(changed):
                        self.m[i][changed] = 0.0
                        self.v[i][changed] = 0.0
                self._prev_masks[i] = mask.copy()

                # Zero gradient for pruned positions so Adam does not
                # accumulate moments for inactive weights
                g *= mask

            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * g ** 2

            m_hat = self.m[i] / (1.0 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1.0 - self.beta2 ** self.t)

            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

            # Re-enforce mask to guarantee pruned weights stay at zero
            if masks is not None and i in masks:
                p.data *= masks[i]

    def zero_grad(self):
        """Set all parameter gradients to None."""
        for p in self.params:
            p.grad = None


class SGD:
    """SGD with momentum, also supporting mask-aware updates."""

    def __init__(self, params, lr=0.01, momentum=0.9):
        self.params = list(params)
        self.lr = lr
        self.momentum = momentum
        self.v = [np.zeros_like(p.data) for p in self.params]
        self._prev_masks: dict[int, np.ndarray] = {}

    def step(self, masks=None):
        """Perform one SGD update."""
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue

            g = p.grad.copy()

            if masks is not None and i in masks:
                mask = masks[i]
                if i in self._prev_masks:
                    changed = self._prev_masks[i] != mask
                    if np.any(changed):
                        self.v[i][changed] = 0.0
                self._prev_masks[i] = mask.copy()
                g *= mask

            self.v[i] = self.momentum * self.v[i] + g
            p.data -= self.lr * self.v[i]

            if masks is not None and i in masks:
                p.data *= masks[i]

    def zero_grad(self):
        """Set all parameter gradients to None."""
        for p in self.params:
            p.grad = None
