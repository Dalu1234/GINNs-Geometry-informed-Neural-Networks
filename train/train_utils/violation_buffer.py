"""
Buffer of (z_base, c) pairs with violation scores for prioritized sampling.

Samples (z_base, c) proportional to violation so training focuses on pairs
where the model currently fails (interface/envelope violation).
"""

import torch
import numpy as np


class ViolationBuffer:
    """
    Stores (z_base, c_key, violation) entries. Supports sampling a batch of
    z_base for a given c_key with probability proportional to violation.
    """

    def __init__(self, nz_base, c_keys, z_interval, initial_size_per_c, device=None, eps=1e-8):
        """
        Args:
            nz_base: dimension of base latent (before conditioning cols).
            c_keys: list of conditioning keys (e.g. [(2,1,0.75), (3,1,1.0), ...]).
            z_interval: (low, high) for uniform sampling of z_base.
            initial_size_per_c: number of (z_base, c_key) pairs to seed per c_key.
            device: device for z tensors.
            eps: small constant added to violations for sampling (avoid zero prob).
        """
        self.nz_base = nz_base
        self.c_keys = list(c_keys)
        self.z_interval = z_interval
        self.eps = eps
        self.device = device or torch.get_default_device()

        # Flat list: each entry is (z_base tensor, c_key, violation float)
        self._entries = []

        for c_key in self.c_keys:
            for _ in range(initial_size_per_c):
                z_base = torch.rand(nz_base, device=self.device) * (z_interval[1] - z_interval[0]) + z_interval[0]
                self._entries.append((z_base, c_key, 1.0))  # initial violation = 1.0 so all get sampled

    def _indices_for_c(self, c_key):
        return [i for i, (_, c, _) in enumerate(self._entries) if c == c_key]

    def sample_batch(self, c_key, size, temperature=1.0, random_fraction=0.0):
        """
        Sample `size` (z_base, c_key) entries with c_key, with probability
        proportional to violation (higher violation -> more likely).
        A fraction of slots can be fresh random z_base to avoid collapsing onto a tiny corner of z space.

        Args:
            c_key: conditioning key to filter by.
            size: number of z_base to return (e.g. ginn_bsize).
            temperature: >1 smooths distribution, <1 sharpens (more focus on high violation).
            random_fraction: fraction of batch to fill with fresh random z_base (e.g. 0.1 = 10%); rest from buffer.

        Returns:
            z_bases: tensor (size, nz_base).
            indices: list of buffer indices or None for random slots (only non-None get updated after step).
        """
        n_random = min(size - 1, max(0, int(round(size * random_fraction))))
        n_from_buffer = size - n_random

        indices = self._indices_for_c(c_key)
        if len(indices) == 0:
            raise ValueError(f"No buffer entries for c_key={c_key}")

        violations = np.array([self._entries[i][2] for i in indices], dtype=np.float64)
        # Sanitize: NaN/inf can come from model NaNs; avoid invalid probs for np.random.choice
        violations = np.nan_to_num(violations, nan=1.0, posinf=1.0, neginf=0.0)
        violations = np.clip(violations, 0, None) + self.eps
        weights = violations ** (1.0 / temperature)
        wsum = weights.sum()
        if wsum <= 0 or not np.isfinite(wsum):
            probs = np.ones(len(indices), dtype=np.float64) / len(indices)
        else:
            probs = weights / wsum

        chosen_local = np.random.choice(len(indices), size=n_from_buffer, replace=True, p=probs)
        chosen_indices = [indices[j] for j in chosen_local]
        z_bases_buffer = torch.stack([self._entries[i][0].clone() for i in chosen_indices], dim=0)

        if n_random > 0:
            z_bases_random = torch.rand(n_random, self.nz_base, device=self.device) * (
                self.z_interval[1] - self.z_interval[0]
            ) + self.z_interval[0]
            indices_random = [None] * n_random
            z_bases = torch.cat([z_bases_buffer, z_bases_random], dim=0)
            indices_out = chosen_indices + indices_random
            perm = np.random.permutation(size)
            z_bases = z_bases[perm]
            indices_out = [indices_out[p] for p in perm]
        else:
            z_bases = z_bases_buffer
            indices_out = chosen_indices

        return z_bases, indices_out

    def update_violations(self, indices, violations):
        """Update violation for the given buffer indices. NaN/inf are stored as 1.0 to keep sampling valid."""
        for i, v in zip(indices, violations):
            z_base, c_key, _ = self._entries[i]
            v_f = float(v)
            if not np.isfinite(v_f) or v_f < 0:
                v_f = 1.0
            self._entries[i] = (z_base, c_key, v_f)

    def add_random(self, c_key, violation=1.0):
        """Add one random (z_base, c_key) with given violation (e.g. to keep diversity)."""
        z_base = torch.rand(self.nz_base, device=self.device) * (
            self.z_interval[1] - self.z_interval[0]
        ) + self.z_interval[0]
        self._entries.append((z_base, c_key, violation))

    def __len__(self):
        return len(self._entries)
