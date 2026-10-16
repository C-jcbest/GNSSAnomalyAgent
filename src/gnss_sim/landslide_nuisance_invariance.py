"""Training interventions that preserve deformation while adding instrument noise."""

from __future__ import annotations

import numpy as np


def nuisance_counterpart(observations, noise, seed, record_number):
    """Add two offsets and four spikes without consulting any motion labels.

    This controlled intervention assumes additive instrument disturbances. It
    preserves the latent physical activity by construction, not necessarily its
    observability. It must not be used to declare field labels unchanged.
    """
    values = np.asarray(observations, dtype=float)
    noise = np.asarray(noise, dtype=float)
    if values.ndim != 2 or values.shape[1] != 3 or len(values) < 180:
        raise ValueError("Nuisance interventions require a complete NEU calendar")
    if noise.shape != (3,) or not np.isfinite(noise).all() or np.any(noise <= 0):
        raise ValueError("Three positive observation noise proxies are required")
    complete = np.isfinite(values).all(axis=1)
    if np.any(np.isfinite(values).any(axis=1) != complete):
        raise ValueError("Original days must be fully observed or fully missing")
    offset_candidates = np.flatnonzero(complete & (np.arange(len(values)) >= 70)
                                       & (np.arange(len(values)) < len(values) - 70))
    if len(offset_candidates) < 2:
        raise ValueError("Two supported offset dates are required")
    random = np.random.default_rng(np.random.SeedSequence([seed, record_number, 0x4E5049]))
    artifact = np.zeros_like(values)
    events = []
    offset_days = random.choice(offset_candidates, 2, replace=False)
    for day in sorted(offset_days):
        direction = random.normal(size=3)
        offset = direction / np.linalg.norm(direction) * random.uniform(8, 40)
        artifact[day:] += offset
        events.append(dict(kind="instrument_offset", day=int(day), vector_mm=offset.tolist()))
    spike_candidates = np.flatnonzero(complete & (np.arange(len(values)) >= 20)
                                      & (np.arange(len(values)) < len(values) - 20))
    spike_candidates = spike_candidates[~np.isin(spike_candidates, offset_days)]
    if len(spike_candidates) < 4:
        raise ValueError("Four separate supported spike dates are required")
    for day in sorted(random.choice(spike_candidates, 4, replace=False)):
        amplitude = random.normal(size=3) * noise * random.uniform(6, 20)
        artifact[day] += amplitude
        events.append(dict(kind="spike", day=int(day), vector_mm=amplitude.tolist()))
    counterpart = values + artifact
    counterpart[~complete] = np.nan
    return counterpart, artifact, events


def shuffled_within_activity(activity, seed):
    """Break parent matching while keeping the consistency pair's activity class."""
    activity = np.asarray(activity)
    if activity.ndim != 1 or not np.isin(activity, [0, 1]).all():
        raise ValueError("Shuffle controls require known binary TRAIN labels")
    indices = np.arange(len(activity))
    random = np.random.default_rng(seed)
    for label in (0, 1):
        selected = np.flatnonzero(activity == label)
        indices[selected] = random.permutation(selected)
    return indices


def nuisance_consistency_loss(original_logits, counterpart_logits):
    """Constrain activity probabilities; do not force nuisance labels to match."""
    import torch

    if original_logits.shape != counterpart_logits.shape:
        raise ValueError("Corresponding activity queries must share their shape")
    return torch.mean((torch.sigmoid(original_logits) - torch.sigmoid(counterpart_logits)) ** 2)
