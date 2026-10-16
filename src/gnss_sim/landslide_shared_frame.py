"""Keep a local GNSS training intervention local in the rendered coordinate frame."""

from __future__ import annotations

import warnings

import numpy as np


def observation_slots(observations, center):
    """Use the original disjoint three-day, two-complete-observation rule."""
    observations = np.asarray(observations, dtype=float)
    if observations.ndim != 2 or observations.shape[1] != 3:
        raise ValueError("Shared charts require an original NEU calendar")
    if type(center) is not int or not 0 <= center < len(observations):
        raise ValueError("Chart center must belong to the original calendar")
    padded = np.pad(observations, ((91, 91), (0, 0)), constant_values=np.nan)
    indices = center + 91 + np.arange(-90, 91, 3)[:, None] + np.array([-1, 0, 1])
    samples = padded[indices]
    complete = np.isfinite(samples).all(axis=-1)
    samples = np.where(complete[..., None], samples, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        medians = np.nanmedian(samples, axis=1)
    medians[complete.sum(axis=1) < 2] = np.nan
    return medians


def shared_frame_pair(original, counterpart, center, noise):
    """Render TRAIN-only paired charts with one reference and one safe span.

    The positive median supplies the reference. The span covers both members,
    preventing the negative from clipping outside the positive's range. It
    retains the original positive's six-noise-unit lower bound. This function
    needs observations only; constructing the physical counterpart is a
    separate training intervention and never an evaluation-time operation.
    """
    original = np.asarray(original, dtype=float)
    counterpart = np.asarray(counterpart, dtype=float)
    noise = np.asarray(noise, dtype=float)
    if original.shape != counterpart.shape:
        raise ValueError("Paired charts must share one original calendar")
    if not np.array_equal(np.isfinite(original), np.isfinite(counterpart)):
        raise ValueError("The intervention must preserve the original missing mask")
    if noise.shape != (3,) or not np.isfinite(noise).all() or np.any(noise <= 0):
        raise ValueError("Shared charts require three positive original noise proxies")
    positive = observation_slots(original, center)
    negative = observation_slots(counterpart, center)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        reference = np.nanmedian(positive, axis=0)
    reference = np.nan_to_num(reference)
    centered = np.stack((positive - reference, negative - reference))
    available = np.isfinite(centered)
    magnitude = np.max(np.where(available, np.abs(centered), 0), axis=(0, 1))
    span = np.maximum(2 * magnitude, 6 * noise)
    coordinates = 31 * (0.5 - centered / span)
    coordinates = np.transpose(coordinates, (0, 2, 1))
    available = np.transpose(available, (0, 2, 1))
    distance = np.abs(np.arange(32)[None, None, :, None] - np.nan_to_num(coordinates)[:, :, None])
    trace = np.maximum(1 - distance, 0) * available[:, :, None]
    mask = np.broadcast_to(available[:, :, None], trace.shape)
    patches = np.concatenate((trace, mask), axis=1).astype(np.float32)
    return patches[0], patches[1], {"reference": reference, "span": span}


def train_shared_pool(inputs, target, positive, negative, arm, width):
    """Hold supervised examples, order and optimization fixed across pair arms."""
    import time

    import torch

    from gnss_sim.landslide_paired_persistence import pair_order_loss, pretrained_activity_head

    seed = 20261020
    torch.manual_seed(seed)
    model = pretrained_activity_head(inputs.shape[1], 0, width)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    if not 0 < target.sum() < len(target):
        raise ValueError("Both activity and stationary training centers are required")
    criterion = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor((len(target) - target.sum()) / target.sum())
    )
    if arm["shuffle"]:
        negative = negative[np.random.default_rng(seed + 1).permutation(len(negative))]
    left, right = torch.from_numpy(positive), torch.from_numpy(negative)
    values = torch.from_numpy(inputs)
    labels = torch.from_numpy(target.astype(np.float32))
    shuffle = np.random.default_rng(seed)
    history = []
    started = time.perf_counter()
    model.train()
    for epoch in range(12):
        supervised_total = paired_total = 0.0
        order = shuffle.permutation(len(target))
        for first in range(0, len(order), 128):
            indices = order[first : first + 128]
            supervised = criterion(model(values[indices]).squeeze(1), labels[indices])
            paired = torch.tensor(0.0)
            if arm["pair"]:
                paired = pair_order_loss(model(left).squeeze(1), model(right).squeeze(1))
            optimizer.zero_grad()
            (supervised + paired).backward()
            optimizer.step()
            supervised_total += float(supervised.detach()) * len(indices)
            paired_total += float(paired.detach()) * len(indices)
        history.append(
            dict(
                epoch=epoch + 1,
                balanced_bce=supervised_total / len(target),
                pair_order_loss=paired_total / len(target),
            )
        )
    return model, history, time.perf_counter() - started, len(target)
