"""Learn GNSS activity from raw chart patches with label-preserving reflections."""

from __future__ import annotations

import warnings

import numpy as np

from gnss_sim.landslide_adaptive_review import raw_motion_evidence


def raw_chart_patches(case, noise, height=32):
    """Rasterize local NEU without changing time or bridging unavailable blocks.

    Each column is a disjoint three-day median requiring two complete NEU days.
    The 61 column centers cover -90..90 original days. The patch's reference
    offset is removed; the vertical span retains at least six daily-noise units.
    Missing columns have no trace. Separate broadcast masks expose availability
    rather than silently presenting an unavailable column as a zero position.
    """
    values = np.array(
        [row if row is not None else [np.nan] * 3 for row in case.displacement_mm],
        dtype=float,
    )
    noise = np.asarray(noise, dtype=float)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("Raw chart patches require original NEU observations")
    if noise.shape != (3,) or not np.isfinite(noise).all() or np.any(noise <= 0):
        raise ValueError("Raw chart patches require three positive noise proxies")
    if type(height) is not int or height < 2:
        raise ValueError("Raster height must contain at least two rows")
    padded = np.pad(values, ((91, 91), (0, 0)), constant_values=np.nan)
    centers = np.arange(len(values))[:, None, None] + 91
    offsets = np.arange(-90, 91, 3)[None, :, None]
    indices = centers + offsets + np.array([-1, 0, 1])[None, None, :]
    samples = padded[indices]
    complete = np.isfinite(samples).all(axis=-1)
    samples = np.where(complete[..., None], samples, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        medians = np.nanmedian(samples, axis=2)
        medians[complete.sum(axis=2) < 2] = np.nan
        reference = np.nanmedian(medians, axis=1, keepdims=True)
    centered = medians - reference
    available = np.isfinite(centered)
    magnitude = np.max(np.where(available, np.abs(centered), 0), axis=1)
    span = np.maximum(2 * magnitude, 6 * noise[None, :])
    coordinates = (height - 1) * (0.5 - centered / span[:, None, :])
    coordinates = np.transpose(coordinates, (0, 2, 1))
    available = np.transpose(available, (0, 2, 1))
    rows = np.arange(height)[None, None, :, None]
    distance = np.abs(rows - np.nan_to_num(coordinates)[:, :, None, :])
    trace = np.maximum(1 - distance, 0) * available[:, :, None, :]
    mask = np.broadcast_to(available[:, :, None, :], trace.shape)
    return np.concatenate((trace, mask), axis=1).astype(np.float32)


def invariance_inputs(case, features):
    """Give numerical competitors and the visual model the same raw84 features."""
    long42 = raw_motion_evidence(case, features)
    long84 = raw_motion_evidence(case, features, 84)
    matrix = np.column_stack(
        (
            features.matrix,
            long42["rate"],
            long42["signal_to_noise"],
            long42["direction"],
            long84["rate"],
            long84["signal_to_noise"],
            long84["direction"],
        )
    )
    support = features.observed & (
        np.isfinite(features.progressive_rate)
        | np.isfinite(long42["rate"])
        | np.isfinite(long84["rate"])
    )
    images = raw_chart_patches(case, features.noise)
    return images, matrix, support


def reflect_chart_axes(images, signs):
    """Reflect each physical component vertically; never reverse the time axis."""
    images = np.asarray(images)
    signs = np.asarray(signs)
    if images.ndim != 4 or images.shape[1] != 6:
        raise ValueError("Reflections require a batch of six-channel chart patches")
    if signs.shape != (len(images), 3) or not np.isin(signs, [-1, 1]).all():
        raise ValueError("Each chart needs three independent coordinate signs")
    transformed = images.copy()
    for axis in range(3):
        selected = signs[:, axis] == -1
        transformed[selected, axis] = images[selected, axis, ::-1, :]
        transformed[selected, axis + 3] = images[selected, axis + 3, ::-1, :]
    return transformed


def chart_numeric_descriptors(images):
    """Expose every raster column's position and availability to strong XGBoost.

    The linear two-row raster preserves each supported column's ordinate. Its
    first moment recovers that ordinate, so this control receives the chart's
    temporal shape information without learning a convolutional encoder.
    """
    images = np.asarray(images)
    if images.ndim != 4 or images.shape[1] != 6 or images.shape[2] < 2:
        raise ValueError("Chart descriptors require six-channel image patches")
    trace = images[:, :3]
    availability = images[:, 3:, 0, :] > 0
    rows = np.arange(images.shape[2])[None, None, :, None]
    mass = trace.sum(axis=2)
    positions = (trace * rows).sum(axis=2) / np.maximum(mass, 1e-12)
    positions = positions / (images.shape[2] - 1) - 0.5
    positions[~availability] = np.nan
    return np.column_stack((positions.reshape(len(images), -1), availability.reshape(len(images), -1)))


def fit_numeric_scaling(matrix):
    """Fit numerical normalization only on the eligible training observations."""
    matrix = np.asarray(matrix, dtype=float)
    finite = np.isfinite(matrix)
    counts = finite.sum(axis=0)
    mean = np.sum(np.where(finite, matrix, 0), axis=0) / np.maximum(counts, 1)
    differences = np.where(finite, matrix - mean, 0)
    variance = np.sum(differences**2, axis=0) / np.maximum(counts, 1)
    scale = np.sqrt(variance)
    scale[scale < 1e-12] = 1
    return {"mean": mean.tolist(), "scale": scale.tolist()}


def numeric_input(matrix, scaling):
    """Expose unavailable numerical values explicitly, using training means."""
    matrix = np.asarray(matrix, dtype=float)
    finite = np.isfinite(matrix)
    scaled = (np.where(finite, matrix, scaling["mean"]) - scaling["mean"]) / scaling["scale"]
    return np.column_stack((scaled, ~finite)).astype(np.float32)


def activity_model(hidden_width, numeric_width, use_image=True, use_numeric=True):
    """Build a small two-convolution chart encoder and one activity head.

    Torch is an existing optional reproduction dependency. Import it only when
    actually training/inferencing; observation rendering needs NumPy alone.
    """
    import torch
    from torch import nn

    if not use_image and not use_numeric:
        raise ValueError("At least one observation modality is required")

    class ChartActivityModel(nn.Module):
        def __init__(self):
            super().__init__()
            if use_image:
                self.image_encoder = nn.Sequential(
                    nn.Conv2d(6, 8, kernel_size=3, stride=2, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d((2, 5)),
                    nn.Flatten(),
                )
            else:
                self.image_encoder = nn.Identity()
            image_width = 160 if use_image else 0
            auxiliary_width = numeric_width if use_numeric else 0
            self.classifier = nn.Sequential(
                nn.Linear(image_width + auxiliary_width, hidden_width),
                nn.ReLU(),
                nn.Linear(hidden_width, 1),
            )

        def forward(self, images, numbers):
            modalities = []
            if use_image:
                modalities.append(self.image_encoder(images))
            if use_numeric:
                modalities.append(numbers)
            return self.classifier(torch.cat(modalities, dim=1)).squeeze(1)

    return ChartActivityModel()
