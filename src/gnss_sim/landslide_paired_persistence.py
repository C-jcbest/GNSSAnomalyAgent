"""Controlled persistence pairs and charts for a frozen image-text encoder."""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw


def offset_counterpart(observations, latent, first, last, step_day):
    """Replace TRAINING latent motion with an equal-displacement instrument step.

    Observational errors and original missing dates are retained. The original
    and counterpart have exactly equal observed endpoint values at first/last.
    The latent component is required by this training intervention; it must
    never be available to the detector or applied to evaluation records.
    """
    observations = np.asarray(observations, dtype=float)
    latent = np.asarray(latent, dtype=float)
    if (observations.ndim != 2 or observations.shape[1] != 3
            or latent.shape != observations.shape or not np.isfinite(latent).all()):
        raise ValueError("Paired observations require equal finite latent NEU calendars")
    if not 0 <= first < step_day <= last < len(observations):
        raise ValueError("The step must lie inside the selected endpoint interval")
    if not np.isfinite(observations[[first, last]]).all():
        raise ValueError("Endpoint equality requires observed original endpoints")
    displacement = latent[last] - latent[first]
    if np.linalg.norm(displacement) <= 1e-10:
        raise ValueError("A zero-displacement pair does not compare persistence and offset")
    replacement = np.broadcast_to(latent[first], latent.shape).copy()
    replacement[step_day:] += displacement
    counterpart = observations - latent + replacement
    # Preserve original endpoint bytes rather than accumulating round-off from
    # subtracting and adding two cumulative reference positions.
    counterpart[first] = observations[first]
    counterpart[last] = observations[last]
    return counterpart, replacement


def render_encoder_chart(patch):
    """Render all original 3-day slots, disconnecting unavailable neighbors."""
    patch = np.asarray(patch, dtype=float)
    if patch.shape != (6, 32, 61) or not np.isfinite(patch).all():
        raise ValueError("Encoder chart requires six original raster/mask channels")
    canvas = Image.new("RGB", (256, 256), "white")
    draw = ImageDraw.Draw(canvas)
    colors = ((20, 105, 130), (190, 85, 40), (80, 135, 75))
    rows = np.arange(32)
    for axis, color in enumerate(colors):
        top = 12 + axis * 73
        draw.rectangle((28, top, 246, top + 60), outline=(200, 200, 200))
        draw.text((7, top + 22), ("N", "E", "U")[axis], fill=color)
        available = patch[axis + 3, 0] > 0
        trace = patch[axis]
        mass = trace.sum(axis=0)
        ordinate = (trace * rows[:, None]).sum(axis=0) / np.maximum(mass, 1e-12)
        x_positions = 29 + np.arange(61) * 216 / 60
        y_positions = top + 1 + ordinate * 58 / 31
        for column in range(61):
            if not available[column]:
                x = x_positions[column]
                draw.rectangle((x - 1.8, top + 1, x + 1.8, top + 59), fill=(231, 231, 231))
                continue
            current = (float(x_positions[column]), float(y_positions[column]))
            if column > 0 and available[column - 1]:
                previous = (float(x_positions[column - 1]), float(y_positions[column - 1]))
                draw.line((previous, current), fill=color, width=1)
            draw.ellipse((current[0] - .8, current[1] - .8, current[0] + .8, current[1] + .8), fill=color)
        draw.line((137, top + 1, 137, top + 59), fill=(170, 170, 170), width=1)
    draw.text((21, 234), "-90", fill="black")
    draw.text((132, 234), "0", fill="black")
    draw.text((231, 234), "90", fill="black")
    draw.text((64, 245), "Original relative day", fill="black")
    return canvas


def pair_order_loss(positive_logits, negative_logits):
    """Require continuing motion to outrank the matched instrument offset."""
    import torch.nn.functional as functional

    if positive_logits.shape != negative_logits.shape or positive_logits.ndim != 1:
        raise ValueError("Matched pair logits must share one original pair dimension")
    return functional.softplus(1 - (positive_logits - negative_logits)).mean()


def pretrained_activity_head(visual_width, numeric_width, hidden_width):
    """Train only a small activity head; the image-text encoder stays frozen."""
    from torch import nn

    return nn.Sequential(nn.Linear(visual_width + numeric_width, hidden_width),
                         nn.ReLU(), nn.Linear(hidden_width, 1))
