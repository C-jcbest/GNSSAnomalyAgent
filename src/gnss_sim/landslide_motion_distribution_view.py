"""Show displacement-change concentration and directional persistence on original slots."""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from gnss_sim.landslide_feature_discovery import discovery_block_matrix


def distribution_view_inputs(values, noise):
    blocks, available, _, _ = discovery_block_matrix(values, noise)
    increments = np.diff(blocks, axis=1)
    edge_available = available[:, :-1] & available[:, 1:]
    magnitude = np.linalg.norm(increments, axis=-1)
    magnitude[~edge_available] = np.nan
    products = np.sum(increments[:, :-1] * increments[:, 1:], axis=-1)
    denominator = magnitude[:, :-1] * magnitude[:, 1:]
    agreement = np.divide(
        products,
        denominator,
        out=np.full_like(products, np.nan),
        where=np.isfinite(denominator) & (denominator > 1e-12),
    )
    agreement = np.clip(agreement, -1, 1)
    agreement_available = np.isfinite(agreement)
    descriptors = np.column_stack((magnitude, agreement, edge_available, agreement_available))
    return dict(magnitude=magnitude, agreement=agreement, descriptors=descriptors)


def view_sequences(magnitude, agreement, permuted=False):
    """The ablation preserves value/missing multisets but removes original order."""
    magnitude = np.asarray(magnitude, dtype=float)
    agreement = np.asarray(agreement, dtype=float)
    if magnitude.shape != (12,) or agreement.shape != (11,):
        raise ValueError(
            "Distribution view needs twelve original edges and eleven adjacent agreements"
        )
    if permuted:
        random = np.random.default_rng(2026101325)
        magnitude = magnitude[random.permutation(12)]
        agreement = agreement[random.permutation(11)]
    return magnitude, agreement


def render_distribution_view(magnitude, agreement, permuted=False):
    """Two compact panels, raw unavailable slots visible, no activity labeling."""
    magnitude, agreement = view_sequences(magnitude, agreement, permuted)
    canvas = Image.new("RGB", (256, 256), "white")
    draw = ImageDraw.Draw(canvas)
    left, right = 28, 246
    times = (-77.5 + 14 * np.arange(12), -70.5 + 14 * np.arange(11))
    x_positions = [left + (time + 91) * (right - left) / 182 for time in times]
    if permuted:
        x_positions = [np.linspace(left + 10, right - 10, count) for count in (12, 11)]
    draw.text((12, 4), "NEU change magnitude / noise", fill=(30, 70, 100))
    finite_magnitude = magnitude[np.isfinite(magnitude)]
    span = max(6.0, float(finite_magnitude.max()) if len(finite_magnitude) else 0.0)
    draw.text((30, 19), f"0 to {span:.2g}", fill=(90, 90, 90))
    draw.rectangle((left, 35, right, 116), outline=(190, 190, 190))
    if not permuted:
        draw.rectangle(
            (left + 84 * (right - left) / 182, 36, left + 98 * (right - left) / 182, 115),
            fill=(242, 242, 225),
        )
    for x, value in zip(x_positions[0], magnitude):
        if not np.isfinite(value):
            draw.rectangle((x - 4, 36, x + 4, 115), fill=(224, 224, 224))
            continue
        draw.rectangle((x - 4, 115 - 78 * value / span, x + 4, 115), fill=(30, 108, 142))
    draw.text((12, 129), "Neighbor direction agreement", fill=(35, 90, 70))
    draw.rectangle((left, 153, right, 225), outline=(190, 190, 190))
    draw.line((left + 1, 189, right - 1, 189), fill=(210, 210, 210))
    draw.text((7, 153), "+1", fill=(90, 90, 90))
    draw.text((7, 215), "-1", fill=(90, 90, 90))
    if not permuted:
        draw.line((137, 154, 137, 224), fill=(190, 180, 130))
    previous = None
    for x, value in zip(x_positions[1], agreement):
        if not np.isfinite(value):
            draw.rectangle((x - 4, 154, x + 4, 224), fill=(224, 224, 224))
            previous = None
            continue
        current = (float(x), float(189 - 34 * value))
        if previous is not None:
            draw.line((previous, current), fill=(66, 117, 88), width=2)
        draw.ellipse(
            (current[0] - 1.2, current[1] - 1.2, current[0] + 1.2, current[1] + 1.2),
            fill=(66, 117, 88),
        )
        previous = current
    if permuted:
        draw.text((35, 237), "Order removed; slot rank", fill=(70, 70, 70))
    else:
        draw.text((23, 234), "-91", fill=(70, 70, 70))
        draw.text((133, 234), "0", fill=(70, 70, 70))
        draw.text((231, 234), "91", fill=(70, 70, 70))
        draw.text((67, 245), "Original relative day", fill=(70, 70, 70))
    return canvas
