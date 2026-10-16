"""Plot original displacement blocks in space while retaining their time order."""

from __future__ import annotations

import numpy as np


def trajectory_segments(blocks):
    """Only adjacent available original slots can form a trajectory segment."""
    blocks = np.asarray(blocks, dtype=float)
    if blocks.shape != (13, 3):
        raise ValueError("Trajectory requires thirteen original NEU blocks")
    valid = np.isfinite(blocks).all(axis=1)
    left = np.flatnonzero(valid[:-1] & valid[1:])
    segments = np.stack((blocks[left], blocks[left + 1]), axis=1)
    return left, segments


def parse_query_labels(payload):
    """Keep all forty-two original queried dates, including explicit unknown."""
    if (not isinstance(payload, dict) or set(payload) != {"labels"}
            or not isinstance(payload["labels"], list) or len(payload["labels"]) != 42
            or any(type(value) is not int or value not in (-1, 0, 1)
                   for value in payload["labels"])):
        raise ValueError("Expected exactly42 original-date integer states")
    return np.array(payload["labels"], dtype=int)


def render_trajectory(blocks, path, time_color):
    """Use equal coordinate scales; gaps never become new observed segments."""
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.colors import Normalize

    blocks = np.asarray(blocks, dtype=float)
    left, segments = trajectory_segments(blocks)
    centers = -84.5 + 14 * np.arange(13)
    valid = np.isfinite(blocks).all(axis=1)
    figure, axes = plt.subplots(1, 3, figsize=(12, 4.5))
    normalizer = Normalize(-84.5, 83.5)
    for panel, (first, second) in zip(axes, ((0, 1), (0, 2), (1, 2)), strict=True):
        projections = segments[:, :, [first, second]]
        colors = plt.colormaps["viridis"](normalizer(centers[left] + 7)) if time_color else "#858585"
        panel.add_collection(LineCollection(projections, colors=colors, linewidths=1.2))
        panel.scatter(blocks[valid, first], blocks[valid, second], s=32,
                      c=centers[valid] if time_color else "#858585",
                      cmap="viridis" if time_color else None,
                      norm=normalizer if time_color else None, zorder=3)
        query = valid & (centers >= -21) & (centers < 21)
        panel.scatter(blocks[query, first], blocks[query, second], s=90,
                      facecolors="none", edgecolors="black", linewidths=1.2, zorder=4)
        # Query slot IDs stay visible in both arms. Other dates remain in the
        # common block table; gray removal does not remove all temporal evidence.
        offsets = {5: (-20, -20), 6: (0, 18), 7: (18, -10)}
        for index in np.flatnonzero(query):
            panel.annotate(str(index), (blocks[index, first], blocks[index, second]),
                           xytext=offsets[index], textcoords="offset points", fontsize=9,
                           arrowprops={"arrowstyle": "-", "color": "#555555", "linewidth": .5})
        panel.set_xlabel(("N", "E", "U")[first] + " / noise proxy")
        panel.set_ylabel(("N", "E", "U")[second] + " / noise proxy")
        # Equal-sized coordinate spans keep every panel square without
        # distorting noise-unit geometry or shrinking a panel to a narrow strip.
        low = np.min(blocks[valid][:, [first, second]], axis=0)
        high = np.max(blocks[valid][:, [first, second]], axis=0)
        center = (low + high) / 2
        half_span = max(float(np.max(high - low)) * .65, 1.)
        panel.set_xlim(center[0] - half_span, center[0] + half_span)
        panel.set_ylim(center[1] - half_span, center[1] + half_span)
        panel.set_aspect("equal", adjustable="box")
        panel.grid(alpha=.2)
    title = "Original block trajectory | chronological color" if time_color else "Original block trajectory | gray control"
    figure.suptitle(title + "\nBlack rings: query blocks5/6/7 | missing blocks disconnected", fontsize=11)
    figure.subplots_adjust(left=.07, right=.88, bottom=.15, top=.80, wspace=.40)
    if time_color:
        color_axis = figure.add_axes((.92, .20, .015, .52))
        scalar = plt.cm.ScalarMappable(norm=normalizer, cmap="viridis")
        figure.colorbar(scalar, cax=color_axis, label="Original relative block center day")
    figure.savefig(path, dpi=120)
    plt.close(figure)
