"""Describe how raw displacement changes are distributed around a query date."""

from __future__ import annotations

import numpy as np

from gnss_sim.landslide_feature_discovery import discovery_block_matrix

SCALES = (2, 3, 6)
GLOBAL_NAMES = (
    "net",
    "path",
    "coherence",
    "largest_share",
    "effective_edges",
    "trimmed_net",
    "supported_fraction",
)
LOCAL_NAMES = ("query_share", "mass_distance")


def motion_mass_descriptors(values, noise):
    """Measure concentration and query location, never label motion or artifacts.

    Thirteen original 14-day medians are reused; adjacent available blocks only
    form an increment. Each scale uses 5/7/13 blocks centered at the query.
    Statistics need at least half the increments and both query-adjacent edges.
    Missing target observations always remain unavailable. There is no hard
    movement threshold, interpolation, jump removal, or inferred phase.
    """
    values = np.asarray(values, dtype=float)
    blocks, available, _, _ = discovery_block_matrix(values, noise)
    increments = np.diff(blocks, axis=1)
    edge_available = available[:, :-1] & available[:, 1:]
    globals_by_scale, locals_by_scale = [], []
    for radius in SCALES:
        first, stop = 6 - radius, 6 + radius
        edges = increments[:, first:stop]
        valid = edge_available[:, first:stop]
        changes = np.where(valid[:, :, None], edges, 0)
        mass = np.linalg.norm(changes, axis=-1)
        path = mass.sum(axis=1)
        total = changes.sum(axis=1)
        net = np.linalg.norm(total, axis=1)
        largest_index = np.argmax(mass, axis=1)
        largest = mass[np.arange(len(values)), largest_index]
        trimmed = np.linalg.norm(total - changes[np.arange(len(values)), largest_index], axis=1)
        denominator = np.maximum(path, 1e-12)
        squared_mass = np.sum(mass**2, axis=1)
        effective = np.divide(
            path**2, squared_mass, out=np.zeros_like(path), where=squared_mass > 0
        )
        global_matrix = np.column_stack(
            (
                net,
                path,
                net / denominator,
                largest / denominator,
                effective,
                trimmed,
                valid.mean(axis=1),
            )
        )
        query_mass = mass[:, radius - 1] + mass[:, radius]
        # Midpoints of adjacent block centers are exactly t-7.5,t+6.5
        # at the query; distances retain original slot positions across gaps.
        relative_midpoints = -77.5 + 14 * np.arange(first, stop)
        distance = (mass * np.abs(relative_midpoints)).sum(axis=1) / denominator
        distance /= 14 * radius
        local_matrix = np.column_stack((query_mass / denominator, distance))
        supported = (valid.sum(axis=1) >= radius) & valid[:, radius - 1] & valid[:, radius]
        supported &= np.isfinite(values).all(axis=1)
        global_matrix[~supported] = np.nan
        local_matrix[~supported] = np.nan
        globals_by_scale.append(global_matrix)
        locals_by_scale.append(local_matrix)
    globals_matrix = np.column_stack(globals_by_scale)
    locality_matrix = np.column_stack(locals_by_scale)
    full = np.column_stack((globals_matrix, locality_matrix))
    # Strong comparison receives the same complete raw block positions, noise
    # units and availability, rather than only the handcrafted summaries.
    profile = np.column_stack((blocks.reshape(len(values), -1), available))
    return dict(
        global_distribution=globals_matrix,
        query_distribution=full,
        raw_profile=profile,
        blocks=blocks,
        edge_available=edge_available,
    )
