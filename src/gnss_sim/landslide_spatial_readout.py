"""Coarse center/context readout from the same frozen chart encoder."""

from __future__ import annotations

import torch.nn.functional as functional


def spatial_chart_readouts(feature_map, projection):
    """Project global, central and flanking pools with the pretrained head.

    The three NEU panels occupy rows 0..6 of the 8x8 map. Columns 3..4
    overlap the marked target region; columns 1..2 and 5..6 are its flanks.
    Attention has already mixed spatial information, so these are readout
    regions, not isolated temporal windows or exact daily token alignment.
    """
    if feature_map.ndim != 4 or feature_map.shape[1:] != (1024, 8, 8):
        raise ValueError("Readout geometry requires MobileCLIP2-S0's 1024x8x8 map")
    global_pool = feature_map.mean(dim=(2, 3))
    center_pool = feature_map[:, :, :7, 3:5].mean(dim=(2, 3))
    flank_pool = feature_map[:, :, :7, [1, 2, 5, 6]].mean(dim=(2, 3))
    global_embedding = functional.normalize(projection(global_pool), dim=-1)
    center_embedding = functional.normalize(projection(center_pool), dim=-1)
    flank_embedding = functional.normalize(projection(flank_pool), dim=-1)
    contrast = (center_embedding - flank_embedding) / 2
    return global_embedding, center_embedding, contrast
