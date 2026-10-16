"""Read original displacement-block changes at the target date, without new labels."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from gnss_sim.landslide_visual_invariance import numeric_input


def original_edges(evidence):
    """Never difference blocks separated by an unavailable original block."""
    blocks = evidence["blocks"]
    edges = np.diff(blocks, axis=1).copy()
    edges[~evidence["edge_available"]] = np.nan
    return edges.reshape(len(blocks), 36)


def dated_inputs(evidence, edge_scaling, numbers, embeddings, kind):
    """Keep query-adjacent edges 5 and 6 fixed in the order-destruction control.

    Changes are computed before permutation on the original calendar. We
    rearrange existing observations and flags together, never fabricate a
    derivative between two nonadjacent blocks or interpolate missing input.
    """
    scaled = numeric_input(original_edges(evidence), edge_scaling)
    edges = np.concatenate(
        (scaled[:, :36].reshape(-1, 12, 3), scaled[:, 36:].reshape(-1, 12, 3)),
        axis=2,
    )
    if kind == "permuted":
        positions = np.array([0, 1, 2, 3, 4, 7, 8, 9, 10, 11])
        random = np.random.default_rng(2026101327)
        for row in range(len(edges)):
            order = random.permutation(positions)
            edges[row, positions] = edges[row, order]
    parts = [edges.reshape(len(edges), 72), numbers]
    if kind == "visual":
        parts.append(embeddings)
    return np.column_stack(parts).astype(np.float32)


class DatedEvidenceHead(nn.Module):
    """A small numerical chronology probe, optionally fused with cached vision."""

    def __init__(self, width, kind):
        super().__init__()
        self.kind = kind
        if kind == "flat":
            head_width = 150
        else:
            self.temporal = nn.Sequential(
                nn.Conv1d(6, width, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv1d(width, width, kernel_size=3, dilation=3, padding=3),
                nn.ReLU(),
            )
            head_width = 2 * width + 78
            if kind == "visual":
                head_width += 512
        self.classifier = nn.Sequential(
            nn.Linear(head_width, width),
            nn.ReLU(),
            nn.Linear(width, 1),
        )

    def forward(self, inputs):
        if self.kind == "flat":
            return self.classifier(inputs)
        edges = inputs[:, :72].reshape(-1, 12, 6).transpose(1, 2)
        encoded = self.temporal(edges)
        if self.kind == "pooled":
            pooled = encoded.mean(dim=2)
            dated = torch.cat((pooled, pooled), dim=1)
        else:
            # These are the two original edges bordering the query block.
            # The convolution mixes nearby evidence; it is not an exact-day
            # visual token alignment or a physical activity estimate.
            dated = torch.cat((encoded[:, :, 5], encoded[:, :, 6]), dim=1)
        return self.classifier(torch.cat((dated, inputs[:, 72:]), dim=1))
