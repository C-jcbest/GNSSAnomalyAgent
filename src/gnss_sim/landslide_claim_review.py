"""Check factual block-change claims without converting them to motion labels."""

from __future__ import annotations

import numpy as np


def parse_block_claims(payload):
    """Require original-date labels and one sign claim per original block pair."""
    if not isinstance(payload, dict) or set(payload) != {"labels", "changes"}:
        raise ValueError("Expected labels and changes only")
    labels = payload["labels"]
    if (not isinstance(labels, list) or len(labels) != 42
            or any(type(value) is not int or value not in (-1, 0, 1) for value in labels)):
        raise ValueError("Expected forty-two original-date integer labels")
    changes = payload["changes"]
    if not isinstance(changes, list) or len(changes) != 12:
        raise ValueError("Expected twelve original adjacent block pairs")
    for claim in changes:
        if claim is None:
            continue
        if (not isinstance(claim, list) or len(claim) != 3
                or any(type(value) is not int or value not in (-1, 0, 1) for value in claim)):
            raise ValueError("Each change is a NEU sign triple or null")
    return np.asarray(labels, dtype=int), changes


def check_block_claims(blocks, claims):
    """Audit arithmetic facts only; a correct sign does not establish activity."""
    blocks = np.asarray(blocks, dtype=float)
    if blocks.shape != (13, 3) or len(claims) != 12:
        raise ValueError("Check thirteen original slots and twelve claims")
    facts = []
    conflicts = []
    for pair in range(12):
        available = np.isfinite(blocks[pair:pair + 2]).all()
        if available:
            difference = blocks[pair + 1] - blocks[pair]
            # This tiny tolerance protects arithmetic zero, not physical motion.
            signs = np.where(np.abs(difference) <= 1e-12, 0, np.sign(difference)).astype(int).tolist()
            values = difference.tolist()
        else:
            signs = None
            values = None
        fact = {"original_pair": [pair, pair + 1], "difference_neu_noise_units": values,
                "actual_signs": signs, "claimed_signs": claims[pair]}
        facts.append(fact)
        # An unreadable available pair is explicitly unknown, not a false
        # arithmetic statement. A concrete claim across a missing slot is false.
        if claims[pair] is not None and claims[pair] != signs:
            conflicts.append(fact)
    return facts, conflicts
