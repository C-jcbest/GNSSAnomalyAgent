"""Retain a prior decision when a successful local review supplies no new decision."""

import numpy as np

from gnss_sim.landslide_adaptive_review import combine_reviews


def combine_partial_reviews(initial, reviews, observed):
    """Distinguish successful abstention from contradictory evidence and failure.

    A successful review may be unable to decide within a noisy crop. If every
    covering review abstains, keep the existing global decision rather than
    interpreting this lack of a local decision as evidence against it. This is
    an inertia policy, not a calibrated confidence claim: wrong global decisions
    can also survive and must be counted in the experiments.

    Definite local answers still use the original strict majority. Opposing
    definite answers, or failure without a strict majority, remain unknown.
    Missing observations and unknown priors never gain a decision from abstention.
    """
    prediction = combine_reviews(initial, reviews, observed)
    coverage = np.zeros(len(initial), dtype=int)
    definite = np.zeros(len(initial), dtype=int)
    failures = np.zeros(len(initial), dtype=int)
    for review in reviews:
        start, stop = review["start"], review["stop"]
        activity = np.asarray(review["activity"], dtype=int)
        coverage[start:stop] += 1
        definite[start:stop] += activity[start:stop] >= 0
        if review["status"] == "failed":
            failures[start:stop] += 1
    no_new_decision = observed & (coverage > 0) & (definite == 0) & (failures == 0)
    prediction.activity[no_new_decision] = initial[no_new_decision]
    prediction.stage = np.where(prediction.activity == 0, 0, -1)
    prediction.validate(len(initial))
    return prediction, no_new_decision
