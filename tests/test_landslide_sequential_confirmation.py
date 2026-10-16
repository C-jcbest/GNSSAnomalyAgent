import numpy as np
from analyze_landslide_sequential_followup import confirmation_decision


def comparison_inputs():
    labels = {"case": [{"activity_label": value} for value in (1, 1, 0, 0)]}
    observed = {"case": np.ones(4, bool)}
    predictions, summaries = {}, {}
    for rep in (0, 1):
        main = f"followup_censored_r{rep}"
        predictions[main] = {"case": {"activity": [1, 1, 0, 0]}}
        summaries[main] = {"f1": 1.0, "weak_recall": 1.0}
        for stem in ("adaptive_no_feedback", "followup_fixed_short", "first_only"):
            control = f"{stem}_r{rep}" + ("__conflict_scope" if stem == "adaptive_no_feedback" else "")
            predictions[control] = {"case": {"activity": [1, 0, 0, 0]}}
            summaries[control] = {"f1": 2 / 3, "weak_recall": .5}
    return labels, observed, predictions, summaries


def test_confirmation_cannot_ignore_a_better_cheaper_first_view():
    labels, observed, predictions, summaries = comparison_inputs()
    for rep in (0, 1):
        control = f"first_only_r{rep}"
        predictions[control] = {"case": {"activity": [1, 1, 0, 0]}}
        summaries[control] = {"f1": 1.0, "weak_recall": 1.0}
    decision = confirmation_decision(labels, observed, predictions, summaries)
    assert not decision["passed"]
    assert [row["passed"] for row in decision["comparisons"]] == [True, True, False]
    assert not decision["default_changed"]


def test_weak_recall_loss_blocks_confirmation_despite_total_f1_gain():
    labels, observed, predictions, summaries = comparison_inputs()
    for rep in (0, 1):
        summaries[f"followup_censored_r{rep}"]["weak_recall"] = .25
    decision = confirmation_decision(labels, observed, predictions, summaries)
    assert not decision["passed"]
    assert decision["comparisons"][0]["mean_f1_difference"] > 0
    assert decision["comparisons"][0]["net_correct_both_repeats"] == 2
    assert not decision["comparisons"][0]["passed"]
