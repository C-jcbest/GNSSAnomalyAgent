"""Probe one doubled raw-evidence scale before spending on new visual reviews."""
from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import numpy as np
from probe_landslide_revision_scope import conflict_neighbourhood
from run_landslide_comparison import ROOT, load_labels, save_json, sha
from run_landslide_window_optimization import load_case_features, read_json

from gnss_sim.landslide_adaptive_review import ReviewPolicy, raw_motion_evidence, review_priorities
from gnss_sim.landslide_evaluation import runs, score_counts

PROTOCOL = {
    "schema": "landslide-longer-evidence-probe-v1", "separation_days": 84,
    "basis": "One doubling of the existing 42-day spacing; same three 14-day raw median blocks",
    "policy": {"maximum_reviews": 2, "minimum_conflict_days": 14,
               "minimum_rate": .03, "minimum_signal_to_noise": 2.0},
    "scope": "Train/validation plus already viewed controlled development records. "
             "Evidence and observation opportunities, not independently confirmed VLM accuracy",
    "rules": "Retain the paired rate/noise/direction of one scale. Add 84-day positive "
             "evidence by OR; never construct a positive by mixing one scale's rate with "
             "another scale's noise ratio. Base conflict neighbourhood=49; newly enabled "
             "84-day conflicts use physical half-footprint=91. Updated selection stays bounded.",
    "candidate_selection": "No rate/noise/budget/window sweep or old-test threshold rescue. "
                           "Use this probe to decide whether a small cached VLM mechanism test is warranted",
    "numerical_controls": "If pursued, give rule and XGBoost this same added scale and "
                          "fit/calibrate on the original train24/validation12, not a reused old feature model",
    "failures": "Long evidence can blur boundaries, react to steps and trigger useless reviews. "
                "No numerical positive is automatically a visual activity label",
}


def scale_positive(evidence, policy):
    return (np.isfinite(evidence["rate"]) & (evidence["rate"] >= policy.minimum_rate)
            & (evidence["signal_to_noise"] >= policy.minimum_signal_to_noise))


def merged_long_evidence(base, extra, policy):
    """Add complete scale-specific evidence, preserving the rate/noise pairing."""
    use_extra = (~np.isfinite(base["rate"]) & np.isfinite(extra["rate"]))
    use_extra |= scale_positive(extra, policy) & ~scale_positive(base, policy)
    return {key: np.where(use_extra, extra[key], base[key]) for key in base}


def expanded_observed(mask, observed, margin):
    expanded = np.zeros(len(mask), dtype=bool)
    for start, stop in runs(mask):
        expanded[max(0, start - margin):min(len(mask), stop + margin)] = True
    return expanded & observed


def longer_conflict_scope(initial, features, base, extra, policy):
    """Original conflict support keeps its radius; only new support needs 91 days."""
    merged = merged_long_evidence(base, extra, policy)
    old_conflict, _ = conflict_neighbourhood(initial, features, base, policy)
    conflict, _ = conflict_neighbourhood(initial, features, merged, policy)
    existing = old_conflict & conflict
    additional = conflict & ~old_conflict
    neighbourhood = (expanded_observed(existing, features.observed, 49)
                     | expanded_observed(additional, features.observed, 91))
    return merged, conflict, neighbourhood, additional


def probe(out):
    base = ROOT / "artifacts/landslide-adaptive-review-2026-10-09"
    training = base / "motion-probe-v2"
    sources = [base / name for name in (
        "mechanism-v1", "confirmation-v1", "interval-confirmation-v1", "revision-scope-confirmation-v1",
    )]
    out.mkdir(parents=True, exist_ok=False)
    save_json(out / "protocol.json", PROTOCOL)
    shutil.copyfile(Path(__file__), out / "source.py")
    policy = ReviewPolicy(**PROTOCOL["policy"])
    selections, evidence_rows, opportunities, bindings = {}, [], [], {}
    folders = [training, *sources]
    for folder in folders:
        if folder == training:
            config = read_json(training / "protocol.json")
            split = config["split"]
            groups = {cid: name for name in ("train", "validation") for cid in split[name]}
            reference = folder / "reference-labels.csv"
            for file, expected in read_json(folder / "manifest.json")["files"].items():
                if sha(folder / file) != expected:
                    raise ValueError("Training evidence changed")
            cache = None
        else:
            config = read_json(folder / "protocol.json")
            groups = dict.fromkeys(config["case_ids"], "viewed_development")
            reference = folder / "reference-subset.csv"
            for file, expected in read_json(folder / "freeze.json").items():
                if sha(folder / file) != expected:
                    raise ValueError("Frozen comparison changed")
            for file, expected in read_json(folder / "prediction-freeze.json").items():
                if sha(folder / file) != expected:
                    raise ValueError("Frozen predictions changed")
            cache = read_json(folder / "predictions.json")
        cases, features = {}, {}
        for cid in groups:
            cases[cid], features[cid] = load_case_features(folder, cid)
        target = out / folder.name
        target.mkdir()
        # The shared loader validates an exact record population. The training
        # reference also contains viewed method-selection records, which this
        # part of the probe must not treat as training/validation examples.
        with reference.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            columns = reader.fieldnames
            subset = [row for row in reader if row["case_id"] in cases]
        subset_path = target / "reference-subset.csv"
        with subset_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(subset)
        labels = load_labels(subset_path, cases)
        for cid in groups:
            feature = features[cid]
            evidence42 = raw_motion_evidence(cases[cid], feature)
            evidence84 = raw_motion_evidence(cases[cid], feature, PROTOCOL["separation_days"])
            merged = merged_long_evidence(evidence42, evidence84, policy)
            actual = np.array([int(row["activity_label"]) for row in labels[cid]])
            weak = np.array([int(row["weak_signal"]) == 1 for row in labels[cid]])
            short = ((feature.progressive_rate >= policy.minimum_rate)
                     & (feature.signal_to_noise >= policy.minimum_signal_to_noise) & feature.observed)
            for name, evidence in (("base42", evidence42), ("extended84", merged)):
                gate = (short | scale_positive(evidence, policy)) & feature.observed
                evidence_rows.append({"source": folder.name, "group": groups[cid], "case_id": cid,
                    "variant": name, **score_counts(int(np.count_nonzero(gate & (actual == 1))),
                                                    int(np.count_nonzero(gate & (actual == 0))),
                                                    int(np.count_nonzero(~gate & (actual == 1)))),
                    "weak_detected_days": int(np.count_nonzero(gate & weak)), "weak_days": int(weak.sum()),
                    "scope": "Numerical trigger positives only, not final visual accuracy"})
            np.savez_compressed(target / f"{cid}-evidence.npz",
                                **{f"h{h}_{key}": value for h, values in ((42, evidence42), (84, evidence84))
                                   for key, value in values.items()})
            if cache is None:
                continue
            old_selections = read_json(folder / "selections.json")
            for rep in range(config["repetitions"]):
                key = f"{cid}-r{rep}"
                initial = np.array(cache[f"direct_visual_r{rep}"][cid]["activity"])
                old_choice = old_selections[key]
                windows = [(row["start"], row["stop"]) for row in old_choice["priorities"]]
                merged, conflict, neighbourhood, additional = longer_conflict_scope(
                    initial, feature, evidence42, evidence84, policy)
                priorities, selected = review_priorities(initial, feature, merged, windows, policy)
                if any(int(conflict[r["start"]:r["stop"]].sum()) != r["conflict_days"] for r in priorities):
                    raise ValueError("New daily conflicts and window priorities differ")
                selections.setdefault(folder.name, {})[key] = {
                    "priorities": priorities, "selected": selected,
                    "neighbourhood_runs": runs(neighbourhood), "additional_conflict_runs": runs(additional),
                }
                for name, chosen in (("base42", old_choice["selected"]), ("extended84", selected)):
                    coverage = np.zeros(len(initial), dtype=bool)
                    for row in chosen:
                        coverage[row["start"]:row["stop"]] = True
                    opportunity = coverage & weak & (initial != 1)
                    requested = {row["index"] for row in chosen}
                    cached = set(old_choice["fixed_indices"]) | {r["index"] for r in old_choice["selected"]}
                    opportunities.append({"source": folder.name, "case_id": cid, "repeat": rep + 1,
                        "variant": name, "targets": len(chosen), "weak_missed_days_observed": int(opportunity.sum()),
                        "nonactivity_days_in_targets": int(np.count_nonzero(coverage & (actual == 0))),
                        "new_uncached_image_queries": len(requested - cached),
                        "target_indices": ",".join(str(i) for i in sorted(requested))})
        bindings[folder.name] = {str(folder / reference.name): sha(reference),
                                str(folder / "protocol.json"): sha(folder / "protocol.json")}
    save_json(out / "selections.json", selections)
    save_json(out / "source-bindings.json", bindings)
    for file, rows in (("trigger-evidence.csv", evidence_rows), ("observation-opportunities.csv", opportunities)):
        with (out / file).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    save_json(out / "manifest.json", {"new_vlm_requests": 0,
        "files": {str(path.relative_to(out)): sha(path) for path in out.rglob("*") if path.is_file()}})
    print("Longer-scale trigger and bounded-observation probe completed without API requests", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    probe(args.out)


if __name__ == "__main__":
    main()
