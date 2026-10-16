"""Post-score diagnosis of two previously observed local-revision failures."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from analyze_landslide_codex_observation_reference import scoring_inputs
from probe_landslide_counterevidence_review import BASE, HOLDOUT
from run_landslide_window_optimization import load_case_features, read_json

from gnss_sim.artifacts import write_json
from gnss_sim.landslide_counterevidence_review import raw_counterevidence
from gnss_sim.landslide_evaluation import runs


def render(out, destination):
    destination.mkdir(parents=True, exist_ok=False)
    predictions = read_json(out / "predictions.json")["new-AI"]
    vetoes = read_json(out / "vetoes.json")["new-AI"]
    labels = scoring_inputs(BASE / "ai-observable-holdout-data-v1/codex-review")[0]
    fig, panels = plt.subplots(5, 2, figsize=(15, 11), sharex="col")
    diagnostics = {}
    for column, (cid, repeat, start, stop) in enumerate((("case_0139", 0, 0, 240), ("case_0143", 1, 220, 440))):
        case, feature = load_case_features(HOLDOUT / "run", cid)
        raw = np.array([row if row is not None else [np.nan] * 3 for row in case.displacement_mm])
        evidence = raw_counterevidence(case, feature.noise)
        actual = np.array([int(row["activity_label"]) for row in labels[cid]])
        for axis, name in enumerate(("N", "E", "U")):
            panels[axis, column].plot(np.arange(start, stop), raw[start:stop, axis], linewidth=.8, color="#344b66")
            panels[axis, column].set_ylabel(name + " (mm)")
        for panel in panels[:, column]:
            for left, right in runs(actual[start:stop] == 1):
                panel.axvspan(start + left, start + right, color="#288b69", alpha=.12)
            for left, right in runs(actual[start:stop] == -1):
                panel.axvspan(start + left, start + right, color="#888888", alpha=.10)
            panel.grid(alpha=.15)
            panel.set_xlim(start, stop - 1)
        for name, color, style in (("direct", "#999999", "-"), ("pure", "#2161a6", "-"),
                                    ("counterevidence", "#df8332", "--"), ("numeric_vote", "#7c3f91", ":")):
            activity = predictions[f"{name}_r{repeat}"][cid]["activity"]
            panels[3, column].step(np.arange(start, stop), activity[start:stop], where="mid",
                                   color=color, linestyle=style, label=name, linewidth=1.6)
        panels[3, column].set_yticks([-1, 0, 1], ["unknown", "still", "active"])
        panels[3, column].set_ylim(-1.15, 1.15)
        panels[3, column].legend(fontsize=8, ncol=2, loc="lower left")
        eligible = np.zeros(len(raw), dtype=bool)
        eligible[vetoes[f"counterevidence_r{repeat}"][cid]["eligible_days"]] = True
        for level, key, color in ((3, "eligible", "#444444"), (2, "supported", "#777777"),
                                   (1, "plateau", "#2377aa"), (0, "motion", "#bc6f2a")):
            mask = eligible if key == "eligible" else evidence[key]
            days = np.flatnonzero(mask & (np.arange(len(raw)) >= start) & (np.arange(len(raw)) < stop))
            panels[4, column].scatter(days, np.full(len(days), level), marker="|", s=35, color=color)
        panels[4, column].set_yticks([0, 1, 2, 3], ["motion", "plateau", "support", "edit"])
        panels[4, column].set_ylim(-.5, 3.5)
        panels[4, column].set_xlabel("Day")
        panels[0, column].set_title(f"{cid} / repeat{repeat}: counterevidence retains pure output")
        subsets = {}
        initial = np.array(predictions[f"direct_r{repeat}"][cid]["activity"])
        pure = np.array(predictions[f"pure_r{repeat}"][cid]["activity"])
        for description, mask in (("new_activity_edits", eligible & (initial == 0) & (pure == 1)),
                                   ("removed_activity_edits", eligible & (initial == 1) & (pure == 0))):
            subsets[description] = {"days": np.flatnonzero(mask).tolist(),
                                    **{key + "_days": int(np.count_nonzero(mask & evidence[key]))
                                       for key in ("supported", "plateau", "motion")}}
        diagnostics[cid] = {"repeat": repeat, "subsets": subsets, "noise_mm": feature.noise.tolist(),
                             "evidence_by_scale": {str(scale): {
                                 key: [None if not np.isfinite(value) else float(value) for value in values]
                                 for key, values in arrays.items()} for scale, arrays in evidence["scales"].items()}}
    fig.suptitle("Post-score failure diagnosis; green = confirmed interior, gray = unknown reference", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, .975))
    fig.savefig(destination / "failures.png", dpi=160)
    plt.close(fig)
    write_json(destination / "diagnostics.json", diagnostics)
    print({cid: row["subsets"] for cid, row in diagnostics.items()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    render(args.out.resolve(), args.destination.resolve())
