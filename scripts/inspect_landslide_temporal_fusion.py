"""Export raw observation evidence for the developer's post-score visual inspection."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from run_landslide_window_optimization import load_case_features, read_json

from gnss_sim.artifacts import write_json
from gnss_sim.landslide_evaluation import runs

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/landslide-adaptive-review-2026-10-09"
RUN = BASE / "temporal-fusion-development-v1"
OUT = BASE / "temporal-fusion-developer-inspection-v1"
WINDOWS = {"case_0159": (420, 730), "case_0163": (480, 680), "case_0164": (260, 440)}
BLOCKS = {
    "case_0159": [(500, 530), (530, 560), (570, 600), (600, 630), (630, 660)],
    "case_0160": [(255, 275), (295, 315), (340, 360)],
    "case_0161": [(300, 330), (350, 380), (390, 420)],
    "case_0163": [(520, 550), (550, 580), (580, 610), (610, 640)],
    "case_0164": [(260, 280), (280, 294), (294, 314), (390, 410), (410, 430)],
    "case_0167": [(210, 240), (270, 300), (330, 360)],
}


def export():
    OUT.mkdir(exist_ok=False)
    (OUT / "source.py").write_bytes(Path(__file__).read_bytes())
    probabilities = read_json(RUN / "numeric-probabilities.json")["primary_ai"]
    threshold = read_json(RUN / "choices.json")["primary_ai"]["joint"]["threshold"]
    evidence = {}
    for cid, blocks in BLOCKS.items():
        case, feature = load_case_features(RUN, cid)
        values = np.asarray(
            [row if row is not None else [np.nan] * 3 for row in case.displacement_mm]
        )
        summaries = []
        for start, stop in blocks:
            raw = values[start:stop]
            summaries.append(
                {
                    "start_inclusive": start,
                    "stop_exclusive": stop,
                    "complete_observations": int(feature.observed[start:stop].sum()),
                    "numeric_supported_days": int(feature.supported[start:stop].sum()),
                    "raw_NEU_median_mm": np.nanmedian(raw, axis=0).tolist(),
                }
            )
        evidence[cid] = {
            "blocks": summaries,
            "numeric_unsupported_intervals": runs(feature.observed & ~feature.supported),
        }
        if cid not in WINDOWS:
            continue
        start, stop = WINDOWS[cid]
        days = np.arange(start, stop)
        fig, axes = plt.subplots(4, 1, figsize=(12, 9), sharex=True)
        for index, component in enumerate(("N", "E", "U")):
            axes[index].plot(days, values[start:stop, index], linewidth=0.9)
            axes[index].set_ylabel(f"{component} (mm)")
            axes[index].grid(alpha=0.2)
        probability = np.clip(np.array(probabilities[cid]), 1e-6, 1 - 1e-6)
        score = np.log(probability / (1 - probability)) - np.log(threshold / (1 - threshold))
        score[~feature.supported] = np.nan
        axes[3].plot(days, score[start:stop], linewidth=0.9)
        axes[3].axhline(0, color="gray", linestyle="--")
        axes[3].set_ylabel("Numeric log odds\n(relative to threshold)")
        axes[3].set_xlabel(
            "Original day; raw observations, no interpolation; score is not confidence"
        )
        fig.suptitle(f"{cid} | Developer diagnostic [{start}, {stop}); not a model query")
        fig.tight_layout()
        fig.savefig(OUT / f"{cid}-{start}-{stop}.png", dpi=150)
        plt.close(fig)
    write_json(OUT / "raw-numeric-evidence.json", evidence)


if __name__ == "__main__":
    export()
