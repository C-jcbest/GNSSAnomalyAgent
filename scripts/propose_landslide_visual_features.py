"""Request bounded GNSS feature definitions from original training windows."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from probe_landslide_endpoint_fit import ROOT, export
from probe_landslide_visual_invariance import TRAINING, calibration_labels
from run_landslide_ai_reference_holdout import verify_bindings
from run_landslide_comparison import save_json
from run_landslide_window_optimization import load_case_features, read_json

from gnss_sim.artifacts import sha
from gnss_sim.landslide_feature_discovery import discovery_block_matrix
from gnss_sim.landslide_visual import VisualRequests

ARMS = ("visual_numeric", "visual_only", "numeric_only")
TASK = """Design three numerical evidence features for sustained GNSS displacement
at the CENTER date. Example labels describe only the center, not the whole window.
Weak cumulative motion is active. An elevated stationary plateau, one isolated step
and measurement noise are not. Prefer ongoing change over historical displacement.
Do not infer latent mechanisms, physical failure or unobserved observations.

Each function takes m: a (13,3) numpy array of chronological nonoverlapping 14-day
NEU block medians. Relative block centers are -84.5+14*j days, j=0..12.
Each block requires >=8 complete original NEU samples. Unavailable slots are NaN.
Do not compress, interpolate, drop time slots or replace missing displacement by zero.
Values are divided by daily-difference noise proxies with reference offsets removed;
these units are not confidence probabilities. The queried date lies in block6.
Long movement elsewhere need not imply center activity. Use only supported contiguous
block increments. Return float('nan') if required evidence is absent, otherwise a scalar.

Use only numpy as np, m and local variables. No imports, files, network, eval, exec,
environment, random or other libraries. Keep each function <=12 lines, with descriptive
snake_case names. No hardcoded example IDs/dates/levels or example-fitted thresholds.
No anomaly threshold or final class. The downstream detector and validation scores
are not provided. Proposed code will be reviewed BEFORE execution. No score-based retry.

Return exactly JSON {"functions":[{"name":"NAME","purpose":"raw evidence measured",
"code":"def NAME(m):\\n    ..."}]} with exactly THREE definitions, each a complete
Python function. No Markdown fences or additional keys. Optional plots/tables are
different presentations of the same original training observations.
"""


def prepare(out):
    out.mkdir(parents=True, exist_ok=False)
    source = TRAINING["primary_process"]
    train = read_json(source / "protocol.json")["split"]["train"]
    pairs = {cid: load_case_features(source, cid) for cid in train}
    labels = calibration_labels(source, pairs)
    candidates, blocks = [], {}
    for cid, (case, features) in pairs.items():
        values = np.array([row if row is not None else [np.nan] * 3 for row in case.displacement_mm])
        blocks[cid] = discovery_block_matrix(values, features.noise)
        for day in range(98, len(values) - 98, 14):
            row = labels[cid][day]
            code = int(row["activity_label"])
            if code < 0 or blocks[cid][1][day].sum() < 10 or not features.observed[day]:
                continue
            kind = "inactive"
            if code == 1:
                kind = "weak_active" if int(row.get("weak_signal", 0)) == 1 else "other_active"
            candidates.append({"case_id": cid, "day": day, "kind": kind})
    rng = np.random.default_rng(20261015)
    selected, used = [], set()
    for kind in ("inactive", "weak_active", "other_active"):
        members = [row for row in candidates if row["kind"] == kind]
        count = 0
        for index in rng.permutation(len(members)):
            row = members[index]
            if row["case_id"] in used:
                continue
            selected.append(row)
            used.add(row["case_id"])
            count += 1
            if count == 2:
                break
        if count != 2:
            raise ValueError("Six distinct-case training examples are unavailable")
    tables = []
    fig, panels = plt.subplots(6, 3, figsize=(15, 11))
    for index, row in enumerate(selected):
        case, features = pairs[row["case_id"]]
        day = row["day"]
        values = np.array([value if value is not None else [np.nan] * 3 for value in case.displacement_mm])
        normalized, available, counts, medians = blocks[row["case_id"]]
        matrix = [[round(float(x), 6) if np.isfinite(x) else None for x in value] for value in normalized[day]]
        tables.append({"example": index + 1, "center_label": row["kind"], "normalized_neu_blocks": matrix,
                       "available": available[day].tolist(), "complete_days": counts[day].tolist()})
        for axis in range(3):
            panel = panels[(index // 3) * 3 + axis, index % 3]
            local = values[day - 91 : day + 91, axis]
            panel.plot(np.arange(-91, 91), local, linewidth=.6, color=("#147d72", "#cc6d4b", "#a38a32")[axis])
            panel.plot(-84.5 + 14 * np.arange(13), medians[day, :, axis], "o", markersize=3, color="#444444")
            panel.axvline(0, color="#426da8", linewidth=.8)
            span = max(10, 6 * features.noise[axis], np.nanmax(local) - np.nanmin(local))
            center = (np.nanmax(local) + np.nanmin(local)) / 2
            panel.set_ylim(center - .65 * span, center + .65 * span)
            panel.set_xlim(-91, 90)
            panel.set_xticks([-84, -42, 0, 42, 84])
            panel.set_ylabel(("N", "E", "U")[axis] + " mm")
            panel.grid(alpha=.2)
            if axis == 0:
                panel.set_title(f"Example {index + 1}: center {row['kind']}")
            if axis == 2:
                panel.set_xlabel("Original days relative to center0")
        export(out / f"example-{index + 1}-input.json", case.model_dump(mode="json"))
    fig.suptitle("Six training windows; center labels do not label the whole window")
    fig.tight_layout()
    fig.savefig(out / "training-atlas.png", dpi=120)
    plt.close(fig)
    export(out / "selected-training-examples.json", selected)
    export(out / "candidate-training-windows.json", candidates)
    export(out / "training-block-tables.json", tables)
    packets = {}
    for arm in ARMS:
        prompt = TASK + "\nCenter labels JSON:\n" + json.dumps([{key: row[key] for key in ("example", "center_label")} for row in tables])
        if arm != "visual_only":
            prompt += "\nMeasured blocks JSON:\n" + json.dumps(tables, separators=(",", ":"))
        packets[arm] = {"prompt": prompt, "images": ["training-atlas.png"] if arm != "numeric_only" else []}
    export(out / "packets.json", packets)
    export(out / "protocol.json", {
        "scope": "Bounded offline feature discovery; no feature execution or effectiveness scores yet",
        "workflow": "academic-research-suite/experiment-agent run/validate inline",
        "model": "qwen3.8-flash", "budget": 6, "repetitions": 2, "features_per_pool": 3,
        "arms": list(ARMS), "examples": "Two inactive/weakactive/otheractive; six distinct process training records; seed20261015",
        "generator_input": "Only original training plots, center labels and/or block measurements; no validation cases, fitted detector or scores supplied",
        "author_exposure": "Author knows historical development errors; training-only remote packets do not restore author blindness",
        "novelty_boundary": "Generic VLM feature discovery exists in arXiv2609.21801v1; GNSS mechanism differences and actual gains still unproven",
        "review_contract": "Review all definitions before any execution; invalid functions recorded, no generation retry or score-based redesign",
        "next_comparison": "Freeze reviewed pools before same24/12 and6/6 calibration. Original support/stage head. All known training days; XGB120trees depth2/4 lr.05 seed20261007,19 thresholds,38 opportunities per arm. No per-feature subset search. Compare base18, generic expert pool, all block information, generated-only and base-union pools; retain both generated repetitions. Main unions must both improve dailyF1/net vs base, benefit >=2 records, improve weak/event recall or FP count, and stay within .02 F1 of all-information control. Multimodal necessity additionally requires main above visual-only and numeric-only generation controls; otherwise do not claim it. If usefulness gate fails stop fixed pools without regeneration/seed/window rescue.",
        "independent_confirmation": False, "new_records": 0,
    })
    export(out / "schedule.json", [{"arm": arm, "repetition": repetition, "request_id": f"{arm}-r{repetition}"}
                                  for repetition in range(2) for arm in rng.permutation(ARMS).tolist()])
    binding = {}
    for path in [Path(__file__), ROOT / "src/gnss_sim/landslide_feature_discovery.py",
                 ROOT / "tests/test_landslide_feature_discovery.py", ROOT / "src/gnss_sim/landslide_visual.py"]:
        copy = out / "source" / path.relative_to(ROOT)
        copy.parent.mkdir(parents=True, exist_ok=True)
        copy.write_bytes(path.read_bytes())
        binding[path.relative_to(ROOT).as_posix()] = sha(path)
    export(out / "source-bindings.json", binding)
    export(out / "state.json", {"status": "prepared", "new_api_calls": 0})
    export(out / "pre-proposal-freeze.json", {path.relative_to(out).as_posix(): sha(path)
                                             for path in out.rglob("*") if path.is_file() and path.name != "state.json"})
    print("Prepared six original training windows and six bounded generation requests", flush=True)


def propose(out):
    verify_bindings(out, "pre-proposal-freeze.json")
    for name, expected in read_json(out / "source-bindings.json").items():
        if sha(ROOT / name) != expected:
            raise ValueError("Frozen generation source changed")
    if read_json(out / "state.json")["status"] != "prepared":
        raise ValueError("Generation starts once from prepared inputs")
    packets = read_json(out / "packets.json")
    transport = VisualRequests(out / "requests", ROOT / ".env", "qwen3.8-flash", 6)
    save_json(out / "state.json", {"status": "running", "budget": 6})
    proposals, failures = {}, {}
    for item in read_json(out / "schedule.json"):
        packet = packets[item["arm"]]
        try:
            value = transport.ask(item["request_id"], packet["prompt"], [out / name for name in packet["images"]])
            if set(value) != {"functions"} or len(value["functions"]) != 3:
                raise ValueError("Expected exactly three functions")
            names = []
            for function in value["functions"]:
                if set(function) != {"name", "purpose", "code"}:
                    raise ValueError("Unexpected feature fields")
                tree = ast.parse(function["code"])
                if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
                    raise ValueError("Expected one function definition")
                definition = tree.body[0]
                if definition.name != function["name"] or len(definition.args.args) != 1 or definition.args.args[0].arg != "m":
                    raise ValueError("Generated function signature mismatch")
                names.append(function["name"])
            if len(set(names)) != 3:
                raise ValueError("Duplicate function names")
            proposals[item["request_id"]] = {**item, "proposal": value, "executed": False, "manual_review_required": True}
        except (ValueError, RuntimeError, TypeError, SyntaxError) as error:
            failures[item["request_id"]] = {"error_type": type(error).__name__, "message": str(error)}
        save_json(out / "proposals.json", proposals)
        save_json(out / "failures.json", failures)
        print(item["request_id"] + (" returned unexecuted code" if item["request_id"] in proposals else " failed"), flush=True)
    save_json(out / "state.json", {"status": "proposals_complete", "attempts": 6,
                                   "parsed_pools": len(proposals), "failures": len(failures), "code_executed": False})
    export(out / "proposal-freeze.json", {path.relative_to(out).as_posix(): sha(path)
                                         for path in out.rglob("*") if path.is_file() and path.name not in ("state.json", "proposal-freeze.json")})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "propose"))
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.mode == "prepare":
        prepare(arguments.out.resolve())
    else:
        propose(arguments.out.resolve())
