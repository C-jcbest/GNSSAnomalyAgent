"""Post-hoc P8a common-success/format diagnostics; primary denominators stay intact."""
import json
from collections import Counter
from pathlib import Path

from gnss_sim.evaluation_views import evaluate_views
from gnss_sim.input_only import sha, write_json
from gnss_sim.p8a import CONDITIONS, ROOT
from gnss_sim.schemas import CaseTruth, RangeResult


def main():
    directory = ROOT / "runs/p8a/experiment"
    run = json.loads((directory / "run.json").read_bytes())
    for name, digest in run["artifacts_sha256"].items():
        if sha(directory / name) != digest:
            raise ValueError("frozen artifact changed")
    rows = {}
    formats = {}
    for c in CONDITIONS:
        rows[c] = {r.case_id: r for r in [RangeResult.model_validate_json(line) for line in
            (directory / c / "predictions.jsonl").read_text(encoding="utf-8").splitlines()]}
        records = [json.loads(p.read_bytes())["record"] for p in
                   (directory / "requests" / c).glob("*/completed.json")]
        counts = Counter()
        for record in records:
            try:
                counts[type(json.loads(record.get("response") or "" )).__name__] += 1
            except ValueError:
                counts["not_plain_json"] += 1
        formats[c] = {"top_level_json_types": dict(counts),
            "max_completion_tokens": max((r.get("usage") or {}).get("completion_tokens", 0) for r in records),
            "failed": [{k: r.get(k) for k in ("case_id", "error_type", "finish_reason")}
                       for r in records if r["status"] != "success"]}
    ids = {cid for cid in rows["V0"] if all(rows[c][cid].status == "success" for c in CONDITIONS)}
    selection = json.loads((ROOT / "runs/p7b/prepared/selection.json").read_bytes())
    truths = {}
    for entry in selection["cases"]:
        cid = entry["case_id"]
        if cid in ids:
            path = ROOT / "data/pilots/pilot-v1/cases" / cid / "truth.json"
            if sha(path) != entry["truth_sha256"]:
                raise ValueError("truth changed")
            truths[cid] = CaseTruth.model_validate_json(path.read_bytes())
    report = {"scope": "post-hoc diagnostic only; excludes failures, not primary ranking",
        "source_sha256": sha(Path(__file__)), "run_sha256": sha(directory / "run.json"),
        "common_success_cases": len(ids), "formats": formats,
        "common_success": {c: evaluate_views(truths, {cid: rows[c][cid] for cid in ids}, "range")
                           for c in CONDITIONS}}
    write_json(ROOT / "runs/p8a/diagnostics.json", report)
    print(json.dumps({"common_success_cases": len(ids), "formats": formats}, indent=2))


if __name__ == "__main__":
    main()
