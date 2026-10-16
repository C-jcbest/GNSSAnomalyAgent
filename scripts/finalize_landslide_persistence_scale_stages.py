"""Export the assistant's explicit visual decisions with measured rate evidence.

The stage plan and the corrections below are authored decisions. Numerical
summaries describe the evidence; they do not generate stage labels.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

CORRECTIONS = {
    ("case_0219", 515, 535): (
        "turning_unclear", "61日速度0.388→0.452→0.509，91日0.382→0.418→0.450；不能确认匀速。"),
    ("case_0221", 550, 575): (
        "insufficient_rate_support", "91日仅14个有效日且首段无值，31日先升后降；无法确认整个区间减速。"),
    ("case_0226", 275, 300): (
        "insufficient_rate_support", "91日完全无支持，61日仅14日且末段无值；保留减速候选为未知。"),
    ("case_0231", 180, 190): (
        "window_conflict", "31日0.352→0.349→0.374，与61/91日下降方向冲突，短段不赋确定减速。"),
    ("case_0233", 515, 545): (
        "window_conflict", "31日0.339→0.359→0.294，61/91日较平；无法确认逐日匀速阶段。"),
    ("case_0234", 150, 160): (
        "measurement_artifact", "原始EU偏移附近，31日先降后升；长窗可能已受随后跳变影响，不能确认加速。"),
    ("case_0234", 260, 285): (
        "window_conflict", "31日0.395→0.349→0.377，长窗下降；局部反弹使阶段判定保留未知。"),
}

UNKNOWN_EVIDENCE = {
    "turning_unclear": "原始持续位移已确认，但斜率变化或近平台转折的阶段边界尚不明确。",
    "window_conflict": "原始持续累积可见，短窗受噪声影响，跨时间尺度不能支持一致的逐日阶段结论。",
    "insufficient_rate_support": "原始位移支持活动；缺口或拟合中断使阶段速率支持不足。",
    "measurement_artifact": "原始位移中有额外分量偏移，不能把偏移造成的速度峰当作运动加速。",
}


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def finalize(directory: Path) -> None:
    target = directory / "stage-decisions.json"
    if target.exists():
        raise FileExistsError("Preserve authored decisions; do not overwrite")
    plan = read_json(directory / "stage-plan.json")
    checks = read_json(directory / "stage-plan-rate-checks.json")
    activity = read_json(directory / "activity-decisions.json")
    cases = {}
    changes = []
    for cid, candidates in plan["cases"].items():
        rows = []
        for candidate, check in zip(candidates, checks[cid], strict=True):
            start, stop, feature, reason = candidate
            correction = CORRECTIONS.get((cid, start, stop))
            if correction is not None:
                reason, explanation = correction
                changes.append({"case_id": cid, "start": start, "stop": stop,
                                "before": feature, "after": "unknown", "evidence": explanation})
                feature = "unknown"
            else:
                explanation = ""
            span = next(span for span in activity["cases"][cid]
                        if span[2] == "activity" and span[0] <= start < span[1])
            measured = []
            for window, values in check["windows"].items():
                thirds = [None if value is None else round(value, 6)
                          for value in values["speed_thirds_mm_day"]]
                measured.append(f"{window}日窗：速度三段中位数{thirds} mm/day，"
                                f"速度有效{values['speed_valid_days']}日、"
                                f"切向加速度有效{values['tangential_valid_days']}日")
            raw = span[3]
            nonzero = ""
            steady = ""
            if feature == "unknown":
                raw += UNKNOWN_EVIDENCE[reason] + explanation
            elif feature == "steady_motion":
                raw += "原始曲线活动中部保持近似线性累积，未把位移平台标为匀速。"
                nonzero = "原始位移持续累积；三尺度拟合速度均明显非零，详见本段数值证据。"
                steady = "对照原始斜率及辅助图，速度在本段近似平台；允许短窗噪声和缓慢变化，不要求严格数学恒速。"
            elif feature == "acceleration":
                raw += "活动内部原始曲线斜率逐步增大，三尺度速度总体上升。"
            else:
                raw += "活动内部原始曲线逐步收敛，三尺度速度总体下降；局部小波动不延长阶段边界。"
            rows.append([start, stop, feature, raw, "；".join(measured),
                         nonzero, steady, reason])
        cases[cid] = rows
    result = {
        "authored_at": datetime.now(timezone.utc).isoformat(),
        "authorship": "Codex actual raw and auxiliary visual inspection, then numerical assistance. Explicit plan and seven authored corrections; no automatic stage-classification algorithm.",
        "specific_process_labels_seen": False, "specific_predictions_seen": False,
        "columns": ["start", "stop", "feature", "raw_evidence", "rate_evidence",
                    "nonzero_evidence", "steady_evidence", "unknown_reason"],
        "cases": cases, "post_visual_numeric_corrections": changes,
    }
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Authored {len(cases)} records, {sum(map(len, cases.values()))} stage segments; "
          f"{len(changes)} explicit corrections")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("review_directory", type=Path)
    finalize(parser.parse_args().review_directory.resolve())
