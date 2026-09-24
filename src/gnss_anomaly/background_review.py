"""Descriptive background audit; diagnostics never approve normal labels."""

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from .contracts import Window
from .storage import child, file_hash, read_json, write_json


def audit_backgrounds(root: Path, out: Path):
    if out.exists():
        raise FileExistsError(out)
    catalog = read_json(root / "backgrounds.json")
    items = []
    for source in catalog["items"]:
        path = child(root, source["file"])
        if file_hash(path) != source["sha256"]:
            raise ValueError("background checksum mismatch")
        x = Window.model_validate(read_json(path)).array()
        if not np.isfinite(x).all():
            raise ValueError("background audit expects complete candidates")
        delta = np.diff(x, axis=0)
        row = {
            k: source[k]
            for k in ("id", "station_alias", "group", "split", "start", "end_exclusive", "sha256")
        }
        for i, axis in enumerate(("n", "e", "u")):
            row[f"{axis}_range_mm"] = float(np.ptp(x[:, i]))
            row[f"{axis}_max_hourly_jump_mm"] = float(np.max(np.abs(delta[:, i])))
            row[f"{axis}_end_median_shift_mm"] = float(np.median(x[-24:, i]) - np.median(x[:24, i]))
        row["max_range_mm"] = float(np.max(np.ptp(x, axis=0)))
        row["max_hourly_jump_mm"] = float(np.max(np.abs(delta)))
        row["review_status"] = "descriptive_audit_only"
        row["approved"] = False
        items.append(row)
    groups = defaultdict(list)
    for item in items:
        groups[item["group"]].append(item)
    summary = [
        {
            "group": g,
            "split": rows[0]["split"],
            "windows": len(rows),
            "minimum_max_range_mm": min(r["max_range_mm"] for r in rows),
            "minimum_max_hourly_jump_mm": min(r["max_hourly_jump_mm"] for r in rows),
        }
        for g, rows in sorted(groups.items())
    ]
    report = {
        "kind": "descriptive_background_audit",
        "source_catalog_sha256": file_hash(root / "backgrounds.json"),
        "unit_basis": "User confirmed platform coordinates in metres on 2026-09-22; displacement differences in mm.",
        "groups": summary,
        "items": items,
        "note": "No detector score, automatic normal approval, or field-cause label is produced.",
    }
    write_json(out / "audit.json", report)
    pd.DataFrame(items).to_csv(out / "audit.csv", index=False, encoding="utf-8-sig")
    lines = [
        "# 完整候选背景审核",
        "",
        "以下是数据描述统计，不是算法检测结果或正常性标签。",
        "",
        "| 来源组 | 划分 | 窗口数 | 各窗最大范围的最小值 / mm | 各窗最大小时跳变的最小值 / mm |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for row in summary:
        lines.append(
            f"| {row['group']} | {row['split']} | {row['windows']} | {row['minimum_max_range_mm']:.2f} | {row['minimum_max_hourly_jump_mm']:.2f} |"
        )
    lines += [
        "",
        "所有原始候选仍保持 approved=false。显著跳变和趋势需独立审核，不能在已有原生异常上仅标注注入事件，就将其当作完整真值。",
    ]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report
