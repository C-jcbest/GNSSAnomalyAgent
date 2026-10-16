"""Cross-check reversible workbook date recovery against official NGL records."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

MONTHS = {name: index for index, name in enumerate(
    ("JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"), 1)}


def ngl_calendar(code):
    return datetime(2000 + int(code[:2]), MONTHS[code[2:5]], int(code[5:]))


def check(field, out):
    out.mkdir(parents=True, exist_ok=False)
    report = {
        "scope": "Date provenance verification only; no field predictions or labels",
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "stations": {},
    }
    for station in ("OREO", "ORE2"):
        source = field / "ngl-source-v1" / f"{station}.tenv3"
        if not source.exists():
            raise ValueError(f"Official daily component record is required: {source}")
        lines = source.read_text().splitlines()
        headers = lines[0].split()
        records = [dict(zip(headers, line.split())) for line in lines[1:] if line.strip()]
        dates = {ngl_calendar(row["YYMMMDD"]): row for row in records}
        exported = []
        decimal_errors = []
        unmatched = []
        with (field / "audit" / f"{station}-raw.csv").open(newline="", encoding="utf-8") as handle:
            for ordinal, row in enumerate(csv.DictReader(handle), 2):
                raw = datetime.fromisoformat(row["YYMMMDD"])
                recovered = datetime(2000 + raw.day, raw.month, raw.year % 100)
                native = dates.get(recovered)
                if native is None:
                    unmatched.append(ordinal)
                else:
                    decimal_errors.append(abs(float(row["yyyy.yyyy"]) - float(native["yyyy.yyyy"])))
                exported.append({
                    "workbook_row": ordinal, "raw_excel_date": row["YYMMMDD"],
                    "recovered_candidate_date": recovered.date().isoformat(),
                    "workbook_decimal_year": row["yyyy.yyyy"],
                    "source_scalar_displacement": row["3D Displacement_Corrected"],
                    "ngl_date_code": native["YYMMMDD"] if native else "",
                    "ngl_mjd": native["__MJD"] if native else "",
                    "ngl_decimal_year": native["yyyy.yyyy"] if native else "",
                })
        candidate_dates = [row["recovered_candidate_date"] for row in exported]
        nonincreasing = sum(right <= left for left, right in zip(candidate_dates, candidate_dates[1:]))
        maximum_error = max(decimal_errors) if decimal_errors else None
        verified = not unmatched and maximum_error is not None and maximum_error <= 1e-8 and nonincreasing == 0
        with (out / f"{station}-date-crosscheck.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(exported[0]))
            writer.writeheader()
            writer.writerows(exported)
        report["stations"][station] = {
            "rows": len(exported), "official_date_matches": len(exported) - len(unmatched),
            "unmatched_workbook_rows": unmatched, "maximum_decimal_year_difference": maximum_error,
            "recovered_date_nonincreasing_pairs": nonincreasing,
            "recovered_date_range": [candidate_dates[0], candidate_dates[-1]],
            "status": "verified_against_official_dates" if verified else "unresolved",
            "recovery": "date(2000 + excel.day, excel.month, excel.year % 100)",
            "official_component_header": headers,
            "official_frame": "IGS20 (current archive), not article IGS14 processed data",
            "official_source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        }
    article = ET.parse(field / "article-fulltext.xml")
    excerpts = []
    for paragraph in article.iter("p"):
        text = " ".join("".join(paragraph.itertext()).split())
        if any(phrase in text for phrase in ("Trimble NetR9", "~0.54 m at OREO", "We use the OREO GPS")):
            excerpts.append(text)
    report["article_evidence"] = {
        "doi": "10.1126/sciadv.adq9399", "pmc": "PMC11482306", "excerpts": excerpts,
        "article_sha256": hashlib.sha256((field / "article-fulltext.xml").read_bytes()).hexdigest(),
    }
    report["unit_and_task_decision"] = (
        "Official component columns explicitly use meters. The workbook scalar column has no unit header; "
        "article meter-scale GPS displacement supports, but does not independently prove, its unit mapping. "
        "Preserve workbook source scalar units. Current IGS20 components are not a byte-identical reproduction "
        "of the article's IGS14 and P227 horizontal correction. No invented ENU, interpolation or field score."
    )
    (out / "date-provenance.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# Oak Ridge 日期来源核验", "", "逐行将工作簿候选恢复日期与官方NGL日期和十进制年份对照。",
             "原工作簿不修改，导出同时保留原日期、候选日期和官方日码。", "",
             "| 站点 | 工作簿行 | 官方日期匹配 | 十进制年份最大差 | 状态 |",
             "|---|---:|---:|---:|---|"]
    for station, result in report["stations"].items():
        lines.append(f"| {station} | {result['rows']} | {result['official_date_matches']} | "
                     f"{result['maximum_decimal_year_difference']} | {result['status']} |")
    lines.extend(["", "官方NGL三分量表头标明米；正文说明2016–2024年的OREO记录、"
                  "IGS14以及P227稳定站水平运动校正。当前官方档案是IGS20，"
                  "不能冒充原论文的逐字节复现。工作簿标量单位尚无直接表头映射证明，未换算。",
                  "", "这解决日期来源核查，不建立活动／A/S/D参考，也不产生现场效果结论。"])
    (out / "date-provenance.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(report["stations"], ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--field", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    check(args.field, args.out)


if __name__ == "__main__":
    main()
