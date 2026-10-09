"""Expose unknown-stage rate evidence without changing the frozen AI review."""

from __future__ import annotations

import argparse
from pathlib import Path

from gnss_sim.artifacts import sha, write_json

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REVIEW = ROOT / "artifacts/landslide-ai-stage-review-2026-10-08/review-v1"


def build(review_directory: Path) -> Path:
    source = review_directory / "public/index.html"
    source_digest = sha(source)
    html = source.read_text(encoding="utf-8")
    # The added view reads the exact frozen payload and shares its static assets.
    # Unknown remains a scientific decision; showing more evidence never assigns a stage.
    marker = '.join("")}</select></label>`'
    if html.count(marker) != 1:
        raise ValueError("Inspect source: unknown-segment renderer no longer matches")
    html = html.replace(
        marker,
        '.join("")}</select></label><label>速率依据／辅助冲突说明'
        '<textarea data-index="${i}" data-field="rate_evidence">'
        '${escapeHtml(s.rate_evidence)}</textarea></label>`',
    )
    html = html.replace("<head>", '<head>\n    <base href="../public/">', 1)
    html = html.replace("gnss-ai-stage-review-v1:", "gnss-ai-stage-review-presentation-v1:")
    notice = "这是已接触历史预测的AI开发标注，不能作为独立参考或准确率证明。"
    if html.count(notice) != 1:
        raise ValueError("Inspect source: AI disclosure no longer matches")
    html = html.replace(
        notice,
        notice + "此展示补充版展开未定片段的速率／多窗冲突依据，标签与冻结版本相同。",
    )
    output_directory = review_directory / "presentation-v1"
    output_directory.mkdir(exist_ok=False)
    page = output_directory / "index.html"
    page.write_text(html, encoding="utf-8")
    if sha(source) != source_digest:
        raise ValueError("Frozen source changed while building presentation")
    write_json(output_directory / "presentation-freeze.json", {
        "source_page_sha256": source_digest,
        "review_freeze_sha256": sha(review_directory / "review-freeze.json"),
        "implementation_sha256": sha(Path(__file__)),
        "page_sha256": sha(page),
        "changes": ["show unknown rate_evidence", "shared assets base", "isolated draft storage"],
        "label_changes": False,
        "browser_rendering": "not_verified",
    })
    return page


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-directory", type=Path, default=DEFAULT_REVIEW)
    args = parser.parse_args()
    print(build(args.review_directory.resolve()))


if __name__ == "__main__":
    main()
