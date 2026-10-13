"""Read saved PDFs/HTML to measure section allocation; never edit the draft.

The primary comparison counts alphabetic characters (including Chinese),
excluding whitespace, punctuation, and numeric-only table entries. Images have
no weight apart from their extractable text. Section page spans are a separate,
approximate layout measure; double-column floats can cross section boundaries.
"""

from bisect import bisect_right
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from lxml import html
import pdfplumber


REPO = Path(__file__).resolve().parents[2]
WORKING = Path(__file__).resolve().parent
CACHE = WORKING / "render-structure"


def letter_count(text):
    return sum(character.isalpha() for character in text)


def html_sections(name, section_ids):
    path = REPO / "papers/evidence" / f"{name}_arxiv_html.html"
    tree = html.fromstring(path.read_text(encoding="utf-8"))
    sections = []
    for section_id in section_ids:
        node = deepcopy(tree.get_element_by_id(section_id))
        heading = node.xpath("./h2")[0].text_content()
        # MathML annotation duplicates the visible mathematical expression.
        for annotation in node.xpath("//*[local-name()='annotation']"):
            annotation.getparent().remove(annotation)
        sections.append({
            "section": " ".join(heading.split()),
            "letters": letter_count(node.text_content()),
        })
    return path, sections


def pdf_sections(path, starts, end, top, bottom, columns=1):
    anchors = [(page, column, y) for _, page, column, y in starts]
    counts = [0] * len(starts)
    with pdfplumber.open(path) as document:
        for page_number, page in enumerate(document.pages, 1):
            if page_number < starts[0][1] or page_number > end[0]:
                continue
            for character in page.chars:
                y = character["top"]
                if not top <= y <= bottom:
                    continue
                if path.name == "crocetti.pdf" and page_number == 1 and character["x0"] < 160:
                    continue
                column = int(columns == 2 and character["x0"] >= page.width / 2)
                position = (page_number, column, y)
                if position < anchors[0] or position >= end:
                    continue
                index = bisect_right(anchors, position) - 1
                counts[index] += letter_count(character["text"])

    def page_position(anchor):
        page, column, y = anchor
        return page - 1 + column / columns + (y - top) / (bottom - top) / columns

    stops = anchors[1:] + [end]
    sections = []
    for index, (name, _, _, _) in enumerate(starts):
        sections.append({
            "section": name,
            "letters": counts[index],
            "start": anchors[index],
            "stop": stops[index],
            "approximate_page_span": page_position(stops[index]) - page_position(anchors[index]),
        })
    return sections


def add_percentages(sections):
    total = sum(section["letters"] for section in sections)
    for section in sections:
        section["text_percent"] = 100 * section["letters"] / total
    if all("approximate_page_span" in section for section in sections):
        total_pages = sum(section["approximate_page_span"] for section in sections)
        for section in sections:
            section["page_span_percent"] = 100 * section["approximate_page_span"] / total_pages


def main():
    records = []
    for name, ids in [
        ("mllm", [f"S{index}" for index in range(1, 8)]),
        ("tama", [f"S{index}" for index in range(1, 7)]),
        ("vlm4ts", [f"Sx{index}" for index in range(1, 6)]),
    ]:
        path, sections = html_sections(name, ids)
        records.append({"id": name, "source": str(path.relative_to(REPO)), "sections": sections})

    specifications = [
        ("crocetti", REPO / "papers/evidence/crocetti.pdf", 72, 778,
         [("Introduction", 1, 0, 543.5), ("Data", 3, 0, 509.7),
          ("Methods", 5, 0, 437.7), ("Results", 9, 0, 434.4),
          ("Conclusions", 20, 0, 555.7)], (21, 0, 372.0)),
        ("timesnet", REPO / "papers/evidence/gnss_timesnet.pdf", 38, 753,
         [("Introduction", 2, 0, 505.4), ("Materials and Methods", 3, 0, 734.1),
          ("Results and discussion", 9, 0, 314.1), ("Conclusion", 15, 0, 557.1)],
         (16, 0, 581.3)),
        ("draft", WORKING / "render-final/paper.pdf", 67, 778,
         [("1 引言", 3, 0, 76.4), ("2 相关工作", 5, 0, 67.4),
          ("3 方法", 7, 0, 389.3), ("4 实验设置", 12, 0, 362.0),
          ("5 结果分析", 17, 0, 423.0), ("6 讨论", 26, 0, 67.4),
          ("7 结论", 28, 0, 438.3)], (29, 0, 138.2)),
    ]
    for name, path, top, bottom, starts, end in specifications:
        records.append({
            "id": name,
            "source": str(path.relative_to(REPO)),
            "pdf_text_area_y": [top, bottom],
            "sections": pdf_sections(path, starts, end, top, bottom),
        })
    for record in records:
        path = REPO / record["source"]
        record["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        add_percentages(record["sections"])
        print(record["id"])
        for section in record["sections"]:
            print(section["section"].encode("ascii", "backslashreplace").decode(),
                  f"text={section['text_percent']:.1f}%",
                  f"pages={section.get('approximate_page_span', 0):.2f}",
                  f"page_share={section.get('page_span_percent', 0):.1f}%")
    result = {
        "date": "2026-10-13",
        "scope": "Approximate section allocation, not a publication requirement or quality score.",
        "metric": "Alphabetic visible-text characters; normalise within each paper. Main body only.",
        "exclusions": "Abstract, references, acknowledgments, declarations and appendices. TimesNet end-matter figure plates excluded.",
        "html_pdf_difference": "HTML uses semantic section ownership; PDF uses observed heading coordinates. Formula/table extraction can differ slightly.",
        "records": records,
    }
    (WORKING / "section-allocation-measurements.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
