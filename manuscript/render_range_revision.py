"""Rasterize this revision through the document skill after native Word PDF export."""

from __future__ import annotations

import importlib.util
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RENDER = ROOT / "working/render-range-20261018"
DOCUMENT = ROOT / "GNSS多类型异常识别论文初稿_范围视觉与形变更新版.docx"
DEPENDENCIES = Path("C:/Users/JC/.cache/codex-runtimes/codex-primary-runtime/dependencies")
RENDERER = Path(
    "C:/Users/JC/.codex/plugins/cache/openai-primary-runtime/documents/26.1015.12040/skills/documents/render_docx.py"
)


def native_pdf(doc_path, user_profile, convert_tmp_dir, stem, verbose):
    pdf = RENDER / "paper.pdf"
    if pdf.stat().st_mtime < DOCUMENT.stat().st_mtime:
        raise ValueError("PDF is older than the revised Word document; export again")
    destination = Path(convert_tmp_dir) / f"{stem}.pdf"
    shutil.copyfile(pdf, destination)
    return (
        str(destination),
        "PDF exported by native Microsoft Word; see renderer-diagnostic.log for unavailable LibreOffice",
    )


def main():
    os.environ["PATH"] = (
        str(DEPENDENCIES / "native/poppler/Library/bin") + os.pathsep + os.environ["PATH"]
    )
    spec = importlib.util.spec_from_file_location("document_skill_renderer", RENDERER)
    renderer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(renderer)
    renderer.convert_to_pdf = native_pdf
    pages = renderer.rasterize(str(DOCUMENT), str(RENDER), 120, False, False)
    print(f"Rendered {len(pages)} pages with the document skill rasterizer")


if __name__ == "__main__":
    main()
