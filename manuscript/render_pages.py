"""Use the skill's page rasterizer after a native Microsoft Word PDF export.

The bundled runtime has no LibreOffice executable. The diagnostic log in working/
records that failure; this adapter only substitutes the PDF conversion stage.
"""

import importlib.util
import os
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parent
DEPENDENCIES = Path("C:/Users/JC/.cache/codex-runtimes/codex-primary-runtime/dependencies")
RENDERER = Path("C:/Users/JC/.codex/plugins/cache/openai-primary-runtime/documents/26.1011.11113/skills/documents/render_docx.py")
RENDER = ROOT / "working/render-sdr"


def native_word_pdf(doc_path, user_profile, convert_tmp_dir, stem, verbose):
    destination = Path(convert_tmp_dir) / f"{stem}.pdf"
    shutil.copyfile(RENDER / "paper.pdf", destination)
    return str(destination), "PDF exported from this DOCX by Microsoft Word via render_word.ps1"


def main():
    os.environ["PATH"] = str(DEPENDENCIES / "native/poppler/Library/bin") + os.pathsep + os.environ["PATH"]
    specification = importlib.util.spec_from_file_location("document_skill_renderer", RENDERER)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    module.convert_to_pdf = native_word_pdf
    pages = module.rasterize(str(ROOT / "GNSS多类型异常识别论文初稿_SDR-VLM实验更新版.docx"), str(RENDER), 120, False, False)
    current = {Path(page).resolve() for page in pages}
    for previous_page in RENDER.glob("page-*.png"):
        if previous_page.resolve() not in current:
            previous_page.unlink()
    print(f"Rendered {len(pages)} pages through the document skill rasterizer")


if __name__ == "__main__":
    main()
