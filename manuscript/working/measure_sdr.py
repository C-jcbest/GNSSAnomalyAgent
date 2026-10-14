"""Measure the updated paper and verify that evidence and historical drafts remain separate."""
from pathlib import Path
from zipfile import ZipFile
import hashlib
import json
import re
import subprocess
import sys
from docx import Document
from lxml import etree
import pdfplumber
from measure_section_allocation import pdf_sections, add_percentages

ROOT = Path(__file__).resolve().parents[1]
WORKING = ROOT / 'working'
DOCX = ROOT / 'GNSS多类型异常识别论文初稿_SDR-VLM实验更新版.docx'
PDF = WORKING / 'render-sdr/paper.pdf'

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    starts = []
    end = None
    with pdfplumber.open(PDF) as pdf:
        pages = len(pdf.pages)
        for number, page in enumerate(pdf.pages, 1):
            for line in page.extract_text_lines():
                text = line['text']
                if re.match(r'^[1-6] (引言|相关工作|位移结构分解|结果与分析|讨论|结论)', text):
                    starts.append((text, number, 0, line['top']))
                if text == '参考文献':
                    end = (number, 0, line['top'])
    assert len(starts) == 6 and end
    sections = pdf_sections(PDF, starts, end, 67, 778)
    add_percentages(sections)
    (WORKING / 'section-allocation-sdr.json').write_text(json.dumps({'date':'2026-10-14','method':'PDF alphabetic characters; body excludes abstracts and references; layout spans include figures and tables. Same method as prior allocation review.','pdf_sha256':digest(PDF),'sections':sections}, ensure_ascii=False,indent=2),encoding='utf-8')
    document = Document(DOCX)
    with ZipFile(DOCX) as archive:
        tree = etree.fromstring(archive.read('word/document.xml'))
    namespaces = {'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main','m':'http://schemas.openxmlformats.org/officeDocument/2006/math'}
    text = ''.join(tree.xpath('//w:t/text() | //m:t/text()',namespaces=namespaces))
    manifest = json.loads((WORKING/'build-manifest-sdr.json').read_text(encoding='utf-8'))
    historical = {}
    for path in ROOT.glob('*.docx'):
        if path == DOCX or path.name.startswith('~$'):
            continue
        repo_path = path.relative_to(ROOT.parent).as_posix()
        previous = subprocess.check_output(['git','show','HEAD:'+repo_path],cwd=ROOT.parent)
        historical[path.name] = {'sha256':digest(path),'unchanged':path.read_bytes()==previous}
    page_hashes = {p.name:digest(p) for p in sorted((WORKING/'render-sdr').glob('page-*.png'))}
    result = {'experiment':'GNSS-VLM-EXP06-v1','pages':pages,'tables':len(document.tables),'figures':len(document.inline_shapes),'equation_groups':manifest['equations'],'references':len(manifest['citations']),'math_objects':len(tree.xpath('//m:oMath',namespaces=namespaces)),'docx_sha256':digest(DOCX),'historical_drafts':historical,'unresolved':re.findall(r'\[\[FIGURE|\[@|TODO|TBD|\\arg|表\[',text),'rendered_pages':page_hashes}
    assert 20<=pages<=30 and result['tables']==10 and result['figures']==5
    assert all(item['unchanged'] for item in historical.values())
    assert not result['unresolved'],result['unresolved']
    (WORKING/'final-integrity-sdr.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'pages':pages,'sections':[{k:s[k] for k in ['section','text_percent','approximate_page_span']} for s in sections],'structure':[result[k] for k in ['tables','figures','equation_groups','references']]},ensure_ascii=False))

if __name__=='__main__':
    main()
