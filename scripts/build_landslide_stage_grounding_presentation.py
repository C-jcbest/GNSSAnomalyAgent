"""Attach post-freeze findings without altering the inference or primary page."""

from __future__ import annotations

import json

from build_landslide_stage_turning_presentation import replace_once
from landslide_stage_grounding import ROOT, RUN

from gnss_sim.artifacts import sha, write_json

NOTES = {
    "case_0001": "定位两遍只有两个A阶段，89个参考日弃判。第2遍称减速未被明确要求，与实际提示deceleration定义矛盾；图号覆盖通过仍不能保证语义执行。",
    "case_0002": "定位两遍同类35／40日，控制15／30日；这是局部开发一致性线索，仍有5个D→A日，不推广成总体改善。",
    "case_0004": "控制两遍越活动而整条拒绝；定位两遍边界合法，但参考阶段全部未定，不能判其正确。控制失败不影响585参考分母，也不能以未知重复一致当可靠。",
    "case_0005": "四配置都覆盖强[525,610)，本轮没有同期漏图改善优势；定位尾部未知使两遍各少10同类日。弱段四配置均S，参考未定，不能当正确。",
    "case_0007": "参考阶段全未定。定位两遍后半段分别S／D，只能报告波动，不能当内容成绩。",
    "case_0008": "定位第1遍20个D参考日判S，第2遍同类。定位标签与证据引用合规不能替代转折判断。",
    "case_0009": "控制两遍15参考日全同类，定位第1遍全未知，第2遍10同类／5异类；最后活动仍阶段未定。",
    "case_0010": "定位第1遍把两个复合活动整段判A，全部25日S变A且33日D变A；第2遍恢复S。引用全合规，阶段分类仍不稳定。",
    "case_0011": "定位两遍均A[285,495)，44日D参考仍被判A。第1遍notes称285–360未标A，但实际区间包含它；引用通过不检查这种矛盾。控制两遍160参考日全同类。",
}


def main():
    destination = RUN / "presentation-v1"
    destination.mkdir(exist_ok=False)
    source = RUN / "public/index.html"
    page = source.read_text(encoding="utf-8")
    page = replace_once(page, "<head>", '<head>\n  <base href="../public/">')
    section = """<section>
    <div class="eyebrow">POST-FREEZE / AUTHOR DIAGNOSIS</div>
    <h2>图号和范围能对上，阶段仍未稳定改善</h2>
    <p>定位组42个阶段提议全部通过机械引用覆盖检查，但同类日两遍由控制427／500变为285／389，重复分歧275→527；本轮暂不替换主配置。</p>
    <p>第2遍定位macro-F1为0.7022，高于控制0.6059，但同类日少111、D同类少126。定位保留S25日而控制全判D；该单记录恒速类别占macro权重的1/3，不能只挑指标宣布更好。</p>
    <p>0005四配置本轮都覆盖强活动，不能凭历史一次漏图宣称清单修复有效。0004控制两遍越界，定位边界合法，但该记录参考全未定；两次失败不影响585参考评分。</p>
    <p>0010定位第1遍与0011两遍仍整段A；0011说明与实际区间直接矛盾。引用合规只说明图号存在及轴范围可覆盖，不证明实际看图、非零运动或正确阶段。</p>
    <div class="links"><a href="../diagnosis-v1.json">逐记录／内部／成本诊断</a><a href="../presentation-v1/report.md">完整结果报告</a><a href="../public/index.html">原冻结网页</a></div>
    <p class="caption">本节与逐记录文字为事后作者诊断，未发送给模型。参考非独立、可判覆盖27.52%、S单记录；真实浏览器渲染未验收。</p>
    </section>
    """
    page = replace_once(page, '  <section>\n    <h2>完整运行与开发诊断',
                        section + '  <section>\n    <h2>完整运行与开发诊断')
    page = replace_once(page, '<h2 id="case-title"></h2>',
                        '<h2 id="case-title"></h2><p id="record-diagnosis" class="notice"></p>')
    notes = json.dumps(NOTES, ensure_ascii=False).replace("<", "\\u003c")
    page = replace_once(page, '      const request = c.requests[arm];',
                        f'      const notes = {notes};\n'
                        '      el("record-diagnosis").textContent = notes[c.case_id] || "无确认活动：跳过，不计正常识别成绩。";\n'
                        '      const request = c.requests[arm];')
    (destination / "index.html").write_text(page, encoding="utf-8")
    (destination / "report.md").write_bytes((ROOT / "docs/landslide-stage-grounding-results.md").read_bytes())
    write_json(destination / "freeze.json", {
        "primary_page_sha256": sha(source), "diagnosis_sha256": sha(RUN / "diagnosis-v1.json"),
        "builder_sha256": sha(ROOT / "scripts/build_landslide_stage_grounding_presentation.py"),
        "replacement_helper_sha256": sha(ROOT / "scripts/build_landslide_stage_turning_presentation.py"),
        "files_sha256": {name: sha(destination / name) for name in ("index.html", "report.md")},
        "inference_unchanged": True,
    })
    print("Supplemental findings saved; primary page unchanged")


if __name__ == "__main__":
    main()
