"""Add explicit contract status and interpretation without changing the frozen page."""

from __future__ import annotations

import json

from landslide_stage_turning import ROOT, RUN

from gnss_sim.artifacts import sha, write_json

NOTES = {
    "case_0001": "三个活动均无候选；四组共同留下15个D→A参考分歧日及25个支撑弃判日，候选说明未解决它们。",
    "case_0002": "唯一峰候选为[390,450)。候选两遍都输出A→D，40参考日全部同类；控制两遍为35／15同类日。此单例不能代表总体。",
    "case_0004": "两活动各有一个峰候选，但参考阶段全部未定，无法评分；控制两遍最终相同，候选两遍有138个确定类别分歧日。",
    "case_0005": "强活动[525,610)无候选。候选第1遍声称原图从685日才开始，漏看已发送全程／中段图中的强活动，使34个D参考日转弃判。弱活动的三个候选未产生稳定分类：两组第1遍均未知，第2遍均S，尚无确定参考可验证。",
    "case_0007": "活动无候选且参考阶段全部未定，候选两遍未知不能当正确率；控制一次输出A一次未知。",
    "case_0008": "第一活动有峰候选，四组55参考日全部同类；候选两遍更稳定的58日S都在参考未定范围，不能当准确性改善。",
    "case_0009": "三个活动均无候选；15参考日同类由控制5／15变为候选0／15。空候选组仍有附加读取说明，变化不能归因于候选位置。",
    "case_0010": "第一活动有三候选。候选第1遍忽略波动并输出S，第2遍按240／270／300日切成D／A／D，恒速参考25日全变异类（20A、5D）；新增分段不自动代表更准确。",
    "case_0011": "候选第1遍保留了转折，但较窄阶段留下33参考日弃判；第2遍输出非法S[720,955)，越出[675,785)，整记录阶段拒绝且160参考日保留弃判，无重试。文字声称720–785未标，与实际stages矛盾。",
}


def replace_once(source, before, after):
    if source.count(before) != 1:
        raise ValueError("Frozen presentation insertion point is not unique")
    return source.replace(before, after)


def main():
    output = RUN / "presentation-v1"
    output.mkdir(exist_ok=False)
    source_path = RUN / "public/index.html"
    diagnosis_path = RUN / "diagnosis-v1.json"
    diagnosis = json.loads(diagnosis_path.read_text(encoding="utf-8"))
    page = source_path.read_text(encoding="utf-8")
    page = replace_once(page, "<head>", '<head>\n  <base href="../public/">')
    section = """<section>
    <div class="eyebrow">POST-FREEZE / AUTHOR DIAGNOSIS</div>
    <h2>本轮未显示稳定收益，保留原配置</h2>
    <p>候选组两遍的开发macro-F1均低于同期控制，重复分歧479→764日。不是独立准确率结论，不能用历史单遍作优化前后对照。</p>
    <p>13个候选只覆盖21个活动中的9个，漏掉0001与0005强段，却在0005弱段产生3个；候选与附加说明共同构成干预，空候选不代表恒速。</p>
    <p>剔除0011的四次共同成功敏感性分析保留425参考日：第1遍控制332／候选308同类，第2遍322／350同类；但候选两遍macro-F1仍较低，恒速第二遍失去全部25日。这个事后成功筛选不替代完整585日主表。</p>
    <p>实际查看原图确认0005全程包含525–610日，模型“图从685日开始”的解释不成立；0010的峰谷拆分造成恒速类别损失。下一步先检验图像范围读错与候选过度分段，再另立短窗支撑实验。</p>
    <div class="links"><a href="../diagnosis-v1.json">逐记录／内部／成本诊断</a><a href="../presentation-v1/report.md">完整结果报告</a><a href="../public/index.html">原冻结页面</a></div>
    <p class="caption">本节及逐记录诊断是推理冻结后的作者解释，未发送给模型。网页真实浏览器渲染尚未验收。</p>
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
    old = '      el("request-status").textContent = `${names[arm]} · ${request.response.status} · ${request.response.error_type || "无传输／解析错误，阶段契约状态另见results.json"}`;'
    new = '''      const contract = data.summary.results.find(result => result.request_id === `${c.case_id}-${arm}`);
      el("request-status").textContent = `${names[arm]} · 请求：${request.response.status} · 阶段契约：${contract.status}${contract.error ? " · " + contract.error.message + " · 整记录阶段未知，无重试" : ""}`;'''
    page = replace_once(page, old, new)
    page = replace_once(page, '      el("receipt").textContent = JSON.stringify(request.receipt_summary, null, 2);',
                        '      el("receipt").textContent = JSON.stringify({ ...request.receipt_summary, stage_contract_status: contract.status, stage_contract_error: contract.error }, null, 2);')
    (output / "index.html").write_text(page, encoding="utf-8")
    report = ROOT / "docs/landslide-stage-turning-results.md"
    (output / "report.md").write_bytes(report.read_bytes())
    write_json(output / "freeze.json", {
        "frozen_primary_page_sha256": sha(source_path), "diagnosis_sha256": sha(diagnosis_path),
        "builder_sha256": sha(ROOT / "scripts/build_landslide_stage_turning_presentation.py"),
        "files_sha256": {name: sha(output / name) for name in ("index.html", "report.md")},
        "total_tokens": diagnosis["total_tokens"], "inference_unchanged": True,
    })
    print("Created supplemental presentation; primary page remains frozen")


if __name__ == "__main__":
    main()
