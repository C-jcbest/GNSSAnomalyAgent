# GNSS视觉数值分工论文草稿

当前主稿为[26页Word范围视觉与形变更新版](GNSS多类型异常识别论文初稿_范围视觉与形变更新版.docx)，题为《面向GNSS多类型异常识别的视觉数值分工方法》。2026-10-18更新，含中英文摘要、六章正文、7幅图、10张表、11组可编辑编号公式及16条引用。分支为`codex/exp45-deformation-parity`。

## 实验与结论的对应关系

- 主比较及暂态视觉消融对应**EXP37的360站独立模拟测试**。单点、阶跃分别报严格发生日日F1；暂态与形变分别报Affiliation指标，并保留日P/R/F1和误报。图像联合暂态Aff-F1为0.9853，文字联合为0.9253，去视觉完整数值组件为0.9046；同时披露联合暂态日F1为0.4095及阴性站11402误报日。
- 形变补充对应**EXP45的108站新开发确认**，不与360站拼接为同批综合成绩。联合形变Aff-F1为0.9802、日F1为0.9324，与数值版持平；最终视觉修边0站，成绩归属数值。
- 去视觉消融保留局部位移向量配对，不把内部数值近邻另列为外部论文方法。不声称已执行TAMA、TSAD-Agents、VLM4TS等完整作者流程，也不声称领域SOTA。

## 正文及证据入口

- `revisions/range-20261018/chapters/`：本次逐章正文；不覆盖旧`chapters/`。
- `revisions/range-20261018/figures/`：由冻结结果及原始观测生成的7幅图。
- `revisions/range-20261018/working/revision-review.md`：逐章文献风格依据、方法代码映射、内容与逐页检查记录。
- 同目录`tables.json`、`evidence-sources.json`和`build-manifest.json`：图表数据、输入哈希和构建记录。
- 同目录`current-citation-check.json`：本轮9条Crossref元数据复核；其余引用沿用已有来源记录。元数据核验不等于重读全文。
- [EXP37评估](../docs/exp37-assessment.md)、[EXP45评估](../docs/exp45-assessment.md)：完整实验来源。

表3为逐类主比较，表4为点类及形变的辅助日指标，表5和表6为暂态消融，表7为开发确认，表8和表9为种子与场景分层，表10为形变补充。图5和图6同时给出补检正例和永久阶跃误报反例。

## 历史版本

[25页SDR-VLM实验更新稿](GNSS多类型异常识别论文初稿_SDR-VLM实验更新版.docx)对应EXP06；[语言润色稿](GNSS多类型异常识别论文初稿_语言润色版.docx)、[结构优化稿](GNSS多类型异常识别论文初稿_结构优化版.docx)及[最初稿](GNSS多类型异常识别论文初稿.docx)对应更早研究。四个文件均与本轮开始时逐字节一致，不作为当前结果依据。旧`chapters/`、`build_manuscript.py`和`supplement/reproducibility.md`仍属历史EXP06材料。

## 本机重建

依赖使用Codex工作区加载器提供的bundled Python。新增绘图及公式依赖在D盘任务临时目录，本机已具备Microsoft Word。病例原始数据只读自旧共享目录`D:/CodeWorkSpace/AgentWorkspace/GNSSAnomalyAgent-sim/artifacts/exp30/data`，研究代码及Git操作仍在D盘当前工作树。

```powershell
$env:PYTHONIOENCODING='utf-8'
$env:PYTHONPATH='D:/CodeWorkSpace/CodexWorktrees/.artifact-runtime/gnss-paper-20261018/gnss-paper-authoring-deps'
$env:MPLCONFIGDIR='D:/CodeWorkSpace/CodexWorktrees/.artifact-runtime/gnss-paper-20261018/mpl'
& 'C:/Users/JC/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe' manuscript/prepare_range_evidence.py
& 'C:/Users/JC/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe' manuscript/build_range_revision.py
$env:TEMP='D:/CodeWorkSpace/CodexWorktrees/.artifact-runtime/gnss-paper-20261018'
$env:TMP=$env:TEMP
./manuscript/render_word.ps1 -DocumentPath './manuscript/GNSS多类型异常识别论文初稿_范围视觉与形变更新版.docx' -OutputDirectory './manuscript/working/render-range-20261018'
& 'C:/Users/JC/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe' manuscript/render_range_revision.py
```

新脚本只读取保存的实验，不发起模型请求。渲染先用隐藏Word实例导出PDF，再调用文档技能的栅格化器；本机无LibreOffice，诊断和转换方式记在内部渲染目录。PDF、PNG均用于内部检查，不作为交付件；重新构建后必须重新检查受影响页面。

本稿是可继续修改的研究草稿。投稿模板、作者单位、资助与贡献等须由作者按实际情况及目标刊物要求补全，不能虚构。
