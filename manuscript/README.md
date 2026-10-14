# GNSS多类型异常识别论文初稿

[Word语言润色稿](GNSS多类型异常识别论文初稿_语言润色版.docx)：《融合数值形态候选与视觉大模型的GNSS多类型异常识别方法》。2026-10-14完成，Microsoft Word实际分页24页，包含中英文摘要、六章正文、5幅图、8张表、10组编号公式和14条实际引用。

本版按作者提供的写作规则突出形态分工贡献，精简重复免责声明，将不利结果写成具体指标取舍。沿用六章结构；正文文字占比依次为14.4%、17.5%、25.4%、33.0%、6.7%、3.1%。全部表格数据与展示公式保留，已完成内容复核及逐页视觉检查。

[原26页结构优化稿](GNSS多类型异常识别论文初稿_结构优化版.docx)与[原29页初稿](GNSS多类型异常识别论文初稿.docx)均保留为历史版本，源文件分别对应提交 `fb97423` 与 `f37f140`。

论文对应 **GNSS-VLM-EXP04-v1：GNSS形态分工视觉数值融合异常识别实验**。正文简称MF-VLM，对应冻结结果中的 `morphology_fusion`。证据范围为两个种子、48个模拟测试站；24站消融单独报告。初稿未新增模型调用或修改冻结实验材料。

## 内容与证据

- `chapters/`：逐章正文源文件，包含摘要和六章正文。
- `working/chapter-source-style-notes.md`：逐章参考论文、读取范围、结构及学术表达借鉴说明。
- `working/extra-references.json`：核实的书目与版本。保留15条检索资料，文末仅列正文实际使用的14条。
- `working/content-review.md`：独立内容审阅及从原始真值、预测复算结果的记录。
- `working/polish-final-check.md`：本版语言修订、证据复核、章节占比与24页视觉检查。
- `working/polish-edit-receipt.json`、`working/polish-review-corrections.json`：逐段润色与复审修改记录。
- `working/revision-final-check.md`：历史结构优化版的26页检查记录。
- `working/content-review-revision.md`：六章结构优化版内容复审与修正记录。
- `working/section-allocation-proposal.md`：先前五篇论文占比调研与历史提案；用户修正后的实际方案见本版检查记录。
- `working/section-allocation-polished.json`：本版文字与版面占比。
- `supplement/reproducibility.md`：由原实验章保留的复现条件、方法映射和运行入口。
- `working/final-check.md`：原29页初稿的历史检查记录。
- `working/build-manifest-polished.json`：章节、引用、图表和公式编号映射。
- `working/final-integrity-polished.json`：最终DOCX哈希及结构检查结果。

图1是根据冻结方法绘制的流程图；图2、图3来自 `results/exp04-two-seeds/comparison.png` 的分面；图4、图5来自 `results/exp04-cases/cases.png` 的暂态与持续形变面板。裁剪只改变布局，没有重算或修改曲线、标签及成绩。

## 本机重建

`build_manuscript.py` 将章节和书目装配为Word，公式为可编辑OMML。它依赖当前Windows环境的Microsoft Office MathML转换样式表、宋体/微软雅黑/Times New Roman，以及bundled Python中的python-docx、lxml和Pillow；`latex2mathml==3.81.1`安装在本任务临时目录 `%TEMP%/gnss-paper-authoring-deps`。临时目录失效时，需要用同一Python重新安装该依赖：

```powershell
& 'C:/Users/JC/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe' -m pip install --target "$env:TEMP/gnss-paper-authoring-deps" 'latex2mathml==3.81.1'
& 'C:/Users/JC/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe' manuscript/build_manuscript.py
& ./manuscript/render_word.ps1
& 'C:/Users/JC/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe' manuscript/render_pages.py
```

渲染使用本机Microsoft Word导出PDF，再调用文档技能的栅格化组件生成120 dpi逐页PNG。`render_pages.py`中的技能及Poppler路径与当前安装绑定，迁移机器时需调整；该脚本只使用前一步已导出的PDF，所以必须按上述顺序执行。内部PDF、PNG和转换诊断已忽略，不作为论文交付物。正文或布局修改后应重新检查受影响页面，本版24页检查记录对应语言润色版DOCX，历史26页与29页记录分别对应此前两版。

正式投稿前仍需根据目标刊物模板、作者及单位信息进行排版，并为读者提供可访问的代码与补充材料入口。研究结论限定于本文模拟条件。
