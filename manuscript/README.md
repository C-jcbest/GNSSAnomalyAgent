# GNSS多类型异常识别论文初稿

[Word初稿](GNSS多类型异常识别论文初稿.docx)：《融合数值形态候选与视觉大模型的GNSS多类型异常识别方法》。2026-10-13完成，Microsoft Word实际分页29页，包含中英文摘要、七章正文、5幅图、10张表、11组编号公式和14条实际引用。

论文对应 **GNSS-VLM-EXP04-v1：GNSS形态分工视觉数值融合异常识别实验**。正文简称MF-VLM，对应冻结结果中的 `morphology_fusion`。证据范围为两个种子、48个模拟测试站；24站消融单独报告。初稿未新增模型调用或修改冻结实验材料。

## 内容与证据

- `chapters/`：逐章正文源文件，包含摘要和七章正文。
- `working/chapter-source-style-notes.md`：逐章参考论文、读取范围、结构及学术表达借鉴说明。
- `working/extra-references.json`：核实的书目与版本。保留15条检索资料，文末仅列正文实际使用的14条。
- `working/content-review.md`：独立内容审阅及从原始真值、预测复算结果的记录。
- `working/final-check.md`：审阅意见落实、文档完整性及29页视觉检查记录。
- `working/build-manifest.json`：章节、引用、图表和公式编号映射。
- `working/final-integrity.json`：最终DOCX哈希及结构检查结果。

图1是根据冻结方法绘制的流程图；图2、图3来自 `results/exp04-two-seeds/comparison.png` 的分面；图4、图5来自 `results/exp04-cases/cases.png` 的暂态与持续形变面板。裁剪只改变布局，没有重算或修改曲线、标签及成绩。

## 本机重建

`build_manuscript.py` 将章节和书目装配为Word，公式为可编辑OMML。它依赖当前Windows环境的Microsoft Office MathML转换样式表、宋体/微软雅黑/Times New Roman，以及bundled Python中的python-docx、lxml和Pillow；`latex2mathml==3.81.1`安装在本任务临时目录 `%TEMP%/gnss-paper-authoring-deps`。临时目录失效时，需要用同一Python重新安装该依赖：

```powershell
& 'C:/Users/JC/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe' -m pip install --target "$env:TEMP/gnss-paper-authoring-deps" 'latex2mathml==3.81.1'
& 'C:/Users/JC/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe' manuscript/build_manuscript.py
& ./manuscript/render_word.ps1
& 'C:/Users/JC/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe' manuscript/render_pages.py
```

渲染使用本机Microsoft Word导出PDF，再调用文档技能的栅格化组件生成120 dpi逐页PNG。`render_pages.py`中的技能及Poppler路径与当前安装绑定，迁移机器时需调整；该脚本只使用前一步已导出的PDF，所以必须按上述顺序执行。内部PDF、PNG和转换诊断已忽略，不作为论文交付物。正文或布局修改后应重新检查受影响页面，当前29页记录只对应本版DOCX。

正式投稿前仍需根据目标刊物模板、作者及单位信息进行排版，并为读者提供可访问的代码与补充材料入口。研究结论限定于本文模拟条件。
