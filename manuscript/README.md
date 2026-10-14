# SDR-VLM论文实验更新版

[25页Word实验更新稿](GNSS多类型异常识别论文初稿_SDR-VLM实验更新版.docx)：《融合位移结构分解与视觉语义修复的GNSS多类型异常识别方法》。2026-10-14完成，包含中英文摘要、六章正文、5幅图、10张表、13组编号公式和17条实际引用。当前分支codex/paper-sdr-vlm。

论文已由EXP04形态分工方案改为当前保留的 **GNSS-VLM-EXP06-v1：GNSS位移结构分解与视觉语义核验实验**。主方法SDR-VLM，对应pair_focus_repair；主实验48站四类等权日F1为0.8652，无视觉结构分解为0.8323。总体改进以数值结构为主体，视觉增加3.30个百分点，配对区间下界为零。

EXP07、EXP08原版SDR-VLM的两批各48站复核另表列示，未将未采纳方案写成生产方法，也未合并为原预登记144站。较低的EXP08结果及种子1412视觉负增量一并保留。原作者PELT与GNSS适配在独立边缘任务中比较，前期MF-VLM和定位融合明确标作本研究前期方案。

六章正文文字占比依次约为15.0%、17.8%、24.5%、30.9%、9.3%、2.6%；统计口径沿用此前篇幅调研。没有独立实验设置章，必要条件融入方法与结果。

## 内容与证据

- chapters/：新版逐章正文。
- figures/exp06/、prepare_sdr_figures.py：由冻结结果和原始观测绘制5图，旧EXP04图保留。
- working/exp06-evidence-audit.md与.json：重算672份主实验输出、192份扩展输出及bootstrap、活动误差、PELT边缘结果。
- working/exp06-method-source-map.md：方法与冻结代码映射，明确均值填补和平衡RF。
- working/exp06-literature-check.md、extra-references.json：新增文献、正式版本和实际阅读范围。
- working/sdr-final-check.md：全稿内容、图表及25页视觉检查。
- working/section-allocation-sdr.json：实际章节文字和版面占比。
- working/build-manifest-sdr.json、final-integrity-sdr.json：图表公式编号、文件哈希、历史稿保留检查。
- supplement/reproducibility.md：EXP06条件及方法键映射；reproducibility-exp04.md保留历史说明。

第4章表3为主比较，表4为各类指标，表5为核心消融，表6为活动边界，表7为作者PELT边缘对照，表8和表9为跨种子与描述性区间，表10为扩展验证。图4同时展示暂态恢复和真实阶跃损失，图5展示原始位移活动与停止平台。

## 历史版本

[24页语言润色稿](GNSS多类型异常识别论文初稿_语言润色版.docx)、[26页结构优化稿](GNSS多类型异常识别论文初稿_结构优化版.docx)和[29页最初稿](GNSS多类型异常识别论文初稿.docx)全部保留，文件与本轮开始时逐字节一致。历史稿对应EXP04；各自审阅记录仍保存在working中，不能作为新版检查凭据。

## 本机重建

使用Codex bundled Python、python-docx、lxml、Pillow、numpy及本机Office的MML2OMML.XSL。任务临时依赖目录为 %TEMP%/gnss-paper-authoring-deps；当前使用latex2mathml 3.81.1、matplotlib 3.10.8及其绘图依赖。目录失效时先安装，再依次运行：

```powershell
& 'C:/Users/JC/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe' -m pip install --target "$env:TEMP/gnss-paper-authoring-deps" 'latex2mathml==3.81.1' 'matplotlib==3.10.8'
& 'C:/Users/JC/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe' manuscript/build_manuscript.py
& ./manuscript/render_word.ps1
& 'C:/Users/JC/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe' manuscript/render_pages.py
& 'C:/Users/JC/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe' manuscript/working/measure_sdr.py
```

生成图表需要本地data/exp06-*和runs/exp06-*材料。公式为可编辑OMML。渲染用独立隐藏Word实例导出PDF，再调用文档技能栅格化组件生成120 dpi逐页PNG；转换路径与本机安装绑定。PDF与PNG仅供内部检查，交付文件为Word。重新构建后须重新审阅受影响页面，原检查记录与哈希绑定。

投稿时按目标刊物模板填写作者、单位与公开材料入口；正文保持作者视角及当前实验的真实证据范围。
