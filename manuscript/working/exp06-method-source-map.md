# EXP06方法章来源映射与复查记录

日期：2026-10-14。范围仅为 `manuscript/chapters/03-method.md`。
论文方法现为SDR-VLM，主结果行 `pair_focus_repair`；旧MF-VLM规则已整体替换。
本次仅修订文字，未重跑实验、未修改预测或参数。

## 来源优先级

1. 两个正式运行的冻结源码：`runs/exp06-seed2026101405-v1/source/`、`runs/exp06-seed2026101406-v1/source/`。
2. 正式方法协议 `source/method-protocol.md` 与 `docs/experiment-exp06.md`。
3. 当前代码用于阅读，与冻结源码逐文件SHA256比对。
4. `docs/exp06-assessment.md`用于判断贡献归属和已观察误差；方法章不重新填写结果数字。

当前及两个冻结版本的 `segmentation.py` SHA256均为：
`3F139E94E4BA68A33665D8F77FBB85515FF741D73A7AD5709F38BAC780166805`。
当前及两个冻结版本的 `segmented_review.py` SHA256均为：
`90B8576D9372911CA71F5CD054342E69B0DCB61E5C59ABACB4C8BA6019775F4C`。

## 段落与实际实现

以下路径简写相对于 `runs/exp06-seed2026101405-v1/source/`。

| 章节 | 实际依据 | 已核查内容 |
|---|---|---|
| 3.1 | method-protocol.md；segmentation.py:135；segmented_review.py:301 | 数值独立检测，视觉仅修复返回对；活动与尖峰不受视觉删除 |
| 3.2 | features.py:22、96；baselines.py:51、145 | 正常线性年周期与MAD；21日30维＋18维短邻域；平衡RF默认predict；均值填补 |
| 3.3 | segmentation.py:17、48、57、135 | 三响应OLS代价；PELT原搜索；7有效点、jump=1、6log(3n)、90日正常尾段；±3日精化和四模型比较 |
| 3.4活动 | segmentation.py:91 | 线性/smoothstep、单活动、15日最短、7日网格和±7日精化、5log(3n)+2log2；停止平台与活动掩码分开 |
| 3.4返回 | segmentation.py:75、135 | 扣选中活动、120日最大距离、30日前后文、独立跳变/脉冲、时间顺序贪心、返回日前一日为暂态末日 |
| 3.5图像 | segmented_review.py:61、78、175 | 21日样条＋跳变联合拟合仅扣连续项；全局原始图＋局部左右并排；±45日局部上下文；隐藏数值分类与分数 |
| 3.5修复 | segmented_review.py:301 | uncertain备选；视觉confirmed transient；两端≤7日；两端仍有step；采用数值边界；每次重算可用阶跃 |
| 3.6 | segmented_review.py:78、301；method-protocol.md | not_requested；数值结果独立、视觉失败保留；请求无重试；共享缺测掩码；离线任务 |

## 引文与解释范围

`Truong2020`：离线变点代价、搜索与惩罚的一般依据；DOI 10.1016/j.sigpro.2019.107299。
`Fearnhead2024`：连续分段线性信号与斜率变化表示的依据；DOI 10.18637/jss.v109.i07。
本子任务读取项目全文阅读笔记 `docs/exp06-reading-notes.md`，没有重新下载全文；主任务负责最终书目绑定与来源复核。
正文未声称运行CPOP，未将CPOP连续模型写成直接支持跳变；实际分段允许独立截距为本方法定义。

## 需要统稿时保持的事实

- 活动改善属于数值模型，慢运动辅助图未证明必要，不能归因视觉活动检测。
- 两端返回修复可以删除同日起始的真实阶跃，结果章需要保留已发生的错误配对。
- `baselines.py`回执文字残留“median imputation”，但真实代码是 `SimpleImputer()` 默认均值，论文按实际实现写均值。
- 主方法尖峰分类器有显式类别平衡，不能称完全默认RF；默认指predict判决，不指全部训练配置。
- `source/protocol.md`是继承的一般旧协议，EXP06特定方案依据 `source/method-protocol.md`。
- 只存在备选返回对才请求视觉，但不是每个触发请求的备选都必然有可修复双阶跃；修复时再次检查。
- 本方法判定不是“完全无参数”；统计惩罚、时间尺度、单活动假设已逐项写入正文。

## 本章完成检查

已通读全部新章并与上述实现对照；汉字3641字、10组展示公式、1张判据表、保留 `[[FIGURE:framework]]`。
旧Viterbi主方法、排名同日交集、每站必请求、失败全零等叙述已移除。
`git diff --check -- manuscript/chapters/03-method.md`通过（仅提示工作树CRLF将规范化为LF）。
公式转换、Word图表与逐页排版由主任务生成新版后完成，本子任务未声称已验证。
