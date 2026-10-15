# EXP26：反事实位移连续性视觉复核（CFV-GNSS）

## Material Passport

- 日期：2026-10-15；状态：请求前登记；任务：simulation / code experiment。
- 父提交：5b88846；分支：codex/exp26-counterfactual-visual-review。
- 备份：backup/pre-exp26-counterfactual-20261015-5b88846；源码ZIP及EXP25完整证据ZIP均保留。
- 研究问题：把候选暂态扣除后的轨迹可视化，能否提高模型对混合异常中返回关系的辨别？

## 数据和分阶段执行

使用新seed2026101517，96训练、48validation、48test，原730日简单生成器和噪声不变。
仅validation开发；按station_id前24站pilot，后24站仅在扩量条件成立时运行。
test不读取标签、不评分。本轮不是重复使用1516 test作未见验证。
数值模型六项：segment_numeric、rf_split_temporal、random_forest、rf_balanced、extra_trees、isolation_forest。
每站所有方法同观测、同候选；无候选不调用，所有站仍计入成绩。

## 新方法及消融

CFV先从数值未配对step构建120日内所有合法候选，与EXP14 free_graph一致。
在每个候选上使用起点前后各最多7日、受相邻jump截断的原位移中位数差估计offset；
只用起点估计，不用终点拟合强制两端吻合。对候选范围扣除此常量向量，显示原轨迹和假设校正轨迹。
不扣除真实长期运动，不生成缺测值，保留其他阶跃。要求模型检查起止连续性与各轴一致性，
内含独立阶跃可保留，不把“校正后平滑”自动当真值。图上保留正常历史残差作为辅助，原始位移同图可核查。
这是本项目待验证设计，不宣称文献已有同一算法，也不是原码复现。

固定两个模型qwen3-vl-plus、qwen3.8-flash，temperature=0，无thinking，无重试。
各模型cf_image及cf_noimage同提示、同候选，只删image项；另两模型raw_image仅保留原始/历史残差图，
以分开候选图变化、提示变化与反事实图增益。cf_text用相同图的数据逐点文本及校正向量替代图片，
是强同信息对照，不能只用缺观测去图来宣称视觉模态优于文字。原SDR-Flash和去慢轨迹SDR-raw各保留。
独立数值对照：原segment_numeric，以及同候选EXP15未调参反向向量近邻。
第一阶段最多8种新输入＋2种原SDR请求/触发站；无候选仅保留数值。

## 指标与扩量/采纳条件

主指标：暂态Affiliation P/R/正站均F1。辅助：暂态逐日P/R/F1、TP/FP/FN、覆盖、四类日F1、
形变Affiliation与日F1、通用日F1、活动漏日/停止后假活动，IoU仅补充。
不point-adjust，不扫真值阈值，不挑触发站作总体。所有失败保留数值，并单列依赖失败。

pilot扩量：任一模型cf_image比segment_numeric暂态Aff-F1至少+.02，且比同模型cf_noimage至少+.02；
若无正向图像差异，取消本方案扩量。满足则所有固定模式均完成余24站，不只扩有利模式。
开发48站升级候选：cf_image相对原SDR-Flash或SDR-raw中较好者Aff-F1至少+.02，
且对同模型cf_noimage正向至少+.02；同时披露日F1、误报、同信息文字和数值近邻结果。
若仅Affiliation改善而重叠/误报恶化，保留范围指标候选但明确取舍，不称精确恢复。
若增益完全由数值近邻或文字匹配实现，不声称视觉独有优势。
达到开发升级后才冻结方案进入新test；未达到不扩正式测试。不得按结果修改本协议。

## 文献和复现边界

实读TAMA arXiv2411.02465v1、VLM4TS arXiv2506.06836v1、VisualTimeAnomaly arXiv2502.17812v2，
2026-10-15公开全文取回成功，原HTML、文本、URL和SHA在papers/exp26-counterfactual。
TAMA提供参考/多尺度复核思路；VLM4TS提供候选与全局上下文分工；VisualTimeAnomaly讨论
粗粒度与细粒度互补及缺测可视化。反事实校正是本项目推导，不冒称三篇论文提出。
范围指标复用既有锁定Affiliation作者源码及GNSS缺测、半开区间、正站均值适配。
本轮不重新执行三篇完整原方法，不把其摘要数字放进GNSS对比表。
