# EXP27文献核对与设计推导

检索：2026-10-15，原始响应及SHA见papers/exp27-trend-cfv/retrieval.json。
此前EXP26相关新近视觉论文仍作为主要方法背景，本轮补充地学趋势估计的早期基础。

## TAMA：局部放大应提供不同视觉证据

《See it, Think it, Sorted: Large Multimodal Models are Few-shot Time Series Anomaly Analyzers》，
arXiv:2411.02465v1，https://arxiv.org/html/2411.02465v1 。本轮重新取得全文HTTP200，
读取§3.2中的Multi-scaled Self-reflection和Results Aggregation。
前者给模型前序输出和异常区局部放大图，指出改变视图可避免对同图重复生成同结果；
后者包含置信分数聚合及阈值。本项目只借鉴局部视图补充证据，没有复现完整TAMA，
没有搬用置信阈值或point-adjustment，也没有填其未经本数据执行的成绩。

EXP27的差别是让模型比较候选暂态扣除前后起止两条边的连续性，
放大视图提供原始每日观测，位置仍由同一数值候选固定。只是推导，不能以文献代替实验验证。

## MIDAS：稳健趋势估计可分离慢变化与跳变

Blewitt、Kreemer、Hammond、Gazeaux，2016，
《MIDAS robust trend estimator for accurate GPS station velocities without step detection》，
DOI https://doi.org/10.1002/2015JB012552 。本轮读取Crossref原出版元数据及结构化摘要，
没有取得并通读出版全文。摘要明确普通Theil-Sen是成对斜率中位数，MIDAS采用约一年间隔、
处理间断与偏斜并重新估计；目的是长期速度，不是本项目的短窗口暂态配对。

本轮使用SciPy现有theilslopes，在每条边两侧分别估计局部斜率，
以共同趋势剥离后的两侧中位数差估跳量。14日是明确披露的新窗口设计参数，
不称MIDAS实现，不采用其一年间隔规则，不把其摘要精度数字当本项目结果。
该适配在局部加速、短片段、离群值或相邻边干扰下仍可能有偏，必须和原7日估计对照。

## CPOP：连续斜率变化与跳变是不同问题

Maidstone、Fearnhead、Letchford，《Detecting changes in slope with an L0 penalty》，
arXiv:1701.01672，取得abs页面HTTP200，页面当前版本v2。
本轮仅阅读作者摘要：连续分段线性拟合、残差平方和与斜率变化L0惩罚。
未阅读全文或执行其代码，不声称其能直接解决三轴GNSS暂态，也不列作已完成基线。
它提醒本轮不要把连续运动曲率当成瞬时位移跳量，实际效果仍要由实验确定。

## 检索限制

Crossref第二次关键词查询返回429，未将未取到的结果写成已读论文。
本轮没有新增文献SOTA数字，没有证明视觉一定胜过同信息文字或数值近邻。
