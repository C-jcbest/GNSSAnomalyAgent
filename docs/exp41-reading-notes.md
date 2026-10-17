# EXP41文献阅读与方法边界

2026-10-17，继续使用academic-research-suite实验工作流。用户已授权方案设计、脚本修改和
现有视觉接口执行，沿用先小批后扩量、请求失败留存及不重试的规则。

## 实际阅读

- [TAMA，2411.02465v1](https://arxiv.org/html/2411.02465v1)：本轮重读§3.2的
  Multimodal Analyzing、Multi-scaled Self-reflection和Results Aggregation。
  作者将检测区间和放大图再交给模型复核，并用置信度聚合；本轮只借鉴全局定位与局部复核，
  不采用其置信度阈值、不复现完整正常参考学习，也不将其point-adjusted F1与本项目Aff直接排名。
- [Can Multimodal LLMs Perform Time Series Anomaly Detection?，2502.17812v2](https://arxiv.org/html/2502.17812v2)：
  本轮重读§6工具池、实验与Checking/Planning/Scanning消融。作者将传统检测与多模态判断
  放入协同流程。本轮借鉴“数值作为工具证据、视觉参与范围决策”，不声称运行其完整智能体。
  两篇原文及纯文本副本沿用`papers/exp38-reference-residual/`，未重新下载同一版本。
- [Supervision Recovery for Time Series Anomaly Detection via Context-Anchored Pairing，2609.33610v1](https://arxiv.org/html/2609.33610v1)：
  新取得全文，实际读引言的matched normal–anomalous supervision与§3的counterfactual
  recomposition说明。CAPS从匹配仿真对学习结构/异常语义，再生成目标上下文中的异常监督，
  不是GNSS边界正反向配对，也不是视觉LLM曲线复核。故本轮不将标题中的pairing当作边界
  方法依据，不新增复杂生成模型，不填其GNSS对比分数。原文、文本和HTTP/SHA收据存于
  `papers/exp41-boundary-mechanism/`。只补读该线索，不称系统性最新文献综述。

## 本项目实际新增的组合

已有`gnss_research.segmentation.jump_evidence`用原始位移局部观测比较连续linear、
连续hinge、step及step_hinge，代价含本站历史噪声归一化下的复杂度惩罚。
本轮不改此工具、不增加物理阈值，而将其结果和7日局部中位数变化交给视觉复核。
借鉴`exp15_vector`的局部变化配对以提供候选，不能把候选当真值或自动接收。
视觉复核直接拥有最终暂态输出，数值不无条件回填；保留同信息文字和去数值证据消融。

这是针对GNSS持续运动/暂态混淆的本项目流程适配，不是论文作者源码复现。
是否有收益需看新站实验，不能从文献或拟合模型名称直接推断。
