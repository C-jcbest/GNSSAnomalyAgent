# EXP45执行补充记录

2026-10-18，延续2026-10-17登记的形变性能保留协议。

## 评分读取修正

18站pilot完成12次请求后，首次评分在整日缺测为JSON `null`的观测处中断，
尚未生成批次分数和准入结果。原`visual_events`直接转numpy数组，误以为缺测总是三项null。
改用项目已有`schema.observed_values`处理整行缺测，端点仍取区间内首末真实观测。
针对性测试已改为实际整行null输入并通过。

不修改已冻结的实验脚本、协议、提示、模型响应或数值候选。使用独立
`scripts/report_exp45.py`仅替换这一转换，再调用原有冻结评分流程。
中断前已写的部分final文件移至`final-incomplete-mask-error-v0`留存，最终结果另写。
每批保留`evaluation-correction.json`，记录修正原因和两个脚本的SHA；0新增请求、0重试。
后续108站沿用同一已验证修正，不据分数调整选择规则。

正式评分命令：

```powershell
.venv/Scripts/python.exe scripts/report_exp45.py --phase pilot
.venv/Scripts/python.exe scripts/report_exp45.py --phase confirmation
```

## 文献复读范围

复读既有全文缓存，未声称完成最新系统综述：

- [TAMA，2411.02465v1](https://arxiv.org/html/2411.02465v1)，§3.2
  Multi-scaled Self-reflection和Results Aggregation：作者用检测局部放大图复核，
  并另有置信度阈值聚合。本实验只借鉴全局/局部查看，不采用其阈值聚合，
  不将作者point-adjusted F1与本实验Affiliation数值直接排名。
- [Can Multimodal LLMs Perform Time Series Anomaly Detection?，2502.17812v2](https://arxiv.org/html/2502.17812v2)，
  §6.2 Planning、Detection、Checking：数值工具与多模态检测协同，Checking检查区间过宽/过窄。
  本实验是数值活动锚定、视觉提供边界候选的简单适配，不是作者完整TSAD-Agents复现。

数值择优只保证本项目拟合准则不劣，不保证真值指标不劣；是否接近依实际开发确认结果判断。
未路由站不请求模型，所有站仍计分。视觉输出失败照常纳入联合方法结果，不重试。
