# 融合数值形态候选与视觉大模型的GNSS多类型异常识别方法

## 摘要

GNSS位移序列中的尖峰、阶跃、暂态偏移与持续形变具有不同的时间支撑和形态证据，统一判决容易混淆点事件、偏移平台及持续活动。针对不同测站不宜共用单一物理位移阈值的问题，本文提出融合数值形态候选与视觉语言模型的GNSS多类型异常识别方法MF-VLM。首先，利用站点正常历史拟合趋势与年周期，以稳健残差尺度构造相对特征；其次，通过随机森林及区间时间解码形成候选，联合全局与局部NEU曲线进行一次视觉审阅；最后，按形态分配判决来源，尖峰沿用数值分类，阶跃采用候选日期与视觉确认的交集，暂态由检查窗约束，持续形变结合数值支持与视觉活动边界。实验采用两个独立生成种子的48个模拟测试站，覆盖四类异常及正常、混合场景，以四类等权日F1为主指标，并报告无点调整的事件F1和通用日F1。MF-VLM的三项指标分别为0.5681、0.5646和0.7095；主指标较平衡随机森林和数值时间解码分别提高14.24和19.47个百分点。在各站后365日有效观测中，相较统一排名视觉融合，尖峰误报由24个降至0个，但真阳性由9个降至6个。消融表明，局部视图与视觉活动边界未提供稳定的整体增益，固定数值活动边界变体的四类等权日F1和通用日F1分别为0.5816和0.7734。结果支持在当前模拟条件下按GNSS异常形态分配数值与视觉判据，改善多类型均衡识别及部分场景的误报控制，同时揭示边界保守化带来的漏检代价。

关键词：GNSS时间序列；异常识别；视觉语言模型；形态分工；多类型评价

## GNSS Multi-Type Anomaly Detection with Numerical Morphology Candidates and a Vision-Language Model

## Abstract

Spikes, steps, transient offsets, and sustained deformation in GNSS displacement time series require different temporal and morphological evidence. Applying a uniform decision rule can confuse point events, residual offsets, and ongoing motion. This study proposes MF-VLM, a morphology-specific fusion method combining numerical candidates with a vision-language model. A station-specific normal history is used to estimate the trend, annual cycle, and robust residual scale. Random forests and temporal decoding then generate candidates for a single visual review of global and local north-east-up curves. Final decisions depend on morphology: spikes retain numerical classifications, steps require agreement between candidate dates and visual confirmation, transients are constrained to candidate windows, and deformation combines numerical support with visually assessed activity boundaries. Experiments use 48 simulated test stations from two independent generation seeds, covering four anomaly types as well as normal and mixed scenarios. The primary endpoint is the equally weighted mean of four type-specific daily F1 scores, with event F1 and generic daily F1 reported without point adjustment. MF-VLM achieves 0.5681, 0.5646, and 0.7095 on these three metrics, respectively. The primary score exceeds balanced random forests and numerical temporal decoding by 14.24 and 19.47 percentage points. Over the observed evaluation days after the normal reference, compared with uniform ranked visual fusion, spike false positives decrease from 24 to zero, while true positives decrease from nine to six. Ablations do not establish consistent overall gains from local views or visual activity boundaries. A variant retaining numerical activity boundaries achieves a type-macro daily F1 of 0.5816 and a generic daily F1 of 0.7734. These findings support assigning numerical and visual evidence according to GNSS anomaly morphology under the simulated conditions, while identifying the missed detections associated with conservative boundary estimates.

Keywords: GNSS time series; anomaly detection; vision-language model; morphology-specific fusion; multi-type evaluation
