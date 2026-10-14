# 融合数值形态候选与视觉大模型的GNSS多类型异常识别方法

## 摘要

GNSS位移序列中的尖峰、阶跃、暂态偏移与持续形变依赖不同的时间和形态证据，准确识别需要结合站点背景分配判据。本文提出数值候选与视觉语言模型协同的形态分工融合方法MF-VLM。利用站点正常历史估计趋势、年周期及稳健残差尺度，构造相对特征；通过随机森林与区间时间解码形成候选，联合全局和局部NEU曲线完成单轮视觉审阅；按形态分别采用数值尖峰判决、候选日期与视觉确认一致的阶跃判决、检查窗内暂态区间及数值支持下的视觉活动边界。在两个独立生成种子的48个模拟测试站上，四类等权日F1、四类等权事件F1（IoU≥0.5）和通用日F1分别为0.5681、0.5646和0.7095。主指标较平衡随机森林和数值时间解码分别提高14.24和19.47个百分点，较使用相同视觉响应的统一排名融合提高6.01个百分点。相较于统一排名融合，评价期内尖峰误报由24个减至0个，命中数由9个变为6个。边界分析显示，视觉活动边界将含活动站停止后的假活动日降至0，同时增加活动漏日；固定数值边界变体的四类等权日F1为0.5816，通用日F1为0.7734。形态分工提高了该模拟任务中的类型均衡识别能力，并提供了可逐类解释的误报、漏检与活动覆盖取舍。

关键词：GNSS时间序列；异常识别；视觉语言模型；形态分工；多类型评价

## GNSS Multi-Type Anomaly Detection with Numerical Morphology Candidates and a Vision-Language Model

## Abstract

Spikes, steps, transient offsets, and sustained deformation in GNSS displacement time series require distinct temporal and morphological evidence. This study proposes MF-VLM, a morphology-specific fusion method that combines numerical candidates with a vision-language model. Normal station histories provide estimates of the trend, annual cycle, and robust residual scale for relative feature construction. Random forests and temporal decoding generate candidates for a single visual review of global and local north-east-up curves. Final decisions retain numerical classifications for spikes, require agreement between candidate dates and visual confirmation for steps, constrain transients to candidate windows, and use visually assessed activity boundaries supported by numerical intervals. Across 48 simulated test stations from two independent generation seeds, MF-VLM achieves a type-macro daily F1 of 0.5681, a type-macro event F1 of 0.5646 at an intersection-over-union threshold of 0.5, and a generic daily F1 of 0.7095. The primary score exceeds balanced random forests and numerical temporal decoding by 14.24 and 19.47 percentage points, respectively, and uniform ranked fusion using the same visual responses by 6.01 percentage points. During the evaluation period, spike false positives decrease from 24 to zero, with six true positives compared with nine for uniform ranked fusion. Visual activity boundaries eliminate false activity days after cessation at stations with sustained activity while increasing missed activity days. A variant retaining numerical boundaries achieves a type-macro daily F1 of 0.5816 and a generic daily F1 of 0.7734. Morphology-specific evidence allocation improves balanced type recognition in this simulated task and provides interpretable tradeoffs among false alarms, missed detections, and activity coverage.

Keywords: GNSS time series; anomaly detection; vision-language model; morphology-specific fusion; multi-type evaluation
