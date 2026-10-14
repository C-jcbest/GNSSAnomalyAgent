# 融合位移结构分解与视觉语义修复的GNSS多类型异常识别方法

## 摘要

GNSS位移序列中的水平跳变、暂态偏移与持续运动具有不同的时间结构，活动停止后的位移平台和运动背景上的返回过程容易造成类别与边界混淆。本文提出融合位移结构分解与视觉语义修复的多类型异常识别方法SDR-VLM。利用测站正常历史估计背景与残差尺度，通过尖峰分类、三分量位移分段、局部边缘判别和有界活动拟合，独立生成尖峰、阶跃、暂态及持续形变结果；针对数值方法未选中的返回配对，联合原始NEU曲线与辅助残差图开展视觉核验，在数值边缘支持下修复暂态类别。该方法通过站点相对尺度适配观测差异，采用固定统计惩罚与时间尺度，无需跨站统一毫米或速率告警阈值。在两个独立生成种子的48个模拟测试站上，四类等权日F1、四类等权事件F1（IoU≥0.5）和通用日F1分别为0.8652、0.8523和0.9178。主指标较同站重跑的既有完整定位融合提高28.35个百分点，其中无视觉结构分解达到0.8323，视觉修复进一步提高3.30个百分点。位移结构建模将16个活动站的漏活动日由944日减少至265日，停止后假活动日由124日减少至6日；视觉修复将暂态日F1由0.8788提高至0.9372。结果表明，位移结构分解与受限语义修复能够分别改善活动范围和返回类别，为GNSS质量核查与形变分析提供可解释的多类型识别结果。

关键词：GNSS时间序列；异常识别；视觉语言模型；位移结构分解；语义修复

## GNSS Multi-Type Anomaly Detection with Displacement Structure Decomposition and Visual Semantic Repair

## Abstract

Level shifts, transient offsets, and sustained motion in GNSS displacement time series have distinct temporal structures. Post-motion displacement plateaus and return patterns superimposed on ongoing motion can cause errors in event classification and boundary estimation. This study proposes SDR-VLM, a multi-type anomaly detection method combining displacement structure decomposition with visual semantic repair. Normal station histories provide background estimates and residual scales. Spike classification, three-component segmentation, local edge classification, and bounded activity fitting independently produce spike, step, transient, and sustained-deformation detections. A vision-language model reviews unresolved return pairs using raw north-east-up curves and auxiliary residual plots, and repairs transient classifications under numerical edge constraints. Station-relative normalization, fixed statistical penalties, and temporal scales support cross-station analysis without a common displacement or velocity alarm threshold. Across 48 simulated test stations from two independent generation seeds, SDR-VLM achieves a type-macro daily F1 of 0.8652, a type-macro event F1 of 0.8523 at an intersection-over-union threshold of 0.5, and a generic daily F1 of 0.9178. The primary score exceeds an earlier complete localization-fusion method rerun on the same stations by 28.35 percentage points. Numerical structure decomposition alone achieves 0.8323, with visual repair adding 3.30 percentage points. At 16 stations with sustained activity, displacement structure modeling reduces missed activity days from 944 to 265 and false activity days after cessation from 124 to six. Visual repair increases transient daily F1 from 0.8788 to 0.9372. These results show that numerical structure modeling and constrained semantic repair improve activity delineation and return-pattern classification, respectively, providing interpretable event detections for GNSS quality assessment and deformation analysis.

Keywords: GNSS time series; anomaly detection; vision-language model; displacement structure decomposition; semantic repair
