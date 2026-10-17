# EXP36：论文依据与本项目适配边界

检索日期2026-10-17。采用academic-research-suite实验执行/核验流程，用户已授权设计、
修改实验与现有视觉接口请求。记录属于方法开发依据，非系统综述、非领域SOTA证书。
原始HTML、提取文本与SHA见`papers/exp36-contrastive-evidence/`。

## 实际阅读与取舍

1. **See it, Think it, Sorted: Large Multimodal Models are Few-shot Time Series
   Anomaly Analyzers（TAMA）**，2411.02465v1。
   来源：https://arxiv.org/html/2411.02465v1 。实际阅读Introduction、Methodology中
   Multimodal Reference Learning / Multimodal Analyzing段与附录参考提示。
   原文使用正常参考图像学习其模式，再检测/分类，并有多尺度复核。
   本轮采用“显式示例能帮助视觉判别”的动机，但固定4个有标签GNSS形态示例
   （其中包含异常），不是原文同通道正常参考，不称TAMA源码复现。
   正常参考、语义类型与多尺度视觉本身都有先例，不作为本项目首次贡献。

2. **AnomaMind: Agentic Time Series Anomaly Detection with Tool-Augmented Reasoning**，
   2602.13807v2，来源：https://arxiv.org/html/2602.13807v2 。实际阅读Introduction、
   IV-A与IV-B的memory/数值工具概述，核实作者README及模型列表。
   原文知识记忆包括训练数据视觉异常模式，数值工具提供可量测证据；
   核心检测策略含强化学习，不能用本项目少量提示适配冒称完整AnomaMind。
   作者库：https://github.com/ustc-time-series/AnomaMind ，README公开0.6B/8B权重。
   本轮不称其未公开；尚未在本项目执行，不能写成已超越该方法。

3. **LLM-Generated Feature Pools for Time Series Anomaly Detection**，2609.21801v1，
   来源：https://arxiv.org/html/2609.21801v1 。实际阅读角色分工、§3.1、§5.2、Appendix C。
   原文在调优侧提供带/不带异常的图像示例生成特征池，随后用固定统计检测器评价；
   LLM不直接看其评价split。它支持把视觉能力用于表征构造而非只作解释这一备选方向。
   本轮未生成/运行该特征池，不转用论文表格成绩，不以其跨域结果证明GNSS收益。
   固定4训练示例是本项目小成本尝试，与该论文的生成特征池方法不同。

4. **Harnessing Vision-Language Models for Time Series Anomaly Detection（VLM4TS）**，
   正确arXiv编号2506.06836v2，来源：https://arxiv.org/html/2506.06836v2 。
   本轮取得正确全文，实际重点阅读作者提交8ab8c164的`vit4ts.py`、`clip_vision.py`、
   `vlm4ts.py`与依赖清单：CLIP视觉候选→VLM全图复核；默认alpha=.01，
   VLM可删除、增补或修改范围。尚未执行完整方法，不填GNSS分数。
   未来执行须保留默认阈值，分别标出单变量输入/缺测、NEU组合及模型接口适配。
   检索时误请求2508.04353v2返回404，保留失败收据；已按项目核实文献纠正，
   错误编号不作为任何方法依据。

## 后续作者对照的准备状态

本轮另外核实AnomaMind作者提交`bf2c0cb3aaeb5a47aa7acabd1714ce123ddcdaa1`，
取得`AnomaAgent.py`、`infer_detector_only.py`和`detector_prompt.py`原文件并保存SHA。
实际读取了它们的流程初始化、原提示/上游状态构造和反思路由相关部分，未执行。
完整流程是Locator→Actor→Detector→可选Evaluator循环→finalize；作者的纯detector脚本
依赖已经生成的上游状态，不能只把0.6B权重接到裸GNSS数组就称完整框架复现。
公开0.6B模型元数据固定为`cf649fb2cb4aed088cca8162e73e0c8a9fb26e0a`，
确认未加访问门槛，尚未下载/推理。模型页面：
https://huggingface.co/YuChongZ/AnomaMind-0.6B 。

下一次公平比较应把“完整作者流程/固定权重”和“仅detector组件适配”分行；
统一数据与通用异常任务，不用GNSS真值把作者类型强行转换成四类后宣称各类更优。
VLM4TS则保留其原alpha=.01、窗口与CLIP设置，输入/接口适配与阈值改动分列。
这些是已经核实的可执行方向，当前没有它们的实验分数，不构成SOTA比较完成。

EXP37测试期间再次核对v2正文的代码脚注：链接为
https://github.com/Xiaoyu-Tao/AnomaMind-TS 。通过`git ls-remote HEAD`确认该库当前
仍指向同一提交`bf2c0cb3aaeb5a47aa7acabd1714ce123ddcdaa1`；固定提交README的
SHA与前述ustc-time-series来源一致。两地址不能当作两个独立对照方法。
实际补读README训练/评价部分：完整复现涉及SFT/RL检测策略和其他角色接口，
不是只换通用API的一个提示。GitHub元数据API遇到限流，错误正文保留；
固定提交原文件成功取得，访问收据见`anomamind-v2-retrieval.json`。

## 本轮可检验的设计

示例图帮助区分暂态矩形偏移和真实持续位移；数值工具验证边界是否支持不连续跳变，
避免视觉把“平滑运动停止后平台”当暂态。数值工具沿EXP29完整冻结实现，
此前它曾拒绝有用视觉候选，本轮因此必须保留去验证与仅精修消融，不能假定硬验证有效。
若示例、图像或数值工具没有单独增益，必须对应地取消该贡献叙述或方案采纳。

仅在本项目模拟开发站的分类型对照中最好，可称“本组开发对照最优”；
领域SOTA仍需更强作者方法同数据公平执行、未使用测试确认以及领域数据证据。
