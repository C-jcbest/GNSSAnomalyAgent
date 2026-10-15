# EXP12：固定视觉任务的模型能力对照

2026-10-15，用户明确同意增加qwen3-vl-plus。保留.env及原模型结果。
在任何该模型请求前登记；复用EXP11开发种子2026101502的同24验证站、训练参考、
图像、提示、数值边界、单次temperature=0、enable_thinking=false和三种活动输入。
同样先6站小批，再18站，不按小批分数选站。原水平修复也使用新模型并单列，
因此比较activity_reference与本模型pair_focus_repair才是活动职责增量；
与flash的差异是模型因素，不应全部算作新活动职责。

这是已见数值基线开发数据上的模型比较，不是独立正式测试，不合并到EXP06成绩。
主候选仍为activity_reference；activity_zero、activity_text为消融，不能事后改主候选。
四类等权日F1、活动日F1、事件F1、通用日F1及各类别均报告，不做point adjustment。

EXP11活动F1基线0.9409969481，原+0.06门槛在此批不可达；保留原协议和其自动判定。
本模型对照仅作开发诊断，无自动生产采纳。若出现实际收益，须另登记新数据确认方案，
可以采用与任务上限相称的门槛，但不得回写本轮为预先通过，也不得只选有利指标。

阿里云官方vision文档已确认qwen3-vl-plus图像能力和enable_thinking=false支持；
本代理服务只确认/models列出该名称，是否支持现有json_schema以小批实际响应为准。
请求模型和服务返回模型都保留；不能独立验证代理后端身份，不将名称当做权重证明。
官方文档还建议一般图像对话避免system message，本轮为隔离模型因素保持原提示组织；
如将来采用该建议，须作为独立提示适配，不能混入本轮。

来源：https://www.alibabacloud.com/help/en/model-studio/vision
本地原文：papers/exp10-visual-evidence/qwen-vl-official.html。
