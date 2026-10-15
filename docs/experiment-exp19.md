# EXP19：固定输入的原生思考模式对照（NTV-VLM）

2026-10-15，请求前登记。Qwen官方vision文档明确qwen3-vl-plus默认enable_thinking=false，
支持true和thinking_budget；Qwen3-VL Technical Report arXiv:2511.21631v1摘要/§1/§2
及§4.5说明其动态分辨率、思考/非思考训练及图像工具思考。已取得全文，不能由报告
推断本代理模型身份，亦不将作者基准改善移作GNSS成绩。
AnomLLM的显式CoT结论不等于更新Qwen原生思考模式必定无益；这里直接验证，
不改变提示词、不要求公开长推理，不接入工具、不增加投票次数。

开发诊断2026101508的48已见验证站，复用EXP18三种已冻结输入：
reference_distribution、distribution_text、no_evidence。每个原request只修改
enable_thinking=true与thinking_budget=2048，保持温度/图片/坐标/提示/schema相同。
不更换模型名字，只用用户授权的qwen3-vl-plus。各方法先首触发站，再全部，失败不补。
检查接口是否实际返回reasoning_content或reasoning_tokens；参数被接受但无可见思考
证据时只能报告“请求了该模式”，不能把相同输出当验证了思考能力。

这不是把EXP18失败重试成成功：每个是独立登记的模式对照，原失败/输出全部保留。
EXP18新1512候选在登记前已固定且不据本轮结果修改。EXP19若开发视觉比EXP18同输入
至少+.01且不低于文字-.005，再在同1512全144验证站增加固定思考对照；
该数据仍是开发，不能因模型模式新就改称正式测试。
其后沿用EXP18的同站采纳门槛与独立正式要求，不扫budget，不据好看F1换主指标。
