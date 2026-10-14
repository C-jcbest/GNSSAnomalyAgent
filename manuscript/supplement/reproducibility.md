# SDR-VLM复现材料与方法映射

本材料对应《融合位移结构分解与视觉语义修复的GNSS多类型异常识别方法》，实验编号GNSS-VLM-EXP06-v1，结果键pair_focus_repair。旧EXP04条件保存在reproducibility-exp04.md，不能用于解释新版表格。

## 数据与冻结运行

开发种子2026101404，使用96训练站和24验证站；主测试种子2026101405、2026101406，各96训练、24验证、24测试。主测试共48站、每场景8站，每站730日、365日正常前缀。每个种子独立训练。生成机制、参数与声明以docs/experiment-exp06.md及冻结source/method-protocol.md为准。

原始观测和真值位于data/exp06-seed2026101405、data/exp06-seed2026101406；冻结运行位于runs/exp06-seed2026101405-v1、runs/exp06-seed2026101406-v1。run内source保存实际源码，predictions保存事件、掩码、请求和响应，registration记录数据和来源哈希。data与runs是本地实验材料，未纳Git；公开复现时需另行提供脱敏完整数据及运行归档。

## 正文与结果身份

| 正文名称 | 结果键 | 身份 |
|---|---|---|
| SDR-VLM | pair_focus_repair | 本文冻结主方法；配对父响应来自pair_focus |
| 位移结构分解 无视觉 | segment_numeric | 与主方法相同数值路径；无视觉修复 |
| 去额外慢运动扣除 | pair_raw_repair | 同候选、同提示角色、同修复规则；父响应pair_raw |
| 默认随机森林 | random_forest | sklearn默认估计器加30维GNSS适配 |
| 平衡随机森林 | rf_balanced | 显式class_weight=balanced |
| 极端随机树 | extra_trees | sklearn默认估计器加GNSS适配 |
| 孤立森林 | isolation_forest | 正常训练数据拟合，默认predict，仅通用异常 |
| 数值时间解码 | rf_split_temporal | 本项目前期48/30维特征与训练侧解码 |
| 纯视觉 | visual_only | 单轮全局曲线检测 |
| 自由候选视觉 | guided_fusion | 旧完整图与候选辅助输出 |
| 既有MF-VLM | morphology_fusion | 本研究前期形态分工方案，同站重跑 |
| 既有完整定位融合 | localized_fusion | 本研究前期数值阶跃定位融合，同站重跑 |

pair_focus和pair_raw仅执行返回配对子任务，不能在完整四类排行榜中解释其缺失类别为模型能力。派生repair行本身不新增请求，主方法实际成本来自pair_focus的7次请求。

## 参数与原作者实现

PELT使用ruptures 1.1.10作者搜索源码，未修改搜索实现。本文添加三响应线性代价，min_size=7有效点、jump=1、惩罚6log(3n)，保留正常尾90日。局部边缘±3日、两侧最多21日、每侧至少5有效点，准则RSS+3p log(3n)。有界活动最短15日、最多1段，7日粗网格与±7日细化，惩罚5log(3n)+2log2。返回间距至多120日、上下文各30日。辅助连续样条结点间隔21日。视觉修复两端允许各±7日，最终采用数值边界。

尖峰使用48维多输出平衡随机森林的尖峰列，100棵树，class_weight=balanced，库predict判决。特征空项由训练集均值填补；历史training.json的median文字与冻结SimpleImputer()不一致，本文按实际均值实现记载。各运行训练随机种子以冻结注册和训练元数据为准，不把随机权重修改称为原库默认。

视觉服务标识qwen3.8-flash，temperature=0，enable_thinking=false，timeout=120秒，无自动重试。服务端权重未本地固定。主方法失败保留数值结果并记录视觉依赖失败；纯视觉和旧完整融合失败计零检测。开发76请求、正式110请求，其中主视觉7、图像消融7、旧完整48、纯视觉48；后续扩展请求另记，未计入186次开发加主测试总数。

作者PELT对照单列：API默认l2及README rbf示例均用pen=10、min_size=2、jump=5。pen=10属于作者示例值；共用正常残差与缺测原时间映射属于输入适配。论文表7报告77个非尖峰真边缘的严格/±3日F1，原库不输出四类标签。

## 评价与证据入口

- results/exp06-main-v1/summary.json：主表、类型计数及成本。
- results/exp06-main-v1/paired-bootstrap.json：合并48站5000次配对抽样；固定种子2026101404，非种子分层。
- results/exp06-main-v1/activity-errors.json：仅16个真活动站的边界误差；全48站活动假日另查类型FP。
- results/exp06-pelt-main-v1/summary.json：原作者PELT与适配边缘结果。
- manuscript/working/exp06-evidence-audit.json与.md：原始预测重算、精确数值、配置和来源复核。
- manuscript/working/exp06-method-source-map.md：方法段落与冻结实现映射。
- manuscript/working/exp06-literature-check.md：文献版本、实读范围及新增引用核查。

正文扩展验证来自EXP07种子2026101408/1409和EXP08种子2026101412/1413的原版SDR-VLM及数值消融；每批48站。配置和源码与EXP06一致，分别报告，未合并成预登记144站，也未纳入未采纳的候选改动。EXP08种子1412的视觉增量为负，正文保留。

主指标先跨站汇总每类TP/FP/FN，再等权平均四类F1。事件评价使用有效日IoU和一对一匹配，不是直接实现文献的range-F1或affiliation。缺测排除计数；区间可跨缺测；图像、CSV和评价共享最终标签掩码。所有模型使用实际预测，未进行point adjustment或测试阈值搜索。

## 论文重建

build_manuscript.py读取chapters和核实书目；prepare_sdr_figures.py从冻结EXP06结果及两例观测重新绘图，输出figures/exp06；不调用模型。render_word.ps1由独立隐藏Word实例导出PDF，render_pages.py使用文档技能组件逐页栅格化。依赖、命令与最终检查见manuscript/README.md。
