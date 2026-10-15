# EXP29 方法与结果键的对应关系

实验名称：**EXP29：视觉先行定位与数值验证（VFN-GNSS）**。
主方法键：`vf_vl_image_complement`。模型在开发结果查看前选定为 qwen3-vl-plus。

| 结果键 | 实际方法 | 回答的问题 |
|---|---|---|
| `vf_vl_image_complement` | VL 看原图自主给范围→原始位移信息准则细化和核验→补充近邻未覆盖的范围 | 是否带来强数值之外的新检出？ |
| `vf_vl_text_complement` | 同模型、同提示，原图换为相同逐日观测的文字，其他流程相同 | 增量是否来自图像模态？ |
| `vf_vl_image_pure`、`vf_vl_text_pure` | 直接采用模型的原始暂态范围，不作数值验证、不混入其他类别 | 模型自主定位能力如何？纯暂态检测器的四类 F1 不用于与全类别方法评优。 |
| `vf_vl_image_snap`、`vf_vl_text_snap` | 数值边界细化后替换近邻全部暂态，不作脉冲/连续/独立阶跃三模型否决 | 近似日期与精细边界的差异。 |
| `vf_vl_image_verified`、`vf_vl_text_verified` | 完整数值验证后替换全部暂态 | 依赖视觉提出完整暂态集合会丢失多少原正确事件？ |
| `vf_vl_image_raw_complement`、`vf_vl_text_raw_complement` | 直接把模型范围补充到近邻，不经数值验证 | 数值验证的收益及代价。 |
| `vf_flash_image_*` | 首轮 60 站的同流程 Flash 模型对照 | 仅作模型敏感性，不混入 360 站表比较。 |
| `nf_vl` | 数值先行 CFV-VL：原数值未配对阶跃候选→原图与扣除图视觉复核 | 原受限数值候选流程与新流程比较。沿用 EXP26 提示和绘图，不等于完整重跑 EXP28 所有变体。 |
| `vector_nearest` | 原自由候选上的三轴相反变化向量近邻 | 强数值对照，无模型请求、无新增调参。 |
| `numeric_all_pairs` | 所有 PELT 阶跃两两组合→与新流程相同的边界细化/验证→近邻互补 | 排除仅扩大数值候选图带来的收益。 |
| 六个原数值方法 | `segment_numeric`、`rf_split_temporal`、`random_forest`、`rf_balanced`、`extra_trees`、`isolation_forest` | 身份和适配沿用 EXP28。Isolation Forest 仅一般异常，不伪造四类/暂态分数。 |

所有方法的 `summary.json`、`comparison.csv` 和 `range-metrics.csv` 使用上述键。`paired-changes.json` 给主方法相对各控制的站级变化及 bootstrap。`visual-role-diagnosis.json` 区分模型没有建议、边界不支持、数值解释否决以及最终新增 TP/FP。

`complement` 是事先选定的 add-only 规则，不能修正近邻既有误报，也不会以另一条近似范围覆盖已存在的暂态。`snap`/`verified` 替换消融保留近邻的其他事件，不把删除暂态的两端重新猜成 step；因此它们用于检验暂态集合与后处理，不能冒充完整的重新分配事件所有权的方法。

主终点对应 `summary.methods.<方法>.range.transient.affiliation.station_macro_f1`，是有真暂态的站点宏平均；正常站不在此均值里，须同时读负站误报和总体 FP。暂态日 F1 对应 `range.transient.daily_micro.f1`，四类等权池化日 F1 对应 `original.types_daily_macro_f1`。报告同时保存 Affiliation precision/recall、日 P/R、TP/FP/FN、形变、范围联合、一般异常、覆盖率及兼容性事件指标。没有 point adjustment，也不把仅接近真区间的 Aff 上升写成真实异常日恢复。

AnomaMind、TimeRLM、DAST、SAGE 与 TimeEvo 为本次实际阅读的流程依据；本轮没有执行其完整作者训练和推理框架，没有这些方法的 GNSS 比较分数。本项目新流程及数值窗口均属明确标注的 GNSS 适配。

登记时序说明：数值基线计算期间，在首次视觉请求及首次评分前，补充了独立阶跃 nuisance 项和全部阶跃组合的无视觉控制。因此数值基线回执保留的早期协议 SHA 与最终视觉协议不同；六个冻结数值方法及其输出未改变。视觉流程以 pilot-registration.json 中的协议与脚本 SHA 为准，扩大阶段逐字节核验同一版本。

首轮报告 v1 在汇总成本时遇到“已准备但未执行”的扩大阶段请求而中止，保留 INCOMPLETE.md；最终首轮报告为 pilot-v2，未重新发出请求或修改分数。完整开发报告为 full-v1。全实验成本包含 60 次 Flash 模型敏感性请求；完整 360 站表本身只比较 VL 和数值方法，两者的成本分开保存。

## 明确区分三个后续登记

S1 `joint_vl_image`、`joint_vl_text`、`joint_numeric_all_pairs`：使用有序边界联合工具替代原独立细化，首轮响应不重采样。其主开发Aff 0.8920，只有1个新增TP日、22个新增FP日，未达到预定采纳条件。

S2 `vnv_image`：视觉定位→数值取证→局部原图/扣除图视觉终审，并允许取消与重新分配原配对；`vnv_text_final` 共享相同图像首轮、终审换同信息文字；`vnv_text_pipeline` 两轮均文字。开发主Aff 0.7427、日F1 0.7992，低于近邻0.8825/0.9005，取消集成。不能把更多视觉调用本身当成收益。

S3 **VRC-GNSS：视觉范围互补的独立确认**，是针对原已列出的 `vf_vl_image_raw_complement` 进行的单独后续研究。先看开发后选它作确认对象，明确披露选择过程；新test开始前单独冻结，方法、模型、提示与融合均不改。S3成功与否都不改变原预选主方法、S1、S2的失败结论。其独立测试表为 `results/exp29-s3-test-full-v1`，没有执行的S2终审或近期论文原方法不填入测试表。

S3最终图像Aff-F1 0.9514、日F1 0.9089、四类F1 0.8431，分别对应上述三个summary字段；
同信息文字为0.9236/0.9021/0.8503，近邻为0.9254/0.9246/0.8559。
图像Aff独立收益复现，但日F1相对近邻下降0.015679超过预设0.01，综合确认 `supported=false`。
只保留为范围研究变体；详见 `confirmation.json`、`contribution-audit.json` 和报告正反案例。
