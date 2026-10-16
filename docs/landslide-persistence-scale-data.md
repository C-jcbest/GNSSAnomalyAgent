# PP-VN方法与120条新增数据说明

> 更新于2026-10-12。新增数据、标签和19方法全120条对比评分均已完成，见[全量评测结果](landslide-persistence-scale-results.md)。本文是封存初稿v54的补充说明，不替换历史实验或负结果。公式使用Markdown LaTeX。

## 方法名称与具体机制

暂定名称为**持续性配对视觉—数值学习（PP-VN，Persistence-Paired Visual–Numeric Learning）**。目标是识别长期GNSS位移中当前是否存在持续运动，尤其是弱运动与一次偏移后平台的混淆。

输入为三分量原始位移、原日历和缺测掩码。每个待识别中心提取局部NEU曲线图，用冻结的MobileCLIP2-S0视觉分支取得512维表征；拼接18项已有数值特征及18项缺值指示，经小型活动头输出活动概率。它使用预训练多模态模型的视觉能力，当前实现不调用语言生成、Agent工具循环或多智能体。

$$
h_c=\frac{E_{\mathrm{frozen}}(I_c)}{\|E_{\mathrm{frozen}}(I_c)\|_2},
\qquad z_c=f_\theta([h_c,n_c]),\qquad p_c=\sigma(z_c).
$$

关键训练干预是构造**同净位移、同误差与同缺测、不同持续性**的配对。仅在已知运动分量的受控训练数据中，把持续运动替换成一次偏移。设原观测为$x_t=m_t+e_t$，端点为$u,v$，阶跃日为$k$：

$$
\widetilde x_t=x_t-m_t+m_u+\mathbf{1}_{t\geq k}(m_v-m_u).
$$

实现直接复制两个端点的原观测，保证端点数值精确相等；原始缺测保持缺测。潜在分量仅用于构造训练对，评估记录输入不含潜在运动或过程标签。两条曲线净变化相同，训练头需要利用中间的持续性区别。负例重新估计数值尺度，因此不能声称两者所有标准化特征完全一致。

监督分类损失加上配对排序约束，让持续运动中心的活动分数高于对应偏移中心：

$$
\mathcal{L}=\mathcal{L}_{\mathrm{weighted\ BCE}}+
\frac{1}{K}\sum_{i=1}^{K}
\log\left(1+\exp\left[1-\left(z_i^+-z_i^-\right)\right]\right).
$$

视觉编码器保持冻结，主要学习一个16或32单元隐层的活动头。A/S/D阶段沿用共同数值头并限制在活动内部。当前技术差异集中于GNSS持续性训练干预；配对学习与排序损失本身已有先例，不能称首创损失或新增Agent推理架构。该方法使用居中上下文，属于离线回溯识别。

## 已经验证的优势及代价

以下是**此前24条未参与方法选择的受控记录**，不是本次新增120条的结果。对照“同数据增广”保持训练样本相同，只去除配对排序项。

| 指标 | PP-VN | 同数据增广、无配对损失 | 原有强数值XGBoost84 |
| --- | ---: | ---: | ---: |
| 活动Precision | 0.921354 | 0.968103 | 0.935460 |
| 活动Recall | 0.828009 | 0.757112 | 0.821444 |
| 活动F1 | 0.872191 | 0.849705 | 0.874752 |
| 弱运动Recall | 0.562735 | 0.425345 | 0.577792 |
| 事件F1，IoU 0.3 | 0.696970 | 0.645161 | 0.567568 |
| 匹配事件起点MAE，日 | 30.304348 | 25.750000 | 22.523810 |
| 匹配事件终点MAE，日 | 28.043478 | 24.500000 | 27.000000 |

受控数据上，PP-VN比同增广对照提升活动F1 **2.25个百分点**、弱运动召回**13.74个百分点**。机制简单，可以用去配对、打乱配对、去视觉、去数值等对照直接检验；推断无需远程模型调用，但仍有局部图编码成本，不能称零成本或比所有数值方法更快。

这项收益伴随Precision下降和边界误差增大。主方法与同增广的整条记录配对bootstrap F1差区间为$[-0.004395,0.054786]$，跨零；24条不能证明稳定的总体显著优势。此前12条AI观测参考上，PP-VN F1为0.897064，同增广为0.915985，**主方法反而下降**。因此目前适合表述为“受控弱运动召回与部分事件指标改善”，不能表述为全面领先或可靠降低平台误报。

完整来源：[旧过程结果](../artifacts/landslide-adaptive-review-2026-10-09/paired-persistence-confirmation-analysis-v1/primary_process/report.md)、[旧观测结果](../artifacts/landslide-adaptive-review-2026-10-09/paired-persistence-confirmation-analysis-v1/secondary_observation/report.md)、[整条记录bootstrap](../artifacts/landslide-adaptive-review-2026-10-09/paired-persistence-confirmation-analysis-v1/primary_process/descriptive-bootstrap.json)。

## 本轮新增规模及标注

本批ID为`case_0217`至`case_0336`，共**120条，每条730日，87,600条过程逐日标签**。有效观测83,956日，缺测3,644日。固定生成种子2026101141、顺序种子2026101142；生成前保存协议与模型选择，不重抽、不替换记录，不使用本批调参。本次扩充的是评价数据，未增加训练数据或重新训练模型。

| 受控过程族 | 全批条数 | 预指定观测复核条数 |
| --- | ---: | ---: |
| 静稳 | 20 | 4 |
| 单次仪器偏移 | 20 | 4 |
| 弱匀速运动 | 20 | 4 |
| 加速—近匀速—减速 | 20 | 4 |
| 间歇再活动 | 20 | 4 |
| 强活动后弱运动 | 20 | 4 |
| 合计 | 120 | 24 |

这六族比例是实验覆盖设计，不代表现场发生比例。测量误差、离群点和缺测复用现有仿真机制；潜在活动采用受控解析过程。增加数量改善同生成体系内的评估规模，不能替代现场或跨生成机制验证。

**全部120条保存过程参考**：活动、内部A/S/D、事件和弱运动属性由已知生成过程记录，缺测日为未知。它适合检验受控过程恢复，不能作为人工观测标签。

**其中24条完成实际观测标注**：在生成前指定前24个匿名ID，每族4条，具体族名对复核阶段隐藏。Codex实际查看48幅原始全局/局部图、读取600个原始30日观测块，先冻结活动；随后查看24幅多尺度辅助图，再用31/61/91日速率作辅助核查。阶段候选、7处数值辅助修正及证据均保存。在观测标签冻结后才打开本批过程参考；没有本批模型预测用于标注。

观测标签共17,520条，确认活动区间24个，确认活动观测日2,843；静稳11,694日，未知2,983日，其中包含722个缺测日。活动内部可确定阶段788日：A129日、S369日、D290日，其余2,055个活动观测日阶段未知。弱位移可被确认而阶段仍未知，两层任务分别保留证据。

上述24条是**AI开发者观测参考**。复核者知道历史方法与历史标签，不是独立领域专家；标注不证明现场形变成因。其余96条仅具有过程参考，没有冒称逐条AI看图复核。

## 验证与复现

校验覆盖120条原始观测唯一性、与旧24条零重复、六族各20条、预指定复核每族4条、730日完整日历、过程活动与已知速度对应、缺测未知、阶段仅在冻结活动内、图/CSV/JSON同标签及掩码、冻结文件SHA。抽取首、中、末三条（0217、0277、0336）重生成，原始观测与过程标签逐值一致。配对机制已有两组测试共5项通过，四个新增脚本Ruff通过；第三方Matplotlib有弃用警告。

数据位置：`artifacts/landslide-adaptive-review-2026-10-09/persistence-scale-data-v1/`。主要文件：

- `protocol.json`：生成前登记的范围、种子、保留策略和冻结模型选择。
- `inputs/<case_id>/input.json`：120条原始NEU观测；同目录含三尺度派生诊断。
- `mechanism-sealed/reference-labels.csv`：全部120条过程逐日参考；生成细节单独保存在`generation-ledger.json`。
- `codex-review/activity-decisions.json`、`activity-freeze.json`：原始位移活动决策及冻结凭证。
- `codex-review/stage-plan.json`、`stage-decisions.json`：实际观测阶段候选、最终判断和证据。
- `codex-review/compiled/daily-reference.csv`：24条实际观测复核逐日标签；同目录含JSON、24幅标签图及SHA冻结。
- `../persistence-scale-validation-v2/validation.json`：数据与标注校验结果。v1校验保留历史，v2将时间字段明确命名为验证器读取时间，避免冒称全会话首次读取时间。

在仓库根目录，使用已配置研究环境运行，输出目录必须是新的目录：

```powershell
$env:PYTHONPATH='D:/CodeWorkSpace/AgentWorkspace/GNSSAnomalyAgent-sim/src'
& artifacts/reproduction-2026-10-07/environment/Scripts/python.exe scripts/prepare_landslide_persistence_scale.py --out artifacts/landslide-adaptive-review-2026-10-09/persistence-scale-reproduction
& artifacts/reproduction-2026-10-07/environment/Scripts/python.exe scripts/validate_landslide_persistence_scale.py --data artifacts/landslide-adaptive-review-2026-10-09/persistence-scale-data-v1 --out artifacts/landslide-adaptive-review-2026-10-09/persistence-scale-validation-new
```

生成命令可复现原始输入和过程标签；观测标注是实际看图判断，复现时应使用已保存的决策和图像SHA，或由另一复核者重新判断。不能把标注导出脚本执行称为新一轮独立看图。

## 实验适合性与论文判断

当前最合适的定位是**预训练视觉与数值融合的GNSS持续性识别研究**。受控实验能隔离“小型持续性训练干预是否改善弱运动召回”；已有强数值、ML、直接VLM和有限复核对照，比只比较提示词或Agent调用次数更有价值。

此前24条确认中，同增广增益的整条记录区间跨零、AI观测参考出现负收益，历史事实保持。新增120条过程参考下，同增广差+0.015448，描述性区间[0.005656,0.025843]；强数值日F1仍未被超过，新增24条观测子集相对同增广的小幅收益区间仍跨零，且低于强数值。现场分布与独立专家参考尚未验证。发表判断仍需结合贡献与期刊要求，不能保证发表。

本批120条冻结模型预测、19方法统一对比、既有机制对照、分组及计算成本**已完成**。过程主F1 0.908419，弱召回0.773599；同增广F1 0.892971，84日XGBoost 0.909885。主方法相对同增广净纠错+477日，但误报增加757日、事件F1下降0.017786。观测主F1 0.944300，低于84日XGBoost 0.966443。全部记录、负结果和历史结果保留，重放一致，详见[全量报告](landslide-persistence-scale-results.md)。如据此进行新的方法选择，应另留未参与选择的评价数据，不把规模增加直接写成方法效果增加。
