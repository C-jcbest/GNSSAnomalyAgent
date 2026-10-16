# 12条原始位移记录的 Codex 视觉与数值辅助复核

2026-10-10。用户授权由当前会话直接使用视觉能力并结合其他辅助完成复核。本轮是 **AI辅助开发参考**，已接触历史方法结果，不是独立专家金标准；复核过程中未读取本批生成标签或旧预测来决定标注。原始未审包保持原字节，新结果另存，不回写旧实验参考或成绩。

## 实际复核与主要判断

逐例查看12张原始全局图及12张固定局部图，读取非重叠30日块中位数/IQR；先保存并冻结完整活动分区，再查看12张31/61/91日速率与切向加速度叠图，检查候选阶段区段的有效支撑和分段速率。辅助拟合共享原始观测，只用于敏感性检查，不能当成三份独立证据。边界采用原始日索引半开区间，保留未知带，不能解释为精确物理起止。

| 记录 | 保守确认活动内部 | 主要复核判断 |
|---|---|---|
| 0049 | 无 | 三分量反复起伏，尖峰没有持续累计 |
| 0050 | 无 | 约205日突跳前后平台；突跳带及末段E弱漂移保留未知 |
| 0051 | [220,390) | 弱持续位移可确认；只有[250,290)粗尺度近恒速，其余阶段未知 |
| 0052 | [135,315) | 持续位移后平台；约390日突变不新建活动；缺口附近阶段未知 |
| 0053 | [115,225)、[510,620) | 两轮运动分开；中间平台禁止阶段；第二轮起段速率冲突保留未知 |
| 0054 | [60,210)、[470,650) | 局部图确认后期弱持续位移；弱段A/S/D仍未知 |
| 0055 | 无 | U暂时偏低后回升，N/E无持续共同累计 |
| 0056 | 无确定内部 | N/E单突跳后平台，U长期缓慢变化与漂移不可辨；[270,730)未知 |
| 0057 | [225,420) | N/E弱累积确认，内部阶段未知；后期U单分量变化保留未知 |
| 0058 | [135,305) | 明显持续位移，缺口影响起段速率支撑，确认内部后半段减速 |
| 0059 | [90,225)、[490,605) | 两轮活动；长窗较平但短窗下降的候选S降为未知 |
| 0060 | [70,225)、[480,640) | 第一轮明显活动及第二轮弱活动；弱段不因速度小而判S |

“无确定内部”不等于整条正常。例如0056未确认活动，也未将其长时U漂移判作静稳。活动指可观测持续位移，单GNSS曲线不能独立证明地质归因。活动外无A/S/D，导数提前响应不扩展活动范围。

共12个确认内部、8,760日历行。缺测380日，两层均未知。第一层有效观测中确认活动1,755日、静稳5,223日、已复核但不确定1,402日；连同缺测，导出活动标签-1为1,782日。第二层在确认活动观测日内：A=114、S=164、D=313、未知=1,164，共591日具有确定阶段。这些是标注覆盖数量，**不是识别准确率或纠错收益**。

数值辅助否定了几项最初候选：0058[185,220)只有8日61日速度支撑、91日支撑为0，不能用附近平滑平台判S；0059[130,155)的31日分段速度0.806/0.748/0.703与长窗较平解释冲突；0059[525,550)的61日速度0.572/0.604/0.641仍在增加。0060的高速度候选及弱段也保留阶段未知。最终分区、理由和这些修正均另存，未为增加阶段覆盖强行赋类。

## 可追溯产物与复现

根目录：[codex-observation-review-v1](../artifacts/landslide-adaptive-review-2026-10-09/codex-observation-review-v1)。

- [活动判断](../artifacts/landslide-adaptive-review-2026-10-09/codex-observation-review-v1/activity-decisions.json)及[先行冻结](../artifacts/landslide-adaptive-review-2026-10-09/codex-observation-review-v1/activity-freeze.json)。
- [阶段判断及修正理由](../artifacts/landslide-adaptive-review-2026-10-09/codex-observation-review-v1/stage-decisions.json)。
- [逐日CSV](../artifacts/landslide-adaptive-review-2026-10-09/codex-observation-review-v1/compiled/daily-reference.csv)、[完整复核JSON](../artifacts/landslide-adaptive-review-2026-10-09/codex-observation-review-v1/compiled/review.json)和[覆盖统计](../artifacts/landslide-adaptive-review-2026-10-09/codex-observation-review-v1/compiled/summary.json)。
- [实际36张看图清单](../artifacts/landslide-adaptive-review-2026-10-09/codex-observation-review-v1/compiled/inspection-ledger.json)、[分段速率检查](../artifacts/landslide-adaptive-review-2026-10-09/codex-observation-review-v1/compiled/rate-checks.json)及[数值支撑清单](../artifacts/landslide-adaptive-review-2026-10-09/codex-observation-review-v1/compiled/support-inventory.json)。
- 12张 `compiled/case_XXXX-review.png`直接使用同一逐日标签；[0054弱尾图](../artifacts/landslide-adaptive-review-2026-10-09/codex-observation-review-v1/compiled/case_0054-review.png)与[0059阶段未知图](../artifacts/landslide-adaptive-review-2026-10-09/codex-observation-review-v1/compiled/case_0059-review.png)可作为示例。

材料准备复用原始块统计和既有Huber局部二次拟合；编译复用仓库的活动分区解析器与阶段参考契约，没有另造自动标注算法。以下命令在**新目录**复现观测图和显式标注编译；视觉判断本身是本会话人工逐例作出的AI判断，重运行绘图不等于重新复核。

```powershell
$env:PYTHONPATH='D:/CodeWorkSpace/AgentWorkspace/GNSSAnomalyAgent-sim/src'
& artifacts/reproduction-2026-10-07/environment/Scripts/python.exe scripts/prepare_landslide_codex_observation_review.py --pack artifacts/landslide-adaptive-review-2026-10-09/independent-review-pack-v1 --out <新目录>
# 将本轮 activity-decisions.json、activity-freeze.json、stage-decisions.json 复制到新目录，保留原冻结时间和哈希。
& artifacts/reproduction-2026-10-07/environment/Scripts/python.exe scripts/compile_landslide_codex_observation_review.py <新目录>
```

本轮新增远程模型请求0，不给当前会话视觉阅读虚构远端token账单。既有累计成本、受控/半合成人口、41组冻结对比均不变。初稿v24使用 `$...$` 和 `$$...$$` LaTeX，新增本轮复核披露；新参考只支持开发分析，正式独立观测效果仍需要可信且未参与方法选择的参考。

验证已通过：原空包/准备材料/冻结标注哈希、活动先行时间顺序、8,760行CSV与JSON一致、图色带掩码一致、缺测与阶段门控、共享阶段参考编译器重放。既有活动分区和阶段参考40项测试通过，三个本轮Python脚本Ruff通过。初稿24个行内公式和4个独立块公式经仓库已有KaTeX 0.18.9严格语法与渲染检查通过，公式预览及核验清单保存于manuscript-v24/math-preview。
