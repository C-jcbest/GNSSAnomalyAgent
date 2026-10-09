# 新背景观测参考复核说明

1. 仅接收公开观测包。按全局原始 N/E/U 位移和三个固定局部窗阅读，不查看机制、设计类型、候选预测或其他复核者答案。模板初始为 `pending`，这是无标签状态。
2. 对可确认的持续变化填 `episodes`。每段给出 `possible_start <= confirmed_start <= confirmed_end <= possible_end` 和 `raw_evidence`，闭区间、从 0 开始的原始日索引。确认起止是保守内部边界，不是精确物理起止。
3. 明确审查过的稳定范围填写 `stable_ranges` 的 start/end/raw_evidence。不得与任何活动的可能范围交叠；不因未填活动而默认全记录稳定。未覆盖、边界待定、无法判定、缺测均未知。
4. 记录实际 `reviewer`、`provenance` 与 `activity_notes` 后改为 `activity_status: reviewed`。来源如“领域专家独立复核”“AI 辅助草案，尚无专家复核”，按实际情况填写；输入哈希／原图哈希不得更换。填写身份不自动赋予独立性。
5. 使用 `activity_review_sha256(review)` 冻结原图活动审查，将所得哈希写入 `stage_activity_sha256`。此后再打开 31/61/91 日辅助图，在确认活动内部填 `stages` 的 start/end/feature/evidence。活动编辑后必须重新审查阶段并重新冻结，不沿用旧阶段哈希。
6. feature 仅为 low_speed_deformation / acceleration / steady_motion / deceleration；无法确定不用填。阶段不能落在平台或边界待定范围。转折存在不确定时留空，不强制把每个活动日分成确定阶段。
7. `compile_observation_review` 统一产出逐日 activity_label=-1/0/1 与 feature。缺测活动与阶段均未知；阶段额外应用固定 61 日支撑，A/D 还需要切向估计支撑。原图活动不因辅助缺测而删除，阶段也不能先于活动。

此轮网页显示的是未发送模型的提示词草案，并无模型回答、专家标签或新分数。独立双人／专家复核与分歧裁决尚未完成，不能把空模板称作标注完成。

项目内提供导出工具。先将真实完成的原图审查写入单条 JSON，在没有 stages 时执行：

```powershell
$env:PYTHONPATH='src;scripts'
& .venv/Scripts/python.exe scripts/export_landslide_observation_review.py --packet artifacts/landslide-background-2026-10-08/batch-v1/public --review <完成的审查.json> --activity-hash-only
```

把打印的活动 SHA-256 写入同一审查 JSON 的 `stage_activity_sha256`，再进行阶段复核。导出时去掉 `--activity-hash-only`，加 `--output <新建的参考目录>`。该工具不会验证填写者是否具备专家资质或独立性；来源仍需研究负责人核实。编译后的标签不能覆盖旧参考文件。

仅凭单个 GNSS 位移序列，仪器平滑漂移与真实缓慢形变可能不可辨。可记录可观测活动，同时把物理归因留待额外质量／场地证据；不可因为隐藏机制为静稳就直接标为稳定负例。
