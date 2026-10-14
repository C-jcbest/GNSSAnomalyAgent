# EXP08：GNSS尖峰独立学习与视觉位移分解融合

2026-10-14，生成新开发数据之前登记。保留稳定EXP06源码`23be8da`和完整备份，
EXP07失败试验提交`a0151f3`另标记`backup/exp07-rejected-a0151f3`。
实验分支`codex/exp08-independent-spike-study`；生产模块暂不改变。

## 假设与候选

现有48维形态RF同时学习四类型，balanced多输出类权重按列相乘。
稀疏尖峰与其他形态共存会改变样本权重；独立二分类可能改善尖峰召回和误报，
避免漏尖峰污染分段。这是待验证的项目训练假设，不是文献已证实的GNSS结论。

- `binary_rf_numeric`/`binary_rf_repair`：48维现有形态特征、单独尖峰标签、
  sklearn RandomForestClassifier，class_weight=balanced，random_state=数据种子，其余默认。
- `binary_et_numeric`/`binary_et_repair`：相同输入、标签、站点与训练范围，
  ExtraTreesClassifier，class_weight=balanced，random_state=种子，其余默认。
- `binary_rf_default_numeric`：同独立任务RF但class_weight=None，作为训练权重消融，
  不参与主候选选择，不额外调用视觉。
- 原`segment_numeric`/`pair_focus_repair`与`rf_balanced`同组控制。

训练仅使用96训练站的可观测异常分析期，和原RF一致；正常历史MAD归一化和训练中位数插补。
二分类predict替换原RF的尖峰集合，不取并集，不改变其余数值与视觉规则。
新增模型只决定尖峰，原PELT/有界位移活动/配对保持原实现，视觉使用原EXP06提示和图。
没有新增跨站物理阈值，不扫描分类概率阈值，不扫模型超参数。
默认n_estimators等参数与balanced/random_state适配分别保存；通用库实现不冒称论文原代码复现。

## 数据、指标与决定

新开发种子2026101410，96训练/24验证，测试分区不读取。
先数值预筛；数值主指标相对原分解增量至少0.02且任一类型不下降超过0.05，
才请求该候选的完整视觉开发对照。原SDR-VLM控制使用同站独立请求。
视觉先固定最多6个观测歧义站验证，再执行剩余请求；无歧义明确不请求。
完整开发四类等权日F1至少+0.02，任一类不下降超过0.05，才冻结最高分候选。
候选差距≤0.005选择RF，便于和原RF直接比较；不纳入消融选择。

仅通过开发时才生成新测试种子2026101411、2026101412各24站，训练各96。
正式48站合并增量至少0.02、两个种子方向为正、任一类型不下降超过0.05才升级。
否则取消集成，保留稳定版和所有负结果。主指标一直为按类汇总计数再等权日F1，
不做point adjustment；事件F1、每类日F1、通用F1、正常站误报与bootstrap同时保存。
旧EXP06/07测试均不作为本轮未见测试，也不参与参数选择。
