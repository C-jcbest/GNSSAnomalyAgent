# EXP37补充领域调研：可比任务与作者实现

2026-10-17，独立测试运行期间的有界补充检索，不用于修改冻结方案。
arXiv按GNSS/GPS、time series、anomaly/change point检索8条上限，实际返回6条；
其中GPS也匹配Gaussian processes缩写，不把这些结果当作GNSS新研究。
Crossref另检索2023年后的GNSS displacement anomaly landslide前6条元数据。
原查询、响应与SHA保存在`papers/exp37-domain-search/`，不是系统综述或最新性保证。

## 实际补读的领域作者实现

**Anomaly detection in weekly SIRGAS GNSS time series: a TimesNet assessment in real data**，
2025-03-15，DOI：https://doi.org/10.1007/s10291-025-01839-3 。
本轮核对已保存正式发表Crossref元数据，补读作者提交
`7bf4e9f31215fa93265d7b59e6fcd744924d163d`的`StationTrainer.train`及参数。
没有把较早ResearchSquare预印本当作2025正式版全文已读；本轮未新读正式版全文。
原作者库：https://github.com/mauriciodev/deep_anomaly_gnss 。

代码中模型拟合与decision_function作用于同站序列，分数再MinMax缩放，
`StationTrainer`默认阈值为该站分数99百分位，默认50 epochs；仓库另有站点参数覆盖。
这与跨站统一毫米阈值不同，但仍不能擅改百分位后称作者默认。
该模型给出通用异常分数，不直接给GNSS单点/阶跃/暂态/持续形变四类语义。
未来对照必须分别标“作者默认同站拟合”和“正常前缀训练等本项目适配”，
统一观测掩码与通用检测指标；不能依靠真值把原模型预测分配成四类。
未执行该模型，因此仍无本数据成绩，不从论文原数据成绩推出本项目优势。

## 仅元数据筛选，未当作方法证据

Crossref还返回SSRN的“A Novel Method for Anomaly Detection and Correction of GNSS
Time Series”、会议摘要和TCN-Transformer滑坡位移文献。
本轮只得到其标题/DOI等元数据，未读正文、未核实代码和评价设置，
不将其列为已复现方法或“优秀近期方法已被超越”。原始命中保留，避免编造细节。

领域对照应优先解决任务一致性：位置位移与IWV/原子钟不是同一检测任务；
永久断点日识别与持续范围识别不同；通用异常与多类语义也不同。
本轮OWV的可支持贡献仍是同信息图/文的暂态范围差异，完整领域SOTA需要更多同任务作者对照。
