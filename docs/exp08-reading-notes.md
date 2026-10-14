# EXP08定向论文与训练机制核查

2026-10-14。近期主要参考保持VLM4TS（AAAI2026）、ChangePointCNN-GNSS（JGR2025）
和AnomLLM（ICLR2025），本轮针对候选定位的尖峰学习环节继续核查。

| 来源与读取范围 | 对本轮的支持 | 边界 |
|---|---|---|
| [VLM4TS](https://doi.org/10.1609/aaai.v40i26.39319)，既有方法全文与正式摘要 | 轻量候选定位与视觉上下文核验的分工；本轮修改候选定位，视觉角色保持原版。 | 作者以预训练ViT定位，不以该论文为独立RF已有效的证据，不称完整复现。 |
| [ChangePointCNN-GNSS](https://doi.org/10.1029/2025JH000910)，既有正式全文§1–2 | GNSS解析/图像模块结合和位置变化形态区分有领域先例。 | 原任务长时间速度估计，不直接复制其阈值循环。 |
| Gabsatarov、Vladimirova，*Machine Learning for GNSS Time Series Analysis in the Time Domain*，2025，[DOI](https://doi.org/10.2205/2025es001018)；本轮新取得Crossref元数据与摘要 | 摘要报告GNSS统计学习建模、跨不同构造地区测试，是进一步研究的领域线索。 | 正文链接访问失败；不能从摘要推断其分类器、参数或异常F1，本轮不填它的对照分数。 |
| Nguyen、Tran，*Detecting outliers in GNSS position time series using machine learning techniques*，2023，[DOI](https://doi.org/10.46326/jmes.2023.64(4).03)；本轮Crossref摘要 | 摘要明确比较Isolation Forest、OC-SVM、LOF，任务为GNSS离群点，说明点型任务与持续形变应区分。 | 摘要的15%离群比例和ROC结果来自作者任务；本项目极稀疏尖峰、四类F1不同，不能直接横比。本轮未获得全文/代码。 |
| [sklearn RF官方文档](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)及已安装`_forest.py`/`_class_weight.py` | 官方明确多输出各列类权重相乘；实际源码传递多输出权重到森林。独立二分类改变样本权重及分裂目标，是可验证的机制。 | 这是库接口/源码事实；并不保证GNSS效果改善。balanced和单目标都是项目适配，其余默认参数实际保存，禁止描述为原论文方法。 |

网页响应/摘要和已安装源码缓存见`papers/exp08-independent-spike/manifest.json`。
论文正文访问失败已记录，未把已读摘要写成已读全文。
不再重复EXP07提示扩展或局部RSS尖峰规则；本轮以训练标签隔离和原提示不变为核心对照。
既有`rf_independent`是30维四个独立分类器，本轮为48维尖峰专属模型，
不把“独立分类”称首次创新，GNSS视觉/结构分工仍为主线。
