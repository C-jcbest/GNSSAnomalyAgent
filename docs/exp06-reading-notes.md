# EXP06：分段位移与视觉核验阅读记录

2026-10-14。针对EXP05暂态/活动混淆继续阅读；不是系统综述。

| 已读来源 | 本轮读取范围 | 可用依据与边界 |
|---|---|---|
| Truong、Oudre、Vayatis，*Selective review of offline change point detection methods*，Signal Processing 2020，[DOI](https://doi.org/10.1016/j.sigpro.2019.107299) | 新获取[arXiv v3全文](https://arxiv.org/html/1801.00718v3)，§4.1.2、§5.1.2、§6.1、§8 | 分段线性代价、PELT与复杂度惩罚有明确来源。惩罚过高漏变点、过低响应噪声；BIC依赖模型假设，不能叫无阈值。本文GNSS三轴代价与类型映射属于适配。 |
| Fearnhead、Grose，*cpop: Detecting Changes in Piecewise-Linear Signals*，JSS 2024，[DOI](https://doi.org/10.18637/jss.v109.i07) | 复核正式元数据与既有§3—4全文缓存 | 原始信号的斜率分段适合提供活动候选；本文不运行作者R包，不将本项目有界轨迹回归冒称CPOP。 |
| He等，*VLM4TS*，AAAI 2026，[DOI](https://doi.org/10.1609/aaai.v40i26.39319) | 复读既有arXiv v2中的resolution-context dilemma和localization/verification段 | 支持数值/视觉分工与全局加局部语义核验。它用预训练ViT筛查，本文用站点标准化位移回归与训练RF尖峰，不能说完整复现或首次两阶段。 |
| Wang等，*ChangePointCNN-GNSS*，JGR: Machine Learning and Computation 2025，[DOI](https://doi.org/10.1029/2025JH000910) | 复读既有正式全文§2—3方法 | GNSS数值变点配置与图像评价的领域先例；原任务为速度估计，长观测跨度、固定幅值和CNN评价阈值不能直接搬入本轮四类识别。 |

新获取文件与已安装PELT/CostLinear实际源码快照、SHA256在`papers/exp06-segmentation/manifest.json`。
另通过Crossref核实Killick等PELT原论文 *Optimal Detection of Changepoints With a Linear Computational Cost*，
JASA 2012，DOI `10.1080/01621459.2012.737745`；本轮该项只读正式元数据，
算法理解依据Truong综述与实际Pelt源文件，不声称读完原论文全文。
GitHub标签API请求失败已记入manifest，不编造已锁定仓库commit；本轮使用确切PyPI版本及安装源码哈希。
作者`CostLinear`接受首列单响应与其余协变量，不能直接塞NEU三响应；新增三响应代价明确为本项目代码。
直接库`Pelt`默认min_size=2、jump=5，`predict`必须提供pen或其他停止条件，没有自动领域惩罚默认。
本轮min_size=7、jump=1、6log(3n)全部属于披露的GNSS配置。
已实际执行Pelt默认构造（l2）和README的`Pelt(model="rbf").predict(pen=10)`，
默认构造与示例完整配置分列；pen=10不自动解释为领域默认。两者边缘任务结果与GNSS适配分列。

拟合中的白噪声、轴间归一化和单活动假设接近本项目简化任务，实际缺测、相关噪声、多期活动需后续验证。
不据仿真高分推断真实滑坡部署效果，不填任何未执行近期模型成绩。
