# EXP05方法实施前的文献续读

阅读日期：2026-10-14。承接2026-10-13真实来源调研，本轮复读已缓存全文的具体方法段，不冒充新增检索或源码执行。

## VLM4TS：定位与全局核验

He等，*Harnessing Vision-Language Models for Time Series Anomaly Detection*，AAAI 2026，
DOI [10.1609/aaai.v40i26.39319](https://ojs.aaai.org/index.php/AAAI/article/view/39319)。
实际方法阅读版本为[arXiv v2](https://arxiv.org/html/2506.06836v2)，
本地文本`papers/evidence/vlm4ts_arxiv_text.txt`的“VLM4TS: Anomaly Verification and Refinement”段。

作者先由ViT4TS提供候选端点，再向视觉模型提供完整序列单图及候选文字，进行全局判断与区间修正。
提示包含候选来源、拒绝符合全局背景的误报、增加遗漏区间等要求，输出置信度1—3并丢弃最低等级。
因此“数值/视觉分工”和“单图+候选”已有近邻，不能将本项目的少图组合包装成首次提出。
该论文的候选由视觉编码器生成，本文由监督RF生成；作者默认GPT-4o，本项目沿用qwen3.8-flash；
作者通用异常区间与本文四形态标签、严格点日评价不同。本轮新定位器不是其源码复现。
作者文中效率提升基于其数据和模型，本项目必须按自己的usage与耗时实测，不能引用为本方法提升幅度。

## ChangePointCNN-GNSS：领域内变点与视觉配置评价

Wang等，*ChangePointCNN-GNSS: An AI Model for Assessing Change Points and Optimizing Site Velocity
Estimation From Global GNSS Data*，2025，DOI [10.1029/2025JH000910](https://doi.org/10.1029/2025JH000910)。
实际读取作者托管正式PDF第3—4页的瞬时/过渡变点及方法段；
文本`papers/optimization-20261013/changepointcnn_2025_text.txt`。

作者将瞬时水平变化与持续非线性位移边界分开，候选方案交给CNN评价长期速度估计适合性。
滑动窗比较相邻30日均值，预先列举水平/垂向厘米阈值，并限制30日候选间隔。
原文还有5米极值清理、360日窗口2.5-sigma迭代预处理及4年以上段限制。
这些属于该论文的实际设计，而非本项目推荐值；不照搬到本站相对尺度与730日四类活动识别。
也不按其大缺口边界加变点规则给本项目缺测日期强行标异常。

## 本轮取舍

EXP05检验的是把视觉确认的永久变化形态与观测驱动的精确日期分开，而非扩大提示或引入多代理。
共享局部趋势加水平跳变是本项目可审查的回归模型；每侧5个观测是披露的最小支持参数。
检验局部图是否有益需要三种输入同站对照；未获得新结果前，不预写“少图更优”。
近期作者源码基线仍待独立执行，不能把上述文献阅读当作与这些方法的性能对比。

## TimeRCD源码与在线演示核查

本轮取得ICML 2026 TimeRCD官方Git仓库，锁定提交`372bb980426b2f67007311c6f3165ab789c79bef`。
读取`detector.py`、`_inference.py`及Tutorial后确认：multi默认窗口5000、batch=1，短序列使用完整长度，
先做全序列均值/标准差归一化，返回softmax异常后验及异常/正常logit差；不处理输入NaN，API不提供默认硬判决。
因此缺测插值及二类argmax均须单列为本项目适配；详情见[补充协议](experiment-exp05-timercd.md)。

官方Hugging Face模型版本已锁为`0880420070a524efe14d7e163b34dcca90d2be1b`，
multi权重大小444841116字节、官方LFS SHA-256为
`a497ac910f24f8fb347d690a21b60d74bc04785ebc777880553c158dea4434d0`。
单连接、镜像与首版分段传输均遇到响应不完整，文件残片未加载或充当完整模型。
后续使用官方hf客户端获取；只有完整哈希核验后才可启动模型实验。

另读作者在线演示源码版本`51ed77549971e9226ccd7359227ca4736d7bad84`的app与TimeRCD模型文件。
演示app仍从`checkpoints.zip`提取旧命名权重，不是当前轻量API所指的`best_model/pretrain_checkpoint_best_multi.pth`。
不能把在线演示输出直接当作锁定的新权重结果；本轮未上传测试数据到该演示、未填在线分数。
来源缓存与权重元数据见`papers/exp05-timercd/`，访问范围如上，不据元数据声称模型执行完成。

## Material Passport

材料类型：针对新实验的全文续读与设计映射。来源：已取得作者/出版来源缓存，版本与读取范围如上。
状态：阅读完成；作者模型本轮未运行；方法增益待新验证与新测试。不是系统综述。
