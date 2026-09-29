# 数据生成

## 背景与分组

协议 `synthetic-v1`：固定2025年365日，参考坐标为零，单位毫米。N/E/U为独立零均值 Gaussian，标准差分别为0.5/0.5/1.0。无季节项、相关噪声或缺测；不裁剪噪声、不按曲线重抽。

`all` 按 Normal、Global Extremum、Trend、Transient Mean Shift 顺序组成同背景四例，数量必须为4的倍数。组号控制分量和正负方向；24例覆盖三轴正负。请求seed和组号共同派生噪声及位置随机种子。相同seed的重复批次不是新独立样本；校准与评价使用不同seed，禁止同背景跨集合。

## 异常与标签

| 类型 | 实现 | 参数 | GT |
| --- | --- | --- | --- |
| Normal | GutenTAG Polynomial / Consolidator | 零多项式、标准空间独立高斯噪声 | 无 |
| Global Extremum | GutenTAG AnomalyExtremum | length=1、local=false、min按方向指定 | 单日Point |
| Trend | TODS collective_trend_outliers | radius=45、斜率绝对值3/89 | 90日Range |
| Transient Mean Shift | GutenTAG AnomalyMean | length=14、offset=±3、creeping_length=0 | 14日Range |

异常活动区间位于日索引60–304内，两端均含。标准空间结果最后逐轴乘σ转换为毫米。参数是本项目配置，不是论文统一默认值。

极值使用上游噪声保护逻辑；零基底下净偏移由 `2 × max(abs(noise))` 决定，然后叠加当天噪声，不固定最终±3σ。趋势首日增量为零，末日累计±3σ；之后保留位移但不扩活动标签。临时均值偏移全区间加±3σ，结束后恢复原背景。范围的进入/退出不另标Point。

## 上游来源

- [GutenTAG](https://github.com/TimeEval/GutenTAG/tree/0bf06e601d57f1f1e4e2d52cfbfa27871dd6a021)：MIT，commit `0bf06e601d57f1f1e4e2d52cfbfa27871dd6a021`。直接调用原生极值、均值、背景和组合器。
- [TODS生成器](https://github.com/datamllab/tods/blob/5bd1fb9b3c574146f89935a944614a5f1f67c915/benchmark/synthetic/Generator/univariate_generator.py)：Apache-2.0，commit `5bd1fb9b3c574146f89935a944614a5f1f67c915`。仅增加显式中心位置参数，原斜率、残余和标签语句保持不变。原文件、适配文件、补丁和许可证见 `vendor/tods/`。
- [VisualTimeAnomaly / TSAD-Agents](https://arxiv.org/html/2502.17812v2)作为研究参考，其趋势代码追溯至TODS。本项目不是完整论文benchmark复现。

源码、下载档案哈希及依赖见 `configs/generator-sources.json`；生成环境锁见 `configs/generator-requirements.txt`。worker启动时校验上游源码和依赖，失败则停止，不回退到自写公式。TODS全局随机状态隔离在子进程中。

## 数据文件

每批数据在 `data/generated/<dataset-id>/`：`manifest.json` 管理生成进度、类型和背景组；`cases/<case-id>/input.json` 仅含观测及确定性派生；`truth.json` 保存噪声、净改变量、原生标签和事件。

H/R3D分别是相对参考坐标的水平和三维偏移模长，不是路径长度。网页可读取真值叠图供研究检查；检测图和检测输入不含真值。
