# 数据获取、收束与来源划分

更新：2026-09-23。当前输入是[每日15点数据包](daily15-data.md)。采集与实验分离：导出后冻结快照，实验不得自动回源补数据；模型API可独立调用。

## 数据链路

| 层级 | 本地路径 | 用途 |
| --- | --- | --- |
| 站点清单 | data/raw/stations-expanded-20260922.json | 私有ID与稳定匿名别名映射 |
| 原始快照 | data/snapshots/history-18months-20260922/ | 原始响应、小时网格、checkpoint和manifest |
| 小时来源 | data/processed/offline-v2/ | 收束边缘全空小时，保留内部缺口和划分 |
| 当前日包 | data/processed/daily15-v2/ | 严格15点的全历史与180天候选 |
| 当前注入比较集 | data/processed/daily-comparison-v1/ | 原划分的六窗、144例相关派生；仅注入标签，背景未认证 |
| 原生缺测敏感性集 | data/processed/daily-comparison-native-v1/ | 同六窗、36例；仅保留来源缺口，不增加缺测，与上行`native`输入和标签相同 |
| 历史开发注入集 | data/processed/daily-pilot-v1/ | 冻结开发预实验48输入/标签，仅追溯 |
| 单位证据 | data/labels/platform-evidence-20260922.json | 用户确认米单位、无基准重置 |

`offline-v1`、`daily15-v1`为中间版本。旧14天及120天小时窗仅追溯，不与日结果混排。

## 采集契约

已查询2025-03-22至2026-09-22（不含终点）22站，18站有观测，共80,506条去重站点小时记录。无时区请求按北京时间解释，内部使用带时区时间，查询为半开区间`[start,end)`。接口名称含Daily，不代表日采样。

| POST接口 | 用途 |
| --- | --- |
| UserLogin/doLogin.php | 登录获取会话 |
| Station/getStationListInfo.php | 站点、平台分组和初始坐标 |
| GNSSData/getDailyGNSSDataInfo.php | 按站点和范围读取N/E/U |

按天请求，业务错误、非整点数据或重复时间戳数值冲突会拒绝完成快照，边界重复去重留痕。保存脱敏原响应、observations.csv及SHA256。`fetch/collect --resume`校验checkpoint后续传；无完成manifest的失败目录不能用于实验。上游分页/截断契约尚未完全独立核实。

固定基准位移为 `1000 × (坐标米值 − 初始坐标米值)`，保留有符号N/E/U，不逐窗口重置，不以合位移替代。用户确认单位为米、无基准重置；补充证据引用源哈希，不改旧manifest核实字段。额外质量字段原样保存，不臆定状态。

## 收束与固定划分

每站仅删首末三分量全空小时；任一分量有效即保留边界，零值有效。内部缺测和原值不变，不插值、去趋势、平滑或剔异常。全空站排除但留记录。小时网格由289,872收束至115,389，80,506有效小时全部保留。

用户确认按平台StationGroupUUID隔离来源，种子20260922，四组固定为开发2/验证1/测试1。SCWM-04所在组固定开发，整站单列辅助，同组其他点仅开发。重叠窗及注入/缺测派生继承分组，不能按站点或时间随机打散、交换验证/测试来源。

早期S04/S07为索引别名，须核对本地映射，不把改名当新来源。完整率仅筛覆盖，不证明正常；未批准候选保持approved=false，审核另存版本，不修改冻结包。

历史122个完整小时候选和16个约120天窗保留清单。验证组33窗最大小时跳变均至少413.6mm只是描述统计，未区分周期与非周期，不能据此排除整组长期用途。

## 重建与审计

仅在需要新版本时执行，输出必须是新目录：

```powershell
uv run gnss-exp prepare-offline --source data/snapshots/history-18months-20260922 --stations data/raw/stations-expanded-20260922.json --references data/labels/real-cases.json --out data/processed/offline-new
uv run gnss-exp prepare-daily --source data/processed/offline-new --out data/processed/daily15-new --days 180
```

coverage.csv记录起止、边缘删除量、内部缺测和最长缺口，bundle.json记录哈希。使用前核对源哈希、时区、采样间隔与同源划分；缺失或损坏报错。清单、坐标、标签和快照仅本地保存，凭据不进入日志或Git。
