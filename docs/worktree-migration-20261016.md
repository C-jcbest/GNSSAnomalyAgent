# 2026-10-16 研究工作树迁移至D盘

用户已设置Codex工作树根目录为`D:/CodeWorkSpace/CodexWorktrees`并授权迁移当前研究。
实际代码及Git工作目录现为：

```text
D:\CodeWorkSpace\CodexWorktrees\gnss-vlm-research\GNSSAnomalyAgent-sim
```

当前分支仍为`codex/exp35-granularity-fusion`，迁移起点145ffe479862b6ad3ffded24953eda11da00faaa。
通过应用在新根目录建立同提交的工作树，复制并校验当前忽略材料，重建环境后转移分支，
最后使用应用的工作树归档工具撤下C盘旧副本。没有再次归档当前聊天或其他聊天。

## 已验证

- 原22,721个跟踪文件与新目录逐字节一致。Git检出时4个历史JSON的换行被规范化，
  已复制原文件字节并刷新Git索引，确认没有新增暂存内容或代码修改。
- 必要忽略文件1,638个、153,889,461字节（数据、数值模型和`.env`）在工作树外备份，
  副本及D盘目标文件均逐项SHA校验通过。未将密钥或配置内容提交Git。
- 私有备份位于`D:/CodeWorkSpace/CodexWorktrees/.migration-backups/gnss-vlm-20261016`，
  数据ZIP为51,946,151字节，本地配置单独私有保全；完整文件清单和校验记录保存在该目录。
- 按uv.lock在D盘重建`.venv`，模型实际加载成功；模型SHA保持
  `cf053d56aa93060633b4a99e949247fe3f6c1205d4322a00072d88ae5a06a96e`。
- 本地模型配置读取检查通过；EXP35两批源summary、逐站预测、脚本及协议共8项SHA验证通过。
- 4项针对性测试通过，原作者指标代码仍有7条已有SyntaxWarning；未修改冻结代码消除警告。
- 应用附件显示D盘工作树为活动，旧C盘工作树为已归档；Git工作树列表不再包含C盘旧目录，
  文件系统核对其目录已不存在。当前会话保留，未发模型请求。

## 路径与历史边界

原`D:/CodeWorkSpace/AgentWorkspace/GNSSAnomalyAgent-sim/artifacts`及其备份和EXP30数据保持原路径，
无需搬移或修改冻结脚本中指向该共享数据目录的常量。
旧D盘模拟检出仍为dd74c65c的detached HEAD，Git主检出仍在`D:/CodeWorkSpace/AgentWorkspace/GNSSAnomalyAgent`。
后续研究命令应在上面的新D盘工作树执行。

旧结果中记录的C盘`source_directory`属于原运行来源，不进行全库替换，以免改变冻结证据SHA。
核验其文件时，将旧工作树前缀映射为新D盘前缀，保留相对路径不变即可。
本次仅迁移清理后实际存续的文件，**不会恢复上轮已缺失的早期C盘忽略材料**；
相关完整性边界继续以[清理恢复记录](cleanup-20261016.md)为准。

以后新建工作树只能使用D盘根目录；当前C盘旧归档仅为恢复记录，不再作为工作环境。
