# TimeRCD源码与获取证据（2026-10-14）

论文正式来源：[ICML 2026/PMLR](https://proceedings.mlr.press/v306/lan26b.html)。
官方代码：https://github.com/thu-sail-lab/Time-RCD 。
代码提交`372bb980426b2f67007311c6f3165ab789c79bef`。

- `upstream/`：未经修改的作者源码、README、Tutorial、许可证；13文件哈希见`source-manifest.json`。
- `acquisition.json`：Hugging Face官方模型元数据请求及锁定版本。
- `official-weight-metadata.json`：官方LFS文件大小与SHA-256。权重须完整核验后才能加载。
- `environment.txt`：已安装的独立Python 3.12 CPU环境包版本，不更改主实验Python环境。
- `demo/`及`demo-source-manifest.json`：实际核查的官方在线演示源码与版本，使用旧命名权重，不等同新API。
- `demo-config.json`、`demo-source-info.json`：官方演示的接口/源版本信息；未上传实验站点数据、未调用其推理。
- `setup-status.json`：下载中断、保留残片、标准客户端下载未完成的真实记录；**TimeRCD推理未执行**。
- `attempts/fetch-ranges-v1.py`：失败的资料获取脚本，仅作获取过程记录，不是方法代码或成功执行证明。

算法/评价协议见`docs/experiment-exp05-timercd.md`，实施脚本`scripts/run_exp05_timercd.py`。
该脚本保留原API缺测输入检查、明确插值适配、连续AP与额外argmax判决，后者不是作者默认判决。
只做通用检测，不填四类分类指标。不以随机权重、未完整权重或旧在线模型替代。

后续取得与官方LFS摘要一致的完整文件后，先6验证站，再24验证，最后同48测试站：

```powershell
vendor/upstream/time-rcd/.venv/Scripts/python.exe scripts/run_exp05_timercd.py --datasets data/exp05-development-small --split validation --check-raw --checkpoint <完整官方multi权重路径> --out runs/exp05-timercd-pilot-v1
```

本轮的EXP05视觉成绩不依赖TimeRCD；无法据这些尚未执行的对照声称超过TimeRCD。
