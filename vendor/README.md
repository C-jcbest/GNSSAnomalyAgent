# 生成器上游来源

`tods/univariate_generator.py` 是固定 commit 的原始文件，逐字节保留；
`univariate_generator_position.py` 仅为趋势方法增加可选显式中心位置。
差异完整保存在 `explicit-position.patch`，不改斜率、残余或标签语句。
修改日期：2026-09-29，GNSS simulation project。

TODS：Apache-2.0，许可证随源码保留。GutenTAG：MIT，完整包在独立环境中安装，
本目录保留许可证。commit、URL、全部运行源码哈希及依赖版本见
`configs/generator-sources.json`。参数和单位映射属于项目适配，
不称完整 benchmark 复现。不得为消除 lint 提示格式化第三方文件。
