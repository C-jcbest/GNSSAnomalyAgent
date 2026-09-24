# P1 正常序列模型

> 版本：2026-09-24 · 下列正常背景是 `event-v2` 的固定组成部分。365 日、参数和噪声类别全部固定；网页仅选择案例类型、seed 与案例数。

## 固定坐标与时间网格

令 $c\in\{N,E,U\}$，$t=0,1,\ldots,364$。第 0 日是 2025-01-01，第 364 日是 2025-12-31，每日每轴恰有一个值。参考坐标始终为

$$
P_0=(N_0,E_0,U_0)=(0,0,0)\ \mathrm{mm}.
$$

一年有 365 个观测日期，而周期分母固定为 $365.25$ 日；后者是生成器采用的近似回归周期，并不改变日期网格。P1 没有可配置的 `days`、起始日期或参考坐标。

## Annual 与 semiannual 正常背景

两种周期信号按轴相加：

$$
B_{t,c}=A_{1,c}\sin\!\left(\frac{2\pi t}{365.25}+\phi_{1,c}\right)
+A_{2,c}\sin\!\left(\frac{4\pi t}{365.25}+\phi_{2,c}\right).
$$

固定幅值为 $A_1=(1.5,1.5,2.0)$ mm、$A_2=(0.5,0.5,1.0)$ mm，顺序均为 N/E/U。$\phi_1$ 与 $\phi_2$ 各含三轴相位，分别由案例的 annual 与 semiannual seed 独立抽样，均匀分布在 $[0,2\pi)$。相位不开放给用户填写，实际抽样值保存在 `truth.json`。因此 $B_{0,c}$ 通常不为零；固定参考坐标不是拿首日观测值重新估计的。

Annual/semiannual 周期项结构可参见 [Khazraei & Amiri-Simkooei (2020)](https://academic.oup.com/gji/article/224/1/257/5911580)。当前幅值与噪声数值由本实验作为改善异常可辨识性的受控 benchmark 设定，**不是**该论文参数表的完整复用，也不代表现场精度。本文不采用线性速度和 flicker noise。

## 独立白噪声与无 secular deformation

每个日期和分量的测量噪声独立抽取：

$$
\epsilon_{t,c}\sim\mathcal N(0,\sigma_c^2),\qquad
(\sigma_N,\sigma_E,\sigma_U)=(0.75,0.75,1.5)\ \mathrm{mm}.
$$

P1 不含 AR(1)、flicker noise 或其他相关噪声，也没有线性速度。用 $D^{\mathrm{def}}_{t,c}$ 表示注入形变、$A^{\mathrm{art}}_{t,c}$ 表示观测伪差，则 P1 中两者恒为零；P2 才定义具体事件及贡献。

## 从成分到观测

统一的生成关系是

$$
O_{t,c}=P_{0,c}+B_{t,c}+D^{\mathrm{def}}_{t,c}
+\epsilon_{t,c}+A^{\mathrm{art}}_{t,c},
\qquad D^{\mathrm{def}}_{t,c}=A^{\mathrm{art}}_{t,c}=0\quad(\mathrm{P1}).
$$

所以 P1 简化为 $\boxed{O_{t,c}=P_{0,c}+B_{t,c}+\epsilon_{t,c}}$。生成器直接保存 `observed_coordinate_mm`、`normal_background_mm` 与 `measurement_noise_mm`；无需另设含义不清的 `true_coordinate` 或 `ideal_coordinate`。Normal truth 的事件列表为空，不提前引入运动状态字段。

保留三轴带符号位移，同时计算水平与三维偏移模长：

$$
\Delta_{t,c}=O_{t,c}-P_{0,c},\qquad
H_t=\sqrt{\Delta_{t,N}^2+\Delta_{t,E}^2},\qquad
R_{3D,t}=\sqrt{\Delta_{t,N}^2+\Delta_{t,E}^2+\Delta_{t,U}^2}.
$$

这里用 $R_{3D}$ 表示三维偏移，避免与未来的注入形变 $D^{\mathrm{def}}$ 混用。H 与 R3D 都是“该日离固定参考位置多远”，不是逐日累计路径；它们没有正负方向。

## 固定参数与图表阅读

| 项 | N | E | U |
| --- | ---: | ---: | ---: |
| Annual 幅值 | 1.5 mm | 1.5 mm | 2.0 mm |
| Semiannual 幅值 | 0.5 mm | 0.5 mm | 1.0 mm |
| 白噪声标准差 | 0.75 mm | 0.75 mm | 1.5 mm |

网页“观测位移”展示 $\Delta_N,\Delta_E,\Delta_U$，可叠加 H 或显示正常背景；“生成成分”按轴展示 $B$ 与 $\epsilon$。背景、相位和 seed 视图只用于理解模拟器，不能馈入未来的检测算法。缩放、日期定位和悬停读数不修改数据。
