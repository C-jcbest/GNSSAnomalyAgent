"""Create a new manuscript revision without altering frozen earlier drafts."""

from __future__ import annotations

import argparse
from pathlib import Path

from run_landslide_window_optimization import read_json

from gnss_sim.artifacts import sha, write_json


def write(root, out):
    previous = root / "docs/gnss-deformation-manuscript-draft-2026-10-12.md"
    base = root / "artifacts/gnss-deformation-diversity-2026-10-12"
    text = previous.read_text(encoding="utf-8")
    text = text.replace("deformation-manuscript-20261012-v1", "deformation-manuscript-20261012-v2")
    text = text.replace(
        "旧v54稿保持原样。",
        "旧v54及本日v1稿保持原样；本版补充共同缺测判定与中心—上下文视觉读取小试。",
    )
    text = text.replace(
        "本文公开完整负结果",
        "后续八类16训练／8验证的中心—上下文视觉差分小试，日F1为0.9093，低于同维度重复全局0.9262与中心无差分0.9331；缺测持续性诊断能改善强数值事件分数，但不是视觉专属贡献。本文公开完整负结果",
    )
    text = text.replace(
        "The findings bound the usefulness of pairing",
        "An additional stratified center/context readout pilot (16 training and 8 exposed validation records) obtains daily F1 of 0.9093 versus 0.9262 for a dimension-matched global control and 0.9331 without context subtraction. A common missing-aware duration correction improves numerical event scores without establishing a visual-specific contribution. The findings bound the usefulness of pairing",
    )
    method = r"""### 3.5 共同缺测持续性与中心—上下文读取小试

原活动判定要求14连续支撑日，但原事件评价允许连接≤7未知日。为避免短原始缺测将既有阳性活动削成不足持续期的两段，本轮建立共同判定：只跨≤7原始缺测连接两侧已有支撑阳性，不跨观测阴性或有观测但无特征支撑日；候选连通段仍至少含14个阳性观测，缺测不计持续日数且输出未知。该规则用于所有新视觉、纯数值、XGBoost、规则及PELT校准，不属于主方法专属创新。快速短事件仍可能受到14观测的限制，未删除这些参考来提高分数。

候选CC-VN（Center-Context Visual-Numeric Readout，中心—上下文视觉数值读取）只改变同一冻结MobileCLIP2-S0的空间读取。令特征图$F\in\mathbb R^{1024\times8\times8}$，沿用原预训练投影$W,b$，中心$C$取行0:7、列3:5，两侧$L\cup R$取相同行、列1,2,5,6：

$$
g=\operatorname{norm}(W\operatorname{avg}(F)+b),\quad
c=\operatorname{norm}(W\operatorname{avg}_{C}(F)+b),\quad
f=\operatorname{norm}(W\operatorname{avg}_{L\cup R}(F)+b).
$$

$$
d=\frac{c-f}{2},\qquad p_t=\sigma\!\left(h_\theta([g;d;n_t])\right).
$$

不再归一化差分，以保留中心／上下文差异幅度。主、重复全局$[g;g;n_t]$、中心无差分$[g;c;n_t]$均1060维及相同头大小；原全局548、纯视觉1024、纯数值36维作为其他模态对照。Attention已混入全局信息，32像素stride也不支持精确逐日定位；它只是粗区域读出。patch读取和视觉数值融合已有VLM4TS/VETime近邻[1,3]，本研究不将这两个操作单独主张为新颖性。

"""
    text = text.replace("## 4 数据与实验协议", method + "## 4 数据与实验协议")
    result = """### 5.5 新缺测诊断与视觉读取小试

在全部7DQ＋9PP保存概率上，仅改变共同缺测持续性判定、不改模型／阈值，raw84日F1从0.776750升至0.783572，事件F1从0.789474升至0.888889，新增TP26/FP14；同增广raw84日F1从0.791519升至0.798098，事件从0.731707升至0.820513。主PP-VN日F1从0.773036升至0.777306，事件从0.666667升至0.684211，TP/FP均新增31。该诊断是暴露验证数据的事后分析，不能撤销既有停止决定，也不能只给视觉方法使用。新共同实现对原16臂×24记录的384组活动／阶段逐值重放一致。

新的空间读取机制先做八类各2训练／1验证，共16／8记录，按原ID顺序抽取而非依据模型效果选择；这八条均已暴露，不是新的保留评价。六神经臂、三强树各38机会，规则及PELT各9机会，共360个配置，所有新学习／规则方法均重新使用共同持续性判定校准。三数值树在同16训练记录使用全部已知支撑日，神经臂同3382个stride3训练中心；数值缩放及阶段头仅使用这16训练记录。主要结果如下，不能与48／24实验分数直接相减解释成推广改进：

| 方法 | Precision | Recall | 日F1 | 事件F1(.3) | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 原全局视觉数值 | 0.975771 | 0.902240 | 0.937566 | 1.000000 | 11 | 48 |
| 同维度重复全局 | 0.931959 | 0.920570 | 0.926230 | 1.000000 | 33 | 39 |
| 中心无差分 | 0.943750 | 0.922607 | 0.933059 | 1.000000 | 27 | 38 |
| **CC-VN中心上下文差分** | **0.981132** | **0.847251** | **0.909290** | **1.000000** | **8** | **75** |
| 去数值差分视觉 | 0.696613 | 0.963340 | 0.808547 | 1.000000 | 206 | 18 |
| 同架构纯数值 | 0.884381 | 0.887984 | 0.886179 | 0.909091 | 57 | 55 |
| raw12 XGBoost | 0.986207 | 0.873727 | 0.926566 | 1.000000 | 6 | 62 |
| raw84 XGBoost | 0.973214 | 0.887984 | 0.928647 | 1.000000 | 12 | 55 |
| 图表信息XGBoost | 0.894309 | 0.896130 | 0.895219 | 0.909091 | 52 | 51 |
| 稳健规则 | 1.000000 | 0.706721 | 0.828162 | 0.888889 | 0 | 144 |
| PELT | 0.984127 | 0.505092 | 0.667564 | 0.888889 | 4 | 243 |

主相对重复全局减少25FP、增加36FN，净−11；对中心无差分净−18，没有一条记录相对重复全局净受益。纯干扰三记录各视觉数值臂均0FP，主没有附加纠错；混合形变／干扰召回0.837104低于重复全局0.923077。预设7条件仅3项通过，STOP_SPATIAL_READOUT_PILOT，不扩大48／24、不改区域或seed救分，不把表现较好的普通全局对照提升为新贡献。

主起止MAE6.2／10.4日，重复全局7.4／7.0日；本组多个方法事件F1均1，仍不等于边界准确或普遍能力。整记录2000次bootstrap有1988次定义良好；主对重复全局日F1差描述区间[−0.03734,−0.00935]，对raw84为[−0.07674,0.01216]，均不是选择校正或独立显著性证据。空间图8526中心编码CPU421.533秒，全局特征与原缓存完全一致；9选中模型、11方法／88记录预测、64240分数及全部活动／阶段评分重放通过。没有新增API或保留评价访问。

评分后实看case_0390和0385原始三轴并辅助计算原位移块：0390两次持续下降被主差分缩短起尾，净−4日；0385是两次转换后的平台，各视觉数值对照和raw84均无形变误报，净0。这些开发者观察只用于原因分析，不修改过程参考，不称独立盲标。完整结果、PNG／PDF与复现入口见[空间小试报告](../artifacts/gnss-deformation-diversity-2026-10-12/spatial-pilot-report-v1/report.md)。

"""
    text = text.replace("## 6 讨论", result + "## 6 讨论")
    text = text.replace(
        "辅助质量监督和逐轴证据也没有解决这一问题。",
        "辅助质量监督、逐轴证据和新的中心—上下文空间差分也没有解决这一问题。共同缺测持续性规则能改善强数值事件识别，提示先保证事件判定公平，再讨论视觉专属增益。",
    )
    text = text.replace(
        "及对应verify/report脚本。",
        "及对应verify/report脚本；新增空间小试由encode_landslide_spatial_readout.py、train_landslide_spatial_readout.py、verify_landslide_spatial_readout.py和report_landslide_spatial_readout.py复现，共同缺测判定位于landslide_observed_duration.py。",
    )
    with out.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(text)
    snapshot = base / "manuscript-v2"
    snapshot.mkdir(exist_ok=False)
    (snapshot / "manuscript.md").write_bytes(out.read_bytes())
    (snapshot / "source.py").write_bytes(Path(__file__).read_bytes())
    write_json(
        snapshot / "evidence.json",
        {
            "manuscript_sha256": sha(out),
            "previous_v1_sha256": sha(previous),
            "spatial_training_manifest_sha256": sha(
                base / "spatial-pilot-calibration-v1/calibration-freeze.json"
            ),
            "replay_sha256": sha(base / "spatial-pilot-replay-v1/verification.json"),
            "report_manifest_sha256": sha(base / "spatial-pilot-report-v1/freeze.json"),
            "diagnostic_manifest_sha256": sha(base / "missing-duration-probe-v1/freeze.json"),
            "decision": read_json(base / "spatial-pilot-calibration-v1/decision.json")["decision"],
            "test_truth_opened": False,
            "historical_drafts_modified": False,
        },
    )
    print(out, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    write(Path(__file__).resolve().parents[1], args.out.resolve())
