"""Add first-observation controls and raw error figures to the frozen development."""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from analyze_landslide_dual_reference_confirmation import score_target
from analyze_landslide_review_utility_v2 import verify_receipts
from matplotlib.colors import ListedColormap
from prepare_landslide_ai_reference_holdout import verify_manifest, write_csv
from probe_landslide_revision_scope import scoped_activity
from run_landslide_censored_followup import BASE, BATCHES, reference_for_batch, verify
from run_landslide_window_optimization import load_case_features, read_json
from write_landslide_dual_reference_report import full_table
from xgboost import XGBClassifier

from gnss_sim.artifacts import sha, write_json
from gnss_sim.landslide_adaptive_review import combine_reviews
from gnss_sim.landslide_local_experiment import learned_stages
from gnss_sim.landslide_review_utility import correction_utility

RUN = BASE / "censored-followup-development-v1"
ANALYSIS = BASE / "censored-followup-development-analysis-v2"
OUT = BASE / "censored-followup-development-report-v1"


def first_only_predictions(batch, process=False):
    comparison = BATCHES[batch]
    original = read_json(ANALYSIS / batch / "predictions.json")
    schedules = read_json(RUN / "schedules.json")[batch]
    head = XGBClassifier()
    head.load_model(comparison / "run/stage-xgboost.json")
    process_codes = read_json(BASE / "dual-reference-process-predictions-v1/stage-codes.json") if process else None
    predictions = {f"first_only_r{rep}": {} for rep in (0, 1)}
    used = {name: {} for name in predictions}
    for cid in read_json(comparison / "run/protocol.json")["case_ids"]:
        feature = load_case_features(comparison / "run", cid)[1]
        codes = np.array(process_codes[cid]) if process else head.predict(feature.matrix).astype(int) + 1
        for rep in (0, 1):
            row = schedules[f"{cid}-r{rep}"]
            initial = np.array(original[f"direct_visual_r{rep}"][cid]["activity"])
            activity = initial.copy()
            keys = [f"{cid}-direct-r{rep}"]
            if row["budget"]:
                first = {**row["first"], **row["first_result"]}
                reviewed = combine_reviews(initial, [first], feature.observed).activity
                scope = np.zeros(len(initial), bool)
                scope[first["start"]:first["stop"]] = True
                scope &= np.array(row["authority"]) & feature.observed
                activity = scoped_activity(initial, reviewed, scope, feature.observed)
                keys.append(row["first_key"])
            method = f"first_only_r{rep}"
            predictions[method][cid] = learned_stages(codes, activity, feature.supported).to_dict()
            used[method][cid] = keys
    return predictions, used


def draw_case(batch, cid, labels, predictions):
    data = Path(read_json(BATCHES[batch] / "source-bindings.json")["data"])
    case = read_json(data / "inputs" / cid / "input.json")
    values = np.array([[np.nan] * 3 if row is None else row for row in case["displacement_mm"]])
    observed = np.isfinite(values).all(axis=1)
    names = ["AI reference"]
    states = [[row["activity_label"] for row in labels]]
    for rep in (0, 1):
        for label, stem in (("Original", "adaptive_no_feedback"), ("First only", "first_only"),
                            ("Censored follow-up", "followup_censored"), ("Wide follow-up", "followup_wide")):
            method = f"{stem}_r{rep}" + ("__conflict_scope" if stem == "adaptive_no_feedback" else "")
            names.append(f"{label} r{rep}")
            states.append(predictions[method][cid]["activity"])
    figure, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True,
        gridspec_kw={"height_ratios": [2, 2, 2, 4]})
    for component, name in enumerate(("N", "E", "U")):
        axes[component].plot(values[:, component], linewidth=.7)
        axes[component].set_ylabel(name + " mm")
        axes[component].grid(alpha=.2)
    codes = np.array(states) + 1
    codes[:, ~observed] = 3
    axes[3].imshow(codes, aspect="auto", interpolation="nearest", vmin=0, vmax=3,
        cmap=ListedColormap(["#edc46a", "#e0e7e0", "#407ea0", "#777777"]),
        extent=(-.5, len(values) - .5, len(names) - .5, -.5))
    axes[3].set_yticks(range(len(names)), names, fontsize=8)
    axes[3].set_xlabel("Original day; blue=activity,pale=stationary,amber=unknown,gray=missing")
    axes[3].set_xlim(0, len(values) - 1)
    figure.suptitle(cid + " | exposed development; post-score diagnosis; frozen reference unchanged")
    figure.tight_layout()
    figure.savefig(OUT / (cid + ".png"), dpi=140)
    plt.close(figure)


def report():
    verify(RUN)
    verify_manifest(ANALYSIS)
    OUT.mkdir(exist_ok=False)
    (OUT / "source.py").write_bytes(Path(__file__).read_bytes())
    receipts = {}
    for comparison in BATCHES.values():
        receipts.update(verify_receipts(comparison / "run"))
    receipts.update(verify_receipts(RUN))
    tables, per_records, supplements, bootstrap = {}, {}, [], []
    for target in ("ai_0133_0144", "ai_0145_0156", "process_0145_0156"):
        batch = "ai_0145_0156" if target.startswith("process") else target
        labels, observed, weak = reference_for_batch(batch)
        if target.startswith("process"):
            labels = read_json(BASE / "dual-reference-confirmation-analysis-v1/secondary_process/reference-labels.json")
            weak = {cid: np.array([row["weak_signal"] == 1 and row["activity_label"] == 1 for row in rows])
                for cid, rows in labels.items()}
        predictions = read_json(ANALYSIS / target / "predictions.json")
        first, used = first_only_predictions(batch, target.startswith("process"))
        results, new_rows, new_records = score_target(labels, observed, weak, first, used, receipts)
        predictions.update(first)
        with (ANALYSIS / target / "summary.csv").open(encoding="utf-8-sig") as stream:
            tables[target] = list(csv.DictReader(stream)) + new_rows
        with (ANALYSIS / target / "per-record.csv").open(encoding="utf-8-sig") as stream:
            per_records[target] = list(csv.DictReader(stream)) + new_records
        folder = OUT / target
        folder.mkdir()
        write_json(folder / "predictions.json", predictions)
        write_json(folder / "first-only-results.json", results)
        write_csv(folder / "summary.csv", tables[target])
        write_csv(folder / "per-record.csv", per_records[target])
        for rep in (0, 1):
            for cid in labels:
                actual = np.array([row["activity_label"] for row in labels[cid]])
                value = correction_utility(actual, np.array(first[f"first_only_r{rep}"][cid]["activity"]),
                    np.array(predictions[f"followup_censored_r{rep}"][cid]["activity"]), observed[cid])
                supplements.append({"target": target, "case_id": cid, "repeat": rep, **value})
        if not target.startswith("process"):
            for cid in (("case_0139", "case_0143") if batch == "ai_0133_0144"
                        else ("case_0148", "case_0154")):
                draw_case(batch, cid, labels[cid], predictions)
    # Resample whole records while keeping the two repeats as separate outcomes.
    combined = per_records["ai_0133_0144"] + per_records["ai_0145_0156"]
    cases = sorted({row["case_id"] for row in combined})
    indices = np.random.default_rng(20261011).integers(0, len(cases), (2000, len(cases)))
    for rep in (0, 1):
        main = f"followup_censored_r{rep}"
        for control in (f"adaptive_no_feedback_r{rep}__conflict_scope", f"followup_fixed_short_r{rep}"):
            values = {}
            for method in (main, control):
                mapping = {row["case_id"]: row for row in combined if row["method"] == method}
                counts = np.array([[int(mapping[cid][key]) for key in ("tp", "fp", "fn")] for cid in cases])
                total = counts[indices].sum(axis=1)
                values[method] = 2 * total[:, 0] / np.maximum(1, 2 * total[:, 0] + total[:, 1] + total[:, 2])
            bootstrap.append({"control": control, "treatment": main,
                "exploratory_95_percentile_interval": np.quantile(values[main] - values[control], [.025, .975]).tolist(),
                "scope": "24 whole-record descriptive bootstrap; repeats not pooled as independent records"})
    write_json(OUT / "bootstrap.json", bootstrap)
    write_csv(OUT / "second-observation-increment.csv", supplements)
    duplicate = "case_0139-window-06-image-r0.response.json"
    write_json(OUT / "avoidable-duplicate.json", {
        "previous": str(BASE / "repair-role-development-v1/requests" / duplicate),
        "current": str(RUN / "requests" / duplicate),
        "same_request_sha256": read_json(BASE / "repair-role-development-v1/requests" / duplicate)["request_sha256"]
            == read_json(RUN / "requests" / duplicate)["request_sha256"],
        "explanation": "Parent-batch cache search did not include previous role probe. One repeat was avoidable; all20 new launches remain in cost. Not a retry or model-selection replicate."})
    text = """# 截断端点驱动的第二次原始观察：有待新冻结确认的开发收益

2026-10-11。预登记条件通过，状态PROMISING_DEVELOPMENT；不改变默认，不称未见验证。收益只在两个案例/三个记录-重复组合，原作者AI开发标签保持。

## 识别机制

第一次观察原方法已选的最大冲突180日目标。局部活动到达该图右边时，这个端点仅说明区间被观察范围截断，不代表真实终止。在之后49日有至少14个原始冲突日时，第二次观察以该边为中心的98日原始图；98来自原±49日证据范围。其余使用原另一目标，原0/1/2调用预算不变。数值只决定取证，不赋活动。采用原严格多数合并重叠回答，冲突未知；缺测/失败不回填。原49日权限与实际目标并集相交，不改变门槛、提示、原始渲染或共同阶段头。

主触发0139第一遍、0148两遍。0143文字端点到达359但该日缺测，未触发确定继续活动；不以结构筛查后的例子降低门槛或扩大边缘容忍。左边界和未触边的漏检不在本机制范围。

## 对比与消融

两完整已暴露AI批次、后批过程目标；原34方法与5方案各2遍共44方法。固定短窗共享触发和98日宽度但以原另一目标中心观察；180日边缘图检验尺度；去数值门槛检验盲目追边；最新回答覆写检验严格多数。评分后追加仅第一次观察作为成本更低的诊断控制，未更改原登记决策，完整报告46方法×3目标。

继续条件推理前登记：24记录合并日计数，两遍F1差平均相对原方法和固定短窗均>0，弱召回平均不低于原方法，对两者净确定纠错总和均>0。这个条件允许其他指标取舍，不要求所有单批次/重复获胜；不得改写上一轮失败条件。这里只准许新记录确认，不允许挑最好重复或消融升级默认。

## 实际结果

旧AI原F1 .8964/.9029，主.9227/.9029；第一遍增40TP、撤43FP，净确定纠错+83，第二遍无作用。后AI原.8536/.8363，主.8692/.8520，两遍弱R均保持.6230/.5234，FP207/187降158/138，两遍净+49。合并两遍主F1平均增.01444，对固定短窗增.01265，弱R平均增.02107，净纠错分别+181/+157。描述bootstrap下限为0，不能称小样本显著性或稳定普遍收益。

第二次观察本身相对第一窗控制：0139第一遍新增45TP且无确定FP增加；0148每遍撤44平台FP且无AI确定TP损失，共净+133。其余相对原方法的改善部分来自不再使用原另一窗的错误改写，不能把全部+181归因于第二张图。

过程目标主F1 .8413/.8204比原.8280/.8073高，但每遍少19个活动TP，弱R .5368/.4732降.5131/.4494，净+31/+31。部分过程活动在AI未知区，因此AI“无TP损失”不等于真实过程无损。严格多数与最新覆写的AI确定F1相同，但主在后AI未知区多弃判49日、过程覆盖减少49日；不把弃判自动算作正确。阶段分数仍来自数值头门控。

较宽180日边缘图后AI F1 .8547/.8368、弱R .5566/.4570，低于98日主；同一区域宽图仍把弱尾和平台混合撤回。去数值门槛旧AI .9146/.9278、后AI .8586/.8659，第二遍F1更高、FP更少，但后批第一遍丢49TP、弱R .5273低于主.6230。主门槛不是各指标最优，不能用此消融挑换主方案。

## 成本与复现边界

20新增启动/20回执，81325报告token、51.109秒，无失败/重试；23第二问题复用，首次回答来自旧父批次。存在一条与上一轮role probe字节相同的可避免重复请求，已保留并计入20成本。缓存日期与新回答不同，不称同时期fresh对照或排除模型漂移。各方法部署调用/token见全表，第一窗控制更便宜。无新记录、训练或标签。

首次评分错误地调用只允许不重叠目标的范围函数，在第一个结果前失败；v2仅按冻结协议计算目标并集，旧源码、空失败目录和全部推理保留。9业务测试/Ruff通过，原68历史分数重现。复现推理脚本run_landslide_censored_followup.py的prepare/run后，使用analyze_landslide_censored_followup_v2.py --run <冻结run> --out <新的空目录>评分；原analyze子命令保留为失败历史，不能作为有效评分入口。

TAMA作者commit db99a75a83d0ef05caf7ff0d199b75b594e98584的main_cli.py 544—582实读仍为固定前/后2/3观察；本候选是局部目标的截断语义决定下一目标，而不是固定多尺度。AnomaMind/TimeRLM等已有有限动态取证，不主张粗到细或有限循环首创。需要新冻结数据证明GNSS特定机制收益。

## 全部统一结果

"""
    for target, rows in tables.items():
        text += f"\n### {target}\n\n" + full_table(rows) + "\n"
    report_path = BASE.parents[1] / "docs/landslide-censored-followup-development-results.md"
    report_path.write_text(text, encoding="utf-8")
    write_json(OUT / "report-binding.json", {"path": str(report_path), "sha256": sha(report_path)})
    write_json(OUT / "manifest.json", {"files": {path.relative_to(OUT).as_posix(): sha(path)
        for path in OUT.rglob("*") if path.is_file()}})


if __name__ == "__main__":
    report()
