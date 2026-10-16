"""Test local observed-change readability before proposing another visual revision."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from encode_landslide_spatial_readout import verify_manifest
from probe_landslide_nuisance_evidence import write_csv
from run_landslide_window_optimization import load_case_features, read_json
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from gnss_sim.artifacts import sha, write_json
from gnss_sim.landslide_feature_discovery import discovery_block_matrix


def prepare_rows(data, encoded, ids, partition):
    features, targets, locations = [], [], []
    for cid in ids:
        case, evidence = load_case_features(data, cid)
        raw = np.asarray([row if row is not None else [np.nan] * 3 for row in case.displacement_mm])
        blocks, available, _, _ = discovery_block_matrix(raw, evidence.noise)
        with np.load(encoded / partition / (cid + ".npz")) as bank:
            selected = bank["selected"] & available[:, 5:8].all(axis=1)
            indices = np.flatnonzero(selected)
            # Both sides refer to the same central raw block. No labels, latent
            # displacement or generator parameters enter these descriptive targets.
            target = np.concatenate((blocks[:, 6] - blocks[:, 5], blocks[:, 7] - blocks[:, 6]), axis=1)
            visual = bank["embeddings"][indices].astype(np.float64)
        features.append(visual)
        targets.append(target[indices])
        locations.extend(dict(case_id=cid, day_index=int(day)) for day in indices)
    return np.concatenate(features), np.concatenate(targets), locations


def score(target, prediction):
    return dict(
        mae_noise_units=float(mean_absolute_error(target, prediction)),
        rmse_noise_units=float(np.sqrt(mean_squared_error(target, prediction))),
        r2_uniform=float(r2_score(target, prediction)),
        r2_per_axis_edges=r2_score(target, prediction, multioutput="raw_values").tolist(),
    )


def probe(data, encoded, out):
    verify_manifest(data, "data-freeze.json")
    verify_manifest(encoded, "encoding-freeze.json")
    split = read_json(encoded / "protocol.json")["split"]
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "protocol.json", dict(
        name="Observed target fidelity;information probe,not an anomaly detector",
        scope="48TRAIN/24exposedVALIDATION;not independent,no reserved contents",
        target="Signed noise-normalized observed changes from block5 to6 and6 to7;13 original14day blocks;>=8complete observations each",
        inputs="Original cached512MobileCLIP2 embedding;no numerical or activity/stage/quality label inputs",
        fitting="TRAIN-only StandardScaler and six-output Ridge(alpha1,solverlsqr,tol1e-8);no search",
        controls="TRAIN mean;zero change;TRAIN embeddings shuffled across same training pool(seed20261013)",
        test="Same validation target and support for all arms;retain six signed outputs and per-record losses",
        interpretation="Fidelity is a necessary diagnostic,not anomaly improvement or causal evidence;low fidelity cannot rule out nonlinear readability",
        continuation="Both aggregate MAE<zero and mean;all six R2>0;real MAE<shuffled. Even pass requires a separately frozen anomaly mechanism",
        failure="Stop this linear-evidence approach if any condition fails;no alpha/seed/model rescue",
        input_bindings={str(data / "data-freeze.json"): sha(data / "data-freeze.json"),
                        str(encoded / "encoding-freeze.json"): sha(encoded / "encoding-freeze.json")},
        source_bindings={str(Path(__file__).resolve()): sha(Path(__file__)),
                         str(Path("src/gnss_sim/landslide_feature_discovery.py").resolve()):
                         sha(Path("src/gnss_sim/landslide_feature_discovery.py"))},
        activity_reference_read=False, reserved_content_loaded=False, new_api_calls=0,
    ))
    (out / "source.py").write_bytes(Path(__file__).read_bytes())
    write_json(out / "pre-probe-freeze.json", {path.name: sha(path) for path in out.iterdir()})
    started = time.perf_counter()
    train_x, train_y, train_locations = prepare_rows(data, encoded, split["train"], "train")
    valid_x, valid_y, valid_locations = prepare_rows(data, encoded, split["validation"], "validation")
    scaler = StandardScaler().fit(train_x)
    train_scaled, valid_scaled = scaler.transform(train_x), scaler.transform(valid_x)
    real = Ridge(alpha=1, solver="lsqr", tol=1e-8).fit(train_scaled, train_y)
    order = np.random.default_rng(20261013).permutation(len(train_x))
    shuffled = Ridge(alpha=1, solver="lsqr", tol=1e-8).fit(train_scaled[order], train_y)
    predictions = dict(
        visual_target=real.predict(valid_scaled), shuffled_target=shuffled.predict(valid_scaled),
        train_mean=np.broadcast_to(train_y.mean(axis=0), valid_y.shape).copy(),
        zero_change=np.zeros_like(valid_y),
    )
    np.savez_compressed(out / "fidelity.npz", targets=valid_y, **predictions,
                        scaler_mean=scaler.mean_, scaler_scale=scaler.scale_,
                        real_coef=real.coef_, real_intercept=real.intercept_,
                        shuffled_coef=shuffled.coef_, shuffled_intercept=shuffled.intercept_,
                        training_mean=train_y.mean(axis=0))
    write_json(out / "locations.json", dict(train=train_locations, validation=valid_locations))
    summary = {name: score(valid_y, prediction) for name, prediction in predictions.items()}
    per_case = []
    case_ids = np.array([row["case_id"] for row in valid_locations])
    for cid in split["validation"]:
        mask = case_ids == cid
        for name, prediction in predictions.items():
            per_case.append(dict(method=name, case_id=cid, observations=int(mask.sum()),
                                 **score(valid_y[mask], prediction[mask])))
    write_csv(out / "per-case.csv", per_case)
    passed = (
        summary["visual_target"]["mae_noise_units"] < summary["zero_change"]["mae_noise_units"]
        and summary["visual_target"]["mae_noise_units"] < summary["train_mean"]["mae_noise_units"]
        and summary["visual_target"]["mae_noise_units"] < summary["shuffled_target"]["mae_noise_units"]
        and min(summary["visual_target"]["r2_per_axis_edges"]) > 0
    )
    # Independently replay exported linear weights on reloaded stored parameters.
    with np.load(out / "fidelity.npz") as stored:
        normalized = (valid_x - stored["scaler_mean"]) / stored["scaler_scale"]
        for name, prefix in (("visual_target", "real"), ("shuffled_target", "shuffled")):
            reconstructed = normalized @ stored[prefix + "_coef"].T + stored[prefix + "_intercept"]
            np.testing.assert_allclose(reconstructed, stored[name], rtol=1e-10, atol=1e-10)
    decision = dict(
        status="LINEAR_TARGET_MECHANISM_PENDING" if passed else "STOP_LINEAR_TARGET_EVIDENCE",
        train_queries=len(train_x), validation_queries=len(valid_x), dimensions=6,
        seconds=time.perf_counter() - started, zero_api_calls=True,
        exported_weight_replay="passed1e-10", activity_reference_read=False,
        anomaly_gain_established=False, reserved_content_loaded=False,
    )
    write_json(out / "summary.json", summary)
    write_json(out / "decision.json", decision)
    text = (
        "# 视觉目标事实可读性：先检验证据，再设计修订\n\n## Material Passport\n\n"
        "- Origin Skill: academic-research-suite / experiment-agent。\n"
        "- 48训练／24已暴露验证；未加载活动参考，不是异常效果实验。\n"
        "- 固定线性探针，无调参／新API；导出权重重新计算通过1e-10容差。\n\n"
        "| 方法 | MAE（噪声单位） | RMSE | 六输出平均R² |\n| --- | ---: | ---: | ---: |\n"
    )
    for name, row in summary.items():
        text += f"| {name} | {row['mae_noise_units']:.6f} | {row['rmse_noise_units']:.6f} | {row['r2_uniform']:.6f} |\n"
    text += (
        "\n决定：" + decision["status"] + "。活动F1未计算，也未证明新的异常识别贡献。"
        "六维分别对应左／右相邻块N/E/U有符号变化。目标块至少八完整观测，缺测不插值。"
        "视觉含相同原始观测，重建是事实读取而非预测未来或独立验证。"
        "相邻日期相关，R²不是显著性；失败仅否定固定线性读出，不能否定全部视觉信息。\n"
    )
    (out / "report.md").write_text(text, encoding="utf-8")
    write_json(out / "freeze.json", {path.name: sha(path) for path in out.iterdir()})
    print(dict(decision=decision, summary=summary), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "encoded", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    arguments = parser.parse_args()
    probe(arguments.data.resolve(), arguments.encoded.resolve(), arguments.out.resolve())
