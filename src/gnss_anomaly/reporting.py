"""Read-only history access. Evaluation labels never flow back into detector prompts."""

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .contracts import Prediction, Window
from .evaluation import METRIC_VERSION
from .experiments import summarize_rows
from .storage import child, digest, file_hash, read_json


class History:
    def __init__(self, workspace: Path):
        self.root = workspace.resolve()
        self.cache = {}

    def paths(self, folder: str, filename: str):
        base = child(self.root, folder)
        if not base.exists():
            return []
        return sorted(p for p in base.rglob(filename) if p.resolve().is_relative_to(base))

    def token(self, path: Path):
        return digest(path.resolve().relative_to(self.root).as_posix())[:20]

    def run_path(self, key: str):
        for path in self.paths("runs", "run.json"):
            if self.token(path) == key:
                return path
        raise ValueError("未找到运行记录，请刷新列表")

    def load_run(self, path: Path):
        files = [path, *sorted((path.parent / "results").glob("*.json"))]
        stamp = tuple((p.name, p.stat().st_mtime_ns, p.stat().st_size) for p in files)
        cached = self.cache.get(str(path))
        if cached and cached[0] == stamp:
            return cached[1]
        meta = read_json(path)
        rows = []
        for p in files[1:]:
            child(self.root, str(p.resolve()))
            row = read_json(p)
            row["result_key"] = self.token(p)
            rows.append(row)
        summary = summarize_rows(meta, rows)
        result = (meta, rows, summary)
        self.cache[str(path)] = (stamp, result)
        return result

    def index(self):
        runs, errors = [], []
        for path in self.paths("runs", "run.json"):
            try:
                meta, rows, summary = self.load_run(path)
                completed = sum(r["metrics"]["completed"] for r in rows)
                runs.append(
                    {
                        "key": self.token(path),
                        "name": path.parent.relative_to(self.root / "runs").as_posix(),
                        "created_at": meta.get("created_at")
                        or datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(),
                        "date_source": "recorded" if meta.get("created_at") else "file_mtime",
                        "split": meta["split"],
                        "kind": meta["dataset_kind"],
                        "sampling_hours": meta.get("sampling_hours", 1),
                        "methods": meta["methods"],
                        "dataset_hash": meta["dataset_sha256"],
                        "run_id": meta["run_id"],
                        "present": summary["records_present"],
                        "expected": summary["records_expected"],
                        "complete": summary["all_records_present"],
                        "failures": len(rows) - completed,
                        "limited": meta.get("limit") is not None,
                    }
                )
            except (ValueError, KeyError, OSError, TypeError):
                errors.append(
                    {"name": path.parent.name, "message": "记录损坏或格式不支持，未纳入汇总"}
                )
        return {"runs": sorted(runs, key=lambda r: r["created_at"], reverse=True), "errors": errors}

    def detail(self, key: str):
        path = self.run_path(key)
        meta, rows, summary = self.load_run(path)
        fields = (
            "case_id",
            "method",
            "repeat",
            "kind",
            "state",
            "semantic",
            "group",
            "metrics",
            "seconds",
            "model_requests",
            "cost",
            "result_key",
        )
        cases = [
            {
                **{f: r[f] for f in fields},
                "source_group_id": r.get("source_group_id", r["group"]),
                "status": r["prediction"]["status"],
            }
            for r in rows
        ]
        safe_meta = {
            k: meta.get(k)
            for k in (
                "run_id",
                "dataset_sha256",
                "dataset_kind",
                "split",
                "model",
                "source",
                "limit",
                "prompt_version",
                "plot_version",
                "sampling_hours",
                "created_at",
            )
        }
        safe_meta["detectors"] = meta["config"]["detectors"]
        safe_meta["model_config"] = meta["config"]["model"]
        safe_meta["visual_protocol"] = meta["config"].get("visual_protocol")
        safe_meta["repeats"] = meta["config"]["repeats"]
        safe_meta["aggregation_version"] = METRIC_VERSION
        notices = []
        if meta["dataset_kind"] == "synthetic_demo_only":
            notices.append("纯合成工程示例，不作为论文方法优劣证据。")
        if meta["dataset_kind"] == "platform_pilot_provisional":
            notices.append(
                "开发预实验：真值只标注额外受控注入，F1是与注入掩码的一致度；原生变化未标注，未注入版本的报警率不是现场误报率；背景仅经智能体目视筛查，未确认为现场正常。不用于正式排名或生成能力卡。"
            )
        if meta.get("limit") is not None:
            notices.append("限量冒烟运行，不代表完整划分。")
        if not summary["all_records_present"]:
            notices.append("运行记录尚不完整；当前指标是已落盘部分的描述，不能用于最终排名。")
        if any(not r["metrics"]["completed"] for r in rows):
            notices.append("包含技术失败或数据不足；异常拒判计漏检，正常拒判另列失败数。")
        return {
            "name": path.parent.name,
            "meta": safe_meta,
            "summary": summary,
            "cases": cases,
            "notices": notices,
        }

    def dataset(self, meta: dict):
        candidates = self.paths("data/processed", "manifest.json")
        if meta.get("dataset_path"):
            try:
                explicit = child(self.root, meta["dataset_path"]) / "manifest.json"
                if explicit.is_file():
                    candidates.insert(0, explicit)
            except ValueError:
                pass
        for path in candidates:
            if file_hash(path) == meta["dataset_sha256"]:
                return path.parent, read_json(path)
        raise ValueError("未找到哈希匹配的本地数据集；可查看指标，但无法还原曲线")

    def case(self, key: str, result_key: str):
        meta, rows, _ = self.load_run(self.run_path(key))
        row = next((r for r in rows if r["result_key"] == result_key), None)
        if row is None:
            raise ValueError("未找到案例记录")
        root, manifest = self.dataset(meta)
        source = next((r for r in manifest["records"] if r["case_id"] == row["case_id"]), None)
        if source is None:
            raise ValueError("运行案例不属于该数据集")
        for name in (source["file"], "labels.json"):
            if file_hash(child(root, name)) != manifest["files"][name]:
                raise ValueError("数据或标签哈希不匹配，停止展示叠图")
        window = Window.model_validate(read_json(child(root, source["file"])))
        truth = Prediction(events=read_json(root / "labels.json")[row["case_id"]]["events"])
        prediction = Prediction.model_validate(row["prediction"])
        truth.mask(len(window.values))
        prediction.mask(len(window.values))
        return window, truth, prediction, row

    def data_index(self, view="prepared"):
        snapshots, errors = [], []
        preparations = self.paths("data/processed", "preparation-report.json")
        preparation = None
        paths = self.paths("data/snapshots", "manifest.json")
        if preparations:
            latest = max(preparations, key=lambda p: p.stat().st_mtime_ns)
            preparation = read_json(latest)
            if view != "raw":
                paths = sorted((latest.parent / "snapshots").glob("*/manifest.json"))
        reference_path = self.root / "data/labels/real-cases.json"
        references = read_json(reference_path).get("cases", []) if reference_path.exists() else []
        for path in paths:
            try:
                m = read_json(path)
                if m.get("kind") not in (
                    "platform_snapshot",
                    "trimmed_platform_snapshot",
                    "daily15_platform_snapshot",
                ):
                    continue
                snapshots.append(
                    {
                        "key": self.token(path),
                        "alias": m["station_alias"],
                        "start": m["start"],
                        "end": m["end_exclusive"],
                        "units_verified": m.get("units_verified", False),
                        "created_at": m["created_at"],
                        "audit": m["audit"],
                        "sampling_hours": m.get("sampling_hours", 1),
                        "group": m.get("site_alias"),
                        "split": (preparation or {})
                        .get("group_splits", {})
                        .get(m.get("site_alias")),
                        "trimmed": m.get("kind") == "trimmed_platform_snapshot",
                        "removed_hours": m.get("provenance", {}).get("removed_hours"),
                        "reference": next(
                            (r for r in references if r["station_alias"] == m["station_alias"]),
                            None,
                        ),
                    }
                )
            except (ValueError, KeyError, OSError):
                errors.append(path.parent.name)
        inventories = self.paths("data/raw", "inventory.json")
        inventory = None
        if inventories:
            m = read_json(max(inventories, key=lambda p: p.stat().st_mtime))
            inventory = {
                "created_at": m["created_at"],
                "station_count": m["station_count"],
                "probed": len(m["items"]),
                "with_records": sum(bool(r.get("raw_records")) for r in m["items"]),
            }
        collections = self.paths("data/snapshots", "collection-report.json")
        collection = None
        if collections:
            path = max(collections, key=lambda p: p.stat().st_mtime)
            report = read_json(path)
            collection = {
                "expected": report["expected_stations"],
                "finished": len(report["results"]),
                "nonempty": sum(
                    r.get("audit", {}).get("observed_timestamps", 0) > 0 for r in report["results"]
                ),
                "observations": sum(
                    r.get("audit", {}).get("observed_timestamps", 0) for r in report["results"]
                ),
                "all_successful": report.get("all_successful", False),
            }
        return {
            "snapshots": snapshots,
            "inventory": inventory,
            "collection": collection,
            "errors": errors,
            "preparation": preparation,
        }

    def snapshot(self, key: str):
        path = next(
            (
                p
                for p in [
                    *self.paths("data/snapshots", "manifest.json"),
                    *self.paths("data/processed", "manifest.json"),
                ]
                if self.token(p) == key
            ),
            None,
        )
        if path is None:
            raise ValueError("未找到快照")
        meta = read_json(path)
        csv = child(self.root, str(path.parent / "observations.csv"))
        if file_hash(csv) != meta["files"]["observations.csv"]:
            raise ValueError("快照哈希不匹配")
        frame = pd.read_csv(csv)
        if len(frame) > 24 * 366 * 2:
            raise ValueError("单次预览最多两年，请先提取窗口")
        values = frame[["n_mm", "e_mm", "u_mm"]].to_numpy(float)
        window = Window(
            case_id=meta["station_alias"],
            sampling_hours=meta.get("sampling_hours", 1),
            timestamps=list(pd.to_datetime(frame.timestamp)),
            values=[[float(v) if np.isfinite(v) else None for v in row] for row in values],
        )
        return window, meta
