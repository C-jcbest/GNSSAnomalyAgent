#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate SFT detector as a pure chatbot (upstream fixed).

Example::

  python3 -m AnomaMind.infer.infer_detector_only \
      --baseline-res-dir /path/data/YAHOO/result/test/API \
      --output-dir /path/data/YAHOO/result/test_all/pure_rl_8B \
      --env-file /path/to/repo/env/your_detector.env \
      --workers 128 \
      --no-artifacts \
      --run-eval
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import time
import threading
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import replace as dataclass_replace
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

RL_SYSTEM_PROMPT = "You are a professional time series anomaly detection expert."


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _default_env_file() -> Path:
    return _repo_root() / "env" / "local.env"


def _load_env_file(path: Path, *, override: bool = True) -> None:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip()
        if not k:
            continue
        if (not override) and (os.getenv(k) is not None):
            continue
        os.environ[k] = v


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        try:
            return json.loads(text, strict=False)
        except Exception:
            return None
    except Exception:
        return None


def _should_skip_existing_output(path: Path) -> bool:
    data = _load_json(path)
    if not isinstance(data, dict):
        return False
    final_output = data.get("final_output") or {}
    has_error = bool(data.get("state_has_error"))
    final_error = bool(final_output.get("error")) if isinstance(final_output, dict) else False
    return (not has_error) and (not final_error)


def _is_retryable_error_message(message: str) -> bool:
    text = str(message or "").strip().lower()
    if not text:
        return False
    retryable_signals = (
        "connection error",
        "timeout",
        "timed out",
        "temporarily unavailable",
        "connection reset",
        "server disconnected",
    )
    return any(signal in text for signal in retryable_signals)


def _detect_with_retries(detector, state: Dict[str, Any], *, max_retries: int, retry_sleep_seconds: float):
    attempts = 0
    max_retries = max(0, int(max_retries))
    retry_sleep_seconds = max(0.0, float(retry_sleep_seconds))

    while True:
        try:
            out = detector.detect(state)
        except Exception as e:
            attempts += 1
            if attempts <= max_retries and _is_retryable_error_message(str(e)):
                if retry_sleep_seconds > 0:
                    time.sleep(retry_sleep_seconds)
                continue
            raise

        if isinstance(out, dict):
            has_error = bool(out.get("has_error"))
            error_message = str(out.get("error_message") or "")
            if has_error and attempts < max_retries and _is_retryable_error_message(error_message):
                attempts += 1
                if retry_sleep_seconds > 0:
                    time.sleep(retry_sleep_seconds)
                continue
        return out


def _coerce_ai_message(resp: Any) -> AIMessage:
    if isinstance(resp, AIMessage):
        return resp
    content = getattr(resp, "content", resp)
    if not isinstance(content, str):
        content = str(content or "")
    return AIMessage(content=content)


def _detector_prompt_text(detector, state: Dict[str, Any]) -> str:
    existing = list(state.get("detector_messages") or [])
    if len(existing) == 1:
        first_existing = existing[0]
        role = ""
        content = ""
        if isinstance(first_existing, dict):
            role = str(first_existing.get("role") or "")
            content = str(first_existing.get("content") or "")
        else:
            role = type(first_existing).__name__.replace("Message", "").lower()
            content = str(getattr(first_existing, "content", "") or "")
        if role == "user" and content.strip():
            return content

    prompt_messages = detector._build_messages(state)  # type: ignore[attr-defined]
    if not prompt_messages:
        return ""
    first = prompt_messages[0]
    content = getattr(first, "content", "") or ""
    return content if isinstance(content, str) else str(content)


def _detect_via_rl_chat(detector, state: Dict[str, Any]) -> Dict[str, Any]:
    if state.get("has_error", False):
        return {}
    llm = getattr(detector, "llm", None)
    if llm is None:
        return {
            "has_error": True,
            "error_message": "detector llm is not configured",
        }

    prompt_text = _detector_prompt_text(detector, state)
    if not prompt_text.strip():
        return {
            "has_error": True,
            "error_message": "detector prompt is empty",
        }
    detector_messages = [
        SystemMessage(content=RL_SYSTEM_PROMPT),
        HumanMessage(content=prompt_text),
    ]
    try:
        ai = _coerce_ai_message(llm.invoke(detector_messages))
    except Exception as exc:
        return {
            "detector_messages": detector_messages,
            "has_error": True,
            "error_message": str(exc),
        }

    detector_messages.append(ai)
    return {
        "detector_messages": detector_messages,
    }


class _DetectorRunner:
    def __init__(self, detector, *, prompt_mode: str):
        self.detector = detector
        self.prompt_mode = str(prompt_mode or "rl")

    def detect(self, state: Dict[str, Any]) -> Dict[str, Any]:
        if self.prompt_mode == "legacy":
            return self.detector.detect(state)
        return _detect_via_rl_chat(self.detector, state)


def _iter_result_jsons(res_dir: Path) -> List[Path]:
    out: List[Path] = []
    for p in sorted(res_dir.glob("*.json")):
        if p.name in {"_smoke_summary.json"}:
            continue
        out.append(p)
    return out


def _write_smoke_summary(output_dir: Path, *, rows: List[Dict[str, Any]], worst_exit_code: int, elapsed_s: float) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "_smoke_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "rows": rows,
                "worst_exit_code": worst_exit_code,
                "count": len(rows),
                "elapsed_s": round(float(elapsed_s), 3),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return summary_path


def _smoke_eval_from_intervals(gt_intervals: List[List[int]], pred_intervals: List[List[int]]) -> Dict[str, int]:
    from AnomaMind.core.reward import intervals_to_set

    gt_set = intervals_to_set(gt_intervals)
    pred_set = intervals_to_set(pred_intervals)
    return {
        "tp_points": len(gt_set & pred_set),
        "fp_points": len(pred_set - gt_set),
        "fn_points": len(gt_set - pred_set),
        "gt_point_count": len(gt_set),
        "pred_point_count": len(pred_set),
    }


def _run_one(
    *,
    baseline_json_path: Path,
    output_dir: Path,
    detector,
    prompt_mode: str,
    copy_artifacts: bool,
    skip_if_exists: bool,
    max_retries: int,
    retry_sleep_seconds: float,
    force_regenerate: bool = False,
    rollouts: int = 1,
    rollout_temp_step: float = 0.12,
    llm_cfg_template: Any = None,
) -> Tuple[int, Dict[str, Any]]:
    from AnomaMind.core.Detector import Detector
    from AnomaMind.core.finalize import finalize_detector_output
    from AnomaMind.core.knowledge_base import KnowledgeBase
    from AnomaMind.core.llm_config import LLMConfig, create_llm
    from AnomaMind.core.utils import load_segment_data
    from AnomaMind.infer.infer import _copy_segment_final_to_output, _write_smoke_gt_pred_png
    from AnomaMind.helpers.message_io import export_messages

    base = _load_json(baseline_json_path)
    if not base:
        return 2, {"segment": baseline_json_path.name, "exit_code": 2, "error": "bad_json"}

    stem = baseline_json_path.stem
    out_json = output_dir / f"{stem}.json"
    if skip_if_exists and out_json.exists():
        return 0, {"segment": stem, "exit_code": 0, "skipped": True, "skip_reason": "exists"}
    if (not force_regenerate) and out_json.exists() and _should_skip_existing_output(out_json):
        return 0, {"segment": stem, "exit_code": 0, "skipped": True, "skip_reason": "existing_success"}

    seg_dir_raw = str(base.get("segment_dir") or "").strip()
    if not seg_dir_raw:
        return 2, {"segment": stem, "exit_code": 2, "error": "missing_segment_dir"}
    seg_dir = Path(seg_dir_raw).resolve()
    if not seg_dir.exists():
        return 2, {"segment": stem, "segment_dir": str(seg_dir), "exit_code": 2, "error": "segment_dir_not_found"}

    # Load segment data + GT from processed directory (authoritative).
    try:
        pack = load_segment_data(seg_dir)
    except Exception as e:
        return 2, {"segment": stem, "segment_dir": str(seg_dir), "exit_code": 2, "error": f"load_segment_data_failed:{type(e).__name__}"}

    gt_intervals: List[List[int]] = list(pack.get("ground_truth") or [])

    state: Dict[str, Any] = {
        "data": pack["segment_data"],
        "type": pack.get("type") or base.get("type") or "TODS",
        "path": str(seg_dir),
        # Upstream fixed: use baseline visual hints and baseline tool transcript.
        "visual_description": base.get("visual_description", ""),
        "visual_localization": base.get("visual_localization", []),
        "visual_localization_items": base.get("visual_localization_items", []),
        "actor_messages": base.get("actor_messages") or base.get("messages") or [],
        "detector_messages": [],
        "has_error": False,
        "error_message": "",
        "final_output": {},
    }

    try:
        out = _detect_with_retries(
            _DetectorRunner(detector, prompt_mode=prompt_mode),
            state,
            max_retries=max_retries,
            retry_sleep_seconds=retry_sleep_seconds,
        )  # returns {"detector_messages": ...} or error
        if isinstance(out, dict):
            state.update(out)
        else:
            state["has_error"] = True
            state["error_message"] = f"detector_returned_non_dict:{type(out).__name__}"
    except Exception as e:
        state["has_error"] = True
        state["error_message"] = f"detector_exception:{type(e).__name__}:{e}"
    state = finalize_detector_output(state)

    fo = state.get("final_output") or {}
    pred_intervals = fo.get("anomaly_intervals")
    pred_intervals = pred_intervals if isinstance(pred_intervals, list) else []

    # Even on error, produce a consistent smoke eval (pred treated as empty).
    smoke = _smoke_eval_from_intervals(gt_intervals, pred_intervals) if not fo.get("error") else _smoke_eval_from_intervals(gt_intervals, [])

    # Keep baseline actor trace as-is (dict JSON); export_messages used to stringify dicts as "unknown".
    actor_export = list(base.get("actor_messages") or [])
    if not actor_export:
        actor_export = export_messages(list(state.get("actor_messages", []) or []))
    detector_export = export_messages(list(state.get("detector_messages", []) or []))
    payload: Dict[str, Any] = {
        "segment_dir": str(seg_dir),
        "state_has_error": bool(state.get("has_error")) or bool(fo.get("error")),
        "final_output": fo,
        "visual_description": state.get("visual_description", ""),
        "visual_localization": state.get("visual_localization", []),
        "visual_localization_items": state.get("visual_localization_items", []),
        "actor_messages": actor_export,
        "detector_messages": detector_export,
        "messages": actor_export + detector_export,
        "ground_truth_intervals": gt_intervals,
        "smoke_point_eval": smoke,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    if copy_artifacts:
        p_gt_pred = output_dir / f"{stem}_smoke_gt_pred.png"
        _write_smoke_gt_pred_png(p_gt_pred, stem, pack, gt_intervals, pred_intervals, smoke)
        _copy_segment_final_to_output(seg_dir, output_dir, stem)

    exit_code = 0 if (not payload["state_has_error"]) else 2
    return exit_code, {
        "segment": stem,
        "segment_dir": str(seg_dir),
        "exit_code": exit_code,
        **smoke,
    }


def _run_one_with_timeout(
    *,
    timeout_seconds: float,
    **kwargs,
) -> Tuple[int, Dict[str, Any]]:
    timeout_seconds = float(timeout_seconds or 0)
    if timeout_seconds <= 0:
        return _run_one(**kwargs)

    baseline_json_path = kwargs.get("baseline_json_path")
    stem = getattr(baseline_json_path, "stem", None) or getattr(baseline_json_path, "name", "unknown")

    ex = ThreadPoolExecutor(max_workers=1)
    fut = ex.submit(_run_one, **kwargs)
    try:
        return fut.result(timeout=timeout_seconds)
    except FutureTimeoutError:
        fut.cancel()
        return 2, {
            "segment": stem,
            "exit_code": 2,
            "error": f"timeout_after_{int(timeout_seconds)}s",
        }
    except Exception as exc:
        return 2, {
            "segment": stem,
            "exit_code": 2,
            "error": f"run_one_wrapper_exception:{type(exc).__name__}:{exc}",
        }
    finally:
        ex.shutdown(wait=False, cancel_futures=True)


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Evaluate SFT detector as chatbot; upstream fixed by baseline results.")
    p.add_argument(
        "--baseline-res-dir",
        "--input-dir",
        dest="baseline_res_dir",
        type=Path,
        required=True,
        help="Input baseline result dir (e.g. /path/data/result/TODS_new).",
    )
    p.add_argument("--output-dir", type=Path, required=True, help="Output result dir (TODS_new-like structure).")
    p.add_argument(
        "--env-file",
        type=Path,
        default=_default_env_file(),
        help="Env file for LLM (LLM_BASE_URL/KEY/AUX_LLM_MODEL).",
    )
    p.add_argument("--max-files", type=int, default=0, help="0=all; else only first N files.")
    p.add_argument("--workers", type=int, default=1, help="Parallel workers (threads).")
    p.add_argument(
        "--prompt-mode",
        choices=["rl", "legacy"],
        default="rl",
        help="Detector prompting mode. 'rl' matches RL/SFT chat packaging; 'legacy' uses Detector.detect(state).",
    )
    p.add_argument("--no-artifacts", action="store_true", help="Do not write png artifacts.")
    p.add_argument(
        "--skip-if-exists",
        action="store_true",
        help="Skip segments whose output json already exists, even if the existing result is an error.",
    )
    p.add_argument("--run-eval", action="store_true", help="Run eval_res_dir_f1.py after generation.")
    p.add_argument("--max-retries", type=int, default=3, help="Retry count for transient detector connection failures.")
    p.add_argument("--retry-sleep-seconds", type=float, default=1.0, help="Sleep between transient detector retries.")
    p.add_argument("--per-sample-timeout-seconds", type=float, default=0.0, help="Per-sample timeout; on timeout mark sample failed and continue.")
    args = p.parse_args(argv)

    _load_env_file(Path(args.env_file), override=True)

    from AnomaMind.core.knowledge_base import KnowledgeBase
    from AnomaMind.core.llm_config import LLMConfig, create_llm
    from AnomaMind.core.Detector import Detector

    # Thread-local detector to avoid sharing clients across threads.
    model = os.getenv("AUX_LLM_MODEL") or os.getenv("DETECTOR_MODEL") or "gpt-4o-mini"
    cfg = LLMConfig(
        model_name=str(model),
        api_key=os.getenv("LLM_API_KEY"),
        base_url=str(os.getenv("LLM_BASE_URL") or "http://localhost:8000/v1"),
        temperature=float(os.getenv("DETECTOR_TEMPERATURE") or os.getenv("LLM_TEMPERATURE") or "0.5"),
        max_tokens=None,
    )
    _tls = threading.local()

    def _get_detector() -> Detector:
        det = getattr(_tls, "detector", None)
        if det is not None:
            return det
        llm = create_llm(cfg)
        kb = KnowledgeBase()
        det = Detector(knowledge_base=kb, llm=llm)
        _tls.detector = det
        return det

    files = _iter_result_jsons(Path(args.baseline_res_dir))
    if args.max_files and args.max_files > 0:
        files = files[: int(args.max_files)]

    rows: List[Dict[str, Any]] = []
    worst = 0
    t0 = time.time()
    workers = max(1, int(args.workers or 1))
    out_dir = Path(args.output_dir)
    copy_artifacts = not bool(args.no_artifacts)
    skip_if_exists = bool(args.skip_if_exists)

    def _task(path: Path) -> Tuple[int, Dict[str, Any]]:
        return _run_one_with_timeout(
            baseline_json_path=path,
            output_dir=out_dir,
            detector=_get_detector(),
            prompt_mode=str(args.prompt_mode),
            copy_artifacts=copy_artifacts,
            skip_if_exists=skip_if_exists,
            max_retries=int(args.max_retries),
            retry_sleep_seconds=float(args.retry_sleep_seconds),
            timeout_seconds=float(args.per_sample_timeout_seconds),
        )

    if workers == 1:
        for f in files:
            rc, row = _task(f)
            rows.append(row)
            worst = max(worst, rc)
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(_task, f) for f in files]
            for fut in as_completed(futs):
                rc, row = fut.result()
                rows.append(row)
                worst = max(worst, rc)

    summary_path = _write_smoke_summary(
        Path(args.output_dir),
        rows=rows,
        worst_exit_code=worst,
        elapsed_s=time.time() - t0,
    )
    print(f"wrote: {summary_path}")

    if args.run_eval:
        cmd = [sys.executable, "-m", "AnomaMind.eval.evaluate_results", str(Path(args.output_dir))]
        subprocess.run(cmd, check=False)

    return 0 if worst == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
