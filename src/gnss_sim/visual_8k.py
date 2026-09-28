"""Future input-only visual request adapter; preserve frozen P6/P7 inference sources."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from gnss_sim.visual_runner import _request

ROOT = Path(__file__).resolve().parents[2]
METHOD = "visual-semantics-v2-8k"
DEFAULT_CONFIG = ROOT / "configs/p8-visual-8k.json"


def validate_config(config: dict) -> dict:
    parent_path = ROOT / "configs/p7a-visual-semantics.json"
    frozen = json.loads((ROOT / "configs/p7a-frozen.json").read_bytes())
    if hashlib.sha256(parent_path.read_bytes()).hexdigest() != frozen["config_sha256"]:
        raise ValueError("frozen parent config changed")
    for name, digest in frozen["source_sha256"].items():
        text = Path(__file__).with_name(name).read_text(encoding="utf-8")
        if hashlib.sha256(text.encode()).hexdigest() != digest:
            raise ValueError("frozen parent inference source changed")
    expected = json.loads(parent_path.read_bytes())
    for key in ("pilot_id", "pilot_manifest_sha256", "pilot_summary_sha256"):
        expected.pop(key)
    expected.update(protocol=METHOD, method=METHOD)
    expected["model"]["max_tokens"] = 8192
    if config != expected:
        raise ValueError("8k config must preserve v2 prompts and settings with max_tokens=8192")
    return config


def load_config(path: Path = DEFAULT_CONFIG) -> dict:
    return validate_config(json.loads(path.read_bytes()))


def request_once(client, base, key, image, task, case_id, config):
    validate_config(config)
    if task not in ("point", "range"):
        raise ValueError("unknown visual task")
    prediction, record = _request(client, base, key, image, task, case_id, config)
    prediction = prediction.model_copy(update={"method": METHOD})
    record.update(method=METHOD, max_tokens=config["model"]["max_tokens"],
                  response_format={"type": "json_object"},
                  completed_at=datetime.now(timezone.utc).isoformat())
    return prediction, record
