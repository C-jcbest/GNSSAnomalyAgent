import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from gnss_sim import p8a_full
from gnss_sim.input_only import sha, write_json


def test_extension_refuses_changed_cached_responses_and_input_count(tmp_path, monkeypatch):
    root = tmp_path / "root"
    run = tmp_path / "run"
    write_json(root / "configs/p8a-range-context.json", {})
    write_json(root / "configs/p8a-frozen.json", {})
    write_json(run / "inputs/manifest.json", {})
    cached = run / "experiment/requests/V0/case_0001/completed.json"
    write_json(cached, {"prediction": {"status": "failed"}})
    ids = [f"case_{i:04d}" for i in range(1, 301)]
    registered = {"inference_source_sha256": {},
        "extension_source_sha256": sha(Path(p8a_full.__file__)),
        "config_sha256": sha(root / "configs/p8a-range-context.json"),
        "parent_frozen_sha256": sha(root / "configs/p8a-frozen.json"),
        "input_manifest_sha256": sha(run / "inputs/manifest.json"),
        "cache_sha256": {"requests/V0/case_0001/completed.json": sha(cached)},
        "cached_cases": ids[:60]}
    write_json(run / "registered.json", registered)
    monkeypatch.setattr(p8a_full, "ROOT", root)
    monkeypatch.setattr(p8a_full.p8a, "source_hashes", lambda: {})
    monkeypatch.setattr(p8a_full, "read_inputs", lambda p: [({}, SimpleNamespace(case_id=c)) for c in ids])
    assert len(p8a_full.validate(run)[1]) == 300
    old = cached.read_bytes()
    cached.write_text(json.dumps({"prediction": {"status": "success"}}))
    with pytest.raises(ValueError, match="cached"):
        p8a_full.validate(run)
    cached.write_bytes(old)
    ids.pop()
    with pytest.raises(ValueError, match="case set"):
        p8a_full.validate(run)
