"""A saved Windows source path must not skip the executing-source check."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_landslide_comparison import sha  # noqa: E402
from verify_landslide_evidence_display_confirmation import verify_live_sources  # noqa: E402


@pytest.mark.parametrize("saved_path", ["source/scripts/runner.py", "source\\scripts\\runner.py"])
def test_source_guard_detects_mutation_with_both_path_separators(tmp_path, saved_path):
    root, run = tmp_path / "repo", tmp_path / "run"
    live, snapshot = root / "scripts/runner.py", run / "source/scripts/runner.py"
    live.parent.mkdir(parents=True)
    snapshot.parent.mkdir(parents=True)
    live.write_text("frozen source", encoding="utf-8")
    snapshot.write_bytes(live.read_bytes())
    (run / "pre-generation-freeze.json").write_text(json.dumps({saved_path: sha(live)}), encoding="utf-8")
    assert len(verify_live_sources(run, root)) == 1
    live.write_text("changed source", encoding="utf-8")
    with pytest.raises(ValueError, match="differs from pre-generation"):
        verify_live_sources(run, root)
