import pytest

from gnss_research.data import simulate
from gnss_research.files import read_json
from gnss_research.upstream import ROOT, prepare_visualtime, setup_sources


def test_original_request_defaults_and_station_adapter(tmp_path):
    if not (ROOT / "vendor/upstream/visualtime/.git").exists():
        pytest.skip("run gnss-research setup-sources for this source integration check")
    observation, _ = simulate(9, "s0000", "normal")
    before = setup_sources()
    prepare_visualtime(observation, tmp_path / "source-run", "test-model")
    after = setup_sources()
    assert before == after
    for axis in "NEU":
        for task in ("point", "range"):
            payload = read_json(tmp_path / "source-run" / f"{axis}-{task}" / "request.json")
            assert payload["temperature"] == 0.4
            assert len(payload["stop"]) == 4
            assert payload["model"] == "test-model"
            assert payload["messages"][0]["content"][1]["image_url"]["url"].startswith(
                "data:image/png;base64,")
