"""Regression checks for upstream portability fixes, not mocked accuracy claims."""
import ast
import importlib.util
import runpy
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

pytest.importorskip("sklearn", reason="Run upstream checks in the dedicated reproduction environment")

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "artifacts/reproduction-2026-10-07"


@pytest.fixture(scope="module")
def tama():
    path = WORK / "runs/tama-pilot-v2/adapted/main_cli.py"
    if not path.exists():
        pytest.skip("Prepare pinned TAMA pilot with scripts/reproduce_tama.py first")
    sys.path.insert(0, str(path.parent))
    namespace = runpy.run_path(str(path))
    yield namespace
    sys.path.remove(str(path.parent))


def test_training_reference_required_even_if_test_has_normal_labels(tama, tmp_path):
    (tmp_path / "structure.yaml").write_text(yaml.safe_dump({"a": {"normal": ["a-0-0"]}}))
    helper = tama["NormalReferenceHelper"](str(tmp_path), str(tmp_path / "structure.yaml"))
    with pytest.raises(RuntimeError, match="Training references required"):
        helper.find_normal_reference("a", 0)


def test_reference_choice_does_not_depend_on_test_labels(tama, tmp_path):
    images = tmp_path / "a/train/image"
    images.mkdir(parents=True)
    for index in range(3):
        (images / f"{index}.png").write_bytes(b"fixture-path-only")
    structure = tmp_path / "structure.yaml"
    structure.write_text(yaml.safe_dump({"a": {"normal": ["a-0-0"]}}))
    first = tama["NormalReferenceHelper"](str(tmp_path), str(structure))
    structure.write_text(yaml.safe_dump({"a": {"normal": ["a-99-99"]}}))
    second = tama["NormalReferenceHelper"](str(tmp_path), str(structure))
    assert first.find_normal_reference("a", 0, fixed=True) == second.find_normal_reference("a", 0, fixed=True)


def test_internal_missing_days_do_not_shift_local_plot_coordinates(tama, tmp_path):
    folder = tmp_path / "a/test"
    folder.mkdir(parents=True)
    values = np.arange(15, dtype=float).reshape(1, 15, 1)
    values[:, 4:7, :] = np.nan
    np.save(folder / "data.npy", values)
    np.save(folder / "labels.npy", np.full_like(values, -1))
    (tmp_path / "background.txt").write_text("")
    dataset = tama["ProcessedDataset"](str(tmp_path), mode="test")
    loaded = dataset.get_data("a", 0, 0)
    assert len(loaded) == 15
    assert np.isnan(loaded[4:7]).all()
    assert loaded[7] == 7


def test_detection_and_review_prompts_have_correct_axes(tama):
    for key in ("normal_reference_prompt", "anormaly_detection_prompt", "double_check_prompt"):
        assert "The horizontal axis represents the time series index." in tama[key]
        assert "The vertical axis represents the value of the time series." in tama[key]


def test_transport_budget_blocks_before_network(monkeypatch, tmp_path, tama):
    monkeypatch.setenv("TAMA_REQUEST_DIR", str(tmp_path))
    monkeypatch.setenv("TAMA_MAX_REQUESTS", "1")
    monkeypatch.setenv("QWEN_MODEL", "fixture-model")
    (tmp_path / "001.started.json").write_text("{}")
    backend = tama["LocalBackend"]()
    # No API key or URL is supplied: the persisted budget must stop before transport access.
    with pytest.raises(RuntimeError, match="budget exhausted"):
        backend.chat([])


def test_source_patcher_retains_detection_and_review_logic():
    original = WORK / "upstream/tama/main_cli.py"
    if not original.exists():
        pytest.skip("Pinned upstream source is not materialized")
    spec = importlib.util.spec_from_file_location("prepare_tama", ROOT / "scripts/reproduce_tama.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    before = ast.parse(original.read_text(encoding="utf-8"))
    after = ast.parse(module.patch_upstream(original.read_text(encoding="utf-8")))
    for name in ("make_anormaly_detection_response_text", "make_double_check_prompt"):
        original_function = next(node for node in before.body if isinstance(node, ast.FunctionDef) and node.name == name)
        adapted_function = next(node for node in after.body if isinstance(node, ast.FunctionDef) and node.name == name)
        assert ast.dump(original_function) == ast.dump(adapted_function)


def test_interval_contract_rejects_out_of_bounds_and_wrong_container():
    spec = importlib.util.spec_from_file_location("contracts", ROOT / "scripts/reproduction_contracts.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with pytest.raises(ValueError, match="outside"):
        module.parse_tama_intervals("[(230, 1100)/4/trend]", 1095)
    with pytest.raises(ValueError, match="string"):
        module.parse_tama_intervals([], 1095)
    with pytest.raises(ValueError, match="Malformed"):
        module.parse_tama_intervals("[(2, 3)/4/trend, extra]", 1095)
    assert module.parse_tama_intervals("[]", 1095) == []
    assert module.parse_tama_intervals("[(0, 1094)/4/trend]", 1095)[0]["end"] == 1094
