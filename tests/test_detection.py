import json
from pathlib import Path

import httpx
import pytest


def test_registered_workflow_isolated_inputs_failures_and_no_overwrite(tmp_path, monkeypatch):
    from gnss_sim import detection, report
    from gnss_sim.schemas import GenerationRequest
    from gnss_sim.storage import DatasetStore

    store = DatasetStore(tmp_path / "data")
    try:
        normal = store.generate_sync(GenerationRequest(seed=701, count=2, case_type="normal"))
        dataset = store.generate_sync(GenerationRequest(seed=702, count=4, case_type="all"))
    finally:
        store.executor.shutdown(wait=True)
    assert normal.status == dataset.status == "complete"
    normal_dir, dataset_dir = store.root / normal.dataset_id, store.root / dataset.dataset_id
    parameters, out = tmp_path / "parameters.json", tmp_path / "run"
    read_bytes = Path.read_bytes

    with monkeypatch.context() as guard:
        def no_truth(path):
            assert path.name != "truth.json"
            return read_bytes(path)

        guard.setattr(Path, "read_bytes", no_truth)
        assert detection.calibrate(normal_dir, parameters)["calibration_cases"] == 2
    with pytest.raises(ValueError, match="different background seeds"):
        detection.prepare(normal_dir, parameters, out)
    assert not out.exists()
    assert detection.prepare(dataset_dir, parameters, out)["maximum_requests"] == 16
    with pytest.raises(FileExistsError):
        detection.prepare(dataset_dir, parameters, out)

    sent = []

    def respond(request):
        sent.append(json.loads(request.content))
        return httpx.Response(200, json={"model": "qwen3.8-flash", "id": "fixture",
            "usage": {"total_tokens": 5}, "choices": [{"finish_reason": "stop",
            "message": {"content": '{"N":[],"E":[],"U":[]}'}}]})

    real_client = httpx.Client
    predict = detection.numerical.predict
    with monkeypatch.context() as guard:
        def inference_inputs_only(path):
            assert path.name not in ("truth.json", "evaluation-registration.json")
            assert not path.is_relative_to(store.root)
            return read_bytes(path)

        def controlled_failure(case, *args):
            if case.case_id == "case_0001":
                raise ValueError("engineering failure")
            return predict(case, *args)

        guard.setattr(Path, "read_bytes", inference_inputs_only)
        guard.setattr(detection.numerical, "predict", controlled_failure)
        guard.setattr(detection, "credentials", lambda _: ("https://example.invalid", "fixture-key"))
        guard.setattr(detection.httpx, "Client", lambda **kwargs: real_client(
            transport=httpx.MockTransport(respond), **kwargs))
        assert detection.run_numerical(out)["failures"] == 1
        assert detection.run_visual(out, None)["calls"] == 8
        with pytest.raises(FileExistsError):
            detection.run_visual(out, None)
        with pytest.raises(FileExistsError):
            detection.run_numerical(out)
    assert len(sent) == 8
    assert not any(b"fixture-key" in p.read_bytes() for p in out.rglob("*.json"))
    summary = report.run(out)
    assert summary["numerical"]["point"]["normal"]["failed_cases"] == 1
    assert summary["visual"]["range"]["daily"]["fn"] == 104
    assert (out / "comparison.html").is_file()
    saved = json.loads((out / "evaluation.json").read_bytes())
    for group in [saved["summary"], *saved["by_type"].values()]:
        for method in group.values():
            for task in ("point", "range"):
                for metrics in ("daily", "affiliation"):
                    assert {"precision", "recall", "f1"} <= method[task][metrics].keys()
    page = (out / "comparison.html").read_text(encoding="utf-8")
    for column in ("Affiliation Precision", "Affiliation Recall", "Affiliation F1"):
        assert page.count(f"<th>{column}</th>") == 2  # Overall and per-type tables.
    with pytest.raises(FileExistsError):
        report.run(out)
    (out / "parameters.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="Registered input/config changed"):
        detection.validate(out)


def test_morphology_separates_isolated_jump_and_return_with_signed_trends():
    import numpy as np

    from gnss_sim import numerical as m

    for sign in (-1, 1):
        values = np.zeros(365)
        values[50] = 8 * sign
        values[120:144] = 5 * sign
        values[240:] = 6 * sign
        isolated, cleaned = m.without_spikes(values, 2.)
        changes = m.jumps(cleaned, 2.)
        point, temporary = m.pair_returns(changes)
        assert isolated == [50] and point == [240] and temporary == [(120, 143)]
        assert np.allclose(m.detrended_jumps(cleaned, changes), 0.)
        trend = sign * np.clip((np.arange(365) - 100) / 80, 0, 1) * 8
        ranges = m.trend_ranges(m.trend_scores(trend), .04, .02)
        assert any(s <= 150 <= e for s, e in ranges)
        assert all(e < 250 for _, e in ranges)
    values = np.random.default_rng(11).normal(size=365)
    assert m.trend_scores(values)[15] == pytest.approx(np.median([(values[j] - values[i]) / (j-i) for i in range(31) for j in range(i+1, 31)]))


def test_point_request_is_bounded_and_never_retries_transport(tmp_path):
    from gnss_sim.detection import make_config
    from gnss_sim.detection import request_case as request_upstream

    config = make_config()
    sent = []

    def respond(request):
        payload = json.loads(request.content)
        sent.append(payload)
        assert payload['model'] == 'qwen3.8-flash'
        assert payload['enable_thinking'] is False and payload['max_tokens'] == 8192
        content = '[]' if len(sent) == 1 else '{"N":[100],"E":[],"U":[]}'
        return httpx.Response(200, json={'model': 'qwen3.8-flash', 'id': 'mock',
            'choices': [{'finish_reason': 'stop', 'message': {'content': content}}],
            'usage': {'total_tokens': 10}})

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        result, records = request_upstream(client, 'https://example.invalid', 'test-key',
                                          'case_0001', 'point', b'image', config, tmp_path / 'ok')
    assert result.status == 'success' and result.predictions.N == [100]
    assert len(records) == 2 and records[0]['error'] == 'structure'
    assert not any('test-key' in p.read_text() for p in (tmp_path / 'ok').glob('*.json'))
    with httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(503))) as client:
        result, records = request_upstream(client, 'https://example.invalid', 'test-key',
                                          'case_0002', 'range', b'image', config, tmp_path / 'bad')
    assert result.status == 'failed' and len(records) == 1
    with pytest.raises(FileExistsError):
        request_upstream(None, '', '', 'case_0002', 'range', b'image', config, tmp_path / 'bad')
