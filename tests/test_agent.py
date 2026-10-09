import json
from datetime import date, timedelta

import httpx
import numpy as np
import pytest

from gnss_sim import agent, numerical
from gnss_sim.schemas import CaseInput, PointResult, RangeResult


@pytest.fixture
def example():
    x = np.random.default_rng(303).normal(size=(365, 3))
    case = CaseInput(case_id="case_0001", dates=[date(2025, 1, 1)+timedelta(days=i) for i in range(365)],
        reference_coordinate_mm=(0, 0, 0), observed_coordinate_mm=x.tolist(),
        displacement_mm=x.tolist(), horizontal_offset_mm=np.linalg.norm(x[:, :2], axis=1).tolist(),
        spatial_offset_mm=np.linalg.norm(x, axis=1).tolist())
    parameters = {"config": numerical.CONFIG, "thresholds": {
        a: {"spike": 3., "jump": 3., "trend_high": .1, "trend_low": .03} for a in numerical.AXES}}
    items = [{"id": "c001", "axis": "N", "start": 50, "end": 50, "sources": ["numerical"]}]
    return case, parameters, items


def envelope(content='{"c001":true}', calls=None, finish="stop", **extra):
    return {"model": "qwen3.8-flash", "usage": {"total_tokens": 10},
            "choices": [{"finish_reason": finish, "message": {"role": "assistant",
                         "content": content, **({"tool_calls": calls} if calls else {}), **extra}}]}


def tool_call(name="local_samples", ids=None):
    return {"id": "call_1", "type": "function", "function": {"name": name,
        "arguments": json.dumps({"candidate_ids": ids or ["c001"]})}}


def test_real_create_agent_tool_loop_and_matched_controls(example, tmp_path, monkeypatch):
    from pathlib import Path

    case, parameters, items = example
    read_bytes = Path.read_bytes
    def no_truth(path):
        assert path.name not in ("truth.json", "evaluation-registration.json", "manifest.json")
        return read_bytes(path)
    monkeypatch.setattr(Path, "read_bytes", no_truth)
    for arm in agent.ARMS:
        sent = []
        def respond(request):
            payload = json.loads(request.content)
            sent.append(payload)
            if payload.get("tools"):
                return httpx.Response(200, json=envelope(None, [tool_call()], "tool_calls"))
            return httpx.Response(200, json=envelope())
        with httpx.Client(transport=httpx.MockTransport(respond)) as client:
            result, info = agent.review(case, b"mock image", items, parameters, "point", arm,
                                       client, "https://fixture.invalid", "secret-fixture", tmp_path / arm)
            assert result.predictions.N == [50] and result.status == "success"
            assert len(sent) == info["calls"] == 2
            assert all(p["model"] == "qwen3.8-flash" and p["enable_thinking"] is False
                       and p["max_tokens"] == 8192 for p in sent)
            if arm in ("langchain_agent", "boundary_agent"):
                assert info["tools"][0]["name"] == "local_samples"
                assert sent[1].get("tools") is None
                assert any(m["role"] == "tool" for m in sent[1]["messages"])
            elif arm == "candidate_visual":
                assert "Numerical evidence:" not in json.dumps(sent)
            else:
                assert "Numerical evidence:" in json.dumps(sent)
            with pytest.raises(FileExistsError):
                agent.review(case, b"", items, parameters, "point", arm, client,
                             "https://fixture.invalid", "secret-fixture", tmp_path / arm)
        assert not any(b"secret-fixture" in p.read_bytes() for p in (tmp_path / arm).glob("*.json"))


@pytest.mark.parametrize("mode,expected", [("structure", 3), ("transport", 1),
                                         ("thinking", 1), ("length", 1), ("invalid_twice", 2)])
def test_budgets_and_failure_contract(example, tmp_path, mode, expected):
    case, parameters, items = example
    sent = []
    def respond(request):
        sent.append(json.loads(request.content))
        if mode == "transport":
            raise httpx.ReadTimeout("fixture")
        if mode == "thinking":
            return httpx.Response(200, json=envelope(reasoning_content="not allowed"))
        if mode == "length":
            return httpx.Response(200, json=envelope(finish="length"))
        return httpx.Response(200, json=envelope("{}" if len(sent) == 1 or mode == "invalid_twice"
                                                else '{"c001":true}'))
    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        if mode == "structure":
            row, info = agent.review(case, b"image", items, parameters, "point", "fixed_fusion",
                                    client, "https://fixture.invalid", "key", tmp_path / "request")
            assert row.status == "success" and info["corrections"] == 1
        else:
            with pytest.raises(RuntimeError):
                agent.review(case, b"image", items, parameters, "point", "fixed_fusion",
                             client, "https://fixture.invalid", "key", tmp_path / "request")
    assert len(sent) == expected


def test_candidates_cannot_expand_dates_or_leak_metadata(example, tmp_path):
    case, parameters, items = example
    point = PointResult(case_id=case.case_id, method="n", status="success", predictions={"N": [50], "E": [], "U": []})
    visual = RangeResult(case_id=case.case_id, method="v", status="success", predictions={"N": [[40, 60]], "E": [], "U": []})
    temporary = visual.model_copy(update={"method": "n"})
    proposed = agent.candidates(point, visual, temporary, "range")
    assert len(proposed) == 1 and set(proposed[0]) == {"id", "axis", "start", "end", "sources"}
    assert agent.result_from_decisions(case.case_id, "range", "test", proposed,
                                       {"c001": True}).predictions.N == [(40, 60)]
    with pytest.raises(ValueError):
        agent.candidates(point, visual.model_copy(update={"status": "failed"}), temporary, "range")
    for content in ('{"c002":true}', '{"c001":1}', '{"c001":true,"c001":false}'):
        with pytest.raises(ValueError):
            agent.decisions(content, items)
    with pytest.raises(ValueError):
        agent.Evidence(case, items, parameters).local_samples(["truth"])
    def forbidden(_):
        raise AssertionError("empty candidate task must not send")
    with httpx.Client(transport=httpx.MockTransport(forbidden)) as client:
        row, info = agent.review(case, b"", [], parameters, "point", "langchain_agent", client,
                                "https://fixture.invalid", "key", tmp_path / "empty")
    assert row.status == "success" and info["calls"] == 0


def test_boundary_contract_and_overlap(example):
    case, _, _ = example
    items = [
        {"id": "c001", "axis": "N", "start": 40, "end": 60},
        {"id": "c002", "axis": "N", "start": 55, "end": 70},
        {"id": "c003", "axis": "E", "start": 0, "end": 10},
    ]
    selected = agent.decisions('{"c001":[50,60],"c002":[58,65],"c003":null}', items, True)
    row = agent.result_from_decisions(case.case_id, "range", "boundary_agent", items, selected)
    assert row.predictions.N == [(50, 65)] and row.predictions.E == []
    for content in (
        '{"c001":[39,60],"c002":null,"c003":null}',
        '{"c001":[40,61],"c002":null,"c003":null}',
        '{"c001":[60,40],"c002":null,"c003":null}',
        '{"c001":[40.0,60],"c002":null,"c003":null}',
        '{"c001":[true,60],"c002":null,"c003":null}',
        '{"c001":true,"c002":null,"c003":null}',
        '{"c001":[],"c002":null,"c003":null}',
        '{"c001":null,"c002":null}',
        '{"c001":null,"c001":null,"c002":null,"c003":null}',
    ):
        with pytest.raises(ValueError):
            agent.decisions(content, items, True)
    selected["c001"] = [39, 60]
    with pytest.raises(ValueError):
        agent.result_from_decisions(case.case_id, "range", "boundary_agent", items, selected)


def test_boundary_agent_tool_loop_and_single_correction(example, tmp_path):
    case, parameters, _ = example
    items = [{"id": "c001", "axis": "N", "start": 40, "end": 60}]
    sent = []
    def respond(request):
        payload = json.loads(request.content)
        sent.append(payload)
        if payload.get("tools"):
            return httpx.Response(200, json=envelope(None, [tool_call()], "tool_calls"))
        content = '{"c001":[39,60]}' if len(sent) == 2 else '{"c001":[45,55]}'
        return httpx.Response(200, json=envelope(content))
    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        row, info = agent.review(case, b"image", items, parameters, "range", "boundary_agent",
                                client, "https://fixture.invalid", "key", tmp_path / "request")
    assert row.predictions.N == [(45, 55)]
    assert len(sent) == 3 and info["corrections"] == 1
    assert any(m["role"] == "tool" for m in sent[1]["messages"])
    assert "inside that candidate" in sent[2]["messages"][-1]["content"]


@pytest.mark.parametrize("design", ["boundary", "localization", "window", "point_context", "point_hypothesis"])
def test_registered_experiment_end_to_end_mocked(tmp_path, monkeypatch, design):
    from gnss_sim import detection, experiment
    from gnss_sim.schemas import GenerationRequest
    from gnss_sim.storage import DatasetStore

    store = DatasetStore(tmp_path / "datasets")
    try:
        calibration = store.generate_sync(GenerationRequest(seed=17701, count=3, case_type="normal"))
    finally:
        store.executor.shutdown(wait=True)
    parameters = tmp_path / "parameters.json"
    detection.calibrate(store.root / calibration.dataset_id, parameters)
    out = tmp_path / "experiment"
    experiment.register(out, 17702, 24, parameters, design, store.root / calibration.dataset_id)
    with pytest.raises(FileExistsError):
        experiment.register(out, 17702, 24, parameters)
    registered = (out / "preregistered.json").read_bytes()
    changed = json.loads(registered)
    changed["seed"] += 1
    (out / "preregistered.json").write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="Preregistered settings changed"):
        experiment.validate(out)
    (out / "preregistered.json").write_bytes(registered)
    monkeypatch.setattr(experiment, "DatasetStore", lambda _: DatasetStore(tmp_path / "datasets"))
    real_client = httpx.Client
    sent = []
    def respond(request):
        payload = json.loads(request.content)
        sent.append(payload)
        first = next(m for m in payload["messages"] if m["role"] == "user")["content"][0]["text"]
        if "Candidates: " in first:
            items = json.loads(first.split("Candidates: ")[1].split("\n")[0])
            ids = [c["id"] for c in items]
            if payload.get("tools"):
                name = payload["tools"][0]["function"]["name"]
                body = envelope(None, [tool_call(name=name, ids=ids)], "tool_calls")
            else:
                shrink = "inclusive [start,end] integer pair" in payload["messages"][0]["content"]
                localize = "supplied boundary proposal" in payload["messages"][0]["content"]
                body = envelope(json.dumps({c["id"]: "original" if localize else
                                            [c["start"], c["end"]] if shrink else True
                                            for c in items}))
        else:
            body = envelope('{"N":[],"E":[],"U":[]}' if "isolated anomalous observations" in first
                            else '{"N":[[100,110]],"E":[],"U":[]}')
        return httpx.Response(200, json=body)
    monkeypatch.setattr(httpx, "Client", lambda **kw: real_client(transport=httpx.MockTransport(respond), **kw))
    for module in (detection, experiment):
        monkeypatch.setattr(module, "credentials", lambda _: ("https://fixture.invalid", "fixture-key"))
    result = experiment.execute(out, None)
    assert result["total_calls"] == len(sent) <= experiment.read(out / "preregistered.json")["maximum_requests"]
    saved = json.loads((out / "evaluation.json").read_bytes())
    assert saved["paired_analysis"]["groups"] == 6
    reg = experiment.read(out / "preregistered.json")
    assert set(saved["summary"]) == set(reg["methods"])
    for arm in reg["arms"]:
        for task in detection.TASKS:
            assert saved["summary"][arm][task]["cases"] == 24
            assert saved["summary"][arm][task]["failed_cases"] == 0
            assert saved["summary"][arm][task]["daily"] == saved["summary"]["support"][task]["daily"]
    assert (out / "comparison.html").exists()
    if design == "window":
        assert reg["maximum_requests"] == 19 * 24
        assert saved["summary"]["window_agent"]["point"] == saved["summary"]["localization_agent"]["point"]
        assert not list((out / "window_agent/requests").glob("*/point/*-started.json"))
        for info in (out / "window_agent/results").glob("*-point.json"):
            assert experiment.read(info)["info"]["shared_task_source"] == "localization_agent"
    if design in ("point_context", "point_hypothesis"):
        assert reg["maximum_requests"] == 22 * 24
        assert reg["primary_metric"] == "point daily F1"
        for arm, shared in reg["shared_task_sources"].items():
            source = shared["range"]
            assert saved["summary"][arm]["range"] == saved["summary"][source]["range"]
            assert not list((out / arm / "requests").glob("*/range/*-started.json"))
        images = experiment.point_context_images(out, reg)
        image = next(out / name for names in images.values() for name in names)
        image.write_bytes(b"tampered")
        with pytest.raises(ValueError, match="Point observation image changed"):
            experiment.point_context_images(out, reg)
    with pytest.raises(FileExistsError):
        experiment.execute(out, None)
    (out / "calibration.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="Calibration changed"):
        experiment.validate(out)


def test_localization_shapes_expansion_and_output_isolation(example, tmp_path, monkeypatch):
    from pathlib import Path

    from gnss_sim import localization

    case, parameters, _ = example
    read_bytes = Path.read_bytes
    def observation_only(path):
        assert path.name not in ("truth.json", "manifest.json", "evaluation-registration.json")
        return read_bytes(path)
    monkeypatch.setattr(Path, "read_bytes", observation_only)
    x = np.zeros((365, 3))
    x[150:168, 0] = 3
    x[:, 1] = 3*np.clip((np.arange(365)-80)/85, 0, 1)
    case = case.model_copy(update={"displacement_mm": x.tolist()})
    items = [{"id": "c001", "axis": "N", "start": 155, "end": 170},
             {"id": "c002", "axis": "E", "start": 90, "end": 160}]
    packages = {c["id"]: localization.propose(case, c, parameters, 1.) for c in items}
    assert packages["c001"]["rule_choice"] == "level"
    assert packages["c001"]["options"]["level"]["start"] == 150
    assert packages["c001"]["options"]["level"]["end"] == 167
    assert packages["c002"]["rule_choice"] == "trend"
    assert packages["c002"]["options"]["trend"]["start"] == 80
    assert packages["c002"]["options"]["trend"]["end"] == 165
    sent = []
    def respond(request):
        payload = json.loads(request.content)
        sent.append(payload)
        if payload.get("tools"):
            return httpx.Response(200, json=envelope(None,
                [tool_call("localize_boundaries", ["c001", "c002"])], "tool_calls"))
        answer = '{"c001":[150,167],"c002":"trend"}' if len(sent) == 2 else '{"c001":"level","c002":"trend"}'
        return httpx.Response(200, json=envelope(answer))
    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        row, info = agent.review(case, b"image", items, parameters, "range", "localization_agent",
            client, "https://fixture.invalid", "key", tmp_path / "request", packages)
    assert row.predictions.N == [(150, 167)] and row.predictions.E == [(80, 165)]
    assert info["calls"] == 3 and info["corrections"] == 1
    assert "rule_choice" not in json.dumps(sent)
    for invalid in ('{"c001":true,"c002":null}', '{"c001":"unknown","c002":null}',
                    '{"c001":null}', '{"c001":null,"c001":"level","c002":null}'):
        with pytest.raises(ValueError):
            localization.decisions(invalid, packages)


def test_context_search_reaches_interior_with_identical_observations(example):
    from gnss_sim import localization

    case, parameters, _ = example
    x = np.zeros((365, 3))
    x[:, 0] = 3*np.clip((np.arange(365)-75)/85, 0, 1)
    case = case.model_copy(update={"displacement_mm": x.tolist()})
    item = {"id": "c001", "axis": "N", "start": 0, "end": 120}
    old = localization.propose(case, item, parameters, 1.)
    new = localization.propose(case, item, parameters, 1., "context")
    assert old["context"] == new["context"] == [0, 182]
    assert old["sigma_mm"] == new["sigma_mm"]
    assert old["raw_samples"] == new["raw_samples"]
    assert old["non_target_score"] == new["non_target_score"]
    assert old["options"]["original"] == new["options"]["original"]
    assert old["search_start"] == [0, 31]
    assert new["search_start"] == new["search_end"] == [0, 182]
    assert new["rule_choice"] == "trend"
    assert new["options"]["trend"]["start"] == 75
    assert new["options"]["trend"]["end"] == 160
    assert new["options"]["trend"]["score"] <= old["options"]["trend"]["score"]
    # The full-context algorithm also supports year-edge candidates without external padding.
    edge = localization.propose(case, {**item, "start": 340, "end": 364}, parameters, 1., "context")
    for option in edge["options"].values():
        assert edge["context"][0] <= option["start"] <= option["end"] <= 364
    with pytest.raises(ValueError):
        localization.propose(case, item, parameters, 1., "truth")


def test_window_range_gate_failure_preserves_shared_point_success_and_failure(example, tmp_path, monkeypatch):
    from gnss_sim import detection, experiment, report
    from gnss_sim.artifacts import sha, write_json, write_rows

    case, parameters, items = example
    cases = [case, case.model_copy(update={"case_id": "case_0002"})]
    rows = [PointResult(case_id=c.case_id, method="localization_agent",
                       status="success" if i == 0 else "failed",
                       predictions={"N": [50] if i == 0 else [], "E": [], "U": []})
            for i, c in enumerate(cases)]
    write_rows(tmp_path / "localization_agent/point.jsonl", rows)
    write_json(tmp_path / "localization_agent/run.json", {
        "files": {"point.jsonl": sha(tmp_path / "localization_agent/point.jsonl")}})
    jobs = [{"case_id": c.case_id, "task": t, "status": "success", "candidates": items}
            for c in cases for t in ("point", "range")]
    write_json(tmp_path / "candidates.json", jobs)
    write_json(tmp_path / "candidate-registration.json", {
        "candidates_sha256": sha(tmp_path / "candidates.json"), "producer_files": {}})
    write_json(tmp_path / "base/parameters.json", parameters)
    write_json(tmp_path / "preregistered.json", {"fixture": True})
    image = tmp_path / "base/inputs/images/case_0001.png"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"fixture image")
    reg = {"arms": ["window_agent"], "agent_domains": {"window_agent": "context"},
           "shared_task_sources": {"window_agent": {"point": "localization_agent"}}, "local_plot_arms": [],
           "arm_maximum_requests": {"window_agent": 6}, "timeout_seconds": 1, "workers": 1}
    monkeypatch.setattr(experiment, "validate", lambda _: reg)
    monkeypatch.setattr(detection, "validate", lambda _: ({}, cases))
    monkeypatch.setattr(experiment, "boundary_evidence", lambda *_: {})
    monkeypatch.setattr(experiment, "credentials", lambda _: ("https://fixture.invalid", "key"))
    attempted = []
    def fail(*args, **kwargs):
        attempted.append((args[0].case_id, args[4]))
        raise RuntimeError("fixture transport failure")
    monkeypatch.setattr(agent, "review", fail)
    info = experiment.run_arm(tmp_path, "window_agent", None)
    assert not info["initial_gate_passed"] and attempted == [("case_0001", "range")]
    actual = report.load_rows(tmp_path / "window_agent/point.jsonl", "point")
    for expected in rows:
        assert actual[expected.case_id].status == expected.status
        assert actual[expected.case_id].predictions == expected.predictions
    assert all(r.status == "failed" for r in report.load_rows(tmp_path / "window_agent/range.jsonl", "range").values())


def test_point_local_views_use_every_raw_day_and_all_candidates(example, tmp_path, monkeypatch):
    from pathlib import Path

    from matplotlib.axes import Axes

    from gnss_sim import rendering

    case, _, _ = example
    items = [{"id": f"c{i:03d}", "axis": axis, "start": day, "end": day}
             for i, (axis, day) in enumerate((("N", 0), ("E", 364), ("U", 50), ("N", 100), ("E", 200)), 1)]
    captured = []
    original_plot = Axes.plot
    original_read = Path.read_bytes
    def plot(ax, x, y, *args, **kwargs):
        captured.append((np.asarray(x), np.asarray(y)))
        return original_plot(ax, x, y, *args, **kwargs)
    def no_truth(path):
        assert path.name not in ("truth.json", "manifest.json", "evaluation-registration.json")
        return original_read(path)
    monkeypatch.setattr(Axes, "plot", plot)
    monkeypatch.setattr(Path, "read_bytes", no_truth)
    paths = rendering.render_point_context(case, items, tmp_path)
    assert len(paths) == 2 and len(captured) == 5
    for item, (days, values) in zip(items, captured):
        lo, hi = max(0, item["start"]-14), min(364, item["start"]+14)
        assert days.tolist() == list(range(lo, hi+1))
        assert values.tolist() == [case.displacement_mm[d][rendering.AXES.index(item["axis"])] for d in days]
    assert rendering.plt.imread(paths[0]).shape[:2] == (1200, 1800)
    assert rendering.plt.imread(paths[1]).shape[:2] == (300, 1800)
    with pytest.raises(FileExistsError):
        rendering.render_point_context(case, items, tmp_path)
    assert rendering.render_point_context(case, [], tmp_path / "empty") == []
    with pytest.raises(ValueError):
        rendering.render_point_context(case, [{**items[0], "end": 1}], tmp_path / "invalid")


@pytest.mark.parametrize("global_arm,local_arm", [("window_agent", "local_point_agent"),
                                                 ("candidate_visual", "local_candidate_visual")])
def test_local_image_factor_preserves_prompts_tools_and_request_contract(example, tmp_path, global_arm, local_arm):
    case, parameters, items = example
    sent = {}
    for arm in (global_arm, local_arm):
        sent[arm] = []
        def respond(request):
            payload = json.loads(request.content)
            sent[arm].append(payload)
            if payload.get("tools"):
                return httpx.Response(200, json=envelope(None, [tool_call()], "tool_calls"))
            return httpx.Response(200, json=envelope())
        with httpx.Client(transport=httpx.MockTransport(respond)) as client:
            result, info = agent.review(case, b"global", items, parameters, "point", arm, client,
                "https://fixture.invalid", "key", tmp_path / arm,
                extra_images=(b"local",) if arm == local_arm else ())
        assert result.status == "success" and info["calls"] == 2
    a, b = sent[global_arm][0], sent[local_arm][0]
    assert a["messages"][0] == b["messages"][0]
    assert a["messages"][1]["content"] == b["messages"][1]["content"][:2]
    assert len(b["messages"][1]["content"]) == 3
    assert a.get("tools") == b.get("tools")
    assert all(p["model"] == "qwen3.8-flash" and p["enable_thinking"] is False
               and p["max_tokens"] == 8192 for p in sent[local_arm])


@pytest.mark.parametrize("arm", ["local_point_agent", "hypothesis_point_agent"])
def test_local_point_gate_failure_preserves_frozen_range(example, tmp_path, monkeypatch, arm):
    from gnss_sim import detection, experiment, report
    from gnss_sim.artifacts import sha, write_json, write_rows

    case, parameters, items = example
    cases = [case, case.model_copy(update={"case_id": "case_0002"})]
    rows = [RangeResult(case_id=c.case_id, method="window_agent",
                        status="success" if i == 0 else "failed",
                        predictions={"N": [[50, 60]] if i == 0 else [], "E": [], "U": []})
            for i, c in enumerate(cases)]
    write_rows(tmp_path / "window_agent/range.jsonl", rows)
    write_json(tmp_path / "window_agent/run.json", {"files": {"range.jsonl": sha(tmp_path / "window_agent/range.jsonl")}})
    write_json(tmp_path / "candidates.json", [
        {"case_id": c.case_id, "task": t, "status": "success", "candidates": items}
        for c in cases for t in ("point", "range")])
    write_json(tmp_path / "candidate-registration.json", {
        "candidates_sha256": sha(tmp_path / "candidates.json"), "producer_files": {}})
    write_json(tmp_path / "base/parameters.json", parameters)
    write_json(tmp_path / "preregistered.json", {"fixture": True})
    image = tmp_path / "base/inputs/images/case_0001.png"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"global")
    reg = {"arms": [arm], "agent_domains": {},
           "shared_task_sources": {arm: {"range": "window_agent"}},
           "local_plot_arms": [arm],
           "arm_maximum_requests": {arm: 6}, "timeout_seconds": 1, "workers": 1}
    monkeypatch.setattr(experiment, "validate", lambda _: reg)
    monkeypatch.setattr(detection, "validate", lambda _: ({}, cases))
    monkeypatch.setattr(experiment, "point_context_images", lambda *_: {})
    monkeypatch.setattr(experiment, "credentials", lambda _: ("https://fixture.invalid", "key"))
    attempted = []
    def fail(*args, **kwargs):
        attempted.append((args[0].case_id, args[4]))
        raise RuntimeError("fixture transport failure")
    monkeypatch.setattr(agent, "review", fail)
    info = experiment.run_arm(tmp_path, arm, None)
    assert not info["initial_gate_passed"] and attempted == [("case_0001", "point")]
    actual = report.load_rows(tmp_path / arm / "range.jsonl", "range")
    for expected in rows:
        assert actual[expected.case_id].status == expected.status
        assert actual[expected.case_id].predictions == expected.predictions
    assert all(r.status == "failed" for r in report.load_rows(tmp_path / arm / "point.jsonl", "point").values())


@pytest.mark.parametrize("old,new", [("local_point_agent", "hypothesis_point_agent"),
                                    ("local_candidate_visual", "hypothesis_visual")])
def test_hypothesis_factor_changes_only_system_and_survives_single_correction(example, tmp_path, monkeypatch, old, new):
    from pathlib import Path

    case, parameters, items = example
    original_read = Path.read_bytes
    def no_truth(path):
        assert path.name not in ("truth.json", "manifest.json", "evaluation-registration.json")
        return original_read(path)
    monkeypatch.setattr(Path, "read_bytes", no_truth)
    sent = {}
    for arm in (old, new):
        sent[arm] = []
        def respond(request):
            payload = json.loads(request.content)
            sent[arm].append(payload)
            if payload.get("tools"):
                return httpx.Response(200, json=envelope(None, [tool_call()], "tool_calls"))
            if arm == new and len(sent[arm]) == (2 if payload["messages"][1]["content"][0]["text"].endswith(agent.TOOL_GUIDANCE) else 1):
                return httpx.Response(200, json=envelope('{}'))
            return httpx.Response(200, json=envelope('{"c001":false}'))
        with httpx.Client(transport=httpx.MockTransport(respond)) as client:
            row, info = agent.review(case, b"global", items, parameters, "point", arm, client,
                "https://fixture.invalid", "secret-fixture", tmp_path / arm, extra_images=(b"local",))
        assert row.status == "success" and row.predictions.N == []
        assert info["calls"] == (3 if arm == new else 2)
    a, b = sent[old][0], sent[new][0]
    assert a["messages"][1:] == b["messages"][1:]
    assert a.get("tools") == b.get("tools")
    assert a["messages"][0]["content"] == agent.SYSTEM
    assert b["messages"][0]["content"] == agent.POINT_EXPLANATION_SYSTEM
    correction = next(p for p in sent[new] if isinstance(p["messages"][-1]["content"], str)
                      and "Correct it once" in p["messages"][-1]["content"])
    assert agent.POINT_EXPLANATION_SYSTEM in correction["messages"][-1]["content"]
    assert all(p["model"] == "qwen3.8-flash" and p["enable_thinking"] is False and p["max_tokens"] == 8192
               for p in sent[new])
    assert not any(b"secret-fixture" in p.read_bytes() for p in (tmp_path / new).glob("*.json"))
