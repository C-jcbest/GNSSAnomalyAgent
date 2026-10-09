"""Prospectively registered comparison of independent detectors and LangChain reviews."""
from __future__ import annotations

import base64
import importlib.metadata
import json
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import httpx

from gnss_sim import agent, detection, localization, rendering, report
from gnss_sim.artifacts import sha, write_json, write_rows
from gnss_sim.metrics import evaluate
from gnss_sim.schemas import CaseInput, CaseTruth, DatasetManifest, GenerationRequest
from gnss_sim.storage import DatasetStore
from gnss_sim.visual import credentials, failed_result

ROOT = Path(__file__).resolve().parents[2]
METHODS = ("numerical", "visual", "union", "intersection", "specialist", "support", *agent.ARMS)
LABELS = dict(zip(METHODS, ("数值 N", "视觉 V", "并集", "交集", "任务分工",
                           "候选全部保留", "候选视觉复核", "固定证据融合", "LangChain 智能体",
                           "边界收缩智能体")))
PAIRS = [("boundary_agent", m) for m in
         ("support", "langchain_agent", "specialist", "candidate_visual", "fixed_fusion")]
PACKAGES = ("numpy", "pydantic", "httpx", "affiliation", "matplotlib", "langchain",
            "langchain-core", "langgraph", "langgraph-prebuilt", "langsmith")
LOCALIZATION_ARMS = ("candidate_visual", "boundary_agent", "localization_agent")
LOCALIZATION_METHODS = (*METHODS[:6], *LOCALIZATION_ARMS, "tool_boundaries")
LABELS.update(localization_agent="定位工具智能体", tool_boundaries="定位工具固定评分")
LOCALIZATION_PAIRS = [("localization_agent", m) for m in
                      ("tool_boundaries", "boundary_agent", "support", "visual", "candidate_visual")]
WINDOW_ARMS = ("candidate_visual", "localization_agent", "window_agent")
WINDOW_METHODS = (*METHODS[:6], *WINDOW_ARMS, "tool_boundaries", "window_tool_boundaries")
WINDOW_PAIRS = [("window_agent", "localization_agent"),
                ("window_agent", "window_tool_boundaries"),
                ("window_tool_boundaries", "tool_boundaries"),
                ("window_agent", "visual"), ("window_agent", "support"),
                ("window_agent", "candidate_visual")]
LABELS.update(window_agent="窗口搜索智能体", window_tool_boundaries="窗口搜索固定评分")
POINT_ARMS = ("candidate_visual", "local_candidate_visual", "window_agent", "local_point_agent")
POINT_METHODS = (*METHODS[:6], *POINT_ARMS, "window_tool_boundaries")
POINT_PAIRS = [("local_point_agent", "window_agent"),
               ("local_candidate_visual", "candidate_visual"),
               ("local_point_agent", "local_candidate_visual"),
               ("local_point_agent", "numerical"), ("window_agent", "candidate_visual")]
LABELS.update(local_point_agent="局部图数值智能体", local_candidate_visual="局部图视觉复核")
HYPOTHESIS_ARMS = ("window_agent", "local_candidate_visual", "hypothesis_visual",
                   "local_point_agent", "hypothesis_point_agent")
HYPOTHESIS_METHODS = (*METHODS[:6], *HYPOTHESIS_ARMS, "window_tool_boundaries")
HYPOTHESIS_PAIRS = [("hypothesis_point_agent", "local_point_agent"),
                    ("hypothesis_visual", "local_candidate_visual"),
                    ("hypothesis_point_agent", "hypothesis_visual"),
                    ("hypothesis_point_agent", "numerical"),
                    ("local_point_agent", "local_candidate_visual")]
LABELS.update(hypothesis_point_agent="双解释数值智能体", hypothesis_visual="双解释视觉复核")


def read(path):
    return json.loads(path.read_bytes())


def source_paths():
    return [*sorted((ROOT / "src/gnss_sim").glob("*.py")), ROOT / "uv.lock",
            ROOT / "pyproject.toml", ROOT / "docs/detection.md",
            ROOT / "configs/generator-sources.json", ROOT / "configs/generator-requirements.txt"]


def register(out, seed, count, parameters, design="boundary", calibration_dataset=None):
    if out.exists():
        raise FileExistsError(out)
    if count < 24 or count % 24:
        raise ValueError("Use complete 24-case axis/sign cycles")
    if design not in ("boundary", "localization", "window", "point_context", "point_hypothesis"):
        raise ValueError("Unknown experiment design")
    GenerationRequest(seed=seed, count=count, case_type="all")
    prior = sorted((ROOT / "data/generated").glob("synthetic-v1-*/manifest.json"))
    if any(read(p)["request"]["seed"] == seed for p in prior):
        raise ValueError("Generation seed already used; do not redraw")
    out.mkdir(parents=True)
    for p in source_paths():
        dest = out / "source" / p.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dest)
    shutil.copy2(parameters, out / "calibration.json")
    arms, methods, pairs = {"boundary": (agent.ARMS, METHODS, PAIRS),
                            "localization": (LOCALIZATION_ARMS, LOCALIZATION_METHODS, LOCALIZATION_PAIRS),
                            "window": (WINDOW_ARMS, WINDOW_METHODS, WINDOW_PAIRS),
                            "point_context": (POINT_ARMS, POINT_METHODS, POINT_PAIRS),
                            "point_hypothesis": (HYPOTHESIS_ARMS, HYPOTHESIS_METHODS, HYPOTHESIS_PAIRS)}[design]
    shared = {"window_agent": {"point": "localization_agent"}} if design == "window" else {
        "local_candidate_visual": {"range": "candidate_visual"},
        "local_point_agent": {"range": "window_agent"}} if design == "point_context" else {}
    if design == "point_hypothesis":
        shared = {arm: {"range": "window_agent"} for arm in HYPOTHESIS_ARMS if arm != "window_agent"}
    local_arms = (["local_candidate_visual", "local_point_agent"] if design == "point_context" else
                  list(agent.LOCAL_POINT_ARMS) if design == "point_hypothesis" else [])
    arm_maximum = {arm: (3 if arm in shared else 6)*count for arm in arms}
    review_maximum = sum(arm_maximum.values())
    reg = {"protocol": "langchain-boundary-v2", "scope": "new-background development",
           "registered_at": datetime.now(timezone.utc).isoformat(),
           "seed": seed, "cases": count, "bootstrap_seed": seed + 1, "bootstrap_repeats": 2000,
           "maximum_requests": 4 * count + review_maximum, "baseline_maximum_requests": 4 * count,
           "review_maximum_requests": review_maximum, "per_review_task_maximum": 3,
           "methods": list(methods), "arms": list(arms), "primary_pair": list(pairs[0]),
           "arm_maximum_requests": arm_maximum,
           "shared_task_sources": shared,
           "local_plot_arms": local_arms,
           "point_context_config": rendering.POINT_CONTEXT if local_arms else None,
           "evidence_domains": [], "agent_domains": {}, "tool_methods": {},
           "secondary_pairs": pairs[1:], "primary_metric": "range daily F1",
           "success_criterion": "primary paired F1 interval lower bound > 0; also report recall, "
                                "normal alarms, failures and TP/FP removed; no post-hoc rerun",
           "design": {"candidates": "N points; V ranges plus N temporary ranges",
                      "single_changed_factor": "range action space in boundary_agent only",
                      "action": "controls: accept/reject entire candidate; boundary_agent range: "
                                "reject/keep/shrink within candidate; point remains boolean",
                      "route": "all inputs; no candidates -> empty success without calls",
                      "model_turns": "two semantic turns and at most one structural correction",
                      "agent": "one batch of one/two read-only tools, then final decision",
                      "boundary_agent": "same tools, evidence, plot and turns as langchain_agent; "
                                        "only range output contract and action instructions differ",
                      "fixed_fusion": "both tools for all candidates, provisional then final",
                      "candidate_visual": "same candidates and plot, provisional then final",
                      "gate": "first nonempty point and range per arm; failure stops that arm",
                      "interpretation": "matched call opportunities; evidence/turn function differ"},
           "model": "qwen3.8-flash", "enable_thinking": False, "max_tokens": 8192,
           "workers": 4, "timeout_seconds": 120,
           "sources": {p.relative_to(ROOT).as_posix(): sha(p) for p in source_paths()},
           "versions": {n: importlib.metadata.version(n) for n in PACKAGES},
           "calibration_sha256": sha(out / "calibration.json"),
           "prior_manifests": {str(p.resolve()): sha(p) for p in prior}}
    if design in ("localization", "window", "point_context", "point_hypothesis"):
        frozen = read(parameters)
        candidates = ([calibration_dataset / "manifest.json"] if calibration_dataset is not None
                      else [p for p in prior if read(p)["request"]["seed"] == frozen["calibration"]["seed"]])
        if len(candidates) != 1:
            raise ValueError("Locate exactly one dedicated calibration dataset")
        manifest = DatasetManifest.model_validate_json(candidates[0].read_bytes())
        if (manifest.status != "complete" or manifest.request.case_type != "normal"
                or manifest.request.seed != frozen["calibration"]["seed"]
                or any(c.case_type != "normal" for c in manifest.cases)
                or len(manifest.cases) != frozen["calibration"]["cases"]):
            raise ValueError("Expected the complete frozen Normal calibration dataset")
        paths = [candidates[0].parent / "cases" / c.case_id / "input.json" for c in manifest.cases]
        if [sha(p) for p in paths] != frozen["calibration"]["inputs"]:
            raise ValueError("Normal calibration observations changed")
        scales = localization.noise_scales([CaseInput.model_validate_json(p.read_bytes()) for p in paths])
        write_json(out / "localization-calibration.json", {
            "config": localization.CONFIG, "sigma_mm": scales,
            "calibration_input_hashes": frozen["calibration"]["inputs"],
            "source_sha256": sha(ROOT / "src/gnss_sim/localization.py")})
        reg.update(protocol="localization-v1",
                   localization_calibration_sha256=sha(out / "localization-calibration.json"))
        reg.update(evidence_domains=["endpoints"], agent_domains={"localization_agent": "endpoints"},
                   tool_methods={"tool_boundaries": "endpoints"})
        reg["design"] = {
            "candidates": "unchanged: N points; V ranges plus N temporary ranges",
            "factor": "dedicated boundary fitting plus proposal-ID selection; complete strategy",
            "search": localization.CONFIG,
            "calibration": "noise scales from the same independent complete Normal observations",
            "tool_rule": "minimum SSE/sigma^2+k*log(n); constant/permanent step => reject; "
                         "temporary level or clipped ramp => select its inclusive active range",
            "agent": "all candidates get identical frozen level/trend proposals and samples; "
                     "one required tool batch, final null/original/level/trend selection",
            "point": "unchanged numerical candidates and original statistical review; "
                     "tool_boundaries point equals numerical; do not attribute point differences to range",
            "gate": "first nonempty point/range per arm; failed gate stops arm",
            "limits": "all inputs, no truth routing; no new candidate regions; expansion is limited "
                      "to observation-defined search windows; near-best spans are not confidence intervals"}
        if design == "window":
            reg.update(protocol="localization-window-v1", evidence_domains=["endpoints", "context"],
                       agent_domains={"localization_agent": "endpoints", "window_agent": "context"},
                       tool_methods={"tool_boundaries": "endpoints", "window_tool_boundaries": "context"})
            reg["design"].update(
                factor="only endpoint search domain; same fits, scoring, context and prompts",
                search="endpoints: each original endpoint +/-31; context: both endpoints anywhere "
                       "in the same original [start-62,end+62] clipped observation context",
                point="window_agent copies identical frozen localization_agent point results "
                      "including failures; no new point requests, to remove service variation",
                interpretation="primary new-vs-old agent tests endpoint domain; secondary new "
                               "agent-vs-new tool tests additional model confirmation")
        if design in ("point_context", "point_hypothesis"):
            reg.update(protocol="point-context-v1", evidence_domains=["context"],
                       agent_domains={"window_agent": "context"},
                       tool_methods={"window_tool_boundaries": "context"},
                       primary_metric="point daily F1",
                       success_criterion="primary paired Point F1 interval lower bound > 0 AND "
                                         "local agent Point TP >= global agent Point TP; "
                                         "report all recall, normal alarms, failures and removed TP/FP")
            reg["design"].update(
                factor="only additional candidate-local observation plots for Point; global plot retained",
                point="same candidates, prompts, numerical tools and two turns; all candidates get "
                      "raw axis/day plots +/-14 days clipped to year, four candidates per page",
                range="unchanged full-context boundary fitting; local visual/agent copies frozen "
                      "range outputs from corresponding global arm, including failures",
                search="both endpoints anywhere in original observation context",
                interpretation="2x2 global/local plot x visual/tool review; primary local-vs-global "
                               "tool agent, secondary image effect and evidence combination; "
                               "extra image pixels and adaptive tool choices belong to full strategy")
        if design == "point_hypothesis":
            reg.update(protocol="point-hypothesis-v1",
                       success_criterion="primary paired Point F1 interval lower bound > 0 AND "
                                         "new agent Point TP >= original agent Point TP; report all "
                                         "recall, normal alarms, failures and removed TP/FP")
            reg["design"].update(
                factor="only Point system decision instructions: compare genuine anomaly vs "
                       "ordinary background spike; no new tools, thresholds or output fields",
                point="all four Point arms share identical global and candidate-local raw images, "
                      "candidates, tool definitions, two turns and boolean output; new instructions "
                      "applied to both visual and tool review, including structural correction",
                range="all four Point arms copy the same frozen window_agent Range outputs "
                      "including failures; Point gate failure never changes shared Range",
                interpretation="2x2 original/new decision instructions x visual/tool review; "
                               "primary new-vs-original tool review, secondary prompt effect "
                               "and tool combination; model explanations are not elicited")
    write_json(out / "preregistered.json", reg)
    write_json(out / "registration-seal.json", {"sha256": sha(out / "preregistered.json")})
    return {k: reg[k] for k in ("seed", "cases", "maximum_requests", "primary_pair")}


def validate(out):
    if sha(out / "preregistered.json") != read(out / "registration-seal.json")["sha256"]:
        raise ValueError("Preregistered settings changed")
    reg = read(out / "preregistered.json")
    if reg["sources"] != {p.relative_to(ROOT).as_posix(): sha(p) for p in source_paths()}:
        raise ValueError("Experiment source changed")
    if reg["versions"] != {n: importlib.metadata.version(n) for n in PACKAGES}:
        raise ValueError("Experiment dependencies changed")
    if reg["calibration_sha256"] != sha(out / "calibration.json"):
        raise ValueError("Calibration changed")
    if reg["evidence_domains"] and reg["localization_calibration_sha256"] != sha(out / "localization-calibration.json"):
        raise ValueError("Localization calibration changed")
    return reg


def generate_base(out, reg):
    """Preparation boundary alone sees labels to audit background separation."""
    for path, digest in reg["prior_manifests"].items():
        if sha(Path(path)) != digest:
            raise ValueError("Prior manifest changed")
    store = DatasetStore(ROOT / "data/generated")
    try:
        meta = store.generate_sync(GenerationRequest(seed=reg["seed"], count=reg["cases"],
                                                     case_type="all"))
    finally:
        store.executor.shutdown(wait=True)
    if meta.status != "complete":
        raise ValueError("Generation failed; no redraw")
    dataset = store.root / meta.dataset_id
    old_seeds, old_noise, new_seeds, new_noise = set(), set(), set(), set()
    for p in [*[Path(x) for x in reg["prior_manifests"]], dataset / "manifest.json"]:
        manifest = DatasetManifest.model_validate_json(p.read_bytes())
        seeds, noises = (new_seeds, new_noise) if p.parent == dataset else (old_seeds, old_noise)
        for c in manifest.cases:
            truth = CaseTruth.model_validate_json((p.parent / "cases" / c.case_id / "truth.json").read_bytes())
            seeds.add(truth.noise_seed)
            import hashlib
            noises.add(hashlib.sha256(json.dumps(truth.measurement_noise_mm).encode()).hexdigest())
    if new_seeds & old_seeds or new_noise & old_noise:
        raise ValueError("Background overlap; stop without redraw")
    write_json(out / "isolation-audit.json", {"new_backgrounds": len(new_seeds),
                "previous_backgrounds": len(old_seeds), "noise_seed_overlap": 0,
                "noise_array_overlap": 0, "dataset": str(dataset.resolve())})
    return detection.prepare(dataset, out / "calibration.json", out / "base")


def prepare_candidates(out):
    """No truth/evaluation registration reads beyond this inference boundary."""
    base = out / "base"
    _, cases = detection.validate(base)
    points = report.load_rows(base / "numerical/point.jsonl", "point")
    visual = report.load_rows(base / "visual/range.jsonl", "range")
    temporary = report.load_rows(base / "numerical/branches/temporary.jsonl", "range")
    jobs = []
    for case in cases:
        cid = case.case_id
        for task in detection.TASKS:
            try:
                items = agent.candidates(points.get(cid), visual.get(cid), temporary.get(cid), task)
                status = "success"
            except ValueError:
                items, status = [], "failed"
            jobs.append({"case_id": cid, "task": task, "status": status, "candidates": items})
    write_json(out / "candidates.json", jobs)
    write_json(out / "candidate-registration.json", {
        "candidates_sha256": sha(out / "candidates.json"),
        "producer_files": {str(p.relative_to(out)): sha(p) for folder in ("numerical", "visual")
                           for p in (base / folder).rglob("*") if p.is_file()}})
    reg = read(out / "preregistered.json")
    if reg["local_plot_arms"]:
        by_case = {case.case_id: case for case in cases}
        images = {}
        for job in (j for j in jobs if j["task"] == "point"):
            cid = job["case_id"]
            paths = rendering.render_point_context(by_case[cid], job["candidates"],
                                                   out / "point-context-images" / cid)
            images[cid] = [p.relative_to(out).as_posix() for p in paths]
        write_json(out / "point-context-registration.json", {
            "candidate_registration_sha256": sha(out / "candidate-registration.json"),
            "config": reg["point_context_config"], "images": images,
            "files": {name: sha(out / name) for names in images.values() for name in names}})
        write_json(out / "point-context-seal.json", {"sha256": sha(out / "point-context-registration.json")})
    domains = reg["evidence_domains"]
    if domains:
        parameters = read(base / "parameters.json")
        scales = read(out / "localization-calibration.json")["sigma_mm"]
        by_case = {case.case_id: case for case in cases}
        for domain in domains:
            prefix = "window" if domain == "context" else "localization"
            evidence = {}
            for job in jobs:
                if job["task"] == "range" and job["status"] == "success":
                    evidence[job["case_id"]] = {c["id"]: localization.propose(
                        by_case[job["case_id"]], c, parameters, scales[c["axis"]], domain)
                        for c in job["candidates"]}
            write_json(out / f"{prefix}-evidence.json", evidence)
            write_json(out / f"{prefix}-evidence-registration.json", {
                "evidence_sha256": sha(out / f"{prefix}-evidence.json"),
                "candidate_registration_sha256": sha(out / "candidate-registration.json"),
                "calibration_sha256": sha(out / "localization-calibration.json")})
    return cases, jobs


def boundary_evidence(out, domain="endpoints"):
    prefix = "window" if domain == "context" else "localization"
    saved = read(out / f"{prefix}-evidence-registration.json")
    for name, key in ((f"{prefix}-evidence.json", "evidence_sha256"),
                      ("candidate-registration.json", "candidate_registration_sha256"),
                      ("localization-calibration.json", "calibration_sha256")):
        if sha(out / name) != saved[key]:
            raise ValueError("Boundary evidence changed")
    return read(out / f"{prefix}-evidence.json")


def run_arm(out, arm, env_file):
    reg = validate(out)
    if arm not in reg["arms"]:
        raise ValueError("Unregistered arm")
    _, cases = detection.validate(out / "base")
    creg = read(out / "candidate-registration.json")
    if sha(out / "candidates.json") != creg["candidates_sha256"]:
        raise ValueError("Candidates changed")
    for name, digest in creg["producer_files"].items():
        if sha(out / name) != digest:
            raise ValueError("Candidate producers changed")
    jobs = read(out / "candidates.json")
    case_map = {c.case_id: c for c in cases}
    parameters = read(out / "base/parameters.json")
    packages = boundary_evidence(out, reg["agent_domains"][arm]) if arm in reg["agent_domains"] else {}
    task_sources = reg["shared_task_sources"].get(arm, {})
    shared_rows = {}
    for task, source in task_sources.items():
        saved = read(out / source / "run.json")
        for name, digest in saved["files"].items():
            if sha(out / source / name) != digest:
                raise ValueError("Shared task source changed")
        shared_rows[task] = report.load_rows(out / source / f"{task}.jsonl", task)
        if set(shared_rows[task]) != set(case_map):
            raise ValueError("Shared task inputs differ")
    local_images = {}
    if arm in reg["local_plot_arms"]:
        local_images = point_context_images(out, reg)
    endpoint, key = credentials(env_file)
    if not endpoint.startswith("https://"):
        raise ValueError("HTTPS required")
    directory = out / arm
    directory.mkdir(exist_ok=False)
    write_json(directory / "started.json", {"registration_sha256": sha(out / "preregistered.json"),
                                            "maximum_requests": reg["arm_maximum_requests"][arm],
                                            "shared_task_sources": task_sources})
    start = time.perf_counter()
    with httpx.Client(timeout=reg["timeout_seconds"], follow_redirects=False) as client:
        def execute(job):
            cid, task = job["case_id"], job["task"]
            row, info = failed_result(cid, task, arm), {"error": None}
            if task in task_sources:
                source = task_sources[task]
                row = shared_rows[task][cid].model_copy(update={"method": arm})
                info = {"shared_task_source": source, "calls": 0,
                        "source_sha256": sha(out / source / f"{task}.jsonl")}
            elif job["status"] == "success":
                try:
                    row, info = agent.review(case_map[cid],
                        (out / "base/inputs/images" / f"{cid}.png").read_bytes(),
                        job["candidates"], parameters, task, arm, client, endpoint, key,
                        directory / "requests" / cid / task,
                        boundary_packages=packages.get(cid) if task == "range" else None,
                        extra_images=tuple((out / name).read_bytes() for name in local_images.get(cid, []))
                        if task == "point" else ())
                except Exception as exc:
                    info = {"error": type(exc).__name__}
            else:
                info = {"error": "candidate_producer_failed"}
            write_json(directory / "results" / f"{cid}-{task}.json", {
                "result": row.model_dump(mode="json"), "info": info})
            print(json.dumps({"arm": arm, "case": cid, "task": task, "status": row.status}), flush=True)
            return job, row
        gate_jobs = []
        for task in detection.TASKS:
            first = next((j for j in jobs if j["task"] == task and j["status"] == "success"
                          and j["candidates"]), None)
            if first:
                gate_jobs.append(first)
        responses = [execute(j) for j in gate_jobs]
        gate = all(row.status == "success" for _, row in responses)
        remaining = [j for j in jobs if j not in gate_jobs]
        if gate:
            with ThreadPoolExecutor(max_workers=reg["workers"]) as pool:
                responses.extend(pool.map(execute, remaining))
        else:
            responses.extend(execute(j) if j["task"] in task_sources else
                             (j, failed_result(j["case_id"], j["task"], arm)) for j in remaining)
    for task in detection.TASKS:
        write_rows(directory / f"{task}.jsonl", sorted(
            [row for j, row in responses if j["task"] == task], key=lambda r: r.case_id))
    records = [read(p) for p in directory.glob("requests/*/*/*-completed.json")]
    calls = len(list(directory.glob("requests/*/*/*-started.json")))
    if calls > reg["arm_maximum_requests"][arm]:
        raise RuntimeError("Review request budget exceeded")
    cost = {"calls": calls, "completed_requests": len(records), "initial_gate_passed": gate,
            "failures": sum(r.status != "success" for _, r in responses),
            "total_tokens": sum((r["usage"] or {}).get("total_tokens", 0) for r in records),
            "usage_missing": sum(r["usage"] is None for r in records), "currency_cost": None,
            "structural_invalid_responses": sum(str(r["error"]).startswith("structure") for r in records),
            "wall_seconds": time.perf_counter() - start,
            "files": {p.relative_to(directory).as_posix(): sha(p)
                      for p in directory.rglob("*") if p.is_file()}}
    write_json(directory / "run.json", cost)
    return {k: v for k, v in cost.items() if k != "files"}


def point_context_images(out, reg):
    """Verify frozen observation plots before supplying them to any model."""
    path = out / "point-context-registration.json"
    saved = read(path)
    if sha(path) != read(out / "point-context-seal.json")["sha256"]:
        raise ValueError("Point image registration changed")
    if saved["candidate_registration_sha256"] != sha(out / "candidate-registration.json"):
        raise ValueError("Point image candidates changed")
    if saved["config"] != reg["point_context_config"]:
        raise ValueError("Point image configuration changed")
    for name, digest in saved["files"].items():
        if sha(out / name) != digest:
            raise ValueError("Point observation image changed")
    return saved["images"]


def score(out):
    reg = validate(out)
    base = out / "base"
    detection.validate(base)
    if (out / "evaluation.json").exists():
        raise FileExistsError("Evaluation exists")
    ereg = read(base / "evaluation-registration.json")
    if sha(base / "evaluation-registration.json") != read(base / "registered.json")["evaluation_registration_sha256"]:
        raise ValueError("Evaluation registration changed")
    for name, digest in ereg["artifacts"].items():
        if sha(Path(name)) != digest:
            raise ValueError("Evaluation data changed")
    creg = read(out / "candidate-registration.json")
    if sha(out / "candidates.json") != creg["candidates_sha256"]:
        raise ValueError("Candidates changed")
    for name, digest in creg["producer_files"].items():
        if sha(out / name) != digest:
            raise ValueError("Producer changed")
    jobs = read(out / "candidates.json")
    ids = sorted({j["case_id"] for j in jobs})
    data = Path(ereg["dataset"])
    metadata = read(data / "manifest.json")["cases"]
    truths = {cid: CaseTruth.model_validate_json((data / "cases" / cid / "truth.json").read_bytes()) for cid in ids}
    pred = {m: {t: report.load_rows(base / m / f"{t}.jsonl", t) for t in detection.TASKS}
            for m in ("numerical", "visual")}
    for m in ("union", "intersection"):
        pred[m] = {t: report.combine(pred["numerical"][t], pred["visual"][t], ids, t, m)
                   for t in detection.TASKS}
    pred["specialist"] = {"point": pred["numerical"]["point"], "range": pred["visual"]["range"]}
    pred["support"] = {t: {} for t in detection.TASKS}
    for j in jobs:
        cid, t, items = j["case_id"], j["task"], j["candidates"]
        pred["support"][t][cid] = (agent.result_from_decisions(cid, t, "support", items,
            {c["id"]: True for c in items}) if j["status"] == "success"
            else failed_result(cid, t, "support"))
    cost = {"visual": read(base / "visual/run.json"), "numerical": read(base / "numerical/run.json")}
    for arm in reg["arms"]:
        cost[arm] = read(out / arm / "run.json")
        for name, digest in cost[arm]["files"].items():
            if sha(out / arm / name) != digest:
                raise ValueError("Review results changed")
        pred[arm] = {t: report.load_rows(out / arm / f"{t}.jsonl", t) for t in detection.TASKS}
    for method, domain in reg["tool_methods"].items():
        packages = boundary_evidence(out, domain)
        pred[method] = {"point": pred["numerical"]["point"], "range": {}}
        for job in (j for j in jobs if j["task"] == "range"):
            cid = job["case_id"]
            pred[method]["range"][cid] = (
                localization.result(cid, method, packages[cid],
                    {key: p["rule_choice"] for key, p in packages[cid].items()})
                if job["status"] == "success" else failed_result(cid, "range", method))
        folder = out / method
        folder.mkdir(exist_ok=False)
        for task in detection.TASKS:
            write_rows(folder / f"{task}.jsonl", list(pred[method][task].values()))
        cost[method] = {"calls": 0, "additional_model_calls": 0,
                        "total_tokens": 0, "wall_seconds": 0}
    calls = sum(c.get("calls", 0) for c in cost.values())
    if calls > reg["maximum_requests"]:
        raise ValueError("Experiment request budget exceeded")
    def summarize(selected):
        return {m: {t: evaluate(selected, {cid: row for cid, row in pred[m][t].items()
                     if cid in selected}, t) for t in detection.TASKS} for m in reg["methods"]}
    result = {"scope": reg["scope"], "summary": summarize(truths),
        "by_type": {kind: summarize({r["case_id"]: truths[r["case_id"]] for r in metadata
                    if r["case_type"] == kind}) for kind in ("normal", "global_extremum", "trend", "mean_shift")},
        "cost": cost, "total_calls": calls,
        "paired_analysis": report.paired_analysis(truths, pred, reg["bootstrap_repeats"],
                                                   reg["bootstrap_seed"], reg["methods"],
                                                   [reg["primary_pair"], *reg["secondary_pairs"]]),
        "cases": [{**r, "truth": [e.model_dump(mode="json") for e in truths[r["case_id"]].events],
                   "predictions": {m: {t: pred[m][t][r["case_id"]].model_dump(mode="json")
                                       for t in detection.TASKS} for m in reg["methods"]}} for r in metadata]}
    write_json(out / "evaluation.json", result)
    for r in result["cases"]:
        r["image_base64"] = base64.b64encode((base / "inputs/images" / f'{r["case_id"]}.png').read_bytes()).decode()
    report.export_html(out, result, reg["methods"], LABELS)
    return {"total_calls": calls, "f1": {m: {t: result["summary"][m][t]["daily"]["f1"]
                                            for t in detection.TASKS} for m in reg["methods"]}}


def execute(out, env_file):
    reg = validate(out)
    write_json(out / "execution-started.json", {"registration_sha256": sha(out / "preregistered.json")})
    generate_base(out, reg)
    detection.run_numerical(out / "base")
    detection.run_visual(out / "base", env_file)
    prepare_candidates(out)
    for arm in reg["arms"]:
        print(json.dumps({"completed_arm": arm, **run_arm(out, arm, env_file)}), flush=True)
    result = score(out)
    write_json(out / "completed.json", {"result": result,
        "files": {p.relative_to(out).as_posix(): sha(p) for p in out.rglob("*") if p.is_file()}})
    return result
