"""The fixed, metadata-stratified P4 development pilot."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from uuid import uuid4

import numpy as np

from gnss_sim import scenarios
from gnss_sim.generator import GENERATOR_VERSION, derive_event_seeds, generate_case
from gnss_sim.schemas import CASE_TYPES, SCENARIO_TYPES, CaseInput, CaseTruth

PILOT_ID = "pilot-v1"
DEFAULT_SEED = 20260925
AXES = ("N", "E", "U")
SINGLE_TYPES = CASE_TYPES[1:]
SOURCE_FILES = ("generator.py", "events.py", "scenarios.py", "pilot.py")


def _json_bytes(value: dict) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
            + "\n").encode("utf-8")


def _seed(seed: int, *namespace: int) -> int:
    return int(np.random.SeedSequence([seed, 0x5034, *namespace]).generate_state(
        1, dtype=np.uint32)[0])


def _single_metadata(case_seed: int) -> tuple[str, str]:
    streams = derive_event_seeds(case_seed)
    axis = AXES[int(np.random.default_rng(streams.shape).integers(0, 3))]
    sign = "positive" if np.random.default_rng(streams.sign).integers(0, 2) else "negative"
    return axis, sign


def _planned_cases(seed: int):
    for index in range(30):
        yield "normal", _seed(seed, 0, index), None, None
    for type_index, kind in enumerate(SINGLE_TYPES, start=1):
        for axis_index, axis in enumerate(AXES):
            for sign_index, sign in enumerate(("positive", "negative")):
                found = 0
                attempt = 0
                while found < 4:
                    case_seed = _seed(seed, 1, type_index, axis_index, sign_index, attempt)
                    attempt += 1
                    if _single_metadata(case_seed) == (axis, sign):
                        yield kind, case_seed, axis, sign
                        found += 1
                    if attempt > 10000:
                        raise RuntimeError(f"seed search failed for {kind}/{axis}/{sign}")
    for scenario_index, scenario in enumerate(SCENARIO_TYPES):
        for index in range(25):
            yield scenario, _seed(seed, 2, scenario_index, index), None, None


def _source_hashes() -> dict[str, str]:
    root = Path(__file__).parent
    return {name: hashlib.sha256((root / name).read_text(encoding="utf-8")
                                 .replace("\r\n", "\n").encode("utf-8")).hexdigest()
            for name in SOURCE_FILES}


def _validate_case(case_input: CaseInput, truth: CaseTruth, kind: str,
                   axis: str | None, sign: str | None) -> None:
    if case_input.case_id != truth.case_id or len(case_input.dates) != 365:
        raise ValueError("case identity or length mismatch")
    if kind == "normal" and truth.events:
        raise ValueError("normal case has events")
    if kind in SINGLE_TYPES:
        if len(truth.events) != 1:
            raise ValueError("single case does not have exactly one event")
        event = truth.events[0]
        magnitude = (event.parameters.final_offset_mm if kind in ("slow_trend", "acceleration")
                     else event.parameters.amplitude_mm)
        if event.type != kind or event.axis != axis or (magnitude > 0) != (sign == "positive"):
            raise ValueError("single stratum mismatch")
    if kind in SCENARIO_TYPES and (truth.scenario_type != kind or not 2 <= len(truth.events) <= 6):
        raise ValueError("multi scenario mismatch")
    if any(event.start_index > event.end_index for event in truth.events):
        raise ValueError("invalid truth interval")
    deformation = np.zeros((365, 3))
    artifact = np.zeros((365, 3))
    for event, part in zip(truth.events, truth.event_contributions):
        magnitude = (event.parameters.final_offset_mm if event.type in
                     ("slow_trend", "acceleration") else event.parameters.amplitude_mm)
        expected, _ = scenarios.MAKERS[event.type](case_input.dates, event.axis,
                                                     event.start_index, magnitude)
        if not np.allclose(part.values_mm, expected, rtol=0, atol=1e-14):
            raise ValueError("event contribution differs from canonical profile")
        if part.component == "injected_deformation":
            deformation += part.values_mm
        else:
            artifact += part.values_mm
    if not np.allclose(deformation, truth.injected_deformation_mm, rtol=0, atol=1e-13):
        raise ValueError("aggregate deformation mismatch")
    if not np.allclose(artifact, truth.observation_artifact_mm, rtol=0, atol=1e-13):
        raise ValueError("aggregate artifact mismatch")
    expected_observed = (np.asarray(case_input.reference_coordinate_mm)
                         + np.asarray(truth.normal_background_mm)
                         + np.asarray(truth.measurement_noise_mm) + deformation + artifact)
    if not np.allclose(case_input.observed_coordinate_mm, expected_observed,
                       rtol=0, atol=1e-13):
        raise ValueError("observed composition mismatch")


def _summary(truths: list[tuple[str, CaseTruth]]) -> dict:
    groups = Counter("normal" if kind == "normal" else "single" if kind in SINGLE_TYPES
                     else "multi" for kind, _ in truths)
    singles = Counter(kind for kind, _ in truths if kind in SINGLE_TYPES)
    single_axes = Counter(event.axis for kind, truth in truths if kind in SINGLE_TYPES
                          for event in truth.events)
    single_signs = Counter("positive" if (event.parameters.final_offset_mm
                           if kind in ("slow_trend", "acceleration") else
                           event.parameters.amplitude_mm) > 0 else "negative"
                           for kind, truth in truths if kind in SINGLE_TYPES
                           for event in truth.events)
    strata = Counter((kind, truth.events[0].axis,
                      "positive" if (truth.events[0].parameters.final_offset_mm
                       if kind in ("slow_trend", "acceleration") else
                       truth.events[0].parameters.amplitude_mm) > 0 else "negative")
                     for kind, truth in truths if kind in SINGLE_TYPES)
    scenarios = Counter(kind for kind, _ in truths if kind in SCENARIO_TYPES)
    multi = [truth for kind, truth in truths if kind in SCENARIO_TYPES]
    counts = Counter(len(truth.events) for truth in multi)
    event_types = Counter(event.type for _, truth in truths for event in truth.events)
    same_axis = sum(len({event.axis for event in truth.events}) == 1 for truth in multi)
    return {
        "total_cases": len(truths), "groups": dict(groups),
        "single_types": dict(singles), "single_axes": dict(single_axes),
        "single_signs": dict(single_signs), "multi_scenarios": dict(scenarios),
        "single_strata": {f"{kind}/{axis}/{sign}": strata[kind, axis, sign]
                          for kind in SINGLE_TYPES for axis in AXES
                          for sign in ("positive", "negative")},
        "multi_events_per_case": {str(i): counts[i] for i in range(2, 7)},
        "multi_axis_layout": {"same_axis": same_axis, "cross_axis": len(multi) - same_axis},
        "event_types": dict(event_types),
    }


def verify_pilot(directory: Path) -> dict:
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest["pilot_id"] != PILOT_ID or manifest["generator_version"] != GENERATOR_VERSION:
        raise ValueError("pilot protocol mismatch")
    if manifest["source_sha256"] != _source_hashes():
        raise ValueError("frozen generator source has changed")
    planned = list(_planned_cases(manifest["seed"]))
    if len(manifest["cases"]) != len(planned):
        raise ValueError("pilot case count mismatch")
    truths = []
    for index, (entry, plan) in enumerate(zip(manifest["cases"], planned), start=1):
        kind, case_seed, axis, sign = plan
        if (entry["case_id"], entry["case_type"], entry["case_seed"],
                entry["axis"], entry["sign"]) != (f"case_{index:04d}", kind,
                                                   case_seed, axis, sign):
            raise ValueError("pilot seed or stratum plan mismatch")
        case_dir = directory / "cases" / entry["case_id"]
        for name in ("input", "truth"):
            raw = (case_dir / f"{name}.json").read_bytes()
            if hashlib.sha256(raw).hexdigest() != entry[f"{name}_sha256"]:
                raise ValueError(f"pilot file hash mismatch: {entry['case_id']}/{name}")
        case_input = CaseInput.model_validate_json((case_dir / "input.json").read_bytes())
        truth = CaseTruth.model_validate_json((case_dir / "truth.json").read_bytes())
        _validate_case(case_input, truth, entry["case_type"], entry["axis"], entry["sign"])
        if len(truth.events) != entry["event_count"]:
            raise ValueError("manifest event count mismatch")
        truths.append((entry["case_type"], truth))
    summary = _summary(truths)
    stored = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    if stored != summary or len(truths) != 300 or summary["groups"] != {
        "normal": 30, "single": 120, "multi": 150
    } or any(count != 4 for count in summary["single_strata"].values()):
        raise ValueError("pilot summary mismatch")
    return manifest


def generate_pilot(root: Path = Path("data/pilots"), seed: int = DEFAULT_SEED) -> Path:
    if not 0 <= seed <= 4294967295:
        raise ValueError("seed must be a uint32")
    directory = root / PILOT_ID
    if directory.exists():
        manifest = verify_pilot(directory)
        if manifest["seed"] != seed:
            raise ValueError("pilot-v1 already exists with a different seed")
        return directory
    root.mkdir(parents=True, exist_ok=True)
    temporary = root / f".{PILOT_ID}-{uuid4().hex}.tmp"
    temporary.mkdir()
    try:
        cases = []
        truths = []
        seen_seeds = set()
        for index, (kind, case_seed, axis, sign) in enumerate(_planned_cases(seed), start=1):
            if case_seed in seen_seeds:
                raise ValueError("case seed collision")
            seen_seeds.add(case_seed)
            case_id = f"case_{index:04d}"
            case_input, truth = generate_case(case_id, case_seed, kind)
            _validate_case(case_input, truth, kind, axis, sign)
            case_dir = temporary / "cases" / case_id
            case_dir.mkdir(parents=True)
            hashes = {}
            for name, model in (("input", case_input), ("truth", truth)):
                raw = _json_bytes(model.model_dump(mode="json"))
                (case_dir / f"{name}.json").write_bytes(raw)
                hashes[f"{name}_sha256"] = hashlib.sha256(raw).hexdigest()
            cases.append({"case_id": case_id, "case_type": kind, "case_seed": case_seed,
                          "axis": axis, "sign": sign, "event_count": len(truth.events), **hashes})
            truths.append((kind, truth))
        summary = _summary(truths)
        if summary["groups"] != {"normal": 30, "single": 120, "multi": 150} or any(
            count != 4 for count in summary["single_strata"].values()
        ):
            raise ValueError("pilot allocation mismatch")
        manifest = {"pilot_id": PILOT_ID, "generator_version": GENERATOR_VERSION,
                    "seed": seed, "numpy_version": np.__version__,
                    "source_sha256": _source_hashes(), "cases": cases}
        (temporary / "manifest.json").write_bytes(_json_bytes(manifest))
        (temporary / "summary.json").write_bytes(_json_bytes(summary))
        temporary.rename(directory)
        return directory
    except Exception:
        import shutil

        shutil.rmtree(temporary)
        raise
