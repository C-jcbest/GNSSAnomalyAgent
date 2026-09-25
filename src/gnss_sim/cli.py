from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import uvicorn

from gnss_sim.api import create_app
from gnss_sim.evaluation import evaluate_pilot, load_results_jsonl
from gnss_sim.pilot import DEFAULT_SEED, generate_pilot
from gnss_sim.schemas import GenerationRequest
from gnss_sim.storage import DatasetStore


def main() -> None:
    parser = argparse.ArgumentParser(prog="gnss-sim", description="Synthetic daily GNSS lab")
    commands = parser.add_subparsers(dest="command", required=True)
    generate = commands.add_parser("generate", help="Generate P2 events or P3 scenarios")
    generate.add_argument("--seed", type=int, default=20260923)
    generate.add_argument("--count", type=int, default=20)
    generate.add_argument(
        "--case-type",
        choices=("all", "all_scenarios", "normal", "spike", "step", "slow_trend", "acceleration", "transient_shift",
                 "multi_spike", "change_with_local", "temporary_with_local", "longterm_with_local",
                 "longterm_with_change", "complex_multiaxis"),
        default="all",
    )
    generate.add_argument("--data-dir", type=Path)
    pilot = commands.add_parser("pilot", help="Create or verify fixed P4 pilot-v1")
    pilot.add_argument("--seed", type=int, default=DEFAULT_SEED)
    pilot.add_argument("--data-dir", type=Path, default=Path("data/pilots"))
    evaluate = commands.add_parser("evaluate", help="Evaluate JSONL predictions on pilot-v1")
    evaluate.add_argument("--predictions", type=Path, required=True)
    evaluate.add_argument("--method", required=True)
    evaluate.add_argument("--pilot-dir", type=Path, default=Path("data/pilots/pilot-v1"))
    evaluate.add_argument("--out", type=Path)
    serve = commands.add_parser("serve", help="Serve the local experiment API and built UI")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8765)
    serve.add_argument("--data-dir", type=Path)
    args = parser.parse_args()
    if args.command == "pilot":
        directory = generate_pilot(args.data_dir, args.seed)
        print(f"{directory.resolve()}: 300 verified cases")
    elif args.command == "evaluate":
        results, errors = load_results_jsonl(args.predictions)
        report = evaluate_pilot(args.pilot_dir, results, args.method)
        report["invalid_result_lines"] = errors
        output = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(output, encoding="utf-8")
        else:
            print(output)
    elif args.command == "generate":
        root = args.data_dir or Path(os.environ.get("GNSS_SIM_DATA_DIR", "data/generated"))
        request = GenerationRequest(seed=args.seed, count=args.count, case_type=args.case_type)
        result = DatasetStore(root).generate_sync(request)
        if result.status != "complete":
            parser.exit(1, f"Generation failed: {result.error}\n")
        print(f"{result.dataset_id}: {result.generated_cases} cases in {root.resolve()}")
    else:
        root = args.data_dir or Path(os.environ.get("GNSS_SIM_DATA_DIR", "data/generated"))
        uvicorn.run(create_app(root), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
