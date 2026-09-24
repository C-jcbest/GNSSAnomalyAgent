from __future__ import annotations

import argparse
import os
from pathlib import Path

import uvicorn

from gnss_sim.api import create_app
from gnss_sim.schemas import GenerationRequest
from gnss_sim.storage import DatasetStore


def main() -> None:
    parser = argparse.ArgumentParser(prog="gnss-sim", description="Synthetic daily GNSS lab")
    commands = parser.add_subparsers(dest="command", required=True)
    generate = commands.add_parser("generate", help="Generate single-event P2 cases")
    generate.add_argument("--seed", type=int, default=20260923)
    generate.add_argument("--count", type=int, default=10)
    generate.add_argument(
        "--case-type",
        choices=("normal", "spike", "step", "slow_trend", "acceleration", "transient_shift"),
        default="normal",
    )
    generate.add_argument("--data-dir", type=Path)
    serve = commands.add_parser("serve", help="Serve the local experiment API and built UI")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8765)
    serve.add_argument("--data-dir", type=Path)
    args = parser.parse_args()
    root = args.data_dir or Path(os.environ.get("GNSS_SIM_DATA_DIR", "data/generated"))

    if args.command == "generate":
        request = GenerationRequest(seed=args.seed, count=args.count, case_type=args.case_type)
        result = DatasetStore(root).generate_sync(request)
        if result.status != "complete":
            parser.exit(1, f"Generation failed: {result.error}\n")
        print(f"{result.dataset_id}: {result.generated_cases} cases in {root.resolve()}")
    else:
        uvicorn.run(create_app(root), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
