"""Bounded multi-station snapshot collection; reusable daily responses enable resumption."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from .acquisition import Platform
from .storage import digest, read_json, write_json


def collect(stations_file: Path, out: Path, start, end, workers=4, resume=False):
    if not 1 <= workers <= 4:
        raise ValueError("workers must be 1–4")
    stations = read_json(stations_file)
    if not stations or len({r["StationUUID"] for r in stations}) != len(stations):
        raise ValueError("empty or duplicate station list")
    identity = digest({"stations": stations, "start": start.isoformat(), "end": end.isoformat()})
    plan_path = out / "collection-plan.json"
    if out.exists():
        if not resume or read_json(plan_path)["collection_id"] != identity:
            raise ValueError("collection exists or query changed; choose new directory or resume")
    else:
        write_json(
            plan_path,
            {
                "collection_id": identity,
                "start": start.isoformat(),
                "end_exclusive": end.isoformat(),
                "stations": len(stations),
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
        )
    platform = Platform()
    platform.login()
    report = {
        "collection_id": identity,
        "expected_stations": len(stations),
        "results": [],
        "all_finished": False,
    }

    def fetch(station):
        alias = "S-" + digest(station["StationUUID"])[:8]

        def progress(days):
            if days % 100 == 0:
                print(f"{alias}: {days} 天已缓存", flush=True)

        try:
            audit = platform.fetch(
                station, start, end, out / alias, alias, resume=resume, progress=progress
            )
            return {"alias": alias, "status": "complete", "audit": audit}
        except (ValueError, RuntimeError, OSError, KeyError) as exc:
            return {"alias": alias, "status": "error", "reason": str(exc)[:300]}

    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(fetch, s) for s in stations]
            for future in as_completed(futures):
                result = future.result()
                report["results"].append(result)
                write_json(out / "collection-report.json", report)
                print(
                    f"{result['alias']}: {result['status']} ({len(report['results'])}/{len(stations)})",
                    flush=True,
                )
        report["all_finished"] = True
        report["all_successful"] = all(r["status"] == "complete" for r in report["results"])
        write_json(out / "collection-report.json", report)
        return report
    finally:
        platform.close()
