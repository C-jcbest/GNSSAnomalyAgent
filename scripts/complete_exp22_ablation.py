"""Complete the fixed seven-station image ablation under a disclosed amendment."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from gnss_research.files import read_json, sha256, write_json
from gnss_research.vision import call_visual, credentials


def main():
    visual = Path("runs/exp22-image-ablation-v1")
    frozen = read_json(visual / "registration.json")
    if not (visual / "pilot-completed.json").exists():
        raise ValueError("pilot must finish before amended completion")
    sentinel = visual / "remaining-started.json"
    if sentinel.exists():
        raise ValueError("remaining phase already started; no retries")
    amendment = Path("docs/experiment-exp22-completion-amendment.md")
    selected = [job for job in frozen["pending"] if job["case_id"] not in frozen["pilot_ids"]]
    write_json(visual / "completion-amendment-registration.json", {
        "amendment_sha256": sha256(amendment), "script_sha256": sha256(Path(__file__)),
        "original_registration_sha256": sha256(visual / "registration.json"),
        "pilot_summary_sha256": sha256(visual / "pilot-summary.json"),
        "original_expand_passed": False, "requests": len(selected), "jobs": selected})
    write_json(sentinel, {"requests": len(selected), "retry_count": 0,
                          "authority": "disclosed fixed-population contribution check amendment"})
    lookup = {case["id"]: case for case in frozen["triggered"]}
    base, key, _ = credentials(Path(".env"))

    def request(job):
        case = lookup[job["case_id"]]
        observation = read_json(Path(case["dataset"]) / case["record"]["observation"])
        work = visual / "predictions" / job["case_id"] / job["mode"]
        if sha256(work / "request.json") != job["request_sha256"]:
            raise ValueError("original registered input changed")
        result = call_visual(observation, "pair_focus", work, (base, key, job["model"]))
        print(job["case_id"], job["mode"], result["status"],
              [(event["kind"], event["start"], event["end"])
               for event in result.get("prediction", {}).get("events", [])], flush=True)

    with ThreadPoolExecutor(max_workers=3) as executor:
        list(executor.map(request, selected))
    write_json(visual / "remaining-completed.json", {"requests": len(selected), "retry_count": 0})


if __name__ == "__main__":
    main()
