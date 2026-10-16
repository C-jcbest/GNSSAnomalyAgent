"""Locate and preserve an immutable TimeRLM code tree, without running models."""

from __future__ import annotations

import argparse
from pathlib import Path

import httpx

from gnss_sim.artifacts import sha, write_json


def retrieve(out):
    out.mkdir(parents=True, exist_ok=False)
    repository = "OpenTSLM/TimeRLM"
    with httpx.Client(timeout=30, follow_redirects=True) as client:
        metadata = client.get("https://api.github.com/repos/" + repository)
        (out / "repository.json").write_bytes(metadata.content)
        metadata.raise_for_status()
        branch = metadata.json()["default_branch"]
        response = client.get(f"https://api.github.com/repos/{repository}/commits/{branch}")
        (out / "commit.json").write_bytes(response.content)
        response.raise_for_status()
        commit = response.json()["sha"]
        response = client.get(f"https://api.github.com/repos/{repository}/git/trees/{commit}", params={"recursive": "1"})
        (out / "tree.json").write_bytes(response.content)
        response.raise_for_status()
        paths = [item["path"] for item in response.json()["tree"] if item["type"] == "blob"]
        if response.json().get("truncated"):
            raise ValueError("Cannot describe a truncated source tree as complete")
        candidates = [path for path in paths if path.endswith(".py") and
                      any(term in path.lower() for term in ("environment", "repl", "plot", "rlm", "agent"))]
        write_json(out / "retrieval.json", dict(
            repository=repository, default_branch=branch, commit=commit, candidates=candidates,
            author_model_run=False, source_files_read=[],
        ))
    (out / "source.py").write_bytes(Path(__file__).read_bytes())
    write_json(out / "freeze.json", {path.name: sha(path) for path in out.iterdir()})
    print(dict(commit=commit, candidates=candidates[:35], candidate_count=len(candidates)), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    retrieve(parser.parse_args().out.resolve())
