"""Locate immutable author code for nuisance-invariance nearest-neighbor checks."""

from __future__ import annotations

import argparse
from pathlib import Path

import httpx

from gnss_sim.artifacts import sha, write_json


def retrieve(out):
    out.mkdir(parents=True, exist_ok=False)
    (out / "source.py").write_bytes(Path(__file__).read_bytes())
    repository = "JxdEngineer/SSRL"
    result = dict(repository=repository, author_model_run=False, source_files_read=[])
    with httpx.Client(timeout=30, follow_redirects=True) as client:
        try:
            response = client.get("https://api.github.com/repos/" + repository)
            (out / "metadata.json").write_bytes(response.content)
            response.raise_for_status()
            branch = response.json()["default_branch"]
            response = client.get(f"https://api.github.com/repos/{repository}/commits/{branch}")
            (out / "commit.json").write_bytes(response.content)
            response.raise_for_status()
            revision = response.json()["sha"]
            response = client.get(f"https://api.github.com/repos/{repository}/git/trees/{revision}", params={"recursive": "1"})
            (out / "tree.json").write_bytes(response.content)
            response.raise_for_status()
            if response.json().get("truncated"):
                raise ValueError("A truncated tree is not a complete author code index")
            result.update(commit=revision, paths=[row["path"] for row in response.json()["tree"]
                                                 if row["type"] == "blob" and row["path"].endswith((".py", ".md"))])
        except (httpx.HTTPError, ValueError) as error:
            result["error"] = str(error)
    write_json(out / "retrieval.json", result)
    write_json(out / "freeze.json", {path.name: sha(path) for path in out.iterdir()})
    print(result, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    retrieve(parser.parse_args().out.resolve())
