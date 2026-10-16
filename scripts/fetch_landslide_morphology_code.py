"""Bind author repository metadata to exact commits before targeted source reading."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

from gnss_sim.artifacts import sha, write_json

REPOSITORIES = ("CSTCloudOps/TShape", "PenyChen/VAN-AD")


def retrieve_repository(repository, out):
    folder = out / repository.split("/")[-1]
    folder.mkdir()
    with httpx.Client(timeout=30, follow_redirects=True) as client:
        metadata = client.get("https://api.github.com/repos/" + repository)
        metadata.raise_for_status()
        info = metadata.json()
        commit = client.get(
            "https://api.github.com/repos/" + repository + "/commits/" + info["default_branch"]
        )
        commit.raise_for_status()
        revision = commit.json()["sha"]
        tree = client.get(
            "https://api.github.com/repos/" + repository + "/git/trees/" + revision,
            params={"recursive": "1"},
        )
        tree.raise_for_status()
        write_json(folder / "metadata.json", info)
        write_json(folder / "commit.json", commit.json())
        write_json(folder / "tree.json", tree.json())
        readme_paths = [
            row["path"]
            for row in tree.json()["tree"]
            if row["type"] == "blob" and row["path"].lower() == "readme.md"
        ]
        ledger = []
        for relative in readme_paths:
            url = f"https://raw.githubusercontent.com/{repository}/{revision}/{relative}"
            response = client.get(url)
            response.raise_for_status()
            (folder / relative).write_bytes(response.content)
            ledger.append(dict(path=relative, url=url, sha256=sha(folder / relative)))
        write_json(folder / "fetch-ledger.json", ledger)
        return dict(repository=repository, commit=revision, truncated=tree.json()["truncated"])


def retrieve(out):
    out.mkdir(parents=True, exist_ok=False)
    (out / "source.py").write_bytes(Path(__file__).read_bytes())
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda name: retrieve_repository(name, out), REPOSITORIES))
    write_json(out / "repositories.json", results)
    # Further selected source fetches are a separate frozen artifact.
    write_json(
        out / "freeze.json",
        {str(p.relative_to(out)): sha(p) for p in out.rglob("*") if p.is_file()},
    )
    print(results, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    retrieve(parser.parse_args().out.resolve())
