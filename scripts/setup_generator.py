"""Install the pinned upstream sources in a dedicated environment."""
from __future__ import annotations

import hashlib
import json
import subprocess
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    lock = json.loads((ROOT / "configs/generator-sources.json").read_text())
    archive = ROOT / "papers/local/gutentag.zip"
    archive.parent.mkdir(parents=True, exist_ok=True)
    if not archive.exists():
        with urllib.request.urlopen(lock["gutentag"]["archive_url"], timeout=120) as response:
            content = response.read()
        if hashlib.sha256(content).hexdigest() != lock["gutentag"]["archive_sha256"]:
            raise RuntimeError("Downloaded GutenTAG archive hash mismatch")
        with archive.open("xb") as output:
            output.write(content)
    if hashlib.sha256(archive.read_bytes()).hexdigest() != lock["gutentag"]["archive_sha256"]:
        raise RuntimeError("Existing GutenTAG archive hash mismatch; refusing overwrite")
    environment = ROOT / ".venv-generator"
    if not environment.exists():
        subprocess.run(["uv", "venv", "--python", "3.11.11", str(environment)], check=True)
    python = environment / "Scripts/python.exe"
    if not python.exists():
        python = environment / "bin/python"
    subprocess.run(["uv", "pip", "sync", "--python", str(python),
                    str(ROOT / "configs/generator-requirements.txt")], check=True)
    subprocess.run(["uv", "pip", "install", "--python", str(python), "--no-deps", str(archive)],
                   check=True)
    subprocess.run([str(python), str(ROOT / "scripts/generate_worker.py"), "--verify"],
                   check=True)


if __name__ == "__main__":
    main()
