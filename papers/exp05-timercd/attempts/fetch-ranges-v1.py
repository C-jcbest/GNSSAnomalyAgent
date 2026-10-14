"""Fetch an immutable public checkpoint in bounded ranges and verify its official hash."""
from __future__ import annotations

import hashlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

from gnss_research.files import read_json, write_json


def main():
    evidence = Path("papers/exp05-timercd")
    acquisition = read_json(evidence / "acquisition.json")
    metadata = read_json(evidence / "official-weight-metadata.json")
    size = metadata["size"]
    url = "https://huggingface.co/thu-sail-lab/Time-RCD/resolve/" + acquisition["revision"]
    url += "/best_model/pretrain_checkpoint_best_multi.pth"
    chunks = Path("vendor/upstream/time-rcd/weights/ranges")
    chunks.mkdir(exist_ok=False)
    block_size = 8 * 1024 * 1024

    def fetch(start):
        end = min(size, start + block_size) - 1
        path = chunks / f"{start:012d}.part"
        with httpx.Client(timeout=60, follow_redirects=True,
                          transport=httpx.HTTPTransport(retries=0)) as client:
            with client.stream("GET", url, headers={"Range": f"bytes={start}-{end}"}) as response:
                response.raise_for_status()
                if response.status_code != 206 or response.headers.get("content-range") != (
                        f"bytes {start}-{end}/{size}"):
                    raise ValueError("server did not return the requested immutable byte range")
                with path.open("xb") as output:
                    for block in response.iter_bytes():
                        output.write(block)
        if path.stat().st_size != end - start + 1:
            raise ValueError("incomplete byte range")
        print(f"complete range {start}-{end}", flush=True)
        return path

    with ThreadPoolExecutor(max_workers=8) as executor:
        parts = list(executor.map(fetch, range(0, size, block_size)))
    target = chunks.parent / "pretrain_checkpoint_best_multi_verified.pth"
    digest = hashlib.sha256()
    with target.open("xb") as output:
        for part in parts:
            with part.open("rb") as stream:
                while block := stream.read(1024 * 1024):
                    output.write(block)
                    digest.update(block)
    if target.stat().st_size != size or digest.hexdigest() != metadata["lfs"]["sha256"]:
        raise ValueError("checkpoint does not match the official LFS digest")
    receipt = {"url": url, "revision": acquisition["revision"], "bytes": size,
               "sha256": digest.hexdigest(), "official_lfs_hash_verified": True,
               "local_path": str(target), "transfer": "8 concurrent immutable HTTP ranges",
               "retry_count": 0}
    write_json(evidence / "checkpoint-ranges.json", receipt)
    print(receipt)


if __name__ == "__main__":
    main()
