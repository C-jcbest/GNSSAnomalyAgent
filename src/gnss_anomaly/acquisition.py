"""Explicit online collection; none of these functions are imported by inference."""

import math
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import httpx
import pandas as pd
from dotenv import dotenv_values

from .storage import digest, file_hash, read_json, verify_files, write_json


def load_environment(path: Path | None = None):
    if path:
        if not path.is_file():
            raise FileNotFoundError("environment file does not exist")
        for key, value in dotenv_values(path).items():
            if value is not None:
                os.environ.setdefault(key.upper(), value)


def local_time(value: str) -> datetime:
    result = datetime.fromisoformat(value)
    if result.tzinfo is None:
        result = result.replace(tzinfo=ZoneInfo("Asia/Shanghai"))
    return result


def redact(value):
    if isinstance(value, dict):
        return {
            k: redact(v)
            for k, v in value.items()
            if not any(s in k.lower() for s in ("password", "session", "token", "secret"))
        }
    if isinstance(value, list):
        return [redact(v) for v in value]
    return value


class Platform:
    def __init__(self, client: httpx.Client | None = None):
        self.base = os.getenv("BEIDOU_API_BASE_URL", "").rstrip("/")
        self.username = os.getenv("BEIDOU_USERNAME", "")
        self.password = os.getenv("BEIDOU_PASSWORD", "")
        if not all((self.base, self.username, self.password)):
            raise ValueError("missing BEIDOU_API_BASE_URL / BEIDOU_USERNAME / BEIDOU_PASSWORD")
        self.client = client or httpx.Client(timeout=30, follow_redirects=False)
        self.session = None

    def close(self):
        self.client.close()

    def post(self, path: str, payload: dict) -> dict:
        try:
            response = self.client.post(f"{self.base}/{path}", json=payload)
            response.raise_for_status()
            body = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            # Do not log URLs, request payloads or authentication responses.
            raise RuntimeError(f"platform request failed ({type(exc).__name__})") from None
        if not isinstance(body, dict) or str(body.get("ResponseCode")) != "200":
            raise RuntimeError("platform business response failed or ResponseCode missing")
        return body

    def login(self):
        if not self.session:
            body = self.post(
                "UserLogin/doLogin.php", {"Username": self.username, "Password": self.password}
            )
            self.session = body.get("SessionUUID")
            if not isinstance(self.session, str) or not self.session:
                raise RuntimeError("platform returned no session")

    def stations(self) -> list[dict]:
        self.login()
        body = self.post(
            "Station/getStationListInfo.php",
            {
                "SessionUUID": self.session,
                "PageInfo": {"PageFlag": "StationNameAsc", "PageNumber": 1, "PageSize": -1},
            },
        )
        rows = body.get("StationList")
        if not isinstance(rows, list) or not all(isinstance(r, dict) for r in rows):
            raise ValueError("invalid station list")
        return rows

    def fetch(
        self,
        station: dict,
        start: datetime,
        end: datetime,
        out: Path,
        alias: str,
        site: str | None = None,
        resume: bool = False,
        progress=None,
    ):
        if start.tzinfo is None or end.tzinfo is None:
            raise ValueError("collection timestamps must include timezone")
        start = start.astimezone(ZoneInfo("Asia/Shanghai"))
        end = end.astimezone(ZoneInfo("Asia/Shanghai"))
        if out.exists() and not resume:
            raise FileExistsError("snapshot already exists; choose a new output directory")
        if start >= end or any(t.minute or t.second or t.microsecond for t in (start, end)):
            raise ValueError("range must be increasing and aligned to whole hours")
        reference = [float(station[k]) for k in ("StationN0", "StationE0", "StationU0")]
        if not all(math.isfinite(v) for v in reference):
            raise ValueError("invalid station reference")
        query_id = digest(
            {
                "station": station["StationUUID"],
                "start": start.isoformat(),
                "end": end.isoformat(),
                "reference": reference,
                "alias": alias,
                "site": site,
            }
        )
        checkpoint_path = out / "checkpoint.json"
        if out.exists():
            checkpoint = read_json(checkpoint_path)
            if checkpoint.get("query_sha256") != query_id:
                raise ValueError("snapshot query changed; cannot resume")
            if (out / "manifest.json").exists():
                manifest = read_json(out / "manifest.json")
                if manifest.get("query_sha256") != query_id:
                    raise ValueError("completed snapshot query mismatch")
                verify_files(out, manifest["files"])
                return manifest["audit"]
        else:
            out.mkdir(parents=True)
            checkpoint = {"query_sha256": query_id, "files": {}, "completed_days": 0, "retries": 0}
            write_json(checkpoint_path, checkpoint)
        self.login()
        records, files = [], dict(checkpoint["files"])
        cursor = start
        count = 0
        while cursor < end:
            upper = min(cursor + timedelta(days=1), end)
            raw_path = out / "raw" / f"{count:04d}.json"
            relative = raw_path.relative_to(out).as_posix()
            if relative in files:
                if file_hash(raw_path) != files[relative]:
                    raise ValueError("cached daily response checksum mismatch")
                body = read_json(raw_path)
            else:
                for attempt in range(3):
                    try:
                        body = self.post(
                            "GNSSData/getDailyGNSSDataInfo.php",
                            {
                                "SessionUUID": self.session,
                                "StationUUID": station["StationUUID"],
                                "BeginTime": cursor.strftime("%Y-%m-%d %H:%M:%S"),
                                "EndTime": upper.strftime("%Y-%m-%d %H:%M:%S"),
                            },
                        )
                        break
                    except RuntimeError:
                        checkpoint["retries"] += 1
                        write_json(checkpoint_path, checkpoint)
                        if attempt == 2:
                            raise
                        time.sleep(2**attempt)
            rows = body.get("Data")
            if not isinstance(rows, list):
                raise ValueError("platform Data is not a list")
            if relative not in files:
                write_json(raw_path, redact(body))
                files[relative] = file_hash(raw_path)
                checkpoint.update(files=files, completed_days=count + 1)
                write_json(checkpoint_path, checkpoint)
            records.extend(rows)
            cursor, count = upper, count + 1
            if progress:
                progress(count)
        frame, audit = normalize(records, reference, start, end)
        path = out / "observations.csv"
        frame.to_csv(path, index=False)
        files[path.name] = file_hash(path)
        write_json(
            out / "manifest.json",
            {
                "schema_version": 1,
                "kind": "platform_snapshot",
                "query_sha256": query_id,
                "station_alias": alias,
                "site_alias": site,
                "source_timezone": "Asia/Shanghai",
                "start": start.isoformat(),
                "end_exclusive": end.isoformat(),
                "created_at": datetime.now(timezone.utc).isoformat(),
                "files": files,
                "reference_m": reference,
                "unit_assumption": "platform coordinates in metres",
                "units_verified": False,
                "reference_id": "station_initial_coordinates",
                "audit": audit,
                "requests": count,
                "retry_attempts": checkpoint["retries"],
                "completeness": "daily queries; upstream truncation contract not verified",
            },
        )
        return audit


def normalize(records: list[dict], reference: list[float], start: datetime, end: datetime):
    rows: dict[datetime, list[float | None]] = {}
    duplicates = outside = 0
    for row in records:
        timestamp = local_time(str(row.get("DataTime") or row.get("data_time") or row.get("Time")))
        timestamp = timestamp.astimezone(timezone.utc)
        if not start <= timestamp < end:
            outside += 1
            continue
        if timestamp.minute or timestamp.second or timestamp.microsecond:
            raise ValueError("off-grid observations: inspect platform sampling before resampling")
        values = []
        for names in (
            ("PJKInfoN", "N", "NData"),
            ("PJKInfoE", "E", "EData"),
            ("PJKInfoU", "U", "UData"),
        ):
            value = next((row[k] for k in names if k in row), None)
            value = float(value) if value not in (None, "") else None
            if value is not None and not math.isfinite(value):
                value = None
            values.append(value)
        if timestamp in rows:
            if rows[timestamp] != values:
                raise ValueError("conflicting observations at the same timestamp")
            duplicates += 1
        rows[timestamp] = values
    index = pd.date_range(
        start.astimezone(timezone.utc), end.astimezone(timezone.utc), freq="h", inclusive="left"
    )
    frame = pd.DataFrame(index=index, columns=["n_m", "e_m", "u_m"], dtype=float)
    for t, values in rows.items():
        frame.loc[t] = values
    for axis, base in zip(("n", "e", "u"), reference):
        frame[f"{axis}_mm"] = (frame[f"{axis}_m"] - base) * 1000
    audit = {
        "raw_count": len(records),
        "observed_timestamps": len(rows),
        "expected_hours": len(frame),
        "duplicates": duplicates,
        "outside_range": outside,
        "missing_fraction": float(frame[["n_m", "e_m", "u_m"]].isna().to_numpy().mean()),
    }
    frame.index.name = "timestamp"
    return frame.reset_index(), audit
