"""Strict temporal contract for TAMA's published interval-string output."""
import re

ENTRY = re.compile(r"\((\d+)(?:,\s*(\d+))?\)/(\d+(?:\.\d+)?)/([a-z_]+)")


def parse_tama_intervals(value: str, length: int) -> list[dict]:
    if not isinstance(value, str):
        raise ValueError("TAMA interval output must be a string")
    if not value.startswith("[") or not value.endswith("]"):
        raise ValueError("TAMA interval output requires brackets")
    interior = value[1:-1].strip()
    if not interior:
        return []
    matches = list(ENTRY.finditer(interior))
    if not matches or ENTRY.sub("", interior).strip(" ,"):
        raise ValueError("Malformed TAMA interval string")
    intervals = []
    for match in matches:
        start = int(match[1])
        end = int(match[2]) if match[2] is not None else start
        if not 0 <= start <= end < length:
            raise ValueError(f"Interval [{start}, {end}] outside [0, {length - 1}]")
        intervals.append({"start": start, "end": end, "confidence": float(match[3]),
                          "type": match[4]})
    return intervals
