#!/usr/bin/env python3
"""
csv2json.py — Minimal CSV → JSON/NDJSON converter for radarlink.

Examples:
  NDJSON (streaming):
    python3 src/radarlink/tools/csv2json.py data/sample_tracks.csv -o data/sample_tracks.ndjson
  JSON array (pretty):
    python3 src/radarlink/tools/csv2json.py data/sample_tracks.csv --format array -o data/sample_tracks.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

# Fields we will read if present (others ignored)
WHITELIST = [
    # time
    "timestamp_ms", "t",
    # polar
    "range", "range_mean",
    "azimuth", "azimuth_mean",
    "elevation", "elevation_mean",
    # velocity (true Doppler)
    "velocity",
    # SNR
    "snr_db", "snr_mean_db",
    # optional metadata
    "x_mean", "y_mean", "z_mean",
    "class_name"
]

def _to_number(s: str | None) -> Any:
    if s is None: return None
    s = s.strip()
    if s == "" or s.lower() == "nan": return None
    try:
        if "." not in s and "e" not in s.lower():
            return int(s)
    except ValueError:
        pass
    try:
        f = float(s)
        if math.isfinite(f) and abs(f - int(f)) < 1e-9:
            return int(f)
        return f
    except ValueError:
        return s

def _normalize(rec: Dict[str, Any]) -> Dict[str, Any]:
    """Map *_mean → canonical key, rename velocity and snr, keep timestamp."""
    out: Dict[str, Any] = {}

    # timestamp: prefer explicit; else map 't'→timestamp_ms
    if "timestamp_ms" in rec and rec["timestamp_ms"] is not None:
        out["timestamp_ms"] = rec["timestamp_ms"]
    elif "t" in rec and rec["t"] is not None:
        out["timestamp_ms"] = rec["t"]

    # polar → canonical
    if rec.get("range") is not None or rec.get("range_mean") is not None:
        out["range"] = rec.get("range", rec.get("range_mean"))
    if rec.get("azimuth") is not None or rec.get("azimuth_mean") is not None:
        out["azimuth"] = rec.get("azimuth", rec.get("azimuth_mean"))
    if rec.get("elevation") is not None or rec.get("elevation_mean") is not None:
        out["elevation"] = rec.get("elevation", rec.get("elevation_mean"))

    # velocity → radial_velocity_mps (true Doppler from dataset)
    if rec.get("velocity") is not None:
        out["radial_velocity_mps"] = rec["velocity"]

    # snr: prefer snr_db, else snr_mean_db
    if rec.get("snr_db") is not None or rec.get("snr_mean_db") is not None:
        out["snr_db"] = rec.get("snr_db", rec.get("snr_mean_db"))

    # optional metadata (kept but NOT required downstream)
    for k in ("x_mean", "y_mean", "z_mean", "class_name"):
        if rec.get(k) is not None:
            out[k] = rec[k]

    return out

def iter_records(csv_path: Path) -> Iterable[Dict[str, Any]]:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise SystemExit("CSV needs a header row.")

        for row in reader:
            raw: Dict[str, Any] = {}
            for k in WHITELIST:
                if k in row and row[k] not in (None, ""):
                    raw[k] = _to_number(row[k])

            rec = _normalize(raw)

            # Emit only if we have enough to be useful:
            # either polar (range+azimuth) OR (x_mean & y_mean) present
            has_polar = ("range" in rec and "azimuth" in rec)
            has_xy    = ("x_mean" in rec and "y_mean" in rec)
            if has_polar or has_xy or ("timestamp_ms" in rec):
                yield rec

def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Convert CSV to NDJSON or JSON array (radarlink).")
    p.add_argument("csv", type=Path, help="Input CSV (with header).")
    p.add_argument("-o", "--out", type=Path, help="Output file (default: stdout).")
    p.add_argument("--format", choices=["ndjson", "array"], default="ndjson")
    ns = p.parse_args(argv)

    recs = iter_records(ns.csv)

    if ns.out:
        ns.out.parent.mkdir(parents=True, exist_ok=True)

    if ns.format == "ndjson":
        out = sys.stdout if not ns.out else ns.out.open("w", encoding="utf-8")
        try:
            for rec in recs:
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
        finally:
            if out is not sys.stdout:
                out.close()
        return 0

    data = list(recs)
    out = sys.stdout if not ns.out else ns.out.open("w", encoding="utf-8")
    try:
        json.dump(data, out, ensure_ascii=False, indent=2)
        out.write("\n")
    finally:
        if out is not sys.stdout:
            out.close()
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
