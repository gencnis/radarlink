# src/radarlink/tools/rel_vector_from_polar.py
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import List, Optional

import numpy as np

from radarlink.geo.loc_convert import (PolarConventions, RadarLocConverter,
                                       guess_az_unit_from_values)
from radarlink.rel_vec import compute  # HT = T - H


def bearing_deg_from_enu(e: float, n: float) -> float:
    return (math.degrees(math.atan2(e, n)) + 360.0) % 360.0

def parse_enutriple(s: str) -> np.ndarray:
    try:
        e, n, u = (float(x) for x in s.split(","))
        return np.array([e, n, u], dtype=float)
    except Exception:
        raise argparse.ArgumentTypeError("Expected 'E,N,U' (e.g. '100,200,10')")

def _env_bool(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "y", "on")

def _resolve_units(ns, sample_lines: List[str]):
    """Return (az_unit, el_unit, az_ref, az_cw, default_el). Env overrides; az_unit auto-detects if unspecified."""
    az_unit = ns.az_unit or os.environ.get("AZ_UNIT")
    el_unit = ns.el_unit or os.environ.get("EL_UNIT") or "deg"
    az_ref  = ns.az_ref  or os.environ.get("AZ_REF")  or "north"
    az_cw   = ns.az_cw if ns.az_cw is not None else _env_bool("AZ_CW", True)
    try:
        default_el = float(ns.default_el if ns.default_el is not None else os.environ.get("DEFAULT_ELEVATION", "0.0"))
    except Exception:
        default_el = 0.0

    if not az_unit:
        # auto-detect from sample_lines
        az_vals = []
        for line in sample_lines:
            try:
                o = json.loads(line)
            except Exception:
                continue
            v = o.get("azimuth", o.get("azimuth_mean", None))
            if v is None:
                continue
            try:
                az_vals.append(float(v))
            except Exception:
                pass
            if len(az_vals) >= 500:
                break
        az_unit = guess_az_unit_from_values(az_vals) if az_vals else "deg"
    return az_unit, el_unit, az_ref, az_cw, default_el

def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="NDJSON (range,az[,elevation]) -> HT relative vector (radar at 0,0,0)."
    )
    p.add_argument("--interceptor", type=parse_enutriple,
                   help='Interceptor ENU as "E,N,U" in meters (radar-origin frame).')
    p.add_argument("--az-unit", choices=["deg","rad"], default=None)
    p.add_argument("--el-unit", choices=["deg","rad"], default=None)
    p.add_argument("--default-el", type=float, default=None,
                   help="Elevation used when missing (in el-unit).")
    p.add_argument("--az-ref", choices=["north","east"], default=None,
                   help="Azimuth reference: 0° at north or east.")
    # tri-state for az_cw: True (default), or False if --az-ccw passed, or env if omitted
    g = p.add_mutually_exclusive_group()
    g.add_argument("--az-cw", dest="az_cw", action="store_true")
    g.add_argument("--az-ccw", dest="az_cw", action="store_false")
    p.set_defaults(az_cw=None)

    p.add_argument("--uav-only", action="store_true",
                   help="If present, drop rows where class_name != 'uav' (when class exists).")
    p.add_argument("--peek", type=int, default=1000,
                   help="How many lines to peek for azimuth unit auto-detection if needed.")
    ns = p.parse_args(argv)

    # Buffer the first N lines to allow auto-detection
    buffer: List[str] = []
    for _ in range(ns.peek):
        pos = sys.stdin.tell() if sys.stdin.seekable() else None
        line = sys.stdin.readline()
        if not line:
            break
        buffer.append(line)
        if pos is None:
            # not seekable; keep buffering
            continue

    # Resolve units (env overrides; auto-detect az if unspecified)
    az_unit, el_unit, az_ref, az_cw, default_el = _resolve_units(ns, buffer)

    conv = PolarConventions(
        az_ref=az_ref, az_cw=az_cw, az_unit=az_unit,
        el_positive_up=True, el_unit=el_unit, default_el=default_el
    )
    cvt = RadarLocConverter(conv)

    # Resolve interceptor vector H (ENU at radar)
    H: Optional[np.ndarray] = None
    if ns.interceptor is not None:
        H = ns.interceptor
    else:
        # try from ENV: either ENU triple or compute elsewhere (orchestrator usually does LLA->ENU)
        val = os.environ.get("INTERCEPTOR_ENU")
        if val:
            H = parse_enutriple(val)
    if H is None:
        print("ERROR: interceptor ENU is required (use --interceptor or INTERCEPTOR_ENU env).", file=sys.stderr)
        return 2

    def process_line(line: str):
        line = line.strip()
        if not line:
            return
        o = json.loads(line)

        if ns.uav_only and o.get("class_name") and o["class_name"] != "uav":
            return

        r = o.get("range") or o.get("range_mean")
        az = o.get("azimuth") or o.get("azimuth_mean")
        el = o.get("elevation", o.get("elevation_mean", None))
        if r is None or az is None:
            return

        x, y, z = cvt.polar_to_cart(float(r), float(az), None if el is None else float(el))
        T = np.array([x, y, z], dtype=float)
        HT = compute(T, H)
        e, n, u = float(HT[0]), float(HT[1]), float(HT[2])
        out = {
            "timestamp_ms": o.get("timestamp_ms"),
            "e": e, "n": n, "u": u,
            "bearing_deg": bearing_deg_from_enu(e, n),
            "slant_m": float(np.linalg.norm(HT)),
        }
        sys.stdout.write(json.dumps(out) + "\n")

    # Process buffered lines
    for line in buffer:
        process_line(line)
    # Process the rest
    for line in sys.stdin:
        process_line(line)
    sys.stdout.flush()
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
