#!/usr/bin/env python3
"""
control_hub — Orchestrator core, usable both as:
  1) a callable module:   from radarlink.tools.control_hub import ControlHubRuntime
  2) a CLI stream tool:   python3 -m radarlink.tools.control_hub < in.ndjson > out.ndjson

What it does per input object (radar JSON):
- Builds radar-centric ENU frame (radar at 0,0,0) from .env
- Gets interceptor ENU from .env (INTERCEPTOR_ENU or LLA via LocalFrame)
- Converts radar polar (range, az[, el]) → target ENU
- Computes relative vector HT = T - H
- Emits bearing_deg (0°=North, CW) and elevation_deg (above horizon)
- Adds radar_id, track_uuid, and echoes radar_data block
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import uuid
from typing import Iterable, List, Optional, Tuple

import numpy as np

from radarlink.geo.loc_convert import (PolarConventions, RadarLocConverter,
                                       guess_az_unit_from_values)
from radarlink.geo_frame import LocalFrame  # LLA -> ENU@radar
from radarlink.rel_vec import compute  # HT = T - H

# ---------- small helpers ----------

def _env_float(name: str) -> float:
    try:
        return float(os.environ[name])
    except KeyError:
        raise RuntimeError(f"Missing required env: {name}")

def _env_opt(name: str) -> Optional[str]:
    v = os.environ.get(name)
    return v if v not in (None, "") else None

def _parse_enutriple(s: str) -> np.ndarray:
    e, n, u = (float(x) for x in s.split(","))
    return np.array([e, n, u], dtype=float)

def _bearing_deg(e: float, n: float) -> float:
    return (math.degrees(math.atan2(e, n)) + 360.0) % 360.0  # 0=N, CW

def _elev_deg(e: float, n: float, u: float) -> float:
    ground = math.hypot(e, n)
    return math.degrees(math.atan2(u, ground))               # +up

def _pick(*vals):
    for v in vals:
        if v is not None:
            return v
    return None

def _stable_radar_id() -> str:
    rlat = _env_float("RADAR_LAT")
    rlon = _env_float("RADAR_LON")
    ralt = _env_float("RADAR_ALT_M")
    name = f"{rlat:.7f},{rlon:.7f},{ralt:.3f}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"radarlink:radar:{name}"))


# ---------- the orchestrator runtime (callable) ----------

class ControlHubRuntime:
    """
    Construct once, then call .process(o: dict) -> dict for each radar sample.
    """

    def __init__(self, radar_id: str, frame: LocalFrame, H_enu: np.ndarray,
                 converter: RadarLocConverter):
        self.radar_id = radar_id
        self.frame = frame        # not used yet beyond init; kept for future LLA usage
        self.H = H_enu.astype(float)
        self.cvt = converter

    @classmethod
    def from_env(cls,
                 az_unit: Optional[str] = None,
                 az_ref: Optional[str] = None,
                 az_cw: Optional[bool] = None,
                 el_unit: Optional[str] = None,
                 default_el: Optional[float] = None) -> "ControlHubRuntime":
        """Builds runtime from environment (.env)."""
        # Build ENU at radar
        rlat = _env_float("RADAR_LAT")
        rlon = _env_float("RADAR_LON")
        ralt = _env_float("RADAR_ALT_M")
        frame = LocalFrame(rlat, rlon, ralt)

        # Interceptor ENU
        enu_str = _env_opt("INTERCEPTOR_ENU")
        if enu_str:
            H = _parse_enutriple(enu_str)
        else:
            ilat = _env_float("INTERCEPTOR_LAT")
            ilon = _env_float("INTERCEPTOR_LON")
            ialt = _env_float("INTERCEPTOR_ALT_M")
            H = frame.geodetic_to_enu(ilat, ilon, ialt)

        # Conventions (prefer explicit args, else env, else defaults)
        if az_unit is None:
            az_unit = _env_opt("AZ_UNIT") or "deg"
        if az_ref is None:
            az_ref = (_env_opt("AZ_REF") or "north").lower()
        if az_cw is None:
            az_cw = (_env_opt("AZ_CW") or "true").strip().lower() in ("1","true","yes","y","on")
        if el_unit is None:
            el_unit = _env_opt("EL_UNIT") or "deg"
        if default_el is None:
            try:
                default_el = float(_env_opt("DEFAULT_ELEVATION") or "0.0")
            except Exception:
                default_el = 0.0

        conv = PolarConventions(
            az_ref=az_ref, az_cw=az_cw, az_unit=az_unit,
            el_positive_up=True, el_unit=el_unit, default_el=default_el
        )
        return cls(_stable_radar_id(), frame, H, RadarLocConverter(conv))

    def set_conventions_from_az_samples(self, az_samples: Iterable[float]) -> None:
        """Optional: auto-detect az_unit from a few values; keeps other settings."""
        az_unit = guess_az_unit_from_values(az_samples)
        c = self.cvt.conv
        self.cvt = RadarLocConverter(PolarConventions(
            az_ref=c.az_ref, az_cw=c.az_cw, az_unit=az_unit,
            el_positive_up=c.el_positive_up, el_unit=c.el_unit, default_el=c.default_el
        ))

    # --- core processing for one radar JSON object ---
    def process(self, o: dict) -> Optional[dict]:
        r = _pick(o.get("range"), o.get("range_mean"))
        az = _pick(o.get("azimuth"), o.get("azimuth_mean"))
        el = _pick(o.get("elevation"), o.get("elevation_mean"))

        if r is None or az is None:
            return None  # cannot compute without polar pair

        # radar polar -> target ENU @ radar
        x, y, z = self.cvt.polar_to_cart(float(r), float(az), None if el is None else float(el))
        T = np.array([x, y, z], dtype=float)

        # relative vector interceptor -> target
        HT = compute(T, self.H)
        e, n, u = (float(HT[0]), float(HT[1]), float(HT[2]))
        slant = float(np.linalg.norm(HT))
        bearing = _bearing_deg(e, n)
        elev = _elev_deg(e, n, u)

        # ids
        src_track = _pick(o.get("track_idx"), o.get("track_id"))
        if src_track is None:
            name = f"{self.radar_id}:{o.get('timestamp_ms')}:{r}:{az}:{el}"
            track_uuid = str(uuid.uuid5(uuid.NAMESPACE_URL, name))
        else:
            name = f"{self.radar_id}:{src_track}"
            track_uuid = str(uuid.uuid5(uuid.NAMESPACE_URL, name))

        return {
            "radar_id": self.radar_id,
            "timestamp_ms": o.get("timestamp_ms"),
            "target_class": o.get("class_name"),
            "track_id": src_track,
            "track_uuid": track_uuid,
            "radar_data": {
                "range": None if r is None else float(r),
                "azimuth": None if az is None else float(az),
                "elevation": None if el is None else (float(el)),
                "snr_db": o.get("snr_db"),
            },
            "computed": {
                "e": e, "n": n, "u": u,
                "bearing_deg": bearing,
                "elevation_deg": elev,
                "range_m": slant
            }
        }


# ---------- CLI stream mode (kept for testing/pipes) ----------

def _auto_detect_az_unit_from_lines(lines: List[str], cap: int = 500) -> str:
    vals = []
    for i, line in enumerate(lines[:cap]):
        try:
            o = json.loads(line)
            v = _pick(o.get("azimuth"), o.get("azimuth_mean"))
            if v is not None:
                vals.append(float(v))
        except Exception:
            pass
    return guess_az_unit_from_values(vals) if vals else "deg"

def main() -> int:
    ap = argparse.ArgumentParser(description="control_hub: NDJSON stdin → enriched NDJSON stdout")
    ap.add_argument("--peek", type=int, default=1000, help="lines to peek for azimuth auto-detect if AZ_UNIT unset")
    ns = ap.parse_args()

    # Build runtime from env (may use defaults)
    rt = ControlHubRuntime.from_env()

    # If AZ_UNIT not set in env, peek a bit and auto-detect
    if _env_opt("AZ_UNIT") is None:
        buffer: List[str] = []
        for _ in range(ns.peek):
            line = sys.stdin.readline()
            if not line:
                break
            buffer.append(line)
        detected = _auto_detect_az_unit_from_lines(buffer)
        rt.set_conventions_from_az_samples([json.loads(l).get("azimuth",
                                 json.loads(l).get("azimuth_mean")) for l in buffer if l.strip()])
        # Process buffered, then rest
        for line in buffer:
            if not line.strip():
                continue
            o = json.loads(line)
            out = rt.process(o)
            if out is not None:
                sys.stdout.write(json.dumps(out) + "\n")
        for line in sys.stdin:
            if not line.strip():
                continue
            o = json.loads(line)
            out = rt.process(o)
            if out is not None:
                sys.stdout.write(json.dumps(out) + "\n")
        sys.stdout.flush()
        return 0

    # No auto-detect needed: straight stream
    for line in sys.stdin:
        if not line.strip():
            continue
        o = json.loads(line)
        out = ControlHubRuntime.from_env().process(o)  # simple, stateless build per run
        if out is not None:
            sys.stdout.write(json.dumps(out) + "\n")
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
