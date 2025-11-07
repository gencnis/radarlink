#!/usr/bin/env python3
"""
Config reader for radar & interceptor positions via environment variables.

Supported env vars:
  RADAR_LAT, RADAR_LON, RADAR_ALT_M
  INTERCEPTOR_LAT, INTERCEPTOR_LON, INTERCEPTOR_ALT_M

All values are optional; conversion to float is validated.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple


def _get_float(name: str) -> Optional[float]:
    v = os.getenv(name)
    if v is None or v.strip() == "":
        return None
    try:
        return float(v)
    except ValueError:
        raise SystemExit(f"Invalid float for {name}: {v!r}")

def get_radar_position() -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Return (lat, lon, alt_m) for the radar."""
    return (
        _get_float("RADAR_LAT"),
        _get_float("RADAR_LON"),
        _get_float("RADAR_ALT_M"),
    )

def get_interceptor_position() -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Return (lat, lon, alt_m) for the interceptor."""
    return (
        _get_float("INTERCEPTOR_LAT"),
        _get_float("INTERCEPTOR_LON"),
        _get_float("INTERCEPTOR_ALT_M"),
    )
