# src/radarlink/geo/loc_convert.py
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Optional, Tuple


@dataclass(frozen=True)
class PolarConventions:
    az_ref: str = "north"      # "north" or "east"
    az_cw: bool = True         # True=CW, False=CCW
    az_unit: str = "deg"       # "deg" or "rad"
    el_positive_up: bool = True
    el_unit: str = "deg"       # "deg" or "rad"
    default_el: float = 0.0

class RadarLocConverter:
    def __init__(self, conv: PolarConventions | None = None):
        self.conv = conv or PolarConventions()

    @staticmethod
    def _deg2rad(v: float) -> float:
        return v * math.pi / 180.0

    def _map_az(self, az: float) -> float:
        θ = self._deg2rad(az) if self.conv.az_unit == "deg" else float(az)
        if self.conv.az_ref.lower() == "east":
            θ -= math.pi / 2.0
        if not self.conv.az_cw:
            θ = -θ
        return θ

    def _map_el(self, el: Optional[float]) -> float:
        if el is None:
            el = self.conv.default_el
        φ = self._deg2rad(el) if self.conv.el_unit == "deg" else float(el)
        return φ if self.conv.el_positive_up else -φ

    def polar_to_cart(
        self, range_m: float, azimuth: float, elevation: Optional[float] = None
    ) -> Tuple[float, float, float]:
        r = float(range_m)
        θ = self._map_az(azimuth)
        φ = self._map_el(elevation)
        r_h = r * math.cos(φ)
        x = r_h * math.sin(θ)   # East
        y = r_h * math.cos(θ)   # North
        z = r * math.sin(φ)     # Up
        return x, y, z

# -------- helpers you can import --------

def guess_az_unit_from_values(values: Iterable[float]) -> str:
    """
    Return 'rad' if max|value| < ~10, else 'deg'.
    Robust for typical radar data where radians are within [-π, +π] or [0, 2π].
    """
    vmax = 0.0
    for v in values:
        try:
            vv = abs(float(v))
            if vv > vmax:
                vmax = vv
        except Exception:
            continue
    return "rad" if vmax < 10.0 else "deg"
