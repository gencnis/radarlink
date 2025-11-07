from __future__ import annotations

import math
from typing import Tuple

import numpy as np

# --- WGS-84 constants (internal) ---
_WGS84_A = 6378137.0
_WGS84_F = 1.0 / 298.257223563
_WGS84_E2 = _WGS84_F * (2.0 - _WGS84_F)

def _geodetic_to_ecef(lat_deg: float, lon_deg: float, alt_m: float) -> Tuple[float, float, float]:
    lat = math.radians(lat_deg); lon = math.radians(lon_deg)
    sL, cL = math.sin(lat), math.cos(lat)
    sO, cO = math.sin(lon), math.cos(lon)
    N = _WGS84_A / math.sqrt(1.0 - _WGS84_E2 * sL * sL)
    x = (N + alt_m) * cL * cO
    y = (N + alt_m) * cL * sO
    z = (N * (1.0 - _WGS84_E2) + alt_m) * sL
    return x, y, z

def _enu_rot_rows(lat0_deg: float, lon0_deg: float) -> np.ndarray:
    """
    Returns 3x3 matrix whose rows are ENU unit vectors expressed in ECEF.
    ENU = R * (ECEF - ECEF0)
    """
    lat0 = math.radians(lat0_deg); lon0 = math.radians(lon0_deg)
    sL, cL = math.sin(lat0), math.cos(lat0)
    sO, cO = math.sin(lon0), math.cos(lon0)
    return np.array([
        [-sO,           cO,          0.0],   # East
        [-sL*cO,       -sL*sO,       cL ],   # North
        [ cL*cO,        cL*sO,       sL ],   # Up
    ], dtype=np.float64)

class LocalFrame:
    """
    Minimal local ENU frame anchored at the radar (lat0, lon0, alt0).
    Public API (only what we need now):
      - geodetic_to_enu(lat, lon, alt) -> [E, N, U]
      - radar_polar_to_enu(range_m, az_deg, el_deg=0.0) -> [E, N, U]
        (az: North=0°, clockwise; el: +Up)
    """
    __slots__ = ("ecef0", "R_ecef_to_enu")

    def __init__(self, lat0_deg: float, lon0_deg: float, alt0_m: float):
        x0, y0, z0 = _geodetic_to_ecef(lat0_deg, lon0_deg, alt0_m)
        self.ecef0 = np.array([x0, y0, z0], dtype=np.float64)
        self.R_ecef_to_enu = _enu_rot_rows(lat0_deg, lon0_deg)

    def geodetic_to_enu(self, lat_deg: float, lon_deg: float, alt_m: float) -> np.ndarray:
        x, y, z = _geodetic_to_ecef(lat_deg, lon_deg, alt_m)
        d = np.array([x, y, z], dtype=np.float64) - self.ecef0
        return self.R_ecef_to_enu @ d  # [E, N, U] meters

    @staticmethod
    def radar_polar_to_enu(range_m: float, az_deg: float, el_deg: float = 0.0) -> np.ndarray:
        # az: North=0°, clockwise; el: +Up
        az = math.radians(az_deg); el = math.radians(el_deg)
        r_h = range_m * math.cos(el)
        N = r_h * math.cos(az)
        E = r_h * math.sin(az)
        U = range_m * math.sin(el)
        return np.array([E, N, U], dtype=np.float64)
