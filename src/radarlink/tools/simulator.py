#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from radarlink.geo.loc_convert import PolarConventions, RadarLocConverter
from radarlink.tools.control_hub import ControlHubRuntime

# ---------------- Config / constants ----------------
NAMES  = ['radar', 'target', 'interceptor']
COLORS = ['tab:blue', 'tab:orange', 'tab:green']

XY_LIM  = 300.0
Z_MIN, Z_MAX = -100.0, 200.0
SQUARE_SIZE = 6.0

DRAG_ENTITY_IDXS = [0, 2]  # radar & interceptor

# ---------------- Small helpers ----------------
def _az_to_deg(conv, az_value) -> float:
    """Convert radar az value to degrees with 0°=North, clockwise."""
    az = float(az_value)
    if conv.az_unit == "deg":
        az = math.radians(az)
    if conv.az_ref.lower() == "east":
        az -= math.pi / 2.0
    if not conv.az_cw:
        az = -az
    return (math.degrees(az) + 360.0) % 360.0

def pairwise_distance(p, q) -> float:
    return float(np.linalg.norm(p - q))

def _deg2rad(d: float) -> float:
    return float(d) * math.pi / 180.0

def _rad2deg(r: float) -> float:
    return float(r) * 180.0 / math.pi

def cube_faces_from_center(center, edge):
    cx, cy, cz = map(float, center)
    h = edge / 2.0
    v000 = (cx - h, cy - h, cz - h)
    v001 = (cx - h, cy - h, cz + h)
    v010 = (cx - h, cy + h, cz - h)
    v011 = (cx - h, cy + h, cz + h)
    v100 = (cx + h, cy - h, cz - h)
    v101 = (cx + h, cy - h, cz + h)
    v110 = (cx + h, cy + h, cz - h)
    v111 = (cx + h, cy + h, cz + h)
    return [
        [v000, v100, v110, v010],  # -X
        [v001, v101, v111, v011],  # +X
        [v000, v001, v011, v010],  # -Y
        [v100, v101, v111, v110],  # +Y
        [v000, v001, v101, v100],  # -Z
        [v010, v011, v111, v110],  # +Z
    ]

def make_or_update_cube(ax3d, cube_artist, faces):
    if cube_artist is None:
        cube = Poly3DCollection(
            faces, alpha=0.25, facecolor='crimson', edgecolor='k', linewidths=0.7
        )
        ax3d.add_collection3d(cube)
        return cube
    cube_artist.set_verts(faces)
    return cube_artist

# ---------------- ENU <-> Polar (consistent with control_hub) ----------------
def enu_to_polar_for(rt: ControlHubRuntime, e: float, n: float, u: float) -> Tuple[float, float, float]:
    """
    Convert ENU (radar-centric) to (range, az, el) in rt's conventions.
    range: slant (m), az: unit/ref/CW per rt, el: +up, unit per rt.
    """
    conv = rt.cvt.conv
    theta = math.atan2(e, n)  # radians (math angle from North, CCW)
    az = theta
    if not conv.az_cw:
        az = -az
    if conv.az_ref.lower() == "east":
        az = az + math.pi / 2.0
    az_out = _rad2deg(az) if conv.az_unit == "deg" else az

    r_h = math.hypot(e, n)
    r = math.hypot(r_h, u)
    el = math.atan2(u, r_h)
    el_out = _rad2deg(el) if conv.el_unit == "deg" else el
    return (r, az_out, el_out)

# ---------------- Synthetic trajectory generator ----------------
def rand2(rs: np.random.RandomState, low, high):
    return rs.uniform(low, high)

def make_random_start(rs: np.random.RandomState,
                      xy_lim: float, z_min: float, z_max: float) -> np.ndarray:
    x = rand2(rs, -0.4*xy_lim, 0.4*xy_lim)
    y = rand2(rs, -0.4*xy_lim, 0.4*xy_lim)
    z = rand2(rs, 0.1*z_min, 0.6*z_max)
    return np.array([x, y, z], dtype=float)

def lissajous_3d(t: float, A: float, B: float, C: float,
                 ax: float, ay: float, az: float,
                 phase_xy: float, phase_z: float) -> np.ndarray:
    x = A * math.sin(ax * t + phase_xy)
    y = B * math.sin(ay * t)
    z = C * math.sin(az * t + phase_z)
    return np.array([x, y, z], dtype=float)

def make_trajectory_fn(rs: np.random.RandomState,
                       xy_lim: float, z_min: float, z_max: float):
    A = rand2(rs, 0.25*xy_lim, 0.6*xy_lim)
    B = rand2(rs, 0.25*xy_lim, 0.6*xy_lim)
    C = rand2(rs, 0.15*(z_max - z_min), 0.45*(z_max - z_min))
    ax = rand2(rs, 0.4, 0.9)
    ay = rand2(rs, 0.5, 1.2)
    az = rand2(rs, 0.3, 0.8)
    pxy = rand2(rs, 0, 2*math.pi)
    pz  = rand2(rs, 0, 2*math.pi)
    center = np.array([rand2(rs, -0.2*xy_lim, 0.2*xy_lim),
                       rand2(rs, -0.2*xy_lim, 0.2*xy_lim),
                       rand2(rs,  0.1*z_min,  0.4*z_max)], float)

    def path(t: float) -> np.ndarray:
        return center + lissajous_3d(t, A, B, C, ax, ay, az, pxy, pz)
    return path

# ---------------- Replay loaders (CSV / NDJSON) ----------------
def load_replay_csv(csv_path: Path) -> List[np.ndarray]:
    rows = []
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return rows
        cols = {k.lower(): k for k in reader.fieldnames}
        have_xyz     = all(c in cols for c in ("x","y","z"))
        have_xyz_est = all(c in cols for c in ("x_est","y_est","z_est"))
        have_polar   = ("range_m" in cols) and ("az_deg" in cols)

        for r in reader:
            try:
                if have_xyz:
                    rows.append(np.array([float(r[cols["x"]]),
                                          float(r[cols["y"]]),
                                          float(r[cols["z"]])], float))
                elif have_xyz_est:
                    rows.append(np.array([float(r[cols["x_est"]]),
                                          float(r[cols["y_est"]]),
                                          float(r[cols["z_est"]])], float))
                elif have_polar:
                    r_m  = float(r[cols["range_m"]])
                    az   = _deg2rad(float(r[cols["az_deg"]]))
                    el   = _deg2rad(float(r[cols["el_deg"]])) if "el_deg" in cols and r[cols["el_deg"]] not in (None,"") else 0.0
                    e = r_m * math.sin(az) * math.cos(el)
                    n = r_m * math.cos(az) * math.cos(el)
                    u = r_m * math.sin(el)
                    rows.append(np.array([e,n,u], float))
            except Exception:
                continue
    return rows

def load_replay_ndjson(ndjson_path: Path) -> List[np.ndarray]:
    rows = []
    with ndjson_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                o = json.loads(line)
            except Exception:
                continue
            if all(k in o for k in ("x","y","z")):
                rows.append(np.array([float(o["x"]), float(o["y"]), float(o["z"])], float))
            elif all(k in o for k in ("e","n","u")):
                rows.append(np.array([float(o["e"]), float(o["n"]), float(o["u"])], float))
            elif "range" in o and "azimuth" in o:
                r = float(o["range"])
                az = float(o["azimuth"])
                el = float(o.get("elevation", 0.0))
                tmp_rt = ControlHubRuntime.from_env()
                cvt = RadarLocConverter(tmp_rt.cvt.conv)
                x,y,z = cvt.polar_to_cart(r, az, el)
                rows.append(np.array([x,y,z], float))
    return rows

# ---------------- Draggable helpers ----------------
class DraggablePoints2D:
    def __init__(self, ax, scatters, xs, ys, zs, on_update, entity_indices, can_drag):
        self.ax = ax
        self.scatters = scatters
        self.xs = xs; self.ys = ys; self.zs = zs
        self.on_update = on_update
        self.entity_indices = entity_indices
        self.can_drag = can_drag
        self.artist_to_idx = {s: i for i, s in enumerate(self.scatters)}
        self._dragging = None
        c = ax.figure.canvas.mpl_connect
        self.cid_press = c('button_press_event', self.on_press)
        self.cid_release = c('button_release_event', self.on_release)
        self.cid_motion = c('motion_notify_event', self.on_motion)

    def on_press(self, event):
        if not self.can_drag(): return
        if event.inaxes != self.ax: return
        for s in self.scatters:
            contains, _ = s.contains(event)
            if contains:
                self._dragging = self.artist_to_idx[s]
                break

    def on_motion(self, event):
        if self._dragging is None or not self.can_drag(): return
        if event.inaxes != self.ax: return
        if event.xdata is None or event.ydata is None: return
        vis_idx = self._dragging
        real_idx = self.entity_indices[vis_idx]
        self.xs[real_idx] = float(event.xdata)
        self.ys[real_idx] = float(event.ydata)
        self.scatters[vis_idx].set_offsets(np.column_stack([self.xs[real_idx], self.ys[real_idx]]))
        self.on_update()

    def on_release(self, event):
        self._dragging = None

class DraggableZOnly:
    def __init__(self, ax, scatters, xs, ys, zs, on_update, entity_indices, can_drag):
        self.ax = ax
        self.scatters = scatters
        self.xs = xs; self.ys = ys; self.zs = zs
        self.on_update = on_update
        self.entity_indices = entity_indices
        self.can_drag = can_drag
        self.artist_to_idx = {s: i for i, s in enumerate(self.scatters)}
        self._dragging = None
        c = ax.figure.canvas.mpl_connect
        self.cid_press = c('button_press_event', self.on_press)
        self.cid_release = c('button_release_event', self.on_release)
        self.cid_motion = c('motion_notify_event', self.on_motion)

    def on_press(self, event):
        if not self.can_drag(): return
        if event.inaxes != self.ax: return
        for s in self.scatters:
            contains, _ = s.contains(event)
            if contains:
                self._dragging = self.artist_to_idx[s]
                break

    def on_motion(self, event):
        if self._dragging is None or not self.can_drag(): return
        if event.inaxes != self.ax: return
        if event.ydata is None: return
        vis_idx = self._dragging
        real_idx = self.entity_indices[vis_idx]
        new_z = float(np.clip(event.ydata, Z_MIN, Z_MAX))
        self.zs[real_idx] = new_z
        self.scatters[vis_idx].set_offsets(np.column_stack([vis_idx, self.zs[real_idx]]))
        self.on_update()

    def on_release(self, event):
        self._dragging = None

# ---------------- Simulator ----------------
def simulator_main():
    ap = argparse.ArgumentParser(description="RadarLink Simulator (synthetic & replay modes)")
    ap.add_argument("--mode", choices=["synthetic","replay"], default="synthetic")
    ap.add_argument("--csv", type=str, help="Replay CSV path (x,y,z or x_est,y_est,z_est or range_m,az_deg[,el_deg])")
    ap.add_argument("--ndjson", type=str, help="Replay NDJSON path (x,y,z) or (e,n,u) or (range,az[,elevation])")
    ap.add_argument("--per-row", type=float, default=0.10, help="Seconds per step while playing")
    ap.add_argument("--xy-lim", type=float, default=XY_LIM)
    ap.add_argument("--z-min", type=float, default=Z_MIN)
    ap.add_argument("--z-max", type=float, default=Z_MAX)
    ap.add_argument("--square", type=float, default=SQUARE_SIZE, help="Red square size (m)")
    ap.add_argument("--seed-positions", type=int, default=42, help="RNG seed for initial radar/interceptor")
    ap.add_argument("--seed-trajectory", type=int, default=7, help="RNG seed for synthetic trajectory")
    args = ap.parse_args()

    rt = ControlHubRuntime.from_env()

    # Initial visual positions
    rs_pos = np.random.RandomState(args.seed_positions)
    xs = np.zeros(3, float); ys = np.zeros(3, float); zs = np.zeros(3, float)
    radar0 = np.array([0.0, 0.0,  1.0]) + np.array([rs_pos.uniform(-10,10), rs_pos.uniform(-10,10), rs_pos.uniform(-3,3)])
    inter0 = radar0 + np.array([rs_pos.uniform(80,140), rs_pos.uniform(160,240), rs_pos.uniform(0,20)])
    xs[0], ys[0], zs[0] = radar0
    xs[2], ys[2], zs[2] = inter0

    # Synthetic or replay
    if args.mode == "synthetic":
        rs_traj = np.random.RandomState(args.seed_trajectory)
        traj = make_trajectory_fn(rs_traj, args.xy_lim, args.z_min, args.z_max)
        def get_target_vec(t): return traj(t)
        total_steps = None
    else:
        if args.csv:
            rows = load_replay_csv(Path(args.csv))
        elif args.ndjson:
            rows = load_replay_ndjson(Path(args.ndjson))
        else:
            raise SystemExit("Replay mode requires --csv or --ndjson")
        if not rows:
            raise SystemExit("No usable rows in replay file")
        def get_target_vec(t_idx): return rows[int(t_idx)]
        total_steps = len(rows)

    # ---- Figure & layout ----
    fig = plt.figure(constrained_layout=True, figsize=(14.5, 11.5))
    gs = GridSpec(3, 2, figure=fig, height_ratios=[1, 1.25, 1.0], width_ratios=[2, 1])

    ax_xy = fig.add_subplot(gs[0, 0])
    ax_z  = fig.add_subplot(gs[0, 1])
    ax_3d = fig.add_subplot(gs[1, 0], projection='3d')
    right_sub = gs[1, 1].subgridspec(2, 1, height_ratios=[1, 1])
    ax_tbl_pos  = fig.add_subplot(right_sub[0, 0])
    ax_tbl_dist = fig.add_subplot(right_sub[1, 0])

    ax_polar_r = fig.add_subplot(gs[2, 0], projection='polar')
    ax_polar_i = fig.add_subplot(gs[2, 1], projection='polar')

    def setup_polar(ax, title):
        ax.set_title(title, fontsize=11)
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)   # clockwise
        ax.set_rlim(0, args.xy_lim * 1.25)
        ax.grid(True, linestyle=':', linewidth=0.7)
        ln, = ax.plot([], [], linestyle=(0, (2, 3)), lw=1.0, alpha=0.9)  # placeholder (unused)
        pt, = ax.plot([], [], marker='s', markersize=6, linestyle='None', color='crimson', alpha=0.7)
        return ln, pt

    pol_line_r, pol_pt_r = setup_polar(ax_polar_r, "Radar POV — 0°=North (CW)")
    pol_line_i, pol_pt_i = setup_polar(ax_polar_i, "Interceptor POV — 0°=North (CW)")

    # ---- XY panel
    ax_xy.set_title('XY — drag radar & interceptor (only when PAUSED)')
    ax_xy.set_xlabel('East (m)'); ax_xy.set_ylabel('North (m)')
    ax_xy.set_xlim(-args.xy_lim, args.xy_lim); ax_xy.set_ylim(-args.xy_lim, args.xy_lim)
    ax_xy.set_aspect('equal', adjustable='box'); ax_xy.grid(True, linestyle=':', linewidth=0.7)
    ax_xy.axhline(0, color='gray', lw=0.8, alpha=0.6); ax_xy.axvline(0, color='gray', lw=0.8, alpha=0.6)

    scat_xy, texts_xy, vis_names = [], [], []
    for vis_idx, ent_idx in enumerate(DRAG_ENTITY_IDXS):
        name = NAMES[ent_idx]; color = COLORS[ent_idx]
        sc = ax_xy.scatter([xs[ent_idx]], [ys[ent_idx]], s=140, color=color,
                           edgecolor='k', picker=8, zorder=3)
        scat_xy.append(sc)
        txt = ax_xy.text(xs[ent_idx]+2, ys[ent_idx]+2,
                         f'{name}\n({xs[ent_idx]:.1f},{ys[ent_idx]:.1f})',
                         fontsize=9, color=color, weight='bold')
        texts_xy.append(txt); vis_names.append(name)

    # ---- Z-only panel
    ax_z.set_title('Z — drag radar & interceptor (only when PAUSED)')
    ax_z.set_xlabel('Entities'); ax_z.set_ylabel('Up (m)')
    ax_z.set_xlim(-0.5, len(DRAG_ENTITY_IDXS)-0.5); ax_z.set_ylim(args.z_min, args.z_max)
    ax_z.set_xticks(range(len(DRAG_ENTITY_IDXS))); ax_z.set_xticklabels(vis_names)
    ax_z.grid(True, linestyle=':', linewidth=0.7); ax_z.axhline(0, color='gray', lw=0.8, alpha=0.6)
    scat_z, texts_z = [], []
    for vis_idx, ent_idx in enumerate(DRAG_ENTITY_IDXS):
        name = NAMES[ent_idx]; color = COLORS[ent_idx]
        sc = ax_z.scatter([vis_idx], [zs[ent_idx]], s=140, color=color,
                          edgecolor='k', picker=8, zorder=3)
        scat_z.append(sc)
        txt = ax_z.text(vis_idx+0.08, zs[ent_idx]+1.5, f'{name}\n(z={zs[ent_idx]:.1f})',
                        fontsize=9, color=color, weight='bold')
        texts_z.append(txt)

    # ---- 3D panel
    ax_3d.set_title('3D (target trajectory)')
    ax_3d.set_xlabel('East'); ax_3d.set_ylabel('North'); ax_3d.set_zlabel('Up')
    ax_3d.set_xlim(-args.xy_lim, args.xy_lim); ax_3d.set_ylim(-args.xy_lim, args.xy_lim)
    ax_3d.set_zlim(args.z_min, args.z_max)
    ax_3d.plot([0,0],[0,0],[args.z_min,args.z_max], color='gray', lw=0.8, alpha=0.5)

    scat_3d = {
        'radar':        ax_3d.scatter([xs[0]], [ys[0]], [zs[0]], s=70, color=COLORS[0], edgecolor='k', depthshade=True),
        'interceptor':  ax_3d.scatter([xs[2]], [ys[2]], [zs[2]], s=70, color=COLORS[2], edgecolor='k', depthshade=True),
        'target':       ax_3d.scatter([0],[0],[0], s=70, color=COLORS[1], edgecolor='k', depthshade=True),
    }

    # Red square marker for HT tip (in XY)
    red_square = ax_xy.scatter([], [], s=120, marker='s', color='crimson', alpha=0.35, edgecolor='k')

    # Cube at target (3D)
    cube_artist = None

    # Tables
    def render_tables(R, H, T, HT, bearing_deg, elev_deg, ground, slant):
        ax_tbl_pos.clear(); ax_tbl_pos.set_title('Positions (E, N, U)', fontsize=11); ax_tbl_pos.axis('off')
        headers_pos = ['Point', 'E', 'N', 'U']
        data_pos = [
            ['radar',       f'{R[0]:.1f}', f'{R[1]:.1f}', f'{R[2]:.1f}'],
            ['interceptor', f'{H[0]:.1f}', f'{H[1]:.1f}', f'{H[2]:.1f}'],
            ['target',      f'{T[0]:.1f}', f'{T[1]:.1f}', f'{T[2]:.1f}'],
        ]
        tp = ax_tbl_pos.table(cellText=data_pos, colLabels=headers_pos, loc='center')
        tp.auto_set_font_size(False); tp.set_fontsize(9); tp.scale(1.0, 1.1)

        ax_tbl_dist.clear(); ax_tbl_dist.set_title('Ranges & Angles', fontsize=11); ax_tbl_dist.axis('off')
        data_dist = [
            ['radar→target ground', f'{math.hypot(T[0]-R[0], T[1]-R[1]):.2f} m'],
            ['radar→target slant',  f'{np.linalg.norm(T-R):.2f} m'],
            ['interceptor→target ground', f'{ground:.2f} m'],
            ['interceptor→target slant',  f'{slant:.2f} m'],
            ['bearing (int→tgt)',   f'{bearing_deg:.2f}°'],
            ['elevation (int→tgt)', f'{elev_deg:.2f}°'],
        ]
        td = ax_tbl_dist.table(cellText=data_dist, colLabels=['Metric','Value'], loc='center')
        td.auto_set_font_size(False); td.set_fontsize(9); td.scale(1.0, 1.1)

    # State
    paused = False

    def set_status():
        status_txt.set_text("PAUSED" if paused else "PLAYING")
        status_txt.set_color("crimson" if paused else "green")
        fig.canvas.draw_idle()

    status_txt = fig.text(
        0.86, 0.985, "PLAYING", fontsize=11, ha="left", va="top",
        bbox=dict(boxstyle="round,pad=0.25", fc="w", ec="0.5", alpha=0.9), color="green"
    )

    def on_key(ev):
        nonlocal paused
        if ev.key in (' ', 'p'):
            paused = not paused
            set_status()
    fig.canvas.mpl_connect('key_press_event', on_key)

    # Draggers (only when paused)
    can_drag = lambda: paused
    _drag_xy = DraggablePoints2D(ax_xy, scat_xy, xs, ys, zs, on_update=lambda: refresh_all(), entity_indices=DRAG_ENTITY_IDXS, can_drag=can_drag)
    _drag_z  = DraggableZOnly(ax_z,  scat_z, xs, ys, zs, on_update=lambda: refresh_all(), entity_indices=DRAG_ENTITY_IDXS, can_drag=can_drag)

    # Refresh everything
    def refresh_all():
        # Update labels in XY/Z
        for vis_idx, ent_idx in enumerate(DRAG_ENTITY_IDXS):
            scat_xy[vis_idx].set_offsets(np.column_stack([xs[ent_idx], ys[ent_idx]]))
            texts_xy[vis_idx].set_position((xs[ent_idx]+2, ys[ent_idx]+2))
            texts_xy[vis_idx].set_text(f'{NAMES[ent_idx]}\n({xs[ent_idx]:.1f},{ys[ent_idx]:.1f})')
            scat_z[vis_idx].set_offsets(np.column_stack([vis_idx, zs[ent_idx]]))
            texts_z[vis_idx].set_position((vis_idx+0.08, zs[ent_idx]+1.5))
            texts_z[vis_idx].set_text(f'{NAMES[ent_idx]}\n(z={zs[ent_idx]:.1f})')

        # Visual ENU
        R_vis = np.array([xs[0], ys[0], zs[0]], float)
        H_vis = np.array([xs[2], ys[2], zs[2]], float)
        T_vis = current_target.copy()

        # Compute-space (radar at origin)
        R = np.array([0.0, 0.0, 0.0], float)
        H = H_vis - R_vis
        T = T_vis - R_vis

        # Sync runtime interceptor
        rt.H[:] = H

        # Feed control_hub with polar derived from T (exercise full pipeline)
        r, az_in, el_in = enu_to_polar_for(rt, T[0], T[1], T[2])
        o = {
            "timestamp_ms": int(time.time() * 1000),
            "range": r,
            "azimuth": az_in,
            "elevation": el_in,
            "snr_db": None,
            "class_name": "uav",
        }
        out = rt.process(o)

        # Pull outputs
        e, n, u = out["computed"]["e"], out["computed"]["n"], out["computed"]["u"]
        bearing_deg = out["computed"]["bearing_deg"]
        elev_deg    = out["computed"]["elevation_deg"]
        slant       = out["computed"]["range_m"]
        ground      = math.hypot(e, n)

        # --- Update 3D points
        scat_3d['radar']._offsets3d       = (np.array([R_vis[0]]), np.array([R_vis[1]]), np.array([R_vis[2]]))
        scat_3d['interceptor']._offsets3d = (np.array([H_vis[0]]), np.array([H_vis[1]]), np.array([H_vis[2]]))
        scat_3d['target']._offsets3d      = (np.array([T_vis[0]]), np.array([T_vis[1]]), np.array([T_vis[2]]))

        # Update cube at target
        nonlocal cube_artist
        cube_artist = make_or_update_cube(ax_3d, cube_artist, cube_faces_from_center(T_vis, 6.0))

        # --- Polar POVs (titles driven by pipeline/control_hub)

        # Radar POV: use azimuth we fed in (converted to 0°=N, CW) and show ground & slant from T
        radar_bearing_deg = _az_to_deg(rt.cvt.conv, az_in)
        theta_r = math.radians(radar_bearing_deg)
        ground_radar = math.hypot(T[0], T[1])
        slant_radar  = float(np.linalg.norm(T))
        pol_pt_r.set_data([theta_r], [ground_radar])
        ax_polar_r.set_rmax(max(ax_polar_r.get_rmax(), ground_radar * 1.05))
        ax_polar_r.set_title(
            f"Radar POV (0°=N) — bearing {radar_bearing_deg:6.2f}° | ground {ground_radar:,.1f} m | slant {slant_radar:,.1f} m",
            fontsize=11
        )

        # Interceptor POV: use computed bearing & ranges from control_hub
        theta_i = math.radians(bearing_deg)
        pol_pt_i.set_data([theta_i], [ground])  # radius = ground range
        ax_polar_i.set_rmax(max(ax_polar_i.get_rmax(), ground * 1.05))
        ax_polar_i.set_title(
            f"Interceptor POV (0°=N) — bearing {bearing_deg:6.2f}° | ground {ground:,.1f} m | slant {slant:,.1f} m",
            fontsize=11
        )

        # --- Red square at HT tip in XY (no line)
        ht_tip_xy = (H_vis[0] + e, H_vis[1] + n)
        red_square.set_offsets(np.array([[ht_tip_xy[0], ht_tip_xy[1]]]))
        red_square.set_sizes([args.square * 10.0])

        # --- Tables
        render_tables(R_vis, H_vis, T_vis, np.array([e,n,u]), bearing_deg, elev_deg, ground, slant)

        fig.canvas.draw_idle()

    # Initial target
    current_target = np.array([0.0,0.0,0.0], float)

    # Main playback loop
    plt.show(block=False)
    set_status()

    step = 0
    while True:
        if args.mode == "synthetic":
            t_real = step * args.per_row
            current_target[:] = get_target_vec(t_real)
        else:
            if step >= total_steps:
                break
            current_target[:] = get_target_vec(step)

        refresh_all()

        t0 = time.time()
        while True:
            plt.pause(0.03)
            if paused:
                refresh_all()
                t0 = time.time()
            if (time.time() - t0) >= args.per_row and not paused:
                break
        step += 1

    print("Done.")
    plt.show()


if __name__ == "__main__":
    simulator_main()
