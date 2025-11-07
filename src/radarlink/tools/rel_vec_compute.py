#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterator, Literal, Tuple

import numpy as np

from radarlink.rel_vec import compute, compute_batch


def _parse_interceptor(s: str) -> Tuple[float,float,float]:
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("Use --interceptor 'E,N,U' (meters).")
    return float(parts[0]), float(parts[1]), float(parts[2])

def _iter_json_ndjson(fh) -> Iterator[dict]:
    for line in fh:
        line = line.strip()
        if not line:
            continue
        yield json.loads(line)

def _iter_json_array(fh) -> Iterator[dict]:
    data = json.load(fh)
    if not isinstance(data, list):
        raise SystemExit("JSON array expected.")
    for obj in data:
        yield obj

def _xyz_from_obj(obj: dict, kx: str, ky: str, kz: str|None) -> Tuple[float,float,float]:
    try:
        x = float(obj[kx]); y = float(obj[ky])
        z = float(obj[kz]) if kz and kz in obj and obj[kz] is not None else 0.0
        return x, y, z
    except Exception as e:
        raise ValueError(f"bad object (need {kx},{ky}[,{kz}]): {obj}") from e

def main(argv=None):
    p = argparse.ArgumentParser(description="HT = T - H from JSON input (array or NDJSON).")
    p.add_argument("--infile", type=Path, help="Input file (default: stdin).")
    p.add_argument("--informat", choices=("ndjson","array"), default="ndjson",
                   help="Input format: ndjson (default) or array.")
    p.add_argument("--interceptor", required=True, type=_parse_interceptor,
                   help="Interceptor vector 'E,N,U' (meters) in same frame as targets.")
    p.add_argument("--keys", default="x,y,z",
                   help="Comma-separated keys for x,y,z in input objects. z is optional. E.g. 'x,y' or 'east,north,up'.")
    p.add_argument("-o","--out", type=Path, help="Output NDJSON file (default: stdout).")
    p.add_argument("--batch", action="store_true", help="Batch mode: load all, vectorized.")
    args = p.parse_args(argv)

    key_parts = [k.strip() for k in args.keys.split(",") if k.strip()!=""]
    if len(key_parts) < 2:
        raise SystemExit("--keys must provide at least x,y")
    kx, ky = key_parts[0], key_parts[1]
    kz = key_parts[2] if len(key_parts) >= 3 else None

    fh_in = (args.infile.open("r", encoding="utf-8") if args.infile else sys.stdin)
    out = (args.out.open("w", encoding="utf-8") if args.out else sys.stdout)

    try:
        iterator = _iter_json_ndjson(fh_in) if args.informat=="ndjson" else _iter_json_array(fh_in)
        h = np.asarray(args.interceptor, dtype=np.float64)

        if args.batch:
            # Load all → vectorized
            buf = []
            for obj in iterator:
                try:
                    buf.append(_xyz_from_obj(obj, kx, ky, kz))
                except ValueError:
                    continue
            if not buf:
                return 0
            arr = np.asarray(buf, dtype=np.float64)  # (N,3)
            ht = compute_batch(arr, h)               # (N,3)
            for e,n,u in ht:
                out.write(json.dumps({"e": float(e), "n": float(n), "u": float(u)}) + "\n")
        else:
            # Stream object-by-object
            for obj in iterator:
                try:
                    x,y,z = _xyz_from_obj(obj, kx, ky, kz)
                except ValueError:
                    continue
                vec = compute(np.array((x,y,z), dtype=np.float64), h)
                out.write(json.dumps({"e": float(vec[0]), "n": float(vec[1]), "u": float(vec[2])}) + "\n")
    finally:
        if fh_in is not sys.stdin: fh_in.close()
        if out is not sys.stdout: out.close()

if __name__ == "__main__":
    main()
