# drop-in test you can run locally
import json
import math


def it_rows(path):
    with open(path) as f:
        data = json.load(f)  # or iterate NDJSON lines if that's your format
        for o in data:
            yield o

prev = None
pairs = []
for o in it_rows("../data/real_samples.json"):
    t  = o["timestamp_ms"] * 1e-3
    x,y,z = o["x_mean"], o["y_mean"], o["z_mean"]
    r  = math.sqrt(x*x + y*y + z*z)
    vm = o["velocity_mean"]  # suspected radial (m/s)
    if prev:
        dt = max(1e-3, t - prev["t"])
        dr = r - prev["r"]
        rdot = dr / dt
        pairs.append((rdot, vm))
    prev = {"t": t, "r": r}

# simple correlation
import statistics

xs = [a for a,b in pairs]; ys = [b for a,b in pairs]
mx,my = statistics.mean(xs), statistics.mean(ys)
sx,sy = statistics.pstdev(xs) or 1e-9, statistics.pstdev(ys) or 1e-9
corr = sum(((x-mx)/sx)*((y-my)/sy) for x,y in pairs)/len(pairs) if pairs else 0.0
print("corr(rdot, velocity_mean) ≈", round(corr,3))
