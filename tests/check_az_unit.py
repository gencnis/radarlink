import numpy as np

PATH = "../data/moving_target_dataset.npy"  # change if needed

obj = np.load(PATH, allow_pickle=True)

def collect_azimuths(root):
    """Return a flat numpy array of all 'azimuth' values found anywhere."""
    out = []

    def visit(x):
        if isinstance(x, dict):
            for k, v in x.items():
                if k == "azimuth":
                    a = np.asarray(v).ravel()
                    out.extend(a.tolist())
                else:
                    visit(v)
        elif isinstance(x, (list, tuple)):
            for v in x:
                visit(v)
        elif isinstance(x, np.ndarray):
            if x.dtype == object:
                for v in x.flat:
                    visit(v)
            elif x.dtype.names:  # structured array
                if "azimuth" in x.dtype.names:
                    out.extend(np.asarray(x["azimuth"]).ravel().tolist())
                else:
                    # try visiting fields that might be dict-like
                    for name in x.dtype.names:
                        visit(x[name])
            else:
                # plain numeric array → nothing to do
                pass

    # npz files expose a dict-like loader
    if isinstance(root, np.lib.npyio.NpzFile):
        for k in root.files:
            visit(root[k])
    else:
        # try dict-in-0D array case
        try:
            visit(root.item())
        except Exception:
            visit(root)
    return np.array(out, dtype=float)

az = collect_azimuths(obj)
print(f"Found {az.size} azimuth samples")
if az.size:
    print(f"min={az.min():.6f}, max={az.max():.6f}")
    guess = "rad" if np.nanmax(az) < 10 else "deg"
    print("Unit guess:", guess)
else:
    print("No 'azimuth' found; check the file/field names.")
