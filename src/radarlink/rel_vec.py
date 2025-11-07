from __future__ import annotations

import numpy as np

Array3 = np.ndarray   # shape (3,)
ArrayNx3 = np.ndarray # shape (N,3)

def compute(target_xyz: Array3, interceptor_xyz: Array3) -> Array3:
    """HT = T - H, both shape (3,), dtype float."""
    return target_xyz - interceptor_xyz

def compute_batch(targets_nx3: ArrayNx3, interceptor_xyz: Array3) -> ArrayNx3:
    """Vectorized HT for (N,3) targets. Broadcasting subtraction."""
    t = np.asarray(targets_nx3, dtype=np.float64)
    h = np.asarray(interceptor_xyz, dtype=np.float64).reshape(1, 3)
    return t - h
