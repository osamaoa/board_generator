import numpy as np

try:
    import cupy as cp  # type: ignore
except Exception:
    cp = None


def gpu_available() -> bool:
    if cp is None:
        return False
    try:
        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


def get_xp(use_gpu: bool):
    if use_gpu and gpu_available():
        return cp
    return np


def gpu_enabled(xp) -> bool:
    return cp is not None and xp is cp


def is_gpu_array(arr) -> bool:
    return cp is not None and isinstance(arr, cp.ndarray)


def to_numpy(arr):
    if is_gpu_array(arr):
        return cp.asnumpy(arr)
    return arr


def seed_all(seed: int):
    np.random.seed(seed)
    if cp is not None:
        try:
            cp.random.seed(seed)
        except Exception:
            pass
