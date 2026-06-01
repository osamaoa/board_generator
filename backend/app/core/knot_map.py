from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from scipy.ndimage import binary_dilation, distance_transform_edt, gaussian_filter, sobel


def _robust_norm(x: np.ndarray, lo: float = 2.0, hi: float = 98.0) -> np.ndarray:
    finite = np.isfinite(x)
    if not np.any(finite):
        return np.zeros_like(x, dtype=np.float32)
    vals = x[finite]
    v_lo = float(np.percentile(vals, lo))
    v_hi = float(np.percentile(vals, hi))
    if not (np.isfinite(v_lo) and np.isfinite(v_hi)) or v_hi <= v_lo + 1e-8:
        return np.zeros_like(x, dtype=np.float32)
    y = (x - v_lo) / (v_hi - v_lo)
    return np.clip(y, 0.0, 1.0).astype(np.float32, copy=False)


def _structure_isotropy(img: np.ndarray, sigma_tensor: float = 2.0) -> Tuple[np.ndarray, np.ndarray]:
    gx = sobel(img, axis=1, mode="reflect")
    gy = sobel(img, axis=0, mode="reflect")
    grad = np.sqrt(gx * gx + gy * gy).astype(np.float32, copy=False)

    jxx = gaussian_filter(gx * gx, sigma=sigma_tensor)
    jyy = gaussian_filter(gy * gy, sigma=sigma_tensor)
    jxy = gaussian_filter(gx * gy, sigma=sigma_tensor)
    trace = jxx + jyy + 1e-8
    delta = np.sqrt(np.maximum((jxx - jyy) ** 2 + 4.0 * (jxy ** 2), 0.0))
    lam1 = 0.5 * (trace + delta)
    lam2 = 0.5 * (trace - delta)
    coherence = (lam1 - lam2) / (lam1 + lam2 + 1e-8)
    isotropy = np.clip(1.0 - coherence, 0.0, 1.0).astype(np.float32, copy=False)
    return isotropy, grad


def build_knot_map_from_fiber_gray01(
    fiber_gray01: np.ndarray,
    *,
    return_diag: bool = False,
) -> np.ndarray | Tuple[np.ndarray, Dict[str, float]]:
    """
    Build a knot-prior map from a single fiber grayscale image in [0,1].
    Clear-wood is near the global gray baseline; knot zones deviate toward black/white.
    """
    fiber = np.asarray(fiber_gray01, dtype=np.float32)
    if fiber.ndim != 2:
        raise ValueError(f"Expected 2D grayscale array, got shape={fiber.shape}.")
    fiber = np.clip(fiber, 0.0, 1.0)

    baseline = float(np.median(fiber))
    dev = fiber - baseline
    abs_dev = np.abs(dev)

    mad = float(np.median(abs_dev))
    robust_sigma = max(1e-4, 1.4826 * mad)
    z = abs_dev / robust_sigma

    strong_dev = np.clip((z - 2.2) / 6.0, 0.0, 1.0).astype(np.float32, copy=False)
    strong_dev = gaussian_filter(strong_dev, sigma=1.1)
    strong_dev_n = _robust_norm(strong_dev, lo=5.0, hi=99.3)

    pos = gaussian_filter(np.clip(dev, 0.0, None), sigma=2.0)
    neg = gaussian_filter(np.clip(-dev, 0.0, None), sigma=2.0)
    bipolar = 2.0 * np.minimum(pos, neg) / (pos + neg + 1e-6)
    bipolar_n = _robust_norm(np.clip(bipolar, 0.0, 1.0), lo=5.0, hi=99.0)

    isotropy, grad = _structure_isotropy(fiber, sigma_tensor=2.2)
    iso_detail = _robust_norm(
        isotropy * _robust_norm(gaussian_filter(grad, sigma=1.2)),
        lo=5.0,
        hi=99.0,
    )

    raw = _robust_norm(
        0.70 * strong_dev_n + 0.20 * (strong_dev_n * bipolar_n) + 0.10 * iso_detail,
        lo=2.0,
        hi=99.0,
    )

    seed_score = strong_dev_n * (0.60 + 0.40 * bipolar_n)
    seed_percentile = 99.4
    seed_threshold = float(np.percentile(seed_score, seed_percentile))
    seed = seed_score >= seed_threshold
    coverage = float(np.mean(seed))

    if coverage < 0.0012:
        seed_percentile = 98.8
        seed_threshold = float(np.percentile(seed_score, seed_percentile))
        seed = seed_score >= seed_threshold
        coverage = float(np.mean(seed))
    if coverage > 0.04:
        seed_percentile = 99.8
        seed_threshold = float(np.percentile(seed_score, seed_percentile))
        seed = seed_score >= seed_threshold
        coverage = float(np.mean(seed))

    seed = binary_dilation(seed, structure=np.ones((3, 3), dtype=bool), iterations=1)
    dist = distance_transform_edt(~seed)
    sigma_px = max(6.0, 0.024 * float(max(seed.shape)))
    surround = np.exp(-(dist ** 2) / (2.0 * sigma_px * sigma_px))
    surround = _robust_norm(surround.astype(np.float32, copy=False), lo=2.0, hi=99.0)

    prior = _robust_norm(0.84 * raw + 0.16 * (surround * raw), lo=2.0, hi=99.0)
    prior = gaussian_filter(prior, sigma=0.9)

    gate = np.clip(0.30 + 0.70 * (0.75 * strong_dev_n + 0.25 * bipolar_n), 0.0, 1.0)
    prior = np.clip(prior * gate, 0.0, 1.0).astype(np.float32, copy=False)

    if not return_diag:
        return prior
    diag = {
        "baseline": float(baseline),
        "mad": float(mad),
        "robust_sigma": float(robust_sigma),
        "seed_percentile": float(seed_percentile),
        "seed_threshold": float(seed_threshold),
        "seed_coverage": float(np.mean(seed)),
        "prior_mean": float(np.mean(prior)),
        "prior_p95": float(np.percentile(prior, 95.0)),
        "prior_p99": float(np.percentile(prior, 99.0)),
        "strong_dev_p95": float(np.percentile(strong_dev_n, 95.0)),
        "bipolar_p95": float(np.percentile(bipolar_n, 95.0)),
    }
    return prior, diag
