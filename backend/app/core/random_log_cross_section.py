from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Sequence

import numpy as np
import scipy.io
from scipy.interpolate import CubicSpline


def _build_circular_distance_matrix(n_angles: int) -> np.ndarray:
    # Matches MATLAB Dist construction in cross_section_generator.m.
    p = np.pi
    base = np.concatenate(
        [
            np.arange(0.0, p + 1e-12, 2.0 * p / n_angles, dtype=float),
            p - np.arange(2.0 * p / n_angles, p - (2.0 * p / n_angles) + 1e-12, 2.0 * p / n_angles, dtype=float),
        ]
    )
    if base.shape[0] != n_angles:
        # Robust fallback if floating-point stepping differs from MATLAB-like lengths.
        idx = np.arange(n_angles, dtype=int)
        base = np.minimum(idx, n_angles - idx) * (2.0 * p / n_angles)

    dist = np.zeros((n_angles, n_angles), dtype=float)
    cols = np.arange(n_angles, dtype=int)
    dist[0, :] = base
    for r in range(1, n_angles):
        cols = np.concatenate(([cols[-1]], cols[:-1]))
        dist[r, :] = dist[0, cols]
    return dist


@dataclass
class _FactorModel:
    factor: float
    weight: float
    basis: np.ndarray
    sqrt_eigvals: np.ndarray


class RandomLogCrossSectionGenerator:
    """
    Runtime log cross-section generator ported from cross_section_generator.m.

    It samples a random radial profile around the log perimeter using the same
    covariance-model logic and then maps it to a real ring-count/radial template
    derived from Harald's dataset.
    """

    # MATLAB defaults from cross_section_generator.m (current project variant).
    _NDFI = 100
    # _ADJUST_X_MM = 12.0
    _ADJUST_X_MM = 4.0
    _ADJUST_SF = 0.2
    _MAX_SAMPLE_RETRIES = 40

    # [test_factor, weight]
    _FACTOR_WEIGHTS = np.asarray(
        [
            [1.00, 0.3],
            [1.10, 0.1],
            [1.20, 0.1],
            [1.25, 0.1],
            [1.30, 0.1],
            [1.35, 0.1],
            [1.40, 0.1],
            [1.45, 0.1],
        ],
        dtype=float,
    )

    def __init__(self, mat_path: str):
        payload = scipy.io.loadmat(mat_path)
        self._all_data_harald = np.asarray(payload["All_data_harald"], dtype=float)
        self._one_line_log_data = np.asarray(payload["One_line_log_data"], dtype=np.int64)
        self._rn_rs_all = np.asarray(payload["Rn_Rs_all"], dtype=float)

        if self._rn_rs_all.ndim != 2 or self._rn_rs_all.shape[1] < 2:
            raise RuntimeError("Invalid Rn_Rs_all in log-generator data.")
        if self._one_line_log_data.ndim != 2 or self._one_line_log_data.shape[1] < 8:
            raise RuntimeError("Invalid One_line_log_data in log-generator data.")

        self._mean_radius = float(np.mean(self._rn_rs_all))
        self._std_radius = float(np.mean(np.std(self._rn_rs_all, axis=0)))
        corr = np.corrcoef(self._rn_rs_all, rowvar=False)
        self._cor_radius_n_s = float(corr[0, 1]) if corr.shape[0] >= 2 else 0.0
        self._r_all = np.mean(self._rn_rs_all[:, 0:2], axis=1)

        self._dist = _build_circular_distance_matrix(self._NDFI)
        self._factor_models = self._build_factor_models()
        weights = np.asarray([m.weight for m in self._factor_models], dtype=float)
        wsum = float(np.sum(weights))
        if not np.isfinite(wsum) or wsum <= 0.0:
            raise RuntimeError("Invalid factor weights for random log generator.")
        self._factor_prob = weights / wsum

    @staticmethod
    def default_data_path_candidates() -> List[str]:
        here = os.path.dirname(__file__)
        repo_root = os.path.abspath(os.path.join(here, "../../../"))
        return [
            os.path.join(repo_root, "data", "from_haralds_data_201008.mat"),
            os.path.join(repo_root, "data", "from_haralds_data_221115.mat"),
            os.path.join(repo_root, ".old_log_generator", "from_haralds_data_201008.mat"),
            os.path.join(repo_root, ".old_log_generator", "from_haralds_data_221115.mat"),
        ]

    @classmethod
    def from_default_paths(cls) -> "RandomLogCrossSectionGenerator":
        for path in cls.default_data_path_candidates():
            if os.path.isfile(path):
                return cls(path)
        raise RuntimeError(
            "Missing random log generator data file. "
            "Expected one of: data/from_haralds_data_201008.mat, "
            "data/from_haralds_data_221115.mat."
        )

    def _build_factor_models(self) -> List[_FactorModel]:
        models: List[_FactorModel] = []
        cor = self._cor_radius_n_s

        for factor, weight in self._FACTOR_WEIGHTS:
            dist_clipped = np.minimum(self._dist, np.pi / factor)
            cm = np.cos(dist_clipped * factor) * ((1.0 - cor) / 2.0) + (1.0 - (1.0 - cor) / 2.0)

            eigvals, eigvecs = np.linalg.eigh(cm)
            max_eval = float(np.max(eigvals)) if eigvals.size else 0.0
            keep = eigvals > (max_eval * 1e-6)
            if not np.any(keep):
                continue

            ev = np.asarray(eigvals[keep], dtype=float)
            vv = np.asarray(eigvecs[:, keep], dtype=float)
            models.append(
                _FactorModel(
                    factor=float(factor),
                    weight=float(weight),
                    basis=vv,
                    sqrt_eigvals=np.sqrt(np.maximum(ev, 0.0)),
                )
            )

        if not models:
            raise RuntimeError("Could not build covariance models for random log generation.")
        return models

    def _sample_profile(self) -> np.ndarray:
        idx = int(np.random.choice(len(self._factor_models), p=self._factor_prob))
        model = self._factor_models[idx]

        z = np.random.randn(model.sqrt_eigvals.shape[0])
        data_real = (z * model.sqrt_eigvals) @ model.basis.T  # shape: (NDFI,)
        radiuses = np.concatenate([data_real, data_real[:1]])
        radiuses = radiuses * self._std_radius + self._mean_radius
        return np.asarray(radiuses, dtype=float)

    def _give_data_for_one_log(self, log_nr_one_based: int) -> tuple[np.ndarray, bool]:
        # Port of give_data_for_one_log_201025.m (only the path used by this project).
        x = float(self._ADJUST_X_MM)
        sf = float(self._ADJUST_SF)

        idx = int(log_nr_one_based) - 1
        if idx < 0 or idx >= self._one_line_log_data.shape[0]:
            return np.empty((0, 4), dtype=float), True

        row_start = int(self._one_line_log_data[idx, 6]) - 1
        row_end = int(self._one_line_log_data[idx, 7]) - 1
        if row_start < 0 or row_end < row_start:
            return np.empty((0, 4), dtype=float), True

        # MATLAB uses -sort(-rows), i.e. descending row order.
        rows = np.arange(row_start, row_end + 1, dtype=int)[::-1]
        rr = np.asarray(self._all_data_harald[rows][:, [4, 5, 8]], dtype=float)
        if rr.size == 0:
            return np.empty((0, 4), dtype=float), True

        nr = int(np.max(rr[:, 0]))
        rr_comp = np.zeros((nr, 4), dtype=float)
        rr_comp[:, 0] = np.arange(1, nr + 1, dtype=float)

        problematic = False
        first_ring = int(rr[0, 0])

        for r in range(rr.shape[0] - 1):
            r1d = int(rr[r, 0])
            r2d = int(rr[r + 1, 0])
            dy = int(r2d - r1d)
            if dy <= 0:
                problematic = True
                break

            dr_dy_n = float((rr[r + 1, 1] - rr[r, 1]) / dy)
            dr_dy_s = float((rr[r + 1, 2] - rr[r, 2]) / dy)
            if dr_dy_n <= 0.0 or dr_dy_s <= 0.0:
                problematic = True
                break

            i0 = r1d - 1
            i1 = r2d
            rr_comp[i0:i1, 1] = np.linspace(rr[r, 1], rr[r + 1, 1], dy + 1)
            rr_comp[i0:i1, 2] = np.linspace(rr[r, 2], rr[r + 1, 2], dy + 1)
            rr_comp[i0, 3] = 1.0
            rr_comp[i1 - 1, 3] = 1.0

        if problematic:
            return np.empty((0, 4), dtype=float), True

        if first_ring == 2 and rr_comp.shape[0] >= 2:
            rr_comp[0, :] = np.asarray([1.0, rr_comp[1, 1] / 2.0, rr_comp[1, 2] / 2.0, 0.0], dtype=float)
        elif first_ring == 3 and rr_comp.shape[0] >= 3:
            rr_comp[0, :] = np.asarray([1.0, rr_comp[2, 1] / 3.0, rr_comp[2, 2] / 3.0, 0.0], dtype=float)
            rr_comp[1, :] = np.asarray([2.0, rr_comp[2, 1] * 2.0 / 3.0, rr_comp[2, 2] * 2.0 / 3.0, 0.0], dtype=float)
        elif first_ring not in (1, 2, 3):
            return np.empty((0, 4), dtype=float), True

        data = np.zeros((rr_comp.shape[0], 4), dtype=float)
        data[:, 0] = rr_comp[:, 1]
        data[:, 1] = rr_comp[:, 2]
        data[:, 2] = rr_comp[:, 3]

        # adjust_raw == 1 path
        data_adj = np.array(data, copy=True)
        nrows = data_adj.shape[0]
        for sn in (0, 1):
            for k in range(1, nrows - 1):
                this_r = float(data[k, sn])
                if this_r <= x:
                    continue
                if this_r >= float(data[nrows - 1, sn] - x):
                    continue

                r1_candidates = np.where(data[:, sn] > (this_r - x))[0]
                r2_candidates = np.where(data[:, sn] < (this_r + x))[0]
                if r1_candidates.size == 0 or r2_candidates.size == 0:
                    continue

                r1 = int(np.min(r1_candidates))
                r2 = int(np.max(r2_candidates))
                r0 = int(k)
                dnr1 = r0 - r1
                dnr2 = r2 - r0
                if dnr1 > dnr2:
                    r1 = r0 - dnr2
                elif dnr1 < dnr2:
                    r2 = r0 + dnr1
                if r2 <= r1:
                    continue

                mean_window = float(np.mean(data[r1 : r2 + 1, sn]))
                dr_local = data[r1 + 1 : r2 + 1, sn] - data[r1:r2, sn]
                mean_dr = float(np.mean(dr_local)) if dr_local.size else 0.0
                data_adj[k, sn] = mean_window + mean_dr * np.random.randn() * sf

        if nrows > 20:
            mdr_15_to_20 = float(np.mean(data_adj[19, 0:2] - data_adj[14, 0:2]) / 5.0)
            if np.isfinite(mdr_15_to_20) and mdr_15_to_20 > 1e-12:
                add_nbr = float(np.mean(data_adj[0, 0:2]) / mdr_15_to_20)
                if np.isfinite(add_nbr) and add_nbr > 1.2:
                    nbr_to_add = int(np.round(add_nbr)) - 1
                    if nbr_to_add > 0:
                        lines = np.zeros((nbr_to_add, 4), dtype=float)
                        base = (
                            np.arange(1, nbr_to_add + 1, dtype=float).reshape(-1, 1)
                            * np.mean(data_adj[0, 0:2])
                            / float(nbr_to_add + 1)
                        )
                        lines[:, 0:2] = base + mdr_15_to_20 * np.random.randn(nbr_to_add, 2) * sf
                        lines[:, 3] = 2.0
                        data_adj = np.vstack([lines, data_adj])

        return data_adj, False

    def _sample_log_ring_radii(self, profile: np.ndarray) -> np.ndarray:
        if profile.size < 2 or not np.isfinite(profile[0]) or abs(float(profile[0])) < 1e-9:
            raise RuntimeError("Invalid sampled radial profile for random cross-section generation.")

        # MATLAB: argmin(abs(R_all - radiuses(1) + rands(391,1)*0.01))
        jitter = np.random.rand(self._r_all.shape[0]) * 0.01
        idx = int(np.argmin(np.abs(self._r_all - float(profile[0]) + jitter)))

        data_this_log, problematic = self._give_data_for_one_log(log_nr_one_based=idx + 1)
        if problematic or data_this_log.size == 0:
            raise RuntimeError("Failed to derive ring radii from selected Harald log.")

        radiuses_mns_real_log = np.mean(data_this_log[:, 0:2], axis=1)
        if radiuses_mns_real_log.size <= 0:
            raise RuntimeError("Selected log produced no valid ring radii.")

        rr = np.outer(radiuses_mns_real_log, profile / float(profile[0]))
        if not np.all(np.isfinite(rr)):
            raise RuntimeError("Generated ring-radius field contains invalid values.")
        return rr

    def generate_ring_splines(self) -> Sequence[CubicSpline]:
        last_error: Exception | None = None
        for _ in range(self._MAX_SAMPLE_RETRIES):
            try:
                profile = self._sample_profile()
                rr = self._sample_log_ring_radii(profile)
                n_rings, n_angles = rr.shape
                if n_rings <= 0 or n_angles <= 3:
                    raise RuntimeError("Insufficient ring data for spline generation.")

                theta = np.linspace(-np.pi, np.pi, n_angles, dtype=float)
                splines: List[CubicSpline] = []
                for k in range(n_rings):
                    rk = np.maximum(rr[k, :], 1e-6)
                    splines.append(CubicSpline(theta, rk, bc_type="not-a-knot", extrapolate=True))

                if splines:
                    return splines
            except Exception as exc:  # retry on stochastic/pathological sample failures
                last_error = exc
                continue

        if last_error is not None:
            raise RuntimeError(
                f"Random log cross-section generation failed after {self._MAX_SAMPLE_RETRIES} retries: {last_error}"
            ) from last_error
        raise RuntimeError("Random log cross-section generation failed.")
