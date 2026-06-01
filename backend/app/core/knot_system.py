import numpy as np
import scipy.io
import os
from pathlib import Path
from typing import List
from .config import BoardConfig
from .array_backend import get_xp, gpu_enabled, to_numpy
from .knot_sequence_model import (
    default_training_mat_path,
    sample_random_knot_log,
    resolve_knot_sequence_runtime_info,
)
from .random_log_cross_section import RandomLogCrossSectionGenerator
try:
    from scipy.spatial import cKDTree
except Exception:
    cKDTree = None

class KnotSystem:
    _mat_cache = {}
    _cross_section_generator = None
    _OVERRIDE_C1_VALUE = -1.458e-3
    _OVERRIDE_AX100_MIN = 32.7
    _OVERRIDE_AX100_MAX = 55.3

    @classmethod
    def _data_path(cls):
        return os.path.join(os.path.dirname(__file__), '../../../data')

    @classmethod
    def _load_mat_cached(cls, mat_path: str):
        if mat_path not in cls._mat_cache:
            cls._mat_cache[mat_path] = scipy.io.loadmat(mat_path)
        return cls._mat_cache[mat_path]

    @classmethod
    def _get_cross_section_generator(cls) -> RandomLogCrossSectionGenerator:
        if cls._cross_section_generator is None:
            cls._cross_section_generator = RandomLogCrossSectionGenerator.from_default_paths()
        return cls._cross_section_generator

    def __init__(self, config: BoardConfig):
        self.config = config
        self.xp = get_xp(config.use_gpu)
        self.gpu_enabled = gpu_enabled(self.xp)
        self.n_knots = 0
        self.splines = []
        
        # Knot Parameters (NumPy arrays)
        self.th0 = None
        self.z0 = None
        self.c1 = None
        self.c2 = None
        self.k = None
        self.kp = None
        self.Abump = None
        self.Bbump = None
        self.Aexp = None
        self.a1 = None
        self.a2 = None
        self.a3 = None
        self.a4 = None
        self.RL = None
        self.RD = None
        self.knot_sequence_info = {
            "mode": "uninitialized",
            "used_pytorch_checkpoint": False,
            "allow_fallback": True,
            "checkpoint_path": "",
            "training_data_path": "",
            "load_note": "",
            "sample_length": 0,
            "slot_count": 0,
            "dz_mm": 0.0,
            "z_min_mm": 0.0,
            "slot_tokens": [],
            "slot_has_knot": [],
            "slot_knot_ids": [],
        }
        
        # Function Handles
        self.crook_x = lambda y: 0
        self.crook_y = lambda y: 0
        self.taper = lambda y: 0
        self._crook_lengths = np.zeros((0,), dtype=np.float64)
        self._crook_amplitudes = np.zeros((0,), dtype=np.float64)
        self._crook_shifts = np.zeros((0,), dtype=np.float64)
        self._crook_dir_x = np.zeros((0,), dtype=np.float64)
        self._crook_dir_y = np.zeros((0,), dtype=np.float64)
        self._taper_coeff_random = 0.0
        self.geometry_randomization_info = {}
        
        self.process(config)

    def _load_random_splines(self):
        generator = self._get_cross_section_generator()
        self.splines = list(generator.generate_ring_splines())

    def _store_knot_parameters(
        self,
        k_th0_deg,
        k_z0,
        k_c1,
        k_c2,
        k_k,
        k_kp,
        k_Abump,
        k_Bbump,
        k_Aexp,
        k_a1,
        k_a2,
        k_a3,
        k_a4,
        k_RL,
        k_RD,
        *,
        enforce_generated_rl_rd_min_gap: bool = False,
    ):
        self.n_knots = len(k_th0_deg)
        if self.n_knots == 0:
            self._set_empty_knot_parameters()
            return 0

        curr_RL = np.asarray(k_RL, dtype=float).copy()
        curr_RD = np.asarray(k_RD, dtype=float).copy()

        adjusted_rl_count = 0
        min_gap_mm = max(
            0.0,
            self._finite_or_default(
                getattr(self.config, "knot_generator_min_rd_minus_rl_mm", 30.0),
                30.0,
            ),
        )
        if enforce_generated_rl_rd_min_gap and min_gap_mm > 0.0:
            # Keep generated dead-knot transition zone wide enough by shifting RL inward
            # when RD - RL is below the configured threshold.
            narrow_mask = (curr_RD - curr_RL) < min_gap_mm
            if np.any(narrow_mask):
                curr_RL[narrow_mask] = curr_RD[narrow_mask] - min_gap_mm
                adjusted_rl_count = int(np.count_nonzero(narrow_mask))

        fallback_gap = max(1.0, min_gap_mm) if enforce_generated_rl_rd_min_gap else 10.0
        mask = curr_RL >= curr_RD
        curr_RD[mask] = curr_RL[mask] + fallback_gap

        def to_shape(arr):
            return self.xp.asarray(np.asarray(arr, dtype=float)).reshape(1, 1, 1, self.n_knots)

        self.th0 = to_shape(np.asarray(k_th0_deg, dtype=float) * np.pi / 180.0)
        self.z0 = to_shape(k_z0)
        self.c1 = to_shape(k_c1)
        self.c2 = to_shape(k_c2)
        self.k = to_shape(k_k)
        self.kp = to_shape(k_kp)
        self.Abump = to_shape(k_Abump)
        self.Bbump = to_shape(k_Bbump)
        self.Aexp = to_shape(k_Aexp)
        self.a1 = to_shape(k_a1)
        self.a2 = to_shape(k_a2)
        self.a3 = to_shape(k_a3)
        self.a4 = to_shape(k_a4)
        self.RL = to_shape(curr_RL)
        self.RD = to_shape(curr_RD)
        return adjusted_rl_count

    @classmethod
    def _sample_override_knot_axis_coefficients(cls, knot_count: int) -> tuple[np.ndarray, np.ndarray]:
        n_knots = max(0, int(knot_count))
        if n_knots == 0:
            return np.zeros((0,), dtype=np.float64), np.zeros((0,), dtype=np.float64)

        ax100 = np.random.uniform(
            low=float(cls._OVERRIDE_AX100_MIN),
            high=float(cls._OVERRIDE_AX100_MAX),
            size=(n_knots,),
        )
        k_c1 = np.full((n_knots,), float(cls._OVERRIDE_C1_VALUE), dtype=np.float64)
        k_c2 = (9.7e-3 * ax100) + 0.1725
        return k_c1, np.asarray(k_c2, dtype=np.float64)

    @staticmethod
    def _apply_dictionary_jitter(data: np.ndarray, jitter_scale: float) -> np.ndarray:
        arr = np.asarray(data, dtype=float).copy()
        if arr.size == 0:
            return arr
        s = max(0.0, float(jitter_scale))
        if s <= 0.0:
            return arr

        n = arr.shape[0]
        arr[:, 0] += np.random.normal(0.0, 6.0 * s, size=n)  # th0
        arr[:, 1] *= 1.0 + np.random.normal(0.0, 0.05 * s, size=n)  # RL
        arr[:, 2] *= 1.0 + np.random.normal(0.0, 0.05 * s, size=n)  # RD
        arr[:, 3:7] *= 1.0 + np.random.normal(0.0, 0.08 * s, size=(n, 4))  # a1..a4
        arr[:, 7] += np.random.normal(0.0, 0.02 * s, size=n)  # c1
        arr[:, 8] += np.random.normal(0.0, 0.02 * s, size=n)  # c2

        arr[:, 1] = np.maximum(arr[:, 1], 1.0)
        arr[:, 2] = np.maximum(arr[:, 2], arr[:, 1] + 1.0)
        return arr

    @staticmethod
    def _finite_or_default(value, default: float) -> float:
        try:
            val = float(value)
        except (TypeError, ValueError):
            return float(default)
        return val if np.isfinite(val) else float(default)

    @staticmethod
    def _vector_with_length(values, count: int, default: float = 0.0) -> np.ndarray:
        n = max(0, int(count))
        if n == 0:
            return np.zeros((0,), dtype=np.float64)

        if isinstance(values, (list, tuple, np.ndarray)):
            arr = np.asarray(values, dtype=np.float64).reshape(-1)
        else:
            arr = np.zeros((0,), dtype=np.float64)

        if arr.size < n:
            arr = np.pad(arr, (0, n - arr.size), mode="constant", constant_values=float(default))
        elif arr.size > n:
            arr = arr[:n]

        bad = ~np.isfinite(arr)
        if np.any(bad):
            arr = arr.copy()
            arr[bad] = float(default)
        return arr.astype(np.float64, copy=False)

    @staticmethod
    def _sanitize_order_list(values) -> List[int]:
        out: List[int] = []
        if not isinstance(values, (list, tuple, np.ndarray)):
            return out
        for raw in values:
            try:
                order = int(raw)
            except (TypeError, ValueError):
                continue
            if order < 1:
                continue
            out.append(int(order))
        return out

    def _evaluate_crook_displacement(self, z_vals, direction_weights_np: np.ndarray):
        xp = self.xp
        z = xp.asarray(z_vals, dtype=float)
        if self._crook_lengths.size == 0:
            return xp.zeros_like(z, dtype=float)

        lengths = xp.asarray(self._crook_lengths, dtype=float)
        amps = xp.asarray(self._crook_amplitudes, dtype=float)
        shifts = xp.asarray(self._crook_shifts, dtype=float)
        weights = xp.asarray(direction_weights_np, dtype=float)
        arg = (2.0 * xp.pi * (z[..., xp.newaxis] + shifts)) / lengths
        wave = xp.sin(arg)
        return xp.sum((weights * amps) * wave, axis=-1)

    def _configure_geometry_deformation(self, p: BoardConfig) -> None:
        randomize = bool(getattr(p, "randomize_crook_taper", True))
        p_terms = max(1, int(self._finite_or_default(getattr(p, "crook_component_count", 8), 8)))
        shift_max_mm = max(
            0.0,
            self._finite_or_default(getattr(p, "crook_shift_max_mm", 8000.0), 8000.0),
        )
        crook_scale_max = max(
            0.0,
            self._finite_or_default(getattr(p, "random_crook_scale_max", 1.0), 1.0),
        )
        taper_max = max(
            0.0,
            self._finite_or_default(getattr(p, "random_taper_max", 1.0 / 160.0), 1.0 / 160.0),
        )
        theta_min_deg = self._finite_or_default(
            getattr(p, "random_crook_theta_min_deg", 0.0),
            0.0,
        )
        theta_max_deg = self._finite_or_default(
            getattr(p, "random_crook_theta_max_deg", 360.0),
            360.0,
        )
        if theta_max_deg < theta_min_deg:
            theta_min_deg, theta_max_deg = theta_max_deg, theta_min_deg

        random_extra_orders = self._sanitize_order_list(
            getattr(p, "random_crook_extra_orders", []),
        )
        base_orders = list(range(1, p_terms + 1))
        component_orders: List[int]
        if randomize:
            component_orders = base_orders + random_extra_orders
        else:
            raw_manual_orders = self._sanitize_order_list(
                getattr(p, "manual_crook_orders", []),
            )
            if raw_manual_orders:
                component_orders = [
                    int(raw_manual_orders[idx]) if idx < len(raw_manual_orders) else int(idx + 1)
                    for idx in range(p_terms)
                ]
            else:
                component_orders = list(base_orders)

        component_orders_arr = np.asarray(component_orders, dtype=np.float64).reshape(-1)
        term_count = int(component_orders_arr.size)
        lengths = (2.0 ** (5.0 - component_orders_arr)) * 1000.0

        raw_amp_values = self._vector_with_length(
            getattr(p, "random_crook_amplitude_max", []),
            max(0, len(getattr(p, "random_crook_amplitude_max", []) or [])),
            0.0,
        )
        amp_max_default = np.zeros((term_count,), dtype=np.float64)
        copy_n = min(int(raw_amp_values.size), term_count)
        if copy_n > 0:
            amp_max_default[:copy_n] = np.maximum(raw_amp_values[:copy_n], 0.0)
        if int(raw_amp_values.size) == 0:
            # Legacy default: when no per-component amplitudes are given, use L_i/320.
            amp_max_default = lengths / 320.0
        elif copy_n < term_count:
            # For user-added random orders without explicit amplitudes, use L_i/320 defaults.
            amp_max_default[copy_n:] = lengths[copy_n:] / 320.0
        amp_max = np.maximum(amp_max_default, 0.0) * crook_scale_max

        if randomize:
            amplitudes = np.random.uniform(0.0, amp_max, size=term_count).astype(np.float64)
            shifts = np.random.uniform(0.0, shift_max_mm, size=term_count).astype(np.float64)
            theta = np.random.uniform(
                np.deg2rad(theta_min_deg),
                np.deg2rad(theta_max_deg),
                size=term_count,
            ).astype(np.float64)
            sampled_taper = float(np.random.uniform(0.0, taper_max)) if taper_max > 0.0 else 0.0
        else:
            amplitudes = self._vector_with_length(
                getattr(p, "manual_crook_amplitudes", []),
                term_count,
                0.0,
            )
            amplitudes = np.maximum(amplitudes, 0.0)
            shifts = self._vector_with_length(
                getattr(p, "manual_crook_shifts_mm", []),
                term_count,
                0.0,
            )
            theta_deg = self._vector_with_length(
                getattr(p, "manual_crook_thetas_deg", []),
                term_count,
                0.0,
            )
            theta = np.deg2rad(theta_deg).astype(np.float64, copy=False)
            sampled_taper = 0.0

        self._crook_lengths = lengths
        self._crook_amplitudes = amplitudes
        self._crook_shifts = shifts
        self._crook_dir_x = np.sin(theta).astype(np.float64)
        self._crook_dir_y = np.cos(theta).astype(np.float64)
        self._taper_coeff_random = float(sampled_taper)

        legacy_crook_x = self._finite_or_default(getattr(p, "manual_crook_x_coeff", 0.0), 0.0)
        legacy_crook_y = self._finite_or_default(getattr(p, "manual_crook_y_coeff", 0.0), 0.0)
        legacy_taper = self._finite_or_default(getattr(p, "manual_taper_coeff", 0.0), 0.0)
        active_legacy_crook_x = (0.0 if randomize else legacy_crook_x)
        active_legacy_crook_y = (0.0 if randomize else legacy_crook_y)
        taper_total = self._taper_coeff_random if randomize else legacy_taper

        self.crook_x = lambda z_vals: (
            self._evaluate_crook_displacement(z_vals, self._crook_dir_x)
            + active_legacy_crook_x * self.xp.asarray(z_vals, dtype=float) ** 2
        )
        self.crook_y = lambda z_vals: (
            self._evaluate_crook_displacement(z_vals, self._crook_dir_y)
            + active_legacy_crook_y * self.xp.asarray(z_vals, dtype=float) ** 2
        )
        self.taper = lambda z_vals: taper_total * self.xp.asarray(z_vals, dtype=float)

        self.geometry_randomization_info = {
            "mode": ("random" if randomize else "manual"),
            "randomize_crook_taper": bool(randomize),
            "crook_component_count": int(term_count),
            "base_crook_component_count": int(p_terms),
            "crook_shift_max_mm": float(shift_max_mm),
            "random_crook_scale_max": float(crook_scale_max),
            "random_crook_amplitude_max": amp_max_default.tolist(),
            "random_crook_extra_orders": [int(v) for v in random_extra_orders],
            "random_crook_theta_min_deg": float(theta_min_deg),
            "random_crook_theta_max_deg": float(theta_max_deg),
            "random_taper_max": float(taper_max),
            "sampled_taper_coeff": float(self._taper_coeff_random),
            "component_orders": [int(v) for v in component_orders],
            "component_amplitudes": amplitudes.tolist(),
            "component_shifts_mm": shifts.tolist(),
            "component_thetas_deg": np.rad2deg(theta).tolist(),
            "legacy_manual_crook_x_coeff": float(legacy_crook_x),
            "legacy_manual_crook_y_coeff": float(legacy_crook_y),
            "active_legacy_manual_crook_x_coeff": float(active_legacy_crook_x),
            "active_legacy_manual_crook_y_coeff": float(active_legacy_crook_y),
            "legacy_manual_taper_coeff": float(legacy_taper),
            "effective_taper_coeff": float(taper_total),
        }

    def _process_manual_input_knots(self, p: BoardConfig):
        knots = p.resolved_input_knots()
        if len(knots) == 0:
            self._set_empty_knot_parameters()
            return

        override_c1_c2 = bool(getattr(p, "knot_sequence_override_c1_c2", False))
        if override_c1_c2:
            k_c1, k_c2 = self._sample_override_knot_axis_coefficients(len(knots))
        else:
            k_c1 = np.asarray([k.c1 for k in knots], dtype=float)
            k_c2 = np.asarray([k.c2 for k in knots], dtype=float)

        self._store_knot_parameters(
            [k.th0_deg for k in knots],
            [k.z0 for k in knots],
            k_c1,
            k_c2,
            [k.k for k in knots],
            [k.kp for k in knots],
            [k.Abump for k in knots],
            [k.Bbump for k in knots],
            [k.Aexp for k in knots],
            [k.a1 for k in knots],
            [k.a2 for k in knots],
            [k.a3 for k in knots],
            [k.a4 for k in knots],
            [k.RL for k in knots],
            [k.RD for k in knots],
        )

    @staticmethod
    def _knot_lx_profile(
        x_vals: np.ndarray,
        *,
        a1: float,
        a2: float,
        a3: float,
        a4: float,
        rl: float,
        rd: float,
        dead_knots: bool,
    ) -> np.ndarray:
        x = np.asarray(x_vals, dtype=np.float64)
        lx = (a1 * x**4) + (a2 * x**3) + (a3 * x**2) + (a4 * x)

        if dead_knots:
            rl_v = float(rl)
            rd_v = float(rd)
            if rd_v > rl_v:
                lx_rl = (
                    a1 * rl_v**4
                    + a2 * rl_v**3
                    + a3 * rl_v**2
                    + a4 * rl_v
                )
                mask = (x >= rl_v) & (x <= rd_v)
                if np.any(mask):
                    t = (x[mask] - rl_v) / (rd_v - rl_v)
                    lx = lx.copy()
                    lx[mask] = lx_rl * (1.0 - t**2)
                lx[x > rd_v] = 0.0
            else:
                lx[x > rd_v] = 0.0

        return np.maximum(lx, 0.0)

    def _sample_knot_axis_envelope(
        self,
        *,
        th0_deg: float,
        z0: float,
        c1: float,
        c2: float,
        a1: float,
        a2: float,
        a3: float,
        a4: float,
        rl: float,
        rd: float,
        dead_knots: bool,
        sample_step_mm: float,
        min_radius_mm: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        rd_v = max(0.0, float(rd))
        if rd_v <= 0.0:
            return np.empty((0, 3), dtype=np.float64), np.empty((0,), dtype=np.float64)

        step = max(0.5, float(sample_step_mm))
        n_samples = int(np.clip(np.ceil(rd_v / step) + 1, 12, 256))
        xr = np.linspace(0.0, rd_v, n_samples, dtype=np.float64)

        lx = self._knot_lx_profile(
            xr,
            a1=float(a1),
            a2=float(a2),
            a3=float(a3),
            a4=float(a4),
            rl=float(rl),
            rd=rd_v,
            dead_knots=bool(dead_knots),
        )
        radius = 0.5 * lx
        keep = radius > float(min_radius_mm)
        if not np.any(keep):
            return np.empty((0, 3), dtype=np.float64), np.empty((0,), dtype=np.float64)

        xr = xr[keep]
        radius = radius[keep]

        th0 = np.deg2rad(float(th0_deg))
        axis_z = (float(c1) * xr**2) + (float(c2) * xr) + float(z0)
        x_local = xr * np.cos(th0)
        y_local = -xr * np.sin(th0)

        crook_x_vals = np.asarray(to_numpy(self.crook_x(axis_z)), dtype=np.float64).reshape(-1)
        crook_y_vals = np.asarray(to_numpy(self.crook_y(axis_z)), dtype=np.float64).reshape(-1)
        if crook_x_vals.shape[0] != xr.shape[0]:
            crook_x_vals = np.resize(crook_x_vals, xr.shape[0])
        if crook_y_vals.shape[0] != xr.shape[0]:
            crook_y_vals = np.resize(crook_y_vals, xr.shape[0])

        x_world = x_local - crook_x_vals
        y_world = y_local - crook_y_vals
        points = np.column_stack([x_world, y_world, axis_z]).astype(np.float64, copy=False)
        return points, radius.astype(np.float64, copy=False)

    def _count_overlapping_knot_pairs(
        self,
        *,
        k_th0: np.ndarray,
        k_z0: np.ndarray,
        k_c1: np.ndarray,
        k_c2: np.ndarray,
        k_a1: np.ndarray,
        k_a2: np.ndarray,
        k_a3: np.ndarray,
        k_a4: np.ndarray,
        k_RL: np.ndarray,
        k_RD: np.ndarray,
        dead_knots: bool,
        sample_step_mm: float = 2.5,
        min_radius_mm: float = 0.25,
        clearance_mm: float = 0.5,
    ) -> int:
        n_knots = int(len(k_th0))
        if n_knots < 2:
            return 0

        pts_list: list[np.ndarray] = []
        rad_list: list[np.ndarray] = []
        knot_id_list: list[np.ndarray] = []

        for idx in range(n_knots):
            pts, rad = self._sample_knot_axis_envelope(
                th0_deg=float(k_th0[idx]),
                z0=float(k_z0[idx]),
                c1=float(k_c1[idx]),
                c2=float(k_c2[idx]),
                a1=float(k_a1[idx]),
                a2=float(k_a2[idx]),
                a3=float(k_a3[idx]),
                a4=float(k_a4[idx]),
                rl=float(k_RL[idx]),
                rd=float(k_RD[idx]),
                dead_knots=bool(dead_knots),
                sample_step_mm=float(sample_step_mm),
                min_radius_mm=float(min_radius_mm),
            )
            if pts.shape[0] <= 0:
                continue
            pts_list.append(pts)
            rad_list.append(rad)
            knot_id_list.append(np.full((pts.shape[0],), int(idx), dtype=np.int32))

        if len(pts_list) < 2:
            return 0

        points = np.concatenate(pts_list, axis=0)
        radii = np.concatenate(rad_list, axis=0)
        knot_ids = np.concatenate(knot_id_list, axis=0)
        if points.shape[0] < 2 or np.unique(knot_ids).size < 2:
            return 0

        max_radius = float(np.max(radii))
        if max_radius <= 0.0:
            return 0

        # Broad-phase with constant search radius, then exact overlap check with per-point radii.
        if cKDTree is not None:
            tree = cKDTree(points)
            try:
                pairs = tree.query_pairs(r=(2.0 * max_radius + 1e-9), output_type="ndarray")
            except TypeError:
                pair_set = tree.query_pairs(r=(2.0 * max_radius + 1e-9))
                if not pair_set:
                    return 0
                pairs = np.asarray(list(pair_set), dtype=np.int64)
        else:
            n_pts = int(points.shape[0])
            pair_rows: list[tuple[int, int]] = []
            search_r = float(2.0 * max_radius + 1e-9)
            for i in range(n_pts - 1):
                d = np.linalg.norm(points[i + 1:] - points[i], axis=1)
                js = np.where(d <= search_r)[0]
                if js.size <= 0:
                    continue
                for j in js.tolist():
                    pair_rows.append((i, i + 1 + int(j)))
            if len(pair_rows) == 0:
                return 0
            pairs = np.asarray(pair_rows, dtype=np.int64)

        if pairs.size <= 0:
            return 0

        ia = pairs[:, 0]
        ib = pairs[:, 1]
        cross_knot = knot_ids[ia] != knot_ids[ib]
        if not np.any(cross_knot):
            return 0
        ia = ia[cross_knot]
        ib = ib[cross_knot]

        d = np.linalg.norm(points[ia] - points[ib], axis=1)
        overlap_threshold = radii[ia] + radii[ib] - float(clearance_mm)
        overlaps = d < overlap_threshold
        if not np.any(overlaps):
            return 0

        ka = knot_ids[ia[overlaps]].astype(np.int64, copy=False)
        kb = knot_ids[ib[overlaps]].astype(np.int64, copy=False)
        pair_ids = np.column_stack([np.minimum(ka, kb), np.maximum(ka, kb)])
        unique_pairs = np.unique(pair_ids, axis=0)
        return int(unique_pairs.shape[0])

    def process(self, p: BoardConfig):
        self._load_random_splines()
        self._configure_geometry_deformation(p)
        override_c1_c2 = bool(getattr(p, "knot_sequence_override_c1_c2", False))
        override_c1_value = float(self._OVERRIDE_C1_VALUE)
        override_ax100_min = float(self._OVERRIDE_AX100_MIN)
        override_ax100_max = float(self._OVERRIDE_AX100_MAX)

        if p.use_input_knots:
            self.knot_sequence_info = {
                "mode": "manual_input",
                "used_pytorch_checkpoint": False,
                "allow_fallback": False,
                "checkpoint_path": "",
                "training_data_path": "",
                "load_note": (
                    "Manual knots enabled; sequence model not used. c1/c2 override active."
                    if override_c1_c2
                    else "Manual knots enabled; sequence model not used."
                ),
                "sample_length": 0,
                "slot_count": 0,
                "knot_sequence_override_c1_c2": bool(override_c1_c2),
                "knot_sequence_override_c1_value": float(override_c1_value),
                "knot_sequence_override_ax100_range": [
                    float(override_ax100_min),
                    float(override_ax100_max),
                ],
                "dz_mm": 0.0,
                "z_min_mm": 0.0,
                "slot_tokens": [],
                "slot_has_knot": [],
                "slot_knot_ids": [],
            }
            self._process_manual_input_knots(p)
            return

        # Continue with rest of process logic
        data_path = self._data_path()
        configured_training_mat_path = str(getattr(p, "knot_sequence_training_data_path", "") or "").strip()
        if configured_training_mat_path:
            training_mat_path = str(Path(configured_training_mat_path).expanduser().resolve())
        else:
            training_mat_path = str(default_training_mat_path().resolve())

        # Load knots data
        d_knots = self._load_mat_cached(training_mat_path)
        Data_all_or = np.asarray(d_knots['Data_all_or'], dtype=float)
        dz = d_knots['dz'][0, 0]
        token_to_rowid_ptr = d_knots.get('token_to_rowid_ptr', None)
        token_to_rowid_flat = d_knots.get('token_to_rowid_flat', None)
        has_cluster_token_map = False
        if token_to_rowid_ptr is not None and token_to_rowid_flat is not None:
            ptr_arr = np.asarray(token_to_rowid_ptr, dtype=np.int64)
            flat_arr = np.asarray(token_to_rowid_flat, dtype=np.int64).reshape(-1)
            if ptr_arr.ndim == 2 and ptr_arr.shape[1] >= 2:
                token_to_rowid_ptr = ptr_arr
                token_to_rowid_flat = flat_arr
                has_cluster_token_map = True
            else:
                token_to_rowid_ptr = None
                token_to_rowid_flat = None
        
        # Load bad knots
        d_bad = self._load_mat_cached(os.path.join(data_path, 'bad_knots.mat'))
        bad_knots = np.asarray(d_bad['bad_knots'][0], dtype=np.int64)
        bad_knots_set = set(int(v) for v in bad_knots.tolist())
        
        # 1. Generate a fresh random knot sequence that matches board length
        # (floored to dz slots), retrying when sampled knots intersect each other.
        board_length = p.board_length_mm()
        slot_count = max(1, int(board_length / float(dz)))
        checkpoint_path = str(getattr(p, "knot_sequence_checkpoint_path", "") or "")
        allow_fallback = bool(getattr(p, "knot_sequence_allow_fallback", True))
        top_k = max(0, int(getattr(p, "knot_sequence_top_k", 0)))
        top_p = float(getattr(p, "knot_sequence_top_p", 0.8))
        reject_intersections = bool(getattr(p, "knot_sequence_reject_intersections", True))
        max_overlap_attempts = max(
            1,
            int(
                self._finite_or_default(
                    getattr(p, "knot_sequence_intersection_max_attempts", 64),
                    64.0,
                )
            ),
        )
        seq_attempt_budget = max_overlap_attempts if reject_intersections else 1
        jitter_scale = max(0.0, float(getattr(p, "knot_dictionary_jitter", 0.0)))

        runtime_info = resolve_knot_sequence_runtime_info(
            checkpoint_path=checkpoint_path,
            training_mat_path=training_mat_path,
            allow_fallback=allow_fallback,
        )

        z_min, _ = p.z_extent()
        overlap_rejected_attempts = 0
        empty_candidate_attempts = 0
        last_sample_length = 0
        best_payload = None
        selected_payload = None

        for _ in range(seq_attempt_budget):
            log_seq = sample_random_knot_log(
                slot_count=slot_count,
                min_tokens=slot_count,
                extra_tokens=0,
                top_k=top_k,
                top_p=top_p,
                checkpoint_path=checkpoint_path,
                training_mat_path=training_mat_path,
                allow_fallback=allow_fallback,
                seed=None,
            )
            last_sample_length = int(len(log_seq))

            if last_sample_length < slot_count:
                padded = np.zeros((slot_count,), dtype=np.int64)
                padded[:last_sample_length] = np.asarray(log_seq, dtype=np.int64).reshape(-1)
                knot_seq_window = padded
            else:
                knot_seq_window = np.asarray(log_seq[:slot_count], dtype=np.int64).reshape(-1)
            slot_tokens = np.asarray(knot_seq_window, dtype=np.int64).reshape(-1)

            # Map each sequence element to its fixed axial location.
            # Zero elements represent an empty dz slot and are removed later.
            slot_z_pos = z_min + (np.arange(1, len(knot_seq_window) + 1) * float(dz))

            knot_ids: list[int] = []
            knot_slot_indices: list[int] = []
            knot_z0_list: list[float] = []
            for slot_idx_raw, (tok_raw, slot_z) in enumerate(zip(knot_seq_window, slot_z_pos)):
                token_id = int(tok_raw)
                if token_id == 0:
                    continue

                if has_cluster_token_map:
                    if token_id < 0 or token_id >= token_to_rowid_ptr.shape[0]:
                        continue
                    row_start = int(token_to_rowid_ptr[token_id, 0])
                    row_len = int(token_to_rowid_ptr[token_id, 1])
                    if row_len <= 0:
                        continue
                    row_end = row_start + row_len
                    if row_start < 0 or row_end > token_to_rowid_flat.shape[0]:
                        continue
                    members = token_to_rowid_flat[row_start:row_end]
                    if members.size <= 0:
                        continue
                    knot_id = int(members[np.random.randint(0, members.size)])
                else:
                    knot_id = token_id

                if knot_id <= 0 or knot_id >= Data_all_or.shape[0]:
                    continue
                if knot_id in bad_knots_set:
                    continue
                knot_ids.append(knot_id)
                knot_slot_indices.append(int(slot_idx_raw))
                knot_z0_list.append(float(slot_z))

            if len(knot_ids) == 0:
                empty_candidate_attempts += 1
                continue

            knot_ids_arr = np.asarray(knot_ids, dtype=np.int64)
            knot_slot_indices_arr = np.asarray(knot_slot_indices, dtype=np.int64)
            knot_z0 = np.asarray(knot_z0_list, dtype=float)
            Data = Data_all_or[knot_ids_arr]
            if jitter_scale > 0.0:
                Data = self._apply_dictionary_jitter(Data, jitter_scale)

            # 2. Calculate L100 and apply range filter.
            L100 = (
                Data[:, 3] * 100**4
                + Data[:, 4] * 100**3
                + Data[:, 5] * 100**2
                + Data[:, 6] * 100
            )
            valid_mask = (L100 >= p.L100_min) & (L100 <= p.L100_max)
            Data = Data[valid_mask]
            L100 = L100[valid_mask]
            knot_ids_arr = knot_ids_arr[valid_mask]
            knot_slot_indices_arr = knot_slot_indices_arr[valid_mask]
            knot_z0 = knot_z0[valid_mask]

            if Data.shape[0] == 0:
                empty_candidate_attempts += 1
                continue

            # 3. Extract parameters for overlap check and later storage.
            k_th0 = np.asarray(Data[:, 0], dtype=float)
            k_RL = np.asarray(Data[:, 1], dtype=float)
            k_RD = np.asarray(Data[:, 2], dtype=float)
            k_a1 = np.asarray(Data[:, 3], dtype=float)
            k_a2 = np.asarray(Data[:, 4], dtype=float)
            k_a3 = np.asarray(Data[:, 5], dtype=float)
            k_a4 = np.asarray(Data[:, 6], dtype=float)
            if override_c1_c2:
                k_c1, k_c2 = self._sample_override_knot_axis_coefficients(Data.shape[0])
            else:
                k_c1 = np.asarray(Data[:, 7], dtype=float)
                k_c2 = np.asarray(Data[:, 8], dtype=float)
            k_z0 = np.asarray(knot_z0, dtype=float)
            slot_has_knot = np.zeros((int(slot_count),), dtype=np.uint8)
            slot_knot_ids = np.zeros((int(slot_count),), dtype=np.int64)
            if k_z0.size > 0:
                slot_idx = np.asarray(knot_slot_indices_arr, dtype=np.int64).reshape(-1)
                valid_slot_mask = (slot_idx >= 0) & (slot_idx < int(slot_count))
                if np.any(valid_slot_mask):
                    slot_has_knot[slot_idx[valid_slot_mask]] = 1
                    slot_knot_ids[slot_idx[valid_slot_mask]] = knot_ids_arr[valid_slot_mask]

            overlap_pair_count = self._count_overlapping_knot_pairs(
                k_th0=k_th0,
                k_z0=k_z0,
                k_c1=k_c1,
                k_c2=k_c2,
                k_a1=k_a1,
                k_a2=k_a2,
                k_a3=k_a3,
                k_a4=k_a4,
                k_RL=k_RL,
                k_RD=k_RD,
                dead_knots=bool(getattr(p, "dead_knots", False)),
                sample_step_mm=2.5,
                min_radius_mm=0.25,
                clearance_mm=0.5,
            )

            candidate = {
                "sample_length": int(last_sample_length),
                "Data": Data,
                "L100": L100,
                "k_th0": k_th0,
                "k_RL": k_RL,
                "k_RD": k_RD,
                "k_a1": k_a1,
                "k_a2": k_a2,
                "k_a3": k_a3,
                "k_a4": k_a4,
                "k_c1": k_c1,
                "k_c2": k_c2,
                "k_z0": k_z0,
                "slot_tokens": slot_tokens,
                "slot_has_knot": slot_has_knot,
                "slot_knot_ids": slot_knot_ids,
                "dz_mm": float(dz),
                "z_min_mm": float(z_min),
                "overlap_pair_count": int(overlap_pair_count),
            }
            if best_payload is None or int(candidate["overlap_pair_count"]) < int(best_payload["overlap_pair_count"]):
                best_payload = candidate

            if reject_intersections and int(overlap_pair_count) > 0:
                overlap_rejected_attempts += 1
                continue

            selected_payload = candidate
            break

        if selected_payload is None:
            selected_payload = best_payload
        if selected_payload is None:
            self.knot_sequence_info = {
                **runtime_info,
                "sample_length": int(last_sample_length),
                "slot_count": int(slot_count),
                "dictionary_tokenization": ("clustered" if has_cluster_token_map else "direct"),
                "dictionary_token_count": (
                    int(max(0, token_to_rowid_ptr.shape[0] - 1))
                    if has_cluster_token_map
                    else int(max(0, Data_all_or.shape[0] - 1))
                ),
                "dictionary_jitter": float(jitter_scale),
                "knot_overlap_rejection_enabled": bool(reject_intersections),
                "knot_overlap_attempt_limit": int(seq_attempt_budget),
                "knot_overlap_rejected_attempts": int(overlap_rejected_attempts),
                "knot_overlap_empty_attempts": int(empty_candidate_attempts),
                "knot_overlap_pair_count_final": 0,
                "knot_overlap_resolved": True,
                "knot_overlap_fallback_used": False,
                "knot_sequence_override_c1_c2": bool(override_c1_c2),
                "knot_sequence_override_c1_value": float(override_c1_value),
                "knot_sequence_override_ax100_range": [
                    float(override_ax100_min),
                    float(override_ax100_max),
                ],
                "dz_mm": float(dz),
                "z_min_mm": float(z_min),
                "slot_tokens": [0] * int(slot_count),
                "slot_has_knot": [0] * int(slot_count),
                "slot_knot_ids": [0] * int(slot_count),
            }
            self._set_empty_knot_parameters()
            return

        k_th0 = np.asarray(selected_payload["k_th0"], dtype=float)
        k_RL = np.asarray(selected_payload["k_RL"], dtype=float)
        k_RD = np.asarray(selected_payload["k_RD"], dtype=float)
        k_a1 = np.asarray(selected_payload["k_a1"], dtype=float)
        k_a2 = np.asarray(selected_payload["k_a2"], dtype=float)
        k_a3 = np.asarray(selected_payload["k_a3"], dtype=float)
        k_a4 = np.asarray(selected_payload["k_a4"], dtype=float)
        k_c1 = np.asarray(selected_payload["k_c1"], dtype=float)
        k_c2 = np.asarray(selected_payload["k_c2"], dtype=float)
        k_z0 = np.asarray(selected_payload["k_z0"], dtype=float)
        L100 = np.asarray(selected_payload["L100"], dtype=float)
        final_overlap_pair_count = int(selected_payload["overlap_pair_count"])

        self.knot_sequence_info = {
            **runtime_info,
            "sample_length": int(selected_payload["sample_length"]),
            "slot_count": int(slot_count),
            "dictionary_tokenization": ("clustered" if has_cluster_token_map else "direct"),
            "dictionary_token_count": (
                int(max(0, token_to_rowid_ptr.shape[0] - 1))
                if has_cluster_token_map
                else int(max(0, Data_all_or.shape[0] - 1))
            ),
            "dictionary_jitter": float(jitter_scale),
            "knot_overlap_rejection_enabled": bool(reject_intersections),
            "knot_overlap_attempt_limit": int(seq_attempt_budget),
            "knot_overlap_rejected_attempts": int(overlap_rejected_attempts),
            "knot_overlap_empty_attempts": int(empty_candidate_attempts),
            "knot_overlap_pair_count_final": int(final_overlap_pair_count),
            "knot_overlap_resolved": bool(final_overlap_pair_count <= 0),
            "knot_overlap_fallback_used": bool(reject_intersections and final_overlap_pair_count > 0),
            "knot_sequence_override_c1_c2": bool(override_c1_c2),
            "knot_sequence_override_c1_value": float(override_c1_value),
            "knot_sequence_override_ax100_range": [
                float(override_ax100_min),
                float(override_ax100_max),
            ],
            "dz_mm": float(selected_payload.get("dz_mm", float(dz))),
            "z_min_mm": float(selected_payload.get("z_min_mm", float(z_min))),
            "slot_tokens": np.asarray(
                selected_payload.get("slot_tokens", np.zeros((int(slot_count),), dtype=np.int64)),
                dtype=np.int64,
            ).reshape(-1).astype(int).tolist(),
            "slot_has_knot": np.asarray(
                selected_payload.get("slot_has_knot", np.zeros((int(slot_count),), dtype=np.uint8)),
                dtype=np.uint8,
            ).reshape(-1).astype(int).tolist(),
            "slot_knot_ids": np.asarray(
                selected_payload.get("slot_knot_ids", np.zeros((int(slot_count),), dtype=np.int64)),
                dtype=np.int64,
            ).reshape(-1).astype(int).tolist(),
        }
        
        # 4. Derived parameters (keep th0 from dictionary; jitter may already have adjusted it)
        n_knots = int(len(k_th0))
        ones = np.ones(n_knots)
        k_Bbump = 2 * ones
        k_Abump = np.abs(0.0217 * L100 - 0.2)
        k_Aexp  = 0.0056 * L100 + 1.96
        # k_Ax100 not used?
        k_k     = 0.99 * ones
        k_kp    = 0.95 * ones
        
        # 5. Store Parameters (reshape to 1x1x1xN for broadcasting with 3D grid)
        adjusted_rl_count = self._store_knot_parameters(
            k_th0,
            k_z0,
            k_c1,
            k_c2,
            k_k,
            k_kp,
            k_Abump,
            k_Bbump,
            k_Aexp,
            k_a1,
            k_a2,
            k_a3,
            k_a4,
            k_RL,
            k_RD,
            enforce_generated_rl_rd_min_gap=True,
        )
        self.knot_sequence_info["knot_generator_rl_adjusted_count"] = int(adjusted_rl_count)
        self.knot_sequence_info["knot_generator_min_rd_minus_rl_mm"] = float(
            max(
                0.0,
                self._finite_or_default(
                    getattr(p, "knot_generator_min_rd_minus_rl_mm", 30.0),
                    30.0,
                ),
            )
        )
        
    def _set_empty_knot_parameters(self):
        self.n_knots = 0
        empty = self.xp.zeros((1, 1, 1, 0), dtype=float)
        self.th0 = empty
        self.z0 = empty
        self.c1 = empty
        self.c2 = empty
        self.k = empty
        self.kp = empty
        self.Abump = empty
        self.Bbump = empty
        self.Aexp = empty
        self.a1 = empty
        self.a2 = empty
        self.a3 = empty
        self.a4 = empty
        self.RL = empty
        self.RD = empty

    def generate_dummy_knot(self):
        # Backward-compatible alias: keep ring splines and clear knot parameters.
        self._set_empty_knot_parameters()

    def calculate_influence(self, x_grid, y_grid, z_grid, Ro, Ri0, flags):
        xp = self.xp
        # Model coordinates:
        # x_grid: Width
        # y_grid: Thickness
        # z_grid: Length (longitudinal axis used for crook/taper)

        # Expand dims for broadcasting against knots
        # Grid: (Ny, Nx, Nz) -> (Ny, Nx, Nz, 1)
        x_board = xp.asarray(x_grid)[..., xp.newaxis]
        y_board = xp.asarray(y_grid)[..., xp.newaxis]
        z_board = xp.asarray(z_grid)[..., xp.newaxis]
        Ro = xp.asarray(Ro)[..., xp.newaxis]
        
        # Unpack flags
        get_knots = flags.get('get_knots', False)
        include_knot_dev = flags.get('include_knot_dev', True)
        dead_knots = flags.get('dead_knots', False)
        
        # 1. Transform transverse coordinates by crook/taper along board length.
        x_transverse = x_board + self.crook_x(z_board)
        y_transverse = y_board + self.crook_y(z_board)
        Ro_mod = Ro - self.taper(z_board)

        # No-knot mode: keep ring/crook/taper field while returning empty knot info.
        if int(getattr(self, "n_knots", 0) or 0) <= 0:
            Ro_base = xp.maximum(Ro_mod[..., 0], 1.0)
            gg_base = x_transverse[..., 0]**2 + y_transverse[..., 0]**2 - Ro_base**2
            info = {}
            if bool(flags.get('get_knots', False)):
                K_empty = xp.full_like(gg_base, xp.nan, dtype=float)
                info['K'] = K_empty
                info['K_live'] = K_empty
                info['K_dead'] = K_empty
            return gg_base, info

        # 2. Rotate the crook-corrected transverse point directly into the knot frame.
        # Equivalent to r*cos(phi+th0), r*sin(phi+th0), but avoids the polar round trip.
        cos_th0 = xp.cos(self.th0)
        sin_th0 = xp.sin(self.th0)
        radial_coord = x_transverse * cos_th0 - y_transverse * sin_th0
        tangential_coord = x_transverse * sin_th0 + y_transverse * cos_th0

        # 3. Knot axis and deviation metric.
        knot_axis_z = self.c1 * radial_coord**2 + self.c2 * radial_coord + self.z0
        longitudinal_offset = z_board - knot_axis_z

        term_ang = xp.arctan2(tangential_coord, radial_coord)**2
        p = xp.sqrt(
            longitudinal_offset**2
            + 1.0 / self.kp**2 * (radial_coord**2 + tangential_coord**2) * term_ang
        )
        
        pmin = float(getattr(self.config, "soft_clamp_pmin", 2.0))
        al = float(getattr(self.config, "soft_clamp_alpha", 1.0))
        # Smooth blending (numerically stable):
        # p_soft = (p*exp(a*p) + pmin*exp(a*pmin)) / (exp(a*p) + exp(a*pmin))
        #       = sigmoid(a*(p-pmin))*p + (1-sigmoid(a*(p-pmin)))*pmin
        d = xp.clip(al * (p - pmin), -60.0, 60.0)
        w = 1.0 / (1.0 + xp.exp(-d))
        p = w * p + (1.0 - w) * pmin
        
        # pmax curve
        # pmax = (Abump * Ro^(Aexp-1) / (1-k))^(1/Bbump)
        # Avoid division by zero if k=1 (k is usually 0.99)
        term_pmax = (self.Abump * Ro_mod**(self.Aexp - 1.0) / (1.0 - self.k))
        pmax = xp.power(term_pmax, 1.0/self.Bbump)
        
        # Conditions (live/dead knot branching mirrors MATLAB KnotSystem.m)
        if dead_knots:
            condRo = (p > pmax) | (radial_coord < 0) | (Ro_mod > self.RD)
            condR2 = (~condRo) & (Ro_mod >= self.RL) & (Ro_mod <= self.RD)
            condRi = (~condRo) & (~condR2)
        else:
            condRo = (p > pmax) | (radial_coord < 0)
            condRi = ~condRo
            condR2 = xp.zeros_like(condRo, dtype=bool)
        
        Ri = Ro_mod
        if include_knot_dev:
            # Ri = condRo*Ro + condRi*(k*Ro + Abump*Ro^Aexp * p^-Bbump)
            # Use masking
            term_knot = self.k * Ro_mod + self.Abump * Ro_mod**self.Aexp * xp.power(p, -self.Bbump)
            if dead_knots:
                S = self.k * self.RL
                T = self.Abump * self.RL**self.Aexp
                Kred = (Ro_mod - self.RL) / (self.RD - self.RL)
                delta_R_red = self.RD - self.RL
                gamma = 1.0 / 1.5
                delta_p_sq = delta_R_red**2 - (gamma * (self.RD - Ro_mod))**2
                # MATLAB uses sqrt(complex(...)); take real part for stable real-valued Ri.
                delta_p = (
                    xp.sqrt(delta_p_sq.astype(xp.complex128)).real
                    - np.sqrt(1.0 - gamma**2) * delta_R_red
                )
                p_alt = p + delta_p
                term_dead = Kred * self.RD + (1.0 - Kred) * (
                    S + T * xp.power(p_alt, -self.Bbump)
                )
                Ri = xp.where(condRi, term_knot, Ri)
                Ri = xp.where(condR2, term_dead, Ri)
            else:
                Ri = xp.where(condRo, Ro_mod, term_knot)
            
        Ri = xp.maximum(Ri, 1.0)
        
        # Growth field in the rotated knot-frame cross-section.
        gg = radial_coord**2 + tangential_coord**2 - Ri**2
        # g = min(real(gg), [], 4)
        # We calculate full gg (with knots dim) here.
        # Caller handles min.
        
        # Info struct
        info = {}
        
        if get_knots:
            # Knot internal structure (Lx)
            Lx = (
                self.a1 * radial_coord**4
                + self.a2 * radial_coord**3
                + self.a3 * radial_coord**2
                + self.a4 * radial_coord
            )
            if dead_knots:
                Lx_RL = (
                    self.a1 * self.RL**4
                    + self.a2 * self.RL**3
                    + self.a3 * self.RL**2
                    + self.a4 * self.RL
                )
                mask_range = (radial_coord >= self.RL) & (radial_coord <= self.RD)
                # Accelerating dead-knot taper (ease-in): starts gentle at RL
                # and diminishes faster toward RD.
                t = (radial_coord - self.RL) / (self.RD - self.RL)
                t = xp.clip(t, 0.0, 1.0)
                h = 1.0 - t**2
                Lx_tapered = Lx_RL * h
                Lx = xp.where(mask_range, Lx_tapered, Lx)
                Lx = xp.where(radial_coord > self.RD, 0.0, Lx)

            # Negative diameters are non-physical and should not reappear as
            # valid knot volume after squaring in the implicit field.
            Lx = xp.maximum(Lx, 0.0)
            
            # K field
            K = longitudinal_offset**2 + tangential_coord**2 / self.kp**2 - (Lx / 2.0)**2
            
            # Masking
            K = xp.where(radial_coord < 0, xp.nan, K)
            K = xp.where(radial_coord > 1.2 * Ro_mod, xp.nan, K)
            
            info['K'] = K
            if dead_knots:
                # Split knot scalar field into live/dead regions for visualization.
                K_live = xp.where(radial_coord <= self.RL, K, xp.nan)
                K_dead = xp.where(
                    (radial_coord > self.RL) & (radial_coord <= self.RD),
                    K,
                    xp.nan,
                )
            else:
                K_live = K
                K_dead = xp.full_like(K, xp.nan)
            info['K_live'] = K_live
            info['K_dead'] = K_dead
            
            method = int(flags.get('calc_fibers_a0_method', 1))
            z_expanded = xp.broadcast_to(z_board, radial_coord.shape)
            f_cross, f_long = self.calculate_flow_field(
                Lx,
                knot_axis_z,
                longitudinal_offset,
                self.kp,
                radial_coord,
                tangential_coord,
                z_expanded,
                method,
            )
            
            info['fiber_cross'] = f_cross
            info['fiber_long'] = f_long
            # Signed longitudinal distance to the knot axis.
            info['length_offset'] = longitudinal_offset
            info['yp'] = longitudinal_offset
            
            # Knot-axis tangent for fiber override.
            # The axis definition uses radial_coord (knot frame), so derivative must
            # use the same coordinate to correctly reflect c1/c2 curvature.
            dz_dx = 2 * self.c1 * radial_coord + self.c2

            # Tangent vector in local knot-frame coordinates:
            # (radial, tangential, longitudinal).
            mag_kf = xp.sqrt(1 + dz_dx**2)
            info['knot_dir_radial'] = 1.0 / mag_kf
            info['knot_dir_transverse'] = xp.zeros_like(mag_kf)
            info['knot_dir_longitudinal'] = dz_dx / mag_kf
            # Compatibility aliases used by the existing fiber-solver path.
            info['tx_kf'] = info['knot_dir_radial']
            info['ty_kf'] = info['knot_dir_longitudinal']
            info['tz_kf'] = info['knot_dir_transverse']
            
        return gg, info

    def calculate_flow_field(
        self,
        Lx,
        knot_axis_z,
        longitudinal_offset,
        kp,
        radial_coord,
        tangential_coord,
        z_coord,
        method,
    ):
        """Compute local fiber flow in the knot frame."""
        xp = self.xp
        br = kp * Lx
        use_approx = method != 1
        G, a = self.solve_flow_parameters(br, Lx, approx=use_approx)

        mask = radial_coord > 0
        dc_dtransverse = xp.ones_like(radial_coord)
        dc_dlongitudinal = xp.zeros_like(radial_coord)

        # Masked update avoids full-domain heavy math while also avoiding host sync.
        tangential_m = tangential_coord[mask]
        z_m = z_coord[mask]
        knot_axis_z_m = knot_axis_z[mask]
        longitudinal_offset_m = longitudinal_offset[mask]
        a_m = a[mask]
        G_m = G[mask]

        den1 = tangential_m**2 + (a_m + longitudinal_offset_m)**2
        den2 = tangential_m**2 + (a_m - longitudinal_offset_m)**2
        term1_dtransverse = (a_m + longitudinal_offset_m) / (den1 + 1e-12)
        term2_dtransverse = (a_m - longitudinal_offset_m) / (den2 + 1e-12)
        dc_dtransverse[mask] = (
            G_m * (term1_dtransverse + term2_dtransverse)
        ) / (2 * xp.pi) + 1.0

        den3 = (a_m + z_m - knot_axis_z_m) ** 2 + tangential_m**2
        den4 = (a_m - z_m + knot_axis_z_m) ** 2 + tangential_m**2
        term1_dlongitudinal = tangential_m / (den3 + 1e-12)
        term2_dlongitudinal = -tangential_m / (den4 + 1e-12)
        dc_dlongitudinal[mask] = (
            G_m * (term1_dlongitudinal + term2_dlongitudinal) * (-0.5)
        ) / xp.pi

        # Fiber direction tangent to stream function c = const
        fiber_cross_raw = -dc_dlongitudinal
        fiber_long_raw = dc_dtransverse
        mag = xp.sqrt(fiber_cross_raw**2 + fiber_long_raw**2)

        fiber_cross = xp.zeros_like(radial_coord)
        fiber_long = xp.zeros_like(radial_coord)
        valid = mag > 1e-12
        fiber_cross[valid] = fiber_cross_raw[valid] / mag[valid]
        fiber_long[valid] = fiber_long_raw[valid] / mag[valid]
        return fiber_cross, fiber_long

    def solve_flow_parameters(self, br, lr, approx=True):
        """Solve flow parameters G0 and a0 (method 1 exact-ish via Newton, 2 approx)."""
        xp = self.xp
        y = br / 2.0
        n = 7.0
        num = xp.sqrt(n) * lr * y
        den = xp.sqrt((2 * n + 1) * lr**2 + (4 * n - 4) * y**2)
        a0_approx = num / (den + 1e-12)

        bad_approx = ~xp.isfinite(a0_approx)
        a0_approx = xp.where(bad_approx, lr / 4.0, a0_approx)

        if approx:
            a0 = a0_approx
        else:
            # Vectorized Newton iterations for flowResiduals from MATLAB.
            a0 = a0_approx.copy()
            a0 = xp.where(xp.abs(a0) < 1e-8, xp.sign(a0) * 1e-8 + 1e-8, a0)
            for _ in range(8):
                A = xp.arctan2(y, -a0) - xp.arctan2(y, a0)
                B = lr**2 - 4.0 * a0**2
                F = 8.0 * a0 * y - A * B
                dA = 2.0 * y / (a0**2 + y**2 + 1e-12)
                dB = -8.0 * a0
                dF = 8.0 * y - (dA * B + A * dB)
                step = xp.where(xp.abs(dF) > 1e-10, F / dF, 0.0)
                a0 = a0 - step
                a0 = xp.where(xp.isfinite(a0), a0, a0_approx)
                a0 = xp.where(xp.abs(a0) < 1e-8, xp.sign(a0) * 1e-8 + 1e-8, a0)

        G0 = (xp.pi * lr**2 - 4 * xp.pi * a0**2) / (4 * a0 + 1e-12)
        return G0, a0
