from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Response
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any, Tuple
import numpy as np
import json
import math
import os
import hashlib
import base64
import gc
from io import BytesIO
import zipfile
import uuid
from PIL import Image, ImageDraw, ImageFilter, ImageOps
import scipy.io
from scipy.interpolate import griddata

class NanSafeEncoder(json.JSONEncoder):
    """JSON encoder that converts NaN/Inf to null."""
    def default(self, obj):
        if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
            return None
        return super().default(obj)
    
    def encode(self, o):
        return super().encode(self._sanitize(o))
    
    def _sanitize(self, obj):
        if isinstance(obj, float):
            if math.isnan(obj) or math.isinf(obj):
                return None
        elif isinstance(obj, dict):
            return {k: self._sanitize(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._sanitize(v) for v in obj]
        return obj

from .core.config import BoardConfig
from .core.knot_system import KnotSystem
from .core.mesh import BoardMesh
from .core.growth import GrowthSimulator
from .core.fiber import FiberSolver
from .core.array_backend import seed_all, to_numpy
from .core.photorealistic_inference import (
    PhotorealisticInferenceError,
    PhotorealisticUnavailableError,
    generate_photorealistic_surfaces,
    get_photorealistic_capability,
    preload_photorealistic_model,
)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1024, compresslevel=6)

_SIM_CACHE: Dict[str, Dict[str, Any]] = {}
_SIM_CACHE_ORDER: List[str] = []
_SIM_CACHE_LIMIT = 1
_SIM_MAX_BOARD_ATTEMPTS = 30
_DEFAULT_FIBER_IRREGULARITY_STRENGTH = 0.35
_DEFAULT_RING_IRREGULARITY_STRENGTH = 0.40


class _RetryablePlacementError(RuntimeError):
    """Per-attempt placement failure that should be retried."""


def swap_yz(point):
    """Swap Y and Z for Three.js (Y-up) from MATLAB (Z=Length=up)."""
    return [point[0], point[2], point[1]]


class ExportContoursRequest(BaseModel):
    simulation_id: Optional[str] = None
    contours: Optional[List[List[List[float]]]] = None
    board_outline: Optional[Dict[str, List[float]]] = None
    show_rings_inside_knots: Optional[bool] = None
    blur_sigma: Optional[float] = None


class ExportMatRequest(BaseModel):
    simulation_id: str


class ExportFibersRequest(BaseModel):
    simulation_id: str
    rand_fibers: Optional[bool] = None
    out_of_plane_threshold: Optional[float] = None
    snr: Optional[float] = None
    blur_sigma: Optional[float] = None


class ExportMatlabImageBundleRequest(BaseModel):
    simulation_id: str
    show_rings_inside_knots: Optional[bool] = None
    rand_fibers: Optional[bool] = None
    out_of_plane_threshold: Optional[float] = None
    snr: Optional[float] = None
    contour_line_width: Optional[float] = None
    contour_blur_sigma: Optional[float] = None
    fiber_blur_sigma: Optional[float] = None
    ring_irregularity_strength: Optional[float] = None
    fiber_irregularity_strength: Optional[float] = None
    imid: Optional[int] = None
    include_middle_surface: Optional[bool] = None


class ExportPhotorealisticRequest(BaseModel):
    simulation_id: str
    show_rings_inside_knots: Optional[bool] = None
    rand_fibers: Optional[bool] = None
    out_of_plane_threshold: Optional[float] = None
    snr: Optional[float] = None
    contour_line_width: Optional[float] = None
    contour_blur_sigma: Optional[float] = None
    fiber_blur_sigma: Optional[float] = None
    ring_irregularity_strength: Optional[float] = None
    fiber_irregularity_strength: Optional[float] = None
    imid: Optional[int] = None
    ddim_steps: Optional[int] = None
    guidance_scale: Optional[float] = None
    use_img2img_strength: Optional[float] = None
    include_knot_maps: Optional[bool] = None
    use_rings_only: Optional[bool] = None
    include_base64: Optional[bool] = None


def _clamp01(value):
    return np.clip(value, 0.0, 1.0)


def _surface_meta(board_outline: Dict[str, List[float]]) -> Dict[str, Dict[str, Any]]:
    mn = board_outline.get("min", [0.0, 0.0, 0.0])
    mx = board_outline.get("max", [1.0, 1.0, 1.0])
    x0, y0, z0 = float(mn[0]), float(mn[1]), float(mn[2])
    x1, y1, z1 = float(mx[0]), float(mx[1]), float(mx[2])

    return {
        "x_min": {"axis": 0, "fixed": x0, "u_axis": 1, "v_axis": 2, "u_min": y0, "u_max": y1, "v_min": z0, "v_max": z1},
        "x_max": {"axis": 0, "fixed": x1, "u_axis": 1, "v_axis": 2, "u_min": y0, "u_max": y1, "v_min": z0, "v_max": z1},
        "z_min": {"axis": 2, "fixed": z0, "u_axis": 0, "v_axis": 1, "u_min": x0, "u_max": x1, "v_min": y0, "v_max": y1},
        "z_max": {"axis": 2, "fixed": z1, "u_axis": 0, "v_axis": 1, "u_min": x0, "u_max": x1, "v_min": y0, "v_max": y1},
    }


def _classify_surface(points: np.ndarray, meta: Dict[str, Dict[str, Any]]) -> Optional[str]:
    if points.shape[0] < 2:
        return None

    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    ranges = maxs - mins
    fixed_axis = int(np.argmin(ranges))

    if fixed_axis not in (0, 2):
        return None

    axis_labels = ["x", "y", "z"]
    axis_label = axis_labels[fixed_axis]
    avg = float(np.mean(points[:, fixed_axis]))
    side = "min" if avg <= (meta[f"{axis_label}_min"]["fixed"] + meta[f"{axis_label}_max"]["fixed"]) * 0.5 else "max"
    return f"{axis_label}_{side}"


def _to_pixels(u: np.ndarray, v: np.ndarray, u_min: float, u_max: float, v_min: float, v_max: float, size: int) -> np.ndarray:
    u_span = max(1e-9, float(u_max - u_min))
    v_span = max(1e-9, float(v_max - v_min))
    x = np.rint(_clamp01((u - u_min) / u_span) * (size - 1)).astype(np.int32)
    y = np.rint((1.0 - _clamp01((v - v_min) / v_span)) * (size - 1)).astype(np.int32)
    return np.column_stack([x, y])


def _render_surface_png(lines: List[np.ndarray], surf_meta: Dict[str, Any], size: int = 512) -> bytes:
    image = Image.new("L", (size, size), color=255)
    draw = ImageDraw.Draw(image)
    u_axis = int(surf_meta["u_axis"])
    v_axis = int(surf_meta["v_axis"])

    for line in lines:
        if line.shape[0] < 2:
            continue
        u = line[:, u_axis]
        v = line[:, v_axis]
        pixels = _to_pixels(
            u,
            v,
            float(surf_meta["u_min"]),
            float(surf_meta["u_max"]),
            float(surf_meta["v_min"]),
            float(surf_meta["v_max"]),
            size,
        )
        draw.line([tuple(pt) for pt in pixels.tolist()], fill=0, width=1)

    png_buffer = BytesIO()
    image.save(png_buffer, format="PNG", optimize=False)
    return png_buffer.getvalue()


def _surface_meta_matlab_model(board_dims: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    x0 = float(board_dims.get("x_min", 0.0))
    x1 = float(board_dims.get("x_max", 1.0))
    y0 = float(board_dims.get("y_min", 0.0))
    y1 = float(board_dims.get("y_max", 1.0))
    z0 = float(board_dims.get("z_min", 0.0))
    z1 = float(board_dims.get("z_max", 1.0))

    # MATLAB saveRings view order:
    # rings_1 -> +Y face, rings_2 -> -Y face, rings_3 -> +X face, rings_4 -> -X face.
    # Image orientation follows save_face conventions used for fiber images:
    # YDir reversed for all, XDir reversed on ids 1 and 4.
    return {
        "y_max": {"u_axis": 0, "v_axis": 2, "u_min": x0, "u_max": x1, "v_min": z0, "v_max": z1, "flip_x": True},
        "y_min": {"u_axis": 0, "v_axis": 2, "u_min": x0, "u_max": x1, "v_min": z0, "v_max": z1, "flip_x": False},
        # Middle XZ slice uses the same orientation convention as y_min (no X flip).
        "y_mid": {"u_axis": 0, "v_axis": 2, "u_min": x0, "u_max": x1, "v_min": z0, "v_max": z1, "flip_x": False},
        "x_max": {"u_axis": 1, "v_axis": 2, "u_min": y0, "u_max": y1, "v_min": z0, "v_max": z1, "flip_x": False},
        "x_min": {"u_axis": 1, "v_axis": 2, "u_min": y0, "u_max": y1, "v_min": z0, "v_max": z1, "flip_x": True},
    }


def _render_surface_png_matlab(
    lines: List[np.ndarray],
    surf_meta: Dict[str, Any],
    size: int = 512,
    line_width: float = 1.0,
) -> bytes:
    image = Image.new("L", (size, size), color=255)
    draw = ImageDraw.Draw(image)
    u_axis = int(surf_meta["u_axis"])
    v_axis = int(surf_meta["v_axis"])
    flip_x = bool(surf_meta.get("flip_x", False))
    width_px = max(1, min(64, int(round(float(line_width)))))

    for line in lines:
        if line.shape[0] < 2:
            continue
        u = line[:, u_axis]
        v = line[:, v_axis]

        u_span = max(1e-9, float(surf_meta["u_max"] - surf_meta["u_min"]))
        v_span = max(1e-9, float(surf_meta["v_max"] - surf_meta["v_min"]))
        x = np.rint(_clamp01((u - float(surf_meta["u_min"])) / u_span) * (size - 1)).astype(np.int32)
        y = np.rint((1.0 - _clamp01((v - float(surf_meta["v_min"])) / v_span)) * (size - 1)).astype(np.int32)
        if flip_x:
            x = (size - 1) - x
        draw.line([tuple(pt) for pt in np.column_stack([x, y]).tolist()], fill=0, width=width_px)

    png_buffer = BytesIO()
    image.save(png_buffer, format="PNG", optimize=False)
    return png_buffer.getvalue()


def _classify_model_side(points: np.ndarray, board_dims: Dict[str, Any]) -> Optional[str]:
    if points.shape[0] < 2:
        return None
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    ranges = maxs - mins
    fixed_axis = int(np.argmin(ranges))

    if fixed_axis == 0:
        x0 = float(board_dims.get("x_min", 0.0))
        x1 = float(board_dims.get("x_max", 1.0))
        side = "min" if float(np.mean(points[:, 0])) <= 0.5 * (x0 + x1) else "max"
        return f"x_{side}"
    if fixed_axis == 1:
        y0 = float(board_dims.get("y_min", 0.0))
        y1 = float(board_dims.get("y_max", 1.0))
        side = "min" if float(np.mean(points[:, 1])) <= 0.5 * (y0 + y1) else "max"
        return f"y_{side}"
    return None


def _build_matlab_ring_pngs(
    contours_mat: List[Any],
    board_dims: Dict[str, Any],
    *,
    size: int = 512,
    line_width: float = 1.0,
) -> Dict[str, bytes]:
    face_meta = _surface_meta_matlab_model(board_dims)
    by_face: Dict[str, List[np.ndarray]] = {k: [] for k in ["y_max", "y_min", "x_max", "x_min"]}

    for line in contours_mat:
        arr = np.asarray(line, dtype=float)
        if arr.ndim != 2 or arr.shape[1] != 3 or arr.shape[0] < 2:
            continue
        key = _classify_model_side(arr, board_dims)
        if key in by_face:
            by_face[key].append(arr)

    # MATLAB saveRings order.
    ordered = [
        ("rings_1", "y_max"),
        ("rings_2", "y_min"),
        ("rings_3", "x_max"),
        ("rings_4", "x_min"),
    ]
    out: Dict[str, bytes] = {}
    for folder_name, face_key in ordered:
        out[folder_name] = _render_surface_png_matlab(
            by_face[face_key],
            face_meta[face_key],
            size=size,
            line_width=line_width,
        )
    return out


def _build_matlab_mid_ring_png(
    contours_mid_mat: List[Any],
    board_dims: Dict[str, Any],
    *,
    size: int = 512,
    line_width: float = 1.0,
) -> bytes:
    face_meta = _surface_meta_matlab_model(board_dims)
    lines: List[np.ndarray] = []
    for line in contours_mid_mat:
        arr = np.asarray(line, dtype=float)
        if arr.ndim != 2 or arr.shape[1] != 3 or arr.shape[0] < 2:
            continue
        lines.append(arr)
    return _render_surface_png_matlab(
        lines,
        face_meta["y_mid"],
        size=size,
        line_width=line_width,
    )


def _sample_board_perimeter_xy(x0: float, x1: float, y0: float, y1: float, samples_per_edge: int = 80):
    n = max(8, int(samples_per_edge))
    xs = []
    ys = []
    sides = []

    # y = y0 (min)
    x = np.linspace(x0, x1, n, endpoint=False)
    xs.extend(x.tolist())
    ys.extend([y0] * len(x))
    sides.extend(["z_min_side"] * len(x))

    # x = x1 (max)
    y = np.linspace(y0, y1, n, endpoint=False)
    xs.extend([x1] * len(y))
    ys.extend(y.tolist())
    sides.extend(["x_max_side"] * len(y))

    # y = y1 (max)
    x = np.linspace(x1, x0, n, endpoint=False)
    xs.extend(x.tolist())
    ys.extend([y1] * len(x))
    sides.extend(["z_max_side"] * len(x))

    # x = x0 (min)
    y = np.linspace(y1, y0, n, endpoint=True)
    xs.extend([x0] * len(y))
    ys.extend(y.tolist())
    sides.extend(["x_min_side"] * len(y))

    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float), np.asarray(sides, dtype=object)


def _evaluate_outer_radius(splines, theta: np.ndarray) -> np.ndarray:
    if not splines:
        return np.full(theta.shape, np.nan, dtype=float)

    theta = np.asarray(theta, dtype=float)
    outer = np.full(theta.shape, -np.inf, dtype=float)

    for spline in splines:
        try:
            breaks = np.asarray(getattr(spline, "x", []), dtype=float).reshape(-1)
            t_eval = theta
            if breaks.size >= 2:
                b0 = float(breaks[0])
                b1 = float(breaks[-1])
                period = b1 - b0
                if np.isfinite(period) and period > 0:
                    t_eval = ((theta - b0) % period) + b0
            r = np.asarray(spline(t_eval), dtype=float)
            outer = np.maximum(outer, r)
        except Exception:
            continue

    outer[~np.isfinite(outer)] = np.nan
    return outer


def _sample_center_within_log(
    splines,
    *,
    width: float,
    thickness: float,
) -> tuple[float, float]:
    if not splines:
        return 0.0, 0.0

    theta = np.linspace(-np.pi, np.pi, 1024, endpoint=False, dtype=np.float64)
    outer = np.asarray(_evaluate_outer_radius(splines, theta), dtype=np.float64)
    valid = outer[np.isfinite(outer)]
    if valid.size == 0:
        return 0.0, 0.0

    min_radius = float(np.min(valid))
    half_diag = 0.5 * float(np.hypot(width, thickness))
    max_center_radius = min_radius - half_diag
    if max_center_radius < 0.0:
        raise _RetryablePlacementError(
            "Board dimensions are too large to fit inside the selected log cross-section."
        )
    if max_center_radius <= 1e-9:
        return 0.0, 0.0

    # Uniform sample over a disk to avoid center bias.
    radius = max_center_radius * float(np.sqrt(np.random.random()))
    angle = 2.0 * float(np.pi) * float(np.random.random())
    return radius * float(np.cos(angle)), radius * float(np.sin(angle))


def _board_fit_warnings(
    config: BoardConfig,
    mesh: BoardMesh,
    k: KnotSystem,
    *,
    force_check: bool = False,
) -> List[str]:
    if (not force_check) and int(config.board_or_log) != 0:
        return []
    if not getattr(k, "splines", None):
        return []

    try:
        x0, x1 = mesh.board_coords["x"]
        y0, y1 = mesh.board_coords["y"]
        z0, z1 = mesh.board_coords["z"]
        px, py, side_tags = _sample_board_perimeter_xy(x0, x1, y0, y1, samples_per_edge=80)
        board_length = max(1.0, abs(float(z1) - float(z0)))
        # Validate the footprint across the full board length because both crook
        # (center shift) and taper (radius reduction) are z-dependent.
        n_z = int(np.clip(np.ceil(board_length / 20.0) + 1, 2, 401))
        z_samples = np.linspace(float(z0), float(z1), n_z, dtype=float)
        crook_x = np.asarray(to_numpy(k.crook_x(z_samples)), dtype=float).reshape(-1)
        crook_y = np.asarray(to_numpy(k.crook_y(z_samples)), dtype=float).reshape(-1)
        taper = np.asarray(to_numpy(k.taper(z_samples)), dtype=float).reshape(-1)

        x_eval = px.reshape(1, -1) + crook_x.reshape(-1, 1)
        y_eval = py.reshape(1, -1) + crook_y.reshape(-1, 1)
        theta = np.arctan2(y_eval, x_eval)
        r_board = np.hypot(x_eval, y_eval)
        r_tree = _evaluate_outer_radius(k.splines, theta.reshape(-1)).reshape(theta.shape)
        r_tree_eff = r_tree - taper.reshape(-1, 1)

        valid = np.isfinite(r_tree_eff)
        if not np.any(valid):
            return []

        outside = valid & (r_board > (r_tree_eff + 1e-6))
        if not np.any(outside):
            return []

        overflow = r_board[outside] - r_tree_eff[outside]
        max_over = float(np.max(overflow)) if overflow.size else 0.0
        pct_outside = 100.0 * float(np.sum(outside)) / float(np.sum(valid))
        bad_sides = sorted(set(side_tags[np.any(outside, axis=0)].tolist()))
        bad_z = z_samples[np.any(outside, axis=1)]

        side_label_map = {
            "x_min_side": "X-min side",
            "x_max_side": "X-max side",
            "z_min_side": "Z-min side",
            "z_max_side": "Z-max side",
        }
        side_text = ", ".join(side_label_map.get(s, s) for s in bad_sides)
        if bad_z.size > 0:
            z_text = f"{float(np.min(bad_z)):.1f}..{float(np.max(bad_z)):.1f}"
        else:
            z_text = f"{float(z0):.1f}..{float(z1):.1f}"

        msg = (
            "Selected tree cross-section is smaller than the requested board footprint "
            "(viewer X/Z directions; model X/Y). "
            "Check includes crook+taper along board length. "
            f"Outside perimeter: {pct_outside:.1f}% (max overflow {max_over:.1f} mm). "
            f"Length interval with outside points: Z={z_text} mm. "
            f"Affected sides: {side_text}. "
            "Consider reducing board extents or generating another random board."
        )
        return [msg]
    except Exception:
        return []


def _cache_simulation(entry: Dict[str, Any]) -> str:
    sim_id = str(uuid.uuid4())
    _SIM_CACHE[sim_id] = entry
    _SIM_CACHE_ORDER.append(sim_id)
    while len(_SIM_CACHE_ORDER) > _SIM_CACHE_LIMIT:
        old_id = _SIM_CACHE_ORDER.pop(0)
        old_entry = _SIM_CACHE.pop(old_id, None)
        if old_entry is not None:
            del old_entry
            gc.collect()
    return sim_id


def _has_any_contours(layers_data: Dict[str, Any]) -> bool:
    return bool(
        (layers_data.get("contours") or [])
        or (layers_data.get("contours_masked") or [])
        or (layers_data.get("contours_unmasked") or [])
    )


def _contours_to_mat_cell(contours: List[Any]) -> np.ndarray:
    n = len(contours)
    cell = np.empty((n, 1), dtype=object)
    for i, line in enumerate(contours):
        arr = np.asarray(line, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != 3:
            arr = np.empty((0, 3), dtype=np.float32)
        cell[i, 0] = arr
    return cell


def _mat_segment_cell(segments: List[np.ndarray]) -> np.ndarray:
    cell = np.empty((len(segments), 1), dtype=object)
    for i, segment in enumerate(segments):
        arr = np.asarray(segment, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != 3:
            arr = np.empty((0, 3), dtype=np.float32)
        cell[i, 0] = arr
    return cell


def _finite_float_or_none(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _finite_float_list(value: Any) -> List[float]:
    if value is None:
        return []
    try:
        arr = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return []
    return [float(v) for v in arr.tolist() if math.isfinite(float(v))]


def _evaluate_exported_crook_centerline(
    z_mm: float,
    geometry_randomization: Dict[str, Any],
) -> Tuple[float, float]:
    if not isinstance(geometry_randomization, dict):
        return 0.0, 0.0

    amplitudes = _finite_float_list(geometry_randomization.get("component_amplitudes"))
    shifts = _finite_float_list(geometry_randomization.get("component_shifts_mm"))
    thetas_deg = _finite_float_list(geometry_randomization.get("component_thetas_deg"))
    orders = [
        max(1, int(math.floor(v)))
        for v in _finite_float_list(geometry_randomization.get("component_orders"))
    ]
    p_count = max(0, int(math.floor(_finite_float_or_none(
        geometry_randomization.get("crook_component_count")
    ) or 0.0)))
    term_count = max(p_count, len(amplitudes), len(shifts), len(thetas_deg), len(orders))

    dx = 0.0
    dy = 0.0
    for idx in range(term_count):
        order = orders[idx] if idx < len(orders) else (idx + 1)
        wavelength_mm = (2.0 ** (5 - order)) * 1000.0
        if not math.isfinite(wavelength_mm) or wavelength_mm <= 0.0:
            continue
        amp = amplitudes[idx] if idx < len(amplitudes) else 0.0
        shift = shifts[idx] if idx < len(shifts) else 0.0
        theta = math.radians(thetas_deg[idx] if idx < len(thetas_deg) else 0.0)
        wave = math.sin((2.0 * math.pi * (z_mm + shift)) / wavelength_mm)
        dx += math.sin(theta) * amp * wave
        dy += math.cos(theta) * amp * wave

    legacy_x = _finite_float_or_none(
        geometry_randomization.get("active_legacy_manual_crook_x_coeff")
    )
    legacy_y = _finite_float_or_none(
        geometry_randomization.get("active_legacy_manual_crook_y_coeff")
    )
    if legacy_x is not None:
        dx += legacy_x * z_mm * z_mm
    if legacy_y is not None:
        dy += legacy_y * z_mm * z_mm

    # The frontend displays the pith centerline opposite to the applied crook.
    return -dx, -dy


def _knot_sequence_segments_to_mat_struct(
    knot_sequence: Dict[str, Any],
    geometry_randomization: Dict[str, Any],
    board_dimensions: Dict[str, Any],
    knots: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    if not isinstance(knot_sequence, dict):
        knot_sequence = {}
    if not isinstance(board_dimensions, dict):
        board_dimensions = {}
    if not isinstance(knots, list):
        knots = []

    slot_count = max(0, int(math.floor(_finite_float_or_none(
        knot_sequence.get("slot_count")
    ) or 0.0)))
    dz_mm = _finite_float_or_none(knot_sequence.get("dz_mm")) or 0.0
    z_min_mm = _finite_float_or_none(knot_sequence.get("z_min_mm"))
    if z_min_mm is None:
        z_min_mm = _finite_float_or_none(board_dimensions.get("z_min")) or 0.0

    z_max_mm = _finite_float_or_none(board_dimensions.get("z_max"))
    length_mm = _finite_float_or_none(board_dimensions.get("length"))
    if length_mm is None and z_max_mm is not None:
        length_mm = abs(z_max_mm - z_min_mm)
    if (slot_count <= 0 or dz_mm <= 0.0) and length_mm is not None and length_mm > 0.0:
        dz_mm = dz_mm if dz_mm > 0.0 else 10.0
        slot_count = max(1, int(math.ceil(length_mm / dz_mm)))

    try:
        raw_occupancy = np.asarray(
            knot_sequence.get("slot_has_knot", []),
            dtype=np.float32,
        ).reshape(-1)
    except (TypeError, ValueError):
        raw_occupancy = np.empty((0,), dtype=np.float32)
    occupancy = np.zeros((slot_count,), dtype=np.float32)
    if slot_count > 0 and raw_occupancy.size > 0:
        keep = min(slot_count, int(raw_occupancy.size))
        occupancy[:keep] = raw_occupancy[:keep]

    for item in knots:
        if not isinstance(item, dict) or slot_count <= 0 or dz_mm <= 0.0:
            continue
        slot_idx_value = _finite_float_or_none(item.get("slot_index"))
        if slot_idx_value is not None:
            slot_idx = int(round(slot_idx_value))
        else:
            z0_mm = _finite_float_or_none(item.get("z0_mm"))
            if z0_mm is None:
                continue
            slot_idx = int(math.floor((z0_mm - z_min_mm) / dz_mm))
            if slot_idx >= slot_count and math.isclose(z0_mm, z_min_mm + slot_count * dz_mm):
                slot_idx = slot_count - 1
        if 0 <= slot_idx < slot_count:
            occupancy[slot_idx] = 1.0

    with_knot: List[np.ndarray] = []
    no_knot: List[np.ndarray] = []

    if slot_count > 0 and math.isfinite(dz_mm) and dz_mm > 0.0:
        for idx in range(slot_count):
            z0 = z_min_mm + (idx * dz_mm)
            z1 = z_min_mm + ((idx + 1) * dz_mm)
            if z_max_mm is not None:
                z1 = min(z1, z_max_mm) if z_max_mm >= z_min_mm else max(z1, z_max_mm)
            if math.isclose(z0, z1):
                continue
            x0, y0 = _evaluate_exported_crook_centerline(z0, geometry_randomization)
            x1, y1 = _evaluate_exported_crook_centerline(z1, geometry_randomization)
            segment = np.asarray([[x0, y0, z0], [x1, y1, z1]], dtype=np.float32)
            has_knot = idx < occupancy.size and float(occupancy[idx]) > 0.0
            if has_knot:
                with_knot.append(segment)
            else:
                no_knot.append(segment)

    return {
        "with_knot": _mat_segment_cell(with_knot),
        "no_knot": _mat_segment_cell(no_knot),
        "slot_count": np.array([[slot_count]], dtype=np.int32),
        "dz_mm": np.array([[float(dz_mm)]], dtype=np.float32),
        "z_min_mm": np.array([[float(z_min_mm)]], dtype=np.float32),
        "coordinate_system": np.array(
            ["segment rows are [X=width, Y=thickness, Z=length]"], dtype=object
        ),
    }


_MAT_MESH_DTYPE = [
    ("vertices", "O"),
    ("faces", "O"),
    ("layer_index", "O"),
    ("part", "O"),
    ("knot_index", "O"),
    ("slot_index", "O"),
    ("z0_mm", "O"),
    ("vertex_colors", "O"),
    ("dead_weight", "O"),
    ("color", "O"),
    ("face_index_base", "O"),
]


_MAT_PHOTOREALISTIC_FACE_DTYPE = [
    ("face", "O"),
    ("filename", "O"),
    ("image", "O"),
    ("flip_x", "O"),
]


def _empty_mat_mesh_struct_array() -> np.ndarray:
    return np.empty((0, 1), dtype=_MAT_MESH_DTYPE)


def _empty_photorealistic_face_struct_array() -> np.ndarray:
    return np.empty((0, 1), dtype=_MAT_PHOTOREALISTIC_FACE_DTYPE)


def _mat_optional_scalar(value: Any, dtype: Any, cast: Any) -> np.ndarray:
    if value is None:
        return np.empty((0, 0), dtype=dtype)
    return np.array([[cast(value)]], dtype=dtype)


def _mat_mesh_payloads_to_struct_array(meshes: List[Dict[str, Any]]) -> np.ndarray:
    if not meshes:
        return _empty_mat_mesh_struct_array()

    arr = np.empty((len(meshes), 1), dtype=_MAT_MESH_DTYPE)
    for i, item in enumerate(meshes):
        vertices = np.asarray(item.get("vertices", np.empty((0, 3))), dtype=np.float32)
        if vertices.ndim != 2 or vertices.shape[1] != 3:
            vertices = np.empty((0, 3), dtype=np.float32)

        faces = np.asarray(item.get("faces", np.empty((0, 3))), dtype=np.float64)
        if faces.ndim != 2 or faces.shape[1] < 3:
            faces = np.empty((0, 3), dtype=np.float64)
        else:
            faces = faces[:, :3]
            if faces.size and np.nanmin(faces) <= 0:
                faces = faces + 1.0

        vertex_colors = np.asarray(item.get("vertex_colors", np.empty((0, 3))), dtype=np.float32)
        if vertex_colors.ndim != 2 or vertex_colors.shape[1] != 3:
            vertex_colors = np.empty((0, 3), dtype=np.float32)

        dead_weight = np.asarray(item.get("dead_weight", np.empty((0, 1))), dtype=np.float32).reshape(-1, 1)

        arr["vertices"][i, 0] = vertices
        arr["faces"][i, 0] = faces
        arr["layer_index"][i, 0] = _mat_optional_scalar(
            item.get("layer_index"),
            np.int32,
            int,
        )
        arr["part"][i, 0] = np.array([str(item.get("part", ""))], dtype=object)
        arr["knot_index"][i, 0] = _mat_optional_scalar(
            item.get("knot_index"),
            np.int32,
            int,
        )
        arr["slot_index"][i, 0] = _mat_optional_scalar(
            item.get("slot_index"),
            np.int32,
            int,
        )
        arr["z0_mm"][i, 0] = _mat_optional_scalar(
            item.get("z0_mm"),
            np.float32,
            float,
        )
        arr["vertex_colors"][i, 0] = vertex_colors
        arr["dead_weight"][i, 0] = dead_weight
        arr["color"][i, 0] = np.array([str(item.get("color", ""))], dtype=object)
        arr["face_index_base"][i, 0] = np.array([[1]], dtype=np.int32)

    return arr


def _surface_to_mat_mesh_payload(surface: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(surface, dict):
        return None

    vertices = np.asarray(surface.get("vertices", np.empty((0, 3))), dtype=np.float32)
    faces = np.asarray(surface.get("faces", np.empty((0, 3))), dtype=np.int32)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or vertices.size == 0:
        return None
    if faces.ndim != 2 or faces.shape[1] < 3 or faces.size == 0:
        return None

    payload: Dict[str, Any] = {
        "vertices": vertices,
        "faces": faces[:, :3],
    }
    layer_index = surface.get("layer_index")
    if layer_index is not None:
        try:
            payload["layer_index"] = int(layer_index)
        except (TypeError, ValueError):
            pass
    return payload


def _surfaces_to_mat_mesh_payloads(surfaces: Any) -> List[Dict[str, Any]]:
    if not isinstance(surfaces, list):
        return []
    payloads: List[Dict[str, Any]] = []
    for surface in surfaces:
        payload = _surface_to_mat_mesh_payload(surface)
        if payload is not None:
            payloads.append(payload)
    return payloads


def _to_float32_3d(value: Any) -> np.ndarray:
    if value is None:
        return np.empty((0, 0, 0), dtype=np.float32)
    try:
        arr = np.asarray(to_numpy(value), dtype=np.float32)
    except Exception:
        return np.empty((0, 0, 0), dtype=np.float32)
    if arr.ndim != 3:
        return np.empty((0, 0, 0), dtype=np.float32)
    return arr


def _growth_fields_to_float32_stack(fields: Any, indices: Any = None) -> Tuple[np.ndarray, np.ndarray]:
    if not isinstance(fields, list) or not fields:
        return (
            np.empty((0, 0, 0, 0), dtype=np.float32),
            np.empty((0,), dtype=np.int32),
        )

    arrays: List[np.ndarray] = []
    kept_indices: List[int] = []
    raw_indices = list(indices) if isinstance(indices, (list, tuple)) else []
    shape: Optional[Tuple[int, int, int]] = None
    for field_idx, field in enumerate(fields):
        arr = _to_float32_3d(field)
        if arr.size == 0:
            continue
        arr_shape = tuple(int(v) for v in arr.shape)
        if shape is None:
            shape = arr_shape
        if arr_shape != shape:
            continue
        arrays.append(arr)
        try:
            layer_idx = int(raw_indices[field_idx]) if field_idx < len(raw_indices) else int(field_idx)
        except Exception:
            layer_idx = int(field_idx)
        kept_indices.append(layer_idx)

    if not arrays:
        return (
            np.empty((0, 0, 0, 0), dtype=np.float32),
            np.empty((0,), dtype=np.int32),
        )
    return (
        np.stack(arrays, axis=3).astype(np.float32, copy=False),
        np.asarray(kept_indices, dtype=np.int32).reshape(-1),
    )


def _png_bytes_to_rgb_array(png_bytes: bytes) -> np.ndarray:
    with Image.open(BytesIO(png_bytes)) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def _photorealistic_faces_to_mat_struct_array(faces: List[Dict[str, Any]]) -> np.ndarray:
    if not faces:
        return _empty_photorealistic_face_struct_array()

    arr = np.empty((len(faces), 1), dtype=_MAT_PHOTOREALISTIC_FACE_DTYPE)
    for i, item in enumerate(faces):
        image = np.asarray(item.get("image", np.empty((0, 0, 3))), dtype=np.uint8)
        if image.ndim != 3 or image.shape[2] != 3:
            image = np.empty((0, 0, 3), dtype=np.uint8)

        arr["face"][i, 0] = np.array([str(item.get("face", ""))], dtype=object)
        arr["filename"][i, 0] = np.array([str(item.get("filename", ""))], dtype=object)
        arr["image"][i, 0] = image
        arr["flip_x"][i, 0] = np.array([[bool(item.get("flip_x", False))]], dtype=bool)

    return arr


def _apply_png_blur_bytes(png_bytes: bytes, sigma: float | None) -> bytes:
    blur_sigma = float(sigma) if sigma is not None else 0.0
    if not np.isfinite(blur_sigma) or blur_sigma <= 0.0:
        return png_bytes
    image = Image.open(BytesIO(png_bytes)).convert("L")
    blurred = image.filter(ImageFilter.GaussianBlur(radius=blur_sigma))
    out = BytesIO()
    blurred.save(out, format="PNG", optimize=False)
    return out.getvalue()


def _flip_png_vertical_bytes(png_bytes: bytes) -> bytes:
    image = Image.open(BytesIO(png_bytes)).convert("L")
    flipped = image.transpose(Image.FLIP_TOP_BOTTOM)
    out = BytesIO()
    flipped.save(out, format="PNG", optimize=False)
    return out.getvalue()


def _png_gray_to_float01(png_bytes: bytes) -> np.ndarray:
    return np.asarray(Image.open(BytesIO(png_bytes)).convert("L"), dtype=np.float32) / 255.0


def _float01_to_png_gray_bytes(img01: np.ndarray) -> bytes:
    arr = np.clip(np.asarray(img01, dtype=np.float32), 0.0, 1.0)
    image = Image.fromarray(np.rint(arr * 255.0).astype(np.uint8), mode="L")
    out = BytesIO()
    image.save(out, format="PNG", optimize=False)
    return out.getvalue()


def _stable_rng_from_png_bytes(png_bytes: bytes, *, salt: str) -> np.random.Generator:
    digest = hashlib.sha256(salt.encode("utf-8") + b":" + png_bytes).digest()
    seed = int.from_bytes(digest[:8], byteorder="little", signed=False)
    return np.random.default_rng(seed)


def _normalize_unit_std(arr: np.ndarray) -> np.ndarray:
    out = np.asarray(arr, dtype=np.float32)
    out = out - np.mean(out, dtype=np.float64)
    sd = float(np.std(out, dtype=np.float64))
    if not np.isfinite(sd) or sd < 1e-8:
        return np.zeros_like(out, dtype=np.float32)
    return (out / sd).astype(np.float32)


def _fractal_band_noise(
    shape: tuple[int, int],
    rng: np.random.Generator,
    *,
    beta: float,
    f_low: float,
    f_high: float,
) -> np.ndarray:
    h, w = int(shape[0]), int(shape[1])
    fy = np.fft.fftfreq(h)
    fx = np.fft.rfftfreq(w)
    yy, xx = np.meshgrid(fy, fx, indexing="ij")
    rr = np.sqrt(xx * xx + yy * yy)
    eps = 1e-6
    amp = np.power(rr + eps, -float(beta))
    lo = max(1e-4, float(f_low))
    hi = max(lo + 1e-4, float(f_high))
    gate_low = 1.0 - np.exp(-np.power(rr / lo, 4.0))
    gate_high = np.exp(-np.power(rr / hi, 4.0))
    amp = amp * gate_low * gate_high
    amp[0, 0] = 0.0
    phase = rng.uniform(0.0, 2.0 * np.pi, size=rr.shape)
    spec = amp * (np.cos(phase) + 1j * np.sin(phase))
    noise = np.fft.irfft2(spec, s=(h, w))
    return _normalize_unit_std(noise)


def _max_filter3x3(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float32)
    pad = np.pad(a, ((1, 1), (1, 1)), mode="edge")
    windows = [
        pad[0:-2, 0:-2], pad[0:-2, 1:-1], pad[0:-2, 2:],
        pad[1:-1, 0:-2], pad[1:-1, 1:-1], pad[1:-1, 2:],
        pad[2:, 0:-2], pad[2:, 1:-1], pad[2:, 2:],
    ]
    return np.maximum.reduce(windows).astype(np.float32)


def _pil_blur_float01(arr: np.ndarray, radius: float) -> np.ndarray:
    radius = float(radius)
    if not np.isfinite(radius) or radius <= 0.0:
        return np.asarray(arr, dtype=np.float32)
    image = Image.fromarray(np.rint(np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8), mode="L")
    blurred = image.filter(ImageFilter.GaussianBlur(radius=radius))
    return np.asarray(blurred, dtype=np.float32) / 255.0


def _apply_fiber_irregularity_bytes(png_bytes: bytes, strength: float | None) -> bytes:
    s = float(strength) if strength is not None else 0.0
    if not np.isfinite(s) or s <= 0.0:
        return png_bytes
    s = float(np.clip(s, 0.0, 2.0))
    img = _png_gray_to_float01(png_bytes)
    if img.ndim != 2 or img.size == 0:
        return png_bytes

    rng = _stable_rng_from_png_bytes(png_bytes, salt="fiber_irregularity_v1")
    median = float(np.median(img))
    dev = np.abs(img - median)

    # Emphasize clear-wood areas while damping perturbation around strong knot deviations.
    clear_w = np.clip(1.0 - (dev / 0.12), 0.0, 1.0)
    clear_w = np.power(clear_w, 1.5).astype(np.float32, copy=False)

    dx = np.zeros_like(img, dtype=np.float32)
    dy = np.zeros_like(img, dtype=np.float32)
    dx[:, 1:] = np.abs(img[:, 1:] - img[:, :-1])
    dy[1:, :] = np.abs(img[1:, :] - img[:-1, :])
    edge = dx + dy
    edge_w = np.clip(1.0 - (edge / 0.05), 0.0, 1.0)

    n_low = _fractal_band_noise(img.shape, rng, beta=1.1, f_low=0.008, f_high=0.06)
    n_mid = _fractal_band_noise(img.shape, rng, beta=0.9, f_low=0.035, f_high=0.22)
    n_sparse = rng.uniform(-1.0, 1.0, size=img.shape).astype(np.float32)
    n_sparse = np.sign(n_sparse) * np.power(np.abs(n_sparse), 3.2)
    n_sparse = _normalize_unit_std(n_sparse)

    noise = _normalize_unit_std((0.62 * n_low) + (0.28 * n_mid) + (0.10 * n_sparse))
    noise = np.sign(noise) * np.power(np.abs(noise), 1.35)
    noise = _normalize_unit_std(noise)

    amplitude = float(0.008 * s)
    weight = clear_w * (0.65 + (0.35 * edge_w))
    perturbed = np.clip(img + (amplitude * weight * noise), 0.0, 1.0)
    return _float01_to_png_gray_bytes(perturbed)


def _apply_ring_irregularity_bytes(png_bytes: bytes, strength: float | None) -> bytes:
    s = float(strength) if strength is not None else 0.0
    if not np.isfinite(s) or s <= 0.0:
        return png_bytes
    s = float(np.clip(s, 0.0, 2.0))
    img = _png_gray_to_float01(png_bytes)
    if img.ndim != 2 or img.size == 0:
        return png_bytes

    rng = _stable_rng_from_png_bytes(png_bytes, salt="ring_irregularity_v1")
    dark_on_light = bool(float(np.mean(img)) >= 0.5)
    line = (1.0 - img) if dark_on_light else img
    line = np.clip(line, 0.0, 1.0).astype(np.float32, copy=False)

    width_noise = _fractal_band_noise(line.shape, rng, beta=1.35, f_low=0.01, f_high=0.09)
    width_gain = np.clip(1.0 + (0.35 * s * width_noise), 0.60, 1.60)

    dilated = _max_filter3x3(line)
    line_var = np.clip((line * width_gain), 0.0, 1.0)
    dilate_w = float(np.clip(0.55 * s, 0.0, 0.85))
    line_mix = np.clip(((1.0 - dilate_w) * line_var) + (dilate_w * dilated), 0.0, 1.0)
    line_blur = _pil_blur_float01(line_mix, radius=(0.30 + (0.55 * s)))
    line_soft = np.clip(line_mix + ((0.40 * s) * line_blur), 0.0, 1.0)

    bg_noise = _fractal_band_noise(line.shape, rng, beta=1.0, f_low=0.02, f_high=0.25)
    bg_noise = np.sign(bg_noise) * np.power(np.abs(bg_noise), 1.2)
    bg_noise = _normalize_unit_std(bg_noise)
    bg_weight = np.clip(1.0 - (2.3 * line_soft), 0.0, 1.0)
    paper = (0.010 * s) * bg_weight * bg_noise

    out = (1.0 - line_soft) if dark_on_light else line_soft
    out = np.clip(out + paper, 0.0, 1.0)
    out = _pil_blur_float01(out, radius=(0.10 + (0.18 * s)))
    return _float01_to_png_gray_bytes(out)


def _contours_mat_to_viewer(contours_mat: List[Any]) -> List[List[List[float]]]:
    out: List[List[List[float]]] = []
    for line in contours_mat:
        arr = np.asarray(line, dtype=float)
        if arr.ndim != 2 or arr.shape[1] != 3 or arr.shape[0] < 2:
            continue
        out.append([[float(p[0]), float(p[2]), float(p[1])] for p in arr])
    return out


def _board_outline_from_cached_entry(entry: Dict[str, Any]) -> Optional[Dict[str, List[float]]]:
    dims = entry.get("board_dimensions") or {}
    try:
        x0 = float(dims["x_min"])
        x1 = float(dims["x_max"])
        y0 = float(dims["y_min"])
        y1 = float(dims["y_max"])
        z0 = float(dims["z_min"])
        z1 = float(dims["z_max"])
    except Exception:
        return None
    return {
        "min": swap_yz([x0, y0, z0]),
        "max": swap_yz([x1, y1, z1]),
    }


def _fiber_orientation_map(fx: np.ndarray, fy: np.ndarray, flip_sign: bool) -> np.ndarray:
    fiber = np.arctan2(fy, fx) - (np.pi / 2.0)
    if flip_sign:
        fiber = -fiber
    fiber = (fiber + (np.pi / 2.0)) / np.pi
    return np.clip(fiber, 0.0, 1.0)


def _render_fiber_face_png(
    x_face: np.ndarray,
    y_face: np.ndarray,
    fx_face: np.ndarray,
    fy_face: np.ndarray,
    *,
    flip_sign: bool,
    flip_x: bool,
    size: int = 512,
) -> bytes:
    x = np.asarray(x_face, dtype=np.float64)
    y = np.asarray(y_face, dtype=np.float64)
    fx = np.asarray(fx_face, dtype=np.float64)
    fy = np.asarray(fy_face, dtype=np.float64)

    img = np.full((size, size), 1.0, dtype=np.float64)
    if x.shape != y.shape or x.shape != fx.shape or x.shape != fy.shape:
        image = Image.fromarray(np.rint(img * 255.0).astype(np.uint8), mode="L")
        buffer = BytesIO()
        image.save(buffer, format="PNG", optimize=False)
        return buffer.getvalue()

    fiber = _fiber_orientation_map(fx, fy, flip_sign=flip_sign)
    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(fiber)
    if np.count_nonzero(mask) >= 3:
        points = np.column_stack([x[mask], y[mask]])
        values = fiber[mask]
        x_min, x_max = float(np.min(points[:, 0])), float(np.max(points[:, 0]))
        y_min, y_max = float(np.min(points[:, 1])), float(np.max(points[:, 1]))
        if x_max > x_min and y_max > y_min:
            xi = np.linspace(x_min, x_max, size, dtype=np.float64)
            yi = np.linspace(y_min, y_max, size, dtype=np.float64)
            XI, YI = np.meshgrid(xi, yi, indexing="xy")
            interp = griddata(points, values, (XI, YI), method="linear")
            if np.isnan(interp).any():
                interp_nn = griddata(points, values, (XI, YI), method="nearest")
                interp = np.where(np.isfinite(interp), interp, interp_nn)
            img = np.clip(np.nan_to_num(interp, nan=1.0), 0.0, 1.0)

    # MATLAB save_face uses YDir reverse for all sides and optional XDir reverse.
    img = img[::-1, :]
    if flip_x:
        img = img[:, ::-1]

    image = Image.fromarray(np.rint(img * 255.0).astype(np.uint8), mode="L")
    buffer = BytesIO()
    image.save(buffer, format="PNG", optimize=False)
    return buffer.getvalue()


def _build_fiber_surface_pngs(
    txx: np.ndarray,
    tyy: np.ndarray,
    tzz: np.ndarray,
    mesh_x: np.ndarray,
    mesh_y: np.ndarray,
    mesh_z: np.ndarray,
    rand_fibers: bool = False,
    out_of_plane_threshold: float = 0.75,
    snr: float = 0.9,
    size: int = 512,
) -> Dict[str, bytes]:
    # Arrays are MATLAB-coordinate fields with shape (ny, nx, nz):
    # X=width, Y=thickness, Z=length.
    ny, nx, nz = txx.shape
    if tyy.shape != (ny, nx, nz) or tzz.shape != (ny, nx, nz):
        raise ValueError("Fiber component arrays have incompatible shapes.")
    if mesh_x.size != nx or mesh_y.size != ny or mesh_z.size != nz:
        raise ValueError("Mesh axes are incompatible with fiber array shape.")

    Y2, Z2 = np.meshgrid(mesh_y, mesh_z, indexing="ij")  # (ny, nz)
    X2, Zx2 = np.meshgrid(mesh_x, mesh_z, indexing="ij")  # (nx, nz)

    tyy_xmin = np.asarray(tyy[:, 0, :], dtype=np.float64).copy()
    tzz_xmin = np.asarray(tzz[:, 0, :], dtype=np.float64).copy()
    tyy_xmax = np.asarray(tyy[:, -1, :], dtype=np.float64).copy()
    tzz_xmax = np.asarray(tzz[:, -1, :], dtype=np.float64).copy()
    txx_ymin = np.asarray(txx[0, :, :], dtype=np.float64).copy()
    tzz_ymin = np.asarray(tzz[0, :, :], dtype=np.float64).copy()
    txx_ymax = np.asarray(txx[-1, :, :], dtype=np.float64).copy()
    tzz_ymax = np.asarray(tzz[-1, :, :], dtype=np.float64).copy()

    if rand_fibers:
        # Match MATLAB/UI behavior: perturb in-plane vectors where out-of-plane component is strong.
        tyy_xmin, tzz_xmin = FiberSolver._apply_noise(
            np.asarray(txx[:, 0, :], dtype=np.float64),
            tyy_xmin,
            tzz_xmin,
            out_of_plane_threshold,
            snr,
        )
        tyy_xmax, tzz_xmax = FiberSolver._apply_noise(
            np.asarray(txx[:, -1, :], dtype=np.float64),
            tyy_xmax,
            tzz_xmax,
            out_of_plane_threshold,
            snr,
        )
        txx_ymin, tzz_ymin = FiberSolver._apply_noise(
            np.asarray(tyy[0, :, :], dtype=np.float64),
            txx_ymin,
            tzz_ymin,
            out_of_plane_threshold,
            snr,
        )
        txx_ymax, tzz_ymax = FiberSolver._apply_noise(
            np.asarray(tyy[-1, :, :], dtype=np.float64),
            txx_ymax,
            tzz_ymax,
            out_of_plane_threshold,
            snr,
        )

    # Match MATLAB saveFibers conventions and map to current 4 side names:
    # x_min/x_max are width-side faces; z_min/z_max correspond to thickness-side faces.
    return {
        # x-min face (use in-plane components: Y/Z).
        "x_min": _render_fiber_face_png(
            Y2, Z2, tyy_xmin, tzz_xmin,
            flip_sign=True, flip_x=True, size=size
        ),
        # x-max face.
        "x_max": _render_fiber_face_png(
            Y2, Z2, tyy_xmax, tzz_xmax,
            flip_sign=False, flip_x=False, size=size
        ),
        # z-min (viewer) == y-min (model thickness).
        "z_min": _render_fiber_face_png(
            X2, Zx2, txx_ymin, tzz_ymin,
            flip_sign=False, flip_x=False, size=size
        ),
        # z-max (viewer) == y-max (model thickness).
        "z_max": _render_fiber_face_png(
            X2, Zx2, txx_ymax, tzz_ymax,
            flip_sign=True, flip_x=True, size=size
        ),
    }


def _render_normal_face_png(
    nx_face: np.ndarray,
    ny_face: np.ndarray,
    nz_face: np.ndarray,
    *,
    flip_x: bool,
    size: int = 512,
) -> bytes:
    nx = np.asarray(nx_face, dtype=np.float32)
    ny = np.asarray(ny_face, dtype=np.float32)
    nz = np.asarray(nz_face, dtype=np.float32)

    if nx.ndim != 2 or ny.ndim != 2 or nz.ndim != 2 or nx.shape != ny.shape or nx.shape != nz.shape:
        image = Image.new("RGB", (size, size), color=(128, 128, 255))
        buffer = BytesIO()
        image.save(buffer, format="PNG", optimize=False)
        return buffer.getvalue()

    vec = np.stack([nx, ny, nz], axis=-1)  # (u, v, 3)
    finite = np.isfinite(vec).all(axis=2)
    vec[~finite] = 0.0

    mag = np.linalg.norm(vec, axis=2)
    safe = mag > 1e-8
    if np.any(safe):
        vec[safe] = vec[safe] / mag[safe, None]
    vec[~safe] = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    # RGB encodes normalized normal components:
    # R=(nx+1)/2, G=(ny+1)/2, B=(nz+1)/2.
    rgb = np.clip(0.5 * (vec + 1.0), 0.0, 1.0)

    # Map to image axes: horizontal=u, vertical=v.
    img = np.transpose(rgb, (1, 0, 2))  # (v, u, 3)

    # Match side-image orientation conventions used across exports/viewer overlays.
    img = img[::-1, :, :]
    if flip_x:
        img = img[:, ::-1, :]

    image = Image.fromarray(np.rint(img * 255.0).astype(np.uint8), mode="RGB")
    if image.size != (size, size):
        image = image.resize((size, size), resample=Image.BILINEAR)

    buffer = BytesIO()
    image.save(buffer, format="PNG", optimize=False)
    return buffer.getvalue()


def _build_normal_surface_pngs(
    nx: np.ndarray,
    ny: np.ndarray,
    nz: np.ndarray,
    *,
    size: int = 512,
) -> Dict[str, bytes]:
    # Arrays are MATLAB-coordinate fields with shape (ny, nx, nz):
    # X=width, Y=thickness, Z=length.
    ny_dim, nx_dim, nz_dim = nx.shape
    if ny.shape != (ny_dim, nx_dim, nz_dim) or nz.shape != (ny_dim, nx_dim, nz_dim):
        raise ValueError("Normal component arrays have incompatible shapes.")

    # Match face naming and orientation conventions used elsewhere:
    # x_min/x_max are width-side faces; z_min/z_max are thickness-side faces.
    return {
        "x_min": _render_normal_face_png(
            nx[:, 0, :], ny[:, 0, :], nz[:, 0, :], flip_x=False, size=size
        ),
        "x_max": _render_normal_face_png(
            nx[:, -1, :], ny[:, -1, :], nz[:, -1, :], flip_x=False, size=size
        ),
        "z_min": _render_normal_face_png(
            nx[0, :, :], ny[0, :, :], nz[0, :, :], flip_x=False, size=size
        ),
        "z_max": _render_normal_face_png(
            nx[-1, :, :], ny[-1, :, :], nz[-1, :, :], flip_x=False, size=size
        ),
    }


def _render_out_of_plane_face_png(
    comp_face: np.ndarray,
    *,
    flip_x: bool,
    size: int = 512,
) -> bytes:
    comp = np.asarray(comp_face, dtype=np.float32)
    if comp.ndim != 2:
        image = Image.new("RGB", (size, size), color="#eef7ff")
        buffer = BytesIO()
        image.save(buffer, format="PNG", optimize=False)
        return buffer.getvalue()

    finite = np.isfinite(comp)
    mag = np.zeros_like(comp, dtype=np.float32)
    mag[finite] = np.clip(np.abs(comp[finite]), 0.0, 1.0)

    # Map to image axes: horizontal=u, vertical=v.
    img = np.transpose(mag, (1, 0))  # (v, u)
    img = img[::-1, :]
    if flip_x:
        img = img[:, ::-1]

    # High out-of-plane -> darker warm tone, low -> light cool tone.
    luma = np.rint((1.0 - img) * 255.0).astype(np.uint8)
    image_l = Image.fromarray(luma, mode="L")
    image_rgb = ImageOps.colorize(image_l, black="#7d0015", white="#eef7ff")

    if image_rgb.size != (size, size):
        image_rgb = image_rgb.resize((size, size), resample=Image.BILINEAR)

    buffer = BytesIO()
    image_rgb.save(buffer, format="PNG", optimize=False)
    return buffer.getvalue()


def _build_fiber_out_of_plane_surface_pngs(
    txx: np.ndarray,
    tyy: np.ndarray,
    *,
    size: int = 512,
) -> Dict[str, bytes]:
    # Arrays are MATLAB-coordinate fields with shape (ny, nx, nz):
    # X=width, Y=thickness, Z=length.
    ny, nx, nz = txx.shape
    if tyy.shape != (ny, nx, nz):
        raise ValueError("Fiber component arrays have incompatible shapes.")

    return {
        # x-min/x-max faces: out-of-plane component is X.
        "x_min": _render_out_of_plane_face_png(txx[:, 0, :], flip_x=False, size=size),
        "x_max": _render_out_of_plane_face_png(txx[:, -1, :], flip_x=False, size=size),
        # z-min/z-max viewer faces correspond to y-min/y-max model faces:
        # out-of-plane component is Y.
        "z_min": _render_out_of_plane_face_png(tyy[0, :, :], flip_x=False, size=size),
        "z_max": _render_out_of_plane_face_png(tyy[-1, :, :], flip_x=False, size=size),
    }


def _build_matlab_bundle_png_payload(
    entry: Dict[str, Any],
    *,
    show_inside: bool = False,
    include_middle_surface: bool = False,
    image_size: int = 512,
    rand_fibers: bool = False,
    out_of_plane_threshold: float = 0.75,
    snr: float = 0.9,
    contour_line_width: float = 1.0,
    contour_blur_sigma: float = 0.0,
    fiber_blur_sigma: float = 0.0,
    ring_irregularity_strength: float = _DEFAULT_RING_IRREGULARITY_STRENGTH,
    fiber_irregularity_strength: float = _DEFAULT_FIBER_IRREGULARITY_STRENGTH,
    imid: int = 1,
    use_rings_only: bool = False,
) -> Dict[str, Any]:
    board_dims = entry.get("board_dimensions") or {}
    required_dim_keys = ["x_min", "x_max", "y_min", "y_max", "z_min", "z_max"]
    if any(k not in board_dims for k in required_dim_keys):
        raise HTTPException(status_code=400, detail="Cached board dimensions are incomplete.")

    if show_inside:
        contours_mat = entry.get("contours_unmasked") or entry.get("contours") or []
    else:
        contours_mat = (
            entry.get("contours_masked_live")
            or entry.get("contours_masked")
            or entry.get("contours")
            or []
        )
    if not contours_mat:
        raise HTTPException(status_code=400, detail="No contour data available in cached simulation.")

    render_size = max(32, int(image_size))

    ring_pngs = _build_matlab_ring_pngs(
        contours_mat,
        board_dims,
        size=render_size,
        line_width=float(contour_line_width),
    )
    if bool(include_middle_surface):
        contours_mid_mat = (
            (entry.get("contours_mid_unmasked") or [])
            if bool(show_inside)
            else (
                (entry.get("contours_mid_masked_live") or [])
                or (entry.get("contours_mid_masked") or [])
            )
        )
        if not contours_mid_mat:
            raise HTTPException(
                status_code=400,
                detail=(
                    "No cached middle-surface contour data available. "
                    "Regenerate board and try the middle-surface export again."
                ),
            )
        ring_pngs["rings_5"] = _build_matlab_mid_ring_png(
            contours_mid_mat,
            board_dims,
            size=render_size,
            line_width=float(contour_line_width),
        )
    ring_pngs_final: Dict[str, bytes] = {}
    ring_folder_order = ["rings_1", "rings_2", "rings_3", "rings_4"]
    if bool(include_middle_surface):
        ring_folder_order.append("rings_5")
    for ring_folder in ring_folder_order:
        png_bytes = _apply_png_blur_bytes(ring_pngs[ring_folder], float(contour_blur_sigma))
        png_bytes = _flip_png_vertical_bytes(png_bytes)
        png_bytes = _apply_ring_irregularity_bytes(png_bytes, float(ring_irregularity_strength))
        ring_pngs_final[ring_folder] = png_bytes

    fiber_pngs_final: Dict[str, bytes] = {}
    if not bool(use_rings_only):
        fibers = entry.get("fibers") or {}
        mesh_axes = entry.get("mesh_axes") or {}
        txx = np.asarray(fibers.get("txx", np.empty((0,), dtype=np.float32)), dtype=np.float64)
        tyy = np.asarray(fibers.get("tyy", np.empty((0,), dtype=np.float32)), dtype=np.float64)
        tzz = np.asarray(fibers.get("tzz", np.empty((0,), dtype=np.float32)), dtype=np.float64)
        if txx.ndim != 3 or tyy.ndim != 3 or tzz.ndim != 3 or txx.size == 0:
            raise HTTPException(
                status_code=400,
                detail="No 3D fiber field available in cached simulation. Generate board with fibers enabled first.",
            )

        mesh_x = np.asarray(mesh_axes.get("x", np.empty((0,), dtype=np.float32)), dtype=np.float64).reshape(-1)
        mesh_y = np.asarray(mesh_axes.get("y", np.empty((0,), dtype=np.float32)), dtype=np.float64).reshape(-1)
        mesh_z = np.asarray(mesh_axes.get("z", np.empty((0,), dtype=np.float32)), dtype=np.float64).reshape(-1)
        if mesh_x.size < 2 or mesh_y.size < 2 or mesh_z.size < 2:
            raise HTTPException(status_code=400, detail="Cached mesh axes are missing. Regenerate board and retry.")

        fiber_pngs_by_side = _build_fiber_surface_pngs(
            txx,
            tyy,
            tzz,
            mesh_x,
            mesh_y,
            mesh_z,
            rand_fibers=bool(rand_fibers),
            out_of_plane_threshold=float(out_of_plane_threshold),
            snr=float(snr),
            size=render_size,
        )

        # MATLAB saveFibers order:
        # fiber_1 -> y_max, fiber_2 -> y_min, fiber_3 -> x_max, fiber_4 -> x_min.
        # In current viewer naming, y_min/y_max map to z_min/z_max respectively.
        fiber_side_order = [
            ("fiber_1", "z_max"),
            ("fiber_2", "z_min"),
            ("fiber_3", "x_max"),
            ("fiber_4", "x_min"),
        ]

        for fiber_folder, side_key in fiber_side_order:
            png_bytes = _apply_png_blur_bytes(fiber_pngs_by_side[side_key], float(fiber_blur_sigma))
            png_bytes = _flip_png_vertical_bytes(png_bytes)
            png_bytes = _apply_fiber_irregularity_bytes(png_bytes, float(fiber_irregularity_strength))
            fiber_pngs_final[fiber_folder] = png_bytes

    imid_safe = max(0, int(imid))
    return {
        "filename": f"{imid_safe:05d}.png",
        "rings": ring_pngs_final,
        "fibers": fiber_pngs_final,
    }


@app.get("/")
def read_root():
    return {"message": "Board Generator API"}


@app.get("/capabilities")
def get_capabilities():
    return {
        "photorealistic_export": get_photorealistic_capability(),
    }


@app.post("/photorealistic/preload")
def preload_photorealistic():
    try:
        preload_info = preload_photorealistic_model()
        return {
            "ok": True,
            "loaded": bool(preload_info.get("loaded")),
            "capability": get_photorealistic_capability(),
        }
    except PhotorealisticUnavailableError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except PhotorealisticInferenceError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to preload photorealistic model: {e}")


@app.post("/export/contours")
def export_contours(req: ExportContoursRequest):
    try:
        contours: List[List[List[float]]] = []
        board_outline: Optional[Dict[str, List[float]]] = None
        blur_sigma = float(req.blur_sigma) if req.blur_sigma is not None else 0.0

        sim_id = str(req.simulation_id or "").strip()
        if sim_id:
            entry = _SIM_CACHE.get(sim_id)
            if entry is None:
                raise HTTPException(status_code=404, detail="Simulation data not found. Regenerate board before export.")

            show_inside = bool(req.show_rings_inside_knots) if req.show_rings_inside_knots is not None else True
            if show_inside:
                contours_mat = entry.get("contours_unmasked") or entry.get("contours") or []
            else:
                contours_mat = (
                    entry.get("contours_masked_live")
                    or entry.get("contours_masked")
                    or entry.get("contours")
                    or []
                )
            contours = _contours_mat_to_viewer(contours_mat)
            board_outline = _board_outline_from_cached_entry(entry)
        else:
            contours = req.contours if isinstance(req.contours, list) else []
            board_outline = req.board_outline if isinstance(req.board_outline, dict) else None

        if len(contours) == 0:
            raise HTTPException(status_code=400, detail="No contours available to export.")
        if not board_outline:
            raise HTTPException(status_code=400, detail="Board outline missing for contour export.")

        surface_info = _surface_meta(board_outline or {})
        per_surface = {key: [] for key in surface_info.keys()}

        for line in contours:
            if not isinstance(line, list) or len(line) < 2:
                continue
            arr = np.asarray(line, dtype=float)
            if arr.ndim != 2 or arr.shape[1] != 3 or arr.shape[0] < 2:
                continue
            key = _classify_surface(arr, surface_info)
            if key in per_surface:
                per_surface[key].append(arr)

        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
            for key in ["x_min", "x_max", "z_min", "z_max"]:
                png_bytes = _render_surface_png(per_surface[key], surface_info[key], size=512)
                png_bytes = _apply_png_blur_bytes(png_bytes, blur_sigma)
                zf.writestr(f"contours_{key}_512.png", png_bytes)

        return Response(
            content=zip_buffer.getvalue(),
            media_type="application/zip",
            headers={"Content-Disposition": "attachment; filename=contour_surfaces_512.zip"},
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to export contours: {e}")


@app.post("/export/mat")
def export_mat(req: ExportMatRequest):
    try:
        sim_id = str(req.simulation_id or "").strip()
        if not sim_id:
            raise HTTPException(status_code=400, detail="simulation_id is required.")
        entry = _SIM_CACHE.get(sim_id)
        if entry is None:
            raise HTTPException(status_code=404, detail="Simulation data not found. Regenerate board before export.")

        fibers = entry.get("fibers") or {}
        normals = entry.get("normals") or {}
        mesh_axes = entry.get("mesh_axes") or {}
        mesh_grids = entry.get("mesh_grids") or {}
        scalar_fields = entry.get("scalar_fields") or {}
        board_dims = entry.get("board_dimensions") or {}
        contours = entry.get("contours") or []
        growth_layers = entry.get("growth_layers") or []
        pith_surface = entry.get("pith_surface")
        knots = entry.get("knots") or []
        photorealistic_faces = entry.get("photorealistic_faces") or []
        export_mode = str(entry.get("export_mode") or "board")
        fiber_domain = str(entry.get("fiber_domain") or export_mode)
        knot_sequence_segments = _knot_sequence_segments_to_mat_struct(
            entry.get("knot_sequence") or {},
            entry.get("geometry_randomization") or {},
            board_dims,
            knots,
        )

        mat_payload = {
            "fiber_txx": np.asarray(fibers.get("txx", np.empty((0,), dtype=np.float32)), dtype=np.float32),
            "fiber_tyy": np.asarray(fibers.get("tyy", np.empty((0,), dtype=np.float32)), dtype=np.float32),
            "fiber_tzz": np.asarray(fibers.get("tzz", np.empty((0,), dtype=np.float32)), dtype=np.float32),
            "normal_nx": np.asarray(normals.get("nx", np.empty((0,), dtype=np.float32)), dtype=np.float32),
            "normal_ny": np.asarray(normals.get("ny", np.empty((0,), dtype=np.float32)), dtype=np.float32),
            "normal_nz": np.asarray(normals.get("nz", np.empty((0,), dtype=np.float32)), dtype=np.float32),
            "contours": _contours_to_mat_cell(contours),
            "contours_masked": _contours_to_mat_cell(entry.get("contours_masked") or []),
            "contours_masked_live": _contours_to_mat_cell(entry.get("contours_masked_live") or []),
            "contours_unmasked": _contours_to_mat_cell(entry.get("contours_unmasked") or []),
            "growth_layers": _mat_mesh_payloads_to_struct_array(growth_layers),
            "pith_surface": _mat_mesh_payloads_to_struct_array([pith_surface] if pith_surface else []),
            "knots": _mat_mesh_payloads_to_struct_array(knots),
            "knot_sequence_segments": knot_sequence_segments,
            "photorealistic_faces": _photorealistic_faces_to_mat_struct_array(photorealistic_faces),
            "mesh_x": np.asarray(mesh_axes.get("x", np.empty((0,), dtype=np.float32)), dtype=np.float32).reshape(-1),
            "mesh_y": np.asarray(mesh_axes.get("y", np.empty((0,), dtype=np.float32)), dtype=np.float32).reshape(-1),
            "mesh_z": np.asarray(mesh_axes.get("z", np.empty((0,), dtype=np.float32)), dtype=np.float32).reshape(-1),
            "mesh_grid_x": _to_float32_3d(mesh_grids.get("x")),
            "mesh_grid_y": _to_float32_3d(mesh_grids.get("y")),
            "mesh_grid_z": _to_float32_3d(mesh_grids.get("z")),
            "knot_field": _to_float32_3d(scalar_fields.get("knot_field")),
            "growth_layer_fields": np.asarray(
                scalar_fields.get("growth_layer_fields", np.empty((0, 0, 0, 0), dtype=np.float32)),
                dtype=np.float32,
            ),
            "growth_layer_indices": np.asarray(
                scalar_fields.get("growth_layer_indices", np.empty((0,), dtype=np.int32)),
                dtype=np.int32,
            ).reshape(-1),
            "raw_field_isovalue": np.array([[0.0]], dtype=np.float32),
            "board_dimensions": {
                "x_min": float(board_dims.get("x_min", 0.0)),
                "x_max": float(board_dims.get("x_max", 0.0)),
                "y_min": float(board_dims.get("y_min", 0.0)),
                "y_max": float(board_dims.get("y_max", 0.0)),
                "z_min": float(board_dims.get("z_min", 0.0)),
                "z_max": float(board_dims.get("z_max", 0.0)),
                "width": float(board_dims.get("width", 0.0)),
                "thickness": float(board_dims.get("thickness", 0.0)),
                "length": float(board_dims.get("length", 0.0)),
            },
            "coordinate_system": np.array(
                ["X=width, Y=thickness, Z=length (MATLAB coordinates)"], dtype=object
            ),
            "export_mode": np.array([export_mode], dtype=object),
            "fiber_domain": np.array([fiber_domain], dtype=object),
            "simulation_id": np.array([sim_id], dtype=object),
        }

        mat_buf = BytesIO()
        scipy.io.savemat(mat_buf, mat_payload, do_compression=True)
        mat_filename = f"board_export_{sim_id}.mat"

        script_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "visualize_exported_board.m")
        )
        zip_buf = BytesIO()
        with zipfile.ZipFile(zip_buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr(mat_filename, mat_buf.getvalue())
            if os.path.exists(script_path):
                with open(script_path, "rb") as f:
                    zf.writestr("visualize_exported_board.m", f.read())
            else:
                zf.writestr(
                    "visualize_exported_board.m",
                    "% visualize_exported_board.m not found on server.\n"
                    "% Ensure the script exists at repository root.\n",
                )

        zip_filename = f"board_export_{sim_id}_with_visualizer.zip"
        return Response(
            content=zip_buf.getvalue(),
            media_type="application/zip",
            headers={"Content-Disposition": f'attachment; filename="{zip_filename}"'},
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to export MAT data: {e}")


@app.post("/export/fibers")
def export_fibers(req: ExportFibersRequest):
    try:
        sim_id = str(req.simulation_id or "").strip()
        if not sim_id:
            raise HTTPException(status_code=400, detail="simulation_id is required.")
        entry = _SIM_CACHE.get(sim_id)
        if entry is None:
            raise HTTPException(status_code=404, detail="Simulation data not found. Regenerate board before export.")

        fibers = entry.get("fibers") or {}
        mesh_axes = entry.get("mesh_axes") or {}

        txx = np.asarray(fibers.get("txx", np.empty((0,), dtype=np.float32)), dtype=np.float64)
        tyy = np.asarray(fibers.get("tyy", np.empty((0,), dtype=np.float32)), dtype=np.float64)
        tzz = np.asarray(fibers.get("tzz", np.empty((0,), dtype=np.float32)), dtype=np.float64)
        if txx.ndim != 3 or tyy.ndim != 3 or tzz.ndim != 3 or txx.size == 0:
            raise HTTPException(
                status_code=400,
                detail="No 3D fiber field available in cached simulation. Generate board with fibers enabled first.",
            )

        mesh_x = np.asarray(mesh_axes.get("x", np.empty((0,), dtype=np.float32)), dtype=np.float64).reshape(-1)
        mesh_y = np.asarray(mesh_axes.get("y", np.empty((0,), dtype=np.float32)), dtype=np.float64).reshape(-1)
        mesh_z = np.asarray(mesh_axes.get("z", np.empty((0,), dtype=np.float32)), dtype=np.float64).reshape(-1)
        if mesh_x.size < 2 or mesh_y.size < 2 or mesh_z.size < 2:
            raise HTTPException(status_code=400, detail="Cached mesh axes are missing. Regenerate board and retry.")

        rand_fibers = bool(req.rand_fibers) if req.rand_fibers is not None else False
        out_of_plane_threshold = float(req.out_of_plane_threshold) if req.out_of_plane_threshold is not None else 0.75
        snr = float(req.snr) if req.snr is not None else 0.9
        blur_sigma = float(req.blur_sigma) if req.blur_sigma is not None else 0.0

        pngs = _build_fiber_surface_pngs(
            txx,
            tyy,
            tzz,
            mesh_x,
            mesh_y,
            mesh_z,
            rand_fibers=rand_fibers,
            out_of_plane_threshold=out_of_plane_threshold,
            snr=snr,
            size=512,
        )

        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
            for key in ["x_min", "x_max", "z_min", "z_max"]:
                zf.writestr(f"fibers_{key}_512.png", _apply_png_blur_bytes(pngs[key], blur_sigma))

        return Response(
            content=zip_buffer.getvalue(),
            media_type="application/zip",
            headers={"Content-Disposition": 'attachment; filename="fiber_surfaces_512.zip"'},
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to export fibers: {e}")


@app.post("/export/matlab-image-bundle")
def export_matlab_image_bundle(req: ExportMatlabImageBundleRequest):
    try:
        sim_id = str(req.simulation_id or "").strip()
        if not sim_id:
            raise HTTPException(status_code=400, detail="simulation_id is required.")

        entry = _SIM_CACHE.get(sim_id)
        if entry is None:
            raise HTTPException(status_code=404, detail="Simulation data not found. Regenerate board before export.")

        show_inside = bool(req.show_rings_inside_knots) if req.show_rings_inside_knots is not None else False
        rand_fibers = bool(req.rand_fibers) if req.rand_fibers is not None else False
        out_of_plane_threshold = float(req.out_of_plane_threshold) if req.out_of_plane_threshold is not None else 0.75
        snr = float(req.snr) if req.snr is not None else 0.9
        contour_line_width = float(req.contour_line_width) if req.contour_line_width is not None else 1.0
        contour_blur_sigma = float(req.contour_blur_sigma) if req.contour_blur_sigma is not None else 0.0
        fiber_blur_sigma = float(req.fiber_blur_sigma) if req.fiber_blur_sigma is not None else 0.0
        ring_irregularity_strength = (
            float(req.ring_irregularity_strength)
            if req.ring_irregularity_strength is not None
            else _DEFAULT_RING_IRREGULARITY_STRENGTH
        )
        fiber_irregularity_strength = (
            float(req.fiber_irregularity_strength)
            if req.fiber_irregularity_strength is not None
            else _DEFAULT_FIBER_IRREGULARITY_STRENGTH
        )
        imid = int(req.imid) if req.imid is not None else 1
        payload = _build_matlab_bundle_png_payload(
            entry,
            show_inside=show_inside,
            include_middle_surface=bool(req.include_middle_surface),
            rand_fibers=rand_fibers,
            out_of_plane_threshold=out_of_plane_threshold,
            snr=snr,
            contour_line_width=contour_line_width,
            contour_blur_sigma=contour_blur_sigma,
            fiber_blur_sigma=fiber_blur_sigma,
            ring_irregularity_strength=ring_irregularity_strength,
            fiber_irregularity_strength=fiber_irregularity_strength,
            imid=imid,
        )

        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
            for ring_folder in sorted(payload["rings"].keys(), key=lambda name: int(str(name).split("_")[-1])):
                zf.writestr(f"output/{ring_folder}/{payload['filename']}", payload["rings"][ring_folder])
            for fiber_folder in ["fiber_1", "fiber_2", "fiber_3", "fiber_4"]:
                zf.writestr(f"output/{fiber_folder}/{payload['filename']}", payload["fibers"][fiber_folder])

        return Response(
            content=zip_buffer.getvalue(),
            media_type="application/zip",
            headers={"Content-Disposition": f'attachment; filename="matlab_image_bundle_{sim_id}.zip"'},
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to export MATLAB image bundle: {e}")


@app.post("/export/photorealistic-surfaces")
def export_photorealistic_surfaces(req: ExportPhotorealisticRequest):
    try:
        sim_id = str(req.simulation_id or "").strip()
        if not sim_id:
            raise HTTPException(status_code=400, detail="simulation_id is required.")

        entry = _SIM_CACHE.get(sim_id)
        if entry is None:
            raise HTTPException(status_code=404, detail="Simulation data not found. Regenerate board before export.")

        capability = get_photorealistic_capability()
        if not bool(capability.get("available")):
            reason = str(capability.get("reason") or "Photorealistic inference is unavailable.")
            raise HTTPException(status_code=503, detail=reason)

        show_inside = bool(req.show_rings_inside_knots) if req.show_rings_inside_knots is not None else False
        rand_fibers = bool(req.rand_fibers) if req.rand_fibers is not None else False
        out_of_plane_threshold = float(req.out_of_plane_threshold) if req.out_of_plane_threshold is not None else 0.75
        snr = float(req.snr) if req.snr is not None else 0.9
        contour_line_width = float(req.contour_line_width) if req.contour_line_width is not None else 1.0
        contour_blur_sigma = float(req.contour_blur_sigma) if req.contour_blur_sigma is not None else 0.0
        fiber_blur_sigma = float(req.fiber_blur_sigma) if req.fiber_blur_sigma is not None else 0.0
        ring_irregularity_strength = (
            float(req.ring_irregularity_strength)
            if req.ring_irregularity_strength is not None
            else _DEFAULT_RING_IRREGULARITY_STRENGTH
        )
        fiber_irregularity_strength = (
            float(req.fiber_irregularity_strength)
            if req.fiber_irregularity_strength is not None
            else _DEFAULT_FIBER_IRREGULARITY_STRENGTH
        )
        imid = int(req.imid) if req.imid is not None else 1
        use_rings_only = bool(req.use_rings_only) if req.use_rings_only is not None else False
        include_knot_maps = bool(req.include_knot_maps) if req.include_knot_maps is not None else False
        if bool(use_rings_only) and bool(include_knot_maps):
            raise HTTPException(
                status_code=400,
                detail="use_rings_only=true cannot be combined with include_knot_maps=true.",
            )

        payload = _build_matlab_bundle_png_payload(
            entry,
            show_inside=show_inside,
            include_middle_surface=False,
            rand_fibers=rand_fibers,
            out_of_plane_threshold=out_of_plane_threshold,
            snr=snr,
            contour_line_width=contour_line_width,
            contour_blur_sigma=contour_blur_sigma,
            fiber_blur_sigma=fiber_blur_sigma,
            ring_irregularity_strength=ring_irregularity_strength,
            fiber_irregularity_strength=fiber_irregularity_strength,
            imid=imid,
            use_rings_only=use_rings_only,
        )

        generated = generate_photorealistic_surfaces(
            payload["rings"],
            payload["fibers"],
            ddim_steps=req.ddim_steps,
            guidance_scale=req.guidance_scale,
            use_img2img_strength=req.use_img2img_strength,
            include_knot_maps=include_knot_maps,
            use_rings_only=use_rings_only,
        )

        # surface_1..4 follow MATLAB-like side order:
        # 1 -> z_max, 2 -> z_min, 3 -> x_max, 4 -> x_min.
        # flip_x follows the same conventions used for bundle ring/fiber images.
        surface_face_map: Dict[str, Dict[str, Any]] = {
            "surface_1": {"face": "z_max", "flip_x": True},
            "surface_2": {"face": "z_min", "flip_x": False},
            "surface_3": {"face": "x_max", "flip_x": False},
            "surface_4": {"face": "x_min", "flip_x": True},
        }

        photorealistic_faces: List[Dict[str, Any]] = []
        for idx in range(1, 5):
            surf_key = f"surface_{idx}"
            png_bytes = generated[surf_key]
            photorealistic_faces.append({
                "face": str(surface_face_map[surf_key]["face"]),
                "flip_x": bool(surface_face_map[surf_key]["flip_x"]),
                "filename": f"photorealistic_{idx}_{payload['filename']}",
                "image": _png_bytes_to_rgb_array(png_bytes),
            })
        entry["photorealistic_faces"] = photorealistic_faces

        zip_filename = f"photorealistic_surfaces_{sim_id}.zip"
        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
            for idx in range(1, 5):
                surf_key = f"surface_{idx}"
                zf.writestr(
                    f"output/photorealistic_{idx}/{payload['filename']}",
                    generated[surf_key],
                )
        zip_bytes = zip_buffer.getvalue()

        if bool(req.include_base64):
            surfaces: Dict[str, Dict[str, Any]] = {}
            for idx in range(1, 5):
                surf_key = f"surface_{idx}"
                face_key = str(surface_face_map[surf_key]["face"])
                surfaces[surf_key] = {
                    "face": face_key,
                    "flip_x": bool(surface_face_map[surf_key]["flip_x"]),
                    "filename": f"photorealistic_{idx}_{payload['filename']}",
                    "png_base64": base64.b64encode(generated[surf_key]).decode("ascii"),
                }

            return {
                "ok": True,
                "simulation_id": sim_id,
                "image_id_filename": payload["filename"],
                "surfaces": surfaces,
                "zip_filename": zip_filename,
                "zip_base64": base64.b64encode(zip_bytes).decode("ascii"),
            }

        return Response(
            content=zip_bytes,
            media_type="application/zip",
            headers={"Content-Disposition": f'attachment; filename="{zip_filename}"'},
        )
    except HTTPException:
        raise
    except PhotorealisticUnavailableError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except PhotorealisticInferenceError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to export photorealistic surfaces: {e}")


@app.post("/simulate")
def simulate(config: BoardConfig):
    try:
        seeded_mode = bool(config.use_seed)
        base_seed = int(config.simulation_seed) if seeded_mode else 0

        # 1. Initialize + random-board retries for UI mode.
        # Without retries, some random cross-sections can miss the board footprint
        # and produce empty contour sets.
        board_mode = int(getattr(config, "board_or_log", 0)) == 0
        if not board_mode:
            # Log-mode fiber display is intentionally disabled in the UI, so the
            # frontend sends calc_fibers=false. Still compute the field for MAT export.
            config.calc_fibers = True
            config.quiver_or_stream = 0
        randomize_extents_from_dims = bool(
            board_mode and getattr(config, "randomize_board_extents_from_dimensions", False)
        )
        if randomize_extents_from_dims:
            if (
                float(getattr(config, "board_width", 0.0)) <= 0.0
                or float(getattr(config, "board_thickness", 0.0)) <= 0.0
                or float(getattr(config, "board_length", 0.0)) <= 0.0
            ):
                raise HTTPException(
                    status_code=422,
                    detail=(
                        "Dimension mode requires positive board dimensions: "
                        "board_width > 0, board_thickness > 0, board_length > 0."
                    ),
                )
        # For seeded runs, keep retry policy aligned between board/log modes so
        # both modes can converge to the same accepted stochastic sample.
        max_attempts = _SIM_MAX_BOARD_ATTEMPTS if (board_mode or seeded_mode) else 1

        warnings: List[str] = []
        last_reject_reason = ""
        retries_used = 0
        k = None
        mesh = None
        layers_data = {}
        mesh_accum = {}

        for attempt_idx in range(max_attempts):
            if seeded_mode:
                seed_all(base_seed + int(attempt_idx))

            k = KnotSystem(config)

            if randomize_extents_from_dims:
                try:
                    width = float(config.board_width)
                    thickness = float(config.board_thickness)
                    length = float(config.board_length)
                    cx, cy = _sample_center_within_log(
                        getattr(k, "splines", []) or [],
                        width=width,
                        thickness=thickness,
                    )
                    half_w = 0.5 * width
                    half_t = 0.5 * thickness
                    config.board_x_min = float(cx - half_w)
                    config.board_x_max = float(cx + half_w)
                    config.board_y_min = float(cy - half_t)
                    config.board_y_max = float(cy + half_t)
                    config.board_z_min = 0.0
                    config.board_z_max = float(length)
                except _RetryablePlacementError as exc:
                    last_reject_reason = str(exc)
                    retries_used += 1
                    continue

            mesh = BoardMesh(config, k)
            enforce_seeded_fit_retry = seeded_mode and (not board_mode)
            fit_warnings = _board_fit_warnings(
                config,
                mesh,
                k,
                force_check=enforce_seeded_fit_retry,
            )
            if (board_mode or seeded_mode) and fit_warnings:
                last_reject_reason = fit_warnings[0]
                retries_used += 1
                continue

            layers_data, mesh_accum = GrowthSimulator.run(config, mesh, k)
            if (board_mode or seeded_mode) and bool(getattr(config, "display_contours", False)) and not _has_any_contours(layers_data):
                last_reject_reason = "Generated board had no ring contours."
                retries_used += 1
                continue

            warnings = list(fit_warnings)
            break
        else:
            reason = (
                f" Last rejection: {last_reject_reason}"
                if last_reject_reason
                else ""
            )
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Failed to generate a valid random board after {max_attempts} attempts."
                    f"{reason}"
                ),
            )

        if retries_used > 0:
            warnings.append(
                f"Auto-retried stochastic geometry generation {retries_used} time(s) to find a valid board/log intersection."
            )

        # 2. Continue with selected successful simulation
        def swap_segments(segments):
            return [[swap_yz(seg[0]), swap_yz(seg[1])] for seg in segments]

        # Normal field in MATLAB coordinates (X width, Y thickness, Z length).
        nx_mat = np.empty((0,), dtype=np.float32)
        ny_mat = np.empty((0,), dtype=np.float32)
        nz_mat = np.empty((0,), dtype=np.float32)
        normal_overlays = None
        normal_vector_data = None
        if config.board_or_log == 0:
            nx_mat = np.asarray(to_numpy(mesh_accum.get('grid_nx', np.empty((0,)))), dtype=np.float32)
            ny_mat = np.asarray(to_numpy(mesh_accum.get('grid_ny', np.empty((0,)))), dtype=np.float32)
            nz_mat = np.asarray(to_numpy(mesh_accum.get('grid_nz', np.empty((0,)))), dtype=np.float32)
            if nx_mat.size and ny_mat.size and nz_mat.size:
                nmag = np.sqrt(nx_mat**2 + ny_mat**2 + nz_mat**2).astype(np.float32, copy=False)
                inv = np.where(nmag > 1e-12, 1.0 / nmag, 0.0).astype(np.float32, copy=False)
                nx_mat = (nx_mat * inv).astype(np.float32, copy=False)
                ny_mat = (ny_mat * inv).astype(np.float32, copy=False)
                nz_mat = np.where(nmag > 1e-12, nz_mat * inv, 1.0).astype(np.float32, copy=False)

            if nx_mat.ndim == 3 and ny_mat.shape == nx_mat.shape and nz_mat.shape == nx_mat.shape and nx_mat.size:
                try:
                    normal_pngs = _build_normal_surface_pngs(nx_mat, ny_mat, nz_mat, size=512)
                    normal_overlays = {}
                    for face_key in ["x_min", "x_max", "z_min", "z_max"]:
                        png_bytes = normal_pngs.get(face_key)
                        if not png_bytes:
                            continue
                        normal_overlays[face_key] = {
                            "filename": f"normals_{face_key}_512.png",
                            "src": f"data:image/png;base64,{base64.b64encode(png_bytes).decode('ascii')}",
                        }
                except Exception as e:
                    print(f"Normal overlay render error: {e}")
                    normal_overlays = None
                try:
                    normal_surface_segments = FiberSolver.build_surface_normal_quiver3d(
                        mesh,
                        nx_mat,
                        ny_mat,
                        nz_mat,
                    )
                    if normal_surface_segments:
                        normal_vector_data = {
                            "surface_quiver3d": swap_segments(normal_surface_segments),
                        }
                except Exception as e:
                    print(f"Normal quiver build error: {e}")
                    normal_vector_data = None
        
        # 3. Build response
        def _surface_to_viewer_payload(surf: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
            if not isinstance(surf, dict):
                return None
            verts = np.asarray(surf.get('vertices', []), dtype=float)
            faces = np.asarray(surf.get('faces', []), dtype=int)
            if verts.ndim != 2 or verts.shape[1] != 3:
                return None
            if faces.ndim != 2 or faces.shape[1] < 3:
                return None

            # Swap Y<->Z for Three.js (Y-up).
            # MATLAB: (X=Width, Y=Thickness, Z=Length)
            # Three.js: (X=Width, Y=Length(up), Z=Thickness)
            swapped_verts = np.column_stack([
                verts[:, 0],  # X stays
                verts[:, 2],  # Z(Length) -> Y(up)
                verts[:, 1],  # Y(Thickness) -> Z
            ])
            payload = {
                'vertices': swapped_verts.tolist(),
                'faces': faces[:, :3].tolist(),
            }
            layer_index = surf.get('layer_index')
            if layer_index is not None:
                try:
                    payload['layer_index'] = int(layer_index)
                except (TypeError, ValueError):
                    pass
            return payload

        response_layers = []
        if 'surfaces' in layers_data:
            for surf in layers_data['surfaces']:
                payload = _surface_to_viewer_payload(surf)
                if payload is not None:
                    response_layers.append(payload)

        response_pith_layer = _surface_to_viewer_payload(layers_data.get('pith_surface'))
        if response_pith_layer is None and len(response_layers) > 0:
            response_pith_layer = response_layers[0]

        growth_layers_mat = _surfaces_to_mat_mesh_payloads(layers_data.get('surfaces') or [])
        pith_surface_mat = _surface_to_mat_mesh_payload(layers_data.get('pith_surface'))

        # 4. Build contour data (board face ring patterns)
        contour_data = []
        contour_data_mat = []
        contour_data_masked = []
        contour_data_masked_live = []
        contour_data_unmasked = []
        contour_data_masked_mat = []
        contour_data_masked_live_mat = []
        contour_data_unmasked_mat = []
        contour_data_mid_masked_mat = []
        contour_data_mid_masked_live_mat = []
        contour_data_mid_unmasked_mat = []
        if 'contours' in layers_data:
            for line_points in layers_data['contours']:
                contour_data_mat.append(np.asarray(line_points, dtype=np.float32))
                swapped = [[p[0], p[2], p[1]] for p in line_points]
                contour_data.append(swapped)
        if 'contours_masked' in layers_data:
            for line_points in layers_data['contours_masked']:
                contour_data_masked_mat.append(np.asarray(line_points, dtype=np.float32))
                contour_data_masked.append([[p[0], p[2], p[1]] for p in line_points])
        if 'contours_masked_live' in layers_data:
            for line_points in layers_data['contours_masked_live']:
                contour_data_masked_live_mat.append(np.asarray(line_points, dtype=np.float32))
                contour_data_masked_live.append([[p[0], p[2], p[1]] for p in line_points])
        if 'contours_unmasked' in layers_data:
            for line_points in layers_data['contours_unmasked']:
                contour_data_unmasked_mat.append(np.asarray(line_points, dtype=np.float32))
                contour_data_unmasked.append([[p[0], p[2], p[1]] for p in line_points])
        if 'contours_mid_masked' in layers_data:
            for line_points in layers_data['contours_mid_masked']:
                contour_data_mid_masked_mat.append(np.asarray(line_points, dtype=np.float32))
        if 'contours_mid_masked_live' in layers_data:
            for line_points in layers_data['contours_mid_masked_live']:
                contour_data_mid_masked_live_mat.append(np.asarray(line_points, dtype=np.float32))
        if 'contours_mid_unmasked' in layers_data:
            for line_points in layers_data['contours_mid_unmasked']:
                contour_data_mid_unmasked_mat.append(np.asarray(line_points, dtype=np.float32))

        # 5. Board outline (wireframe box)
        bc = mesh.board_coords
        x0, x1 = bc['x']
        y0, y1 = bc['y']  # Thickness
        z0, z1 = bc['z']  # Length
        
        # 8 vertices of the box, swapped for Three.js
        board_outline = {
            'min': swap_yz([x0, y0, z0]),
            'max': swap_yz([x1, y1, z1]),
        }
        board_dimensions = {
            "x_min": float(x0),
            "x_max": float(x1),
            "y_min": float(y0),
            "y_max": float(y1),
            "z_min": float(z0),
            "z_max": float(z1),
            "width": float(abs(x1 - x0)),
            "thickness": float(abs(y1 - y0)),
            "length": float(abs(z1 - z0)),
        }
        mesh_axes = {
            # Three.js coordinates (X=width, Y=length, Z=thickness)
            'x': np.asarray(getattr(mesh, 'x_coords', []), dtype=float).tolist(),
            'y': np.asarray(getattr(mesh, 'z_coords', []), dtype=float).tolist(),
            'z': np.asarray(getattr(mesh, 'y_coords', []), dtype=float).tolist(),
        }
        mesh_axes_mat = {
            "x": np.asarray(getattr(mesh, 'x_coords', []), dtype=np.float32),
            "y": np.asarray(getattr(mesh, 'y_coords', []), dtype=np.float32),
            "z": np.asarray(getattr(mesh, 'z_coords', []), dtype=np.float32),
        }

        # 6. Knot isosurface (single board/log mesh-based knot field surface)
        knot_data = []
        knot_data_mat = []
        if layers_data.get('ttt') is not None:
            try:
                from skimage.measure import marching_cubes

                # Scale to world coordinates
                x_coords = (
                    np.asarray(mesh.x_coords, dtype=float)
                    if getattr(mesh, 'x_coords', None) is not None
                    else np.asarray(np.linspace(mesh.X.min(), mesh.X.max(), mesh.X.shape[1]), dtype=float)
                )
                y_coords = (
                    np.asarray(mesh.y_coords, dtype=float)
                    if getattr(mesh, 'y_coords', None) is not None
                    else np.asarray(np.linspace(mesh.Y.min(), mesh.Y.max(), mesh.Y.shape[0]), dtype=float)
                )
                z_coords = (
                    np.asarray(mesh.z_coords, dtype=float)
                    if getattr(mesh, 'z_coords', None) is not None
                    else np.asarray(np.linspace(mesh.Z.min(), mesh.Z.max(), mesh.Z.shape[2]), dtype=float)
                )

                def _sample_trilinear(volume: np.ndarray, pts_ijk: np.ndarray) -> np.ndarray:
                    # marching_cubes vertex coordinates are in (y_idx, x_idx, z_idx).
                    ny, nx, nz = volume.shape
                    pts = np.asarray(pts_ijk, dtype=np.float32)
                    py = np.clip(pts[:, 0], 0.0, max(0.0, ny - 1.0))
                    px = np.clip(pts[:, 1], 0.0, max(0.0, nx - 1.0))
                    pz = np.clip(pts[:, 2], 0.0, max(0.0, nz - 1.0))

                    y0 = np.floor(py).astype(np.int32)
                    x0 = np.floor(px).astype(np.int32)
                    z0 = np.floor(pz).astype(np.int32)
                    y1 = np.minimum(y0 + 1, ny - 1)
                    x1 = np.minimum(x0 + 1, nx - 1)
                    z1 = np.minimum(z0 + 1, nz - 1)

                    wy = py - y0
                    wx = px - x0
                    wz = pz - z0

                    c000 = volume[y0, x0, z0]
                    c100 = volume[y1, x0, z0]
                    c010 = volume[y0, x1, z0]
                    c110 = volume[y1, x1, z0]
                    c001 = volume[y0, x0, z1]
                    c101 = volume[y1, x0, z1]
                    c011 = volume[y0, x1, z1]
                    c111 = volume[y1, x1, z1]

                    c00 = c000 * (1.0 - wy) + c100 * wy
                    c01 = c001 * (1.0 - wy) + c101 * wy
                    c10 = c010 * (1.0 - wy) + c110 * wy
                    c11 = c011 * (1.0 - wy) + c111 * wy
                    c0 = c00 * (1.0 - wx) + c10 * wx
                    c1 = c01 * (1.0 - wx) + c11 * wx
                    return c0 * (1.0 - wz) + c1 * wz

                def _build_knot_payload(
                    knot_field: np.ndarray,
                    *,
                    part: str,
                    knot_index: Optional[int] = None,
                    slot_index: Optional[int] = None,
                    z0_mm: Optional[float] = None,
                    color: str = '#222222',
                    dead_field: Optional[np.ndarray] = None,
                ) -> Optional[Tuple[Dict[str, Any], Dict[str, Any]]]:
                    field_local = np.asarray(knot_field, dtype=np.float32)
                    if field_local.ndim != 3:
                        return None
                    finite_mask = np.isfinite(field_local)
                    if not np.any(finite_mask):
                        return None
                    finite_vals = field_local[finite_mask]
                    if finite_vals.size <= 0 or not (np.min(finite_vals) <= 0.0 <= np.max(finite_vals)):
                        return None

                    field_eval = np.where(finite_mask, field_local, np.max(finite_vals) + 1.0)
                    verts, faces, _, _ = marching_cubes(field_eval, 0, mask=finite_mask)
                    if verts.size <= 0 or faces.size <= 0:
                        return None

                    scaled = np.zeros_like(verts)
                    scaled[:, 0] = np.interp(verts[:, 0], np.arange(len(y_coords)), y_coords)
                    scaled[:, 1] = np.interp(verts[:, 1], np.arange(len(x_coords)), x_coords)
                    scaled[:, 2] = np.interp(verts[:, 2], np.arange(len(z_coords)), z_coords)

                    # Reorder to (X, Y, Z) then swap for Three.js
                    world_verts = np.column_stack([
                        scaled[:, 1],  # X
                        scaled[:, 2],  # Z(Length) -> Y(up)
                        scaled[:, 0],  # Y(Thickness) -> Z
                    ])
                    matlab_verts = np.column_stack([
                        scaled[:, 1],  # X (width)
                        scaled[:, 0],  # Y (thickness)
                        scaled[:, 2],  # Z (length)
                    ])

                    knot_payload: Dict[str, Any] = {
                        'vertices': world_verts.tolist(),
                        'faces': faces.tolist(),
                        'part': part,
                        'color': color,
                    }
                    knot_payload_mat: Dict[str, Any] = {
                        'vertices': matlab_verts.astype(np.float32, copy=False),
                        'faces': faces[:, :3].astype(np.int32, copy=False),
                        'part': part,
                        'color': color,
                    }
                    if knot_index is not None:
                        knot_payload['knot_index'] = int(knot_index)
                        knot_payload_mat['knot_index'] = int(knot_index)
                    if slot_index is not None:
                        knot_payload['slot_index'] = int(slot_index)
                        knot_payload_mat['slot_index'] = int(slot_index)
                    if z0_mm is not None and math.isfinite(float(z0_mm)):
                        knot_payload['z0_mm'] = float(z0_mm)
                        knot_payload_mat['z0_mm'] = float(z0_mm)

                    if bool(getattr(config, 'dead_knots', False)) and dead_field is not None:
                        dead_mask = np.isfinite(np.asarray(dead_field, dtype=float)).astype(np.float32, copy=False)
                        dead_w = np.clip(_sample_trilinear(dead_mask, verts), 0.0, 1.0)
                        live_rgb = (np.array([166.0, 120.0, 67.0], dtype=np.float32) / 255.0)
                        dead_rgb = (np.array([122.0, 31.0, 43.0], dtype=np.float32) / 255.0)
                        vertex_colors = (
                            (1.0 - dead_w)[:, None] * live_rgb[None, :]
                            + dead_w[:, None] * dead_rgb[None, :]
                        )
                        knot_payload['vertex_colors'] = vertex_colors.astype(
                            np.float32, copy=False
                        ).tolist()
                        knot_payload_mat['vertex_colors'] = vertex_colors.astype(
                            np.float32, copy=False
                        )
                        knot_payload_mat['dead_weight'] = dead_w.astype(np.float32, copy=False)

                    return knot_payload, knot_payload_mat

                field_raw = layers_data.get('ttt')
                if field_raw is not None:
                    field = np.asarray(to_numpy(field_raw), dtype=np.float32)
                    if field.ndim == 3:
                        info = layers_data.get('knot_influence_info') or {}
                        per_knot_raw = info.get('K')
                        dead_per_knot_raw = info.get('K_dead')
                        dead_field_raw = layers_data.get('ttt_dead')
                        z0_vals = np.asarray(to_numpy(getattr(k, 'z0', np.zeros((0,), dtype=np.float32))), dtype=np.float32).reshape(-1)
                        dz_mm = float((getattr(k, 'knot_sequence_info', {}) or {}).get('dz_mm') or 0.0)
                        z_min_mm = float((getattr(k, 'knot_sequence_info', {}) or {}).get('z_min_mm') or 0.0)
                        slot_count = int((getattr(k, 'knot_sequence_info', {}) or {}).get('slot_count') or 0)

                        def _matching_per_knot_field(field_raw_candidate) -> Tuple[bool, int]:
                            if field_raw_candidate is None or getattr(field_raw_candidate, "ndim", 0) != 4:
                                return False, 0
                            raw_shape = tuple(int(v) for v in field_raw_candidate.shape)
                            if raw_shape[:3] != tuple(int(v) for v in field.shape):
                                return False, 0
                            return True, int(raw_shape[3])

                        per_knot_count = 0
                        per_knot_available, per_knot_count = _matching_per_knot_field(per_knot_raw)
                        dead_per_knot_available, dead_per_knot_count = _matching_per_knot_field(dead_per_knot_raw)
                        
                        if per_knot_available:
                            for knot_idx in range(int(per_knot_count)):
                                slot_index = None
                                z0_mm = None
                                if knot_idx < z0_vals.size and math.isfinite(float(z0_vals[knot_idx])):
                                    z0_mm = float(z0_vals[knot_idx])
                                    if dz_mm > 0.0:
                                        slot_est = int(np.rint((z0_mm - z_min_mm) / dz_mm - 1.0))
                                        if 0 <= slot_est < max(slot_count, slot_est + 1):
                                            slot_index = slot_est

                                knot_field_single = np.asarray(
                                    to_numpy(per_knot_raw[:, :, :, knot_idx]),
                                    dtype=np.float32,
                                )
                                dead_field_single = None
                                if dead_per_knot_available and knot_idx < int(dead_per_knot_count):
                                    dead_field_single = np.asarray(
                                        to_numpy(dead_per_knot_raw[:, :, :, knot_idx]),
                                        dtype=np.float32,
                                    )
                                knot_payloads = _build_knot_payload(
                                    knot_field_single,
                                    part='single',
                                    knot_index=knot_idx,
                                    slot_index=slot_index,
                                    z0_mm=z0_mm,
                                    dead_field=dead_field_single,
                                )
                                if knot_payloads is not None:
                                    knot_payload, knot_payload_mat = knot_payloads
                                    knot_data.append(knot_payload)
                                    knot_data_mat.append(knot_payload_mat)

                        if not knot_data:
                            dead_field = None
                            if dead_field_raw is not None:
                                dead_field = np.asarray(to_numpy(dead_field_raw), dtype=np.float32)
                            knot_payloads = _build_knot_payload(
                                field,
                                part='combined',
                                dead_field=dead_field,
                            )
                            if knot_payloads is not None:
                                knot_payload, knot_payload_mat = knot_payloads
                                knot_data.append(knot_payload)
                                knot_data_mat.append(knot_payload_mat)

                        knot_data.sort(
                            key=lambda item: (
                                0 if isinstance(item.get('slot_index'), int) else 1,
                                int(item.get('slot_index') or 0),
                                int(item.get('knot_index') or 0),
                            )
                        )
                        knot_data_mat.sort(
                            key=lambda item: (
                                0 if isinstance(item.get('slot_index'), int) else 1,
                                int(item.get('slot_index') or 0),
                                int(item.get('knot_index') or 0),
                            )
                        )
            except Exception as e:
                print(f"Knot isosurface error: {e}")

        # 7. Fiber plot data (all quiver display modes)
        fiber_data = None
        fiber_out_of_plane_overlays = None
        fiber_components_mat = {
            "txx": np.empty((0,), dtype=np.float32),
            "tyy": np.empty((0,), dtype=np.float32),
            "tzz": np.empty((0,), dtype=np.float32),
        }
        if config.calc_fibers:
            try:
                txx, tyy, tzz = FiberSolver.solve(
                    config,
                    mesh,
                    k,
                    mesh_accum,
                    precomputed_info=layers_data.get('knot_influence_info')
                )

                if config.board_or_log != 0:
                    log_surface_field = layers_data.get('last_g')
                    if log_surface_field is not None:
                        try:
                            log_g = np.asarray(to_numpy(log_surface_field), dtype=np.float32)
                            if log_g.shape == np.asarray(txx).shape:
                                inside_log = np.isfinite(log_g) & (log_g <= 0.0)
                                txx = np.where(inside_log, txx, np.nan)
                                tyy = np.where(inside_log, tyy, np.nan)
                                tzz = np.where(inside_log, tzz, np.nan)
                        except Exception as e:
                            print(f"Log fiber mask error: {e}")

                plot_data = FiberSolver.build_plot_data_all(config, mesh, txx, tyy, tzz)
                fiber_components_mat = {
                    "txx": np.asarray(txx, dtype=np.float32),
                    "tyy": np.asarray(tyy, dtype=np.float32),
                    "tzz": np.asarray(tzz, dtype=np.float32),
                }

                if config.board_or_log == 0:
                    fiber_data = {
                        'surface_quiver3d': swap_segments(plot_data.get('surface_quiver3d', [])),
                        'volume_quiver3d': swap_segments(plot_data.get('volume_quiver3d', [])),
                        'quiver2d': swap_segments(plot_data.get('quiver2d', [])),
                        'quiver2d_clean': swap_segments(plot_data.get('quiver2d_clean', [])),
                        'quiver2d_rand': swap_segments(plot_data.get('quiver2d_rand', [])),
                    }

                    try:
                        oop_pngs = _build_fiber_out_of_plane_surface_pngs(
                            np.asarray(txx, dtype=np.float32),
                            np.asarray(tyy, dtype=np.float32),
                            size=512,
                        )
                        fiber_out_of_plane_overlays = {}
                        for face_key in ["x_min", "x_max", "z_min", "z_max"]:
                            png_bytes = oop_pngs.get(face_key)
                            if not png_bytes:
                                continue
                            fiber_out_of_plane_overlays[face_key] = {
                                "filename": f"fiber_oop_{face_key}_512.png",
                                "src": f"data:image/png;base64,{base64.b64encode(png_bytes).decode('ascii')}",
                            }
                    except Exception as e:
                        print(f"Fiber out-of-plane overlay render error: {e}")
                        fiber_out_of_plane_overlays = None
            except Exception as e:
                print(f"Fiber plot error: {e}")

        growth_layer_fields_mat, growth_layer_indices_mat = _growth_fields_to_float32_stack(
            layers_data.get("growth_layer_fields") or [],
            layers_data.get("growth_layer_indices") or [],
        )
        mesh_grids_mat = {
            "x": _to_float32_3d(getattr(mesh, "X", None)),
            "y": _to_float32_3d(getattr(mesh, "Y", None)),
            "z": _to_float32_3d(getattr(mesh, "Z", None)),
        }
        scalar_fields_mat = {
            "knot_field": _to_float32_3d(layers_data.get("ttt")),
            "growth_layer_fields": growth_layer_fields_mat,
            "growth_layer_indices": growth_layer_indices_mat,
        }

        # The full knot influence payload can be very large; after building
        # knot meshes and any fiber outputs, it should not survive into cache.
        layers_data.pop("knot_influence_info", None)

        sim_id = _cache_simulation({
            "export_mode": "log" if int(config.board_or_log) != 0 else "board",
            "fiber_domain": "log" if int(config.board_or_log) != 0 else "board",
            "fibers": fiber_components_mat,
            "normals": {
                "nx": nx_mat,
                "ny": ny_mat,
                "nz": nz_mat,
            },
            "contours": contour_data_mat,
            "contours_masked": contour_data_masked_mat,
            "contours_masked_live": contour_data_masked_live_mat,
            "contours_unmasked": contour_data_unmasked_mat,
            "contours_mid_masked": contour_data_mid_masked_mat,
            "contours_mid_masked_live": contour_data_mid_masked_live_mat,
            "contours_mid_unmasked": contour_data_mid_unmasked_mat,
            "board_dimensions": board_dimensions,
            "mesh_axes": mesh_axes_mat,
            "mesh_grids": mesh_grids_mat,
            "scalar_fields": scalar_fields_mat,
            "growth_layers": growth_layers_mat,
            "pith_surface": pith_surface_mat,
            "knots": knot_data_mat,
            "knot_sequence": dict(getattr(k, "knot_sequence_info", {}) or {}),
            "geometry_randomization": dict(getattr(k, "geometry_randomization_info", {}) or {}),
        })

        result = {
            "export_mode": "log" if int(config.board_or_log) != 0 else "board",
            "layers": response_layers,
            "pith_layer": response_pith_layer,
            "contours": contour_data,
            "contours_masked": contour_data_masked,
            "contours_masked_live": contour_data_masked_live,
            "contours_unmasked": contour_data_unmasked,
            "board_outline": board_outline,
            "board_dimensions": board_dimensions,
            "mesh_axes": mesh_axes,
            "knots": knot_data,
            "fibers": fiber_data,
            "normal_vectors": normal_vector_data,
            "normal_overlays": normal_overlays,
            "fiber_out_of_plane_overlays": fiber_out_of_plane_overlays,
            "knot_sequence": dict(getattr(k, "knot_sequence_info", {}) or {}),
            "geometry_randomization": dict(getattr(k, "geometry_randomization_info", {}) or {}),
            "warnings": warnings,
            "simulation_id": sim_id,
            "gpu_active": bool(getattr(k, 'gpu_enabled', False)),
            "gpu_requested": bool(config.use_gpu),
        }
        
        # Single-pass JSON serialization (avoids decode/re-encode overhead).
        return Response(
            content=NanSafeEncoder().encode(result),
            media_type="application/json"
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8100)
