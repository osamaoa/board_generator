from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Response
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any, Tuple
import numpy as np
import json
import math
import os
import hashlib
import base64
import gc
from pathlib import Path
from io import BytesIO
import zipfile
import uuid
from PIL import Image, ImageDraw, ImageFilter, ImageOps
import scipy.io
from scipy.interpolate import griddata
from scipy.ndimage import map_coordinates

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
_VENEER_INTERNAL_MESH_SIZE_MM = 20.0
_DEFAULT_FIBER_IRREGULARITY_STRENGTH = 0.35
_DEFAULT_RING_IRREGULARITY_STRENGTH = 0.40
_DEMO_MODE = str(os.environ.get("BOARD_GENERATOR_DEMO", "")).strip().lower() in {"1", "true", "yes", "on"}
_PHOTOREALISTIC_DISABLED_REASON = "Photorealistic generation is disabled for this CPU demo."
_DEFAULT_RING_COLOR_STOPS = [
    (-0.60, "#d8c8ae"),
    (-0.08, "#d3bea0"),
    (0.00, "#cbae8c"),
    (0.08, "#d3bea0"),
    (0.60, "#d8c8ae"),
]
_DEFAULT_RING_COLOR_STOPS_TEXT = "-0.6:#d8c8ae;-0.08:#d3bea0;0:#cbae8c;0.08:#d3bea0;0.6:#d8c8ae"
_DEFAULT_KNOT_STAIN_DARKNESS = 0.30
_DEFAULT_KNOT_STAIN_SPREAD_MM = 40.0
_DEFAULT_KNOT_STAIN_COLOR = "#8f705b"
_DEFAULT_KNOT_STAIN_OPACITY = 1.0
_DEFAULT_KNOT_CORE_STRENGTH = 0.56
_DEFAULT_KNOT_RING_STRENGTH = 2.0
_DEFAULT_KNOT_REACTION_STRENGTH = 2.0
_DEFAULT_VENEER_FIBER_TEXTURE_STRENGTH = 0.65
_DEFAULT_VENEER_FIBER_TEXTURE_SCALE_MM = 0.70
_DEFAULT_VENEER_FIBER_TEXTURE_LENGTH_MM = 80.0


def _frontend_dist_dir() -> Path:
    configured = str(os.environ.get("BOARD_GENERATOR_FRONTEND_DIST") or "").strip()
    if configured:
        return Path(configured).expanduser()
    # backend/app/main.py -> backend/app -> backend -> repository root
    return Path(__file__).resolve().parents[2] / "frontend_dist"


def _frontend_index_path() -> Path:
    return _frontend_dist_dir() / "index.html"


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


class RenderRingColorOverlaysRequest(BaseModel):
    simulation_id: str
    ring_color_stops: Optional[Any] = None
    ring_color_clip: Optional[float] = None
    ring_color_knot_darkness: Optional[float] = None
    ring_color_knot_spread_mm: Optional[float] = None
    ring_color_knot_stain_color: Optional[str] = None
    ring_color_knot_opacity: Optional[float] = None
    show_rings_inside_knots: Optional[bool] = None
    knot_inside_limit: Optional[float] = None
    size: Optional[int] = None


def _clamp01(value):
    return np.clip(value, 0.0, 1.0)


def _pil_bilinear() -> int:
    try:
        return int(Image.Resampling.BILINEAR)
    except AttributeError:
        return int(Image.BILINEAR)


def _pil_lanczos() -> int:
    try:
        return int(Image.Resampling.LANCZOS)
    except AttributeError:
        return int(Image.LANCZOS)


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


def _hex_to_rgb_float(value: Any) -> np.ndarray:
    text = str(value or "").strip()
    if text.startswith("#"):
        text = text[1:]
    if len(text) == 3:
        text = "".join(ch * 2 for ch in text)
    if len(text) != 6:
        raise ValueError(f"Invalid hex color: {value!r}")
    try:
        rgb = [int(text[idx : idx + 2], 16) for idx in (0, 2, 4)]
    except ValueError as exc:
        raise ValueError(f"Invalid hex color: {value!r}") from exc
    return np.asarray(rgb, dtype=np.float32) / 255.0


def _parse_ring_color_stops(raw: Any = None) -> List[Tuple[float, np.ndarray]]:
    if raw is None or raw == "":
        items = list(_DEFAULT_RING_COLOR_STOPS)
    elif isinstance(raw, str):
        items = []
        for piece in raw.split(";"):
            token = piece.strip()
            if not token:
                continue
            if ":" not in token:
                raise ValueError(
                    "ring_color_stops string entries must be level:#rrggbb separated by semicolons."
                )
            level_text, color_text = token.split(":", 1)
            items.append((float(level_text.strip()), color_text.strip()))
    elif isinstance(raw, (list, tuple)):
        items = []
        for idx, item in enumerate(raw):
            if isinstance(item, dict):
                level = item.get("level", item.get("value"))
                color = item.get("color")
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                level, color = item[0], item[1]
            else:
                raise ValueError(f"Invalid ring color stop at index {idx}: {item!r}")
            items.append((float(level), color))
    else:
        raise ValueError("ring_color_stops must be a string or a list.")

    parsed = [(float(level), _hex_to_rgb_float(color)) for level, color in items]
    parsed = sorted(
        [(level, color) for level, color in parsed if np.isfinite(level)],
        key=lambda item: item[0],
    )
    if len(parsed) < 2:
        raise ValueError("At least two ring color stops are required.")
    unique: List[Tuple[float, np.ndarray]] = []
    for level, color in parsed:
        if unique and level <= unique[-1][0]:
            level = unique[-1][0] + 1e-6
        unique.append((level, color))
    return unique


def _nearest_ring_normalized_field(
    growth_layer_fields: Any,
    *,
    clip: float = 1.0,
) -> np.ndarray:
    fields: List[np.ndarray] = []
    shape: Optional[Tuple[int, int, int]] = None

    if isinstance(growth_layer_fields, list):
        raw_fields = growth_layer_fields
    else:
        try:
            stacked = np.asarray(to_numpy(growth_layer_fields), dtype=np.float32)
        except Exception:
            stacked = np.empty((0,), dtype=np.float32)
        if stacked.ndim == 4:
            raw_fields = [stacked[..., idx] for idx in range(stacked.shape[-1])]
        elif stacked.ndim == 3:
            raw_fields = [stacked]
        else:
            raw_fields = []

    if not raw_fields:
        return np.empty((0, 0, 0), dtype=np.float32)

    for field in raw_fields:
        arr = np.asarray(to_numpy(field), dtype=np.float32)
        if arr.ndim != 3:
            continue
        arr_shape = tuple(int(v) for v in arr.shape)
        if shape is None:
            shape = arr_shape
        if arr_shape == shape:
            fields.append(arr)
    if not fields:
        return np.empty((0, 0, 0), dtype=np.float32)

    return _nearest_ring_normalized_arrays(fields, clip=clip)


def _nearest_ring_normalized_arrays(
    fields: List[np.ndarray],
    *,
    clip: float = 1.0,
) -> np.ndarray:
    if not fields:
        return np.empty((0, 0), dtype=np.float32)
    shape = tuple(int(v) for v in np.asarray(fields[0]).shape)
    usable = []
    for field in fields:
        arr = np.asarray(field, dtype=np.float32)
        if arr.shape == shape:
            usable.append(arr)
    if not usable:
        return np.empty((0, 0), dtype=np.float32)

    stack = np.stack(usable, axis=-1).astype(np.float32, copy=False)
    finite = np.isfinite(stack)
    abs_stack = np.where(finite, np.abs(stack), np.inf)
    nearest_idx = np.argmin(abs_stack, axis=-1)
    has_value = np.isfinite(np.min(abs_stack, axis=-1))
    nearest = np.take_along_axis(stack, nearest_idx[..., None], axis=-1)[..., 0]

    n_layers = int(stack.shape[-1])
    if n_layers >= 2:
        prev_idx = np.maximum(nearest_idx - 1, 0)
        next_idx = np.minimum(nearest_idx + 1, n_layers - 1)
        prev_val = np.take_along_axis(stack, prev_idx[..., None], axis=-1)[..., 0]
        next_val = np.take_along_axis(stack, next_idx[..., None], axis=-1)[..., 0]
        prev_spacing = np.where(nearest_idx > 0, np.abs(nearest - prev_val), np.nan)
        next_spacing = np.where(nearest_idx < (n_layers - 1), np.abs(next_val - nearest), np.nan)
        spacing = np.nanmin(np.stack([prev_spacing, next_spacing], axis=0), axis=0)
        finite_spacing = spacing[np.isfinite(spacing) & (spacing > 1e-9)]
        fallback = float(np.median(finite_spacing)) if finite_spacing.size else 1.0
        spacing = np.where(np.isfinite(spacing) & (spacing > 1e-9), spacing, fallback)
    else:
        finite_abs = np.abs(nearest[np.isfinite(nearest)])
        spacing = float(np.percentile(finite_abs, 95)) if finite_abs.size else 1.0
        if not np.isfinite(spacing) or spacing <= 1e-9:
            spacing = 1.0

    normalized = nearest / np.maximum(spacing, 1e-9)
    clip_value = max(1e-6, float(clip))
    normalized = np.clip(normalized, -clip_value, clip_value)
    normalized = np.where(has_value & np.isfinite(normalized), normalized, np.nan)
    return normalized.astype(np.float32, copy=False)


def _ring_interval_phase_arrays(
    fields: List[np.ndarray],
    *,
    clip: float = 1.0,
) -> np.ndarray:
    if not fields:
        return np.empty((0, 0), dtype=np.float32)
    shape = tuple(int(v) for v in np.asarray(fields[0]).shape)
    usable = []
    for field in fields:
        arr = np.asarray(field, dtype=np.float32)
        if arr.shape == shape:
            usable.append(arr)
    if len(usable) < 2:
        return _nearest_ring_normalized_arrays(usable, clip=clip)

    stack = np.stack(usable, axis=-1).astype(np.float32, copy=False)
    inner = stack[..., :-1]
    outer = stack[..., 1:]
    finite_pair = np.isfinite(inner) & np.isfinite(outer)
    bracket = finite_pair & (inner >= 0.0) & (outer <= 0.0)
    score = np.where(bracket, np.abs(inner) + np.abs(outer), np.inf)
    pair_idx = np.argmin(score, axis=-1)
    has_pair = np.isfinite(np.min(score, axis=-1))

    inner_sel = np.take_along_axis(inner, pair_idx[..., None], axis=-1)[..., 0]
    outer_sel = np.take_along_axis(outer, pair_idx[..., None], axis=-1)[..., 0]
    denom = inner_sel - outer_sel
    with np.errstate(invalid="ignore", divide="ignore"):
        phase = inner_sel / denom
    phase = np.clip(phase, 0.0, 1.0) - 0.5

    fallback = _nearest_ring_normalized_arrays(usable, clip=clip)
    values = np.where(has_pair & np.isfinite(phase), phase, fallback)
    clip_value = max(1e-6, float(clip))
    values = np.clip(values, -clip_value, clip_value)
    return values.astype(np.float32, copy=False)


def _colorize_normalized_ring_values(values: np.ndarray, stops: List[Tuple[float, np.ndarray]]) -> np.ndarray:
    vals = np.asarray(values, dtype=np.float32)
    levels = np.asarray([level for level, _ in stops], dtype=np.float32)
    colors = np.stack([color for _, color in stops], axis=0).astype(np.float32)
    flat = vals.reshape(-1)
    out = np.ones((flat.size, 3), dtype=np.float32)
    finite = np.isfinite(flat)
    if np.any(finite):
        clipped = np.clip(flat[finite], float(levels[0]), float(levels[-1]))
        for channel in range(3):
            out[finite, channel] = np.interp(clipped, levels, colors[:, channel])
    return np.rint(np.clip(out.reshape(vals.shape + (3,)), 0.0, 1.0) * 255.0).astype(np.uint8)


def _apply_knot_stain_to_rgb(
    rgb: np.ndarray,
    knot_field_image: Optional[np.ndarray],
    *,
    strength: float = 0.0,
    spread_mm: float = _DEFAULT_KNOT_STAIN_SPREAD_MM,
    stain_color: Any = _DEFAULT_KNOT_STAIN_COLOR,
    opacity: float = _DEFAULT_KNOT_STAIN_OPACITY,
    knot_inside_limit: float = -20.0,
) -> np.ndarray:
    s = float(strength)
    spread = float(spread_mm)
    alpha_scale = float(opacity)
    if (
        not np.isfinite(s)
        or not np.isfinite(spread)
        or not np.isfinite(alpha_scale)
        or s <= 0.0
        or spread <= 1e-9
        or alpha_scale <= 0.0
    ):
        return rgb
    knot_field = np.asarray(knot_field_image, dtype=np.float32)
    if knot_field.shape != rgb.shape[:2]:
        return rgb
    finite = np.isfinite(knot_field)
    if not np.any(finite):
        return rgb

    strength_safe = float(np.clip(s, 0.0, 1.0))
    opacity_safe = float(np.clip(alpha_scale, 0.0, 1.0))
    spread_safe = max(1e-6, spread)
    limit = float(knot_inside_limit)
    if not np.isfinite(limit):
        limit = -20.0

    inside = finite & (knot_field <= limit)
    outside = finite & ~inside
    distance = np.zeros_like(knot_field, dtype=np.float32)
    distance[outside] = np.sqrt(np.maximum(knot_field[outside] - limit, 0.0))
    weight = np.zeros_like(knot_field, dtype=np.float32)
    weight[inside] = 1.0
    weight[outside] = np.exp(-0.5 * (distance[outside] / spread_safe) ** 2)

    base = np.asarray(rgb, dtype=np.float32) / 255.0
    stain_rgb = _hex_to_rgb_float(stain_color)
    # Strength controls darkness while the color picker controls hue.
    target = np.clip(stain_rgb * (1.25 - 0.75 * strength_safe), 0.0, 1.0).reshape(1, 1, 3)
    alpha = np.clip(weight * opacity_safe, 0.0, 1.0)
    out = base.copy()
    out[..., :3] = ((1.0 - alpha[..., None]) * base[..., :3]) + (alpha[..., None] * target)
    return np.rint(np.clip(out, 0.0, 1.0) * 255.0).astype(np.uint8)


def _positive_robust_scale(values: np.ndarray, percentile: float = 96.0) -> float:
    arr = np.asarray(values, dtype=np.float32)
    finite = np.isfinite(arr) & (arr > 1e-8)
    if not np.any(finite):
        return 0.0
    positive = arr[finite]
    scale = float(np.percentile(positive, float(percentile)))
    if not np.isfinite(scale) or scale <= 1e-8:
        scale = float(np.max(positive))
    return scale if np.isfinite(scale) and scale > 1e-8 else 0.0


def _apply_veneer_knot_color_layers_to_rgb(
    rgb: np.ndarray,
    ring_phase_image: np.ndarray,
    knot_field_image: Optional[np.ndarray],
    deviation_image: Optional[np.ndarray],
    reaction_lobe_image: Optional[np.ndarray],
    *,
    strength: float = 0.0,
    stain_color: Any = _DEFAULT_KNOT_STAIN_COLOR,
    opacity: float = _DEFAULT_KNOT_STAIN_OPACITY,
    core_strength: float = _DEFAULT_KNOT_CORE_STRENGTH,
    ring_strength: float = _DEFAULT_KNOT_RING_STRENGTH,
    reaction_strength: float = _DEFAULT_KNOT_REACTION_STRENGTH,
) -> np.ndarray:
    s = float(strength)
    alpha_scale = float(opacity)
    if (
        not np.isfinite(s)
        or not np.isfinite(alpha_scale)
        or s <= 0.0
        or alpha_scale <= 0.0
    ):
        return rgb

    base_uint8 = np.asarray(rgb, dtype=np.uint8)
    phase = np.asarray(ring_phase_image, dtype=np.float32)
    if base_uint8.ndim != 3 or base_uint8.shape[-1] != 3 or phase.shape != base_uint8.shape[:2]:
        return rgb

    finite_phase = np.isfinite(phase)
    if not np.any(finite_phase):
        return rgb

    core = np.zeros_like(phase, dtype=np.float32)
    if knot_field_image is not None:
        knot_field = np.asarray(knot_field_image, dtype=np.float32)
        if knot_field.shape == phase.shape:
            finite_knot = np.isfinite(knot_field)
            penetration = np.zeros_like(knot_field, dtype=np.float32)
            penetration[finite_knot] = np.sqrt(np.maximum(-knot_field[finite_knot], 0.0))
            scale = _positive_robust_scale(penetration, percentile=88.0)
            if scale > 0.0:
                core = np.clip(penetration / scale, 0.0, 1.0) ** 0.55
                core = np.where(finite_knot & (knot_field < 0.0), core, 0.0).astype(np.float32, copy=False)
                if np.any(core > 1e-4):
                    core = _pil_blur_float01(core, radius=0.75)

    deviation = np.zeros_like(phase, dtype=np.float32)
    if deviation_image is not None:
        dev_arr = np.asarray(deviation_image, dtype=np.float32)
        if dev_arr.shape == phase.shape:
            finite_dev = np.isfinite(dev_arr) & (dev_arr > 1e-6)
            scale = _positive_robust_scale(dev_arr, percentile=96.0)
            if scale > 0.0:
                normalized = np.zeros_like(dev_arr, dtype=np.float32)
                normalized[finite_dev] = np.clip(dev_arr[finite_dev] / scale, 0.0, 1.65)
                deviation[finite_dev] = 1.0 - np.exp(-1.75 * np.sqrt(normalized[finite_dev]))
                deviation = np.clip(deviation, 0.0, 1.0)

    phase_abs = np.abs(np.nan_to_num(phase, nan=1.0, posinf=1.0, neginf=1.0))
    latewood = np.exp(-0.5 * (phase_abs / 0.075) ** 2).astype(np.float32, copy=False)
    disturbed_latewood = np.clip(deviation * latewood * (1.0 - (0.82 * core)), 0.0, 1.0)
    if np.any(disturbed_latewood > 1e-4):
        disturbed_latewood = _pil_blur_float01(disturbed_latewood, radius=0.35)

    reaction = np.zeros_like(phase, dtype=np.float32)
    if reaction_lobe_image is not None:
        reaction_arr = np.asarray(reaction_lobe_image, dtype=np.float32)
        if reaction_arr.shape == phase.shape:
            reaction = np.clip(np.nan_to_num(reaction_arr, nan=0.0, posinf=0.0, neginf=0.0), 0.0, 1.0)
            reaction = reaction * (0.35 + (0.65 * deviation)) * (1.0 - (0.58 * core))
            reaction = np.clip(reaction, 0.0, 1.0)
            if np.any(reaction > 1e-4):
                reaction = _pil_blur_float01(reaction, radius=1.0)

    finite_mask = finite_phase.astype(np.float32)
    core = np.clip(core * finite_mask, 0.0, 1.0)
    disturbed_latewood = np.clip(disturbed_latewood * finite_mask, 0.0, 1.0)
    reaction = np.clip(reaction * finite_mask, 0.0, 1.0)
    if not (np.any(core > 1e-4) or np.any(disturbed_latewood > 1e-4) or np.any(reaction > 1e-4)):
        return rgb

    strength_safe = float(np.clip(s, 0.0, 1.0))
    opacity_safe = float(np.clip(alpha_scale, 0.0, 1.0))
    core_strength_safe = float(np.clip(float(core_strength), 0.0, 2.0))
    ring_strength_safe = float(np.clip(float(ring_strength), 0.0, 2.0))
    reaction_strength_safe = float(np.clip(float(reaction_strength), 0.0, 2.0))
    base = base_uint8.astype(np.float32) / 255.0
    stain_rgb = _hex_to_rgb_float(stain_color).reshape(1, 1, 3)
    out = base.copy()

    ring_alpha = np.clip(
        opacity_safe * ring_strength_safe * (0.16 + (0.28 * strength_safe)) * disturbed_latewood,
        0.0,
        0.80,
    )
    ring_target = np.clip((out[..., :3] * (0.93 - (0.12 * strength_safe))) + (stain_rgb * 0.055), 0.0, 1.0)
    out[..., :3] = ((1.0 - ring_alpha[..., None]) * out[..., :3]) + (ring_alpha[..., None] * ring_target)

    reaction_alpha = np.clip(
        opacity_safe * reaction_strength_safe * (0.10 + (0.20 * strength_safe)) * reaction,
        0.0,
        0.65,
    )
    reaction_target = np.clip((out[..., :3] * (0.97 - (0.08 * strength_safe))) + (stain_rgb * 0.055), 0.0, 1.0)
    out[..., :3] = ((1.0 - reaction_alpha[..., None]) * out[..., :3]) + (reaction_alpha[..., None] * reaction_target)

    core_alpha = np.clip(opacity_safe * core_strength_safe * (0.24 + (0.56 * strength_safe)) * core, 0.0, 0.95)
    core_target = np.clip(stain_rgb * (1.06 - (0.30 * strength_safe)), 0.0, 1.0)
    core_target = np.minimum(out[..., :3] * (0.98 - (0.10 * strength_safe)), core_target)
    out[..., :3] = ((1.0 - core_alpha[..., None]) * out[..., :3]) + (core_alpha[..., None] * core_target)
    return np.rint(np.clip(out, 0.0, 1.0) * 255.0).astype(np.uint8)


def _render_color_matrix_png(
    matrix_uv: np.ndarray,
    *,
    stops: List[Tuple[float, np.ndarray]],
    size: int,
    flip_x: bool = False,
    transparent_nan: bool = False,
    knot_field_uv: Any = None,
    knot_inside_limit: float = -20.0,
    knot_darkness: float = 0.0,
    knot_darkness_spread_mm: float = _DEFAULT_KNOT_STAIN_SPREAD_MM,
    knot_stain_color: Any = _DEFAULT_KNOT_STAIN_COLOR,
    knot_opacity: float = _DEFAULT_KNOT_STAIN_OPACITY,
) -> bytes:
    matrix = np.asarray(matrix_uv, dtype=np.float32)
    if matrix.ndim != 2 or matrix.size == 0:
        image = Image.new("RGBA" if transparent_nan else "RGB", (int(size), int(size)), color=(255, 255, 255, 0) if transparent_nan else (255, 255, 255))
    else:
        # matrix axes are (u, v). Image columns are u, image rows are reversed v.
        finite = np.isfinite(matrix[:, ::-1].T)
        stain_field_image = None
        if knot_field_uv is not None:
            stain_matrix = np.asarray(knot_field_uv, dtype=np.float32)
            if stain_matrix.shape == matrix.shape:
                stain_field_image = stain_matrix[:, ::-1].T
        image_arr = _colorize_normalized_ring_values(matrix[:, ::-1].T, stops)
        image_arr = _apply_knot_stain_to_rgb(
            image_arr,
            stain_field_image,
            strength=knot_darkness,
            spread_mm=knot_darkness_spread_mm,
            stain_color=knot_stain_color,
            opacity=knot_opacity,
            knot_inside_limit=knot_inside_limit,
        )
        if bool(transparent_nan):
            alpha = np.where(finite, 255, 0).astype(np.uint8)
            image_arr = np.dstack([image_arr, alpha])
        if bool(flip_x):
            image_arr = image_arr[:, ::-1, :]
        image = Image.fromarray(image_arr, mode="RGBA" if transparent_nan else "RGB")
        if image.size != (int(size), int(size)):
            image = image.resize((int(size), int(size)), resample=_pil_bilinear())
    png_buffer = BytesIO()
    image.save(png_buffer, format="PNG", optimize=False)
    return png_buffer.getvalue()


def _build_growth_color_surface_pngs(
    growth_layer_fields: Any,
    *,
    size: int = 512,
    color_stops: Any = None,
    clip: float = 1.0,
    knot_mask: Any = None,
    knot_field: Any = None,
    knot_inside_limit: float = -20.0,
    knot_darkness: float = 0.0,
    knot_darkness_spread_mm: float = _DEFAULT_KNOT_STAIN_SPREAD_MM,
    knot_stain_color: Any = _DEFAULT_KNOT_STAIN_COLOR,
    knot_opacity: float = _DEFAULT_KNOT_STAIN_OPACITY,
) -> Dict[str, bytes]:
    normalized = _nearest_ring_normalized_field(growth_layer_fields, clip=float(clip))
    if normalized.ndim != 3 or normalized.size == 0:
        return {}
    stain_field = None
    if knot_field is not None:
        try:
            knot_arr = np.asarray(to_numpy(knot_field), dtype=np.float32)
            if knot_arr.shape == normalized.shape:
                stain_field = knot_arr
        except Exception:
            stain_field = None
    if knot_mask is not None:
        try:
            mask_arr = np.asarray(to_numpy(knot_mask), dtype=bool)
            if mask_arr.shape == normalized.shape:
                normalized = normalized.copy()
                normalized[mask_arr] = np.nan
                if stain_field is not None:
                    stain_field = stain_field.copy()
                    stain_field[mask_arr] = np.nan
        except Exception:
            pass
    stops = _parse_ring_color_stops(color_stops)
    ny, nx, nz = normalized.shape
    if ny <= 0 or nx <= 0 or nz <= 0:
        return {}

    y_mid_float = 0.5 * float(ny - 1)
    y0 = int(np.floor(y_mid_float))
    y1 = min(y0 + 1, ny - 1)
    alpha = float(y_mid_float - y0)
    y_mid_matrix = (1.0 - alpha) * normalized[y0, :, :] + alpha * normalized[y1, :, :]
    y_mid_stain = None
    if stain_field is not None:
        y_mid_stain = (1.0 - alpha) * stain_field[y0, :, :] + alpha * stain_field[y1, :, :]

    def render(matrix, *, flip_x=False, stain_matrix=None):
        return _render_color_matrix_png(
            matrix,
            stops=stops,
            size=size,
            flip_x=flip_x,
            knot_field_uv=stain_matrix,
            knot_inside_limit=knot_inside_limit,
            knot_darkness=knot_darkness,
            knot_darkness_spread_mm=knot_darkness_spread_mm,
            knot_stain_color=knot_stain_color,
            knot_opacity=knot_opacity,
        )

    return {
        "ring_color_1": render(normalized[-1, :, :], flip_x=True, stain_matrix=None if stain_field is None else stain_field[-1, :, :]),
        "ring_color_2": render(normalized[0, :, :], stain_matrix=None if stain_field is None else stain_field[0, :, :]),
        "ring_color_3": render(normalized[:, -1, :], stain_matrix=None if stain_field is None else stain_field[:, -1, :]),
        "ring_color_4": render(normalized[:, 0, :], flip_x=True, stain_matrix=None if stain_field is None else stain_field[:, 0, :]),
        "ring_color_5": render(y_mid_matrix, stain_matrix=y_mid_stain),
        "ring_color_bottom": render(normalized[:, :, 0].T, stain_matrix=None if stain_field is None else stain_field[:, :, 0].T),
        "ring_color_top": render(normalized[:, :, -1].T, stain_matrix=None if stain_field is None else stain_field[:, :, -1].T),
    }


def _build_growth_color_viewer_overlay_pngs(
    growth_layer_fields: Any,
    *,
    size: int = 512,
    color_stops: Any = None,
    clip: float = 1.0,
    knot_mask: Any = None,
    knot_field: Any = None,
    knot_inside_limit: float = -20.0,
    knot_darkness: float = 0.0,
    knot_darkness_spread_mm: float = _DEFAULT_KNOT_STAIN_SPREAD_MM,
    knot_stain_color: Any = _DEFAULT_KNOT_STAIN_COLOR,
    knot_opacity: float = _DEFAULT_KNOT_STAIN_OPACITY,
) -> Dict[str, bytes]:
    normalized = _nearest_ring_normalized_field(growth_layer_fields, clip=float(clip))
    if normalized.ndim != 3 or normalized.size == 0:
        return {}
    stain_field = None
    if knot_field is not None:
        try:
            knot_arr = np.asarray(to_numpy(knot_field), dtype=np.float32)
            if knot_arr.shape == normalized.shape:
                stain_field = knot_arr
        except Exception:
            stain_field = None
    if knot_mask is not None:
        try:
            mask_arr = np.asarray(to_numpy(knot_mask), dtype=bool)
            if mask_arr.shape == normalized.shape:
                normalized = normalized.copy()
                normalized[mask_arr] = np.nan
                if stain_field is not None:
                    stain_field = stain_field.copy()
                    stain_field[mask_arr] = np.nan
        except Exception:
            pass
    stops = _parse_ring_color_stops(color_stops)
    def render(matrix, *, stain_matrix=None):
        return _render_color_matrix_png(
            matrix,
            stops=stops,
            size=size,
            flip_x=False,
            knot_field_uv=stain_matrix,
            knot_inside_limit=knot_inside_limit,
            knot_darkness=knot_darkness,
            knot_darkness_spread_mm=knot_darkness_spread_mm,
            knot_stain_color=knot_stain_color,
            knot_opacity=knot_opacity,
        )
    return {
        "x_min": render(normalized[:, 0, :], stain_matrix=None if stain_field is None else stain_field[:, 0, :]),
        "x_max": render(normalized[:, -1, :], stain_matrix=None if stain_field is None else stain_field[:, -1, :]),
        "z_min": render(normalized[0, :, :], stain_matrix=None if stain_field is None else stain_field[0, :, :]),
        "z_max": render(normalized[-1, :, :], stain_matrix=None if stain_field is None else stain_field[-1, :, :]),
        # Model Z is board length and viewer Y. These are the two end cross-sections.
        "y_min": render(normalized[:, :, 0].T, stain_matrix=None if stain_field is None else stain_field[:, :, 0].T),
        "y_max": render(normalized[:, :, -1].T, stain_matrix=None if stain_field is None else stain_field[:, :, -1].T),
    }


def _build_growth_color_log_cap_overlay_pngs(
    growth_layer_fields: Any,
    outer_field: Any,
    *,
    size: int = 512,
    color_stops: Any = None,
    clip: float = 1.0,
    knot_field: Any = None,
    knot_inside_limit: float = -20.0,
    knot_darkness: float = 0.0,
    knot_darkness_spread_mm: float = _DEFAULT_KNOT_STAIN_SPREAD_MM,
    knot_stain_color: Any = _DEFAULT_KNOT_STAIN_COLOR,
    knot_opacity: float = _DEFAULT_KNOT_STAIN_OPACITY,
) -> Dict[str, bytes]:
    normalized = _nearest_ring_normalized_field(growth_layer_fields, clip=float(clip))
    if normalized.ndim != 3 or normalized.size == 0:
        return {}
    stain_field = None
    if knot_field is not None:
        try:
            knot_arr = np.asarray(to_numpy(knot_field), dtype=np.float32)
            if knot_arr.shape == normalized.shape:
                stain_field = knot_arr
        except Exception:
            stain_field = None

    try:
        outer = np.asarray(to_numpy(outer_field), dtype=np.float32)
        if outer.shape == normalized.shape:
            normalized = normalized.copy()
            normalized[~np.isfinite(outer) | (outer > 0.0)] = np.nan
            if stain_field is not None:
                stain_field = stain_field.copy()
                stain_field[~np.isfinite(outer) | (outer > 0.0)] = np.nan
    except Exception:
        pass

    stops = _parse_ring_color_stops(color_stops)
    def render(matrix, *, stain_matrix=None):
        return _render_color_matrix_png(
            matrix,
            stops=stops,
            size=size,
            flip_x=False,
            transparent_nan=True,
            knot_field_uv=stain_matrix,
            knot_inside_limit=knot_inside_limit,
            knot_darkness=knot_darkness,
            knot_darkness_spread_mm=knot_darkness_spread_mm,
            knot_stain_color=knot_stain_color,
            knot_opacity=knot_opacity,
        )
    return {
        # Log length is model Z, which maps to viewer Y. These are the two cut caps.
        "y_min": render(
            normalized[:, :, 0].T,
            stain_matrix=None if stain_field is None else stain_field[:, :, 0].T,
        ),
        "y_max": render(
            normalized[:, :, -1].T,
            stain_matrix=None if stain_field is None else stain_field[:, :, -1].T,
        ),
    }


def _encode_ring_color_overlay_pngs(
    color_pngs: Dict[str, bytes],
    face_keys: List[str],
    *,
    size: int = 512,
) -> Optional[Dict[str, Dict[str, str]]]:
    overlays: Dict[str, Dict[str, str]] = {}
    size_safe = max(16, int(size))
    for face_key in face_keys:
        png_bytes = color_pngs.get(face_key)
        if not png_bytes:
            continue
        overlays[face_key] = {
            "filename": f"ring_color_{face_key}_{size_safe}.png",
            "src": f"data:image/png;base64,{base64.b64encode(png_bytes).decode('ascii')}",
        }
    return overlays or None


def _build_board_ring_color_overlay_payload(
    growth_layer_fields: Any,
    *,
    knot_field: Any = None,
    show_rings_inside_knots: bool = True,
    knot_inside_limit: float = -20.0,
    color_stops: Any = None,
    clip: float = 1.0,
    knot_darkness: float = 0.0,
    knot_darkness_spread_mm: float = _DEFAULT_KNOT_STAIN_SPREAD_MM,
    knot_stain_color: Any = _DEFAULT_KNOT_STAIN_COLOR,
    knot_opacity: float = _DEFAULT_KNOT_STAIN_OPACITY,
    size: int = 512,
) -> Optional[Dict[str, Dict[str, str]]]:
    knot_mask = None
    if not bool(show_rings_inside_knots) and knot_field is not None:
        try:
            knot_arr = np.asarray(to_numpy(knot_field), dtype=np.float32)
            knot_mask = knot_arr <= float(knot_inside_limit)
        except Exception:
            knot_mask = None

    size_safe = max(16, int(size))
    color_pngs = _build_growth_color_viewer_overlay_pngs(
        growth_layer_fields,
        size=size_safe,
        color_stops=color_stops,
        clip=float(clip),
        knot_mask=knot_mask,
        knot_field=knot_field,
        knot_inside_limit=float(knot_inside_limit),
        knot_darkness=float(knot_darkness),
        knot_darkness_spread_mm=float(knot_darkness_spread_mm),
        knot_stain_color=knot_stain_color,
        knot_opacity=float(knot_opacity),
    )
    return _encode_ring_color_overlay_pngs(
        color_pngs,
        ["x_min", "x_max", "z_min", "z_max", "y_min", "y_max"],
        size=size_safe,
    )


def _build_log_ring_color_overlay_payload(
    growth_layer_fields: Any,
    outer_field: Any,
    *,
    knot_field: Any = None,
    knot_inside_limit: float = -20.0,
    color_stops: Any = None,
    clip: float = 1.0,
    knot_darkness: float = 0.0,
    knot_darkness_spread_mm: float = _DEFAULT_KNOT_STAIN_SPREAD_MM,
    knot_stain_color: Any = _DEFAULT_KNOT_STAIN_COLOR,
    knot_opacity: float = _DEFAULT_KNOT_STAIN_OPACITY,
    size: int = 512,
) -> Optional[Dict[str, Dict[str, str]]]:
    size_safe = max(16, int(size))
    color_pngs = _build_growth_color_log_cap_overlay_pngs(
        growth_layer_fields,
        outer_field,
        size=size_safe,
        color_stops=color_stops,
        clip=float(clip),
        knot_field=knot_field,
        knot_inside_limit=float(knot_inside_limit),
        knot_darkness=float(knot_darkness),
        knot_darkness_spread_mm=float(knot_darkness_spread_mm),
        knot_stain_color=knot_stain_color,
        knot_opacity=float(knot_opacity),
    )
    return _encode_ring_color_overlay_pngs(
        color_pngs,
        ["y_min", "y_max"],
        size=size_safe,
    )


def _axis_fractional_indices(values: np.ndarray, axis_values: Any) -> np.ndarray:
    coords = np.asarray(axis_values, dtype=np.float32)
    vals = np.asarray(values, dtype=np.float32)
    out = np.full(vals.shape, np.nan, dtype=np.float32)
    finite = np.isfinite(vals)
    if coords.ndim != 1 or coords.size < 2 or not np.any(finite):
        return out
    if coords[0] > coords[-1]:
        coords = coords[::-1]
        reverse = True
    else:
        reverse = False
    idx = np.arange(coords.size, dtype=np.float32)
    interp = np.interp(vals[finite], coords, idx, left=np.nan, right=np.nan).astype(np.float32)
    if reverse:
        interp = (coords.size - 1) - interp
    out[finite] = interp
    return out


def _sample_volume_on_xyz(volume: Any, mesh: Any, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    arr = np.asarray(to_numpy(volume), dtype=np.float32)
    if arr.ndim != 3 or arr.size == 0:
        return np.full(np.asarray(x).shape, np.nan, dtype=np.float32)
    xi = _axis_fractional_indices(x, getattr(mesh, "x_coords", []))
    yi = _axis_fractional_indices(y, getattr(mesh, "y_coords", []))
    zi = _axis_fractional_indices(z, getattr(mesh, "z_coords", []))
    valid = np.isfinite(xi) & np.isfinite(yi) & np.isfinite(zi)
    sampled = np.full(np.asarray(x).shape, np.nan, dtype=np.float32)
    if not np.any(valid):
        return sampled
    coords = np.vstack([yi[valid], xi[valid], zi[valid]])
    sampled[valid] = map_coordinates(arr, coords, order=1, mode="constant", cval=np.nan).astype(np.float32)
    return sampled


def _reduce_surface_knot_field(field: Any) -> Optional[np.ndarray]:
    if field is None:
        return None
    arr = np.asarray(to_numpy(field), dtype=np.float32)
    if arr.ndim == 4:
        finite = np.isfinite(arr)
        arr = np.min(np.where(finite, arr, np.inf), axis=3)
        arr = np.where(np.any(finite, axis=3), arr, np.nan)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if arr.ndim != 2:
        return None
    return arr.astype(np.float32, copy=False)


def _evaluate_np_field_function(func: Any, values: np.ndarray) -> np.ndarray:
    vals = np.asarray(values, dtype=np.float32)
    try:
        out = np.asarray(to_numpy(func(vals)), dtype=np.float32)
    except Exception:
        out = np.zeros_like(vals, dtype=np.float32)
    if out.shape != vals.shape:
        out = np.resize(out, vals.shape).astype(np.float32, copy=False)
    return np.where(np.isfinite(out), out, 0.0).astype(np.float32, copy=False)


def _as_float_array(value: Any) -> np.ndarray:
    return np.asarray(to_numpy(value), dtype=np.float32).reshape(-1)


def _smooth_sigmoid_np(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -60.0, 60.0)))


def _evaluate_veneer_smooth_radial_fields(
    config: BoardConfig,
    knot_system: Any,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    z_grid: np.ndarray,
    *,
    clip: float = 1.0,
    return_deviation: bool = False,
) -> Optional[Any]:
    splines = list(getattr(knot_system, "splines", []) or [])
    if not splines:
        return None
    try:
        x_np = np.asarray(x_grid, dtype=np.float32)
        y_np = np.asarray(y_grid, dtype=np.float32)
        z_np = np.asarray(z_grid, dtype=np.float32)
        if x_np.ndim != 2 or y_np.shape != x_np.shape or z_np.shape != x_np.shape:
            return None

        h, w = x_np.shape
        out = np.full((h, w), np.nan, dtype=np.float32)
        out_deviation = np.full((h, w), np.nan, dtype=np.float32) if bool(return_deviation) else None
        max_points = 90_000
        chunk_cols = max(4, min(w, int(max_points / max(1, h))))
        n_knots = int(getattr(knot_system, "n_knots", 0) or 0)

        if n_knots > 0:
            th0 = _as_float_array(getattr(knot_system, "th0", []))
            z0 = _as_float_array(getattr(knot_system, "z0", []))
            c1 = _as_float_array(getattr(knot_system, "c1", []))
            c2 = _as_float_array(getattr(knot_system, "c2", []))
            k = _as_float_array(getattr(knot_system, "k", []))
            kp = _as_float_array(getattr(knot_system, "kp", []))
            abump = _as_float_array(getattr(knot_system, "Abump", []))
            aexp = _as_float_array(getattr(knot_system, "Aexp", []))
            bbump = _as_float_array(getattr(knot_system, "Bbump", []))
            usable_knots = min(
                n_knots,
                th0.size,
                z0.size,
                c1.size,
                c2.size,
                k.size,
                kp.size,
                abump.size,
                aexp.size,
                bbump.size,
            )
            if usable_knots <= 0:
                n_knots = 0
            else:
                th0 = th0[:usable_knots]
                z0 = z0[:usable_knots]
                c1 = c1[:usable_knots]
                c2 = c2[:usable_knots]
                k = k[:usable_knots]
                kp = np.maximum(kp[:usable_knots], 1e-6)
                abump = abump[:usable_knots]
                aexp = aexp[:usable_knots]
                bbump = bbump[:usable_knots]
                n_knots = usable_knots

        pmin = float(getattr(config, "soft_clamp_pmin", 2.0) or 2.0)
        alpha = float(getattr(config, "soft_clamp_alpha", 1.0) or 1.0)
        knot_gate_mm = 4.0
        radial_gate_mm = 3.0

        for c0 in range(0, w, chunk_cols):
            c1_idx = min(w, c0 + chunk_cols)
            x_chunk = x_np[:, c0:c1_idx]
            y_chunk = y_np[:, c0:c1_idx]
            z_chunk = z_np[:, c0:c1_idx]
            taper = _evaluate_np_field_function(getattr(knot_system, "taper", lambda z: 0.0), z_chunk)
            x_trans = x_chunk + _evaluate_np_field_function(getattr(knot_system, "crook_x", lambda z: 0.0), z_chunk)
            y_trans = y_chunk + _evaluate_np_field_function(getattr(knot_system, "crook_y", lambda z: 0.0), z_chunk)
            radius_sq = x_trans * x_trans + y_trans * y_trans
            theta = np.arctan2(y_chunk, x_chunk).astype(np.float32, copy=False)

            if n_knots > 0:
                cos_th0 = np.cos(th0)[None, None, :]
                sin_th0 = np.sin(th0)[None, None, :]
                x_e = x_trans[..., None]
                y_e = y_trans[..., None]
                z_e = z_chunk[..., None]
                radial_coord = x_e * cos_th0 - y_e * sin_th0
                tangential_coord = x_e * sin_th0 + y_e * cos_th0
                knot_axis_z = c1[None, None, :] * radial_coord**2 + c2[None, None, :] * radial_coord + z0[None, None, :]
                longitudinal_offset = z_e - knot_axis_z
                term_ang = np.arctan2(tangential_coord, radial_coord) ** 2
                p = np.sqrt(
                    longitudinal_offset**2
                    + (radial_coord**2 + tangential_coord**2)
                    * term_ang
                    / (kp[None, None, :] ** 2)
                )
                d = np.clip(alpha * (p - pmin), -60.0, 60.0)
                w_soft = 1.0 / (1.0 + np.exp(-d))
                p = w_soft * p + (1.0 - w_soft) * pmin
                radial_gate = _smooth_sigmoid_np(radial_coord / radial_gate_mm)

            fields: List[np.ndarray] = []
            chunk_deviation = np.zeros_like(x_chunk, dtype=np.float32) if out_deviation is not None else None
            for spline in splines:
                ro = np.asarray(spline(theta), dtype=np.float32) - taper
                ro = np.maximum(ro, 1.0)
                if n_knots > 0:
                    ro_e = ro[..., None]
                    term_knot = (
                        k[None, None, :] * ro_e
                        + abump[None, None, :]
                        * np.power(ro_e, aexp[None, None, :])
                        * np.power(np.maximum(p, 1e-6), -bbump[None, None, :])
                    )
                    denom = np.maximum(1e-6, 1.0 - k[None, None, :])
                    pmax_base = abump[None, None, :] * np.power(ro_e, aexp[None, None, :] - 1.0) / denom
                    pmax = np.power(np.maximum(pmax_base, 1e-9), 1.0 / bbump[None, None, :])
                    influence_gate = _smooth_sigmoid_np((pmax - p) / knot_gate_mm) * radial_gate
                    delta = np.maximum(term_knot - ro_e, 0.0) * influence_gate
                    delta = np.where(np.isfinite(delta), delta, 0.0)
                    delta = np.minimum(delta, 0.6 * ro_e)
                    delta_combined = np.sqrt(np.sum(delta * delta, axis=-1))
                    if chunk_deviation is not None:
                        chunk_deviation = np.maximum(chunk_deviation, delta_combined.astype(np.float32, copy=False))
                    ro_eff = np.maximum(ro + delta_combined, 1.0)
                else:
                    ro_eff = ro
                fields.append((radius_sq - ro_eff * ro_eff).astype(np.float32, copy=False))

            chunk_values = _ring_interval_phase_arrays(fields, clip=float(clip))
            if chunk_values.shape == x_chunk.shape:
                out[:, c0:c1_idx] = chunk_values
                if out_deviation is not None and chunk_deviation is not None:
                    out_deviation[:, c0:c1_idx] = np.where(
                        np.isfinite(chunk_values),
                        chunk_deviation,
                        np.nan,
                    )

        if not np.any(np.isfinite(out)):
            return None
        if out_deviation is not None:
            return out.astype(np.float32, copy=False), out_deviation.astype(np.float32, copy=False)
        return out.astype(np.float32, copy=False)
    except Exception as exc:
        print(f"Veneer smooth radial field error: {exc}")
        return None


def _evaluate_veneer_knot_field(
    config: BoardConfig,
    knot_system: Any,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    z_grid: np.ndarray,
) -> Optional[np.ndarray]:
    splines = list(getattr(knot_system, "splines", []) or [])
    n_knots = int(getattr(knot_system, "n_knots", 0) or 0)
    if not splines or n_knots <= 0:
        return None
    try:
        x_np = np.asarray(x_grid, dtype=np.float32)
        y_np = np.asarray(y_grid, dtype=np.float32)
        z_np = np.asarray(z_grid, dtype=np.float32)
        if x_np.ndim != 2 or y_np.shape != x_np.shape or z_np.shape != x_np.shape:
            return None

        th0 = _as_float_array(getattr(knot_system, "th0", []))
        z0 = _as_float_array(getattr(knot_system, "z0", []))
        c1 = _as_float_array(getattr(knot_system, "c1", []))
        c2 = _as_float_array(getattr(knot_system, "c2", []))
        kp = _as_float_array(getattr(knot_system, "kp", []))
        a1 = _as_float_array(getattr(knot_system, "a1", []))
        a2 = _as_float_array(getattr(knot_system, "a2", []))
        a3 = _as_float_array(getattr(knot_system, "a3", []))
        a4 = _as_float_array(getattr(knot_system, "a4", []))
        rl = _as_float_array(getattr(knot_system, "RL", []))
        usable_knots = min(
            n_knots,
            th0.size,
            z0.size,
            c1.size,
            c2.size,
            kp.size,
            a1.size,
            a2.size,
            a3.size,
            a4.size,
            rl.size,
        )
        if usable_knots <= 0:
            return None

        th0 = th0[:usable_knots]
        z0 = z0[:usable_knots]
        c1 = c1[:usable_knots]
        c2 = c2[:usable_knots]
        kp = np.maximum(kp[:usable_knots], 1e-6)
        a1 = a1[:usable_knots]
        a2 = a2[:usable_knots]
        a3 = a3[:usable_knots]
        a4 = a4[:usable_knots]
        rl = rl[:usable_knots]

        h, w = x_np.shape
        out = np.full((h, w), np.nan, dtype=np.float32)
        max_values = 750_000
        chunk_cols = max(4, min(w, int(max_values / max(1, h * usable_knots))))
        dead_knots = bool(getattr(config, "dead_knots", False))
        outer_spline = splines[-1]

        for c0 in range(0, w, chunk_cols):
            c1_idx = min(w, c0 + chunk_cols)
            x_chunk = x_np[:, c0:c1_idx]
            y_chunk = y_np[:, c0:c1_idx]
            z_chunk = z_np[:, c0:c1_idx]
            theta = np.arctan2(y_chunk, x_chunk).astype(np.float32, copy=False)
            ro = np.asarray(outer_spline(theta), dtype=np.float32)
            ro_mod = ro - _evaluate_np_field_function(getattr(knot_system, "taper", lambda z: 0.0), z_chunk)
            x_trans = x_chunk + _evaluate_np_field_function(getattr(knot_system, "crook_x", lambda z: 0.0), z_chunk)
            y_trans = y_chunk + _evaluate_np_field_function(getattr(knot_system, "crook_y", lambda z: 0.0), z_chunk)

            cos_th0 = np.cos(th0)[None, None, :]
            sin_th0 = np.sin(th0)[None, None, :]
            radial_coord = x_trans[..., None] * cos_th0 - y_trans[..., None] * sin_th0
            tangential_coord = x_trans[..., None] * sin_th0 + y_trans[..., None] * cos_th0
            knot_axis_z = c1[None, None, :] * radial_coord**2 + c2[None, None, :] * radial_coord + z0[None, None, :]
            longitudinal_offset = z_chunk[..., None] - knot_axis_z

            lx = (
                a1[None, None, :] * radial_coord**4
                + a2[None, None, :] * radial_coord**3
                + a3[None, None, :] * radial_coord**2
                + a4[None, None, :] * radial_coord
            )
            lx = np.maximum(lx, 0.0)
            k_field = longitudinal_offset**2 + tangential_coord**2 / (kp[None, None, :] ** 2) - (lx / 2.0) ** 2
            k_field = np.where(radial_coord < 0.0, np.nan, k_field)
            k_field = np.where(radial_coord > (1.2 * ro_mod[..., None]), np.nan, k_field)
            if dead_knots:
                k_field = np.where(radial_coord <= rl[None, None, :], k_field, np.nan)
            k_field = np.where(np.isfinite(k_field), k_field, np.nan)

            finite = np.isfinite(k_field)
            if np.any(finite):
                reduced = np.min(np.where(finite, k_field, np.inf), axis=2)
                out[:, c0:c1_idx] = np.where(np.any(finite, axis=2), reduced, np.nan)

        if not np.any(np.isfinite(out)):
            return None
        return out.astype(np.float32, copy=False)
    except Exception as exc:
        print(f"Veneer knot field error: {exc}")
        return None


def _evaluate_veneer_reaction_lobe_field(
    config: BoardConfig,
    knot_system: Any,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    z_grid: np.ndarray,
    *,
    spread_mm: float = _DEFAULT_KNOT_STAIN_SPREAD_MM,
) -> Optional[np.ndarray]:
    splines = list(getattr(knot_system, "splines", []) or [])
    n_knots = int(getattr(knot_system, "n_knots", 0) or 0)
    if not splines or n_knots <= 0:
        return None
    try:
        x_np = np.asarray(x_grid, dtype=np.float32)
        y_np = np.asarray(y_grid, dtype=np.float32)
        z_np = np.asarray(z_grid, dtype=np.float32)
        if x_np.ndim != 2 or y_np.shape != x_np.shape or z_np.shape != x_np.shape:
            return None

        th0 = _as_float_array(getattr(knot_system, "th0", []))
        z0 = _as_float_array(getattr(knot_system, "z0", []))
        c1_arr = _as_float_array(getattr(knot_system, "c1", []))
        c2_arr = _as_float_array(getattr(knot_system, "c2", []))
        kp = _as_float_array(getattr(knot_system, "kp", []))
        a1 = _as_float_array(getattr(knot_system, "a1", []))
        a2 = _as_float_array(getattr(knot_system, "a2", []))
        a3 = _as_float_array(getattr(knot_system, "a3", []))
        a4 = _as_float_array(getattr(knot_system, "a4", []))
        usable_knots = min(
            n_knots,
            th0.size,
            z0.size,
            c1_arr.size,
            c2_arr.size,
            kp.size,
            a1.size,
            a2.size,
            a3.size,
            a4.size,
        )
        if usable_knots <= 0:
            return None

        th0 = th0[:usable_knots]
        z0 = z0[:usable_knots]
        c1_arr = c1_arr[:usable_knots]
        c2_arr = c2_arr[:usable_knots]
        kp = np.maximum(kp[:usable_knots], 1e-6)
        a1 = a1[:usable_knots]
        a2 = a2[:usable_knots]
        a3 = a3[:usable_knots]
        a4 = a4[:usable_knots]

        spread = float(spread_mm)
        if not np.isfinite(spread) or spread <= 0.0:
            spread = _DEFAULT_KNOT_STAIN_SPREAD_MM
        spread = max(1e-6, spread)

        h, w = x_np.shape
        out = np.zeros((h, w), dtype=np.float32)
        max_values = 650_000
        chunk_cols = max(4, min(w, int(max_values / max(1, h * usable_knots))))
        outer_spline = splines[-1]

        for c0 in range(0, w, chunk_cols):
            c1_idx = min(w, c0 + chunk_cols)
            x_chunk = x_np[:, c0:c1_idx]
            y_chunk = y_np[:, c0:c1_idx]
            z_chunk = z_np[:, c0:c1_idx]
            theta = np.arctan2(y_chunk, x_chunk).astype(np.float32, copy=False)
            taper = _evaluate_np_field_function(getattr(knot_system, "taper", lambda z: 0.0), z_chunk)
            ro_mod = np.asarray(outer_spline(theta), dtype=np.float32) - taper
            ro_mod = np.maximum(ro_mod, 1.0)
            x_trans = x_chunk + _evaluate_np_field_function(getattr(knot_system, "crook_x", lambda z: 0.0), z_chunk)
            y_trans = y_chunk + _evaluate_np_field_function(getattr(knot_system, "crook_y", lambda z: 0.0), z_chunk)

            cos_th0 = np.cos(th0)[None, None, :]
            sin_th0 = np.sin(th0)[None, None, :]
            radial_coord = x_trans[..., None] * cos_th0 - y_trans[..., None] * sin_th0
            tangential_coord = x_trans[..., None] * sin_th0 + y_trans[..., None] * cos_th0
            knot_axis_z = (
                c1_arr[None, None, :] * radial_coord**2
                + c2_arr[None, None, :] * radial_coord
                + z0[None, None, :]
            )
            longitudinal_offset = z_chunk[..., None] - knot_axis_z

            lx = (
                a1[None, None, :] * radial_coord**4
                + a2[None, None, :] * radial_coord**3
                + a3[None, None, :] * radial_coord**2
                + a4[None, None, :] * radial_coord
            )
            branch_radius = np.maximum(0.5 * lx, 0.0)
            branch_radius_safe = np.maximum(branch_radius, 0.25)
            axis_dist = np.sqrt(longitudinal_offset**2 + (tangential_coord / kp[None, None, :]) ** 2)

            outside_dist = np.maximum(axis_dist - branch_radius_safe, 0.0)
            zone = (spread * 1.25) + (branch_radius_safe * 0.45)
            axis_gate = np.exp(-0.5 * (outside_dist / np.maximum(zone, 1e-6)) ** 2)
            inside_gate = _smooth_sigmoid_np((branch_radius_safe - axis_dist) / max(1.0, spread * 0.18))

            radial_soft = max(2.0, spread * 0.18)
            radial_gate = (
                _smooth_sigmoid_np(radial_coord / radial_soft)
                * _smooth_sigmoid_np(((1.16 * ro_mod[..., None]) - radial_coord) / max(2.0, spread * 0.25))
            )

            side_width = np.maximum(np.maximum(branch_radius_safe * 0.80, spread * 0.35), 2.0)
            side_bias = 0.35 + (0.65 * _smooth_sigmoid_np(tangential_coord / side_width))
            long_width = (spread * 2.25) + (branch_radius_safe * 1.40)
            long_tail = np.exp(-0.5 * (longitudinal_offset / np.maximum(long_width, 1e-6)) ** 2)
            lobe = axis_gate * (1.0 - (0.45 * inside_gate)) * radial_gate * side_bias * (0.35 + (0.65 * long_tail))

            valid = (
                np.isfinite(lobe)
                & (radial_coord > 0.0)
                & (radial_coord < (1.20 * ro_mod[..., None]))
            )
            lobe = np.where(valid, lobe, 0.0)
            out[:, c0:c1_idx] = np.maximum(out[:, c0:c1_idx], np.max(lobe, axis=2).astype(np.float32, copy=False))

        out = np.clip(out, 0.0, 1.0)
        if not np.any(out > 1e-4):
            return None
        return out.astype(np.float32, copy=False)
    except Exception as exc:
        print(f"Veneer reaction lobe field error: {exc}")
        return None


def _smooth_1d(values: np.ndarray, sigma: float) -> np.ndarray:
    sigma_safe = float(sigma)
    arr = np.asarray(values, dtype=np.float32)
    if not np.isfinite(sigma_safe) or sigma_safe <= 0.35 or arr.size < 3:
        return arr.astype(np.float32, copy=False)
    radius = int(min(96, max(1, math.ceil(3.0 * sigma_safe))))
    xx = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-0.5 * (xx / sigma_safe) ** 2)
    kernel = kernel / max(1e-12, float(np.sum(kernel)))
    padded = np.pad(arr, (radius, radius), mode="reflect")
    return np.convolve(padded, kernel.astype(np.float32), mode="valid").astype(np.float32, copy=False)


def _resize_signed_noise(noise: np.ndarray, *, width: int, height: int) -> np.ndarray:
    n = _normalize_unit_std(np.asarray(noise, dtype=np.float32))
    image01 = np.clip(0.5 + (0.16 * np.clip(n, -3.0, 3.0)), 0.0, 1.0)
    image = Image.fromarray(np.rint(image01 * 255.0).astype(np.uint8), mode="L")
    resized = image.resize((int(width), int(height)), resample=_pil_lanczos())
    arr = (np.asarray(resized, dtype=np.float32) / 255.0 - 0.5) / 0.16
    return _normalize_unit_std(arr)


def _apply_veneer_fiber_texture_to_rgb(
    rgb: np.ndarray,
    ring_phase: np.ndarray,
    knot_field_image: Optional[np.ndarray],
    *,
    strength: float = _DEFAULT_VENEER_FIBER_TEXTURE_STRENGTH,
    scale_mm: float = _DEFAULT_VENEER_FIBER_TEXTURE_SCALE_MM,
    streak_length_mm: float = _DEFAULT_VENEER_FIBER_TEXTURE_LENGTH_MM,
    sheet_length_mm: float = 1.0,
    log_length_mm: float = 1.0,
    knot_inside_limit: float = -20.0,
    knot_spread_mm: float = _DEFAULT_KNOT_STAIN_SPREAD_MM,
) -> np.ndarray:
    s = float(strength)
    scale = float(scale_mm)
    streak_length = float(streak_length_mm)
    if (
        not np.isfinite(s)
        or not np.isfinite(scale)
        or not np.isfinite(streak_length)
        or s <= 0.0
        or scale <= 0.0
        or streak_length <= 0.0
    ):
        return rgb

    base_uint8 = np.asarray(rgb, dtype=np.uint8)
    phase = np.asarray(ring_phase, dtype=np.float32)
    if base_uint8.ndim != 3 or base_uint8.shape[-1] != 3 or phase.shape != base_uint8.shape[:2]:
        return rgb
    h, w = int(phase.shape[0]), int(phase.shape[1])
    if h < 4 or w < 4:
        return rgb

    finite = np.isfinite(phase)
    if not np.any(finite):
        return rgb

    s = float(np.clip(s, 0.0, 2.0))
    px_per_mm_x = (w - 1) / max(1e-6, float(sheet_length_mm))
    px_per_mm_y = (h - 1) / max(1e-6, float(log_length_mm))
    scale_px = float(np.clip(scale * px_per_mm_x, 0.45, 12.0))
    streak_px = float(np.clip(streak_length * px_per_mm_y, 6.0, max(6.0, h * 1.5)))

    digest = hashlib.sha256()
    digest.update(b"veneer_fiber_texture_v1")
    digest.update(np.asarray([h, w, s, scale, streak_length, sheet_length_mm, log_length_mm], dtype=np.float32).tobytes())
    sy = max(1, h // 96)
    sx = max(1, w // 160)
    digest.update(np.nan_to_num(phase[::sy, ::sx], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32).tobytes())
    if knot_field_image is not None:
        knot_arr_for_seed = np.asarray(knot_field_image, dtype=np.float32)
        if knot_arr_for_seed.shape == phase.shape:
            digest.update(np.nan_to_num(knot_arr_for_seed[::sy, ::sx], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32).tobytes())
    rng = np.random.default_rng(int.from_bytes(digest.digest()[:8], byteorder="little", signed=False))

    coarse_h = int(np.clip(round(h / max(1.0, streak_px / 7.5)), 6, h))
    long_noise = _resize_signed_noise(rng.normal(0.0, 1.0, size=(coarse_h, w)).astype(np.float32), width=w, height=h)

    column_noise = _smooth_1d(rng.normal(0.0, 1.0, size=w).astype(np.float32), sigma=max(0.5, scale_px * 0.9))
    column_noise = _normalize_unit_std(column_noise)
    column_noise = np.broadcast_to(column_noise[None, :], (h, w)).astype(np.float32, copy=False)

    micro_h = max(12, min(h, int(round(h / 3.0))))
    micro_w = max(32, min(w, int(round(w / max(1.0, scale_px)))))
    micro_noise = _resize_signed_noise(
        rng.normal(0.0, 1.0, size=(micro_h, micro_w)).astype(np.float32),
        width=w,
        height=h,
    )

    texture = _normalize_unit_std((0.50 * long_noise) + (0.35 * column_noise) + (0.15 * micro_noise))

    knot_influence = None
    if knot_field_image is not None:
        knot_field = np.asarray(knot_field_image, dtype=np.float32)
        if knot_field.shape == phase.shape:
            finite_knot = np.isfinite(knot_field)
            if np.any(finite_knot):
                limit = float(knot_inside_limit)
                if not np.isfinite(limit):
                    limit = -20.0
                spread = max(1e-6, float(knot_spread_mm))
                influence = np.zeros_like(knot_field, dtype=np.float32)
                inside = finite_knot & (knot_field <= limit)
                outside = finite_knot & ~inside
                influence[inside] = 1.0
                distance = np.zeros_like(knot_field, dtype=np.float32)
                distance[outside] = np.sqrt(np.maximum(knot_field[outside] - limit, 0.0))
                influence[outside] = np.exp(-0.5 * (distance[outside] / spread) ** 2)
                if np.any(influence > 1e-4):
                    blur_px = float(np.clip(spread * 0.20 * max(px_per_mm_x, px_per_mm_y), 1.0, 18.0))
                    knot_influence = _pil_blur_float01(np.clip(influence, 0.0, 1.0), radius=blur_px)
                    gy, gx = np.gradient(knot_influence)
                    grad_mag = np.abs(gx) + np.abs(gy)
                    norm = float(np.percentile(grad_mag[grad_mag > 0.0], 98.0)) if np.any(grad_mag > 0.0) else 0.0
                    if np.isfinite(norm) and norm > 1e-8:
                        warp_px = float(np.clip(spread * px_per_mm_x * (0.20 + 0.35 * s), 1.0, 28.0))
                        yy, xx = np.meshgrid(
                            np.arange(h, dtype=np.float32),
                            np.arange(w, dtype=np.float32),
                            indexing="ij",
                        )
                        coords = np.asarray([
                            yy + (0.45 * warp_px * gy / norm),
                            xx + (warp_px * gx / norm),
                        ], dtype=np.float32)
                        texture = map_coordinates(texture, coords, order=1, mode="reflect").astype(np.float32, copy=False)

    phase_abs = np.abs(np.nan_to_num(phase, nan=1.0, posinf=1.0, neginf=1.0))
    latewood = np.exp(-0.5 * (phase_abs / 0.10) ** 2).astype(np.float32, copy=False)
    ring_weight = np.clip(0.62 + (0.48 * latewood), 0.0, 1.15)
    if knot_influence is not None:
        ring_weight = ring_weight * np.clip(1.0 - (0.22 * knot_influence), 0.70, 1.0)

    texture = _normalize_unit_std(texture)
    texture = np.clip(texture, -2.7, 2.7)
    dark = np.clip(texture, 0.0, 2.7)
    light = np.clip(-texture, 0.0, 2.7)
    delta = (0.026 * s) * ring_weight * ((0.58 * light) - (0.72 * dark))
    delta[finite] = delta[finite] - float(np.mean(delta[finite], dtype=np.float64))
    delta[~finite] = 0.0

    base = base_uint8.astype(np.float32) / 255.0
    gain = np.asarray([0.90, 0.98, 1.08], dtype=np.float32).reshape(1, 1, 3)
    out = base.copy()
    out[..., :3] = np.where(
        finite[..., None],
        np.clip(base[..., :3] + (delta[..., None] * gain), 0.0, 1.0),
        base[..., :3],
    )
    return np.rint(out * 255.0).astype(np.uint8)


def _build_veneer_payload(
    config: BoardConfig,
    knot_system: Any,
    mesh: Any,
    layers_data: Dict[str, Any],
    *,
    color_stops: Any = None,
    clip: float = 1.0,
    knot_darkness: float = 0.0,
    knot_darkness_spread_mm: float = _DEFAULT_KNOT_STAIN_SPREAD_MM,
    knot_stain_color: Any = _DEFAULT_KNOT_STAIN_COLOR,
    knot_opacity: float = _DEFAULT_KNOT_STAIN_OPACITY,
    knot_core_strength: float = _DEFAULT_KNOT_CORE_STRENGTH,
    knot_ring_strength: float = _DEFAULT_KNOT_RING_STRENGTH,
    knot_reaction_strength: float = _DEFAULT_KNOT_REACTION_STRENGTH,
) -> Optional[Dict[str, Any]]:
    growth_layer_fields = layers_data.get("growth_layer_fields") or []
    normalized = _nearest_ring_normalized_field(growth_layer_fields, clip=float(clip))
    if normalized.ndim != 3 or normalized.size == 0:
        return None

    try:
        x_coords = np.asarray(getattr(mesh, "x_coords", []), dtype=np.float32)
        y_coords = np.asarray(getattr(mesh, "y_coords", []), dtype=np.float32)
        z_coords = np.asarray(getattr(mesh, "z_coords", []), dtype=np.float32)
        if x_coords.size < 2 or y_coords.size < 2 or z_coords.size < 2:
            return None

        x_center = 0.5 * (float(config.board_x_min) + float(config.board_x_max))
        y_center = 0.5 * (float(config.board_y_min) + float(config.board_y_max))
        z_min = float(np.min(z_coords))
        z_max = float(np.max(z_coords))
        log_length = max(1e-6, z_max - z_min)

        outer = max(1e-6, float(getattr(config, "veneer_outer_radius_mm", 50.0) or 50.0))
        inner = max(0.0, float(getattr(config, "veneer_inner_radius_mm", 20.0) or 20.0))
        if inner >= outer:
            inner = max(0.0, outer - 1.0)
        thickness = max(1e-6, float(getattr(config, "veneer_thickness_mm", 3.0) or 3.0))
        requested_length = max(0.0, float(getattr(config, "veneer_length_mm", 1000.0) or 1000.0))
        turns_to_inner = max(0.0, (outer - inner) / thickness)
        theta_limit = 2.0 * math.pi * turns_to_inner
        if theta_limit <= 1e-6:
            return None

        # In rotary peeling, one full revolution removes one veneer thickness.
        dense_count = max(512, min(20000, int(math.ceil(theta_limit * 48.0))))
        theta_dense = np.linspace(0.0, theta_limit, dense_count, dtype=np.float32)
        radius_dense = outer - (thickness * theta_dense / (2.0 * math.pi))
        radius_dense = np.maximum(radius_dense, inner)
        dtheta = np.diff(theta_dense)
        radius_mid = 0.5 * (radius_dense[:-1] + radius_dense[1:])
        dr_dtheta = thickness / (2.0 * math.pi)
        ds = np.sqrt(radius_mid**2 + dr_dtheta**2) * dtheta
        arc_dense = np.concatenate([[0.0], np.cumsum(ds)]).astype(np.float32)
        max_length = float(arc_dense[-1])
        target_length = min(max_length, requested_length) if requested_length > 0.0 else max_length
        if target_length <= 1e-6:
            return None

        length_samples = max(64, min(2400, int(getattr(config, "veneer_sheet_samples_length", 900) or 900)))
        width_samples = max(32, min(1200, int(getattr(config, "veneer_sheet_samples_width", 260) or 260)))
        sheet_s = np.linspace(0.0, target_length, length_samples, dtype=np.float32)
        theta = np.interp(sheet_s, arc_dense, theta_dense).astype(np.float32)
        radius = np.interp(sheet_s, arc_dense, radius_dense).astype(np.float32)
        z_vals = np.linspace(z_min, z_max, width_samples, dtype=np.float32)

        x_curve = x_center + radius * np.cos(theta)
        y_curve = y_center + radius * np.sin(theta)
        x_grid = np.broadcast_to(x_curve[None, :], (width_samples, length_samples))
        y_grid = np.broadcast_to(y_curve[None, :], (width_samples, length_samples))
        z_grid = np.broadcast_to(z_vals[:, None], (width_samples, length_samples))

        fiber_texture_strength = float(
            getattr(config, "veneer_fiber_texture_strength", _DEFAULT_VENEER_FIBER_TEXTURE_STRENGTH)
            if getattr(config, "veneer_fiber_texture_strength", None) is not None
            else _DEFAULT_VENEER_FIBER_TEXTURE_STRENGTH
        )
        fiber_texture_scale_mm = float(
            getattr(config, "veneer_fiber_texture_scale_mm", _DEFAULT_VENEER_FIBER_TEXTURE_SCALE_MM)
            or _DEFAULT_VENEER_FIBER_TEXTURE_SCALE_MM
        )
        fiber_texture_length_mm = float(
            getattr(config, "veneer_fiber_texture_length_mm", _DEFAULT_VENEER_FIBER_TEXTURE_LENGTH_MM)
            or _DEFAULT_VENEER_FIBER_TEXTURE_LENGTH_MM
        )

        knot_color_active = float(knot_darkness) > 0.0 and float(knot_opacity) > 0.0
        knot_sheet = None
        reaction_lobe = None
        if knot_color_active or float(fiber_texture_strength) > 0.0:
            knot_sheet = _evaluate_veneer_knot_field(config, knot_system, x_grid, y_grid, z_grid)
            if knot_sheet is None:
                knot_field = layers_data.get("ttt_live")
                if knot_field is None:
                    knot_field = layers_data.get("ttt")
                if knot_field is not None:
                    knot_sheet = _sample_volume_on_xyz(knot_field, mesh, x_grid, y_grid, z_grid)
        if knot_color_active:
            reaction_lobe = _evaluate_veneer_reaction_lobe_field(
                config,
                knot_system,
                x_grid,
                y_grid,
                z_grid,
                spread_mm=float(knot_darkness_spread_mm),
            )

        sheet_result = _evaluate_veneer_smooth_radial_fields(
            config,
            knot_system,
            x_grid,
            y_grid,
            z_grid,
            clip=float(clip),
            return_deviation=True,
        )
        sheet_deviation = None
        if isinstance(sheet_result, tuple) and len(sheet_result) >= 2:
            sheet_values = sheet_result[0]
            sheet_deviation = sheet_result[1]
        else:
            sheet_values = sheet_result

        if sheet_values is None:
            sheet_values = _sample_volume_on_xyz(normalized, mesh, x_grid, y_grid, z_grid)
        finite_sheet = np.isfinite(sheet_values)
        if not np.any(finite_sheet):
            return None

        stops = _parse_ring_color_stops(color_stops)
        image_arr = _colorize_normalized_ring_values(sheet_values, stops)
        image_arr = _apply_veneer_knot_color_layers_to_rgb(
            image_arr,
            sheet_values,
            knot_sheet,
            sheet_deviation,
            reaction_lobe,
            strength=float(knot_darkness),
            stain_color=knot_stain_color,
            opacity=float(knot_opacity),
            core_strength=float(knot_core_strength),
            ring_strength=float(knot_ring_strength),
            reaction_strength=float(knot_reaction_strength),
        )
        image_arr = _apply_veneer_fiber_texture_to_rgb(
            image_arr,
            sheet_values,
            knot_sheet,
            strength=fiber_texture_strength,
            scale_mm=fiber_texture_scale_mm,
            streak_length_mm=fiber_texture_length_mm,
            sheet_length_mm=float(target_length),
            log_length_mm=float(log_length),
            knot_inside_limit=float(config.knot_inside_limit),
            knot_spread_mm=float(knot_darkness_spread_mm),
        )
        alpha = np.where(finite_sheet, 255, 0).astype(np.uint8)
        sheet_rgba = np.dstack([image_arr, alpha])
        sheet_img = Image.fromarray(sheet_rgba, mode="RGBA")
        sheet_physical_aspect = max(1e-6, float(target_length) / max(1e-6, float(log_length)))
        display_height = max(32, min(220, int(width_samples)))
        display_width = max(64, int(round(display_height * sheet_physical_aspect)))
        max_display_width = 9000
        if display_width > max_display_width:
            scale = max_display_width / float(display_width)
            display_width = max_display_width
            display_height = max(32, int(round(display_height * scale)))
        if display_width != int(length_samples) or display_height != int(width_samples):
            sheet_img = sheet_img.resize((int(display_width), int(display_height)), resample=_pil_lanczos())

        sheet_buffer = BytesIO()
        sheet_img.save(sheet_buffer, format="PNG", optimize=False)

        preview_size = 512
        xx = np.linspace(float(np.min(x_coords)), float(np.max(x_coords)), preview_size, dtype=np.float32)
        yy = np.linspace(float(np.max(y_coords)), float(np.min(y_coords)), preview_size, dtype=np.float32)
        px, py = np.meshgrid(xx, yy)
        pz = np.full_like(px, z_max)
        preview_values = _sample_volume_on_xyz(normalized, mesh, px, py, pz)
        preview_arr = _colorize_normalized_ring_values(preview_values, stops)
        preview_alpha = np.where(np.isfinite(preview_values), 255, 36).astype(np.uint8)
        preview_img = Image.fromarray(np.dstack([preview_arr, preview_alpha]), mode="RGBA")
        preview_draw = ImageDraw.Draw(preview_img)

        def to_preview_xy(xv: np.ndarray, yv: np.ndarray) -> List[Tuple[float, float]]:
            x_min = float(np.min(x_coords))
            x_max = float(np.max(x_coords))
            y_min = float(np.min(y_coords))
            y_max = float(np.max(y_coords))
            pxs = (np.asarray(xv) - x_min) / max(1e-9, x_max - x_min) * (preview_size - 1)
            pys = (1.0 - ((np.asarray(yv) - y_min) / max(1e-9, y_max - y_min))) * (preview_size - 1)
            return list(zip(pxs.astype(float).tolist(), pys.astype(float).tolist()))

        preview_theta = np.linspace(0.0, float(theta[-1]), max(128, min(4000, int(length_samples * 1.5))), dtype=np.float32)
        preview_radius = outer - (thickness * preview_theta / (2.0 * math.pi))
        preview_radius = np.maximum(preview_radius, inner)
        preview_x = x_center + preview_radius * np.cos(preview_theta)
        preview_y = y_center + preview_radius * np.sin(preview_theta)
        preview_draw.line(to_preview_xy(preview_x, preview_y), fill=(0, 146, 200, 255), width=3)
        for rr, color in [(outer, (0, 146, 200, 130)), (inner, (0, 146, 200, 90))]:
            bbox = to_preview_xy(
                np.asarray([x_center - rr, x_center + rr], dtype=np.float32),
                np.asarray([y_center + rr, y_center - rr], dtype=np.float32),
            )
            preview_draw.ellipse([bbox[0][0], bbox[0][1], bbox[1][0], bbox[1][1]], outline=color, width=1)

        preview_buffer = BytesIO()
        preview_img.save(preview_buffer, format="PNG", optimize=False)

        return {
            "sheet": {
                "filename": "veneer_sheet_color.png",
                "src": f"data:image/png;base64,{base64.b64encode(sheet_buffer.getvalue()).decode('ascii')}",
                "width_px": int(display_width),
                "height_px": int(display_height),
                "sample_width_px": int(length_samples),
                "sample_height_px": int(width_samples),
                "physical_width_mm": float(target_length),
                "physical_height_mm": float(log_length),
            },
            "preview": {
                "filename": "veneer_spiral_preview.png",
                "src": f"data:image/png;base64,{base64.b64encode(preview_buffer.getvalue()).decode('ascii')}",
                "width_px": int(preview_size),
                "height_px": int(preview_size),
            },
            "params": {
                "outer_radius_mm": float(outer),
                "inner_radius_mm": float(inner),
                "thickness_mm": float(thickness),
                "requested_length_mm": float(requested_length),
                "actual_length_mm": float(target_length),
                "log_length_mm": float(log_length),
                "turns": float(theta[-1] / (2.0 * math.pi)),
                "knot_color_model": "layered_core_latewood_reaction",
                "knot_color_metric": "knot_implicit_field + growth_layer_radial_deviation_mm + branch_axis_reaction_lobe",
                "knot_core_strength": float(knot_core_strength),
                "knot_ring_strength": float(knot_ring_strength),
                "knot_reaction_strength": float(knot_reaction_strength),
                "fiber_texture_strength": float(fiber_texture_strength),
                "fiber_texture_scale_mm": float(fiber_texture_scale_mm),
                "fiber_texture_length_mm": float(fiber_texture_length_mm),
            },
        }
    except Exception as exc:
        print(f"Veneer render error: {exc}")
        return None


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
    index_path = _frontend_index_path()
    if index_path.is_file():
        return FileResponse(index_path)
    return {"message": "Board Generator API"}


@app.get("/capabilities")
def get_capabilities():
    if _DEMO_MODE:
        photorealistic_capability = {
            "available": False,
            "reason": _PHOTOREALISTIC_DISABLED_REASON,
            "device": "",
            "cuda_available": False,
            "recommended_ddim_steps": 50,
            "loaded": False,
        }
    else:
        photorealistic_capability = get_photorealistic_capability()
    return {
        "photorealistic_export": photorealistic_capability,
    }


@app.post("/photorealistic/preload")
def preload_photorealistic():
    try:
        if _DEMO_MODE:
            raise HTTPException(status_code=503, detail=_PHOTOREALISTIC_DISABLED_REASON)
        preload_info = preload_photorealistic_model()
        return {
            "ok": True,
            "loaded": bool(preload_info.get("loaded")),
            "capability": get_photorealistic_capability(),
        }
    except HTTPException:
        raise
    except PhotorealisticUnavailableError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except PhotorealisticInferenceError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to preload photorealistic model: {e}")


@app.post("/render/ring-color-overlays")
def render_ring_color_overlays(req: RenderRingColorOverlaysRequest):
    try:
        sim_id = str(req.simulation_id or "").strip()
        if not sim_id:
            raise HTTPException(status_code=400, detail="simulation_id is required.")
        entry = _SIM_CACHE.get(sim_id)
        if entry is None:
            raise HTTPException(status_code=404, detail="Simulation data not found. Regenerate board before rendering color overlays.")

        scalar_fields = entry.get("scalar_fields") or {}
        growth_fields = scalar_fields.get("growth_layer_fields")
        if growth_fields is None:
            raise HTTPException(status_code=400, detail="Cached growth-layer scalar fields are missing. Regenerate board and retry.")

        size = max(16, int(req.size or 512))
        clip = max(1e-6, float(req.ring_color_clip if req.ring_color_clip is not None else 1.0))
        knot_darkness = float(
            req.ring_color_knot_darkness
            if req.ring_color_knot_darkness is not None
            else _DEFAULT_KNOT_STAIN_DARKNESS
        )
        knot_darkness_spread = max(
            1e-6,
            float(
                req.ring_color_knot_spread_mm
                if req.ring_color_knot_spread_mm is not None
                else _DEFAULT_KNOT_STAIN_SPREAD_MM
            ),
        )
        knot_stain_color = str(req.ring_color_knot_stain_color or _DEFAULT_KNOT_STAIN_COLOR)
        knot_opacity = float(np.clip(
            float(
                req.ring_color_knot_opacity
                if req.ring_color_knot_opacity is not None
                else _DEFAULT_KNOT_STAIN_OPACITY
            ),
            0.0,
            1.0,
        ))
        color_stops = req.ring_color_stops
        if str(entry.get("export_mode") or "board") in {"log", "veneer"}:
            overlays = _build_log_ring_color_overlay_payload(
                growth_fields,
                scalar_fields.get("outer_log_field"),
                knot_field=scalar_fields.get("knot_field"),
                knot_inside_limit=float(req.knot_inside_limit) if req.knot_inside_limit is not None else -20.0,
                size=size,
                color_stops=color_stops,
                clip=clip,
                knot_darkness=knot_darkness,
                knot_darkness_spread_mm=knot_darkness_spread,
                knot_stain_color=knot_stain_color,
                knot_opacity=knot_opacity,
            )
        else:
            overlays = _build_board_ring_color_overlay_payload(
                growth_fields,
                knot_field=scalar_fields.get("knot_field"),
                show_rings_inside_knots=bool(req.show_rings_inside_knots) if req.show_rings_inside_knots is not None else True,
                knot_inside_limit=float(req.knot_inside_limit) if req.knot_inside_limit is not None else -20.0,
                size=size,
                color_stops=color_stops,
                clip=clip,
                knot_darkness=knot_darkness,
                knot_darkness_spread_mm=knot_darkness_spread,
                knot_stain_color=knot_stain_color,
                knot_opacity=knot_opacity,
            )
        return {"ring_color_overlays": overlays or {}}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to render ring color overlays: {e}")


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
        if _DEMO_MODE:
            raise HTTPException(status_code=503, detail=_PHOTOREALISTIC_DISABLED_REASON)

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
        if _DEMO_MODE:
            config.use_gpu = False

        seeded_mode = bool(config.use_seed)
        base_seed = int(config.simulation_seed) if seeded_mode else 0

        # 1. Initialize + random-board retries for UI mode.
        # Without retries, some random cross-sections can miss the board footprint
        # and produce empty contour sets.
        mode = int(getattr(config, "board_or_log", 0))
        board_mode = mode == 0
        veneer_mode = mode == 2
        export_mode_name = "veneer" if veneer_mode else ("log" if not board_mode else "board")
        if not board_mode:
            config.quiver_or_stream = 0
            if veneer_mode:
                # Veneer mode renders the flattened color field directly.
                config.calc_fibers = False
                config.display_contours = False
                for attr in ("mesh_size_x_mm", "mesh_size_y_mm", "mesh_size_z_mm"):
                    try:
                        mesh_size = float(getattr(config, attr) or _VENEER_INTERNAL_MESH_SIZE_MM)
                    except (TypeError, ValueError):
                        mesh_size = _VENEER_INTERNAL_MESH_SIZE_MM
                    setattr(config, attr, max(mesh_size, _VENEER_INTERNAL_MESH_SIZE_MM))
            else:
                # Log-mode fiber display is intentionally disabled in the UI, so the
                # frontend sends calc_fibers=false. Still compute the field for MAT export.
                config.calc_fibers = True
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

        ring_color_overlays = None
        veneer_payload = None
        ring_color_kwargs = {
            "size": 512,
            "color_stops": (getattr(config, "ring_color_stops", "") or _DEFAULT_RING_COLOR_STOPS_TEXT),
            "clip": float(getattr(config, "ring_color_clip", 1.0) or 1.0),
            "knot_darkness": float(
                getattr(config, "ring_color_knot_darkness", _DEFAULT_KNOT_STAIN_DARKNESS)
                if getattr(config, "ring_color_knot_darkness", None) is not None
                else _DEFAULT_KNOT_STAIN_DARKNESS
            ),
            "knot_darkness_spread_mm": float(
                getattr(config, "ring_color_knot_spread_mm", _DEFAULT_KNOT_STAIN_SPREAD_MM)
                or _DEFAULT_KNOT_STAIN_SPREAD_MM
            ),
            "knot_stain_color": str(
                getattr(config, "ring_color_knot_stain_color", _DEFAULT_KNOT_STAIN_COLOR)
                or _DEFAULT_KNOT_STAIN_COLOR
            ),
            "knot_opacity": float(np.clip(
                float(
                    getattr(config, "ring_color_knot_opacity", _DEFAULT_KNOT_STAIN_OPACITY)
                    if getattr(config, "ring_color_knot_opacity", None) is not None
                    else _DEFAULT_KNOT_STAIN_OPACITY
                ),
                0.0,
                1.0,
            )),
        }
        knot_field = layers_data.get("ttt_live")
        if knot_field is None:
            knot_field = layers_data.get("ttt")
        if bool(getattr(config, "display_ring_color", False)):
            try:
                if mode == 0:
                    ring_color_overlays = _build_board_ring_color_overlay_payload(
                        layers_data.get("growth_layer_fields") or [],
                        knot_field=knot_field,
                        show_rings_inside_knots=bool(getattr(config, "display_rings_inside_knots", True)),
                        knot_inside_limit=float(config.knot_inside_limit),
                        **ring_color_kwargs,
                    )
                else:
                    ring_color_overlays = _build_log_ring_color_overlay_payload(
                        layers_data.get("growth_layer_fields") or [],
                        layers_data.get("last_g"),
                        knot_field=knot_field,
                        knot_inside_limit=float(config.knot_inside_limit),
                        **ring_color_kwargs,
                    )
            except Exception as e:
                print(f"Ring color overlay render error: {e}")
                ring_color_overlays = None
        if veneer_mode:
            try:
                veneer_kwargs = dict(ring_color_kwargs)
                veneer_kwargs.pop("size", None)
                veneer_kwargs.update({
                    "knot_core_strength": float(np.clip(
                        float(
                            getattr(config, "ring_color_knot_core_strength", _DEFAULT_KNOT_CORE_STRENGTH)
                            if getattr(config, "ring_color_knot_core_strength", None) is not None
                            else _DEFAULT_KNOT_CORE_STRENGTH
                        ),
                        0.0,
                        2.0,
                    )),
                    "knot_ring_strength": float(np.clip(
                        float(
                            getattr(config, "ring_color_knot_ring_strength", _DEFAULT_KNOT_RING_STRENGTH)
                            if getattr(config, "ring_color_knot_ring_strength", None) is not None
                            else _DEFAULT_KNOT_RING_STRENGTH
                        ),
                        0.0,
                        2.0,
                    )),
                    "knot_reaction_strength": float(np.clip(
                        float(
                            getattr(config, "ring_color_knot_reaction_strength", _DEFAULT_KNOT_REACTION_STRENGTH)
                            if getattr(config, "ring_color_knot_reaction_strength", None) is not None
                            else _DEFAULT_KNOT_REACTION_STRENGTH
                        ),
                        0.0,
                        2.0,
                    )),
                })
                veneer_payload = _build_veneer_payload(
                    config,
                    k,
                    mesh,
                    layers_data,
                    **veneer_kwargs,
                )
                if veneer_payload is None:
                    warnings.append("Veneer rendering did not produce a valid sheet. Check spiral radius, mesh extent, and ring fields.")
            except Exception as e:
                print(f"Veneer render error: {e}")
                veneer_payload = None
                warnings.append(f"Veneer rendering failed: {e}")

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
            "outer_log_field": _to_float32_3d(layers_data.get("last_g")),
            "growth_layer_fields": growth_layer_fields_mat,
            "growth_layer_indices": growth_layer_indices_mat,
        }

        # The full knot influence payload can be very large; after building
        # knot meshes and any fiber outputs, it should not survive into cache.
        layers_data.pop("knot_influence_info", None)

        sim_id = _cache_simulation({
            "export_mode": export_mode_name,
            "fiber_domain": "log" if not board_mode else "board",
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
            "veneer": veneer_payload,
            "knot_sequence": dict(getattr(k, "knot_sequence_info", {}) or {}),
            "geometry_randomization": dict(getattr(k, "geometry_randomization_info", {}) or {}),
        })

        result = {
            "export_mode": export_mode_name,
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
            "ring_color_overlays": ring_color_overlays,
            "veneer": veneer_payload,
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


@app.get("/{asset_path:path}", include_in_schema=False)
def serve_frontend_asset(asset_path: str):
    frontend_dir = _frontend_dist_dir()
    index_path = frontend_dir / "index.html"
    if not index_path.is_file():
        raise HTTPException(status_code=404, detail="Frontend build not found.")

    requested = (frontend_dir / str(asset_path or "")).resolve()
    try:
        requested.relative_to(frontend_dir.resolve())
    except Exception:
        raise HTTPException(status_code=404, detail="Asset not found.")

    if requested.is_file():
        return FileResponse(requested)
    return FileResponse(index_path)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8100)
