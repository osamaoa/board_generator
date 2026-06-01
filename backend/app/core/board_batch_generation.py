from __future__ import annotations

import json
import multiprocessing as mp
import shutil
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from app.core.array_backend import seed_all
from app.core.config import BoardConfig
from app.core.fiber import FiberSolver
from app.core.growth import GrowthSimulator
from app.core.knot_system import KnotSystem
from app.core.mesh import BoardMesh
from app.core.photorealistic_inference import (
    generate_photorealistic_surfaces_batch,
    get_photorealistic_capability,
    preload_photorealistic_model,
)
from app.main import (
    _apply_fiber_irregularity_bytes,
    _apply_png_blur_bytes,
    _apply_ring_irregularity_bytes,
    _board_fit_warnings,
    _build_fiber_surface_pngs,
    _build_matlab_mid_ring_png,
    _build_matlab_ring_pngs,
    _evaluate_outer_radius,
    _flip_png_vertical_bytes,
    _render_surface_png_matlab,
)


_ALLOWED_OUTPUTS = {"rings", "fibers", "middle", "top_bottom", "photorealistic"}
_IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")
_EXTENT_KEYS = (
    "board_x_min",
    "board_x_max",
    "board_y_min",
    "board_y_max",
    "board_z_min",
    "board_z_max",
)
_DIMENSION_KEYS = ("board_width", "board_thickness", "board_length")
_RELEVANT_CONFIG_KEYS = {
    # Geometry / board extents
    "board_x_min",
    "board_x_max",
    "board_y_min",
    "board_y_max",
    "board_z_min",
    "board_z_max",
    # Board dimensions for random placement mode
    "board_width",
    "board_thickness",
    "board_length",
    "randomize_board_extents_from_dimensions",
    # Mesh controls
    "mesh_size_x_mm",
    "mesh_size_y_mm",
    "mesh_size_z_mm",
    # Reproducibility / runtime
    "use_seed",
    "simulation_seed",
    "use_gpu",
    # Knot controls
    "use_input_knots",
    "input_knot_count",
    "input_knots",
    "randomize_crook_taper",
    "crook_component_count",
    "crook_shift_max_mm",
    "random_crook_scale_max",
    "random_crook_amplitude_max",
    "random_crook_extra_orders",
    "random_crook_theta_min_deg",
    "random_crook_theta_max_deg",
    "random_crook_abs_max",
    "random_taper_max",
    "manual_crook_amplitudes",
    "manual_crook_shifts_mm",
    "manual_crook_thetas_deg",
    "manual_crook_orders",
    "manual_crook_x_coeff",
    "manual_crook_y_coeff",
    "manual_taper_coeff",
    "include_knot_dev",
    "dead_knots",
    "knot_inside_limit",
    "knot_generator_min_rd_minus_rl_mm",
    "soft_clamp_alpha",
    "soft_clamp_pmin",
    "L100_min",
    "L100_max",
    "knot_sequence_top_k",
    "knot_sequence_top_p",
    "knot_sequence_min_tokens",
    "knot_sequence_extra_tokens",
    "knot_sequence_checkpoint_path",
    "knot_sequence_training_data_path",
    "knot_sequence_allow_fallback",
    "knot_sequence_reject_intersections",
    "knot_sequence_intersection_max_attempts",
    "knot_dictionary_jitter",
    "knot_sequence_override_c1_c2",
    # Fiber controls used for image export
    "calc_fibers_a0_method",
    "knot_fiber_field_override",
    "multi_knot_fiber_selection_rule",
    "multi_knot_fiber_selection_sigma",
    "multi_knot_fiber_selection_min_weight",
    "knot_fiber_disable_dead_override",
    "knot_fiber_reverse_above_axis",
    "rand_fibers",
    "out_of_plane_threshold",
    "snr",
}


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "yes", "on"}:
            return True
        if text in {"0", "false", "no", "off"}:
            return False
    return bool(value)


def _parse_float_list_arg(value: Any, *, flag_name: str) -> Optional[List[float]]:
    if value is None:
        return None
    if isinstance(value, (list, tuple, np.ndarray)):
        pieces = [str(v).strip() for v in value]
    else:
        text = str(value).strip()
        if not text:
            return None
        pieces = [p.strip() for p in text.split(",")]
    out: List[float] = []
    for idx, token in enumerate(pieces):
        if token == "":
            continue
        try:
            num = float(token)
        except (TypeError, ValueError):
            raise RuntimeError(
                f"Invalid float token '{token}' for {flag_name} at index {idx + 1}."
            )
        if not np.isfinite(num):
            raise RuntimeError(
                f"Non-finite value '{token}' for {flag_name} at index {idx + 1}."
            )
        out.append(float(num))
    return out


def _parse_int_list_arg(
    value: Any,
    *,
    flag_name: str,
    minimum: Optional[int] = None,
) -> Optional[List[int]]:
    if value is None:
        return None
    if isinstance(value, (list, tuple, np.ndarray)):
        pieces = [str(v).strip() for v in value]
    else:
        text = str(value).strip()
        if not text:
            return None
        pieces = [p.strip() for p in text.split(",")]
    out: List[int] = []
    for idx, token in enumerate(pieces):
        if token == "":
            continue
        try:
            raw_num = float(token)
        except (TypeError, ValueError):
            raise RuntimeError(
                f"Invalid integer token '{token}' for {flag_name} at index {idx + 1}."
            )
        if not np.isfinite(raw_num):
            raise RuntimeError(
                f"Non-finite value '{token}' for {flag_name} at index {idx + 1}."
            )
        num = int(round(raw_num))
        if not np.isclose(raw_num, float(num), atol=1e-9):
            raise RuntimeError(
                f"Invalid integer token '{token}' for {flag_name} at index {idx + 1}."
            )
        if minimum is not None and num < int(minimum):
            raise RuntimeError(
                f"Invalid value {num} for {flag_name} at index {idx + 1}; "
                f"minimum is {int(minimum)}."
            )
        out.append(int(num))
    return out


def _parse_float_or_range_arg(
    value: Any,
    *,
    flag_name: str,
    allow_none: bool = False,
    minimum: Optional[float] = None,
    maximum: Optional[float] = None,
) -> Optional[Tuple[float, float]]:
    if value is None:
        return None if allow_none else (0.0, 0.0)

    pieces: List[Any]
    if isinstance(value, (list, tuple, np.ndarray)):
        pieces = list(value)
    else:
        text = str(value).strip()
        if text == "":
            return None if allow_none else (0.0, 0.0)
        if "," in text:
            pieces = [p.strip() for p in text.split(",") if str(p).strip() != ""]
        else:
            pieces = [text]

    if len(pieces) == 1:
        pieces = [pieces[0], pieces[0]]
    elif len(pieces) != 2:
        raise RuntimeError(
            f"{flag_name} must be a single float or two comma-separated floats (a,b)."
        )

    try:
        lo = float(pieces[0])
        hi = float(pieces[1])
    except (TypeError, ValueError):
        raise RuntimeError(
            f"{flag_name} must be a single float or two comma-separated floats (a,b)."
        )

    if not (np.isfinite(lo) and np.isfinite(hi)):
        raise RuntimeError(f"{flag_name} contains non-finite values.")

    if lo > hi:
        lo, hi = hi, lo

    if minimum is not None:
        lo = max(float(minimum), lo)
        hi = max(float(minimum), hi)
    if maximum is not None:
        lo = min(float(maximum), lo)
        hi = min(float(maximum), hi)
    if lo > hi:
        raise RuntimeError(
            f"{flag_name} range is invalid after applying bounds "
            f"[{minimum if minimum is not None else '-inf'}, {maximum if maximum is not None else '+inf'}]."
        )
    return float(lo), float(hi)


def _parse_float_or_range_or_list_arg(
    value: Any,
    *,
    flag_name: str,
    allow_none: bool = False,
    minimum: Optional[float] = None,
    maximum: Optional[float] = None,
) -> Optional[Tuple[float, ...]]:
    if value is None:
        return None if allow_none else (0.0, 0.0)

    pieces: List[Any]
    if isinstance(value, (list, tuple, np.ndarray)):
        pieces = list(value)
    else:
        text = str(value).strip()
        if text == "":
            return None if allow_none else (0.0, 0.0)
        if "," in text:
            pieces = [p.strip() for p in text.split(",") if str(p).strip() != ""]
        else:
            pieces = [text]

    if len(pieces) <= 0:
        return None if allow_none else (0.0, 0.0)

    if len(pieces) <= 2:
        range_spec = _parse_float_or_range_arg(
            pieces,
            flag_name=flag_name,
            allow_none=allow_none,
            minimum=minimum,
            maximum=maximum,
        )
        return tuple(range_spec) if range_spec is not None else None

    values: List[float] = []
    for idx, token in enumerate(pieces):
        try:
            num = float(token)
        except (TypeError, ValueError):
            raise RuntimeError(
                f"{flag_name} must be a single float, range a,b, or list a,b,c,... "
                f"(invalid token '{token}' at index {idx + 1})."
            )
        if not np.isfinite(num):
            raise RuntimeError(
                f"{flag_name} contains a non-finite value at index {idx + 1}."
            )
        if minimum is not None:
            num = max(float(minimum), num)
        if maximum is not None:
            num = min(float(maximum), num)
        values.append(float(num))
    if len(values) < 3:
        raise RuntimeError(
            f"{flag_name} discrete list mode requires at least 3 values."
        )
    return tuple(values)


def _sample_chunk_value(
    spec: Optional[Tuple[float, float]],
    *,
    chunk_index: int,
    cache: Dict[int, float],
) -> Optional[float]:
    if spec is None:
        return None
    cached = cache.get(int(chunk_index))
    if cached is not None:
        return float(cached)
    lo, hi = spec
    if np.isclose(lo, hi):
        sampled = float(lo)
    else:
        sampled = float(np.random.uniform(lo, hi))
    cache[int(chunk_index)] = float(sampled)
    return float(sampled)


def _sample_chunk_value_from_range_or_list(
    spec: Optional[Tuple[float, ...]],
    *,
    chunk_index: int,
    cache: Dict[int, float],
) -> Optional[float]:
    if spec is None:
        return None
    cached = cache.get(int(chunk_index))
    if cached is not None:
        return float(cached)
    if len(spec) <= 2:
        sampled = _sample_chunk_value(
            (float(spec[0]), float(spec[1] if len(spec) > 1 else spec[0])),
            chunk_index=chunk_index,
            cache=cache,
        )
        return None if sampled is None else float(sampled)
    idx = int(np.random.randint(0, len(spec)))
    sampled = float(spec[idx])
    cache[int(chunk_index)] = float(sampled)
    return float(sampled)


def _is_range_spec(spec: Optional[Tuple[float, float]]) -> bool:
    if spec is None:
        return False
    return not bool(np.isclose(spec[0], spec[1]))


def _is_discrete_list_spec(spec: Optional[Tuple[float, ...]]) -> bool:
    return bool(spec is not None and len(spec) > 2)


def _serialize_range_spec(spec: Optional[Tuple[float, float]]) -> Any:
    if spec is None:
        return None
    lo, hi = float(spec[0]), float(spec[1])
    if np.isclose(lo, hi):
        return float(lo)
    return [lo, hi]


def _serialize_range_or_list_spec(spec: Optional[Tuple[float, ...]]) -> Any:
    if spec is None:
        return None
    if len(spec) <= 2:
        lo = float(spec[0])
        hi = float(spec[1] if len(spec) > 1 else spec[0])
        return _serialize_range_spec((lo, hi))
    return [float(v) for v in spec]


def _load_json_payload(path: str) -> Dict[str, Any]:
    text = str(path or "").strip()
    if not text:
        return {}
    p = Path(text)
    if not p.is_file():
        raise RuntimeError(f"JSON file not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected JSON object in {p}, got {type(payload).__name__}.")
    return dict(payload)


def _extract_board_config_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(payload.get("config"), dict):
        return dict(payload["config"])
    return dict(payload)


def _filter_relevant_board_config(cfg_data: Dict[str, Any]) -> tuple[Dict[str, Any], List[str]]:
    filtered: Dict[str, Any] = {}
    ignored: List[str] = []
    for key, value in cfg_data.items():
        if str(key) in _RELEVANT_CONFIG_KEYS:
            filtered[str(key)] = value
        else:
            ignored.append(str(key))
    return filtered, sorted(ignored)


def _extract_manual_knots_from_payload(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    candidates: Any = None
    if isinstance(payload.get("manual_knots"), list):
        candidates = payload.get("manual_knots")
    elif isinstance(payload.get("config"), dict) and isinstance(payload["config"].get("input_knots"), list):
        candidates = payload["config"].get("input_knots")
    elif isinstance(payload.get("input_knots"), list):
        candidates = payload.get("input_knots")
    else:
        return []

    out: List[Dict[str, Any]] = []
    for idx, item in enumerate(candidates or []):
        if not isinstance(item, dict):
            raise RuntimeError(f"Manual knot #{idx + 1} in --config-json is not a JSON object.")
        out.append(dict(item))
    return out


def _has_explicit_manual_knots_payload(payload: Dict[str, Any]) -> bool:
    if isinstance(payload.get("manual_knots"), list):
        return True
    if isinstance(payload.get("input_knots"), list):
        return True
    if isinstance(payload.get("config"), dict) and isinstance(payload["config"].get("input_knots"), list):
        return True
    return False


def _load_manual_knots(path: str) -> List[Dict[str, Any]]:
    text = str(path or "").strip()
    if not text:
        return []
    p = Path(text)
    if not p.is_file():
        raise RuntimeError(f"Manual knots JSON file not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload, dict) and isinstance(payload.get("input_knots"), list):
        knots = payload.get("input_knots")
    elif isinstance(payload, list):
        knots = payload
    else:
        raise RuntimeError(
            f"Manual knots JSON must be a list or object with 'input_knots' list: {p}"
        )

    out: List[Dict[str, Any]] = []
    for idx, item in enumerate(knots):
        if not isinstance(item, dict):
            raise RuntimeError(f"Manual knot #{idx + 1} in {p} is not a JSON object.")
        out.append(dict(item))
    return out


def _parse_outputs(raw: str) -> Set[str]:
    text = str(raw or "").strip().lower()
    if not text:
        return {"rings", "fibers", "middle", "top_bottom"}

    aliases = {
        "rings": "rings",
        "rings4": "rings",
        "ring4": "rings",
        "fibers": "fibers",
        "fibers4": "fibers",
        "fiber4": "fibers",
        "middle": "middle",
        "middle_ring": "middle",
        "middle_surface": "middle",
        "top_bottom": "top_bottom",
        "top_bottom_rings": "top_bottom",
        "cross_sections": "top_bottom",
        "photorealistic": "photorealistic",
        "photo": "photorealistic",
    }

    tokens = [t.strip() for t in text.split(",") if t.strip()]
    if any(t == "all" for t in tokens):
        return set(_ALLOWED_OUTPUTS)
    if any(t == "none" for t in tokens):
        return set()

    out: Set[str] = set()
    for token in tokens:
        mapped = aliases.get(token)
        if mapped is None:
            raise RuntimeError(
                "Unsupported output token "
                f"'{token}'. Allowed: rings, fibers, middle, top_bottom, photorealistic, all."
            )
        out.add(mapped)
    unknown = out - _ALLOWED_OUTPUTS
    if unknown:
        raise RuntimeError(f"Unsupported outputs requested: {sorted(unknown)}")
    return out


def _build_image_stem_map(folder: Path) -> Dict[str, Path]:
    if not folder.is_dir():
        raise RuntimeError(f"Missing image folder: {folder}")
    mapping: Dict[str, Path] = {}
    for entry in folder.iterdir():
        if not entry.is_file():
            continue
        if entry.suffix.lower() not in _IMG_EXTS:
            continue
        if entry.stem not in mapping:
            mapping[entry.stem] = entry
    if not mapping:
        raise RuntimeError(f"No images found in folder: {folder}")
    return mapping


def _parse_stems_csv(raw: Any) -> List[str]:
    text = str(raw or "").strip()
    if not text:
        return []
    return [token.strip() for token in text.split(",") if token.strip()]


def _model_dump(cfg: BoardConfig) -> Dict[str, Any]:
    if hasattr(cfg, "model_dump"):
        return cfg.model_dump()
    return cfg.dict()


def _write_png(path: Path, png_bytes: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(png_bytes)


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def _build_top_bottom_ring_pngs(
    contours_mat: Sequence[Any],
    board_dims: Dict[str, Any],
    *,
    size: int,
    line_width: float,
) -> Dict[str, bytes]:
    x0 = float(board_dims.get("x_min", 0.0))
    x1 = float(board_dims.get("x_max", 1.0))
    y0 = float(board_dims.get("y_min", 0.0))
    y1 = float(board_dims.get("y_max", 1.0))
    z0 = float(board_dims.get("z_min", 0.0))
    z1 = float(board_dims.get("z_max", 1.0))
    z_mid = 0.5 * (z0 + z1)

    by_face: Dict[str, List[np.ndarray]] = {"z_min": [], "z_max": []}
    for line in contours_mat:
        arr = np.asarray(line, dtype=float)
        if arr.ndim != 2 or arr.shape[1] != 3 or arr.shape[0] < 2:
            continue
        mins = arr.min(axis=0)
        maxs = arr.max(axis=0)
        fixed_axis = int(np.argmin(maxs - mins))
        if fixed_axis != 2:
            continue
        side = "z_min" if float(np.mean(arr[:, 2])) <= z_mid else "z_max"
        by_face[side].append(arr)

    meta = {
        "z_min": {"u_axis": 0, "v_axis": 1, "u_min": x0, "u_max": x1, "v_min": y0, "v_max": y1, "flip_x": False},
        "z_max": {"u_axis": 0, "v_axis": 1, "u_min": x0, "u_max": x1, "v_min": y0, "v_max": y1, "flip_x": False},
    }

    raw_bottom = _render_surface_png_matlab(
        by_face["z_min"],
        meta["z_min"],
        size=int(size),
        line_width=float(line_width),
    )
    raw_top = _render_surface_png_matlab(
        by_face["z_max"],
        meta["z_max"],
        size=int(size),
        line_width=float(line_width),
    )
    return {
        # Keep top/bottom cross-sections as direct contour rasters; skip the
        # synthetic blur/irregularity used for side-ring image exports.
        "rings_bottom": _flip_png_vertical_bytes(raw_bottom),
        "rings_top": _flip_png_vertical_bytes(raw_top),
    }


@dataclass
class _PhotorealisticJob:
    filename: str
    ring_paths: Dict[str, Path]
    fiber_paths: Dict[str, Path]
    temp_input_dir: Optional[Path]
    metadata_path: Path


@dataclass
class _PlacementPolicy:
    mode: str  # "extents" or "dimensions"
    board_width: float
    board_thickness: float
    board_length: float
    randomize_crook_taper: bool
    crook_component_count: int
    crook_shift_max_mm: float
    random_crook_theta_min_deg: float
    random_crook_theta_max_deg: float
    random_crook_scale_max: float
    random_taper_max: float


class _RetryablePlacementError(RuntimeError):
    """Per-attempt placement failure that should count as rejection, not crash generation."""


def _detect_cuda_device_count() -> int:
    count = 0
    try:
        import torch  # type: ignore
    except Exception:
        torch = None
    if torch is not None:
        try:
            if bool(torch.cuda.is_available()):
                count = max(count, int(torch.cuda.device_count()))
        except Exception:
            pass
    try:
        import cupy as cp  # type: ignore
        count = max(count, int(cp.cuda.runtime.getDeviceCount()))
    except Exception:
        pass
    return max(0, int(count))


def _set_current_cuda_device(device_index: int) -> None:
    idx = int(device_index)
    if idx < 0:
        return
    try:
        import torch  # type: ignore
        if bool(torch.cuda.is_available()):
            n = int(torch.cuda.device_count())
            if n > 0:
                torch.cuda.set_device(int(idx % n))
    except Exception:
        pass
    try:
        import cupy as cp  # type: ignore
        n = int(cp.cuda.runtime.getDeviceCount())
        if n > 0:
            cp.cuda.Device(int(idx % n)).use()
    except Exception:
        pass


def _board_generate_worker_entry(payload: Dict[str, Any], result_queue: Any) -> None:
    worker_id = int(payload.get("worker_id", 0))
    try:
        args_ns = SimpleNamespace(**dict(payload.get("args_dict") or {}))
        _set_current_cuda_device(int(getattr(args_ns, "_multi_gpu_device_index", -1)))
        generate_boards_dataset(args_ns)
        result_queue.put({"ok": True, "worker_id": worker_id})
    except Exception as exc:
        result_queue.put(
            {
                "ok": False,
                "worker_id": worker_id,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )


def _run_multi_gpu_board_generation(
    args: Any,
    *,
    root: Path,
    num_boards: int,
    imid_start: int,
    max_attempts: int,
    worker_count: int,
    device_count: int,
) -> Dict[str, Any]:
    n_workers = max(1, min(int(worker_count), int(num_boards)))
    base = int(num_boards // n_workers)
    rem = int(num_boards % n_workers)

    worker_specs: List[Dict[str, Any]] = []
    next_imid = int(imid_start)
    for worker_id in range(n_workers):
        shard_boards = int(base + (1 if worker_id < rem else 0))
        if shard_boards <= 0:
            continue
        shard_attempts = int(
            max(
                shard_boards,
                int(np.ceil((float(max_attempts) * float(shard_boards)) / float(max(1, num_boards)))),
            )
        )
        manifest_path = root / f"manifest.worker_{worker_id}.json"
        spec = {
            "worker_id": int(worker_id),
            "device_index": int(worker_id % max(1, device_count)),
            "num_boards": int(shard_boards),
            "imid_start": int(next_imid),
            "max_attempts": int(shard_attempts),
            "manifest_path": manifest_path,
        }
        worker_specs.append(spec)
        next_imid += shard_boards

    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    processes: List[Any] = []
    for spec in worker_specs:
        args_dict = dict(vars(args))
        args_dict["num_boards"] = int(spec["num_boards"])
        args_dict["imid_start"] = int(spec["imid_start"])
        args_dict["max_attempts"] = int(spec["max_attempts"])
        args_dict["output_dir"] = str(root)
        args_dict["gpu_workers"] = 1
        args_dict["_multi_gpu_worker"] = True
        args_dict["_multi_gpu_device_index"] = int(spec["device_index"])
        args_dict["_multi_gpu_seed_offset"] = int(spec["worker_id"]) * 1_000_000
        args_dict["_multi_gpu_manifest_path"] = str(spec["manifest_path"])
        args_dict["_multi_gpu_worker_id"] = int(spec["worker_id"])
        args_dict["_multi_gpu_worker_count"] = int(len(worker_specs))
        payload = {"worker_id": int(spec["worker_id"]), "args_dict": args_dict}
        proc = ctx.Process(
            target=_board_generate_worker_entry,
            args=(payload, result_queue),
            daemon=False,
        )
        proc.start()
        processes.append(proc)

    for proc in processes:
        proc.join()

    worker_results: Dict[int, Dict[str, Any]] = {}
    for _ in processes:
        try:
            msg = result_queue.get(timeout=5.0)
        except Exception:
            continue
        if isinstance(msg, dict):
            worker_results[int(msg.get("worker_id", -1))] = msg

    failures: List[str] = []
    for proc, spec in zip(processes, worker_specs):
        wid = int(spec["worker_id"])
        result_msg = worker_results.get(wid, {})
        if proc.exitcode != 0:
            tb = str(result_msg.get("traceback") or "").strip()
            failures.append(
                f"worker {wid} failed (exitcode={proc.exitcode})."
                + (f"\n{tb}" if tb else "")
            )
        elif not bool(result_msg.get("ok", False)):
            err = str(result_msg.get("error") or "unknown worker error")
            tb = str(result_msg.get("traceback") or "").strip()
            failures.append(
                f"worker {wid} failed: {err}"
                + (f"\n{tb}" if tb else "")
            )

    if failures:
        raise RuntimeError(
            "Multi-GPU board generation failed.\n"
            + "\n\n".join(failures)
        )

    worker_manifests: List[Dict[str, Any]] = []
    for spec in worker_specs:
        path = Path(spec["manifest_path"])
        if not path.is_file():
            raise RuntimeError(f"Missing worker manifest: {path}")
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise RuntimeError(f"Failed to parse worker manifest {path}: {exc}") from exc
        if not isinstance(data, dict):
            raise RuntimeError(f"Invalid worker manifest format: {path}")
        worker_manifests.append(data)

    if not worker_manifests:
        raise RuntimeError("No worker manifests were produced.")

    aggregate = dict(worker_manifests[0])
    aggregate["created_at_utc"] = datetime.now(timezone.utc).isoformat()
    aggregate["num_boards_requested"] = int(num_boards)
    aggregate["num_boards_generated"] = int(
        sum(int(m.get("num_boards_generated", 0) or 0) for m in worker_manifests)
    )
    aggregate["max_attempts"] = int(
        sum(int(m.get("max_attempts", 0) or 0) for m in worker_manifests)
    )
    aggregate["num_rejected_outside_log"] = int(
        sum(int(m.get("num_rejected_outside_log", 0) or 0) for m in worker_manifests)
    )
    aggregate["num_rejected_low_knot_count"] = int(
        sum(int(m.get("num_rejected_low_knot_count", 0) or 0) for m in worker_manifests)
    )
    generated_filenames: List[str] = []
    for m in worker_manifests:
        generated_filenames.extend([str(v) for v in (m.get("generated_filenames") or [])])
    aggregate["generated_filenames"] = sorted(set(generated_filenames))
    aggregate["multi_gpu"] = {
        "enabled": True,
        "gpu_workers": int(len(worker_specs)),
        "cuda_device_count_visible": int(device_count),
        "workers": [
            {
                "worker_id": int(spec["worker_id"]),
                "device_index": int(spec["device_index"]),
                "num_boards_requested": int(spec["num_boards"]),
                "imid_start": int(spec["imid_start"]),
                "max_attempts": int(spec["max_attempts"]),
            }
            for spec in worker_specs
        ],
    }

    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    for spec in worker_specs:
        try:
            Path(spec["manifest_path"]).unlink(missing_ok=True)
        except Exception:
            pass
    print(
        f"[board-cli] done (multi-gpu). generated={aggregate['num_boards_generated']}, "
        f"rejected_outside_log={aggregate['num_rejected_outside_log']}, "
        f"rejected_low_knot_count={aggregate['num_rejected_low_knot_count']}, "
        f"manifest={manifest_path}"
    )
    return aggregate


def _sample_center_within_log(
    splines: Sequence[Any],
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
            "Board dimensions are too large to fit inside the selected log cross-section. "
            f"min_log_radius={min_radius:.2f} mm, board_half_diagonal={half_diag:.2f} mm."
        )

    if max_center_radius <= 1e-9:
        return 0.0, 0.0

    # Uniform sample over a disk to avoid clustering near the center.
    radius = max_center_radius * float(np.sqrt(np.random.random()))
    angle = 2.0 * float(np.pi) * float(np.random.random())
    return radius * float(np.cos(angle)), radius * float(np.sin(angle))


def _apply_randomized_crook_taper(cfg: BoardConfig, policy: _PlacementPolicy) -> None:
    cfg.randomize_crook_taper = bool(policy.randomize_crook_taper)
    cfg.crook_component_count = int(max(1, policy.crook_component_count))
    cfg.crook_shift_max_mm = float(max(0.0, policy.crook_shift_max_mm))
    cfg.random_crook_theta_min_deg = float(policy.random_crook_theta_min_deg)
    cfg.random_crook_theta_max_deg = float(policy.random_crook_theta_max_deg)
    cfg.random_crook_scale_max = float(max(0.0, policy.random_crook_scale_max))
    cfg.random_taper_max = float(max(0.0, policy.random_taper_max))


def _apply_random_board_extents_for_dimensions_mode(
    cfg: BoardConfig,
    k: KnotSystem,
    policy: _PlacementPolicy,
) -> None:
    cx, cy = _sample_center_within_log(
        getattr(k, "splines", []) or [],
        width=float(policy.board_width),
        thickness=float(policy.board_thickness),
    )
    half_w = 0.5 * float(policy.board_width)
    half_t = 0.5 * float(policy.board_thickness)

    cfg.board_x_min = float(cx - half_w)
    cfg.board_x_max = float(cx + half_w)
    cfg.board_y_min = float(cy - half_t)
    cfg.board_y_max = float(cy + half_t)


def _normalize_generation_config(
    args: Any,
    outputs: Set[str],
    *,
    config_payload: Dict[str, Any],
    random_crook_scale_max: float,
    random_taper_max: float,
) -> tuple[BoardConfig, _PlacementPolicy]:
    cfg_data = _extract_board_config_payload(config_payload)
    cfg_data, ignored_keys = _filter_relevant_board_config(cfg_data)
    if (
        "random_crook_scale_max" not in cfg_data
        and cfg_data.get("random_crook_abs_max") is not None
    ):
        cfg_data["random_crook_scale_max"] = cfg_data.get("random_crook_abs_max")
    if ignored_keys:
        preview = ", ".join(ignored_keys[:12])
        suffix = "" if len(ignored_keys) <= 12 else f", ... (+{len(ignored_keys) - 12} more)"
        print(
            "[board-cli] note: ignoring config fields not relevant to image generation: "
            f"{preview}{suffix}"
        )
    manual_knots_path = str(args.manual_knots_json or "").strip()
    manual_knots = _load_manual_knots(manual_knots_path)
    manual_knots_explicit = bool(manual_knots_path)
    if not manual_knots_explicit and not manual_knots:
        manual_knots = _extract_manual_knots_from_payload(config_payload)
        manual_knots_explicit = _has_explicit_manual_knots_payload(config_payload)
    if manual_knots_explicit or manual_knots:
        cfg_data["input_knots"] = manual_knots
        cfg_data["input_knot_count"] = len(manual_knots)
        cfg_data["use_input_knots"] = True

    overrides: Dict[str, Any] = {}

    def ov(name: str, value: Any) -> None:
        if value is not None:
            overrides[name] = value

    def explicit_in_cfg(name: str) -> bool:
        return name in cfg_data and cfg_data.get(name) is not None

    extent_cli = {
        "board_x_min": args.board_x_min,
        "board_x_max": args.board_x_max,
        "board_y_min": args.board_y_min,
        "board_y_max": args.board_y_max,
        "board_z_min": args.board_z_min,
        "board_z_max": args.board_z_max,
    }
    dim_cli = {
        "board_width": args.board_width_mm,
        "board_thickness": args.board_thickness_mm,
        "board_length": args.board_length_mm,
    }
    extent_explicit_keys = {k for k in _EXTENT_KEYS if explicit_in_cfg(k)}
    extent_explicit_keys.update(k for k, v in extent_cli.items() if v is not None)
    dim_explicit_keys = {k for k in _DIMENSION_KEYS if explicit_in_cfg(k)}
    dim_explicit_keys.update(k for k, v in dim_cli.items() if v is not None)

    has_extents = bool(extent_explicit_keys)
    has_dimensions = bool(dim_explicit_keys)
    if has_extents and has_dimensions:
        raise RuntimeError(
            "Define either board extents or board dimensions, not both. "
            "Extents: board_x_min/max, board_y_min/max, board_z_min/max. "
            "Dimensions: board_width/board_thickness/board_length (or --board-*-mm flags)."
        )
    if not has_extents and not has_dimensions:
        raise RuntimeError(
            "Missing board definition. Define either board extents "
            "(board_x_min/max, board_y_min/max, board_z_min/max) "
            "or board dimensions (board_width/board_thickness/board_length or --board-*-mm flags)."
        )

    if has_extents and len(extent_explicit_keys) != len(_EXTENT_KEYS):
        missing = sorted(set(_EXTENT_KEYS) - extent_explicit_keys)
        raise RuntimeError(
            "Extent mode requires all six extent values. "
            f"Missing: {', '.join(missing)}."
        )
    if has_dimensions and len(dim_explicit_keys) != len(_DIMENSION_KEYS):
        missing = sorted(set(_DIMENSION_KEYS) - dim_explicit_keys)
        raise RuntimeError(
            "Dimension mode requires all three dimensions. "
            f"Missing: {', '.join(missing)}."
        )

    ov("board_x_min", args.board_x_min)
    ov("board_x_max", args.board_x_max)
    ov("board_y_min", args.board_y_min)
    ov("board_y_max", args.board_y_max)
    ov("board_z_min", args.board_z_min)
    ov("board_z_max", args.board_z_max)
    ov("board_width", args.board_width_mm)
    ov("board_thickness", args.board_thickness_mm)
    ov("board_length", args.board_length_mm)
    ov("mesh_size_x_mm", args.mesh_size_x_mm)
    ov("mesh_size_y_mm", args.mesh_size_y_mm)
    ov("mesh_size_z_mm", args.mesh_size_z_mm)
    ov("crook_component_count", args.crook_component_count)
    ov("crook_shift_max_mm", args.crook_shift_max_mm)
    ov("random_crook_scale_max", args.random_crook_scale_max)
    random_crook_amplitude_max = _parse_float_list_arg(
        args.random_crook_amplitudes_max,
        flag_name="--random-crook-amplitudes-max",
    )
    if random_crook_amplitude_max is not None:
        overrides["random_crook_amplitude_max"] = random_crook_amplitude_max
    random_crook_extra_orders = _parse_int_list_arg(
        args.random_crook_extra_orders,
        flag_name="--random-crook-extra-orders",
        minimum=1,
    )
    if random_crook_extra_orders is not None:
        overrides["random_crook_extra_orders"] = random_crook_extra_orders
    ov("random_crook_theta_min_deg", args.random_crook_theta_min_deg)
    ov("random_crook_theta_max_deg", args.random_crook_theta_max_deg)
    ov("random_taper_max", args.random_taper_max)
    ov("manual_crook_x_coeff", args.manual_crook_x_coeff)
    ov("manual_crook_y_coeff", args.manual_crook_y_coeff)
    ov("manual_taper_coeff", args.manual_taper_coeff)
    manual_crook_amplitudes = _parse_float_list_arg(
        args.manual_crook_amplitudes,
        flag_name="--manual-crook-amplitudes",
    )
    manual_crook_shifts_mm = _parse_float_list_arg(
        args.manual_crook_shifts_mm,
        flag_name="--manual-crook-shifts-mm",
    )
    manual_crook_thetas_deg = _parse_float_list_arg(
        args.manual_crook_thetas_deg,
        flag_name="--manual-crook-thetas-deg",
    )
    manual_crook_orders = _parse_int_list_arg(
        args.manual_crook_orders,
        flag_name="--manual-crook-orders",
        minimum=1,
    )
    if manual_crook_amplitudes is not None:
        overrides["manual_crook_amplitudes"] = manual_crook_amplitudes
    if manual_crook_shifts_mm is not None:
        overrides["manual_crook_shifts_mm"] = manual_crook_shifts_mm
    if manual_crook_thetas_deg is not None:
        overrides["manual_crook_thetas_deg"] = manual_crook_thetas_deg
    if manual_crook_orders is not None:
        overrides["manual_crook_orders"] = manual_crook_orders
    ov("knot_inside_limit", args.knot_inside_limit)
    ov("knot_generator_min_rd_minus_rl_mm", args.knot_generator_min_rd_minus_rl_mm)
    ov("L100_min", args.l100_min)
    ov("L100_max", args.l100_max)
    ov("knot_sequence_top_k", args.knot_seq_top_k)
    ov("knot_sequence_top_p", args.knot_seq_top_p)
    ov("knot_sequence_min_tokens", args.knot_seq_min_tokens)
    ov("knot_sequence_extra_tokens", args.knot_seq_extra_tokens)
    ov("knot_sequence_checkpoint_path", args.knot_seq_checkpoint_path)
    ov("knot_sequence_training_data_path", args.knot_seq_training_data_path)
    ov("knot_dictionary_jitter", args.knot_dictionary_jitter)
    ov("soft_clamp_alpha", args.soft_clamp_alpha)
    ov("soft_clamp_pmin", args.soft_clamp_pmin)
    ov("out_of_plane_threshold", args.out_of_plane_threshold)
    ov("snr", args.snr)
    ov("calc_fibers_a0_method", args.calc_fibers_a0_method)
    ov("multi_knot_fiber_selection_rule", args.multi_knot_fiber_selection_rule)
    ov("input_knot_count", args.input_knot_count)
    ov("simulation_seed", args.simulation_seed)

    bool_overrides = {
        "use_gpu": args.use_gpu,
        "use_seed": args.use_seed,
        "use_input_knots": args.use_input_knots,
        "include_knot_dev": args.include_knot_dev,
        "dead_knots": args.dead_knots,
        "randomize_crook_taper": args.randomize_crook_taper,
        "knot_fiber_field_override": args.knot_fiber_field_override,
        "knot_fiber_disable_dead_override": args.knot_fiber_disable_dead_override,
        "knot_fiber_reverse_above_axis": args.knot_fiber_reverse_above_axis,
        "rand_fibers": args.rand_fibers,
        "knot_sequence_allow_fallback": args.knot_seq_allow_fallback,
        "knot_sequence_override_c1_c2": args.knot_seq_override_c1_c2,
    }
    for key, value in bool_overrides.items():
        if value is not None:
            overrides[key] = _as_bool(value)

    cfg_data.update(overrides)
    cfg_data["board_or_log"] = 0

    needs_contours = bool(outputs & {"rings", "middle", "top_bottom", "photorealistic"})
    needs_fibers = bool(outputs & {"fibers", "photorealistic"})

    cfg = BoardConfig(**cfg_data)
    cfg.board_or_log = 0
    cfg.display_contours = bool(needs_contours)
    cfg.calc_fibers = bool(needs_fibers)
    cfg.crook_component_count = max(1, int(cfg.crook_component_count))
    cfg.crook_shift_max_mm = max(0.0, float(cfg.crook_shift_max_mm))
    cfg.random_crook_scale_max = max(0.0, float(cfg.random_crook_scale_max))
    cfg.random_taper_max = max(0.0, float(cfg.random_taper_max))
    cfg.knot_generator_min_rd_minus_rl_mm = max(0.0, float(cfg.knot_generator_min_rd_minus_rl_mm))
    cfg.random_crook_amplitude_max = [
        max(0.0, float(v))
        for v in _parse_float_list_arg(
            cfg.random_crook_amplitude_max,
            flag_name="config.random_crook_amplitude_max",
        ) or []
    ]
    cfg.random_crook_extra_orders = _parse_int_list_arg(
        cfg.random_crook_extra_orders,
        flag_name="config.random_crook_extra_orders",
        minimum=1,
    ) or []
    cfg.manual_crook_orders = _parse_int_list_arg(
        cfg.manual_crook_orders,
        flag_name="config.manual_crook_orders",
        minimum=1,
    ) or []
    theta_min = float(cfg.random_crook_theta_min_deg)
    theta_max = float(cfg.random_crook_theta_max_deg)
    if not (np.isfinite(theta_min) and np.isfinite(theta_max)):
        raise RuntimeError("Crook theta bounds must be finite numbers.")
    if theta_max < theta_min:
        theta_min, theta_max = theta_max, theta_min
    cfg.random_crook_theta_min_deg = theta_min
    cfg.random_crook_theta_max_deg = theta_max

    if has_dimensions:
        if float(cfg.board_width) <= 0.0 or float(cfg.board_thickness) <= 0.0 or float(cfg.board_length) <= 0.0:
            raise RuntimeError(
                "Dimension mode requires positive board dimensions: "
                "board_width > 0, board_thickness > 0, board_length > 0."
            )

    if not (np.isfinite(random_crook_scale_max) and random_crook_scale_max >= 0.0):
        raise RuntimeError("--random-crook-scale-max must be a finite non-negative number.")
    if not (np.isfinite(random_taper_max) and random_taper_max >= 0.0):
        raise RuntimeError("--random-taper-max must be a finite non-negative number.")
    cfg.random_crook_scale_max = float(random_crook_scale_max)
    cfg.random_taper_max = float(random_taper_max)

    policy = _PlacementPolicy(
        mode="extents" if has_extents else "dimensions",
        board_width=float(max(1e-6, cfg.board_width)),
        board_thickness=float(max(1e-6, cfg.board_thickness)),
        board_length=float(max(1e-6, cfg.board_length)),
        randomize_crook_taper=bool(cfg.randomize_crook_taper),
        crook_component_count=int(cfg.crook_component_count),
        crook_shift_max_mm=float(cfg.crook_shift_max_mm),
        random_crook_theta_min_deg=float(cfg.random_crook_theta_min_deg),
        random_crook_theta_max_deg=float(cfg.random_crook_theta_max_deg),
        random_crook_scale_max=float(random_crook_scale_max),
        random_taper_max=float(random_taper_max),
    )
    return cfg, policy


def generate_boards_dataset(args: Any) -> Dict[str, Any]:
    config_payload = _load_json_payload(str(args.config_json or ""))
    board_gen_cfg = (
        dict(config_payload.get("boards_generate"))
        if isinstance(config_payload.get("boards_generate"), dict)
        else {}
    )
    board_cfg_payload = _extract_board_config_payload(config_payload)

    def resolve(name: str, cli_value: Any, default: Any) -> Any:
        if cli_value is not None:
            return cli_value
        if name in board_gen_cfg and board_gen_cfg.get(name) is not None:
            return board_gen_cfg.get(name)
        # Accept both snake_case and kebab-case keys from config JSON.
        alt_name = name.replace("_", "-")
        if alt_name in board_gen_cfg and board_gen_cfg.get(alt_name) is not None:
            return board_gen_cfg.get(alt_name)
        return default

    output_dir_raw = str(resolve("output_dir", args.output_dir, "") or "").strip()
    if not output_dir_raw:
        raise RuntimeError(
            "Missing output directory. Set --output-dir or boards_generate.output_dir in --config-json."
        )

    outputs = _parse_outputs(str(resolve("outputs", args.outputs, "rings,fibers,middle,top_bottom")))
    if not outputs:
        raise RuntimeError("No outputs requested. Set --outputs to at least one output type.")

    num_boards = max(1, int(resolve("num_boards", args.num_boards, 100)))
    min_knot_count = max(0, int(resolve("min_knot_count", args.min_knot_count, 0)))
    image_size = max(32, int(resolve("image_size", args.image_size, 512)))
    imid_start = max(0, int(resolve("imid_start", args.imid_start, 1)))
    contour_line_width_spec = _parse_float_or_range_arg(
        resolve("contour_line_width", args.contour_line_width, 1.0),
        flag_name="contour_line_width",
        allow_none=False,
        minimum=1.0,
    )
    contour_blur_sigma = max(0.0, float(resolve("contour_blur_sigma", args.contour_blur_sigma, 0.0)))
    fiber_blur_raw = resolve(
        "fiber_blur_sigma",
        args.fiber_blur_sigma,
        board_gen_cfg.get("fiber_blur_segma", 0.0),
    )
    fiber_blur_spec = _parse_float_or_range_arg(
        fiber_blur_raw,
        flag_name="fiber_blur_sigma",
        allow_none=False,
        minimum=0.0,
    )
    fiber_irregularity_spec = _parse_float_or_range_arg(
        resolve("fiber_irregularity_strength", args.fiber_irregularity_strength, 0.35),
        flag_name="fiber_irregularity_strength",
        allow_none=False,
        minimum=0.0,
        maximum=2.0,
    )
    ring_irregularity_spec = _parse_float_or_range_arg(
        resolve("ring_irregularity_strength", args.ring_irregularity_strength, 0.40),
        flag_name="ring_irregularity_strength",
        allow_none=False,
        minimum=0.0,
        maximum=2.0,
    )
    show_inside = _as_bool(resolve("show_rings_inside_knots", args.show_rings_inside_knots, False))

    photo_steps = resolve("photorealistic_ddim_steps", args.photorealistic_ddim_steps, None)
    photo_guidance_spec = _parse_float_or_range_arg(
        resolve("photorealistic_guidance_scale", args.photorealistic_guidance_scale, None),
        flag_name="photorealistic_guidance_scale",
        allow_none=True,
    )
    photo_img2img_spec = _parse_float_or_range_or_list_arg(
        resolve("photorealistic_img2img_strength", args.photorealistic_img2img_strength, None),
        flag_name="photorealistic_img2img_strength",
        allow_none=True,
        minimum=0.0,
        maximum=1.0,
    )
    photo_batch = max(1, int(resolve("photorealistic_batch_size", args.photorealistic_batch_size, 4)))
    photo_include_knot_maps = _as_bool(
        resolve(
            "photorealistic_include_knot_maps",
            getattr(args, "photorealistic_include_knot_maps", None),
            False,
        )
    )
    photo_use_rings_only = _as_bool(
        resolve(
            "photorealistic_use_rings_only",
            getattr(args, "photorealistic_use_rings_only", None),
            False,
        )
    )
    if bool(photo_use_rings_only) and bool(photo_include_knot_maps):
        raise RuntimeError(
            "photorealistic_use_rings_only=true cannot be combined with "
            "photorealistic_include_knot_maps=true."
        )
    default_cfg = BoardConfig()
    cfg_crook_scale_default = board_cfg_payload.get(
        "random_crook_scale_max",
        board_cfg_payload.get("random_crook_abs_max", float(default_cfg.random_crook_scale_max)),
    )
    cfg_taper_max_default = board_cfg_payload.get("random_taper_max", float(default_cfg.random_taper_max))
    random_crook_scale_max = max(
        0.0,
        float(
            resolve(
                "random_crook_scale_max",
                args.random_crook_scale_max,
                resolve("random_crook_abs_max", args.random_crook_scale_max, cfg_crook_scale_default),
            )
        ),
    )
    random_taper_max = max(
        0.0,
        float(resolve("random_taper_max", args.random_taper_max, cfg_taper_max_default)),
    )

    max_attempts = int(resolve("max_attempts", args.max_attempts, 0))
    if max_attempts <= 0:
        max_attempts = max(num_boards * 20, num_boards)
    if max_attempts < num_boards:
        raise RuntimeError("--max-attempts must be >= --num-boards.")

    root = Path(output_dir_raw).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    metadata_dir = root / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    base_cfg, placement_policy = _normalize_generation_config(
        args,
        outputs,
        config_payload=config_payload,
        random_crook_scale_max=random_crook_scale_max,
        random_taper_max=random_taper_max,
    )
    base_cfg.calc_fibers = bool(("fibers" in outputs) or ("photorealistic" in outputs and not bool(photo_use_rings_only)))
    is_multi_gpu_worker = bool(getattr(args, "_multi_gpu_worker", False))
    worker_id = int(getattr(args, "_multi_gpu_worker_id", 0) or 0)
    worker_count_hint = int(getattr(args, "_multi_gpu_worker_count", 1) or 1)
    worker_seed_offset = int(getattr(args, "_multi_gpu_seed_offset", 0) or 0)
    worker_device_index = int(getattr(args, "_multi_gpu_device_index", -1) or -1)
    if bool(base_cfg.use_gpu) and worker_device_index >= 0:
        _set_current_cuda_device(worker_device_index)

    gpu_workers_raw = resolve("gpu_workers", getattr(args, "gpu_workers", None), None)
    requested_gpu_workers = 0
    if gpu_workers_raw is not None:
        try:
            requested_gpu_workers = int(gpu_workers_raw)
        except (TypeError, ValueError):
            raise RuntimeError("gpu_workers must be an integer.")

    visible_cuda_devices = _detect_cuda_device_count() if bool(base_cfg.use_gpu) else 0
    if (
        not is_multi_gpu_worker
        and bool(base_cfg.use_gpu)
        and visible_cuda_devices > 1
        and int(num_boards) > 1
    ):
        target_workers = int(visible_cuda_devices if requested_gpu_workers <= 0 else requested_gpu_workers)
        target_workers = max(1, min(int(target_workers), int(visible_cuda_devices), int(num_boards)))
        if target_workers > 1:
            print(
                f"[board-cli] multi-gpu mode enabled: workers={target_workers}, "
                f"visible_cuda_devices={visible_cuda_devices}"
            )
            return _run_multi_gpu_board_generation(
                args,
                root=root,
                num_boards=int(num_boards),
                imid_start=int(imid_start),
                max_attempts=int(max_attempts),
                worker_count=int(target_workers),
                device_count=int(visible_cuda_devices),
            )

    worker_prefix = (
        f"[board-cli][worker {worker_id + 1}/{max(1, worker_count_hint)}] "
        if is_multi_gpu_worker
        else "[board-cli] "
    )
    print(
        f"{worker_prefix}generating boards: requested={num_boards}, max_attempts={max_attempts}, "
        f"outputs={','.join(sorted(outputs))}, output_dir={root}, "
        f"min_knot_count={min_knot_count}, "
        f"placement_mode={placement_policy.mode}"
    )
    if placement_policy.mode == "dimensions":
        print(
            "[board-cli] dimension mode: random board placement per attempt "
            f"(width={placement_policy.board_width:.2f} mm, "
            f"thickness={placement_policy.board_thickness:.2f} mm, "
            f"length={placement_policy.board_length:.2f} mm, "
            "fixed_z=[0.00, length] mm)"
        )
    if placement_policy.randomize_crook_taper:
        print(
            "[board-cli] random crook/taper enabled: "
            f"p={placement_policy.crook_component_count}, "
            f"shift_max_mm={placement_policy.crook_shift_max_mm:.6g}, "
            f"theta_deg=[{placement_policy.random_crook_theta_min_deg:.6g}, "
            f"{placement_policy.random_crook_theta_max_deg:.6g}], "
            f"crook_scale_max={placement_policy.random_crook_scale_max:.6g}, "
            f"taper_max={placement_policy.random_taper_max:.6g}"
        )
    else:
        print(
            "[board-cli] random crook/taper disabled: using manual component values "
            f"(p={placement_policy.crook_component_count}) and manual_taper_coeff."
        )
    if _is_range_spec(fiber_blur_spec):
        print(
            "[board-cli] chunk-randomized fiber blur enabled: "
            f"fiber_blur_sigma in [{fiber_blur_spec[0]:.6g}, {fiber_blur_spec[1]:.6g}] "
            f"sampled once per chunk (chunk_size={photo_batch})."
        )
    if _is_range_spec(fiber_irregularity_spec):
        print(
            "[board-cli] chunk-randomized fiber irregularity enabled: "
            f"fiber_irregularity_strength in [{fiber_irregularity_spec[0]:.6g}, {fiber_irregularity_spec[1]:.6g}] "
            f"sampled once per chunk (chunk_size={photo_batch})."
        )
    if _is_range_spec(ring_irregularity_spec):
        print(
            "[board-cli] chunk-randomized ring irregularity enabled: "
            f"ring_irregularity_strength in [{ring_irregularity_spec[0]:.6g}, {ring_irregularity_spec[1]:.6g}] "
            f"sampled once per chunk (chunk_size={photo_batch})."
        )
    if _is_range_spec(contour_line_width_spec):
        print(
            "[board-cli] chunk-randomized contour line width enabled: "
            f"contour_line_width in [{contour_line_width_spec[0]:.6g}, {contour_line_width_spec[1]:.6g}] "
            f"sampled once per chunk (chunk_size={photo_batch})."
        )
    if _is_range_spec(photo_guidance_spec):
        print(
            "[board-cli] chunk-randomized photorealistic guidance enabled: "
            f"guidance_scale in [{photo_guidance_spec[0]:.6g}, {photo_guidance_spec[1]:.6g}] "
            f"sampled once per chunk (chunk_size={photo_batch})."
        )
    if _is_discrete_list_spec(photo_img2img_spec):
        values_text = ", ".join(f"{float(v):.6g}" for v in (photo_img2img_spec or ()))
        print(
            "[board-cli] chunk-randomized photorealistic img2img enabled: "
            f"img2img_strength sampled from [{values_text}] once per chunk (chunk_size={photo_batch})."
        )
    elif _is_range_spec(photo_img2img_spec):
        print(
            "[board-cli] chunk-randomized photorealistic img2img enabled: "
            f"img2img_strength in [{photo_img2img_spec[0]:.6g}, {photo_img2img_spec[1]:.6g}] "
            f"sampled once per chunk (chunk_size={photo_batch})."
        )
    if bool("photorealistic" in outputs):
        if bool(photo_use_rings_only):
            print("[board-cli] photorealistic rings-only conditioning enabled: ring maps only.")
        elif bool(photo_include_knot_maps):
            print(
                "[board-cli] photorealistic knot maps enabled: "
                "conditioning uses ring+fiber+derived-knot-map."
            )
    accepted = 0
    rejected_outside_log = 0
    rejected_low_knot_count = 0
    photorealistic_jobs: List[_PhotorealisticJob] = []
    accepted_filenames: List[str] = []
    fiber_blur_chunk_values: Dict[int, float] = {}
    fiber_irregularity_chunk_values: Dict[int, float] = {}
    ring_irregularity_chunk_values: Dict[int, float] = {}
    contour_line_width_chunk_values: Dict[int, float] = {}
    photo_guidance_chunk_values: Dict[int, float] = {}
    photo_img2img_chunk_values: Dict[int, float] = {}

    for attempt in range(max_attempts):
        if accepted >= num_boards:
            break

        cfg = BoardConfig(**_model_dump(base_cfg))
        if bool(cfg.use_seed):
            cfg.simulation_seed = int(base_cfg.simulation_seed) + int(attempt) + int(worker_seed_offset)
            seed_all(int(cfg.simulation_seed))

        _apply_randomized_crook_taper(cfg, placement_policy)
        if placement_policy.mode == "dimensions":
            # Provisional XY extents are centered until spline-aware placement is sampled.
            cfg.board_x_min = -0.5 * float(placement_policy.board_width)
            cfg.board_x_max = 0.5 * float(placement_policy.board_width)
            cfg.board_y_min = -0.5 * float(placement_policy.board_thickness)
            cfg.board_y_max = 0.5 * float(placement_policy.board_thickness)
            cfg.board_z_min = 0.0
            cfg.board_z_max = float(placement_policy.board_length)

        k = KnotSystem(cfg)
        knot_count = int(getattr(k, "n_knots", 0) or 0)
        if knot_count < min_knot_count:
            rejected_low_knot_count += 1
            if (rejected_low_knot_count % 25) == 0:
                print(
                    f"[board-cli] rejected_low_knot_count={rejected_low_knot_count} "
                    f"(attempt={attempt + 1}/{max_attempts})"
                )
            continue

        if placement_policy.mode == "dimensions":
            try:
                _apply_random_board_extents_for_dimensions_mode(cfg, k, placement_policy)
            except _RetryablePlacementError:
                rejected_outside_log += 1
                if (rejected_outside_log % 25) == 0:
                    print(
                        f"[board-cli] rejected_outside_log={rejected_outside_log} "
                        f"(attempt={attempt + 1}/{max_attempts})"
                    )
                continue

        mesh = BoardMesh(cfg, k)
        warnings = _board_fit_warnings(cfg, mesh, k)
        if warnings:
            rejected_outside_log += 1
            if (rejected_outside_log % 25) == 0:
                print(
                    f"[board-cli] rejected_outside_log={rejected_outside_log} "
                    f"(attempt={attempt + 1}/{max_attempts})"
                )
            continue

        layers_data, mesh_accum = GrowthSimulator.run(cfg, mesh, k)
        contours_masked = [
            np.asarray(line, dtype=np.float32)
            for line in (layers_data.get("contours_masked") or [])
        ]
        contours_masked_live = [
            np.asarray(line, dtype=np.float32)
            for line in (layers_data.get("contours_masked_live") or [])
        ]
        contours_unmasked = [
            np.asarray(line, dtype=np.float32)
            for line in (layers_data.get("contours_unmasked") or [])
        ]
        contours_mid_masked = [
            np.asarray(line, dtype=np.float32)
            for line in (layers_data.get("contours_mid_masked") or [])
        ]
        contours_mid_masked_live = [
            np.asarray(line, dtype=np.float32)
            for line in (layers_data.get("contours_mid_masked_live") or [])
        ]
        contours_mid_unmasked = [
            np.asarray(line, dtype=np.float32)
            for line in (layers_data.get("contours_mid_unmasked") or [])
        ]

        if show_inside:
            contours_main = contours_unmasked
            contours_mid = contours_mid_unmasked
        else:
            contours_main = contours_masked_live or contours_masked
            contours_mid = contours_mid_masked_live or contours_mid_masked

        bc = mesh.board_coords
        board_dims = {
            "x_min": float(bc["x"][0]),
            "x_max": float(bc["x"][1]),
            "y_min": float(bc["y"][0]),
            "y_max": float(bc["y"][1]),
            "z_min": float(bc["z"][0]),
            "z_max": float(bc["z"][1]),
            "width": float(abs(bc["x"][1] - bc["x"][0])),
            "thickness": float(abs(bc["y"][1] - bc["y"][0])),
            "length": float(abs(bc["z"][1] - bc["z"][0])),
        }
        chunk_index = int(accepted // photo_batch)
        fiber_blur_sigma_value = _sample_chunk_value(
            fiber_blur_spec,
            chunk_index=chunk_index,
            cache=fiber_blur_chunk_values,
        )
        fiber_irregularity_value = _sample_chunk_value(
            fiber_irregularity_spec,
            chunk_index=chunk_index,
            cache=fiber_irregularity_chunk_values,
        )
        ring_irregularity_value = _sample_chunk_value(
            ring_irregularity_spec,
            chunk_index=chunk_index,
            cache=ring_irregularity_chunk_values,
        )
        contour_line_width_value = _sample_chunk_value(
            contour_line_width_spec,
            chunk_index=chunk_index,
            cache=contour_line_width_chunk_values,
        )
        photo_guidance_value = _sample_chunk_value(
            photo_guidance_spec,
            chunk_index=chunk_index,
            cache=photo_guidance_chunk_values,
        )
        photo_img2img_value = _sample_chunk_value_from_range_or_list(
            photo_img2img_spec,
            chunk_index=chunk_index,
            cache=photo_img2img_chunk_values,
        )

        rings_need = bool(outputs & {"rings", "middle", "top_bottom", "photorealistic"})
        fibers_need = bool(("fibers" in outputs) or ("photorealistic" in outputs and not bool(photo_use_rings_only)))
        ring_pngs: Dict[str, bytes] = {}
        fiber_pngs: Dict[str, bytes] = {}
        if rings_need:
            if not contours_main:
                rejected_outside_log += 1
                if (rejected_outside_log % 25) == 0:
                    print(
                        f"[board-cli] rejected_outside_log={rejected_outside_log} "
                        f"(attempt={attempt + 1}/{max_attempts})"
                    )
                continue

            raw_rings = _build_matlab_ring_pngs(
                contours_main,
                board_dims,
                size=image_size,
                line_width=float(contour_line_width_value if contour_line_width_value is not None else 1.0),
            )
            for key, value in raw_rings.items():
                ring_png = _flip_png_vertical_bytes(
                    _apply_png_blur_bytes(value, contour_blur_sigma)
                )
                ring_png = _apply_ring_irregularity_bytes(
                    ring_png,
                    float(ring_irregularity_value if ring_irregularity_value is not None else 0.0),
                )
                ring_pngs[key] = ring_png

            if "middle" in outputs:
                if not contours_mid:
                    rejected_outside_log += 1
                    if (rejected_outside_log % 25) == 0:
                        print(
                            f"[board-cli] rejected_outside_log={rejected_outside_log} "
                            f"(attempt={attempt + 1}/{max_attempts})"
                        )
                    continue
                raw_mid = _build_matlab_mid_ring_png(
                    contours_mid,
                    board_dims,
                    size=image_size,
                    line_width=float(contour_line_width_value if contour_line_width_value is not None else 1.0),
                )
                mid_png = _flip_png_vertical_bytes(
                    _apply_png_blur_bytes(raw_mid, contour_blur_sigma)
                )
                mid_png = _apply_ring_irregularity_bytes(
                    mid_png,
                    float(ring_irregularity_value if ring_irregularity_value is not None else 0.0),
                )
                ring_pngs["rings_5"] = mid_png

            if "top_bottom" in outputs:
                top_bottom_pngs = _build_top_bottom_ring_pngs(
                    contours_main,
                    board_dims,
                    size=image_size,
                    line_width=float(contour_line_width_value if contour_line_width_value is not None else 1.0),
                )
                ring_pngs.update(top_bottom_pngs)

        if fibers_need:
            txx, tyy, tzz = FiberSolver.solve(
                cfg,
                mesh,
                k,
                mesh_accum,
                precomputed_info=layers_data.get("knot_influence_info"),
            )
            mesh_x = np.asarray(getattr(mesh, "x_coords", []), dtype=np.float64).reshape(-1)
            mesh_y = np.asarray(getattr(mesh, "y_coords", []), dtype=np.float64).reshape(-1)
            mesh_z = np.asarray(getattr(mesh, "z_coords", []), dtype=np.float64).reshape(-1)
            raw_fibers_by_side = _build_fiber_surface_pngs(
                np.asarray(txx, dtype=np.float64),
                np.asarray(tyy, dtype=np.float64),
                np.asarray(tzz, dtype=np.float64),
                mesh_x,
                mesh_y,
                mesh_z,
                rand_fibers=bool(cfg.rand_fibers),
                out_of_plane_threshold=float(cfg.out_of_plane_threshold),
                snr=float(cfg.snr),
                size=image_size,
            )
            for folder, side in [
                ("fiber_1", "z_max"),
                ("fiber_2", "z_min"),
                ("fiber_3", "x_max"),
                ("fiber_4", "x_min"),
            ]:
                fiber_png = _flip_png_vertical_bytes(
                    _apply_png_blur_bytes(
                        raw_fibers_by_side[side],
                        float(fiber_blur_sigma_value if fiber_blur_sigma_value is not None else 0.0),
                    )
                )
                fiber_png = _apply_fiber_irregularity_bytes(
                    fiber_png,
                    float(fiber_irregularity_value if fiber_irregularity_value is not None else 0.0),
                )
                fiber_pngs[folder] = fiber_png

        accepted += 1
        filename = f"{(imid_start + accepted - 1):05d}.png"
        file_stem = Path(filename).stem

        if "rings" in outputs:
            for folder in ["rings_1", "rings_2", "rings_3", "rings_4"]:
                _write_png(root / folder / filename, ring_pngs[folder])
        if "middle" in outputs:
            _write_png(root / "rings_5" / filename, ring_pngs["rings_5"])
        if "top_bottom" in outputs:
            _write_png(root / "rings_top" / filename, ring_pngs["rings_top"])
            _write_png(root / "rings_bottom" / filename, ring_pngs["rings_bottom"])
        if "fibers" in outputs:
            for folder in ["fiber_1", "fiber_2", "fiber_3", "fiber_4"]:
                _write_png(root / folder / filename, fiber_pngs[folder])

        temp_input_dir: Optional[Path] = None
        ring_input_paths: Dict[str, Path] = {}
        fiber_input_paths: Dict[str, Path] = {}
        metadata_path = metadata_dir / f"{file_stem}.json"
        if "photorealistic" in outputs:
            if not rings_need or (not bool(photo_use_rings_only) and not fibers_need):
                raise RuntimeError(
                    "Internal error: photorealistic export requires ring generation, "
                    "and fiber generation unless rings-only mode is enabled."
                )
            temp_input_dir = None
            if "rings" in outputs:
                for idx in range(1, 5):
                    ring_input_paths[f"rings_{idx}"] = (
                        root / f"rings_{idx}" / filename
                    )
            else:
                temp_input_dir = root / ".photorealistic_inputs" / file_stem
                for idx in range(1, 5):
                    path = temp_input_dir / f"rings_{idx}" / filename
                    _write_png(path, ring_pngs[f"rings_{idx}"])
                    ring_input_paths[f"rings_{idx}"] = path

            if not bool(photo_use_rings_only):
                if "fibers" in outputs:
                    for idx in range(1, 5):
                        fiber_input_paths[f"fiber_{idx}"] = (
                            root / f"fiber_{idx}" / filename
                        )
                else:
                    if temp_input_dir is None:
                        temp_input_dir = root / ".photorealistic_inputs" / file_stem
                    for idx in range(1, 5):
                        path = temp_input_dir / f"fiber_{idx}" / filename
                        _write_png(path, fiber_pngs[f"fiber_{idx}"])
                        fiber_input_paths[f"fiber_{idx}"] = path

            photorealistic_jobs.append(
                _PhotorealisticJob(
                    filename=filename,
                    ring_paths=ring_input_paths,
                    fiber_paths=fiber_input_paths,
                    temp_input_dir=temp_input_dir,
                    metadata_path=metadata_path,
                )
            )

        # --- Extract knot parameters from KnotSystem for ML dataset -----------
        def _knot_params_to_list(arr):
            """Flatten a (1,1,1,K) or (K,) array to a plain Python list."""
            if arr is None:
                return []
            import numpy as _np
            a = _np.asarray(arr).reshape(-1)
            return [float(v) for v in a]

        knot_params_export = {
            "n_knots":  int(k.n_knots),
            # Per-knot physical parameters  (each list has length n_knots)
            "th0_rad":  _knot_params_to_list(k.th0),   # azimuth angle [rad]
            "z0":       _knot_params_to_list(k.z0),    # axial position [mm]
            "RL":       _knot_params_to_list(k.RL),    # live-knot radius [mm]
            "RD":       _knot_params_to_list(k.RD),    # dead-knot radius [mm]
            "c1":       _knot_params_to_list(k.c1),    # knot axis curvature
            "c2":       _knot_params_to_list(k.c2),    # knot axis slope
            "a1":       _knot_params_to_list(k.a1),    # knot shape polynomial
            "a2":       _knot_params_to_list(k.a2),
            "a3":       _knot_params_to_list(k.a3),
            "a4":       _knot_params_to_list(k.a4),
            "Abump":    _knot_params_to_list(k.Abump),
            "Bbump":    _knot_params_to_list(k.Bbump),
            "Aexp":     _knot_params_to_list(k.Aexp),
            "k_val":    _knot_params_to_list(k.k),
            "kp":       _knot_params_to_list(k.kp),
            # Sequence info (fixed-length slot representation used by knot model)
            "knot_sequence_info": {
                "mode":           str(k.knot_sequence_info.get("mode", "")),
                "dz_mm":          float(k.knot_sequence_info.get("dz_mm", 0.0)),
                "z_min_mm":       float(k.knot_sequence_info.get("z_min_mm", 0.0)),
                "slot_count":     int(k.knot_sequence_info.get("slot_count", 0)),
                # slot_tokens : integer token id per slot (0 = empty)
                "slot_tokens":    [int(t) for t in
                                   k.knot_sequence_info.get("slot_tokens", [])],
                # slot_has_knot : bool per slot
                "slot_has_knot":  [bool(b) for b in
                                   k.knot_sequence_info.get("slot_has_knot", [])],
            },
            # Board geometry (needed to normalise coordinates at training time)
            "board_extents": {
                "x_min": float(cfg.board_x_min), "x_max": float(cfg.board_x_max),
                "y_min": float(cfg.board_y_min), "y_max": float(cfg.board_y_max),
                "z_min": float(cfg.board_z_min), "z_max": float(cfg.board_z_max),
            },
            # Geometry randomisation (crook / taper)
            "geometry_randomization_info": (
                k.geometry_randomization_info
                if isinstance(k.geometry_randomization_info, dict) else {}
            ),
        }
        # -----------------------------------------------------------------------

        board_meta = {
            "board_index": accepted,
            "filename": filename,
            "attempt_index": attempt + 1,
            "simulation_seed": int(cfg.simulation_seed) if bool(cfg.use_seed) else None,
            "chunk_index": int(chunk_index),
            "chunk_sampled_params": {
                "contour_line_width": float(
                    contour_line_width_value if contour_line_width_value is not None else 1.0
                ),
                "fiber_blur_sigma": float(
                    fiber_blur_sigma_value if fiber_blur_sigma_value is not None else 0.0
                ),
                "fiber_irregularity_strength": float(
                    fiber_irregularity_value if fiber_irregularity_value is not None else 0.0
                ),
                "ring_irregularity_strength": float(
                    ring_irregularity_value if ring_irregularity_value is not None else 0.0
                ),
                "photorealistic_guidance_scale": (
                    None if photo_guidance_value is None else float(photo_guidance_value)
                ),
                "photorealistic_img2img_strength": (
                    None if photo_img2img_value is None else float(photo_img2img_value)
                ),
                "photorealistic_include_knot_maps": bool(photo_include_knot_maps),
                "photorealistic_use_rings_only": bool(photo_use_rings_only),
            },
            "board_config": _model_dump(cfg),
            "outputs_requested": sorted(outputs),
            "outputs_saved": {
                "rings_1_to_4": bool("rings" in outputs),
                "rings_5_middle": bool("middle" in outputs),
                "rings_top_bottom": bool("top_bottom" in outputs),
                "fibers_1_to_4": bool("fibers" in outputs),
                "photorealistic_1_to_4": False,
            },
            "knot_params": knot_params_export,
        }
        metadata_path.write_text(
            json.dumps(board_meta, indent=2),
            encoding="utf-8",
        )
        accepted_filenames.append(filename)
        print(
            f"[board-cli] accepted {accepted}/{num_boards} "
            f"(attempt {attempt + 1}/{max_attempts}, rejected_outside_log={rejected_outside_log}, "
            f"rejected_low_knot_count={rejected_low_knot_count})"
        )

    if accepted < num_boards:
        hint = (
            "reducing board extents."
            if placement_policy.mode == "extents"
            else "reducing board dimensions."
        )
        raise RuntimeError(
            "Could not generate enough valid boards within max attempts. "
            f"requested={num_boards}, accepted={accepted}, rejected_outside_log={rejected_outside_log}, "
            f"rejected_low_knot_count={rejected_low_knot_count}, "
            f"max_attempts={max_attempts}. "
            f"Try increasing --max-attempts or {hint}"
        )

    if "photorealistic" in outputs:
        capability = get_photorealistic_capability()
        if not bool(capability.get("available")):
            reason = str(capability.get("reason") or "Photorealistic inference unavailable.")
            raise RuntimeError(reason)
        preload_photorealistic_model()
        total_jobs = len(photorealistic_jobs)
        print(
            f"[board-cli] generating photorealistic surfaces for {total_jobs} boards "
            f"(boards_per_batch={photo_batch})"
        )
        for start in range(0, total_jobs, photo_batch):
            chunk = photorealistic_jobs[start:start + photo_batch]
            chunk_index = int(start // photo_batch)
            chunk_guidance = _sample_chunk_value(
                photo_guidance_spec,
                chunk_index=chunk_index,
                cache=photo_guidance_chunk_values,
            )
            chunk_img2img = _sample_chunk_value(
                photo_img2img_spec,
                chunk_index=chunk_index,
                cache=photo_img2img_chunk_values,
            )
            chunk_inputs: List[Dict[str, Dict[str, bytes]]] = []
            for job in chunk:
                rings = {k: p.read_bytes() for k, p in job.ring_paths.items()}
                payload: Dict[str, Dict[str, bytes]] = {"rings": rings}
                if job.fiber_paths:
                    fibers = {k: p.read_bytes() for k, p in job.fiber_paths.items()}
                    payload["fibers"] = fibers
                chunk_inputs.append(payload)

            chunk_outputs = generate_photorealistic_surfaces_batch(
                chunk_inputs,
                ddim_steps=photo_steps,
                guidance_scale=chunk_guidance,
                use_img2img_strength=chunk_img2img,
                include_knot_maps=bool(photo_include_knot_maps),
                use_rings_only=bool(photo_use_rings_only),
                boards_per_batch=photo_batch,
            )
            if len(chunk_outputs) != len(chunk):
                raise RuntimeError(
                    "Photorealistic batch output size mismatch. "
                    f"expected={len(chunk)}, got={len(chunk_outputs)}."
                )

            for job, generated in zip(chunk, chunk_outputs):
                for idx in range(1, 5):
                    _write_png(
                        root / f"photorealistic_{idx}" / job.filename,
                        generated[f"surface_{idx}"],
                    )

                meta_path = job.metadata_path
                try:
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                except Exception:
                    meta = {}
                if isinstance(meta, dict):
                    outputs_saved = meta.get("outputs_saved")
                    if isinstance(outputs_saved, dict):
                        outputs_saved["photorealistic_1_to_4"] = True
                    else:
                        meta["outputs_saved"] = {"photorealistic_1_to_4": True}
                    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

                if job.temp_input_dir is not None and job.temp_input_dir.is_dir():
                    shutil.rmtree(job.temp_input_dir, ignore_errors=True)
                    parent = job.temp_input_dir.parent
                    if parent.name == ".photorealistic_inputs":
                        try:
                            parent.rmdir()
                        except OSError:
                            pass

            print(
                f"[board-cli] photorealistic progress: {min(start + photo_batch, total_jobs)}/{total_jobs}"
            )

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "num_boards_requested": num_boards,
        "num_boards_generated": accepted,
        "max_attempts": max_attempts,
        "min_knot_count": int(min_knot_count),
        "num_rejected_outside_log": rejected_outside_log,
        "num_rejected_low_knot_count": rejected_low_knot_count,
        "outputs_requested": sorted(outputs),
        "image_size": image_size,
        "contour_line_width": _serialize_range_spec(contour_line_width_spec),
        "contour_blur_sigma": contour_blur_sigma,
        "fiber_blur_sigma": _serialize_range_spec(fiber_blur_spec),
        "fiber_irregularity_strength": _serialize_range_spec(fiber_irregularity_spec),
        "ring_irregularity_strength": _serialize_range_spec(ring_irregularity_spec),
        "show_rings_inside_knots": bool(show_inside),
        "placement": {
            "mode": placement_policy.mode,
            "board_dimensions_mm": {
                "width": float(placement_policy.board_width),
                "thickness": float(placement_policy.board_thickness),
                "length": float(placement_policy.board_length),
            },
            "dimension_mode_z_extent_mm": {
                "min": 0.0,
                "max": float(placement_policy.board_length),
            },
        },
        "auto_randomization": {
            "randomize_crook_taper": bool(placement_policy.randomize_crook_taper),
            "crook_component_count": int(placement_policy.crook_component_count),
            "crook_shift_max_mm": float(placement_policy.crook_shift_max_mm),
            "random_crook_amplitude_max": list(base_cfg.random_crook_amplitude_max or []),
            "random_crook_extra_orders": list(base_cfg.random_crook_extra_orders or []),
            "random_crook_theta_min_deg": float(placement_policy.random_crook_theta_min_deg),
            "random_crook_theta_max_deg": float(placement_policy.random_crook_theta_max_deg),
            "random_crook_scale_max": float(placement_policy.random_crook_scale_max),
            "random_taper_max": float(placement_policy.random_taper_max),
            "manual_crook_orders": list(base_cfg.manual_crook_orders or []),
        },
        "photorealistic": {
            "enabled": bool("photorealistic" in outputs),
            "ddim_steps": photo_steps,
            "guidance_scale": _serialize_range_spec(photo_guidance_spec),
            "img2img_strength": _serialize_range_or_list_spec(photo_img2img_spec),
            "include_knot_maps": bool(photo_include_knot_maps),
            "use_rings_only": bool(photo_use_rings_only),
            "boards_per_batch": int(photo_batch),
        },
        "chunk_sampled_params": {
            "chunk_size_boards": int(photo_batch),
            "contour_line_width_by_chunk": {
                str(k): float(v)
                for k, v in sorted(contour_line_width_chunk_values.items(), key=lambda kv: int(kv[0]))
            },
            "fiber_blur_sigma_by_chunk": {
                str(k): float(v)
                for k, v in sorted(fiber_blur_chunk_values.items(), key=lambda kv: int(kv[0]))
            },
            "fiber_irregularity_strength_by_chunk": {
                str(k): float(v)
                for k, v in sorted(fiber_irregularity_chunk_values.items(), key=lambda kv: int(kv[0]))
            },
            "ring_irregularity_strength_by_chunk": {
                str(k): float(v)
                for k, v in sorted(ring_irregularity_chunk_values.items(), key=lambda kv: int(kv[0]))
            },
            "photorealistic_guidance_scale_by_chunk": {
                str(k): float(v)
                for k, v in sorted(photo_guidance_chunk_values.items(), key=lambda kv: int(kv[0]))
            },
            "photorealistic_img2img_strength_by_chunk": {
                str(k): float(v)
                for k, v in sorted(photo_img2img_chunk_values.items(), key=lambda kv: int(kv[0]))
            },
        },
        "generated_filenames": accepted_filenames,
        "metadata_folder": "metadata",
        "base_config": _model_dump(base_cfg),
    }
    manifest_path_raw = str(getattr(args, "_multi_gpu_manifest_path", "") or "").strip()
    manifest_path = (
        Path(manifest_path_raw).expanduser().resolve()
        if manifest_path_raw
        else (root / "manifest.json")
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(
        f"[board-cli] done. generated={accepted}, rejected_outside_log={rejected_outside_log}, "
        f"rejected_low_knot_count={rejected_low_knot_count}, "
        f"manifest={manifest_path}"
    )
    return manifest


def regenerate_photorealistic_for_existing_boards(args: Any) -> None:
    config_payload = _load_json_payload(str(args.config_json or ""))
    board_gen_cfg = (
        dict(config_payload.get("boards_generate"))
        if isinstance(config_payload.get("boards_generate"), dict)
        else {}
    )

    def resolve(name: str, cli_value: Any, default: Any) -> Any:
        if cli_value is not None:
            return cli_value
        if name in board_gen_cfg and board_gen_cfg.get(name) is not None:
            return board_gen_cfg.get(name)
        # Accept both snake_case and kebab-case keys from config JSON.
        alt_name = name.replace("_", "-")
        if alt_name in board_gen_cfg and board_gen_cfg.get(alt_name) is not None:
            return board_gen_cfg.get(alt_name)
        return default

    root_raw = str(resolve("output_dir", args.data_root, "") or "").strip()
    if not root_raw:
        raise RuntimeError(
            "Missing data root. Set --data-root or boards_generate.output_dir in --config-json."
        )
    root = Path(root_raw).expanduser().resolve()
    if not root.is_dir():
        raise RuntimeError(f"Data root does not exist: {root}")

    photo_steps_raw = resolve("photorealistic_ddim_steps", args.photorealistic_ddim_steps, None)
    photo_steps = None if photo_steps_raw is None else max(1, int(photo_steps_raw))
    photo_guidance_spec = _parse_float_or_range_arg(
        resolve("photorealistic_guidance_scale", args.photorealistic_guidance_scale, None),
        flag_name="photorealistic_guidance_scale",
        allow_none=True,
    )
    photo_img2img_spec = _parse_float_or_range_or_list_arg(
        resolve("photorealistic_img2img_strength", args.photorealistic_img2img_strength, None),
        flag_name="photorealistic_img2img_strength",
        allow_none=True,
        minimum=0.0,
        maximum=1.0,
    )
    fiber_irregularity_spec = _parse_float_or_range_arg(
        resolve("fiber_irregularity_strength", getattr(args, "fiber_irregularity_strength", None), 0.35),
        flag_name="fiber_irregularity_strength",
        allow_none=False,
        minimum=0.0,
        maximum=2.0,
    )
    ring_irregularity_spec = _parse_float_or_range_arg(
        resolve("ring_irregularity_strength", getattr(args, "ring_irregularity_strength", None), 0.40),
        flag_name="ring_irregularity_strength",
        allow_none=False,
        minimum=0.0,
        maximum=2.0,
    )
    photo_batch = max(1, int(resolve("photorealistic_batch_size", args.photorealistic_batch_size, 4)))
    photo_include_knot_maps = _as_bool(
        resolve(
            "photorealistic_include_knot_maps",
            getattr(args, "photorealistic_include_knot_maps", None),
            False,
        )
    )
    photo_use_rings_only = _as_bool(
        resolve(
            "photorealistic_use_rings_only",
            getattr(args, "photorealistic_use_rings_only", None),
            False,
        )
    )
    if bool(photo_use_rings_only) and bool(photo_include_knot_maps):
        raise RuntimeError(
            "photorealistic_use_rings_only=true cannot be combined with "
            "photorealistic_include_knot_maps=true."
        )
    overwrite = _as_bool(resolve("overwrite", args.overwrite, True))

    ring_maps: Dict[int, Dict[str, Path]] = {}
    for idx in range(1, 5):
        ring_maps[idx] = _build_image_stem_map(root / f"rings_{idx}")
    fiber_maps: Dict[int, Dict[str, Path]] = {}
    if not bool(photo_use_rings_only):
        for idx in range(1, 5):
            fiber_maps[idx] = _build_image_stem_map(root / f"fiber_{idx}")

    all_maps: List[Dict[str, Path]] = [*ring_maps.values(), *fiber_maps.values()]
    common_stems = sorted(set.intersection(*[set(m.keys()) for m in all_maps])) if all_maps else []
    if not common_stems:
        if bool(photo_use_rings_only):
            raise RuntimeError("No shared filename stems found across rings_1..4 folders.")
        raise RuntimeError(
            "No shared filename stems found across rings_1..4 and fiber_1..4 folders."
        )

    requested_stems = _parse_stems_csv(getattr(args, "stems", ""))
    if requested_stems:
        missing = [
            stem for stem in requested_stems
            if not all(stem in mapping for mapping in all_maps)
        ]
        if missing:
            preview = ", ".join(missing[:10])
            suffix = "" if len(missing) <= 10 else f", ... (+{len(missing) - 10})"
            raise RuntimeError(
                "Requested stems are missing required photorealistic inputs: "
                f"{preview}{suffix}"
            )
        stems = list(dict.fromkeys(requested_stems))
    else:
        stems = list(common_stems)

    limit_raw = resolve("limit", args.limit, None)
    limit = 0 if limit_raw is None else max(0, int(limit_raw))
    if limit > 0:
        stems = stems[:limit]
    if not stems:
        raise RuntimeError("No stems selected for photorealistic regeneration.")

    existing_photo_maps: Dict[int, Dict[str, Path]] = {}
    for idx in range(1, 5):
        folder = root / f"photorealistic_{idx}"
        if folder.is_dir():
            try:
                existing_photo_maps[idx] = _build_image_stem_map(folder)
            except RuntimeError:
                existing_photo_maps[idx] = {}
        else:
            existing_photo_maps[idx] = {}

    selected_stems = list(stems)
    if not overwrite:
        selected_stems = [
            stem for stem in stems
            if not all(stem in existing_photo_maps[idx] for idx in range(1, 5))
        ]

    if not selected_stems:
        print("[board-cli] no boards require photorealistic regeneration (overwrite=false).")
        summary = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "data_root": str(root),
            "num_candidates": int(len(stems)),
            "num_selected": 0,
            "num_skipped_existing": int(len(stems)),
            "overwrite": bool(overwrite),
            "photorealistic": {
                "ddim_steps": photo_steps,
                "guidance_scale": _serialize_range_spec(photo_guidance_spec),
                "img2img_strength": _serialize_range_or_list_spec(photo_img2img_spec),
                "include_knot_maps": bool(photo_include_knot_maps),
                "use_rings_only": bool(photo_use_rings_only),
                "boards_per_batch": int(photo_batch),
                "fiber_irregularity_strength": _serialize_range_spec(fiber_irregularity_spec),
                "ring_irregularity_strength": _serialize_range_spec(ring_irregularity_spec),
            },
            "generated_filenames": [],
        }
        _write_json(root / "photorealistic_regen_summary.json", summary)
        return

    capability = get_photorealistic_capability()
    if not bool(capability.get("available")):
        reason = str(capability.get("reason") or "Photorealistic inference unavailable.")
        raise RuntimeError(reason)
    preload_photorealistic_model()

    metadata_dir = root / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[board-cli] regenerating photorealistic faces for {len(selected_stems)} boards "
        f"(root={root}, boards_per_batch={photo_batch}, overwrite={overwrite}, "
        f"include_knot_maps={bool(photo_include_knot_maps)}, "
        f"use_rings_only={bool(photo_use_rings_only)})"
    )

    photo_guidance_chunk_values: Dict[int, float] = {}
    photo_img2img_chunk_values: Dict[int, float] = {}
    fiber_irregularity_chunk_values: Dict[int, float] = {}
    ring_irregularity_chunk_values: Dict[int, float] = {}
    generated_filenames: List[str] = []

    for start in range(0, len(selected_stems), photo_batch):
        chunk_stems = selected_stems[start:start + photo_batch]
        chunk_index = int(start // photo_batch)
        chunk_guidance = _sample_chunk_value(
            photo_guidance_spec,
            chunk_index=chunk_index,
            cache=photo_guidance_chunk_values,
        )
        chunk_img2img = _sample_chunk_value_from_range_or_list(
            photo_img2img_spec,
            chunk_index=chunk_index,
            cache=photo_img2img_chunk_values,
        )
        chunk_fiber_irregularity = None
        if not bool(photo_use_rings_only):
            chunk_fiber_irregularity = _sample_chunk_value(
                fiber_irregularity_spec,
                chunk_index=chunk_index,
                cache=fiber_irregularity_chunk_values,
            )
        chunk_ring_irregularity = _sample_chunk_value(
            ring_irregularity_spec,
            chunk_index=chunk_index,
            cache=ring_irregularity_chunk_values,
        )

        chunk_inputs: List[Dict[str, Dict[str, bytes]]] = []
        for stem in chunk_stems:
            rings = {
                f"rings_{idx}": _apply_ring_irregularity_bytes(
                    ring_maps[idx][stem].read_bytes(),
                    float(chunk_ring_irregularity if chunk_ring_irregularity is not None else 0.0),
                )
                for idx in range(1, 5)
            }
            payload: Dict[str, Dict[str, bytes]] = {"rings": rings}
            if not bool(photo_use_rings_only):
                fibers = {
                    f"fiber_{idx}": _apply_fiber_irregularity_bytes(
                        fiber_maps[idx][stem].read_bytes(),
                        float(chunk_fiber_irregularity if chunk_fiber_irregularity is not None else 0.0),
                    )
                    for idx in range(1, 5)
                }
                payload["fibers"] = fibers
            chunk_inputs.append(payload)

        chunk_outputs = generate_photorealistic_surfaces_batch(
            chunk_inputs,
            ddim_steps=photo_steps,
            guidance_scale=chunk_guidance,
            use_img2img_strength=chunk_img2img,
            include_knot_maps=bool(photo_include_knot_maps),
            use_rings_only=bool(photo_use_rings_only),
            boards_per_batch=photo_batch,
        )
        if len(chunk_outputs) != len(chunk_stems):
            raise RuntimeError(
                "Photorealistic batch output size mismatch. "
                f"expected={len(chunk_stems)}, got={len(chunk_outputs)}."
            )

        for stem, generated in zip(chunk_stems, chunk_outputs):
            filename = f"{stem}.png"
            for idx in range(1, 5):
                _write_png(
                    root / f"photorealistic_{idx}" / filename,
                    generated[f"surface_{idx}"],
                )
            generated_filenames.append(filename)

            metadata_path = metadata_dir / f"{stem}.json"
            if metadata_path.is_file():
                try:
                    meta_payload = json.loads(metadata_path.read_text(encoding="utf-8"))
                except Exception:
                    meta_payload = {}
                if isinstance(meta_payload, dict):
                    outputs_saved = meta_payload.get("outputs_saved")
                    if isinstance(outputs_saved, dict):
                        outputs_saved["photorealistic_1_to_4"] = True
                    else:
                        meta_payload["outputs_saved"] = {"photorealistic_1_to_4": True}
                    metadata_path.write_text(
                        json.dumps(meta_payload, indent=2),
                        encoding="utf-8",
                    )

        print(
            f"[board-cli] photorealistic regeneration progress: "
            f"{min(start + photo_batch, len(selected_stems))}/{len(selected_stems)}"
        )

    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_root": str(root),
        "num_candidates": int(len(stems)),
        "num_selected": int(len(selected_stems)),
        "num_skipped_existing": int(len(stems) - len(selected_stems)),
        "overwrite": bool(overwrite),
        "photorealistic": {
            "ddim_steps": photo_steps,
            "guidance_scale": _serialize_range_spec(photo_guidance_spec),
            "img2img_strength": _serialize_range_or_list_spec(photo_img2img_spec),
            "include_knot_maps": bool(photo_include_knot_maps),
            "use_rings_only": bool(photo_use_rings_only),
            "boards_per_batch": int(photo_batch),
            "fiber_irregularity_strength": _serialize_range_spec(fiber_irregularity_spec),
            "ring_irregularity_strength": _serialize_range_spec(ring_irregularity_spec),
        },
        "chunk_sampled_params": {
            "chunk_size_boards": int(photo_batch),
            "photorealistic_guidance_scale_by_chunk": {
                str(k): float(v)
                for k, v in sorted(photo_guidance_chunk_values.items(), key=lambda kv: int(kv[0]))
            },
            "photorealistic_img2img_strength_by_chunk": {
                str(k): float(v)
                for k, v in sorted(photo_img2img_chunk_values.items(), key=lambda kv: int(kv[0]))
            },
            "fiber_irregularity_strength_by_chunk": {
                str(k): float(v)
                for k, v in sorted(fiber_irregularity_chunk_values.items(), key=lambda kv: int(kv[0]))
            },
            "ring_irregularity_strength_by_chunk": {
                str(k): float(v)
                for k, v in sorted(ring_irregularity_chunk_values.items(), key=lambda kv: int(kv[0]))
            },
        },
        "generated_filenames": generated_filenames,
    }
    _write_json(root / "photorealistic_regen_summary.json", summary)
    print(
        f"[board-cli] done. regenerated={len(generated_filenames)}, "
        f"summary={root / 'photorealistic_regen_summary.json'}"
    )
