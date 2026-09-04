from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence


_IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
_SOURCE_FOLDERS = {
    "photorealistic": "photorealistic",
    "ring-color": "ring_color",
}


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read JSON file {path}: {exc}") from exc
    return payload if isinstance(payload, dict) else {}


def _normalise_stem(stem: str) -> str:
    value = Path(str(stem).strip()).stem
    if not value or not re.fullmatch(r"[A-Za-z0-9_.-]+", value):
        raise ValueError("--stem must be a plain filename stem, for example 00001.")
    return value


def _first_manifest_stem(data_root: Path) -> str | None:
    manifest = _read_json(data_root / "manifest.json")
    filenames = manifest.get("generated_filenames")
    if isinstance(filenames, list):
        for filename in filenames:
            if isinstance(filename, str) and filename.strip():
                return _normalise_stem(filename)
    return None


def _candidate_stems(data_root: Path) -> Iterable[str]:
    manifest_stem = _first_manifest_stem(data_root)
    if manifest_stem:
        yield manifest_stem
    for parent in (data_root / "photorealistic_1", data_root / "ring_color_1"):
        if not parent.is_dir():
            continue
        for path in sorted(parent.iterdir()):
            if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES:
                yield path.stem


def _resolve_stem(data_root: Path, stem: str) -> str:
    if str(stem).strip():
        return _normalise_stem(stem)
    for candidate in _candidate_stems(data_root):
        return candidate
    raise ValueError(
        "Could not choose a board automatically. Pass --stem, or provide manifest.json "
        "with generated_filenames."
    )


def _find_surface_file(folder: Path, stem: str) -> Path | None:
    for suffix in _IMAGE_SUFFIXES:
        path = folder / f"{stem}{suffix}"
        if path.is_file():
            return path.resolve()
    return None


def _surface_paths(data_root: Path, stem: str, surface_source: str) -> tuple[str, list[Path]]:
    source = str(surface_source).strip().lower().replace("_", "-")
    if source == "auto":
        candidates: Sequence[str] = ("photorealistic", "ring-color")
    elif source in _SOURCE_FOLDERS:
        candidates = (source,)
    else:
        raise ValueError("--surface-source must be auto, photorealistic, or ring-color.")

    incomplete: list[str] = []
    for candidate in candidates:
        prefix = _SOURCE_FOLDERS[candidate]
        paths = [_find_surface_file(data_root / f"{prefix}_{index}", stem) for index in range(1, 5)]
        if all(path is not None for path in paths):
            return candidate, [path for path in paths if path is not None]
        missing = [str(index) for index, path in enumerate(paths, start=1) if path is None]
        incomplete.append(f"{prefix}_{{{','.join(missing)}}}")

    raise ValueError(
        f"No complete four-face surface set was found for stem {stem}. "
        f"Missing folders/files: {'; '.join(incomplete)}."
    )


def _positive_span(extents: Dict[str, Any], low: str, high: str) -> float | None:
    try:
        value = abs(float(extents[high]) - float(extents[low]))
    except (KeyError, TypeError, ValueError):
        return None
    return value if value > 0.0 else None


def _dimensions_from_metadata(metadata: Dict[str, Any]) -> Dict[str, float] | None:
    knot_params = metadata.get("knot_params")
    extents = knot_params.get("board_extents") if isinstance(knot_params, dict) else None
    if isinstance(extents, dict):
        width = _positive_span(extents, "x_min", "x_max")
        thickness = _positive_span(extents, "y_min", "y_max")
        length = _positive_span(extents, "z_min", "z_max")
        if width and thickness and length:
            return {"width": width, "thickness": thickness, "length": length}

    config = metadata.get("board_config")
    if isinstance(config, dict):
        width = _positive_span(config, "board_x_min", "board_x_max")
        thickness = _positive_span(config, "board_y_min", "board_y_max")
        length = _positive_span(config, "board_z_min", "board_z_max")
        if width and thickness and length:
            return {"width": width, "thickness": thickness, "length": length}
        try:
            values = {
                "width": float(config["board_width"]),
                "thickness": float(config["board_thickness"]),
                "length": float(config["board_length"]),
            }
        except (KeyError, TypeError, ValueError):
            values = {}
        if values and all(value > 0.0 for value in values.values()):
            return values
    return None


def _load_dimensions_mm(data_root: Path, stem: str) -> tuple[Dict[str, float], Path | None]:
    metadata_path = data_root / "metadata" / f"{stem}.json"
    metadata = _read_json(metadata_path)
    dimensions = _dimensions_from_metadata(metadata)
    if dimensions is not None:
        return dimensions, metadata_path.resolve()

    manifest = _read_json(data_root / "manifest.json")
    placement = manifest.get("placement")
    board_dimensions = placement.get("board_dimensions_mm") if isinstance(placement, dict) else None
    if isinstance(board_dimensions, dict):
        try:
            dimensions = {
                "width": float(board_dimensions["width"]),
                "thickness": float(board_dimensions["thickness"]),
                "length": float(board_dimensions["length"]),
            }
        except (KeyError, TypeError, ValueError):
            dimensions = {}
        if dimensions and all(value > 0.0 for value in dimensions.values()):
            return dimensions, None

    raise ValueError(
        f"Board dimensions for {stem} were not found in metadata/{stem}.json or manifest.json."
    )


def resolve_board_export_input(
    data_root: str | Path,
    *,
    stem: str = "",
    surface_source: str = "auto",
) -> Dict[str, Any]:
    root = Path(data_root).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"Board dataset root does not exist: {root}")
    resolved_stem = _resolve_stem(root, stem)
    resolved_source, surfaces = _surface_paths(root, resolved_stem, surface_source)
    dimensions, metadata_path = _load_dimensions_mm(root, resolved_stem)
    return {
        "data_root": root,
        "stem": resolved_stem,
        "surface_source": resolved_source,
        "surface_paths": surfaces,
        "dimensions_mm": dimensions,
        "metadata_path": metadata_path,
    }


def _is_wsl() -> bool:
    if os.name == "nt":
        return False
    try:
        return "microsoft" in Path("/proc/sys/kernel/osrelease").read_text(encoding="utf-8").lower()
    except OSError:
        return False


def _find_blender_executable(explicit: str = "") -> Path:
    requested = str(explicit or os.environ.get("BLENDER_EXECUTABLE", "")).strip()
    if requested:
        path = Path(requested).expanduser()
        if not path.is_file():
            raise ValueError(f"Blender executable does not exist: {path}")
        return path.resolve()

    discovered = shutil.which("blender")
    if discovered:
        return Path(discovered).resolve()

    candidates: list[Path] = []
    if _is_wsl():
        candidates.extend(Path("/mnt/c/Program Files/Blender Foundation").glob("Blender */blender.exe"))
    elif os.name == "nt":
        candidates.extend(Path("C:/Program Files/Blender Foundation").glob("Blender */blender.exe"))
    candidates = sorted((path for path in candidates if path.is_file()), reverse=True)
    if candidates:
        return candidates[0].resolve()
    raise ValueError(
        "Blender was not found. Put blender on PATH, set BLENDER_EXECUTABLE, or pass "
        "--blender-executable."
    )


def _path_for_child_process(path: Path, executable: Path) -> str:
    if _is_wsl() and executable.suffix.lower() == ".exe":
        completed = subprocess.run(
            ["wslpath", "-w", str(path)],
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()
    return str(path)


def export_board_to_blender(
    data_root: str | Path,
    *,
    stem: str = "",
    output_path: str | Path = "",
    surface_source: str = "auto",
    blender_executable: str = "",
    render_preview: bool = True,
    preview_path: str | Path = "",
    render_engine: str = "eevee",
    samples: int = 64,
    pack_images: bool = True,
) -> Dict[str, Any]:
    resolved = resolve_board_export_input(
        data_root,
        stem=stem,
        surface_source=surface_source,
    )
    root: Path = resolved["data_root"]
    resolved_stem: str = resolved["stem"]
    export_dir = root / "blender"

    blend_path = Path(output_path).expanduser() if str(output_path).strip() else export_dir / f"{resolved_stem}.blend"
    if blend_path.suffix.lower() != ".blend":
        blend_path = blend_path.with_suffix(".blend")
    blend_path = blend_path.resolve()
    blend_path.parent.mkdir(parents=True, exist_ok=True)

    image_path = Path(preview_path).expanduser() if str(preview_path).strip() else blend_path.with_name(f"{blend_path.stem}_preview.png")
    if image_path.suffix.lower() != ".png":
        image_path = image_path.with_suffix(".png")
    image_path = image_path.resolve()
    if render_preview:
        image_path.parent.mkdir(parents=True, exist_ok=True)

    engine = str(render_engine).strip().lower()
    if engine not in {"eevee", "cycles"}:
        raise ValueError("--render-engine must be eevee or cycles.")

    executable = _find_blender_executable(blender_executable)
    scene_script = Path(__file__).with_name("_blender_scene.py").resolve()
    payload = {
        "stem": resolved_stem,
        "surface_source": resolved["surface_source"],
        "surface_paths": [str(path) for path in resolved["surface_paths"]],
        "dimensions_mm": resolved["dimensions_mm"],
        "blend_path": str(blend_path),
        "preview_path": str(image_path),
        "render_preview": bool(render_preview),
        "render_engine": engine,
        "samples": max(1, int(samples)),
        "pack_images": bool(pack_images),
    }

    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".json",
        prefix=f"board-{resolved_stem}-",
        dir=blend_path.parent,
        encoding="utf-8",
        delete=False,
    ) as handle:
        payload_path = Path(handle.name).resolve()
        if _is_wsl() and executable.suffix.lower() == ".exe":
            converted = dict(payload)
            converted["surface_paths"] = [
                _path_for_child_process(Path(path), executable) for path in resolved["surface_paths"]
            ]
            converted["blend_path"] = _path_for_child_process(blend_path, executable)
            converted["preview_path"] = _path_for_child_process(image_path, executable)
            json.dump(converted, handle, indent=2)
        else:
            json.dump(payload, handle, indent=2)

    command = [
        str(executable),
        "--background",
        "--factory-startup",
        "--python",
        _path_for_child_process(scene_script, executable),
        "--",
        "--payload",
        _path_for_child_process(payload_path, executable),
    ]
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"Blender export failed with exit code {exc.returncode}.") from exc
    finally:
        payload_path.unlink(missing_ok=True)

    if not blend_path.is_file():
        raise RuntimeError(f"Blender reported success but did not create {blend_path}")
    if render_preview and not image_path.is_file():
        raise RuntimeError(f"Blender reported success but did not create preview {image_path}")

    return {
        "stem": resolved_stem,
        "surface_source": resolved["surface_source"],
        "surface_paths": [str(path) for path in resolved["surface_paths"]],
        "dimensions_mm": resolved["dimensions_mm"],
        "blend_path": str(blend_path),
        "preview_path": str(image_path) if render_preview else None,
        "blender_executable": str(executable),
        "images_packed": bool(pack_images),
        "camera": "orthographic three-quarter long-side view; end cross-sections edge-on",
    }
