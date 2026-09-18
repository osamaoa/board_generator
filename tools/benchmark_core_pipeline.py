#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from app.core.array_backend import seed_all, to_numpy  # noqa: E402
from app.core.config import BoardConfig, InputKnot  # noqa: E402
from app.core.fiber import FiberSolver  # noqa: E402
from app.core.growth import GrowthSimulator  # noqa: E402
from app.core.knot_system import KnotSystem  # noqa: E402
from app.core.mesh import BoardMesh  # noqa: E402
from app.main import _build_fiber_surface_pngs, _build_matlab_ring_pngs  # noqa: E402


def _sync_cuda() -> None:
    try:
        import cupy as cp

        cp.cuda.get_current_stream().synchronize()
    except Exception:
        pass


def _timed(name: str, timings: dict[str, float], fn):
    _sync_cuda()
    started = time.perf_counter()
    result = fn()
    _sync_cuda()
    timings[name] = time.perf_counter() - started
    return result


def _manual_knots(length_mm: float, count: int) -> list[InputKnot]:
    knots: list[InputKnot] = []
    z_values = np.linspace(-30.0, length_mm + 30.0, count, dtype=np.float64)
    for idx, z0 in enumerate(z_values):
        phase = float(idx) / max(1, count - 1)
        knots.append(
            InputKnot(
                th0_deg=float((23.0 + idx * 137.507764) % 360.0),
                L100=float(12.0 + 40.0 * ((idx % 5) / 4.0)),
                z0=float(z0),
                c1=float(-0.00035 - 0.0027 * ((idx % 4) / 3.0)),
                c2=float(0.18 + 0.72 * phase),
                k=0.99,
                kp=0.95,
                Abump=float(0.08 + 0.42 * ((idx % 3) / 2.0)),
                Aexp=float(2.02 + 0.12 * ((idx % 4) / 3.0)),
                Bbump=2.0,
                RL=float(55.0 + 25.0 * ((idx % 4) / 3.0)),
                RD=float(105.0 + 35.0 * ((idx % 4) / 3.0)),
                a1=-2.0e-7,
                a2=7.5e-5,
                a3=-0.010,
                a4=float(0.55 + 0.45 * ((idx % 5) / 4.0)),
            )
        )
    return knots


def _config(args: argparse.Namespace) -> BoardConfig:
    knots = _manual_knots(args.length, args.knots)
    return BoardConfig(
        board_width=args.width,
        board_thickness=args.thickness,
        board_length=args.length,
        board_x_min=-0.5 * args.width,
        board_x_max=0.5 * args.width,
        board_y_min=-0.5 * args.thickness,
        board_y_max=0.5 * args.thickness,
        board_z_min=0.0,
        board_z_max=args.length,
        mesh_size_x_mm=args.mesh_mm,
        mesh_size_y_mm=args.mesh_mm,
        mesh_size_z_mm=args.mesh_mm,
        use_gpu=not args.cpu,
        use_seed=True,
        simulation_seed=args.seed,
        use_input_knots=True,
        input_knot_count=len(knots),
        input_knots=knots,
        randomize_crook_taper=False,
        knot_sequence_override_c1_c2=False,
        knot_axis_calibration_enabled=False,
        dead_knots=False,
        include_knot_dev=True,
        calc_fibers=True,
        calc_fibers_a0_method=1,
        display_contours=True,
        display_rings=False,
        display_pith=False,
        display_ring_color=False,
        display_rings_inside_knots=False,
    )


def _array_digest(value: np.ndarray) -> str:
    arr = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(arr.dtype).encode("ascii"))
    digest.update(str(arr.shape).encode("ascii"))
    digest.update(arr.view(np.uint8))
    return digest.hexdigest()


def _png_digests(images: dict[str, bytes]) -> dict[str, str]:
    return {key: hashlib.sha256(value).hexdigest() for key, value in sorted(images.items())}


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark deterministic pre-photorealistic board generation.")
    parser.add_argument("--width", type=float, default=145.0)
    parser.add_argument("--thickness", type=float, default=45.0)
    parser.add_argument("--length", type=float, default=435.0)
    parser.add_argument("--mesh-mm", type=float, default=2.0)
    parser.add_argument("--knots", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260918)
    parser.add_argument("--render-size", type=int, default=256)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--batch-mode", action="store_true")
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    seed_all(args.seed)
    cfg = _config(args)
    timings: dict[str, float] = {}
    started = time.perf_counter()

    knots = _timed("knot_system_s", timings, lambda: KnotSystem(cfg))
    mesh = _timed("mesh_s", timings, lambda: BoardMesh(cfg, knots))
    growth_kwargs = (
        {
            "contour_variant": "masked_live",
            "retain_growth_layer_fields": False,
            "extract_pith_surface": False,
        }
        if args.batch_mode
        else {}
    )
    layers, mesh_accum = _timed(
        "growth_s", timings, lambda: GrowthSimulator.run(cfg, mesh, knots, **growth_kwargs)
    )
    txx, tyy, tzz = _timed(
        "fiber_s",
        timings,
        lambda: FiberSolver.solve(
            cfg,
            mesh,
            knots,
            mesh_accum,
            precomputed_info=layers.get("knot_influence_info"),
        ),
    )

    contours = layers.get("contours_masked_live") or layers.get("contours_masked") or []
    board = mesh.board_coords
    board_dims = {
        "x_min": float(board["x"][0]),
        "x_max": float(board["x"][1]),
        "y_min": float(board["y"][0]),
        "y_max": float(board["y"][1]),
        "z_min": float(board["z"][0]),
        "z_max": float(board["z"][1]),
    }
    ring_pngs = _timed(
        "ring_render_s",
        timings,
        lambda: _build_matlab_ring_pngs(
            contours,
            board_dims,
            size=(args.render_size, args.render_size * int(round(args.length / 145.0))),
            line_width=1.0,
        ),
    )
    fiber_pngs = _timed(
        "fiber_render_s",
        timings,
        lambda: _build_fiber_surface_pngs(
            txx,
            tyy,
            tzz,
            np.asarray(mesh.x_coords),
            np.asarray(mesh.y_coords),
            np.asarray(mesh.z_coords),
            rand_fibers=False,
            size=(args.render_size, args.render_size * int(round(args.length / 145.0))),
        ),
    )

    arrays = {
        "txx": np.asarray(txx),
        "tyy": np.asarray(tyy),
        "tzz": np.asarray(tzz),
        "grid_nx": np.asarray(to_numpy(mesh_accum["grid_nx"])),
        "grid_ny": np.asarray(to_numpy(mesh_accum["grid_ny"])),
        "grid_nz": np.asarray(to_numpy(mesh_accum["grid_nz"])),
        "g_best": np.asarray(to_numpy(mesh_accum["g_best"])),
        "last_g": np.asarray(to_numpy(layers["last_g"])),
        "ttt": np.asarray(to_numpy(layers["ttt"])),
    }
    if args.snapshot:
        args.snapshot.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.snapshot, **arrays)

    timings["total_s"] = time.perf_counter() - started
    report = {
        "config": {
            "width_mm": args.width,
            "thickness_mm": args.thickness,
            "length_mm": args.length,
            "mesh_mm": args.mesh_mm,
            "knots": args.knots,
            "seed": args.seed,
            "use_gpu": not args.cpu,
            "batch_mode": args.batch_mode,
        },
        "mesh_shape": list(mesh.X.shape),
        "growth_layers": len(knots.splines),
        "contour_count": len(contours),
        "timings": timings,
        "array_digests": {key: _array_digest(value) for key, value in arrays.items()},
        "ring_png_digests": _png_digests(ring_pngs),
        "fiber_png_digests": _png_digests(fiber_pngs),
    }
    encoded = json.dumps(report, indent=2, sort_keys=True)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()
