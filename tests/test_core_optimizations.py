from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.spatial import Delaunay


ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from app.core.array_backend import seed_all  # noqa: E402
from app.core.config import BoardConfig, InputKnot  # noqa: E402
from app.core.growth import GrowthSimulator  # noqa: E402
from app.core.knot_system import KnotSystem  # noqa: E402
from app.core.mesh import BoardMesh  # noqa: E402
from app.main import _render_fiber_face_png  # noqa: E402


def _small_system() -> tuple[BoardConfig, KnotSystem, BoardMesh]:
    seed_all(1234)
    knots = [
        InputKnot(th0_deg=35.0, z0=35.0, L100=24.0, c1=-0.0012, c2=0.42),
        InputKnot(th0_deg=210.0, z0=92.0, L100=31.0, c1=-0.0021, c2=0.66),
    ]
    cfg = BoardConfig(
        board_width=80.0,
        board_thickness=35.0,
        board_length=145.0,
        board_x_min=-40.0,
        board_x_max=40.0,
        board_y_min=-17.5,
        board_y_max=17.5,
        board_z_min=0.0,
        board_z_max=145.0,
        mesh_size_x_mm=12.0,
        mesh_size_y_mm=12.0,
        mesh_size_z_mm=12.0,
        use_gpu=False,
        use_input_knots=True,
        input_knot_count=len(knots),
        input_knots=knots,
        randomize_crook_taper=False,
        knot_sequence_override_c1_c2=False,
        dead_knots=False,
        display_contours=True,
        display_rings=False,
        display_pith=False,
        calc_fibers=True,
    )
    system = KnotSystem(cfg)
    return cfg, system, BoardMesh(cfg, system)


def test_cached_knot_geometry_preserves_growth_field_exactly() -> None:
    cfg, system, mesh = _small_system()
    outer = system.splines[-1]
    ro_outer = outer(mesh.TH)
    flags = {
        "get_knots": True,
        "include_knot_dev": cfg.include_knot_dev,
        "calc_fibers_a0_method": cfg.calc_fibers_a0_method,
        "dead_knots": cfg.dead_knots,
    }
    _, info = system.calculate_influence(
        mesh.X, mesh.Y, mesh.Z, ro_outer, float(outer(0)), flags
    )

    ring = system.splines[len(system.splines) // 2]
    ro_ring = ring(mesh.TH)
    uncached, _ = system.calculate_influence(
        mesh.X,
        mesh.Y,
        mesh.Z,
        ro_ring,
        float(ring(0)),
        {**flags, "get_knots": False},
    )
    cached, _ = system.calculate_influence(
        mesh.X,
        mesh.Y,
        mesh.Z,
        ro_ring,
        float(ring(0)),
        {**flags, "get_knots": False, "geometry_cache": info["geometry_cache"]},
    )
    np.testing.assert_array_equal(cached, uncached)


def test_selected_live_contours_match_full_growth_output() -> None:
    cfg, system, mesh = _small_system()
    full, _ = GrowthSimulator.run(
        cfg,
        mesh,
        system,
        extract_pith_surface=False,
        retain_growth_layer_fields=False,
    )
    selected, _ = GrowthSimulator.run(
        cfg,
        mesh,
        system,
        contour_variant="masked_live",
        extract_pith_surface=False,
        retain_growth_layer_fields=False,
    )
    assert selected["contours"] == full["contours_masked_live"]
    assert selected["contours_mid_masked_live"] == full["contours_mid_masked_live"]


def test_reused_fiber_triangulation_preserves_png_bytes() -> None:
    x_axis = np.linspace(-12.0, 12.0, 7)
    y_axis = np.linspace(0.0, 145.0, 13)
    x, y = np.meshgrid(x_axis, y_axis, indexing="ij")
    fx = np.cos(0.01 * y) + 0.02 * x
    fy = np.sin(0.01 * y) - 0.01 * x
    triangulation = Delaunay(np.column_stack([x.ravel(), y.ravel()]))

    baseline = _render_fiber_face_png(
        x, y, fx, fy, flip_sign=True, flip_x=True, size=(96, 192)
    )
    reused = _render_fiber_face_png(
        x,
        y,
        fx,
        fy,
        flip_sign=True,
        flip_x=True,
        size=(96, 192),
        triangulation=triangulation,
    )
    assert reused == baseline
