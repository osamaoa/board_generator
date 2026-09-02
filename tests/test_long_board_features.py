from __future__ import annotations

from io import BytesIO

import numpy as np
from PIL import Image

from app.core.knot_system import KnotSystem, resolve_knot_sequence_layout
from app.core.photorealistic_inference import (
    compute_long_face_tile_starts,
    stitch_long_face_tiles,
)


def _solid_png(value: int, size: int = 16) -> bytes:
    buffer = BytesIO()
    Image.fromarray(np.full((size, size, 3), value, dtype=np.uint8)).save(buffer, format="PNG")
    return buffer.getvalue()


def test_context_layout_places_origins_outside_visible_board() -> None:
    layout = resolve_knot_sequence_layout(
        board_length_mm=435.0,
        dz_mm=10.0,
        visible_z_min_mm=0.0,
        context_enabled=True,
        context_before_mm=100.0,
        context_after_mm=100.0,
    )
    positions = layout["slot_z_positions_mm"]
    assert layout["visible_slot_count"] == 43
    assert layout["slot_count"] == 63
    assert positions[0] == -90.0
    assert 0.0 in positions
    assert positions[-1] == 530.0


def test_paired_axis_calibration_reconstructs_coefficients(tmp_path) -> None:
    profile = tmp_path / "profile.json"
    profile.write_text(
        '{"schema_version":1,"observations":[{"dz50_mm":20,"dz100_mm":30}],'
        '"sampling":{"jitter_std_dz50_mm":0,"jitter_std_dz100_mm":0}}',
        encoding="utf-8",
    )
    c1, c2, calibrated = KnotSystem._sample_calibrated_knot_axis_coefficients(
        2,
        profile_path=str(profile),
        source_c1=np.zeros(2),
        source_c2=np.zeros(2),
        calibrated_mix=1.0,
    )
    assert calibrated.tolist() == [True, True]
    assert np.allclose(c1 * 50**2 + c2 * 50, 20.0)
    assert np.allclose(c1 * 100**2 + c2 * 100, 30.0)


def test_tile_layout_and_stitch_preserve_long_output() -> None:
    starts = compute_long_face_tile_starts(48, 16, 4)
    assert starts[0] == 0
    assert starts[-1] == 32
    stitched = stitch_long_face_tiles(
        [_solid_png(20), _solid_png(100), _solid_png(180), _solid_png(240)],
        starts,
        output_height=48,
        tile_size=16,
    )
    image = Image.open(BytesIO(stitched))
    assert image.size == (16, 48)
    pixels = np.asarray(image)
    assert np.isfinite(pixels).all()
    assert int(pixels[0].mean()) == 20
    assert int(pixels[-1].mean()) == 240
