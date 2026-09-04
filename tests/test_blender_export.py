from __future__ import annotations

import json

from PIL import Image

from app.cli import _build_parser
from app.core.board_batch_generation import _export_generated_boards_to_blender
from app.core.blender_export import resolve_board_export_input


def _write_surface_set(root, prefix: str, stem: str = "00007") -> None:
    for index in range(1, 5):
        folder = root / f"{prefix}_{index}"
        folder.mkdir(parents=True)
        Image.new("RGB", (12, 24), color=(80 + index * 20, 50, 25)).save(folder / f"{stem}.png")


def _write_metadata(root, stem: str = "00007") -> None:
    folder = root / "metadata"
    folder.mkdir(parents=True)
    payload = {
        "knot_params": {
            "board_extents": {
                "x_min": -72.5,
                "x_max": 72.5,
                "y_min": -22.5,
                "y_max": 22.5,
                "z_min": 10.0,
                "z_max": 445.0,
            }
        }
    }
    (folder / f"{stem}.json").write_text(json.dumps(payload), encoding="utf-8")


def test_export_input_prefers_complete_photorealistic_set(tmp_path) -> None:
    _write_surface_set(tmp_path, "photorealistic")
    _write_surface_set(tmp_path, "ring_color")
    _write_metadata(tmp_path)
    (tmp_path / "manifest.json").write_text(
        json.dumps({"generated_filenames": ["00007.png"]}),
        encoding="utf-8",
    )

    resolved = resolve_board_export_input(tmp_path)

    assert resolved["stem"] == "00007"
    assert resolved["surface_source"] == "photorealistic"
    assert [path.parent.name for path in resolved["surface_paths"]] == [
        "photorealistic_1",
        "photorealistic_2",
        "photorealistic_3",
        "photorealistic_4",
    ]
    assert resolved["dimensions_mm"] == {
        "width": 145.0,
        "thickness": 45.0,
        "length": 435.0,
    }


def test_export_input_falls_back_to_ring_color(tmp_path) -> None:
    _write_surface_set(tmp_path, "ring_color")
    _write_metadata(tmp_path)

    resolved = resolve_board_export_input(tmp_path, stem="00007.png", surface_source="auto")

    assert resolved["surface_source"] == "ring-color"
    assert resolved["surface_paths"][0].parent.name == "ring_color_1"


def test_cli_parser_exposes_blender_export_options() -> None:
    args = _build_parser().parse_args(
        [
            "boards",
            "export-blender",
            "--data-root",
            "/tmp/boards",
            "--stem",
            "00007",
            "--surface-source",
            "ring-color",
            "--render-preview",
            "false",
        ]
    )

    assert args.group == "boards"
    assert args.command == "export-blender"
    assert args.surface_source == "ring-color"
    assert args.render_preview == "false"


def test_generation_config_runs_blender_post_export(tmp_path, monkeypatch) -> None:
    calls = []

    def fake_export(data_root, **kwargs):
        calls.append((data_root, kwargs))
        return {"stem": kwargs["stem"], "blend_path": str(kwargs["output_path"])}

    monkeypatch.setattr("app.core.blender_export.export_board_to_blender", fake_export)
    summary = _export_generated_boards_to_blender(
        {
            "blender_export": {
                "enabled": True,
                "surface_source": "photorealistic",
                "render_preview": False,
                "pack_images": True,
            }
        },
        root=tmp_path,
        filenames=["00001.png", "00002.png"],
    )

    assert summary is not None
    assert summary["enabled"] is True
    assert [call[1]["stem"] for call in calls] == ["00001", "00002"]
    assert calls[0][1]["surface_source"] == "photorealistic"
    assert calls[0][1]["render_preview"] is False
