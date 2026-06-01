#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image


_COLUMN_KIND = {
    0: "rings",
    1: "fiber",
    2: "pred",
    3: "target",
}


def _image_stats(image: Image.Image) -> Dict[str, float]:
    arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    r = arr[..., 0]
    g = arr[..., 1]
    b = arr[..., 2]
    luma = 0.2126 * r + 0.7152 * g + 0.0722 * b
    sat = np.maximum.reduce([r, g, b]) - np.minimum.reduce([r, g, b])

    return {
        "mean_r": float(r.mean()),
        "mean_g": float(g.mean()),
        "mean_b": float(b.mean()),
        "mean_luma": float(luma.mean()),
        "p95_luma": float(np.quantile(luma, 0.95)),
        "p99_luma": float(np.quantile(luma, 0.99)),
        "clip_high_pct": float((arr >= (250.0 / 255.0)).any(axis=2).mean() * 100.0),
        "clip_low_pct": float((arr <= (5.0 / 255.0)).all(axis=2).mean() * 100.0),
        "mean_sat": float(sat.mean()),
    }


def _mean_stats(items: List[Dict[str, float]]) -> Dict[str, float]:
    if not items:
        return {}
    keys = list(items[0].keys())
    return {k: float(np.mean([x[k] for x in items])) for k in keys}


def _tile_bounds(width: int, height: int, row: int, col: int, rows: int, cols: int) -> Tuple[int, int, int, int]:
    x0 = (width * col) // cols
    x1 = (width * (col + 1)) // cols
    y0 = (height * row) // rows
    y1 = (height * (row + 1)) // rows
    return x0, y0, x1, y1


def analyze_panel(image_path: Path, out_root: Path, rows: int, cols: int) -> Dict[str, object]:
    image = Image.open(image_path).convert("RGB")
    width, height = image.size

    panel_dir = out_root / image_path.stem
    tiles_dir = panel_dir / "tiles"
    tiles_dir.mkdir(parents=True, exist_ok=True)

    pred_stats: List[Dict[str, float]] = []
    target_stats: List[Dict[str, float]] = []
    tile_index: List[Dict[str, object]] = []

    for row in range(rows):
        for col in range(cols):
            x0, y0, x1, y1 = _tile_bounds(width, height, row, col, rows, cols)
            tile = image.crop((x0, y0, x1, y1))
            kind = _COLUMN_KIND.get(col, f"col{col}")
            tile_name = f"r{row+1:02d}_c{col+1:02d}_{kind}.png"
            tile_path = tiles_dir / tile_name
            tile.save(tile_path)

            s = _image_stats(tile)
            tile_index.append(
                {
                    "row": row + 1,
                    "col": col + 1,
                    "kind": kind,
                    "path": str(tile_path),
                    "bounds": [x0, y0, x1, y1],
                    "stats": s,
                }
            )

            if kind == "pred":
                pred_stats.append(s)
            elif kind == "target":
                target_stats.append(s)

    summary = {
        "image": str(image_path),
        "size": [width, height],
        "rows": rows,
        "cols": cols,
        "pred_mean_stats": _mean_stats(pred_stats),
        "target_mean_stats": _mean_stats(target_stats),
        "delta_pred_minus_target": {
            key: float(_mean_stats(pred_stats).get(key, 0.0) - _mean_stats(target_stats).get(key, 0.0))
            for key in _mean_stats(pred_stats).keys()
        },
        "tiles": tile_index,
    }

    with (panel_dir / "analysis.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Split W&B panel images into tiles and compute separate stats for pred/target columns."
    )
    parser.add_argument("images", nargs="+", type=str, help="Panel image paths (e.g., i1.png i2.png).")
    parser.add_argument(
        "--out-dir",
        type=str,
        default=".run/wandb_panel_check",
        help="Output directory for crops and stats.",
    )
    parser.add_argument("--rows", type=int, default=4, help="Number of panel rows (default: 4).")
    parser.add_argument("--cols", type=int, default=4, help="Number of panel columns (default: 4).")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    rows = int(args.rows)
    cols = int(args.cols)
    if rows <= 0 or cols <= 0:
        raise SystemExit("rows and cols must be > 0")

    out_root = Path(args.out_dir).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    summaries: List[Dict[str, object]] = []
    for raw in args.images:
        image_path = Path(raw).expanduser().resolve()
        if not image_path.is_file():
            raise SystemExit(f"Image does not exist: {image_path}")
        summaries.append(analyze_panel(image_path=image_path, out_root=out_root, rows=rows, cols=cols))

    combined_path = out_root / "summary.json"
    with combined_path.open("w", encoding="utf-8") as handle:
        json.dump(summaries, handle, indent=2)

    for item in summaries:
        pred = item.get("pred_mean_stats", {})
        target = item.get("target_mean_stats", {})
        delta = item.get("delta_pred_minus_target", {})
        print(f"image={item.get('image')}")
        print(f"  pred_mean_luma={pred.get('mean_luma'):.6f} target_mean_luma={target.get('mean_luma'):.6f} delta={delta.get('mean_luma'):.6f}")
        print(f"  pred_clip_high_pct={pred.get('clip_high_pct'):.6f} target_clip_high_pct={target.get('clip_high_pct'):.6f} delta={delta.get('clip_high_pct'):.6f}")
        print(f"  pred_mean_sat={pred.get('mean_sat'):.6f} target_mean_sat={target.get('mean_sat'):.6f} delta={delta.get('mean_sat'):.6f}")

    print(f"saved={combined_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
