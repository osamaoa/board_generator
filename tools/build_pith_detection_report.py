#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import csv
import html
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional


FACE_ORDER = [
    ("surface_1", "1", "y_max", "x"),
    ("surface_2", "2", "y_min", "x"),
    ("surface_3", "3", "x_max", "y"),
    ("surface_4", "4", "x_min", "y"),
]


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _finite_float_list(value: Any) -> List[float]:
    if not isinstance(value, list):
        return []
    out: List[float] = []
    for raw in value:
        try:
            val = float(raw)
        except (TypeError, ValueError):
            continue
        if math.isfinite(val):
            out.append(val)
    return out


def _bool_list(value: Any) -> List[bool]:
    if not isinstance(value, list):
        return []
    return [bool(v) for v in value]


def _safe_pct(num: int, den: int) -> str:
    if den <= 0:
        return "0.0%"
    return f"{100.0 * float(num) / float(den):.1f}%"


def _image_data_uri(path: Path) -> Optional[str]:
    if not path.is_file():
        return None
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _choose_image_path(data_root: Path, stem: str, surface_idx: str, image_prefix: str) -> Path:
    filename = f"{stem}.png"
    preferred = data_root / f"{image_prefix}_{surface_idx}" / filename
    if preferred.is_file():
        return preferred
    for fallback_prefix in ("photorealistic", "rings", "fiber"):
        fallback = data_root / f"{fallback_prefix}_{surface_idx}" / filename
        if fallback.is_file():
            return fallback
    return preferred


def _svg_path(points: List[tuple[float, float]]) -> str:
    if not points:
        return ""
    first_x, first_y = points[0]
    rest = " ".join(f"L {x:.2f} {y:.2f}" for x, y in points[1:])
    return f"M {first_x:.2f} {first_y:.2f} {rest}".strip()


def _render_face_svg(
    *,
    data_root: Path,
    label: Dict[str, Any],
    surface_key: str,
    surface_idx: str,
    face_name: str,
    horizontal_axis: str,
    image_prefix: str,
) -> str:
    stem = str(label.get("stem") or Path(str(label.get("filename") or "")).stem)
    face = (label.get("faces") or {}).get(surface_key) or {}
    centerline = label.get("pith_centerline") or {}
    image_x_norm = _finite_float_list(face.get("image_x_norm"))
    z_norm = _finite_float_list(centerline.get("z_norm"))
    projected_on_face = _bool_list(face.get("projected_on_face"))
    perpendicular = _finite_float_list(face.get("perpendicular_distance_mm"))

    n = min(len(image_x_norm), len(z_norm))
    if n <= 0:
        image_x_norm = [0.5, 0.5]
        z_norm = [0.0, 1.0]
        projected_on_face = [True, True]
        perpendicular = [0.0, 0.0]
        n = 2

    image_x_norm = image_x_norm[:n]
    z_norm = z_norm[:n]
    projected_on_face = projected_on_face[:n] if len(projected_on_face) >= n else [False] * n
    perpendicular = perpendicular[:n] if len(perpendicular) >= n else [0.0] * n

    domain_min = min(0.0, min(image_x_norm))
    domain_max = max(1.0, max(image_x_norm))
    domain_span = max(1e-6, domain_max - domain_min)
    pad_domain = max(0.08, 0.08 * domain_span)
    domain_min -= pad_domain
    domain_max += pad_domain
    domain_span = max(1e-6, domain_max - domain_min)

    width = 420.0
    height = 340.0
    pad_left = 34.0
    pad_top = 22.0
    plot_w = 360.0
    plot_h = 280.0

    def sx(value: float) -> float:
        return pad_left + ((float(value) - domain_min) / domain_span) * plot_w

    def sy(value: float) -> float:
        return pad_top + max(0.0, min(1.0, float(value))) * plot_h

    image_x0 = sx(0.0)
    image_x1 = sx(1.0)
    image_y0 = sy(0.0)
    image_y1 = sy(1.0)
    image_uri = _image_data_uri(_choose_image_path(data_root, stem, surface_idx, image_prefix))

    step = max(1, n // 256)
    points = [(sx(image_x_norm[i]), sy(z_norm[i])) for i in range(0, n, step)]
    if points[-1] != (sx(image_x_norm[-1]), sy(z_norm[-1])):
        points.append((sx(image_x_norm[-1]), sy(z_norm[-1])))
    path_data = _svg_path(points)

    on_count = sum(1 for v in projected_on_face if v)
    dist_min = min(perpendicular) if perpendicular else 0.0
    dist_max = max(perpendicular) if perpendicular else 0.0
    status = (
        f"projected in image: {_safe_pct(on_count, n)}, "
        f"hidden-axis distance {dist_min:.1f}..{dist_max:.1f} mm"
    )

    image_markup = (
        f'<image href="{image_uri}" x="{image_x0:.2f}" y="{image_y0:.2f}" '
        f'width="{(image_x1 - image_x0):.2f}" height="{(image_y1 - image_y0):.2f}" '
        'preserveAspectRatio="none" />'
        if image_uri
        else (
            f'<rect x="{image_x0:.2f}" y="{image_y0:.2f}" '
            f'width="{(image_x1 - image_x0):.2f}" height="{(image_y1 - image_y0):.2f}" '
            'fill="#f3f4f6" />'
        )
    )

    return f"""
      <div class="face-card">
        <div class="face-title">{html.escape(surface_key)} - {html.escape(face_name)} ({html.escape(horizontal_axis.upper())} vs Z)</div>
        <svg viewBox="0 0 {width:.0f} {height:.0f}" role="img" aria-label="Pith projection for {html.escape(surface_key)}">
          <rect x="0" y="0" width="{width:.0f}" height="{height:.0f}" rx="8" fill="#ffffff" />
          {image_markup}
          <rect x="{image_x0:.2f}" y="{image_y0:.2f}" width="{(image_x1 - image_x0):.2f}" height="{(image_y1 - image_y0):.2f}" fill="none" stroke="#111827" stroke-width="1.2" />
          <path d="{html.escape(path_data)}" fill="none" stroke="#e11d48" stroke-width="2.6" stroke-linecap="round" stroke-linejoin="round" />
          <circle cx="{points[0][0]:.2f}" cy="{points[0][1]:.2f}" r="4" fill="#16a34a" />
          <circle cx="{points[-1][0]:.2f}" cy="{points[-1][1]:.2f}" r="4" fill="#7c3aed" />
          <line x1="{image_x0:.2f}" y1="{image_y1 + 8.0:.2f}" x2="{image_x1:.2f}" y2="{image_y1 + 8.0:.2f}" stroke="#6b7280" stroke-width="1" />
          <text x="{image_x0:.2f}" y="{image_y1 + 24.0:.2f}" font-size="11" fill="#374151">face min</text>
          <text x="{image_x1:.2f}" y="{image_y1 + 24.0:.2f}" font-size="11" fill="#374151" text-anchor="end">face max</text>
        </svg>
        <div class="face-status">{html.escape(status)}</div>
      </div>
    """


def _write_centerline_csv(labels: List[Dict[str, Any]], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "stem",
                "row",
                "z_mm",
                "x_mm",
                "y_mm",
                "z_norm",
                "x_norm",
                "y_norm",
                "inside_board_xy",
            ],
        )
        writer.writeheader()
        for label in labels:
            stem = str(label.get("stem") or Path(str(label.get("filename") or "")).stem)
            centerline = label.get("pith_centerline") or {}
            rows = centerline.get("row") or []
            z_mm = centerline.get("z_mm") or []
            x_mm = centerline.get("x_mm") or []
            y_mm = centerline.get("y_mm") or []
            z_norm = centerline.get("z_norm") or []
            x_norm = centerline.get("x_norm") or []
            y_norm = centerline.get("y_norm") or []
            inside = centerline.get("inside_board_xy") or []
            n = min(
                len(rows),
                len(z_mm),
                len(x_mm),
                len(y_mm),
                len(z_norm),
                len(x_norm),
                len(y_norm),
                len(inside),
            )
            for idx in range(n):
                writer.writerow(
                    {
                        "stem": stem,
                        "row": rows[idx],
                        "z_mm": z_mm[idx],
                        "x_mm": x_mm[idx],
                        "y_mm": y_mm[idx],
                        "z_norm": z_norm[idx],
                        "x_norm": x_norm[idx],
                        "y_norm": y_norm[idx],
                        "inside_board_xy": bool(inside[idx]),
                    }
                )


def build_report(args: argparse.Namespace) -> None:
    data_root = Path(args.data_root).expanduser().resolve()
    label_dir = data_root / "pith_labels"
    label_paths = sorted(label_dir.glob("*.json"))
    if args.limit is not None:
        label_paths = label_paths[: max(0, int(args.limit))]
    if not label_paths:
        raise RuntimeError(f"No pith label JSON files found in {label_dir}")

    labels = [_read_json(path) for path in label_paths]
    labels = [label for label in labels if label]
    if not labels:
        raise RuntimeError(f"No valid pith label JSON files found in {label_dir}")

    csv_path = Path(args.output_csv).expanduser().resolve() if args.output_csv else label_dir / "pith_centerlines.csv"
    _write_centerline_csv(labels, csv_path)

    board_sections: List[str] = []
    for label in labels:
        stem = str(label.get("stem") or Path(str(label.get("filename") or "")).stem)
        ext = label.get("board_extents") or {}
        centerline = label.get("pith_centerline") or {}
        x_vals = _finite_float_list(centerline.get("x_mm"))
        y_vals = _finite_float_list(centerline.get("y_mm"))
        inside = _bool_list(centerline.get("inside_board_xy"))
        inside_text = _safe_pct(sum(1 for v in inside if v), len(inside))
        summary = (
            f"X {min(x_vals):.1f}..{max(x_vals):.1f} mm, "
            f"Y {min(y_vals):.1f}..{max(y_vals):.1f} mm, "
            f"inside board XY: {inside_text}"
            if x_vals and y_vals
            else "No centerline samples."
        )
        faces = "\n".join(
            _render_face_svg(
                data_root=data_root,
                label=label,
                surface_key=surface_key,
                surface_idx=surface_idx,
                face_name=face_name,
                horizontal_axis=horizontal_axis,
                image_prefix=str(args.image_prefix),
            )
            for surface_key, surface_idx, face_name, horizontal_axis in FACE_ORDER
        )
        board_sections.append(
            f"""
            <section class="board-section">
              <h2>Board {html.escape(stem)}</h2>
              <div class="board-meta">
                <span>{html.escape(summary)}</span>
                <span>Extents: X [{float(ext.get('x_min', 0.0)):.1f}, {float(ext.get('x_max', 0.0)):.1f}], Y [{float(ext.get('y_min', 0.0)):.1f}, {float(ext.get('y_max', 0.0)):.1f}], Z [{float(ext.get('z_min', 0.0)):.1f}, {float(ext.get('z_max', 0.0)):.1f}] mm</span>
              </div>
              <div class="face-grid">{faces}</div>
            </section>
            """
        )

    output_html = Path(args.output_html).expanduser().resolve()
    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_html.write_text(
        f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Pith Detection Data Report</title>
  <style>
    body {{
      margin: 0;
      font-family: Arial, Helvetica, sans-serif;
      background: #f8fafc;
      color: #111827;
    }}
    header {{
      padding: 24px 28px 12px;
      border-bottom: 1px solid #d1d5db;
      background: #ffffff;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 24px;
    }}
    h2 {{
      margin: 0;
      font-size: 18px;
    }}
    .subtle {{
      color: #4b5563;
      font-size: 14px;
    }}
    .board-section {{
      margin: 18px 28px;
      padding: 18px;
      background: #ffffff;
      border: 1px solid #d1d5db;
      border-radius: 8px;
    }}
    .board-meta {{
      display: flex;
      flex-wrap: wrap;
      gap: 12px 20px;
      margin: 8px 0 14px;
      color: #374151;
      font-size: 13px;
    }}
    .face-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      gap: 14px;
    }}
    .face-card {{
      border: 1px solid #e5e7eb;
      border-radius: 8px;
      padding: 10px;
      background: #ffffff;
    }}
    .face-title {{
      font-weight: 700;
      font-size: 13px;
      margin: 0 0 6px;
    }}
    .face-status {{
      color: #4b5563;
      font-size: 12px;
      margin-top: 6px;
    }}
    svg {{
      display: block;
      width: 100%;
      height: auto;
      background: #f9fafb;
      border-radius: 8px;
    }}
  </style>
</head>
<body>
  <header>
    <h1>Pith Detection Data Report</h1>
    <div class="subtle">Data root: {html.escape(str(data_root))}</div>
    <div class="subtle">Centerline CSV: {html.escape(str(csv_path))}</div>
    <div class="subtle">Green marker is z_min/top row; purple marker is z_max/bottom row. Red pith lines can extend outside the face image when the projection is off-face.</div>
  </header>
  {"".join(board_sections)}
</body>
</html>
""",
        encoding="utf-8",
    )
    print(f"Wrote {output_html}")
    print(f"Wrote {csv_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an HTML sanity report for generated pith-detection labels."
    )
    parser.add_argument("--data-root", required=True, help="Generated board dataset root.")
    parser.add_argument("--output-html", required=True, help="HTML report path.")
    parser.add_argument(
        "--output-csv",
        default="",
        help="Optional centerline CSV path. Defaults to data-root/pith_labels/pith_centerlines.csv.",
    )
    parser.add_argument(
        "--image-prefix",
        default="photorealistic",
        help="Side-image folder prefix, e.g. photorealistic or rings.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional max boards to include.")
    return parser.parse_args()


def main() -> None:
    build_report(parse_args())


if __name__ == "__main__":
    main()
