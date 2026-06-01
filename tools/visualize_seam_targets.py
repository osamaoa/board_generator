#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw


_IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")


def _build_stem_map(folder: Path) -> Dict[str, Path]:
    if not folder.is_dir():
        raise RuntimeError(f"Missing dataset folder: {folder}")
    out: Dict[str, Path] = {}
    for entry in sorted(folder.iterdir()):
        if not entry.is_file():
            continue
        if entry.suffix.lower() not in _IMG_EXTS:
            continue
        if entry.stem not in out:
            out[entry.stem] = entry
    return out


def _resolve_stem(data_root: Path, requested: str) -> Tuple[str, List[Path]]:
    color_dirs = [data_root / f"color_{i}" for i in range(1, 5)]
    if requested:
        paths: List[Path] = []
        for d in color_dirs:
            found: Path | None = None
            for ext in _IMG_EXTS:
                candidate = d / f"{requested}{ext}"
                if candidate.is_file():
                    found = candidate
                    break
            if found is None:
                # Fallback: case-insensitive stem match.
                for entry in d.iterdir():
                    if not entry.is_file():
                        continue
                    if entry.suffix.lower() not in _IMG_EXTS:
                        continue
                    if entry.stem == requested:
                        found = entry
                        break
            if found is None:
                raise RuntimeError(f"Requested stem '{requested}' not found in folder: {d}")
            paths.append(found)
        return requested, paths

    maps = [_build_stem_map(d) for d in color_dirs]
    stems = sorted(set.intersection(*[set(m.keys()) for m in maps]))
    if not stems:
        raise RuntimeError("No common stems found across color_1..4.")

    stem = stems[0]
    paths = [maps[i][stem] for i in range(4)]
    return stem, paths


def _load_and_resize_rgb(path: Path, image_size: int) -> Image.Image:
    image = Image.open(path).convert("RGB")
    if image_size > 0 and image.size != (image_size, image_size):
        image = image.resize((image_size, image_size), resample=Image.BICUBIC)
    return image


def _edge_mask(size: Tuple[int, int], edge: str, strip_width_latent: int, latent_divisor: int) -> np.ndarray:
    w, h = size
    lat_w = max(1, int(round(w / float(latent_divisor))))
    lat_h = max(1, int(round(h / float(latent_divisor))))
    sw = max(1, int(strip_width_latent))

    mask_lat = np.zeros((lat_h, lat_w), dtype=np.uint8)
    if edge == "left":
        mask_lat[:, :sw] = 255
    elif edge == "right":
        mask_lat[:, max(0, lat_w - sw):] = 255
    elif edge == "top":
        mask_lat[:sw, :] = 255
    elif edge == "bottom":
        mask_lat[max(0, lat_h - sw):, :] = 255
    else:
        raise RuntimeError(f"Unsupported edge: {edge}")

    mask_img = Image.fromarray(mask_lat, mode="L").resize((w, h), resample=Image.Resampling.NEAREST)
    return np.asarray(mask_img, dtype=np.uint8)


def _overlay_mask(image: Image.Image, mask: np.ndarray, color: Tuple[int, int, int], alpha: int = 110) -> Image.Image:
    arr = np.asarray(image).astype(np.float32)
    mask_f = (mask > 0).astype(np.float32)[..., None]
    color_arr = np.asarray(color, dtype=np.float32).reshape(1, 1, 3)
    out = arr * (1.0 - mask_f * (alpha / 255.0)) + color_arr * (mask_f * (alpha / 255.0))
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), mode="RGB")


def _extract_strip(image: Image.Image, edge: str, strip_width_latent: int, latent_divisor: int) -> Image.Image:
    w, h = image.size
    lat_w = max(1, int(round(w / float(latent_divisor))))
    sw = max(1, int(strip_width_latent))
    px = max(1, int(round((sw / float(lat_w)) * w)))
    if edge == "left":
        box = (0, 0, px, h)
    elif edge == "right":
        box = (max(0, w - px), 0, w, h)
    elif edge == "top":
        box = (0, 0, w, px)
    elif edge == "bottom":
        box = (0, max(0, h - px), w, h)
    else:
        raise RuntimeError(f"Unsupported edge: {edge}")
    return image.crop(box)


def _label_image(image: Image.Image, text: str) -> Image.Image:
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, image.width, 24), fill=(0, 0, 0))
    draw.text((6, 6), text, fill=(255, 255, 255))
    return image


def _make_faces_panel(face_images: List[Image.Image], out_path: Path) -> None:
    # 2x2 panel: (surface_1, surface_2, surface_3, surface_4)
    w, h = face_images[0].size
    panel = Image.new("RGB", (w * 2, h * 2), color=(255, 255, 255))
    panel.paste(face_images[0], (0, 0))
    panel.paste(face_images[1], (w, 0))
    panel.paste(face_images[2], (0, h))
    panel.paste(face_images[3], (w, h))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    panel.save(out_path)


def _make_pairs_panel(
    base_faces: List[Image.Image],
    seam_pairs: List[Tuple[int, str, int, str]],
    pair_colors: List[Tuple[int, int, int]],
    strip_width_latent: int,
    latent_divisor: int,
    out_path: Path,
) -> None:
    rows: List[Image.Image] = []
    for pair_idx, ((fa, ea, fb, eb), color) in enumerate(zip(seam_pairs, pair_colors), start=1):
        face_a = base_faces[fa].copy()
        face_b = base_faces[fb].copy()

        mask_a = _edge_mask(face_a.size, ea, strip_width_latent, latent_divisor)
        mask_b = _edge_mask(face_b.size, eb, strip_width_latent, latent_divisor)
        face_a = _overlay_mask(face_a, mask_a, color=color, alpha=120)
        face_b = _overlay_mask(face_b, mask_b, color=color, alpha=120)

        face_a = _label_image(face_a, f"pair {pair_idx}: surface_{fa+1} {ea}")
        face_b = _label_image(face_b, f"pair {pair_idx}: surface_{fb+1} {eb}")

        strip_a = _extract_strip(base_faces[fa], ea, strip_width_latent, latent_divisor)
        strip_b = _extract_strip(base_faces[fb], eb, strip_width_latent, latent_divisor)
        strip_a = strip_a.resize((180, face_a.height), resample=Image.Resampling.BICUBIC)
        strip_b = strip_b.resize((180, face_b.height), resample=Image.Resampling.BICUBIC)
        strip_a = _label_image(strip_a, "edge strip A")
        strip_b = _label_image(strip_b, "edge strip B")

        row = Image.new(
            "RGB",
            (face_a.width + face_b.width + strip_a.width + strip_b.width, face_a.height),
            color=(255, 255, 255),
        )
        x = 0
        for img in [face_a, face_b, strip_a, strip_b]:
            row.paste(img, (x, 0))
            x += img.width
        rows.append(row)

    if not rows:
        return
    panel_w = max(r.width for r in rows)
    panel_h = sum(r.height for r in rows)
    panel = Image.new("RGB", (panel_w, panel_h), color=(255, 255, 255))
    y = 0
    for row in rows:
        panel.paste(row, (0, y))
        y += row.height
    out_path.parent.mkdir(parents=True, exist_ok=True)
    panel.save(out_path)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Visualize the exact seam regions targeted by seam-consistency loss on one training sample.",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="/mnt/e/SCRATCH/Osama/RGB_for_labeling/images",
        help="Dataset root containing color_1..4 folders.",
    )
    parser.add_argument("--stem", type=str, default="", help="Image stem to visualize; default is first common stem.")
    parser.add_argument("--image-size", type=int, default=512, help="Resize size used by training pipeline.")
    parser.add_argument(
        "--strip-width-latent",
        type=int,
        default=2,
        help="Latent strip width from seam loss (default matches training flag seam_strip_width=2).",
    )
    parser.add_argument(
        "--latent-divisor",
        type=int,
        default=8,
        help="VAE downscale factor for SD2 latent space.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".run/seam_target_check",
        help="Output folder for generated visualization images.",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root).expanduser().resolve()
    stem, color_paths = _resolve_stem(data_root, args.stem)

    seam_pairs: List[Tuple[int, str, int, str]] = [
        (0, "left", 2, "right"),
        (0, "right", 3, "left"),
        (1, "right", 2, "left"),
        (1, "left", 3, "right"),
    ]
    pair_colors: List[Tuple[int, int, int]] = [
        (230, 57, 70),
        (29, 161, 242),
        (76, 175, 80),
        (255, 152, 0),
    ]

    base_faces = [_load_and_resize_rgb(p, int(args.image_size)) for p in color_paths]
    annotated_faces = [img.copy() for img in base_faces]

    # Apply all seam masks on each face to show the full targeted border regions.
    face_pair_notes: Dict[int, List[str]] = {0: [], 1: [], 2: [], 3: []}
    for pair_idx, ((fa, ea, fb, eb), color) in enumerate(zip(seam_pairs, pair_colors), start=1):
        mask_a = _edge_mask(annotated_faces[fa].size, ea, int(args.strip_width_latent), int(args.latent_divisor))
        mask_b = _edge_mask(annotated_faces[fb].size, eb, int(args.strip_width_latent), int(args.latent_divisor))
        annotated_faces[fa] = _overlay_mask(annotated_faces[fa], mask_a, color=color, alpha=120)
        annotated_faces[fb] = _overlay_mask(annotated_faces[fb], mask_b, color=color, alpha=120)
        face_pair_notes[fa].append(f"P{pair_idx}:{ea}")
        face_pair_notes[fb].append(f"P{pair_idx}:{eb}")

    for face_idx in range(4):
        label = f"surface_{face_idx+1} | " + ", ".join(face_pair_notes[face_idx])
        annotated_faces[face_idx] = _label_image(annotated_faces[face_idx], label)

    out_root = Path(args.output_dir).expanduser().resolve() / stem
    out_root.mkdir(parents=True, exist_ok=True)

    _make_faces_panel(annotated_faces, out_root / "seam_targets_faces.png")
    _make_pairs_panel(
        base_faces=base_faces,
        seam_pairs=seam_pairs,
        pair_colors=pair_colors,
        strip_width_latent=int(args.strip_width_latent),
        latent_divisor=int(args.latent_divisor),
        out_path=out_root / "seam_targets_pairs.png",
    )

    # Save individual surfaces for easier zooming.
    for i, img in enumerate(annotated_faces, start=1):
        img.save(out_root / f"surface_{i}_seam_targets.png")

    summary = {
        "stem": stem,
        "data_root": str(data_root),
        "color_paths": [str(p) for p in color_paths],
        "image_size": int(args.image_size),
        "strip_width_latent": int(args.strip_width_latent),
        "latent_divisor": int(args.latent_divisor),
        "strip_width_pixels_approx": int(round((args.strip_width_latent / (args.image_size / args.latent_divisor)) * args.image_size)),
        "surface_order": {
            "0": "surface_1",
            "1": "surface_2",
            "2": "surface_3",
            "3": "surface_4",
        },
        "seam_pairs": [
            {"pair": idx + 1, "face_a": f"surface_{fa+1}", "edge_a": ea, "face_b": f"surface_{fb+1}", "edge_b": eb}
            for idx, (fa, ea, fb, eb) in enumerate(seam_pairs)
        ],
        "outputs": {
            "faces_panel": str(out_root / "seam_targets_faces.png"),
            "pairs_panel": str(out_root / "seam_targets_pairs.png"),
            "surface_1": str(out_root / "surface_1_seam_targets.png"),
            "surface_2": str(out_root / "surface_2_seam_targets.png"),
            "surface_3": str(out_root / "surface_3_seam_targets.png"),
            "surface_4": str(out_root / "surface_4_seam_targets.png"),
        },
    }
    with (out_root / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
