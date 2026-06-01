#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation, distance_transform_edt, gaussian_filter, sobel


_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


@dataclass(frozen=True)
class FiberSample:
    stem: str
    face_idx: int
    fiber_path: Path


def _iter_images(folder: Path) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    if not folder.is_dir():
        return out
    for entry in sorted(folder.iterdir()):
        if not entry.is_file():
            continue
        if entry.suffix.lower() not in _IMAGE_EXTS:
            continue
        if entry.stem not in out:
            out[entry.stem] = entry
    return out


def _load_gray(path: Path) -> np.ndarray:
    arr = np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0
    if arr.ndim != 2:
        raise RuntimeError(f"Expected grayscale image at {path}.")
    return np.clip(arr, 0.0, 1.0)


def _robust_norm(x: np.ndarray, lo: float = 2.0, hi: float = 98.0) -> np.ndarray:
    finite = np.isfinite(x)
    if not np.any(finite):
        return np.zeros_like(x, dtype=np.float32)
    vals = x[finite]
    v_lo = float(np.percentile(vals, lo))
    v_hi = float(np.percentile(vals, hi))
    if not (np.isfinite(v_lo) and np.isfinite(v_hi)) or v_hi <= v_lo + 1e-8:
        return np.zeros_like(x, dtype=np.float32)
    y = (x - v_lo) / (v_hi - v_lo)
    return np.clip(y, 0.0, 1.0).astype(np.float32, copy=False)


def _structure_isotropy(img: np.ndarray, sigma_tensor: float = 2.0) -> Tuple[np.ndarray, np.ndarray]:
    gx = sobel(img, axis=1, mode="reflect")
    gy = sobel(img, axis=0, mode="reflect")
    grad = np.sqrt(gx * gx + gy * gy).astype(np.float32, copy=False)
    jxx = gaussian_filter(gx * gx, sigma=sigma_tensor)
    jyy = gaussian_filter(gy * gy, sigma=sigma_tensor)
    jxy = gaussian_filter(gx * gy, sigma=sigma_tensor)
    trace = jxx + jyy + 1e-8
    delta = np.sqrt(np.maximum((jxx - jyy) ** 2 + 4.0 * (jxy ** 2), 0.0))
    lam1 = 0.5 * (trace + delta)
    lam2 = 0.5 * (trace - delta)
    coherence = (lam1 - lam2) / (lam1 + lam2 + 1e-8)
    isotropy = np.clip(1.0 - coherence, 0.0, 1.0).astype(np.float32, copy=False)
    return isotropy, grad


def build_knot_map_from_fiber(fiber_gray: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Knot map heuristic using fiber only.
    Observation used: clear wood is near gray baseline; knots shift toward white/black.
    """
    fiber = np.clip(fiber_gray, 0.0, 1.0).astype(np.float32, copy=False)

    baseline = float(np.median(fiber))
    dev = fiber - baseline
    abs_dev = np.abs(dev)

    mad = float(np.median(abs_dev))
    robust_sigma = max(1e-4, 1.4826 * mad)
    z = abs_dev / robust_sigma

    strong_dev = np.clip((z - 2.2) / 6.0, 0.0, 1.0).astype(np.float32, copy=False)
    strong_dev = gaussian_filter(strong_dev, sigma=1.1)
    strong_dev_n = _robust_norm(strong_dev, lo=5.0, hi=99.3)

    pos = gaussian_filter(np.clip(dev, 0.0, None), sigma=2.0)
    neg = gaussian_filter(np.clip(-dev, 0.0, None), sigma=2.0)
    bipolar = 2.0 * np.minimum(pos, neg) / (pos + neg + 1e-6)
    bipolar_n = _robust_norm(np.clip(bipolar, 0.0, 1.0), lo=5.0, hi=99.0)

    isotropy, grad = _structure_isotropy(fiber, sigma_tensor=2.2)
    iso_detail = _robust_norm(isotropy * _robust_norm(gaussian_filter(grad, sigma=1.2)), lo=5.0, hi=99.0)

    raw = _robust_norm(
        0.70 * strong_dev_n + 0.20 * (strong_dev_n * bipolar_n) + 0.10 * iso_detail,
        lo=2.0,
        hi=99.0,
    )

    seed_score = strong_dev_n * (0.60 + 0.40 * bipolar_n)
    seed_percentile = 99.4
    seed_threshold = float(np.percentile(seed_score, seed_percentile))
    seed = seed_score >= seed_threshold
    coverage = float(np.mean(seed))

    if coverage < 0.0012:
        seed_percentile = 98.8
        seed_threshold = float(np.percentile(seed_score, seed_percentile))
        seed = seed_score >= seed_threshold
        coverage = float(np.mean(seed))
    if coverage > 0.04:
        seed_percentile = 99.8
        seed_threshold = float(np.percentile(seed_score, seed_percentile))
        seed = seed_score >= seed_threshold
        coverage = float(np.mean(seed))

    seed = binary_dilation(seed, structure=np.ones((3, 3), dtype=bool), iterations=1)
    dist = distance_transform_edt(~seed)
    sigma_px = max(6.0, 0.024 * float(max(seed.shape)))
    surround = np.exp(-(dist ** 2) / (2.0 * sigma_px * sigma_px))
    surround = _robust_norm(surround.astype(np.float32, copy=False), lo=2.0, hi=99.0)

    prior = _robust_norm(0.84 * raw + 0.16 * (surround * raw), lo=2.0, hi=99.0)
    prior = gaussian_filter(prior, sigma=0.9)

    gate = np.clip(0.30 + 0.70 * (0.75 * strong_dev_n + 0.25 * bipolar_n), 0.0, 1.0)
    prior = np.clip(prior * gate, 0.0, 1.0).astype(np.float32, copy=False)

    diag = {
        "baseline": float(baseline),
        "mad": float(mad),
        "robust_sigma": float(robust_sigma),
        "seed_percentile": float(seed_percentile),
        "seed_threshold": float(seed_threshold),
        "seed_coverage": float(np.mean(seed)),
        "prior_mean": float(np.mean(prior)),
        "prior_p95": float(np.percentile(prior, 95.0)),
        "prior_p99": float(np.percentile(prior, 99.0)),
        "strong_dev_p95": float(np.percentile(strong_dev_n, 95.0)),
        "bipolar_p95": float(np.percentile(bipolar_n, 95.0)),
    }
    return prior, diag


def _colormap_turbo_like(x: np.ndarray) -> np.ndarray:
    x = np.clip(x.astype(np.float32), 0.0, 1.0)
    r = np.clip(1.52 * x - 0.22, 0.0, 1.0)
    g = np.clip(1.85 * (1.0 - np.abs(x - 0.50) * 1.78), 0.0, 1.0)
    b = np.clip(1.35 * (0.82 - x), 0.0, 1.0)
    return np.stack([r, g, b], axis=-1)


def _to_u8_gray(arr01: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(arr01 * 255.0), 0, 255).astype(np.uint8)


def _to_u8_rgb(arr01: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(arr01 * 255.0), 0, 255).astype(np.uint8)


def _save_preview_png(arr: np.ndarray, out_path: Path, preview_size: int) -> None:
    img = Image.fromarray(arr)
    if int(preview_size) > 0 and img.size != (preview_size, preview_size):
        img = img.resize((preview_size, preview_size), resample=Image.BICUBIC)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, format="PNG", optimize=False)


def _build_html(samples: Sequence[Dict[str, object]], meta: Dict[str, object]) -> str:
    samples_json = json.dumps(list(samples), separators=(",", ":"))
    meta_json = json.dumps(meta, separators=(",", ":"))
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>Fiber-Only Knot Map Review</title>
  <style>
    :root {{
      --bg: #0b1016;
      --panel: #131b25;
      --ink: #e6edf3;
      --muted: #9db0c4;
      --line: #2c3745;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--ink);
      background:
        radial-gradient(1200px 660px at 8% -12%, #1a2838 0%, var(--bg) 45%),
        linear-gradient(165deg, #0b1016 0%, #0a0e14 100%);
      font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
      padding: 22px;
    }}
    .top {{
      display: grid;
      gap: 10px;
      margin-bottom: 16px;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 14px;
    }}
    h1 {{ margin: 0; font-size: 20px; }}
    .sub {{ color: var(--muted); font-size: 13px; line-height: 1.4; }}
    .stats {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      font-size: 12px;
    }}
    .chip {{
      border: 1px solid #314355;
      background: #111824;
      border-radius: 999px;
      padding: 5px 10px;
      color: #cce6ff;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(520px, 1fr));
      gap: 12px;
    }}
    .card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 10px;
      display: grid;
      gap: 8px;
    }}
    .card h3 {{
      margin: 0;
      font-size: 14px;
      font-weight: 600;
      color: #d8ecff;
    }}
    .meta {{
      color: var(--muted);
      font-size: 12px;
      line-height: 1.35;
    }}
    .imgs {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 8px;
    }}
    figure {{
      margin: 0;
      display: grid;
      gap: 4px;
    }}
    img {{
      width: 100%;
      aspect-ratio: 1 / 1;
      object-fit: cover;
      border-radius: 8px;
      border: 1px solid #2f3c4d;
      background: #fff;
    }}
    figcaption {{
      color: #b7cadc;
      font-size: 11px;
      text-align: center;
    }}
  </style>
</head>
<body>
  <section class="top">
    <h1>Fiber-Only Knot Map Review</h1>
    <div class="sub">
      Knot map inferred from fiber images only.<br/>
      Signal: deviation from clear-wood gray baseline (white/black shifts) + local bright/dark coexistence + isotropy cue.
    </div>
    <div class="stats" id="stats"></div>
  </section>
  <section class="grid" id="grid"></section>
<script>
const META = {meta_json};
const SAMPLES = {samples_json};
const statsEl = document.getElementById("stats");
const gridEl = document.getElementById("grid");

function addChip(text) {{
  const s = document.createElement("span");
  s.className = "chip";
  s.textContent = text;
  statsEl.appendChild(s);
}}

addChip("data_root: " + META.data_root);
addChip("samples: " + String(META.num_samples));
addChip("seed: " + String(META.seed));
addChip("faces: " + String(META.num_faces));
addChip("avg baseline: " + Number(META.avg_baseline).toFixed(4));
addChip("avg seed coverage: " + Number(META.avg_seed_coverage).toFixed(4));
addChip("avg prior mean: " + Number(META.avg_prior_mean).toFixed(4));
addChip("avg prior p95: " + Number(META.avg_prior_p95).toFixed(4));

for (const s of SAMPLES) {{
  const card = document.createElement("article");
  card.className = "card";

  const title = document.createElement("h3");
  title.textContent =
    "#" + String(s.sample_index).padStart(3, "0")
    + "  stem=" + s.stem
    + "  face=" + String(s.face_idx);
  card.appendChild(title);

  const meta = document.createElement("div");
  meta.className = "meta";
  meta.textContent =
    "base=" + s.baseline.toFixed(4)
    + "  mad=" + s.mad.toFixed(4)
    + "  seed_pct=" + s.seed_percentile.toFixed(1)
    + "  seed_cov=" + s.seed_coverage.toFixed(4)
    + "  prior_p95=" + s.prior_p95.toFixed(4)
    + "  bipolar_p95=" + s.bipolar_p95.toFixed(4);
  card.appendChild(meta);

  const imgs = document.createElement("div");
  imgs.className = "imgs";
  const entries = [
    ["fiber", s.fiber_rel],
    ["knot map", s.prior_rel],
    ["overlay", s.overlay_rel],
  ];
  for (const [label, rel] of entries) {{
    const fig = document.createElement("figure");
    const im = document.createElement("img");
    im.src = rel;
    im.loading = "lazy";
    im.alt = label + " " + s.stem + " face " + String(s.face_idx);
    const cap = document.createElement("figcaption");
    cap.textContent = label;
    fig.appendChild(im);
    fig.appendChild(cap);
    imgs.appendChild(fig);
  }}
  card.appendChild(imgs);
  gridEl.appendChild(card);
}}
</script>
</body>
</html>
"""


def _collect_fiber_samples(data_root: Path, fiber_prefix: str, num_faces: int) -> List[FiberSample]:
    fiber_maps: Dict[int, Dict[str, Path]] = {}
    for idx in range(1, num_faces + 1):
        fiber_dir = data_root / f"{fiber_prefix}{idx}"
        fiber_maps[idx] = _iter_images(fiber_dir)
        if not fiber_maps[idx]:
            raise RuntimeError(f"No images found in fiber folder: {fiber_dir}")

    common_stems = set.intersection(*[set(m.keys()) for m in fiber_maps.values()])
    if not common_stems:
        raise RuntimeError("No common stems across requested fiber face folders.")

    samples: List[FiberSample] = []
    for stem in sorted(common_stems):
        for idx in range(1, num_faces + 1):
            samples.append(FiberSample(stem=stem, face_idx=idx, fiber_path=fiber_maps[idx][stem]))
    return samples


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build an HTML gallery reviewing knot maps derived from fiber images only."
    )
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--fiber-prefix", type=str, default="fiber_")
    parser.add_argument("--num-faces", type=int, default=4)
    parser.add_argument("--num-samples", type=int, default=240)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--preview-size", type=int, default=320)
    parser.add_argument("--output-html", type=str, default="./runs/knot_fiber_review/index.html")
    args = parser.parse_args()

    data_root = Path(args.data_root).expanduser().resolve()
    if not data_root.is_dir():
        raise SystemExit(f"[knot-fiber] data root does not exist: {data_root}")

    num_faces = max(1, int(args.num_faces))
    num_samples = max(1, int(args.num_samples))
    preview_size = max(64, int(args.preview_size))
    rng = random.Random(int(args.seed))

    all_samples = _collect_fiber_samples(
        data_root=data_root,
        fiber_prefix=str(args.fiber_prefix),
        num_faces=num_faces,
    )
    if num_samples >= len(all_samples):
        chosen = list(all_samples)
    else:
        chosen = rng.sample(all_samples, num_samples)
    chosen = sorted(chosen, key=lambda p: (p.stem, p.face_idx))

    output_html = Path(args.output_html).expanduser().resolve()
    out_dir = output_html.parent
    assets_dir = out_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    samples: List[Dict[str, object]] = []
    seed_cov: List[float] = []
    prior_means: List[float] = []
    prior_p95: List[float] = []
    baselines: List[float] = []

    for i, sample in enumerate(chosen, start=1):
        fiber = _load_gray(sample.fiber_path)
        prior, diag = build_knot_map_from_fiber(fiber)

        heat = _colormap_turbo_like(prior)
        fiber_rgb = np.repeat(fiber[..., np.newaxis], 3, axis=2)
        alpha = np.clip(prior ** 0.88, 0.0, 1.0) * 0.78
        overlay = fiber_rgb * (1.0 - alpha[..., np.newaxis]) + heat * alpha[..., np.newaxis]

        base = f"s{i:04d}_{sample.stem}_f{sample.face_idx}"
        fiber_rel = Path("assets") / f"{base}_fiber.png"
        prior_rel = Path("assets") / f"{base}_knotmap.png"
        overlay_rel = Path("assets") / f"{base}_overlay.png"

        _save_preview_png(_to_u8_gray(fiber), assets_dir / fiber_rel.name, preview_size=preview_size)
        _save_preview_png(_to_u8_gray(prior), assets_dir / prior_rel.name, preview_size=preview_size)
        _save_preview_png(_to_u8_rgb(overlay), assets_dir / overlay_rel.name, preview_size=preview_size)

        seed_cov.append(float(diag["seed_coverage"]))
        prior_means.append(float(diag["prior_mean"]))
        prior_p95.append(float(diag["prior_p95"]))
        baselines.append(float(diag["baseline"]))

        samples.append(
            {
                "sample_index": int(i),
                "stem": sample.stem,
                "face_idx": int(sample.face_idx),
                "fiber_rel": str(fiber_rel).replace("\\", "/"),
                "prior_rel": str(prior_rel).replace("\\", "/"),
                "overlay_rel": str(overlay_rel).replace("\\", "/"),
                "baseline": float(diag["baseline"]),
                "mad": float(diag["mad"]),
                "robust_sigma": float(diag["robust_sigma"]),
                "seed_percentile": float(diag["seed_percentile"]),
                "seed_threshold": float(diag["seed_threshold"]),
                "seed_coverage": float(diag["seed_coverage"]),
                "prior_mean": float(diag["prior_mean"]),
                "prior_p95": float(diag["prior_p95"]),
                "prior_p99": float(diag["prior_p99"]),
                "strong_dev_p95": float(diag["strong_dev_p95"]),
                "bipolar_p95": float(diag["bipolar_p95"]),
            }
        )

    meta = {
        "data_root": str(data_root),
        "fiber_prefix": str(args.fiber_prefix),
        "num_faces": int(num_faces),
        "num_samples": int(len(samples)),
        "seed": int(args.seed),
        "preview_size": int(preview_size),
        "avg_baseline": float(np.mean(baselines)) if baselines else 0.0,
        "avg_seed_coverage": float(np.mean(seed_cov)) if seed_cov else 0.0,
        "avg_prior_mean": float(np.mean(prior_means)) if prior_means else 0.0,
        "avg_prior_p95": float(np.mean(prior_p95)) if prior_p95 else 0.0,
    }

    html = _build_html(samples=samples, meta=meta)
    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_html.write_text(html, encoding="utf-8")
    (out_dir / "summary.json").write_text(
        json.dumps({"meta": meta, "samples": samples}, indent=2),
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "output_html": str(output_html),
                "output_dir": str(out_dir),
                "num_samples": int(len(samples)),
                "data_root": str(data_root),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
