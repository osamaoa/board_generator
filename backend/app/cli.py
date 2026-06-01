from __future__ import annotations

import argparse
import json
import sys

import numpy as np


def _add_optional_bool_arg(parser: argparse.ArgumentParser, flag: str, help_text: str) -> None:
    parser.add_argument(
        flag,
        type=str,
        choices=["true", "false", "1", "0", "yes", "no", "on", "off"],
        default=None,
        help=help_text,
    )


def _as_bool_or_default(value: str | None, default: bool) -> bool:
    if value is None:
        return bool(default)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return bool(default)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="board-cli",
        description="CLI for board-generator tasks that are outside the web UI.",
    )

    top = parser.add_subparsers(dest="group", required=True)

    diffusion = top.add_parser("diffusion", help="Diffusion-model tasks.")
    diffusion_sub = diffusion.add_subparsers(dest="command", required=True)

    train = diffusion_sub.add_parser(
        "train",
        help=(
            "Train a new photorealistic diffusion model from conditioning maps to RGB pairs "
            "(ring+fiber by default, or ring-only with --use-rings-only)."
        ),
    )

    train.add_argument(
        "--data-root",
        type=str,
        default="/mnt/e/SCRATCH/Osama/RGB_for_labeling/images",
        help=(
            "Dataset root containing color_1..4 and ring_pred_new_1..4. "
            "fiber_1..4 is required unless --use-rings-only=true."
        ),
    )
    train.add_argument(
        "--output-dir",
        type=str,
        default="./runs/wood_sd2_concat_joint_v2",
        help="Directory for training states and exported checkpoints.",
    )
    train.add_argument(
        "--sd2-model-dir",
        type=str,
        default="",
        help="Local SD2 folder (defaults to repo/SD2_model or PHOTOREALISTIC_SD2_MODEL_DIR).",
    )
    train.add_argument(
        "--sdxl-model-dir",
        type=str,
        default="",
        help="Local SDXL folder (defaults to repo/SDXL_model or PHOTOREALISTIC_SDXL_MODEL_DIR).",
    )
    train.add_argument(
        "--model-family",
        type=str,
        choices=["sd2", "sdxl"],
        default="sd2",
        help="Base diffusion family used for photorealistic training.",
    )

    train.add_argument("--image-size", type=int, default=512)
    train.add_argument("--boards-per-batch", type=int, default=2)
    train.add_argument("--grad-accum-steps", type=int, default=1)
    train.add_argument("--num-workers", type=int, default=4)
    train.add_argument(
        "--train-pin-memory",
        type=str,
        default="false",
        help="Training dataloader pin_memory true/false (default: false).",
    )

    train.add_argument("--max-train-steps", type=int, default=325000)
    train.add_argument("--lr", type=float, default=2e-5)
    train.add_argument("--weight-decay", type=float, default=0.0)
    train.add_argument(
        "--optimizer",
        type=str,
        choices=["auto", "adamw", "adafactor"],
        default="auto",
        help="Optimizer for UNet/null-token training. 'auto' picks Adafactor for SDXL and AdamW for SD2.",
    )
    train.add_argument("--num-warmup-steps", type=int, default=2000)
    train.add_argument("--max-grad-norm", type=float, default=1.0)

    train.add_argument("--guidance-drop-prob", type=float, default=0.1)
    train.add_argument(
        "--include-knot-maps",
        type=str,
        default="false",
        help=(
            "If true, derive a knot map from each fiber image and use ring+fiber+knot "
            "as conditioning channels during training (true/false). "
            "Requires --use-rings-only=false."
        ),
    )
    train.add_argument(
        "--use-rings-only",
        type=str,
        default="false",
        help=(
            "If true, train with ring maps only (fiber channel is omitted/zeroed). "
            "Cannot be combined with --include-knot-maps=true."
        ),
    )
    train.add_argument("--num-train-timesteps", type=int, default=1000)
    train.add_argument(
        "--prediction-type",
        type=str,
        choices=["epsilon", "v_prediction"],
        default="epsilon",
        help="Must match inference scheduler prediction type.",
    )
    train.add_argument(
        "--min-snr-gamma",
        type=float,
        default=5.0,
        help="Set 0 to disable Min-SNR reweighting.",
    )
    train.add_argument(
        "--latent-recon-weight",
        type=float,
        default=0.1,
        help="Weight for latent x0 reconstruction loss term.",
    )
    train.add_argument(
        "--cross-surface-consistency-weight",
        type=float,
        default=0.0,
        help=(
            "Optional board-level cross-face consistency loss weight. "
            "0.0 keeps current behavior; small values like 0.02-0.10 can improve face-to-face coherence."
        ),
    )
    train.add_argument(
        "--seam-consistency-weight",
        type=float,
        default=0.0,
        help=(
            "Optional seam-aware loss weight for adjacent face borders. "
            "0.0 disables it; start around 0.02-0.08."
        ),
    )
    train.add_argument(
        "--seam-strip-width",
        type=int,
        default=2,
        help="Latent border-strip width for seam consistency (pixels in latent space).",
    )
    train.add_argument(
        "--shared-board-noise",
        type=str,
        default="false",
        help="Use shared diffusion noise across the 4 faces of each board during training (true/false).",
    )

    train.add_argument("--ema-decay", type=float, default=0.9999)
    train.add_argument("--ema-update-every", type=int, default=1)
    train.add_argument(
        "--use-ema",
        type=str,
        default="auto",
        help="Enable EMA shadow weights during training/export (true/false/auto).",
    )

    train.add_argument("--val-ratio", type=float, default=0.1)
    train.add_argument("--validate-every", type=int, default=1000)
    train.add_argument("--val-boards", type=int, default=2)
    train.add_argument("--val-ddim-steps", type=int, default=30)
    train.add_argument("--val-guidance-scale", type=float, default=1.5)
    train.add_argument(
        "--val-num-workers",
        type=int,
        default=0,
        help="Validation dataloader workers. 0 is safest for memory stability.",
    )
    train.add_argument(
        "--val-pin-memory",
        type=str,
        default="false",
        help="Validation dataloader pin_memory true/false.",
    )
    train.add_argument(
        "--validate-on-main-process-only",
        type=str,
        default="true",
        help="Run validation only on rank0 in multi-GPU (true/false).",
    )

    train.add_argument("--export-every", type=int, default=5000)
    train.add_argument("--export-ddim-steps", type=int, default=50)
    train.add_argument("--export-guidance-scale", type=float, default=1.5)
    train.add_argument("--export-img2img-strength", type=float, default=0.0)

    train.add_argument(
        "--mixed-precision",
        type=str,
        choices=["no", "fp16", "bf16"],
        default="fp16",
    )
    train.add_argument(
        "--enable-gradient-checkpointing",
        type=str,
        default="true",
        help="true/false",
    )
    train.add_argument(
        "--enable-xformers",
        type=str,
        default="true",
        help="true/false",
    )

    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--resume-state-dir", type=str, default="")
    train.add_argument(
        "--show-cli-progress",
        type=str,
        default="true",
        help="Print training progress in terminal while training (true/false).",
    )
    train.add_argument(
        "--cli-progress-every",
        type=int,
        default=10,
        help="Print one CLI progress update every N optimizer steps.",
    )

    train.add_argument(
        "--disable-wandb",
        action="store_true",
        help="Disable Weights & Biases logging.",
    )
    train.add_argument("--wandb-project", type=str, default="wood-sd2-concat")
    train.add_argument("--wandb-run", type=str, default="osama-joint-photorealistic-v2")
    train.add_argument(
        "--wandb-mode",
        type=str,
        choices=["online", "offline", "disabled"],
        default="online",
    )

    knots = top.add_parser(
        "knots",
        help="Knot-sequence model tasks (dictionary data prep, training, and sampling).",
    )
    knots_sub = knots.add_subparsers(dest="command", required=True)

    prepare = knots_sub.add_parser(
        "prepare-data",
        help="Build knot-sequence training_data MAT from raw logs_data/*.mat files.",
    )
    prepare.add_argument(
        "--logs-dir",
        type=str,
        default="./.old_knot_generator/logs_data",
        help="Folder containing knot_data_*.mat files.",
    )
    prepare.add_argument(
        "--output-mat-path",
        type=str,
        default="./knot_model_checkpoint/training_data_new_2025.mat",
        help="Output training_data MAT path.",
    )
    prepare.add_argument("--dz", type=float, default=10.0, help="Axial slot size in mm (default: 10).")
    prepare.add_argument(
        "--n-samples",
        type=int,
        default=101,
        help="Sequence length used for training segments (default: 101).",
    )
    prepare.add_argument(
        "--n-overlap",
        type=int,
        default=97,
        help="Overlap between consecutive segments (default: 97).",
    )
    prepare.add_argument(
        "--cluster-count",
        type=int,
        default=512,
        help="Number of knot clusters (0 disables clustering and keeps direct knot IDs).",
    )
    prepare.add_argument(
        "--cluster-seed",
        type=int,
        default=42,
        help="Random seed for knot dictionary clustering.",
    )
    prepare.add_argument(
        "--cluster-max-iter",
        type=int,
        default=40,
        help="Max k-means iterations for dictionary clustering.",
    )

    knot_train = knots_sub.add_parser(
        "train",
        help="Train a PyTorch LSTM knot-sequence model from training_data MAT.",
    )
    knot_train.add_argument(
        "--training-mat-path",
        type=str,
        default="./knot_model_checkpoint/training_data_new_2025.mat",
        help="Path to training_data MAT (contains inps, outs, and embedding matrix).",
    )
    knot_train.add_argument(
        "--output-checkpoint-path",
        type=str,
        default="./knot_model_checkpoint/knot_sequence_model.pt",
        help="Output model checkpoint path.",
    )
    knot_train.add_argument(
        "--output-history-path",
        type=str,
        default="",
        help="Optional JSON history path (default: same as checkpoint with .json).",
    )
    knot_train.add_argument("--hidden-size", type=int, default=128)
    knot_train.add_argument("--num-layers", type=int, default=1)
    knot_train.add_argument("--dropout", type=float, default=0.0)
    knot_train.add_argument("--batch-size", type=int, default=64)
    knot_train.add_argument("--epochs", type=int, default=60)
    knot_train.add_argument("--learning-rate", type=float, default=1e-3)
    knot_train.add_argument("--weight-decay", type=float, default=0.0)
    knot_train.add_argument("--grad-clip", type=float, default=1.0)
    knot_train.add_argument("--val-ratio", type=float, default=0.1)
    _add_optional_bool_arg(
        knot_train,
        "--early-stop-enabled",
        "Enable early stopping (default: true).",
    )
    knot_train.add_argument("--early-stop-patience", type=int, default=8)
    knot_train.add_argument(
        "--early-stop-min-delta",
        type=float,
        default=0.0,
        help="Minimum monitored metric improvement required to reset early-stop patience.",
    )
    knot_train.add_argument(
        "--early-stop-monitor",
        type=str,
        default="val_loss",
        choices=["val_loss", "train_loss", "val_acc", "train_acc"],
        help="Metric tracked for best checkpoint and early stopping.",
    )
    knot_train.add_argument("--seed", type=int, default=42)
    knot_train.add_argument(
        "--no-knot-weight",
        type=float,
        default=0.35,
        help="Class weight for token 0 (no-knot); lower gives relatively more weight to non-zero knot IDs.",
    )
    knot_train.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    _add_optional_bool_arg(
        knot_train,
        "--freeze-embedding",
        "Freeze embedding matrix during training (default: true).",
    )

    sample = knots_sub.add_parser(
        "sample",
        help="Sample a fresh knot token sequence from the trained model.",
    )
    sample.add_argument("--length", type=int, default=400, help="Total tokens to sample.")
    sample.add_argument("--top-k", type=int, default=0)
    sample.add_argument(
        "--top-p",
        type=float,
        default=0.8,
        help="Nucleus sampling threshold in (0,1]; 0 disables top-p filtering.",
    )
    sample.add_argument("--seed", type=int, default=None, help="Optional RNG seed for deterministic sampling.")
    sample.add_argument(
        "--checkpoint-path",
        type=str,
        default="./knot_model_checkpoint/knot_sequence_model.pt",
        help="PyTorch checkpoint path.",
    )
    sample.add_argument(
        "--training-mat-path",
        type=str,
        default="./knot_model_checkpoint/training_data_new_2025.mat",
        help="Fallback training MAT path (used if checkpoint is unavailable).",
    )
    _add_optional_bool_arg(
        sample,
        "--allow-fallback",
        "Allow fallback Markov sampler when checkpoint is unavailable (default: true).",
    )
    sample.add_argument(
        "--device",
        type=str,
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device for PyTorch checkpoint sampling (default: auto).",
    )
    sample.add_argument(
        "--output-mat-path",
        type=str,
        default="",
        help="Optional path to save sampled sequence as MATLAB file with variable 'log'.",
    )
    sample.add_argument(
        "--print-limit",
        type=int,
        default=40,
        help="Number of initial tokens to print in CLI summary.",
    )

    evaluate = knots_sub.add_parser(
        "evaluate",
        help="Evaluate generated knot sequences against training sequences and export an HTML report.",
    )
    evaluate.add_argument(
        "--training-mat-path",
        type=str,
        default="./knot_model_checkpoint/training_data_new_2025.mat",
        help="Path to training_data MAT used as reference.",
    )
    evaluate.add_argument(
        "--checkpoint-path",
        type=str,
        default="./knot_model_checkpoint/knot_sequence_model.pt",
        help="PyTorch checkpoint path used for sequence sampling.",
    )
    _add_optional_bool_arg(
        evaluate,
        "--allow-fallback",
        "Allow fallback Markov sampler when checkpoint is unavailable (default: true).",
    )
    evaluate.add_argument(
        "--device",
        type=str,
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device for PyTorch checkpoint sampling during evaluation (default: auto).",
    )
    evaluate.add_argument(
        "--num-generated-sequences",
        type=int,
        default=500,
        help="Number of generated sequences to sample for comparison.",
    )
    evaluate.add_argument(
        "--sequence-length",
        type=int,
        default=0,
        help="Generated sequence length. 0 uses reconstructed full training sequence length from MAT.",
    )
    evaluate.add_argument("--top-k", type=int, default=0, help="Top-k filtering during sampling.")
    evaluate.add_argument(
        "--top-p",
        type=float,
        default=0.8,
        help="Nucleus sampling threshold in (0,1]; 0 disables top-p filtering.",
    )
    evaluate.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base RNG seed for generated-sequence sampling and deterministic report decoding.",
    )
    evaluate.add_argument(
        "--output-html-path",
        type=str,
        default="./knot_model_checkpoint/knot_sequence_eval_report.html",
        help="Output HTML report path.",
    )
    evaluate.add_argument(
        "--output-plot-data-mat-path",
        type=str,
        default="",
        help="Output MATLAB data file for train/generated parameter plots. Default: HTML path with _plot_data.mat.",
    )
    evaluate.add_argument(
        "--output-matlab-script-path",
        type=str,
        default="",
        help="Output MATLAB script that recreates histogram and whisker figures. Default: HTML path with _plots.m.",
    )
    evaluate.add_argument(
        "--title",
        type=str,
        default="Knot Sequence Generator Evaluation",
        help="Title shown in the HTML report.",
    )

    boards = top.add_parser("boards", help="Board generation tasks.")
    boards_sub = boards.add_subparsers(dest="command", required=True)

    generate = boards_sub.add_parser(
        "generate",
        help=(
            "Generate many valid boards and export selected image outputs. "
            "Boards outside the log are rejected automatically."
        ),
    )
    generate.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=(
            "Root folder where output-type subfolders (rings_*, fiber_*, photorealistic_*) are written. "
            "If omitted, uses boards_generate.output_dir from --config-json."
        ),
    )
    generate.add_argument(
        "--num-boards",
        type=int,
        default=None,
        help="Number of accepted boards to generate (default: 100).",
    )
    generate.add_argument(
        "--max-attempts",
        type=int,
        default=None,
        help="Maximum attempts including rejected boards (default: auto=20x requested).",
    )
    generate.add_argument(
        "--gpu-workers",
        type=int,
        default=None,
        help=(
            "Parallel GPU workers for boards generation. "
            "Set <=0 (or omit) to auto-use all visible GPUs when use_gpu=true."
        ),
    )
    generate.add_argument(
        "--min-knot-count",
        type=int,
        default=None,
        help=(
            "Minimum required knot count per generated board; "
            "boards with fewer knots are rejected and retried (default: 0)."
        ),
    )
    generate.add_argument(
        "--outputs",
        type=str,
        default=None,
        help=(
            "Comma-separated outputs: rings, fibers, middle, top_bottom, photorealistic. "
            "Use 'all' for every output. Default: rings,fibers,middle,top_bottom."
        ),
    )
    generate.add_argument("--image-size", type=int, default=None, help="PNG resolution per side (default: 512).")
    generate.add_argument("--imid-start", type=int, default=None, help="Start index for output image filenames (default: 1).")
    generate.add_argument(
        "--contour-line-width",
        type=str,
        default=None,
        help=(
            "Contour line width in px. Use a single value (e.g. 1.0) or a range a,b "
            "(e.g. 1.0,3.0) to sample once per batch chunk."
        ),
    )
    generate.add_argument("--contour-blur-sigma", type=float, default=None)
    generate.add_argument(
        "--fiber-blur-sigma",
        type=str,
        default=None,
        help=(
            "Fiber blur sigma. Use a single value (e.g. 0.6) or a range a,b "
            "(e.g. 0.2,0.8) to sample once per batch chunk."
        ),
    )
    generate.add_argument(
        "--fiber-irregularity-strength",
        type=str,
        default=None,
        help=(
            "Extra non-Gaussian irregularity for generated fiber maps. "
            "Use a single value (e.g. 0.35) or a range a,b (e.g. 0.2,0.6) to sample once per batch chunk. "
            "Default: 0.35."
        ),
    )
    generate.add_argument(
        "--ring-irregularity-strength",
        type=str,
        default=None,
        help=(
            "Hand-drawn realism perturbation for generated ring maps. "
            "Use a single value (e.g. 0.40) or a range a,b (e.g. 0.2,0.6) to sample once per batch chunk. "
            "Default: 0.40."
        ),
    )
    generate.add_argument(
        "--show-rings-inside-knots",
        type=str,
        default=None,
        choices=["true", "false", "1", "0", "yes", "no", "on", "off"],
        help="If true, do not mask rings inside knots for ring exports (default: false).",
    )

    generate.add_argument(
        "--config-json",
        type=str,
        default="",
        help=(
            "Optional JSON file for board generation. "
            "If it contains {\"config\": {...}}, that inner object is used. "
            "UI/3D-only fields are ignored."
        ),
    )
    generate.add_argument(
        "--manual-knots-json",
        type=str,
        default="",
        help=(
            "Optional JSON for manual knots (list of knot objects or {\"input_knots\": [...]}) "
            "to populate BoardConfig.input_knots."
        ),
    )

    # Geometry / mesh
    generate.add_argument("--board-x-min", type=float, default=None)
    generate.add_argument("--board-x-max", type=float, default=None)
    generate.add_argument("--board-y-min", type=float, default=None)
    generate.add_argument("--board-y-max", type=float, default=None)
    generate.add_argument("--board-z-min", type=float, default=None)
    generate.add_argument("--board-z-max", type=float, default=None)
    generate.add_argument(
        "--board-width-mm",
        type=float,
        default=None,
        help="Board size along X (use with --board-thickness-mm and --board-length-mm; mutually exclusive with board extents).",
    )
    generate.add_argument(
        "--board-thickness-mm",
        type=float,
        default=None,
        help="Board size along Y (use with --board-width-mm and --board-length-mm; mutually exclusive with board extents).",
    )
    generate.add_argument(
        "--board-length-mm",
        type=float,
        default=None,
        help="Board size along Z (use with --board-width-mm and --board-thickness-mm; mutually exclusive with board extents).",
    )
    generate.add_argument("--mesh-size-x-mm", type=float, default=None)
    generate.add_argument("--mesh-size-y-mm", type=float, default=None)
    generate.add_argument("--mesh-size-z-mm", type=float, default=None)
    _add_optional_bool_arg(
        generate,
        "--randomize-crook-taper",
        "Enable stochastic crook/taper sampling per generated board (default: true).",
    )
    generate.add_argument(
        "--crook-component-count",
        type=int,
        default=None,
        help="Number of sinusoidal crook components p (default: 8).",
    )
    generate.add_argument(
        "--crook-shift-max-mm",
        type=float,
        default=None,
        help="Maximum longitudinal phase shift z0_i in mm for crook components (default: 8000).",
    )
    generate.add_argument(
        "--random-crook-amplitudes-max",
        type=str,
        default=None,
        help=(
            "Comma-separated per-component amplitude maxima for random mode "
            "(default from legacy MATLAB script: 50,25,12.5,5,2.5,1.25,0.625,0.3125)."
        ),
    )
    generate.add_argument(
        "--random-crook-extra-orders",
        type=str,
        default=None,
        help=(
            "Comma-separated extra random crook orders appended after 1..p "
            "(for example: 9,10)."
        ),
    )
    generate.add_argument(
        "--random-crook-theta-min-deg",
        type=float,
        default=None,
        help="Lower bound for random theta_i in degrees (default: 0).",
    )
    generate.add_argument(
        "--random-crook-theta-max-deg",
        type=float,
        default=None,
        help="Upper bound for random theta_i in degrees (default: 360).",
    )
    generate.add_argument(
        "--manual-crook-amplitudes",
        type=str,
        default=None,
        help="Comma-separated manual amplitudes a_i (used when --randomize-crook-taper=false).",
    )
    generate.add_argument(
        "--manual-crook-shifts-mm",
        type=str,
        default=None,
        help="Comma-separated manual z0_i shifts in mm (used when --randomize-crook-taper=false).",
    )
    generate.add_argument(
        "--manual-crook-thetas-deg",
        type=str,
        default=None,
        help="Comma-separated manual theta_i in degrees (used when --randomize-crook-taper=false).",
    )
    generate.add_argument(
        "--manual-crook-orders",
        type=str,
        default=None,
        help=(
            "Comma-separated manual per-component orders (used when --randomize-crook-taper=false). "
            "If fewer values than p are provided, missing entries fall back to 1..p by index."
        ),
    )
    generate.add_argument("--manual-crook-x-coeff", type=float, default=None)
    generate.add_argument("--manual-crook-y-coeff", type=float, default=None)
    generate.add_argument("--manual-taper-coeff", type=float, default=None)
    generate.add_argument(
        "--random-crook-scale-max",
        "--random-crook-abs-max",
        dest="random_crook_scale_max",
        type=float,
        default=None,
        help="Global multiplier applied to random crook amplitude maxima (default: 1.0).",
    )
    generate.add_argument(
        "--random-taper-max",
        type=float,
        default=None,
        help="Max for per-board random taper coefficient sampled from U(0, max) (default: 0.00625).",
    )

    # Knots
    generate.add_argument("--input-knot-count", type=int, default=None)
    generate.add_argument("--knot-inside-limit", type=float, default=None)
    generate.add_argument(
        "--knot-generator-min-rd-minus-rl-mm",
        type=float,
        default=None,
        help=(
            "Minimum RD-RL (mm) for sequence-generated knots. "
            "If smaller, RL is shifted to RD-minus-this-value (default: 30)."
        ),
    )
    generate.add_argument("--l100-min", type=float, default=None)
    generate.add_argument("--l100-max", type=float, default=None)
    generate.add_argument("--soft-clamp-alpha", type=float, default=None)
    generate.add_argument("--soft-clamp-pmin", type=float, default=None)
    generate.add_argument("--knot-seq-top-k", type=int, default=None)
    generate.add_argument(
        "--knot-seq-top-p",
        type=float,
        default=None,
        help="Nucleus sampling threshold in (0,1] for knot sequence generation; 0 disables top-p.",
    )
    generate.add_argument("--knot-seq-min-tokens", type=int, default=None)
    generate.add_argument("--knot-seq-extra-tokens", type=int, default=None)
    generate.add_argument("--knot-seq-checkpoint-path", type=str, default=None)
    generate.add_argument("--knot-seq-training-data-path", type=str, default=None)
    generate.add_argument(
        "--knot-dictionary-jitter",
        type=float,
        default=None,
        help="Small random jitter scale applied to decoded dictionary knot parameters (default: 0.0).",
    )
    _add_optional_bool_arg(
        generate,
        "--knot-seq-override-c1-c2",
        (
            "Override knot c1/c2 with c1=-1.458e-3 and "
            "c2=9.7e-3*Ax100+0.1725 where Ax100 is uniformly sampled in [32.7, 55.3]."
        ),
    )
    _add_optional_bool_arg(
        generate,
        "--knot-seq-allow-fallback",
        "Allow fallback knot-sequence sampling when checkpoint is unavailable.",
    )

    # Fibers
    generate.add_argument("--calc-fibers-a0-method", type=int, default=None, choices=[1, 2])
    generate.add_argument(
        "--multi-knot-fiber-selection-rule",
        type=str,
        default=None,
        choices=["weighted_deviation", "longitudinal"],
        help=(
            "Rule used to choose among candidate fiber fields from multiple knots. "
            "Defaults to weighted_deviation unless the loaded config overrides it."
        ),
    )
    generate.add_argument("--out-of-plane-threshold", type=float, default=None)
    generate.add_argument("--snr", type=float, default=None)

    # Seeding / execution mode
    generate.add_argument("--simulation-seed", type=int, default=None)
    _add_optional_bool_arg(generate, "--use-gpu", "Override use_gpu.")
    _add_optional_bool_arg(generate, "--use-seed", "Override use_seed.")
    _add_optional_bool_arg(generate, "--use-input-knots", "Override use_input_knots.")
    _add_optional_bool_arg(generate, "--include-knot-dev", "Override include_knot_dev.")
    _add_optional_bool_arg(generate, "--dead-knots", "Override dead_knots.")
    _add_optional_bool_arg(
        generate,
        "--knot-fiber-field-override",
        "Override knot_fiber_field_override.",
    )
    _add_optional_bool_arg(
        generate,
        "--knot-fiber-disable-dead-override",
        "Override knot_fiber_disable_dead_override.",
    )
    _add_optional_bool_arg(
        generate,
        "--knot-fiber-reverse-above-axis",
        "Override knot_fiber_reverse_above_axis.",
    )
    _add_optional_bool_arg(generate, "--rand-fibers", "Override rand_fibers.")

    # Photorealistic inference
    generate.add_argument("--photorealistic-ddim-steps", type=int, default=None)
    generate.add_argument(
        "--photorealistic-guidance-scale",
        type=str,
        default=None,
        help=(
            "CFG guidance scale. Use a single value (e.g. 4.0) or a range a,b "
            "(e.g. 3.0,5.0) to sample once per diffusion batch chunk."
        ),
    )
    generate.add_argument(
        "--photorealistic-img2img-strength",
        type=str,
        default=None,
        help=(
            "Img2img strength in [0,1]. Use a single value (e.g. 0.2) or a range a,b "
            "(e.g. 0.1,0.4). You can also pass a discrete list a,b,c,... "
            "(e.g. 0.0,0.1,0.2) to sample one value per diffusion batch chunk."
        ),
    )
    generate.add_argument(
        "--photorealistic-batch-size",
        type=int,
        default=None,
        help="Boards per diffusion batch during photorealistic generation (default: 4).",
    )
    _add_optional_bool_arg(
        generate,
        "--photorealistic-include-knot-maps",
        "If true, include a fiber-derived knot map channel in photorealistic conditioning.",
    )
    _add_optional_bool_arg(
        generate,
        "--photorealistic-use-rings-only",
        "If true, run photorealistic inference using only ring maps (no fiber inputs).",
    )
    regenerate_photo = boards_sub.add_parser(
        "regenerate-photorealistic",
        help=(
            "Generate photorealistic_1..4 for existing boards using rings_1..4 "
            "(and optionally fiber_1..4 unless rings-only mode is enabled)."
        ),
    )
    regenerate_photo.add_argument(
        "--data-root",
        type=str,
        default=None,
        help=(
            "Existing board dataset root containing rings_1..4 "
            "(and fiber_1..4 unless --photorealistic-use-rings-only=true). "
            "If omitted, uses boards_generate.output_dir from --config-json."
        ),
    )
    regenerate_photo.add_argument(
        "--config-json",
        type=str,
        default="",
        help=(
            "Optional JSON file in the same format as boards generate. "
            "Only relevant photorealistic settings are used."
        ),
    )
    regenerate_photo.add_argument(
        "--stems",
        type=str,
        default="",
        help="Optional comma-separated filename stems to process (e.g. 00001,00002).",
    )
    regenerate_photo.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional max number of stems to process after filtering/order.",
    )
    regenerate_photo.add_argument(
        "--overwrite",
        type=str,
        default=None,
        choices=["true", "false", "1", "0", "yes", "no", "on", "off"],
        help="Overwrite existing photorealistic_1..4 files for selected stems (default: true).",
    )
    regenerate_photo.add_argument("--photorealistic-ddim-steps", type=int, default=None)
    regenerate_photo.add_argument(
        "--photorealistic-guidance-scale",
        type=str,
        default=None,
        help=(
            "CFG guidance scale. Use a single value (e.g. 4.0) or a range a,b "
            "(e.g. 3.0,5.0) to sample once per diffusion batch chunk."
        ),
    )
    regenerate_photo.add_argument(
        "--photorealistic-img2img-strength",
        type=str,
        default=None,
        help=(
            "Img2img strength in [0,1]. Use a single value (e.g. 0.2) or a range a,b "
            "(e.g. 0.1,0.4). You can also pass a discrete list a,b,c,... "
            "(e.g. 0.0,0.1,0.2) to sample one value per diffusion batch chunk."
        ),
    )
    regenerate_photo.add_argument(
        "--photorealistic-batch-size",
        type=int,
        default=None,
        help="Boards per diffusion batch during photorealistic generation (default: 4).",
    )
    _add_optional_bool_arg(
        regenerate_photo,
        "--photorealistic-include-knot-maps",
        "If true, include a fiber-derived knot map channel in photorealistic conditioning.",
    )
    _add_optional_bool_arg(
        regenerate_photo,
        "--photorealistic-use-rings-only",
        "If true, run photorealistic inference using only ring maps (no fiber inputs).",
    )
    regenerate_photo.add_argument(
        "--fiber-irregularity-strength",
        type=str,
        default=None,
        help=(
            "Extra non-Gaussian irregularity for loaded fiber_1..4 maps before diffusion. "
            "Use a single value or a range a,b sampled once per batch chunk. Default: 0.35."
        ),
    )
    regenerate_photo.add_argument(
        "--ring-irregularity-strength",
        type=str,
        default=None,
        help=(
            "Hand-drawn realism perturbation for loaded rings_1..4 maps before diffusion. "
            "Use a single value or a range a,b sampled once per batch chunk. Default: 0.40."
        ),
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.group == "knots" and args.command == "prepare-data":
        try:
            from app.core.knot_sequence_model import KnotDatasetPrepConfig, prepare_knot_training_data

            cfg = KnotDatasetPrepConfig(
                logs_dir=str(args.logs_dir),
                output_mat_path=str(args.output_mat_path),
                dz=float(args.dz),
                n_samples=int(args.n_samples),
                n_overlap=int(args.n_overlap),
                cluster_count=int(args.cluster_count),
                cluster_seed=int(args.cluster_seed),
                cluster_max_iter=int(args.cluster_max_iter),
            )
            summary = prepare_knot_training_data(cfg)
            print(json.dumps(summary, indent=2))
            return 0
        except (RuntimeError, ImportError, ValueError) as exc:
            print(f"[board-cli] configuration error: {exc}", file=sys.stderr)
            return 2

    if args.group == "knots" and args.command == "train":
        try:
            from app.core.knot_sequence_model import KnotSequenceTrainConfig, train_knot_sequence_model

            cfg = KnotSequenceTrainConfig(
                training_mat_path=str(args.training_mat_path),
                output_checkpoint_path=str(args.output_checkpoint_path),
                output_history_path=str(args.output_history_path),
                hidden_size=int(args.hidden_size),
                num_layers=int(args.num_layers),
                dropout=float(args.dropout),
                batch_size=int(args.batch_size),
                epochs=int(args.epochs),
                learning_rate=float(args.learning_rate),
                weight_decay=float(args.weight_decay),
                grad_clip=float(args.grad_clip),
                val_ratio=float(args.val_ratio),
                early_stop_enabled=_as_bool_or_default(args.early_stop_enabled, True),
                early_stop_patience=int(args.early_stop_patience),
                early_stop_min_delta=float(args.early_stop_min_delta),
                early_stop_monitor=str(args.early_stop_monitor),
                seed=int(args.seed),
                freeze_embedding=_as_bool_or_default(args.freeze_embedding, True),
                no_knot_weight=float(args.no_knot_weight),
                device=str(args.device),
            )
            summary = train_knot_sequence_model(cfg)
            print(json.dumps(summary, indent=2))
            return 0
        except (RuntimeError, ImportError, ValueError) as exc:
            print(f"[board-cli] configuration error: {exc}", file=sys.stderr)
            return 2

    if args.group == "knots" and args.command == "sample":
        try:
            from pathlib import Path

            import scipy.io

            from app.core.knot_sequence_model import get_knot_sequence_runtime

            runtime = get_knot_sequence_runtime(
                checkpoint_path=str(args.checkpoint_path),
                training_mat_path=str(args.training_mat_path),
                allow_fallback=_as_bool_or_default(args.allow_fallback, True),
                device=str(args.device),
            )
            seq = runtime.sample(
                length=max(1, int(args.length)),
                temperature=1.15,
                top_k=max(0, int(args.top_k)),
                top_p=float(args.top_p),
                seed=(None if args.seed is None else int(args.seed)),
            )

            output_mat = str(args.output_mat_path or "").strip()
            if output_mat:
                out_path = Path(output_mat).expanduser().resolve()
                out_path.parent.mkdir(parents=True, exist_ok=True)
                scipy.io.savemat(out_path, {"log": seq.reshape(1, -1)})
            else:
                out_path = None

            preview_len = max(1, int(args.print_limit))
            summary = {
                "mode": runtime.mode(),
                "load_note": runtime.load_note(),
                "length": int(seq.shape[0]),
                "non_zero_count": int(np.count_nonzero(seq)),
                "preview": seq[:preview_len].tolist(),
                "output_mat_path": str(out_path) if out_path is not None else None,
            }
            print(json.dumps(summary, indent=2))
            return 0
        except (RuntimeError, ImportError, ValueError) as exc:
            print(f"[board-cli] configuration error: {exc}", file=sys.stderr)
            return 2

    if args.group == "knots" and args.command == "evaluate":
        try:
            from app.core.knot_sequence_model import KnotSequenceEvalConfig, evaluate_knot_sequence_generator

            cfg = KnotSequenceEvalConfig(
                training_mat_path=str(args.training_mat_path),
                checkpoint_path=str(args.checkpoint_path),
                output_html_path=str(args.output_html_path),
                output_plot_data_mat_path=str(args.output_plot_data_mat_path),
                output_matlab_script_path=str(args.output_matlab_script_path),
                num_generated_sequences=max(1, int(args.num_generated_sequences)),
                sequence_length=int(args.sequence_length),
                top_k=max(0, int(args.top_k)),
                top_p=float(args.top_p),
                allow_fallback=_as_bool_or_default(args.allow_fallback, True),
                device=str(args.device),
                seed=(None if args.seed is None else int(args.seed)),
                title=str(args.title),
            )
            summary = evaluate_knot_sequence_generator(cfg)
            print(json.dumps(summary, indent=2))
            return 0
        except (RuntimeError, ImportError, ValueError) as exc:
            print(f"[board-cli] configuration error: {exc}", file=sys.stderr)
            return 2

    if args.group == "diffusion" and args.command == "train":
        try:
            from app.core.photorealistic_training import config_from_args, train_photorealistic_diffusion

            cfg = config_from_args(args)
            train_photorealistic_diffusion(cfg)
            return 0
        except (RuntimeError, ImportError) as exc:
            print(f"[board-cli] configuration error: {exc}", file=sys.stderr)
            return 2

    if args.group == "boards" and args.command == "generate":
        try:
            from app.core.board_batch_generation import generate_boards_dataset

            generate_boards_dataset(args)
            return 0
        except (RuntimeError, ImportError, ValueError) as exc:
            print(f"[board-cli] configuration error: {exc}", file=sys.stderr)
            return 2

    if args.group == "boards" and args.command == "regenerate-photorealistic":
        try:
            from app.core.board_batch_generation import regenerate_photorealistic_for_existing_boards

            regenerate_photorealistic_for_existing_boards(args)
            return 0
        except (RuntimeError, ImportError, ValueError) as exc:
            print(f"[board-cli] configuration error: {exc}", file=sys.stderr)
            return 2

    parser.error("Unsupported command.")
    return 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("[board-cli] interrupted", file=sys.stderr)
        raise SystemExit(130)
