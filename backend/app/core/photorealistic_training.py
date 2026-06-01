from __future__ import annotations

import json
import os
import shutil
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDIMScheduler, DDPMScheduler, UNet2DConditionModel
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers.training_utils import EMAModel
from PIL import Image
from safetensors.torch import save_file
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from .knot_map import build_knot_map_from_fiber_gray01


_IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")
_DEFAULT_LATENT_SCALE = 0.18215
_MIN_FREE_BYTES_FOR_STATE_SAVE = 8 * (1024 ** 3)  # 8 GiB
_SDXL_NUM_TIME_IDS = 6


class TrainingConfigurationError(RuntimeError):
    """Raised when training configuration is invalid or dependencies are missing."""


@dataclass
class TrainingConfig:
    data_root: str = "/mnt/e/SCRATCH/Osama/RGB_for_labeling/images"
    output_dir: str = "./runs/wood_sd2_concat_joint_v2"
    sd2_model_dir: str = ""
    model_family: str = "sd2"
    sdxl_model_dir: str = ""

    image_size: int = 512
    boards_per_batch: int = 2
    grad_accum_steps: int = 1
    num_workers: int = 4
    train_pin_memory: bool = False

    max_train_steps: int = 325000
    lr: float = 2e-5
    weight_decay: float = 0.0
    optimizer: str = "auto"
    num_warmup_steps: int = 2000
    max_grad_norm: float = 1.0

    guidance_drop_prob: float = 0.1
    include_knot_maps: bool = False
    use_rings_only: bool = False
    num_train_timesteps: int = 1000
    prediction_type: str = "epsilon"
    min_snr_gamma: float = 5.0
    latent_recon_weight: float = 0.1
    cross_surface_consistency_weight: float = 0.0
    seam_consistency_weight: float = 0.0
    seam_strip_width: int = 2
    shared_board_noise: bool = False

    ema_decay: float = 0.9999
    ema_update_every: int = 1
    use_ema: bool = True

    val_ratio: float = 0.1
    validate_every: int = 1000
    val_boards: int = 2
    val_ddim_steps: int = 30
    val_guidance_scale: float = 1.5
    val_num_workers: int = 0
    val_pin_memory: bool = False
    validate_on_main_process_only: bool = True

    export_every: int = 5000
    export_ddim_steps: int = 50
    export_guidance_scale: float = 1.5
    export_img2img_strength: float = 0.0

    mixed_precision: str = "fp16"
    enable_gradient_checkpointing: bool = True
    enable_xformers: bool = True

    seed: int = 42

    show_cli_progress: bool = True
    cli_progress_every: int = 10

    use_wandb: bool = True
    wandb_project: str = "wood-sd2-concat"
    wandb_run: str = "osama-joint-photorealistic-v2"
    wandb_mode: str = "online"

    resume_state_dir: str = ""


class NullContextEmbedding(nn.Module):
    def __init__(self, dim: int, seq_len: int = 77, init_std: float = 0.02):
        super().__init__()
        token = torch.zeros(1, seq_len, dim)
        nn.init.normal_(token, std=init_std)
        self.token = nn.Parameter(token)

    def forward(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return self.token.expand(batch_size, -1, -1).to(device=device, dtype=dtype)


class NullContextEMA:
    def __init__(self, module: NullContextEmbedding, decay: float):
        self.decay = float(decay)
        self.shadow = module.token.detach().clone()
        self._backup: Optional[torch.Tensor] = None

    @torch.no_grad()
    def step(self, module: NullContextEmbedding) -> None:
        self.shadow.mul_(self.decay).add_(module.token.detach(), alpha=1.0 - self.decay)

    @torch.no_grad()
    def store(self, module: NullContextEmbedding) -> None:
        self._backup = module.token.detach().clone()

    @torch.no_grad()
    def copy_to(self, module: NullContextEmbedding) -> None:
        module.token.data.copy_(self.shadow)

    @torch.no_grad()
    def restore(self, module: NullContextEmbedding) -> None:
        if self._backup is None:
            return
        module.token.data.copy_(self._backup)
        self._backup = None

    def state_dict(self) -> Dict[str, Any]:
        return {"decay": self.decay, "shadow": self.shadow}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        if "shadow" not in state_dict:
            raise ValueError("Invalid NullContextEMA state_dict: missing 'shadow'.")
        self.decay = float(state_dict.get("decay", self.decay))
        self.shadow = state_dict["shadow"].detach().clone()


def _default_sd2_model_dir() -> Path:
    env_path = str(
        os.environ.get("PHOTOREALISTIC_SD2_MODEL_DIR")
        or os.environ.get("SD2_MODEL_DIR")
        or ""
    ).strip()
    if env_path:
        return Path(env_path).expanduser().resolve()
    return Path(__file__).resolve().parents[3] / "SD2_model"


def _default_sdxl_model_dir() -> Path:
    env_path = str(
        os.environ.get("PHOTOREALISTIC_SDXL_MODEL_DIR")
        or os.environ.get("SDXL_MODEL_DIR")
        or ""
    ).strip()
    if env_path:
        return Path(env_path).expanduser().resolve()
    return Path(__file__).resolve().parents[3] / "SDXL_model"


def _normalize_model_family(value: Any) -> str:
    text = str(value or "sd2").strip().lower()
    if text in {"sd2", "sdxl"}:
        return text
    raise TrainingConfigurationError(
        f"Unsupported model_family '{value}'. Expected one of: sd2, sdxl."
    )


def _resolve_model_root(cfg: TrainingConfig) -> Path:
    family = _normalize_model_family(cfg.model_family)
    if family == "sdxl":
        model_dir = str(cfg.sdxl_model_dir or "").strip()
        if not model_dir:
            model_dir = str(_default_sdxl_model_dir())
        return Path(model_dir).expanduser().resolve()
    model_dir = str(cfg.sd2_model_dir or "").strip()
    if not model_dir:
        model_dir = str(_default_sd2_model_dir())
    return Path(model_dir).expanduser().resolve()


def _resolve_cross_attention_dim(unet: UNet2DConditionModel) -> int:
    raw = getattr(unet.config, "cross_attention_dim", 0)
    if isinstance(raw, (list, tuple)):
        if not raw:
            raise TrainingConfigurationError("UNet cross_attention_dim is empty.")
        raw = raw[0]
    dim = int(raw)
    if dim <= 0:
        raise TrainingConfigurationError(
            f"Invalid UNet cross_attention_dim: {raw!r}."
        )
    return dim


@dataclass
class _ModelConditioningSpec:
    model_family: str
    pooled_embed_dim: int = 0
    num_time_ids: int = 0

    @property
    def requires_added_cond_kwargs(self) -> bool:
        return bool(self.model_family == "sdxl")


def _build_model_conditioning_spec(
    *,
    model_family: str,
    unet: UNet2DConditionModel,
) -> _ModelConditioningSpec:
    family = _normalize_model_family(model_family)
    if family != "sdxl":
        return _ModelConditioningSpec(model_family=family)

    add_time_dim = int(getattr(unet.config, "addition_time_embed_dim", 256) or 256)
    if add_time_dim <= 0:
        raise TrainingConfigurationError(
            f"Invalid SDXL addition_time_embed_dim: {add_time_dim}."
        )

    linear_1 = getattr(getattr(unet, "add_embedding", None), "linear_1", None)
    if linear_1 is None or not hasattr(linear_1, "in_features"):
        raise TrainingConfigurationError(
            "SDXL UNet missing add_embedding.linear_1; cannot infer pooled text embedding size."
        )

    total_in = int(linear_1.in_features)
    pooled_dim = total_in - int(_SDXL_NUM_TIME_IDS * add_time_dim)
    if pooled_dim <= 0:
        raise TrainingConfigurationError(
            "Failed to infer SDXL pooled text embedding dimension from UNet config."
        )

    return _ModelConditioningSpec(
        model_family=family,
        pooled_embed_dim=int(pooled_dim),
        num_time_ids=int(_SDXL_NUM_TIME_IDS),
    )


def _build_added_cond_kwargs(
    *,
    spec: _ModelConditioningSpec,
    batch_size: int,
    image_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Optional[Dict[str, torch.Tensor]]:
    if not spec.requires_added_cond_kwargs:
        return None
    text_embeds = torch.zeros(
        (int(batch_size), int(spec.pooled_embed_dim)),
        device=device,
        dtype=dtype,
    )
    time_row = torch.tensor(
        [
            float(image_size),
            float(image_size),
            0.0,
            0.0,
            float(image_size),
            float(image_size),
        ],
        device=device,
        dtype=dtype,
    )
    time_ids = time_row.unsqueeze(0).repeat(int(batch_size), 1)
    return {
        "text_embeds": text_embeds,
        "time_ids": time_ids,
    }


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _resolve_use_ema(value: Any, model_family: str) -> bool:
    raw = str(value if value is not None else "auto").strip().lower()
    family = _normalize_model_family(model_family)
    if raw in {"", "auto"}:
        # SDXL full-model training is memory-heavy; default to no EMA there.
        return bool(family != "sdxl")
    return _as_bool(raw)


def _build_stem_path_map(folder: Path) -> Dict[str, Path]:
    if not folder.is_dir():
        raise TrainingConfigurationError(f"Missing dataset folder: {folder}")
    mapping: Dict[str, Path] = {}
    for entry in folder.iterdir():
        if not entry.is_file():
            continue
        if entry.suffix.lower() not in _IMG_EXTS:
            continue
        stem = entry.stem
        if stem not in mapping:
            mapping[stem] = entry
    return mapping


def _split_train_val(stems: List[str], ratio: float, seed: int) -> Tuple[List[str], List[str]]:
    if not stems:
        return [], []
    rng = torch.Generator(device="cpu")
    rng.manual_seed(int(seed))
    perm = torch.randperm(len(stems), generator=rng).tolist()
    shuffled = [stems[i] for i in perm]
    val_count = int(round(len(shuffled) * float(ratio)))
    if len(shuffled) > 1:
        val_count = min(max(val_count, 1), len(shuffled) - 1)
    else:
        val_count = 0
    val_stems = shuffled[:val_count]
    train_stems = shuffled[val_count:]
    return train_stems, val_stems


class WoodPhotorealisticDataset(Dataset):
    """One item is one board containing 4 surfaces."""

    def __init__(
        self,
        root: Path,
        image_size: int,
        stems: Optional[List[str]] = None,
        include_knot_maps: bool = False,
        use_rings_only: bool = False,
    ):
        self.root = root
        self.image_size = int(image_size)
        self.include_knot_maps = bool(include_knot_maps)
        self.use_rings_only = bool(use_rings_only)

        self.ring_dirs = [self.root / f"ring_pred_new_{i}" for i in range(1, 5)]
        self.color_dirs = [self.root / f"color_{i}" for i in range(1, 5)]
        self.fiber_dirs = [self.root / f"fiber_{i}" for i in range(1, 5)]

        self.ring_maps = [_build_stem_path_map(path) for path in self.ring_dirs]
        self.fiber_maps = (
            [_build_stem_path_map(path) for path in self.fiber_dirs]
            if not self.use_rings_only
            else []
        )
        self.color_maps = [_build_stem_path_map(path) for path in self.color_dirs]

        if stems is None:
            sets = [set(m.keys()) for m in (self.ring_maps + self.color_maps)]
            if not self.use_rings_only:
                sets.extend([set(m.keys()) for m in self.fiber_maps])
            matched = sorted(set.intersection(*sets)) if sets else []
            if not matched:
                if self.use_rings_only:
                    raise TrainingConfigurationError(
                        "No matching stems found across ring_pred_new_* and color_* folders."
                    )
                raise TrainingConfigurationError(
                    "No matching stems found across ring_pred_new_*, fiber_*, and color_* folders."
                )
            self.stems = matched
        else:
            self.stems = list(stems)

        self.to_tensor = transforms.Compose(
            [
                transforms.Resize(
                    (self.image_size, self.image_size),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.ToTensor(),
            ]
        )

    def __len__(self) -> int:
        return len(self.stems)

    def _load_gray(self, path: Path) -> torch.Tensor:
        return self.to_tensor(Image.open(path).convert("L")).mul(2.0).sub(1.0)

    def _load_rgb(self, path: Path) -> torch.Tensor:
        return self.to_tensor(Image.open(path).convert("RGB")).mul(2.0).sub(1.0)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        stem = self.stems[index]
        cond_surfaces: List[torch.Tensor] = []
        target_surfaces: List[torch.Tensor] = []

        for face_idx in range(4):
            ring = self._load_gray(self.ring_maps[face_idx][stem])
            fiber = (
                torch.zeros_like(ring)
                if self.use_rings_only
                else self._load_gray(self.fiber_maps[face_idx][stem])
            )
            color = self._load_rgb(self.color_maps[face_idx][stem])
            cond_channels = [ring, fiber]
            if self.include_knot_maps:
                fiber01 = ((fiber[0].detach().cpu().numpy() + 1.0) * 0.5).astype(np.float32, copy=False)
                knot01 = build_knot_map_from_fiber_gray01(fiber01)
                knot = torch.from_numpy(knot01).unsqueeze(0).to(dtype=fiber.dtype).mul(2.0).sub(1.0)
                cond_channels.append(knot)
            if self.use_rings_only:
                cond_surfaces.append(ring)
            else:
                cond_surfaces.append(torch.cat(cond_channels, dim=0))
            target_surfaces.append(color)

        return {
            "cond_nch": torch.stack(cond_surfaces, dim=0),
            "target_rgb": torch.stack(target_surfaces, dim=0),
            "stem": stem,
        }


def _collate_boards(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "cond_nch": torch.stack([item["cond_nch"] for item in batch], dim=0),
        "target_rgb": torch.stack([item["target_rgb"] for item in batch], dim=0),
        "stem": [item["stem"] for item in batch],
    }


def _expand_unet_in_channels(unet: UNet2DConditionModel, new_in_channels: int = 8) -> UNet2DConditionModel:
    old_conv: nn.Conv2d = unet.conv_in
    if int(old_conv.in_channels) == int(new_in_channels):
        return unet
    if int(old_conv.in_channels) != 4:
        raise TrainingConfigurationError(
            f"Unexpected UNet input channels: expected 4, found {old_conv.in_channels}."
        )

    new_conv = nn.Conv2d(
        in_channels=new_in_channels,
        out_channels=old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=(old_conv.bias is not None),
    ).to(device=old_conv.weight.device, dtype=old_conv.weight.dtype)

    with torch.no_grad():
        new_conv.weight.zero_()
        new_conv.weight[:, :4, :, :] = old_conv.weight
        if old_conv.bias is not None:
            new_conv.bias.copy_(old_conv.bias)

    unet.conv_in = new_conv
    unet.config.in_channels = int(new_in_channels)
    return unet


def _extract_to_shape(values: torch.Tensor, timesteps: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    out = values.to(device=timesteps.device, dtype=reference.dtype)[timesteps]
    while out.ndim < reference.ndim:
        out = out.unsqueeze(-1)
    return out


def _resolve_dtype(mixed_precision: str) -> torch.dtype:
    if mixed_precision == "fp16":
        return torch.float16
    if mixed_precision == "bf16":
        return torch.bfloat16
    return torch.float32


def _resolve_optimizer_name(cfg: TrainingConfig) -> str:
    name = str(getattr(cfg, "optimizer", "auto") or "auto").strip().lower()
    if name not in {"auto", "adamw", "adafactor"}:
        raise TrainingConfigurationError(
            f"Unsupported optimizer '{cfg.optimizer}'. Expected one of: auto, adamw, adafactor."
        )
    if name == "auto":
        return "adafactor" if str(cfg.model_family) == "sdxl" else "adamw"
    return name


def _resolve_vae_dtype(
    *,
    vae: AutoencoderKL,
    requested_dtype: torch.dtype,
    device: torch.device,
) -> torch.dtype:
    force_upcast = bool(getattr(getattr(vae, "config", None), "force_upcast", False))
    enable_upcast = str(os.environ.get("PHOTOREALISTIC_ENABLE_VAE_FORCE_UPCAST", "")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if (
        bool(enable_upcast)
        and bool(force_upcast)
        and device.type == "cuda"
        and requested_dtype in (torch.float16, torch.bfloat16)
    ):
        # SDXL VAE commonly requires fp32 encode/decode for numerical stability.
        return torch.float32
    return requested_dtype


def _ensure_finite_tensor(name: str, tensor: torch.Tensor) -> None:
    finite = torch.isfinite(tensor)
    if bool(torch.all(finite).item()):
        return
    total = int(tensor.numel())
    bad = int((~finite).sum().item())
    detached = tensor.detach().float()
    safe = torch.nan_to_num(detached, nan=0.0, posinf=0.0, neginf=0.0)
    raise TrainingConfigurationError(
        f"Non-finite values detected in {name}: {bad}/{total} elements are NaN/Inf. "
        f"finite_min={float(safe.min().item()):.6f}, finite_max={float(safe.max().item()):.6f}."
    )


def _sanitize_non_finite(
    tensor: torch.Tensor,
    *,
    clamp_abs: float = 100.0,
) -> Tuple[torch.Tensor, bool]:
    finite = torch.isfinite(tensor)
    if bool(torch.all(finite).item()):
        return tensor, False
    fixed = torch.nan_to_num(
        tensor,
        nan=0.0,
        posinf=float(clamp_abs),
        neginf=-float(clamp_abs),
    )
    fixed = fixed.clamp(min=-float(clamp_abs), max=float(clamp_abs))
    return fixed, True


def _get_prediction_target(
    noise_scheduler: DDPMScheduler,
    prediction_type: str,
    z0: torch.Tensor,
    noise: torch.Tensor,
    timesteps: torch.Tensor,
) -> torch.Tensor:
    if prediction_type == "epsilon":
        return noise
    if prediction_type == "v_prediction":
        return noise_scheduler.get_velocity(z0, noise, timesteps)
    raise TrainingConfigurationError(f"Unsupported prediction_type: {prediction_type}")


def _compute_x0_prediction(
    prediction_type: str,
    z_t: torch.Tensor,
    model_output: torch.Tensor,
    alpha_t: torch.Tensor,
    sigma_t: torch.Tensor,
) -> torch.Tensor:
    if prediction_type == "epsilon":
        return (z_t - sigma_t * model_output) / torch.clamp(alpha_t, min=1e-6)
    if prediction_type == "v_prediction":
        return alpha_t * z_t - sigma_t * model_output
    raise TrainingConfigurationError(f"Unsupported prediction_type: {prediction_type}")


def _min_snr_weights(
    prediction_type: str,
    alpha_cumprod: torch.Tensor,
    timesteps: torch.Tensor,
    gamma: float,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    alpha = alpha_cumprod.to(device=device, dtype=dtype)[timesteps]
    snr = alpha / torch.clamp(1.0 - alpha, min=1e-8)
    gamma_t = torch.full_like(snr, float(gamma))
    if prediction_type == "epsilon":
        return torch.minimum(snr, gamma_t) / torch.clamp(snr, min=1e-8)
    if prediction_type == "v_prediction":
        return torch.minimum(snr, gamma_t) / torch.clamp(snr + 1.0, min=1e-8)
    raise TrainingConfigurationError(f"Unsupported prediction_type: {prediction_type}")


def _cross_surface_consistency_loss(
    x0_pred: torch.Tensor,
    z0_target: torch.Tensor,
    boards: int,
    surfaces: int,
) -> torch.Tensor:
    # Encourage consistent board-level appearance across the 4 faces by matching
    # pairwise face-to-face latent moment differences to the target board.
    if boards <= 0 or surfaces <= 1:
        return x0_pred.new_tensor(0.0)

    pred = x0_pred.float().view(boards, surfaces, *x0_pred.shape[1:])
    tgt = z0_target.float().view(boards, surfaces, *z0_target.shape[1:])

    pred_mu = pred.mean(dim=(-1, -2))  # [B,S,C]
    tgt_mu = tgt.mean(dim=(-1, -2))
    pred_std = pred.std(dim=(-1, -2), unbiased=False)  # [B,S,C]
    tgt_std = tgt.std(dim=(-1, -2), unbiased=False)

    pred_mu_d = pred_mu[:, :, None, :] - pred_mu[:, None, :, :]  # [B,S,S,C]
    tgt_mu_d = tgt_mu[:, :, None, :] - tgt_mu[:, None, :, :]
    pred_std_d = pred_std[:, :, None, :] - pred_std[:, None, :, :]
    tgt_std_d = tgt_std[:, :, None, :] - tgt_std[:, None, :, :]

    loss_mu = F.smooth_l1_loss(pred_mu_d, tgt_mu_d)
    loss_std = F.smooth_l1_loss(pred_std_d, tgt_std_d)
    return loss_mu + loss_std


def _edge_profile(face_latent: torch.Tensor, edge: str, strip_width: int) -> torch.Tensor:
    # face_latent: [B, C, H, W] -> profile [B, C, L]
    width = max(1, int(strip_width))
    if edge == "left":
        return face_latent[:, :, :, :width].mean(dim=-1)
    if edge == "right":
        return face_latent[:, :, :, -width:].mean(dim=-1)
    if edge == "top":
        return face_latent[:, :, :width, :].mean(dim=-2)
    if edge == "bottom":
        return face_latent[:, :, -width:, :].mean(dim=-2)
    raise ValueError(f"Unsupported edge: {edge}")


def _seam_consistency_loss(
    x0_pred: torch.Tensor,
    z0_target: torch.Tensor,
    boards: int,
    surfaces: int,
    strip_width: int = 2,
) -> torch.Tensor:
    # Expected side-face order:
    # 0->surface_1, 1->surface_2, 2->surface_3, 3->surface_4.
    # We compare seam relationships for adjacent side pairs.
    if boards <= 0 or surfaces < 4:
        return x0_pred.new_tensor(0.0)

    pred = x0_pred.float().view(boards, surfaces, *x0_pred.shape[1:])
    tgt = z0_target.float().view(boards, surfaces, *z0_target.shape[1:])

    seam_pairs = [
        (0, "left", 2, "right"),
        (0, "right", 3, "left"),
        (1, "right", 2, "left"),
        (1, "left", 3, "right"),
    ]

    losses: List[torch.Tensor] = []
    for fa, ea, fb, eb in seam_pairs:
        p_a = _edge_profile(pred[:, fa], ea, strip_width)
        p_b = _edge_profile(pred[:, fb], eb, strip_width)
        t_a = _edge_profile(tgt[:, fa], ea, strip_width)
        t_b = _edge_profile(tgt[:, fb], eb, strip_width)

        p_d = p_a - p_b
        p_d_rev = p_a - p_b.flip(-1)
        t_d = t_a - t_b
        t_d_rev = t_a - t_b.flip(-1)

        # Orientation may differ by face convention; choose lowest-cost alignment.
        l00 = F.smooth_l1_loss(p_d, t_d)
        l01 = F.smooth_l1_loss(p_d, t_d_rev)
        l10 = F.smooth_l1_loss(p_d_rev, t_d)
        l11 = F.smooth_l1_loss(p_d_rev, t_d_rev)
        losses.append(torch.minimum(torch.minimum(l00, l01), torch.minimum(l10, l11)))

    if not losses:
        return x0_pred.new_tensor(0.0)
    return torch.stack(losses).mean()


def _save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def _format_duration(seconds: float) -> str:
    if not np.isfinite(seconds) or seconds < 0.0:
        return "?"
    total = int(round(float(seconds)))
    hours = total // 3600
    minutes = (total % 3600) // 60
    secs = total % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _bytes_to_gib(value: int) -> float:
    return float(max(0, int(value))) / float(1024 ** 3)


def _disk_usage_summary(path: Path) -> str:
    usage = shutil.disk_usage(str(path))
    return (
        f"free={_bytes_to_gib(usage.free):.2f} GiB, "
        f"used={_bytes_to_gib(usage.used):.2f} GiB, "
        f"total={_bytes_to_gib(usage.total):.2f} GiB"
    )


def _looks_like_filesystem_write_error(exc: BaseException) -> bool:
    text = str(exc).lower()
    return (
        "inline_container" in text
        or "pytorchstreamwriter" in text
        or "no space left on device" in text
        or "unexpected pos" in text
        or "i/o error" in text
        or "input/output error" in text
    )


def _looks_like_cuda_oom(exc: BaseException) -> bool:
    text = str(exc).lower()
    return "out of memory" in text and "cuda" in text


def _ensure_checkpoint_free_space(path: Path, *, context: str) -> None:
    usage = shutil.disk_usage(str(path))
    if int(usage.free) >= int(_MIN_FREE_BYTES_FOR_STATE_SAVE):
        return
    need_gib = _bytes_to_gib(_MIN_FREE_BYTES_FOR_STATE_SAVE)
    raise TrainingConfigurationError(
        f"Insufficient free space for checkpointing during {context}. "
        f"Need at least ~{need_gib:.1f} GiB free, but {_disk_usage_summary(path)} "
        f"on filesystem containing '{path}'. "
        "Use --output-dir on a volume with more space (e.g. /mnt/d) or free disk space."
    )


def _build_cond_rgb_from_cond_nch(cond_nch: torch.Tensor) -> torch.Tensor:
    if cond_nch.ndim != 4:
        raise TrainingConfigurationError(
            f"Expected cond tensor rank=4 [N,C,H,W], got shape={tuple(cond_nch.shape)}."
        )
    channels = int(cond_nch.shape[1])
    if channels == 1:
        zero = torch.zeros_like(cond_nch[:, :1])
        return torch.cat([cond_nch, zero, zero], dim=1)
    if channels == 2:
        zero = torch.zeros_like(cond_nch[:, :1])
        return torch.cat([cond_nch, zero], dim=1)
    if channels == 3:
        return cond_nch
    raise TrainingConfigurationError(
        f"Unsupported conditioning channels: {channels}. "
        "Expected 1 (ring), 2 (ring+fiber), or 3 (ring+fiber+knot)."
    )


@torch.no_grad()
def _sample_validation_batch(
    *,
    vae: AutoencoderKL,
    unet: UNet2DConditionModel,
    null_embed: NullContextEmbedding,
    scheduler: DDIMScheduler,
    conditioning_spec: _ModelConditioningSpec,
    cond_boards: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
    latent_scale: float,
    ddim_steps: int,
    guidance_scale: float,
) -> torch.Tensor:
    bsz, surfaces, _, height, width = cond_boards.shape
    cond_channels = int(cond_boards.shape[2])
    cond_flat = cond_boards.view(bsz * surfaces, cond_channels, height, width)
    vae_dtype = next(vae.parameters()).dtype
    cond_rgb = _build_cond_rgb_from_cond_nch(cond_flat).to(device=device, dtype=vae_dtype)

    cond_latent = (vae.encode(cond_rgb).latent_dist.mean * float(latent_scale)).to(dtype=dtype)
    cond_latent, _ = _sanitize_non_finite(cond_latent)
    scheduler.set_timesteps(int(ddim_steps), device=device)
    x = torch.randn_like(cond_latent, dtype=dtype)
    zero_cond_latent = torch.zeros_like(cond_latent)
    cfg_scale = float(guidance_scale)

    token = null_embed(cond_latent.shape[0], device=device, dtype=dtype)
    added_cond_kwargs = _build_added_cond_kwargs(
        spec=conditioning_spec,
        batch_size=int(cond_latent.shape[0]),
        image_size=int(width),
        device=device,
        dtype=dtype,
    )
    for timestep in scheduler.timesteps:
        t_tensor = timestep if torch.is_tensor(timestep) else torch.tensor(timestep, device=device)
        autocast_ctx = (
            torch.autocast(device_type="cuda", dtype=dtype)
            if device.type == "cuda" and dtype in (torch.float16, torch.bfloat16)
            else nullcontext()
        )
        with autocast_ctx:
            # Run CFG branches sequentially to reduce peak memory during validation.
            x_in = x.to(dtype=dtype)
            unet_in_uncond = torch.cat([x_in, zero_cond_latent], dim=1)
            pred = unet(
                unet_in_uncond,
                t_tensor,
                encoder_hidden_states=token,
                added_cond_kwargs=added_cond_kwargs,
            ).sample

            if cfg_scale != 0.0:
                unet_in_cond = torch.cat([x_in, cond_latent], dim=1)
                pred_cond = unet(
                    unet_in_cond,
                    t_tensor,
                    encoder_hidden_states=token,
                    added_cond_kwargs=added_cond_kwargs,
                ).sample
                # In-place CFG merge avoids extra temporary tensors:
                # pred <- pred + scale * (pred_cond - pred)
                pred.mul_(1.0 - cfg_scale).add_(pred_cond, alpha=cfg_scale)
                del pred_cond

            x = scheduler.step(pred, t_tensor, x_in).prev_sample.to(dtype=dtype)

    rgb = vae.decode((x / float(latent_scale)).to(dtype=vae_dtype)).sample
    rgb, _ = _sanitize_non_finite(rgb, clamp_abs=1.0)
    return (rgb.clamp(-1.0, 1.0) + 1.0) * 0.5


def _to_uint8_image(tensor_chw: torch.Tensor) -> Image.Image:
    arr = tensor_chw.detach().cpu().permute(1, 2, 0).numpy()
    arr = (arr.clip(0.0, 1.0) * 255.0).round().astype("uint8")
    return Image.fromarray(arr)


def _build_validation_panel(
    cond_nch: torch.Tensor,
    pred_rgb: torch.Tensor,
    target_rgb: torch.Tensor,
) -> Image.Image:
    rows: List[Image.Image] = []
    for i in range(cond_nch.shape[0]):
        channels = int(cond_nch.shape[1])
        ring = cond_nch[i, 0:1].repeat(3, 1, 1)
        ring = (ring.clamp(-1.0, 1.0) + 1.0) * 0.5

        images: List[Image.Image] = [_to_uint8_image(ring)]
        if channels >= 2:
            fiber = cond_nch[i, 1:2].repeat(3, 1, 1)
            fiber = (fiber.clamp(-1.0, 1.0) + 1.0) * 0.5
            images.append(_to_uint8_image(fiber))
        if channels >= 3:
            knot = cond_nch[i, 2:3].repeat(3, 1, 1)
            knot = (knot.clamp(-1.0, 1.0) + 1.0) * 0.5
            images.append(_to_uint8_image(knot))
        images.append(_to_uint8_image(pred_rgb[i]))
        images.append(_to_uint8_image((target_rgb[i].clamp(-1.0, 1.0) + 1.0) * 0.5))
        width = sum(img.width for img in images)
        height = max(img.height for img in images)
        row = Image.new("RGB", (width, height), color=(255, 255, 255))
        offset_x = 0
        for img in images:
            row.paste(img, (offset_x, 0))
            offset_x += img.width
        rows.append(row)

    if not rows:
        return Image.new("RGB", (1, 1), color=(255, 255, 255))

    panel_w = max(img.width for img in rows)
    panel_h = sum(img.height for img in rows)
    panel = Image.new("RGB", (panel_w, panel_h), color=(255, 255, 255))
    offset_y = 0
    for row in rows:
        panel.paste(row, (0, offset_y))
        offset_y += row.height
    return panel


def _export_checkpoint_bundle(
    *,
    cfg: TrainingConfig,
    model_family: str,
    model_root: Path,
    accelerator: Accelerator,
    unet: UNet2DConditionModel,
    null_embed: NullContextEmbedding,
    step: int,
    use_ema: bool,
    suffix: str,
) -> None:
    if not accelerator.is_main_process:
        return

    out_dir = Path(cfg.output_dir).resolve() / f"export_step_{step:07d}_{suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)

    unet_unwrapped = accelerator.unwrap_model(unet)
    null_unwrapped = accelerator.unwrap_model(null_embed)

    unet_state = {k: v.detach().cpu() for k, v in unet_unwrapped.state_dict().items()}
    null_state = {"token": null_unwrapped.token.detach().cpu()}

    save_file(unet_state, str(out_dir / "unet.safetensors"))
    save_file(null_state, str(out_dir / "null_embed.safetensors"))

    model_dir_name = "SDXL_model" if str(model_family).strip().lower() == "sdxl" else "SD2_model"
    export_cfg = {
        "image_size": int(cfg.image_size),
        "ddim_steps": int(cfg.export_ddim_steps),
        "guidance_scale": float(cfg.export_guidance_scale),
        "use_img2img_strength": float(cfg.export_img2img_strength),
        "include_knot_maps": bool(cfg.include_knot_maps),
        "use_rings_only": bool(cfg.use_rings_only),
        "model_family": str(model_family),
        "base_model_dir": str(model_dir_name),
        # Keep sd2_model_dir for backward compatibility with older runtimes.
        "sd2_model_dir": str(model_dir_name),
        "prediction_type": str(cfg.prediction_type),
        "trained_with_ema": bool(use_ema),
        "export_step": int(step),
    }
    _save_json(out_dir / "config.json", export_cfg)


def _get_wandb_client(cfg: TrainingConfig):
    if not cfg.use_wandb:
        return None
    try:
        import wandb  # type: ignore
    except Exception as exc:  # pragma: no cover - import error path
        raise TrainingConfigurationError(
            f"W&B requested but import failed: {exc}. Install wandb or use --disable-wandb."
        ) from exc
    return wandb


def _validate_cfg(cfg: TrainingConfig) -> None:
    _normalize_model_family(cfg.model_family)
    _resolve_optimizer_name(cfg)
    if cfg.prediction_type not in {"epsilon", "v_prediction"}:
        raise TrainingConfigurationError("prediction_type must be 'epsilon' or 'v_prediction'.")
    if cfg.max_train_steps <= 0:
        raise TrainingConfigurationError("max_train_steps must be > 0.")
    if cfg.boards_per_batch <= 0:
        raise TrainingConfigurationError("boards_per_batch must be > 0.")
    if cfg.grad_accum_steps <= 0:
        raise TrainingConfigurationError("grad_accum_steps must be > 0.")
    if cfg.image_size <= 0:
        raise TrainingConfigurationError("image_size must be > 0.")
    if cfg.ema_update_every < 1:
        raise TrainingConfigurationError("ema_update_every must be >= 1.")
    if cfg.val_ratio < 0.0 or cfg.val_ratio >= 1.0:
        raise TrainingConfigurationError("val_ratio must be in [0, 1).")
    if bool(cfg.use_rings_only) and bool(cfg.include_knot_maps):
        raise TrainingConfigurationError(
            "use_rings_only=true cannot be combined with include_knot_maps=true."
        )
    if int(cfg.cli_progress_every) <= 0:
        raise TrainingConfigurationError("cli_progress_every must be > 0.")


def train_photorealistic_diffusion(cfg: TrainingConfig) -> None:
    _validate_cfg(cfg)

    cfg.model_family = _normalize_model_family(cfg.model_family)
    model_root = _resolve_model_root(cfg)
    if cfg.model_family == "sdxl" and not cfg.sdxl_model_dir:
        cfg.sdxl_model_dir = str(model_root)
    if cfg.model_family == "sd2" and not cfg.sd2_model_dir:
        cfg.sd2_model_dir = str(model_root)

    data_root = Path(cfg.data_root).expanduser().resolve()
    output_root = Path(cfg.output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    _ensure_checkpoint_free_space(output_root, context="training start")

    if not data_root.is_dir():
        raise TrainingConfigurationError(f"data_root does not exist: {data_root}")
    if not model_root.is_dir():
        raise TrainingConfigurationError(
            f"{cfg.model_family}_model_dir does not exist: {model_root}"
        )

    accelerator = Accelerator(gradient_accumulation_steps=cfg.grad_accum_steps, mixed_precision=cfg.mixed_precision)
    set_seed(int(cfg.seed) + int(accelerator.process_index))

    wandb = None
    if accelerator.is_main_process:
        wandb = _get_wandb_client(cfg)
        if wandb is not None:
            wandb.init(
                project=cfg.wandb_project,
                name=cfg.wandb_run,
                mode=cfg.wandb_mode,
                config=asdict(cfg),
            )

    dataset_all = WoodPhotorealisticDataset(
        root=data_root,
        image_size=cfg.image_size,
        include_knot_maps=bool(cfg.include_knot_maps),
        use_rings_only=bool(cfg.use_rings_only),
    )
    train_stems, val_stems = _split_train_val(dataset_all.stems, cfg.val_ratio, cfg.seed)
    train_ds = WoodPhotorealisticDataset(
        root=data_root,
        image_size=cfg.image_size,
        stems=train_stems,
        include_knot_maps=bool(cfg.include_knot_maps),
        use_rings_only=bool(cfg.use_rings_only),
    )
    val_ds = (
        WoodPhotorealisticDataset(
            root=data_root,
            image_size=cfg.image_size,
            stems=val_stems,
            include_knot_maps=bool(cfg.include_knot_maps),
            use_rings_only=bool(cfg.use_rings_only),
        )
        if val_stems
        else None
    )

    train_dl = DataLoader(
        train_ds,
        batch_size=cfg.boards_per_batch,
        shuffle=True,
        num_workers=max(0, int(cfg.num_workers)),
        pin_memory=bool(cfg.train_pin_memory),
        drop_last=True,
        collate_fn=_collate_boards,
    )
    if len(train_dl) == 0:
        raise TrainingConfigurationError(
            "Training dataloader is empty. Reduce --boards-per-batch or provide more matched boards."
        )

    val_dl = None
    if val_ds is not None and len(val_ds) > 0:
        val_dl = DataLoader(
            val_ds,
            batch_size=max(1, min(int(cfg.val_boards), 4)),
            shuffle=True,
            num_workers=max(0, int(cfg.val_num_workers)),
            pin_memory=bool(cfg.val_pin_memory),
            drop_last=False,
            collate_fn=_collate_boards,
        )
    val_iter = iter(val_dl) if val_dl is not None else None

    torch_dtype = _resolve_dtype(cfg.mixed_precision)

    vae = AutoencoderKL.from_pretrained(
        str(model_root),
        subfolder="vae",
        local_files_only=True,
        torch_dtype=torch_dtype,
    )
    vae.requires_grad_(False)
    vae_runtime_dtype = _resolve_vae_dtype(
        vae=vae,
        requested_dtype=torch_dtype,
        device=accelerator.device,
    )
    vae_scaling_factor = float(getattr(vae.config, "scaling_factor", _DEFAULT_LATENT_SCALE))

    unet = UNet2DConditionModel.from_pretrained(
        str(model_root),
        subfolder="unet",
        local_files_only=True,
    )
    unet = _expand_unet_in_channels(unet, new_in_channels=8)
    conditioning_spec = _build_model_conditioning_spec(
        model_family=cfg.model_family,
        unet=unet,
    )

    if cfg.enable_gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    null_embed = NullContextEmbedding(
        dim=_resolve_cross_attention_dim(unet),
        seq_len=77,
    )

    if cfg.enable_xformers:
        try:
            unet.enable_xformers_memory_efficient_attention()
        except Exception:
            pass

    # Keep train/noise and sample/validation schedules aligned with base scheduler config
    # to avoid systematic tone/color drift from schedule mismatch.
    base_ddim_scheduler = DDIMScheduler.from_pretrained(
        str(model_root),
        subfolder="scheduler",
        local_files_only=True,
    )
    scheduler_cfg = dict(base_ddim_scheduler.config)
    scheduler_cfg["num_train_timesteps"] = int(cfg.num_train_timesteps)
    noise_scheduler = DDPMScheduler.from_config(
        scheduler_cfg,
        prediction_type=cfg.prediction_type,
    )
    val_ddim_scheduler = DDIMScheduler.from_config(
        scheduler_cfg,
        prediction_type=cfg.prediction_type,
    )

    trainable_params = list(unet.parameters()) + list(null_embed.parameters())
    optimizer_name = _resolve_optimizer_name(cfg)
    if optimizer_name == "adafactor":
        try:
            from transformers.optimization import Adafactor  # type: ignore
        except Exception as exc:
            raise TrainingConfigurationError(
                f"optimizer=adafactor requested but transformers Adafactor is unavailable: {exc}"
            ) from exc
        optimizer = Adafactor(
            trainable_params,
            lr=float(cfg.lr),
            weight_decay=float(cfg.weight_decay),
            scale_parameter=False,
            relative_step=False,
            warmup_init=False,
        )
    else:
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=float(cfg.lr),
            betas=(0.9, 0.999),
            weight_decay=float(cfg.weight_decay),
            foreach=False,
        )

    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(cfg.num_warmup_steps),
        num_training_steps=int(cfg.max_train_steps),
    )

    unet, null_embed, optimizer, train_dl, lr_scheduler = accelerator.prepare(
        unet, null_embed, optimizer, train_dl, lr_scheduler
    )
    vae = vae.to(accelerator.device, dtype=vae_runtime_dtype)
    vae.eval()
    vae_dtype = next(vae.parameters()).dtype
    if accelerator.is_main_process and vae_dtype != torch_dtype:
        accelerator.print(
            "[info] VAE is running in float32 (force_upcast) while UNet runs in "
            f"{str(torch_dtype)}."
        )

    ema_unet: Optional[EMAModel] = None
    ema_null: Optional[NullContextEMA] = None
    if bool(cfg.use_ema):
        ema_unet = EMAModel(
            accelerator.unwrap_model(unet).parameters(),
            decay=float(cfg.ema_decay),
            update_after_step=0,
            use_ema_warmup=False,
        )
        ema_null = NullContextEMA(accelerator.unwrap_model(null_embed), decay=float(cfg.ema_decay))
        accelerator.register_for_checkpointing(ema_unet)
        accelerator.register_for_checkpointing(ema_null)

    step = 0
    epoch = 0

    if cfg.resume_state_dir:
        resume_dir = Path(cfg.resume_state_dir).expanduser().resolve()
        accelerator.print(f"[resume] Loading accelerator state from {resume_dir}")
        accelerator.load_state(str(resume_dir))
        meta_path = resume_dir / "meta.json"
        if meta_path.is_file():
            with meta_path.open("r", encoding="utf-8") as handle:
                meta = json.load(handle)
            step = int(meta.get("step", 0))
            epoch = int(meta.get("epoch", 0))

    progress_start_step = int(step)
    progress_start_time = time.perf_counter()

    if accelerator.is_main_process:
        cond_channels = 1 if bool(cfg.use_rings_only) else (3 if bool(cfg.include_knot_maps) else 2)
        accelerator.print(
            f"[info] optimizer={optimizer_name} (requested={str(getattr(cfg, 'optimizer', 'auto'))})"
        )
        train_info = {
            "train_boards": len(train_ds),
            "val_boards": len(val_ds) if val_ds is not None else 0,
            "world_size": accelerator.num_processes,
            "batch_per_device": int(cfg.boards_per_batch),
            "grad_accum_steps": int(cfg.grad_accum_steps),
            "effective_board_batch": int(cfg.boards_per_batch) * accelerator.num_processes * int(cfg.grad_accum_steps),
            "max_train_steps": int(cfg.max_train_steps),
            "data_root": str(data_root),
            "model_family": str(cfg.model_family),
            "base_model_dir": str(model_root),
            "sd2_model_dir": str(model_root),
            "optimizer": str(getattr(cfg, "optimizer", "auto")),
            "optimizer_resolved": str(optimizer_name),
            "prediction_type": cfg.prediction_type,
            "cond_channels": int(cond_channels),
            "include_knot_maps": bool(cfg.include_knot_maps),
            "use_rings_only": bool(cfg.use_rings_only),
            "use_ema": bool(cfg.use_ema),
        }
        _save_json(output_root / "train_setup.json", train_info)
        _save_json(output_root / "train_config.json", asdict(cfg))
        if bool(cfg.show_cli_progress):
            accelerator.print(
                "[train] CLI progress enabled "
                f"(every {int(cfg.cli_progress_every)} steps, "
                f"start_step={int(step)}, max_steps={int(cfg.max_train_steps)})."
            )

    alpha_cumprod = noise_scheduler.alphas_cumprod

    while step < int(cfg.max_train_steps):
        epoch += 1
        for batch in train_dl:
            sanitized_non_finite = False
            cond_boards = batch["cond_nch"].to(accelerator.device)
            target_boards = batch["target_rgb"].to(accelerator.device)

            boards, surfaces, _, height, width = target_boards.shape
            batch_flat = boards * surfaces

            cond_channels = int(cond_boards.shape[2])
            cond_flat = cond_boards.view(batch_flat, cond_channels, height, width)
            target_flat = target_boards.view(batch_flat, 3, height, width)

            with torch.no_grad():
                cond_rgb = _build_cond_rgb_from_cond_nch(cond_flat).to(dtype=vae_dtype)
                target_flat_vae = target_flat.to(dtype=vae_dtype)
                cond_latent = vae.encode(cond_rgb).latent_dist.mean * float(vae_scaling_factor)
                z0 = vae.encode(target_flat_vae).latent_dist.mean * float(vae_scaling_factor)
                cond_latent, had_bad_cond = _sanitize_non_finite(cond_latent)
                z0, had_bad_z0 = _sanitize_non_finite(z0)
                sanitized_non_finite = bool(sanitized_non_finite or had_bad_cond or had_bad_z0)

                timesteps = torch.randint(
                    low=0,
                    high=noise_scheduler.config.num_train_timesteps,
                    size=(batch_flat,),
                    device=accelerator.device,
                    dtype=torch.long,
                )
                if bool(cfg.shared_board_noise):
                    shared = torch.randn(
                        (boards, 1, *z0.shape[1:]),
                        device=z0.device,
                        dtype=z0.dtype,
                    )
                    noise = shared.expand(-1, surfaces, *z0.shape[1:]).reshape_as(z0)
                else:
                    noise = torch.randn_like(z0)
                z_t = noise_scheduler.add_noise(z0, noise, timesteps)

            if cfg.guidance_drop_prob > 0.0:
                drop_mask = (torch.rand((batch_flat, 1, 1, 1), device=accelerator.device) < float(cfg.guidance_drop_prob)).to(
                    dtype=cond_latent.dtype
                )
                cond_latent_train = cond_latent * (1.0 - drop_mask)
            else:
                cond_latent_train = cond_latent

            with accelerator.accumulate(unet):
                with accelerator.autocast():
                    unet_in = torch.cat([z_t, cond_latent_train], dim=1)
                    token = null_embed(batch_flat, accelerator.device, dtype=unet_in.dtype)
                    added_cond_kwargs = _build_added_cond_kwargs(
                        spec=conditioning_spec,
                        batch_size=int(batch_flat),
                        image_size=int(width),
                        device=accelerator.device,
                        dtype=unet_in.dtype,
                    )
                    model_output = unet(
                        unet_in,
                        timesteps,
                        encoder_hidden_states=token,
                        added_cond_kwargs=added_cond_kwargs,
                    ).sample
                    model_output, had_bad_model_output = _sanitize_non_finite(model_output)
                    sanitized_non_finite = bool(sanitized_non_finite or had_bad_model_output)

                    target = _get_prediction_target(
                        noise_scheduler=noise_scheduler,
                        prediction_type=cfg.prediction_type,
                        z0=z0,
                        noise=noise,
                        timesteps=timesteps,
                    )
                    target, had_bad_target = _sanitize_non_finite(target)
                    sanitized_non_finite = bool(sanitized_non_finite or had_bad_target)

                    loss_per = F.mse_loss(model_output.float(), target.float(), reduction="none").mean(dim=(1, 2, 3))
                    if float(cfg.min_snr_gamma) > 0.0:
                        weights = _min_snr_weights(
                            prediction_type=cfg.prediction_type,
                            alpha_cumprod=alpha_cumprod,
                            timesteps=timesteps,
                            gamma=float(cfg.min_snr_gamma),
                            dtype=loss_per.dtype,
                            device=loss_per.device,
                        )
                        noise_loss = (loss_per * weights).mean()
                    else:
                        noise_loss = loss_per.mean()

                    a_t = _extract_to_shape(alpha_cumprod, timesteps, z_t).sqrt()
                    sigma_t = torch.sqrt(torch.clamp(1.0 - _extract_to_shape(alpha_cumprod, timesteps, z_t), min=1e-8))
                    x0_pred = _compute_x0_prediction(
                        prediction_type=cfg.prediction_type,
                        z_t=z_t,
                        model_output=model_output,
                        alpha_t=a_t,
                        sigma_t=sigma_t,
                    )
                    x0_pred, had_bad_x0 = _sanitize_non_finite(x0_pred)
                    sanitized_non_finite = bool(sanitized_non_finite or had_bad_x0)
                    recon_loss = F.smooth_l1_loss(x0_pred.float(), z0.float())
                    consistency_loss = x0_pred.new_tensor(0.0)
                    if float(cfg.cross_surface_consistency_weight) > 0.0:
                        consistency_loss = _cross_surface_consistency_loss(
                            x0_pred=x0_pred,
                            z0_target=z0,
                            boards=boards,
                            surfaces=surfaces,
                        )
                    seam_loss = x0_pred.new_tensor(0.0)
                    if float(cfg.seam_consistency_weight) > 0.0:
                        seam_loss = _seam_consistency_loss(
                            x0_pred=x0_pred,
                            z0_target=z0,
                            boards=boards,
                            surfaces=surfaces,
                            strip_width=int(cfg.seam_strip_width),
                        )

                    total_loss = (
                        noise_loss
                        + float(cfg.latent_recon_weight) * recon_loss
                        + float(cfg.cross_surface_consistency_weight) * consistency_loss
                        + float(cfg.seam_consistency_weight) * seam_loss
                    )
                    _ensure_finite_tensor("train/total_loss", total_loss)

                accelerator.backward(total_loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        list(unet.parameters()) + list(null_embed.parameters()),
                        max_norm=float(cfg.max_grad_norm),
                    )

                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                lr_scheduler.step()

                if (
                    bool(cfg.use_ema)
                    and ema_unet is not None
                    and ema_null is not None
                    and accelerator.sync_gradients
                    and (step % int(cfg.ema_update_every) == 0)
                ):
                    ema_unet.step(accelerator.unwrap_model(unet).parameters())
                    if accelerator.is_main_process:
                        ema_null.step(accelerator.unwrap_model(null_embed))

            if not accelerator.sync_gradients:
                continue

            step += 1

            if accelerator.is_main_process:
                log_payload = {
                    "train/loss_total": float(total_loss.detach().item()),
                    "train/loss_noise": float(noise_loss.detach().item()),
                    "train/loss_recon_latent": float(recon_loss.detach().item()),
                    "train/loss_cross_surface": float(consistency_loss.detach().item()),
                    "train/loss_seam": float(seam_loss.detach().item()),
                    "train/sanitized_non_finite": int(1 if sanitized_non_finite else 0),
                    "train/lr": float(lr_scheduler.get_last_lr()[0]),
                    "train/epoch": int(epoch),
                    "train/step": int(step),
                }
                if wandb is not None:
                    wandb.log(log_payload, step=step)
                else:
                    accelerator.print(
                        f"[step {step}] total={log_payload['train/loss_total']:.5f} "
                        f"noise={log_payload['train/loss_noise']:.5f} recon={log_payload['train/loss_recon_latent']:.5f} "
                        f"sanitized={log_payload['train/sanitized_non_finite']}"
                    )
                if sanitized_non_finite:
                    accelerator.print(
                        f"[warn] step={step}: non-finite tensors were sanitized to keep training stable."
                    )
                if bool(cfg.show_cli_progress):
                    report_every = max(1, int(cfg.cli_progress_every))
                    if step == 1 or step == int(cfg.max_train_steps) or (step % report_every == 0):
                        elapsed = max(1e-9, float(time.perf_counter() - progress_start_time))
                        progressed_steps = max(0, int(step - progress_start_step))
                        steps_per_sec = float(progressed_steps) / elapsed if progressed_steps > 0 else 0.0
                        remaining_steps = max(0, int(cfg.max_train_steps) - int(step))
                        eta_seconds = (
                            float(remaining_steps) / steps_per_sec
                            if steps_per_sec > 0.0
                            else float("inf")
                        )
                        pct = 100.0 * float(step) / float(max(1, int(cfg.max_train_steps)))
                        accelerator.print(
                            f"[train] step {step}/{int(cfg.max_train_steps)} ({pct:5.1f}%) "
                            f"loss={log_payload['train/loss_total']:.5f} "
                            f"lr={log_payload['train/lr']:.3e} "
                            f"speed={steps_per_sec:.2f} step/s "
                            f"eta={_format_duration(eta_seconds)}"
                        )

            if step % int(cfg.export_every) == 0 or step == int(cfg.max_train_steps):
                state_dir = output_root / f"state_{step:07d}"
                _ensure_checkpoint_free_space(output_root, context=f"checkpoint save at step {step}")
                try:
                    if accelerator.is_main_process:
                        state_dir.mkdir(parents=True, exist_ok=True)
                        _save_json(state_dir / "meta.json", {"step": step, "epoch": epoch})
                    accelerator.save_state(str(state_dir))

                    if bool(cfg.use_ema) and ema_unet is not None and ema_null is not None:
                        ema_unet.store(accelerator.unwrap_model(unet).parameters())
                        ema_unet.copy_to(accelerator.unwrap_model(unet).parameters())
                        if accelerator.is_main_process:
                            ema_null.store(accelerator.unwrap_model(null_embed))
                            ema_null.copy_to(accelerator.unwrap_model(null_embed))

                        _export_checkpoint_bundle(
                            cfg=cfg,
                            model_family=str(cfg.model_family),
                            model_root=model_root,
                            accelerator=accelerator,
                            unet=unet,
                            null_embed=null_embed,
                            step=step,
                            use_ema=True,
                            suffix="ema",
                        )

                        if accelerator.is_main_process:
                            ema_null.restore(accelerator.unwrap_model(null_embed))
                        ema_unet.restore(accelerator.unwrap_model(unet).parameters())

                    _export_checkpoint_bundle(
                        cfg=cfg,
                        model_family=str(cfg.model_family),
                        model_root=model_root,
                        accelerator=accelerator,
                        unet=unet,
                        null_embed=null_embed,
                        step=step,
                        use_ema=False,
                        suffix="raw",
                    )
                except TrainingConfigurationError:
                    raise
                except (RuntimeError, OSError) as exc:
                    if _looks_like_filesystem_write_error(exc):
                        raise TrainingConfigurationError(
                            f"Checkpoint export failed at step {step}. "
                            f"Output root: '{output_root}'. "
                            f"Filesystem usage: {_disk_usage_summary(output_root)}. "
                            "This is usually caused by insufficient disk space or filesystem write errors "
                            "while writing model/optimizer state. "
                            "Free disk space or switch --output-dir to a roomier filesystem, "
                            "then resume from the latest valid state directory."
                        ) from exc
                    raise

            if val_dl is not None and (step % int(cfg.validate_every) == 0 or step == int(cfg.max_train_steps)):
                should_run_here = (
                    accelerator.is_main_process if bool(cfg.validate_on_main_process_only) else True
                )
                accelerator.wait_for_everyone()
                if should_run_here:
                    vae.eval()
                    if accelerator.device.type == "cuda":
                        torch.cuda.empty_cache()
                    try:
                        if val_iter is None:
                            val_iter = iter(val_dl)
                        val_batch = next(val_iter)
                    except StopIteration:
                        val_iter = iter(val_dl)
                        val_batch = next(val_iter)

                    val_count = max(1, int(cfg.val_boards))
                    cond_cpu = val_batch["cond_nch"][:val_count]
                    # Validation targets are only used for panel rendering; keep them on CPU to
                    # avoid unnecessary VRAM pressure during SDXL validation sampling.
                    target_cpu = val_batch["target_rgb"][:val_count]
                    preds_cpu: List[torch.Tensor] = []

                    ema_swapped = False
                    if bool(cfg.use_ema) and ema_unet is not None and ema_null is not None:
                        ema_unet.store(accelerator.unwrap_model(unet).parameters())
                        ema_unet.copy_to(accelerator.unwrap_model(unet).parameters())
                        if accelerator.is_main_process:
                            ema_null.store(accelerator.unwrap_model(null_embed))
                            ema_null.copy_to(accelerator.unwrap_model(null_embed))
                        ema_swapped = True

                    try:
                        with torch.no_grad():
                            # Sample one board at a time to keep validation VRAM bounded on SDXL.
                            for bidx in range(int(cond_cpu.shape[0])):
                                cond_chunk = cond_cpu[bidx : bidx + 1].to(accelerator.device)
                                pred_chunk = _sample_validation_batch(
                                    vae=vae,
                                    unet=accelerator.unwrap_model(unet),
                                    null_embed=accelerator.unwrap_model(null_embed),
                                    scheduler=val_ddim_scheduler,
                                    conditioning_spec=conditioning_spec,
                                    cond_boards=cond_chunk,
                                    device=accelerator.device,
                                    dtype=torch_dtype,
                                    latent_scale=float(vae_scaling_factor),
                                    ddim_steps=int(cfg.val_ddim_steps),
                                    guidance_scale=float(cfg.val_guidance_scale),
                                )
                                preds_cpu.append(pred_chunk.detach().cpu())
                                del cond_chunk
                                del pred_chunk

                        if accelerator.is_main_process and wandb is not None:
                            boards = int(cond_cpu.shape[0])
                            panels: List[Any] = []
                            for bidx in range(boards):
                                panel = _build_validation_panel(
                                    cond_nch=cond_cpu[bidx],
                                    pred_rgb=preds_cpu[bidx],
                                    target_rgb=target_cpu[bidx],
                                )
                                panels.append(wandb.Image(panel, caption=f"step={step} board={val_batch['stem'][bidx]}"))
                            wandb.log({"val/panels": panels, "val/step": step}, step=step)
                    except RuntimeError as exc:
                        if _looks_like_cuda_oom(exc):
                            if accelerator.is_main_process:
                                accelerator.print(
                                    f"[warn] step={step}: validation skipped due CUDA OOM; training will continue."
                                )
                            if accelerator.device.type == "cuda":
                                torch.cuda.empty_cache()
                        else:
                            raise
                    finally:
                        if ema_swapped and ema_unet is not None and ema_null is not None:
                            if accelerator.is_main_process:
                                ema_null.restore(accelerator.unwrap_model(null_embed))
                            ema_unet.restore(accelerator.unwrap_model(unet).parameters())
                    if accelerator.device.type == "cuda":
                        torch.cuda.empty_cache()
                accelerator.wait_for_everyone()

            if step >= int(cfg.max_train_steps):
                break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process and wandb is not None:
        wandb.finish()


def config_from_args(args: Any) -> TrainingConfig:
    model_family = _normalize_model_family(str(getattr(args, "model_family", "sd2")))
    use_ema = _resolve_use_ema(getattr(args, "use_ema", "auto"), model_family)
    cfg = TrainingConfig(
        data_root=str(args.data_root),
        output_dir=str(args.output_dir),
        sd2_model_dir=str(args.sd2_model_dir),
        model_family=str(model_family),
        sdxl_model_dir=str(getattr(args, "sdxl_model_dir", "")),
        image_size=int(args.image_size),
        boards_per_batch=int(args.boards_per_batch),
        grad_accum_steps=int(args.grad_accum_steps),
        num_workers=int(args.num_workers),
        train_pin_memory=_as_bool(args.train_pin_memory),
        max_train_steps=int(args.max_train_steps),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        optimizer=str(getattr(args, "optimizer", "auto")),
        num_warmup_steps=int(args.num_warmup_steps),
        max_grad_norm=float(args.max_grad_norm),
        guidance_drop_prob=float(args.guidance_drop_prob),
        include_knot_maps=_as_bool(getattr(args, "include_knot_maps", False)),
        use_rings_only=_as_bool(getattr(args, "use_rings_only", False)),
        num_train_timesteps=int(args.num_train_timesteps),
        prediction_type=str(args.prediction_type),
        min_snr_gamma=float(args.min_snr_gamma),
        latent_recon_weight=float(args.latent_recon_weight),
        cross_surface_consistency_weight=float(args.cross_surface_consistency_weight),
        seam_consistency_weight=float(args.seam_consistency_weight),
        seam_strip_width=int(args.seam_strip_width),
        shared_board_noise=_as_bool(args.shared_board_noise),
        ema_decay=float(args.ema_decay),
        ema_update_every=int(args.ema_update_every),
        use_ema=bool(use_ema),
        val_ratio=float(args.val_ratio),
        validate_every=int(args.validate_every),
        val_boards=int(args.val_boards),
        val_ddim_steps=int(args.val_ddim_steps),
        val_guidance_scale=float(args.val_guidance_scale),
        val_num_workers=int(args.val_num_workers),
        val_pin_memory=_as_bool(args.val_pin_memory),
        validate_on_main_process_only=_as_bool(args.validate_on_main_process_only),
        export_every=int(args.export_every),
        export_ddim_steps=int(args.export_ddim_steps),
        export_guidance_scale=float(args.export_guidance_scale),
        export_img2img_strength=float(args.export_img2img_strength),
        mixed_precision=str(args.mixed_precision),
        enable_gradient_checkpointing=_as_bool(args.enable_gradient_checkpointing),
        enable_xformers=_as_bool(args.enable_xformers),
        seed=int(args.seed),
        show_cli_progress=_as_bool(getattr(args, "show_cli_progress", True)),
        cli_progress_every=int(getattr(args, "cli_progress_every", 10)),
        use_wandb=not _as_bool(args.disable_wandb),
        wandb_project=str(args.wandb_project),
        wandb_run=str(args.wandb_run),
        wandb_mode=str(args.wandb_mode),
        resume_state_dir=str(args.resume_state_dir),
    )
    cfg.model_family = _normalize_model_family(cfg.model_family)
    if cfg.model_family == "sdxl":
        if not cfg.sdxl_model_dir:
            cfg.sdxl_model_dir = str(_default_sdxl_model_dir())
    else:
        if not cfg.sd2_model_dir:
            cfg.sd2_model_dir = str(_default_sd2_model_dir())
    return cfg
