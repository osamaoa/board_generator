from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import importlib.util
import json
import os
from pathlib import Path
import threading
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from PIL import Image

from .knot_map import build_knot_map_from_fiber_gray01


_DEFAULT_LATENT_SCALE = 0.18215
_SDXL_NUM_TIME_IDS = 6


class PhotorealisticUnavailableError(RuntimeError):
    """Raised when photorealistic inference cannot run in current environment."""


class PhotorealisticInferenceError(RuntimeError):
    """Raised when photorealistic inference fails while generating outputs."""


@dataclass
class PhotorealisticDefaults:
    image_size: int = 512
    ddim_steps: int = 50
    guidance_scale: float = 2.0
    use_img2img_strength: float = 0.0
    include_knot_maps: bool = False
    use_rings_only: bool = False
    model_family: str = "sd2"
    base_model_dir: str = ""
    # Keep for backward compatibility with old checkpoint config.json files.
    sd2_model_dir: str = ""
    prediction_type: str = "epsilon"


@dataclass
class _ModelConditioningSpec:
    model_family: str
    pooled_embed_dim: int = 0
    num_time_ids: int = 0

    @property
    def requires_added_cond_kwargs(self) -> bool:
        return bool(self.model_family == "sdxl")


class _PhotorealisticRuntime:
    def __init__(self, checkpoint_dir: Optional[Path] = None):
        self._checkpoint_dir = checkpoint_dir or self._default_checkpoint_dir()
        self._lock = threading.Lock()
        self._is_loaded = False

        self._torch = None
        self._torch_dtype = None
        self._device = None

        self._vae = None
        self._unet = None
        self._scheduler = None
        self._null_token = None
        self._vae_dtype = None
        self._model_conditioning_spec = _ModelConditioningSpec(model_family="sd2")
        self._latent_scale = float(_DEFAULT_LATENT_SCALE)

        self._defaults = PhotorealisticDefaults()

    @staticmethod
    def _repo_root() -> Path:
        # backend/app/core -> backend/app -> backend -> repo root
        return Path(__file__).resolve().parents[3]

    @staticmethod
    def _default_checkpoint_dir() -> Path:
        env_dir = str(os.environ.get("PHOTOREALISTIC_CHECKPOINT_DIR") or "").strip()
        if env_dir:
            return Path(env_dir).expanduser()
        repo_root = _PhotorealisticRuntime._repo_root()
        preferred = repo_root / "photorealistic_model_checkpoint"
        if preferred.exists():
            return preferred
        # Backward compatibility for an old misspelled checkpoint folder.
        legacy = repo_root / "photorealisic_model_checkpoint"
        if legacy.exists():
            return legacy
        return preferred

    @staticmethod
    def _default_sd2_model_dir() -> Path:
        env_dir = str(
            os.environ.get("PHOTOREALISTIC_SD2_MODEL_DIR")
            or os.environ.get("SD2_MODEL_DIR")
            or ""
        ).strip()
        if env_dir:
            return Path(env_dir).expanduser()
        return _PhotorealisticRuntime._repo_root() / "SD2_model"

    @staticmethod
    def _default_sdxl_model_dir() -> Path:
        env_dir = str(
            os.environ.get("PHOTOREALISTIC_SDXL_MODEL_DIR")
            or os.environ.get("SDXL_MODEL_DIR")
            or ""
        ).strip()
        if env_dir:
            return Path(env_dir).expanduser()
        return _PhotorealisticRuntime._repo_root() / "SDXL_model"

    @staticmethod
    def _normalize_model_family(value: Any, *, default: str = "sd2") -> str:
        text = str(value or default).strip().lower()
        if text in {"sd2", "sdxl"}:
            return text
        return str(default)

    def _resolve_base_model_dir(self, defaults: Optional[PhotorealisticDefaults] = None) -> Path:
        cfg = defaults or self._defaults
        family = self._normalize_model_family(getattr(cfg, "model_family", "sd2"))
        fallback = self._default_sdxl_model_dir() if family == "sdxl" else self._default_sd2_model_dir()
        configured = str(getattr(cfg, "base_model_dir", "") or "").strip()
        if not configured:
            configured = str(getattr(cfg, "sd2_model_dir", "") or "").strip()
        if not configured:
            return fallback

        configured_path = Path(configured).expanduser()
        if not configured_path.is_absolute():
            return (self._repo_root() / configured_path).resolve()

        try:
            configured_path.resolve().relative_to(self._repo_root().resolve())
        except Exception:
            # Ignore stale absolute paths from another environment; use current-project model folder.
            return fallback
        return configured_path

    @staticmethod
    def _resolve_cross_attention_dim(unet) -> int:
        raw = getattr(unet.config, "cross_attention_dim", 0)
        if isinstance(raw, (list, tuple)):
            if not raw:
                raise ValueError("UNet cross_attention_dim is empty.")
            raw = raw[0]
        dim = int(raw)
        if dim <= 0:
            raise ValueError(f"Invalid UNet cross_attention_dim: {raw!r}.")
        return dim

    def _build_model_conditioning_spec(self, *, unet) -> _ModelConditioningSpec:
        family = self._normalize_model_family(getattr(self._defaults, "model_family", "sd2"))
        if family != "sdxl":
            return _ModelConditioningSpec(model_family=family)

        add_time_dim = int(getattr(unet.config, "addition_time_embed_dim", 256) or 256)
        if add_time_dim <= 0:
            raise ValueError(f"Invalid SDXL addition_time_embed_dim: {add_time_dim}.")

        linear_1 = getattr(getattr(unet, "add_embedding", None), "linear_1", None)
        if linear_1 is None or not hasattr(linear_1, "in_features"):
            raise ValueError(
                "SDXL UNet missing add_embedding.linear_1; cannot infer pooled text embedding size."
            )
        total_in = int(linear_1.in_features)
        pooled_dim = total_in - int(_SDXL_NUM_TIME_IDS * add_time_dim)
        if pooled_dim <= 0:
            raise ValueError(
                "Failed to infer SDXL pooled text embedding dimension from UNet config."
            )
        return _ModelConditioningSpec(
            model_family=family,
            pooled_embed_dim=int(pooled_dim),
            num_time_ids=int(_SDXL_NUM_TIME_IDS),
        )

    def _build_added_cond_kwargs(
        self,
        *,
        batch_size: int,
        image_size: int,
        dtype,
    ) -> Optional[Dict[str, Any]]:
        spec = self._model_conditioning_spec
        if not spec.requires_added_cond_kwargs:
            return None
        text_embeds = self._torch.zeros(
            (int(batch_size), int(spec.pooled_embed_dim)),
            device=self._device,
            dtype=dtype,
        )
        time_row = self._torch.tensor(
            [
                float(image_size),
                float(image_size),
                0.0,
                0.0,
                float(image_size),
                float(image_size),
            ],
            device=self._device,
            dtype=dtype,
        )
        time_ids = time_row.unsqueeze(0).repeat(int(batch_size), 1)
        return {
            "text_embeds": text_embeds,
            "time_ids": time_ids,
        }

    @staticmethod
    def _module_exists(module_name: str) -> bool:
        try:
            return importlib.util.find_spec(module_name) is not None
        except Exception:
            return False

    def _detect_cuda(self) -> tuple[bool, str]:
        if self._torch is not None and self._device is not None:
            if self._device.type == "cuda":
                try:
                    dev_index = self._device.index
                    if dev_index is None:
                        dev_index = int(self._torch.cuda.current_device())
                    return True, str(self._torch.cuda.get_device_name(dev_index))
                except Exception:
                    return True, "CUDA"
            return False, "CPU"

        try:
            import torch
        except Exception:
            return False, "CPU"

        try:
            if bool(torch.cuda.is_available()):
                dev_index = int(torch.cuda.current_device())
                return True, str(torch.cuda.get_device_name(dev_index))
        except Exception:
            return False, "CPU"
        return False, "CPU"

    def _predict_unet(self, unet_in, t_tensor, enc_states, *, added_cond_kwargs=None):
        if self._unet is None:
            raise RuntimeError("UNet is not loaded.")
        return self._unet(
            unet_in,
            t_tensor,
            encoder_hidden_states=enc_states,
            added_cond_kwargs=added_cond_kwargs,
        ).sample

    def capability(self) -> Dict[str, Any]:
        missing = self._missing_checkpoint_files()
        if missing:
            return {
                "available": False,
                "reason": f"Missing checkpoint files: {', '.join(missing)}",
                "loaded": bool(self._is_loaded),
            }

        defaults = self._defaults if self._is_loaded else self._load_defaults()
        model_family = self._normalize_model_family(getattr(defaults, "model_family", "sd2"))
        base_model_dir = self._resolve_base_model_dir(defaults)
        if not base_model_dir.is_dir():
            return {
                "available": False,
                "reason": f"Missing local {model_family.upper()} model folder: {base_model_dir}",
                "loaded": bool(self._is_loaded),
            }
        missing_model_parts = [
            name for name in ["vae", "unet", "scheduler"] if not (base_model_dir / name).is_dir()
        ]
        if missing_model_parts:
            return {
                "available": False,
                "reason": (
                    f"{model_family.upper()} model is missing required subfolders: "
                    f"{', '.join(missing_model_parts)}"
                ),
                "loaded": bool(self._is_loaded),
            }

        if not self._module_exists("torch"):
            return {
                "available": False,
                "reason": "PyTorch is not installed.",
                "loaded": bool(self._is_loaded),
            }

        if not self._module_exists("diffusers") or not self._module_exists("safetensors.torch"):
            return {
                "available": False,
                "reason": "Required inference dependencies are missing (diffusers/safetensors).",
                "loaded": bool(self._is_loaded),
            }

        cuda_ok, gpu_name = self._detect_cuda()
        if not cuda_ok:
            return {
                "available": False,
                "reason": "Photorealistic generation requires a CUDA GPU.",
                "cuda_available": False,
                "loaded": bool(self._is_loaded),
            }

        return {
            "available": True,
            "reason": "",
            "checkpoint_dir": str(self._checkpoint_dir),
            "model_family": str(model_family),
            "base_model_dir": str(base_model_dir),
            "sd2_model_dir": str(base_model_dir),
            "device": "cuda",
            "gpu_name": gpu_name,
            "cuda_available": True,
            "recommended_ddim_steps": int(defaults.ddim_steps),
            "loaded": bool(self._is_loaded),
        }

    def _missing_checkpoint_files(self) -> list[str]:
        needed = [
            "config.json",
            "unet.safetensors",
            "null_embed.safetensors",
        ]
        return [name for name in needed if not (self._checkpoint_dir / name).is_file()]

    def _load_defaults(self) -> PhotorealisticDefaults:
        cfg_path = self._checkpoint_dir / "config.json"
        if not cfg_path.is_file():
            return PhotorealisticDefaults()

        try:
            with cfg_path.open("r", encoding="utf-8") as f:
                raw = json.load(f)
        except Exception:
            return PhotorealisticDefaults()

        def to_int(val: Any, fallback: int) -> int:
            try:
                parsed = int(val)
                return parsed if parsed > 0 else fallback
            except Exception:
                return fallback

        def to_float(val: Any, fallback: float) -> float:
            try:
                return float(val)
            except Exception:
                return fallback

        prediction_type = str(raw.get("prediction_type") or "epsilon")
        if prediction_type not in {"epsilon", "v_prediction"}:
            prediction_type = "epsilon"
        model_family = self._normalize_model_family(raw.get("model_family", "sd2"))
        base_model_dir = str(raw.get("base_model_dir") or "").strip()
        legacy_sd2_dir = str(raw.get("sd2_model_dir") or "").strip()
        if not base_model_dir and legacy_sd2_dir:
            base_model_dir = legacy_sd2_dir
        if not base_model_dir:
            if model_family == "sdxl":
                base_model_dir = str(self._default_sdxl_model_dir())
            else:
                base_model_dir = str(self._default_sd2_model_dir())

        return PhotorealisticDefaults(
            image_size=to_int(raw.get("image_size", 512), 512),
            ddim_steps=to_int(raw.get("ddim_steps", 50), 50),
            guidance_scale=to_float(raw.get("guidance_scale", 2.0), 2.0),
            use_img2img_strength=to_float(raw.get("use_img2img_strength", 0.0), 0.0),
            include_knot_maps=self._to_bool(raw.get("include_knot_maps", False), default=False),
            use_rings_only=self._to_bool(raw.get("use_rings_only", False), default=False),
            model_family=str(model_family),
            base_model_dir=str(base_model_dir),
            sd2_model_dir=str(base_model_dir),
            prediction_type=prediction_type,
        )

    @staticmethod
    def _to_bool(value: Any, default: bool = False) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            text = value.strip().lower()
            if text in {"1", "true", "yes", "on"}:
                return True
            if text in {"0", "false", "no", "off"}:
                return False
            return bool(default)
        if value is None:
            return bool(default)
        return bool(value)

    @staticmethod
    def _load_state(base_path: Path, load_file, torch):
        st_path = str(base_path) + ".safetensors"
        pt_path = str(base_path) + ".pt"
        if Path(st_path).is_file():
            return load_file(st_path, device="cpu")
        if Path(pt_path).is_file():
            return torch.load(pt_path, map_location="cpu")
        raise FileNotFoundError(f"Neither {st_path} nor {pt_path} exists.")

    @staticmethod
    def _expand_unet_in_channels(unet, torch, new_in: int = 8):
        old_conv = unet.conv_in
        if old_conv.in_channels == new_in:
            return unet
        if old_conv.in_channels != 4:
            raise ValueError(f"Expected UNet input channels=4, got {old_conv.in_channels}.")

        new_conv = torch.nn.Conv2d(
            in_channels=new_in,
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
        unet.config.in_channels = new_in
        return unet

    def _ensure_loaded(self) -> None:
        if self._is_loaded:
            return

        with self._lock:
            if self._is_loaded:
                return

            cap = self.capability()
            if not bool(cap.get("available")):
                raise PhotorealisticUnavailableError(
                    str(cap.get("reason") or "Photorealistic inference unavailable.")
                )

            import torch
            from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
            from safetensors.torch import load_file

            self._defaults = self._load_defaults()

            if not bool(torch.cuda.is_available()):
                raise PhotorealisticUnavailableError(
                    "Photorealistic generation requires a CUDA GPU."
                )

            # CUDA runtime tuning for better inference throughput.
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            try:
                torch.set_float32_matmul_precision("high")
            except Exception:
                pass

            self._torch = torch
            self._device = torch.device("cuda")
            self._torch_dtype = torch.float16
            model_family = self._normalize_model_family(getattr(self._defaults, "model_family", "sd2"))
            base_model_dir = self._resolve_base_model_dir(self._defaults)

            vae = AutoencoderKL.from_pretrained(
                str(base_model_dir),
                subfolder="vae",
                torch_dtype=self._torch_dtype,
                local_files_only=True,
            )
            vae_force_upcast = bool(getattr(getattr(vae, "config", None), "force_upcast", False))
            enable_force_upcast_env = str(
                os.environ.get("PHOTOREALISTIC_ENABLE_VAE_FORCE_UPCAST") or ""
            ).strip().lower()
            if enable_force_upcast_env:
                enable_force_upcast = enable_force_upcast_env in {"1", "true", "yes", "on"}
            else:
                # SDXL commonly requires fp32 VAE decode for stable, non-collapsed outputs.
                enable_force_upcast = bool(model_family == "sdxl")
            vae_runtime_dtype = self._torch_dtype
            if (
                bool(enable_force_upcast)
                and self._device.type == "cuda"
                and bool(vae_force_upcast)
                and self._torch_dtype in (torch.float16, torch.bfloat16)
            ):
                # SDXL VAE is numerically unstable in fp16/bf16; run VAE in fp32.
                vae_runtime_dtype = torch.float32
            vae = vae.to(self._device, dtype=vae_runtime_dtype)
            vae.eval()
            self._latent_scale = float(getattr(vae.config, "scaling_factor", _DEFAULT_LATENT_SCALE))

            # Faster startup: build UNet from config only, then load fine-tuned checkpoint weights.
            # This avoids loading base-model UNet weights that are immediately overwritten.
            unet_config = UNet2DConditionModel.load_config(
                str(base_model_dir),
                subfolder="unet",
                local_files_only=True,
            )
            unet = UNet2DConditionModel.from_config(unet_config)
            unet = self._expand_unet_in_channels(unet, torch=torch, new_in=8)
            unet_state = self._load_state(self._checkpoint_dir / "unet", load_file=load_file, torch=torch)
            unet.load_state_dict(unet_state, strict=True)
            unet = unet.to(self._device, dtype=self._torch_dtype)
            unet.eval()

            null_state = self._load_state(self._checkpoint_dir / "null_embed", load_file=load_file, torch=torch)
            null_token = null_state.get("token") if isinstance(null_state, dict) else None
            if null_token is None:
                raise RuntimeError("null_embed checkpoint does not contain 'token'.")
            null_token = null_token.to(self._device, dtype=self._torch_dtype)
            if null_token.ndim != 3 or null_token.shape[0] != 1:
                raise RuntimeError(f"Unexpected null token shape: {tuple(null_token.shape)}")
            expected_cross = self._resolve_cross_attention_dim(unet)
            if int(null_token.shape[-1]) != int(expected_cross):
                raise RuntimeError(
                    "null_embed token dim does not match UNet cross_attention_dim "
                    f"({int(null_token.shape[-1])} vs {int(expected_cross)})."
                )

            try:
                unet.enable_xformers_memory_efficient_attention()
            except Exception:
                pass

            scheduler = DDIMScheduler.from_pretrained(
                str(base_model_dir),
                subfolder="scheduler",
                local_files_only=True,
            )
            scheduler.config.prediction_type = self._defaults.prediction_type

            self._vae = vae
            self._vae_dtype = vae_runtime_dtype
            self._unet = unet
            self._scheduler = scheduler
            self._null_token = null_token
            self._model_conditioning_spec = self._build_model_conditioning_spec(unet=unet)
            self._is_loaded = True

    def preload(self) -> Dict[str, Any]:
        self._ensure_loaded()
        return {"loaded": bool(self._is_loaded)}

    def _gray_tensor_from_png_bytes(self, png_bytes: bytes, image_size: int):
        if not png_bytes:
            raise PhotorealisticInferenceError("Empty input image bytes.")

        image = Image.open(BytesIO(png_bytes)).convert("L")
        if image.size != (image_size, image_size):
            image = image.resize((image_size, image_size), resample=Image.BICUBIC)
        arr = np.asarray(image, dtype=np.float32) / 255.0
        tensor = self._torch.from_numpy(arr).unsqueeze(0)
        return tensor.mul(2.0).sub(1.0)

    @staticmethod
    def _tensor_to_png_bytes(tensor) -> bytes:
        img = tensor.detach().permute(1, 2, 0).float().cpu().numpy()
        img_u8 = np.clip(np.rint(img * 255.0), 0, 255).astype(np.uint8)
        buf = BytesIO()
        Image.fromarray(img_u8).save(buf, format="PNG", optimize=False)
        return buf.getvalue()

    def _resolve_inference_params(
        self,
        *,
        ddim_steps: Optional[int],
        guidance_scale: Optional[float],
        use_img2img_strength: Optional[float],
    ) -> tuple[int, float, float]:
        default_steps = int(self._defaults.ddim_steps)
        steps = max(1, int(ddim_steps if ddim_steps is not None else default_steps))
        guidance = float(guidance_scale if guidance_scale is not None else self._defaults.guidance_scale)
        img2img = float(
            use_img2img_strength
            if use_img2img_strength is not None
            else self._defaults.use_img2img_strength
        )
        img2img = min(1.0, max(0.0, img2img))
        return steps, guidance, img2img

    def _build_condition_batch(
        self,
        board_inputs: Sequence[Dict[str, Dict[str, bytes]]],
        *,
        image_size: int,
        include_knot_maps: bool,
        use_rings_only: bool,
    ):
        torch = self._torch
        cond_rgb_list = []
        for board in board_inputs:
            ring_pngs = board.get("rings") or {}
            fiber_pngs = board.get("fibers") or {}
            knot_pngs = board.get("knot_maps") or board.get("knots") or {}
            for idx in range(1, 5):
                ring_key = f"rings_{idx}"
                fiber_key = f"fiber_{idx}"
                if ring_key not in ring_pngs:
                    raise PhotorealisticInferenceError(
                        f"Missing required input image: {ring_key}."
                    )

                ring = self._gray_tensor_from_png_bytes(ring_pngs[ring_key], image_size)
                if bool(use_rings_only):
                    fiber = torch.zeros_like(ring)
                else:
                    if fiber_key not in fiber_pngs:
                        raise PhotorealisticInferenceError(
                            f"Missing required input image: {fiber_key}."
                        )
                    fiber = self._gray_tensor_from_png_bytes(fiber_pngs[fiber_key], image_size)

                knot = None
                if include_knot_maps:
                    for knot_key in (f"knot_{idx}", f"knot_map_{idx}", f"knots_{idx}"):
                        if knot_key in knot_pngs:
                            knot = self._gray_tensor_from_png_bytes(knot_pngs[knot_key], image_size)
                            break
                    if knot is None:
                        fiber01 = ((fiber[0].detach().cpu().numpy() + 1.0) * 0.5).astype(np.float32, copy=False)
                        knot01 = build_knot_map_from_fiber_gray01(fiber01)
                        knot = torch.from_numpy(knot01).unsqueeze(0).to(dtype=fiber.dtype).mul(2.0).sub(1.0)
                else:
                    knot = torch.zeros_like(ring)

                cond_rgb = torch.stack([ring[0], fiber[0], knot[0]], dim=0)
                cond_rgb_list.append(cond_rgb)

        if not cond_rgb_list:
            raise PhotorealisticInferenceError("No photorealistic board inputs were provided.")

        return torch.stack(cond_rgb_list, dim=0).to(self._device, dtype=self._torch_dtype)

    def _run_diffusion_from_condition_batch(
        self,
        cond_rgb_batch,
        *,
        steps: int,
        guidance: float,
        img2img: float,
    ):
        torch = self._torch
        self._scheduler.set_timesteps(int(steps), device=self._device)
        vae_dtype = self._vae_dtype if self._vae_dtype is not None else self._torch_dtype

        with torch.inference_mode():
            cond_rgb_for_vae = cond_rgb_batch.to(dtype=vae_dtype)
            cond_latent = self._vae.encode(cond_rgb_for_vae).latent_dist.sample() * float(self._latent_scale)
            cond_latent = cond_latent.to(dtype=self._torch_dtype)
            cond_latent = torch.nan_to_num(cond_latent, nan=0.0, posinf=100.0, neginf=-100.0)
            cond_latent = cond_latent.clamp(min=-100.0, max=100.0)
            with torch.autocast(device_type="cuda", dtype=self._torch_dtype):
                cond_in = torch.cat([torch.zeros_like(cond_latent), cond_latent], dim=0)
                enc_states = self._null_token.expand(cond_latent.shape[0], -1, -1)
                enc_states = torch.cat([enc_states, enc_states], dim=0)
                added_cond_kwargs = self._build_added_cond_kwargs(
                    batch_size=int(cond_in.shape[0]),
                    image_size=int(cond_rgb_batch.shape[-1]),
                    dtype=self._torch_dtype,
                )

                if img2img > 0.0:
                    t_index = int((len(self._scheduler.timesteps) - 1) * img2img)
                    t_start = self._scheduler.timesteps[t_index]
                    x = self._scheduler.add_noise(cond_latent, torch.randn_like(cond_latent), t_start)
                    start_idx = t_index
                else:
                    x = torch.randn_like(cond_latent)
                    start_idx = 0
                for t in self._scheduler.timesteps[start_idx:]:
                    t_tensor = t if torch.is_tensor(t) else torch.tensor(t, device=self._device)

                    # Optimized CFG: run conditional and unconditional branches in one UNet forward pass.
                    latent_in = torch.cat([x, x], dim=0)
                    unet_in = torch.cat([latent_in, cond_in], dim=1)
                    eps_all = self._predict_unet(
                        unet_in,
                        t_tensor,
                        enc_states,
                        added_cond_kwargs=added_cond_kwargs,
                    )
                    eps_all = torch.nan_to_num(eps_all, nan=0.0, posinf=100.0, neginf=-100.0)
                    eps_all = eps_all.clamp(min=-100.0, max=100.0)
                    eps_uncond, eps_cond = eps_all.chunk(2, dim=0)
                    eps = eps_uncond + guidance * (eps_cond - eps_uncond)
                    x = self._scheduler.step(eps, t_tensor, x).prev_sample
                    x = torch.nan_to_num(x, nan=0.0, posinf=100.0, neginf=-100.0)
                    x = x.clamp(min=-100.0, max=100.0)
            # Keep VAE decode out of mixed-precision autocast; SDXL VAE often needs higher precision.
            rgb = self._vae.decode((x / float(self._latent_scale)).to(dtype=vae_dtype)).sample
            rgb = torch.nan_to_num(rgb, nan=0.0, posinf=1.0, neginf=-1.0)
            return (rgb.clamp(-1, 1) + 1.0) * 0.5

    def generate_batch(
        self,
        board_inputs: Sequence[Dict[str, Dict[str, bytes]]],
        *,
        ddim_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        use_img2img_strength: Optional[float] = None,
        include_knot_maps: Optional[bool] = None,
        use_rings_only: Optional[bool] = None,
        boards_per_batch: Optional[int] = None,
    ) -> List[Dict[str, bytes]]:
        self._ensure_loaded()

        try:
            if not board_inputs:
                return []

            image_size = int(self._defaults.image_size)
            steps, guidance, img2img = self._resolve_inference_params(
                ddim_steps=ddim_steps,
                guidance_scale=guidance_scale,
                use_img2img_strength=use_img2img_strength,
            )
            use_knot_maps = self._to_bool(
                include_knot_maps,
                default=bool(self._defaults.include_knot_maps),
            )
            use_rings_only_flag = self._to_bool(
                use_rings_only,
                default=bool(self._defaults.use_rings_only),
            )
            if bool(use_rings_only_flag) and bool(use_knot_maps):
                raise PhotorealisticInferenceError(
                    "rings-only conditioning cannot be combined with include_knot_maps=true."
                )
            if boards_per_batch is None:
                boards_per_batch = 4
            boards_per_batch = max(1, int(boards_per_batch))

            out: List[Dict[str, bytes]] = []
            total = len(board_inputs)
            for start in range(0, total, boards_per_batch):
                chunk = board_inputs[start:start + boards_per_batch]
                cond_rgb_batch = self._build_condition_batch(
                    chunk,
                    image_size=image_size,
                    include_knot_maps=bool(use_knot_maps),
                    use_rings_only=bool(use_rings_only_flag),
                )
                rgb = self._run_diffusion_from_condition_batch(
                    cond_rgb_batch,
                    steps=steps,
                    guidance=guidance,
                    img2img=img2img,
                )

                expected_faces = 4 * len(chunk)
                if int(rgb.shape[0]) != expected_faces:
                    raise PhotorealisticInferenceError(
                        f"Unexpected diffusion output batch size: got {int(rgb.shape[0])}, expected {expected_faces}."
                    )
                for board_idx in range(len(chunk)):
                    base = 4 * board_idx
                    out.append(
                        {
                            "surface_1": self._tensor_to_png_bytes(rgb[base + 0]),
                            "surface_2": self._tensor_to_png_bytes(rgb[base + 1]),
                            "surface_3": self._tensor_to_png_bytes(rgb[base + 2]),
                            "surface_4": self._tensor_to_png_bytes(rgb[base + 3]),
                        }
                    )
            return out
        except PhotorealisticInferenceError:
            raise
        except Exception as exc:
            raise PhotorealisticInferenceError(f"Photorealistic inference failed: {exc}") from exc

    def generate(
        self,
        ring_pngs: Dict[str, bytes],
        fiber_pngs: Optional[Dict[str, bytes]] = None,
        *,
        ddim_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        use_img2img_strength: Optional[float] = None,
        include_knot_maps: Optional[bool] = None,
        use_rings_only: Optional[bool] = None,
    ) -> Dict[str, bytes]:
        outputs = self.generate_batch(
            [{"rings": ring_pngs, "fibers": fiber_pngs or {}}],
            ddim_steps=ddim_steps,
            guidance_scale=guidance_scale,
            use_img2img_strength=use_img2img_strength,
            include_knot_maps=include_knot_maps,
            use_rings_only=use_rings_only,
            boards_per_batch=1,
        )
        if not outputs:
            raise PhotorealisticInferenceError("Photorealistic inference returned no output.")
        return outputs[0]


_RUNTIME = _PhotorealisticRuntime()


def get_photorealistic_capability() -> Dict[str, Any]:
    return _RUNTIME.capability()


def preload_photorealistic_model() -> Dict[str, Any]:
    return _RUNTIME.preload()


def generate_photorealistic_surfaces(
    ring_pngs: Dict[str, bytes],
    fiber_pngs: Optional[Dict[str, bytes]] = None,
    *,
    ddim_steps: Optional[int] = None,
    guidance_scale: Optional[float] = None,
    use_img2img_strength: Optional[float] = None,
    include_knot_maps: Optional[bool] = None,
    use_rings_only: Optional[bool] = None,
) -> Dict[str, bytes]:
    return _RUNTIME.generate(
        ring_pngs,
        fiber_pngs,
        ddim_steps=ddim_steps,
        guidance_scale=guidance_scale,
        use_img2img_strength=use_img2img_strength,
        include_knot_maps=include_knot_maps,
        use_rings_only=use_rings_only,
    )


def generate_photorealistic_surfaces_batch(
    board_inputs: Sequence[Dict[str, Dict[str, bytes]]],
    *,
    ddim_steps: Optional[int] = None,
    guidance_scale: Optional[float] = None,
    use_img2img_strength: Optional[float] = None,
    include_knot_maps: Optional[bool] = None,
    use_rings_only: Optional[bool] = None,
    boards_per_batch: Optional[int] = None,
) -> List[Dict[str, bytes]]:
    return _RUNTIME.generate_batch(
        board_inputs,
        ddim_steps=ddim_steps,
        guidance_scale=guidance_scale,
        use_img2img_strength=use_img2img_strength,
        include_knot_maps=include_knot_maps,
        use_rings_only=use_rings_only,
        boards_per_batch=boards_per_batch,
    )
