# Diffusion Model Theory and Implementation

## 1. Scope

This document explains the photorealistic module used for ring+fiber to RGB surface generation.
It maps equations and modeling choices to:

- `backend/app/core/photorealistic_training.py`
- `backend/app/core/photorealistic_inference.py`

## 2. Data Representation

Each board has 4 side faces.
For each face `f in {1,2,3,4}`:

- conditioning images: ring map + fiber map (grayscale)
- target image: RGB photograph-like surface

Training dataset folders under `data_root`:

- `ring_pred_new_1..4`
- `fiber_1..4`
- `color_1..4`

Files are matched by stem across all folders.

## 3. Latent Diffusion Setup

The system uses SD2 components (VAE + UNet + scheduler).

## 3.1 Conditioning Construction

Per face, the 2-channel conditioning is converted to pseudo-RGB:

\[
I_{cond} = [ring, fiber, 0]
\]

Then VAE encoding gives condition latent `z_cond`.
Target RGB is encoded to latent `z_0`.

## 3.2 UNet Input Channels

Base SD2 UNet expects 4 latent channels.
This project expands `conv_in` to 8 channels and concatenates:

\[
[ z_t, z_{cond} ]
\]

Initialization keeps original 4-channel SD2 weights and zero-inits added channels.

## 3.3 Prompt-Free Context

A learned null-context token (`77 x d`) replaces text prompts.
It is optimized jointly with UNet.

## 4. Forward and Reverse Processes

Noising (training) uses DDPM-style process:

\[
z_t = \sqrt{\bar\alpha_t} z_0 + \sqrt{1-\bar\alpha_t}\,\epsilon
\]

UNet predicts either:

- `epsilon` (`prediction_type=epsilon`)
- velocity `v` (`prediction_type=v_prediction`)

Sampling (validation/inference) uses DDIM scheduler configured to match prediction type.

## 5. Training Objective

Total loss:

\[
L = L_{noise} + \lambda_{recon} L_{recon} + \lambda_{cross} L_{cross} + \lambda_{seam} L_{seam}
\]

## 5.1 Noise Loss with Min-SNR Reweighting

Base loss is MSE between model output and target (`epsilon` or `v`).
If `min_snr_gamma > 0`, per-timestep weights are applied using clipped SNR.

## 5.2 Latent Reconstruction Loss

`x0_pred` is reconstructed from `(z_t, model_output, t)` and compared to `z_0` with SmoothL1:

\[
L_{recon} = SmoothL1(x0_{pred}, z_0)
\]

Weight: `latent_recon_weight`.

## 5.3 Cross-Surface Consistency Loss

For each board, per-face latent means/stds are computed, then pairwise face-difference tensors are matched between prediction and target.

This promotes board-level coherence across the 4 faces.
Weight: `cross_surface_consistency_weight`.

## 5.4 Seam Consistency Loss

Adjacent face pairs are compared on edge strips in latent space.
The loss uses the minimum across orientation variants (normal/reversed alignment).

Pairs used:

- `(surface_1 left) <-> (surface_3 right)`
- `(surface_1 right) <-> (surface_4 left)`
- `(surface_2 right) <-> (surface_3 left)`
- `(surface_2 left) <-> (surface_4 right)`

Edge-strip width is controlled by:

- `seam_strip_width` in latent pixels (not RGB pixels)

Weight: `seam_consistency_weight`.

## 6. Shared-Board Noise Option

If `shared_board_noise=true`, the same diffusion noise sample is shared across the 4 faces of each board (per step).
This can stabilize inter-face coherence.

## 7. Validation and Export During Training

- validation uses DDIM sampling on held-out boards
- EMA and raw checkpoints are exported periodically
- exported inference bundle includes:
  - `unet.safetensors`
  - `null_embed.safetensors`
  - `config.json` (steps, guidance, img2img defaults, prediction type)

## 8. Inference Runtime Model

Inference runtime (`photorealistic_inference.py`) uses the CUDA GPU path with PyTorch fp16. If CUDA is unavailable, the photorealistic capability is disabled.

Capability endpoint reports availability, active device, and recommended DDIM steps.

## 9. Inference Parameters

Main controls:

- `ddim_steps`
- `guidance_scale`
- `use_img2img_strength` in `[0,1]`
- `boards_per_batch`

Classifier-free guidance is applied with standard formula:

\[
\hat\epsilon = \epsilon_{uncond} + s(\epsilon_{cond} - \epsilon_{uncond})
\]

where `s = guidance_scale`.

## 10. Practical Notes

- Keep training and inference prediction type aligned (`epsilon` vs `v_prediction`).
- Seam/cross-surface losses are optional; start with small weights.
- On constrained GPUs, disable pin-memory and reduce boards-per-batch.
