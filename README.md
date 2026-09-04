# Board Generator

Board Generator creates synthetic wood boards from a 3D log model. It combines growth layers, knots, board placement, fiber orientations, and photorealistic face generation in one simulation package.

The package provides:

- a web UI for interactive board and log generation, 3D inspection, MATLAB export, image-map export, and photorealistic face export
- a CLI for batch board exports, render-ready Blender export, knot-sequence model data preparation/training/evaluation, and photorealistic diffusion training

## Hugging Face Demo

Public demo: https://osamaabdeljaber-board-generator.hf.space

**Note: This demo is CPU-only and does not include photorealistic face generation.**

## Start

Install the Python dependencies without the photorealistic feature:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
```

Use the photorealistic requirement set instead when CUDA photorealistic generation or diffusion training is needed:

```bash
pip install -r backend/requirements-photorealistic.txt
```

### Photorealistic Model Assets

Photorealistic generation needs two local model folders that are not stored in Git:

- `SDXL_model/`: the original SDXL base model from Hugging Face, `stabilityai/stable-diffusion-xl-base-1.0`
- `photorealistic_model_checkpoint/`: the Board Generator fine-tuned checkpoint

Download the SDXL base model files required by this project:

```bash
pip install -U huggingface_hub
hf auth login
hf download stabilityai/stable-diffusion-xl-base-1.0 \
  model_index.json \
  scheduler/scheduler_config.json \
  unet/config.json \
  unet/diffusion_pytorch_model.safetensors \
  vae/config.json \
  vae/diffusion_pytorch_model.safetensors \
  --local-dir SDXL_model
```

`hf auth login` is only needed when your Hugging Face session has not already accepted the SDXL license or when the Hub asks for authentication. The command above downloads the app-required files from the original SDXL repository; downloading the entire repository is much larger and is not needed for Board Generator.

Download the Board Generator checkpoint from the public Hugging Face model repository:

```bash
hf download OsamaAbdeljaber/photorealistic-wood-board-sdxl \
  config.json \
  unet.safetensors \
  null_embed.safetensors \
  --local-dir photorealistic_model_checkpoint
```

The checkpoint repository is https://huggingface.co/OsamaAbdeljaber/photorealistic-wood-board-sdxl. It contains the Board Generator fine-tuned checkpoint only: `photorealistic_model_checkpoint/config.json`, `photorealistic_model_checkpoint/unet.safetensors`, and `photorealistic_model_checkpoint/null_embed.safetensors`. The checkpoint is derived from SDXL and the repository includes the OpenRAIL++ license and attribution. Download the SDXL base files separately from the original Stability AI repository above. To keep the folders elsewhere, set `PHOTOREALISTIC_SDXL_MODEL_DIR` and `PHOTOREALISTIC_CHECKPOINT_DIR` before starting the backend.

Install the frontend dependencies:

```bash
cd frontend
npm install
```

Run the backend and frontend in separate terminals:

```bash
cd backend
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8100
```

```bash
cd frontend
npm run dev -- --host 0.0.0.0 --port 5175
```

Open the UI at `http://localhost:5175`. The CLI entrypoint is `./board_cli.py`.

Export one generated board to a packed `.blend` scene (Blender must be installed):

```bash
./board_cli.py boards export-blender --data-root /tmp/boards_out --stem 00001
```

The scene maps the four generated long-face images to separate materials and
includes a low three-quarter perspective camera that keeps the unsupported end
cross-sections mostly out of view.

## Documentation

- Paper PDF: [docs/paper.pdf](docs/paper.pdf)
- Getting started: `docs/getting_started.md`
- UI manual: `docs/web_app.md`
- CLI manual: `docs/cli.md`
