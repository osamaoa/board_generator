# Board Generator

Board Generator creates synthetic wood boards from a 3D log model. It combines growth layers, knots, board placement, fiber orientations, and photorealistic face generation in one simulation package.

The package provides:

- a web UI for interactive board and log generation, 3D inspection, MATLAB export, image-map export, and photorealistic face export
- a CLI for batch board exports, knot-sequence model data preparation/training/evaluation, and photorealistic diffusion training

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

`hf auth login` is only needed when your Hugging Face session has not already accepted the SDXL license or when the Hub asks for authentication. The command above downloads the app-required files from the original SDXL repository; a full repository mirror is much larger and is not needed for Board Generator.

Download the Board Generator checkpoint after it has been uploaded:

```bash
curl -L -o photorealistic_checkpoint.zip \
  https://structuralvibration.com/photorealistic_checkpoint.zip
unzip -o photorealistic_checkpoint.zip
```

The zip extracts `photorealistic_model_checkpoint/config.json`, `photorealistic_model_checkpoint/unet.safetensors`, and `photorealistic_model_checkpoint/null_embed.safetensors`. To keep the folders elsewhere, set `PHOTOREALISTIC_SDXL_MODEL_DIR` and `PHOTOREALISTIC_CHECKPOINT_DIR` before starting the backend.

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

## Documentation

- Paper PDF: `docs/paper.pdf` will be added for the release theory reference.
- Getting started: `docs/getting_started.md`
- UI manual: `docs/web_app.md`
- CLI manual: `docs/cli.md`
