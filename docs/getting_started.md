# Getting Started

## Requirements

- Python 3.10 or newer
- Node.js 18 or newer
- `npm`

MATLAB is optional and is only needed to run the exported MATLAB visualization script. Photorealistic generation requires a CUDA-capable GPU. Without CUDA, the UI and CLI keep the non-photorealistic simulation and export features available.

## Install

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
cd frontend
npm install
```

Use the photorealistic requirement set instead when CUDA photorealistic generation or diffusion training is needed. It includes the base backend requirements:

```bash
pip install -r backend/requirements-photorealistic.txt
```

## Install Photorealistic Models

Photorealistic generation is disabled until both model asset folders are present:

- `SDXL_model/`: the original SDXL base model from Hugging Face, `stabilityai/stable-diffusion-xl-base-1.0`
- `photorealistic_model_checkpoint/`: the Board Generator fine-tuned checkpoint

The backend looks for these folders in the repository root by default. You can keep them elsewhere by setting:

```bash
export PHOTOREALISTIC_SDXL_MODEL_DIR=/absolute/path/to/SDXL_model
export PHOTOREALISTIC_CHECKPOINT_DIR=/absolute/path/to/photorealistic_model_checkpoint
```

### SDXL Base Model

Install the Hugging Face CLI and download the SDXL base files used by this project:

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

`hf auth login` is only required when your machine is not already authenticated for the SDXL model page. The files still come from the original Hugging Face model repository; the command lists only the parts Board Generator loads locally. If you want a full local mirror of the Hugging Face repository, omit the file list, but expect a much larger download containing assets this app does not use.

The expected minimum layout is:

```text
SDXL_model/
  model_index.json
  scheduler/scheduler_config.json
  unet/config.json
  unet/diffusion_pytorch_model.safetensors
  vae/config.json
  vae/diffusion_pytorch_model.safetensors
```

### Fine-Tuned Photorealistic Checkpoint

After the checkpoint has been uploaded, download it from:

```text
https://structuralvibration.com/photorealistic_checkpoint.zip
```

From the repository root:

```bash
curl -L -o photorealistic_checkpoint.zip \
  https://structuralvibration.com/photorealistic_checkpoint.zip
unzip -o photorealistic_checkpoint.zip
```

The archive extracts the runtime checkpoint folder:

```text
photorealistic_model_checkpoint/
  config.json
  unet.safetensors
  null_embed.safetensors
```

Do not put the SDXL base model files into `photorealistic_model_checkpoint/`. The base model and fine-tuned checkpoint are loaded separately.

## Run The UI

Start the API:

```bash
cd backend
source ../.venv/bin/activate
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8100
```

Start the frontend in another terminal:

```bash
cd frontend
npm run dev -- --host 0.0.0.0 --port 5175
```

Open `http://localhost:5175`.

## Check The CLI

```bash
./board_cli.py --help
./board_cli.py boards --help
./board_cli.py knots --help
./board_cli.py diffusion --help
```

For a small batch-generation smoke test:

```bash
./board_cli.py boards generate \
  --config-json path/to/boards_config.json \
  --output-dir /tmp/board_generator_smoke \
  --num-boards 2 \
  --outputs rings
```
