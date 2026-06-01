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
  --config-json docs/boards_generate_full_config.example.json \
  --output-dir /tmp/board_generator_smoke \
  --num-boards 2 \
  --outputs rings
```
