# Board Generator

Board Generator creates synthetic wood boards from a 3D log model. It combines growth layers, knots, board placement, fiber orientations, and photorealistic face generation in one simulation package.

The package provides:

- a web UI for interactive board and log generation, 3D inspection, MATLAB export, image-map export, and photorealistic face export
- a CLI for batch board exports, knot-sequence model data preparation/training/evaluation, and photorealistic diffusion training

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
- UI manual: `docs/web_app.md`
- CLI manual: `docs/cli.md`

Setup notes and optional dependency notes are in `docs/getting_started.md`.
