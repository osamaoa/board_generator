FROM node:20-bookworm AS frontend-build

WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
ENV VITE_BOARD_GENERATOR_DEMO=1
ENV VITE_API_BASE_URL=
RUN npm run build

FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV BOARD_GENERATOR_DEMO=1
ENV BOARD_GENERATOR_FRONTEND_DIST=/app/frontend_dist
ENV BOARD_GENERATOR_KNOT_MODEL_REPO=OsamaAbdeljaber/board-generator-knot-model
ENV BOARD_GENERATOR_KNOT_MODEL_REVISION=main
ENV BOARD_GENERATOR_KNOT_MODEL_CACHE_DIR=/tmp/board_generator_knot_model

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements-demo.txt ./backend/requirements-demo.txt
RUN python -m pip install --upgrade pip \
    && pip install --no-cache-dir -r ./backend/requirements-demo.txt

COPY backend ./backend
COPY data ./data
COPY visualize_exported_board.m ./visualize_exported_board.m
COPY --from=frontend-build /app/frontend/dist ./frontend_dist

EXPOSE 7860

CMD ["sh", "-c", "echo '[demo] starting uvicorn on port 7860' && exec python -m uvicorn app.main:app --app-dir /app/backend --host 0.0.0.0 --port 7860 --proxy-headers"]
