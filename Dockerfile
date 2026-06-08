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

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements-demo.txt ./backend/requirements-demo.txt
RUN python -m pip install --upgrade pip \
    && pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch \
    && pip install --no-cache-dir -r ./backend/requirements-demo.txt

COPY backend ./backend
COPY data ./data
COPY visualize_exported_board.m ./visualize_exported_board.m
COPY knot_model_checkpoint/knot_sequence_model.pt ./knot_model_checkpoint/knot_sequence_model.pt
COPY knot_model_checkpoint/knot_sequence_model.json ./knot_model_checkpoint/knot_sequence_model.json
COPY knot_model_checkpoint/training_data_new_2025.mat ./knot_model_checkpoint/training_data_new_2025.mat
COPY --from=frontend-build /app/frontend/dist ./frontend_dist

EXPOSE 7860

CMD ["python", "-m", "uvicorn", "app.main:app", "--app-dir", "/app/backend", "--host", "0.0.0.0", "--port", "7860", "--proxy-headers"]
