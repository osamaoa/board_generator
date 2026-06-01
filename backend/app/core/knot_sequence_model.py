from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from html import escape
import json
import math
from pathlib import Path
import threading
from typing import Any, Dict, Iterable, Optional

import numpy as np
import scipy.io
from scipy.cluster.vq import kmeans2


def _repo_root() -> Path:
    # backend/app/core -> backend/app -> backend -> repo root
    return Path(__file__).resolve().parents[3]


def default_training_mat_path() -> Path:
    return _repo_root() / "knot_model_checkpoint" / "training_data_new_2025.mat"


def default_checkpoint_path() -> Path:
    return _repo_root() / "knot_model_checkpoint" / "knot_sequence_model.pt"


def _resolve_path(path_value: str, fallback: Path) -> Path:
    text = str(path_value or "").strip()
    if text:
        return Path(text).expanduser().resolve()
    return fallback.resolve()


def _normalize_range_per_column(data: np.ndarray) -> np.ndarray:
    mins = np.min(data, axis=0, keepdims=True)
    maxs = np.max(data, axis=0, keepdims=True)
    span = maxs - mins
    span[span <= 1e-12] = 1.0
    return (data - mins) / span


def _normalize_knots_alt(data: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mins = np.min(data[:, 0:3], axis=0)
    mins = np.concatenate(
        [
            mins,
            np.full(10, float(np.min(data[:, 3:13]))),
            np.full(10, float(np.min(data[:, 13:23]))),
        ]
    )
    maxs = np.max(data[:, 0:3], axis=0)
    maxs = np.concatenate(
        [
            maxs,
            np.full(10, float(np.max(data[:, 3:13]))),
            np.full(10, float(np.max(data[:, 13:23]))),
        ]
    )
    span = maxs - mins
    span[span <= 1e-12] = 1.0
    scaled = (data - mins.reshape(1, -1)) / span.reshape(1, -1)
    return mins.reshape(1, -1), maxs.reshape(1, -1), scaled


def _iter_knot_data_files(logs_dir: Path) -> Iterable[Path]:
    return sorted(p for p in logs_dir.glob("knot_data_*.mat") if p.is_file())


@dataclass
class KnotDatasetPrepConfig:
    logs_dir: str = str((_repo_root() / ".old_knot_generator" / "logs_data").resolve())
    output_mat_path: str = str(default_training_mat_path())
    dz: float = 10.0
    n_samples: int = 101
    n_overlap: int = 97
    cluster_count: int = 512
    cluster_seed: int = 42
    cluster_max_iter: int = 40


def _segment_with_overlap(signal: np.ndarray, n_samples: int, n_overlap: int) -> np.ndarray:
    if n_samples <= 1:
        raise RuntimeError("n_samples must be >= 2.")
    if n_overlap < 0 or n_overlap >= n_samples:
        raise RuntimeError("n_overlap must satisfy 0 <= n_overlap < n_samples.")

    step = n_samples - n_overlap
    out: list[np.ndarray] = []
    last_start = int(signal.shape[0]) - n_samples
    if last_start < 0:
        return np.empty((0, n_samples), dtype=signal.dtype)

    for start in range(0, last_start + 1, step):
        out.append(signal[start : start + n_samples])

    return np.stack(out, axis=0)


def prepare_knot_training_data(cfg: KnotDatasetPrepConfig) -> Dict[str, Any]:
    logs_dir = Path(cfg.logs_dir).expanduser().resolve()
    if not logs_dir.is_dir():
        raise RuntimeError(f"logs_dir does not exist: {logs_dir}")

    output_path = Path(cfg.output_mat_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dz = float(cfg.dz)
    n_samples = int(cfg.n_samples)
    n_overlap = int(cfg.n_overlap)

    eval_locs_l = np.arange(0.1, 1.01, 0.1, dtype=np.float64)
    eval_locs_z = np.arange(0.1, 1.01, 0.1, dtype=np.float64)

    knot_or_not_all: list[np.ndarray] = []
    data_all_rows: list[np.ndarray] = []
    data_all_or_rows: list[np.ndarray] = []
    knot_id = 1

    files = list(_iter_knot_data_files(logs_dir))
    if not files:
        raise RuntimeError(f"No knot_data_*.mat files found under: {logs_dir}")

    for mat_file in files:
        payload = scipy.io.loadmat(mat_file)
        alfa = np.asarray(payload["Knot_param__Alfa_L100_Rlivelim_Rtot_Abump_Aexp"], dtype=np.float64)
        vert = np.asarray(payload["Knot_param__Vertdiamparam"], dtype=np.float64)
        zc = np.asarray(payload["Knot_param__Zcparam"], dtype=np.float64)

        if alfa.ndim != 2 or vert.ndim != 2 or zc.ndim != 2:
            raise RuntimeError(f"Invalid knot matrices in {mat_file}")
        if alfa.shape[0] != vert.shape[0] or alfa.shape[0] != zc.shape[0]:
            raise RuntimeError(f"Row-count mismatch in {mat_file}")
        if alfa.shape[1] < 4 or vert.shape[1] < 4 or zc.shape[1] < 3:
            raise RuntimeError(f"Unexpected knot matrix column count in {mat_file}")

        data = np.concatenate([alfa[:, [0, 2, 3]], vert[:, 0:4], zc[:, 0:2]], axis=1)
        z0 = zc[:, 2]

        z0_d = np.round(z0 / dz) * dz
        for i in range(1, z0_d.shape[0]):
            if z0_d[i - 1] == z0_d[i]:
                z0_d[i] = z0_d[i] + dz

        valid = z0_d >= 0.0
        data = data[valid]
        z0_d = z0_d[valid]
        data_or = np.asarray(data, dtype=np.float64).copy()

        data_alt = np.zeros((data.shape[0], 23), dtype=np.float64)
        if data.shape[0] > 0:
            data_alt[:, 0:3] = data[:, 0:3]
            for row_idx in range(data.shape[0]):
                rt = float(data[row_idx, 2])
                poly_l = np.array([data[row_idx, 3], data[row_idx, 4], data[row_idx, 5], data[row_idx, 6], 0.0], dtype=np.float64)
                poly_z = np.array([data[row_idx, 7], data[row_idx, 8], 0.0], dtype=np.float64)
                data_alt[row_idx, 3:13] = np.polyval(poly_l, rt * eval_locs_l)
                data_alt[row_idx, 13:23] = np.polyval(poly_z, rt * eval_locs_z)

        n_slots = int((np.max(z0_d) / dz)) if z0_d.size > 0 else 0
        knot_or_not = np.zeros(n_slots, dtype=np.uint16)
        for slot_i in range(n_slots):
            if np.any(z0_d == (dz * slot_i)):
                knot_or_not[slot_i] = np.uint16(knot_id)
                knot_id += 1

        knot_or_not_all.append(knot_or_not)
        data_all_rows.append(data_alt)
        data_all_or_rows.append(data_or)

    if not data_all_rows or not data_all_or_rows:
        raise RuntimeError("No valid knot rows were collected from logs_dir.")

    data_all = np.concatenate(data_all_rows, axis=0)
    data_all_or = np.concatenate(data_all_or_rows, axis=0)

    for col in range(3, data_all.shape[1]):
        p95 = float(np.percentile(data_all[:, col], 95))
        data_all[:, col] = np.minimum(data_all[:, col], p95)
    for col in range(13, data_all.shape[1]):
        p05 = float(np.percentile(data_all[:, col], 5))
        data_all[:, col] = np.maximum(data_all[:, col], p05)
    data_all[:, 3:13] = np.maximum(data_all[:, 3:13], 0.0)

    mins, maxs, data_all_scaled = _normalize_knots_alt(data_all)
    data_all_scaled = np.vstack([np.zeros((1, data_all_scaled.shape[1]), dtype=np.float64), data_all_scaled])
    data_all = np.vstack([np.zeros((1, data_all.shape[1]), dtype=np.float64), data_all])

    data_all_or_scaled = _normalize_range_per_column(data_all_or)
    data_all_or = np.vstack([np.zeros((1, data_all_or.shape[1]), dtype=np.float64), data_all_or])
    data_all_or_scaled = np.vstack([np.zeros((1, data_all_or_scaled.shape[1]), dtype=np.float64), data_all_or_scaled])

    n_dictionary_rows = int(data_all_or.shape[0] - 1)
    requested_cluster_count = max(0, int(cfg.cluster_count))
    cluster_enabled = requested_cluster_count > 0 and n_dictionary_rows > 0
    tokenization_mode = 0  # 0=direct, 1=clustered

    rowid_to_token = np.zeros((n_dictionary_rows + 1,), dtype=np.int64)
    token_to_rowid_ptr: np.ndarray
    token_to_rowid_flat: np.ndarray
    token_embedding = np.asarray(data_all_or_scaled, dtype=np.float64)

    if not cluster_enabled:
        rowid_to_token[1:] = np.arange(1, n_dictionary_rows + 1, dtype=np.int64)
        token_to_rowid_ptr = np.zeros((n_dictionary_rows + 1, 2), dtype=np.int64)
        token_to_rowid_ptr[1:, 0] = np.arange(0, n_dictionary_rows, dtype=np.int64)
        token_to_rowid_ptr[1:, 1] = 1
        token_to_rowid_flat = np.arange(1, n_dictionary_rows + 1, dtype=np.int64)
        token_count = int(n_dictionary_rows)
    else:
        features = np.asarray(data_all_or_scaled[1:, :], dtype=np.float64)
        cluster_k = min(max(1, int(requested_cluster_count)), int(features.shape[0]))
        centroids, labels = kmeans2(
            features,
            cluster_k,
            iter=max(1, int(cfg.cluster_max_iter)),
            minit="points",
            seed=int(cfg.cluster_seed),
        )
        labels = np.asarray(labels, dtype=np.int64).reshape(-1)
        if labels.shape[0] != features.shape[0]:
            raise RuntimeError("Unexpected kmeans label shape while clustering knot dictionary.")

        used = np.unique(labels)
        remap_lut = -np.ones((cluster_k,), dtype=np.int64)
        remap_lut[used] = np.arange(used.shape[0], dtype=np.int64)
        labels = remap_lut[labels]

        token_count = int(used.shape[0])
        rowid_to_token[1:] = labels + 1
        token_embedding = np.zeros((token_count + 1, features.shape[1]), dtype=np.float64)
        token_embedding[1:, :] = np.asarray(centroids[used], dtype=np.float64)
        token_embedding[1:, :] = np.clip(token_embedding[1:, :], 0.0, 1.0)
        tokenization_mode = 1

        token_to_rowid_ptr = np.zeros((token_count + 1, 2), dtype=np.int64)
        flat_members: list[int] = []
        for token_id in range(1, token_count + 1):
            members = np.where(rowid_to_token[1:] == token_id)[0] + 1
            start = len(flat_members)
            length = int(members.shape[0])
            token_to_rowid_ptr[token_id, 0] = int(start)
            token_to_rowid_ptr[token_id, 1] = int(length)
            flat_members.extend(int(v) for v in members.tolist())
        token_to_rowid_flat = np.asarray(flat_members, dtype=np.int64)

    all_segments: list[np.ndarray] = []
    for seq in knot_or_not_all:
        segments = _segment_with_overlap(seq, n_samples=n_samples, n_overlap=n_overlap)
        if segments.shape[0] > 0:
            all_segments.append(segments)
    if not all_segments:
        raise RuntimeError(
            "No full training segments were generated. "
            "Decrease --n-samples or provide longer knot log sequences."
        )
    all_sigs = np.asarray(np.concatenate(all_segments, axis=0), dtype=np.int64)
    token_sigs = np.asarray(all_sigs, dtype=np.int64)
    mask_nonzero = token_sigs > 0
    if np.any(mask_nonzero):
        row_ids = token_sigs[mask_nonzero]
        valid_row_ids = (row_ids >= 0) & (row_ids < rowid_to_token.shape[0])
        mapped = np.zeros_like(row_ids)
        mapped[valid_row_ids] = rowid_to_token[row_ids[valid_row_ids]]
        token_sigs[mask_nonzero] = mapped

    inps = np.asarray(token_sigs[:, :-1], dtype=np.uint16)
    outs = np.asarray(token_sigs[:, 1:], dtype=np.uint16)

    scipy.io.savemat(
        output_path,
        {
            "inps": inps,
            "outs": outs,
            "Data_all_scaled": data_all_scaled,
            "Data_all": data_all,
            "Data_all_or_scaled": token_embedding,
            "Data_all_or_scaled_full": data_all_or_scaled,
            "Data_all_or": data_all_or,
            "dz": np.asarray([[dz]], dtype=np.float64),
            "mins": mins,
            "maxs": maxs,
            "eval_locs_Z": eval_locs_z.reshape(1, -1),
            "eval_locs_L": eval_locs_l.reshape(1, -1),
            "tokenization_mode": np.asarray([[int(tokenization_mode)]], dtype=np.int64),
            "token_cluster_count": np.asarray([[int(token_count)]], dtype=np.int64),
            "rowid_to_token": rowid_to_token.reshape(1, -1),
            "token_to_rowid_ptr": token_to_rowid_ptr,
            "token_to_rowid_flat": token_to_rowid_flat.reshape(1, -1),
            "cluster_count_requested": np.asarray([[int(requested_cluster_count)]], dtype=np.int64),
            "cluster_seed": np.asarray([[int(cfg.cluster_seed)]], dtype=np.int64),
            "cluster_max_iter": np.asarray([[int(cfg.cluster_max_iter)]], dtype=np.int64),
        },
    )

    return {
        "output_mat_path": str(output_path),
        "num_logs": int(len(files)),
        "num_dictionary_entries_without_no_knot": int(data_all_or.shape[0] - 1),
        "vocab_size_with_no_knot": int(token_count + 1),
        "tokenization_mode": ("clustered" if tokenization_mode == 1 else "direct"),
        "cluster_count_requested": int(requested_cluster_count),
        "cluster_count_effective": int(token_count),
        "num_sequences": int(inps.shape[0]),
        "sequence_length": int(inps.shape[1]),
        "dz_mm": float(dz),
    }


def _import_torch():
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
    except Exception as exc:
        raise RuntimeError(
            "PyTorch is required for knot-sequence training/inference. "
            "Install torch in this environment."
        ) from exc
    return torch, nn, DataLoader, TensorDataset


class _KnotSequenceLSTM:
    def __init__(
        self,
        *,
        torch: Any,
        nn: Any,
        vocab_size: int,
        embedding_dim: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        embedding_weights: np.ndarray,
        freeze_embedding: bool,
    ):
        class _Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = nn.Embedding(vocab_size, embedding_dim)
                self.embedding.weight.data.copy_(torch.as_tensor(embedding_weights, dtype=torch.float32))
                self.embedding.weight.requires_grad = not bool(freeze_embedding)
                self.lstm = nn.LSTM(
                    embedding_dim,
                    hidden_size,
                    num_layers=num_layers,
                    batch_first=True,
                    dropout=(dropout if num_layers > 1 else 0.0),
                )
                self.head = nn.Linear(hidden_size, vocab_size)

            def forward(self, tokens, hidden=None):
                x = self.embedding(tokens)
                y, hidden_out = self.lstm(x, hidden)
                logits = self.head(y)
                return logits, hidden_out

            def step(self, token, hidden=None):
                # token shape: (batch,)
                x = self.embedding(token.unsqueeze(1))
                y, hidden_out = self.lstm(x, hidden)
                logits = self.head(y[:, -1, :])
                return logits, hidden_out

        self.model = _Model()


@dataclass
class KnotSequenceTrainConfig:
    training_mat_path: str = str(default_training_mat_path())
    output_checkpoint_path: str = str(default_checkpoint_path())
    output_history_path: str = ""
    hidden_size: int = 128
    num_layers: int = 1
    dropout: float = 0.0
    batch_size: int = 64
    epochs: int = 60
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    val_ratio: float = 0.1
    early_stop_enabled: bool = True
    early_stop_patience: int = 8
    early_stop_min_delta: float = 0.0
    early_stop_monitor: str = "val_loss"  # val_loss|train_loss|val_acc|train_acc
    seed: int = 42
    freeze_embedding: bool = True
    no_knot_weight: float = 0.35
    device: str = "auto"  # auto|cpu|cuda


def train_knot_sequence_model(cfg: KnotSequenceTrainConfig) -> Dict[str, Any]:
    torch, nn, DataLoader, TensorDataset = _import_torch()

    training_path = _resolve_path(cfg.training_mat_path, default_training_mat_path())
    if not training_path.is_file():
        raise RuntimeError(f"Training MAT file not found: {training_path}")

    payload = scipy.io.loadmat(training_path)
    inps = np.asarray(payload["inps"], dtype=np.int64)
    outs = np.asarray(payload["outs"], dtype=np.int64)
    if inps.shape != outs.shape or inps.ndim != 2:
        raise RuntimeError(f"Invalid inps/outs shapes in {training_path}: {inps.shape} vs {outs.shape}")

    if "Data_all_or_scaled" in payload:
        embedding = np.asarray(payload["Data_all_or_scaled"], dtype=np.float32)
    elif "Data_all_scaled" in payload:
        embedding = np.asarray(payload["Data_all_scaled"], dtype=np.float32)
    else:
        raise RuntimeError(
            f"{training_path} must contain Data_all_or_scaled or Data_all_scaled for embedding initialization."
        )
    if embedding.ndim != 2:
        raise RuntimeError(f"Invalid embedding matrix shape: {embedding.shape}")

    vocab_tokens = int(max(int(np.max(inps)), int(np.max(outs)))) + 1
    if embedding.shape[0] < vocab_tokens:
        padded = np.zeros((vocab_tokens, embedding.shape[1]), dtype=np.float32)
        padded[: embedding.shape[0], :] = embedding
        embedding = padded

    vocab_size = int(embedding.shape[0])
    embedding_dim = int(embedding.shape[1])
    if vocab_size <= 1 or embedding_dim <= 0:
        raise RuntimeError("Invalid vocabulary/embedding dimensions.")

    seed = int(cfg.seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    requested_device = str(cfg.device).strip().lower()
    if requested_device == "cpu":
        device = torch.device("cpu")
    else:
        cuda_available = bool(torch.cuda.is_available())
        if requested_device == "cuda" and not cuda_available:
            raise RuntimeError("Requested --device cuda but CUDA is unavailable.")
        device = torch.device("cuda" if cuda_available else "cpu")

    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    n_rows = int(inps.shape[0])
    if n_rows < 1:
        raise RuntimeError("Training dataset has zero rows.")

    permutation = np.random.permutation(n_rows)
    val_ratio = float(min(max(cfg.val_ratio, 0.0), 0.5))
    if n_rows == 1:
        val_count = 0
    else:
        val_count = int(round(n_rows * val_ratio))
        val_count = max(1, min(n_rows - 1, val_count))
    train_idx = permutation[val_count:]
    val_idx = permutation[:val_count]

    if train_idx.size == 0:
        train_idx = permutation
        val_idx = np.empty((0,), dtype=np.int64)

    x_train = torch.from_numpy(inps[train_idx]).long()
    y_train = torch.from_numpy(outs[train_idx]).long()
    train_loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=max(1, int(cfg.batch_size)),
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    val_loader = None
    if val_idx.size > 0:
        x_val = torch.from_numpy(inps[val_idx]).long()
        y_val = torch.from_numpy(outs[val_idx]).long()
        val_loader = DataLoader(
            TensorDataset(x_val, y_val),
            batch_size=max(1, int(cfg.batch_size)),
            shuffle=False,
            num_workers=0,
            pin_memory=(device.type == "cuda"),
        )

    model_wrap = _KnotSequenceLSTM(
        torch=torch,
        nn=nn,
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_size=max(4, int(cfg.hidden_size)),
        num_layers=max(1, int(cfg.num_layers)),
        dropout=max(0.0, float(cfg.dropout)),
        embedding_weights=embedding,
        freeze_embedding=bool(cfg.freeze_embedding),
    )
    model = model_wrap.model.to(device)

    class_weights = torch.ones(vocab_size, dtype=torch.float32, device=device)
    class_weights[0] = float(max(1e-4, cfg.no_knot_weight))
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.learning_rate),
        weight_decay=float(max(0.0, cfg.weight_decay)),
    )

    requested_monitor = str(cfg.early_stop_monitor).strip().lower()
    supported_monitors = {"val_loss", "train_loss", "val_acc", "train_acc"}
    if requested_monitor not in supported_monitors:
        raise RuntimeError(
            f"Unsupported early_stop_monitor={cfg.early_stop_monitor!r}. "
            f"Supported: {sorted(supported_monitors)}"
        )

    resolved_monitor = requested_monitor
    if val_loader is None and requested_monitor == "val_loss":
        resolved_monitor = "train_loss"
        print(
            "[knots/train] no validation split available; "
            "early-stop monitor changed from val_loss to train_loss."
        )
    elif val_loader is None and requested_monitor == "val_acc":
        resolved_monitor = "train_acc"
        print(
            "[knots/train] no validation split available; "
            "early-stop monitor changed from val_acc to train_acc."
        )

    monitor_is_accuracy = resolved_monitor.endswith("_acc")
    best_metric = (-math.inf if monitor_is_accuracy else math.inf)
    patience_metric = (-math.inf if monitor_is_accuracy else math.inf)
    best_epoch = -1
    best_state: Dict[str, Any] = {}
    patience = int(cfg.early_stop_patience)
    min_delta = float(max(0.0, cfg.early_stop_min_delta))
    early_stop_enabled = bool(cfg.early_stop_enabled) and patience > 0
    stale_epochs = 0
    train_history: list[float] = []
    val_history: list[Optional[float]] = []
    train_acc_history: list[float] = []
    val_acc_history: list[Optional[float]] = []
    if early_stop_enabled:
        print(
            "[knots/train] early_stop enabled "
            f"monitor={resolved_monitor} patience={patience} min_delta={min_delta:.6g}"
        )
    else:
        print(
            "[knots/train] early_stop disabled "
            f"(enabled={bool(cfg.early_stop_enabled)}, patience={patience})"
        )

    def evaluate(loader) -> tuple[float, float]:
        model.eval()
        total = 0.0
        steps = 0
        correct = 0
        count = 0
        with torch.no_grad():
            for x_batch, y_batch in loader:
                x_batch = x_batch.to(device, non_blocking=True)
                y_batch = y_batch.to(device, non_blocking=True)
                logits, _ = model(x_batch)
                loss = criterion(logits.reshape(-1, vocab_size), y_batch.reshape(-1))
                total += float(loss.item())
                steps += 1
                preds = torch.argmax(logits, dim=-1)
                correct += int((preds == y_batch).sum().item())
                count += int(y_batch.numel())
        if steps == 0:
            return math.inf, 0.0
        return total / float(steps), (float(correct) / float(max(1, count)))

    epochs = max(1, int(cfg.epochs))
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        train_steps = 0
        train_correct = 0
        train_count = 0
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(x_batch)
            loss = criterion(logits.reshape(-1, vocab_size), y_batch.reshape(-1))
            loss.backward()
            if float(cfg.grad_clip) > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(cfg.grad_clip))
            optimizer.step()

            epoch_loss += float(loss.item())
            train_steps += 1
            preds = torch.argmax(logits, dim=-1)
            train_correct += int((preds == y_batch).sum().item())
            train_count += int(y_batch.numel())

        avg_train = (epoch_loss / float(train_steps)) if train_steps > 0 else math.inf
        train_acc = float(train_correct) / float(max(1, train_count))
        train_history.append(avg_train)
        train_acc_history.append(train_acc)

        avg_val: Optional[float]
        val_acc: Optional[float]
        if val_loader is not None:
            avg_val, val_acc = evaluate(val_loader)
        else:
            avg_val = None
            val_acc = None
        val_history.append(avg_val)
        val_acc_history.append(val_acc)

        monitor_value: float
        if resolved_monitor == "val_loss":
            monitor_value = float(avg_val if avg_val is not None else avg_train)
        elif resolved_monitor == "train_loss":
            monitor_value = float(avg_train)
        elif resolved_monitor == "val_acc":
            monitor_value = float(val_acc if val_acc is not None else train_acc)
        elif resolved_monitor == "train_acc":
            monitor_value = float(train_acc)
        else:
            raise RuntimeError(f"Unexpected early-stop monitor: {resolved_monitor}")

        print(
            f"[knots/train] epoch {epoch}/{epochs} "
            f"train_loss={avg_train:.6f} "
            f"train_acc={train_acc:.4f} "
            f"val_loss={(f'{avg_val:.6f}' if avg_val is not None else 'n/a')} "
            f"val_acc={(f'{val_acc:.4f}' if val_acc is not None else 'n/a')}"
        )

        if monitor_is_accuracy:
            improved_for_best = monitor_value > (best_metric + 1e-12)
            improved_for_patience = monitor_value > (patience_metric + min_delta)
        else:
            improved_for_best = monitor_value < (best_metric - 1e-12)
            improved_for_patience = monitor_value < (patience_metric - min_delta)

        if improved_for_best:
            best_metric = monitor_value
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if improved_for_patience:
            patience_metric = monitor_value
            stale_epochs = 0
        else:
            stale_epochs += 1
            if early_stop_enabled and stale_epochs >= patience:
                print(
                    f"[knots/train] early stop at epoch {epoch} "
                    f"(best_epoch={best_epoch}, best_metric={best_metric:.6f}, monitor={resolved_monitor})"
                )
                break

    if not best_state:
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        best_epoch = epochs
        best_metric = float(train_history[-1]) if train_history else math.inf

    model.load_state_dict(best_state, strict=True)

    checkpoint_path = Path(cfg.output_checkpoint_path).expanduser().resolve()
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    history_path = (
        Path(cfg.output_history_path).expanduser().resolve()
        if str(cfg.output_history_path or "").strip()
        else checkpoint_path.with_suffix(".json")
    )
    history_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "format_version": 1,
        "model_state": best_state,
        "model_hparams": {
            "hidden_size": int(max(4, cfg.hidden_size)),
            "num_layers": int(max(1, cfg.num_layers)),
            "dropout": float(max(0.0, cfg.dropout)),
            "freeze_embedding": bool(cfg.freeze_embedding),
        },
        "vocab_size": int(vocab_size),
        "embedding_dim": int(embedding_dim),
        "embedding_weights": model.embedding.weight.detach().cpu(),
        "training_mat_path": str(training_path),
        "dz_mm": float(np.asarray(payload.get("dz", [[10.0]])).reshape(-1)[0]),
        "best_epoch": int(best_epoch),
        "best_metric": float(best_metric),
        "best_metric_name": str(resolved_monitor),
        "train_config": asdict(cfg),
    }
    torch.save(checkpoint, checkpoint_path)

    history_payload = {
        "checkpoint_path": str(checkpoint_path),
        "training_mat_path": str(training_path),
        "train_loss": train_history,
        "val_loss": val_history,
        "train_acc": train_acc_history,
        "val_acc": val_acc_history,
        "best_epoch": int(best_epoch),
        "best_metric": float(best_metric),
        "best_metric_name": str(resolved_monitor),
        "device": str(device),
        "config": asdict(cfg),
    }
    history_path.write_text(json.dumps(history_payload, indent=2), encoding="utf-8")

    summary = {
        "checkpoint_path": str(checkpoint_path),
        "history_path": str(history_path),
        "training_rows": int(train_idx.size),
        "validation_rows": int(val_idx.size),
        "vocab_size": int(vocab_size),
        "embedding_dim": int(embedding_dim),
        "best_epoch": int(best_epoch),
        "best_metric": float(best_metric),
        "best_metric_name": str(resolved_monitor),
        "train_acc_last": float(train_acc_history[-1]) if train_acc_history else 0.0,
        "val_acc_at_best_epoch": (
            float(val_acc_history[best_epoch - 1])
            if (best_epoch >= 1 and best_epoch <= len(val_acc_history) and val_acc_history[best_epoch - 1] is not None)
            else None
        ),
        "device": str(device),
    }
    print(
        "[knots/train] saved checkpoint="
        f"{summary['checkpoint_path']} best_epoch={summary['best_epoch']} "
        f"best_metric={summary['best_metric']:.6f} monitor={summary['best_metric_name']}"
    )
    return summary


class KnotSequenceRuntime:
    def __init__(
        self,
        *,
        checkpoint_path: str = "",
        training_mat_path: str = "",
        allow_fallback: bool = True,
        device: str = "auto",
    ):
        self.checkpoint_path = _resolve_path(checkpoint_path, default_checkpoint_path())
        self.training_mat_path = _resolve_path(training_mat_path, default_training_mat_path())
        self.allow_fallback = bool(allow_fallback)
        self.requested_device = str(device or "auto").strip().lower()

        self._lock = threading.Lock()
        self._loaded = False
        self._mode = "uninitialized"
        self._load_note = ""

        self._torch: Any = None
        self._model: Any = None
        self._device: Any = None
        self._vocab_size: int = 0

        self._fallback_transitions: Dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self._fallback_unigram: tuple[np.ndarray, np.ndarray] = (np.asarray([0], dtype=np.int64), np.asarray([1.0], dtype=np.float64))

    def mode(self) -> str:
        self._ensure_loaded()
        return str(self._mode)

    def load_note(self) -> str:
        self._ensure_loaded()
        return str(self._load_note)

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        with self._lock:
            if self._loaded:
                return
            error = self._try_load_pytorch_checkpoint()
            if error is None:
                print(f"[knot-seq] loaded checkpoint sampler: {self.checkpoint_path}")
                self._loaded = True
                return

            if not self.allow_fallback:
                raise RuntimeError(error)

            self._build_fallback_sampler()
            self._mode = "fallback_markov"
            self._load_note = error
            print(
                "[knot-seq] checkpoint sampler unavailable; using fallback Markov sampler. "
                f"reason={error} fallback_data={self.training_mat_path}"
            )
            self._loaded = True

    def _try_load_pytorch_checkpoint(self) -> Optional[str]:
        if not self.checkpoint_path.is_file():
            return f"Checkpoint not found: {self.checkpoint_path}"

        try:
            torch, nn, _, _ = _import_torch()
        except Exception as exc:
            return str(exc)

        try:
            checkpoint = torch.load(self.checkpoint_path, map_location="cpu")
        except Exception as exc:
            return f"Failed to load checkpoint {self.checkpoint_path}: {exc}"

        if not isinstance(checkpoint, dict):
            return f"Unsupported checkpoint format in {self.checkpoint_path}"

        model_hparams = dict(checkpoint.get("model_hparams") or {})
        embedding_obj = checkpoint.get("embedding_weights")
        if embedding_obj is None:
            return f"Checkpoint missing embedding_weights: {self.checkpoint_path}"

        if hasattr(embedding_obj, "detach"):
            embedding_np = np.asarray(embedding_obj.detach().cpu().numpy(), dtype=np.float32)
        else:
            embedding_np = np.asarray(embedding_obj, dtype=np.float32)
        if embedding_np.ndim != 2:
            return f"Invalid embedding shape in checkpoint: {embedding_np.shape}"

        vocab_size = int(checkpoint.get("vocab_size", embedding_np.shape[0]))
        if vocab_size <= 1:
            vocab_size = int(embedding_np.shape[0])
        if embedding_np.shape[0] < vocab_size:
            padded = np.zeros((vocab_size, embedding_np.shape[1]), dtype=np.float32)
            padded[: embedding_np.shape[0], :] = embedding_np
            embedding_np = padded
        elif embedding_np.shape[0] > vocab_size:
            vocab_size = int(embedding_np.shape[0])

        model_wrap = _KnotSequenceLSTM(
            torch=torch,
            nn=nn,
            vocab_size=vocab_size,
            embedding_dim=int(embedding_np.shape[1]),
            hidden_size=int(model_hparams.get("hidden_size", 128)),
            num_layers=int(model_hparams.get("num_layers", 1)),
            dropout=float(model_hparams.get("dropout", 0.0)),
            embedding_weights=embedding_np,
            freeze_embedding=True,
        )
        model = model_wrap.model

        state_dict = checkpoint.get("model_state")
        if not isinstance(state_dict, dict):
            return f"Checkpoint missing model_state dictionary: {self.checkpoint_path}"
        try:
            model.load_state_dict(state_dict, strict=True)
        except Exception as exc:
            return f"Failed to restore model_state from {self.checkpoint_path}: {exc}"

        requested_device = self.requested_device
        if requested_device == "cpu":
            device = torch.device("cpu")
        else:
            cuda_available = bool(torch.cuda.is_available())
            if requested_device == "cuda" and not cuda_available:
                return "Requested CUDA device for knot sampler, but CUDA is unavailable."
            if requested_device not in {"auto", "cuda"}:
                return f"Unsupported knot sampler device={self.requested_device!r}. Supported: auto, cpu, cuda."
            device = torch.device("cuda" if cuda_available else "cpu")

        model.eval()
        model.to(device)
        self._torch = torch
        self._model = model
        self._device = device
        self._vocab_size = int(vocab_size)
        self._mode = "pytorch_lstm"
        self._load_note = f"device={device}"
        return None

    def _build_fallback_sampler(self) -> None:
        if not self.training_mat_path.is_file():
            raise RuntimeError(
                "Knot-sequence checkpoint is unavailable and fallback training MAT is missing: "
                f"{self.training_mat_path}"
            )

        payload = scipy.io.loadmat(self.training_mat_path)
        inps = np.asarray(payload["inps"], dtype=np.int64)
        outs = np.asarray(payload["outs"], dtype=np.int64)
        if inps.shape != outs.shape or inps.ndim != 2:
            raise RuntimeError(
                f"Invalid inps/outs in fallback training data: {self.training_mat_path}"
            )

        flat_prev = inps.reshape(-1)
        flat_next = outs.reshape(-1)

        if flat_prev.size == 0:
            self._fallback_transitions = {}
            self._fallback_unigram = (
                np.asarray([0], dtype=np.int64),
                np.asarray([1.0], dtype=np.float64),
            )
            self._vocab_size = 1
            return

        pairs = np.column_stack([flat_prev, flat_next])
        unique_pairs, counts = np.unique(pairs, axis=0, return_counts=True)

        transitions: Dict[int, tuple[np.ndarray, np.ndarray]] = {}
        start = 0
        while start < unique_pairs.shape[0]:
            prev = int(unique_pairs[start, 0])
            end = start + 1
            while end < unique_pairs.shape[0] and int(unique_pairs[end, 0]) == prev:
                end += 1

            tokens = np.asarray(unique_pairs[start:end, 1], dtype=np.int64)
            probs = np.asarray(counts[start:end], dtype=np.float64)
            probs = probs / max(1.0, float(np.sum(probs)))
            transitions[prev] = (tokens, probs)
            start = end

        uni_tokens, uni_counts = np.unique(flat_next, return_counts=True)
        uni_probs = np.asarray(uni_counts, dtype=np.float64)
        uni_probs = uni_probs / max(1.0, float(np.sum(uni_probs)))

        self._fallback_transitions = transitions
        self._fallback_unigram = (
            np.asarray(uni_tokens, dtype=np.int64),
            np.asarray(uni_probs, dtype=np.float64),
        )
        self._vocab_size = int(max(int(np.max(flat_prev)), int(np.max(flat_next)))) + 1

    @staticmethod
    def _seed_to_int(seed: Optional[int]) -> int:
        if seed is None:
            return int(np.random.randint(0, 2**31 - 1))
        return int(seed) % (2**31 - 1)

    @staticmethod
    def _normalize_top_p(top_p: float) -> float:
        p = float(top_p)
        if not math.isfinite(p) or p <= 0.0:
            return 0.0
        if p >= 1.0:
            return 1.0
        return p

    @staticmethod
    def _apply_logits_top_k_top_p(
        logits,
        *,
        top_k: int,
        top_p: float,
        torch: Any,
    ):
        filtered = logits

        k = max(0, int(top_k))
        vocab_dim = int(filtered.shape[-1])
        if k > 0 and k < vocab_dim:
            vals, _ = torch.topk(filtered, k=k, dim=-1)
            kth = vals[..., -1, None]
            filtered = filtered.masked_fill(filtered < kth, float("-inf"))

        p = KnotSequenceRuntime._normalize_top_p(top_p)
        if p > 0.0 and p < 1.0:
            sorted_logits, sorted_indices = torch.sort(filtered, descending=True, dim=-1)
            sorted_probs = torch.softmax(sorted_logits, dim=-1)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            sorted_remove = cumulative_probs > p
            sorted_remove[..., 1:] = sorted_remove[..., :-1].clone()
            sorted_remove[..., 0] = False

            remove_mask = torch.zeros_like(sorted_remove, dtype=torch.bool)
            if filtered.ndim > 1:
                remove_mask.scatter_(-1, sorted_indices, sorted_remove)
            else:
                remove_mask.scatter_(0, sorted_indices, sorted_remove)
            filtered = filtered.masked_fill(remove_mask, float("-inf"))

        return filtered

    def _sample_pytorch(
        self,
        *,
        length: int,
        temperature: float,
        top_k: int,
        top_p: float,
        seed: Optional[int],
    ) -> np.ndarray:
        return self._sample_pytorch_many(
            length=length,
            count=1,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            seed=seed,
        )[0]

    def _sample_pytorch_many(
        self,
        *,
        length: int,
        count: int,
        temperature: float,
        top_k: int,
        top_p: float,
        seed: Optional[int],
    ) -> np.ndarray:
        torch = self._torch
        model = self._model
        if torch is None or model is None:
            raise RuntimeError("PyTorch runtime is not initialized.")

        device = self._device if self._device is not None else torch.device("cpu")
        gen = torch.Generator(device=device)
        gen.manual_seed(self._seed_to_int(seed))

        n = max(1, int(count))
        seq_len = max(1, int(length))
        temp = max(float(temperature), 1e-4)
        top_k = max(0, int(top_k))
        top_p = self._normalize_top_p(top_p)
        out = torch.zeros((n, seq_len), dtype=torch.long, device=device)
        token = torch.zeros((n,), dtype=torch.long, device=device)
        hidden = None

        with torch.inference_mode():
            for i in range(1, seq_len):
                logits, hidden = model.step(token, hidden)
                logits = logits.to(dtype=torch.float32)
                if not math.isclose(temp, 1.0):
                    logits = logits / temp

                logits_filtered = self._apply_logits_top_k_top_p(
                    logits,
                    top_k=top_k,
                    top_p=top_p,
                    torch=torch,
                )
                probs = torch.softmax(logits_filtered, dim=-1)
                probs_sum = probs.sum(dim=-1)
                bad = (~torch.isfinite(probs).all(dim=-1)) | (~torch.isfinite(probs_sum)) | (probs_sum <= 0.0)
                if bool(torch.any(bad).item()):
                    fallback_probs = torch.softmax(logits, dim=-1)
                    probs = torch.where(bad.unsqueeze(-1), fallback_probs, probs)
                next_token = torch.multinomial(probs, num_samples=1, generator=gen).squeeze(-1)

                out[:, i] = next_token
                token = next_token

        return np.asarray(out.detach().cpu().numpy(), dtype=np.int64)

    @staticmethod
    def _apply_temperature_top_k_top_p(
        tokens: np.ndarray,
        probs: np.ndarray,
        *,
        temperature: float,
        top_k: int,
        top_p: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        p = np.asarray(probs, dtype=np.float64)
        t = max(float(temperature), 1e-4)
        if not math.isclose(t, 1.0):
            logits = np.log(np.clip(p, 1e-12, None)) / t
            logits -= float(np.max(logits))
            p = np.exp(logits)
        p_sum = float(np.sum(p))
        if p_sum <= 0.0 or not np.isfinite(p_sum):
            p = np.full_like(p, 1.0 / float(p.shape[0]), dtype=np.float64)
        else:
            p = p / p_sum

        k = max(0, int(top_k))
        if k > 0 and k < p.shape[0]:
            idx = np.argpartition(p, -k)[-k:]
            tokens = tokens[idx]
            p = p[idx]
            p = p / float(np.sum(p))

        nucleus_p = KnotSequenceRuntime._normalize_top_p(top_p)
        if nucleus_p > 0.0 and nucleus_p < 1.0 and p.shape[0] > 1:
            order = np.argsort(-p)
            sorted_tokens = tokens[order]
            sorted_probs = p[order]
            cumulative = np.cumsum(sorted_probs)
            keep = cumulative <= nucleus_p
            keep[0] = True  # always keep at least the most probable token
            sorted_tokens = sorted_tokens[keep]
            sorted_probs = sorted_probs[keep]

            if sorted_probs.size == 0:
                tokens = tokens[order[:1]]
                p = np.asarray([1.0], dtype=np.float64)
            else:
                tokens = sorted_tokens
                p = sorted_probs / float(np.sum(sorted_probs))
        return tokens, p

    def _sample_fallback(
        self,
        *,
        length: int,
        temperature: float,
        top_k: int,
        top_p: float,
        seed: Optional[int],
    ) -> np.ndarray:
        out = np.zeros(int(length), dtype=np.int64)
        rng = np.random.default_rng(self._seed_to_int(seed))

        uni_tokens, uni_probs = self._fallback_unigram
        for i in range(1, int(length)):
            prev = int(out[i - 1])
            tokens, probs = self._fallback_transitions.get(prev, (uni_tokens, uni_probs))
            tokens_adj, probs_adj = self._apply_temperature_top_k_top_p(
                tokens,
                probs,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
            out[i] = int(rng.choice(tokens_adj, p=probs_adj))
        return out

    def _sample_fallback_many(
        self,
        *,
        length: int,
        count: int,
        temperature: float,
        top_k: int,
        top_p: float,
        seed: Optional[int],
    ) -> np.ndarray:
        base_seed = None if seed is None else self._seed_to_int(seed)
        rows = []
        for idx in range(max(1, int(count))):
            seed_i = None if base_seed is None else int(base_seed + idx)
            rows.append(
                self._sample_fallback(
                    length=length,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    seed=seed_i,
                )
            )
        return np.stack(rows, axis=0)

    def sample(
        self,
        *,
        length: int,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 0.8,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        self._ensure_loaded()
        length = max(1, int(length))

        if self._mode == "pytorch_lstm":
            return self._sample_pytorch(
                length=length,
                temperature=float(temperature),
                top_k=int(top_k),
                top_p=float(top_p),
                seed=seed,
            )

        return self._sample_fallback(
            length=length,
            temperature=float(temperature),
            top_k=int(top_k),
            top_p=float(top_p),
            seed=seed,
        )

    def sample_many(
        self,
        *,
        count: int,
        length: int,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 0.8,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        self._ensure_loaded()
        length = max(1, int(length))
        count = max(1, int(count))

        if self._mode == "pytorch_lstm":
            return self._sample_pytorch_many(
                length=length,
                count=count,
                temperature=float(temperature),
                top_k=int(top_k),
                top_p=float(top_p),
                seed=seed,
            )

        return self._sample_fallback_many(
            length=length,
            count=count,
            temperature=float(temperature),
            top_k=int(top_k),
            top_p=float(top_p),
            seed=seed,
        )


_RUNTIME_CACHE: Dict[tuple[str, str, bool, str], KnotSequenceRuntime] = {}
_RUNTIME_CACHE_LOCK = threading.Lock()


def get_knot_sequence_runtime(
    *,
    checkpoint_path: str = "",
    training_mat_path: str = "",
    allow_fallback: bool = True,
    device: str = "auto",
) -> KnotSequenceRuntime:
    key = (
        str(_resolve_path(checkpoint_path, default_checkpoint_path())),
        str(_resolve_path(training_mat_path, default_training_mat_path())),
        bool(allow_fallback),
        str(device or "auto").strip().lower(),
    )
    with _RUNTIME_CACHE_LOCK:
        runtime = _RUNTIME_CACHE.get(key)
        if runtime is None:
            runtime = KnotSequenceRuntime(
                checkpoint_path=checkpoint_path,
                training_mat_path=training_mat_path,
                allow_fallback=allow_fallback,
                device=device,
            )
            _RUNTIME_CACHE[key] = runtime
    return runtime


def sample_random_knot_log(
    *,
    slot_count: int,
    min_tokens: int = 400,
    extra_tokens: int = 200,
    temperature: float = 1.15,
    top_k: int = 0,
    top_p: float = 0.8,
    checkpoint_path: str = "",
    training_mat_path: str = "",
    allow_fallback: bool = True,
    device: str = "auto",
    seed: Optional[int] = None,
) -> np.ndarray:
    total_tokens = max(int(slot_count), int(min_tokens), int(slot_count) + int(max(0, extra_tokens)))
    runtime = get_knot_sequence_runtime(
        checkpoint_path=checkpoint_path,
        training_mat_path=training_mat_path,
        allow_fallback=allow_fallback,
        device=device,
    )
    return runtime.sample(
        length=total_tokens,
        temperature=float(temperature),
        top_k=int(top_k),
        top_p=float(top_p),
        seed=seed,
    )


def resolve_knot_sequence_runtime_info(
    *,
    checkpoint_path: str = "",
    training_mat_path: str = "",
    allow_fallback: bool = True,
    device: str = "auto",
) -> Dict[str, Any]:
    runtime = get_knot_sequence_runtime(
        checkpoint_path=checkpoint_path,
        training_mat_path=training_mat_path,
        allow_fallback=allow_fallback,
        device=device,
    )
    mode = str(runtime.mode())
    note = str(runtime.load_note() or "")
    return {
        "mode": mode,
        "used_pytorch_checkpoint": bool(mode == "pytorch_lstm"),
        "allow_fallback": bool(allow_fallback),
        "checkpoint_path": str(runtime.checkpoint_path),
        "training_data_path": str(runtime.training_mat_path),
        "load_note": note,
    }


@dataclass
class KnotSequenceEvalConfig:
    training_mat_path: str = str(default_training_mat_path())
    checkpoint_path: str = str(default_checkpoint_path())
    output_html_path: str = str((_repo_root() / "knot_model_checkpoint" / "knot_sequence_eval_report.html").resolve())
    output_plot_data_mat_path: str = ""
    output_matlab_script_path: str = ""
    num_generated_sequences: int = 500
    sequence_length: int = 0  # 0 uses reconstructed full training sequence length.
    top_k: int = 0
    top_p: float = 0.8
    allow_fallback: bool = True
    device: str = "auto"
    seed: Optional[int] = 42
    title: str = "Knot Sequence Generator Evaluation"


def _safe_float(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except Exception:
        return None
    if not np.isfinite(out):
        return None
    return out


def _format_number(value: Any, digits: int = 4) -> str:
    val = _safe_float(value)
    if val is None:
        return "n/a"
    if val == 0.0:
        return "0"
    aval = abs(val)
    if aval >= 10000.0 or aval < 1e-4:
        return f"{val:.{digits}e}"
    return f"{val:.{digits}f}"


def _format_percent(value: Any, digits: int = 2) -> str:
    val = _safe_float(value)
    if val is None:
        return "n/a"
    return f"{val:.{digits}f}%"


def _summary_stats(values: np.ndarray) -> Dict[str, Optional[float]]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "p05": None,
            "p25": None,
            "p50": None,
            "p75": None,
            "p95": None,
            "max": None,
        }
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "p05": float(np.percentile(arr, 5)),
        "p25": float(np.percentile(arr, 25)),
        "p50": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(np.max(arr)),
    }


def _extract_run_lengths(tokens_2d: np.ndarray, target_nonzero: bool) -> np.ndarray:
    arr = np.asarray(tokens_2d, dtype=np.int64)
    if arr.ndim != 2:
        raise RuntimeError(f"Expected 2D token matrix, got shape {arr.shape}")
    runs: list[int] = []
    for row in arr:
        mask = (row > 0) if target_nonzero else (row == 0)
        run = 0
        for flag in mask.tolist():
            if flag:
                run += 1
            elif run > 0:
                runs.append(run)
                run = 0
        if run > 0:
            runs.append(run)
    return np.asarray(runs, dtype=np.int64)


def _token_metrics(tokens_2d: np.ndarray) -> Dict[str, Any]:
    arr = np.asarray(tokens_2d, dtype=np.int64)
    if arr.ndim != 2:
        raise RuntimeError(f"Expected 2D token matrix, got shape {arr.shape}")
    total_slots = int(arr.size)
    knot_slots = int(np.count_nonzero(arr > 0))
    no_knot_slots = int(total_slots - knot_slots)
    knot_pct = (100.0 * knot_slots / total_slots) if total_slots > 0 else 0.0
    no_knot_pct = (100.0 * no_knot_slots / total_slots) if total_slots > 0 else 0.0

    knots_per_seq = np.sum(arr > 0, axis=1).astype(np.float64, copy=False)
    knot_runs = _extract_run_lengths(arr, target_nonzero=True)
    no_knot_runs = _extract_run_lengths(arr, target_nonzero=False)

    return {
        "sequence_count": int(arr.shape[0]),
        "sequence_length": int(arr.shape[1]),
        "total_slots": total_slots,
        "knot_slots": knot_slots,
        "no_knot_slots": no_knot_slots,
        "knot_slot_pct": float(knot_pct),
        "no_knot_slot_pct": float(no_knot_pct),
        "knots_per_sequence": _summary_stats(knots_per_seq),
        "knot_run_lengths": _summary_stats(knot_runs),
        "no_knot_run_lengths": _summary_stats(no_knot_runs),
    }


def _extract_token_row_lookup(payload: Dict[str, Any]) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    ptr_raw = payload.get("token_to_rowid_ptr", None)
    flat_raw = payload.get("token_to_rowid_flat", None)
    if ptr_raw is None or flat_raw is None:
        return None, None
    ptr = np.asarray(ptr_raw, dtype=np.int64)
    flat = np.asarray(flat_raw, dtype=np.int64).reshape(-1)
    if ptr.ndim != 2 or ptr.shape[1] < 2:
        return None, None
    return ptr, flat


def _decode_tokens_to_dictionary_rows(
    tokens_2d: np.ndarray,
    *,
    data_all_or: np.ndarray,
    token_to_rowid_ptr: Optional[np.ndarray],
    token_to_rowid_flat: Optional[np.ndarray],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    tokens = np.asarray(tokens_2d, dtype=np.int64).reshape(-1)
    nonzero = tokens > 0
    if not np.any(nonzero):
        return (
            np.empty((0, int(data_all_or.shape[1])), dtype=np.float64),
            np.empty((0,), dtype=np.int64),
        )

    tok_nz = tokens[nonzero]
    row_ids = np.zeros(tok_nz.shape[0], dtype=np.int64)
    has_lookup = token_to_rowid_ptr is not None and token_to_rowid_flat is not None

    if has_lookup:
        ptr = np.asarray(token_to_rowid_ptr, dtype=np.int64)
        flat = np.asarray(token_to_rowid_flat, dtype=np.int64).reshape(-1)
        valid_tok = (tok_nz >= 0) & (tok_nz < ptr.shape[0])
        if np.any(valid_tok):
            tok_valid = tok_nz[valid_tok]
            unique_tokens, inverse = np.unique(tok_valid, return_inverse=True)
            decoded = np.zeros(tok_valid.shape[0], dtype=np.int64)
            for idx, token_id in enumerate(unique_tokens.tolist()):
                start = int(ptr[token_id, 0])
                length = int(ptr[token_id, 1])
                if length <= 0:
                    continue
                end = start + length
                if start < 0 or end > flat.shape[0]:
                    continue
                members = flat[start:end]
                if members.size == 0:
                    continue
                hit = np.where(inverse == idx)[0]
                picks = rng.integers(0, members.size, size=hit.size)
                decoded[hit] = members[picks]
            row_ids[valid_tok] = decoded
    else:
        row_ids[:] = tok_nz

    valid_rows = (row_ids > 0) & (row_ids < int(data_all_or.shape[0]))
    if not np.any(valid_rows):
        return (
            np.empty((0, int(data_all_or.shape[1])), dtype=np.float64),
            np.empty((0,), dtype=np.int64),
        )
    valid_row_ids = row_ids[valid_rows]
    return np.asarray(data_all_or[valid_row_ids, :], dtype=np.float64), valid_row_ids


def _append_derived_knot_params(params: np.ndarray) -> tuple[np.ndarray, list[str]]:
    arr = np.asarray(params, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] < 9:
        return np.empty((0, 0), dtype=np.float64), []
    core = arr[:, :9]
    l100 = (
        core[:, 3] * (100.0**4)
        + core[:, 4] * (100.0**3)
        + core[:, 5] * (100.0**2)
        + core[:, 6] * 100.0
    )
    rd_minus_rl = core[:, 2] - core[:, 1]
    out = np.column_stack([core, l100, rd_minus_rl])
    names = [
        "th0_deg",
        "RL_mm",
        "RD_mm",
        "a1",
        "a2",
        "a3",
        "a4",
        "c1",
        "c2",
        "L100_mm",
        "RD_minus_RL_mm",
    ]
    return out, names


def _top_tokens_from_counts(
    counts: np.ndarray,
    *,
    top_n: int = 15,
    skip_zero: bool = True,
) -> list[Dict[str, Any]]:
    cnt = np.asarray(counts, dtype=np.int64).reshape(-1)
    total = int(np.sum(cnt))
    if total <= 0:
        return []
    start_idx = 1 if skip_zero and cnt.shape[0] > 1 else 0
    if start_idx > 0:
        work = cnt[start_idx:]
    else:
        work = cnt
    if work.size == 0:
        return []
    order = np.argsort(-work)
    out: list[Dict[str, Any]] = []
    for rank, idx in enumerate(order[: max(1, int(top_n))], start=1):
        token_id = int(idx + start_idx)
        count = int(work[idx])
        pct = (100.0 * count / total) if total > 0 else 0.0
        out.append({
            "rank": int(rank),
            "token": token_id,
            "count": count,
            "pct_all_slots": float(pct),
        })
    return out


def _js_divergence_from_counts(counts_a: np.ndarray, counts_b: np.ndarray) -> float:
    a = np.asarray(counts_a, dtype=np.float64).reshape(-1)
    b = np.asarray(counts_b, dtype=np.float64).reshape(-1)
    if a.shape != b.shape:
        raise RuntimeError(f"Count vectors must have matching shapes, got {a.shape} vs {b.shape}")
    sa = float(np.sum(a))
    sb = float(np.sum(b))
    if sa <= 0.0 and sb <= 0.0:
        return 0.0
    if sa <= 0.0:
        return math.inf
    if sb <= 0.0:
        return math.inf

    p = a / sa
    q = b / sb
    m = 0.5 * (p + q)

    def _kl(x: np.ndarray, y: np.ndarray) -> float:
        mask = x > 0.0
        if not np.any(mask):
            return 0.0
        return float(np.sum(x[mask] * np.log2(x[mask] / np.clip(y[mask], 1e-12, None))))

    return 0.5 * (_kl(p, m) + _kl(q, m))


def _match_sequence_length(tokens_2d: np.ndarray, sequence_length: int) -> np.ndarray:
    arr = np.asarray(tokens_2d, dtype=np.int64)
    if arr.ndim != 2:
        raise RuntimeError(f"Expected 2D token matrix, got shape {arr.shape}")
    target_len = max(1, int(sequence_length))
    if arr.shape[1] == target_len:
        return arr
    if arr.shape[1] > target_len:
        return np.asarray(arr[:, :target_len], dtype=np.int64)
    pad = np.zeros((arr.shape[0], target_len - arr.shape[1]), dtype=np.int64)
    return np.concatenate([arr, pad], axis=1)


def _reconstruct_full_training_sequences(inps: np.ndarray, outs: np.ndarray) -> np.ndarray:
    x = np.asarray(inps, dtype=np.int64)
    y = np.asarray(outs, dtype=np.int64)
    if x.shape != y.shape or x.ndim != 2 or x.shape[0] <= 0 or x.shape[1] <= 0:
        raise RuntimeError(f"Invalid inps/outs shapes in training MAT: {x.shape} vs {y.shape}")
    if x.shape[1] > 1 and not np.array_equal(x[:, 1:], y[:, :-1]):
        raise RuntimeError(
            "Training MAT inps/outs are not consistent next-token pairs; "
            "cannot reconstruct full reference sequences."
        )
    return np.concatenate([x[:, :1], y], axis=1)


def _parameter_histogram_payload(
    train_values: np.ndarray,
    gen_values: np.ndarray,
    *,
    bins: int = 32,
) -> Optional[Dict[str, Any]]:
    train = np.asarray(train_values, dtype=np.float64).reshape(-1)
    gen = np.asarray(gen_values, dtype=np.float64).reshape(-1)
    train = train[np.isfinite(train)]
    gen = gen[np.isfinite(gen)]
    combined = np.concatenate([train, gen])
    if combined.size == 0:
        return None

    x_min = float(np.min(combined))
    x_max = float(np.max(combined))
    if not np.isfinite(x_min) or not np.isfinite(x_max):
        return None
    if x_min == x_max:
        pad = max(1.0, abs(x_min) * 0.05)
        x_min -= pad
        x_max += pad

    bin_count = max(4, int(bins))
    edges = np.linspace(x_min, x_max, bin_count + 1, dtype=np.float64)
    train_counts, _ = np.histogram(train, bins=edges)
    gen_counts, _ = np.histogram(gen, bins=edges)
    train_total = int(np.sum(train_counts))
    gen_total = int(np.sum(gen_counts))
    train_share = train_counts.astype(np.float64) / float(max(1, train_total))
    gen_share = gen_counts.astype(np.float64) / float(max(1, gen_total))

    return {
        "edges": edges.tolist(),
        "train_share": train_share.tolist(),
        "gen_share": gen_share.tolist(),
        "train_count": train_total,
        "gen_count": gen_total,
        "x_min": x_min,
        "x_max": x_max,
        "y_max": float(max(np.max(train_share, initial=0.0), np.max(gen_share, initial=0.0), 1e-12)),
    }


def _histogram_svg(name: str, histogram: Optional[Dict[str, Any]]) -> str:
    if histogram is None:
        return "<p class=\"muted small\">No finite values available for this parameter.</p>"

    width = 720.0
    height = 260.0
    left = 52.0
    right = 18.0
    top = 18.0
    bottom = 42.0
    plot_w = width - left - right
    plot_h = height - top - bottom

    train_share = np.asarray(histogram["train_share"], dtype=np.float64)
    gen_share = np.asarray(histogram["gen_share"], dtype=np.float64)
    edges = np.asarray(histogram["edges"], dtype=np.float64)
    bin_count = int(train_share.shape[0])
    y_max = float(max(1e-12, histogram["y_max"]))
    bin_w = plot_w / float(max(1, bin_count))

    parts: list[str] = [
        f'<svg class="hist-svg" viewBox="0 0 {int(width)} {int(height)}" role="img" '
        f'aria-label="{escape(name)} histogram comparing training and generated distributions">',
        f'<rect x="0" y="0" width="{width:.0f}" height="{height:.0f}" rx="10" class="hist-bg" />',
        f'<line x1="{left:.1f}" y1="{top + plot_h:.1f}" x2="{left + plot_w:.1f}" y2="{top + plot_h:.1f}" class="hist-axis" />',
        f'<line x1="{left:.1f}" y1="{top:.1f}" x2="{left:.1f}" y2="{top + plot_h:.1f}" class="hist-axis" />',
    ]

    for i in range(bin_count):
        x0 = left + float(i) * bin_w
        train_h = plot_h * float(train_share[i]) / y_max
        gen_h = plot_h * float(gen_share[i]) / y_max
        train_x = x0 + bin_w * 0.10
        gen_x = x0 + bin_w * 0.52
        bar_w = max(1.0, bin_w * 0.36)
        train_y = top + plot_h - train_h
        gen_y = top + plot_h - gen_h
        title = (
            f"{name}: bin [{_format_number(edges[i], 5)}, {_format_number(edges[i + 1], 5)}), "
            f"train={100.0 * float(train_share[i]):.2f}%, gen={100.0 * float(gen_share[i]):.2f}%"
        )
        parts.append(
            f'<rect x="{train_x:.2f}" y="{train_y:.2f}" width="{bar_w:.2f}" height="{train_h:.2f}" class="hist-train">'
            f"<title>{escape(title)}</title></rect>"
        )
        parts.append(
            f'<rect x="{gen_x:.2f}" y="{gen_y:.2f}" width="{bar_w:.2f}" height="{gen_h:.2f}" class="hist-gen">'
            f"<title>{escape(title)}</title></rect>"
        )

    y_label = f"{100.0 * y_max:.1f}%"
    x_min = _format_number(histogram["x_min"], 5)
    x_max = _format_number(histogram["x_max"], 5)
    x_mid = _format_number(0.5 * (float(histogram["x_min"]) + float(histogram["x_max"])), 5)
    parts.extend(
        [
            f'<text x="{left - 8:.1f}" y="{top + 4:.1f}" class="hist-label" text-anchor="end">{escape(y_label)}</text>',
            f'<text x="{left:.1f}" y="{height - 14:.1f}" class="hist-label" text-anchor="middle">{escape(x_min)}</text>',
            f'<text x="{left + plot_w * 0.5:.1f}" y="{height - 14:.1f}" class="hist-label" text-anchor="middle">{escape(x_mid)}</text>',
            f'<text x="{left + plot_w:.1f}" y="{height - 14:.1f}" class="hist-label" text-anchor="middle">{escape(x_max)}</text>',
            f'<text x="{left:.1f}" y="{height - 2:.1f}" class="hist-label" text-anchor="start">parameter value</text>',
            f'<text x="{left - 38:.1f}" y="{top + plot_h * 0.5:.1f}" class="hist-label" text-anchor="middle" transform="rotate(-90 {left - 38:.1f} {top + plot_h * 0.5:.1f})">share per bin</text>',
            "</svg>",
        ]
    )
    return "".join(parts)


def _box_whisker_svg(name: str, train_stats: Dict[str, Any], gen_stats: Dict[str, Any]) -> str:
    width = 720.0
    height = 170.0
    left = 62.0
    right = 22.0
    top = 18.0
    bottom = 34.0
    plot_w = width - left - right
    plot_h = height - top - bottom
    train_y = top + plot_h * 0.35
    gen_y = top + plot_h * 0.72
    box_h = 22.0

    keys = ("p05", "p25", "p50", "p75", "p95")
    train_vals = [_safe_float(train_stats.get(k)) for k in keys]
    gen_vals = [_safe_float(gen_stats.get(k)) for k in keys]
    finite_vals = [v for v in train_vals + gen_vals if v is not None and np.isfinite(v)]
    if not finite_vals:
        return "<p class=\"muted small\">No finite values available for this parameter.</p>"

    x_min = float(min(finite_vals))
    x_max = float(max(finite_vals))
    if x_min == x_max:
        pad = max(1.0, abs(x_min) * 0.05)
        x_min -= pad
        x_max += pad
    span = max(1e-12, x_max - x_min)

    def sx(value: Any) -> Optional[float]:
        val = _safe_float(value)
        if val is None:
            return None
        return left + (float(val) - x_min) * plot_w / span

    def series(label: str, stats: Dict[str, Any], y: float, cls: str) -> str:
        w0 = sx(stats.get("p05"))
        q1 = sx(stats.get("p25"))
        med = sx(stats.get("p50"))
        q3 = sx(stats.get("p75"))
        w1 = sx(stats.get("p95"))
        mean = sx(stats.get("mean"))
        if any(v is None for v in (w0, q1, med, q3, w1)):
            return (
                f'<text x="{left:.1f}" y="{y + 4:.1f}" class="hist-label">'
                f"{escape(label)}: insufficient finite values</text>"
            )
        assert w0 is not None and q1 is not None and med is not None and q3 is not None and w1 is not None
        title = (
            f"{label} {name}: p05={_format_number(stats.get('p05'), 5)}, "
            f"p25={_format_number(stats.get('p25'), 5)}, "
            f"median={_format_number(stats.get('p50'), 5)}, "
            f"p75={_format_number(stats.get('p75'), 5)}, "
            f"p95={_format_number(stats.get('p95'), 5)}, "
            f"mean={_format_number(stats.get('mean'), 5)}"
        )
        parts = [
            f'<g class="{cls}"><title>{escape(title)}</title>',
            f'<text x="{left - 10:.1f}" y="{y + 4:.1f}" class="hist-label" text-anchor="end">{escape(label)}</text>',
            f'<line x1="{w0:.2f}" y1="{y:.2f}" x2="{q1:.2f}" y2="{y:.2f}" class="box-line" />',
            f'<line x1="{q3:.2f}" y1="{y:.2f}" x2="{w1:.2f}" y2="{y:.2f}" class="box-line" />',
            f'<line x1="{w0:.2f}" y1="{y - box_h * 0.35:.2f}" x2="{w0:.2f}" y2="{y + box_h * 0.35:.2f}" class="box-line" />',
            f'<line x1="{w1:.2f}" y1="{y - box_h * 0.35:.2f}" x2="{w1:.2f}" y2="{y + box_h * 0.35:.2f}" class="box-line" />',
            f'<rect x="{q1:.2f}" y="{y - box_h * 0.5:.2f}" width="{max(1.0, q3 - q1):.2f}" height="{box_h:.2f}" class="box-rect" />',
            f'<line x1="{med:.2f}" y1="{y - box_h * 0.5:.2f}" x2="{med:.2f}" y2="{y + box_h * 0.5:.2f}" class="box-median" />',
        ]
        if mean is not None:
            parts.append(
                f'<circle cx="{mean:.2f}" cy="{y:.2f}" r="3.2" class="box-mean">'
                f"<title>{escape(label)} mean={_format_number(stats.get('mean'), 5)}</title></circle>"
            )
        parts.append("</g>")
        return "".join(parts)

    tick_values = [x_min, 0.5 * (x_min + x_max), x_max]
    parts: list[str] = [
        f'<svg class="box-svg" viewBox="0 0 {int(width)} {int(height)}" role="img" '
        f'aria-label="{escape(name)} box and whisker comparison">',
        f'<rect x="0" y="0" width="{width:.0f}" height="{height:.0f}" rx="10" class="hist-bg" />',
        f'<line x1="{left:.1f}" y1="{top + plot_h:.1f}" x2="{left + plot_w:.1f}" y2="{top + plot_h:.1f}" class="hist-axis" />',
    ]
    for val in tick_values:
        x = sx(val)
        if x is None:
            continue
        parts.extend(
            [
                f'<line x1="{x:.1f}" y1="{top + plot_h:.1f}" x2="{x:.1f}" y2="{top + plot_h + 5:.1f}" class="hist-axis" />',
                f'<text x="{x:.1f}" y="{height - 11:.1f}" class="hist-label" text-anchor="middle">{escape(_format_number(val, 5))}</text>',
            ]
        )
    parts.extend(
        [
            series("Training", train_stats, train_y, "box-train"),
            series("Generated", gen_stats, gen_y, "box-gen"),
            f'<text x="{left:.1f}" y="{height - 1:.1f}" class="hist-label" text-anchor="start">whiskers=p05/p95, box=p25-p75, line=median, dot=mean</text>',
            "</svg>",
        ]
    )
    return "".join(parts)


def _matlab_cellstr(values: list[str]) -> np.ndarray:
    arr = np.empty((1, len(values)), dtype=object)
    for idx, value in enumerate(values):
        arr[0, idx] = str(value)
    return arr


def _matlab_string_literal(value: str) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _default_eval_aux_path(output_html_path: Path, suffix: str) -> Path:
    return output_html_path.with_name(f"{output_html_path.stem}{suffix}")


def _write_knot_eval_matlab_exports(
    *,
    output_html_path: Path,
    output_plot_data_mat_path: str,
    output_matlab_script_path: str,
    train_params: np.ndarray,
    gen_params: np.ndarray,
    param_names: list[str],
) -> tuple[Path, Path]:
    data_path = (
        Path(output_plot_data_mat_path).expanduser().resolve()
        if str(output_plot_data_mat_path or "").strip()
        else _default_eval_aux_path(output_html_path, "_plot_data.mat")
    )
    script_path = (
        Path(output_matlab_script_path).expanduser().resolve()
        if str(output_matlab_script_path or "").strip()
        else _default_eval_aux_path(output_html_path, "_plots.m")
    )
    data_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.parent.mkdir(parents=True, exist_ok=True)

    scipy.io.savemat(
        data_path,
        {
            "train_params": np.asarray(train_params, dtype=np.float64),
            "gen_params": np.asarray(gen_params, dtype=np.float64),
            "param_names": _matlab_cellstr(param_names),
        },
    )

    relative_data_path = data_path.name if data_path.parent == script_path.parent else str(data_path)
    script = f"""% Auto-generated by board_cli.py knots evaluate.
% Generates train-vs-generated histograms and box/whisker comparisons for knot parameters.

clearvars;
close all;

script_dir = fileparts(mfilename('fullpath'));
if isempty(script_dir)
    script_dir = pwd;
end

data_path = fullfile(script_dir, {_matlab_string_literal(relative_data_path)});
if ~isfile(data_path)
    data_path = {_matlab_string_literal(str(data_path))};
end

S = load(data_path);
train_params = S.train_params;
gen_params = S.gen_params;
param_names = S.param_names;
if isstring(param_names)
    param_names = cellstr(param_names);
end
param_names = param_names(:)';
n_params = numel(param_names);

n_cols = ceil(sqrt(n_params));
n_rows = ceil(n_params / n_cols);

figure('Name', 'Knot Parameter Histograms', 'Color', 'w');
tiledlayout(n_rows, n_cols, 'TileSpacing', 'compact', 'Padding', 'compact');
for i = 1:n_params
    nexttile;
    train_i = train_params(:, i);
    gen_i = gen_params(:, i);
    train_i = train_i(isfinite(train_i));
    gen_i = gen_i(isfinite(gen_i));
    combined = [train_i; gen_i];
    if isempty(combined)
        title(param_names{{i}}, 'Interpreter', 'none');
        axis off;
        continue;
    end
    edges = linspace(min(combined), max(combined), 33);
    if numel(unique(edges)) < 2
        edges = linspace(min(combined) - 0.5, max(combined) + 0.5, 33);
    end
    histogram(train_i, edges, 'Normalization', 'probability', ...
        'FaceAlpha', 0.45, 'EdgeAlpha', 0.25, 'DisplayName', 'Training');
    hold on;
    histogram(gen_i, edges, 'Normalization', 'probability', ...
        'FaceAlpha', 0.45, 'EdgeAlpha', 0.25, 'DisplayName', 'Generated');
    hold off;
    title(param_names{{i}}, 'Interpreter', 'none');
    xlabel('Value');
    ylabel('Probability');
    grid on;
    if i == 1
        legend('Location', 'best');
    end
end

figure('Name', 'Knot Parameter Box/Whisker Comparison', 'Color', 'w');
tiledlayout(n_rows, n_cols, 'TileSpacing', 'compact', 'Padding', 'compact');
for i = 1:n_params
    nexttile;
    train_i = train_params(:, i);
    gen_i = gen_params(:, i);
    train_i = train_i(isfinite(train_i));
    gen_i = gen_i(isfinite(gen_i));
    if isempty(train_i) && isempty(gen_i)
        title(param_names{{i}}, 'Interpreter', 'none');
        axis off;
        continue;
    end
    hold on;
    draw_p05_p95_box(1, train_i, [0.00 0.45 0.62]);
    draw_p05_p95_box(2, gen_i, [0.85 0.40 0.00]);
    hold off;
    xlim([0.4 2.6]);
    set(gca, 'XTick', [1 2], 'XTickLabel', {{'Training', 'Generated'}});
    title(param_names{{i}}, 'Interpreter', 'none');
    ylabel('Value');
    grid on;
end

function draw_p05_p95_box(x, values, color)
    values = values(isfinite(values));
    if isempty(values)
        return;
    end
    q = local_percentiles(values, [5 25 50 75 95]);
    mean_value = mean(values);
    box_width = 0.34;
    cap_width = 0.22;
    line_width = 1.4;
    patch( ...
        [x - box_width, x + box_width, x + box_width, x - box_width], ...
        [q(2), q(2), q(4), q(4)], ...
        color, ...
        'FaceAlpha', 0.25, ...
        'EdgeColor', color, ...
        'LineWidth', line_width);
    plot([x x], [q(1) q(2)], '-', 'Color', color, 'LineWidth', line_width);
    plot([x x], [q(4) q(5)], '-', 'Color', color, 'LineWidth', line_width);
    plot([x - cap_width, x + cap_width], [q(1) q(1)], '-', 'Color', color, 'LineWidth', line_width);
    plot([x - cap_width, x + cap_width], [q(5) q(5)], '-', 'Color', color, 'LineWidth', line_width);
    plot([x - box_width, x + box_width], [q(3) q(3)], '-', 'Color', color, 'LineWidth', line_width + 0.8);
    plot(x, mean_value, 'o', 'MarkerFaceColor', color, 'MarkerEdgeColor', 'w', 'MarkerSize', 4);
end

function q = local_percentiles(values, pct)
    values = sort(values(:));
    n = numel(values);
    if n == 1
        q = repmat(values(1), size(pct));
        return;
    end
    positions = 1 + (pct(:)' ./ 100) .* (n - 1);
    lo = floor(positions);
    hi = ceil(positions);
    frac = positions - lo;
    q = values(lo)' .* (1 - frac) + values(hi)' .* frac;
end
"""
    script_path.write_text(script, encoding="utf-8")
    return data_path, script_path


def _build_knot_eval_report_html(
    *,
    title: str,
    generated_at_utc: str,
    config: KnotSequenceEvalConfig,
    runtime_mode: str,
    runtime_note: str,
    training_path: Path,
    checkpoint_path: Path,
    training_metrics: Dict[str, Any],
    generated_metrics: Dict[str, Any],
    js_all: float,
    js_nonzero: float,
    coverage_pct: float,
    bad_rate_training: Optional[float],
    bad_rate_generated: Optional[float],
    param_rows: list[Dict[str, Any]],
    top_training_tokens: list[Dict[str, Any]],
    top_generated_tokens: list[Dict[str, Any]],
) -> str:
    def delta(a: Any, b: Any) -> Optional[float]:
        aa = _safe_float(a)
        bb = _safe_float(b)
        if aa is None or bb is None:
            return None
        return bb - aa

    key_rows = [
        (
            "Segments with knots",
            _format_percent(training_metrics["knot_slot_pct"]),
            _format_percent(generated_metrics["knot_slot_pct"]),
            _format_percent(delta(training_metrics["knot_slot_pct"], generated_metrics["knot_slot_pct"])),
        ),
        (
            "Segments with no knot",
            _format_percent(training_metrics["no_knot_slot_pct"]),
            _format_percent(generated_metrics["no_knot_slot_pct"]),
            _format_percent(delta(training_metrics["no_knot_slot_pct"], generated_metrics["no_knot_slot_pct"])),
        ),
        (
            "Average knots per sequence",
            _format_number(training_metrics["knots_per_sequence"]["mean"], 3),
            _format_number(generated_metrics["knots_per_sequence"]["mean"], 3),
            _format_number(
                delta(training_metrics["knots_per_sequence"]["mean"], generated_metrics["knots_per_sequence"]["mean"]),
                3,
            ),
        ),
        (
            "Average knot cluster length (slots)",
            _format_number(training_metrics["knot_run_lengths"]["mean"], 3),
            _format_number(generated_metrics["knot_run_lengths"]["mean"], 3),
            _format_number(
                delta(training_metrics["knot_run_lengths"]["mean"], generated_metrics["knot_run_lengths"]["mean"]),
                3,
            ),
        ),
        (
            "Average continuous no-knot length (slots)",
            _format_number(training_metrics["no_knot_run_lengths"]["mean"], 3),
            _format_number(generated_metrics["no_knot_run_lengths"]["mean"], 3),
            _format_number(
                delta(training_metrics["no_knot_run_lengths"]["mean"], generated_metrics["no_knot_run_lengths"]["mean"]),
                3,
            ),
        ),
    ]

    if bad_rate_training is not None and bad_rate_generated is not None:
        key_rows.append(
            (
                "Decoded bad-knot row rate (diagnostic)",
                _format_percent(bad_rate_training),
                _format_percent(bad_rate_generated),
                _format_percent(delta(bad_rate_training, bad_rate_generated)),
            )
        )

    diag_rows = [
        ("JS divergence (all tokens)", _format_number(js_all, 6)),
        ("JS divergence (non-zero tokens)", _format_number(js_nonzero, 6)),
        ("Generated non-zero token coverage vs training", _format_percent(coverage_pct)),
        ("Training sequences", str(int(training_metrics["sequence_count"]))),
        ("Generated sequences", str(int(generated_metrics["sequence_count"]))),
        ("Sequence length", str(int(generated_metrics["sequence_length"]))),
    ]

    def _run_rows(label: str, stats: Dict[str, Optional[float]]) -> str:
        return (
            "<tr>"
            f"<td>{escape(label)}</td>"
            f"<td>{int(stats.get('count') or 0)}</td>"
            f"<td>{_format_number(stats.get('mean'), 3)}</td>"
            f"<td>{_format_number(stats.get('std'), 3)}</td>"
            f"<td>{_format_number(stats.get('p50'), 3)}</td>"
            f"<td>{_format_number(stats.get('p95'), 3)}</td>"
            f"<td>{_format_number(stats.get('max'), 3)}</td>"
            "</tr>"
        )

    def _token_rows(rows: list[Dict[str, Any]]) -> str:
        parts: list[str] = []
        for row in rows:
            parts.append(
                "<tr>"
                f"<td>{int(row['rank'])}</td>"
                f"<td>{int(row['token'])}</td>"
                f"<td>{int(row['count'])}</td>"
                f"<td>{_format_percent(row['pct_all_slots'], 3)}</td>"
                "</tr>"
            )
        return "\n".join(parts)

    param_rows_html = []
    for row in param_rows:
        param_rows_html.append(
            "<tr>"
            f"<td>{escape(str(row['name']))}</td>"
            f"<td>{_format_number(row['train']['mean'], 5)}</td>"
            f"<td>{_format_number(row['gen']['mean'], 5)}</td>"
            f"<td>{_format_number(delta(row['train']['mean'], row['gen']['mean']), 5)}</td>"
            f"<td>{_format_number(row['train']['std'], 5)}</td>"
            f"<td>{_format_number(row['gen']['std'], 5)}</td>"
            f"<td>{_format_number(row['train']['p05'], 5)}</td>"
            f"<td>{_format_number(row['gen']['p05'], 5)}</td>"
            f"<td>{_format_number(row['train']['p50'], 5)}</td>"
            f"<td>{_format_number(row['gen']['p50'], 5)}</td>"
            f"<td>{_format_number(row['train']['p95'], 5)}</td>"
            f"<td>{_format_number(row['gen']['p95'], 5)}</td>"
            "</tr>"
        )

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{escape(title)}</title>
  <style>
    :root {{
      --bg: #f4f7fb;
      --card: #ffffff;
      --ink: #163047;
      --muted: #6a7d8f;
      --line: #d9e2ea;
      --accent: #0f7a9a;
      --accent-soft: #e7f5fb;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Segoe UI", Tahoma, Geneva, Verdana, sans-serif;
      color: var(--ink);
      background: linear-gradient(170deg, #eef4fb 0%, var(--bg) 100%);
    }}
    .wrap {{ max-width: 1480px; margin: 0 auto; padding: 22px 26px 30px; }}
    h1 {{ margin: 0 0 8px; font-size: 28px; }}
    h2 {{ margin: 0 0 12px; font-size: 20px; }}
    p.meta {{ margin: 0; color: var(--muted); }}
    .grid {{ display: grid; grid-template-columns: repeat(12, 1fr); gap: 14px; margin-top: 16px; }}
    .card {{
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 14px 16px;
      box-shadow: 0 3px 10px rgba(16, 45, 64, 0.05);
    }}
    .col-12 {{ grid-column: span 12; }}
    .col-8 {{ grid-column: span 8; }}
    .col-6 {{ grid-column: span 6; }}
    .col-4 {{ grid-column: span 4; }}
    .chip {{
      display: inline-block;
      padding: 3px 8px;
      border-radius: 999px;
      font-size: 12px;
      background: var(--accent-soft);
      color: var(--accent);
      margin-right: 6px;
      margin-top: 4px;
    }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    th, td {{ border: 1px solid var(--line); padding: 7px 8px; text-align: right; }}
    th:first-child, td:first-child {{ text-align: left; }}
    th {{ background: #eff5fa; }}
    .mono {{ font-family: Consolas, Menlo, Monaco, monospace; }}
    .muted {{ color: var(--muted); }}
    .small {{ font-size: 12px; }}
    .hist-grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 14px; margin-top: 14px; }}
    .hist-card {{ border: 1px solid var(--line); border-radius: 10px; padding: 10px; background: #fbfdff; }}
    .hist-card h3 {{ margin: 0 0 6px; font-size: 15px; }}
    .hist-legend {{ display: flex; gap: 14px; align-items: center; margin: 0 0 6px; color: var(--muted); font-size: 12px; }}
    .swatch {{ display: inline-block; width: 11px; height: 11px; border-radius: 2px; margin-right: 5px; vertical-align: -1px; }}
    .swatch-train {{ background: #0f7a9a; }}
    .swatch-gen {{ background: #d97706; }}
    .hist-svg {{ width: 100%; height: auto; display: block; }}
    .box-svg {{ width: 100%; height: auto; display: block; margin: 6px 0 10px; }}
    .hist-bg {{ fill: #ffffff; }}
    .hist-axis {{ stroke: #9eb0bf; stroke-width: 1; }}
    .hist-label {{ fill: var(--muted); font-size: 11px; font-family: "Segoe UI", Tahoma, Geneva, Verdana, sans-serif; }}
    .hist-train {{ fill: #0f7a9a; opacity: 0.72; }}
    .hist-gen {{ fill: #d97706; opacity: 0.62; }}
    .box-line {{ stroke-width: 2; fill: none; }}
    .box-rect {{ opacity: 0.30; }}
    .box-median {{ stroke-width: 2.5; }}
    .box-mean {{ stroke: #ffffff; stroke-width: 1; }}
    .box-train .box-line, .box-train .box-median {{ stroke: #0f7a9a; }}
    .box-train .box-rect, .box-train .box-mean {{ fill: #0f7a9a; }}
    .box-gen .box-line, .box-gen .box-median {{ stroke: #d97706; }}
    .box-gen .box-rect, .box-gen .box-mean {{ fill: #d97706; }}
    @media (max-width: 1100px) {{
      .col-8, .col-6, .col-4 {{ grid-column: span 12; }}
      .hist-grid {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>{escape(title)}</h1>
    <p class="meta">
      Generated at {escape(generated_at_utc)} UTC
      &nbsp;|&nbsp; Runtime mode: <span class="mono">{escape(runtime_mode)}</span>
    </p>
    <p class="meta">
      Training MAT: <span class="mono">{escape(str(training_path))}</span><br />
      Checkpoint: <span class="mono">{escape(str(checkpoint_path))}</span>
    </p>
    {"<p class='meta small'>Runtime note: " + escape(runtime_note) + "</p>" if runtime_note else ""}

    <div class="grid">
      <section class="card col-12">
        <h2>Configuration</h2>
        <span class="chip">n_generated={int(config.num_generated_sequences)}</span>
        <span class="chip">sequence_length={int(generated_metrics['sequence_length'])}</span>
        <span class="chip">top_k={int(config.top_k)}</span>
        <span class="chip">top_p={_format_number(config.top_p, 4)}</span>
        <span class="chip">device={escape(str(config.device))}</span>
        <span class="chip">allow_fallback={str(bool(config.allow_fallback)).lower()}</span>
      </section>

      <section class="card col-8">
        <h2>Requested Metric Comparison</h2>
        <table>
          <thead>
            <tr><th>Metric</th><th>Training</th><th>Generated</th><th>Delta</th></tr>
          </thead>
          <tbody>
            {"".join(f"<tr><td>{escape(r[0])}</td><td>{r[1]}</td><td>{r[2]}</td><td>{r[3]}</td></tr>" for r in key_rows)}
          </tbody>
        </table>
      </section>

      <section class="card col-4">
        <h2>Additional Diagnostics</h2>
        <table>
          <tbody>
            {"".join(f"<tr><td>{escape(k)}</td><td>{v}</td></tr>" for k, v in diag_rows)}
          </tbody>
        </table>
      </section>

      <section class="card col-12">
        <h2>Knot Parameter Distribution Comparison</h2>
        <table>
          <thead>
            <tr>
              <th>Parameter</th>
              <th>Train mean</th><th>Gen mean</th><th>Delta mean</th>
              <th>Train std</th><th>Gen std</th>
              <th>Train p05</th><th>Gen p05</th>
              <th>Train p50</th><th>Gen p50</th>
              <th>Train p95</th><th>Gen p95</th>
            </tr>
          </thead>
          <tbody>
            {"".join(param_rows_html)}
          </tbody>
        </table>
        <p class="muted small">
          For clustered dictionaries, each non-zero token is decoded to a dictionary row by uniform random member
          sampling within that token cluster. The same decoding rule is applied to training and generated sequences.
        </p>
        <div class="hist-grid">
          {"".join(
              "<div class='hist-card'>"
              f"<h3>{escape(str(row['name']))}</h3>"
              "<div class='hist-legend'>"
              "<span><span class='swatch swatch-train'></span>Training</span>"
              "<span><span class='swatch swatch-gen'></span>Generated</span>"
              f"<span>n_train={int((row.get('histogram') or {}).get('train_count') or 0)}</span>"
              f"<span>n_gen={int((row.get('histogram') or {}).get('gen_count') or 0)}</span>"
              "</div>"
              f"{_box_whisker_svg(str(row['name']), row['train'], row['gen'])}"
              f"{_histogram_svg(str(row['name']), row.get('histogram'))}"
              "</div>"
              for row in param_rows
          )}
        </div>
      </section>

      <section class="card col-6">
        <h2>Run Length Statistics (Training)</h2>
        <table>
          <thead>
            <tr><th>Run Type</th><th>Count</th><th>Mean</th><th>Std</th><th>P50</th><th>P95</th><th>Max</th></tr>
          </thead>
          <tbody>
            {_run_rows("Knot runs (token > 0)", training_metrics["knot_run_lengths"])}
            {_run_rows("No-knot runs (token = 0)", training_metrics["no_knot_run_lengths"])}
          </tbody>
        </table>
      </section>

      <section class="card col-6">
        <h2>Run Length Statistics (Generated)</h2>
        <table>
          <thead>
            <tr><th>Run Type</th><th>Count</th><th>Mean</th><th>Std</th><th>P50</th><th>P95</th><th>Max</th></tr>
          </thead>
          <tbody>
            {_run_rows("Knot runs (token > 0)", generated_metrics["knot_run_lengths"])}
            {_run_rows("No-knot runs (token = 0)", generated_metrics["no_knot_run_lengths"])}
          </tbody>
        </table>
      </section>

      <section class="card col-6">
        <h2>Top Non-Zero Tokens (Training)</h2>
        <table>
          <thead>
            <tr><th>Rank</th><th>Token</th><th>Count</th><th>Share of all slots</th></tr>
          </thead>
          <tbody>
            {_token_rows(top_training_tokens)}
          </tbody>
        </table>
      </section>

      <section class="card col-6">
        <h2>Top Non-Zero Tokens (Generated)</h2>
        <table>
          <thead>
            <tr><th>Rank</th><th>Token</th><th>Count</th><th>Share of all slots</th></tr>
          </thead>
          <tbody>
            {_token_rows(top_generated_tokens)}
          </tbody>
        </table>
      </section>
    </div>
  </div>
</body>
</html>
"""
    return html


def evaluate_knot_sequence_generator(cfg: KnotSequenceEvalConfig) -> Dict[str, Any]:
    training_path = _resolve_path(cfg.training_mat_path, default_training_mat_path())
    checkpoint_path = _resolve_path(cfg.checkpoint_path, default_checkpoint_path())
    output_html_path = Path(cfg.output_html_path).expanduser().resolve()

    if not training_path.is_file():
        raise RuntimeError(f"Training MAT file not found: {training_path}")

    payload = scipy.io.loadmat(training_path)
    inps = np.asarray(payload["inps"], dtype=np.int64)
    outs = np.asarray(payload["outs"], dtype=np.int64)
    full_training_sequences = _reconstruct_full_training_sequences(inps, outs)

    seq_len = int(cfg.sequence_length) if int(cfg.sequence_length) > 0 else int(full_training_sequences.shape[1])
    seq_len = max(1, seq_len)
    training_tokens = _match_sequence_length(full_training_sequences, seq_len)

    data_all_or = np.asarray(payload["Data_all_or"], dtype=np.float64)
    if data_all_or.ndim != 2 or data_all_or.shape[1] < 9:
        raise RuntimeError(f"Invalid Data_all_or shape in training MAT: {data_all_or.shape}")
    token_to_rowid_ptr, token_to_rowid_flat = _extract_token_row_lookup(payload)

    runtime = get_knot_sequence_runtime(
        checkpoint_path=str(checkpoint_path),
        training_mat_path=str(training_path),
        allow_fallback=bool(cfg.allow_fallback),
        device=str(cfg.device),
    )

    n_gen = max(1, int(cfg.num_generated_sequences))
    base_seed = (None if cfg.seed is None else int(cfg.seed))
    generated_tokens = runtime.sample_many(
        count=n_gen,
        length=seq_len,
        temperature=1.15,
        top_k=max(0, int(cfg.top_k)),
        top_p=float(cfg.top_p),
        seed=base_seed,
    )

    decode_seed = int(base_seed if base_seed is not None else np.random.randint(0, 2**31 - 1))
    rng_train = np.random.default_rng(decode_seed + 11)
    rng_gen = np.random.default_rng(decode_seed + 29)
    train_params_raw, train_rows = _decode_tokens_to_dictionary_rows(
        training_tokens,
        data_all_or=data_all_or,
        token_to_rowid_ptr=token_to_rowid_ptr,
        token_to_rowid_flat=token_to_rowid_flat,
        rng=rng_train,
    )
    gen_params_raw, gen_rows = _decode_tokens_to_dictionary_rows(
        generated_tokens,
        data_all_or=data_all_or,
        token_to_rowid_ptr=token_to_rowid_ptr,
        token_to_rowid_flat=token_to_rowid_flat,
        rng=rng_gen,
    )

    train_params, param_names = _append_derived_knot_params(train_params_raw)
    gen_params, _ = _append_derived_knot_params(gen_params_raw)
    if train_params.ndim != 2 or gen_params.ndim != 2 or not param_names:
        raise RuntimeError("Failed to derive knot parameter statistics from decoded rows.")

    training_metrics = _token_metrics(training_tokens)
    generated_metrics = _token_metrics(generated_tokens)

    vocab_size = int(
        max(
            int(np.max(training_tokens)) if training_tokens.size > 0 else 0,
            int(np.max(generated_tokens)) if generated_tokens.size > 0 else 0,
            int(data_all_or.shape[0] - 1),
            int(token_to_rowid_ptr.shape[0] - 1) if token_to_rowid_ptr is not None else 0,
        )
        + 1
    )
    train_counts = np.bincount(training_tokens.reshape(-1), minlength=vocab_size).astype(np.int64, copy=False)
    gen_counts = np.bincount(generated_tokens.reshape(-1), minlength=vocab_size).astype(np.int64, copy=False)

    js_all = _js_divergence_from_counts(train_counts, gen_counts)
    if train_counts.shape[0] > 1 and gen_counts.shape[0] > 1:
        js_nonzero = _js_divergence_from_counts(train_counts[1:], gen_counts[1:])
    else:
        js_nonzero = 0.0

    train_nonzero = set((np.where(train_counts[1:] > 0)[0] + 1).tolist()) if train_counts.shape[0] > 1 else set()
    gen_nonzero = set((np.where(gen_counts[1:] > 0)[0] + 1).tolist()) if gen_counts.shape[0] > 1 else set()
    if train_nonzero:
        coverage_pct = 100.0 * float(len(train_nonzero & gen_nonzero)) / float(len(train_nonzero))
    else:
        coverage_pct = 0.0

    bad_path = _repo_root() / "data" / "bad_knots.mat"
    bad_rate_training: Optional[float] = None
    bad_rate_generated: Optional[float] = None
    if bad_path.is_file():
        try:
            bad_payload = scipy.io.loadmat(bad_path)
            bad_raw = np.asarray(bad_payload.get("bad_knots", np.empty((0,))), dtype=np.int64).reshape(-1)
            bad_set = set(int(v) for v in bad_raw.tolist())
            if bad_set:
                if train_rows.size > 0:
                    bad_rate_training = 100.0 * float(np.sum(np.isin(train_rows, list(bad_set)))) / float(train_rows.size)
                else:
                    bad_rate_training = 0.0
                if gen_rows.size > 0:
                    bad_rate_generated = 100.0 * float(np.sum(np.isin(gen_rows, list(bad_set)))) / float(gen_rows.size)
                else:
                    bad_rate_generated = 0.0
        except Exception:
            bad_rate_training = None
            bad_rate_generated = None

    param_rows: list[Dict[str, Any]] = []
    for idx, name in enumerate(param_names):
        train_stat = _summary_stats(train_params[:, idx] if train_params.size > 0 else np.empty((0,)))
        gen_stat = _summary_stats(gen_params[:, idx] if gen_params.size > 0 else np.empty((0,)))
        param_rows.append({
            "name": name,
            "train": train_stat,
            "gen": gen_stat,
            "histogram": _parameter_histogram_payload(
                train_params[:, idx] if train_params.size > 0 else np.empty((0,)),
                gen_params[:, idx] if gen_params.size > 0 else np.empty((0,)),
            ),
        })

    top_training_tokens = _top_tokens_from_counts(train_counts, top_n=15, skip_zero=True)
    top_generated_tokens = _top_tokens_from_counts(gen_counts, top_n=15, skip_zero=True)

    output_html_path.parent.mkdir(parents=True, exist_ok=True)
    generated_at_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    html_report = _build_knot_eval_report_html(
        title=str(cfg.title or "Knot Sequence Generator Evaluation"),
        generated_at_utc=generated_at_utc,
        config=cfg,
        runtime_mode=str(runtime.mode()),
        runtime_note=str(runtime.load_note() or ""),
        training_path=training_path,
        checkpoint_path=checkpoint_path,
        training_metrics=training_metrics,
        generated_metrics=generated_metrics,
        js_all=js_all,
        js_nonzero=js_nonzero,
        coverage_pct=coverage_pct,
        bad_rate_training=bad_rate_training,
        bad_rate_generated=bad_rate_generated,
        param_rows=param_rows,
        top_training_tokens=top_training_tokens,
        top_generated_tokens=top_generated_tokens,
    )
    output_html_path.write_text(html_report, encoding="utf-8")
    plot_data_mat_path, matlab_script_path = _write_knot_eval_matlab_exports(
        output_html_path=output_html_path,
        output_plot_data_mat_path=str(cfg.output_plot_data_mat_path),
        output_matlab_script_path=str(cfg.output_matlab_script_path),
        train_params=train_params,
        gen_params=gen_params,
        param_names=param_names,
    )

    return {
        "output_html_path": str(output_html_path),
        "output_plot_data_mat_path": str(plot_data_mat_path),
        "output_matlab_script_path": str(matlab_script_path),
        "training_mat_path": str(training_path),
        "checkpoint_path": str(checkpoint_path),
        "runtime_mode": str(runtime.mode()),
        "runtime_note": str(runtime.load_note() or ""),
        "num_generated_sequences": int(n_gen),
        "sequence_length": int(seq_len),
        "training_sequence_count": int(training_tokens.shape[0]),
        "training_total_slots": int(training_metrics["total_slots"]),
        "generated_total_slots": int(generated_metrics["total_slots"]),
        "knot_slot_pct_training": float(training_metrics["knot_slot_pct"]),
        "knot_slot_pct_generated": float(generated_metrics["knot_slot_pct"]),
        "no_knot_slot_pct_training": float(training_metrics["no_knot_slot_pct"]),
        "no_knot_slot_pct_generated": float(generated_metrics["no_knot_slot_pct"]),
        "avg_knot_cluster_length_training": _safe_float(training_metrics["knot_run_lengths"]["mean"]),
        "avg_knot_cluster_length_generated": _safe_float(generated_metrics["knot_run_lengths"]["mean"]),
        "avg_no_knot_run_length_training": _safe_float(training_metrics["no_knot_run_lengths"]["mean"]),
        "avg_no_knot_run_length_generated": _safe_float(generated_metrics["no_knot_run_lengths"]["mean"]),
        "js_divergence_all_tokens": float(js_all),
        "js_divergence_nonzero_tokens": float(js_nonzero),
        "generated_nonzero_token_coverage_pct": float(coverage_pct),
        "decoded_bad_knot_row_rate_training_pct": bad_rate_training,
        "decoded_bad_knot_row_rate_generated_pct": bad_rate_generated,
    }
