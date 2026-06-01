# Knot Sequence Model Guide

## 1. Scope

This guide documents the knot-sequence subsystem:

- training data preparation
- LSTM training and sampling
- runtime use inside board generation
- HTML-based model evaluation

Core implementation:

- `backend/app/core/knot_sequence_model.py`
- runtime integration in `backend/app/core/knot_system.py`

## 2. Data Pipeline

## 2.1 Input

Raw knot logs are read from MATLAB files (`knot_data_*.mat`).

## 2.2 Output MAT

`knots prepare-data` builds `training_data_new_2025.mat` with key arrays such as:

- `inps`, `outs` (token sequences)
- `Data_all_or` (dictionary rows with knot parameters)
- token-to-row lookup tables for direct or clustered tokenization

Optional clustering (`cluster_count > 0`) compresses dictionary IDs into token clusters.

## 3. Training Model

`knots train` fits a PyTorch LSTM next-token model.

Important controls:

- hidden size, layers, dropout
- class weighting for token `0` (`no_knot_weight`)
- early stopping (`early_stop_*`)
- optional embedding freeze

Output artifacts:

- checkpoint (`.pt`)
- optional JSON training history

## 4. Runtime Sampling in Board Generation

When manual knots are disabled, each board gets a fresh sampled sequence.

Runtime steps:

1. sample token sequence matching board slot count
2. map tokens to dictionary rows
3. remove bad IDs (`bad_knots`)
4. filter by L100 range
5. derive secondary parameters (`Abump`, `Aexp`, etc.)
6. enforce minimum dead-zone span (`RD-RL`)
7. reject sequence and resample if knot-knot intersections are detected

If checkpoint is unavailable and fallback is allowed, a fallback sampler is used.

## 5. Intersection Rejection

The generated knot sequencies are checked for overlap. If two or more knots overlap, the sequence is rejected and a new sequence is generated.

## 6. Evaluation Report (`knots evaluate`)

`knots evaluate` generates an HTML report comparing generated sequences vs training sequences.

## 6.1 Requested Metrics

- knot-slot percentage
- no-knot-slot percentage
- average knot cluster length (run length of token > 0)
- average continuous no-knot length (run length of token = 0)
- parameter distribution statistics (mean, std, p05, p25, p50, p75, p95)

## 6.2 Additional Diagnostics

- Jensen-Shannon divergence (all tokens and non-zero tokens)
- generated non-zero token coverage vs training vocabulary
- optional decoded bad-knot row rate
- top non-zero token frequency tables

## 6.3 Parameter Statistics Basis

Parameter comparison is computed after decoding non-zero tokens to dictionary rows.
For clustered tokenization, a random member row is selected from each token cluster.

## 7. Key Runtime Controls

- `knot_sequence_top_k`
- `knot_sequence_top_p`
- `knot_sequence_allow_fallback`
- `knot_sequence_reject_intersections`
- `knot_sequence_intersection_max_attempts`
- `knot_generator_min_rd_minus_rl_mm`
- `knot_dictionary_jitter`
- `knot_sequence_override_c1_c2`

## 8. Known Assumptions

- token `0` represents no-knot slots
- knot realism depends on quality of `Data_all_or` and bad-knot filtering
- sequence model is unconditional (no direct conditioning on board geometry)

## 9. CLI Quick Commands

```bash
./board_cli.py knots prepare-data --help
./board_cli.py knots train --help
./board_cli.py knots sample --help
./board_cli.py knots evaluate --help
```
