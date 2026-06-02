# CLI Manual

The CLI entrypoint is:

```bash
./board_cli.py
```

It uses the same backend simulation core as the UI. The public command groups are:

- `boards`: batch board generation and photorealistic face regeneration
- `knots`: knot-sequence data preparation, training, sampling, and evaluation
- `diffusion`: photorealistic diffusion training

Inspect the live command surface with:

```bash
./board_cli.py --help
./board_cli.py boards --help
./board_cli.py knots --help
./board_cli.py diffusion --help
```

## Batch Board Exports

Start from your own JSON config and override the run-specific output path:

```bash
./board_cli.py boards generate \
  --config-json path/to/boards_config.json \
  --output-dir /tmp/boards_out \
  --num-boards 50 \
  --outputs rings,fibers,middle,photorealistic
```

`boards generate` supports explicit board extents or sampled placement from board dimensions. In sampled-placement mode the generator retries placements until it accepts the requested number of valid boards or reaches `--max-attempts`.

The output selector accepts:

- `rings`: side-face ring maps in `rings_1..4`
- `fibers`: side-face fiber maps in `fiber_1..4`
- `middle`: middle ring surface in `rings_5`
- `top_bottom`: top and bottom ring maps in `rings_top` and `rings_bottom`
- `photorealistic`: diffusion-generated side faces in `photorealistic_1..4`
- `all`

Every accepted board also writes metadata under `metadata/`. A run-level `manifest.json` records the generated filenames, parameters, requested outputs, and accepted/rejected attempt counts.

### Configuration

JSON configs place batch settings under `boards_generate` and model settings under `config`. CLI arguments override matching JSON values.

High-impact batch settings include:

- output count, attempt limit, filename start, and output folders
- mesh spacing and board extents or dimensions
- seed and GPU controls
- random or manual crook/taper controls
- generated or manual knot controls
- contour and fiber map rendering settings
- photorealistic inference settings

Range syntax is supported for selected map and photorealistic settings. A value such as `0.1,0.4` samples one value for each batch chunk. `--photorealistic-img2img-strength` also accepts a longer discrete list.

### Photorealistic Batch Modes

Photorealistic generation can use ring and fiber maps or ring-only conditioning:

```bash
./board_cli.py boards generate \
  --config-json path/to/boards_config.json \
  --output-dir /tmp/ring_only_faces \
  --outputs photorealistic \
  --photorealistic-use-rings-only true
```

Knot-map conditioning is derived from fiber maps and is enabled with:

```bash
--photorealistic-include-knot-maps true
```

Ring-only conditioning and knot-map conditioning are mutually exclusive.

### Multi-GPU Generation

When `use_gpu=true` and multiple CUDA devices are visible, board generation can shard work:

```bash
./board_cli.py boards generate \
  --config-json path/to/boards_config.json \
  --gpu-workers 2
```

Set `--gpu-workers 0` or omit it to let the generator choose from visible GPUs.

### Regenerate Photorealistic Faces

Use the regeneration command when `rings_1..4` already exist and you only want `photorealistic_1..4`:

```bash
./board_cli.py boards regenerate-photorealistic \
  --config-json path/to/boards_config.json
```

Or point directly at an existing dataset:

```bash
./board_cli.py boards regenerate-photorealistic \
  --data-root /tmp/boards_out \
  --stems 00001,00002,00003 \
  --photorealistic-batch-size 8 \
  --overwrite true
```

Regeneration needs `fiber_1..4` unless `--photorealistic-use-rings-only true` is used.

## Knot-Sequence Model

Prepare knot-model training data:

```bash
./board_cli.py knots prepare-data \
  --logs-dir ./.old_knot_generator/logs_data \
  --output-mat-path ./knot_model_checkpoint/training_data_new_2025.mat
```

Train the LSTM sampler:

```bash
./board_cli.py knots train \
  --training-mat-path ./knot_model_checkpoint/training_data_new_2025.mat \
  --output-checkpoint-path ./knot_model_checkpoint/knot_sequence_model.pt
```

Sample a new token sequence:

```bash
./board_cli.py knots sample \
  --length 400 \
  --top-p 0.8 \
  --checkpoint-path ./knot_model_checkpoint/knot_sequence_model.pt
```

Evaluate sampled sequences against training data:

```bash
./board_cli.py knots evaluate \
  --training-mat-path ./knot_model_checkpoint/training_data_new_2025.mat \
  --checkpoint-path ./knot_model_checkpoint/knot_sequence_model.pt \
  --output-dir ./runs/knot_eval
```

Run `./board_cli.py knots <command> --help` for the full parameter set and output paths.

## Photorealistic Diffusion Training

Diffusion training uses ring/fiber conditioning maps by default and can be configured for ring-only training:

```bash
./board_cli.py diffusion train --help
```

Install the photorealistic dependency set first:

```bash
pip install -r backend/requirements-photorealistic.txt
```

Training checkpoints and runtime checkpoint paths are separate from batch export settings. Keep the selected model assets and a CUDA-capable PyTorch runtime available to the backend before running UI or CLI photorealistic inference.
