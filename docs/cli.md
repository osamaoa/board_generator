# CLI Manual

The CLI entrypoint is:

```bash
./board_cli.py
```

It uses the same backend simulation core as the UI. The public command groups are:

- `boards`: batch generation, photorealistic face regeneration, and Blender export
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

### Long Boards And Knot Context

Board dimensions remain ordinary configuration values. For example, set
`board_length` to `435` to generate a three-segment board. Knot-sequence context
is optional and permits knots whose origins lie outside the visible board to
intersect an end face:

```bash
./board_cli.py boards generate \
  --config-json path/to/boards_config.json \
  --knot-seq-context-enabled true \
  --knot-seq-context-before-mm 100 \
  --knot-seq-context-after-mm 100
```

The visible board coordinates are unchanged. Only the sampled knot sequence is
extended before and after it, and the chosen layout is written to board
metadata. Disable the option to retain the legacy sequence extent.

Empirical knot-axis calibration is also optional. A calibration profile stores
paired axial displacements at 50 and 100 mm, allowing the generator to sample
`c1` and `c2` together rather than independently:

```bash
./board_cli.py boards generate \
  --config-json path/to/boards_config.json \
  --knot-axis-calibration-enabled true \
  --knot-axis-calibration-path /path/to/knot_axis_profile.json \
  --knot-axis-calibration-mix 0.8
```

The mix is the probability of using an empirical observation; the remainder
uses the knot library. Calibration takes precedence over the legacy fixed
`c1`/`c2` override while enabled.

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

For a face longer than the diffusion model's native square field of view,
optional long-face translation divides each conditioning map into overlapping
square windows and feather-blends the generated windows:

```bash
./board_cli.py boards generate \
  --config-json path/to/boards_config.json \
  --photorealistic-long-face-enabled true \
  --photorealistic-tile-length-mm 145 \
  --photorealistic-tile-overlap-px 96
```

The side-map height is derived from `board_length / tile_length_mm`; top and
bottom maps remain square. Disable this option to use the original single-image
translation path.

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
  --photorealistic-long-face-enabled true \
  --photorealistic-tile-overlap-px 96 \
  --overwrite true
```

Regeneration needs `fiber_1..4` unless `--photorealistic-use-rings-only true` is used.

### Export A Board To Blender

Turn an existing generated board into a self-contained, render-ready Blender file:

```bash
./board_cli.py boards export-blender \
  --data-root /tmp/boards_out \
  --stem 00001
```

The command uses `photorealistic_1..4` when all four images exist. Otherwise,
`--surface-source auto` falls back to `ring_color_1..4`. Select a source
explicitly with `--surface-source photorealistic` or
`--surface-source ring-color`.

By default it writes:

```text
<data-root>/blender/00001.blend
<data-root>/blender/00001_preview.png
```

The Blender scene preserves the generator's physical dimensions and face
order: surface 1 is +Y, surface 2 is -Y, surface 3 is +X, and surface 4 is -X.
Each surface gets its own UV-mapped material with image-derived grain bump,
micro-roughness, soft milled edge bevels, and studio lighting. Source images
are packed into the `.blend` file by default.

The top and bottom end cross-sections are not photorealistic generator outputs.
They receive a clearly named procedural placeholder material. The supplied
orthographic three-quarter camera is perpendicular to the board length, making
both end faces edge-on and concealing them in the default render.

Useful options include:

```bash
./board_cli.py boards export-blender \
  --data-root /tmp/boards_out \
  --stem 00001 \
  --output-path /tmp/exports/my_board.blend \
  --render-engine cycles \
  --samples 128 \
  --render-preview true \
  --pack-images true
```

Blender is discovered from `BLENDER_EXECUTABLE`, `blender` on `PATH`, or the
newest Windows Blender installation visible under `/mnt/c/Program Files` when
the CLI runs in WSL. Override discovery with `--blender-executable`.

To export every accepted board automatically after `boards generate`, add a
top-level block to the generation JSON:

```json
{
  "config": {},
  "boards_generate": {
    "outputs": "rings,fibers,photorealistic"
  },
  "blender_export": {
    "enabled": true,
    "surface_source": "photorealistic",
    "render_preview": true,
    "render_engine": "eevee",
    "samples": 64,
    "pack_images": true
  }
}
```

This post-generation step runs only after all requested photorealistic faces
have been written. `output_dir` and `blender_executable` are optional; the same
defaults as `boards export-blender` are used when they are omitted.

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
