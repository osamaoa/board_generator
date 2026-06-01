# UI Manual

The web UI is the interactive path through Board Generator. It configures a board or log, runs the shared Python simulation backend, renders the result in 3D, and exposes exports for downstream image and MATLAB workflows.

## Run

Start the FastAPI backend on port `8100` and the Vite frontend on port `5175` as described in `docs/getting_started.md`.

## Workflow

1. Choose board mode or log mode and set geometry in the control panel.
2. Configure knot generation or enter manual knots.
3. Select the growth, contour, knot, pith, and fiber calculations needed for the run.
4. Click `Generate Board`.
5. Inspect the 3D result and enable overlays as needed.
6. Export data, image maps, or photorealistic surfaces from the Export tab.

Changing simulation parameters requires a new generation run. Viewer toggles only change the current display.

## Control Panel

### Geometry

Geometry controls define the model domain:

- board extents or board dimensions
- mesh spacing along width, thickness, and length
- board placement behavior inside the generated log
- crook and taper randomization or manual crook/taper values

Dimensions mode samples valid board placement attempts. A generated board can be rejected when the board footprint leaves the log after the selected log geometry is applied.

### Knots

The knot controls choose between generated knot sequences and manual knot parameters. The generated path uses the trained knot-sequence assets available to the backend. Manual mode exposes knot angle, axial position, dead/live radii, axis shape, and growth-bump parameters.

### Fibers

Fiber controls select whether fiber orientations are calculated and which quiver view is shown in the UI. Export-oriented fiber settings such as face-map noise and out-of-plane handling are kept with the simulation config so the image exports and photorealistic conditioning use the same run.

### Simulation

Simulation controls include CPU/GPU selection, reproducibility seed settings, and the calculation switches used before packaging the 3D result.

### View

The view controls show or hide:

- the board/log shell
- contours and growth layers
- modeled pith geometry
- knots and knot-sequence slots
- fiber vectors and surface overlays
- generated photorealistic face overlays after photorealistic export

Board opacity is useful when checking interior knots, growth layers, and fiber vectors.

### Export

The Export tab keeps the outputs used by the public package:

- MATLAB export ZIP containing `board_export_*.mat` and `visualize_exported_board.m`
- MATLAB-style ring and fiber image ZIPs, with an optional middle ring surface
- photorealistic side-face export and ZIP download

The `.mat` export contains the data needed by the MATLAB visualization script for the current run, including geometry, contours, optional 3D growth layers, knots, knot-sequence slot segments, available photorealistic faces, and calculated fiber fields.

Photorealistic export uses generated ring maps and, unless rings-only conditioning is selected, generated fiber maps. The model can be preloaded from the Export tab before generating faces. Generated faces can be shown immediately as overlays in the 3D viewer.

## Viewer Conventions

The simulation model uses:

- `X`: board width
- `Y`: board thickness
- `Z`: board length

The Three.js viewer presents length vertically for inspection. The MATLAB export and MATLAB visualization script keep the simulation coordinate convention.

The projection and grid buttons in the viewer only affect camera inspection. They do not alter exported data.

## Troubleshooting

For slow runs, increase mesh spacing or disable calculations you do not need for that export. For a missing photorealistic export button state, check the backend capability status and model checkpoint setup. For empty export data, regenerate after enabling the calculation that produces that field; log-mode MAT export computes the log fiber field even though the UI does not render it.
