import numpy as np
import warnings
from .config import BoardConfig
from .mesh import BoardMesh
from .knot_system import KnotSystem as KnotSystemType
from .array_backend import to_numpy

class GrowthSimulator:
    @staticmethod
    def run(p: BoardConfig, mesh: BoardMesh, k: 'KnotSystemType'):
        xp = getattr(k, 'xp', np)
        needs_normal_accum = bool(p.calc_fibers)
        base_grid = xp.asarray(mesh.X)
        empty_field = xp.asarray(np.empty((0,), dtype=np.float32))
        layers = {}
        if needs_normal_accum:
            mesh_accum = {
                'grid_nx': xp.zeros_like(base_grid),
                'grid_ny': xp.zeros_like(base_grid),
                'grid_nz': xp.zeros_like(base_grid),
                'g_best': xp.full_like(base_grid, xp.inf),
            }
        else:
            mesh_accum = {
                'grid_nx': empty_field,
                'grid_ny': empty_field,
                'grid_nz': empty_field,
                'g_best': empty_field,
            }
        
        X_base = base_grid
        Y_base = xp.asarray(mesh.Y)
        Z_base = xp.asarray(mesh.Z)
        x_coords_base = xp.asarray(
            mesh.x_coords if mesh.x_coords is not None else to_numpy(X_base[0, :, 0])
        )
        y_coords_base = xp.asarray(
            mesh.y_coords if mesh.y_coords is not None else to_numpy(Y_base[:, 0, 0])
        )
        z_coords_base = xp.asarray(
            mesh.z_coords if mesh.z_coords is not None else to_numpy(Z_base[0, 0, :])
        )

        if not k.splines:
            return layers, mesh_accum

        # Pre-calculate influence (for knot field ttt)
        s_end = k.splines[-1]
        Ri0 = float(s_end(0))
        Ro = xp.asarray(s_end(to_numpy(mesh.TH)))

        flags = {
            'get_knots': True,
            'include_knot_dev': p.include_knot_dev,
            'meshdensity': p.meshdensity_effective(),
            'calc_fibers_a0_method': p.calc_fibers_a0_method,
            'dead_knots': p.dead_knots
        }
        
        _, info = k.calculate_influence(X_base, Y_base, Z_base, Ro, Ri0, flags)
        tt = info.get('K', xp.zeros_like(X_base))

        def _reduce_k_field(field):
            if field is None:
                return None
            if getattr(field, "ndim", 0) == 4:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message="All-NaN slice encountered", category=RuntimeWarning)
                    return xp.nanmin(field, axis=3)
            return field

        ttt = _reduce_k_field(tt)
        ttt_live = _reduce_k_field(info.get('K_live'))
        ttt_dead = _reduce_k_field(info.get('K_dead'))

        layers['ttt'] = ttt
        layers['ttt_live'] = ttt_live
        layers['ttt_dead'] = ttt_dead
        # Reuse this full knot-influence payload in FiberSolver to avoid recomputing
        # an equivalent get_knots=True influence call.
        layers['knot_influence_info'] = info
        layers['surfaces'] = []
        layers['pith_surface'] = None
        layers['contours'] = []
        layers['contours_masked'] = []
        layers['contours_masked_live'] = []
        layers['contours_unmasked'] = []
        layers['contours_mid_masked'] = []
        layers['contours_mid_masked_live'] = []
        layers['contours_mid_unmasked'] = []
        layers['growth_layer_fields'] = []
        layers['growth_layer_indices'] = []
        total_layers = len(k.splines)

        # Loop over splines (Growth Layers)
        board = mesh.board_coords
        contour_axes_board = None
        ttt_cpu = None
        ttt_live_cpu = None
        if p.board_or_log == 0 and ttt is not None:
            ttt_cpu = to_numpy(ttt)
        if p.board_or_log == 0 and ttt_live is not None:
            ttt_live_cpu = to_numpy(ttt_live)
        if p.board_or_log == 0 and p.display_contours:
            contour_axes_board = (
                to_numpy(x_coords_base),
                to_numpy(y_coords_base),
                to_numpy(z_coords_base),
            )
        log_visual_stride = max(1, int(getattr(p, "log_layer_stride", 5)))

        def _emit_layer_visual(layer_idx: int) -> bool:
            if p.board_or_log != 1:
                return True
            is_last = (layer_idx == (total_layers - 1))
            if is_last:
                return True
            offset_from_last = (total_layers - 1) - layer_idx
            return bool((offset_from_last % log_visual_stride) == 0)

        # In log mode, only compute growth fields for layers that can actually be
        # visualized (last, and every n-th layer counting inward from the outside),
        # plus the first layer so the optional pith proxy remains available.
        # When fibers are requested, compute every layer so the nearest growth-ring
        # normal field used by the fiber solver covers the full log volume.
        if p.board_or_log == 1:
            if p.calc_fibers:
                layer_indices = list(range(total_layers))
            else:
                layer_indices = sorted({0, *[idx for idx in range(total_layers) if _emit_layer_visual(idx)]})
        else:
            layer_indices = list(range(total_layers))

        def _accumulate_normals_from_layer(g_layer, x_coords, y_coords, z_coords, grad_xp) -> None:
            if not p.calc_fibers:
                return
            gy, gx, gz = GrowthSimulator._compute_gradients(
                g_layer, x_coords, y_coords, z_coords, grad_xp
            )
            closer = grad_xp.abs(g_layer) < grad_xp.abs(mesh_accum['g_best'])
            mesh_accum['g_best'][closer] = g_layer[closer]
            mesh_accum['grid_nx'][closer] = gx[closer]
            mesh_accum['grid_ny'][closer] = gy[closer]
            mesh_accum['grid_nz'][closer] = gz[closer]

        def _finalize_layer_visuals(
            layer_idx: int,
            g_layer,
            x_coords,
            y_coords,
            z_coords,
        ) -> None:
            g_surface_visual = g_layer
            emit_layer_visual = _emit_layer_visual(layer_idx)

            layers['growth_layer_fields'].append(
                np.asarray(to_numpy(g_surface_visual), dtype=np.float32)
            )
            layers['growth_layer_indices'].append(int(layer_idx))

            if p.display_contours and emit_layer_visual:
                if contour_axes_board is None:
                    x_vals = to_numpy(x_coords)
                    y_vals = to_numpy(y_coords)
                    z_vals = to_numpy(z_coords)
                else:
                    x_vals, y_vals, z_vals = contour_axes_board

                x_vals = np.asarray(x_vals, dtype=np.float64).reshape(-1)
                y_vals = np.asarray(y_vals, dtype=np.float64).reshape(-1)
                z_vals = np.asarray(z_vals, dtype=np.float64).reshape(-1)
                contour_board = board
                if p.board_or_log != 0:
                    contour_board = {
                        'x': [float(x_vals[0]), float(x_vals[-1])],
                        'y': [float(y_vals[0]), float(y_vals[-1])],
                        'z': [float(z_vals[0]), float(z_vals[-1])],
                    }

                # Copy once, then slice on CPU to reduce GPU sync/copy overhead and
                # preserve the same contour behavior as the previous matplotlib path.
                g_cpu = to_numpy(g_layer)

                g_faces_unmasked = (
                    g_cpu[:, 0, :],
                    g_cpu[:, -1, :],
                    g_cpu[0, :, :],
                    g_cpu[-1, :, :],
                    g_cpu[:, :, 0],
                    g_cpu[:, :, -1],
                )
                contour_lines_unmasked = GrowthSimulator._extract_contours(
                    x_vals, y_vals, z_vals, g_faces_unmasked, contour_board
                )
                layers['contours_unmasked'].extend(contour_lines_unmasked)
                contour_lines_mid_unmasked = GrowthSimulator._extract_midplane_contours(
                    x_vals, y_vals, z_vals, g_cpu, contour_board
                )
                layers['contours_mid_unmasked'].extend(contour_lines_mid_unmasked)

                if p.board_or_log == 0:
                    if ttt_cpu is not None:
                        g_masked_cpu = g_cpu.copy()
                        g_masked_cpu[ttt_cpu < p.knot_inside_limit] = np.nan
                    else:
                        g_masked_cpu = g_cpu
                    g_faces_masked = (
                        g_masked_cpu[:, 0, :],
                        g_masked_cpu[:, -1, :],
                        g_masked_cpu[0, :, :],
                        g_masked_cpu[-1, :, :],
                        g_masked_cpu[:, :, 0],
                        g_masked_cpu[:, :, -1],
                    )
                    contour_lines_masked = GrowthSimulator._extract_contours(
                        x_vals, y_vals, z_vals, g_faces_masked, contour_board
                    )
                    layers['contours_masked'].extend(contour_lines_masked)
                    contour_lines_mid_masked = GrowthSimulator._extract_midplane_contours(
                        x_vals, y_vals, z_vals, g_masked_cpu, contour_board
                    )
                    layers['contours_mid_masked'].extend(contour_lines_mid_masked)

                    if ttt_live_cpu is not None:
                        g_masked_live_cpu = g_cpu.copy()
                        g_masked_live_cpu[ttt_live_cpu < p.knot_inside_limit] = np.nan
                    else:
                        # Fallback: preserve previous masked behavior when live split is unavailable.
                        g_masked_live_cpu = g_masked_cpu
                    g_faces_masked_live = (
                        g_masked_live_cpu[:, 0, :],
                        g_masked_live_cpu[:, -1, :],
                        g_masked_live_cpu[0, :, :],
                        g_masked_live_cpu[-1, :, :],
                        g_masked_live_cpu[:, :, 0],
                        g_masked_live_cpu[:, :, -1],
                    )
                    contour_lines_masked_live = GrowthSimulator._extract_contours(
                        x_vals, y_vals, z_vals, g_faces_masked_live, contour_board
                    )
                    layers['contours_masked_live'].extend(contour_lines_masked_live)
                    contour_lines_mid_masked_live = GrowthSimulator._extract_midplane_contours(
                        x_vals, y_vals, z_vals, g_masked_live_cpu, contour_board
                    )
                    layers['contours_mid_masked_live'].extend(contour_lines_mid_masked_live)

                    if p.display_rings_inside_knots:
                        layers['contours'].extend(contour_lines_unmasked)
                    else:
                        layers['contours'].extend(contour_lines_masked)
                else:
                    layers['contours'].extend(contour_lines_unmasked)

            # Isosurface extraction.
            # - Full 3D rings are exported only when requested (display_rings).
            # - The first layer (pith proxy) is always exported for live "Display Pith".
            needs_ring_surface = bool(p.display_rings)
            needs_pith_surface = (layer_idx == 0)
            emit_ring_surface = bool(needs_ring_surface and emit_layer_visual)

            if emit_ring_surface or needs_pith_surface:
                surface = GrowthSimulator._extract_isosurface(
                    g_surface_visual,
                    x_coords,
                    y_coords,
                    z_coords,
                )
                if surface is not None:
                    surface['layer_index'] = int(layer_idx)
                    if emit_ring_surface:
                        layers['surfaces'].append(surface)
                    if needs_pith_surface:
                        layers['pith_surface'] = surface

        # Dead-knot mode: enforce nested non-intersection directly on g-fields
        # (layer i must stay inside layer i+1 => g_i >= g_{i+1} pointwise).
        # Apply this in both board and log modes so live-knot bumps do not
        # protrude through later dead-knot growth layers.
        enforce_dead_nesting = bool(getattr(p, "dead_knots", False))
        stored_layers = []
        pending_idx = None
        pending_g = None
        pending_coords = None

        for i in layer_indices:
            s = k.splines[i]
            Ri0 = float(s(0))
            
            # Use the base mesh in both board and log modes.
            # In log mode, BoardMesh already expands XY bounds to cover the full
            # crook+taper-deformed log envelope with extra margin.
            X, Y, Z = X_base, Y_base, Z_base
            x_coords, y_coords, z_coords = x_coords_base, y_coords_base, z_coords_base
            TH_eval = to_numpy(mesh.TH)

            Ro = xp.asarray(s(TH_eval))
            
            flags_iter = flags.copy()
            flags_iter['get_knots'] = False

            g, _ = k.calculate_influence(X, Y, Z, Ro, Ri0, flags_iter)
            
            # Reduce over knots dimension
            if g.ndim == 4:
                g = xp.min(g, axis=3)
            
            if enforce_dead_nesting:
                stored_layers.append((int(i), g, x_coords, y_coords, z_coords))
            else:
                # In non-dead mode, normals come from raw g directly.
                _accumulate_normals_from_layer(g, x_coords, y_coords, z_coords, xp)
                # Non-dead mode keeps the previous streaming behavior.
                if pending_g is not None and pending_coords is not None and pending_idx is not None:
                    px, py, pz = pending_coords
                    _finalize_layer_visuals(
                        pending_idx,
                        pending_g,
                        px,
                        py,
                        pz,
                    )
                pending_idx = int(i)
                pending_g = g
                pending_coords = (x_coords, y_coords, z_coords)
        
        if enforce_dead_nesting:
            n_layers = len(stored_layers)
            corrected_fields = [None] * n_layers
            if n_layers > 0:
                corrected_fields[-1] = stored_layers[-1][1]
                for ridx in range(n_layers - 2, -1, -1):
                    g_curr = stored_layers[ridx][1]
                    g_outer = corrected_fields[ridx + 1]
                    corrected_fields[ridx] = xp.maximum(g_curr, g_outer)

            for idx, layer_payload in enumerate(stored_layers):
                layer_idx, _, px, py, pz = layer_payload
                corrected_g = corrected_fields[idx]
                # In dead-knot mode, normals are based on corrected nested fields.
                _accumulate_normals_from_layer(corrected_g, px, py, pz, xp)
                _finalize_layer_visuals(
                    layer_idx,
                    corrected_g,
                    px,
                    py,
                    pz,
                )
            layers['last_g'] = corrected_fields[-1] if n_layers > 0 else None
        else:
            # Flush final layer.
            if pending_g is not None and pending_coords is not None and pending_idx is not None:
                px, py, pz = pending_coords
                _finalize_layer_visuals(
                    pending_idx,
                    pending_g,
                    px,
                    py,
                    pz,
                )
            layers['last_g'] = pending_g
        return layers, mesh_accum

    @staticmethod
    def _extract_isosurface(g, x_coords, y_coords, z_coords):
        try:
            from skimage.measure import marching_cubes
            g_cpu = to_numpy(g)
            verts, faces, normals, _ = marching_cubes(g_cpu, 0)

            # Scale from index space to world coordinates.
            # verts are in (i, j, k) index space.
            x_coords_cpu = to_numpy(x_coords)
            y_coords_cpu = to_numpy(y_coords)
            z_coords_cpu = to_numpy(z_coords)

            # marching_cubes returns (row, col, depth) = (y_idx, x_idx, z_idx)
            scaled_verts = np.zeros_like(verts)
            scaled_verts[:, 0] = np.interp(
                verts[:, 0], np.arange(len(y_coords_cpu)), y_coords_cpu
            )  # Y
            scaled_verts[:, 1] = np.interp(
                verts[:, 1], np.arange(len(x_coords_cpu)), x_coords_cpu
            )  # X
            scaled_verts[:, 2] = np.interp(
                verts[:, 2], np.arange(len(z_coords_cpu)), z_coords_cpu
            )  # Z

            # Reorder to (X, Y, Z) = (Width, Thickness, Length)
            world_verts = np.column_stack([
                scaled_verts[:, 1],  # X (Width)
                scaled_verts[:, 0],  # Y (Thickness)
                scaled_verts[:, 2],  # Z (Length)
            ])

            return {
                'vertices': world_verts,
                'faces': faces,
                'normals': normals
            }
        except ImportError:
            print("skimage not installed, skipping isosurface")
        except Exception:
            pass
        return None

    @staticmethod
    def _compute_gradients(g, x_coords, y_coords, z_coords, xp):
        """Compute gradients on a separable (possibly nonuniform) grid."""
        gy = GrowthSimulator._axis_gradient(g, y_coords, axis=0, xp=xp)
        gx = GrowthSimulator._axis_gradient(g, x_coords, axis=1, xp=xp)
        gz = GrowthSimulator._axis_gradient(g, z_coords, axis=2, xp=xp)
        return gy, gx, gz

    @staticmethod
    def _axis_gradient(values, coords, axis: int, xp):
        try:
            coords_arr = xp.asarray(coords, dtype=values.dtype).reshape(-1)
        except Exception:
            # Handles mixed NumPy/CuPy inputs when one backend cannot consume
            # the other's array type (e.g. np.asarray(cupy_array)).
            coords_arr = xp.asarray(to_numpy(coords), dtype=values.dtype).reshape(-1)
        n = int(values.shape[axis])
        grad = xp.zeros_like(values)
        if n <= 1:
            return grad

        def slc(idx):
            s = [slice(None)] * values.ndim
            s[axis] = idx
            return tuple(s)

        eps = values.dtype.type(1e-12) if hasattr(values.dtype, 'type') else 1e-12

        # Forward/backward at boundaries.
        d0 = coords_arr[1] - coords_arr[0]
        grad[slc(0)] = (values[slc(1)] - values[slc(0)]) / (d0 + eps)
        d1 = coords_arr[-1] - coords_arr[-2]
        grad[slc(-1)] = (values[slc(-1)] - values[slc(-2)]) / (d1 + eps)

        if n == 2:
            return grad

        # Centered differences in the interior.
        s_mid = slc(slice(1, n - 1))
        s_prev = slc(slice(0, n - 2))
        s_next = slc(slice(2, n))
        denom = coords_arr[2:] - coords_arr[:-2]
        shape = [1] * values.ndim
        shape[axis] = n - 2
        grad[s_mid] = (values[s_next] - values[s_prev]) / (denom.reshape(shape) + eps)
        return grad

    @staticmethod
    def _extract_contours(x_vals, y_vals, z_vals, g_faces, board):
        """Extract contour lines at g=0 on the 6 board faces.
        
        Like MATLAB's contourslice(X, Y, Z, g, board.x, board.y, board.z, [0 0])
        """
        contour_lines = []
        gx0, gx1, gy0, gy1, gz0, gz1 = g_faces
        face_specs = [
            # (slice_data_2d, axis1_coords, axis2_coords, fixed_axis, fixed_val, transpose)
            # x-faces: slice g[:, ix, :] -> 2D array (ny, nz), axes are (y, z)
            (gx0, y_vals, z_vals, 'x', board['x'][0], True),
            (gx1, y_vals, z_vals, 'x', board['x'][1], True),
            # y-faces: slice g[iy, :, :] -> 2D array (nx, nz), axes are (x, z) 
            (gy0, x_vals, z_vals, 'y', board['y'][0], True),
            (gy1, x_vals, z_vals, 'y', board['y'][1], True),
            # z-faces: slice g[:, :, iz] -> 2D array (ny, nx), axes are (x, y)
            # No transpose needed here because the matrix is already (len(y), len(x)).
            (gz0, x_vals, y_vals, 'z', board['z'][0], False),
            (gz1, x_vals, y_vals, 'z', board['z'][1], False),
        ]

        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()

        try:
            for slice_2d, a1, a2, fixed_axis, fixed_val, transpose in face_specs:
                if slice_2d is None:
                    continue
                if np.all(np.isnan(slice_2d)):
                    continue
                # Skip faces where g=0 cannot exist.
                finite = slice_2d[np.isfinite(slice_2d)]
                if finite.size == 0:
                    continue
                if not (np.min(finite) <= 0.0 <= np.max(finite)):
                    continue

                contour_matrix = np.asarray(slice_2d.T if transpose else slice_2d, dtype=np.float64)
                segments = []
                try:
                    ax.clear()
                    cs = ax.contour(a1, a2, contour_matrix, levels=[0])
                    if hasattr(cs, 'allsegs') and cs.allsegs:
                        segments = cs.allsegs[0]
                    elif hasattr(cs, 'collections'):
                        for collection in cs.collections:
                            for path in collection.get_paths():
                                segments.append(path.vertices)
                except Exception:
                    segments = []

                if not segments:
                    continue

                for vertices in segments:
                    if vertices is None or len(vertices) < 2:
                        continue
                    if fixed_axis == 'x':
                        points_3d = np.column_stack([
                            np.full((len(vertices),), fixed_val),
                            vertices[:, 0],
                            vertices[:, 1],
                        ])
                    elif fixed_axis == 'y':
                        points_3d = np.column_stack([
                            vertices[:, 0],
                            np.full((len(vertices),), fixed_val),
                            vertices[:, 1],
                        ])
                    else:
                        points_3d = np.column_stack([
                            vertices[:, 0],
                            vertices[:, 1],
                            np.full((len(vertices),), fixed_val),
                        ])
                    contour_lines.append(points_3d.tolist())
        finally:
            if fig is not None:
                plt.close(fig)
        
        return contour_lines

    @staticmethod
    def _extract_midplane_contours(x_vals, y_vals, z_vals, g_volume, board):
        """Extract contour lines at g=0 on the mid XZ plane (fixed Y = board midpoint)."""
        contour_lines = []
        g_cpu = np.asarray(g_volume, dtype=np.float64)
        if g_cpu.ndim != 3:
            return contour_lines

        x_vals = np.asarray(x_vals, dtype=np.float64).reshape(-1)
        y_vals = np.asarray(y_vals, dtype=np.float64).reshape(-1)
        z_vals = np.asarray(z_vals, dtype=np.float64).reshape(-1)
        if x_vals.size < 2 or y_vals.size < 2 or z_vals.size < 2:
            return contour_lines
        if g_cpu.shape != (y_vals.size, x_vals.size, z_vals.size):
            return contour_lines

        y_mid = 0.5 * (float(board['y'][0]) + float(board['y'][1]))

        # Interpolate g on the exact midpoint plane.
        if y_mid <= y_vals[0]:
            g_mid = g_cpu[0, :, :]
        elif y_mid >= y_vals[-1]:
            g_mid = g_cpu[-1, :, :]
        else:
            idx_hi = int(np.searchsorted(y_vals, y_mid, side='right'))
            idx_hi = min(max(idx_hi, 1), y_vals.size - 1)
            idx_lo = idx_hi - 1
            y0 = float(y_vals[idx_lo])
            y1 = float(y_vals[idx_hi])
            denom = max(1e-12, y1 - y0)
            alpha = np.clip((y_mid - y0) / denom, 0.0, 1.0)
            g_mid = (1.0 - alpha) * g_cpu[idx_lo, :, :] + alpha * g_cpu[idx_hi, :, :]

        if np.all(np.isnan(g_mid)):
            return contour_lines
        finite = g_mid[np.isfinite(g_mid)]
        if finite.size == 0:
            return contour_lines
        if not (np.min(finite) <= 0.0 <= np.max(finite)):
            return contour_lines

        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        try:
            # g_mid is (nx, nz), contour expects (len(z), len(x)).
            contour_matrix = np.asarray(g_mid.T, dtype=np.float64)
            segments = []
            try:
                ax.clear()
                cs = ax.contour(x_vals, z_vals, contour_matrix, levels=[0])
                if hasattr(cs, 'allsegs') and cs.allsegs:
                    segments = cs.allsegs[0]
                elif hasattr(cs, 'collections'):
                    for collection in cs.collections:
                        for path in collection.get_paths():
                            segments.append(path.vertices)
            except Exception:
                segments = []

            for vertices in segments:
                if vertices is None or len(vertices) < 2:
                    continue
                points_3d = np.column_stack([
                    vertices[:, 0],  # X
                    np.full((len(vertices),), y_mid),  # fixed Y midpoint
                    vertices[:, 1],  # Z
                ])
                contour_lines.append(points_3d.tolist())
        finally:
            if fig is not None:
                plt.close(fig)

        return contour_lines
