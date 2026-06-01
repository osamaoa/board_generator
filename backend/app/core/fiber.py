import numpy as np
from .config import BoardConfig
from .mesh import BoardMesh
from .knot_system import KnotSystem as KnotSystemType
from .array_backend import to_numpy


class FiberSolver:
    @staticmethod
    def _stack_segments(starts, vecs, scale, y_keep_max=None, z_keep_max=None, clip_bounds=None):
        starts = np.asarray(starts).reshape(-1, 3)
        vecs = np.asarray(vecs).reshape(-1, 3)
        valid = np.isfinite(starts).all(axis=1) & np.isfinite(vecs).all(axis=1)
        if not np.any(valid):
            return []
        starts = starts[valid]
        vecs = vecs[valid]
        ends = starts + scale * vecs

        if y_keep_max is not None:
            y_lim = float(y_keep_max)
            keep = (starts[:, 1] <= y_lim) & (ends[:, 1] <= y_lim)
            starts = starts[keep]
            ends = ends[keep]
        if z_keep_max is not None:
            z_lim = float(z_keep_max)
            keep = (starts[:, 2] <= z_lim) & (ends[:, 2] <= z_lim)
            starts = starts[keep]
            ends = ends[keep]
        if starts.size == 0:
            return []

        if clip_bounds is not None:
            (x_min, x_max), (y_min, y_max), (z_min, z_max) = clip_bounds
            starts[:, 0] = np.clip(starts[:, 0], x_min, x_max)
            starts[:, 1] = np.clip(starts[:, 1], y_min, y_max)
            starts[:, 2] = np.clip(starts[:, 2], z_min, z_max)
            ends[:, 0] = np.clip(ends[:, 0], x_min, x_max)
            ends[:, 1] = np.clip(ends[:, 1], y_min, y_max)
            ends[:, 2] = np.clip(ends[:, 2], z_min, z_max)
        return np.stack([starts, ends], axis=1).tolist()

    @staticmethod
    def solve(
        p: BoardConfig,
        mesh: BoardMesh,
        k: 'KnotSystemType',
        mesh_accum: dict,
        precomputed_info: dict | None = None,
    ):
        """Port of FiberSolver.solve from MATLAB."""
        xp = getattr(k, 'xp', np)
        X = xp.asarray(mesh.X)
        Y = xp.asarray(mesh.Y)
        Z = xp.asarray(mesh.Z)
        if not k.splines or k.n_knots == 0:
            return np.zeros_like(mesh.X), np.zeros_like(mesh.X), np.ones_like(mesh.X)

        th0 = k.th0  # (1, 1, 1, n_knots)

        # 1) Flow field around knots.
        info = precomputed_info if isinstance(precomputed_info, dict) else None
        if (
            info is None
            or info.get('K') is None
            or info.get('fiber_cross') is None
            or info.get('fiber_long') is None
        ):
            s = k.splines[-1]
            Ri0 = float(s(0))
            th_eval = to_numpy(mesh.TH)
            Ro = xp.asarray(s(th_eval))
            flags = {
                'get_knots': True,
                'include_knot_dev': p.include_knot_dev,
                'meshdensity': p.meshdensity_effective(),
                'calc_fibers_a0_method': p.calc_fibers_a0_method,
                'dead_knots': p.dead_knots
            }
            _, info = k.calculate_influence(X, Y, Z, Ro, Ri0, flags)

        tt = info.get('K')
        fiber_cross = info.get('fiber_cross')
        fiber_long = info.get('fiber_long')
        knot_dir_radial = info.get('knot_dir_radial')
        if knot_dir_radial is None:
            knot_dir_radial = info.get('tx_kf')
        knot_dir_longitudinal = info.get('knot_dir_longitudinal')
        if knot_dir_longitudinal is None:
            knot_dir_longitudinal = info.get('ty_kf')
        knot_dir_transverse = info.get('knot_dir_transverse')
        if knot_dir_transverse is None:
            knot_dir_transverse = info.get('tz_kf')
        length_offset = info.get('length_offset')
        if length_offset is None:
            length_offset = info.get('yp')
        tt_dead = info.get('K_dead')

        if tt is None or fiber_cross is None or fiber_long is None:
            return np.zeros_like(X), np.zeros_like(X), np.ones_like(X)

        # 2) Rotate from knot frame to board coordinates.
        txx = fiber_cross * np.sin(th0)
        tyy = fiber_cross * np.cos(th0)
        tzz = fiber_long

        # 3) Growth-ring normals from nearest layer.
        drdxx = xp.asarray(mesh_accum['grid_nx']).copy()
        drdyy = xp.asarray(mesh_accum['grid_ny']).copy()
        drdzz = xp.asarray(mesh_accum['grid_nz']).copy()
        mag_n = xp.sqrt(drdxx**2 + drdyy**2 + drdzz**2)
        mag_n[mag_n == 0] = 1.0
        drdxx = (drdxx / mag_n)[..., np.newaxis]
        drdyy = (drdyy / mag_n)[..., np.newaxis]
        drdzz = (drdzz / mag_n)[..., np.newaxis]

        # 4) Tangency correction.
        delta_x = -(drdxx * txx + drdyy * tyy + drdzz * tzz) / (
            drdxx * xp.cos(th0) - drdyy * xp.sin(th0) + 1e-12
        )
        txx = txx + delta_x * xp.cos(th0)
        tyy = tyy - delta_x * xp.sin(th0)

        # 5) Normalize per knot.
        mag = xp.sqrt(txx**2 + tyy**2 + tzz**2)
        txx = txx / (mag + 1e-12)
        tyy = tyy / (mag + 1e-12)
        tzz = tzz / (mag + 1e-12)

        # 6) Multi-knot selection.
        if k.n_knots > 1 and tzz.ndim == 4:
            fallback_idx = xp.argmax(xp.abs(tzz), axis=3)
            selection_rule = str(getattr(p, "multi_knot_fiber_selection_rule", "weighted_deviation") or "").strip().lower()

            def local_knot_relevance_weights():
                x_board = X[..., np.newaxis]
                y_board = Y[..., np.newaxis]
                z_board = Z[..., np.newaxis]
                x_transverse = x_board + k.crook_x(z_board)
                y_transverse = y_board + k.crook_y(z_board)
                cos_th0 = xp.cos(th0)
                sin_th0 = xp.sin(th0)
                radial_coord = x_transverse * cos_th0 - y_transverse * sin_th0
                tangential_coord = x_transverse * sin_th0 + y_transverse * cos_th0

                if length_offset is not None and getattr(length_offset, "ndim", 0) == 4:
                    longitudinal_offset = length_offset
                else:
                    knot_axis_z = k.c1 * radial_coord**2 + k.c2 * radial_coord + k.z0
                    longitudinal_offset = z_board - knot_axis_z

                Lx = k.a1 * radial_coord**4 + k.a2 * radial_coord**3 + k.a3 * radial_coord**2 + k.a4 * radial_coord
                if bool(getattr(p, "dead_knots", False)):
                    Lx_RL = k.a1 * k.RL**4 + k.a2 * k.RL**3 + k.a3 * k.RL**2 + k.a4 * k.RL
                    mask_range = (radial_coord >= k.RL) & (radial_coord <= k.RD)
                    taper_t = xp.clip((radial_coord - k.RL) / (k.RD - k.RL + 1e-12), 0.0, 1.0)
                    Lx_tapered = Lx_RL * (1.0 - taper_t**2)
                    Lx = xp.where(mask_range, Lx_tapered, Lx)
                    Lx = xp.where(radial_coord > k.RD, 0.0, Lx)
                Lx = xp.maximum(Lx, 0.0)

                obstacle_radius = 0.5 * Lx
                local_distance = xp.sqrt(longitudinal_offset**2 + tangential_coord**2 / (k.kp**2 + 1e-12))
                rho = local_distance / (obstacle_radius + 1e-12)
                sigma = max(1e-6, float(getattr(p, "multi_knot_fiber_selection_sigma", 1.5)))
                weights = xp.exp(-((xp.maximum(rho - 1.0, 0.0) / sigma) ** 2))
                valid_weight = xp.isfinite(weights) & (obstacle_radius > 1e-9) & (radial_coord >= 0.0)
                if tt is not None and getattr(tt, "ndim", 0) == 4:
                    valid_weight = valid_weight & xp.isfinite(tt)
                return xp.where(valid_weight, weights, 0.0)

            if selection_rule in {"original", "longitudinal", "abs_longitudinal"}:
                idx = fallback_idx
            else:
                # Prefer the knot whose local obstacle is influential and whose
                # candidate produces the strongest transverse deviation. Fall back
                # to the original longitudinal rule away from all knots.
                weights = local_knot_relevance_weights()
                transverse_deviation = xp.sqrt(txx**2 + tyy**2)
                selection_score = weights * transverse_deviation
                selection_score = xp.where(xp.isfinite(selection_score), selection_score, 0.0)
                influence_idx = xp.argmax(selection_score, axis=3)
                max_weight = xp.max(weights, axis=3)
                min_weight = max(0.0, float(getattr(p, "multi_knot_fiber_selection_min_weight", 1e-4)))
                idx = xp.where(max_weight > min_weight, influence_idx, fallback_idx)
            txx = xp.take_along_axis(txx, idx[..., xp.newaxis], axis=3)[..., 0]
            tyy = xp.take_along_axis(tyy, idx[..., xp.newaxis], axis=3)[..., 0]
            tzz = xp.take_along_axis(tzz, idx[..., xp.newaxis], axis=3)[..., 0]
        else:
            txx = txx[..., 0] if txx.ndim == 4 else txx
            tyy = tyy[..., 0] if tyy.ndim == 4 else tyy
            tzz = tzz[..., 0] if tzz.ndim == 4 else tzz

        # 7) Default direction in invalid regions.
        nan_mask = xp.isnan(txx) | xp.isnan(tyy) | xp.isnan(tzz)
        txx[nan_mask] = 0.0
        tyy[nan_mask] = 0.0
        tzz[nan_mask] = 1.0

        # 8) Optional override inside knots with knot-axis field.
        if (
            p.knot_fiber_field_override
            and knot_dir_radial is not None
            and knot_dir_longitudinal is not None
            and knot_dir_transverse is not None
            and tt.ndim == 4
        ):
            kf_x = knot_dir_radial * xp.cos(th0) + knot_dir_transverse * xp.sin(th0)
            kf_y = -knot_dir_radial * xp.sin(th0) + knot_dir_transverse * xp.cos(th0)
            kf_z = knot_dir_longitudinal
            reverse_above = bool(getattr(p, 'knot_fiber_reverse_above_axis', False))
            disable_dead_override = bool(getattr(p, 'knot_fiber_disable_dead_override', True))
            has_dead_field = (
                disable_dead_override
                and tt_dead is not None
                and getattr(tt_dead, 'ndim', 0) == 4
            )
            for kn in range(k.n_knots):
                mask = tt[:, :, :, kn] < p.knot_inside_limit
                if has_dead_field:
                    mask = mask & (~xp.isfinite(tt_dead[:, :, :, kn]))
                if reverse_above and length_offset is not None and getattr(length_offset, 'ndim', 0) == 4:
                    # Reverse the override on the positive-longitudinal side of the knot axis.
                    sign = xp.where(length_offset[:, :, :, kn] > 0.0, -1.0, 1.0)
                    txx[mask] = (kf_x[:, :, :, kn] * sign)[mask]
                    tyy[mask] = (kf_y[:, :, :, kn] * sign)[mask]
                    tzz[mask] = (kf_z[:, :, :, kn] * sign)[mask]
                else:
                    txx[mask] = kf_x[:, :, :, kn][mask]
                    tyy[mask] = kf_y[:, :, :, kn][mask]
                    tzz[mask] = kf_z[:, :, :, kn][mask]

        return to_numpy(txx), to_numpy(tyy), to_numpy(tzz)

    @staticmethod
    def build_plot_data_all(p: BoardConfig, mesh: BoardMesh, txx, tyy, tzz):
        """Create plot-ready fiber segments for all quiver display modes."""
        if p.board_or_log != 0:
            return {
                'surface_quiver3d': [],
                'volume_quiver3d': [],
                'quiver2d': [],
                'quiver2d_clean': [],
                'quiver2d_rand': [],
            }

        # Face slices: x-min/x-max/y-min/y-max.
        slices = [
            ('x', 0),
            ('x', -1),
            ('y', 0),
            ('y', -1),
        ]
        surface_segments = []
        quiver2d_segments_clean = []
        quiver2d_segments_rand = []
        surface_scale = FiberSolver._arrow_scale(mesh, 1.5)
        volume_scale = FiberSolver._arrow_scale(mesh, 0.45)
        bx0, bx1 = mesh.board_coords['x']
        by0, by1 = mesh.board_coords['y']
        bz0, bz1 = mesh.board_coords['z']
        clip_bounds_2d = ((bx0, bx1), (by0, by1), (bz0, bz1))
        # Keep full face coverage and rely on endpoint clipping to keep arrows inside bounds.
        y_keep_max = None
        z_keep_max = None

        for axis, idx in slices:
            if axis == 'x':
                xs = mesh.X[:, idx, :]
                ys = mesh.Y[:, idx, :]
                zs = mesh.Z[:, idx, :]
                ux = txx[:, idx, :].copy()
                uy = tyy[:, idx, :].copy()
                uz = tzz[:, idx, :].copy()
            else:
                xs = mesh.X[idx, :, :]
                ys = mesh.Y[idx, :, :]
                zs = mesh.Z[idx, :, :]
                ux = txx[idx, :, :].copy()
                uy = tyy[idx, :, :].copy()
                uz = tzz[idx, :, :].copy()

            # Quiver 3D on board surfaces.
            FiberSolver._append_quiver_segments(
                surface_segments, xs, ys, zs, ux, uy, uz, surface_scale, y_keep_max, z_keep_max, clip_bounds=None
            )

            # Quiver 2D (project onto the board faces).
            if axis == 'x':
                qx = np.zeros_like(ux)
                qy = uy.copy()
                qz = uz.copy()
                qx_rand = qx.copy()
                qy_rand = qy.copy()
                qz_rand = qz.copy()
                qy_rand, qz_rand = FiberSolver._apply_noise(
                    ux, qy_rand, qz_rand, p.out_of_plane_threshold, p.snr
                )
            else:
                qx = ux.copy()
                qy = np.zeros_like(uy)
                qz = uz.copy()
                qx_rand = qx.copy()
                qy_rand = qy.copy()
                qz_rand = qz.copy()
                qx_rand, qz_rand = FiberSolver._apply_noise(
                    uy, qx_rand, qz_rand, p.out_of_plane_threshold, p.snr
                )
            FiberSolver._append_quiver_segments(
                quiver2d_segments_clean, xs, ys, zs, qx, qy, qz, surface_scale, y_keep_max, z_keep_max, clip_bounds_2d
            )
            FiberSolver._append_quiver_segments(
                quiver2d_segments_rand, xs, ys, zs, qx_rand, qy_rand, qz_rand, surface_scale, y_keep_max, z_keep_max, clip_bounds_2d
            )

        # Quiver 3D through the full board volume.
        volume_segments = []
        FiberSolver._append_volume_quiver_segments(
            volume_segments, mesh.X, mesh.Y, mesh.Z, txx, tyy, tzz, volume_scale, y_keep_max, z_keep_max, clip_bounds=None
        )

        quiver2d_segments = quiver2d_segments_rand if bool(p.rand_fibers) else quiver2d_segments_clean
        return {
            'surface_quiver3d': surface_segments,
            'volume_quiver3d': volume_segments,
            'quiver2d': quiver2d_segments,
            'quiver2d_clean': quiver2d_segments_clean,
            'quiver2d_rand': quiver2d_segments_rand,
        }

    @staticmethod
    def build_surface_normal_quiver3d(mesh: BoardMesh, nx, ny, nz):
        """Create 3D quiver segments on the board faces from surface normal fields."""
        if mesh is None:
            return []

        slices = [
            ('x', 0),
            ('x', -1),
            ('y', 0),
            ('y', -1),
        ]
        segments = []
        surface_scale = FiberSolver._arrow_scale(mesh, 0.75)

        for axis, idx in slices:
            if axis == 'x':
                xs = mesh.X[:, idx, :]
                ys = mesh.Y[:, idx, :]
                zs = mesh.Z[:, idx, :]
                ux = np.asarray(nx[:, idx, :]).copy()
                uy = np.asarray(ny[:, idx, :]).copy()
                uz = np.asarray(nz[:, idx, :]).copy()
            else:
                xs = mesh.X[idx, :, :]
                ys = mesh.Y[idx, :, :]
                zs = mesh.Z[idx, :, :]
                ux = np.asarray(nx[idx, :, :]).copy()
                uy = np.asarray(ny[idx, :, :]).copy()
                uz = np.asarray(nz[idx, :, :]).copy()

            FiberSolver._append_quiver_segments(
                segments,
                xs,
                ys,
                zs,
                ux,
                uy,
                uz,
                surface_scale,
                y_keep_max=None,
                z_keep_max=None,
                clip_bounds=None,
            )

        return segments

    @staticmethod
    def _arrow_scale(mesh: BoardMesh, multiplier: float = 0.75):
        if getattr(mesh, 'x_coords', None) is not None and len(mesh.x_coords) > 1:
            dx = float(np.median(np.abs(np.diff(np.asarray(mesh.x_coords, dtype=float)))))
        else:
            dx = abs(mesh.X[0, 1, 0] - mesh.X[0, 0, 0]) if mesh.X.shape[1] > 1 else 1.0

        if getattr(mesh, 'y_coords', None) is not None and len(mesh.y_coords) > 1:
            dy = float(np.median(np.abs(np.diff(np.asarray(mesh.y_coords, dtype=float)))))
        else:
            dy = abs(mesh.Y[1, 0, 0] - mesh.Y[0, 0, 0]) if mesh.Y.shape[0] > 1 else 1.0

        if getattr(mesh, 'z_coords', None) is not None and len(mesh.z_coords) > 1:
            dz = float(np.median(np.abs(np.diff(np.asarray(mesh.z_coords, dtype=float)))))
        else:
            dz = abs(mesh.Z[0, 0, 1] - mesh.Z[0, 0, 0]) if mesh.Z.shape[2] > 1 else 1.0

        # Keep arrows compact for dense quiver rendering.
        return float(max(dx, dy, dz) * multiplier)

    @staticmethod
    def _append_quiver_segments(
        segments,
        xs,
        ys,
        zs,
        ux,
        uy,
        uz,
        scale,
        y_keep_max=None,
        z_keep_max=None,
        clip_bounds=None,
    ):
        starts = np.stack([xs, ys, zs], axis=-1)
        vecs = np.stack([ux, uy, uz], axis=-1)
        segments.extend(
            FiberSolver._stack_segments(
                starts,
                vecs,
                scale,
                y_keep_max=y_keep_max,
                z_keep_max=z_keep_max,
                clip_bounds=clip_bounds,
            )
        )

    @staticmethod
    def _append_volume_quiver_segments(
        segments,
        xs,
        ys,
        zs,
        ux,
        uy,
        uz,
        scale,
        y_keep_max=None,
        z_keep_max=None,
        clip_bounds=None,
    ):
        starts = np.stack([xs, ys, zs], axis=-1)
        vecs = np.stack([ux, uy, uz], axis=-1)
        segments.extend(
            FiberSolver._stack_segments(
                starts,
                vecs,
                scale,
                y_keep_max=y_keep_max,
                z_keep_max=z_keep_max,
                clip_bounds=clip_bounds,
            )
        )

    @staticmethod
    def _apply_noise(v1, v2, v3, thresh, snr_db):
        mask = np.abs(v1) > thresh
        if not np.any(mask):
            return v2, v3

        def awgn_like(signal):
            out = signal.copy()
            signal_power = float(np.mean(signal[mask] ** 2))
            if not np.isfinite(signal_power) or signal_power <= 1e-12:
                return out
            noise_power = signal_power / (10 ** (snr_db / 10.0))
            sigma = np.sqrt(max(noise_power, 1e-12))
            noise = np.random.normal(0.0, sigma, size=signal.shape)
            out[mask] = out[mask] + noise[mask]
            return out

        return awgn_like(v2), awgn_like(v3)
