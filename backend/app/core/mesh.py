import numpy as np
from .config import BoardConfig
from .knot_system import KnotSystem as KnotSystemType
from .array_backend import to_numpy

class BoardMesh:
    LOG_DOMAIN_MARGIN_MM = 20.0
    LOG_DOMAIN_THETA_SAMPLES = 1024
    LOG_DOMAIN_Z_STEP_MM = 5.0
    LOG_DOMAIN_Z_MIN_SAMPLES = 65
    LOG_DOMAIN_Z_MAX_SAMPLES = 2001

    def __init__(self, config: BoardConfig, knot_system: 'KnotSystemType'):
        self.config = config
        self.board_coords = {
            'x': [0.0, 0.0],
            'y': [0.0, 0.0],
            'z': [0.0, 0.0]
        }
        
        self.X = None
        self.Y = None
        self.Z = None
        self.TH = None
        self.R = None
        self.initial_Ri0 = 0.0
        self.x_coords = None
        self.y_coords = None
        self.z_coords = None
        
        self.determine_position(config, knot_system)
        self.generate_grid(config, knot_system)

    def determine_position(self, p: BoardConfig, k: 'KnotSystemType'):
        # Use explicit extents from config
        x0, x1 = p.x_extent()
        y0, y1 = p.y_extent()
        z0, z1 = p.z_extent()
        self.board_coords['x'] = [x0, x1]
        self.board_coords['y'] = [y0, y1]
        self.board_coords['z'] = [z0, z1]

    @staticmethod
    def _sample_count_from_step(
        length_mm: float,
        step_mm: float,
        min_samples: int,
        max_samples: int,
    ) -> int:
        safe_len = max(0.0, float(length_mm))
        safe_step = max(1e-6, float(step_mm))
        raw = int(np.ceil(safe_len / safe_step)) + 1
        return int(np.clip(raw, min_samples, max_samples))

    @staticmethod
    def estimate_log_xy_bounds(
        p: BoardConfig,
        k: 'KnotSystemType',
        z0: float,
        z1: float,
        *,
        splines=None,
        margin_mm: float = LOG_DOMAIN_MARGIN_MM,
    ):
        z_min = float(min(z0, z1))
        z_max = float(max(z0, z1))
        z_len = max(1e-6, z_max - z_min)
        mesh_step_z = getattr(p, "mesh_size_z_mm", BoardMesh.LOG_DOMAIN_Z_STEP_MM)
        try:
            mesh_step_z = float(mesh_step_z)
        except Exception:
            mesh_step_z = BoardMesh.LOG_DOMAIN_Z_STEP_MM
        if not np.isfinite(mesh_step_z) or mesh_step_z <= 0.0:
            mesh_step_z = BoardMesh.LOG_DOMAIN_Z_STEP_MM
        z_step = max(0.5, min(BoardMesh.LOG_DOMAIN_Z_STEP_MM, mesh_step_z))

        z_count = BoardMesh._sample_count_from_step(
            z_len,
            z_step,
            BoardMesh.LOG_DOMAIN_Z_MIN_SAMPLES,
            BoardMesh.LOG_DOMAIN_Z_MAX_SAMPLES,
        )
        z_samples = np.linspace(z_min, z_max, z_count, dtype=np.float64)

        def _safe_eval_vector(func, default: float = 0.0) -> np.ndarray:
            try:
                vals = np.asarray(to_numpy(func(z_samples)), dtype=np.float64).reshape(-1)
            except Exception:
                vals = np.full((z_samples.shape[0],), float(default), dtype=np.float64)
            if vals.shape[0] != z_samples.shape[0]:
                vals = np.resize(vals, z_samples.shape[0])
            bad = ~np.isfinite(vals)
            if np.any(bad):
                vals = vals.copy()
                vals[bad] = float(default)
            return vals

        crook_x_vals = _safe_eval_vector(getattr(k, "crook_x", lambda z: 0.0), default=0.0)
        crook_y_vals = _safe_eval_vector(getattr(k, "crook_y", lambda z: 0.0), default=0.0)
        taper_vals = _safe_eval_vector(getattr(k, "taper", lambda z: 0.0), default=0.0)

        center_x_min = float(np.min(-crook_x_vals))
        center_x_max = float(np.max(-crook_x_vals))
        center_y_min = float(np.min(-crook_y_vals))
        center_y_max = float(np.max(-crook_y_vals))
        taper_min = float(np.min(taper_vals))

        resolved_splines = list(splines) if splines is not None else list(getattr(k, "splines", []) or [])
        theta = np.linspace(
            -np.pi,
            np.pi,
            BoardMesh.LOG_DOMAIN_THETA_SAMPLES,
            endpoint=False,
            dtype=np.float64,
        )

        radius_max = 0.0
        for spline in resolved_splines:
            try:
                radii = np.asarray(spline(theta), dtype=np.float64).reshape(-1)
            except Exception:
                continue
            finite = radii[np.isfinite(radii)]
            if finite.size == 0:
                continue
            candidate = float(np.max(finite) - taper_min)
            if np.isfinite(candidate):
                radius_max = max(radius_max, candidate)

        if not np.isfinite(radius_max) or radius_max <= 0.0:
            fallback = 0.0
            for spline in resolved_splines:
                try:
                    v = float(spline(0.0))
                except Exception:
                    continue
                if np.isfinite(v):
                    fallback = max(fallback, abs(v))
            radius_max = max(100.0, fallback)

        pad = max(0.0, float(margin_mm))
        x_min = center_x_min - radius_max - pad
        x_max = center_x_max + radius_max + pad
        y_min = center_y_min - radius_max - pad
        y_max = center_y_max + radius_max + pad

        if not np.isfinite(x_min) or not np.isfinite(x_max) or x_max <= x_min:
            span = max(200.0, 2.0 * (radius_max + pad))
            x_min, x_max = -0.5 * span, 0.5 * span
        if not np.isfinite(y_min) or not np.isfinite(y_max) or y_max <= y_min:
            span = max(200.0, 2.0 * (radius_max + pad))
            y_min, y_max = -0.5 * span, 0.5 * span

        return float(x_min), float(x_max), float(y_min), float(y_max)

    def generate_grid(self, p: BoardConfig, k: 'KnotSystemType'):
        if k.splines:
            self.initial_Ri0 = float(k.splines[-1](0))
        else:
            self.initial_Ri0 = 100.0
        
        if p.board_or_log == 0:
            x0, x1 = self.board_coords['x']
            y0, y1 = self.board_coords['y']
            z0, z1 = self.board_coords['z']
            nx, ny, nz = p.mesh_counts_for_lengths(abs(x1 - x0), abs(y1 - y0), abs(z1 - z0))
            # Board mode: grid spans board dimensions
            x = np.linspace(x0, x1, nx)
            y = np.linspace(y0, y1, ny)
            z = np.linspace(z0, z1, nz)
        else:
            # Log mode: domain spans full deformed envelope (crook+taper) with margin.
            z0, z1 = self.board_coords['z']
            x0, x1, y0, y1 = self.estimate_log_xy_bounds(
                p,
                k,
                z0,
                z1,
                splines=getattr(k, "splines", None),
                margin_mm=self.LOG_DOMAIN_MARGIN_MM,
            )
            nx, ny, nz = p.mesh_counts_for_lengths(abs(x1 - x0), abs(y1 - y0), abs(z1 - z0))
            x = np.linspace(x0, x1, nx)
            y = np.linspace(y0, y1, ny)
            z = np.linspace(z0, z1, nz)

        self.x_coords = np.asarray(x, dtype=float)
        self.y_coords = np.asarray(y, dtype=float)
        self.z_coords = np.asarray(z, dtype=float)

        # MATLAB: [X, Y, Z] = meshgrid(x, y, z)
        # numpy meshgrid with 'xy' indexing matches MATLAB behavior
        self.X, self.Y, self.Z = np.meshgrid(x, y, z, indexing='xy')
        
        # MATLAB: [TH, R, ~] = cart2pol(X, Y, Z)
        # cart2pol(x, y) => (atan2(y, x), hypot(x, y))
        self.R = np.hypot(self.X, self.Y)
        self.TH = np.arctan2(self.Y, self.X)
