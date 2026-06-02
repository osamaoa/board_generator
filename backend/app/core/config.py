from pydantic import BaseModel, Field
from typing import List, Optional
import math


class InputKnot(BaseModel):
    # Geometry defaults used by both the UI and CLI simulation paths.
    th0_deg: float = 0.0
    L100: float = 20.0
    z0: float = 60.0
    c1: float = -1.458e-3
    c2: float = 0.5608
    k: float = 0.99
    kp: float = 0.95
    Abump: float = 0.668
    Aexp: float = 2.184
    Bbump: float = 2.0
    RL: float = 100.0
    RD: float = 100.0
    a1: float = -1e-7
    a2: float = 3e-5
    a3: float = -4e-3
    a4: float = 0.6

class BoardConfig(BaseModel):
    # Board dimensions (mm) - legacy fields kept for compatibility
    board_width: float = 145.0
    board_thickness: float = 45.0
    board_length: float = 145.0
    # Board mode helper: when enabled, sample board extents randomly inside
    # the generated log using board_width/board_thickness/board_length.
    randomize_board_extents_from_dimensions: bool = False

    # Board extents (mm) used by the web app UI
    board_x_min: float = -72.5
    board_x_max: float = 72.5
    board_y_min: float = -22.5
    board_y_max: float = 22.5
    board_z_min: float = 0.0
    board_z_max: float = 145.0

    # General Settings
    board_or_log: int = 0  # 0 for board; 1 for log
    # Preferred element size (mm) along each axis.
    mesh_size_x_mm: Optional[float] = 2.0
    mesh_size_y_mm: Optional[float] = 2.0
    mesh_size_z_mm: Optional[float] = 2.0
    use_seed: bool = False
    simulation_seed: int = 100
    use_gpu: bool = True

    # Manual knot sequence input
    use_input_knots: bool = False
    input_knot_count: int = 1
    input_knots: List[InputKnot] = Field(default_factory=lambda: [InputKnot()])
    # Stochastic crook/taper geometry model.
    # Crook uses p sinusoidal components:
    # D_x(z) = sum_i sin(theta_i) * a_i * sin(2*pi*(z + z0_i)/L_i)
    # D_y(z) = sum_i cos(theta_i) * a_i * sin(2*pi*(z + z0_i)/L_i)
    # with L_i = 2^(5-i) * 1000 mm and a_i ~ U(0, random_crook_scale_max * L_i/320).
    randomize_crook_taper: bool = True
    crook_component_count: int = 8
    crook_shift_max_mm: float = 8000.0
    random_crook_scale_max: float = 1.0
    random_crook_theta_min_deg: float = 0.0
    random_crook_theta_max_deg: float = 360.0
    # Per-component random amplitude maxima (mm) guided by legacy MATLAB script.
    random_crook_amplitude_max: List[float] = Field(
        default_factory=lambda: [50.0, 25.0, 12.5, 5.0, 2.5, 1.25, 0.625, 0.3125]
    )
    # Additional random crook component orders appended after default orders 1..8.
    # Example: [9, 10] adds two higher-frequency terms.
    random_crook_extra_orders: List[int] = Field(default_factory=list)
    random_taper_max: float = 1.0 / 160.0

    # Manual deterministic crook/taper parameters used when randomize_crook_taper=False.
    # Arrays are resized to match crook_component_count (trim or zero-pad).
    # Defaults represent a typical realization using means/representative values
    # from the random model used in the legacy MATLAB script.
    manual_crook_amplitudes: List[float] = Field(
        default_factory=lambda: [25.0, 12.5, 6.25, 2.5, 1.25, 0.625, 0.3125, 0.15625]
    )
    manual_crook_shifts_mm: List[float] = Field(
        default_factory=lambda: [4000.0] * 8
    )
    manual_crook_thetas_deg: List[float] = Field(
        default_factory=lambda: [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0]
    )
    # Manual per-component orders. When empty, defaults to 1..crook_component_count.
    manual_crook_orders: List[int] = Field(
        default_factory=lambda: [1, 2, 3, 4, 5, 6, 7, 8]
    )

    # Legacy polynomial coefficients kept for backward compatibility.
    manual_crook_x_coeff: float = 0.0
    manual_crook_y_coeff: float = 0.0
    manual_taper_coeff: float = 1.0 / 160.0

    # Knot Settings
    include_knot_dev: bool = True
    dead_knots: bool = True
    knot_inside_limit: float = -20.0
    # Soft clamping parameters used by the knot/growth distortion model.
    soft_clamp_alpha: float = 1.0
    soft_clamp_pmin: float = 2.0
    
    # Knot filtering
    L100_min: float = 5.0
    L100_max: float = 70.0
    # For sequence-generated knots, enforce a minimum dead-zone span (RD - RL).
    # When a generated knot has RD - RL below this value, RL is shifted to RD - value.
    knot_generator_min_rd_minus_rl_mm: float = 30.0

    # Knot-sequence generator (new random sequence per board/log sample)
    knot_sequence_top_k: int = 0
    knot_sequence_top_p: float = 0.8
    knot_sequence_min_tokens: int = 400
    knot_sequence_extra_tokens: int = 200
    knot_sequence_checkpoint_path: str = ""
    knot_sequence_training_data_path: str = ""
    knot_sequence_allow_fallback: bool = True
    # Reject sampled knot sequences when generated knot volumes intersect.
    knot_sequence_reject_intersections: bool = True
    knot_sequence_intersection_max_attempts: int = 64
    knot_dictionary_jitter: float = 0.0
    # Override knot c1/c2 using:
    #   c1 = -1.458e-3
    #   Ax100 ~ U(32.7, 55.3)
    #   c2 = 9.7e-3 * Ax100 + 0.1725
    # Applied to both generated and manual knots.
    knot_sequence_override_c1_c2: bool = True

    # Visualization Flags
    display_rings: bool = False
    display_knots: bool = True
    display_normal_vectors_surface: bool = False
    display_rings_inside_knots: bool = True
    display_pith: bool = False
    # Log-mode ring-surface decimation: render every Nth growth layer
    # while always keeping the first and last layer.
    log_layer_stride: int = 5
    display_knot_axes: bool = False
    display_contours: bool = True
    display_surface_mesh: bool = False
    
    transparent: bool = True
    board_opacity: float = 0.8

    # Fiber Calculation Settings
    calc_fibers: bool = True
    calc_fibers_a0_method: int = 1  # 1: exact (slow), 2: approx (fast)
    knot_fiber_field_override: bool = True
    multi_knot_fiber_selection_rule: str = "weighted_deviation"
    multi_knot_fiber_selection_sigma: float = 1.5
    multi_knot_fiber_selection_min_weight: float = 1e-4
    # When enabled, suppress knot-axis fiber override inside dead knot region.
    knot_fiber_disable_dead_override: bool = True
    # When enabled, reverse knot override direction where the longitudinal offset is positive.
    knot_fiber_reverse_above_axis: bool = False
    quiver_or_stream: int = 3  # 0: off, 1: quiver3d surfaces, 2: quiver3d volume, 3: quiver2d

    # Fiber Randomness / Noise
    rand_fibers: bool = False
    out_of_plane_threshold: float = 0.75
    snr: float = 0.9
    blur_segma: float = 0.1

    # Image Saving Settings
    save_rings: bool = False
    save_fibers: bool = False
    imid: int = 1
    rings_dpi: int = 60
    contour_line_width: float = 3.00
    fiber_line_width: float = 2.00

    def apply_overrides(self):
        if self.save_rings:
            self.display_rings = False
            self.display_knots = False
            self.display_rings_inside_knots = False
            self.display_pith = False
            self.display_knot_axes = False
            self.display_contours = True
            self.transparent = False
            self.quiver_or_stream = 3

    @staticmethod
    def _ordered_extent(v0: float, v1: float):
        return (v0, v1) if v0 <= v1 else (v1, v0)

    def x_extent(self):
        return self._ordered_extent(self.board_x_min, self.board_x_max)

    def y_extent(self):
        return self._ordered_extent(self.board_y_min, self.board_y_max)

    def z_extent(self):
        return self._ordered_extent(self.board_z_min, self.board_z_max)

    def board_length_mm(self):
        z0, z1 = self.z_extent()
        return max(z1 - z0, 1.0)

    @staticmethod
    def _axis_count_from_size(
        length_mm: float,
        size_mm: Optional[float],
        default_size_mm: float = 2.0,
    ) -> int:
        if not (isinstance(length_mm, (int, float)) and math.isfinite(length_mm)):
            return 2

        if isinstance(size_mm, (int, float)) and math.isfinite(size_mm) and size_mm > 0:
            resolved_size = float(size_mm)
        else:
            resolved_size = float(default_size_mm)

        if not (math.isfinite(resolved_size) and resolved_size > 0):
            resolved_size = 2.0

        intervals = int(math.ceil((abs(float(length_mm)) / resolved_size) - 1e-9))
        return max(2, intervals + 1)

    def mesh_counts_for_lengths(self, x_length_mm: float, y_length_mm: float, z_length_mm: float):
        nx = self._axis_count_from_size(x_length_mm, self.mesh_size_x_mm, 2.0)
        ny = self._axis_count_from_size(y_length_mm, self.mesh_size_y_mm, 2.0)
        nz = self._axis_count_from_size(z_length_mm, self.mesh_size_z_mm, 2.0)
        return nx, ny, nz

    def mesh_counts(self):
        x0, x1 = self.x_extent()
        y0, y1 = self.y_extent()
        z0, z1 = self.z_extent()
        return self.mesh_counts_for_lengths(abs(x1 - x0), abs(y1 - y0), abs(z1 - z0))

    def meshdensity_effective(self):
        nx, ny, nz = self.mesh_counts()
        return max(nx, ny, nz)

    def resolved_input_knots(self) -> List[InputKnot]:
        count = max(0, int(self.input_knot_count))
        knots = list(self.input_knots[:count])
        while len(knots) < count:
            knots.append(InputKnot())
        return knots
