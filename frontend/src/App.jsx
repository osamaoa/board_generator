import React, { useEffect, useState } from 'react';
import axios from 'axios';
import ControlPanel from './components/ControlPanel';
import Viewer3D from './components/Viewer3D';
import './App.css';

const isDemoMode = ['1', 'true', 'yes', 'on'].includes(
  String(import.meta.env.VITE_BOARD_GENERATOR_DEMO || '').trim().toLowerCase()
);
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || (import.meta.env.DEV ? 'http://localhost:8100' : '');
const defaultPhotorealisticCapability = {
  available: false,
  reason: '',
  device: '',
  cuda_available: null,
  recommended_ddim_steps: 50,
  loaded: false,
};
const photorealisticSurfaceFaceOrder = {
  surface_1: 'z_max',
  surface_2: 'z_min',
  surface_3: 'x_max',
  surface_4: 'x_min',
};

const defaultInputKnot = {
  th0_deg: 0.0,
  L100: 20.0,
  z0: 60.0,
  c1: -1.458e-3,
  c2: 0.5608,
  k: 0.99,
  kp: 0.95,
  Abump: 0.668,
  Aexp: 2.184,
  Bbump: 2.0,
  RL: 100.0,
  RD: 100.0,
  a1: -1e-7,
  a2: 3e-5,
  a3: -4e-3,
  a4: 0.6
};
const defaultCrookComponentCount = 8;
const defaultRandomCrookAmplitudeMax = [50.0, 25.0, 12.5, 5.0, 2.5, 1.25, 0.625, 0.3125];
const defaultManualCrookAmplitudes = [25.0, 12.5, 6.25, 2.5, 1.25, 0.625, 0.3125, 0.15625];
const defaultManualCrookShiftsMm = Array.from({ length: defaultCrookComponentCount }, () => 4000.0);
const defaultManualCrookThetasDeg = [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0];
const defaultManualCrookOrders = [1, 2, 3, 4, 5, 6, 7, 8];

// Default config matching Python backend default
export const defaultConfig = {
  board_width: 145.0,
  board_thickness: 45.0,
  board_length: 145.0,
  randomize_board_extents_from_dimensions: false,
  board_x_min: -72.5,
  board_x_max: 72.5,
  board_y_min: -22.5,
  board_y_max: 22.5,
  board_z_min: 0.0,
  board_z_max: 145.0,
  board_or_log: 0,
  mesh_size_x_mm: 2.0,
  mesh_size_y_mm: 2.0,
  mesh_size_z_mm: 2.0,
  use_seed: false,
  simulation_seed: 100,
  use_gpu: !isDemoMode,
  use_input_knots: false,
  input_knot_count: 1,
  input_knots: [{ ...defaultInputKnot }],
  randomize_crook_taper: true,
  crook_component_count: defaultCrookComponentCount,
  crook_shift_max_mm: 8000.0,
  random_crook_scale_max: 1.0,
  random_crook_amplitude_max: [...defaultRandomCrookAmplitudeMax],
  random_crook_extra_orders: [],
  random_crook_theta_min_deg: 0.0,
  random_crook_theta_max_deg: 360.0,
  random_taper_max: 1.0 / 160.0,
  manual_crook_amplitudes: [...defaultManualCrookAmplitudes],
  manual_crook_shifts_mm: [...defaultManualCrookShiftsMm],
  manual_crook_thetas_deg: [...defaultManualCrookThetasDeg],
  manual_crook_orders: [...defaultManualCrookOrders],
  manual_crook_x_coeff: 0.0,
  manual_crook_y_coeff: 0.0,
  manual_taper_coeff: 1.0 / 160.0,
  include_knot_dev: true,
  dead_knots: true,
  knot_inside_limit: -20.0,
  knot_generator_min_rd_minus_rl_mm: 30.0,
  soft_clamp_alpha: 1.0,
  soft_clamp_pmin: 2.0,
  L100_min: 5.0,
  L100_max: 70.0,
  knot_sequence_top_k: 0,
  knot_sequence_top_p: 0.80,
  knot_dictionary_jitter: 0.0,
  knot_sequence_override_c1_c2: true,
  display_rings: false,
  display_knots: true,
  display_knot_slots: false,
  display_normal_vectors_surface: false,
  display_rings_inside_knots: true,
  display_pith: false,
  log_layer_stride: 5,
  display_knot_axes: false,
  display_contours: true,
  display_surface_mesh: false,
  display_board: true,
  board_opacity: 0.8,
  contour_line_width: 3.0,
  fiber_line_width: 2.0,
  calc_fibers: true,
  calc_fibers_a0_method: 1,
  knot_fiber_field_override: true,
  multi_knot_fiber_selection_rule: 'weighted_deviation',
  knot_fiber_disable_dead_override: true,
  knot_fiber_reverse_above_axis: false,
  quiver_or_stream: 3,
  rand_fibers: false,
  out_of_plane_threshold: 0.75,
  snr: 0.9,
  blur_segma: 0.1,
  export_contour_line_width: 1.0,
  export_surface_blur_sigma: 0.0,
  export_fiber_blur_sigma: 0.0,
  export_fiber_irregularity_strength: 0.35,
  export_ring_irregularity_strength: 0.40,
  export_show_rings_inside_knots: false,
  photorealistic_guidance_scale: 2.0,
  photorealistic_img2img_strength: 0.0,
  photorealistic_include_knot_maps: false,
  photorealistic_use_rings_only: false,
  photorealistic_steps: 50,
  imid: 1,
  save_rings: false,
  save_fibers: false
};

export const normalizeLoadedConfig = (raw) => {
  const candidate = (raw && typeof raw === 'object' && raw.config && typeof raw.config === 'object')
    ? raw.config
    : raw;
  if (!candidate || typeof candidate !== 'object') {
    throw new Error('Invalid configuration payload');
  }

  const next = { ...defaultConfig };
  for (const key of Object.keys(defaultConfig)) {
    if (Object.prototype.hasOwnProperty.call(candidate, key)) {
      next[key] = candidate[key];
    }
  }

  const toNumberOr = (value, fallback) => {
    const num = Number(value);
    return Number.isFinite(num) ? num : fallback;
  };
  const toBoolOr = (value, fallback) => {
    if (typeof value === 'boolean') return value;
    if (typeof value === 'string') {
      const s = value.trim().toLowerCase();
      if (['1', 'true', 'yes', 'on'].includes(s)) return true;
      if (['0', 'false', 'no', 'off'].includes(s)) return false;
    }
    return fallback;
  };
  const defaultValueAt = (defaults, index, fallback = 0) => {
    if (!Array.isArray(defaults) || defaults.length === 0) return fallback;
    if (index < defaults.length) return toNumberOr(defaults[index], fallback);
    return toNumberOr(defaults[defaults.length - 1], fallback);
  };
  const fitCrookArray = (rawArray, defaults, fallback = 0) => {
    const values = Array.isArray(rawArray) ? rawArray : [];
    const out = [];
    for (let i = 0; i < next.crook_component_count; i += 1) {
      const fill = defaultValueAt(defaults, i, fallback);
      out.push(toNumberOr(values[i], fill));
    }
    return out;
  };
  const parsePositiveIntList = (rawValue) => {
    const values = Array.isArray(rawValue)
      ? rawValue
      : (typeof rawValue === 'string' ? rawValue.split(',') : []);
    const out = [];
    for (const token of values) {
      const num = Math.floor(Number(token));
      if (Number.isFinite(num) && num >= 1) {
        out.push(num);
      }
    }
    return out;
  };
  const fitManualOrderArray = (rawArray) => {
    const values = parsePositiveIntList(rawArray);
    const out = [];
    for (let i = 0; i < next.crook_component_count; i += 1) {
      const fallback = i + 1;
      const val = Number.isFinite(values[i]) ? values[i] : fallback;
      out.push(Math.max(1, Math.floor(val)));
    }
    return out;
  };

  next.board_opacity = Math.min(1, Math.max(0, toNumberOr(next.board_opacity, defaultConfig.board_opacity)));
  next.mesh_size_x_mm = Math.max(0.05, toNumberOr(next.mesh_size_x_mm, defaultConfig.mesh_size_x_mm));
  next.mesh_size_y_mm = Math.max(0.05, toNumberOr(next.mesh_size_y_mm, defaultConfig.mesh_size_y_mm));
  next.mesh_size_z_mm = Math.max(0.05, toNumberOr(next.mesh_size_z_mm, defaultConfig.mesh_size_z_mm));
  next.board_width = Math.max(1e-6, toNumberOr(next.board_width, defaultConfig.board_width));
  next.board_thickness = Math.max(1e-6, toNumberOr(next.board_thickness, defaultConfig.board_thickness));
  next.board_length = Math.max(1e-6, toNumberOr(next.board_length, defaultConfig.board_length));
  next.crook_component_count = Math.max(1, Math.floor(toNumberOr(next.crook_component_count, defaultConfig.crook_component_count)));
  next.crook_shift_max_mm = Math.max(0, toNumberOr(next.crook_shift_max_mm, defaultConfig.crook_shift_max_mm));
  next.random_crook_scale_max = Math.max(0, toNumberOr(next.random_crook_scale_max, defaultConfig.random_crook_scale_max));
  next.random_crook_theta_min_deg = toNumberOr(next.random_crook_theta_min_deg, defaultConfig.random_crook_theta_min_deg);
  next.random_crook_theta_max_deg = toNumberOr(next.random_crook_theta_max_deg, defaultConfig.random_crook_theta_max_deg);
  next.random_taper_max = Math.max(0, toNumberOr(next.random_taper_max, defaultConfig.random_taper_max));
  next.manual_taper_coeff = toNumberOr(next.manual_taper_coeff, defaultConfig.manual_taper_coeff);
  next.knot_generator_min_rd_minus_rl_mm = Math.max(
    0,
    toNumberOr(next.knot_generator_min_rd_minus_rl_mm, defaultConfig.knot_generator_min_rd_minus_rl_mm)
  );
  next.random_crook_amplitude_max = fitCrookArray(
    next.random_crook_amplitude_max,
    defaultRandomCrookAmplitudeMax,
    0
  ).map((v) => Math.max(0, v));
  next.random_crook_extra_orders = parsePositiveIntList(next.random_crook_extra_orders);
  next.manual_crook_amplitudes = fitCrookArray(
    next.manual_crook_amplitudes,
    defaultManualCrookAmplitudes,
    0
  ).map((v) => Math.max(0, v));
  next.manual_crook_shifts_mm = fitCrookArray(next.manual_crook_shifts_mm, defaultManualCrookShiftsMm, 0);
  next.manual_crook_thetas_deg = fitCrookArray(next.manual_crook_thetas_deg, defaultManualCrookThetasDeg, 0);
  next.manual_crook_orders = fitManualOrderArray(next.manual_crook_orders);
  if (!['weighted_deviation', 'longitudinal'].includes(next.multi_knot_fiber_selection_rule)) {
    next.multi_knot_fiber_selection_rule = defaultConfig.multi_knot_fiber_selection_rule;
  }
  next.export_contour_line_width = Math.max(1, toNumberOr(next.export_contour_line_width, defaultConfig.export_contour_line_width));
  next.export_fiber_irregularity_strength = Math.min(2, Math.max(0, toNumberOr(next.export_fiber_irregularity_strength, defaultConfig.export_fiber_irregularity_strength)));
  next.export_ring_irregularity_strength = Math.min(2, Math.max(0, toNumberOr(next.export_ring_irregularity_strength, defaultConfig.export_ring_irregularity_strength)));
  next.photorealistic_guidance_scale = Math.max(0, toNumberOr(next.photorealistic_guidance_scale, defaultConfig.photorealistic_guidance_scale));
  next.photorealistic_img2img_strength = Math.min(1, Math.max(0, toNumberOr(next.photorealistic_img2img_strength, defaultConfig.photorealistic_img2img_strength)));
  next.photorealistic_steps = Math.max(1, Math.floor(toNumberOr(next.photorealistic_steps, defaultConfig.photorealistic_steps)));
  next.photorealistic_include_knot_maps = toBoolOr(
    next.photorealistic_include_knot_maps,
    defaultConfig.photorealistic_include_knot_maps
  );
  next.photorealistic_use_rings_only = toBoolOr(
    next.photorealistic_use_rings_only,
    defaultConfig.photorealistic_use_rings_only
  );
  if (next.photorealistic_use_rings_only) {
    next.photorealistic_include_knot_maps = false;
  }
  next.input_knot_count = Math.max(0, Math.floor(toNumberOr(next.input_knot_count, defaultConfig.input_knot_count)));
  next.knot_sequence_top_k = Math.max(0, Math.floor(toNumberOr(next.knot_sequence_top_k, defaultConfig.knot_sequence_top_k)));
  next.knot_sequence_top_p = Math.min(1, Math.max(0, toNumberOr(next.knot_sequence_top_p, defaultConfig.knot_sequence_top_p)));
  next.knot_dictionary_jitter = Math.max(0, toNumberOr(next.knot_dictionary_jitter, defaultConfig.knot_dictionary_jitter));
  next.knot_sequence_override_c1_c2 = toBoolOr(
    next.knot_sequence_override_c1_c2,
    defaultConfig.knot_sequence_override_c1_c2
  );

  const loadedKnots = Array.isArray(next.input_knots) ? next.input_knots : [];
  const normalizedKnots = loadedKnots.map((knot) => ({ ...defaultInputKnot, ...(knot || {}) }));
  while (normalizedKnots.length < next.input_knot_count) {
    normalizedKnots.push({ ...defaultInputKnot });
  }
  next.input_knots = normalizedKnots.slice(0, next.input_knot_count);

  return next;
};

const estimateSimulationDurationMs = (cfg) => {
  const useDimensionMode = !!cfg.randomize_board_extents_from_dimensions;
  const lenX = useDimensionMode
    ? Math.max(1, Number(cfg.board_width) || 1)
    : Math.max(1, Math.abs(Number(cfg.board_x_max) - Number(cfg.board_x_min)));
  const lenY = useDimensionMode
    ? Math.max(1, Number(cfg.board_thickness) || 1)
    : Math.max(1, Math.abs(Number(cfg.board_y_max) - Number(cfg.board_y_min)));
  const lenZ = useDimensionMode
    ? Math.max(1, Number(cfg.board_length) || 1)
    : Math.max(1, Math.abs(Number(cfg.board_z_max) - Number(cfg.board_z_min)));
  const hx = Math.max(0.05, Number(cfg.mesh_size_x_mm) || (145 / 69));
  const hy = Math.max(0.05, Number(cfg.mesh_size_y_mm) || (45 / 69));
  const hz = Math.max(0.05, Number(cfg.mesh_size_z_mm) || (145 / 69));
  const nx = Math.max(2, Math.ceil((lenX / hx) - 1e-9) + 1);
  const ny = Math.max(2, Math.ceil((lenY / hy) - 1e-9) + 1);
  const nz = Math.max(2, Math.ceil((lenZ / hz) - 1e-9) + 1);
  const points = Math.max(1, nx * ny * nz);
  const baseline = 70 * 70 * 70;
  const gpuFactor = cfg.use_gpu ? 1.0 : 1.85;
  let estimate = 1400 + Math.round((points / baseline) * 7000 * gpuFactor);
  if (cfg.display_contours) estimate += 900;
  if (cfg.calc_fibers) estimate += cfg.use_gpu ? 1400 : 2600;
  if (cfg.calc_fibers && Number(cfg.quiver_or_stream) === 2) estimate += cfg.use_gpu ? 900 : 1800;
  if (cfg.display_rings) estimate += 900;
  if (cfg.display_knots) estimate += 700;
  return Math.max(2800, Math.min(45000, Math.round(estimate)));
};

const buildLoadingPlan = (cfg) => {
  const stages = [
    { key: 'prep', label: 'Preparing mesh and knot system', weight: 1.0 },
    { key: 'growth', label: 'Computing growth layers', weight: 4.0 },
    ...(cfg.display_contours ? [{ key: 'contours', label: 'Extracting ring contours', weight: 0.8 }] : []),
    ...(cfg.calc_fibers ? [{ key: 'fibers', label: 'Computing fiber orientations', weight: 1.7 }] : []),
    { key: 'package', label: 'Packaging and transferring data', weight: 1.0 }
  ];
  const totalWeight = stages.reduce((sum, stage) => sum + stage.weight, 0) || 1;
  let cumulative = 0;
  const stageStarts = stages.map((stage) => {
    const start = cumulative;
    cumulative += (stage.weight / totalWeight) * 100;
    return start;
  });
  return {
    stages,
    stageStarts,
    estimateMs: estimateSimulationDurationMs(cfg)
  };
};

const base64ToBlob = (base64Data, mimeType) => {
  const binary = window.atob(base64Data);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) {
    bytes[i] = binary.charCodeAt(i);
  }
  return new Blob([bytes], { type: mimeType });
};

const downloadBlob = (blob, filename) => {
  const url = window.URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  window.URL.revokeObjectURL(url);
};

const normalizePhotorealisticCapability = (raw) => ({
  ...defaultPhotorealisticCapability,
  ...(raw && typeof raw === 'object' ? raw : {}),
});

const basenameFromPath = (pathValue) => {
  const text = typeof pathValue === 'string' ? pathValue.trim() : '';
  if (!text) return '';
  const normalized = text.replace(/\\/g, '/');
  const idx = normalized.lastIndexOf('/');
  return idx >= 0 ? normalized.slice(idx + 1) : normalized;
};

const buildKnotSequenceIndicator = (raw) => {
  if (!raw || typeof raw !== 'object') return null;
  const mode = String(raw.mode || '').trim().toLowerCase();
  const checkpointName = basenameFromPath(raw.checkpoint_path);
  const note = String(raw.load_note || '').trim();

  if (mode === 'pytorch_lstm') {
    return {
      level: 'success',
      message: `Knot model: checkpoint sampler active (${checkpointName || 'knot_sequence_model.pt'}).`,
    };
  }
  if (mode === 'fallback_markov') {
    const suffix = note ? ` Reason: ${note}` : '';
    return {
      level: 'warning',
      message: `Knot model: fallback sampler active (checkpoint not used).${suffix}`,
    };
  }
  if (mode === 'manual_input') {
    return {
      level: 'info',
      message: 'Knot model: manual knot input mode (sequence model not used).',
    };
  }
  return {
    level: 'info',
    message: `Knot model status: ${mode || 'unknown'}.`,
  };
};

const applyLogModeVisualizationDefaults = (cfg) => {
  if (!cfg || Number(cfg.board_or_log) !== 1) return cfg;
  return {
    ...cfg,
    calc_fibers: false,
    quiver_or_stream: 0,
  };
};

function App() {
  const [config, setConfig] = useState(defaultConfig);
  const [simulationData, setSimulationData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [exportingMat, setExportingMat] = useState(false);
  const [exportingMatlabBundle, setExportingMatlabBundle] = useState(false);
  const [exportingPhotorealistic, setExportingPhotorealistic] = useState(false);
  const [preloadingPhotorealistic, setPreloadingPhotorealistic] = useState(false);
  const [capabilities, setCapabilities] = useState({
    photorealistic_export: defaultPhotorealisticCapability,
  });
  const [error, setError] = useState(null);
  const [warnings, setWarnings] = useState([]);
  const [configFeedback, setConfigFeedback] = useState(null);
  const [loadingProgress, setLoadingProgress] = useState(0);
  const [loadingStageIndex, setLoadingStageIndex] = useState(0);
  const [loadingStages, setLoadingStages] = useState([]);
  const [loadingStageStarts, setLoadingStageStarts] = useState([]);
  const [loadingEstimateMs, setLoadingEstimateMs] = useState(10000);
  const [loadingStartedAt, setLoadingStartedAt] = useState(0);
  const [loadingElapsedMs, setLoadingElapsedMs] = useState(0);
  const [photorealisticOverlays, setPhotorealisticOverlays] = useState(null);
  const [showPhotorealisticOverlay, setShowPhotorealisticOverlay] = useState(false);
  const [showNormalOverlay, setShowNormalOverlay] = useState(false);
  const [showFiberOutOfPlaneOverlay, setShowFiberOutOfPlaneOverlay] = useState(false);
  const [photorealisticZipBundle, setPhotorealisticZipBundle] = useState(null);
  useEffect(() => {
    if (!loading) return undefined;

    const updateProgress = () => {
      const elapsed = Date.now() - loadingStartedAt;
      setLoadingElapsedMs(elapsed);

      const raw = (elapsed / Math.max(1000, loadingEstimateMs)) * 100;
      const clamped = Math.min(97, raw);
      setLoadingProgress((prev) => Math.max(prev, clamped));

      let activeIndex = 0;
      for (let i = 0; i < loadingStageStarts.length; i += 1) {
        if (clamped >= loadingStageStarts[i]) activeIndex = i;
        else break;
      }
      setLoadingStageIndex(activeIndex);
    };

    updateProgress();
    const intervalId = window.setInterval(updateProgress, 120);
    return () => window.clearInterval(intervalId);
  }, [loading, loadingStartedAt, loadingEstimateMs, loadingStageStarts]);

  useEffect(() => {
    if (!configFeedback) return undefined;
    const timeoutId = window.setTimeout(() => setConfigFeedback(null), 3600);
    return () => window.clearTimeout(timeoutId);
  }, [configFeedback]);

  useEffect(() => {
    let isMounted = true;

    const loadCapabilities = async () => {
      try {
        const response = await axios.get(`${API_BASE_URL}/capabilities`);
        if (!isMounted) return;

        const payload = response?.data && typeof response.data === 'object' ? response.data : {};
        const photorealistic = normalizePhotorealisticCapability(payload.photorealistic_export);
        setCapabilities({
          photorealistic_export: photorealistic,
        });
      } catch (err) {
        console.error(err);
        if (!isMounted) return;
        setCapabilities({
          photorealistic_export: normalizePhotorealisticCapability(null),
        });
      }
    };

    loadCapabilities();
    return () => { isMounted = false; };
  }, []);

  const buildMatlabBundleRequest = (simId, options = {}) => ({
    simulation_id: simId,
    show_rings_inside_knots: !!config.export_show_rings_inside_knots,
    include_middle_surface: !!options.includeMiddleSurface,
    rand_fibers: !!config.rand_fibers,
    out_of_plane_threshold: Number(config.out_of_plane_threshold),
    snr: Number(config.snr),
    contour_line_width: Math.max(1, Number(config.export_contour_line_width) || 1),
    contour_blur_sigma: Math.max(0, Number(config.export_surface_blur_sigma) || 0),
    fiber_blur_sigma: Math.max(0, Number(config.export_fiber_blur_sigma) || 0),
    fiber_irregularity_strength: Math.min(2, Math.max(0, Number(config.export_fiber_irregularity_strength) || 0)),
    ring_irregularity_strength: Math.min(2, Math.max(0, Number(config.export_ring_irregularity_strength) || 0)),
    imid: Number.isFinite(Number(config.imid)) ? Math.max(0, Math.floor(Number(config.imid))) : 1,
  });

  const buildPhotorealisticRequest = (simId) => ({
    ...(buildMatlabBundleRequest(simId)),
    guidance_scale: Math.max(0, Number(config.photorealistic_guidance_scale) || 0),
    use_img2img_strength: Math.min(1, Math.max(0, Number(config.photorealistic_img2img_strength) || 0)),
    include_knot_maps: !!config.photorealistic_include_knot_maps && !config.photorealistic_use_rings_only,
    use_rings_only: !!config.photorealistic_use_rings_only,
    ddim_steps: Math.max(1, Math.floor(Number(config.photorealistic_steps) || 1)),
  });

  const handleSimulate = async () => {
    const simulationConfig = applyLogModeVisualizationDefaults(config);
    const plan = buildLoadingPlan(simulationConfig);
    setLoading(true);
    setError(null);
    setWarnings([]);
    setLoadingStages(plan.stages);
    setLoadingStageStarts(plan.stageStarts);
    setLoadingEstimateMs(plan.estimateMs);
    setLoadingStartedAt(Date.now());
    setLoadingElapsedMs(0);
    setLoadingProgress(1);
    setLoadingStageIndex(0);
    try {
      const response = await axios.post(`${API_BASE_URL}/simulate`, simulationConfig);
      setSimulationData(response.data);
      const dims = response?.data?.board_dimensions;
      if (dims && typeof dims === 'object' && config.randomize_board_extents_from_dimensions) {
        setConfig((prev) => ({
          ...prev,
          board_x_min: Number.isFinite(Number(dims.x_min)) ? Number(dims.x_min) : prev.board_x_min,
          board_x_max: Number.isFinite(Number(dims.x_max)) ? Number(dims.x_max) : prev.board_x_max,
          board_y_min: Number.isFinite(Number(dims.y_min)) ? Number(dims.y_min) : prev.board_y_min,
          board_y_max: Number.isFinite(Number(dims.y_max)) ? Number(dims.y_max) : prev.board_y_max,
          board_z_min: Number.isFinite(Number(dims.z_min)) ? Number(dims.z_min) : prev.board_z_min,
          board_z_max: Number.isFinite(Number(dims.z_max)) ? Number(dims.z_max) : prev.board_z_max,
          board_width: Number.isFinite(Number(dims.width)) ? Number(dims.width) : prev.board_width,
          board_thickness: Number.isFinite(Number(dims.thickness)) ? Number(dims.thickness) : prev.board_thickness,
          board_length: Number.isFinite(Number(dims.length)) ? Number(dims.length) : prev.board_length,
        }));
      }
      setPhotorealisticOverlays(null);
      setShowPhotorealisticOverlay(false);
      setShowNormalOverlay(false);
      setShowFiberOutOfPlaneOverlay(false);
      setPhotorealisticZipBundle(null);
      const responseWarnings = Array.isArray(response.data?.warnings)
        ? response.data.warnings.filter((w) => typeof w === 'string' && w.trim().length > 0)
        : [];
      setWarnings(responseWarnings);
      console.log("Simulation data received:", response.data);
    } catch (err) {
      console.error(err);
      setError("Simulation failed. Check console or backend connection.");
      setWarnings([]);
    } finally {
      setLoadingProgress(100);
      setLoadingStageIndex(Math.max(0, plan.stages.length - 1));
      await new Promise((resolve) => setTimeout(resolve, 120));
      setLoading(false);
    }
  };

  const handleExportMat = async () => {
    const simId = String(simulationData?.simulation_id || '').trim();
    if (!simId) {
      setError('No simulation export snapshot available. Generate a board first.');
      return;
    }

    setError(null);
    setExportingMat(true);
    try {
      const response = await axios.post(
        `${API_BASE_URL}/export/mat`,
        { simulation_id: simId },
        { responseType: 'blob' }
      );
      const blob = new Blob([response.data], { type: 'application/octet-stream' });
      const contentDisposition = String(response.headers?.['content-disposition'] || '');
      const match = contentDisposition.match(/filename="?([^"]+)"?/i);
      const filename = match?.[1] || `board_export_${simId}_with_visualizer.zip`;
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      a.remove();
      window.URL.revokeObjectURL(url);
    } catch (err) {
      console.error(err);
      setError('MATLAB export failed. Regenerate board and try again.');
    } finally {
      setExportingMat(false);
    }
  };

  const handleExportMatlabBundle = async () => {
    const simId = String(simulationData?.simulation_id || '').trim();
    if (!simId) {
      setError('No simulation export snapshot available. Generate a board first.');
      return;
    }

    setError(null);
    setExportingMatlabBundle(true);
    try {
      const response = await axios.post(
        `${API_BASE_URL}/export/matlab-image-bundle`,
        buildMatlabBundleRequest(simId, { includeMiddleSurface: false }),
        { responseType: 'blob' }
      );
      const blob = new Blob([response.data], { type: 'application/zip' });
      const contentDisposition = String(response.headers?.['content-disposition'] || '');
      const match = contentDisposition.match(/filename="?([^"]+)"?/i);
      const filename = match?.[1] || `matlab_image_bundle_${simId}.zip`;
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      a.remove();
      window.URL.revokeObjectURL(url);
    } catch (err) {
      console.error(err);
      setError('MATLAB-style image bundle export failed. Generate a board with fibers enabled and try again.');
    } finally {
      setExportingMatlabBundle(false);
    }
  };

  const handleExportMatlabBundleWithMiddle = async () => {
    const simId = String(simulationData?.simulation_id || '').trim();
    if (!simId) {
      setError('No simulation export snapshot available. Generate a board first.');
      return;
    }

    setError(null);
    setExportingMatlabBundle(true);
    try {
      const response = await axios.post(
        `${API_BASE_URL}/export/matlab-image-bundle`,
        buildMatlabBundleRequest(simId, { includeMiddleSurface: true }),
        { responseType: 'blob' }
      );
      const blob = new Blob([response.data], { type: 'application/zip' });
      const contentDisposition = String(response.headers?.['content-disposition'] || '');
      const match = contentDisposition.match(/filename="?([^"]+)"?/i);
      const filename = match?.[1] || `matlab_image_bundle_${simId}.zip`;
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      a.remove();
      window.URL.revokeObjectURL(url);
    } catch (err) {
      console.error(err);
      setError('MATLAB image bundle export with middle surface failed. Regenerate board and try again.');
    } finally {
      setExportingMatlabBundle(false);
    }
  };

  const handleExportPhotorealistic = async () => {
    const simId = String(simulationData?.simulation_id || '').trim();
    if (!simId) {
      setError('No simulation export snapshot available. Generate a board first.');
      return;
    }

    setError(null);
    setExportingPhotorealistic(true);
    try {
      const response = await axios.post(
        `${API_BASE_URL}/export/photorealistic-surfaces`,
        {
          ...buildPhotorealisticRequest(simId),
          include_base64: true,
        }
      );
      const payload = response?.data && typeof response.data === 'object' ? response.data : {};
      const surfaces = payload.surfaces && typeof payload.surfaces === 'object' ? payload.surfaces : null;
      if (!surfaces) {
        throw new Error('Backend response missing surfaces payload.');
      }

      const nextOverlays = {};
      for (let idx = 1; idx <= 4; idx += 1) {
        const surfKey = `surface_${idx}`;
        const surfPayload = surfaces[surfKey] && typeof surfaces[surfKey] === 'object'
          ? surfaces[surfKey]
          : {};
        const face = typeof surfPayload.face === 'string' && surfPayload.face
          ? surfPayload.face
          : photorealisticSurfaceFaceOrder[surfKey];
        const filename = typeof surfPayload.filename === 'string' && surfPayload.filename
          ? surfPayload.filename
          : `photorealistic_${idx}.png`;
        const pngBase64 = typeof surfPayload.png_base64 === 'string' ? surfPayload.png_base64.trim() : '';
        if (!pngBase64) {
          throw new Error(`Missing PNG data for ${surfKey}.`);
        }
        nextOverlays[face] = {
          src: `data:image/png;base64,${pngBase64}`,
          flipX: !!surfPayload.flip_x,
          filename,
        };
      }

      const zipBase64 = typeof payload.zip_base64 === 'string' ? payload.zip_base64.trim() : '';
      const zipFilename = typeof payload.zip_filename === 'string' && payload.zip_filename.trim()
        ? payload.zip_filename.trim()
        : `photorealistic_surfaces_${simId}.zip`;
      setPhotorealisticZipBundle(zipBase64 ? { zipBase64, zipFilename } : null);
      setPhotorealisticOverlays(nextOverlays);
      setShowPhotorealisticOverlay(true);
      setConfigFeedback({
        type: 'success',
        message: 'Photorealistic surfaces generated and shown as overlays. Use the download button to save the ZIP.',
      });
    } catch (err) {
      console.error(err);
      let detail = '';
      if (typeof err?.response?.data?.detail === 'string') {
        detail = err.response.data.detail;
      }
      setPhotorealisticZipBundle(null);
      setError(detail || 'Photorealistic export failed. Ensure backend photorealistic dependencies are available.');
    } finally {
      setExportingPhotorealistic(false);
    }
  };

  const handleDownloadPhotorealisticZip = () => {
    const zipBase64 = typeof photorealisticZipBundle?.zipBase64 === 'string'
      ? photorealisticZipBundle.zipBase64.trim()
      : '';
    if (!zipBase64) {
      setError('No generated photorealistic ZIP is available yet. Run photorealistic generation first.');
      return;
    }
    const filename = typeof photorealisticZipBundle?.zipFilename === 'string' && photorealisticZipBundle.zipFilename.trim()
      ? photorealisticZipBundle.zipFilename.trim()
      : 'photorealistic_surfaces.zip';
    const zipBlob = base64ToBlob(zipBase64, 'application/zip');
    downloadBlob(zipBlob, filename);
  };

  const handlePreloadPhotorealistic = async () => {
    setError(null);
    setPreloadingPhotorealistic(true);
    try {
      await axios.post(`${API_BASE_URL}/photorealistic/preload`, {});
      const response = await axios.get(`${API_BASE_URL}/capabilities`);
      const payload = response?.data && typeof response.data === 'object' ? response.data : {};
      const photorealistic = normalizePhotorealisticCapability(payload.photorealistic_export);
      setCapabilities({
        photorealistic_export: photorealistic,
      });
      setConfigFeedback({
        type: 'success',
        message: 'Photorealistic model is loaded and ready.',
      });
    } catch (err) {
      console.error(err);
      const detail = typeof err?.response?.data?.detail === 'string'
        ? err.response.data.detail
        : '';
      setError(detail || 'Photorealistic model preload failed.');
    } finally {
      setPreloadingPhotorealistic(false);
    }
  };

  const photorealisticCapability = capabilities?.photorealistic_export || defaultPhotorealisticCapability;
  const photorealisticAvailable = !!photorealisticCapability.available;
  const photorealisticLoaded = !!photorealisticCapability.loaded;
  const photorealisticReason = String(photorealisticCapability.reason || '');
  const viewerConfig = applyLogModeVisualizationDefaults(config);
  const isLogMode = Number(config.board_or_log) === 1;
  const normalOverlayAvailable = !!(
    simulationData
    && simulationData.normal_overlays
    && typeof simulationData.normal_overlays === 'object'
    && Object.keys(simulationData.normal_overlays).length > 0
  );
  const fiberOutOfPlaneOverlayAvailable = !!(
    simulationData
    && simulationData.fiber_out_of_plane_overlays
    && typeof simulationData.fiber_out_of_plane_overlays === 'object'
    && Object.keys(simulationData.fiber_out_of_plane_overlays).length > 0
  );
  const knotSequenceIndicator = buildKnotSequenceIndicator(simulationData?.knot_sequence);

  const handleSaveConfig = () => {
    try {
      const payload = {
        format: 'board-generator-ui-config',
        version: 1,
        saved_at: new Date().toISOString(),
        config,
      };
      const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' });
      const stamp = new Date().toISOString().replace(/[:.]/g, '-');
      const filename = `board_ui_config_${stamp}.json`;
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      a.remove();
      window.URL.revokeObjectURL(url);
      setError(null);
    } catch (err) {
      console.error(err);
      setError('Saving UI configuration failed.');
    }
  };

  const handleLoadConfig = async (file) => {
    if (!file) return;
    setError(null);
    try {
      const text = await file.text();
      const parsed = JSON.parse(text);
      const nextConfig = normalizeLoadedConfig(parsed);
      setConfig(nextConfig);
      setConfigFeedback({
        type: 'success',
        message: `UI configuration loaded successfully (${file.name}).`,
      });
    } catch (err) {
      console.error(err);
      setConfigFeedback({
        type: 'error',
        message: 'Loading UI configuration failed. Use a valid exported JSON config file.',
      });
    }
  };

  return (
    <div className="App">
      <aside className="control-column">
        <ControlPanel
          config={config}
          onConfigChange={setConfig}
          isLogMode={isLogMode}
          onSimulate={handleSimulate}
          loading={loading}
          onExportMat={handleExportMat}
          onExportMatlabBundle={handleExportMatlabBundle}
          onExportMatlabBundleWithMiddle={handleExportMatlabBundleWithMiddle}
          onExportPhotorealistic={handleExportPhotorealistic}
          onDownloadPhotorealisticZip={handleDownloadPhotorealisticZip}
          onPreloadPhotorealistic={handlePreloadPhotorealistic}
          onSaveConfig={handleSaveConfig}
          onLoadConfig={handleLoadConfig}
          exportMatDisabled={loading || exportingMat || !simulationData || !simulationData.simulation_id}
          exportMatLoading={exportingMat}
          exportMatlabBundleDisabled={loading || exportingMatlabBundle || !simulationData || !simulationData.simulation_id || !simulationData.fibers}
          exportMatlabBundleLoading={exportingMatlabBundle}
          exportPhotorealisticDisabled={
            loading
            || exportingPhotorealistic
            || !simulationData
            || !simulationData.simulation_id
            || !simulationData.fibers
            || !photorealisticAvailable
            || isDemoMode
          }
          exportPhotorealisticLoading={exportingPhotorealistic}
          downloadPhotorealisticZipDisabled={
            loading
            || exportingPhotorealistic
            || !photorealisticZipBundle
            || !photorealisticZipBundle.zipBase64
          }
          preloadPhotorealisticDisabled={
            loading
            || preloadingPhotorealistic
            || photorealisticLoaded
            || !photorealisticAvailable
            || isDemoMode
          }
          preloadPhotorealisticLoading={preloadingPhotorealistic}
          photorealisticAvailable={photorealisticAvailable}
          photorealisticLoaded={photorealisticLoaded}
          photorealisticReason={photorealisticReason}
          demoMode={isDemoMode}
          showPhotorealisticOverlay={showPhotorealisticOverlay}
          photorealisticOverlayAvailable={!!photorealisticOverlays}
          onTogglePhotorealisticOverlay={(next) => setShowPhotorealisticOverlay(!!next)}
          showNormalOverlay={showNormalOverlay}
          normalOverlayAvailable={normalOverlayAvailable}
          onToggleNormalOverlay={(next) => setShowNormalOverlay(!!next)}
          showFiberOutOfPlaneOverlay={showFiberOutOfPlaneOverlay}
          fiberOutOfPlaneOverlayAvailable={fiberOutOfPlaneOverlayAvailable}
          onToggleFiberOutOfPlaneOverlay={(next) => setShowFiberOutOfPlaneOverlay(!!next)}
        />
      </aside>

      <main className="viewer-container">
        {(error || warnings.length > 0 || !!configFeedback || !!knotSequenceIndicator) && (
          <div className="message-stack">
            {error && <div className="error-overlay">{error}</div>}
            {warnings.map((warning, index) => (
              <div className="warning-overlay" key={`warn-${index}`}>{warning}</div>
            ))}
            {knotSequenceIndicator && (
              <div className={`knot-seq-overlay ${knotSequenceIndicator.level}`}>
                {knotSequenceIndicator.message}
              </div>
            )}
            {configFeedback && (
              <div className={`config-feedback-overlay ${configFeedback.type}`}>{configFeedback.message}</div>
            )}
          </div>
        )}

        {simulationData && (
          <div className={`viewer-status ${simulationData.gpu_active ? 'gpu-on' : 'gpu-off'}`}>
            GPU: {simulationData.gpu_active ? 'Active' : (simulationData.gpu_requested ? 'Requested, fallback to CPU' : 'Off')}
          </div>
        )}

        <Viewer3D
          data={simulationData}
          config={viewerConfig}
          showKnotSequenceSlots={!!viewerConfig.display_knot_slots}
          photorealisticOverlays={photorealisticOverlays}
          showPhotorealisticOverlay={showPhotorealisticOverlay}
          normalOverlays={simulationData?.normal_overlays || null}
          showNormalOverlay={showNormalOverlay}
          fiberOutOfPlaneOverlays={simulationData?.fiber_out_of_plane_overlays || null}
          showFiberOutOfPlaneOverlay={showFiberOutOfPlaneOverlay}
        />

        {loading && (
          <div className="loading-overlay">
            <div className="loading-header">
              <div className="spinner"></div>
              <div className="loading-copy">
                <p className="loading-title">Generating board data...</p>
                <p className="loading-stage">{loadingStages[loadingStageIndex]?.label || 'Running simulation'}</p>
              </div>
            </div>
            <div className="loading-bar">
              <div className="loading-bar-fill" style={{ width: `${Math.max(1, Math.min(100, loadingProgress))}%` }} />
            </div>
            <div className="loading-progress-row">
              <span>{Math.round(loadingProgress)}%</span>
              <span>{(loadingElapsedMs / 1000).toFixed(1)}s elapsed</span>
            </div>
            <div className="loading-stage-list">
              {loadingStages.map((stage, index) => {
                const className = index < loadingStageIndex
                  ? 'loading-stage-item done'
                  : index === loadingStageIndex
                    ? 'loading-stage-item active'
                    : 'loading-stage-item';
                return (
                  <div className={className} key={stage.key}>{stage.label}</div>
                );
              })}
            </div>
          </div>
        )}
      </main>
    </div>
  );
}

export default App;
