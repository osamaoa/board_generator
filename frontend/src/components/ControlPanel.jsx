import React, { useEffect, useMemo, useRef, useState } from 'react';
import './ControlPanel.css';

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
    a4: 0.6,
};
const defaultCrookComponentCount = 8;
const defaultRandomCrookAmplitudeMax = [50.0, 25.0, 12.5, 5.0, 2.5, 1.25, 0.625, 0.3125];
const defaultManualCrookAmplitudes = [25.0, 12.5, 6.25, 2.5, 1.25, 0.625, 0.3125, 0.15625];
const defaultManualCrookShiftsMm = Array.from({ length: defaultCrookComponentCount }, () => 4000.0);
const defaultManualCrookThetasDeg = [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0];
const defaultManualCrookOrders = [1, 2, 3, 4, 5, 6, 7, 8];
const defaultCrookShiftMaxMm = 8000.0;
const defaultRandomTaperMax = 1.0 / 160.0;
const defaultManualTaperCoeff = 1.0 / 160.0;

const sectionTabs = [
    { id: 'geometry', label: 'GEOM' },
    { id: 'knots', label: 'KNOT' },
    { id: 'fibers', label: 'FIBER' },
    { id: 'simulation', label: 'SIM' },
    { id: 'visuals', label: 'VIEW' },
    { id: 'export', label: 'EXPORT' },
];

const knotParamFields = [
    { key: 'th0_deg', label: 'th0 (deg)', step: 1 },
    { key: 'L100', label: 'L100', step: 1 },
    { key: 'z0', label: 'z0', step: 1 },
    { key: 'c1', label: 'c1', step: 0.0001 },
    { key: 'c2', label: 'c2', step: 0.0001 },
    { key: 'k', label: 'k', step: 0.01 },
    { key: 'kp', label: 'kp', step: 0.01 },
    { key: 'Abump', label: 'Abump', step: 0.001 },
    { key: 'Aexp', label: 'Aexp', step: 0.001 },
    { key: 'Bbump', label: 'Bbump', step: 0.1 },
    { key: 'RL', label: 'RL', step: 1 },
    { key: 'RD', label: 'RD', step: 1 },
    { key: 'a1', label: 'a1', step: 0.0000001 },
    { key: 'a2', label: 'a2', step: 0.00001 },
    { key: 'a3', label: 'a3', step: 0.0001 },
    { key: 'a4', label: 'a4', step: 0.01 },
];

const toNumber = (value, fallback = 0) => {
    const parsed = typeof value === 'number' ? value : parseFloat(value);
    return Number.isFinite(parsed) ? parsed : fallback;
};

const toInt = (value, fallback = 0) => {
    const parsed = typeof value === 'number' ? value : parseInt(value, 10);
    return Number.isFinite(parsed) ? parsed : fallback;
};
const normalizePositiveIntArray = (raw) => {
    const values = Array.isArray(raw)
        ? raw
        : (typeof raw === 'string' ? raw.split(',') : []);
    const out = [];
    for (const token of values) {
        const val = Math.floor(Number(token));
        if (Number.isFinite(val) && val >= 1) out.push(val);
    }
    return out;
};
const formatPositiveIntArray = (raw) => normalizePositiveIntArray(raw).join(', ');

const ControlPanel = ({
    config,
    onConfigChange,
    isLogMode = false,
    onSimulate,
    loading,
    onExportMat,
    onExportMatlabBundle,
    onExportMatlabBundleWithMiddle,
    onExportPhotorealistic,
    onDownloadPhotorealisticZip,
    onPreloadPhotorealistic,
    onSaveConfig,
    onLoadConfig,
    exportMatDisabled,
    exportMatLoading,
    exportMatlabBundleDisabled,
    exportMatlabBundleLoading,
    exportPhotorealisticDisabled,
    exportPhotorealisticLoading,
    downloadPhotorealisticZipDisabled,
    preloadPhotorealisticDisabled,
    preloadPhotorealisticLoading,
    photorealisticAvailable,
    photorealisticLoaded,
    photorealisticReason,
    showPhotorealisticOverlay,
    photorealisticOverlayAvailable,
    onTogglePhotorealisticOverlay,
    showNormalOverlay,
    normalOverlayAvailable,
    onToggleNormalOverlay,
    showFiberOutOfPlaneOverlay,
    fiberOutOfPlaneOverlayAvailable,
    onToggleFiberOutOfPlaneOverlay,
    demoMode = false,
}) => {
    const [activeSection, setActiveSection] = useState('geometry');
    const [selectedKnotIndex, setSelectedKnotIndex] = useState(0);
    const [knotInsideLimitDraft, setKnotInsideLimitDraft] = useState(String(toNumber(config.knot_inside_limit, 0)));
    const loadConfigInputRef = useRef(null);

    const normalizeInputKnots = (cfg) => {
        const count = Math.max(0, toInt(cfg.input_knot_count, 0));
        const knots = Array.isArray(cfg.input_knots)
            ? cfg.input_knots.map((knot) => ({ ...defaultInputKnot, ...(knot || {}) }))
            : [];
        while (knots.length < count) {
            knots.push({ ...defaultInputKnot });
        }
        return knots.slice(0, count);
    };
    const defaultCrookValueAt = (defaults, index, fill = 0) => {
        if (!Array.isArray(defaults) || defaults.length === 0) return fill;
        if (index < defaults.length) return toNumber(defaults[index], fill);
        return toNumber(defaults[defaults.length - 1], fill);
    };
    const normalizeCrookArray = (raw, count, defaults = [], fill = 0) => {
        const values = Array.isArray(raw) ? raw : [];
        const out = [];
        for (let i = 0; i < count; i += 1) {
            const fallback = defaultCrookValueAt(defaults, i, fill);
            out.push(toNumber(values[i], fallback));
        }
        return out;
    };
    const crookComponentCount = Math.max(1, toInt(config.crook_component_count, defaultCrookComponentCount));
    const randomCrookAmplitudeMax = useMemo(
        () => normalizeCrookArray(config.random_crook_amplitude_max, crookComponentCount, defaultRandomCrookAmplitudeMax, 0).map((v) => Math.max(0, v)),
        [config.random_crook_amplitude_max, crookComponentCount]
    );
    const manualCrookAmplitudes = useMemo(
        () => normalizeCrookArray(config.manual_crook_amplitudes, crookComponentCount, defaultManualCrookAmplitudes, 0).map((v) => Math.max(0, v)),
        [config.manual_crook_amplitudes, crookComponentCount]
    );
    const manualCrookShifts = useMemo(
        () => normalizeCrookArray(config.manual_crook_shifts_mm, crookComponentCount, defaultManualCrookShiftsMm, 0),
        [config.manual_crook_shifts_mm, crookComponentCount]
    );
    const manualCrookThetas = useMemo(
        () => normalizeCrookArray(config.manual_crook_thetas_deg, crookComponentCount, defaultManualCrookThetasDeg, 0),
        [config.manual_crook_thetas_deg, crookComponentCount]
    );
    const manualCrookOrders = useMemo(
        () => normalizeCrookArray(config.manual_crook_orders, crookComponentCount, defaultManualCrookOrders, 1)
            .map((v, idx) => Math.max(1, Math.floor(toNumber(v, idx + 1)))),
        [config.manual_crook_orders, crookComponentCount]
    );
    const randomCrookExtraOrdersText = useMemo(
        () => formatPositiveIntArray(config.random_crook_extra_orders),
        [config.random_crook_extra_orders]
    );

    const manualKnots = useMemo(
        () => normalizeInputKnots(config),
        [config.input_knot_count, config.input_knots]
    );

    useEffect(() => {
        if (manualKnots.length === 0) {
            if (selectedKnotIndex !== 0) setSelectedKnotIndex(0);
            return;
        }
        if (selectedKnotIndex > manualKnots.length - 1) {
            setSelectedKnotIndex(manualKnots.length - 1);
        }
    }, [manualKnots.length, selectedKnotIndex]);

    useEffect(() => {
        setKnotInsideLimitDraft(String(toNumber(config.knot_inside_limit, 0)));
    }, [config.knot_inside_limit]);

    const handleChange = (key, value, type = 'string') => {
        let val = value;
        if (type === 'number') val = toNumber(value, toNumber(config[key], 0));
        if (type === 'int') val = toInt(value, toInt(config[key], 0));
        if (type === 'bool') val = value;

        const nextConfig = { ...config, [key]: val };
        if (key === 'mesh_size_x_mm' || key === 'mesh_size_y_mm' || key === 'mesh_size_z_mm') {
            const sx = Math.max(0.05, toNumber(nextConfig.mesh_size_x_mm, toNumber(config.mesh_size_x_mm, 2.0)));
            const sy = Math.max(0.05, toNumber(nextConfig.mesh_size_y_mm, toNumber(config.mesh_size_y_mm, 2.0)));
            const sz = Math.max(0.05, toNumber(nextConfig.mesh_size_z_mm, toNumber(config.mesh_size_z_mm, 2.0)));
            nextConfig.mesh_size_x_mm = sx;
            nextConfig.mesh_size_y_mm = sy;
            nextConfig.mesh_size_z_mm = sz;
        }
        if (key === 'board_width' || key === 'board_thickness' || key === 'board_length') {
            nextConfig.board_width = Math.max(1e-6, toNumber(nextConfig.board_width, toNumber(config.board_width, 145.0)));
            nextConfig.board_thickness = Math.max(1e-6, toNumber(nextConfig.board_thickness, toNumber(config.board_thickness, 45.0)));
            nextConfig.board_length = Math.max(1e-6, toNumber(nextConfig.board_length, toNumber(config.board_length, 145.0)));
        }
        if (key === 'crook_shift_max_mm') {
            nextConfig.crook_shift_max_mm = Math.max(0, toNumber(nextConfig.crook_shift_max_mm, defaultCrookShiftMaxMm));
        }
        if (key === 'log_layer_stride') {
            nextConfig.log_layer_stride = Math.max(1, toInt(nextConfig.log_layer_stride, 5));
        }
        if (key === 'random_crook_scale_max') {
            nextConfig.random_crook_scale_max = Math.max(0, toNumber(nextConfig.random_crook_scale_max, 1.0));
        }
        if (key === 'random_taper_max') {
            nextConfig.random_taper_max = Math.max(0, toNumber(nextConfig.random_taper_max, defaultRandomTaperMax));
        }
        if (key === 'knot_generator_min_rd_minus_rl_mm') {
            nextConfig.knot_generator_min_rd_minus_rl_mm = Math.max(0, toNumber(nextConfig.knot_generator_min_rd_minus_rl_mm, 30.0));
        }
        if (key === 'input_knot_count' || key === 'use_input_knots' || key === 'input_knots') {
            nextConfig.input_knot_count = Math.max(0, toInt(nextConfig.input_knot_count, 0));
            nextConfig.input_knots = normalizeInputKnots(nextConfig);
            if (selectedKnotIndex > nextConfig.input_knot_count - 1) {
                setSelectedKnotIndex(Math.max(0, nextConfig.input_knot_count - 1));
            }
        }
        if (key === 'photorealistic_use_rings_only' && !!nextConfig.photorealistic_use_rings_only) {
            nextConfig.photorealistic_include_knot_maps = false;
        }
        if (key === 'photorealistic_include_knot_maps' && !!nextConfig.photorealistic_use_rings_only) {
            nextConfig.photorealistic_include_knot_maps = false;
        }
        onConfigChange(nextConfig);
    };

    const setKnotCount = (countRaw) => {
        const count = Math.max(0, toInt(countRaw, 0));
        const nextConfig = { ...config, input_knot_count: count };
        nextConfig.input_knots = normalizeInputKnots(nextConfig);
        onConfigChange(nextConfig);
        if (count === 0) {
            setSelectedKnotIndex(0);
        } else if (selectedKnotIndex > count - 1) {
            setSelectedKnotIndex(count - 1);
        }
    };

    const saveKnots = (nextKnots) => {
        const count = Math.max(0, nextKnots.length);
        onConfigChange({
            ...config,
            input_knot_count: count,
            input_knots: nextKnots.map((knot) => ({ ...defaultInputKnot, ...knot })),
        });
    };

    const handleKnotParamChange = (index, key, value) => {
        const val = toNumber(value, toNumber(manualKnots[index]?.[key], 0));
        const knots = manualKnots.map((knot) => ({ ...knot }));
        const nextKnot = { ...knots[index], [key]: val };
        if (key === 'L100') {
            nextKnot.Abump = Math.abs(0.0217 * val - 0.2);
            nextKnot.Aexp = 0.0056 * val + 1.96;
            nextKnot.a4 = 0.01 * val + 0.2;
        }
        knots[index] = nextKnot;
        saveKnots(knots);
    };

    const addKnot = () => {
        const knots = manualKnots.map((knot) => ({ ...knot }));
        knots.push({ ...defaultInputKnot });
        saveKnots(knots);
        setSelectedKnotIndex(knots.length - 1);
    };

    const duplicateSelectedKnot = () => {
        if (manualKnots.length === 0) return;
        const knots = manualKnots.map((knot) => ({ ...knot }));
        const source = knots[selectedKnotIndex] ?? defaultInputKnot;
        knots.splice(selectedKnotIndex + 1, 0, { ...source });
        saveKnots(knots);
        setSelectedKnotIndex(selectedKnotIndex + 1);
    };

    const removeSelectedKnot = () => {
        if (manualKnots.length === 0) return;
        const knots = manualKnots.filter((_, idx) => idx !== selectedKnotIndex);
        saveKnots(knots);
        if (knots.length === 0) {
            setSelectedKnotIndex(0);
        } else if (selectedKnotIndex > knots.length - 1) {
            setSelectedKnotIndex(knots.length - 1);
        }
    };

    const resetSelectedKnot = () => {
        if (manualKnots.length === 0) return;
        const knots = manualKnots.map((knot) => ({ ...knot }));
        knots[selectedKnotIndex] = { ...defaultInputKnot };
        saveKnots(knots);
    };

    const selectedKnot = manualKnots[selectedKnotIndex] ?? null;

    const handleLoadConfigFile = async (event) => {
        const file = event?.target?.files?.[0];
        if (!file || !onLoadConfig) return;
        await onLoadConfig(file);
        event.target.value = '';
    };

    const handleKnotInsideLimitInput = (raw) => {
        setKnotInsideLimitDraft(raw);
        if (raw === '' || raw === '-' || raw === '.' || raw === '-.') return;
        const parsed = Number(raw);
        if (Number.isFinite(parsed)) {
            handleChange('knot_inside_limit', parsed, 'number');
        }
    };

    const commitKnotInsideLimitInput = () => {
        const parsed = Number(knotInsideLimitDraft);
        const safeValue = Number.isFinite(parsed) ? parsed : toNumber(config.knot_inside_limit, 0);
        setKnotInsideLimitDraft(String(safeValue));
        handleChange('knot_inside_limit', safeValue, 'number');
    };
    const setCrookComponentCount = (raw) => {
        const nextCount = Math.max(1, toInt(raw, crookComponentCount));
        onConfigChange({
            ...config,
            crook_component_count: nextCount,
            random_crook_amplitude_max: normalizeCrookArray(config.random_crook_amplitude_max, nextCount, defaultRandomCrookAmplitudeMax, 0).map((v) => Math.max(0, v)),
            manual_crook_amplitudes: normalizeCrookArray(config.manual_crook_amplitudes, nextCount, defaultManualCrookAmplitudes, 0).map((v) => Math.max(0, v)),
            manual_crook_shifts_mm: normalizeCrookArray(config.manual_crook_shifts_mm, nextCount, defaultManualCrookShiftsMm, 0),
            manual_crook_thetas_deg: normalizeCrookArray(config.manual_crook_thetas_deg, nextCount, defaultManualCrookThetasDeg, 0),
            manual_crook_orders: normalizeCrookArray(config.manual_crook_orders, nextCount, defaultManualCrookOrders, 1)
                .map((v, idx) => Math.max(1, Math.floor(toNumber(v, idx + 1)))),
        });
    };
    const updateCrookArrayValue = (key, index, rawValue, defaults = [], nonNegative = false) => {
        const current = normalizeCrookArray(config[key], crookComponentCount, defaults, 0);
        let nextVal = toNumber(rawValue, current[index] ?? defaultCrookValueAt(defaults, index, 0));
        if (nonNegative) nextVal = Math.max(0, nextVal);
        current[index] = nextVal;
        onConfigChange({
            ...config,
            [key]: current,
        });
    };
    const updateCrookIntArrayValue = (key, index, rawValue, defaults = [], minimum = 1) => {
        const current = normalizeCrookArray(config[key], crookComponentCount, defaults, minimum)
            .map((v, idx) => Math.max(minimum, Math.floor(toNumber(v, idx + minimum))));
        const nextVal = Math.max(minimum, toInt(rawValue, current[index] ?? minimum));
        current[index] = nextVal;
        onConfigChange({
            ...config,
            [key]: current,
        });
    };
    const setRandomCrookExtraOrdersFromText = (rawText) => {
        onConfigChange({
            ...config,
            random_crook_extra_orders: normalizePositiveIntArray(rawText),
        });
    };
    const logLengthMm = Math.max(
        0.1,
        Math.abs(toNumber(config.board_z_max, 145.0) - toNumber(config.board_z_min, 0.0))
    );
    const setLogLength = (rawValue) => {
        const currentZMin = toNumber(config.board_z_min, 0.0);
        const nextLength = Math.max(0.1, toNumber(rawValue, logLengthMm));
        onConfigChange({
            ...config,
            board_z_min: currentZMin,
            board_z_max: currentZMin + nextLength,
        });
    };
    return (
        <div className="control-panel">
            <nav className="section-nav">
                {sectionTabs.map((tab) => (
                    <button
                        key={tab.id}
                        className={`section-tab ${activeSection === tab.id ? 'active' : ''}`}
                        onClick={() => setActiveSection(tab.id)}
                        type="button"
                    >
                        {tab.label}
                    </button>
                ))}
            </nav>

            <div className="scroll-container">
                {activeSection === 'geometry' && (
                    <section className="panel-section">
                        <h3>Geometry</h3>
                        <p className="section-copy">Define board extents or use board dimensions with random in-log placement, plus stochastic crook/taper limits.</p>
                        <div className="field-grid single">
                            <label className="field">
                                <span>Mode</span>
                                <select value={config.board_or_log} onChange={(e) => handleChange('board_or_log', e.target.value, 'int')}>
                                    <option value={0}>Board</option>
                                    <option value={1}>Log</option>
                                </select>
                            </label>
                        </div>

                        {toInt(config.board_or_log, 0) === 0 && (
                            <label className="toggle">
                                <input
                                    type="checkbox"
                                    checked={!!config.randomize_board_extents_from_dimensions}
                                    onChange={(e) => handleChange('randomize_board_extents_from_dimensions', e.target.checked, 'bool')}
                                />
                                <span>Random Board Placement From Dimensions</span>
                            </label>
                        )}

                        {toInt(config.board_or_log, 0) === 0 && !config.randomize_board_extents_from_dimensions && (
                            <div className="field-grid">
                                <label className="field">
                                    <span>X Min</span>
                                    <input type="number" value={config.board_x_min} onChange={(e) => handleChange('board_x_min', e.target.value, 'number')} />
                                </label>
                                <label className="field">
                                    <span>X Max</span>
                                    <input type="number" value={config.board_x_max} onChange={(e) => handleChange('board_x_max', e.target.value, 'number')} />
                                </label>
                                <label className="field">
                                    <span>Y Min</span>
                                    <input type="number" value={config.board_y_min} onChange={(e) => handleChange('board_y_min', e.target.value, 'number')} />
                                </label>
                                <label className="field">
                                    <span>Y Max</span>
                                    <input type="number" value={config.board_y_max} onChange={(e) => handleChange('board_y_max', e.target.value, 'number')} />
                                </label>
                                <label className="field">
                                    <span>Z Min</span>
                                    <input type="number" value={config.board_z_min} onChange={(e) => handleChange('board_z_min', e.target.value, 'number')} />
                                </label>
                                <label className="field">
                                    <span>Z Max</span>
                                    <input type="number" value={config.board_z_max} onChange={(e) => handleChange('board_z_max', e.target.value, 'number')} />
                                </label>
                            </div>
                        )}

                        {toInt(config.board_or_log, 0) === 0 && config.randomize_board_extents_from_dimensions && (
                            <>
                                <div className="field-grid">
                                    <label className="field">
                                        <span>Board Width (mm)</span>
                                        <input
                                            type="number"
                                            min={0.1}
                                            step="0.1"
                                            value={toNumber(config.board_width, 145)}
                                            onChange={(e) => handleChange('board_width', e.target.value, 'number')}
                                        />
                                    </label>
                                    <label className="field">
                                        <span>Board Thickness (mm)</span>
                                        <input
                                            type="number"
                                            min={0.1}
                                            step="0.1"
                                            value={toNumber(config.board_thickness, 45)}
                                            onChange={(e) => handleChange('board_thickness', e.target.value, 'number')}
                                        />
                                    </label>
                                    <label className="field">
                                        <span>Board Length (mm)</span>
                                        <input
                                            type="number"
                                            min={0.1}
                                            step="0.1"
                                            value={toNumber(config.board_length, 145)}
                                            onChange={(e) => handleChange('board_length', e.target.value, 'number')}
                                        />
                                    </label>
                                </div>
                                <p className="geometry-note">
                                    For each simulation, X/Y extents are sampled randomly inside the generated log.
                                    Z extents are set to [0, Board Length].
                                </p>
                            </>
                        )}

                        {toInt(config.board_or_log, 0) === 1 && (
                            <>
                                <div className="field-grid">
                                    <label className="field">
                                        <span>X Min</span>
                                        <input
                                            type="number"
                                            value={toNumber(config.board_x_min, -72.5)}
                                            onChange={(e) => handleChange('board_x_min', e.target.value, 'number')}
                                        />
                                    </label>
                                    <label className="field">
                                        <span>X Max</span>
                                        <input
                                            type="number"
                                            value={toNumber(config.board_x_max, 72.5)}
                                            onChange={(e) => handleChange('board_x_max', e.target.value, 'number')}
                                        />
                                    </label>
                                    <label className="field">
                                        <span>Y Min</span>
                                        <input
                                            type="number"
                                            value={toNumber(config.board_y_min, -22.5)}
                                            onChange={(e) => handleChange('board_y_min', e.target.value, 'number')}
                                        />
                                    </label>
                                    <label className="field">
                                        <span>Y Max</span>
                                        <input
                                            type="number"
                                            value={toNumber(config.board_y_max, 22.5)}
                                            onChange={(e) => handleChange('board_y_max', e.target.value, 'number')}
                                        />
                                    </label>
                                </div>
                                <div className="field-grid single">
                                    <label className="field">
                                        <span>Log Length (mm)</span>
                                        <input
                                            type="number"
                                            min={0.1}
                                            step="1"
                                            value={logLengthMm}
                                            onChange={(e) => setLogLength(e.target.value)}
                                        />
                                    </label>
                                </div>
                                <div className="field-grid single">
                                    <label className="field">
                                        <span>Log Layer Stride n</span>
                                        <input
                                            type="number"
                                            min={1}
                                            step={1}
                                            value={Math.max(1, toInt(config.log_layer_stride, 5))}
                                            onChange={(e) => handleChange('log_layer_stride', e.target.value, 'int')}
                                        />
                                    </label>
                                </div>
                                <p className="geometry-note">
                                    Log mode board placement uses X/Y extents.
                                    Z extent is set by `board_z_max - board_z_min` (Log Length).
                                    Growth-layer rendering uses every n-th layer (plus first and last).
                                </p>
                            </>
                        )}
                        <label className="toggle">
                            <input
                                type="checkbox"
                                checked={!!config.randomize_crook_taper}
                                onChange={(e) => handleChange('randomize_crook_taper', e.target.checked, 'bool')}
                            />
                            <span>Randomize Crook/Taper</span>
                        </label>

                        <div className="field-grid single">
                            <label className="field">
                                <span>Crook Components (p)</span>
                                <input
                                    type="number"
                                    min={1}
                                    step={1}
                                    value={crookComponentCount}
                                    onChange={(e) => setCrookComponentCount(e.target.value)}
                                />
                            </label>
                        </div>

                        {!!config.randomize_crook_taper ? (
                            <>
                                <div className="field-grid">
                                    <label className="field">
                                        <span>Crook Amplitude Scale Max</span>
                                        <input
                                            type="number"
                                            min={0}
                                            step="0.01"
                                            value={toNumber(config.random_crook_scale_max, 1.0)}
                                            onChange={(e) => handleChange('random_crook_scale_max', e.target.value, 'number')}
                                        />
                                    </label>
                                    <label className="field">
                                        <span>Crook Shift Max (mm)</span>
                                        <input
                                            type="number"
                                            min={0}
                                            step="1"
                                            value={toNumber(config.crook_shift_max_mm, defaultCrookShiftMaxMm)}
                                            onChange={(e) => handleChange('crook_shift_max_mm', e.target.value, 'number')}
                                        />
                                    </label>
                                </div>
                                <div className="field-grid single">
                                    <label className="field">
                                        <span>Extra Crook Orders (CSV)</span>
                                        <input
                                            type="text"
                                            placeholder="e.g. 9,10"
                                            value={randomCrookExtraOrdersText}
                                            onChange={(e) => setRandomCrookExtraOrdersFromText(e.target.value)}
                                        />
                                    </label>
                                </div>
                                {Array.from({ length: crookComponentCount }).map((_, idx) => (
                                    <div key={`random-crook-component-${idx}`} className="field-grid single">
                                        <label className="field">
                                            <span>{`Amplitude Max a${idx + 1} (mm)`}</span>
                                            <input
                                                type="number"
                                                min={0}
                                                step="0.01"
                                                value={toNumber(randomCrookAmplitudeMax[idx], 0)}
                                                onChange={(e) => updateCrookArrayValue('random_crook_amplitude_max', idx, e.target.value, defaultRandomCrookAmplitudeMax, true)}
                                            />
                                        </label>
                                    </div>
                                ))}
                                <div className="field-grid">
                                    <label className="field">
                                        <span>Crook Theta Min (deg)</span>
                                        <input
                                            type="number"
                                            step="1"
                                            value={toNumber(config.random_crook_theta_min_deg, 0.0)}
                                            onChange={(e) => handleChange('random_crook_theta_min_deg', e.target.value, 'number')}
                                        />
                                    </label>
                                    <label className="field">
                                        <span>Crook Theta Max (deg)</span>
                                        <input
                                            type="number"
                                            step="1"
                                            value={toNumber(config.random_crook_theta_max_deg, 360.0)}
                                            onChange={(e) => handleChange('random_crook_theta_max_deg', e.target.value, 'number')}
                                        />
                                    </label>
                                </div>
                                <div className="field-grid single">
                                    <label className="field">
                                        <span>Taper Coeff Max</span>
                                        <input
                                            type="number"
                                            min={0}
                                            step="0.0001"
                                            value={toNumber(config.random_taper_max, defaultRandomTaperMax)}
                                            onChange={(e) => handleChange('random_taper_max', e.target.value, 'number')}
                                        />
                                    </label>
                                </div>
                                <p className="geometry-note">
                                    Crook and taper are re-sampled automatically each time you press Generate Board.
                                </p>
                            </>
                        ) : (
                            <>
                                <div className="field-grid single">
                                    <label className="field">
                                        <span>Manual Taper Coeff</span>
                                        <input
                                            type="number"
                                            step="0.0001"
                                            value={toNumber(config.manual_taper_coeff, defaultManualTaperCoeff)}
                                            onChange={(e) => handleChange('manual_taper_coeff', e.target.value, 'number')}
                                        />
                                    </label>
                                </div>
                                {Array.from({ length: crookComponentCount }).map((_, idx) => (
                                    <div key={`crook-component-${idx}`} className="knot-card">
                                        <h4>{`Crook Component ${idx + 1}`}</h4>
                                        <div className="field-grid">
                                            <label className="field">
                                                <span>{`Amplitude a${idx + 1}`}</span>
                                                <input
                                                    type="number"
                                                    min={0}
                                                    step="0.01"
                                                    value={toNumber(manualCrookAmplitudes[idx], 0)}
                                                    onChange={(e) => updateCrookArrayValue('manual_crook_amplitudes', idx, e.target.value, defaultManualCrookAmplitudes, true)}
                                                />
                                            </label>
                                            <label className="field">
                                                <span>{`Shift z0${idx + 1} (mm)`}</span>
                                                <input
                                                    type="number"
                                                    step="1"
                                                    value={toNumber(manualCrookShifts[idx], 0)}
                                                    onChange={(e) => updateCrookArrayValue('manual_crook_shifts_mm', idx, e.target.value, defaultManualCrookShiftsMm, false)}
                                                />
                                            </label>
                                        </div>
                                        <div className="field-grid">
                                            <label className="field">
                                                <span>{`Order${idx + 1}`}</span>
                                                <input
                                                    type="number"
                                                    min={1}
                                                    step={1}
                                                    value={toInt(manualCrookOrders[idx], idx + 1)}
                                                    onChange={(e) => updateCrookIntArrayValue('manual_crook_orders', idx, e.target.value, defaultManualCrookOrders, 1)}
                                                />
                                            </label>
                                            <label className="field">
                                                <span>{`Theta${idx + 1} (deg)`}</span>
                                                <input
                                                    type="number"
                                                    step="1"
                                                    value={toNumber(manualCrookThetas[idx], 0)}
                                                    onChange={(e) => updateCrookArrayValue('manual_crook_thetas_deg', idx, e.target.value, defaultManualCrookThetasDeg, false)}
                                                />
                                            </label>
                                        </div>
                                    </div>
                                ))}
                            </>
                        )}
                    </section>
                )}

                {activeSection === 'simulation' && (
                    <section className="panel-section">
                        <h3>Simulation</h3>
                        <p className="section-copy">Define per-axis element size in mm, deterministic seed, and compute mode.</p>
                        <div className="field-grid">
                            <label className="field">
                                <span>Element Size X (mm)</span>
                                <input type="number" min={0.05} step={0.05} value={toNumber(config.mesh_size_x_mm, 2.0)} onChange={(e) => handleChange('mesh_size_x_mm', e.target.value, 'number')} />
                            </label>
                            <label className="field">
                                <span>Element Size Y (mm)</span>
                                <input type="number" min={0.05} step={0.05} value={toNumber(config.mesh_size_y_mm, 2.0)} onChange={(e) => handleChange('mesh_size_y_mm', e.target.value, 'number')} />
                            </label>
                        </div>
                        <div className="field-grid single">
                            <label className="field">
                                <span>Element Size Z (mm)</span>
                                <input type="number" min={0.05} step={0.05} value={toNumber(config.mesh_size_z_mm, 2.0)} onChange={(e) => handleChange('mesh_size_z_mm', e.target.value, 'number')} />
                            </label>
                        </div>

                        <label className="toggle">
                            <input type="checkbox" checked={config.use_seed} onChange={(e) => handleChange('use_seed', e.target.checked, 'bool')} />
                            <span>Use Deterministic Seed</span>
                        </label>

                        <label className="field">
                            <span>Simulation Seed</span>
                            <input type="number" step="1" value={config.simulation_seed} onChange={(e) => handleChange('simulation_seed', e.target.value, 'int')} />
                        </label>

                        <label className="toggle">
                            <input
                                type="checkbox"
                                checked={!demoMode && !!config.use_gpu}
                                disabled={!!demoMode}
                                onChange={(e) => handleChange('use_gpu', e.target.checked, 'bool')}
                            />
                            <span>{demoMode ? 'Use GPU When Available (Demo Disabled)' : 'Use GPU When Available'}</span>
                        </label>
                    </section>
                )}

                {activeSection === 'knots' && (
                    <section className="panel-section">
                        <h3>Knot System</h3>
                        <p className="section-copy">Switch between random data-driven knots and direct knot input.</p>

                        <div className="field-grid">
                            <label className="field">
                                <span>Knot Inside Limit</span>
                                <input
                                    type="text"
                                    inputMode="decimal"
                                    value={knotInsideLimitDraft}
                                    onChange={(e) => handleKnotInsideLimitInput(e.target.value)}
                                    onBlur={commitKnotInsideLimitInput}
                                />
                            </label>
                            <label className="field">
                                <span>Input Knot Count</span>
                                <input type="number" min={0} step="1" value={config.input_knot_count} onChange={(e) => setKnotCount(e.target.value)} />
                            </label>
                        </div>
                        <div className="field-grid">
                            <label className="field">
                                <span>Soft Clamp alpha</span>
                                <input type="number" step="0.01" value={toNumber(config.soft_clamp_alpha, 1.0)} onChange={(e) => handleChange('soft_clamp_alpha', e.target.value, 'number')} />
                            </label>
                            <label className="field">
                                <span>Soft Clamp pmin</span>
                                <input type="number" step="0.01" value={toNumber(config.soft_clamp_pmin, 2.0)} onChange={(e) => handleChange('soft_clamp_pmin', e.target.value, 'number')} />
                            </label>
                        </div>

                        <label className="toggle">
                            <input type="checkbox" checked={config.include_knot_dev} onChange={(e) => handleChange('include_knot_dev', e.target.checked, 'bool')} />
                            <span>Include Flow Deviation</span>
                        </label>

                        <label className="toggle">
                            <input type="checkbox" checked={config.dead_knots} onChange={(e) => handleChange('dead_knots', e.target.checked, 'bool')} />
                            <span>Dead Knots</span>
                        </label>

                        <label className="toggle">
                            <input type="checkbox" checked={config.use_input_knots} onChange={(e) => handleChange('use_input_knots', e.target.checked, 'bool')} />
                            <span>Use Manual Knot Sequence</span>
                        </label>

                        <div className="field-grid">
                            <div className="field">
                                <span>Override Knot c1/c2</span>
                                <label className="toggle" style={{ marginTop: 6 }}>
                                    <input
                                        type="checkbox"
                                        checked={!!config.knot_sequence_override_c1_c2}
                                        onChange={(e) => handleChange('knot_sequence_override_c1_c2', e.target.checked, 'bool')}
                                    />
                                    <span>Use c1=-1.458e-3 and c2 from Ax100 model for all knots</span>
                                </label>
                            </div>
                        </div>

                        {!config.use_input_knots && (
                            <>
                                <div className="field-grid">
                                    <label className="field">
                                        <span>L100 Min</span>
                                        <input type="number" step="1" value={config.L100_min} onChange={(e) => handleChange('L100_min', e.target.value, 'number')} />
                                    </label>
                                    <label className="field">
                                        <span>L100 Max</span>
                                        <input type="number" step="1" value={config.L100_max} onChange={(e) => handleChange('L100_max', e.target.value, 'number')} />
                                    </label>
                                </div>
                                <div className="field-grid single">
                                    <label className="field">
                                        <span>Min RD-RL Gap (Generated, mm)</span>
                                        <input
                                            type="number"
                                            min={0}
                                            step="1"
                                            value={toNumber(config.knot_generator_min_rd_minus_rl_mm, 30.0)}
                                            onChange={(e) => handleChange('knot_generator_min_rd_minus_rl_mm', e.target.value, 'number')}
                                        />
                                    </label>
                                </div>
                                <div className="field-grid single">
                                    <label className="field">
                                        <span>Sequence Top-K</span>
                                        <input
                                            type="number"
                                            min={0}
                                            step="1"
                                            value={toInt(config.knot_sequence_top_k, 0)}
                                            onChange={(e) => handleChange('knot_sequence_top_k', e.target.value, 'int')}
                                        />
                                    </label>
                                </div>
                                <div className="field-grid single">
                                    <label className="field">
                                        <span>Sequence Top-P</span>
                                        <input
                                            type="number"
                                            min={0}
                                            max={1}
                                            step="0.01"
                                            value={toNumber(config.knot_sequence_top_p, 0.80)}
                                            onChange={(e) => handleChange('knot_sequence_top_p', e.target.value, 'number')}
                                        />
                                    </label>
                                </div>
                                <div className="field-grid single">
                                    <label className="field">
                                        <span>Dictionary Jitter</span>
                                        <input
                                            type="number"
                                            min={0}
                                            step="0.01"
                                            value={toNumber(config.knot_dictionary_jitter, 0)}
                                            onChange={(e) => handleChange('knot_dictionary_jitter', e.target.value, 'number')}
                                        />
                                    </label>
                                </div>
                            </>
                        )}

                        {config.use_input_knots && (
                            <div className="knot-manager">
                                <div className="knot-toolbar">
                                    <button type="button" onClick={addKnot}>Add Knot</button>
                                    <button type="button" onClick={duplicateSelectedKnot} disabled={manualKnots.length === 0}>Duplicate</button>
                                    <button type="button" onClick={removeSelectedKnot} disabled={manualKnots.length === 0}>Remove</button>
                                    <button type="button" onClick={resetSelectedKnot} disabled={manualKnots.length === 0}>Reset</button>
                                    <span className="knot-count">Total: {manualKnots.length}</span>
                                </div>

                                <div className="knot-layout">
                                    <div className="knot-list">
                                        {manualKnots.length === 0 && (
                                            <p className="section-copy">No knots defined. Add one to start editing.</p>
                                        )}
                                        {manualKnots.map((knot, index) => (
                                            <button
                                                key={`knot-${index}`}
                                                type="button"
                                                className={`knot-chip ${selectedKnotIndex === index ? 'active' : ''}`}
                                                onClick={() => setSelectedKnotIndex(index)}
                                            >
                                                <strong>Knot {index + 1}</strong>
                                                <small>th0 {toNumber(knot.th0_deg).toFixed(1)} deg | z0 {toNumber(knot.z0).toFixed(1)} | L100 {toNumber(knot.L100).toFixed(1)}</small>
                                            </button>
                                        ))}
                                    </div>

                                    <div className="knot-editor">
                                        {selectedKnot && (
                                            <>
                                                <h4>Selected Knot Parameters</h4>
                                                <div className="field-grid">
                                                    {knotParamFields.map((field) => (
                                                        <label className="field" key={field.key}>
                                                            <span>{field.label}</span>
                                                            <input
                                                                type="number"
                                                                step={field.step}
                                                                value={selectedKnot[field.key]}
                                                                disabled={
                                                                    !!config.knot_sequence_override_c1_c2 &&
                                                                    (field.key === 'c1' || field.key === 'c2')
                                                                }
                                                                onChange={(e) => handleKnotParamChange(selectedKnotIndex, field.key, e.target.value)}
                                                            />
                                                        </label>
                                                    ))}
                                                </div>
                                            </>
                                        )}
                                    </div>
                                </div>
                            </div>
                        )}
                    </section>
                )}

                {activeSection === 'visuals' && (
                    <section className="panel-section">
                        <h3>Visualization</h3>
                        <p className="section-copy">Tune what gets rendered and how visible each feature is.</p>
                        {isLogMode && (
                            <p className="section-copy">
                                In log mode, fibers are hidden, but rings, contours, knots, and board visibility can be toggled.
                            </p>
                        )}

                        <div className="toggle-grid">
                            <label className="toggle">
                                <input
                                    type="checkbox"
                                    checked={!!config.display_contours}
                                    onChange={(e) => handleChange('display_contours', e.target.checked, 'bool')}
                                />
                                <span>Display Contours</span>
                            </label>
                            <label className="toggle">
                                <input type="checkbox" checked={!!config.display_surface_mesh} onChange={(e) => handleChange('display_surface_mesh', e.target.checked, 'bool')} />
                                <span>Display Mesh (4 Surfaces)</span>
                            </label>
                            <label className="toggle">
                                <input type="checkbox" checked={!!config.display_board} onChange={(e) => handleChange('display_board', e.target.checked, 'bool')} />
                                <span>Display Board</span>
                            </label>
                            <label className="toggle">
                                <input
                                    type="checkbox"
                                    checked={!!config.display_rings}
                                    onChange={(e) => handleChange('display_rings', e.target.checked, 'bool')}
                                />
                                <span>Display Rings (3D, requires new Generate Board)</span>
                            </label>
                            <label className="toggle">
                                <input
                                    type="checkbox"
                                    checked={!!config.display_knots}
                                    onChange={(e) => handleChange('display_knots', e.target.checked, 'bool')}
                                />
                                <span>Display Knots</span>
                            </label>
                            <label className="toggle">
                                <input
                                    type="checkbox"
                                    checked={!!config.display_knot_slots}
                                    onChange={(e) => handleChange('display_knot_slots', e.target.checked, 'bool')}
                                />
                                <span>Display Knot Slots</span>
                            </label>
                            <label className="toggle">
                                <input type="checkbox" checked={config.display_rings_inside_knots} onChange={(e) => handleChange('display_rings_inside_knots', e.target.checked, 'bool')} />
                                <span>Rings Inside Knots</span>
                            </label>
                            <label className="toggle">
                                <input type="checkbox" checked={config.display_pith} onChange={(e) => handleChange('display_pith', e.target.checked, 'bool')} />
                                <span>Display Pith (live innermost ring)</span>
                            </label>
                            <label className="toggle">
                                <input type="checkbox" checked={config.display_knot_axes} onChange={(e) => handleChange('display_knot_axes', e.target.checked, 'bool')} />
                                <span>Display Knot Axes</span>
                            </label>
                            <label className="toggle">
                                <input
                                    type="checkbox"
                                    checked={!!showNormalOverlay}
                                    disabled={!normalOverlayAvailable}
                                    onChange={(e) => onToggleNormalOverlay && onToggleNormalOverlay(e.target.checked)}
                                />
                                <span>Show Normal Overlay</span>
                            </label>
                            <label className="toggle">
                                <input
                                    type="checkbox"
                                    checked={!!config.display_normal_vectors_surface}
                                    onChange={(e) => handleChange('display_normal_vectors_surface', e.target.checked, 'bool')}
                                />
                                <span>Display Normal Vectors</span>
                            </label>
                            <label className="toggle">
                                <input
                                    type="checkbox"
                                    checked={!!showFiberOutOfPlaneOverlay}
                                    disabled={!fiberOutOfPlaneOverlayAvailable}
                                    onChange={(e) => onToggleFiberOutOfPlaneOverlay && onToggleFiberOutOfPlaneOverlay(e.target.checked)}
                                />
                                <span>Show Fiber Out-of-Plane Overlay</span>
                            </label>
                            <label className="toggle">
                                <input
                                    type="checkbox"
                                    checked={!!showPhotorealisticOverlay}
                                    disabled={!photorealisticOverlayAvailable}
                                    onChange={(e) => onTogglePhotorealisticOverlay && onTogglePhotorealisticOverlay(e.target.checked)}
                                />
                                <span>Show Photorealistic Overlay</span>
                            </label>
                        </div>

                        <p className="section-copy">
                            Display Rings (3D) updates only after Generate Board. Display Pith updates on the fly.
                        </p>
                        <p className="section-copy">
                            Normal overlay becomes available after Generate Board and shows RGB-encoded surface normal components.
                        </p>
                        <p className="section-copy">
                            Normal vectors use the same board-surface samples and are drawn as 3D arrows on the board faces.
                        </p>
                        <p className="section-copy">
                            Fiber out-of-plane overlay becomes available after Generate Board with fibers enabled.
                        </p>
                        <p className="section-copy">
                            Photorealistic overlay becomes available after you run `Export Photorealistic Surfaces`.
                        </p>

                        <label className="field">
                            <span>Board Opacity</span>
                            <div className="range-row">
                                <input type="range" min={0} max={1} step={0.01} value={config.board_opacity} onChange={(e) => handleChange('board_opacity', e.target.value, 'number')} />
                                <strong>{toNumber(config.board_opacity).toFixed(2)}</strong>
                            </div>
                        </label>

                        <div className="field-grid single">
                            <label className="field">
                                <span>Contour Line Width</span>
                                <div className="range-row">
                                    <input type="range" min={0.5} max={16} step={0.5} value={config.contour_line_width} onChange={(e) => handleChange('contour_line_width', e.target.value, 'number')} />
                                    <strong>{toNumber(config.contour_line_width).toFixed(1)}</strong>
                                </div>
                            </label>
                        </div>

                        <div className="field-grid">
                            <label className="field">
                                <span>Fiber Display</span>
                                <select
                                    value={isLogMode ? 0 : config.quiver_or_stream}
                                    disabled={isLogMode}
                                    onChange={(e) => handleChange('quiver_or_stream', e.target.value, 'int')}
                                >
                                    <option value={0}>Off</option>
                                    <option value={1}>Quiver 3D (on surfaces)</option>
                                    <option value={2}>Quiver 3D (volume)</option>
                                    <option value={3}>Quiver 2D</option>
                                </select>
                            </label>
                            <label className="field">
                                <span>Fiber Line Width</span>
                                <div className="range-row">
                                    <input type="range" min={0.5} max={12} step={0.5} value={config.fiber_line_width ?? 2} onChange={(e) => handleChange('fiber_line_width', e.target.value, 'number')} />
                                    <strong>{toNumber(config.fiber_line_width ?? 2).toFixed(1)}</strong>
                                </div>
                            </label>
                        </div>
                    </section>
                )}

                {activeSection === 'fibers' && (
                    <section className="panel-section">
                        <h3>Fibers</h3>
                        <p className="section-copy">Adjust solver behavior, display mode, and noise controls.</p>
                        {isLogMode && (
                            <p className="section-copy">Fiber computation is disabled in log mode.</p>
                        )}

                        <label className="toggle">
                            <input
                                type="checkbox"
                                checked={isLogMode ? false : !!config.calc_fibers}
                                disabled={isLogMode}
                                onChange={(e) => handleChange('calc_fibers', e.target.checked, 'bool')}
                            />
                            <span>Calculate Fibers</span>
                        </label>

                        <label className="toggle">
                            <input type="checkbox" checked={config.knot_fiber_field_override} onChange={(e) => handleChange('knot_fiber_field_override', e.target.checked, 'bool')} />
                            <span>Knot Fiber Field Override</span>
                        </label>

                        <label className="toggle">
                            <input
                                type="checkbox"
                                checked={!!config.knot_fiber_disable_dead_override}
                                onChange={(e) => handleChange('knot_fiber_disable_dead_override', e.target.checked, 'bool')}
                            />
                            <span>Disable Override In Dead Knots</span>
                        </label>

                        <label className="toggle">
                            <input type="checkbox" checked={!!config.knot_fiber_reverse_above_axis} onChange={(e) => handleChange('knot_fiber_reverse_above_axis', e.target.checked, 'bool')} />
                            <span>Reverse Override Above Knot Axis</span>
                        </label>

                        <div className="field-grid single">
                            <label className="field">
                                <span>a0 Method</span>
                                <select value={config.calc_fibers_a0_method} onChange={(e) => handleChange('calc_fibers_a0_method', e.target.value, 'int')}>
                                    <option value={1}>Exact</option>
                                    <option value={2}>Approximate</option>
                                </select>
                            </label>
                            <label className="field">
                                <span>Multi-Knot Selection</span>
                                <select
                                    value={config.multi_knot_fiber_selection_rule || 'weighted_deviation'}
                                    onChange={(e) => handleChange('multi_knot_fiber_selection_rule', e.target.value)}
                                >
                                    <option value="weighted_deviation">Weighted Deviation</option>
                                    <option value="longitudinal">Original Longitudinal</option>
                                </select>
                            </label>
                        </div>

                        <label className="toggle">
                            <input type="checkbox" checked={config.rand_fibers} onChange={(e) => handleChange('rand_fibers', e.target.checked, 'bool')} />
                            <span>Randomize Fibers</span>
                        </label>

                        <div className="field-grid">
                            <label className="field">
                                <span>Out-of-Plane Threshold</span>
                                <input type="number" step="0.05" value={config.out_of_plane_threshold} onChange={(e) => handleChange('out_of_plane_threshold', e.target.value, 'number')} />
                            </label>
                            <label className="field">
                                <span>SNR (dB)</span>
                                <input type="number" step="0.5" value={config.snr} onChange={(e) => handleChange('snr', e.target.value, 'number')} />
                            </label>
                        </div>

                    </section>
                )}

                {activeSection === 'export' && (
                    <section className="panel-section">
                        <h3>Export</h3>
                        <p className="section-copy">Configure and export a MATLAB-style rings + fibers image bundle.</p>

                        <label className="toggle">
                            <input
                                type="checkbox"
                                checked={!!config.export_show_rings_inside_knots}
                                onChange={(e) => handleChange('export_show_rings_inside_knots', e.target.checked, 'bool')}
                            />
                            <span>Show Rings Inside Knots</span>
                        </label>

                        <div className="field-grid">
                            <label className="field">
                                <span>Contour Line Thickness (px)</span>
                                <input
                                    type="number"
                                    min={1}
                                    step="1"
                                    value={toNumber(config.export_contour_line_width, 1)}
                                    onChange={(e) => handleChange('export_contour_line_width', e.target.value, 'number')}
                                />
                            </label>
                            <label className="field">
                                <span>Surface Blur Sigma</span>
                                <input
                                    type="number"
                                    min={0}
                                    step="0.05"
                                    value={toNumber(config.export_surface_blur_sigma, 0)}
                                    onChange={(e) => handleChange('export_surface_blur_sigma', e.target.value, 'number')}
                                />
                            </label>
                            <label className="field">
                                <span>Fiber Blur Sigma</span>
                                <input
                                    type="number"
                                    min={0}
                                    step="0.05"
                                    value={toNumber(config.export_fiber_blur_sigma, 0)}
                                    onChange={(e) => handleChange('export_fiber_blur_sigma', e.target.value, 'number')}
                                />
                            </label>
                            <label className="field">
                                <span>Fiber Irregularity</span>
                                <input
                                    type="number"
                                    min={0}
                                    max={2}
                                    step="0.05"
                                    value={toNumber(config.export_fiber_irregularity_strength, 0.35)}
                                    onChange={(e) => handleChange('export_fiber_irregularity_strength', e.target.value, 'number')}
                                />
                            </label>
                            <label className="field">
                                <span>Ring Irregularity</span>
                                <input
                                    type="number"
                                    min={0}
                                    max={2}
                                    step="0.05"
                                    value={toNumber(config.export_ring_irregularity_strength, 0.40)}
                                    onChange={(e) => handleChange('export_ring_irregularity_strength', e.target.value, 'number')}
                                />
                            </label>
                        </div>

                        <div className="field-grid single">
                            <label className="field">
                                <span>MATLAB Image ID (imid)</span>
                                <input
                                    type="number"
                                    min={0}
                                    step="1"
                                    value={toInt(config.imid, 1)}
                                    onChange={(e) => handleChange('imid', e.target.value, 'int')}
                                />
                            </label>
                        </div>

                        <button
                            type="button"
                            className="export-btn"
                            onClick={onExportMatlabBundle}
                            disabled={!!exportMatlabBundleDisabled}
                        >
                            {exportMatlabBundleLoading ? 'Exporting...' : 'Export Rings + Fibers Bundle'}
                        </button>

                        <p className="section-copy export-note">
                            Output ZIP uses MATLAB folder layout: `output/rings_1..4/00001.png` and `output/fiber_1..4/00001.png`.
                        </p>

                        <button
                            type="button"
                            className="export-btn"
                            onClick={onExportMatlabBundleWithMiddle}
                            disabled={!!exportMatlabBundleDisabled}
                        >
                            {exportMatlabBundleLoading ? 'Exporting...' : 'Export Rings + Fibers + Middle Surface'}
                        </button>

                        <p className="section-copy export-note">
                            Same MATLAB bundle plus middle XZ ring contours in `output/rings_5/00001.png` (total ring images: 5).
                        </p>

                        <button
                            type="button"
                            className="export-btn"
                            onClick={onPreloadPhotorealistic}
                            disabled={!!preloadPhotorealisticDisabled}
                        >
                            {preloadPhotorealisticLoading
                                ? 'Loading Model...'
                                : (photorealisticLoaded ? 'Photorealistic Model Loaded' : 'Load Photorealistic Model')}
                        </button>

                        <p className="section-copy export-note">
                            Preloads the diffusion model so image generation starts immediately when you export.
                        </p>

                        <div className="field-grid">
                            <label className="field">
                                <span>Guidance Scale</span>
                                <input
                                    type="number"
                                    min={0}
                                    step="0.1"
                                    value={toNumber(config.photorealistic_guidance_scale, 2.0)}
                                    onChange={(e) => handleChange('photorealistic_guidance_scale', e.target.value, 'number')}
                                />
                            </label>
                            <label className="field">
                                <span>Img2Img Strength</span>
                                <input
                                    type="number"
                                    min={0}
                                    max={1}
                                    step="0.05"
                                    value={toNumber(config.photorealistic_img2img_strength, 0.0)}
                                    onChange={(e) => handleChange('photorealistic_img2img_strength', e.target.value, 'number')}
                                />
                            </label>
                            <label className="field">
                                <span>Diffusion Steps</span>
                                <input
                                    type="number"
                                    min={1}
                                    step="1"
                                    value={toInt(config.photorealistic_steps, 50)}
                                    onChange={(e) => handleChange('photorealistic_steps', e.target.value, 'int')}
                                />
                            </label>
                        </div>

                        <label className="toggle">
                            <input
                                type="checkbox"
                                checked={!!config.photorealistic_use_rings_only}
                                onChange={(e) => handleChange('photorealistic_use_rings_only', e.target.checked, 'bool')}
                            />
                            <span>Use Rings Only (No Fiber Maps)</span>
                        </label>

                        <label className="toggle">
                            <input
                                type="checkbox"
                                checked={!!config.photorealistic_include_knot_maps}
                                disabled={!!config.photorealistic_use_rings_only}
                                onChange={(e) => handleChange('photorealistic_include_knot_maps', e.target.checked, 'bool')}
                            />
                            <span>Include Knot Maps (from fibers)</span>
                        </label>

                        <p className="section-copy export-note">
                            Guidance Scale: Higher values keep the result closer to your conditioning maps (rings, and optionally fibers). Lower values allow more variation in look.
                        </p>
                        <p className="section-copy export-note">
                            Img2Img Strength: Higher values preserve more of the input image layout. Lower values let the model change the surface more.
                        </p>
                        <p className="section-copy export-note">
                            Diffusion Steps: More steps can give cleaner details, but takes longer. Fewer steps are faster.
                        </p>

                        <button
                            type="button"
                            className="export-btn"
                            onClick={onExportPhotorealistic}
                            disabled={!!exportPhotorealisticDisabled}
                        >
                            {exportPhotorealisticLoading ? 'Generating...' : 'Export Photorealistic Surfaces'}
                        </button>

                        <p className="section-copy export-note">
                            Generates 4 diffusion-model surfaces and applies them as overlays on the 3D side faces.
                        </p>

                        <button
                            type="button"
                            className="export-btn"
                            onClick={onDownloadPhotorealisticZip}
                            disabled={!!downloadPhotorealisticZipDisabled}
                        >
                            Download Photorealistic ZIP
                        </button>

                        <p className="section-copy export-note">
                            Downloads the most recently generated 4 photorealistic surfaces as a single ZIP file.
                        </p>

                        {!photorealisticAvailable && !!String(photorealisticReason || '').trim() && (
                            <p className="section-copy export-note export-warning">
                                Photorealistic export warning: {photorealisticReason}
                            </p>
                        )}
                        <button
                            type="button"
                            className="export-btn"
                            onClick={onExportMat}
                            disabled={!!exportMatDisabled}
                        >
                            {exportMatLoading ? 'Exporting...' : 'Export MATLAB .mat Data'}
                        </button>

                        <div className="config-action-row">
                            <button
                                type="button"
                                className="export-btn"
                                onClick={onSaveConfig}
                                disabled={loading}
                            >
                                Save UI Config
                            </button>
                            <button
                                type="button"
                                className="export-btn"
                                onClick={() => loadConfigInputRef.current?.click()}
                                disabled={loading}
                            >
                                Load UI Config
                            </button>
                            <input
                                ref={loadConfigInputRef}
                                type="file"
                                accept=".json,application/json"
                                onChange={handleLoadConfigFile}
                                className="hidden-file-input"
                            />
                        </div>
                    </section>
                )}
            </div>

            <div className="actions">
                <button className="simulate-btn" onClick={onSimulate} disabled={loading} type="button">
                    {loading ? 'Simulating...' : 'Generate Board'}
                </button>
            </div>
        </div>
    );
};

export default ControlPanel;
