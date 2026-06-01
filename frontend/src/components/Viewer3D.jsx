import React, { useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, PerspectiveCamera, OrthographicCamera, GizmoHelper, GizmoViewport, Line } from '@react-three/drei';
import * as THREE from 'three';

/* Knot-surface stain overlay intentionally disabled permanently. */
const KnotSurfaceStains = () => null;

const BoardOutline = ({ outline, boardOpacity, knots }) => {
    const { min, max } = outline;
    const sizeX = max[0] - min[0];
    const sizeY = max[1] - min[1];
    const sizeZ = max[2] - min[2];
    const size = [sizeX, sizeY, sizeZ];
    const center = [(min[0] + max[0]) / 2, (min[1] + max[1]) / 2, (min[2] + max[2]) / 2];
    const opacity = Math.min(1, Math.max(0, boardOpacity ?? 0.12));
    const enableOcclusion = opacity > 0.02;
    const edgeSegments = useMemo(() => {
        const hx = sizeX * 0.5;
        const hy = sizeY * 0.5;
        const hz = sizeZ * 0.5;
        return [
            [-hx, -hy, -hz], [hx, -hy, -hz],
            [hx, -hy, -hz], [hx, hy, -hz],
            [hx, hy, -hz], [-hx, hy, -hz],
            [-hx, hy, -hz], [-hx, -hy, -hz],

            [-hx, -hy, hz], [hx, -hy, hz],
            [hx, -hy, hz], [hx, hy, hz],
            [hx, hy, hz], [-hx, hy, hz],
            [-hx, hy, hz], [-hx, -hy, hz],

            [-hx, -hy, -hz], [-hx, -hy, hz],
            [hx, -hy, -hz], [hx, -hy, hz],
            [hx, hy, -hz], [hx, hy, hz],
            [-hx, hy, -hz], [-hx, hy, hz],
        ];
    }, [sizeX, sizeY, sizeZ]);

    return (
        <>
            <group position={center}>
                {/* Semi-transparent white fill */}
                <mesh>
                    <boxGeometry args={size} />
                    <meshStandardMaterial
                        color="#ffffff"
                        emissive="#ffffff"
                        emissiveIntensity={0.34}
                        roughness={0.9}
                        metalness={0.0}
                        transparent={opacity < 0.999}
                        opacity={opacity}
                        side={THREE.FrontSide}
                        depthWrite={enableOcclusion}
                        polygonOffset
                        polygonOffsetFactor={2}
                        polygonOffsetUnits={2}
                    />
                </mesh>
                {/* 3D view-only board boundary edges */}
                <Line
                    points={edgeSegments}
                    segments
                    color="#000000"
                    lineWidth={2.4}
                    transparent
                    opacity={0.98}
                    depthTest={enableOcclusion}
                    depthWrite={false}
                    renderOrder={1100}
                />
            </group>
            <KnotSurfaceStains knots={knots} min={min} max={max} boardOpacity={boardOpacity} />
        </>
    );
};

const useBoardFaceOverlayGeometry = (boardOutline) => useMemo(() => {
    const min = boardOutline?.min;
    const max = boardOutline?.max;
    if (!Array.isArray(min) || !Array.isArray(max) || min.length !== 3 || max.length !== 3) {
        return [];
    }

    const [x0, y0, z0] = min;
    const [x1, y1, z1] = max;
    const maxDim = Math.max(Math.abs(x1 - x0), Math.abs(y1 - y0), Math.abs(z1 - z0), 1e-6);
    const eps = Math.max(0.04, 0.001 * maxDim);
    const uv = new Float32Array([0, 0, 1, 0, 1, 1, 0, 1]);
    const indicesStd = new Uint16Array([0, 1, 2, 0, 2, 3]);
    const indicesRev = new Uint16Array([0, 2, 1, 0, 3, 2]);
    const mkFace = (key, verts, reverse = false) => ({
        key,
        positions: new Float32Array(verts.flat()),
        uvs: uv,
        indices: reverse ? indicesRev : indicesStd,
    });

    return [
        // z faces: u -> X(width), v -> Y(length)
        mkFace('z_max', [
            [x0, y0, z1 + eps],
            [x1, y0, z1 + eps],
            [x1, y1, z1 + eps],
            [x0, y1, z1 + eps],
        ], false),
        mkFace('z_min', [
            [x0, y0, z0 - eps],
            [x1, y0, z0 - eps],
            [x1, y1, z0 - eps],
            [x0, y1, z0 - eps],
        ], true),
        // x faces: u -> Z(thickness), v -> Y(length)
        mkFace('x_max', [
            [x1 + eps, y0, z0],
            [x1 + eps, y0, z1],
            [x1 + eps, y1, z1],
            [x1 + eps, y1, z0],
        ], true),
        mkFace('x_min', [
            [x0 - eps, y0, z0],
            [x0 - eps, y0, z1],
            [x0 - eps, y1, z1],
            [x0 - eps, y1, z0],
        ], false),
    ];
}, [boardOutline]);

const NormalFaceOverlays = ({ boardOutline, overlays, visible }) => {
    const faces = useBoardFaceOverlayGeometry(boardOutline);
    const texturesByFace = useMemo(() => {
        const loader = new THREE.TextureLoader();
        const next = {};
        for (const faceKey of ['x_min', 'x_max', 'z_min', 'z_max']) {
            const source = typeof overlays?.[faceKey]?.src === 'string' ? overlays[faceKey].src.trim() : '';
            if (!source) continue;
            const tex = loader.load(source);
            tex.colorSpace = THREE.SRGBColorSpace;
            tex.wrapS = THREE.ClampToEdgeWrapping;
            tex.wrapT = THREE.ClampToEdgeWrapping;
            tex.repeat.set(1, 1);
            tex.offset.set(0, 0);
            tex.needsUpdate = true;
            next[faceKey] = tex;
        }
        return next;
    }, [overlays]);

    useEffect(() => () => {
        Object.values(texturesByFace).forEach((tex) => tex.dispose());
    }, [texturesByFace]);

    if (!visible || faces.length === 0) return null;

    return (
        <group renderOrder={930}>
            {faces.map((face) => {
                const texture = texturesByFace[face.key];
                if (!texture) return null;
                return (
                    <mesh key={`normal-overlay-${face.key}`} renderOrder={930}>
                        <bufferGeometry>
                            <bufferAttribute attach="attributes-position" args={[face.positions, 3]} />
                            <bufferAttribute attach="attributes-uv" args={[face.uvs, 2]} />
                            <bufferAttribute attach="index" args={[face.indices, 1]} />
                        </bufferGeometry>
                        <meshBasicMaterial
                            map={texture}
                            transparent
                            opacity={0.92}
                            side={THREE.FrontSide}
                            depthTest
                            depthWrite={false}
                            polygonOffset
                            polygonOffsetFactor={-1}
                            polygonOffsetUnits={-1}
                            toneMapped={false}
                        />
                    </mesh>
                );
            })}
        </group>
    );
};

const FiberOutOfPlaneFaceOverlays = ({ boardOutline, overlays, visible }) => {
    const faces = useBoardFaceOverlayGeometry(boardOutline);
    const texturesByFace = useMemo(() => {
        const loader = new THREE.TextureLoader();
        const next = {};
        for (const faceKey of ['x_min', 'x_max', 'z_min', 'z_max']) {
            const source = typeof overlays?.[faceKey]?.src === 'string' ? overlays[faceKey].src.trim() : '';
            if (!source) continue;
            const tex = loader.load(source);
            tex.colorSpace = THREE.SRGBColorSpace;
            tex.wrapS = THREE.ClampToEdgeWrapping;
            tex.wrapT = THREE.ClampToEdgeWrapping;
            tex.repeat.set(1, 1);
            tex.offset.set(0, 0);
            tex.needsUpdate = true;
            next[faceKey] = tex;
        }
        return next;
    }, [overlays]);

    useEffect(() => () => {
        Object.values(texturesByFace).forEach((tex) => tex.dispose());
    }, [texturesByFace]);

    if (!visible || faces.length === 0) return null;

    return (
        <group renderOrder={940}>
            {faces.map((face) => {
                const texture = texturesByFace[face.key];
                if (!texture) return null;
                return (
                    <mesh key={`fiber-oop-overlay-${face.key}`} renderOrder={940}>
                        <bufferGeometry>
                            <bufferAttribute attach="attributes-position" args={[face.positions, 3]} />
                            <bufferAttribute attach="attributes-uv" args={[face.uvs, 2]} />
                            <bufferAttribute attach="index" args={[face.indices, 1]} />
                        </bufferGeometry>
                        <meshBasicMaterial
                            map={texture}
                            transparent
                            opacity={0.9}
                            side={THREE.FrontSide}
                            depthTest
                            depthWrite={false}
                            polygonOffset
                            polygonOffsetFactor={-1.2}
                            polygonOffsetUnits={-1.2}
                            toneMapped={false}
                        />
                    </mesh>
                );
            })}
        </group>
    );
};

const PhotorealisticFaceOverlays = ({ boardOutline, overlays, visible }) => {
    const faces = useBoardFaceOverlayGeometry(boardOutline);
    const texturesByFace = useMemo(() => {
        const loader = new THREE.TextureLoader();
        const next = {};
        for (const faceKey of ['x_min', 'x_max', 'z_min', 'z_max']) {
            const source = typeof overlays?.[faceKey]?.src === 'string' ? overlays[faceKey].src.trim() : '';
            if (!source) continue;
            const flipX = !!overlays?.[faceKey]?.flipX;
            const tex = loader.load(source);
            tex.colorSpace = THREE.SRGBColorSpace;
            tex.wrapS = flipX ? THREE.RepeatWrapping : THREE.ClampToEdgeWrapping;
            // View-only vertical flip for all photorealistic overlays.
            tex.wrapT = THREE.RepeatWrapping;
            tex.repeat.set(flipX ? -1 : 1, -1);
            tex.offset.set(flipX ? 1 : 0, 1);
            tex.needsUpdate = true;
            next[faceKey] = tex;
        }
        return next;
    }, [overlays]);

    useEffect(() => () => {
        Object.values(texturesByFace).forEach((tex) => tex.dispose());
    }, [texturesByFace]);

    if (!visible || faces.length === 0) return null;

    return (
        <group renderOrder={1000}>
            {faces.map((face) => {
                const texture = texturesByFace[face.key];
                if (!texture) return null;
                return (
                    <mesh key={`photorealistic-overlay-${face.key}`} renderOrder={1000}>
                        <bufferGeometry>
                            <bufferAttribute attach="attributes-position" args={[face.positions, 3]} />
                            <bufferAttribute attach="attributes-uv" args={[face.uvs, 2]} />
                            <bufferAttribute attach="index" args={[face.indices, 1]} />
                        </bufferGeometry>
                        <meshBasicMaterial
                            map={texture}
                            transparent
                            opacity={1}
                            side={THREE.FrontSide}
                            depthTest
                            depthWrite={false}
                            polygonOffset
                            polygonOffsetFactor={-2}
                            polygonOffsetUnits={-2}
                            toneMapped={false}
                        />
                    </mesh>
                );
            })}
        </group>
    );
};

/* ── Contour lines (ring patterns on board faces) ── */
const ContourLines = ({ contours, boardOpacity, contourLineWidth, boardOutline }) => {
    const lines = useMemo(() => {
        if (!Array.isArray(contours) || contours.length === 0) return [];
        const outlineMin = boardOutline?.min;
        const outlineMax = boardOutline?.max;
        const hasOutline = Array.isArray(outlineMin) && Array.isArray(outlineMax) && outlineMin.length === 3 && outlineMax.length === 3;
        const center = hasOutline
            ? [
                (outlineMin[0] + outlineMax[0]) * 0.5,
                (outlineMin[1] + outlineMax[1]) * 0.5,
                (outlineMin[2] + outlineMax[2]) * 0.5,
            ]
            : [0, 0, 0];
        const maxDim = hasOutline
            ? Math.max(
                Math.abs(outlineMax[0] - outlineMin[0]),
                Math.abs(outlineMax[1] - outlineMin[1]),
                Math.abs(outlineMax[2] - outlineMin[2]),
            )
            : 1.0;
        // Small outward nudge keeps contours from z-fighting against the white board face.
        const epsilon = hasOutline ? Math.max(0.05, 0.0015 * maxDim) : 0.0;

        return contours.map((line) => {
            if (!Array.isArray(line) || line.length < 2) return [];

            let processed = line.map((p) => [p[0], p[1], p[2]]);

            if (hasOutline && epsilon > 0) {
                const mins = [Infinity, Infinity, Infinity];
                const maxs = [-Infinity, -Infinity, -Infinity];
                const sums = [0, 0, 0];
                for (const p of processed) {
                    if (p[0] < mins[0]) mins[0] = p[0];
                    if (p[1] < mins[1]) mins[1] = p[1];
                    if (p[2] < mins[2]) mins[2] = p[2];
                    if (p[0] > maxs[0]) maxs[0] = p[0];
                    if (p[1] > maxs[1]) maxs[1] = p[1];
                    if (p[2] > maxs[2]) maxs[2] = p[2];
                    sums[0] += p[0];
                    sums[1] += p[1];
                    sums[2] += p[2];
                }

                const ranges = [maxs[0] - mins[0], maxs[1] - mins[1], maxs[2] - mins[2]];
                let normalAxis = 0;
                if (ranges[1] < ranges[normalAxis]) normalAxis = 1;
                if (ranges[2] < ranges[normalAxis]) normalAxis = 2;
                const avgAxis = sums[normalAxis] / processed.length;
                const side = avgAxis >= center[normalAxis] ? 1 : -1;
                const offset = side * epsilon;
                processed = processed.map((p) => {
                    const q = [p[0], p[1], p[2]];
                    q[normalAxis] += offset;
                    return q;
                });
            }

            if (processed.length >= 4) {
                const a = processed[0];
                const b = processed[processed.length - 1];
                const closeDist = Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
                const isClosed = closeDist <= Math.max(1e-6, 2 * epsilon);
                const curve = new THREE.CatmullRomCurve3(
                    processed.map((p) => new THREE.Vector3(p[0], p[1], p[2])),
                    isClosed,
                    'catmullrom',
                    0.05
                );
                const sampleCount = Math.min(700, Math.max(processed.length * 2, 48));
                processed = curve.getPoints(sampleCount).map((v) => [v.x, v.y, v.z]);
            }

            return processed;
        }).filter((line) => line.length >= 2);
    }, [contours, boardOutline]);
    const enableOcclusion = (boardOpacity ?? 0) > 0.02;
    const width = Math.max(0.5, Number(contourLineWidth ?? 4));

    return (
        <group>
            {lines.map((points, i) => (
                <Line
                    key={i}
                    points={points}
                    color="#121212"
                    lineWidth={width}
                    transparent
                    opacity={1}
                    depthTest={enableOcclusion}
                    depthWrite={false}
                    renderOrder={12}
                />
            ))}
        </group>
    );
};

/* Board surface mesh lines (4 side faces) */
const SurfaceMeshLines = ({ boardOutline, meshAxes, boardOpacity, meshElementSize }) => {
    const segments = useMemo(() => {
        const min = boardOutline?.min;
        const max = boardOutline?.max;
        if (!Array.isArray(min) || !Array.isArray(max) || min.length !== 3 || max.length !== 3) {
            return [];
        }

        const [x0, y0, z0] = min;
        const [x1, y1, z1] = max;
        const axisCountFromSize = (vmin, vmax, h, fallback) => {
            const length = Math.abs(vmax - vmin);
            if (Number.isFinite(h) && h > 0) return Math.max(2, Math.ceil((length / h) - 1e-9) + 1);
            return fallback;
        };
        const fallbackCountX = axisCountFromSize(x0, x1, Number(meshElementSize?.x), 30);
        const fallbackCountY = axisCountFromSize(y0, y1, Number(meshElementSize?.y), 30);
        const fallbackCountZ = axisCountFromSize(z0, z1, Number(meshElementSize?.z), 30);

        const makeAxis = (vals, vmin, vmax, fallbackCount) => {
            if (Array.isArray(vals) && vals.length >= 2) {
                const finite = vals.filter((v) => Number.isFinite(v));
                if (finite.length >= 2) return finite;
            }
            return Array.from({ length: fallbackCount }, (_, i) => (
                vmin + ((vmax - vmin) * i) / Math.max(1, fallbackCount - 1)
            ));
        };

        const xs = makeAxis(meshAxes?.x, x0, x1, fallbackCountX);
        const ys = makeAxis(meshAxes?.y, y0, y1, fallbackCountY);
        const zs = makeAxis(meshAxes?.z, z0, z1, fallbackCountZ);

        const maxDim = Math.max(Math.abs(x1 - x0), Math.abs(y1 - y0), Math.abs(z1 - z0), 1e-6);
        const eps = Math.max(0.03, 0.0012 * maxDim);
        const lines = [];
        const add = (a, b) => {
            lines.push([a[0], a[1], a[2]], [b[0], b[1], b[2]]);
        };

        // X-min / X-max faces
        const xMinFace = x0 - eps;
        const xMaxFace = x1 + eps;
        for (const z of zs) {
            add([xMinFace, y0, z], [xMinFace, y1, z]);
            add([xMaxFace, y0, z], [xMaxFace, y1, z]);
        }
        for (const y of ys) {
            add([xMinFace, y, z0], [xMinFace, y, z1]);
            add([xMaxFace, y, z0], [xMaxFace, y, z1]);
        }

        // Z-min / Z-max faces
        const zMinFace = z0 - eps;
        const zMaxFace = z1 + eps;
        for (const x of xs) {
            add([x, y0, zMinFace], [x, y1, zMinFace]);
            add([x, y0, zMaxFace], [x, y1, zMaxFace]);
        }
        for (const y of ys) {
            add([x0, y, zMinFace], [x1, y, zMinFace]);
            add([x0, y, zMaxFace], [x1, y, zMaxFace]);
        }

        return lines;
    }, [boardOutline, meshAxes, meshElementSize]);

    if (segments.length < 2) return null;
    const enableOcclusion = (boardOpacity ?? 0) > 0.02;

    return (
        <Line
            points={segments}
            segments
            color="#00d7ff"
            lineWidth={1.2}
            transparent
            opacity={0.9}
            depthTest={enableOcclusion}
            depthWrite={false}
            renderOrder={10}
        />
    );
};

/* Fiber vectors */
const FiberPlot = ({ segments, mode, boardOpacity, fiberLineWidth, color = '#ff5a1f' }) => {
    const baseOpacity = mode === 2 ? 0.35 : 0.7;
    const opacity = baseOpacity;
    const is2D = mode === 3;
    const enableOcclusion = (boardOpacity ?? 0) > 0.02;
    const lineWidth = Math.max(0.5, Number(fiberLineWidth ?? 2));
    const renderOrder = is2D ? 11 : 5;
    const headMeshRef = useRef(null);

    const arrowData = useMemo(() => {
        const shaftPoints = [];
        const heads = [];

        for (const seg of segments || []) {
            if (!seg || seg.length !== 2) continue;
            const a = seg[0];
            const b = seg[1];
            if (!a || !b) continue;
            if (
                !Number.isFinite(a[0]) || !Number.isFinite(a[1]) || !Number.isFinite(a[2]) ||
                !Number.isFinite(b[0]) || !Number.isFinite(b[1]) || !Number.isFinite(b[2])
            ) continue;

            const dx = b[0] - a[0];
            const dy = b[1] - a[1];
            const dz = b[2] - a[2];
            const len = Math.hypot(dx, dy, dz);
            if (!Number.isFinite(len) || len < 1e-6) continue;

            const invLen = 1.0 / len;
            const nx = dx * invLen;
            const ny = dy * invLen;
            const nz = dz * invLen;

            const preferredHead = 0.45 + 0.2 * lineWidth;
            let headLength = Math.min(preferredHead, len * 0.55);
            headLength = Math.max(headLength, len * 0.22);
            if (headLength >= len) headLength = len * 0.8;

            const shaftEnd = [
                b[0] - nx * headLength,
                b[1] - ny * headLength,
                b[2] - nz * headLength,
            ];
            shaftPoints.push([a[0], a[1], a[2]], shaftEnd);

            const center = [
                b[0] - nx * (0.5 * headLength),
                b[1] - ny * (0.5 * headLength),
                b[2] - nz * (0.5 * headLength),
            ];
            const headRadius = Math.max(0.12, headLength * 0.34);
            heads.push({
                center,
                dir: [nx, ny, nz],
                headLength,
                headRadius,
            });
        }

        return { shaftPoints, heads };
    }, [segments, lineWidth]);

    useLayoutEffect(() => {
        const mesh = headMeshRef.current;
        if (!mesh) return;

        const up = new THREE.Vector3(0, 1, 0);
        const dir = new THREE.Vector3();
        const tmp = new THREE.Object3D();

        for (let i = 0; i < arrowData.heads.length; i += 1) {
            const h = arrowData.heads[i];
            dir.set(h.dir[0], h.dir[1], h.dir[2]);
            tmp.position.set(h.center[0], h.center[1], h.center[2]);
            tmp.quaternion.setFromUnitVectors(up, dir);
            tmp.scale.set(h.headRadius, h.headLength, h.headRadius);
            tmp.updateMatrix();
            mesh.setMatrixAt(i, tmp.matrix);
        }
        mesh.instanceMatrix.needsUpdate = true;
    }, [arrowData]);

    if (arrowData.shaftPoints.length === 0) return null;

    return (
        <group renderOrder={renderOrder}>
            <Line
                points={arrowData.shaftPoints}
                segments
                color={color}
                lineWidth={lineWidth}
                transparent
                opacity={opacity}
                depthTest={enableOcclusion}
                depthWrite={false}
                renderOrder={renderOrder}
            />
            {arrowData.heads.length > 0 && (
                <instancedMesh
                    ref={headMeshRef}
                    args={[null, null, arrowData.heads.length]}
                    frustumCulled={false}
                    renderOrder={renderOrder}
                >
                    <coneGeometry args={[1, 1, 10, 1]} />
                    <meshBasicMaterial
                        color={color}
                        transparent
                        opacity={opacity}
                        depthTest={enableOcclusion}
                        depthWrite={false}
                    />
                </instancedMesh>
            )}
        </group>
    );
};

const toFiniteNumberArray = (values) => {
    if (!Array.isArray(values)) return [];
    return values.map((v) => Number(v)).filter((v) => Number.isFinite(v));
};

const evaluateCrookCenterlinePoint = (yCoord, geometryRandomization) => {
    const y = Number(yCoord);
    if (!Number.isFinite(y)) return [0, 0];
    const info = (geometryRandomization && typeof geometryRandomization === 'object')
        ? geometryRandomization
        : null;
    if (!info) return [0, 0];

    const amplitudes = toFiniteNumberArray(info.component_amplitudes);
    const shifts = toFiniteNumberArray(info.component_shifts_mm);
    const thetasDeg = toFiniteNumberArray(info.component_thetas_deg);
    const componentOrders = toFiniteNumberArray(info.component_orders)
        .map((v) => Math.max(1, Math.floor(v)));
    const pCount = Math.max(0, Math.floor(Number(info.crook_component_count) || 0));
    const termCount = Math.max(
        pCount,
        amplitudes.length,
        shifts.length,
        thetasDeg.length,
        componentOrders.length
    );

    let dx = 0.0;
    let dz = 0.0;
    for (let idx = 0; idx < termCount; idx += 1) {
        const order = Number.isFinite(componentOrders[idx]) ? componentOrders[idx] : (idx + 1);
        const lengthMm = (2 ** (5 - order)) * 1000.0;
        if (!Number.isFinite(lengthMm) || lengthMm <= 0.0) continue;
        const amp = Number.isFinite(amplitudes[idx]) ? amplitudes[idx] : 0.0;
        const shift = Number.isFinite(shifts[idx]) ? shifts[idx] : 0.0;
        const thetaRad = (Number.isFinite(thetasDeg[idx]) ? thetasDeg[idx] : 0.0) * (Math.PI / 180.0);
        const wave = Math.sin((2.0 * Math.PI * (y + shift)) / lengthMm);
        dx += Math.sin(thetaRad) * amp * wave;
        dz += Math.cos(thetaRad) * amp * wave;
    }

    const legacyCrookX = Number(info.active_legacy_manual_crook_x_coeff);
    const legacyCrookZ = Number(info.active_legacy_manual_crook_y_coeff);
    if (Number.isFinite(legacyCrookX)) dx += legacyCrookX * y * y;
    if (Number.isFinite(legacyCrookZ)) dz += legacyCrookZ * y * y;

    // pith centerline in world coordinates is opposite the applied crook shift
    return [-dx, -dz];
};

const KnotSequenceSlotSegments = ({
    knotSequence,
    geometryRandomization,
    config,
}) => {
    const slotSegments = useMemo(() => {
        const slotCount = Math.max(0, Math.floor(Number(knotSequence?.slot_count) || 0));
        const dz = Number(knotSequence?.dz_mm);
        if (slotCount <= 0 || !Number.isFinite(dz) || dz <= 0) {
            return { withKnot: [], empty: [], withKnotStarts: [], emptyStarts: [] };
        }

        const zMinFromInfo = Number(knotSequence?.z_min_mm);
        const cfgZ0 = Number(config?.board_z_min);
        const cfgZ1 = Number(config?.board_z_max);
        const zMinFromConfig = (
            Number.isFinite(cfgZ0) && Number.isFinite(cfgZ1)
                ? Math.min(cfgZ0, cfgZ1)
                : 0
        );
        const zMin = Number.isFinite(zMinFromInfo) ? zMinFromInfo : zMinFromConfig;
        const occupancy = Array.isArray(knotSequence?.slot_has_knot)
            ? knotSequence.slot_has_knot
            : [];

        const withKnot = [];
        const empty = [];
        const withKnotStarts = [];
        const emptyStarts = [];

        const appendSegment = (idx, end) => {
            const y0 = zMin + (idx * dz);
            const [x0, z0] = evaluateCrookCenterlinePoint(y0, geometryRandomization);
            const start = [x0, y0, z0];
            const hasKnot = Number(occupancy[idx] || 0) > 0;
            if (hasKnot) {
                withKnot.push(start, end);
                withKnotStarts.push(start);
            } else {
                empty.push(start, end);
                emptyStarts.push(start);
            }
        };

        for (let idx = 0; idx < slotCount; idx += 1) {
            const y1 = zMin + ((idx + 1) * dz);
            const [x1, z1] = evaluateCrookCenterlinePoint(y1, geometryRandomization);
            appendSegment(idx, [x1, y1, z1]);
        }

        return { withKnot, empty, withKnotStarts, emptyStarts };
    }, [knotSequence, geometryRandomization, config?.board_z_min, config?.board_z_max]);

    if (
        slotSegments.withKnot.length === 0
        && slotSegments.empty.length === 0
        && slotSegments.withKnotStarts.length === 0
        && slotSegments.emptyStarts.length === 0
    ) return null;

    const emptyStartBuffer = new Float32Array(slotSegments.emptyStarts.flat());
    const knotStartBuffer = new Float32Array(slotSegments.withKnotStarts.flat());

    return (
        <group renderOrder={42}>
            {slotSegments.empty.length > 0 && (
                <Line
                    points={slotSegments.empty}
                    segments
                    color="#e24b4b"
                    lineWidth={4.2}
                    transparent
                    opacity={0.94}
                    depthTest={false}
                    depthWrite={false}
                    renderOrder={42}
                />
            )}
            {slotSegments.withKnot.length > 0 && (
                <Line
                    points={slotSegments.withKnot}
                    segments
                    color="#2ecf62"
                    lineWidth={4.2}
                    transparent
                    opacity={0.96}
                    depthTest={false}
                    depthWrite={false}
                    renderOrder={43}
                />
            )}
            {emptyStartBuffer.length > 0 && (
                <points renderOrder={44}>
                    <bufferGeometry>
                        <bufferAttribute attach="attributes-position" args={[emptyStartBuffer, 3]} />
                    </bufferGeometry>
                    <pointsMaterial
                        color="#ff7d7d"
                        size={9}
                        sizeAttenuation={false}
                        transparent
                        opacity={0.98}
                        depthTest={false}
                        depthWrite={false}
                    />
                </points>
            )}
            {knotStartBuffer.length > 0 && (
                <points renderOrder={45}>
                    <bufferGeometry>
                        <bufferAttribute attach="attributes-position" args={[knotStartBuffer, 3]} />
                    </bufferGeometry>
                    <pointsMaterial
                        color="#64ff93"
                        size={9}
                        sizeAttenuation={false}
                        transparent
                        opacity={0.98}
                        depthTest={false}
                        depthWrite={false}
                    />
                </points>
            )}
        </group>
    );
};

/* ── Growth ring isosurfaces ── */
const GrowthLayer = ({ layer, index, isPith = false, isLogMode = false }) => {
    const geometry = useMemo(() => {
        const geom = new THREE.BufferGeometry();
        const verts = new Float32Array(layer.vertices.flat());
        const indices = new Uint32Array(layer.faces.flat());
        geom.setAttribute('position', new THREE.BufferAttribute(verts, 3));
        geom.setIndex(new THREE.BufferAttribute(indices, 1));
        geom.computeVertexNormals();
        return geom;
    }, [layer]);

    useEffect(() => () => {
        geometry.dispose();
    }, [geometry]);

    const color = isPith ? '#ff9f1c' : (index % 2 === 0 ? '#66bb6a' : '#ef5350');
    const opacity = isPith ? (isLogMode ? 0.72 : 0.62) : (isLogMode ? 0.34 : 0.25);
    const roughness = isLogMode ? 0.42 : 1.0;
    const metalness = isLogMode ? 0.08 : 0.0;

    return (
        <mesh geometry={geometry}>
            <meshStandardMaterial
                color={color}
                transparent
                opacity={opacity}
                side={THREE.DoubleSide}
                roughness={roughness}
                metalness={metalness}
            />
        </mesh>
    );
};

const isValidLayer = (layer) => (
    !!layer &&
    Array.isArray(layer.vertices) &&
    layer.vertices.length > 0 &&
    Array.isArray(layer.faces) &&
    layer.faces.length > 0
);

/* ── Knot isosurfaces ── */
const KnotSurface = ({ knot, boardOpacity }) => {
    const { geometry, hasVertexColors } = useMemo(() => {
        const geom = new THREE.BufferGeometry();
        const verts = new Float32Array(knot.vertices.flat());
        const indices = new Uint32Array(knot.faces.flat());
        geom.setAttribute('position', new THREE.BufferAttribute(verts, 3));
        geom.setIndex(new THREE.BufferAttribute(indices, 1));
        let hasColors = false;
        if (Array.isArray(knot?.vertex_colors) && knot.vertex_colors.length === knot.vertices.length) {
            const colorArray = new Float32Array(knot.vertex_colors.flat());
            if (colorArray.length === verts.length) {
                geom.setAttribute('color', new THREE.BufferAttribute(colorArray, 3));
                hasColors = true;
            }
        }
        geom.computeVertexNormals();
        return { geometry: geom, hasVertexColors: hasColors };
    }, [knot]);

    useEffect(() => () => {
        geometry.dispose();
    }, [geometry]);

    const boardAlpha = Math.min(1, Math.max(0, Number(boardOpacity ?? 0.8)));
    // Gradually reveal knots as board opacity decreases:
    // board_opacity=1 -> knot opacity=0, board_opacity=0 -> knot opacity=0.65
    const knotOpacity = Math.max(0, Math.min(0.65, 0.65 * (1 - boardAlpha)));
    const knotColor = (typeof knot?.color === 'string' && knot.color.trim())
        ? knot.color
        : '#222222';

    if (knotOpacity <= 1e-4) return null;

    return (
        <mesh geometry={geometry} renderOrder={30}>
            <meshStandardMaterial
                color={hasVertexColors ? '#ffffff' : knotColor}
                vertexColors={hasVertexColors}
                transparent
                opacity={knotOpacity}
                side={THREE.DoubleSide}
                depthTest={false}
                depthWrite={false}
            />
        </mesh>
    );
};

/* ── Main Viewer ── */
const Viewer3D = ({
    data,
    config,
    photorealisticOverlays,
    showPhotorealisticOverlay,
    normalOverlays,
    showNormalOverlay,
    fiberOutOfPlaneOverlays,
    showFiberOutOfPlaneOverlay,
    showKnotSequenceSlots = false,
}) => {
    const [showGrid, setShowGrid] = useState(true);
    const [useOrthographic, setUseOrthographic] = useState(false);
    const isLogMode = Number(config?.board_or_log) === 1;
    const viewerBackground = '#dfe8f3';

    const pithLayer = useMemo(() => {
        if (isValidLayer(data?.pith_layer)) return data.pith_layer;
        if (data && Array.isArray(data.layers) && data.layers.length > 0 && isValidLayer(data.layers[0])) {
            return data.layers[0];
        }
        return null;
    }, [data]);

    const ringLayersToRender = useMemo(() => {
        if (!config.display_rings || !data || !Array.isArray(data.layers)) return [];
        const validLayers = data.layers.filter(isValidLayer);
        if (!config.display_pith || !isValidLayer(pithLayer)) return validLayers;

        const hasLayerIndices = validLayers.some((layer) => Number.isFinite(Number(layer?.layer_index)));
        if (hasLayerIndices) {
            return validLayers.filter((layer) => Number(layer?.layer_index) !== 0);
        }

        return validLayers.slice(1);
    }, [config.display_rings, config.display_pith, data, pithLayer]);

    const activeFiberSegments = useMemo(() => {
        if (!data || !data.fibers) return [];

        // New payload: all modes available in one simulation response.
        if (data.fibers.surface_quiver3d || data.fibers.volume_quiver3d || data.fibers.quiver2d) {
            if (config.quiver_or_stream === 0) return [];
            if (config.quiver_or_stream === 1) return data.fibers.surface_quiver3d || [];
            if (config.quiver_or_stream === 2) return data.fibers.volume_quiver3d || [];
            if (config.rand_fibers && Array.isArray(data.fibers.quiver2d_rand)) {
                return data.fibers.quiver2d_rand;
            }
            if (!config.rand_fibers && Array.isArray(data.fibers.quiver2d_clean)) {
                return data.fibers.quiver2d_clean;
            }
            return data.fibers.quiver2d || [];
        }

        // Backward compatibility with older responses.
        if (config.quiver_or_stream === 0) return [];
        if (data.fibers.segments) return data.fibers.segments;
        return [];
    }, [data, config.quiver_or_stream, config.rand_fibers]);

    const activeNormalVectorSegments = useMemo(() => {
        if (!data || !data.normal_vectors) return [];
        if (Array.isArray(data.normal_vectors.surface_quiver3d)) {
            return data.normal_vectors.surface_quiver3d;
        }
        if (Array.isArray(data.normal_vectors.surface_quiver2d)) {
            return data.normal_vectors.surface_quiver2d;
        }
        if (Array.isArray(data.normal_vectors.segments)) {
            return data.normal_vectors.segments;
        }
        return [];
    }, [data]);

    const sceneCenter = useMemo(() => {
        const min = data?.board_outline?.min;
        const max = data?.board_outline?.max;
        if (!Array.isArray(min) || !Array.isArray(max) || min.length !== 3 || max.length !== 3) {
            return [0, 0, 0];
        }
        return [
            (min[0] + max[0]) * 0.5,
            (min[1] + max[1]) * 0.5,
            (min[2] + max[2]) * 0.5,
        ];
    }, [data?.board_outline]);
    const sceneOffset = useMemo(
        () => sceneCenter.map((value) => -value),
        [sceneCenter]
    );
    return (
        <div style={{ width: '100%', height: '100%', position: 'relative' }}>
            <button
                type="button"
                onClick={() => setUseOrthographic((prev) => !prev)}
                style={{
                    position: 'absolute',
                    // Stack above grid toggle and clear the bottom-right gizmo.
                    bottom: 124,
                    right: 12,
                    zIndex: 20,
                    padding: '6px 10px',
                    borderRadius: 8,
                    border: '1px solid rgba(24, 34, 46, 0.35)',
                    background: 'rgba(250, 252, 255, 0.88)',
                    color: '#122031',
                    fontSize: 12,
                    fontWeight: 600,
                    letterSpacing: '0.02em',
                    cursor: 'pointer',
                    boxShadow: '0 2px 8px rgba(16, 24, 40, 0.16)',
                    backdropFilter: 'blur(2px)',
                }}
            >
                Projection: {useOrthographic ? 'Orthographic' : 'Perspective'}
            </button>
            <button
                type="button"
                onClick={() => setShowGrid((prev) => !prev)}
                style={{
                    position: 'absolute',
                    // Keep clear of bottom-right gizmo.
                    bottom: 86,
                    right: 12,
                    zIndex: 20,
                    padding: '6px 10px',
                    borderRadius: 8,
                    border: '1px solid rgba(24, 34, 46, 0.35)',
                    background: 'rgba(250, 252, 255, 0.88)',
                    color: '#122031',
                    fontSize: 12,
                    fontWeight: 600,
                    letterSpacing: '0.02em',
                    cursor: 'pointer',
                    boxShadow: '0 2px 8px rgba(16, 24, 40, 0.16)',
                    backdropFilter: 'blur(2px)',
                }}
            >
                {showGrid ? 'Hide Grid' : 'Show Grid'}
            </button>
            <Canvas
                dpr={[1, 2]}
                gl={{ antialias: true, logarithmicDepthBuffer: true, powerPreference: 'high-performance' }}
            >
                <color attach="background" args={[viewerBackground]} />
                {useOrthographic ? (
                    <OrthographicCamera makeDefault position={[200, 250, 150]} zoom={1} near={0.1} far={5000} />
                ) : (
                    <PerspectiveCamera makeDefault position={[200, 250, 150]} fov={50} />
                )}
                <OrbitControls makeDefault />

                {isLogMode ? (
                    <>
                        {/* Higher directional contrast to reveal subtle knot bumps in log-mode rings. */}
                        <ambientLight intensity={0.44} />
                        <hemisphereLight
                            skyColor="#f6fbff"
                            groundColor="#73879a"
                            intensity={0.28}
                        />
                        <directionalLight position={[170, 220, 140]} intensity={1.08} color="#ffffff" />
                        <directionalLight position={[-220, 95, -140]} intensity={0.56} color="#d5e7ff" />
                        <directionalLight position={[0, 140, -220]} intensity={0.32} color="#ffd8c2" />
                    </>
                ) : (
                    <>
                        <ambientLight intensity={0.88} />
                        <directionalLight position={[100, 200, 100]} intensity={0.72} />
                        <directionalLight position={[-100, 100, -50]} intensity={0.24} />
                    </>
                )}

                {/* Grid/axes helper */}
                {showGrid && (
                    <>
                        <axesHelper args={[50]} />
                        <GizmoHelper alignment="bottom-right" margin={[60, 60]}>
                            <GizmoViewport
                                labelColor="#222222"
                                axisHeadScale={0.8}
                                labels={['X', 'Z', 'Y']}
                                axisColors={['#ff2060', '#2080ff', '#20df80']}
                            />
                        </GizmoHelper>
                        <gridHelper args={[400, 40, '#8ea2b5', '#bfd0df']} />
                    </>
                )}

                <group position={sceneCenter}>
                    <group position={sceneOffset}>
                        {/* Board outline */}
                        {data && data.board_outline && (config.display_board !== false) && (
                            <BoardOutline outline={data.board_outline} boardOpacity={config.board_opacity} knots={data.knots} />
                        )}

                        {/* Surface mesh on the 4 side faces */}
                        {data && data.board_outline && config.display_surface_mesh && (
                            <SurfaceMeshLines
                                boardOutline={data.board_outline}
                                meshAxes={data.mesh_axes}
                                boardOpacity={config.board_opacity}
                                meshElementSize={{
                                    x: Number(config.mesh_size_x_mm),
                                    y: Number(config.mesh_size_z_mm),
                                    z: Number(config.mesh_size_y_mm),
                                }}
                            />
                        )}

                        {/* Growth ring contours on board faces */}
                        {data && config.display_contours && data.contours && data.contours.length > 0 && (
                            <ContourLines
                                contours={data.contours}
                                boardOpacity={config.board_opacity}
                                contourLineWidth={config.contour_line_width}
                                boardOutline={data.board_outline}
                            />
                        )}

                        {/* Growth ring isosurfaces */}
                        {data && ringLayersToRender.map((layer, i) => (
                            <GrowthLayer key={i} layer={layer} index={i} isLogMode={isLogMode} />
                        ))}
                        {data && config.display_pith && pithLayer && (
                            <GrowthLayer key="pith-layer" layer={pithLayer} index={0} isPith isLogMode={isLogMode} />
                        )}

                        {data && data.knot_sequence && showKnotSequenceSlots && (
                            <KnotSequenceSlotSegments
                                knotSequence={data.knot_sequence}
                                geometryRandomization={data.geometry_randomization}
                                config={config}
                            />
                        )}

                        {/* Knots */}
                        {data && data.knots && config.display_knots && data.knots.map((knot, i) => (
                            <KnotSurface
                                key={i}
                                knot={knot}
                                boardOpacity={config.board_opacity}
                            />
                        ))}

                        {/* Fiber orientation plot */}
                        {data && data.fibers && config.calc_fibers && activeFiberSegments.length > 0 && (
                            <FiberPlot
                                segments={activeFiberSegments}
                                mode={config.quiver_or_stream}
                                boardOpacity={config.board_opacity}
                                fiberLineWidth={config.fiber_line_width}
                            />
                        )}

                        {data && config.display_normal_vectors_surface && activeNormalVectorSegments.length > 0 && (
                            <FiberPlot
                                segments={activeNormalVectorSegments}
                                mode={1}
                                boardOpacity={config.board_opacity}
                                fiberLineWidth={config.fiber_line_width}
                                color="#1c86ff"
                            />
                        )}

                        {/* Surface normal colormap overlays */}
                        {data && data.board_outline && (
                            <NormalFaceOverlays
                                boardOutline={data.board_outline}
                                overlays={normalOverlays}
                                visible={!!showNormalOverlay}
                            />
                        )}

                        {/* Fiber out-of-plane component overlays */}
                        {data && data.board_outline && (
                            <FiberOutOfPlaneFaceOverlays
                                boardOutline={data.board_outline}
                                overlays={fiberOutOfPlaneOverlays}
                                visible={!!showFiberOutOfPlaneOverlay}
                            />
                        )}

                        {/* Generated photorealistic face overlays */}
                        {data && data.board_outline && (
                            <PhotorealisticFaceOverlays
                                boardOutline={data.board_outline}
                                overlays={photorealisticOverlays}
                                visible={!!showPhotorealisticOverlay}
                            />
                        )}
                    </group>
                </group>
            </Canvas>
        </div>
    );
};

export default Viewer3D;
