function visualize_exported_board(mat_file, opts)
%VISUALIZE_EXPORTED_BOARD Visualize exported board MAT data from the web app.
%
% Usage:
%   visualize_exported_board('board_export_xxx.mat');
%
%   opts = struct();
%   opts.show_contours = true;
%   opts.hide_contours_inside_knots = false;
%   opts.contours_cross_sections_only = true;
%   opts.show_board_box = false;
%   opts.show_photorealistic_faces = true;
%   opts.flip_photorealistic_faces_vertical = false;
%   opts.show_pith = true;
%   opts.show_growth_layers_3d = false;
%   opts.show_knots_3d = true;
%   opts.show_knot_sequence_segments = true;
%   opts.show_fibers_3d = false;
%   opts.show_fibers_2d = false;
%   opts.show_normals_3d = false;
%   opts.view_angle = [-37.5 30];
%   opts.contour_color = [0 0 0];
%   opts.growth_layer_colors = [0.40 0.73 0.42; 0.94 0.33 0.31];
%   opts.knot_live_color = [0.65 0.47 0.26];
%   opts.knot_dead_color = [0.48 0.12 0.17];
%   opts.knot_sequence_knot_color = [0.18 0.81 0.38];
%   opts.knot_sequence_no_knot_color = [0.89 0.29 0.29];
%   opts.knot_sequence_line_width = 3.0;
%   opts.knot_sequence_endpoint_color = [0 0 0];
%   opts.knot_sequence_endpoint_marker_size = 6;
%   opts.board_alpha = 1.00;
%   opts.photorealistic_face_alpha = 1.0;
%   opts.growth_layer_alpha = 0.25;
%   opts.knot_alpha = 0.50;
%   opts.fiber_straight_direction = [0 0 1];
%   opts.fiber_deviation_max_angle_deg = 90;
%   opts.fiber_deviation_colormap = 'turbo';
%   opts.show_fiber_deviation_colorbar = true;
%   opts.normal_color = [0.00 0.45 0.90];
%   opts.normal_scale = 1.8;
%   opts.normal_line_width = 0.8;
%
% Lighting options:
%   opts.lighting_enabled = true;
%   opts.lighting_method = 'gouraud';     % 'gouraud', 'flat', or 'none'
%   opts.light_style = 'infinite';        % 'infinite' or 'local'
%   opts.light_positions = [0 -1 1; -1 -0.5 0.6];
%   opts.light_color = [1 1 1];
%   opts.material_finish = 'dull';        % e.g. 'dull', 'shiny', 'metal'
%   visualize_exported_board('board_export_xxx.mat', opts);

if nargin < 1 || isempty(mat_file)
    [f, p] = uigetfile('*.mat', 'Select exported board MAT file');
    if isequal(f, 0)
        return;
    end
    mat_file = fullfile(p, f);
end

if nargin < 2 || isempty(opts)
    opts = struct();
end

opts = apply_defaults(opts);
S = load(mat_file);

required_fields = {'board_dimensions', 'mesh_x', 'mesh_y', 'mesh_z'};
for i = 1:numel(required_fields)
    if ~isfield(S, required_fields{i})
        error('Missing required field "%s" in MAT file.', required_fields{i});
    end
end

bd = S.board_dimensions;
xmin = scalar_field(bd, 'x_min');
xmax = scalar_field(bd, 'x_max');
ymin = scalar_field(bd, 'y_min');
ymax = scalar_field(bd, 'y_max');
zmin = scalar_field(bd, 'z_min');
zmax = scalar_field(bd, 'z_max');

mesh_x = S.mesh_x(:);
mesh_y = S.mesh_y(:);
mesh_z = S.mesh_z(:);

figure('Color', 'w', 'Name', 'Exported Board Visualization');
ax = axes('Projection', 'perspective');
hold(ax, 'on');
axis(ax, 'equal');
axis(ax, 'vis3d');
grid(ax, 'on');
apply_view_angle(ax, opts.view_angle);
xlabel(ax, 'X (width)');
ylabel(ax, 'Y (thickness)');
zlabel(ax, 'Z (length)');

has_photorealistic_faces = opts.show_photorealistic_faces ...
    && isfield(S, 'photorealistic_faces') ...
    && ~isempty(struct_items(S.photorealistic_faces));

if opts.show_board
    draw_board_block(ax, xmin, xmax, ymin, ymax, zmin, zmax, opts, has_photorealistic_faces);
end

if has_photorealistic_faces
    plot_photorealistic_faces(ax, S.photorealistic_faces, xmin, xmax, ymin, ymax, zmin, zmax, opts);
end

has_pith_surface = opts.show_pith && isfield(S, 'pith_surface') ...
    && ~isempty(struct_items(S.pith_surface));

if opts.show_growth_layers && isfield(S, 'growth_layers')
    plot_growth_layers(ax, S.growth_layers, opts, has_pith_surface);
end

if has_pith_surface
    plot_pith_surface(ax, S.pith_surface, opts);
end

if opts.show_knots && isfield(S, 'knots')
    plot_knots(ax, S.knots, opts);
end

if opts.show_knot_sequence_segments && isfield(S, 'knot_sequence_segments')
    plot_knot_sequence_segments(ax, S.knot_sequence_segments, opts);
end

if opts.show_contours
    contours_to_plot = select_contours(S, opts);
    plot_contours(ax, contours_to_plot, opts, zmin, zmax);
end

has_fibers = isfield(S, 'fiber_txx') && isfield(S, 'fiber_tyy') && isfield(S, 'fiber_tzz') ...
    && isnumeric(S.fiber_txx) && ~isempty(S.fiber_txx);
has_normals = isfield(S, 'normal_nx') && isfield(S, 'normal_ny') && isfield(S, 'normal_nz') ...
    && isnumeric(S.normal_nx) && ~isempty(S.normal_nx);

if has_fibers && opts.show_fibers_2d
    plot_surface_quivers(ax, mesh_x, mesh_y, mesh_z, S.fiber_txx, S.fiber_tyy, S.fiber_tzz, ...
        opts.step_x, opts.step_y, opts.step_z, opts.fiber_scale, opts);
end

if has_fibers && opts.show_fibers_3d
    plot_fiber_volume_quivers(ax, mesh_x, mesh_y, mesh_z, S.fiber_txx, S.fiber_tyy, S.fiber_tzz, ...
        opts.step_x, opts.step_y, opts.step_z, opts.fiber_scale, opts);
end

if has_normals && opts.show_normals_3d
    plot_volume_quivers(ax, mesh_x, mesh_y, mesh_z, S.normal_nx, S.normal_ny, S.normal_nz, ...
        opts.step_x, opts.step_y, opts.step_z, opts.normal_scale, opts.normal_color, ...
        opts.normal_line_width);
end

apply_scene_lighting(ax, opts);
title(ax, sprintf('Board Export: %s', mat_file), 'Interpreter', 'none');
end

function opts = apply_defaults(opts)
show_knot_sequence_segments_was_set = isfield(opts, 'show_knot_sequence_segments') ...
    || isfield(opts, 'show_knot_sequence_slots');
if isfield(opts, 'show_fiber_2d') && ~isfield(opts, 'show_fibers_2d')
    opts.show_fibers_2d = opts.show_fiber_2d;
end
if isfield(opts, 'show_fiber_3d') && ~isfield(opts, 'show_fibers_3d')
    opts.show_fibers_3d = opts.show_fiber_3d;
end
if isfield(opts, 'show_normals') && ~isfield(opts, 'show_normals_3d')
    opts.show_normals_3d = opts.show_normals;
end
if isfield(opts, 'show_growth_layers_3d') && ~isfield(opts, 'show_growth_layers')
    opts.show_growth_layers = opts.show_growth_layers_3d;
end
if isfield(opts, 'show_knots_3d') && ~isfield(opts, 'show_knots')
    opts.show_knots = opts.show_knots_3d;
end
if isfield(opts, 'show_board_box') && ~isfield(opts, 'show_board')
    opts.show_board = opts.show_board_box;
end
if isfield(opts, 'show_cross_section_contours_only') && ~isfield(opts, 'contours_cross_sections_only')
    opts.contours_cross_sections_only = opts.show_cross_section_contours_only;
end
if isfield(opts, 'hide_contours_in_knots') && ~isfield(opts, 'hide_contours_inside_knots')
    opts.hide_contours_inside_knots = opts.hide_contours_in_knots;
end
if isfield(opts, 'show_knot_sequence_slots') && ~isfield(opts, 'show_knot_sequence_segments')
    opts.show_knot_sequence_segments = opts.show_knot_sequence_slots;
end

defaults = struct();
defaults.show_contours = true;
defaults.hide_contours_inside_knots = false;
defaults.contours_cross_sections_only = false;
defaults.show_board = true;
defaults.show_photorealistic_faces = true;
defaults.flip_photorealistic_faces_vertical = true;
defaults.show_pith = true;
defaults.show_growth_layers = true;
defaults.show_knots = true;
defaults.show_knot_sequence_segments = false;
defaults.show_fibers_3d = false;
defaults.show_fibers_2d = true;
defaults.show_normals_3d = false;
defaults.step_x = 4;
defaults.step_y = 3;
defaults.step_z = 6;
defaults.view_angle = [-37.5 30];
defaults.fiber_scale = 2.0;
defaults.normal_scale = 1.8;
defaults.contour_color = [0 0 0];
defaults.contour_line_width = 1.1;
defaults.board_color = [1 1 1];
defaults.board_edge_color = [0 0 0];
defaults.board_alpha = 0.22;
defaults.photorealistic_face_alpha = 1.0;
defaults.growth_layer_colors = [
    0.40 0.73 0.42;
    0.94 0.33 0.31;
    0.18 0.55 0.86;
    0.96 0.68 0.20;
    0.55 0.39 0.75
];
defaults.growth_layer_alpha = 0.25;
defaults.pith_color = [1.00 0.62 0.11];
defaults.pith_alpha = 0.62;
defaults.knot_live_color = [0.65 0.47 0.26];
defaults.knot_dead_color = [0.48 0.12 0.17];
defaults.knot_alpha = 0.50;
defaults.knot_use_exported_vertex_colors = false;
defaults.knot_sequence_knot_color = [0.18 0.81 0.38];
defaults.knot_sequence_no_knot_color = [0.89 0.29 0.29];
defaults.knot_sequence_line_width = 3.0;
defaults.knot_sequence_endpoint_color = [0 0 0];
defaults.knot_sequence_endpoint_marker_size = 6.0;
defaults.fiber_straight_direction = [0 0 1];
defaults.fiber_deviation_max_angle_deg = 90;
defaults.fiber_deviation_colormap = 'turbo';
defaults.fiber_deviation_color_bins = 32;
defaults.show_fiber_deviation_colorbar = true;
defaults.normal_color = [0.00 0.45 0.90];
defaults.fiber_line_width = 0.8;
defaults.normal_line_width = 0.8;
defaults.lighting_enabled = true;
defaults.lighting_method = 'gouraud';
defaults.light_style = 'infinite';
defaults.light_positions = [0 -1 1; -1 -0.5 0.6];
defaults.light_color = [1 1 1];
defaults.material_finish = 'dull';

fn = fieldnames(defaults);
for i = 1:numel(fn)
    k = fn{i};
    if ~isfield(opts, k)
        opts.(k) = defaults.(k);
    end
end

opts.contour_color = rgb_value(opts.contour_color, defaults.contour_color);
opts.board_color = rgb_value(opts.board_color, defaults.board_color);
opts.board_edge_color = rgb_value(opts.board_edge_color, defaults.board_edge_color);
opts.pith_color = rgb_value(opts.pith_color, defaults.pith_color);
opts.knot_live_color = rgb_value(opts.knot_live_color, defaults.knot_live_color);
opts.knot_dead_color = rgb_value(opts.knot_dead_color, defaults.knot_dead_color);
opts.knot_sequence_knot_color = rgb_value(opts.knot_sequence_knot_color, defaults.knot_sequence_knot_color);
opts.knot_sequence_no_knot_color = rgb_value(opts.knot_sequence_no_knot_color, defaults.knot_sequence_no_knot_color);
opts.knot_sequence_endpoint_color = rgb_value(opts.knot_sequence_endpoint_color, defaults.knot_sequence_endpoint_color);
opts.normal_color = rgb_value(opts.normal_color, defaults.normal_color);
opts.growth_layer_colors = rgb_matrix(opts.growth_layer_colors, defaults.growth_layer_colors);
opts.light_color = rgb_value(opts.light_color, defaults.light_color);

opts.board_alpha = clamp01(opts.board_alpha);
opts.photorealistic_face_alpha = clamp01(opts.photorealistic_face_alpha);
opts.growth_layer_alpha = clamp01(opts.growth_layer_alpha);
opts.pith_alpha = clamp01(opts.pith_alpha);
opts.knot_alpha = clamp01(opts.knot_alpha);
opts.step_x = max(1, round(double(opts.step_x)));
opts.step_y = max(1, round(double(opts.step_y)));
opts.step_z = max(1, round(double(opts.step_z)));
opts.knot_sequence_line_width = positive_scalar_value(opts.knot_sequence_line_width, defaults.knot_sequence_line_width);
opts.knot_sequence_endpoint_marker_size = positive_scalar_value( ...
    opts.knot_sequence_endpoint_marker_size, defaults.knot_sequence_endpoint_marker_size);
opts.view_angle = normalize_view_angle(opts.view_angle, defaults.view_angle);
opts.fiber_straight_direction = normalize_direction(opts.fiber_straight_direction, defaults.fiber_straight_direction);
opts.fiber_deviation_max_angle_deg = positive_scalar_value(opts.fiber_deviation_max_angle_deg, defaults.fiber_deviation_max_angle_deg);
opts.fiber_deviation_color_bins = max(2, round(positive_scalar_value(opts.fiber_deviation_color_bins, defaults.fiber_deviation_color_bins)));
opts.show_fiber_deviation_colorbar = logical_scalar_value(opts.show_fiber_deviation_colorbar, defaults.show_fiber_deviation_colorbar);
opts.show_normals_3d = logical_scalar_value(opts.show_normals_3d, defaults.show_normals_3d);
if ~show_knot_sequence_segments_was_set
    opts.show_knot_sequence_segments = opts.show_knots ...
        && ~opts.show_contours ...
        && ~opts.show_board ...
        && ~opts.show_photorealistic_faces ...
        && ~opts.show_pith ...
        && ~opts.show_growth_layers ...
        && ~opts.show_fibers_3d ...
        && ~opts.show_fibers_2d ...
        && ~opts.show_normals_3d;
end
opts.show_knot_sequence_segments = logical_scalar_value(opts.show_knot_sequence_segments, defaults.show_knot_sequence_segments);
opts.lighting_enabled = logical_scalar_value(opts.lighting_enabled, defaults.lighting_enabled);
opts.lighting_method = option_string(opts.lighting_method, defaults.lighting_method, {'gouraud', 'flat', 'none'});
opts.light_style = option_string(opts.light_style, defaults.light_style, {'infinite', 'local'});
opts.light_positions = normalize_light_positions(opts.light_positions, defaults.light_positions);
opts.material_finish = option_string(opts.material_finish, defaults.material_finish, {'dull', 'shiny', 'metal', 'default'});
opts.show_growth_layers_3d = opts.show_growth_layers;
opts.show_knots_3d = opts.show_knots;
opts.show_board_box = opts.show_board;
end

function apply_view_angle(ax, view_angle)
if isnumeric(view_angle) && isscalar(view_angle)
    view(ax, double(view_angle));
elseif isnumeric(view_angle) && (numel(view_angle) == 2 || numel(view_angle) == 3)
    view(ax, double(view_angle(:)'));
else
    view(ax, 3);
end
end

function view_angle = normalize_view_angle(value, fallback)
if isnumeric(value) && isscalar(value)
    view_angle = double(value);
elseif isnumeric(value) && (numel(value) == 2 || numel(value) == 3)
    view_angle = double(value(:)');
else
    view_angle = fallback;
end
end

function apply_scene_lighting(ax, opts)
delete(findall(ax, 'Type', 'light'));
patch_objects = findall(ax, 'Type', 'patch');
surface_objects = findall(ax, 'Type', 'surface');
lit_objects = [
    patch_objects(:);
    surface_objects(:)
];
if ~opts.lighting_enabled || strcmpi(opts.lighting_method, 'none')
    if ~isempty(lit_objects)
        set(lit_objects, 'FaceLighting', 'none');
    end
    return;
end

positions = opts.light_positions;
for i = 1:size(positions, 1)
    light( ...
        'Parent', ax, ...
        'Position', positions(i, :), ...
        'Style', opts.light_style, ...
        'Color', opts.light_color);
end
if ~isempty(lit_objects)
    set(lit_objects, 'FaceLighting', opts.lighting_method);
end
apply_material_finish(lit_objects, opts.material_finish);
set(findall(ax, 'Tag', 'PhotorealisticFace'), 'FaceLighting', 'none');
end

function apply_material_finish(objects, finish)
if isempty(objects) || strcmpi(finish, 'default')
    return;
end
switch lower(finish)
    case 'dull'
        props = {'AmbientStrength', 0.30, 'DiffuseStrength', 0.80, ...
            'SpecularStrength', 0.05, 'SpecularExponent', 10, ...
            'SpecularColorReflectance', 1.00};
    case 'shiny'
        props = {'AmbientStrength', 0.25, 'DiffuseStrength', 0.65, ...
            'SpecularStrength', 0.75, 'SpecularExponent', 30, ...
            'SpecularColorReflectance', 1.00};
    case 'metal'
        props = {'AmbientStrength', 0.25, 'DiffuseStrength', 0.35, ...
            'SpecularStrength', 0.95, 'SpecularExponent', 25, ...
            'SpecularColorReflectance', 0.00};
    otherwise
        return;
end
set(objects, props{:});
end

function positions = normalize_light_positions(value, fallback)
positions = fallback;
if ~isnumeric(value) || isempty(value)
    return;
end
arr = double(value);
if isvector(arr) && numel(arr) == 3
    positions = arr(:)';
elseif ndims(arr) == 2 && size(arr, 2) == 3
    positions = arr;
end
end

function val = logical_scalar_value(value, fallback)
val = fallback;
if ~(islogical(value) || isnumeric(value)) || isempty(value)
    return;
end
val = logical(value(1));
end

function val = positive_scalar_value(value, fallback)
val = fallback;
if isnumeric(value) && ~isempty(value) && isfinite(value(1)) && value(1) > 0
    val = double(value(1));
end
end

function direction = normalize_direction(value, fallback)
direction = fallback;
if isnumeric(value) && numel(value) == 3 && all(isfinite(value(:)))
    candidate = double(value(:))';
    mag = norm(candidate);
    if mag > 0
        direction = candidate ./ mag;
        return;
    end
end
direction = fallback ./ max(norm(fallback), eps);
end

function val = option_string(value, fallback, allowed)
val = fallback;
if ischar(value)
    candidate = lower(strtrim(value));
elseif (exist('isstring', 'builtin') || exist('isstring', 'file')) && isstring(value) && numel(value) == 1
    candidate = lower(strtrim(char(value)));
else
    return;
end
for i = 1:numel(allowed)
    if strcmp(candidate, allowed{i})
        val = candidate;
        return;
    end
end
end

function val = scalar_field(st, name)
if ~isfield(st, name)
    error('board_dimensions is missing field "%s".', name);
end
v = st.(name);
if isnumeric(v)
    val = double(v(1));
else
    error('board_dimensions.%s must be numeric.', name);
end
end

function draw_board_block(ax, x0, x1, y0, y1, z0, z1, opts, end_cross_sections_only)
V = [
    x0 y0 z0;
    x1 y0 z0;
    x1 y1 z0;
    x0 y1 z0;
    x0 y0 z1;
    x1 y0 z1;
    x1 y1 z1;
    x0 y1 z1;
];
F = [
    1 2 3 4;
    5 6 7 8;
    1 2 6 5;
    2 3 7 6;
    3 4 8 7;
    4 1 5 8;
];
if end_cross_sections_only
    % MATLAB Z is board length; these are the two end cross-section faces.
    F = F([1, 2], :);
end
patch(ax, 'Vertices', V, 'Faces', F, ...
    'FaceColor', opts.board_color, ...
    'EdgeColor', opts.board_edge_color, ...
    'LineWidth', 0.8, ...
    'FaceAlpha', opts.board_alpha);
end

function plot_photorealistic_faces(ax, photo_faces, x0, x1, y0, y1, z0, z1, opts)
items = struct_items(photo_faces);
if isempty(items)
    return;
end

eps_val = max([abs(x1 - x0), abs(y1 - y0), abs(z1 - z0), 1.0]) * 1e-4;
for i = 1:numel(items)
    item = items{i};
    face_key = string_field(item, 'face', '');
    img = field_value(item, 'image', []);
    if isempty(face_key) || isempty(img) || ndims(img) ~= 3 || size(img, 3) ~= 3
        continue;
    end
    if ~isa(img, 'uint8')
        img = double(img);
        if max(img(:)) <= 1.0
            img = img * 255.0;
        end
        img = uint8(min(255, max(0, round(img))));
    end

    if opts.flip_photorealistic_faces_vertical
        img = flipud(img);
    end
    flip_x = logical_scalar_field(item, 'flip_x', false);
    if flip_x
        img = fliplr(img);
    end

    [h, w, ~] = size(img);
    switch lower(face_key)
        case 'z_max'
            [Xg, Zg] = meshgrid(linspace(x0, x1, w), linspace(z0, z1, h));
            Yg = (y1 + eps_val) * ones(size(Xg));
        case 'z_min'
            [Xg, Zg] = meshgrid(linspace(x0, x1, w), linspace(z0, z1, h));
            Yg = (y0 - eps_val) * ones(size(Xg));
        case 'x_max'
            [Yg, Zg] = meshgrid(linspace(y0, y1, w), linspace(z0, z1, h));
            Xg = (x1 + eps_val) * ones(size(Yg));
        case 'x_min'
            [Yg, Zg] = meshgrid(linspace(y0, y1, w), linspace(z0, z1, h));
            Xg = (x0 - eps_val) * ones(size(Yg));
        otherwise
            continue;
    end

    surface(ax, Xg, Yg, Zg, ...
        'CData', img, ...
        'FaceColor', 'texturemap', ...
        'EdgeColor', 'none', ...
        'FaceAlpha', opts.photorealistic_face_alpha, ...
        'FaceLighting', 'none', ...
        'Tag', 'PhotorealisticFace');
end
end

function contours_cell = select_contours(S, opts)
contours_cell = {};
if opts.hide_contours_inside_knots
    if isfield(S, 'contours_masked_live') && ~isempty(S.contours_masked_live)
        contours_cell = S.contours_masked_live;
    elseif isfield(S, 'contours_masked') && ~isempty(S.contours_masked)
        contours_cell = S.contours_masked;
    elseif isfield(S, 'contours')
        contours_cell = S.contours;
    end
elseif isfield(S, 'contours')
    contours_cell = S.contours;
end
end

function plot_contours(ax, contours_cell, opts, zmin, zmax)
if isempty(contours_cell) || ~iscell(contours_cell)
    return;
end
for i = 1:numel(contours_cell)
    c = contours_cell{i};
    if ~isnumeric(c) || size(c, 2) ~= 3 || size(c, 1) < 2
        continue;
    end
    if opts.contours_cross_sections_only && ~is_cross_section_contour(c, zmin, zmax)
        continue;
    end
    plot3(ax, c(:, 1), c(:, 2), c(:, 3), '-', ...
        'Color', opts.contour_color, ...
        'LineWidth', opts.contour_line_width);
end
end

function ok = is_cross_section_contour(points, zmin, zmax)
ok = false;
z = points(:, 3);
finite = z(isfinite(z));
if isempty(finite)
    return;
end
span = max([abs(zmax - zmin), 1.0]);
tol = max(1e-6, 1e-5 * span);
z_range = max(finite) - min(finite);
if z_range > tol
    return;
end
z_mean = mean(finite);
ok = abs(z_mean - zmin) <= tol || abs(z_mean - zmax) <= tol;
end

function plot_growth_layers(ax, growth_layers, opts, skip_pith_layer)
items = struct_items(growth_layers);
if isempty(items)
    return;
end
draw_idx = 0;
for i = 1:numel(items)
    layer = items{i};
    layer_index = optional_scalar_field(layer, 'layer_index');
    if skip_pith_layer && ~isempty(layer_index) && layer_index == 0
        continue;
    end
    draw_idx = draw_idx + 1;
    color = color_row(opts.growth_layer_colors, draw_idx);
    draw_mesh_surface(ax, layer, color, opts.growth_layer_alpha);
end
end

function plot_pith_surface(ax, pith_surface, opts)
items = struct_items(pith_surface);
if isempty(items)
    return;
end
draw_mesh_surface(ax, items{1}, opts.pith_color, opts.pith_alpha);
end

function plot_knots(ax, knots, opts)
items = struct_items(knots);
if isempty(items)
    return;
end
for i = 1:numel(items)
    knot = items{i};
    [vertices, faces] = mesh_vertices_faces(knot);
    if isempty(vertices) || isempty(faces)
        continue;
    end

    dead_weight = numeric_field(knot, 'dead_weight', []);
    if numel(dead_weight) == size(vertices, 1)
        w = clamp01(dead_weight(:));
        cdata = (1.0 - w) * opts.knot_live_color + w * opts.knot_dead_color;
        patch(ax, 'Vertices', vertices, 'Faces', faces, ...
            'FaceVertexCData', cdata, ...
            'FaceColor', 'interp', ...
            'EdgeColor', 'none', ...
            'FaceAlpha', opts.knot_alpha);
    elseif opts.knot_use_exported_vertex_colors
        vertex_colors = numeric_field(knot, 'vertex_colors', []);
        if size(vertex_colors, 1) == size(vertices, 1) && size(vertex_colors, 2) == 3
            patch(ax, 'Vertices', vertices, 'Faces', faces, ...
                'FaceVertexCData', clamp01(vertex_colors), ...
                'FaceColor', 'interp', ...
                'EdgeColor', 'none', ...
                'FaceAlpha', opts.knot_alpha);
        else
            draw_mesh_surface(ax, knot, opts.knot_live_color, opts.knot_alpha);
        end
    else
        draw_mesh_surface(ax, knot, opts.knot_live_color, opts.knot_alpha);
    end
end
end

function plot_knot_sequence_segments(ax, segments, opts)
items = struct_items(segments);
if isempty(items) && isstruct(segments)
    items = {segments};
end
if isempty(items)
    return;
end
for i = 1:numel(items)
    item = items{i};
    no_knot_segments = field_value(item, 'no_knot', {});
    knot_segments = field_value(item, 'with_knot', {});
    draw_knot_sequence_segment_lines( ...
        ax, no_knot_segments, ...
        opts.knot_sequence_no_knot_color, opts.knot_sequence_line_width);
    draw_knot_sequence_segment_lines( ...
        ax, knot_segments, ...
        opts.knot_sequence_knot_color, opts.knot_sequence_line_width);
    draw_knot_sequence_endpoint_dots(ax, no_knot_segments, opts);
    draw_knot_sequence_endpoint_dots(ax, knot_segments, opts);
end
end

function draw_knot_sequence_segment_lines(ax, raw_segments, color, line_width)
segments = segment_items(raw_segments);
if isempty(segments)
    return;
end

points = [];
for i = 1:numel(segments)
    pts = segments{i};
    if size(pts, 1) < 2 || size(pts, 2) ~= 3
        continue;
    end
    valid = all(isfinite(pts), 2);
    pts = pts(valid, :);
    if size(pts, 1) < 2
        continue;
    end
    points = [points; pts; [NaN NaN NaN]]; %#ok<AGROW>
end

if isempty(points)
    return;
end
plot3(ax, points(:, 1), points(:, 2), points(:, 3), ...
    'Color', color, ...
    'LineWidth', line_width, ...
    'Clipping', 'off');
end

function draw_knot_sequence_endpoint_dots(ax, raw_segments, opts)
segments = segment_items(raw_segments);
if isempty(segments)
    return;
end

endpoints = [];
for i = 1:numel(segments)
    pts = segments{i};
    if size(pts, 1) < 2 || size(pts, 2) ~= 3
        continue;
    end
    valid = all(isfinite(pts), 2);
    pts = pts(valid, :);
    if size(pts, 1) < 2
        continue;
    end
    endpoints = [endpoints; pts(1, :); pts(end, :)]; %#ok<AGROW>
end

if isempty(endpoints)
    return;
end
plot3(ax, endpoints(:, 1), endpoints(:, 2), endpoints(:, 3), 'o', ...
    'LineStyle', 'none', ...
    'MarkerSize', opts.knot_sequence_endpoint_marker_size, ...
    'MarkerFaceColor', opts.knot_sequence_endpoint_color, ...
    'MarkerEdgeColor', opts.knot_sequence_endpoint_color, ...
    'Clipping', 'off');
end

function draw_mesh_surface(ax, surface, face_color, face_alpha)
[vertices, faces] = mesh_vertices_faces(surface);
if isempty(vertices) || isempty(faces)
    return;
end
patch(ax, 'Vertices', vertices, 'Faces', faces, ...
    'FaceColor', face_color, ...
    'EdgeColor', 'none', ...
    'FaceAlpha', face_alpha, ...
    'FaceLighting', 'gouraud', ...
    'BackFaceLighting', 'reverselit');
end

function [vertices, faces] = mesh_vertices_faces(surface)
vertices = numeric_field(surface, 'vertices', []);
faces = numeric_field(surface, 'faces', []);
if isempty(vertices) || isempty(faces)
    vertices = [];
    faces = [];
    return;
end
if size(vertices, 2) ~= 3 || size(faces, 2) < 3
    vertices = [];
    faces = [];
    return;
end
faces = round(faces(:, 1:3));
if ~isempty(faces) && min(faces(:)) == 0
    faces = faces + 1;
end
valid = all(isfinite(faces), 2) ...
    & all(faces >= 1, 2) ...
    & all(faces <= size(vertices, 1), 2);
faces = faces(valid, :);
end

function items = struct_items(value)
items = {};
if isempty(value)
    return;
end
if iscell(value)
    for i = 1:numel(value)
        item = value{i};
        if isstruct(item)
            for j = 1:numel(item)
                items{end + 1} = item(j); %#ok<AGROW>
            end
        end
    end
elseif isstruct(value)
    for i = 1:numel(value)
        items{end + 1} = value(i); %#ok<AGROW>
    end
end
end

function segments = segment_items(value)
segments = {};
if isempty(value)
    return;
end
if iscell(value)
    for i = 1:numel(value)
        item = value{i};
        if isnumeric(item)
            arr = double(item);
            if ndims(arr) == 2 && size(arr, 2) == 3
                segments{end + 1} = arr; %#ok<AGROW>
            end
        end
    end
elseif isnumeric(value)
    arr = double(value);
    if ndims(arr) == 2 && size(arr, 2) == 3
        segments{end + 1} = arr; %#ok<AGROW>
    end
end
end

function val = numeric_field(st, name, fallback)
val = fallback;
raw = field_value(st, name, []);
if isnumeric(raw)
    val = double(raw);
end
end

function val = logical_scalar_field(st, name, fallback)
val = fallback;
raw = field_value(st, name, []);
if islogical(raw) || isnumeric(raw)
    if ~isempty(raw)
        val = logical(raw(1));
    end
end
end

function val = string_field(st, name, fallback)
val = fallback;
raw = field_value(st, name, []);
if iscell(raw) && numel(raw) == 1
    raw = raw{1};
end
if ischar(raw)
    val = strtrim(raw);
    return;
end
if (exist('isstring', 'builtin') || exist('isstring', 'file')) && isstring(raw)
    val = strtrim(char(raw));
end
end

function raw = field_value(st, name, fallback)
raw = fallback;
if ~isstruct(st) || ~isfield(st, name)
    return;
end
raw = st.(name);
if iscell(raw) && numel(raw) == 1
    raw = raw{1};
end
end

function val = optional_scalar_field(st, name)
raw = numeric_field(st, name, []);
if isempty(raw)
    val = [];
else
    val = double(raw(1));
end
end

function plot_surface_quivers(ax, x, y, z, ux, uy, uz, sx, sy, sz, scale, opts)
nx = numel(x);
ny = numel(y);
nz = numel(z);
if nx < 2 || ny < 2 || nz < 2
    return;
end
if ndims(ux) ~= 3 || ndims(uy) ~= 3 || ndims(uz) ~= 3 ...
        || any(size(ux) ~= [ny, nx, nz]) ...
        || any(size(uy) ~= [ny, nx, nz]) ...
        || any(size(uz) ~= [ny, nx, nz])
    return;
end

ix = unique([1:sx:nx, nx]);
iy = unique([1:sy:ny, ny]);
iz = sampled_indices_excluding_upper(nz, sz);
if numel(iz) > 1
    iz = iz(1:end - 1);
end
cmap = colormap_values(opts.fiber_deviation_colormap, opts.fiber_deviation_color_bins);
drew_any = false;

for x_idx = [1, nx]
    [Yg, Zg] = ndgrid(y(iy), z(iz));
    Xg = x(x_idx) * ones(size(Yg));

    U = squeeze(ux(iy, x_idx, iz));
    V = squeeze(uy(iy, x_idx, iz));
    W = squeeze(uz(iy, x_idx, iz));
    U(:) = 0;
    drew_any = plot_colored_fiber_quivers(ax, Xg, Yg, Zg, U, V, W, scale, opts, cmap) || drew_any;
end

for y_idx = [1, ny]
    [Xg, Zg] = ndgrid(x(ix), z(iz));
    Yg = y(y_idx) * ones(size(Xg));

    U = squeeze(ux(y_idx, ix, iz));
    V = squeeze(uy(y_idx, ix, iz));
    W = squeeze(uz(y_idx, ix, iz));
    V(:) = 0;
    drew_any = plot_colored_fiber_quivers(ax, Xg, Yg, Zg, U, V, W, scale, opts, cmap) || drew_any;
end

if drew_any
    apply_fiber_deviation_colorbar(ax, opts, cmap);
end
end

function plot_volume_quivers(ax, x, y, z, ux, uy, uz, sx, sy, sz, scale, color, line_width, omit_upper_z_boundary)
if nargin < 14
    omit_upper_z_boundary = false;
end
nx = numel(x);
ny = numel(y);
nz = numel(z);
if nx < 2 || ny < 2 || nz < 2
    return;
end
if ndims(ux) ~= 3 || ndims(uy) ~= 3 || ndims(uz) ~= 3 ...
        || any(size(ux) ~= [ny, nx, nz]) ...
        || any(size(uy) ~= [ny, nx, nz]) ...
        || any(size(uz) ~= [ny, nx, nz])
    return;
end

ix = unique([1:sx:nx, nx]);
iy = unique([1:sy:ny, ny]);
if omit_upper_z_boundary
    iz = sampled_indices_excluding_upper(nz, sz);
else
    iz = unique([1:sz:nz, nz]);
end

[Xg, Yg, Zg] = meshgrid(x(ix), y(iy), z(iz));
U = ux(iy, ix, iz);
V = uy(iy, ix, iz);
W = uz(iy, ix, iz);
quiver3(ax, Xg, Yg, Zg, U, V, W, scale, ...
    'Color', color, 'LineWidth', line_width, 'MaxHeadSize', 0.35);
end

function plot_fiber_volume_quivers(ax, x, y, z, ux, uy, uz, sx, sy, sz, scale, opts)
nx = numel(x);
ny = numel(y);
nz = numel(z);
if nx < 2 || ny < 2 || nz < 2
    return;
end
if ndims(ux) ~= 3 || ndims(uy) ~= 3 || ndims(uz) ~= 3 ...
        || any(size(ux) ~= [ny, nx, nz]) ...
        || any(size(uy) ~= [ny, nx, nz]) ...
        || any(size(uz) ~= [ny, nx, nz])
    return;
end

ix = unique([1:sx:nx, nx]);
iy = unique([1:sy:ny, ny]);
iz = sampled_indices_excluding_upper(nz, sz);

[Xg, Yg, Zg] = meshgrid(x(ix), y(iy), z(iz));
U = ux(iy, ix, iz);
V = uy(iy, ix, iz);
W = uz(iy, ix, iz);

cmap = colormap_values(opts.fiber_deviation_colormap, opts.fiber_deviation_color_bins);
if plot_colored_fiber_quivers(ax, Xg, Yg, Zg, U, V, W, scale, opts, cmap)
    apply_fiber_deviation_colorbar(ax, opts, cmap);
end
end

function drew_any = plot_colored_fiber_quivers(ax, Xg, Yg, Zg, U, V, W, scale, opts, cmap)
drew_any = false;
deviation = fiber_deviation_degrees(U, V, W, opts.fiber_straight_direction);
valid = isfinite(U) & isfinite(V) & isfinite(W) & isfinite(deviation);
if ~any(valid(:))
    return;
end

max_angle = opts.fiber_deviation_max_angle_deg;
bin_count = size(cmap, 1);
bin_value = min(1, max(0, deviation ./ max_angle));
bin_idx = min(bin_count, max(1, floor(bin_value .* bin_count) + 1));

for b = 1:bin_count
    mask = valid & (bin_idx == b);
    if ~any(mask(:))
        continue;
    end
    Uc = U;
    Vc = V;
    Wc = W;
    Uc(~mask) = NaN;
    Vc(~mask) = NaN;
    Wc(~mask) = NaN;
    quiver3(ax, Xg, Yg, Zg, Uc, Vc, Wc, scale, ...
        'Color', cmap(b, :), ...
        'LineWidth', opts.fiber_line_width, ...
        'MaxHeadSize', 0.35);
    drew_any = true;
end
end

function apply_fiber_deviation_colorbar(ax, opts, cmap)
if ~opts.show_fiber_deviation_colorbar
    return;
end
colormap(ax, cmap);
caxis(ax, [0 opts.fiber_deviation_max_angle_deg]);
cb = colorbar(ax);
ylabel(cb, 'Fiber deviation from straight direction (deg)');
end

function deviation = fiber_deviation_degrees(U, V, W, direction)
mag = sqrt(U.^2 + V.^2 + W.^2);
dotv = (U .* direction(1) + V .* direction(2) + W .* direction(3)) ./ max(mag, eps);
% Fiber orientation is axial, so opposite signed longitudinal vectors are still straight.
dotv = abs(min(1, max(-1, dotv)));
deviation = acosd(dotv);
deviation(mag <= eps) = NaN;
end

function cmap = colormap_values(value, n)
if isnumeric(value) && size(value, 2) == 3 && size(value, 1) >= 2
    source = clamp01(double(value));
    xi = linspace(1, size(source, 1), n);
    cmap = interp1(1:size(source, 1), source, xi, 'linear');
    return;
end
name = 'turbo';
if (exist('isstring', 'builtin') || exist('isstring', 'file')) && isstring(value)
    value = char(value);
end
if ischar(value) && ~isempty(strtrim(value))
    name = strtrim(value);
end
try
    cmap = feval(name, n);
catch
    if strcmpi(name, 'turbo')
        cmap = jet(n);
    else
        try
            cmap = parula(n);
        catch
            cmap = jet(n);
        end
    end
end
cmap = clamp01(double(cmap(:, 1:3)));
end

function idx = sampled_indices_excluding_upper(n, step)
upper = max(1, n - 1);
idx = unique(1:step:upper);
if isempty(idx)
    idx = 1;
end
end

function rgb = color_row(colors, idx)
if isempty(colors) || size(colors, 2) ~= 3
    rgb = [0.5 0.5 0.5];
    return;
end
row = mod(idx - 1, size(colors, 1)) + 1;
rgb = colors(row, :);
end

function out = rgb_matrix(value, fallback)
if isnumeric(value) && size(value, 2) == 3
    out = clamp01(double(value));
else
    out = fallback;
end
end

function rgb = rgb_value(value, fallback)
rgb = fallback;
if isnumeric(value) && numel(value) == 3
    rgb = clamp01(double(value(:))');
    return;
end
if (exist('isstring', 'builtin') || exist('isstring', 'file')) && isstring(value)
    value = char(value);
end
if ischar(value)
    s = strtrim(value);
    if numel(s) == 7 && s(1) == '#'
        vals = sscanf(s(2:end), '%2x%2x%2x');
        if numel(vals) == 3
            rgb = double(vals(:))' ./ 255.0;
        end
    end
end
end

function y = clamp01(x)
y = min(1, max(0, double(x)));
end
