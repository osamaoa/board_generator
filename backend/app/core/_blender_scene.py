"""Blender-side scene builder invoked by ``boards export-blender``.

This file intentionally imports bpy and must run inside Blender's Python runtime.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import bpy
from mathutils import Matrix, Vector


def _input(node, name: str):
    socket = node.inputs.get(name)
    if socket is None:
        raise RuntimeError(f"Blender node {node.name!r} has no input {name!r}.")
    return socket


def _set_principled_defaults(shader, *, color, roughness: float) -> None:
    _input(shader, "Base Color").default_value = (*color, 1.0)
    _input(shader, "Roughness").default_value = roughness
    metallic = shader.inputs.get("Metallic")
    if metallic is not None:
        metallic.default_value = 0.0
    specular = shader.inputs.get("Specular IOR Level") or shader.inputs.get("Specular")
    if specular is not None:
        specular.default_value = 0.28
    coat = shader.inputs.get("Coat Weight") or shader.inputs.get("Clearcoat")
    if coat is not None:
        coat.default_value = 0.035


def _wood_material(name: str, image_path: str):
    material = bpy.data.materials.new(name=name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()

    output = nodes.new("ShaderNodeOutputMaterial")
    output.location = (720, 30)
    shader = nodes.new("ShaderNodeBsdfPrincipled")
    shader.location = (430, 30)
    _set_principled_defaults(shader, color=(0.52, 0.27, 0.10), roughness=0.42)
    links.new(shader.outputs["BSDF"], output.inputs["Surface"])

    texcoord = nodes.new("ShaderNodeTexCoord")
    texcoord.location = (-900, 20)
    mapping = nodes.new("ShaderNodeMapping")
    mapping.location = (-700, 20)
    links.new(texcoord.outputs["UV"], mapping.inputs["Vector"])

    image = bpy.data.images.load(str(image_path), check_existing=True)
    try:
        image.colorspace_settings.name = "sRGB"
    except TypeError:
        pass
    texture = nodes.new("ShaderNodeTexImage")
    texture.name = f"{name}_SurfaceImage"
    texture.label = Path(image_path).name
    texture.image = image
    texture.interpolation = "Smart"
    texture.extension = "EXTEND"
    texture.location = (-480, 170)
    links.new(mapping.outputs["Vector"], texture.inputs["Vector"])

    color_adjust = nodes.new("ShaderNodeHueSaturation")
    color_adjust.location = (0, 210)
    color_adjust.inputs["Saturation"].default_value = 0.92
    color_adjust.inputs["Value"].default_value = 0.96
    links.new(texture.outputs["Color"], color_adjust.inputs["Color"])
    links.new(color_adjust.outputs["Color"], shader.inputs["Base Color"])

    grayscale = nodes.new("ShaderNodeRGBToBW")
    grayscale.location = (-80, -40)
    links.new(texture.outputs["Color"], grayscale.inputs["Color"])

    roughness_ramp = nodes.new("ShaderNodeValToRGB")
    roughness_ramp.location = (160, -125)
    roughness_ramp.color_ramp.elements[0].position = 0.18
    roughness_ramp.color_ramp.elements[0].color = (0.31, 0.31, 0.31, 1.0)
    roughness_ramp.color_ramp.elements[1].position = 0.82
    roughness_ramp.color_ramp.elements[1].color = (0.49, 0.49, 0.49, 1.0)
    links.new(grayscale.outputs["Val"], roughness_ramp.inputs["Fac"])
    links.new(roughness_ramp.outputs["Color"], shader.inputs["Roughness"])

    grain_bump = nodes.new("ShaderNodeBump")
    grain_bump.location = (170, -330)
    grain_bump.inputs["Strength"].default_value = 0.16
    grain_bump.inputs["Distance"].default_value = 0.0005
    links.new(grayscale.outputs["Val"], grain_bump.inputs["Height"])

    noise = nodes.new("ShaderNodeTexNoise")
    noise.location = (-300, -440)
    noise.noise_dimensions = "3D"
    noise.inputs["Scale"].default_value = 180.0
    noise.inputs["Detail"].default_value = 3.5
    noise.inputs["Roughness"].default_value = 0.72
    noise.inputs["Distortion"].default_value = 0.12
    links.new(mapping.outputs["Vector"], noise.inputs["Vector"])

    micro_bump = nodes.new("ShaderNodeBump")
    micro_bump.location = (410, -285)
    micro_bump.inputs["Strength"].default_value = 0.11
    micro_bump.inputs["Distance"].default_value = 0.00012
    links.new(noise.outputs["Fac"], micro_bump.inputs["Height"])
    links.new(grain_bump.outputs["Normal"], micro_bump.inputs["Normal"])
    links.new(micro_bump.outputs["Normal"], shader.inputs["Normal"])
    return material


def _end_material():
    material = bpy.data.materials.new(name="EndCrossSections_Placeholder")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    shader = nodes.get("Principled BSDF")
    output = nodes.get("Material Output")
    _set_principled_defaults(shader, color=(0.26, 0.105, 0.035), roughness=0.58)

    texture = nodes.new("ShaderNodeTexNoise")
    texture.inputs["Scale"].default_value = 7.0
    texture.inputs["Detail"].default_value = 3.0
    texture.inputs["Roughness"].default_value = 0.75
    ramp = nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].color = (0.075, 0.025, 0.008, 1.0)
    ramp.color_ramp.elements[1].color = (0.36, 0.15, 0.045, 1.0)
    bump = nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.14
    bump.inputs["Distance"].default_value = 0.00025
    links.new(texture.outputs["Fac"], ramp.inputs["Fac"])
    links.new(ramp.outputs["Color"], shader.inputs["Base Color"])
    links.new(texture.outputs["Fac"], bump.inputs["Height"])
    links.new(bump.outputs["Normal"], shader.inputs["Normal"])
    links.new(shader.outputs["BSDF"], output.inputs["Surface"])
    return material


def _board_mesh(length: float, width: float, thickness: float, materials):
    lx = 0.5 * length
    wy = 0.5 * width
    z0 = 0.001
    z1 = z0 + thickness
    vertices = [
        (-lx, -wy, z0),
        (lx, -wy, z0),
        (lx, wy, z0),
        (-lx, wy, z0),
        (-lx, -wy, z1),
        (lx, -wy, z1),
        (lx, wy, z1),
        (-lx, wy, z1),
    ]
    # The generator's coordinates are remapped as Z->X, X->Y, Y->Z.
    # Material slots follow the generator's source coordinates. After the
    # Z->X, X->Y, Y->Z remap these become Blender +Z, -Z, +Y, and -Y.
    # The generated images already use the MATLAB face orientation, whose
    # compatible corner pairs are 1L-3R, 1R-4L, 2R-3L, and 2L-4R.
    faces = [
        (4, 5, 6, 7),  # top, original +Y, surface 1
        (0, 3, 2, 1),  # bottom, original -Y, surface 2
        (3, 7, 6, 2),  # back, original +X, surface 3
        (0, 1, 5, 4),  # front, original -X, surface 4
        (0, 4, 7, 3),  # -length end
        (1, 2, 6, 5),  # +length end
    ]
    # Image U crosses each physical face; image V follows the board length.
    face_uvs = [
        ((1, 0), (1, 1), (0, 1), (0, 0)),
        ((0, 0), (1, 0), (1, 1), (0, 1)),
        ((0, 0), (1, 0), (1, 1), (0, 1)),
        ((1, 0), (1, 1), (0, 1), (0, 0)),
        ((0, 0), (1, 0), (1, 1), (0, 1)),
        ((0, 0), (1, 0), (1, 1), (0, 1)),
    ]

    mesh = bpy.data.meshes.new("GeneratedBoardMesh")
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    uv_layer = mesh.uv_layers.new(name="BoardSurfaceUV")
    for polygon, coordinates in zip(mesh.polygons, face_uvs):
        for loop_index, uv in zip(polygon.loop_indices, coordinates):
            uv_layer.data[loop_index].uv = uv

    board = bpy.data.objects.new("GeneratedBoard", mesh)
    bpy.context.collection.objects.link(board)
    for material in materials:
        mesh.materials.append(material)
    for index, polygon in enumerate(mesh.polygons):
        polygon.material_index = index if index < 4 else 4

    bevel = board.modifiers.new(name="SoftMilledEdges", type="BEVEL")
    bevel.width = 0.005
    bevel.segments = 5
    bevel.limit_method = "ANGLE"
    bevel.angle_limit = math.radians(25.0)
    bevel.harden_normals = True
    return board


def _add_longitudinal_spin(board, *, width: float, thickness: float):
    """Animate one seamless revolution about the board's geometric long axis."""
    scene = bpy.context.scene
    centre_z = 0.001 + 0.5 * thickness
    rig = bpy.data.objects.new("BoardLongitudinalSpinRig", None)
    bpy.context.collection.objects.link(rig)
    rig.empty_display_type = "PLAIN_AXES"
    rig.empty_display_size = max(0.04, 1.4 * thickness)
    rig.location = (0.0, 0.0, centre_z)
    rig.rotation_mode = "XYZ"

    # Keep frame 1 visually unchanged while moving the pivot to the centreline.
    board.parent = rig
    board.matrix_parent_inverse = Matrix.Identity(4)
    board.location = (0.0, 0.0, -centre_z)

    # Drivers avoid Blender-version-dependent Bezier defaults. The board first
    # lifts clear of the studio floor, rotates at constant angular velocity,
    # then settles back into the exact frame-1 pose.
    rig.rotation_euler = (0.0, 0.0, 0.0)
    spin_curve = rig.driver_add("rotation_euler", 0)
    spin_curve.driver.type = "SCRIPTED"
    spin_curve.driver.expression = (
        "0.0 if frame <= 31 else "
        "(6.283185307179586 if frame >= 211 else "
        "0.03490658503988659 * (frame - 31))"
    )

    clearance_centre_z = math.hypot(0.5 * width, 0.5 * thickness) + 0.002
    lift = max(0.0, clearance_centre_z - centre_z)
    lift_curve = rig.driver_add("location", 2)
    lift_curve.driver.type = "SCRIPTED"
    lift_curve.driver.expression = (
        f"{centre_z:.12g} + {lift:.12g} * ("
        "(0.5 - 0.5 * cos(0.10471975511965977 * (frame - 1))) "
        "if frame < 31 else "
        "(1.0 if frame <= 211 else "
        "0.5 - 0.5 * cos(0.10471975511965977 * (241 - frame))))"
    )

    rig["animation"] = "lift, 360 degree longitudinal-axis spin, and settle"
    rig["loop_frames"] = "1-240; frame 241 duplicates frame 1"
    rig["motion_timing"] = "lift 1-31, spin 31-211, settle 211-241"
    scene.frame_start = 1
    scene.frame_end = 240
    scene.render.fps = 30
    scene.render.fps_base = 1.0
    scene.frame_set(1)
    return rig


def _ground_material():
    material = bpy.data.materials.new(name="StudioGround")
    material.use_nodes = True
    shader = material.node_tree.nodes.get("Principled BSDF")
    _set_principled_defaults(shader, color=(0.018, 0.022, 0.030), roughness=0.76)
    return material


def _look_at(obj, target) -> None:
    direction = Vector(target) - obj.location
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def _area_light(name: str, location, energy: float, size: float, color, target):
    data = bpy.data.lights.new(name=name, type="AREA")
    data.energy = energy
    data.shape = "DISK"
    data.size = size
    data.color = color
    obj = bpy.data.objects.new(name, data)
    bpy.context.collection.objects.link(obj)
    obj.location = location
    _look_at(obj, target)
    return obj


def _build_scene(payload) -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for datablocks in (bpy.data.meshes, bpy.data.curves, bpy.data.materials, bpy.data.cameras, bpy.data.lights):
        for datablock in list(datablocks):
            if datablock.users == 0:
                datablocks.remove(datablock)

    dimensions = payload["dimensions_mm"]
    width = float(dimensions["width"]) / 1000.0
    thickness = float(dimensions["thickness"]) / 1000.0
    length = float(dimensions["length"]) / 1000.0
    if min(width, thickness, length) <= 0.0:
        raise RuntimeError("Board dimensions must all be positive.")

    face_names = (
        "Surface_1_SourcePositiveY_BlenderPositiveZ_Top",
        "Surface_2_SourceNegativeY_BlenderNegativeZ_Bottom",
        "Surface_3_SourcePositiveX_BlenderPositiveY_Back",
        "Surface_4_SourceNegativeX_BlenderNegativeY_Front",
    )
    materials = [
        _wood_material(name, image_path)
        for name, image_path in zip(face_names, payload["surface_paths"])
    ]
    materials.append(_end_material())
    board = _board_mesh(length, width, thickness, materials)
    spin_rig = _add_longitudinal_spin(board, width=width, thickness=thickness)
    board["source_stem"] = payload["stem"]
    board["surface_source"] = payload["surface_source"]
    board["dimensions_mm"] = [float(dimensions[key]) for key in ("length", "width", "thickness")]
    board["face_mapping"] = (
        "source 1:+Y=Blender +Z/top, 2:-Y=Blender -Z/bottom, "
        "3:+X=Blender +Y/back, 4:-X=Blender -Y/front"
    )
    board["seam_mapping"] = "1L-3R, 1R-4L, 2R-3L, 2L-4R"
    board["end_cross_sections"] = "procedural placeholder; presentation camera hides these faces"
    board["animation_rig"] = spin_rig.name

    bpy.ops.mesh.primitive_plane_add(size=max(2.4 * length, 1.8), location=(0.0, 0.0, 0.0))
    ground = bpy.context.object
    ground.name = "StudioGround"
    ground.data.materials.append(_ground_material())

    target = (0.0, 0.0, 0.55 * thickness)
    _area_light(
        "Key_Softbox",
        (-0.18 * length, -0.72 * length, 0.88 * length),
        7.0,
        0.72 * length,
        (1.0, 0.78, 0.58),
        target,
    )

    camera_data = bpy.data.cameras.new("PresentationCamera")
    camera = bpy.data.objects.new("PresentationCamera", camera_data)
    bpy.context.collection.objects.link(camera)
    # Match the manually approved 00002 presentation: a 50 mm perspective
    # camera pitched down 45 degrees, centered across the board length.  The
    # distance scales with length while the aim height follows board thickness,
    # retaining the same composition for other board dimensions.
    camera_distance = 1.23478943 * length
    camera.location = (
        0.0,
        -camera_distance,
        camera_distance + 0.96018444 * thickness,
    )
    camera.rotation_euler = (math.radians(45.0), 0.0, 0.0)
    camera.data.type = "PERSP"
    camera.data.lens = 50.00163269
    bpy.context.scene.camera = camera

    scene = bpy.context.scene
    if payload["render_engine"] == "eevee":
        engine_items = scene.render.bl_rna.properties["engine"].enum_items.keys()
        scene.render.engine = "BLENDER_EEVEE_NEXT" if "BLENDER_EEVEE_NEXT" in engine_items else "BLENDER_EEVEE"
    else:
        scene.render.engine = "CYCLES"
    if scene.render.engine == "CYCLES":
        scene.cycles.samples = int(payload["samples"])
        scene.cycles.use_denoising = True
    else:
        scene.render.image_settings.color_mode = "RGBA"
    scene.render.resolution_x = 1200
    scene.render.resolution_y = 700
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.film_transparent = False
    scene.render.filepath = payload["preview_path"]
    scene.render.image_settings.color_mode = "RGB"
    scene.world.use_nodes = True
    world_background = scene.world.node_tree.nodes.get("Background")
    world_background.inputs["Color"].default_value = (0.012, 0.016, 0.024, 1.0)
    world_background.inputs["Strength"].default_value = 0.12
    scene.render.film_transparent = False
    scene.render.use_file_extension = True
    try:
        scene.view_settings.look = "AgX - Medium High Contrast"
    except TypeError:
        pass
    scene.view_settings.exposure = 0.0

    if payload["render_preview"]:
        bpy.ops.render.render(write_still=True)
    if payload["pack_images"]:
        bpy.ops.file.pack_all()
    bpy.ops.wm.save_as_mainfile(filepath=payload["blend_path"], check_existing=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--payload", required=True)
    argv = []
    if "--" in __import__("sys").argv:
        argv = __import__("sys").argv[__import__("sys").argv.index("--") + 1 :]
    args = parser.parse_args(argv)
    payload = json.loads(Path(args.payload).read_text(encoding="utf-8"))
    _build_scene(payload)


if __name__ == "__main__":
    main()
