"""Capturing the maps that go to ComfyUI, and snapshotting the projection.

Everything the backend receives is a plain render:

  beauty      Workbench studio render       Ă˘â‚¬â€ť the img2img reference
  depth       Cycles/EEVEE emission render  Ă˘â‚¬â€ť camera-space Z, normalised
  normal      Cycles/EEVEE emission render  Ă˘â‚¬â€ť camera-space normal
  visibility  Cycles emission render        Ă˘â‚¬â€ť 1 - occluded, times angle falloff

Rendering rather than baking is deliberate. A render resolves occlusion with the
z-buffer for free, and Ă˘â‚¬â€ť crucially Ă˘â‚¬â€ť the projection UV produced by a UV_PROJECT
modifier *is* screen space, so a screen-space render is already in the right
coordinate space to be sampled through that UV map. Baking these through an
overlapping projection UV would let back faces fight front faces for the same
texel; rendering simply cannot have that problem.

This also avoids Blender 5.x's compositor rework entirely: ``scene.node_tree`` is
gone in 5.1 (replaced by ``scene.compositing_node_group``), and the render-layer
pass sockets no longer appear on the node, so the classic
"Render Layers -> File Output" pass export does not port. None of that is needed
here.
"""

import glob
import os
import tempfile
from contextlib import contextmanager, nullcontext

import bpy
import numpy as np
from mathutils import Vector

CAMERA_NAME = "PL_Projector"
UV_PREFIX = "PL_Proj_"
UV_ATTR_PREFIX = "PL_ProjUV_"

# Fallback studio backdrop. It doubles as the light source when the scene has
# no world of its own, so it cannot be too dark or the mesh renders black.
BACKDROP = (0.35, 0.35, 0.35)

# Guide maps and masks are data. Any world would show up as a non-zero
# background in them Ă˘â‚¬â€ť which is exactly what made the mask look "shaded" and fed
# ControlNet a grey backdrop.
BLACK_WORLD = (0.0, 0.0, 0.0)


# Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬ camera Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬


def ensure_camera(context):
    """Return the addon's projector camera, creating it if needed."""
    cam = bpy.data.objects.get(CAMERA_NAME)
    if cam is None or cam.type != "CAMERA":
        data = bpy.data.cameras.new(CAMERA_NAME)
        cam = bpy.data.objects.new(CAMERA_NAME, data)
        context.scene.collection.objects.link(cam)
        cam.location = (0.0, -3.0, 1.5)
        cam.rotation_euler = (1.2, 0.0, 0.0)
    if context.scene.camera is None:
        context.scene.camera = cam
    return cam


def _find_view3d(context):
    """(window, area, window_region, space, region_3d) for a 3D viewport."""
    area = context.area if getattr(context, "area", None) else None
    region = context.region if getattr(context, "region", None) else None
    region_3d = getattr(context, "region_data", None)

    if area is not None and area.type == "VIEW_3D" and region_3d is not None:
        if region is None:
            region = next((r for r in area.regions if r.type == "WINDOW"), None)
        return context.window, area, region, area.spaces.active, region_3d

    for window in context.window_manager.windows:
        for candidate in window.screen.areas:
            if candidate.type != "VIEW_3D":
                continue
            space = candidate.spaces.active
            if space is None or space.region_3d is None:
                continue
            region = next((r for r in candidate.regions if r.type == "WINDOW"), None)
            return window, candidate, region, space, space.region_3d
    return None, None, None, None, None


def snap_to_view(context, camera):
    """Line the projector camera up with what the artist is looking at.

    Prefers Blender's own "align camera to view", which also matches the
    viewport zoom; copying the view matrix alone gets the angle right but can
    leave the framing slightly off.
    """
    window, area, region, space, region_3d = _find_view3d(context)
    if region_3d is None:
        raise RuntimeError("Open a 3D viewport first, then snap the camera to it.")

    context.scene.camera = camera
    was_perspective = region_3d.view_perspective

    if window is not None and area is not None and region is not None:
        override = {
            "window": window,
            "screen": window.screen,
            "area": area,
            "region": region,
            "space_data": space,
            "region_data": region_3d,
            "scene": context.scene,
        }
        try:
            with context.temp_override(**override):
                bpy.ops.view3d.camera_to_view()
            _restore_perspective(region_3d, was_perspective)
            return camera
        except Exception:  # noqa: BLE001 Ă˘â‚¬â€ť fall back to copying the matrix
            pass

    camera.matrix_world = region_3d.view_matrix.inverted()
    return camera


def _restore_perspective(region_3d, was_perspective):
    """Stay in whatever view the artist was using.

    Aligning the camera to the view can drop the viewport into camera view;
    being left looking through the projector after every Generate is jarring, so
    put the view back the way it was.
    """
    try:
        if was_perspective and region_3d.view_perspective != was_perspective:
            region_3d.view_perspective = was_perspective
    except (TypeError, AttributeError):
        pass


# Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬ projection snapshot Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬
#
# Blender silently caps UV layers at eight (asking for 40 creates 8, with no
# error), so a UV map per layer cannot hold 30 projections. The projection is
# stored as a FLOAT_VECTOR / CORNER attribute instead Ă˘â‚¬â€ť attributes have no such
# cap, and a corner FLOAT_VECTOR deliberately does NOT show up in the UV Maps
# list (a FLOAT2 corner attribute would).
#
# A UV_PROJECT modifier can only write into a UV layer, so one scratch layer is
# created, filled, copied into the attribute, and removed again. Painting needs a
# real UV map too, but only one layer is ever painted at a time, so a single
# shared scratch map is enough: its contents are copied from whichever layer's
# attribute is being painted.

TEMP_UV_NAME = "PL_TmpProject"
PAINT_UV_NAME = "PL_Paint"


def next_uv_name(obj):
    index = 0
    existing = {uv.name for uv in obj.data.uv_layers}
    while f"{UV_PREFIX}{index:03d}" in existing:
        index += 1
    return f"{UV_PREFIX}{index:03d}"


def _next_attr_name(obj):
    index = 0
    existing = {attr.name for attr in obj.data.attributes}
    while f"{UV_ATTR_PREFIX}{index:03d}" in existing:
        index += 1
    return f"{UV_ATTR_PREFIX}{index:03d}"


def _uv_layer_to_attribute(obj, uv_name, attr_name):
    """Copy a UV layer's data into a corner FLOAT_VECTOR attribute."""
    uv = obj.data.uv_layers.get(uv_name)
    if uv is None:
        return False
    loops = len(obj.data.loops)
    flat = np.empty(loops * 2, dtype=np.float32)
    uv.data.foreach_get("uv", flat)

    vec = np.zeros(loops * 3, dtype=np.float32)
    vec[0::3] = flat[0::2]
    vec[1::3] = flat[1::2]

    existing = obj.data.attributes.get(attr_name)
    if existing is not None:
        obj.data.attributes.remove(existing)
    attr = obj.data.attributes.new(name=attr_name, type="FLOAT_VECTOR",
                                   domain="CORNER")
    attr.data.foreach_set("vector", vec)
    obj.data.update()
    return True


def attribute_to_paint_uv(obj, attr_name, uv_name=PAINT_UV_NAME):
    """Copy a layer's projection attribute into the shared paint UV map.

    Returns the UV map name, or None if there is nowhere to put it (the mesh
    already uses all eight UV slots and has no scratch map yet).
    """
    attr = obj.data.attributes.get(attr_name)
    if attr is None:
        return None

    uv = obj.data.uv_layers.get(uv_name)
    if uv is None:
        if len(obj.data.uv_layers) >= 8:
            return None
        uv = obj.data.uv_layers.new(name=uv_name)

    loops = len(obj.data.loops)
    vec = np.empty(loops * 3, dtype=np.float32)
    attr.data.foreach_get("vector", vec)
    flat = np.empty(loops * 2, dtype=np.float32)
    flat[0::2] = vec[0::3]
    flat[1::2] = vec[1::3]
    uv.data.foreach_set("uv", flat)
    obj.data.update()
    return uv_name


def snapshot_projection(context, obj, camera, attr_name=None):
    """Freeze the camera projection of ``obj`` into a corner attribute.

    Uses a UV_PROJECT modifier with the camera as projector on a scratch UV
    layer, then moves the result into an attribute and deletes the scratch layer
    so the mesh's UV slot count is unchanged.
    """
    if obj.data.users > 1:
        raise RuntimeError(
            f"'{obj.name}' shares its mesh data with another object. "
            "Make it single-user (Object > Relations > Make Single User) first."
        )

    attr_name = attr_name or _next_attr_name(obj)
    scratch = obj.data.uv_layers.get(TEMP_UV_NAME)
    if scratch is None:
        if len(obj.data.uv_layers) >= 8:
            raise RuntimeError(
                f"'{obj.name}' already uses all 8 UV maps. Remove one, or bake "
                "the stack first."
            )
        scratch = obj.data.uv_layers.new(name=TEMP_UV_NAME)

    res_x, res_y = projection_resolution(context, camera)
    # Store names, not references: applying a modifier rebuilds the mesh data,
    # which leaves any previously held uv_layers reference dangling.
    previous_active = obj.data.uv_layers.active
    was_active = previous_active.name if previous_active is not None else ""
    mode = obj.mode

    try:
        if mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")
        _select_only(context, obj)

        mod = obj.modifiers.new(name="PL_UVProject", type="UV_PROJECT")
        mod.projector_count = 1
        mod.projectors[0].object = camera
        mod.uv_layer = TEMP_UV_NAME
        obj.data.uv_layers.active = obj.data.uv_layers.get(TEMP_UV_NAME)
        aspect = res_x / max(res_y, 1)
        mod.aspect_x = aspect if aspect > 1.0 else 1.0
        mod.aspect_y = (1.0 / aspect) if aspect < 1.0 else 1.0

        bpy.ops.object.modifier_apply(modifier=mod.name)
        _uv_layer_to_attribute(obj, TEMP_UV_NAME, attr_name)
    finally:
        leftover = obj.data.uv_layers.get(TEMP_UV_NAME)
        if leftover is not None:
            obj.data.uv_layers.remove(leftover)
        if was_active and was_active in obj.data.uv_layers:
            obj.data.uv_layers.active = obj.data.uv_layers[was_active]
        if not obj.data.uv_layers.active and len(obj.data.uv_layers):
            obj.data.uv_layers.active = obj.data.uv_layers[0]
        if mode != "OBJECT":
            try:
                bpy.ops.object.mode_set(mode=mode)
            except RuntimeError:
                pass

    return attr_name


def matrix_from_projection(projection):
    """Rebuild the stored camera matrix.

    A ``subtype="MATRIX"`` FloatVectorProperty iterates as four row Vectors but
    slices into something Matrix() will not take, while a plain list of 16
    floats iterates as scalars. Accept both shapes rather than assuming one.
    """
    from mathutils import Matrix

    flat = []
    for item in projection.cam_matrix:
        try:
            flat.extend(float(x) for x in item)
        except TypeError:
            flat.append(float(item))
    if len(flat) != 16:
        return Matrix.Identity(4)
    return Matrix((flat[0:4], flat[4:8], flat[8:12], flat[12:16]))


def camera_basis(matrix):
    """(right, up, forward) unit vectors of a camera matrix."""
    quat = matrix.to_quaternion()
    return (
        quat @ Vector((1.0, 0.0, 0.0)),
        quat @ Vector((0.0, 1.0, 0.0)),
        quat @ Vector((0.0, 0.0, -1.0)),
    )


def projection_is_valid(projection):
    """False when no camera matrix was ever recorded.

    An untouched ``cam_matrix`` is all zeros, which would invert to nonsense and
    project every loop to the same point. Callers fall back to the stamped
    attribute in that case.
    """
    try:
        flat = []
        for item in projection.cam_matrix:
            try:
                flat.extend(float(x) for x in item)
            except TypeError:
                flat.append(float(item))
    except (AttributeError, TypeError):
        return False
    return len(flat) == 16 and any(abs(value) > 1e-9 for value in flat)


def projection_half_extents(projection):
    """Half the camera frame at unit depth, honouring sensor fit.

    ``sensor_width`` is the physical film size, so a 50 mm lens on a 36 mm sensor
    sees 18/50 of a unit at depth 1 across the fitted axis.
    """
    lens = max(float(projection.lens), 1e-6)
    sensor = max(float(projection.sensor_width), 1e-6)
    res_x = max(int(projection.res_x), 1)
    res_y = max(int(projection.res_y), 1)
    fit = str(projection.sensor_fit).upper()

    portrait = fit == "VERTICAL" or (fit == "AUTO" and res_y > res_x)
    if portrait:
        half_y = (sensor * 0.5) / lens
        half_x = half_y * (res_x / res_y)
    else:
        half_x = (sensor * 0.5) / lens
        half_y = half_x * (res_y / res_x)
    return half_x, half_y


def project_loops_uv(obj, projection):
    """Per-loop projection UVs, rebuilt from the stored camera.

    This is the mapping UV_PROJECT produces, derived analytically from the camera
    matrix plus lens/sensor, so a mesh needs no stamped attribute of its own.
    That is what lets atlas flattening include meshes that were never projected.

    Returns a ``(loops, 2)`` float array, or None when there is nothing to do.
    """
    mesh = obj.data
    loops = len(mesh.loops)
    if not loops or not projection_is_valid(projection):
        return None

    matrix = matrix_from_projection(projection)
    inverse = np.array(matrix.inverted_safe(), dtype=np.float64)

    loop_vertex = np.empty(loops, dtype=np.int32)
    mesh.loops.foreach_get("vertex_index", loop_vertex)

    count = len(mesh.vertices)
    co = np.empty(count * 3, dtype=np.float64)
    mesh.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)

    world = np.array(obj.matrix_world, dtype=np.float64)
    ones = np.ones((count, 1), dtype=np.float64)
    world_points = (world @ np.hstack([co, ones]).T).T[:, :3]

    camera_space = (inverse @ np.hstack([world_points, ones]).T).T[:, :3]

    # The camera looks down -Z, so depth is positive in front of it.
    depth = -camera_space[:, 2]
    safe = np.where(np.abs(depth) < 1e-6, 1e-6, depth)

    half_x, half_y = projection_half_extents(projection)
    u = 0.5 + (camera_space[:, 0] / safe) / (2.0 * half_x) + float(projection.shift_x)
    v = 0.5 + (camera_space[:, 1] / safe) / (2.0 * half_y) + float(projection.shift_y)

    uvs = np.empty((loops, 2), dtype=np.float64)
    uvs[:, 0] = u[loop_vertex]
    uvs[:, 1] = v[loop_vertex]
    return uvs



def projection_resolution(context, camera):
    from ..props import resolution

    return resolution(context)


# Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬ render plumbing Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬


class _RenderState:
    """Save and restore everything the capture passes touch."""

    def __init__(self, context):
        self.context = context
        self.scene = context.scene
        self.saved = {}

    def __enter__(self):
        r = self.scene.render
        view = self.scene.view_settings
        self.saved = {
            "engine": r.engine,
            "filepath": r.filepath,
            "format": r.image_settings.file_format,
            "color_mode": r.image_settings.color_mode,
            "res_x": r.resolution_x,
            "res_y": r.resolution_y,
            "pct": r.resolution_percentage,
            "transparent": r.film_transparent,
            "camera": self.scene.camera,
            "use_nodes": self.scene.use_nodes,
            "view_transform": view.view_transform,
            "look": view.look,
        }
        if hasattr(r, "use_persistent_data"):
            self.saved["persistent"] = r.use_persistent_data
        return self

    def __exit__(self, *exc):
        r = self.scene.render
        view = self.scene.view_settings
        r.engine = self.saved["engine"]
        r.filepath = self.saved["filepath"]
        r.image_settings.file_format = self.saved["format"]
        r.image_settings.color_mode = self.saved["color_mode"]
        r.resolution_x = self.saved["res_x"]
        r.resolution_y = self.saved["res_y"]
        r.resolution_percentage = self.saved["pct"]
        r.film_transparent = self.saved["transparent"]
        self.scene.camera = self.saved["camera"]
        try:
            self.scene.use_nodes = self.saved["use_nodes"]
        except Exception:
            pass
        try:
            view.view_transform = self.saved["view_transform"]
            view.look = self.saved["look"]
        except Exception:
            pass
        if "persistent" in self.saved:
            r.use_persistent_data = self.saved["persistent"]
        return False

    def apply(self, width, height, camera):
        r = self.scene.render
        r.resolution_x = width
        r.resolution_y = height
        r.resolution_percentage = 100
        r.image_settings.file_format = "PNG"
        r.image_settings.color_mode = "RGB"
        r.film_transparent = False
        self.scene.camera = camera

        # Everything the addon writes is data (a guide map, a mask, a reference
        # to hand to the backend), so the display transform must not touch it.
        # Left on the default (AgX) a mask of 1.0 saves as ~0.77 and the
        # projection can never reach full strength.
        try:
            self.scene.view_settings.view_transform = "Standard"
            self.scene.view_settings.look = "None"
        except TypeError:
            pass


# Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬ temp material helpers Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬


class _TempMaterial:
    """Swap an object's materials for one temporary material, then restore.

    Bakes and renders read the object's material slots, so this is the least
    invasive way to force a specific shader for one pass. The face
    ``material_index`` values are untouched, so restoring the identical slot
    list puts everything back exactly as it was.
    """

    def __init__(self, obj, material):
        self.obj = obj
        self.material = material
        self.original = list(obj.data.materials)

    def __enter__(self):
        self.obj.data.materials.clear()
        self.obj.data.materials.append(self.material)
        return self.material

    def __exit__(self, *exc):
        self.obj.data.materials.clear()
        for mat in self.original:
            self.obj.data.materials.append(mat)
        if self.material.name in bpy.data.materials:
            bpy.data.materials.remove(self.material)
        return False


def _emission_material(name):
    """A bare material whose surface is an Emission shader."""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    out.location = (400, 0)
    emis = nt.nodes.new("ShaderNodeEmission")
    emis.location = (100, 0)
    emis.inputs["Strength"].default_value = 1.0
    nt.links.new(emis.outputs["Emission"], out.inputs["Surface"])
    return mat, emis


def _new_node(nodes, kind, x, y, **props):
    node = nodes.new(kind)
    node.location = (x, y)
    for key, value in props.items():
        setattr(node, key, value)
    return node


def _const_vector(nodes, name, vector, x, y):
    node = _new_node(nodes, "ShaderNodeCombineXYZ", x, y)
    node.label = name
    node.inputs[0].default_value = vector[0]
    node.inputs[1].default_value = vector[1]
    node.inputs[2].default_value = vector[2]
    return node


def _math(nodes, operation, x, y, defaults=None):
    """A Math node. ``defaults`` maps input index -> default value."""
    node = _new_node(nodes, "ShaderNodeMath", x, y, operation=operation, use_clamp=False)
    for index, value in (defaults or {}).items():
        node.inputs[index].default_value = value
    return node


def _camera_basis(camera):
    """(right, up, forward) unit vectors of the camera in world space."""
    quat = camera.matrix_world.to_quaternion()
    right = quat @ Vector((1.0, 0.0, 0.0))
    up = quat @ Vector((0.0, 1.0, 0.0))
    forward = quat @ Vector((0.0, 0.0, -1.0))
    return right, up, forward


def _camera_space_vector(nodes, links, camera, source_socket, x, y, label):
    """Express a world-space vector in camera space via three dot products."""
    right, up, forward = _camera_basis(camera)
    out = []
    for index, (name, axis) in enumerate(
        (("Right", right), ("Up", up), ("Forward", forward))
    ):
        const = _const_vector(nodes, f"{label} {name}", axis, x, y - index * 180)
        dot = _new_node(nodes, "ShaderNodeVectorMath", x + 180, y - index * 180)
        dot.operation = "DOT_PRODUCT"
        links.new(source_socket, dot.inputs[0])
        links.new(const.outputs["Vector"], dot.inputs[1])
        out.append(dot.outputs["Value"])
    return out


def _object_depth_range(obj, camera):
    """Camera-space Z range of the object's vertices, used to normalise depth."""
    _, _, forward = _camera_basis(camera)
    origin = camera.matrix_world.translation
    matrix = obj.matrix_world
    depths = [
        (matrix @ v.co - origin).dot(forward) for v in obj.data.vertices
    ]
    if not depths:
        return 0.0, 1.0
    near, far = min(depths), max(depths)
    if far - near < 1e-6:
        far = near + 1e-3
    return near, far


# Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬ the passes Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬


class _TempWorld:
    """Force a neutral, evenly-lit world for the beauty pass.

    EEVEE needs light from somewhere; a grey background gives flat ambient
    lighting so the reference render is readable even in a scene with no lamps,
    and so previously projected layers show up in it.
    """

    def __init__(self, context, color=BACKDROP):
        self.scene = context.scene
        self.color = color
        self.created = False
        self.world = None
        self.saved = {}

    def __enter__(self):
        world = self.scene.world
        if world is None:
            world = bpy.data.worlds.new("PL_tmp_world")
            self.scene.world = world
            self.created = True
        self.world = world
        self.saved["use_nodes"] = world.use_nodes
        world.use_nodes = True
        tree = world.node_tree
        background = None
        for node in tree.nodes:
            if node.type == "BACKGROUND":
                background = node
                break
        if background is None:
            background = tree.nodes.new("ShaderNodeBackground")
            out = next((n for n in tree.nodes if n.type == "OUTPUT_WORLD"), None)
            if out is None:
                out = tree.nodes.new("ShaderNodeOutputWorld")
                out.location = (300, 0)
            tree.links.new(background.outputs["Background"], out.inputs["Surface"])
        self.background = background
        self.saved["color"] = tuple(background.inputs["Color"].default_value)
        self.saved["strength"] = background.inputs["Strength"].default_value
        background.inputs["Color"].default_value = (*self.color, 1.0)
        background.inputs["Strength"].default_value = 1.0
        return world

    def __exit__(self, *exc):
        if self.world is None:
            return False
        self.background.inputs["Color"].default_value = self.saved["color"]
        self.background.inputs["Strength"].default_value = self.saved["strength"]
        self.world.use_nodes = self.saved["use_nodes"]
        if self.created:
            bpy.data.worlds.remove(self.world)
            self.scene.world = None
        return False


def _collections_hidden_in_viewport(scene):
    """Names of collections hidden in the viewport, nested ones included.

    The eye on a collection is its own flag; it does not propagate down to
    ``Object.hide_viewport``, so objects inside an eye-hidden collection report
    themselves as visible.
    """
    hidden = set()

    def walk(collection, inherited):
        flagged = inherited or bool(getattr(collection, "hide_viewport", False))
        if flagged:
            hidden.add(collection.name)
        for child in collection.children:
            walk(child, flagged)

    try:
        walk(scene.collection, False)
    except Exception:  # noqa: BLE001
        return hidden
    return hidden


@contextmanager
def respect_viewport_visibility(context):
    """Hide from the render whatever is hidden in the viewport.

    Blender separates the two: the eye icon and the H key set *viewport* flags
    that a render ignores, so an object you hid still turns up in the reference
    image. This makes "what you see is what gets sent" true.

    Both flags have to be checked. They are different properties, and neither
    implies the other:

    * ``hide_viewport`` Ă˘â‚¬â€ť the outliner eye icon. ``hide_get()`` returns **False**
      for these, which is why an earlier version of this only caught the H key.
    * ``hide_get()`` Ă˘â‚¬â€ť the H key (temporary per-view-layer hide).
    * plus any collection hidden by its own eye, which propagates to nothing.

    Local View isolation is still not handled Ă˘â‚¬â€ť that state lives on the 3D view,
    not on the objects, so it cannot be read reliably from here.
    """
    changed = []
    try:
        scene_objects = context.scene.objects
    except AttributeError:
        scene_objects = ()
    hidden_collections = _collections_hidden_in_viewport(context.scene)
    for obj in scene_objects:
        if getattr(obj, "hide_render", False):
            continue  # already excluded, and not ours to restore
        try:
            hidden = bool(obj.hide_get()) or bool(
                getattr(obj, "hide_viewport", False))
        except Exception:  # noqa: BLE001
            hidden = bool(getattr(obj, "hide_viewport", False))
        if not hidden and hidden_collections:
            hidden = any(
                collection.name in hidden_collections
                for collection in getattr(obj, "users_collection", ())
            )
        if hidden:
            obj.hide_render = True
            changed.append(obj)
    try:
        yield len(changed)
    finally:
        for obj in changed:
            obj.hide_render = False


@contextmanager
def only_visible(context, keep_obj):
    """Hide every other object from the render, then restore.

    A batch run walks a collection of co-located meshes (variants, or objects in
    place). Without this, every *other* object would appear in the reference
    render and occlude the one being textured, which makes the sent image
    useless - and the resulting mask would be carved up by neighbours.

    Cameras and lights are left alone: hiding a light would render the reference
    black, which is a far worse failure than a stray object.
    """
    changed = []
    try:
        scene_objects = list(context.scene.objects)
    except AttributeError:
        scene_objects = []
    for obj in scene_objects:
        if keep_obj is not None and obj.as_pointer() == keep_obj.as_pointer():
            continue
        if getattr(obj, "type", "") in {"CAMERA", "LIGHT"}:
            continue
        if getattr(obj, "hide_render", False):
            continue  # already excluded, and not ours to restore
        obj.hide_render = True
        changed.append(obj)
    try:
        yield len(changed)
    finally:
        for obj in changed:
            obj.hide_render = False


@contextmanager
def flat_colour(context, obj):
    """Render the layer stack unlit, the way a diffuse colour view shows it.

    Rewires the active material's Output Surface to an Emission fed by whatever
    drives Base Color, so the whole stack still evaluates but no lighting or
    specular is added. Restored afterwards, link and all.
    """
    material = obj.active_material
    tree = material.node_tree if material is not None and material.use_nodes else None
    if tree is None:
        yield False
        return

    output = None
    for node in tree.nodes:
        if node.type == "OUTPUT_MATERIAL" and getattr(node, "is_active_output", False):
            output = node
            break
    if output is None:
        yield False
        return

    surface = output.inputs["Surface"]
    saved = surface.links[0].from_socket if surface.links else None

    emission = tree.nodes.new("ShaderNodeEmission")
    emission.name = "PL_flat_colour"
    emission.location = (output.location.x - 220, output.location.y - 220)

    bsdf = next((n for n in tree.nodes if n.type == "BSDF_PRINCIPLED"), None)
    if bsdf is not None and bsdf.inputs["Base Color"].links:
        tree.links.new(bsdf.inputs["Base Color"].links[0].from_socket,
                       emission.inputs["Color"])
    elif bsdf is not None:
        emission.inputs["Color"].default_value = tuple(
            bsdf.inputs["Base Color"].default_value)
    elif saved is not None:
        tree.links.new(saved, emission.inputs["Color"])
    tree.links.new(emission.outputs["Emission"], surface)

    try:
        yield True
    finally:
        try:
            tree.nodes.remove(emission)
        except Exception:  # noqa: BLE001
            pass
        if saved is not None:
            tree.links.new(saved, surface)


def render_beauty(context, obj, camera, width, height, filepath, backdrop=BACKDROP):
    """Render used as the img2img reference.

    EEVEE rather than Workbench on purpose: EEVEE evaluates the shader tree, so
    layers projected from earlier angles appear in the reference and new
    generations stay consistent with them. Workbench cannot see node trees.

    The scene's own world is left alone when it has one Ă˘â‚¬â€ť the artist's lighting
    beats anything we would invent. Only an empty scene gets the neutral studio
    world, which also supplies the light (a dark backdrop would leave the mesh
    unlit and black).

    ``flat=True`` sends the stack's own colour unlit instead of a shaded render,
    which is usually a better img2img reference than a lit one.
    """
    scene = context.scene
    settings = getattr(scene, "projlayer", None)
    flat = bool(getattr(settings, "send_what", "BEAUTY") == "FLAT")

    with _RenderState(context) as state:
        state.apply(width, height, camera)
        if flat:
            # Nothing needs to light an emission shader, and a black world keeps
            # the background from tinting the flat colour.
            _configure_emission_engine(context, samples=1)
            with flat_colour(context, obj):
                with _TempWorld(context, BLACK_WORLD):
                    return _render_to_file(context, filepath)

        _configure_emission_engine(context, samples=16)
        if scene.world is None:
            with _TempWorld(context, backdrop):
                return _render_to_file(context, filepath)
        return _render_to_file(context, filepath)


def render_depth(context, obj, camera, width, height, filepath):
    """Camera-space depth as a grayscale map (near = white)."""
    near, far = _object_depth_range(obj, camera)
    mat, emis = _emission_material("PL_depth_tmp")
    nt = mat.node_tree
    nodes, links = nt.nodes, nt.links

    geom = _new_node(nodes, "ShaderNodeNewGeometry", -800, 0)
    cam_loc = _const_vector(nodes, "Camera", camera.matrix_world.translation, -800, -220)
    sub = _new_node(nodes, "ShaderNodeVectorMath", -600, 0, operation="SUBTRACT")
    links.new(geom.outputs["Position"], sub.inputs[0])
    links.new(cam_loc.outputs["Vector"], sub.inputs[1])

    # depth along the view axis, not radial distance
    comps = _camera_space_vector(nodes, links, camera, sub.outputs["Vector"], -400, 0, "Depth")
    forward_component = comps[2]

    map_range = _new_node(nodes, "ShaderNodeMapRange", 100, -100)
    map_range.clamp = True
    map_range.inputs["From Min"].default_value = near
    map_range.inputs["From Max"].default_value = far
    map_range.inputs["To Min"].default_value = 1.0  # near = white
    map_range.inputs["To Max"].default_value = 0.0
    links.new(forward_component, map_range.inputs["Value"])

    combine = _new_node(nodes, "ShaderNodeCombineXYZ", 300, -100)
    for index in range(3):
        links.new(map_range.outputs["Result"], combine.inputs[index])
    links.new(combine.outputs["Vector"], emis.inputs["Color"])

    with _RenderState(context) as state:
        state.apply(width, height, camera)
        _configure_emission_engine(context)
        with _TempWorld(context, BLACK_WORLD):
            with _TempMaterial(obj, mat):
                return _render_to_file(context, filepath)


def render_normal(context, obj, camera, width, height, filepath):
    """Camera-space normal map guide (RGB = XYZ * 0.5 + 0.5)."""
    mat, emis = _emission_material("PL_normal_tmp")
    nodes, links = mat.node_tree.nodes, mat.node_tree.links

    geom = _new_node(nodes, "ShaderNodeNewGeometry", -800, 0)
    comps = _camera_space_vector(nodes, links, camera, geom.outputs["Normal"], -400, 0, "Normal")

    scaled = []
    for index, comp in enumerate(comps):
        # camera space here is (right, up, forward); for a normal map we want
        # +Z toward the viewer, so the forward component is flipped.
        sign = _math(nodes, "MULTIPLY", 100, -index * 180, {1: -1.0 if index == 2 else 1.0})
        links.new(comp, sign.inputs[0])
        remap = _math(nodes, "MULTIPLY_ADD", 280, -index * 180, {1: 0.5, 2: 0.5})
        links.new(sign.outputs[0], remap.inputs[0])
        scaled.append(remap.outputs[0])

    combine = _new_node(nodes, "ShaderNodeCombineXYZ", 460, -180)
    for index, socket in enumerate(scaled):
        links.new(socket, combine.inputs[index])
    links.new(combine.outputs["Vector"], emis.inputs["Color"])

    with _RenderState(context) as state:
        state.apply(width, height, camera)
        _configure_emission_engine(context)
        with _TempWorld(context, BLACK_WORLD):
            with _TempMaterial(obj, mat):
                return _render_to_file(context, filepath)


def _occlusion_socket(nodes, links, camera, geom, x, y):
    """1 where the camera can see the fragment, 0 where something is in front.

    Deliberately *binary*. Angle cutoff and distance limits are NOT baked in
    here Ă˘â‚¬â€ť they are computed live in the material instead, so the mask image
    stays a clean black-and-white occlusion map and those two controls respond
    instantly instead of needing a re-render.
    """
    cam_loc = _const_vector(nodes, "Camera", camera.matrix_world.translation, x, y - 260)

    to_frag = _new_node(nodes, "ShaderNodeVectorMath", x + 200, y, operation="SUBTRACT")
    links.new(geom.outputs["Position"], to_frag.inputs[0])
    links.new(cam_loc.outputs["Vector"], to_frag.inputs[1])

    # The Direction input of a Raycast node is expected to be unit length; an
    # unnormalised direction makes the ray overshoot by the distance factor and
    # hit geometry behind the surface, which reads as everything being occluded.
    direction = _new_node(nodes, "ShaderNodeVectorMath", x + 380, y, operation="NORMALIZE")
    links.new(to_frag.outputs["Vector"], direction.inputs[0])

    dist = _new_node(nodes, "ShaderNodeVectorMath", x + 380, y - 260, operation="LENGTH")
    links.new(to_frag.outputs["Vector"], dist.inputs[0])

    length_eps = _math(nodes, "SUBTRACT", x + 560, y - 260, {1: 1e-3})
    links.new(dist.outputs["Value"], length_eps.inputs[0])

    ray = _new_node(nodes, "ShaderNodeRaycast", x + 740, y - 260)
    links.new(cam_loc.outputs["Vector"], ray.inputs["Position"])
    links.new(direction.outputs["Vector"], ray.inputs["Direction"])
    links.new(length_eps.outputs[0], ray.inputs["Length"])

    visible = _math(nodes, "SUBTRACT", x + 940, y - 260, {0: 1.0})
    links.new(ray.outputs["Is Hit"], visible.inputs[1])
    return visible.outputs[0]


def render_visibility(context, obj, camera, width, height, filepath):
    """Binary occlusion mask, in screen space.

    Because the layer's projection UV is screen space, this renders straight
    into the coordinate space the mask needs Ă˘â‚¬â€ť no bake, and occlusion resolves
    through the z-buffer instead of fighting overlapping UVs. Angle cutoff and
    distance limiting happen live in the material instead of here.
    """
    mat, emis = _emission_material("PL_vis_tmp")
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    geom = _new_node(nodes, "ShaderNodeNewGeometry", -1600, 0)
    total = _occlusion_socket(nodes, links, camera, geom, -1400, 0)

    combine = _new_node(nodes, "ShaderNodeCombineXYZ", 200, -60)
    for index in range(3):
        links.new(total, combine.inputs[index])
    links.new(combine.outputs["Vector"], emis.inputs["Color"])

    with _RenderState(context) as state:
        state.apply(width, height, camera)
        _configure_cycles(context)
        with _TempWorld(context, BLACK_WORLD):
            with _TempMaterial(obj, mat):
                return _render_to_file(context, filepath)


def render_coverage(context, obj, camera, width, height, filepath, layers):
    """How much of this view is already painted, in this camera's screen space.

    Each existing layer's mask is read through *its own* projection UV, so the
    result correctly reflects paint applied from other angles. Multiplied by the
    visibility of this camera, so the inpaint mask only asks for areas the
    camera can actually see.
    """
    mat, emis = _emission_material("PL_coverage_tmp")
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    geom = _new_node(nodes, "ShaderNodeNewGeometry", -1600, 0)

    coverage = None
    for index, layer in enumerate(layers):
        if layer.mask_image is None:
            continue
        y = 1200 + index * 420
        uv = _new_node(nodes, "ShaderNodeAttribute", 0, y)
        uv.attribute_type = "GEOMETRY"
        uv.attribute_name = layer.uv_attr
        tex = _new_node(nodes, "ShaderNodeTexImage", 180, y)
        tex.image = layer.mask_image
        tex.extension = "CLIP"
        links.new(uv.outputs["Vector"], tex.inputs["Vector"])
        bw = _new_node(nodes, "ShaderNodeRGBToBW", 360, y)
        links.new(tex.outputs["Color"], bw.inputs["Color"])
        if coverage is None:
            coverage = bw.outputs["Val"]
        else:
            combine = _math(nodes, "MAXIMUM", 540, y)
            links.new(coverage, combine.inputs[0])
            links.new(bw.outputs["Val"], combine.inputs[1])
            coverage = combine.outputs[0]

    if coverage is None:
        # Nothing painted yet: the whole visible surface is up for grabs.
        coverage = _math(nodes, "MULTIPLY", 540, 1200, {0: 0.0}).outputs[0]

    visibility = _occlusion_socket(nodes, links, camera, geom, -1400, 0)

    # inpaint mask: 1 where nothing is painted yet, gated by visibility
    uncovered = _math(nodes, "SUBTRACT", 1000, 1200, {0: 1.0})
    links.new(coverage, uncovered.inputs[1])
    final = _math(nodes, "MULTIPLY", 1180, 1000)
    links.new(uncovered.outputs[0], final.inputs[0])
    links.new(visibility, final.inputs[1])

    combine = _new_node(nodes, "ShaderNodeCombineXYZ", 1380, 1000)
    for index in range(3):
        links.new(final.outputs[0], combine.inputs[index])
    links.new(combine.outputs["Vector"], emis.inputs["Color"])

    with _RenderState(context) as state:
        state.apply(width, height, camera)
        _configure_cycles(context)
        with _TempWorld(context, BLACK_WORLD):
            with _TempMaterial(obj, mat):
                return _render_to_file(context, filepath)


def bake_projected_image(context, obj, screen_image, src_attr,
                         target_image, uv_name):
    # Left in place but deliberately unusable: this path wrote an all-black
    # image on Blender 5.1 in every configuration tried. See
    # transfer_projection_to_uv for what actually runs.
    raise RuntimeError(
        "bake_projected_image is unused and unreliable on Blender 5.1; "
        "use transfer_projection_to_uv instead."
    )
    """Resample a screen-space image into the object's own UV layout.

    Source coordinates come from the projection attribute already stored on the
    mesh, so this is a plain texture transfer: no raycasting runs inside the
    bake.

    That matters. Baked directly, ``ShaderNodeRaycast`` reported *every* texel as
    occluded Ă˘â‚¬â€ť the same node chain renders correctly (mean 0.505 on the test
    sphere) but comes out solid black under ``object.bake``. Rendering the
    visibility first and transferring it sidesteps the whole question, and is
    faster because the bake no longer casts a ray per texel.
    """
    mat, emis = _emission_material("PL_bake_project")
    nodes, links = mat.node_tree.nodes, mat.node_tree.links

    attr = _new_node(nodes, "ShaderNodeAttribute", -600, 0)
    attr.attribute_type = "GEOMETRY"
    attr.attribute_name = src_attr

    tex = _new_node(nodes, "ShaderNodeTexImage", -400, 0)
    tex.image = screen_image
    tex.extension = "CLIP"
    tex.interpolation = "Linear"
    links.new(attr.outputs["Vector"], tex.inputs["Vector"])
    links.new(tex.outputs["Color"], emis.inputs["Color"])

    target = _new_node(nodes, "ShaderNodeTexImage", 420, 0)
    target.image = target_image

    scene = context.scene
    saved = {
        "engine": scene.render.engine,
        "samples": scene.cycles.samples,
        "bounces": scene.cycles.max_bounces,
        "margin": scene.render.bake.margin,
        "uv": obj.data.uv_layers.active.name if obj.data.uv_layers.active else "",
        "mode": obj.mode,
    }
    if hasattr(scene.render, "use_persistent_data"):
        saved["persistent"] = scene.render.use_persistent_data
    try:
        if obj.mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")
        _select_only(context, obj)
        if uv_name in obj.data.uv_layers:
            obj.data.uv_layers.active = obj.data.uv_layers[uv_name]
        _configure_cycles(context)
        # Persistent data caches the scene between renders, and with it on the
        # bake silently wrote an all-black image Ă˘â‚¬â€ť the temp material's shader
        # never reached the bake. The guide renders like it; the bake cannot.
        if hasattr(scene.render, "use_persistent_data"):
            scene.render.use_persistent_data = False
        scene.render.bake.margin = 8
        with _TempMaterial(obj, mat):
            # Assert the target node *after* the material swap: assigning the
            # material clears the tree's active/selected node.
            target.image = target_image
            nodes.active = target
            for node in nodes:
                node.select = node.as_pointer() == target.as_pointer()
            bpy.ops.object.bake(type="EMIT",
                                width=target_image.size[0],
                                height=target_image.size[1])
        target_image.update()
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"[Projection Layers] bake failed: {exc}")
        return False
    finally:
        scene.render.engine = saved["engine"]
        scene.cycles.samples = saved["samples"]
        scene.cycles.max_bounces = saved["bounces"]
        scene.render.bake.margin = saved["margin"]
        if "persistent" in saved:
            scene.render.use_persistent_data = saved["persistent"]
        if saved["uv"] and saved["uv"] in obj.data.uv_layers:
            obj.data.uv_layers.active = obj.data.uv_layers[saved["uv"]]
        if saved["mode"] != "OBJECT":
            try:
                bpy.ops.object.mode_set(mode=saved["mode"])
            except RuntimeError:
                pass


#: Cached, layer-independent rasterisation state. A flatten calls the transfer
#: once per (mesh, layer), and almost all of the work - destination UVs, bounding
#: boxes, barycentric weights, destination texel indices - depends only on the
#: mesh's UVs and the canvas size. Building it once per mesh removes the
#: dominant term from the per-layer cost.
_RASTER_CACHE = {}
_RASTER_CACHE_ENTRIES = 2  # fewer entries so the raised budget cannot compound
#: Upper bound on cached fill texels. A cached texel costs 32 bytes (three
#: float64 weights + two int32 index arrays), so this caps the cache near 48 MB.
_RASTER_BUDGET_TEXELS = 8_000_000  # ~288 MB per mesh worst case at 36 bytes/texel


class _MeshRaster:
    """Everything about the transfer that does not vary with the layer."""

    __slots__ = ("keep", "min_x", "max_x", "min_y", "max_y", "denom", "dest",
                 "world_centre", "world_normal", "order", "texels", "bytes",
                 "assign_dest", "assign_tri",
                 "assign_w0", "assign_w1", "assign_w2")


def _raster_key(obj, uv_name, size, screen_shape, uv_bytes, tri_bytes):
    """Key on everything that changes the result, including the UV data itself.

    A stale cache would bake the wrong texture silently, which is worse than
    being slow - so the raw UV and triangle bytes are hashed rather than trusting
    mesh name or counts, which do not change when the artist edits UVs in place.
    """
    return (obj.data.as_pointer(), uv_name, int(size), int(screen_shape[0]),
            int(screen_shape[1]), hash(uv_bytes), hash(tri_bytes))


def _raster_for(obj, mesh, uv_layer, uv_buf, tri_loops, count, size,
                screen_shape):
    key = _raster_key(obj, uv_layer.name, size, screen_shape,
                      uv_buf.tobytes(), tri_loops.tobytes())
    raster = _RASTER_CACHE.get(key)
    if raster is None:
        raster = _build_raster(obj, mesh, uv_buf, tri_loops, count, size)
        if len(_RASTER_CACHE) >= _RASTER_CACHE_ENTRIES:
            _RASTER_CACHE.clear()
        _RASTER_CACHE[key] = raster
    return raster


def _build_raster(obj, mesh, uv_buf, tri_loops, count, size):
    """Compute the layer-independent half of the transfer for one mesh."""
    raster = _MeshRaster()
    dest = uv_buf.reshape(-1, 2)[tri_loops.reshape(-1, 3)] * size

    xs = dest[:, :, 0]
    ys = dest[:, :, 1]
    x0, x1, x2 = xs[:, 0], xs[:, 1], xs[:, 2]
    y0, y1, y2 = ys[:, 0], ys[:, 1], ys[:, 2]

    raster.dest = dest
    raster.min_x = np.maximum(0.0, np.floor(xs.min(axis=1)))
    raster.max_x = np.minimum(size - 1.0, np.ceil(xs.max(axis=1)))
    raster.min_y = np.maximum(0.0, np.floor(ys.min(axis=1)))
    raster.max_y = np.minimum(size - 1.0, np.ceil(ys.max(axis=1)))
    raster.denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)

    keep = (raster.max_x >= raster.min_x) & (raster.max_y >= raster.min_y)
    keep &= np.abs(raster.denom) >= 1e-12
    raster.keep = keep

    # World centres and normals for the per-layer facing test. loop_triangles
    # expose neither to foreach_get, so both come from the corner vertices; the
    # cross product's direction matches the winding normal, which is all the
    # facing test needs.
    tri_verts = np.empty(count * 3, dtype=np.int32)
    mesh.loop_triangles.foreach_get("vertices", tri_verts)
    tri_verts = tri_verts.reshape(-1, 3)
    coords = np.empty(len(mesh.vertices) * 3, dtype=np.float64)
    mesh.vertices.foreach_get("co", coords)
    coords = coords.reshape(-1, 3)

    v0 = coords[tri_verts[:, 0]]
    v1 = coords[tri_verts[:, 1]]
    v2 = coords[tri_verts[:, 2]]
    centres = (v0 + v1 + v2) / 3.0
    normals = np.cross(v1 - v0, v2 - v0)

    matrix = obj.matrix_world
    rotation = np.array(matrix.to_3x3(), dtype=np.float64)
    raster.world_centre = centres @ rotation.T + np.array(matrix.translation)
    raster.world_normal = normals @ rotation.T

    raster.order = np.nonzero(keep)[0]
    raster.assign_dest = None
    raster.assign_tri = None
    raster.assign_w0 = None
    raster.assign_w1 = None
    raster.assign_w2 = None
    raster.texels = 0

    if raster.order.size:
        areas = ((raster.max_x - raster.min_x + 1.0)
                 * (raster.max_y - raster.min_y + 1.0))
        total = float(areas[keep].sum())
    else:
        total = 0.0

    # Below the budget, precompute every triangle's covered texels and their
    # barycentric weights. Above it, leave fills unset and let the caller take
    # the old per-triangle path: recomputing is slow, a stale or blown-up cache
    # is worse.
    if 0.0 < total <= _RASTER_BUDGET_TEXELS:
        selected = raster.order
        widths = (raster.max_x[selected] - raster.min_x[selected] + 1.0).astype(np.int64)
        heights = (raster.max_y[selected] - raster.min_y[selected] + 1.0).astype(np.int64)
        counts = widths * heights

        # Expand every triangle's bounding box in a few whole-array operations
        # instead of a Python loop per triangle. The loop was the same
        # tiny-array numpy overhead that dominated the old per-layer rasteriser,
        # just moved into the cache build - so it had to go too.
        parts_w0, parts_w1, parts_w2, parts_tri, parts_dest = [], [], [], [], []
        block = 0
        while block < selected.size:
            # Take as many triangles as fit the transient budget in one go.
            running = np.cumsum(counts[block:])
            take = int(np.searchsorted(running, _RASTER_BUDGET_TEXELS, side="right"))
            take = max(1, min(take, selected.size - block))
            chunk = slice(block, block + take)
            block += take

            chunk_counts = counts[chunk]
            grand = int(chunk_counts.sum())
            if grand <= 0:
                continue

            ids = np.repeat(selected[chunk], chunk_counts)
            width_rep = np.repeat(widths[chunk], chunk_counts)
            starts = np.cumsum(chunk_counts) - chunk_counts
            local = np.arange(grand, dtype=np.int64) - np.repeat(starts, chunk_counts)
            dx = local % width_rep
            dy = local // width_rep
            lo_x_rep = np.repeat(raster.min_x[selected[chunk]], chunk_counts)
            lo_y_rep = np.repeat(raster.min_y[selected[chunk]], chunk_counts)
            texel_x = lo_x_rep + dx + 0.5
            texel_y = lo_y_rep + dy + 0.5

            p0 = dest[ids, 0]
            p1 = dest[ids, 1]
            p2 = dest[ids, 2]
            d = raster.denom[ids]

            w0 = ((p1[:, 1] - p2[:, 1]) * (texel_x - p2[:, 0])
                  + (p2[:, 0] - p1[:, 0]) * (texel_y - p2[:, 1])) / d
            w1 = ((p2[:, 1] - p0[:, 1]) * (texel_x - p2[:, 0])
                  + (p0[:, 0] - p2[:, 0]) * (texel_y - p2[:, 1])) / d
            w2 = 1.0 - w0 - w1
            inside = (w0 >= -1e-6) & (w1 >= -1e-6) & (w2 >= -1e-6)
            if not inside.any():
                continue

            parts_w0.append(w0[inside])
            parts_w1.append(w1[inside])
            parts_w2.append(w2[inside])
            parts_tri.append(ids[inside])
            # Integer texel coordinates: the bounding boxes are float, and a
            # float index would not be usable downstream.
            dest_rows = (lo_y_rep[inside] + dy[inside]).astype(np.int64)
            dest_cols = (lo_x_rep[inside] + dx[inside]).astype(np.int64)
            parts_dest.append(dest_rows * size + dest_cols)

        if parts_dest:
            weights0 = np.concatenate(parts_w0)
            weights1 = np.concatenate(parts_w1)
            weights2 = np.concatenate(parts_w2)
            texel_tri = np.concatenate(parts_tri)
            texel_dest = np.concatenate(parts_dest)

            # Sort by destination once. Within a destination the stable order is
            # ascending triangle, which is exactly the "last triangle wins" rule
            # the per-triangle loop had - so the winner can be picked
            # arithmetically instead of by writing every triangle in turn.
            order = np.argsort(texel_dest, kind="stable")
            raster.assign_dest = texel_dest[order]
            raster.assign_tri = texel_tri[order]
            raster.assign_w0 = weights0[order]
            raster.assign_w1 = weights1[order]
            raster.assign_w2 = weights2[order]
            raster.texels = int(texel_dest.size)
            del weights0, weights1, weights2, texel_tri, texel_dest

    raster.bytes = int(48 * count + raster.texels * 36)
    return raster


def transfer_projection_to_uv(obj, screen, src_attr, uv_name, size, camera=None,
                              proj_uv=None, eye=None):
    """Resample a screen-space image into the mesh's own UV layout.

    ``proj_uv`` supplies the screen coordinates directly, as a ``(loops, 2)``
    array. When it is given, ``src_attr`` is ignored and the mesh needs no
    stamped projection attribute - which is what lets atlas flattening include
    meshes that were never projected from that camera. Otherwise the coordinates
    are read from the named corner attribute, as before.

    ``camera``/``eye`` enable back-face culling, which matters: a sphere's far
    side projects onto the *same* screen pixels as its near side, so without
    culling it copies the near side's "visible" value and the mask leaks over
    the whole hidden half. That showed up as ~94% white for a single front view,
    where roughly half is the honest answer. ``eye`` is a world position, for
    callers that have a camera matrix but no camera object.

    A small software rasteriser rather than a Cycles bake. ``object.bake`` was
    tried three ways here and wrote an all-black image every time Ă˘â‚¬â€ť constant
    emission, texture through a UV map, texture through a projection attribute,
    with and without persistent data Ă˘â‚¬â€ť while the identical node chain renders
    correctly. Rather than keep chasing it, this does the transfer directly.

    Each triangle carries two coordinate sets: its object-space UV (from the
    mesh) and its screen-space UV (from the projection attribute). Mapping one
    onto the other per texel is all a projection bake really is.

    Overlapping UV islands still fight over texels, as they would in any bake;
    the last triangle written wins.
    """
    mesh = obj.data
    uv_layer = mesh.uv_layers.get(uv_name)
    if uv_layer is None:
        return None

    if proj_uv is None:
        attr = mesh.attributes.get(src_attr)
        if attr is None:
            return None

    loops = len(mesh.loops)
    triangles = mesh.loop_triangles
    count = len(triangles)

    screen_h, screen_w = screen.shape[:2]
    out = np.zeros((size, size) + screen.shape[2:], dtype=np.float32)
    if not count:
        return out

    if eye is None and camera is not None:
        eye = camera.matrix_world.translation

    uv_buf = np.empty(loops * 2, dtype=np.float64)
    uv_layer.data.foreach_get("uv", uv_buf)

    tri_loops = np.empty(count * 3, dtype=np.int32)
    triangles.foreach_get("loops", tri_loops)

    # Layer-independent state, built once per mesh and reused for every layer.
    raster = _raster_for(obj, mesh, uv_layer, uv_buf, tri_loops, count, size,
                         screen.shape)
    tri_loops = tri_loops.reshape(-1, 3)

    if proj_uv is None:
        vec_buf = np.empty(loops * 3, dtype=np.float64)
        attr.data.foreach_get("vector", vec_buf)
        proj_uv = vec_buf.reshape(-1, 3)[:, :2]

    src = proj_uv[tri_loops] * (screen_w, screen_h)      # (T, 3, 2) screen px

    facing = None
    if eye is not None:
        # Row-wise dot, exactly as before: einsum sums in a fixed order, so the
        # facing test stays bit-identical to the pre-cache implementation.
        facing = np.einsum("ij,ij->i", raster.world_normal,
                           np.asarray(eye, dtype=np.float64) - raster.world_centre)

    if raster.assign_dest is not None:
        # Fast path. Everything geometric was computed and sorted once per mesh,
        # so a layer is now a handful of whole-array operations rather than a
        # Python loop over every triangle - which is what actually dominated.
        if facing is not None:
            positions = np.nonzero(facing[raster.assign_tri] > 0.0)[0]
        else:
            positions = np.arange(raster.assign_dest.size)
        if positions.size == 0:
            return out

        destination = raster.assign_dest[positions]

        # Keep only the last writer per destination texel, in ascending triangle
        # order, reproducing the original "last triangle wins" exactly.
        tail = np.empty(positions.size, dtype=bool)
        tail[:-1] = destination[1:] != destination[:-1]
        tail[-1] = True
        winners = positions[tail]

        triangle_ids = raster.assign_tri[winners]
        corners = src[triangle_ids]                     # (m, 3, 2)
        sx = (raster.assign_w0[winners] * corners[:, 0, 0]
              + raster.assign_w1[winners] * corners[:, 1, 0]
              + raster.assign_w2[winners] * corners[:, 2, 0])
        sy = (raster.assign_w0[winners] * corners[:, 0, 1]
              + raster.assign_w1[winners] * corners[:, 1, 1]
              + raster.assign_w2[winners] * corners[:, 2, 1])

        sampled = screen[np.clip(sy.astype(np.int32), 0, screen_h - 1),
                         np.clip(sx.astype(np.int32), 0, screen_w - 1)]
        # out may carry more than one channel (RGB + alpha travel together in the
        # flatten), so assign whole rows; a single-channel target stays 1-D.
        target = out.reshape(size * size, -1) if out.ndim > 2 else out.reshape(-1)
        target[raster.assign_dest[winners]] = sampled
        return out

    # Fallback for meshes too dense to cache: same arithmetic as before, done
    # per layer. Recomputation is slower, but a cache that silently goes stale
    # or exhausts memory would be worse.
    ramp = np.arange(size, dtype=np.float64) + 0.5
    keep = raster.keep
    if facing is not None:
        keep = keep & (facing > 0.0)

    for index in np.nonzero(keep)[0]:
        p = raster.dest[index]
        s = src[index]
        d = raster.denom[index]

        lo_x = int(raster.min_x[index])
        hi_x = int(raster.max_x[index])
        lo_y = int(raster.min_y[index])
        hi_y = int(raster.max_y[index])

        dx = ramp[lo_x:hi_x + 1] - p[2, 0]      # (w,)
        dy = ramp[lo_y:hi_y + 1] - p[2, 1]      # (h,)
        px0, py0 = p[0]
        px1, py1 = p[1]

        w0 = ((py1 - p[2, 1]) * dx[None, :] + (p[2, 0] - px1) * dy[:, None]) / d
        w1 = ((p[2, 1] - py0) * dx[None, :] + (px0 - p[2, 0]) * dy[:, None]) / d
        w2 = 1.0 - w0 - w1
        inside = (w0 >= -1e-6) & (w1 >= -1e-6) & (w2 >= -1e-6)
        if not inside.any():
            continue

        sx = w0 * s[0, 0] + w1 * s[1, 0] + w2 * s[2, 0]
        sy = w0 * s[0, 1] + w1 * s[1, 1] + w2 * s[2, 1]
        sampled = screen[np.clip(sy.astype(np.int32), 0, screen_h - 1),
                         np.clip(sx.astype(np.int32), 0, screen_w - 1)]

        rows, cols = np.nonzero(inside)
        out[rows + lo_y, cols + lo_x] = sampled[rows, cols]

    return out


def load_gray_image(path, colorspace=None):
    """Load a PNG as an image datablock (kept alive, unlike load_gray).

    Pass ``colorspace="Non-Color"`` for data written with the Standard view
    transform, or Blender gamma-decodes it a second time on the way in.
    """
    image = bpy.data.images.load(path, check_existing=False)
    if colorspace:
        try:
            image.colorspace_settings.name = colorspace
        except TypeError:
            pass
    return image


def bake_occlusion(context, obj, camera, image, uv_name):
    """Legacy path: bake occlusion with a raycast inside the bake.

    Kept only so old call sites fail loudly rather than silently. Prefer
    :func:`bake_projected_image`.
    """
    raise RuntimeError(
        "bake_occlusion is unreliable inside a bake on Blender 5.1; use "
        "bake_projected_image instead."
    )


def render_image_through_camera(context, obj, camera, image, uv_name,
                                width, height, filepath):
    """Render a UV-space image from the camera, i.e. back into screen space.

    Needed because the pre-mask is painted in UV space but the backend wants a
    screen-space inpaint mask for the current view.
    """
    mat, emis = _emission_material("PL_screen_from_uv")
    nodes, links = mat.node_tree.nodes, mat.node_tree.links

    uv = _new_node(nodes, "ShaderNodeUVMap", -500, 0)
    uv.uv_map = uv_name
    tex = _new_node(nodes, "ShaderNodeTexImage", -300, 0)
    tex.image = image
    tex.extension = "CLIP"
    links.new(uv.outputs["UV"], tex.inputs["Vector"])
    links.new(tex.outputs["Color"], emis.inputs["Color"])

    with _RenderState(context) as state:
        state.apply(width, height, camera)
        _configure_emission_engine(context)
        with _TempWorld(context, BLACK_WORLD):
            with _TempMaterial(obj, mat):
                return _render_to_file(context, filepath)


def _configure_emission_engine(context, samples=8):
    """EEVEE is plenty for a flat emission pass and much faster than Cycles."""
    scene = context.scene
    scene.render.engine = "BLENDER_EEVEE"
    eevee = getattr(scene, "eevee", None)
    if eevee is not None:
        for attr, value in (
            ("taa_render_samples", samples),
            ("use_raytracing", False),
            ("use_shadows", False),
        ):
            if hasattr(eevee, attr):
                setattr(eevee, attr, value)


def _configure_cycles(context):
    """Cycles, because ShaderNodeRaycast is a Cycles feature."""
    scene = context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = 1
    scene.cycles.max_bounces = 0
    scene.cycles.use_denoising = False
    if hasattr(scene.render, "use_persistent_data"):
        scene.render.use_persistent_data = True


def _render_to_file(context, filepath):
    """Render a still and return the path Blender actually wrote."""
    scene = context.scene
    directory = os.path.dirname(filepath)
    stem = os.path.splitext(os.path.basename(filepath))[0]
    os.makedirs(directory, exist_ok=True)
    scene.render.filepath = filepath
    bpy.ops.render.render(write_still=True)

    if os.path.exists(filepath):
        return filepath
    # Blender sometimes appends the frame number.
    matches = sorted(
        glob.glob(os.path.join(directory, f"{stem}*")), key=os.path.getmtime
    )
    if not matches:
        raise RuntimeError(f"Render produced no file for '{stem}'.")
    return matches[-1]


# Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬ mask post-processing Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬


def load_gray(path):
    """Load a PNG as a float32 (H, W) array in 0..1, top row first."""
    img = bpy.data.images.load(path, check_existing=False)
    try:
        width, height = img.size
        buffer = np.empty(width * height * 4, dtype=np.float32)
        img.pixels.foreach_get(buffer)
        return buffer.reshape((height, width, 4))[:, :, 0]
    finally:
        bpy.data.images.remove(img)


def image_gray(image):
    """Read a Blender image's first channel as a float32 (H, W) array."""
    width, height = image.size
    buffer = np.empty(width * height * 4, dtype=np.float32)
    image.pixels.foreach_get(buffer)
    return buffer.reshape((height, width, 4))[:, :, 0]


def image_rgb(image):
    """Read a Blender image as a float32 (H, W, 3) array."""
    width, height = image.size
    buffer = np.empty(width * height * 4, dtype=np.float32)
    image.pixels.foreach_get(buffer)
    return buffer.reshape((height, width, 4))[:, :, :3]


def image_alpha(image):
    """Read a Blender image's alpha as a float32 (H, W) array."""
    width, height = image.size
    buffer = np.empty(width * height * 4, dtype=np.float32)
    image.pixels.foreach_get(buffer)
    return buffer.reshape((height, width, 4))[:, :, 3]


def resize_nearest(array, size):
    """Nearest-neighbour resize of a 2D or 3D float array to (size, size)."""
    height, width = array.shape[:2]
    ys = (np.arange(size) * height // size).clip(0, height - 1)
    xs = (np.arange(size) * width // size).clip(0, width - 1)
    return array[ys][:, xs]


def write_rgb(image, array):
    """Write a float32 (H, W, 3) array into a Blender image."""
    height, width = array.shape[:2]
    buffer = np.empty((height, width, 4), dtype=np.float32)
    buffer[:, :, :3] = array
    buffer[:, :, 3] = 1.0
    image.pixels.foreach_set(buffer.ravel())
    image.update()
    image.pack()


def save_image_as_png(image, path):
    """Write an existing Blender image out to ``path`` as a PNG."""
    copy = image.copy()
    try:
        copy.filepath_raw = path
        copy.file_format = "PNG"
        copy.save()
    finally:
        bpy.data.images.remove(copy)
    return path


def save_gray(array, path):
    """Write a float32 (H, W) array out as an 8-bit PNG via a packed image."""
    height, width = array.shape
    rgba = np.empty((height, width, 4), dtype=np.float32)
    rgba[:, :, 0] = array
    rgba[:, :, 1] = array
    rgba[:, :, 2] = array
    rgba[:, :, 3] = 1.0
    img = bpy.data.images.new("PL_tmp_mask", width, height, alpha=False)
    try:
        img.pixels.foreach_set(rgba.ravel())
        img.filepath_raw = path
        img.file_format = "PNG"
        img.save()
    finally:
        bpy.data.images.remove(img)


def feather_mask(array, pixels):
    """Fade the frame border inwards and soften with a small blur.

    Done in numpy on the rendered mask rather than in the shader: it is a
    screen-space image, so this is both simpler and much cheaper.
    """
    if pixels <= 0:
        return array
    height, width = array.shape
    ys = np.arange(height, dtype=np.float32)[:, None]
    xs = np.arange(width, dtype=np.float32)[None, :]
    ramp = np.minimum(
        np.minimum(xs, width - 1 - xs), np.minimum(ys, height - 1 - ys)
    )
    ramp = np.clip(ramp / float(pixels), 0.0, 1.0)
    faded = array * (ramp * ramp * (3.0 - 2.0 * ramp))  # smoothstep

    radius = max(1, int(pixels) // 2)
    return _box_blur(faded, radius)


def _box_blur(array, radius):
    """Separable box blur via cumulative sums Ă˘â‚¬â€ť fast and dependency-free."""
    if radius < 1:
        return array

    def blur_axis(data, axis):
        size = data.shape[axis]
        pad = [(0, 0)] * data.ndim
        pad[axis] = (radius + 1, radius)
        padded = np.pad(data, pad, mode="edge")
        cumsum = np.cumsum(padded, axis=axis)
        length = size + 2 * radius + 1
        index = np.arange(size)
        hi = np.take(cumsum, index + 2 * radius + 1, axis=axis)
        lo = np.take(cumsum, index, axis=axis)
        del length
        return (hi - lo) / float(2 * radius + 1)

    return blur_axis(blur_axis(array, 0), 1)


def temp_dir():
    directory = os.path.join(tempfile.gettempdir(), "projlayer")
    os.makedirs(directory, exist_ok=True)
    return directory


def temp_path(name):
    return os.path.join(temp_dir(), name)


def read_bytes(path):
    with open(path, "rb") as handle:
        return handle.read()


def _select_only(context, obj):
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    context.view_layer.objects.active = obj
