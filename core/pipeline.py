"""The generate pipeline.

Split across the thread boundary on purpose:

  prepare()  main thread Ä‚ËĂ˘â€šÂ¬Ă˘â‚¬ĹĄ render the maps, upload them, patch the workflow
  run()      worker thread Ä‚ËĂ˘â€šÂ¬Ă˘â‚¬ĹĄ queue the prompt and wait for the result
  apply()    main thread Ä‚ËĂ˘â€šÂ¬Ă˘â‚¬ĹĄ snapshot the projection UV and create the layer

Only ``run`` touches the network; everything that touches Blender data stays on
the main thread.
"""

import math
import os
import random
import time
import uuid
from contextlib import ExitStack, nullcontext

import bpy
import numpy as np
from mathutils import Vector

from .. import prefs as prefs_module
from ..props import (
    current_mode,
    ensure_prompt_slots,
    find_role_layer,
    is_paint_layer,
    mode_layers,
    resolution,
    restore_after_capture,
    role_layers,
    set_current_mode,
    set_pre_modes,
    settings,
    stack,
    unhide_for_capture,
    visible_layers,
)
from . import comfy, graphs, nodes, render


class Payload:
    """Everything `apply` needs, gathered while preparing."""

    def __init__(self):
        self.obj_name = ""
        self.material_name = ""
        self.camera_name = ""
        self.mode = "COLOR"
        self.width = 0
        self.height = 0
        self.prompt = ""
        self.seed = 0
        self.graph = {}
        self.slots = {}
        self.output_node = ""
        self.premask_image = ""
        self.bypassed = []
        self.base_url = ""
        self.mask_path = ""
        self.feather = 6
        self.falloff = 1.0
        self.angle_cut = 75.0
        self.max_distance = 0.0
        self.started = 0.0
        #: False for queued views: their camera is already positioned, so
        #: snapping it to the viewport would throw that framing away.
        self.snap_camera = True
        #: True during a batch: hide the other objects while rendering, or
        #: co-located meshes would appear in - and occlude - the reference.
        self.isolate = False


def _set_status(context, text):
    scene = getattr(context, "scene", None)
    if scene is not None and hasattr(scene, "projlayer"):
        scene.projlayer.status = text


def flatten_color_stack(context, obj, size, uv_name, progress=None):
    """Composite every visible colour layer into one UV-space image.

    Done in numpy rather than with a Cycles bake, for the same reason the mask
    transfer is: ``object.bake`` produced black every way it was tried here.
    Each layer's picture is mapped from its camera projection into the object's
    UVs, then composited with its own mask and opacity.
    """
    color_layers = [l for l in visible_layers(context, "COLOR") if l.source_image]
    if not color_layers:
        return None

    out = np.zeros((size, size, 3), dtype=np.float32)

    for index, layer in enumerate(color_layers):
        if progress is not None:
            progress(index + 1, len(color_layers), "colour")
        source = render.image_rgb(layer.source_image)
        alpha_src = render.image_alpha(layer.source_image)

        if is_paint_layer(layer):
            color = render.resize_nearest(source, size)
            alpha = render.resize_nearest(alpha_src, size)
        else:
            color = render.transfer_projection_to_uv(
                obj, source, layer.uv_attr, uv_name, size)
            alpha = render.transfer_projection_to_uv(
                obj, alpha_src, layer.uv_attr, uv_name, size)
            if color is None:
                continue

        if layer.mask_image is not None:
            mask = render.resize_nearest(render.image_gray(layer.mask_image), size)
        else:
            mask = np.ones((size, size), dtype=np.float32)

        fac = (mask * float(layer.opacity) * alpha)[:, :, None]
        color = _apply_hsv(color, layer)
        if layer.blend == "MULTIPLY":
            blended = out * color
        elif layer.blend == "SCREEN":
            blended = 1.0 - (1.0 - out) * (1.0 - color)
        elif layer.blend == "ADD":
            blended = out + color
        else:
            blended = color
        out = np.clip(out * (1.0 - fac) + blended * fac, 0.0, 1.0)

    return out


def _hue_matrix(shift):
    """SVG-style hue rotation, with 0.5 meaning no change."""
    angle = (float(shift) - 0.5) * 2.0 * math.pi
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    return np.array([
        [0.213 + cos_a * 0.787 - sin_a * 0.213,
         0.715 - cos_a * 0.715 - sin_a * 0.715,
         0.072 - cos_a * 0.072 + sin_a * 0.928],
        [0.213 - cos_a * 0.213 + sin_a * 0.143,
         0.715 + cos_a * 0.285 + sin_a * 0.140,
         0.072 - cos_a * 0.072 - sin_a * 0.283],
        [0.213 - cos_a * 0.213 - sin_a * 0.787,
         0.715 - cos_a * 0.715 + sin_a * 0.715,
         0.072 + cos_a * 0.928 + sin_a * 0.072],
    ])


def _apply_hsv(color, layer):
    """Same adjustments the shader chain applies, in numpy."""
    if abs(layer.hue - 0.5) > 1e-4:
        color = np.clip(color @ _hue_matrix(layer.hue).T, 0.0, 1.0)
    if abs(layer.saturation - 1.0) > 1e-4:
        grey = (color * np.array([0.2126, 0.7152, 0.0722])).sum(axis=2, keepdims=True)
        color = np.clip(grey + (color - grey) * float(layer.saturation), 0.0, 1.0)
    if abs(layer.value - 1.0) > 1e-4:
        color = np.clip(color * float(layer.value), 0.0, 1.0)
    return color


def flatten_normal_stack(context, obj, size, uv_name, progress=None):
    """Composite every visible normal layer into one UV-space image.

    Straight whiteout blending: the X and Y slopes add, Z multiplies, and the
    result is renormalised.
    """
    layers = [l for l in visible_layers(context, "NORMAL") if l.source_image]
    if not layers:
        return None

    result = None
    for index, layer in enumerate(layers):
        if progress is not None:
            progress(index + 1, len(layers), "normals")
        source = render.image_rgb(layer.source_image)
        alpha_src = render.image_alpha(layer.source_image)
        if is_paint_layer(layer):
            color = render.resize_nearest(source, size)
            alpha = render.resize_nearest(alpha_src, size)
        else:
            color = render.transfer_projection_to_uv(
                obj, source, layer.uv_attr, uv_name, size)
            alpha = render.transfer_projection_to_uv(
                obj, alpha_src, layer.uv_attr, uv_name, size)
            if color is None:
                continue

        if layer.mask_image is not None:
            mask = render.resize_nearest(render.image_gray(layer.mask_image), size)
        else:
            mask = np.ones((size, size), dtype=np.float32)

        fac = (mask * float(layer.opacity) * alpha)[:, :, None]
        vec = color * 2.0 - 1.0

        if result is None:
            result = vec
        else:
            blended = result.copy()
            blended[:, :, 0] = result[:, :, 0] + vec[:, :, 0]
            blended[:, :, 1] = result[:, :, 1] + vec[:, :, 1]
            blended[:, :, 2] = result[:, :, 2] * vec[:, :, 2]
            result = result * (1.0 - fac) + blended * fac

        length = np.linalg.norm(result, axis=2, keepdims=True)
        result = result / np.maximum(1e-6, length)

    return np.clip(result * 0.5 + 0.5, 0.0, 1.0)


def _allocate_pass_index(obj):
    """An object pass index nothing else in the scene is using."""
    used = {other.pass_index for other in bpy.data.objects if other is not obj}
    index = 1
    while index in used:
        index += 1
    obj.pass_index = index
    return index


def _stamp_projection_on_sharers(context, obj, camera, attr_name):
    """Give every other mesh sharing this material the same projection.

    A layer's projection lives in a per-MESH attribute, so an object without it
    samples a missing attribute and smears one texel over its whole surface.
    Stamping the same attribute, from the same camera, onto the other meshes is
    what makes a projection genuinely bleed across a shared material.
    """
    material = obj.active_material
    if material is None:
        return 0
    stamped = 0
    for other in bpy.data.objects:
        if other is obj or other.type != "MESH" or other.data is None:
            continue
        if not any(slot.material is material for slot in other.material_slots):
            continue
        try:
            render.snapshot_projection(context, other, camera, attr_name=attr_name)
            stamped += 1
        except RuntimeError:
            pass  # multi-user mesh or no free slot; the target still works
    return stamped


def material_shared_with(obj) -> bool:
    """Is another mesh using this object's active material?

    When it is, the two share a layer stack, and a projection made for one will
    appear on the other - because the stack is stored on the material. Nothing
    about that is recoverable per object: the projection attribute the layers
    sample only exists on the mesh that was projected.
    """
    material = obj.active_material
    if material is None:
        return False
    for other in bpy.data.objects:
        if other is obj or other.type != "MESH":
            continue
        for slot in other.material_slots:
            if slot.material is material:
                return True
    return False


def mesh_uv_name(obj) -> str:
    """The UV map a mask or sketch should live in: the active one, else the first."""
    uv = obj.data.uv_layers.active
    if uv is None and len(obj.data.uv_layers):
        uv = obj.data.uv_layers[0]
    return uv.name if uv is not None else ""


def _run_dir():
    path = os.path.join(render.temp_dir(), uuid.uuid4().hex[:8])
    os.makedirs(path, exist_ok=True)
    return path


def prepare(context, payload):
    """MAIN THREAD. Render the guides, upload them, and patch the workflow."""
    scene = context.scene
    conf = settings(context)
    prefs = prefs_module.prefs(context)

    obj = context.active_object
    if obj is None or obj.type != "MESH":
        raise RuntimeError("Select a mesh object first.")
    if not conf.prompt.strip():
        raise RuntimeError("Type a prompt before generating.")

    # An operator is a safe place to write to scene data (unlike a draw).
    ensure_prompt_slots(conf)

    # A queued view already has its own framed camera. Only the plain Generate
    # button falls back to the add-on's projector camera and snaps it to the
    # viewport Ă˘â‚¬â€ť this used to run unconditionally, which overwrote every view's
    # camera with the projector and made a batch render the same angle N times.
    if payload.snap_camera:
        camera = render.ensure_camera(context)
    else:
        camera = scene.camera
        if camera is None or getattr(camera, "type", "") != "CAMERA":
            camera = render.ensure_camera(context)

    # Frame the capture from what the artist is actually looking at. Doing this
    # on every Generate means the Snap button is only for checking.
    if conf.auto_snap_camera and payload.snap_camera:
        try:
            render.snap_to_view(context, camera)
        except RuntimeError:
            pass  # no 3D view to copy from (background mode, or no viewport open)
    scene.camera = camera

    # Layer stacks live on the MATERIAL, so two meshes sharing one material

    payload.obj_name = obj.name
    payload.material_name = obj.active_material.name if obj.active_material else ""
    payload.camera_name = camera.name
    payload.mode = current_mode(context)
    payload.width, payload.height = resolution(context)
    payload.prompt = conf.prompt.strip()
    payload.seed = (
        random.randint(0, 2**31 - 1) if conf.randomise_seed else int(conf.seed)
    )
    payload.base_url = prefs.server_address
    payload.started = time.time()
    payload.feather = 6

    # Only this workflow's own layers count as "already painted" Ä‚ËĂ˘â€šÂ¬Ă˘â‚¬ĹĄ normal layers
    # must not stop colour inpainting, or vice versa.
    sources = visible_layers(context, current_mode(context))
    if sources:
        projection = sources[-1].projection
        payload.falloff = projection.falloff
        payload.angle_cut = projection.angle_cut
        payload.feather = projection.feather_px
        payload.max_distance = projection.max_distance

    run_dir = _run_dir()
    base_url = prefs.server_address
    uploads = {}

    def _isolate():
        """Everything that has to be hidden for one render.

        Two independent reasons to exclude an object, and the mask render needs
        both: batch isolation (hide the rest of the scene) and "Visible Only"
        (the eye icon and the H key are viewport flags a render otherwise
        ignores). Composed so both restore in the right order.
        """
        stack = ExitStack()
        if payload.isolate:
            stack.enter_context(render.only_visible(context, obj))
        # Unconditional: whatever the artist hid in the viewport is not part of
        # the render. This used to be gated behind a setting, and a scene saved
        # with that setting off kept rendering hidden objects no matter what the
        # default said - changing an RNA default does not update stored values.
        stack.enter_context(render.respect_viewport_visibility(context))
        return stack

    # Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬ beauty reference Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬
    _set_status(context, "Rendering referenceÄ‚ËĂ˘â€šÂ¬Ă‚Â¦")
    reference_path = os.path.join(run_dir, "reference.png")
    # The pre-mask draws itself as flat red. That must not end up in the
    # reference the backend sees, so hide it for this one render and put it
    # straight back afterwards.
    unhide_for_capture(context)
    try:
        isolate = _isolate()
        with isolate:
            if conf.send_visible_only:
                # The eye icon and the H key set viewport-only flags that a
                # render ignores, so honouring them means hiding from the render
                # for the duration of this one pass.
                with render.respect_viewport_visibility(context):
                    render.render_beauty(context, obj, camera, payload.width,
                                         payload.height, reference_path)
            else:
                render.render_beauty(context, obj, camera, payload.width,
                                     payload.height, reference_path)
    finally:
        restore_after_capture(context)
    uploads["reference"] = _upload(base_url, reference_path, "reference.png")

    # The mask is no longer rendered here: it is baked straight into the mesh's
    # own UV space once the result arrives (see apply). That costs one render
    # less per generation and gives the artist a mask they can paint normally.

    # Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬ optional guides Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬
    if conf.send_depth:
        _set_status(context, "Rendering depthÄ‚ËĂ˘â€šÂ¬Ă‚Â¦")
        depth_path = os.path.join(run_dir, "depth.png")
        with _isolate():
            render.render_depth(context, obj, camera, payload.width, payload.height, depth_path)
        uploads["depth"] = _upload(base_url, depth_path, "depth.png")

    if conf.send_normal:
        _set_status(context, "Rendering normalÄ‚ËĂ˘â€šÂ¬Ă‚Â¦")
        normal_path = os.path.join(run_dir, "normal.png")
        with _isolate():
            render.render_normal(context, obj, camera, payload.width, payload.height, normal_path)
        uploads["normal"] = _upload(base_url, normal_path, "normal.png")

    # Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬ workflow (resolved before the mask so we know if it wants one) Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬
    _set_status(context, "Preparing workflowÄ‚ËĂ˘â€šÂ¬Ă‚Â¦")
    graph, source_label = prefs_module.resolve_workflow(context)
    titles = prefs_module.slot_titles(context)
    slots = graphs.detect_slots(graph)

    if titles:
        slots.update(_slots_from_titles(graph, titles))

    missing = [key for key in ("reference", "positive", "output") if key not in slots]
    if missing:
        raise RuntimeError(
            "Could not find these nodes in the workflow: "
            + ", ".join(missing)
            + f" ({source_label}). Check the slot titles in the add-on preferences."
        )

    # Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬ inpaint mask: a painted pre-mask wins over the auto coverage mask Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬
    premask = find_role_layer(context, "PREMASK") if conf.premask_on else None
    if premask is not None and premask.mask_image is not None:
        payload.premask_image = premask.mask_image.name
        # The pre-mask is painted in the object's UV space, so render it back
        # through the camera to get the screen-space mask the backend wants.
        if "mask" in slots:
            _set_status(context, "Sending the pre-maskÄ‚ËĂ˘â€šÂ¬Ă‚Â¦")
            premask_path = os.path.join(run_dir, "premask.png")
            render.render_image_through_camera(
                context, obj, camera, premask.mask_image,
                premask.mask_uv or mesh_uv_name(obj),
                payload.width, payload.height, premask_path,
            )
            uploads["mask"] = _upload(base_url, premask_path, "premask.png")
    elif conf.only_fill_unpainted and sources and "mask" in slots:
        _set_status(context, "Building inpaint maskÄ‚ËĂ˘â€šÂ¬Ă‚Â¦")
        cover_path = os.path.join(run_dir, "coverage.png")
        render.render_coverage(
            context, obj, camera, payload.width, payload.height, cover_path, sources
        )
        uploads["mask"] = _upload(base_url, cover_path, "coverage.png")

    values = {
        "reference": uploads.get("reference"),
        "positive": payload.prompt,
        "negative": conf.negative_prompt,
        "seed": payload.seed,
    }
    # Only stamp the render size into the workflow when the user wants it: some
    # graphs own their output size via a preset and reject a mismatched value.
    if prefs.patch_resolution:
        values["width"] = payload.width
        values["height"] = payload.height
    for key in ("depth", "normal", "mask"):
        if key in uploads:
            values[key] = uploads[key]

    payload.graph = graphs.apply_slots(graph, slots, values)

    # Cut the guide branches we are not feeding. Skipping a slot is not enough:
    # the workflow's own LoadImage would still run against whatever default
    # filename is saved in it, and a stale one makes ComfyUI reject the whole
    # prompt ("invalid ... .png").
    payload.bypassed = graphs.disable_unused_guides(payload.graph, slots, uploads)

    payload.slots = slots
    payload.output_node = str(slots.get("output") or "")
    return payload


def _slots_from_titles(graph, titles):
    """Resolve user-specified node titles to node ids."""
    out = {}
    for key, title in titles.items():
        needle = title.strip().lower()
        if not needle:
            continue
        for node_id, node in graph.items():
            if not isinstance(node, dict):
                continue
            label = node.get("_meta", {}).get("title", "")
            if label.strip().lower() == needle:
                out[key] = node_id
                break
    return out


def _upload(base_url, path, filename):
    """Upload one PNG and return the *filename* to put in a LoadImage node.

    ComfyUI's /upload/image replies with a JSON object; a LoadImage node wants
    the bare name (optionally prefixed with its subfolder), not the object.
    """
    info = comfy.upload_image(base_url, render.read_bytes(path), filename)
    if isinstance(info, dict):
        name = str(info.get("name") or "")
        subfolder = str(info.get("subfolder") or "")
        if not name:
            raise comfy.ComfyError(
                f"ComfyUI accepted '{filename}' but returned no filename for it."
            )
        return f"{subfolder}/{name}" if subfolder else name
    return str(info or "")


def run(payload, job):
    """WORKER THREAD. Queue the workflow and wait for the images."""
    job.set_status("GeneratingÄ‚ËĂ˘â€šÂ¬Ă‚Â¦")
    client_id = uuid.uuid4().hex
    return comfy.run_prompt(
        payload.base_url,
        payload.graph,
        client_id,
        on_progress=lambda done, total: job.set_progress(done, total),
        is_cancelled=job.is_cancelled,
        timeout=900.0,
        # Graphs often carry several image outputs (a PreviewImage beside the
        # real SaveImage); ask for the one the output slot points at.
        prefer_node=payload.output_node or None,
    )


def _activate_target(context, obj, material_name):
    """Make the payload's object, and its named material, the active ones.

    A batch prepares one object per job and applies the result later, on the
    main thread, by which time the artist may have selected something else.
    ``props.stack()`` resolves from ``context.active_object``, so without this
    the layer would land in whichever material happened to be active.
    """
    view_layer = getattr(context, "view_layer", None)
    objects = getattr(view_layer, "objects", None) if view_layer is not None else None
    if objects is not None:
        if obj.name not in objects:
            raise RuntimeError(
                f"'{obj.name}' is not in the active view layer (its collection "
                "may be excluded), so it cannot be textured."
            )
        try:
            view_layer.objects.active = obj
            obj.select_set(True)
        except RuntimeError:
            pass  # hidden objects refuse selection; the active flag is enough

    if not material_name:
        return

    material = obj.active_material
    if material is not None and material.name == material_name:
        return

    wanted = bpy.data.materials.get(material_name)
    if wanted is None:
        raise RuntimeError(f"Material '{material_name}' no longer exists.")
    for index, slot in enumerate(obj.material_slots):
        if slot.material == wanted:
            obj.active_material_index = index
            return
    raise RuntimeError(
        f"'{material_name}' is no longer a material on '{obj.name}'."
    )


def apply(context, payload, images):
    """MAIN THREAD. Snapshot the projection and add the layer."""
    conf = settings(context)
    obj = bpy.data.objects.get(payload.obj_name)
    if obj is None:
        raise RuntimeError(f"'{payload.obj_name}' no longer exists.")

    if not images:
        raise RuntimeError("ComfyUI finished but produced no image.")

    # Resolve the target before touching the stack: see _activate_target.
    _activate_target(context, obj, payload.material_name)

    elapsed = time.time() - payload.started
    is_normal = payload.mode == "NORMAL"
    target_stack = stack(context, create=True)
    layers = target_stack.normal_layers if is_normal else target_stack.color_layers
    index = len(layers)
    stamp = time.strftime("%H%M%S")

    camera = bpy.data.objects.get(payload.camera_name) or render.ensure_camera(context)

    # A normal map must be read as data, or Blender gamma-decodes it and the
    # lighting goes subtly wrong.
    source = _image_from_bytes(
        f"PL_gen_{stamp}_{index:02d}", images[0],
        "Non-Color" if is_normal else "sRGB",
    )

    # Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬ freeze this camera into a projection attribute Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬
    # Done first: the bake below uses these coordinates as its source UVs.
    attr_name = render.snapshot_projection(context, obj, camera)
    # Sharing a material means sharing the layer, so the other meshes need the
    # same projection attribute too - otherwise they sample a missing attribute.
    if not conf.only_selected:
        # A confined layer is gated off on the other meshes anyway, so there
        # is no point stamping the projection attribute onto them.
        _stamp_projection_on_sharers(context, obj, camera, attr_name)

    # Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬ the layer's mask, transferred into the object's own UV space Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬
    uv_name = mesh_uv_name(obj)
    if not uv_name:
        raise RuntimeError(
            f"'{obj.name}' has no UV map. Unwrap it first Ä‚ËĂ˘â€šÂ¬Ă˘â‚¬ĹĄ masks live in UV space "
            "so they can be painted normally."
        )

    # Pure white by default. The layers are not clipped to "what this camera
    # could see" -- the material's live angle cutoff and max-distance gates
    # already stop the projection on surfaces facing away or too far off, and
    # they do it per fragment with a soft edge. Baking a visibility mask on top
    # of that only made the mask unpredictable between views.
    mask = bpy.data.images.new(f"PL_mask_{stamp}_{index:02d}",
                               payload.width, payload.height, alpha=False)
    try:
        mask.colorspace_settings.name = "Non-Color"
    except TypeError:
        pass

    _write_gray_into(mask, np.ones((payload.height, payload.width),
                                   dtype=np.float32))
    if payload.feather > 0:
        _write_gray_into(mask, render._box_blur(render.image_gray(mask),
                                                int(payload.feather)))
    if payload.premask_image:
        pre = bpy.data.images.get(payload.premask_image)
        if pre is not None and tuple(pre.size) == tuple(mask.size):
            _write_gray_into(mask, render.image_gray(pre) * render.image_gray(mask))
    mask.pack()

    # Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬ create the layer Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬
    layer = layers.add()
    layer.name = _layer_name(payload.prompt, camera, index)
    layer.source_image = source
    layer.mask_image = mask
    layer.mask_uv = uv_name
    layer.uv_attr = attr_name
    layer.opacity = 1.0
    layer.prompt = payload.prompt
    layer.seed = payload.seed
    layer.bake_seconds = elapsed
    _capture_projection(layer.projection, camera, payload)

    # The layer exists now, so it can record the object it belongs to.
    if conf.only_selected:
        layer.owner_pass = _allocate_pass_index(obj)


    # Show the stack the layer landed in.
    set_current_mode(context, payload.mode)
    active_stack = stack(context, create=True)
    if is_normal:
        active_stack.active_normal = len(active_stack.normal_layers) - 1
    else:
        active_stack.active_color = len(active_stack.color_layers) - 1

    # Pre-paint has done its job once the backend has seen the reference.
    for layer in role_layers(context, "PREPAINT"):
        _drop_layer(context, layer)

    rebuild(context)

    # The pre-mask has done its job Ä‚ËĂ˘â€šÂ¬Ă˘â‚¬ĹĄ the new layer carries a copy of it Ä‚ËĂ˘â€šÂ¬Ă˘â‚¬ĹĄ so
    # exit the mode. Leaving it armed would keep the red overlay sitting on top
    # of the result and make the next paint stroke go to the wrong place.
    if payload.premask_image:
        set_pre_modes(context, premask=False)
    return layer


def _write_gray_into(image, array):
    height, width = array.shape
    buffer = np.empty((height, width, 4), dtype=np.float32)
    for channel in range(3):
        buffer[:, :, channel] = array
    buffer[:, :, 3] = 1.0
    image.pixels.foreach_set(buffer.ravel())
    image.update()
    image.pack()


def _drop_layer(context, layer):
    """Remove one layer and the images it owns."""
    active = stack(context)
    ptr = layer.as_pointer()
    for coll in (active.color_layers, active.normal_layers):
        for index, candidate in enumerate(coll):
            if candidate.as_pointer() != ptr:
                continue
            images = [candidate.source_image, candidate.mask_image]
            coll.remove(index)
            for image in images:
                if image is not None and image.users == 0:
                    bpy.data.images.remove(image)
            return True
    return False


def _capture_projection(projection, camera, payload):
    # mathutils presents a matrix row-wise with the translation in the fourth
    # *column*, but a subtype="MATRIX" FloatVectorProperty keeps its array
    # column-major. Flatten by columns so the value read back is the real
    # matrix Ä‚ËĂ˘â€šÂ¬Ă˘â‚¬ĹĄ flatten by rows and it comes back transposed, which silently
    # drops the translation on restore.
    projection.cam_matrix = [v for col in camera.matrix_world.col for v in col]
    projection.lens = camera.data.lens
    projection.sensor_width = camera.data.sensor_width
    projection.sensor_fit = camera.data.sensor_fit
    projection.shift_x = camera.data.shift_x
    projection.shift_y = camera.data.shift_y
    projection.clip_start = camera.data.clip_start
    projection.clip_end = camera.data.clip_end
    projection.res_x = payload.width
    projection.res_y = payload.height
    projection.falloff = payload.falloff
    projection.angle_cut = payload.angle_cut
    projection.max_distance = payload.max_distance


def _layer_name(prompt, camera, index):
    """Name a layer from the prompt and the camera's bearing, not "Layer 4"."""
    words = [w for w in prompt.strip().split() if w.isalpha()][:3]
    label = " ".join(words).capitalize() if words else f"Layer {index + 1}"

    direction = camera.matrix_world.to_quaternion() @ Vector((0.0, 0.0, -1.0))
    return f"{label} {_bearing_label(direction)}".strip()


def _bearing_label(direction):
    angle = math.degrees(math.atan2(direction.x, -direction.y)) % 360.0
    names = [
        "front", "front-left", "left", "back-left",
        "back", "back-right", "right", "front-right",
    ]
    return names[int((angle + 22.5) % 360 // 45)]


def _image_from_bytes(name, data, colorspace):
    path = render.temp_path(f"{name}.png")
    with open(path, "wb") as handle:
        handle.write(data)
    return _image_from_path(name, path, colorspace)


def _image_from_path(name, path, colorspace):
    image = bpy.data.images.load(path, check_existing=False)
    image.name = name
    try:
        image.colorspace_settings.name = colorspace
    except TypeError:
        pass
    image.pack()
    return image


def rebuild(context):
    """Rebuild the material stack for the active object."""
    obj = context.active_object
    if obj is None or obj.type != "MESH":
        return
    # Also materialises a material when the mesh has none, so a stack always
    # has somewhere to live.
    stack(context, create=True)
    material = obj.active_material
    if material is None:
        return

    color_layers = [l for l in visible_layers(context, "COLOR") if l.source_image]
    normal_layers = [l for l in visible_layers(context, "NORMAL") if l.source_image]
    blend = normal_layers[0].normal_blend if normal_layers else "WHITEOUT"

    nodes.rebuild(material, color_layers, normal_layers, blend)
    return material
