"""The generate pipeline.

Split across the thread boundary on purpose:

  prepare()  main thread Ă„â€šĂ‹ÂÄ‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ä‚ËĂ˘â€šÂ¬ÄąÄ„ render the maps, upload them, patch the workflow
  run()      worker thread Ă„â€šĂ‹ÂÄ‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ä‚ËĂ˘â€šÂ¬ÄąÄ„ queue the prompt and wait for the result
  apply()    main thread Ă„â€šĂ‹ÂÄ‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ä‚ËĂ˘â€šÂ¬ÄąÄ„ snapshot the projection UV and create the layer

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


def meshes_sharing_material(material):
    """Every mesh whose material slots reference this material.

    Layer stacks live on the material, so this is also the set of meshes that
    share one stack - and, when they are UV-atlased, one texture.
    """
    if material is None:
        return []
    meshes = []
    for other in bpy.data.objects:
        if other.type != "MESH" or other.data is None:
            continue
        for slot in other.material_slots:
            if slot.material is material:
                meshes.append(other)
                break
    return meshes


def _sample_layer_in_atlas(meshes, layer, size, uv_name):
    """One layer's colour and alpha, mapped into the shared atlas.

    Every mesh sharing the material owns different UV islands, so the atlas is
    assembled by taking each mesh's pixels wherever it covers. Coverage comes
    from alpha rather than a separate pass: a texel no mesh rasterised stays at
    alpha 0 and contributes nothing, which is also the right answer for
    genuinely transparent pixels.

    RGB and alpha travel in a single four-channel transfer, which the rasteriser
    already supports, so this costs one pass per mesh per layer.

    The projection is re-derived from the layer's stored camera rather than read
    from a stamped per-mesh attribute. That attribute only exists on meshes that
    were projected and were reachable when the projection was stamped; relying on
    it left most of an atlas blank, or mapped a stale projection over it. With
    the matrix, any mesh sharing the material can be projected correctly - and
    back-face culling works again, since the camera position is known.
    """
    if is_paint_layer(layer):
        # Already painted in UV space, so it is identical for every mesh.
        color = render.resize_nearest(render.image_rgb(layer.source_image), size)
        alpha = render.resize_nearest(render.image_alpha(layer.source_image), size)
        return color, alpha

    packed = np.dstack([
        render.image_rgb(layer.source_image),
        render.image_alpha(layer.source_image),
    ])

    has_camera = render.projection_is_valid(layer.projection)
    eye = None
    if has_camera:
        eye = render.matrix_from_projection(layer.projection).translation

    color = np.zeros((size, size, 3), dtype=np.float32)
    alpha = np.zeros((size, size), dtype=np.float32)
    for mesh in meshes:
        if mesh.data is None:
            continue
        if has_camera:
            projected = render.project_loops_uv(mesh, layer.projection)
            if projected is None:
                continue
            mapped = render.transfer_projection_to_uv(
                mesh, packed, None, uv_name, size, proj_uv=projected, eye=eye)
        else:
            # No camera was ever recorded for this layer: fall back to whatever
            # the mesh has stamped, if anything.
            if mesh.data.attributes.get(layer.uv_attr) is None:
                continue
            mapped = render.transfer_projection_to_uv(
                mesh, packed, layer.uv_attr, uv_name, size)
        if mapped is None:
            continue
        better = mapped[:, :, 3] >= alpha
        color[better] = mapped[:, :, :3][better]
        alpha = np.maximum(alpha, mapped[:, :, 3])

    return color, alpha


def bake_material_to_image(context, material, image, bake_type="EMIT",
                           on_progress=None):
    """Bake a material's evaluated surface into an image, for every mesh using it.

    One bake covers the whole atlas: all the meshes share the material, and
    therefore share its active image node, so Cycles writes each of them into the
    same target. It also bakes the EVALUATED mesh, so modifiers and subdivision
    come along - something the numpy rasteriser could never do.

    ``EMIT`` collects only emission, and a Principled BSDF surface has none, so
    the surface is made emissive for the duration of the bake. That is exactly why
    an earlier attempt at native bake wrote black: it was baking a BSDF as EMIT,
    which is black by definition.

    The meshes are baked ONE AT A TIME rather than as a single selection. One
    bake of everything is faster, but it blocks the main thread with no chance
    to repaint, which the artist experiences as a freeze on a large atlas. Per
    object, ``on_progress(done, total, mesh_name)`` runs between them and can
    drive a real progress readout. Each bake accumulates into the same image, so
    the resulting atlas is the same either way.
    """
    if material is None or not material.use_nodes or material.node_tree is None:
        return False

    view_layer = context.view_layer
    meshes = [m for m in meshes_sharing_material(material)
              if m.name in view_layer.objects]
    if not meshes:
        return False

    scene = context.scene
    tree = material.node_tree
    nodes, links = tree.nodes, tree.links

    saved = {
        "engine": scene.render.engine,
        "samples": scene.cycles.samples,
        "bounces": scene.cycles.max_bounces,
        "persistent": getattr(scene.render, "use_persistent_data", False),
        "margin": scene.render.bake.margin,
        "selected_to_active": scene.render.bake.use_selected_to_active,
        "active": view_layer.objects.active,
        "selection": [o for o in view_layer.objects if o.select_get()],
    }

    target = None
    emission = None
    surface = None
    saved_surface = None
    try:
        active = view_layer.objects.active
        if active is not None and active.mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")

        # Objects are selected one at a time inside the loop below.
        output = next((n for n in nodes
                       if n.type == "OUTPUT_MATERIAL"
                       and getattr(n, "is_active_output", False)), None)
        if output is None:
            return False
        surface = output.inputs["Surface"]

        if bake_type == "EMIT":
            saved_surface = surface.links[0].from_socket if surface.links else None
            emission = nodes.new("ShaderNodeEmission")
            emission.name = "PL_bake_emission"
            bsdf = next((n for n in nodes if n.type == "BSDF_PRINCIPLED"), None)
            if bsdf is not None and bsdf.inputs["Base Color"].links:
                links.new(bsdf.inputs["Base Color"].links[0].from_socket,
                          emission.inputs["Color"])
            elif bsdf is not None:
                emission.inputs["Color"].default_value = tuple(
                    bsdf.inputs["Base Color"].default_value)
            elif saved_surface is not None:
                links.new(saved_surface, emission.inputs["Color"])
            links.new(emission.outputs["Emission"], surface)

        target = nodes.new("ShaderNodeTexImage")
        target.name = "PL_bake_target"
        target.image = image
        # Selected and active LAST: the bake looks for this pair, and silently
        # writes nothing when it cannot find it.
        #
        # Compare by RNA pointer, NOT `node is target`: iterating a bpy
        # collection hands back a fresh Python wrapper each time, so identity
        # comparison is always False and nothing gets selected. That is exactly
        # why every earlier attempt at native bake wrote an all-black image while
        # reporting success.
        target_ptr = target.as_pointer()
        for node in nodes:
            node.select = node.as_pointer() == target_ptr
        nodes.active = target

        scene.render.engine = "CYCLES"
        scene.cycles.device = "CPU"
        scene.cycles.samples = 4
        scene.cycles.max_bounces = 0
        scene.cycles.use_denoising = False
        scene.render.use_persistent_data = False
        # No bake margin: Blender's dilation floods a connected empty region
        # rather than stopping after N pixels, and on an atlas that means the
        # unpainted half of the texture gets filled with island colour. Measured
        # at 128px: margin=0 left the empty half black, margin=8 filled it.
        scene.render.bake.margin = 0
        scene.render.bake.use_selected_to_active = False

        # Clear once, then bake every mesh with use_clear=False so each one
        # lands in the same atlas instead of erasing the previous object.
        # Prefer the operator's own clear: filling the pixels by hand wrote an
        # opaque alpha background that the bake then composited against, which
        # darkened the result (0.906 instead of 1.000 on a known colour).
        width, height = image.size
        try:
            supports_clear = "use_clear" in bpy.ops.object.bake.get_rna_type().properties
        except Exception:  # noqa: BLE001
            supports_clear = False
        if not supports_clear:
            blank = np.zeros(width * height * 4, dtype=np.float32)
            blank[3::4] = 1.0
            image.pixels.foreach_set(blank)
            image.update()

        baked_count = 0
        for index, mesh in enumerate(meshes):
            try:
                bpy.ops.object.select_all(action="DESELECT")
                mesh.select_set(True)
                view_layer.objects.active = mesh
                if supports_clear:
                    bpy.ops.object.bake(type=bake_type, width=width,
                                        height=height,
                                        use_clear=(index == 0))
                else:
                    bpy.ops.object.bake(type=bake_type, width=width,
                                        height=height)
                baked_count += 1
            except Exception as exc:  # noqa: BLE001
                # One object failing should not cost the whole atlas.
                print(f"[Projection Layers] bake failed for {mesh.name}: {exc}")
            if on_progress is not None:
                try:
                    on_progress(index + 1, len(meshes), mesh.name)
                except Exception:  # noqa: BLE001
                    pass

        image.update()
        return baked_count > 0
    except Exception as exc:  # noqa: BLE001
        print(f"[Projection Layers] native bake failed: {exc}")
        return False
    finally:
        try:
            if target is not None:
                nodes.remove(target)
        except Exception:  # noqa: BLE001
            pass
        try:
            if emission is not None:
                nodes.remove(emission)
            if surface is not None and saved_surface is not None:
                links.new(saved_surface, surface)
        except Exception:  # noqa: BLE001
            pass
        scene.render.engine = saved["engine"]
        scene.cycles.samples = saved["samples"]
        scene.cycles.max_bounces = saved["bounces"]
        scene.render.use_persistent_data = saved["persistent"]
        scene.render.bake.margin = saved["margin"]
        scene.render.bake.use_selected_to_active = saved["selected_to_active"]
        try:
            bpy.ops.object.select_all(action="DESELECT")
            for obj in saved["selection"]:
                obj.select_set(True)
            view_layer.objects.active = saved["active"]
        except Exception:  # noqa: BLE001
            pass


def _try_native_bake(context, material, size, bake_type, on_progress=None):
    """Bake the live material into an array, or None if the bake cannot run.

    A completely black result counts as failure so the caller can fall back to
    the numpy compositor, which cannot fail silently.
    """
    probe = bpy.data.images.new("PL_bake_probe", size, size, alpha=False)
    try:
        if bake_type != "EMIT":
            try:
                probe.colorspace_settings.name = "Non-Color"
            except TypeError:
                pass
        if not bake_material_to_image(context, material, probe, bake_type,
                                      on_progress=on_progress):
            return None
        array = render.image_rgb(probe)
        if float(array.max()) <= 0.0:
            return None
        return array
    finally:
        bpy.data.images.remove(probe)


def flatten_color_stack(context, obj, size, uv_name, progress=None):
    """Composite every visible colour layer into one UV-space atlas.

    Done in numpy rather than with a Cycles bake, for the same reason the mask
    transfer is: ``object.bake`` produced black every way it was tried here.
    Each layer's picture is mapped from its camera projection into the UVs, then
    composited with its own mask and opacity.

    Every mesh sharing the material contributes its own islands. Baking only the
    active object left the rest of an atlas black.
    """
    color_layers = [l for l in visible_layers(context, "COLOR") if l.source_image]
    if not color_layers:
        return None

    # Native Cycles bake first: one C pass over every mesh sharing the material,
    # sampling the evaluated mesh rather than the base one. The numpy compositor
    # below stays as the fallback for when the bake cannot run or comes out black.
    baked = _try_native_bake(
        context, obj.active_material, size, "EMIT",
        on_progress=(lambda done, total, _name:
                     progress(done, total, "colour"))
        if progress is not None else None)
    if baked is not None:
        return baked

    meshes = meshes_sharing_material(obj.active_material) or [obj]
    out = np.zeros((size, size, 3), dtype=np.float32)

    for index, layer in enumerate(color_layers):
        if progress is not None:
            progress(index + 1, len(color_layers), "colour")
        color, alpha = _sample_layer_in_atlas(meshes, layer, size, uv_name)

        if not alpha.any():
            continue  # covers no texel at all: compositing it is a no-op

        # Composite only where the layer actually lands. Outside this rectangle
        # the factor is zero, so the result is bit-identical - but these arrays
        # are size*size*3 and one object in an atlas usually occupies a small
        # share of them. This, not the rasteriser, is where flatten spends its
        # time once the atlas gets large.
        rows = np.any(alpha > 0.0, axis=1)
        cols = np.any(alpha > 0.0, axis=0)
        top = int(np.argmax(rows))
        bottom = size - 1 - int(np.argmax(rows[::-1]))
        left = int(np.argmax(cols))
        right = size - 1 - int(np.argmax(cols[::-1]))

        sub_color = _apply_hsv(color[top:bottom + 1, left:right + 1], layer)
        sub_mask = None
        if layer.mask_image is not None:
            mask_gray = render.image_gray(layer.mask_image)
            if float(mask_gray.min()) == float(mask_gray.max()):
                # A uniform mask multiplies by a constant, so the resample and
                # the multiply are pure waste. Masks are pure white by default
                # now, which makes this the common case.
                level = float(mask_gray.min())
                if level >= 0.999:
                    sub_mask = None
                    mask_level = 1.0
                else:
                    mask_level = level
            else:
                sub_mask = render.resize_nearest(mask_gray, size)[
                    top:bottom + 1, left:right + 1]
                mask_level = 1.0
        else:
            mask_level = 1.0

        sub_alpha = alpha[top:bottom + 1, left:right + 1]
        scale = float(layer.opacity) * mask_level
        if sub_mask is None:
            fac = (scale * sub_alpha)[:, :, None]
        else:
            fac = (sub_mask * scale * sub_alpha)[:, :, None]

        region = out[top:bottom + 1, left:right + 1]
        if layer.blend == "MIX":
            # out + (color - out) * fac is the same lerp with fewer temporaries,
            # and cannot leave [0, 1], so it needs no clip.
            region += (sub_color - region) * fac
        else:
            if layer.blend == "MULTIPLY":
                blended = region * sub_color
            elif layer.blend == "SCREEN":
                blended = 1.0 - (1.0 - region) * (1.0 - sub_color)
            elif layer.blend == "ADD":
                blended = region + sub_color
            else:
                blended = sub_color
            region[:] = np.clip(region * (1.0 - fac) + blended * fac, 0.0, 1.0)

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
    """Composite every visible normal layer into one UV-space atlas.

    Straight whiteout blending: the X and Y slopes add, Z multiplies, and the
    result is renormalised. Every mesh sharing the material contributes its own
    UV islands, exactly as the colour stack does.
    """
    layers = [l for l in visible_layers(context, "NORMAL") if l.source_image]
    if not layers:
        return None

    # NORMAL needs no emissive rewiring: Cycles reads the real surface normal.
    baked = _try_native_bake(
        context, obj.active_material, size, "NORMAL",
        on_progress=(lambda done, total, _name:
                     progress(done, total, "normals"))
        if progress is not None else None)
    if baked is not None:
        return baked

    meshes = meshes_sharing_material(obj.active_material) or [obj]
    result = None
    for index, layer in enumerate(layers):
        if progress is not None:
            progress(index + 1, len(layers), "normals")
        color, alpha = _sample_layer_in_atlas(meshes, layer, size, uv_name)

        # A uniform mask multiplies by a constant, so resampling it into a
        # size*size array is pure waste - and masks are pure white by default
        # now, making that the common case. Same treatment as the colour path.
        mask_level = 1.0
        sub_mask = None
        if layer.mask_image is not None:
            mask_gray = render.image_gray(layer.mask_image)
            if float(mask_gray.min()) == float(mask_gray.max()):
                mask_level = float(mask_gray.min())
            else:
                sub_mask = render.resize_nearest(mask_gray, size)

        if sub_mask is None:
            fac = (mask_level * float(layer.opacity) * alpha)[:, :, None]
        else:
            fac = (sub_mask * float(layer.opacity) * alpha)[:, :, None]
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


def _allocate_owner_id(obj):
    """An object-colour ALPHA value nothing else in the scene is using.

    The alpha channel, not RGB: RGB is the viewport object colour, so writing
    there would visibly recolour the object. Object Index would avoid both but
    EEVEE Next fails to compile a material that uses it inside a nested group.
    """
    used = {round(other.color[3], 3) for other in bpy.data.objects if other is not obj}
    value = 0.01
    while round(value, 3) in used and value < 0.98:
        value += 0.01
    obj.color = (obj.color[0], obj.color[1], obj.color[2], value)
    return value


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
    # viewport Ä‚ËĂ˘â€šÂ¬Ă˘â‚¬ĹĄ this used to run unconditionally, which overwrote every view's
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

    # Only this workflow's own layers count as "already painted" Ă„â€šĂ‹ÂÄ‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ä‚ËĂ˘â€šÂ¬ÄąÄ„ normal layers
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
        if payload.isolate or conf.hide_others:
            stack.enter_context(render.only_visible(context, obj))
        # Unconditional: whatever the artist hid in the viewport is not part of
        # the render. This used to be gated behind a setting, and a scene saved
        # with that setting off kept rendering hidden objects no matter what the
        # default said - changing an RNA default does not update stored values.
        stack.enter_context(render.respect_viewport_visibility(context))
        return stack

    # Ă„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄ„Ä‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ă„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄ„Ä‚ËĂ˘â‚¬ĹˇĂ‚Â¬ beauty reference Ă„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄ„Ä‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ă„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄ„Ä‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ă„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄ„Ä‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ă„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄ„Ä‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ă„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄ„Ä‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ă„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄ„Ä‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ă„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄ„Ä‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ă„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄ„Ä‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ă„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄ„Ä‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ă„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄ„Ä‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ă„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄ„Ä‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ă„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄ„Ä‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ă„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄ„Ä‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ă„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄ„Ä‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ă„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄ„Ä‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ă„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄ„Ä‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ă„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄ„Ä‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ă„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄ„Ä‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ă„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄ„Ä‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ă„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄ„Ä‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ă„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄ„Ä‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ă„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄ„Ä‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ă„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄ„Ä‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ă„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄ„Ä‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ă„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄ„Ä‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ă„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄ„Ä‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ă„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄ„Ä‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ă„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄ„Ä‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ă„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄ„Ä‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ă„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄ„Ä‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ă„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄ„Ä‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ă„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄ„Ä‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ă„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄ„Ä‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ă„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄ„Ä‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ă„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄ„Ä‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ă„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄ„Ä‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ă„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄ„Ä‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ă„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄ„Ä‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ă„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄ„Ä‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ă„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄ„Ä‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ă„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄ„Ä‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ă„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄ„Ä‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ă„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄ„Ä‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ă„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄ„Ä‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ă„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄ„Ä‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ă„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄ„Ä‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ă„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄ„Ä‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ă„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄ„Ä‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ă„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄ„Ä‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ă„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄ„Ä‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ă„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄ„Ä‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ă„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄ„Ä‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ă„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄ„Ä‚ËĂ˘â‚¬ĹˇĂ‚Â¬
    _set_status(context, "Rendering referenceĂ„â€šĂ‹ÂÄ‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ä‚â€šĂ‚Â¦")
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

    # Ă„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄ„Ä‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ă„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄ„Ä‚ËĂ˘â‚¬ĹˇĂ‚Â¬ optional guides Ă„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄ„Ä‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ă„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄ„Ä‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ă„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄ„Ä‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ă„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄ„Ä‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ă„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄ„Ä‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ă„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄ„Ä‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ă„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄ„Ä‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ă„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄ„Ä‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ă„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄ„Ä‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ă„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄ„Ä‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ă„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄ„Ä‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ă„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄ„Ä‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ă„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄ„Ä‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ă„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄ„Ä‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ă„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄ„Ä‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ă„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄ„Ä‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ă„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄ„Ä‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ă„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄ„Ä‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ă„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄ„Ä‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ă„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄ„Ä‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ă„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄ„Ä‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ă„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄ„Ä‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ă„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄ„Ä‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ă„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄ„Ä‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ă„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄ„Ä‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ă„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄ„Ä‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ă„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄ„Ä‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ă„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄ„Ä‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ă„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄ„Ä‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ă„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄ„Ä‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ă„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄ„Ä‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ă„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄ„Ä‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ă„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄ„Ä‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ă„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄ„Ä‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ă„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄ„Ä‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ă„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄ„Ä‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ă„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄ„Ä‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ă„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄ„Ä‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ă„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄ„Ä‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ă„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄ„Ä‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ă„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄ„Ä‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ă„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄ„Ä‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ă„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄ„Ä‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ă„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄ„Ä‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ă„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄ„Ä‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ă„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄ„Ä‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ă„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄ„Ä‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ă„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄ„Ä‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ă„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄ„Ä‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ă„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄ„Ä‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ă„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄ„Ä‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ă„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄ„Ä‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ă„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄ„Ä‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ă„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄ„Ä‚ËĂ˘â‚¬ĹˇĂ‚Â¬
    if conf.send_depth:
        _set_status(context, "Rendering depthĂ„â€šĂ‹ÂÄ‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ä‚â€šĂ‚Â¦")
        depth_path = os.path.join(run_dir, "depth.png")
        with _isolate():
            render.render_depth(context, obj, camera, payload.width, payload.height, depth_path)
        uploads["depth"] = _upload(base_url, depth_path, "depth.png")

    if conf.send_normal:
        _set_status(context, "Rendering normalĂ„â€šĂ‹ÂÄ‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ä‚â€šĂ‚Â¦")
        normal_path = os.path.join(run_dir, "normal.png")
        with _isolate():
            render.render_normal(context, obj, camera, payload.width, payload.height, normal_path)
        uploads["normal"] = _upload(base_url, normal_path, "normal.png")

    # Ă„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄ„Ä‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ă„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄ„Ä‚ËĂ˘â‚¬ĹˇĂ‚Â¬ workflow (resolved before the mask so we know if it wants one) Ă„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄ„Ä‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ă„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄ„Ä‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ă„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄ„Ä‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ă„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄ„Ä‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ă„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄ„Ä‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ă„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄ„Ä‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ă„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄ„Ä‚ËĂ˘â‚¬ĹˇĂ‚Â¬
    _set_status(context, "Preparing workflowĂ„â€šĂ‹ÂÄ‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ä‚â€šĂ‚Â¦")
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

    # Ă„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄ„Ä‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ă„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄ„Ä‚ËĂ˘â‚¬ĹˇĂ‚Â¬ inpaint mask: a painted pre-mask wins over the auto coverage mask Ă„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄ„Ä‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ă„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄ„Ä‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ă„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄ„Ä‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ă„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄ„Ä‚ËĂ˘â‚¬ĹˇĂ‚Â¬
    premask = find_role_layer(context, "PREMASK") if conf.premask_on else None
    if premask is not None and premask.mask_image is not None:
        payload.premask_image = premask.mask_image.name
        # The pre-mask is painted in the object's UV space, so render it back
        # through the camera to get the screen-space mask the backend wants.
        if "mask" in slots:
            _set_status(context, "Sending the pre-maskĂ„â€šĂ‹ÂÄ‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ä‚â€šĂ‚Â¦")
            premask_path = os.path.join(run_dir, "premask.png")
            render.render_image_through_camera(
                context, obj, camera, premask.mask_image,
                premask.mask_uv or mesh_uv_name(obj),
                payload.width, payload.height, premask_path,
            )
            uploads["mask"] = _upload(base_url, premask_path, "premask.png")
    elif conf.only_fill_unpainted and sources and "mask" in slots:
        _set_status(context, "Building inpaint maskĂ„â€šĂ‹ÂÄ‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ä‚â€šĂ‚Â¦")
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
    job.set_status("GeneratingĂ„â€šĂ‹ÂÄ‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ä‚â€šĂ‚Â¦")
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

    # Ă„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄ„Ä‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ă„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄ„Ä‚ËĂ˘â‚¬ĹˇĂ‚Â¬ freeze this camera into a projection attribute Ă„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄ„Ä‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ă„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄ„Ä‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ă„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄ„Ä‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ă„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄ„Ä‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ă„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄ„Ä‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ă„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄ„Ä‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ă„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄ„Ä‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ă„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄ„Ä‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ă„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄ„Ä‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ă„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄ„Ä‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ă„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄ„Ä‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ă„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄ„Ä‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ă„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄ„Ä‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ă„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄ„Ä‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ă„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄ„Ä‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ă„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄ„Ä‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ă„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄ„Ä‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ă„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄ„Ä‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ă„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄ„Ä‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ă„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄ„Ä‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ă„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄ„Ä‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ă„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄ„Ä‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ă„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄ„Ä‚ËĂ˘â‚¬ĹˇĂ‚Â¬
    # Done first: the bake below uses these coordinates as its source UVs.
    attr_name = render.snapshot_projection(context, obj, camera)
    # Sharing a material means sharing the layer, so the other meshes need the
    # same projection attribute too - otherwise they sample a missing attribute.
    if not conf.only_selected:
        # A confined layer is gated off on the other meshes anyway, so there
        # is no point stamping the projection attribute onto them.
        _stamp_projection_on_sharers(context, obj, camera, attr_name)

    # Ă„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄ„Ä‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ă„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄ„Ä‚ËĂ˘â‚¬ĹˇĂ‚Â¬ the layer's mask, transferred into the object's own UV space Ă„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄ„Ä‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ă„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄ„Ä‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ă„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄ„Ä‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ă„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄ„Ä‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ă„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄ„Ä‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ă„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄ„Ä‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ă„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄ„Ä‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ă„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄ„Ä‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ă„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄ„Ä‚ËĂ˘â‚¬ĹˇĂ‚Â¬
    uv_name = mesh_uv_name(obj)
    if not uv_name:
        raise RuntimeError(
            f"'{obj.name}' has no UV map. Unwrap it first Ă„â€šĂ‹ÂÄ‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ä‚ËĂ˘â€šÂ¬ÄąÄ„ masks live in UV space "
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

    # Ă„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄ„Ä‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ă„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄ„Ä‚ËĂ˘â‚¬ĹˇĂ‚Â¬ create the layer Ă„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄ„Ä‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ă„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄ„Ä‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ă„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄ„Ä‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ă„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄ„Ä‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ă„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄ„Ä‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ă„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄ„Ä‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ă„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄ„Ä‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ă„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄ„Ä‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ă„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄ„Ä‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ă„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄ„Ä‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ă„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄ„Ä‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ă„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄ„Ä‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ă„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄ„Ä‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ă„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄ„Ä‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ă„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄ„Ä‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ă„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄ„Ä‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ă„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄ„Ä‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ă„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄ„Ä‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ă„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄ„Ä‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ă„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄ„Ä‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ă„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄ„Ä‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ă„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄ„Ä‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ă„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄ„Ä‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ă„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄ„Ä‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ă„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄ„Ä‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ă„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄ„Ä‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ă„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄ„Ä‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ă„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄ„Ä‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ă„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄ„Ä‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ă„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄ„Ä‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ă„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄ„Ä‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ă„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄ„Ä‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ă„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄ„Ä‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ă„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄ„Ä‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ă„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄ„Ä‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ă„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄ„Ä‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ă„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄ„Ä‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ă„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄ„Ä‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ă„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄ„Ä‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ă„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄ„Ä‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ă„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄ„Ä‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ă„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄ„Ä‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ă„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄ„Ä‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ă„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄ„Ä‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ă„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄ„Ä‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ă„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄ„Ä‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ă„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄ„Ä‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ă„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄ„Ä‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ă„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄ„Ä‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ă„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄ„Ä‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ă„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄ„Ä‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ă„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄ„Ä‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ă„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄ„Ä‚ËĂ˘â‚¬ĹˇĂ‚Â¬
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
        layer.owner_id = _allocate_owner_id(obj)


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

    # The pre-mask has done its job Ă„â€šĂ‹ÂÄ‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ä‚ËĂ˘â€šÂ¬ÄąÄ„ the new layer carries a copy of it Ă„â€šĂ‹ÂÄ‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ä‚ËĂ˘â€šÂ¬ÄąÄ„ so
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
    # matrix Ă„â€šĂ‹ÂÄ‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ä‚ËĂ˘â€šÂ¬ÄąÄ„ flatten by rows and it comes back transposed, which silently
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

    # EEVEE Next cannot compile Object Info inside a nested node group and shows
    # the material magenta; drop the per-object gate there rather than pink.
    engine = getattr(context.scene.render, "engine", "")
    nodes.OWNER_GATE_ENABLED = True  # gate is not what broke EEVEE

    nodes.rebuild(material, color_layers, normal_layers, blend)
    return material
