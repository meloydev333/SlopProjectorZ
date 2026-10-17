"""Property groups â€” the addon's data model.

Layer stacks live on Materials, one stack per material, so two objects with
different materials keep independent layers. Everything else (prompt, guides,
resolution, the generation queue) is scene-level on ``Scene.projlayer``.

Images are referenced by pointer and packed, so a saved .blend round-trips. Each
layer keeps its projection source (camera matrix + lens data) so it can be
re-projected later without regenerating.
"""

import bpy
from bpy.props import (
    BoolProperty,
    CollectionProperty,
    EnumProperty,
    FloatProperty,
    FloatVectorProperty,
    IntProperty,
    PointerProperty,
    StringProperty,
)

# â”€â”€ enums â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

ROLE_ITEMS = [
    ("LAYER", "Layer", "An ordinary projected layer"),
    ("PREMASK", "Pre-mask", "Painted before generating; becomes the new layer's mask"),
    ("PREPAINT", "Pre-paint", "Painted before generating; sent as the reference and removed afterwards"),
]

BLEND_ITEMS = [
    ("MIX", "Mix", "Straight alpha blend"),
    ("MULTIPLY", "Multiply", "Darkens â€” good for dirt and grime"),
    ("SCREEN", "Screen", "Lightens"),
    ("OVERLAY", "Overlay", "Boosts contrast"),
    ("ADD", "Add", "Linear add â€” good for emissive detail"),
    ("SOFT_LIGHT", "Soft Light", "Gentle contrast"),
    ("DARKEN", "Darken", "Keeps the darker of the two"),
    ("LIGHTEN", "Lighten", "Keeps the lighter of the two"),
]

NORMAL_BLEND_ITEMS = [
    ("WHITEOUT", "Whiteout", "Cheap, stable, good for most detail"),
    ("RNM", "Reoriented (RNM)", "Higher quality, avoids flattened detail"),
    ("MIX", "Mix (flat)", "Plain vector mix â€” flattens stacked detail"),
]

#: The two independent workflows. Shared by the layer stack and by each saved
#: view, so a view remembers which workflow it belongs to.
MODE_ITEMS = [
    ("COLOR", "Color", "Project base colour layers"),
    ("NORMAL", "Normal", "Project normal-map layers"),
]

RESOLUTION_ITEMS = [
    ("512", "512", "Draft"),
    ("1024", "1024", "Fast"),
    ("2048", "2048", "Production"),
    ("4096", "4096", "Heavy"),
]

ASPECT_ITEMS = [
    ("1:1", "1:1", "Square"),
    ("4:3", "4:3", "Landscape"),
    ("3:2", "3:2", "Landscape"),
    ("16:9", "16:9", "Wide"),
    ("3:4", "3:4", "Portrait"),
    ("2:3", "2:3", "Portrait"),
    ("9:16", "9:16", "Tall"),
]

ASPECT_RATIOS = {
    "1:1": (1.0, 1.0),
    "4:3": (4.0, 3.0),
    "3:2": (3.0, 2.0),
    "16:9": (16.0, 9.0),
    "3:4": (3.0, 4.0),
    "2:3": (2.0, 3.0),
    "9:16": (9.0, 16.0),
}


# â”€â”€ live rebuild â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

_rebuild_queued = False


def _on_layer_change(self, context):
    """Rebuild the material when a layer property changes.

    Debounced through a one-shot timer: dragging a slider fires this on every
    tick, and rebuilding the node tree each time would stutter.
    """
    global _rebuild_queued
    if _rebuild_queued:
        return
    _rebuild_queued = True

    def _run():
        global _rebuild_queued
        _rebuild_queued = False
        from .core import pipeline

        try:
            pipeline.rebuild(bpy.context)
        except Exception:  # noqa: BLE001 â€” never break the UI over a redraw
            pass
        return None  # unregister

    try:
        bpy.app.timers.register(_run, first_interval=0.05)
    except Exception:  # noqa: BLE001
        _rebuild_queued = False


def _on_projection_change(self, context):
    """Same debounced rebuild, for the projection's own controls.

    Without this, Angle and Max Distance only reached the shader the next time
    something else happened to trigger a rebuild.
    """
    _on_layer_change(self, context)


#: Enum identifier meaning "sample through this layer's own camera projection"
#: rather than one of the mesh's UV maps. A real identifier is needed because
#: Blender warns when a dynamic enum's stored value is an empty string.
UV_PROJECTION = "PROJECTION"


def is_paint_layer(layer) -> bool:
    """True when the layer paints on a UV map instead of a camera projection."""
    return bool(layer.uv_map) and layer.uv_map != UV_PROJECTION


def _uv_map_items(self, context):
    """UV maps of the active mesh, for layers that paint on one directly."""
    items = [(UV_PROJECTION, "(camera projection)",
              "Sample through this layer's own projection")]
    obj = getattr(context, "active_object", None)
    if obj is not None and obj.type == "MESH":
        for uv in obj.data.uv_layers:
            items.append((uv.name, uv.name, f"Paint on '{uv.name}'"))
    return items


def _material_slot_items(self, context):
    """Material slots of the active mesh, for the stack panel's selector.

    Empty lists are not allowed from a dynamic-items callback, hence the
    placeholder â€” a dynamic enum also requires an integer default.
    """
    items = []
    obj = getattr(context, "active_object", None)
    if obj is not None and getattr(obj, "type", "") == "MESH":
        for index, slot in enumerate(obj.material_slots):
            label = slot.material.name if slot.material else "(empty slot)"
            items.append((str(index), label, f"Show the layers of {label}"))
    return items or [("0", "(no material)", "This mesh has no material slots")]


def _on_material_slot(self, context):
    """Show the stack of the slot the artist picked."""
    obj = getattr(context, "active_object", None)
    if obj is None or getattr(obj, "type", "") != "MESH":
        return
    try:
        index = int(self.material_slot)
    except (TypeError, ValueError):
        return
    if 0 <= index < len(obj.material_slots):
        obj.active_material_index = index
    tag_redraw(context)


# â”€â”€ projection source â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


class PL_Projection(bpy.types.PropertyGroup):
    """Frozen camera state for one projection, so a layer can be re-baked."""

    cam_matrix: FloatVectorProperty(name="Camera Matrix", size=16, subtype="MATRIX")
    lens: FloatProperty(name="Lens", default=50.0)
    sensor_width: FloatProperty(name="Sensor Width", default=36.0)
    sensor_fit: StringProperty(name="Sensor Fit", default="AUTO")
    shift_x: FloatProperty(name="Shift X", default=0.0)
    shift_y: FloatProperty(name="Shift Y", default=0.0)
    clip_start: FloatProperty(name="Clip Start", default=0.1)
    clip_end: FloatProperty(name="Clip End", default=100.0)

    res_x: IntProperty(name="Width", default=1024)
    res_y: IntProperty(name="Height", default=1024)

    feather_px: IntProperty(
        name="Soften", default=0, min=0, max=128, subtype="PIXEL",
        update=_on_projection_change,
        description="Softens the projection edge so it fades out instead of cutting off",
    )
    angle_cut: FloatProperty(
        name="Angle", default=75.0, min=5.0, max=90.0, subtype="ANGLE",
        update=_on_projection_change,
        description="Surfaces facing further away than this are dropped from the projection",
    )
    falloff: FloatProperty(
        name="Falloff", default=1.0, min=0.1, max=8.0,
        description=(
            "Shapes the short soft lip at the silhouette. 1 is a straight ramp; "
            "higher values hold the mask solid for longer before it fades"
        ),
    )
    max_distance: FloatProperty(
        name="Max Distance", default=0.0, min=0.0, max=10.0, soft_max=10.0,
        update=_on_projection_change,
        description=(
            "Do not project onto surfaces further than this from the camera. "
            "Stops a projection leaking onto stray geometry. 0 means no limit"
        ),
    )


# â”€â”€ layer â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


class PL_Layer(bpy.types.PropertyGroup):
    name: StringProperty(name="Name", default="Layer")

    # The raw image returned by ComfyUI. Never mutated, so Re-project can always
    # re-derive everything from scratch.
    source_image: PointerProperty(
        name="Image", type=bpy.types.Image, update=_on_layer_change,
    )
    mask_image: PointerProperty(
        name="Mask", type=bpy.types.Image, update=_on_layer_change,
    )

    # Name of the FLOAT_VECTOR/CORNER attribute holding this layer's projection
    # UVs. An attribute rather than a UV map because Blender caps UV layers at
    # eight; see core/render.py.
    uv_attr: StringProperty(name="Projection", default="", update=_on_layer_change)

    # Set for paint layers that work on one of the mesh's own UV maps instead of
    # a camera projection. Empty means "use the projection attribute".
    uv_map: EnumProperty(
        name="UV Map", items=_uv_map_items,
        # With a dynamic items callback the default must be an index; item 0 is
        # always UV_PROJECTION.
        default=0,
        update=_on_layer_change,
    )

    #: Which of the mesh's UV maps the mask lives in. Masks are stored in the
    #: object's own UV space (not the camera projection) so they can be painted
    #: from any angle with the ordinary texture tools.
    mask_uv: StringProperty(name="Mask UV", default="", update=_on_layer_change)

    #: Object pass index this layer is confined to; 0 means every object sharing
    #: the material (the default, which lets a projection bleed).
    owner_pass: IntProperty(name="Owner Index", default=0, min=0)

    role: EnumProperty(name="Role", items=ROLE_ITEMS, default="LAYER")

    #: Flat red instead of the layer's picture, to preview where it lands.
    show_coverage: BoolProperty(
        name="Show Coverage", default=False, update=_on_layer_change,
        description="Draw this layer flat red so you can see exactly where it lands",
    )

    #: Purely a visual tag, so a long stack stays readable.
    tag_color: FloatVectorProperty(
        name="Colour", subtype="COLOR", size=4,
        default=(0.42, 0.58, 0.92, 1.0), min=0.0, max=1.0,
    )

    projection: PointerProperty(name="Projection", type=PL_Projection)

    enabled: BoolProperty(name="Visible", default=True, update=_on_layer_change)
    solo: BoolProperty(name="Solo", default=False, update=_on_layer_change)
    opacity: FloatProperty(
        name="Opacity", default=1.0, min=0.0, max=1.0, subtype="FACTOR",
        update=_on_layer_change,
    )
    blend: EnumProperty(
        name="Blend", items=BLEND_ITEMS, default="MIX", update=_on_layer_change,
    )

    # 0.5 is neutral for a Hue/Saturation/Value node â€” the socket's range is
    # 0..1 with the centre meaning "no shift". Defaulting this to 0.0 rotated
    # every projection's hue by half a turn.
    hue: FloatProperty(
        name="Hue", default=0.5, min=0.0, max=1.0, subtype="FACTOR",
        update=_on_layer_change,
    )
    saturation: FloatProperty(
        name="Saturation", default=1.0, min=0.0, max=2.0, update=_on_layer_change,
    )
    value: FloatProperty(
        name="Value", default=1.0, min=0.0, max=2.0, update=_on_layer_change,
    )

    normal_strength: FloatProperty(
        name="Strength", default=1.0, min=0.0, max=4.0, update=_on_layer_change,
    )
    normal_flip_y: BoolProperty(
        name="Flip Y (DirectX)", default=False, update=_on_layer_change,
    )
    normal_blend: EnumProperty(
        name="Blend", items=NORMAL_BLEND_ITEMS, default="WHITEOUT",
        update=_on_layer_change,
    )

    # Provenance, shown in the UI.
    prompt: StringProperty(name="Prompt", default="")
    seed: IntProperty(name="Seed", default=0)
    bake_seconds: FloatProperty(name="Bake Time", default=0.0)


class PL_PromptSlot(bpy.types.PropertyGroup):
    """One saved prompt/negative pair, switchable from the prompt tabs."""

    prompt: StringProperty(name="Prompt", default="")
    negative: StringProperty(name="Negative", default="")


# â”€â”€ layer stack (per material) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


class PL_LayerStack(bpy.types.PropertyGroup):
    """The layer stacks belonging to one material.

    Stacks live on Materials rather than the Scene. Two objects with different
    materials are different surfaces, so sharing one stack meant painting a
    layer on one silently added it to the other.
    """

    # Colour and normal are two independent workflows with their own stacks, so
    # each keeps its own layer list and its own selection.
    mode: EnumProperty(
        name="Workflow", items=MODE_ITEMS, default="COLOR",
        update=_on_layer_change,
    )

    color_layers: CollectionProperty(type=PL_Layer)
    active_color: IntProperty(name="Active Color Layer", default=0, min=0)
    normal_layers: CollectionProperty(type=PL_Layer)
    active_normal: IntProperty(name="Active Normal Layer", default=0, min=0)


class PL_View(bpy.types.PropertyGroup):
    """One camera to generate from, for sequential multi-view runs."""

    camera: PointerProperty(name="Camera", type=bpy.types.Object)
    mode: EnumProperty(name="Workflow", items=MODE_ITEMS, default="COLOR")
    #: Per-view run state, so the panel can show what is done and what is next.
    state: EnumProperty(
        name="State",
        items=(
            ("PENDING", "Pending", "Not generated yet"),
            ("ACTIVE", "Generating", "Running now"),
            ("DONE", "Done", "Generated"),
            ("FAILED", "Failed", "Stopped on this view"),
        ),
        default="PENDING",
    )


# â”€â”€ scene settings â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

PROMPT_SLOTS = 5


def ensure_prompt_slots(conf) -> None:
    """Top the prompt tabs up to PROMPT_SLOTS entries.

    Never call this while drawing. Blender refuses to write to ID data (the
    Scene) from a UI draw callback, so this belongs in operators, handlers and
    registration â€” and the draw code has to cope with whatever it finds.
    """
    while len(conf.prompt_slots) < PROMPT_SLOTS:
        conf.prompt_slots.add()


def ensure_all_prompt_slots(*_args) -> None:
    """Top up every scene's prompt tabs. Safe from a handler, timer or register.

    ``bpy.data`` is a restricted object while the add-on is being registered, so
    this bails out quietly there and the one-shot timer registered alongside
    picks it up a moment later.
    """
    try:
        scenes = bpy.data.scenes
    except AttributeError:
        return
    for scene in scenes:
        conf = getattr(scene, "projlayer", None)
        if conf is None:
            continue
        try:
            ensure_prompt_slots(conf)
        except Exception:  # noqa: BLE001 â€” never break a file load over this
            pass


def _deferred_prompt_slots():
    """One-shot timer: finish registration-time work in a normal context."""
    ensure_all_prompt_slots()
    return None


class PL_Settings(bpy.types.PropertyGroup):
    # Layer stacks are per Material (``stack()``). This one is the fallback used
    # when there is no mesh to hang a stack on â€” with no active object, or a
    # selection that is not a mesh, the panel still draws and must never raise.
    fallback: PointerProperty(type=PL_LayerStack)

    #: Which material slot the stack panel is showing. Kept in sync with
    #: ``active_material_index`` so switching material in Blender's own slot UI
    #: also switches the stack.
    material_slot: EnumProperty(
        name="Material", items=_material_slot_items, default=0,
        update=_on_material_slot,
    )

    prompt: StringProperty(
        name="Prompt", default="",
        description="What to paint onto the mesh",
    )
    negative_prompt: StringProperty(name="Negative", default="")

    # Saved prompt sets. ``prompt``/``negative_prompt`` above stay the live
    # values; the tabs just save and restore them.
    prompt_slots: CollectionProperty(type=PL_PromptSlot)
    active_prompt: IntProperty(name="Prompt Tab", default=0, min=0)

    #: Draw the capture frame and status HUD over the viewport. The draw
    #: handlers are always attached; this flag decides whether they paint.
    #: Apply the next generation only to the active object, not to every mesh
    #: sharing its material.
    only_selected: BoolProperty(
        name="Only This Object", default=False,
        description=(
            "Apply the generation only to the active object. Other meshes "
            "sharing the material are gated off by object index"
        ),
    )

    show_frame: BoolProperty(
        name="Capture Frame", default=True,
        description="Outline the area that will be sent, in the viewport",
    )

    #: What the reference render should look like.
    send_what: EnumProperty(
        name="Send",
        items=(
            ("BEAUTY", "Beauty", "Shaded render, lit by the scene"),
            ("FLAT", "Flat Colour",
             "The stack's colour unlit, matching a diffuse colour view"),
        ),
        default="BEAUTY",
        description="What to send as the reference image",
    )

    #: Send only what is actually visible in the viewport.
    send_visible_only: BoolProperty(
        name="Visible Only", default=True,
        description=(
            "Also hide objects that are hidden in the viewport but not from "
            "renders, so the reference matches what you see"
        ),
    )

    #: Resolution to bake down to when flattening the stack.
    flatten_resolution: IntProperty(
        name="Bake Resolution", default=2048, min=64, max=8192,
        description="Size of the image a flatten bakes into",
    )

    #: Transient: what a long main-thread job (flatten) is currently doing, so
    #: the viewport HUD can show progress for it.
    busy_text: StringProperty(default="")
    seed: IntProperty(name="Seed", default=0, min=0)
    randomise_seed: BoolProperty(name="Randomise", default=True)

    resolution: EnumProperty(name="Resolution", items=RESOLUTION_ITEMS, default="1024")
    aspect: EnumProperty(name="Aspect", items=ASPECT_ITEMS, default="1:1")

    only_fill_unpainted: BoolProperty(
        name="Only fill unpainted areas", default=False,
        description=(
            "Send the union of all existing masks as an inpaint mask, so the "
            "backend only fills the parts of the mesh that are still blank"
        ),
    )

    send_depth: BoolProperty(name="Depth", default=True)
    send_normal: BoolProperty(name="Normal", default=True)

    auto_repaint_beauty: BoolProperty(
        name="Send previous layers as reference", default=True,
        description="Composite existing layers into the reference image so new generations match them",
    )

    auto_snap_camera: BoolProperty(
        name="Snap the camera to my view on Generate", default=True,
        description=(
            "Move the projector camera to match the 3D viewport right before "
            "rendering, so Generate always captures the view you are looking at"
        ),
    )

    # Pre-mask / pre-paint modes. Turning a mode off only hides its layer â€” the
    # painted work is kept so re-enabling picks up where you left off.
    premask_on: BoolProperty(
        name="Pre-mask", default=False,
        description="Paint the area to regenerate before generating",
    )
    prepaint_on: BoolProperty(
        name="Pre-paint", default=False,
        description="Sketch on the mesh before generating; the sketch feeds the reference image",
    )

    # â”€â”€ multi-camera sequential generation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    views: CollectionProperty(type=PL_View)
    view_index: IntProperty(name="View", default=0, min=0)

    # The queue is transient: which saved view we are on, and whether a run is
    # in progress. Views are referenced by index rather than by camera name
    # because a PL_View also carries a workflow, which a bare name cannot.
    queue_index: IntProperty(default=0, min=0)
    queue_active: BoolProperty(default=False)

    # â”€â”€ batch texturing over a collection â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # One job per (object x view). The cross-product is flattened into
    # queue_index so a batch and a plain view run share one chain:
    #   object_index = queue_index // len(views)
    #   view_index   = queue_index %  len(views)
    target_collection: PointerProperty(
        name="Collection", type=bpy.types.Collection,
        description="Every mesh in this collection is textured, one at a time",
    )
    isolate_batch: BoolProperty(
        name="Isolate Object", default=True,
        description=(
            "Hide the other objects while rendering, so co-located meshes do "
            "not appear in (and occlude) each other's reference image"
        ),
    )
    batch_active: BoolProperty(default=False)
    #: Object names for the running batch, "|"-separated. Names rather than
    #: references because the collection can change between jobs.
    batch_objects: StringProperty(default="")
    #: Names skipped at scan time (no mesh, no UVs, no material), for reporting.
    batch_skipped: StringProperty(default="")
    batch_object_index: IntProperty(default=0, min=0)
    batch_object_total: IntProperty(default=0, min=0)

    # â”€â”€ transient job status, read by the panel â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    status: StringProperty(name="Status", default="")
    progress: FloatProperty(name="Progress", default=0.0, min=0.0, max=1.0)
    progress_text: StringProperty(name="Progress Text", default="")
    is_running: BoolProperty(name="Running", default=False)
    last_error: StringProperty(name="Error", default="")


CLASSES = (
    PL_Projection,
    PL_Layer,
    PL_PromptSlot,
    PL_LayerStack,
    PL_View,
    PL_Settings,
)


def tag_redraw(context) -> None:
    """Force a sidebar repaint. Property writes from a timer do not repaint on
    their own, which is why the mode buttons looked stuck on."""
    wm = getattr(context, "window_manager", None)
    if wm is None:
        return
    for window in wm.windows:
        screen = window.screen
        if screen is None:
            continue
        for area in screen.areas:
            if area.type in {"VIEW_3D", "PROPERTIES"}:
                area.tag_redraw()


def set_pre_modes(context, premask=None, prepaint=None, rebuild=True) -> None:
    """Arm or disarm the pre-mask / pre-paint modes in one place.

    The layers are never deleted â€” only enabled or disabled â€” so leaving a mode
    and coming back keeps the painting. Doing this in one helper matters: the
    flags and the layers have to move together, or the buttons and the viewport
    disagree.
    """
    conf = settings(context)
    if premask is not None:
        conf.premask_on = bool(premask)
    if prepaint is not None:
        conf.prepaint_on = bool(prepaint)

    for role, active in (("PREMASK", conf.premask_on),
                         ("PREPAINT", conf.prepaint_on)):
        layer = find_role_layer(context, role)
        if layer is None:
            continue
        layer.enabled = bool(active)
        if role == "PREMASK":
            # The pre-mask shows itself as flat red while it is armed, so the
            # painted area is obvious.
            layer.show_coverage = bool(active)

    if rebuild:
        try:
            from .core import pipeline

            pipeline.rebuild(context)
        except Exception:  # noqa: BLE001
            pass
    tag_redraw(context)


def unhide_for_capture(context) -> None:
    """Hide the pre-mask's red overlay so it does not tint the reference image."""
    conf = settings(context)
    if not conf.premask_on:
        return
    layer = find_role_layer(context, "PREMASK")
    if layer is not None and layer.enabled:
        layer.enabled = False
        layer.show_coverage = False
        try:
            from .core import pipeline

            pipeline.rebuild(context)
        except Exception:  # noqa: BLE001
            pass


def restore_after_capture(context) -> None:
    """Put the pre-mask overlay back once the request has gone out."""
    conf = settings(context)
    if not conf.premask_on:
        return
    layer = find_role_layer(context, "PREMASK")
    if layer is not None and not layer.enabled:
        layer.enabled = True
        layer.show_coverage = True
        try:
            from .core import pipeline

            pipeline.rebuild(context)
        except Exception:  # noqa: BLE001
            pass
    tag_redraw(context)


def _sync_material_slot(context):
    """Keep the Material dropdown following Blender's own active slot.

    Switching material through Blender's slot list would otherwise leave the
    dropdown showing the previous material while the stack panel below it had
    already moved to the new one.
    """
    conf = getattr(getattr(context, "scene", None), "projlayer", None)
    obj = getattr(context, "active_object", None)
    if conf is None or obj is None or getattr(obj, "type", "") != "MESH":
        return
    index = int(getattr(obj, "active_material_index", 0))
    if not (0 <= index < len(obj.material_slots)):
        return
    wanted = str(index)
    if conf.material_slot != wanted:
        conf.material_slot = wanted


def _watch_paint_mode():
    """Disarm the pre-mask / pre-paint modes when the artist leaves painting.

    Wrapped whole: an exception escaping a timer makes Blender *unregister* it,
    so a single bad tick would silently disable this for the rest of the
    session â€” which is exactly what happened before.
    """
    try:
        if not bpy.app.background:
            context = bpy.context
            conf = getattr(getattr(context, "scene", None), "projlayer", None)
            if conf is not None:
                _sync_material_slot(context)
                if conf.premask_on or conf.prepaint_on:
                    if getattr(context, "mode", "") != "PAINT_TEXTURE":
                        set_pre_modes(context, premask=False, prepaint=False)
    except Exception:  # noqa: BLE001
        pass
    return 0.4


def start_watcher():
    if not bpy.app.timers.is_registered(_watch_paint_mode):
        bpy.app.timers.register(_watch_paint_mode, first_interval=0.4)


def stop_watcher():
    if bpy.app.timers.is_registered(_watch_paint_mode):
        bpy.app.timers.unregister(_watch_paint_mode)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Scene.projlayer = PointerProperty(type=PL_Settings)
    # Stacks hang off Materials, so two objects with different materials keep
    # independent layer lists.
    bpy.types.Material.projlayer = PointerProperty(type=PL_LayerStack)
    # Scenes that already exist (and every scene in every file loaded from now
    # on) get their prompt tabs here, never during a draw. Registration itself
    # runs with bpy.data restricted, hence the deferred timer.
    ensure_all_prompt_slots()
    if not bpy.app.timers.is_registered(_deferred_prompt_slots):
        bpy.app.timers.register(_deferred_prompt_slots, first_interval=0.2)
    if ensure_all_prompt_slots not in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.append(ensure_all_prompt_slots)
    start_watcher()


def unregister():
    stop_watcher()
    if bpy.app.timers.is_registered(_deferred_prompt_slots):
        bpy.app.timers.unregister(_deferred_prompt_slots)
    if ensure_all_prompt_slots in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.remove(ensure_all_prompt_slots)
    del bpy.types.Scene.projlayer
    del bpy.types.Material.projlayer
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)


# â”€â”€ helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


def settings(context) -> PL_Settings:
    return context.scene.projlayer


def stack(context, create=False) -> PL_LayerStack:
    """The layer stack for the active mesh's active material.

    ``create`` is False by default because this is reachable from panel draw
    code: creating a material writes to ID data, which Blender forbids during a
    draw. Operators and :func:`pipeline.rebuild` pass ``create=True``, which is
    where a missing material should actually get made.

    When there is nothing to hang a stack on, the scene's fallback stack is
    returned so a panel can always draw.
    """
    conf = settings(context)
    obj = getattr(context, "active_object", None)
    if obj is None or getattr(obj, "type", "") != "MESH":
        return conf.fallback

    material = getattr(obj, "active_material", None)
    if material is None:
        if not create:
            return conf.fallback
        material = bpy.data.materials.new(f"{obj.name} Material")
        material.use_nodes = True
        if len(obj.data.materials) == 0:
            obj.data.materials.append(material)
        else:
            index = max(0, min(obj.active_material_index, len(obj.data.materials) - 1))
            obj.data.materials[index] = material

    return material.projlayer


def current_mode(context) -> str:
    """The workflow the current stack is showing: "COLOR" or "NORMAL"."""
    return stack(context).mode


def set_current_mode(context, value) -> None:
    stack(context).mode = value


def mode_layers(context):
    """The layer collection belonging to the workflow currently selected."""
    s = stack(context)
    return s.normal_layers if s.mode == "NORMAL" else s.color_layers


def mode_index_prop(context) -> str:
    """Name of the active-index property for the current workflow."""
    return "active_normal" if stack(context).mode == "NORMAL" else "active_color"


def mode_active_index(context) -> int:
    s = stack(context)
    return s.active_normal if s.mode == "NORMAL" else s.active_color


def set_mode_active_index(context, value):
    s = stack(context)
    if s.mode == "NORMAL":
        s.active_normal = value
    else:
        s.active_color = value


def active_layer(context):
    """The selected layer of the current workflow, or None."""
    layers = mode_layers(context)
    if not layers:
        return None
    return layers[max(0, min(mode_active_index(context), len(layers) - 1))]


def visible_layers(context, mode=None):
    """Layers of ``mode`` (default: both) that should contribute, honouring solo."""
    s = stack(context)
    groups = []
    if mode in (None, "COLOR"):
        groups.append(s.color_layers)
    if mode in (None, "NORMAL"):
        groups.append(s.normal_layers)

    out = []
    for layers in groups:
        solo = [l for l in layers if l.solo and l.enabled]
        out.extend(solo if solo else [l for l in layers if l.enabled])
    return out


def find_role_layer(context, role):
    """The first layer with a given role, looking in both stacks."""
    s = stack(context)
    for layers in (s.color_layers, s.normal_layers):
        for layer in layers:
            if layer.role == role:
                return layer
    return None


def role_layers(context, role):
    """Every layer with a given role."""
    s = stack(context)
    return [l for layers in (s.color_layers, s.normal_layers)
            for l in layers if l.role == role]


def resolution(context):
    s = settings(context)
    w, h = ASPECT_RATIOS.get(s.aspect, (1.0, 1.0))
    base = int(s.resolution)
    if w >= h:
        return base, max(1, int(round(base * h / w)))
    return max(1, int(round(base * w / h))), base
