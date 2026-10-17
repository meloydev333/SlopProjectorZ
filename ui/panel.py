"""The sidebar UI.

Deliberately small: two panels. Generate does camera, prompt and the button.
Layers holds the Color/Normal switch and the stack, where every layer carries
its own thumbnail, tweaks and paint buttons â€” there is no separate settings
panel to jump between.

Pre-mask and Pre-paint are working modes rather than layers: they never appear
in the stack. Their sole job is to shape the *next* generation.
"""

import bpy

from .. import prefs as prefs_module
from ..core import state
from . import overlay
from ..props import (
    PROMPT_SLOTS,
    active_layer,
    current_mode,
    is_paint_layer,
    mode_active_index,
    mode_layers,
    settings,
    stack,
)

TAB = overlay.CATEGORY


def _thumb_icon(image):
    """Preview icon id for an image datablock, or 0 if unavailable."""
    if image is None:
        return 0
    try:
        image.preview_ensure()
        return image.preview.icon_id
    except Exception:
        return 0


def _thumb_scale(context):
    """template_icon's scale multiplier, driven by the size preference."""
    try:
        size = prefs_module.prefs(context).thumbnail_size
    except Exception:
        size = 48
    return max(1.0, float(size) / 16.0)


def _progress(layout, factor, text):
    row = layout.row()
    drawer = getattr(row, "progress", None)
    if callable(drawer):
        drawer(factor=factor, text=text)
        return
    row.label(text=text, icon="TIME")


def _stack_layers(context):
    """Only real layers: the pre-mask/pre-paint working layers stay hidden."""
    return [l for l in mode_layers(context) if l.role == "LAYER"]


# â”€â”€ generate â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


class PL_PT_generate(bpy.types.Panel):
    bl_idname = "PL_PT_generate"
    bl_label = "Projection Layers"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = TAB

    def draw_header(self, context):
        conf = settings(context)
        if state.is_running():
            self.layout.label(text="", icon="TIME")
        elif conf.last_error:
            self.layout.label(text="", icon="ERROR")
        else:
            self.layout.label(text="", icon="SHADERFX")

    def draw(self, context):
        layout = self.layout
        conf = settings(context)
        overlay.note_panel_drawn()
        try:
            address = prefs_module.prefs(context).server_address
        except Exception:
            address = ""

        row = layout.row(align=True)
        row.label(text="", icon="ERROR" if conf.last_error else "CHECKMARK")
        row.label(text=address or "no server set")
        row.operator("projlayer.test_connection", text="", icon="FILE_REFRESH")
        row.operator("projlayer.detect_slots", text="", icon="ZOOM_SELECTED")

        layout.separator(factor=0.5)

        # â”€â”€ camera (Generate snaps to the view on its own) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        row = layout.row(align=True)
        row.operator("projlayer.fit_camera", icon="FULLSCREEN_ENTER")
        row.prop(conf, "show_frame", text="", toggle=True, icon="VIEW3D")
        row.prop(conf, "resolution", text="")
        row.prop(conf, "aspect", text="")

        layout.separator(factor=0.5)

        # â”€â”€ prompts, with saved tabs â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        box = layout.box()
        # Draw only the tabs that exist: writing to ID data from draw() is not
        # allowed, so filling them in is registration's and the operators' job.
        tab_count = min(PROMPT_SLOTS, len(conf.prompt_slots))
        if tab_count:
            tabs = box.row(align=True)
            tabs.label(text="")
            for index in range(tab_count):
                op = tabs.operator(
                    "projlayer.select_prompt", text=str(index + 1),
                    depress=(index == conf.active_prompt),
                )
                op.index = index

        prompt_col = box.column()
        prompt_col.scale_y = 2.6
        prompt_col.prop(conf, "prompt", text="")

        neg_row = box.row(align=True)
        neg_row.prop(conf, "negative_prompt", text="", icon="TRASH")

        row = box.row(align=True)
        row.prop(conf, "send_depth", toggle=True, icon="MOD_MASK")
        row.prop(conf, "send_normal", toggle=True, icon="NORMALS_FACE")

        row = box.row(align=True)
        # No "Visible Only" switch on purpose: hiding from the render is now
        # unconditional, so a toggle here would misrepresent what it does.
        row.prop(conf, "send_what", text="")
        row = box.row(align=True)
        row.prop(conf, "only_selected", text="Only This Object", toggle=True)

        # â”€â”€ views: generate the same prompt from several cameras â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        views = layout.box()
        head = views.row(align=True)
        head.label(text="Views", icon="CAMERA_DATA")
        head.operator("projlayer.view_add", text="", icon="ADD")
        head.operator("projlayer.view_remove", text="", icon="X")
        if conf.views:
            icons = {
                "PENDING": "CAMERA_DATA",
                "ACTIVE": "TIME",
                "DONE": "CHECKMARK",
                "FAILED": "ERROR",
            }
            done = sum(1 for view in conf.views if view.state == "DONE")
            if state.is_running() or done:
                head.label(text=f"{done}/{len(conf.views)}")

            column = views.column(align=True)
            for index, view in enumerate(conf.views):
                camera = view.camera
                label = camera.name if camera is not None else "(missing camera)"
                row = column.row(align=True)
                op = row.operator(
                    "projlayer.view_select", text=label,
                    icon=icons.get(view.state, "CAMERA_DATA"),
                    emboss=(index == conf.view_index),
                )
                op.index = index
                if view.state == "ACTIVE" and state.is_running():
                    job = state.current()
                    _progress(row, job.progress if job else 0.0, "")

            row = views.row()
            row.scale_y = 1.2
            row.enabled = not state.is_running()
            row.operator("projlayer.generate_views", icon="RENDER_STILL")
        else:
            views.label(text="Save a view to batch several angles.")

        # â”€â”€ batch texturing over a collection â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        batch = views.box()
        row = batch.row(align=True)
        row.prop(conf, "target_collection", text="")
        row.prop(conf, "isolate_batch", text="", toggle=True, icon="RESTRICT_RENDER_ON")

        jobs = 0
        collection = conf.target_collection
        if collection is not None:
            meshes = sum(1 for o in collection.all_objects
                         if o.type == "MESH" and o.data is not None)
            jobs = meshes * len(conf.views)

        row = batch.row()
        row.scale_y = 1.2
        row.enabled = (not state.is_running()) and jobs > 0
        if jobs:
            row.operator("projlayer.texture_collection",
                         text=f"Texture Collection ({jobs})", icon="FILE_REFRESH")
        else:
            row.operator("projlayer.texture_collection",
                         text="Texture Collection", icon="FILE_REFRESH")

        if conf.batch_active and conf.batch_object_total:
            batch.label(
                text=f"Object {conf.batch_object_index + 1}"
                     f"/{conf.batch_object_total}",
                icon="OBJECT_DATA",
            )

        layout.separator(factor=0.6)

        # â”€â”€ the button â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        if state.is_running():
            job = state.current()
            _progress(layout, job.progress if job else 0.0,
                      conf.progress_text or "workingâ€¦")
            layout.operator("projlayer.cancel", icon="X")
        else:
            row = layout.row()
            row.scale_y = 1.7
            row.operator("projlayer.generate", icon="SHADERFX")

        # â”€â”€ paint before you generate â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        row = layout.row(align=True)
        row.scale_y = 1.2
        row.operator("projlayer.toggle_premask", icon="MOD_MASK",
                     depress=conf.premask_on)
        row.operator("projlayer.toggle_prepaint", icon="BRUSH_DATA",
                     depress=conf.prepaint_on)

        if conf.premask_on:
            box = layout.box()
            box.label(text="Paint where to regenerate", icon="MOD_MASK")
            box.label(text="Painted area shows red. Then hit Generate.")
            box.operator("projlayer.clear_pre", icon="TRASH")
        elif conf.prepaint_on:
            box = layout.box()
            box.label(text="Sketch on the mesh", icon="BRUSH_DATA")
            box.label(text="Your sketch feeds the reference image, then goes.")
            box.operator("projlayer.clear_pre", icon="TRASH")

        if conf.last_error:
            box = layout.box()
            box.alert = True
            box.label(text=conf.last_error, icon="ERROR")
        elif conf.status:
            layout.label(text=conf.status, icon="INFO")


# â”€â”€ layers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


class PL_PT_layers(bpy.types.Panel):
    bl_idname = "PL_PT_layers"
    bl_label = "Layers"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = TAB

    def draw_header(self, context):
        count = len(_stack_layers(context))
        if count:
            self.layout.label(text=f"({count})")

    def draw(self, context):
        layout = self.layout
        conf = settings(context)
        overlay.note_panel_drawn()

        # Stacks belong to the material, so with more than one slot the artist
        # needs to be able to say which material they are looking at.
        obj = getattr(context, "active_object", None)
        if obj is not None and len(getattr(obj, "material_slots", ())) > 1:
            layout.prop(conf, "material_slot", text="Material")

        row = layout.row(align=True)
        row.scale_y = 1.3
        row.prop(stack(context, create=False), "mode", expand=True)

        layers = mode_layers(context)
        active = mode_active_index(context)
        shown = _stack_layers(context)

        row = layout.row(align=True)
        row.operator("projlayer.add_layer", icon="ADD")
        row.operator("projlayer.duplicate_layer", text="", icon="DUPLICATE")
        row.operator("projlayer.move_layer", text="", icon="TRIA_UP").direction = "UP"
        row.operator("projlayer.move_layer", text="", icon="TRIA_DOWN").direction = "DOWN"
        row.operator("projlayer.flatten", text="", icon="RENDER_STILL")

        row = layout.row(align=True)
        row.label(text="Bake", icon="IMAGE_DATA")
        row.prop(conf, "flatten_resolution", text="")

        if not shown:
            box = layout.box()
            box.label(text="No layers yet.", icon="INFO")
            box.label(text="Frame your mesh and hit Generate.")
            return

        # Topmost layer first, the way a paint program shows it.
        for index in range(len(layers) - 1, -1, -1):
            layer = layers[index]
            if layer.role != "LAYER":
                continue  # working layers are not part of the stack
            _draw_layer_entry(layout, context, layer, index, index == active)


def _draw_layer_entry(layout, context, layer, index, is_active):
    box = layout.box()
    is_normal = current_mode(context) == "NORMAL"

    split = box.split(factor=0.30)

    # â”€â”€ thumbnail â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    thumb = split.column()
    preview = _thumb_icon(layer.source_image)
    if preview:
        # template_icon is the only layout widget that scales an icon preview,
        # so this is what actually makes the thumbnail bigger.
        thumb.template_icon(icon_value=preview, scale=_thumb_scale(context))
    else:
        thumb.label(text="", icon="IMAGE_DATA")

    # â”€â”€ everything else â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    body = split.column()

    header = body.row(align=True)
    header.prop(layer, "tag_color", text="")          # colour tag swatch
    op = header.operator(
        "projlayer.select_layer",
        text=layer.name,
        emboss=is_active,
        depress=is_active,
    )
    op.index = index
    header.prop(
        layer, "enabled", text="",
        icon="HIDE_OFF" if layer.enabled else "HIDE_ON",
    )
    header.prop(
        layer, "solo", text="", toggle=True,
        icon="SOLO_ON" if layer.solo else "SOLO_OFF",
    )
    header.prop(
        layer, "show_coverage", text="", toggle=True,
        icon="IMAGE_RGB_ALPHA",
    )
    cam = header.operator("projlayer.focus_projection", text="", icon="CAMERA_DATA")
    cam.index = index
    remove = header.operator("projlayer.remove_layer", text="", icon="X")
    remove.index = index

    # A paint layer works on one of the mesh's own UV maps, so let it be picked.
    if not layer.uv_attr or is_paint_layer(layer):
        body.prop(layer, "uv_map", text="UV")

    body.prop(layer, "opacity", text="Op", slider=True)

    if is_normal:
        body.prop(layer, "normal_strength", text="Str", slider=True)
        row = body.row(align=True)
        row.prop(layer, "normal_blend", text="")
        row.prop(layer, "normal_flip_y", text="FlipY", toggle=True)
    else:
        body.prop(layer, "hue", text="Hue", slider=True)
        body.prop(layer, "saturation", text="Sat", slider=True)
        body.prop(layer, "value", text="Val", slider=True)

    # â”€â”€ projection: how the mask is generated, then the two apply buttons â”€â”€â”€â”€
    row = body.row(align=True)
    row.prop(layer.projection, "feather_px", text="Sft", slider=True)
    row.operator("projlayer.feather_mask", text="", icon="MOD_SMOOTH")
    row.operator("projlayer.reproject", text="", icon="FILE_REFRESH")
    row = body.row(align=True)
    row.prop(layer.projection, "angle_cut", text="Ang", slider=True)
    row.prop(layer.projection, "max_distance", text="Max", slider=True)

    # â”€â”€ paint â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    row = body.row(align=True)
    row.scale_y = 1.2
    row.operator("projlayer.paint_color",
                 text="Normal" if is_normal else "Color", icon="BRUSH_DATA")
    row.operator("projlayer.paint_mask", text="Mask", icon="MOD_MASK")
    row.operator("projlayer.invert_mask", text="", icon="IMAGE_ALPHA")
    row.operator("projlayer.fill_mask", text="", icon="CHECKBOX_HLT").value = 1.0
    row.operator("projlayer.fill_mask", text="", icon="CHECKBOX_DEHLT").value = 0.0


# â”€â”€ registration â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


CLASSES = (PL_PT_generate, PL_PT_layers)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)
    # The viewport HUD attaches once and decides per-scene whether to paint, so
    # there is no state to keep in sync with the toggle. Never in background:
    # a GPU call there crashes Blender rather than raising.
    try:
        from . import overlay

        overlay.enable()
    except Exception:  # noqa: BLE001
        pass
    try:
        from ..ops import keymap

        keymap.register()
    except Exception:  # noqa: BLE001
        pass


def unregister():
    try:
        from ..ops import keymap

        keymap.unregister()
    except Exception:  # noqa: BLE001
        pass
    try:
        from . import overlay

        overlay.disable()
    except Exception:  # noqa: BLE001
        pass
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
