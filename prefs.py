"""Add-on preferences: where ComfyUI lives and which workflow to run."""

import os

import bpy
from bpy.props import (
    BoolProperty,
    EnumProperty,
    FloatProperty,
    IntProperty,
    StringProperty,
)

from .core import comfy, graphs

# Slot keys we know how to fill, and a short artist-facing hint for each.
SLOT_HINTS = {
    "reference": "Your render from the camera",
    "depth": "Greyscale depth guide map",
    "normal": "Surface-direction guide map",
    "mask": "Inpaint mask (only used in 'fill unpainted' mode)",
    "positive": "The prompt you type",
    "negative": "Negative prompt",
    "seed": "Seed",
    "steps": "Sampler steps",
    "cfg": "CFG scale",
    "width": "Render width",
    "height": "Render height",
    "output": "Where the result is read back from",
}


class PL_AddonPreferences(bpy.types.AddonPreferences):
    bl_idname = __package__

    server_address: StringProperty(
        name="Server",
        default=comfy.DEFAULT_BASE_URL,
        description="Address of your ComfyUI server, host:port",
    )

    workflow_mode: EnumProperty(
        name="Workflow",
        items=[
            ("DEFAULT", "Built-in img2img + ControlNet",
             "Use the workflow that ships with the addon — works out of the box"),
            ("FILE", "My own workflow file",
             "Load an API-format workflow JSON exported from ComfyUI via Save (API Format)"),
        ],
        default="DEFAULT",
    )

    workflow_path: StringProperty(
        name="Workflow File", default="", subtype="FILE_PATH",
        description="API-format workflow exported with ComfyUI's 'Save (API Format)'",
    )

    slot_overrides: BoolProperty(
        name="Override slot detection", default=False,
        description="Fill these in by hand if auto-detection picks the wrong nodes",
    )

    # One title per slot. Empty means "use whatever auto-detection found".
    slot_reference: StringProperty(name="Reference image", default="")
    slot_depth: StringProperty(name="Depth map", default="")
    slot_normal: StringProperty(name="Normal map", default="")
    slot_mask: StringProperty(name="Inpaint mask", default="")
    slot_positive: StringProperty(name="Positive prompt", default="")
    slot_negative: StringProperty(name="Negative prompt", default="")
    slot_seed: StringProperty(name="Seed", default="")
    slot_steps: StringProperty(name="Steps", default="")
    slot_cfg: StringProperty(name="CFG", default="")
    slot_width: StringProperty(name="Width", default="")
    slot_height: StringProperty(name="Height", default="")
    slot_output: StringProperty(name="Output", default="")

    timeout: FloatProperty(
        name="Timeout", default=900.0, min=30.0, max=7200.0, subtype="TIME_ABSOLUTE",
        description="Give up if a generation takes longer than this",
    )

    patch_resolution: BoolProperty(
        name="Set the workflow's width and height", default=True,
        description=(
            "Write the add-on's resolution into the workflow's width/height inputs. "
            "Turn this off if your workflow controls its own output size (for example "
            "a node with a 'dimensions' preset) and rejects a mismatched size — the "
            "returned image only has to match the render's aspect ratio, not its "
            "pixel count"
        ),
    )

    # ── UI preferences ───────────────────────────────────────────────────────
    thumbnail_size: IntProperty(
        name="Thumbnail Size", default=48, min=24, max=96, subtype="PIXEL",
    )
    confirm_delete: BoolProperty(name="Confirm before deleting a layer", default=True)

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False

        col = layout.column()
        col.prop(self, "server_address")
        row = col.row(align=True)
        row.operator("projlayer.test_connection", icon="PLUGIN")
        row.prop(self, "timeout")
        col.prop(self, "patch_resolution")

        layout.separator()
        col = layout.column()
        col.prop(self, "workflow_mode")
        if self.workflow_mode == "FILE":
            col.prop(self, "workflow_path")
        else:
            note = col.row()
            note.label(
                text="Built-in workflow ships with the addon.",
                icon="INFO",
            )

        layout.separator()
        box = layout.box()
        box.label(text="Node Slots", icon="NODETREE")
        box.label(
            text="Nodes are matched by their title in the ComfyUI workflow.",
            icon="INFO",
        )
        box.operator("projlayer.detect_slots", icon="ZOOM_SELECTED")
        box.prop(self, "slot_overrides")
        if self.slot_overrides:
            col = box.column(align=True)
            for key in graphs.SLOT_ORDER:
                col.prop(self, f"slot_{key}", text=SLOT_HINTS.get(key, key))

        layout.separator()
        box = layout.box()
        box.label(text="Interface", icon="TOOL_SETTINGS")
        box.prop(self, "tab_name")
        box.prop(self, "thumbnail_size")
        box.prop(self, "compact_rows")
        box.prop(self, "confirm_delete")


CLASSES = (PL_AddonPreferences,)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)


def prefs(context) -> PL_AddonPreferences:
    return context.preferences.addons[__package__].preferences


def resolve_workflow(context) -> dict:
    """Return the API-format graph to run, plus a human-readable source label."""
    p = prefs(context)
    if p.workflow_mode == "FILE" and p.workflow_path:
        path = bpy.path.abspath(p.workflow_path)
        if os.path.exists(path):
            import json

            with open(path, "r", encoding="utf-8") as handle:
                graph = json.load(handle)
            # ComfyUI's "Save" writes {"prompt": {...}}; accept both shapes.
            if isinstance(graph, dict) and isinstance(graph.get("prompt"), dict):
                graph = graph["prompt"]
            return graph, os.path.basename(path)
        raise comfy.ComfyError(
            f"Workflow file not found: {path}"
        )
    return graphs.DEFAULT_IMG2IMG, "built-in workflow"


def slot_titles(context) -> dict:
    """Slot key -> node title override (only the ones the user filled in)."""
    p = prefs(context)
    if not p.slot_overrides:
        return {}
    out = {}
    for key in graphs.SLOT_ORDER:
        value = getattr(p, f"slot_{key}", "")
        if value:
            out[key] = value
    return out
