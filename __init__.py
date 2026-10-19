"""Projection Layers â€” an artist-facing AI projection texturing addon.

Workflow: frame your mesh with the camera, type a prompt, hit Generate. The
viewport render goes to ComfyUI along with depth and normal guide maps, the
result comes back, and it lands on the mesh as a layer with its own paintable
mask. Repeat from any angle and stack the layers.

Architecture note: every layer owns a UV map that *is* the camera projection
(a snapshot of a UV_PROJECT modifier). Because that UV map is a real UV map,
Blender's native texture-paint tools map strokes into the right place, so masks
are painted with the full brush engine rather than a custom brush.

See projection-layers-plan.md for the full design.
"""

bl_info = {
    "name": "Projection Layers",
    "author": "ProjLayers",
    "version": (0, 1, 0),
    "blender": (4, 1, 0),
    "location": "View3D > Sidebar > Proj Layers",
    "description": "Generate textures with ComfyUI and project them as paintable layers",
    "category": "Paint",
}

import bpy

from . import prefs, props
from .ops import operators
from .ui import panel

_MODULES = (prefs, props, operators, panel)


def register():
    for module in _MODULES:
        module.register()


def unregister():
    for module in reversed(_MODULES):
        module.unregister()
