"""Default keyboard shortcuts.

Registered on the add-on's own key configuration, so nothing the user has bound
is ever overwritten, and the bindings appear under Preferences > Keymap where
they can be rebound or removed like any other.
"""

import bpy

KEYMAP_NAME = "3D View"

#: ``(operator, key, ctrl, shift, alt, props)``. ``props`` may be omitted.
BINDINGS = (
    ("projlayer.generate", "G", True, True, False),
    ("projlayer.toggle_premask", "M", True, True, False),
    ("projlayer.toggle_prepaint", "P", True, True, False),
    ("projlayer.paint_mask", "B", True, True, False),
    ("projlayer.paint_color", "C", True, True, False),
    ("projlayer.add_layer", "L", True, True, False),
    ("projlayer.flatten", "F", True, True, False),
    # Shift+X samples only the ACTIVE image, and a fresh paint layer or a
    # pre-paint sketch starts fully transparent - so it returns black. That is
    # Blender being literal, not a bug. This binding samples the merged result
    # instead, which is the colour actually on screen.
    ("paint.sample_color", "X", True, True, False, {"merged": True}),
)

_items = []


def _combo(key, ctrl, shift, alt):
    parts = []
    if ctrl:
        parts.append("Ctrl")
    if shift:
        parts.append("Shift")
    if alt:
        parts.append("Alt")
    parts.append(key)
    return "+".join(parts)


def binding_text():
    """``(combo, label)`` pairs, for showing the shortcuts in the UI."""
    return [
        (_combo(binding[1], binding[2], binding[3], binding[4]),
         binding[0].split(".")[-1].replace("_", " ").title())
        for binding in BINDINGS
    ]


def register():
    window_manager = getattr(bpy.context, "window_manager", None)
    keyconfigs = getattr(window_manager, "keyconfigs", None)
    keyconfig = getattr(keyconfigs, "addon", None) if keyconfigs else None
    if keyconfig is None:
        return  # background, or no window manager: nothing to bind
    try:
        keymap = keyconfig.keymaps.get(KEYMAP_NAME)
        if keymap is None:
            keymap = keyconfig.keymaps.new(name=KEYMAP_NAME, space_type="VIEW_3D")
    except Exception:  # noqa: BLE001
        return

    for binding in BINDINGS:
        idname, key, ctrl, shift, alt = binding[:5]
        settings = binding[5] if len(binding) > 5 else {}
        try:
            item = keymap.keymap_items.new(
                idname, type=key, value="PRESS",
                ctrl=ctrl, shift=shift, alt=alt,
            )
            for name, value in settings.items():
                try:
                    setattr(item.properties, name, value)
                except Exception:  # noqa: BLE001 - operator may not expose it
                    pass
            _items.append((keymap, item))
        except Exception:  # noqa: BLE001
            pass


def unregister():
    for keymap, item in _items:
        try:
            keymap.keymap_items.remove(item)
        except Exception:  # noqa: BLE001
            pass
    _items.clear()
