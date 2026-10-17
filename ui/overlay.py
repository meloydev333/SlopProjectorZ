"""Viewport HUD: the output frame guide and generation status.

Both are flat screen-space elements drawn in region pixels (``POST_PIXEL``).
The frame is *not* a projection of the camera frustum: projecting world-space
corners made it shrink and grow as you zoomed, which is wrong for a frame guide.
It is simply a centred rectangle whose aspect matches the render resolution, so
it marks the shape of the output and holds still while you navigate.

The HUD only paints while this add-on's sidebar tab is open. Blender does not
expose the active sidebar category through RNA, so the panel tells us instead:
:func:`note_panel_drawn` is called from the panel's ``draw`` and the HUD goes
quiet if that has not happened for a while.

A note on verification: calling into ``gpu`` from ``--background`` crashes
Blender outright (EXCEPTION_ACCESS_VIOLATION — there is no GPU context), so the
drawing cannot be exercised by the test suite. Everything computable without a
GPU lives in plain functions — :func:`render_aspect`, :func:`safe_frame_rect`,
:func:`hud_rows` — and those *are* tested.
"""

import time

import bpy

from ..core import state
from ..props import settings

FRAME_COLOR = (1.0, 0.42, 0.05, 1.0)
PANEL_BG = (0.04, 0.04, 0.04, 0.70)
TEXT_COLOR = (0.90, 0.90, 0.90, 1.0)
ACCENT_COLOR = (1.0, 0.55, 0.15, 1.0)
ERROR_COLOR = (1.0, 0.34, 0.34, 1.0)

#: How long after its last draw the panel still counts as open. Generous on
#: purpose: the UI region is not redrawn on every viewport update, so a tight
#: window made the HUD flicker out mid-interaction.
#: Sidebar tab this add-on's panels live under, so the panel and the HUD agree.
CATEGORY = "Proj Layers"

#: Fallback only: how long after its last draw the panel counts as open. Used
#: when the sidebar's active tab cannot be read.
PANEL_GRACE = 2.5

#: Fraction of the viewport the guide occupies, leaving room to see around it.
FRAME_MARGIN = 0.88

_handle = None
_last_panel_draw = 0.0


# ── panel visibility ────────────────────────────────────────────────────────


def note_panel_drawn():
    """Called from the sidebar panel's draw(), so the HUD can tell whether this
    add-on's tab is the one showing."""
    global _last_panel_draw
    _last_panel_draw = time.monotonic()


def panel_visible():
    return (time.monotonic() - _last_panel_draw) < PANEL_GRACE


def active_category(context):
    """The sidebar tab currently showing, or '' if it cannot be read."""
    area = getattr(context, "area", None)
    if area is None:
        return ""
    for region in getattr(area, "regions", ()):
        if getattr(region, "type", "") != "UI":
            continue
        try:
            return region.active_panel_category or ""
        except Exception:  # noqa: BLE001
            return ""
    return ""


def panel_is_active(context):
    """Is this add-on's tab the one showing?

    Reads ``Region.active_panel_category``, the sidebar's real active tab, so
    switching away hides the HUD at once. The old timestamp guess flickered
    because the UI region is not redrawn on every viewport update.
    """
    category = active_category(context)
    if category:
        return category == CATEGORY
    return panel_visible()


# ── geometry and text: pure, and covered by the test suite ───────────────────


def render_aspect(scene):
    """Width over height of the image that will be rendered."""
    render = scene.render
    width = float(render.resolution_x) * float(render.pixel_aspect_x or 1.0)
    height = float(render.resolution_y) * float(render.pixel_aspect_y or 1.0)
    if height <= 0.0:
        return 1.0
    return width / height


def safe_frame_rect(region_width, region_height, aspect, margin=FRAME_MARGIN):
    """A centred rectangle of ``aspect``, fitted into the region.

    Returns ``(left, bottom, right, top)`` in region pixels, or None if the
    inputs are degenerate. Pure arithmetic, so it holds its size no matter how
    the viewport is zoomed or orbited.
    """
    try:
        region_width = float(region_width)
        region_height = float(region_height)
        aspect = float(aspect)
    except (TypeError, ValueError):
        return None
    if region_width <= 1.0 or region_height <= 1.0 or aspect <= 0.0:
        return None

    inner_width = region_width * float(margin)
    inner_height = region_height * float(margin)
    if inner_width / inner_height > aspect:
        height = inner_height
        width = height * aspect
    else:
        width = inner_width
        height = width / aspect

    left = (region_width - width) * 0.5
    bottom = (region_height - height) * 0.5
    return (left, bottom, left + width, bottom + height)


def hud_rows(context):
    """``(text, colour)`` rows for the status panel. No GPU involved.

    Shown while a generation is running, while a bake is stepping, when the
    pre-mask is armed, or when something failed. An idle viewport stays clean.
    """
    conf = settings(context)
    rows = []

    if state.is_running():
        job = state.current()
        stage = conf.status or (job.status if job else "") or "working…"
        rows.append((stage, ACCENT_COLOR))
        if job is not None:
            # Elapsed time is the point: a remote backend can sit silent for a
            # minute and you need to know it is alive rather than hung.
            rows.append((f"{job.elapsed:6.1f}s    {job.progress * 100:3.0f}%",
                         TEXT_COLOR))
    elif conf.busy_text:
        rows.append((conf.busy_text, ACCENT_COLOR))

    if conf.premask_on:
        rows.append(("PRE-MASK ON — painting the next generation", ACCENT_COLOR))

    if conf.prepaint_on:
        rows.append(("PRE-PAINT ON — your sketch is the reference", ACCENT_COLOR))

    if conf.last_error:
        rows.append((str(conf.last_error)[:72], ERROR_COLOR))

    return rows


def pulse_alpha(period=1.1, low=0.35, high=1.0):
    """Breathing alpha for the liveness dot, so a stalled job is obvious."""
    phase = (time.monotonic() % period) / period
    return low + (high - low) * abs(1.0 - 2.0 * phase)


# ── drawing: thin, guarded, and untestable headlessly ───────────────────────


def _draw():
    try:
        context = bpy.context
        region = getattr(context, "region", None)
        scene = getattr(context, "scene", None)
        if region is None or scene is None:
            return
        space = getattr(context, "space_data", None)
        if space is None or getattr(space, "type", "") != "VIEW_3D":
            return
        if getattr(space, "show_region_ui", True) is False:
            return
        # Only while this add-on's tab is the one on screen.
        if not panel_is_active(context):
            return

        conf = settings(context)
        import blf
        import gpu
        from gpu_extras.batch import batch_for_shader

        try:
            scale = max(0.5, float(context.preferences.system.ui_scale or 1.0))
        except Exception:  # noqa: BLE001
            scale = 1.0

        shader = gpu.shader.from_builtin("UNIFORM_COLOR")
        gpu.state.blend_set("ALPHA")

        # ── output frame guide ──────────────────────────────────────────────
        if conf.show_frame:
            rect = safe_frame_rect(region.width, region.height,
                                   render_aspect(scene))
            if rect is not None:
                left, bottom, right, top = rect
                tick = 18.0 * scale
                lines = [
                    # full outline
                    (left, bottom), (right, bottom),
                    (right, bottom), (right, top),
                    (right, top), (left, top),
                    (left, top), (left, bottom),
                    # heavier corner marks, so it reads as a guide rather than
                    # a selection box
                    (left, bottom), (left + tick, bottom),
                    (left, bottom), (left, bottom + tick),
                    (right, top), (right - tick, top),
                    (right, top), (right, top - tick),
                ]
                gpu.state.line_width_set(1.0)
                shader.uniform_float("color", FRAME_COLOR)
                batch_for_shader(shader, "LINES", {"pos": lines}).draw(shader)

        # ── status, bottom centre ───────────────────────────────────────────
        rows = hud_rows(context)
        if rows:
            font = 0
            size = 12.0 * scale
            try:
                blf.size(font, size)
            except TypeError:  # older Blender signature
                blf.size(font, size, 72)

            pad = 9.0 * scale
            line_height = size * 1.55
            width = max(blf.dimensions(font, text)[0] for text, _ in rows) + pad * 2
            height = line_height * len(rows) + pad

            left = (region.width - width) * 0.5
            right = left + width
            bottom = pad * 2.0
            top = bottom + height

            shader.uniform_float("color", PANEL_BG)
            batch_for_shader(shader, "TRI_FAN", {"pos": [
                (left, bottom), (right, bottom), (right, top), (left, top),
            ]}).draw(shader)

            for index, (text, colour) in enumerate(rows):
                baseline = top - pad - line_height * index - size
                blf.color(font, *colour)
                blf.position(font, left + pad, baseline, 0)
                blf.draw(font, text)

            if state.is_running():
                radius = 3.0 * scale
                centre_x = right + pad * 1.4
                centre_y = top - pad - size * 0.5
                shader.uniform_float("color", (1.0, 0.55, 0.15, pulse_alpha()))
                batch_for_shader(shader, "TRI_FAN", {"pos": [
                    (centre_x - radius, centre_y),
                    (centre_x, centre_y - radius),
                    (centre_x + radius, centre_y),
                    (centre_x, centre_y + radius),
                ]}).draw(shader)

        gpu.state.blend_set("NONE")
    except Exception:  # noqa: BLE001 — a HUD must never break the viewport
        pass


# ── registration ─────────────────────────────────────────────────────────────


def is_enabled():
    return _handle is not None


def enable():
    """Attach the draw handler. Never in background: there is no GPU context and
    the call crashes Blender rather than raising."""
    global _handle
    if _handle is not None or bpy.app.background:
        return
    try:
        _handle = bpy.types.SpaceView3D.draw_handler_add(
            _draw, (), "WINDOW", "POST_PIXEL")
    except Exception:  # noqa: BLE001
        _handle = None


def disable():
    global _handle
    if _handle is None:
        return
    try:
        bpy.types.SpaceView3D.draw_handler_remove(_handle, "WINDOW")
    except Exception:  # noqa: BLE001
        pass
    _handle = None
