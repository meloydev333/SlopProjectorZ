"""All operators."""

import os

import bpy
import numpy as np

from .. import prefs as prefs_module
from ..core import comfy, graphs, nodes, pipeline, render, state
from ..props import (
    active_layer,
    current_mode,
    ensure_prompt_slots,
    find_role_layer,
    is_paint_layer,
    mode_active_index,
    mode_layers,
    resolution,
    role_layers,
    set_mode_active_index,
    set_current_mode,
    set_pre_modes,
    settings,
    stack,
    visible_layers,
)


def _obj(context):
    obj = context.active_object
    if obj is None or obj.type != "MESH":
        raise RuntimeError("Select a mesh object first.")
    return obj


def _fail(operator, context, message):
    settings(context).last_error = message
    operator.report({"ERROR"}, message)
    return {"CANCELLED"}


# Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬ generate Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬


def _finish(payload, job, result):
    """Runs on the main thread once the worker thread is done."""
    context = bpy.context
    conf = settings(context)

    if job.cancelled:
        conf.status = "Cancelled."
        conf.queue_active = False
        conf.batch_active = False
        return
    if job.error:
        conf.last_error = job.error
        conf.status = "Failed."
        conf.queue_active = False
        conf.batch_active = False
        return

    try:
        layer = pipeline.apply(context, payload, result)
    except Exception as exc:  # noqa: BLE001 Ä‚ËĂ˘â€šÂ¬Ă˘â‚¬ĹĄ surfaced in the sidebar
        conf.last_error = str(exc)
        conf.status = "Failed to apply the result."
        conf.queue_active = False
        conf.batch_active = False
        return

    conf.last_error = ""
    conf.status = (
        f"{layer.name} Ä‚ËĂ˘â€šÂ¬Ă˘â‚¬ĹĄ {layer.projection.res_x}Ă„â€šĂ˘â‚¬â€ť{layer.projection.res_y} "
        f"in {layer.bake_seconds:.1f}s"
    )

    # Multi-view chaining: start the next camera the moment this one lands, so
    # the queue runs continuously without ever blocking the main thread.
    if conf.queue_active:
        # The view that just landed is the one before the index we advanced to.
        # Modulo, because a batch wraps the view list once per object.
        finished = (conf.queue_index - 1) % max(1, len(conf.views))
        if 0 <= finished < len(conf.views):
            conf.views[finished].state = "DONE"
        _advance_queue(context)


class PL_OT_generate(bpy.types.Operator):
    bl_idname = "projlayer.generate"
    bl_label = "Generate"
    bl_description = "Render the current view, send it to ComfyUI, and project the result as a new layer"
    bl_options = {"REGISTER"}

    def execute(self, context):
        conf = settings(context)

        if state.is_running():
            self.report({"WARNING"}, "A generation is already running.")
            return {"CANCELLED"}

        conf.last_error = ""
        # A single Generate always leaves any multi-view run behind, so a stale
        # queue can never chain off it.
        conf.queue_active = False
        conf.batch_active = False
        conf.status = "Preparing"

        payload = pipeline.Payload()
        try:
            pipeline.prepare(context, payload)
        except comfy.ComfyError as exc:
            return _fail(self, context, str(exc))
        except RuntimeError as exc:
            return _fail(self, context, str(exc))
        except Exception as exc:  # noqa: BLE001
            import traceback

            traceback.print_exc()
            return _fail(self, context, f"Could not prepare the generation: {exc}")

        try:
            bpy.ops.ed.undo_push(message="Generate layer")
        except Exception:
            pass  # undo is unavailable in background mode

        state.start(
            context.scene.name,
            "Generate",
            lambda job: pipeline.run(payload, job),
            lambda job, result: _finish(payload, job, result),
        )
        return {"FINISHED"}


def _batch_objects(conf):
    """The object names queued for the running batch, or [] for a view run."""
    if not conf.batch_active:
        return []
    return [name for name in conf.batch_objects.split("|") if name]


def _activate_object(context, obj):
    """Make ``obj`` the active object, which is what prepare() reads."""
    view_layer = getattr(context, "view_layer", None)
    objects = getattr(view_layer, "objects", None) if view_layer is not None else None
    if objects is None:
        return
    if obj.name not in objects:
        raise RuntimeError(
            f"'{obj.name}' is not in the active view layer (its collection may "
            "be excluded), so it cannot be textured."
        )
    try:
        for candidate in objects:
            candidate.select_set(candidate is obj)
    except RuntimeError:
        pass
    view_layer.objects.active = obj


def _reset_view_states(conf):
    for view in conf.views:
        view.state = "PENDING"


def _queue_slots(index, view_count, object_count):
    """(object_index, view_index) for a flat queue position.

    A batch flattens the object x view cross-product into one index: object 1
    takes the first ``view_count`` slots, object 2 the next, and so on. Kept
    separate from _advance_queue so the mapping can be tested without a backend.
    """
    if view_count <= 0:
        return 0, 0
    if object_count <= 0:
        return 0, index
    return index // view_count, index % view_count


def _batch_candidates(collection):
    """Split a collection into ``(usable names, skipped descriptions)``.

    Scanned up front so unusable objects are reported together instead of
    aborting the batch halfway through and leaving it half textured.
    """
    usable = []
    skipped = []
    for obj in collection.all_objects:
        if obj.type != "MESH" or obj.data is None:
            skipped.append(obj.name)
            continue
        if not pipeline.mesh_uv_name(obj):
            skipped.append(f"{obj.name} (no UVs)")
            continue
        if not obj.material_slots:
            skipped.append(f"{obj.name} (no material)")
            continue
        usable.append(obj.name)
    return usable, skipped


def _advance_queue(context):
    """Start the next job in the queue, or finish if there is none left.

    The queue is one flat list of jobs. A plain view run has one job per view; a
    batch flattens the object x view cross-product into the same index, so both
    share this one chain:

        object_index = queue_index // len(views)
        view_index   = queue_index %  len(views)

    Called from the job's completion callback, which ``state`` invokes on the
    main thread with the previous job already cleared - so starting the next one
    here is safe and keeps the sequence going without a modal loop.

    Returns True when another generation was started.
    """
    conf = settings(context)
    views = len(conf.views)
    objects = _batch_objects(conf)
    is_batch = bool(objects)
    total = (len(objects) if is_batch else 1) * views

    if views == 0 or total == 0:
        conf.queue_active = False
        conf.batch_active = False
        conf.queue_index = 0
        return False

    while conf.queue_index < total:
        index = conf.queue_index
        conf.queue_index += 1

        object_index, view_index = _queue_slots(
            index, views, len(objects) if is_batch else 0
        )
        view = conf.views[view_index]
        camera = view.camera
        if camera is None or camera.type != "CAMERA":
            continue  # the camera was deleted since the view was saved

        obj = None
        if is_batch:
            obj = bpy.data.objects.get(objects[object_index])
            if obj is None:
                continue  # deleted mid-batch
            # A fresh object starts the view list over, so the checkmarks in the
            # panel describe the object being worked on, not the last one.
            if view_index == 0:
                _reset_view_states(conf)
            conf.batch_object_index = object_index

        view.state = "ACTIVE"
        set_current_mode(context, view.mode)
        if is_batch:
            conf.status = (
                f"Object {object_index + 1}/{len(objects)} | "
                f"view {view_index + 1}/{views}"
            )
        else:
            conf.status = f"View {view_index + 1} of {views}..."

        try:
            if obj is not None:
                _activate_object(context, obj)
        except RuntimeError as exc:
            view.state = "FAILED"
            conf.last_error = f"{conf.status}: {exc}"
            conf.status = "Stopped."
            conf.queue_active = False
            conf.batch_active = False
            return False

        context.scene.camera = camera

        payload = pipeline.Payload()
        payload.snap_camera = False  # the view's camera is already framed
        payload.isolate = is_batch and bool(conf.isolate_batch)
        try:
            pipeline.prepare(context, payload)
        except Exception as exc:  # noqa: BLE001 - reported, queue stops
            view.state = "FAILED"
            conf.last_error = f"{conf.status}: {exc}" if is_batch else str(exc)
            conf.status = "Stopped."
            conf.queue_active = False
            conf.batch_active = False
            return False

        state.start(
            context.scene.name,
            "Generate",
            lambda job, p=payload: pipeline.run(p, job),
            lambda job, result, p=payload: _finish(p, job, result),
        )
        return True

    conf.queue_active = False
    conf.queue_index = 0
    if is_batch:
        conf.batch_active = False
        skipped = conf.batch_skipped
        conf.status = "Batch finished."
        if skipped:
            conf.status = f"Batch finished. Skipped: {skipped}"
    else:
        conf.status = "All views generated."
    return False


class PL_OT_generate_views(bpy.types.Operator):
    bl_idname = "projlayer.generate_views"
    bl_label = "Generate All Views"
    bl_description = (
        "Generate a layer for every saved view, one after another. Each view "
        "stops the run if it fails"
    )
    bl_options = {"REGISTER"}

    def execute(self, context):
        conf = settings(context)

        if state.is_running():
            self.report({"WARNING"}, "A generation is already running.")
            return {"CANCELLED"}

        usable = [v for v in conf.views
                  if v.camera is not None and v.camera.type == "CAMERA"]
        if not usable:
            return _fail(self, context, "Add at least one view first.")

        conf.last_error = ""
        conf.queue_index = 0
        conf.queue_active = True
        # A view run must not inherit a queued batch's object list.
        conf.batch_active = False

        if not _advance_queue(context):
            # _advance_queue already recorded the reason.
            message = conf.last_error or "Could not start the first view."
            conf.queue_active = False
            return _fail(self, context, message)
        return {"FINISHED"}


VIEWS_COLLECTION = "PL_Views"


class PL_OT_texture_collection(bpy.types.Operator):
    bl_idname = "projlayer.texture_collection"
    bl_label = "Texture Collection"
    bl_description = (
        "Generate every saved view for every mesh in the target collection, one "
        "object at a time. Can be a lot of paid generations"
    )
    bl_options = {"REGISTER"}

    @classmethod
    def poll(cls, context):
        return not state.is_running()

    def invoke(self, context, event):
        # N objects x M views can be expensive against a paid backend, so make
        # the artist confirm rather than discovering it on the invoice.
        return context.window_manager.invoke_confirm(self, event)

    def execute(self, context):
        conf = settings(context)

        if state.is_running():
            return _fail(self, context, "A generation is already running.")
        if not conf.views:
            return _fail(self, context, "Add at least one view first.")
        collection = conf.target_collection
        if collection is None:
            return _fail(self, context, "Pick a collection to texture.")
        if not conf.prompt.strip():
            return _fail(self, context, "Type a prompt before generating.")

        # Scan up front so unusable objects are reported together instead of
        # aborting the batch halfway through and leaving it half textured.
        usable, skipped = _batch_candidates(collection)

        if not usable:
            return _fail(
                self, context,
                "Nothing in that collection can be textured - each object needs "
                f"a mesh, UVs and a material. Skipped: {', '.join(skipped)}",
            )

        conf.last_error = ""
        conf.batch_objects = "|".join(usable)
        conf.batch_skipped = ", ".join(skipped)
        conf.batch_object_total = len(usable)
        conf.batch_object_index = 0
        conf.batch_active = True
        conf.queue_index = 0
        conf.queue_active = True
        _reset_view_states(conf)

        if not _advance_queue(context):
            message = conf.last_error or "Could not start the batch."
            conf.queue_active = False
            conf.batch_active = False
            return _fail(self, context, message)
        return {"FINISHED"}


def _views_collection(scene):
    """The collection the per-view cameras live in, so the outliner stays tidy."""
    collection = bpy.data.collections.get(VIEWS_COLLECTION)
    if collection is None:
        collection = bpy.data.collections.new(VIEWS_COLLECTION)
        scene.collection.children.link(collection)
    return collection


class PL_OT_view_add(bpy.types.Operator):
    bl_idname = "projlayer.view_add"
    bl_label = "Add View"
    bl_description = "Save the current view as its own camera, to generate from later"

    def execute(self, context):
        conf = settings(context)
        scene = context.scene
        try:
            source = render.ensure_camera(context)
        except RuntimeError as exc:
            return _fail(self, context, str(exc))

        # Every view gets its own camera. Sharing one was the bug: with a single
        # camera in the scene, all the views were the same angle, so a batch run
        # generated the same shot N times.
        camera = source.copy()
        camera.data = source.data.copy()
        camera.name = f"PL_View_{len(conf.views) + 1:02d}"
        camera.data.name = f"{camera.name}_data"
        _views_collection(scene).objects.link(camera)

        # Frame it the way Generate would, or the saved view would not match
        # what the artist is looking at.
        if conf.auto_snap_camera:
            try:
                render.snap_to_view(context, camera)
            except RuntimeError:
                pass

        scene.camera = camera
        view = conf.views.add()
        view.camera = camera
        view.mode = current_mode(context)
        view.state = "PENDING"
        conf.view_index = len(conf.views) - 1
        conf.status = f"View {len(conf.views)} saved as {camera.name}."
        return {"FINISHED"}


class PL_OT_view_remove(bpy.types.Operator):
    bl_idname = "projlayer.view_remove"
    bl_label = "Remove View"
    bl_description = "Remove the selected view from the list"

    def execute(self, context):
        conf = settings(context)
        if not conf.views:
            return _fail(self, context, "There are no views to remove.")
        index = max(0, min(conf.view_index, len(conf.views) - 1))
        camera = conf.views[index].camera
        conf.views.remove(index)
        conf.view_index = max(0, min(index, len(conf.views) - 1))

        # Drop the camera this view owned, unless something still points at it.
        if camera is not None and not any(v.camera == camera for v in conf.views):
            try:
                bpy.data.objects.remove(camera, do_unlink=True)
            except (ReferenceError, RuntimeError):
                pass
        return {"FINISHED"}


class PL_OT_view_select(bpy.types.Operator):
    bl_idname = "projlayer.view_select"
    bl_label = "Select View"
    bl_description = "Make this view the active camera"

    index: bpy.props.IntProperty(default=0, min=0)

    def execute(self, context):
        conf = settings(context)
        if not conf.views:
            return {"CANCELLED"}
        index = max(0, min(self.index, len(conf.views) - 1))
        conf.view_index = index
        view = conf.views[index]
        if view.camera is not None and view.camera.type == "CAMERA":
            context.scene.camera = view.camera
            # Jump the viewport through it, so selecting a view shows exactly
            # what that view will capture.
            _look_through(context, view.camera)
        return {"FINISHED"}


class PL_OT_cancel(bpy.types.Operator):
    bl_idname = "projlayer.cancel"
    bl_label = "Cancel"
    bl_description = "Stop the running generation"

    def execute(self, context):
        state.cancel_current()
        conf = settings(context)
        # Cancel stops the whole batch, not just the job in flight: the chain
        # only advances from _finish, which will not run again once we clear this.
        conf.queue_active = False
        conf.batch_active = False
        conf.status = "Cancelling"
        return {"FINISHED"}


class PL_OT_test_connection(bpy.types.Operator):
    bl_idname = "projlayer.test_connection"
    bl_label = "Test Connection"
    bl_description = "Check that ComfyUI is reachable"

    def execute(self, context):
        prefs = prefs_module.prefs(context)
        ok, message = comfy.ping(prefs.server_address)
        settings(context).last_error = "" if ok else message
        settings(context).status = message
        self.report({"INFO"} if ok else {"ERROR"}, message)
        return {"FINISHED"}


class PL_OT_detect_slots(bpy.types.Operator):
    bl_idname = "projlayer.detect_slots"
    bl_label = "Detect Slots"
    bl_description = "Read the workflow and report which nodes each input will use"

    def execute(self, context):
        prefs = prefs_module.prefs(context)
        try:
            graph, label = prefs_module.resolve_workflow(context)
        except Exception as exc:  # noqa: BLE001
            return _fail(self, context, str(exc))

        slots = graphs.detect_slots(graph)
        found = ", ".join(sorted(slots)) or "nothing"
        settings(context).status = f"{label}: found {found}"
        self.report({"INFO"}, f"Detected {len(slots)} slots in {label}: {found}")
        return {"FINISHED"}


# Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬ camera Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬


class PL_OT_snap_camera(bpy.types.Operator):
    bl_idname = "projlayer.snap_camera"
    bl_label = "Snap to View"
    bl_description = "Move the projector camera to match what you are looking at"

    def execute(self, context):
        camera = render.ensure_camera(context)
        try:
            render.snap_to_view(context, camera)
        except RuntimeError as exc:
            return _fail(self, context, str(exc))
        settings(context).status = "Camera snapped to the viewport."
        return {"FINISHED"}


class PL_OT_fit_camera(bpy.types.Operator):
    bl_idname = "projlayer.fit_camera"
    bl_label = "Frame Selected"
    bl_description = "Point the projector camera at the selected object"

    def execute(self, context):
        camera = render.ensure_camera(context)
        obj = context.active_object
        if obj is None:
            return _fail(self, context, "Select something to frame.")

        import mathutils

        points = [obj.matrix_world @ mathutils.Vector(corner) for corner in obj.bound_box]
        centre = sum(points, mathutils.Vector()) / 8.0
        radius = max((p - centre).length for p in points) or 1.0

        direction = camera.matrix_world.to_quaternion() @ mathutils.Vector((0.0, 0.0, -1.0))
        distance = radius * 2.6
        camera.location = centre - direction * distance
        context.scene.camera = camera
        settings(context).status = "Camera framed the selection."
        return {"FINISHED"}


# Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬ layers Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬


def _activate_layer(context, layer) -> bool:
    """Select ``layer`` whichever stack it lives in."""
    active = stack(context)
    ptr = layer.as_pointer()
    for mode, coll in (("COLOR", active.color_layers), ("NORMAL", active.normal_layers)):
        for index, candidate in enumerate(coll):
            if candidate.as_pointer() == ptr:
                active.mode = mode
                if mode == "NORMAL":
                    active.active_normal = index
                else:
                    active.active_color = index
                return True
    return False


def _prepare_paint_uv(obj, layer, for_mask=False):
    """Which UV map strokes should go through, or None if nowhere to paint.

    Everything paintable now lives in the object's own UV space: masks in
    ``mask_uv``, sketches and paint layers in ``uv_map``. Only a projected
    layer's *colour* is screen-space, and that is not painted.
    """
    wanted = None
    if for_mask and layer.mask_uv:
        wanted = layer.mask_uv
    elif is_paint_layer(layer):
        wanted = layer.uv_map
    if wanted and wanted in [uv.name for uv in obj.data.uv_layers]:
        return wanted
    return None


def _new_image(name, color, width, height, non_color=False, alpha=False):
    image = bpy.data.images.new(name, width, height, alpha=alpha)
    image.generated_color = color
    if non_color:
        try:
            image.colorspace_settings.name = "Non-Color"
        except TypeError:
            pass
    image.pack()
    # Generated images have no other owner. Without a fake user Blender can
    # treat them as orphans, and a saved file comes back with the layer's mask
    # pointer empty - which reads as "this layer has no mask".
    image.use_fake_user = True
    return image


_PREPAINT_COLOR = (0.62, 0.62, 0.62, 1.0)


class PL_OT_add_layer(bpy.types.Operator):
    bl_idname = "projlayer.add_layer"
    bl_label = "New Paint Layer"
    bl_description = (
        "Add a layer you can paint on, using one of the mesh's own UV maps "
        "(the active one by default)"
    )

    def execute(self, context):
        try:
            obj = _obj(context)
        except RuntimeError as exc:
            return _fail(self, context, str(exc))

        uv = obj.data.uv_layers.active
        if uv is None and len(obj.data.uv_layers):
            uv = obj.data.uv_layers[0]
        if uv is None:
            return _fail(self, context, "This mesh has no UV map to paint on.")

        conf = settings(context)
        layers = mode_layers(context)
        is_normal = current_mode(context) == "NORMAL"
        width, height = resolution(context)
        name = f"PL_paint_{len(layers):02d}"

        layer = layers.add()
        layer.name = f"Paint ({uv.name})"
        layer.role = "LAYER"
        layer.uv_map = uv.name
        layer.mask_uv = uv.name
        # Transparent to start with, so an untouched paint layer contributes
        # nothing and the layers underneath stay visible.
        layer.source_image = _new_image(
            name, (0.0, 0.0, 0.0, 0.0) if not is_normal else (0.5, 0.5, 1.0, 0.0),
            width, height, non_color=is_normal, alpha=True,
        )
        layer.mask_image = _new_image(f"{name}_mask", (1.0, 1.0, 1.0, 1.0), width, height)
        layer.projection.res_x = width
        layer.projection.res_y = height
        set_mode_active_index(context, len(layers) - 1)

        pipeline.rebuild(context)

        # Drop straight into painting on it: the layer is brand new and empty,
        # so making the artist press Color next would just be a wasted step.
        try:
            bpy.ops.projlayer.paint_color()
        except RuntimeError as exc:
            conf.status = f"Added {layer.name} (could not enter paint mode: {exc})"
        else:
            _setup_brush(context)
            conf.status = f"Painting {layer.name}"
        return {"FINISHED"}


class PL_OT_select_prompt(bpy.types.Operator):
    bl_idname = "projlayer.select_prompt"
    bl_label = "Prompt Tab"
    bl_description = "Switch to a saved prompt. The current one is stored first"

    index: bpy.props.IntProperty(default=0, min=0)

    def execute(self, context):
        conf = settings(context)
        ensure_prompt_slots(conf)
        count = len(conf.prompt_slots)
        current = max(0, min(conf.active_prompt, count - 1))

        # Save what is on screen, then load the tab being switched to.
        conf.prompt_slots[current].prompt = conf.prompt
        conf.prompt_slots[current].negative = conf.negative_prompt

        target = max(0, min(self.index, count - 1))
        conf.active_prompt = target
        conf.prompt = conf.prompt_slots[target].prompt
        conf.negative_prompt = conf.prompt_slots[target].negative
        return {"FINISHED"}


def _bake_progress(context, text):
    """Report flatten progress and force a repaint.

    A flatten runs on the main thread, so without pumping a redraw the artist
    would see nothing at all until it finished.
    """
    conf = settings(context)
    conf.busy_text = text
    conf.status = text
    try:
        bpy.ops.wm.redraw_timer(type="DRAW_WIN_SWAP", iterations=1)
    except Exception:  # noqa: BLE001 - background or no window
        pass


def _leave_paint_mode(context):
    """Return to Object Mode.

    Switching a pre-mode off should also take the brush away: staying in texture
    paint leaves the mesh armed for strokes that now go somewhere the artist did
    not intend.
    """
    obj = getattr(context, "active_object", None)
    if obj is None or getattr(obj, "mode", "OBJECT") == "OBJECT":
        return
    try:
        bpy.ops.object.mode_set(mode="OBJECT")
    except RuntimeError:
        pass


BRUSH_PRESETS = ("Paint Soft", "Soft", "Paint Hard")


def _set_colour(brush, attr, values):
    """Assign an RGB or RGBA brush colour, whatever length the property is.

    ``Brush.color`` is three components on some brush types and four on others,
    and assigning a 3-tuple to a 4-component property raises TypeError. An
    earlier version swallowed that, so the colour silently never changed and the
    brush kept whatever the artist last used.
    """
    try:
        length = len(getattr(brush, attr))
    except (AttributeError, TypeError):
        return False
    padded = tuple(float(value) for value in values) + (1.0,) * 4
    try:
        setattr(brush, attr, padded[:length])
        return True
    except (AttributeError, TypeError):
        return False


def _brush_candidates():
    """Image-paint brushes available in this session, softest-named first.

    Blender 5.x ships brushes as assets, so ``bpy.data.brushes`` is empty until
    one has been used. Rather than depend on an exact datablock name, prefer any
    brush that can paint images and whose name mentions "soft" - which is what
    the "Paint Soft" preset is.
    """
    usable = [b for b in bpy.data.brushes if getattr(b, "use_paint_image", False)]
    usable.sort(key=lambda b: (0 if "soft" in b.name.lower() else 1, b.name))
    return usable


def _setup_brush(context, color=(1.0, 1.0, 1.0), secondary=(0.0, 0.0, 0.0)):
    """White-on-black soft brush Ä‚ËĂ˘â€šÂ¬Ă˘â‚¬ĹĄ what masking wants.

    Blender 5.x ships brushes as *assets*, so ``bpy.data.brushes`` is often
    empty until one has been used; the named preset is therefore best-effort and
    the soft settings are applied either way.
    """
    tool_settings = getattr(context, "tool_settings", None)
    paint = getattr(tool_settings, "image_paint", None)
    if paint is None:
        return

    for candidate in _brush_candidates():
        try:
            paint.brush = candidate
        except Exception:  # noqa: BLE001
            continue
        break

    brush = getattr(paint, "brush", None)
    if brush is None:
        return

    _set_colour(brush, "color", color)
    _set_colour(brush, "secondary_color", secondary)

    for attr, value in (
        ("strength", 1.0),
        ("use_pressure_strength", True),
        # A soft round brush, not the airbrush: airbrush keeps depositing while
        # the cursor sits still, which is wrong for blocking in a mask.
        ("stroke_method", "SPACE"),
        ("spacing", 0.10),
        ("hardness", 0.0),
        ("falloff_shape", "SPHERE"),
        ("curve_distance_falloff", 0.0),
    ):
        try:
            setattr(brush, attr, value)
        except (AttributeError, TypeError):
            pass


class PL_OT_clear_pre(bpy.types.Operator):
    bl_idname = "projlayer.clear_pre"
    bl_label = "Clear"
    bl_description = "Wipe the pre-mask (or the pre-paint sketch) and start over"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        conf = settings(context)
        return conf.premask_on or conf.prepaint_on

    def execute(self, context):
        conf = settings(context)
        cleared = []

        if conf.premask_on:
            layer = find_role_layer(context, "PREMASK")
            if layer is not None and layer.mask_image is not None:
                image = layer.mask_image
                width, height = image.size
                buffer = np.zeros((width * height * 4), dtype=np.float32)
                buffer[3::4] = 1.0
                image.pixels.foreach_set(buffer)
                image.update()
                image.pack()
                cleared.append("pre-mask")

        if conf.prepaint_on:
            layer = find_role_layer(context, "PREPAINT")
            if layer is not None and layer.source_image is not None:
                image = layer.source_image
                width, height = image.size
                buffer = np.zeros((width * height * 4), dtype=np.float32)
                # Transparent, not black: a blank sketch must show nothing.
                image.pixels.foreach_set(buffer)
                image.update()
                image.pack()
                cleared.append("pre-paint")

        pipeline.rebuild(context)
        conf.status = f"Cleared {' and '.join(cleared)}." if cleared else "Nothing to clear."
        return {"FINISHED"}


class PL_OT_toggle_premask(bpy.types.Operator):
    bl_idname = "projlayer.toggle_premask"
    bl_label = "Pre-mask"
    bl_description = (
        "Paint the area to regenerate. The painted mask is sent to the backend "
        "and becomes the new layer's mask, so only that part is replaced"
    )

    def execute(self, context):
        try:
            obj = _obj(context)
        except RuntimeError as exc:
            return _fail(self, context, str(exc))

        conf = settings(context)
        layer = find_role_layer(context, "PREMASK")
        conf.premask_on = not conf.premask_on

        if not conf.premask_on:
            # Keep the work: just stop it affecting the render.
            set_pre_modes(context, premask=False)
            _leave_paint_mode(context)
            conf.status = "Pre-mask off Ä‚ËĂ˘â€šÂ¬Ă˘â‚¬ĹĄ your painted mask is kept."
            return {"FINISHED"}

        width, height = resolution(context)
        uv_name = pipeline.mesh_uv_name(obj)
        if not uv_name:
            conf.premask_on = False
            return _fail(
                self, context,
                f"'{obj.name}' has no UV map. Unwrap it first Ä‚ËĂ˘â€šÂ¬Ă˘â‚¬ĹĄ the pre-mask is "
                "painted in the object's UV space so you can paint it from any angle.",
            )

        if layer is None:
            layer = stack(context, create=True).color_layers.add()
            layer.name = "Pre-mask"
            layer.role = "PREMASK"
            layer.blend = "MIX"
            # Kept well below opaque so you can still read the model under the
            # red overlay.
            layer.opacity = 0.30
            layer.uv_map = uv_name
            layer.mask_uv = uv_name
            # Transparent: the pre-mask shows itself through its mask, which is
            # drawn flat red so the painted area is obvious while you work.
            layer.source_image = _new_image(
                "PL_premask", (0.0, 0.0, 0.0, 0.0), width, height, alpha=True)
            layer.mask_image = _new_image(
                "PL_premask_mask", (0.0, 0.0, 0.0, 1.0), width, height, non_color=True)
        else:
            layer.uv_map = uv_name
            layer.mask_uv = uv_name

        _activate_layer(context, layer)
        set_pre_modes(context, premask=True)
        conf.status = "Pre-mask on Ä‚ËĂ˘â€šÂ¬Ă˘â‚¬ĹĄ paint the area to regenerate."
        try:
            bpy.ops.projlayer.paint_mask()
        except RuntimeError as exc:
            conf.status = f"Pre-mask on (could not switch to paint mode: {exc})"
        # Set the brush *after* entering paint mode: switching modes resets it,
        # which is why the colour never came out white before.
        _setup_brush(context)
        return {"FINISHED"}

class PL_OT_toggle_prepaint(bpy.types.Operator):
    bl_idname = "projlayer.toggle_prepaint"
    bl_label = "Pre-paint"
    bl_description = (
        "Sketch on the mesh in its own UV space. The sketch is sent as the "
        "reference, then discarded once the result arrives"
    )

    def execute(self, context):
        conf = settings(context)
        layer = find_role_layer(context, "PREPAINT")
        conf.prepaint_on = not conf.prepaint_on

        if not conf.prepaint_on:
            set_pre_modes(context, prepaint=False)
            _leave_paint_mode(context)
            conf.status = "Pre-paint off â€” your sketch is kept."
            return {"FINISHED"}

        try:
            obj = _obj(context)
        except RuntimeError as exc:
            conf.prepaint_on = False
            return _fail(self, context, str(exc))

        width, height = resolution(context)
        uv_name = pipeline.mesh_uv_name(obj)
        if not uv_name:
            conf.prepaint_on = False
            return _fail(
                self, context,
                f"'{obj.name}' has no UV map. Unwrap it first â€” the sketch is "
                "painted in the object's UV space.",
            )

        if layer is None:
            layer = stack(context, create=True).color_layers.add()
            layer.name = "Pre-paint"
            layer.role = "PREPAINT"
            # Full opacity: a sketch is a reference for the backend, not an
            # overlay to read the model through. Only the pre-mask is faint.
            layer.opacity = 1.0
            layer.uv_map = uv_name
            layer.source_image = _new_image(
                "PL_prepaint", (0.0, 0.0, 0.0, 0.0), width, height, alpha=True)
        else:
            layer.uv_map = uv_name

        _activate_layer(context, layer)
        set_pre_modes(context, prepaint=True)
        conf.status = "Pre-paint on â€” sketch, then hit Generate."
        try:
            bpy.ops.projlayer.paint_color()
        except RuntimeError as exc:
            conf.status = f"Pre-paint on (could not switch to paint mode: {exc})"
        _setup_brush(context)
        return {"FINISHED"}


class PL_OT_duplicate_layer(bpy.types.Operator):
    bl_idname = "projlayer.duplicate_layer"
    bl_label = "Duplicate Layer"

    def execute(self, context):
        source = active_layer(context)
        if source is None:
            return _fail(self, context, "No layer to duplicate.")

        layers = mode_layers(context)
        index = mode_active_index(context)

        layer = layers.add()
        for prop in source.bl_rna.properties:
            key = prop.identifier
            if key in {"rna_type"} or prop.is_readonly:
                continue
            try:
                setattr(layer, key, getattr(source, key))
            except Exception:
                pass
        layer.name = f"{source.name} copy"
        layer.solo = False

        # Copy the images so editing one does not affect the other.
        layer.source_image = _copy_image(source.source_image)
        layer.mask_image = _copy_image(source.mask_image)

        target = min(index + 1, len(layers) - 1)
        if target != len(layers) - 1:
            layers.move(len(layers) - 1, target)
        set_mode_active_index(context, target)
        pipeline.rebuild(context)
        return {"FINISHED"}


def _copy_image(image):
    if image is None:
        return None
    copy = image.copy()
    copy.name = f"{image.name}_copy"
    copy.pack()
    return copy


class PL_OT_remove_layer(bpy.types.Operator):
    bl_idname = "projlayer.remove_layer"
    bl_label = "Remove Layer"
    bl_description = "Delete the selected layer"

    @classmethod
    def poll(cls, context):
        return bool(mode_layers(context))

    # -1 means "the selected layer"; the per-row X sets a real index.
    index: bpy.props.IntProperty(default=-1)

    def invoke(self, context, event):
        if prefs_module.prefs(context).confirm_delete:
            return context.window_manager.invoke_confirm(self, event)
        return self.execute(context)

    def execute(self, context):
        layers = mode_layers(context)
        if self.index >= 0:
            index = max(0, min(self.index, len(layers) - 1))
        else:
            index = max(0, min(mode_active_index(context), len(layers) - 1))
        layer = layers[index]

        # Keep the mesh clean: drop this layer's projection attribute.
        for obj in bpy.data.objects:
            if obj.type != "MESH" or not layer.uv_attr:
                continue
            attr = obj.data.attributes.get(layer.uv_attr)
            if attr is not None and obj.data.users == 1:
                obj.data.attributes.remove(attr)

        layers.remove(index)
        set_mode_active_index(context, max(0, min(index, len(layers) - 1)))
        pipeline.rebuild(context)
        return {"FINISHED"}


class PL_OT_move_layer(bpy.types.Operator):
    bl_idname = "projlayer.move_layer"
    bl_label = "Move Layer"

    direction: bpy.props.EnumProperty(items=[("UP", "Up", ""), ("DOWN", "Down", "")])

    @classmethod
    def poll(cls, context):
        return len(mode_layers(context)) > 1

    def execute(self, context):
        layers = mode_layers(context)
        index = mode_active_index(context)
        # The panel lists topmost-first, so "up" means later in the stack.
        target = index + 1 if self.direction == "UP" else index - 1
        if 0 <= target < len(layers):
            layers.move(index, target)
            set_mode_active_index(context, target)
            pipeline.rebuild(context)
        return {"FINISHED"}


class PL_OT_select_layer(bpy.types.Operator):
    bl_idname = "projlayer.select_layer"
    bl_label = "Select Layer"

    index: bpy.props.IntProperty(default=0)

    def execute(self, context):
        layers = mode_layers(context)
        set_mode_active_index(context, max(0, min(self.index, len(layers) - 1)))
        pipeline.rebuild(context)
        return {"FINISHED"}


class PL_OT_toggle_solo(bpy.types.Operator):
    bl_idname = "projlayer.toggle_solo"
    bl_label = "Solo"

    index: bpy.props.IntProperty(default=0)

    def execute(self, context):
        layers = mode_layers(context)
        layer = layers[max(0, min(self.index, len(layers) - 1))]
        layer.solo = not layer.solo
        pipeline.rebuild(context)
        return {"FINISHED"}


class PL_OT_toggle_visible(bpy.types.Operator):
    bl_idname = "projlayer.toggle_visible"
    bl_label = "Toggle Visibility"

    index: bpy.props.IntProperty(default=0)

    def execute(self, context):
        layers = mode_layers(context)
        layer = layers[max(0, min(self.index, len(layers) - 1))]
        layer.enabled = not layer.enabled
        pipeline.rebuild(context)
        return {"FINISHED"}


class PL_OT_move_layer_to(bpy.types.Operator):
    bl_idname = "projlayer.move_layer_to"
    bl_label = "Reorder Layer"
    bl_description = "Move a layer to a new position in the stack"

    from_index: bpy.props.IntProperty(default=0)
    to_index: bpy.props.IntProperty(default=0)

    def execute(self, context):
        layers = mode_layers(context)
        count = len(layers)
        source = max(0, min(self.from_index, count - 1))
        target = max(0, min(self.to_index, count - 1))
        if source != target:
            layers.move(source, target)
            set_mode_active_index(context, target)
            pipeline.rebuild(context)
        return {"FINISHED"}


# Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬ painting Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬


def _activate_paint_target(context, obj, material, image, uv_name):
    """Point Blender's texture-paint tools at the right image and UV map.

    Blender paints into whichever image texture node is active in the material,
    mapping strokes through the active UV map. Our real texture nodes now live
    inside node groups, and a node nested in a group cannot be the material's
    active node Ä‚ËĂ˘â€šÂ¬Ă˘â‚¬ĹĄ so one reusable top-level node is kept pointing at whatever is
    being painted. It references the same image datablock, so the strokes land in
    the layer either way.
    """
    if uv_name and uv_name in obj.data.uv_layers:
        obj.data.uv_layers.active = obj.data.uv_layers[uv_name]

    paint = context.tool_settings.image_paint
    paint.canvas = image

    if material is None or not material.use_nodes or material.node_tree is None:
        return

    tree = material.node_tree
    target = tree.nodes.get("PL_PaintTarget")
    if target is None:
        target = tree.nodes.new("ShaderNodeTexImage")
        target.name = "PL_PaintTarget"
        target.label = "Paint target"
        target.location = (-300, -600)
    target.image = image

    for node in tree.nodes:
        node.select = node == target
    tree.nodes.active = target


class PL_OT_paint_mask(bpy.types.Operator):
    bl_idname = "projlayer.paint_mask"
    bl_label = "Paint Mask"
    bl_description = "Paint on this layer's mask (white shows the layer, black hides it)"

    @classmethod
    def poll(cls, context):
        return active_layer(context) is not None

    def execute(self, context):
        try:
            obj = _obj(context)
        except RuntimeError as exc:
            return _fail(self, context, str(exc))

        layer = active_layer(context)
        if layer.mask_image is None:
            # Old files, and flattened layers, can arrive without a mask.
            # Creating a white one is far better than refusing to paint.
            layer.mask_image = _new_image(
                f"PL_mask_{layer.name}", (1.0, 1.0, 1.0, 1.0),
                *resolution(context))

        # Using an ordinary layer's brush leaves the pre-mask / pre-paint modes.
        if layer.role == "LAYER":
            set_pre_modes(context, premask=False, prepaint=False)

        if obj.mode != "TEXTURE_PAINT":
            previous = obj.mode
        else:
            previous = "OBJECT"
        context.scene["pl_previous_mode"] = previous

        uv_name = _prepare_paint_uv(obj, layer, for_mask=True)
        if uv_name is None:
            return _fail(
                self, context,
                f"'{obj.name}' has no UV map this mask can be painted through.",
            )
        _activate_paint_target(context, obj, obj.active_material, layer.mask_image, uv_name)
        bpy.ops.object.mode_set(mode="TEXTURE_PAINT")
        # After the mode switch: entering paint mode resets the brush.
        _setup_brush(context)
        settings(context).status = f"Painting mask: {layer.name}"
        return {"FINISHED"}


class PL_OT_paint_color(bpy.types.Operator):
    bl_idname = "projlayer.paint_color"
    bl_label = "Paint Color"
    bl_description = "Paint directly on this layer's image"

    @classmethod
    def poll(cls, context):
        return active_layer(context) is not None

    def execute(self, context):
        try:
            obj = _obj(context)
        except RuntimeError as exc:
            return _fail(self, context, str(exc))

        layer = active_layer(context)
        if layer.source_image is None:
            return _fail(self, context, "This layer has no image.")

        conf = settings(context)
        if layer.role == "LAYER":
            set_pre_modes(context, premask=False, prepaint=False)

        if obj.mode != "TEXTURE_PAINT":
            context.scene["pl_previous_mode"] = obj.mode

        uv_name = _prepare_paint_uv(obj, layer)
        if uv_name is None:
            if layer.uv_attr and not is_paint_layer(layer):
                return _fail(
                    self, context,
                    f"'{layer.name}' gets its colour from the camera projection, "
                    "so there is nothing to paint. Paint its Mask instead, or add "
                    "a paint layer.",
                )
            return _fail(
                self, context,
                f"'{obj.name}' has no UV map this layer can paint through.",
            )
        _activate_paint_target(context, obj, obj.active_material, layer.source_image, uv_name)
        bpy.ops.object.mode_set(mode="TEXTURE_PAINT")
        # After the mode switch: entering paint mode resets the brush.
        _setup_brush(context)
        settings(context).status = f"Painting: {layer.name}"
        return {"FINISHED"}


class PL_OT_invert_mask(bpy.types.Operator):
    bl_idname = "projlayer.invert_mask"
    bl_label = "Invert Mask"
    bl_description = "Swap what the mask hides and shows"

    @classmethod
    def poll(cls, context):
        layer = active_layer(context)
        return layer is not None and layer.mask_image is not None

    def execute(self, context):
        layer = active_layer(context)
        image = layer.mask_image
        width, height = image.size
        buffer = np.empty(width * height * 4, dtype=np.float32)
        image.pixels.foreach_get(buffer)
        buffer = buffer.reshape((-1, 4))
        for channel in range(3):
            buffer[:, channel] = 1.0 - buffer[:, channel]
        image.pixels.foreach_set(buffer.ravel())
        image.update()
        image.pack()
        settings(context).status = "Mask inverted."
        return {"FINISHED"}


class PL_OT_fill_mask(bpy.types.Operator):
    bl_idname = "projlayer.fill_mask"
    bl_label = "Fill Mask"
    bl_description = "Fill this layer's whole mask with white (show all) or black (hide all)"
    bl_options = {"REGISTER", "UNDO"}

    value: bpy.props.FloatProperty(
        name="Value", default=1.0, min=0.0, max=1.0,
        description="1 fills white (the layer shows everywhere), 0 fills black",
    )

    @classmethod
    def poll(cls, context):
        layer = active_layer(context)
        return layer is not None and layer.mask_image is not None

    def execute(self, context):
        layer = active_layer(context)
        image = layer.mask_image
        width, height = image.size
        buffer = np.full((width * height * 4,), self.value, dtype=np.float32)
        buffer[3::4] = 1.0
        image.pixels.foreach_set(buffer)
        image.update()
        image.pack()
        settings(context).status = (
            "Mask filled white." if self.value > 0.5 else "Mask filled black."
        )
        return {"FINISHED"}


class PL_OT_feather_mask(bpy.types.Operator):
    bl_idname = "projlayer.feather_mask"
    bl_label = "Soften Seams"
    bl_description = (
        "Blur this layer's mask by the Soften amount so the projection fades "
        "out instead of stopping abruptly"
    )
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        layer = active_layer(context)
        return layer is not None and layer.mask_image is not None

    # 0 means "use the layer's Soften setting" so the slider drives this button.
    radius: bpy.props.IntProperty(default=0, min=0, max=64)

    def execute(self, context):
        layer = active_layer(context)
        image = layer.mask_image
        radius = self.radius or max(1, int(layer.projection.feather_px))
        width, height = image.size
        buffer = np.empty(width * height * 4, dtype=np.float32)
        image.pixels.foreach_get(buffer)
        grey = buffer.reshape((height, width, 4))[:, :, 0]
        blurred = render._box_blur(grey, radius)
        buffer = buffer.reshape((height, width, 4))
        for channel in range(3):
            buffer[:, :, channel] = blurred
        image.pixels.foreach_set(buffer.ravel())
        image.update()
        image.pack()
        settings(context).status = f"Mask softened ({radius}px)."
        return {"FINISHED"}


# Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬ projection Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬


def _apply_projection_to_camera(camera, projection):
    camera.matrix_world = render.matrix_from_projection(projection)
    camera.data.lens = projection.lens
    camera.data.sensor_width = projection.sensor_width
    try:
        camera.data.sensor_fit = projection.sensor_fit
    except TypeError:
        pass
    camera.data.shift_x = projection.shift_x
    camera.data.shift_y = projection.shift_y
    camera.data.clip_start = projection.clip_start
    camera.data.clip_end = projection.clip_end


def _look_through(context, camera):
    """Point the 3D viewport through ``camera``."""
    window, area, region, space, region_3d = render._find_view3d(context)
    if region_3d is None:
        return False
    context.scene.camera = camera
    try:
        region_3d.view_perspective = "CAMERA"
    except TypeError:
        return False
    return True


class PL_OT_focus_projection(bpy.types.Operator):
    bl_idname = "projlayer.focus_projection"
    bl_label = "Go to Projection"
    bl_description = (
        "Move the projector camera back to where this layer was captured from, "
        "without changing the layer"
    )

    index: bpy.props.IntProperty(default=-1)

    @classmethod
    def poll(cls, context):
        return active_layer(context) is not None

    def execute(self, context):
        layers = mode_layers(context)
        if self.index >= 0:
            layer = layers[max(0, min(self.index, len(layers) - 1))]
        else:
            layer = active_layer(context)
        if layer is None:
            return _fail(self, context, "No layer selected.")

        camera = render.ensure_camera(context)
        _apply_projection_to_camera(camera, layer.projection)
        context.scene.camera = camera
        if self.index >= 0:
            set_mode_active_index(context, max(0, min(self.index, len(layers) - 1)))
        # Actually look through that camera, otherwise nothing visibly happens.
        looked = _look_through(context, camera)
        settings(context).status = (
            f"Looking through '{layer.name}'." if looked
            else f"Camera moved to '{layer.name}' (open a 3D viewport to see it)."
        )
        return {"FINISHED"}


class PL_OT_reproject(bpy.types.Operator):
    bl_idname = "projlayer.reproject"
    bl_label = "Re-project"
    bl_description = (
        "Re-snapshot the UV map and re-render the mask from this layer's stored "
        "camera, using the current feather and angle settings"
    )
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        layer = active_layer(context)
        return layer is not None and layer.source_image is not None

    def execute(self, context):
        try:
            obj = _obj(context)
        except RuntimeError as exc:
            return _fail(self, context, str(exc))

        layer = active_layer(context)
        projection = layer.projection
        camera = render.ensure_camera(context)
        _apply_projection_to_camera(camera, projection)
        context.scene.camera = camera

        width = projection.res_x or resolution(context)[0]
        height = projection.res_y or resolution(context)[1]

        render.snapshot_projection(context, obj, camera, attr_name=layer.uv_attr)

        # Re-transfer the visibility into the object's UV space for this camera.
        if layer.mask_image is not None and layer.mask_uv:
            vis_path = render.temp_path("reproject_vis.png")
            render.render_visibility(context, obj, camera, width, height, vis_path)
            transferred = render.transfer_projection_to_uv(
                obj, render.load_gray(vis_path), layer.uv_attr,
                layer.mask_uv, layer.mask_image.size[0], camera=camera,
            )
            if transferred is not None:
                _write_gray(layer.mask_image, transferred)
                if projection.feather_px > 0:
                    _write_gray(
                        layer.mask_image,
                        render._box_blur(render.image_gray(layer.mask_image),
                                         int(projection.feather_px)),
                    )

        pipeline.rebuild(context)
        settings(context).status = f"Re-projected '{layer.name}'."
        return {"FINISHED"}


def _write_gray(image, array):
    height, width = array.shape
    buffer = np.empty((height, width, 4), dtype=np.float32)
    for channel in range(3):
        buffer[:, :, channel] = array
    buffer[:, :, 3] = 1.0
    image.pixels.foreach_set(buffer.ravel())
    image.update()
    image.pack()


class PL_OT_rebuild_material(bpy.types.Operator):
    bl_idname = "projlayer.rebuild_material"
    bl_label = "Rebuild Material"
    bl_description = "Re-composite every layer into the material"

    def execute(self, context):
        material = pipeline.rebuild(context)
        if material is None:
            return _fail(self, context, "Select a mesh with a material.")
        settings(context).status = "Material rebuilt."
        return {"FINISHED"}


# Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬ flatten Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬


class PL_OT_flatten(bpy.types.Operator):
    bl_idname = "projlayer.flatten"
    bl_label = "Flatten to Texture"
    bl_description = (
        "Bake the whole stack into ordinary textures on the mesh's real UV map. "
        "The layers are kept but muted, so nothing is lost"
    )
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return bool(visible_layers(context))

    def invoke(self, context, event):
        return context.window_manager.invoke_confirm(self, event)

    def execute(self, context):
        try:
            obj = _obj(context)
        except RuntimeError as exc:
            return _fail(self, context, str(exc))

        if obj.active_material is None:
            return _fail(self, context, "The object has no material.")

        target_uv = pipeline.mesh_uv_name(obj)
        if not target_uv:
            return _fail(self, context, "This mesh has no UV map to flatten into.")

        conf = settings(context)
        size = int(conf.flatten_resolution)

        colour = [l for l in visible_layers(context, "COLOR") if l.source_image]
        normals = [l for l in visible_layers(context, "NORMAL") if l.source_image]
        if not colour and not normals:
            return _fail(self, context, "There is nothing to flatten.")

        total = len(colour) + len(normals)
        done = [0]

        def step(_index, _count, kind):
            done[0] += 1
            _bake_progress(context, f"Baking {kind}  {done[0]}/{total}")

        try:
            colour_result = (pipeline.flatten_color_stack(
                context, obj, size, target_uv, progress=step) if colour else None)
            normal_result = (pipeline.flatten_normal_stack(
                context, obj, size, target_uv, progress=step) if normals else None)
        finally:
            conf.busy_text = ""

        if colour_result is None and normal_result is None:
            return _fail(self, context, "Baking produced nothing.")

        _bake_progress(context, "Replacing layers")

        # Everything that was baked in is dropped, so the result really is all
        # that is left - both stacks, not just the one being flattened.
        active = stack(context, create=True)
        for layer in list(active.color_layers):
            pipeline._drop_layer(context, layer)
        for layer in list(active.normal_layers):
            pipeline._drop_layer(context, layer)

        if colour_result is not None:
            image = bpy.data.images.new(f"PL_{obj.name}_albedo", size, size,
                                        alpha=False)
            render.write_rgb(image, colour_result)
            layer = active.color_layers.add()
            layer.name = "Flattened"
            layer.role = "LAYER"
            layer.uv_map = target_uv
            layer.mask_uv = target_uv
            layer.source_image = image

        if normal_result is not None:
            image = bpy.data.images.new(f"PL_{obj.name}_normal", size, size,
                                        alpha=False)
            render.write_rgb(image, normal_result)
            try:
                image.colorspace_settings.name = "Non-Color"
            except TypeError:
                pass
            layer = active.normal_layers.add()
            layer.name = "Flattened"
            layer.role = "LAYER"
            layer.uv_map = target_uv
            layer.mask_uv = target_uv
            layer.source_image = image

        # A flattened layer must carry a mask too, or every later attempt to
        # paint it reports "this layer has no mask".
        for _layer in list(active.color_layers) + list(active.normal_layers):
            if _layer.mask_image is None:
                _layer.mask_image = _new_image(
                    f"PL_{_layer.name}_mask", (1.0, 1.0, 1.0, 1.0), size, size)

        active.active_color = 0
        active.active_normal = 0
        pipeline.rebuild(context)
        # Clear it here too: "Replacing layers" is reported after the progress
        # loop finishes, so it would otherwise stay on screen for good.
        conf.busy_text = ""
        conf.status = f"Flattened {total} layer(s) into {size}px."
        self.report({"INFO"}, conf.status)
        return {"FINISHED"}
        self.report({"INFO"}, f"Flattened {count} layer(s).")
        return {"FINISHED"}

        target_uv = _ensure_real_uv(context, obj)
        if target_uv is None:
            return _fail(self, context, "Could not create a UV map to bake into.")

        width = int(settings(context).resolution)
        results = []

        for kind, suffix, bake_type in (
            ("COLOR", "albedo", "EMIT"),
            ("NORMAL", "normal", "NORMAL"),
        ):
            layers = [l for l in visible_layers(context, kind) if l.source_image]
            if not layers:
                continue
            image = bpy.data.images.new(
                f"PL_{obj.name}_{suffix}", width, width, alpha=False
            )
            if bake_type == "NORMAL":
                image.colorspace_settings.name = "Non-Color"
            if _bake(context, obj, material, image, target_uv, bake_type):
                image.pack()
                results.append(suffix)
            else:
                bpy.data.images.remove(image)

        if not results:
            return _fail(self, context, "Baking produced nothing Ä‚ËĂ˘â€šÂ¬Ă˘â‚¬ĹĄ see the console.")

        for layer in visible_layers(context):
            layer.enabled = False
        pipeline.rebuild(context)
        settings(context).status = f"Flattened: {', '.join(results)} at {width}px."
        self.report({"INFO"}, f"Baked {', '.join(results)}.")
        return {"FINISHED"}


def _ensure_real_uv(context, obj):
    for uv in obj.data.uv_layers:
        if not uv.name.startswith(render.UV_PREFIX):
            obj.data.uv_layers.active = uv
            return uv.name
    if obj.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")
    render._select_only(context, obj)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    try:
        bpy.ops.uv.smart_project(angle_limit=1.15, island_margin=0.02)
    finally:
        bpy.ops.object.mode_set(mode="OBJECT")
    active = obj.data.uv_layers.active
    return active.name if active else None


def _bake(context, obj, material, image, uv_name, bake_type):
    """Bake the current shading into ``image`` through ``uv_name``."""
    scene = context.scene
    saved = {
        "engine": scene.render.engine,
        "samples": getattr(scene.cycles, "samples", 1),
        "bounces": getattr(scene.cycles, "max_bounces", 0),
        "direct": scene.render.bake.use_pass_direct,
        "indirect": scene.render.bake.use_pass_indirect,
        "margin": scene.render.bake.margin,
        "uv": obj.data.uv_layers.active.name if obj.data.uv_layers.active else "",
        "mode": obj.mode,
    }
    nodes_tree = material.node_tree
    tex_node = nodes_tree.nodes.new("ShaderNodeTexImage")
    tex_node.image = image
    tex_node.name = "PL_BakeTarget"

    output = nodes.find_output(material)
    bsdf = nodes.find_principled(material)
    rewire = None

    try:
        if obj.mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")
        obj.data.uv_layers.active = obj.data.uv_layers[uv_name]
        for node in nodes_tree.nodes:
            node.select = node == tex_node
        nodes_tree.nodes.active = tex_node

        scene.render.engine = "CYCLES"
        scene.cycles.samples = 1
        scene.cycles.max_bounces = 0
        scene.render.bake.use_pass_direct = False
        scene.render.bake.use_pass_indirect = False
        scene.render.bake.margin = 16

        if bake_type == "EMIT":
            # Route the finished Base Color straight into an Emission shader.
            source = bsdf.inputs["Base Color"].links[0].from_socket
            emis = nodes_tree.nodes.new("ShaderNodeEmission")
            emis.name = "PL_BakeEmission"
            nodes_tree.links.new(source, emis.inputs["Color"])
            for link in list(output.inputs["Surface"].links):
                nodes_tree.links.remove(link)
            nodes_tree.links.new(emis.outputs["Emission"], output.inputs["Surface"])
            rewire = emis

        render._select_only(context, obj)
        bpy.ops.object.bake(type=bake_type, width=image.size[0], height=image.size[1])
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"[Projection Layers] bake failed: {exc}")
        return False
    finally:
        if rewire is not None:
            nodes_tree.nodes.remove(rewire)
            for link in list(output.inputs["Surface"].links):
                nodes_tree.links.remove(link)
            if bsdf is not None:
                nodes_tree.links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])
        nodes_tree.nodes.remove(tex_node)
        scene.render.engine = saved["engine"]
        scene.cycles.samples = saved["samples"]
        scene.cycles.max_bounces = saved["bounces"]
        scene.render.bake.use_pass_direct = saved["direct"]
        scene.render.bake.use_pass_indirect = saved["indirect"]
        scene.render.bake.margin = saved["margin"]
        if saved["uv"] in obj.data.uv_layers:
            obj.data.uv_layers.active = obj.data.uv_layers[saved["uv"]]
        if saved["mode"] != "OBJECT":
            try:
                bpy.ops.object.mode_set(mode=saved["mode"])
            except RuntimeError:
                pass


# Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬ layer assets Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬Ä‚ËĂ˘â‚¬ĹĄĂ˘â€šÂ¬


class PL_OT_export_layer(bpy.types.Operator):
    bl_idname = "projlayer.export_layer"
    bl_label = "Export Layer Image"
    bl_description = "Save this layer's image to disk"

    filepath: bpy.props.StringProperty(subtype="FILE_PATH")
    filter_glob: bpy.props.StringProperty(default="*.png", options={"HIDDEN"})

    @classmethod
    def poll(cls, context):
        layer = active_layer(context)
        return layer is not None and layer.source_image is not None

    def invoke(self, context, event):
        layer = active_layer(context)
        self.filepath = bpy.path.abspath(f"//{layer.name.replace(' ', '_')}.png")
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}

    def execute(self, context):
        layer = active_layer(context)
        image = layer.source_image.copy()
        try:
            image.filepath_raw = self.filepath
            image.file_format = "PNG"
            image.save()
        finally:
            bpy.data.images.remove(image)
        settings(context).status = f"Exported to {self.filepath}"
        return {"FINISHED"}


CLASSES = (
    PL_OT_generate,
    PL_OT_generate_views,
    PL_OT_texture_collection,
    PL_OT_view_add,
    PL_OT_view_remove,
    PL_OT_view_select,
    PL_OT_cancel,
    PL_OT_test_connection,
    PL_OT_detect_slots,
    PL_OT_snap_camera,
    PL_OT_fit_camera,
    PL_OT_add_layer,
    PL_OT_select_prompt,
    PL_OT_clear_pre,
    PL_OT_toggle_premask,
    PL_OT_toggle_prepaint,
    PL_OT_duplicate_layer,
    PL_OT_remove_layer,
    PL_OT_move_layer,
    PL_OT_move_layer_to,
    PL_OT_select_layer,
    PL_OT_toggle_solo,
    PL_OT_toggle_visible,
    PL_OT_paint_mask,
    PL_OT_paint_color,
    PL_OT_invert_mask,
    PL_OT_fill_mask,
    PL_OT_feather_mask,
    PL_OT_focus_projection,
    PL_OT_reproject,
    PL_OT_rebuild_material,
    PL_OT_flatten,
    PL_OT_export_layer,
)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
