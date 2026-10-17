"""Background job runner.

Network work happens on a daemon thread; everything that touches Blender data
happens on the main thread via a ``bpy.app.timers`` poll. There is deliberately
no modal operator — Blender stays fully interactive (you can keep orbiting the
mesh) while a generation is running, and the sidebar shows live progress.
"""

import threading
import time
import traceback

import bpy

# How often the main-thread poll runs. 0.1 s is smooth without being wasteful.
TICK = 0.1


class Job:
    """State shared between the worker thread and the main thread.

    Only plain attribute reads/writes happen across the thread boundary, which
    is safe enough for progress reporting.
    """

    def __init__(self, label):
        self.label = label
        self.started = time.monotonic()
        self.progress = 0.0
        self.progress_text = ""
        self.status = ""
        self.error = ""
        self.finished = False
        self.cancelled = False
        self.result = None
        self.traceback = ""
        self._on_done = None

    # called from the worker thread
    def set_progress(self, done, total=0):
        if total:
            self.progress = max(0.0, min(1.0, float(done) / float(total)))
            self.progress_text = f"{done}/{total}"
        else:
            self.progress_text = str(done)
            self.progress = min(0.99, self.progress + 0.01)

    def set_status(self, text):
        self.status = text

    @property
    def elapsed(self):
        """Seconds since the job started. The HUD shows this because ComfyUI
        gives no real progress without a websocket, so elapsed time plus a
        pulsing dot is the only honest liveness signal."""
        return max(0.0, time.monotonic() - self.started)

    def is_cancelled(self):
        return self.cancelled

    def cancel(self):
        self.cancelled = True


_current = None
_lock = threading.Lock()


def current() -> Job:
    """The running job, or None."""
    return _current


def is_running() -> bool:
    job = _current
    return job is not None and not job.finished


def cancel_current():
    job = _current
    if job is not None:
        job.cancel()


def start(scene_name, label, worker, on_done):
    """Run ``worker(job)`` on a daemon thread.

    ``worker`` must not touch ``bpy`` data — pass results back through the
    returned value. ``on_done(job, result)`` runs on the main thread, and is
    also called when the job fails (inspect ``job.error``).
    """
    global _current

    if is_running():
        raise RuntimeError("A generation is already running.")

    job = Job(label)
    job._on_done = on_done

    with _lock:
        _current = job

    def _run():
        try:
            job.result = worker(job)
        except Exception as exc:  # noqa: BLE001 — reported to the artist
            job.error = str(exc) or exc.__class__.__name__
            job.traceback = traceback.format_exc()
        finally:
            job.finished = True

    threading.Thread(target=_run, name="projlayer-job", daemon=True).start()
    _ensure_timer(scene_name)
    return job


def _ensure_timer(scene_name):
    if not bpy.app.timers.is_registered(_tick):
        bpy.app.timers.register(
            lambda: _tick(scene_name), first_interval=TICK
        )


def _tick(scene_name):
    """Main-thread poll: sync progress into the scene, redraw, finish up."""
    global _current

    job = _current
    scene = bpy.data.scenes.get(scene_name)
    settings = getattr(scene, "projlayer", None) if scene else None

    if job is None:
        if settings is not None:
            settings.is_running = False
        return None  # unregister

    if settings is not None:
        settings.is_running = not job.finished
        settings.progress = job.progress
        settings.progress_text = job.progress_text
        settings.status = job.status
        settings.last_error = job.error

    _redraw()

    if not job.finished:
        return TICK

    # Job is done — hand back to the operator on the main thread.
    callback = job._on_done
    with _lock:
        _current = None
    if settings is not None:
        settings.is_running = False
        settings.progress = 0.0
        settings.progress_text = ""

    if callback is not None:
        try:
            callback(job, job.result)
        except Exception:  # noqa: BLE001
            traceback.print_exc()
            if settings is not None:
                settings.last_error = "Finished, but the result could not be applied (see console)."

    _redraw()
    return None  # unregister


def _redraw():
    """Tag 3D viewports (and the sidebar) for redraw so progress animates."""
    wm = bpy.context.window_manager
    if wm is None:
        return
    for window in wm.windows:
        screen = window.screen
        if screen is None:
            continue
        for area in screen.areas:
            if area.type in {"VIEW_3D", "PROPERTIES"}:
                area.tag_redraw()
