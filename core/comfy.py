"""ComfyUI transport for the projlayer addon.

Design notes
------------
* **Standard library only.** HTTP goes through :mod:`urllib.request` and the
  multipart upload body is assembled by hand. Nothing here imports ``requests``
  (let alone ``websocket-client``), so the module cannot break because Blender's
  bundled interpreter happens to be missing a package.
* **WebSocket is optional.** If ``websocket-client`` *is* importable we stream
  live progress and images from ``/ws``; otherwise (or if the socket fails at any
  point) every code path degrades gracefully to HTTP polling of ``/history``.
* **Thread safe and bpy-free.** Each function is a self-contained blocking call
  that never touches ``bpy``; callers run them on a worker thread and marshal
  results back to the main thread themselves.
* **PNG bytes in, PNG bytes out.** Images are never round-tripped through the
  filesystem, so no temporary files are left behind.
"""

import json
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid

try:  # optional dependency — live streaming is a bonus, never a requirement
    import websocket as _websocket
except Exception:  # noqa: BLE001 — any import failure simply disables the ws path
    _websocket = None


#: Address an artist is most likely to be running ComfyUI on.
DEFAULT_BASE_URL = "127.0.0.1:8188"

_HTTP_TIMEOUT = 10.0        # ordinary API calls
_UPLOAD_TIMEOUT = 60.0      # uploads can be a few megabytes
_INTERRUPT_TIMEOUT = 5.0    # fire-and-forget cancellation
_WS_CONNECT_TIMEOUT = 5.0
_WS_IDLE_TIMEOUT = 1.0      # so cancellation/deadlines are noticed promptly
_COMFY_WS_HEADER_BYTES = 8  # ComfyUI prefixes every binary frame with 8 bytes


class ComfyError(Exception):
    """Raised for any user-reportable ComfyUI failure."""


class _WebSocketUnavailable(Exception):
    """Internal: the WebSocket path failed, fall back to HTTP polling."""


# ---------------------------------------------------------------------------
# Addresses, error text
# ---------------------------------------------------------------------------


def _split_base(base_url: str) -> tuple[str, str]:
    """Return ``(http_base, ws_base)`` for a user-supplied ComfyUI address.

    Accepts ``127.0.0.1:8188``, ``http://host:port`` and ``https://host`` alike.
    """
    raw = (base_url or "").strip() or DEFAULT_BASE_URL
    if "://" not in raw:
        raw = "http://" + raw
    raw = raw.rstrip("/")
    if raw.startswith("https://"):
        return raw, "wss://" + raw[len("https://"):]
    if raw.startswith("http://"):
        return raw, "ws://" + raw[len("http://"):]
    return raw, raw


def _display_address(url: str) -> str:
    """Artist-facing form of an address, e.g. ``127.0.0.1:8188``."""
    raw = (url or "").strip().rstrip("/")
    if "://" in raw:
        netloc = urllib.parse.urlsplit(raw).netloc
        if netloc:
            return netloc
    return raw or DEFAULT_BASE_URL


def _safe_read(exc: urllib.error.HTTPError) -> str:
    """Best-effort read of an error response body (never raises)."""
    try:
        return exc.read().decode("utf-8", "replace")
    except Exception:  # noqa: BLE001 — the body is only used for diagnostics
        return ""


def _describe_node_error(node_id, info) -> str:
    """Turn one entry of ComfyUI's ``node_errors`` mapping into a sentence."""
    class_type = ""
    entries = info
    if isinstance(info, dict):
        class_type = str(info.get("class_type") or "")
        entries = info.get("errors") or info.get("error") or []
    if isinstance(entries, dict):
        entries = [entries]

    texts = []
    if isinstance(entries, (list, tuple)):
        for entry in entries:
            if isinstance(entry, dict):
                text = entry.get("message") or entry.get("type") or ""
                details = entry.get("details")
                if details:
                    text = f"{text}: {details}" if text else str(details)
            else:
                text = entry
            text = str(text).strip()
            if text:
                texts.append(text)

    label = f"{class_type} (node {node_id})" if class_type else f"node {node_id}"
    return f"{label}: {'; '.join(texts)}" if texts else label


def _format_error_body(body_text: str) -> str:
    """Render ComfyUI's JSON error body as one readable sentence.

    Handles the ``/prompt`` validation shape (``error`` + ``node_errors``) as well
    as plain strings, and falls back to the raw text when the body is not JSON.
    """
    text = (body_text or "").strip()
    if not text:
        return ""
    try:
        payload = json.loads(text)
    except ValueError:
        return text[:400]
    if not isinstance(payload, dict):
        return str(payload)[:400]

    parts = []
    error = payload.get("error")
    if isinstance(error, dict):
        head = error.get("message") or error.get("type")
        details = error.get("details")
        if head and details:
            parts.append(f"{head} ({details})")
        elif head:
            parts.append(str(head))
    elif isinstance(error, str) and error.strip():
        parts.append(error.strip())

    node_errors = payload.get("node_errors")
    if isinstance(node_errors, dict):
        for node_id in sorted(node_errors, key=lambda key: str(key)):
            parts.append(_describe_node_error(node_id, node_errors[node_id]))

    return " ".join(parts).strip() or text[:400]


def _failure_message(method: str, url: str, exc: BaseException, action: str) -> str:
    """One readable sentence describing why an HTTP call failed."""
    if isinstance(exc, urllib.error.HTTPError):
        detail = _format_error_body(_safe_read(exc))
        message = f"ComfyUI returned HTTP {exc.code} while trying to {action}."
        return f"{message} {detail}".strip()
    reason = getattr(exc, "reason", None) or exc
    address = _display_address(url)
    return f"Couldn't reach ComfyUI at {address} while trying to {action} — {reason}."


# ---------------------------------------------------------------------------
# Small request helpers
# ---------------------------------------------------------------------------


def _json_request(method: str, url: str, payload: dict | None = None,
                  timeout: float = _HTTP_TIMEOUT, action: str = "talk to ComfyUI"):
    """Perform one JSON HTTP request, returning the decoded body.

    Returns ``None`` for an empty body. Raises :class:`ComfyError` with an
    artist-readable message for every transport or HTTP failure. Every call is
    bounded by ``timeout`` seconds.
    """
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=max(1.0, float(timeout))) as response:
            body = response.read()
    except Exception as exc:  # noqa: BLE001 — HTTPError, URLError, timeouts, resets
        raise ComfyError(_failure_message(method, url, exc, action)) from exc

    text = body.decode("utf-8", "replace").strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except ValueError as exc:
        address = _display_address(url)
        raise ComfyError(
            f"ComfyUI at {address} sent a reply that isn't valid JSON "
            f"while trying to {action}."
        ) from exc


def _get_bytes(url: str, timeout: float = _HTTP_TIMEOUT,
               action: str = "download an image") -> bytes:
    """GET a URL and return the raw body (used for ``/view`` image downloads)."""
    request = urllib.request.Request(
        url, headers={"Accept": "image/png, */*"}, method="GET"
    )
    try:
        with urllib.request.urlopen(request, timeout=max(1.0, float(timeout))) as response:
            return response.read()
    except Exception as exc:  # noqa: BLE001
        raise ComfyError(_failure_message("GET", url, exc, action)) from exc


def _interrupt(base_url: str) -> None:
    """Ask ComfyUI to stop whatever it is running; failures are ignored."""
    try:
        _json_request("POST", f"{base_url}/interrupt", {}, timeout=_INTERRUPT_TIMEOUT,
                      action="stop the running job")
    except Exception:  # noqa: BLE001 — cancellation is best effort
        pass


def _close_quietly(socket) -> None:
    """Close a websocket without letting a teardown error escape."""
    try:
        socket.close()
    except Exception:  # noqa: BLE001
        pass


def _as_int(value, default: int = 0) -> int:
    """Coerce a websocket progress value to ``int`` without raising."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _timeout_message(budget: float) -> str:
    """Readable timeout sentence (``budget`` is the caller's timeout in seconds)."""
    return (f"ComfyUI didn't finish within {int(budget)} seconds, so projlayer stopped "
            f"waiting. Try a lighter workflow or increase the timeout.")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def ping(base_url: str, timeout: float = 1.5) -> tuple[bool, str]:
    """Check whether ComfyUI is reachable.

    Uses a cheap HEAD (falling back to GET) on ``/queue`` and returns
    ``(ok, human_message)``. Never raises — this is meant to be shown verbatim in
    the sidebar before a render starts.
    """
    http_base, _ = _split_base(base_url)
    address = _display_address(http_base)
    url = f"{http_base}/queue"

    last_status = None
    for method in ("HEAD", "GET"):
        request = urllib.request.Request(
            url, headers={"Accept": "application/json"}, method=method
        )
        try:
            with urllib.request.urlopen(request, timeout=max(0.2, float(timeout))):
                return True, f"ComfyUI is responding at {address}."
        except urllib.error.HTTPError as exc:
            last_status = exc.code  # the server answered, just not happily
        except Exception:  # noqa: BLE001 — connection refused, DNS, timeout, ...
            last_status = None

    if last_status is not None:
        return False, (f"ComfyUI answered at {address}, but its queue endpoint returned "
                       f"HTTP {last_status}.")
    return False, f"ComfyUI isn't responding at {address} — is it running?"


def _safe_filename(filename: str) -> str:
    """Reduce any user-supplied name to a bare, quoted-safe ``*.png`` name."""
    name = str(filename or "").replace("\\", "/").split("/")[-1]
    name = name.replace('"', "").strip()
    if not name:
        name = "projlayer.png"
    if not name.lower().endswith(".png"):
        name += ".png"
    return name


def _build_multipart(boundary: str, filename: str, content: bytes,
                     fields: dict) -> bytes:
    """Assemble a ``multipart/form-data`` body by hand (no external dependency)."""
    lines = []
    for key, value in fields.items():
        lines.append(f"--{boundary}".encode("utf-8"))
        lines.append(f'Content-Disposition: form-data; name="{key}"'.encode("utf-8"))
        lines.append(b"")
        lines.append(str(value).encode("utf-8"))

    lines.append(f"--{boundary}".encode("utf-8"))
    lines.append(
        f'Content-Disposition: form-data; name="image"; filename="{filename}"'.encode("utf-8")
    )
    lines.append(b"Content-Type: image/png")
    lines.append(b"")
    lines.append(content)
    lines.append(f"--{boundary}--".encode("utf-8"))
    lines.append(b"")
    return b"\r\n".join(lines)


def upload_image(base_url: str, png_bytes: bytes, filename: str) -> dict:
    """Upload PNG bytes to ComfyUI's input folder.

    Returns the parsed JSON reply, which always contains ``name`` (and usually
    ``subfolder``/``type``). That name is what a ``LoadImage`` node needs.
    """
    http_base, _ = _split_base(base_url)
    if not png_bytes:
        raise ComfyError("There's no image data to upload to ComfyUI.")

    safe_name = _safe_filename(filename)
    boundary = "----projlayer" + uuid.uuid4().hex
    body = _build_multipart(
        boundary, safe_name, png_bytes,
        {"type": "input", "subfolder": "", "overwrite": "true"},
    )
    upload_url = f"{http_base}/upload/image"
    request = urllib.request.Request(
        upload_url,
        data=body,
        headers={
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "Accept": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=_UPLOAD_TIMEOUT) as response:
            raw = response.read()
    except Exception as exc:  # noqa: BLE001
        raise ComfyError(_failure_message("POST", upload_url, exc, "upload an image")) from exc

    try:
        payload = json.loads(raw.decode("utf-8", "replace") or "{}")
    except ValueError as exc:
        raise ComfyError("ComfyUI accepted the image upload but sent an unreadable reply.") from exc
    if not isinstance(payload, dict) or not payload.get("name"):
        raise ComfyError("ComfyUI accepted the image upload but didn't return a filename.")
    return payload


def queue_prompt(base_url: str, graph: dict, client_id: str) -> str:
    """Send an API-format graph to ComfyUI and return its ``prompt_id``.

    On HTTP 400 ComfyUI's validation body (top-level error plus per-node errors)
    is turned into a readable sentence and raised as :class:`ComfyError`.
    """
    http_base, _ = _split_base(base_url)
    if not isinstance(graph, dict) or not graph:
        raise ComfyError("There's nothing to send — the ComfyUI graph is empty.")

    payload = _json_request(
        "POST",
        f"{http_base}/prompt",
        {"prompt": graph, "client_id": str(client_id)},
        timeout=_HTTP_TIMEOUT,
        action="queue the graph",
    )

    prompt_id = ""
    if isinstance(payload, dict):
        prompt_id = str(payload.get("prompt_id") or "")
    if not prompt_id:
        raise ComfyError("ComfyUI accepted the graph but didn't return a job id.")
    return prompt_id


def _view_url(http_base: str, image_info: dict) -> str:
    """Build the ``/view`` URL for one entry of a history node's ``images`` list."""
    query = urllib.parse.urlencode({
        "filename": image_info.get("filename", ""),
        "subfolder": image_info.get("subfolder", "") or "",
        "type": image_info.get("type", "output") or "output",
    })
    return f"{http_base}/view?{query}"


def _images_from_outputs(http_base: str, outputs: dict,
                         prefer_node=None) -> list[bytes]:
    """Download the raw PNG bytes for every image referenced by a history entry.

    ``prefer_node`` puts one node's images first. Graphs often contain several
    image outputs (a PreviewImage next to the real SaveImage, a comparison node,
    a service node), and the first one in history order is not necessarily the
    one the artist means.
    """
    images = []
    failures = []
    if not isinstance(outputs, dict):
        return images

    items = list(outputs.items())
    if prefer_node is not None:
        wanted = str(prefer_node)
        items.sort(key=lambda kv: 0 if str(kv[0]) == wanted else 1)

    for _node_id, node_output in items:
        if not isinstance(node_output, dict):
            continue
        for info in node_output.get("images") or []:
            if not isinstance(info, dict) or not info.get("filename"):
                continue
            try:
                images.append(_get_bytes(_view_url(http_base, info),
                                         action="download a finished image"))
            except ComfyError as exc:
                failures.append(str(exc))

    # A single unreadable temp file shouldn't sink the batch, but if *nothing*
    # came back the artist deserves to know why.
    if not images and failures:
        raise ComfyError(failures[0])
    return images


def _history_entry(http_base: str, prompt_id: str,
                   timeout: float = _HTTP_TIMEOUT) -> dict | None:
    """Return the ``/history`` record for ``prompt_id``, or ``None`` while pending."""
    payload = _json_request(
        "GET",
        f"{http_base}/history/{urllib.parse.quote(str(prompt_id))}",
        timeout=timeout,
        action="check the run history",
    )
    if not isinstance(payload, dict):
        return None
    entry = payload.get(str(prompt_id))
    if isinstance(entry, dict):
        return entry
    if len(payload) == 1:  # some builds return just the single entry
        only = next(iter(payload.values()))
        if isinstance(only, dict):
            return only
    return None


def _node_class_type(graph: dict, node_id) -> str:
    """Look up a node's ``class_type`` so errors can name the node type."""
    if not isinstance(graph, dict) or node_id is None:
        return ""
    node = graph.get(str(node_id))
    return str(node.get("class_type") or "") if isinstance(node, dict) else ""


def _format_execution_error(info, graph: dict) -> str:
    """Readable sentence for an ``execution_error`` payload (node type + message)."""
    if not isinstance(info, dict):
        return "ComfyUI reported an execution error."
    node_id = info.get("node_id") or info.get("node")
    node_type = str(info.get("node_type") or "") or _node_class_type(graph, node_id)
    exception = str(
        info.get("exception_message") or info.get("exception_type") or "unknown error"
    )
    if node_type and node_id is not None:
        where = f"{node_type} (node {node_id})"
    elif node_type:
        where = node_type
    elif node_id is not None:
        where = f"node {node_id}"
    else:
        where = "a node"
    return f"{where} failed: {exception}"


def _history_error(entry: dict, graph: dict) -> str | None:
    """Return a readable message if a history entry records a failure."""
    status = entry.get("status") if isinstance(entry, dict) else None
    if not isinstance(status, dict):
        return None
    if str(status.get("status_str") or "").lower() != "error":
        return None
    for message in status.get("messages") or []:
        if (isinstance(message, (list, tuple)) and len(message) == 2
                and message[0] == "execution_error"):
            return _format_execution_error(message[1], graph)
    return "ComfyUI reported an execution error."


def _history_complete(entry: dict) -> bool:
    """True once ComfyUI considers the prompt finished (successfully or not)."""
    if not isinstance(entry, dict):
        return False
    status = entry.get("status") if isinstance(entry.get("status"), dict) else {}
    if status.get("completed") is True:
        return True
    return bool(entry.get("outputs"))


def fetch_outputs_from_history(base_url: str, prompt_id: str,
                               prefer_node=None) -> list[bytes]:
    """Return the raw PNG bytes of every image ComfyUI recorded for a prompt."""
    http_base, _ = _split_base(base_url)
    entry = _history_entry(http_base, str(prompt_id))
    if not isinstance(entry, dict):
        return []
    return _images_from_outputs(http_base, entry.get("outputs") or {},
                                prefer_node=prefer_node)


def _queue_counts(http_base: str, timeout: float = _HTTP_TIMEOUT) -> tuple[int, int] | None:
    """Return ``(running, pending)`` job counts from ``/queue``, or ``None``."""
    try:
        payload = _json_request("GET", f"{http_base}/queue", timeout=timeout,
                                action="read the queue")
    except ComfyError:
        return None
    if not isinstance(payload, dict):
        return None
    running = payload.get("queue_running") or []
    pending = payload.get("queue_pending") or []
    return (len(running) if isinstance(running, list) else 0,
            len(pending) if isinstance(pending, list) else 0)


def run_prompt(base_url: str, graph: dict, client_id: str,
               on_progress=None, is_cancelled=None,
               timeout: float = 900.0, poll_interval: float = 0.4,
               prefer_node=None) -> list[bytes]:
    """Queue ``graph`` and block until ComfyUI has produced its images.

    Returns a list of raw PNG bytes (one per generated image, possibly empty).
    ``on_progress(done, total)`` is called from this thread while waiting — the
    WebSocket path reports real sampler steps, the polling path reports a
    monotonically increasing count with ``total`` unknown (``0``). If
    ``is_cancelled()`` returns True the server job is interrupted and ``[]`` is
    returned immediately. Raises :class:`ComfyError` on timeout or execution
    failure.
    """
    http_base, ws_base = _split_base(base_url)
    cancel = is_cancelled if callable(is_cancelled) else (lambda: False)
    report = on_progress if callable(on_progress) else None
    budget = max(1.0, float(timeout))
    deadline = time.monotonic() + budget
    interval = max(0.05, float(poll_interval))

    if cancel():
        return []

    prompt_id = queue_prompt(http_base, graph, client_id)

    if _websocket is not None:
        try:
            images = _run_via_websocket(ws_base, http_base, graph, client_id, prompt_id,
                                        report, cancel, deadline, budget)
        except _WebSocketUnavailable:
            images = []
        else:
            if images:
                return images
            if cancel():
                return []
            # Streaming finished without inline frames: the graph probably saves
            # to disk instead of using SaveImageWebsocket, so read the files.
            return fetch_outputs_from_history(http_base, prompt_id,
                                              prefer_node=prefer_node)

    return _run_via_polling(http_base, graph, prompt_id, report, cancel,
                            deadline, budget, interval, prefer_node=prefer_node)


def _run_via_websocket(ws_base: str, http_base: str, graph: dict, client_id: str,
                       prompt_id: str, report, cancel, deadline: float,
                       budget: float) -> list[bytes]:
    """Stream progress and images from ``/ws``.

    Raises :class:`_WebSocketUnavailable` (never a user-facing error) whenever the
    socket cannot be used, so :func:`run_prompt` can fall back to HTTP polling.
    """
    timeout_exc = getattr(_websocket, "WebSocketTimeoutException", TimeoutError)
    socket = _websocket.WebSocket()
    try:
        socket.connect(f"{ws_base}/ws?clientId={urllib.parse.quote(str(client_id))}",
                       timeout=_WS_CONNECT_TIMEOUT)
        socket.settimeout(_WS_IDLE_TIMEOUT)
    except Exception as exc:  # noqa: BLE001 — server without ws, blocked port, ...
        _close_quietly(socket)
        raise _WebSocketUnavailable(str(exc)) from exc

    frames = []
    try:
        while True:
            if cancel():
                _interrupt(http_base)
                return []
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                _interrupt(http_base)
                raise ComfyError(_timeout_message(budget))

            try:
                frame = socket.recv()
            except timeout_exc:
                frame = ""  # idle tick: reuse it to double-check /history
            except Exception as exc:  # noqa: BLE001 — drop back to polling
                raise _WebSocketUnavailable(str(exc)) from exc
            if frame is None:
                raise _WebSocketUnavailable("the socket closed unexpectedly")

            if isinstance(frame, bytes):
                # Binary frames are images with ComfyUI's 8-byte header stripped.
                if len(frame) > _COMFY_WS_HEADER_BYTES:
                    frames.append(frame[_COMFY_WS_HEADER_BYTES:])
                continue

            if frame:
                try:
                    message = json.loads(frame)
                except ValueError:
                    continue
                if not isinstance(message, dict):
                    continue
                kind = str(message.get("type") or "")
                data = message.get("data") if isinstance(message.get("data"), dict) else {}
                if kind == "progress":
                    if report is not None:
                        report(_as_int(data.get("value")), _as_int(data.get("max")))
                elif kind == "execution_error":
                    _interrupt(http_base)
                    raise ComfyError(
                        f"ComfyUI stopped with an error — "
                        f"{_format_execution_error(data, graph)}."
                    )
                elif kind in ("execution_interrupted", "execution_cancelled"):
                    break  # the job was aborted; nothing more is coming
                elif kind == "executing" and data.get("node") is None:
                    break  # ComfyUI's "the queue is done with this prompt" signal
                continue

            # Idle tick: /history is the source of truth in case we missed the
            # completion message or connected after the job had already ended.
            entry = _history_entry(http_base, prompt_id,
                                   timeout=max(1.0, min(_HTTP_TIMEOUT, remaining)))
            if not isinstance(entry, dict):
                continue
            error = _history_error(entry, graph)
            if error:
                raise ComfyError(f"ComfyUI stopped with an error — {error}.")
            if _history_complete(entry):
                break
    finally:
        _close_quietly(socket)

    if frames:
        return frames
    return fetch_outputs_from_history(http_base, prompt_id)


def _run_via_polling(http_base: str, graph: dict, prompt_id: str, report, cancel,
                     deadline: float, budget: float, interval: float,
                     prefer_node=None) -> list[bytes]:
    """Fallback path: watch ``/history`` for completion and ``/queue`` for progress."""
    step = 0
    high_water = 0
    while True:
        if cancel():
            _interrupt(http_base)
            return []
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            _interrupt(http_base)
            raise ComfyError(_timeout_message(budget))
        request_timeout = max(1.0, min(_HTTP_TIMEOUT, remaining))

        entry = _history_entry(http_base, prompt_id, timeout=request_timeout)
        if isinstance(entry, dict):
            error = _history_error(entry, graph)
            if error:
                raise ComfyError(f"ComfyUI stopped with an error — {error}.")
            if _history_complete(entry):
                return _images_from_outputs(http_base, entry.get("outputs") or {},
                                            prefer_node=prefer_node)

        # ComfyUI reports what is *left* in the queue, not what is done, so turn
        # that into a monotonically increasing step count. ``total`` stays 0
        # because the real number of sampler steps is unknown on this path.
        counts = _queue_counts(http_base, timeout=request_timeout)
        if counts is None:
            step += 1
        else:
            running, pending = counts
            remaining_jobs = running + pending
            high_water = max(high_water, remaining_jobs)
            step = max(step + 1, high_water - remaining_jobs)
        if report is not None:
            report(step, 0)

        time.sleep(min(interval, max(0.05, deadline - time.monotonic())))
