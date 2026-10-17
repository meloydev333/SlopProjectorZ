"""Graph templates and slot plumbing for the projlayer addon.

The addon never asks an artist to edit JSON. It exposes a handful of named
*slots* ("Reference image", "Depth map", "Seed", ...), guesses which node of the
current ComfyUI graph owns each slot, and patches a copy of that graph before it
is sent to the server.

Everything here is pure data manipulation — no ``bpy``, no network, no file
I/O — so it is safe to call from any thread and trivial to unit test.
"""

import copy

# ---------------------------------------------------------------------------
# Slots
# ---------------------------------------------------------------------------

#: Canonical slot keys mapped to the label an artist sees. The bundled graphs
#: use these exact strings as ``_meta.title`` so detection is unambiguous.
SLOT_LABELS = {
    "reference": "Reference image",
    "depth": "Depth map",
    "normal": "Normal map",
    "mask": "Mask",
    "positive": "Positive prompt",
    "negative": "Negative prompt",
    "seed": "Seed",
    "steps": "Steps",
    "cfg": "CFG scale",
    "width": "Width",
    "height": "Height",
    "output": "Output",
}

#: Slot keys in sidebar order. ``prefs.py`` iterates this to build its per-slot
#: title overrides, so keep it in step with :data:`SLOT_LABELS`.
SLOT_ORDER = ("reference", "depth", "normal", "mask", "positive", "negative",
              "seed", "steps", "cfg", "width", "height", "output")

#: Lowercase title fragments accepted per slot, most specific first. The first
#: entry doubles as the exact label match (see :func:`_detect_by_title`).
_TITLE_ALIASES = {
    "reference": ("reference image", "reference", "ref image", "source image",
                  "input image", "input"),
    "depth": ("depth map", "depthmap", "depth image", "depth"),
    "normal": ("normal map", "normalmap", "normals", "normal image", "normal"),
    "mask": ("inpaint mask", "mask"),
    # "negative" is resolved before "positive" so that a bare "prompt" fallback
    # cannot claim a node titled "Negative prompt".
    "negative": ("negative prompt", "negative"),
    "positive": ("positive prompt", "positive", "input prompt", "text prompt",
                 "prompt", "input text", "text"),
    "seed": ("seed",),
    "steps": ("steps",),
    "cfg": ("cfg scale", "cfg", "guidance"),
    "width": ("width",),
    "height": ("height",),
    # "preview image" is deliberately absent: a PreviewImage node often sits
    # alongside the real SaveImage, and titles saying "Output image" must win
    # the exact pass. PreviewImage is still picked up as a last-resort fallback
    # by _detect_by_class_type.
    "output": ("output image", "output", "saveimagewebsocket", "save image",
               "result"),
}

#: Slots resolved in this order during title matching. Text slots come before
#: the image slots only where a substring would otherwise be ambiguous.
_SLOT_MATCH_ORDER = ("reference", "depth", "normal", "mask", "negative",
                     "positive", "seed", "steps", "cfg", "width", "height",
                     "output")

#: A node whose title contains any of these must not fill the "positive" slot.
_NEGATIVE_TITLE_MARKERS = ("negative", "neg ")

_IMAGE_SLOTS = ("reference", "depth", "normal", "mask")
_TEXT_SLOTS = ("positive", "negative")
_NUMBER_SLOTS = ("seed", "steps", "cfg", "width", "height")

#: Input names to write, in order, when a slot carries an image or a string.
_IMAGE_INPUT_NAMES = ("image", "filename")
#: "value" covers PrimitiveStringMultiline; "positivePrompt" covers Runware-style
#: inference nodes; "text" is CLIPTextEncode.
_TEXT_INPUT_NAMES = ("text", "prompt", "positivePrompt", "value", "string")

#: Order in which leftover LoadImage nodes fill the image slots.
_LOAD_IMAGE_ORDER = ("reference", "depth", "normal", "mask")


def slot_map() -> dict:
    """Return the canonical slot keys and their human labels."""
    return dict(SLOT_LABELS)


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------


def _title_of(node: dict) -> str:
    """Lowercased ``_meta.title`` of a node (empty string when absent)."""
    meta = node.get("_meta")
    if not isinstance(meta, dict):
        return ""
    return str(meta.get("title") or "").strip().lower()


def _id_sort_key(node_id: str):
    """Sort numeric node ids numerically so "2" comes before "10"."""
    text = str(node_id)
    return (0, int(text), "") if text.isdigit() else (1, 0, text)


def _detect_by_title(nodes: dict, found: dict) -> None:
    """Fill ``found`` from node titles: exact label matches, then substrings."""
    ordered = sorted(nodes.items(), key=lambda item: _id_sort_key(item[0]))
    for exact in (True, False):
        for node_id, node in ordered:
            title = _title_of(node)
            if not title:
                continue
            for slot in _SLOT_MATCH_ORDER:
                if slot in found:
                    continue
                aliases = _TITLE_ALIASES[slot]
                # Never let a generic "prompt" alias claim a negative node.
                if slot == "positive" and any(m in title for m in _NEGATIVE_TITLE_MARKERS):
                    continue
                if exact:
                    if title in aliases:
                        found[slot] = node_id
                elif any(alias in title for alias in aliases):
                    found[slot] = node_id


def _detect_by_class_type(nodes: dict, found: dict) -> None:
    """Best-effort fallback for graphs whose node titles say nothing useful."""
    image_loaders = []
    text_encoders = []
    websocket_saves = []
    save_nodes = []
    preview_nodes = []

    for node_id, node in sorted(nodes.items(), key=lambda item: _id_sort_key(item[0])):
        class_type = str(node.get("class_type") or "")
        lowered = class_type.lower()
        inputs = node.get("inputs") if isinstance(node.get("inputs"), dict) else {}

        if lowered == "loadimage":
            image_loaders.append(node_id)
        if "cliptextencode" in lowered or "primitivestring" in lowered:
            text_encoders.append(node_id)
        # Prefer a websocket saver (images arrive inline), then a real saver
        # (readable from /history), and only then a preview.
        if "saveimagewebsocket" in lowered:
            websocket_saves.append(node_id)
        elif "previewimage" in lowered:
            preview_nodes.append(node_id)
        elif "saveimage" in lowered:
            save_nodes.append(node_id)

        # Numeric knobs announce themselves by their input names.
        for slot in _NUMBER_SLOTS:
            if slot not in found and slot in inputs:
                found[slot] = node_id

    # First unused LoadImage fills reference, then depth, normal, mask.
    spare_loaders = [node_id for node_id in image_loaders if node_id not in found.values()]
    for slot in _LOAD_IMAGE_ORDER:
        if slot not in found and spare_loaders:
            found[slot] = spare_loaders.pop(0)

    spare_encoders = [node_id for node_id in text_encoders if node_id not in found.values()]
    for slot in _TEXT_SLOTS:
        if slot not in found and spare_encoders:
            found[slot] = spare_encoders.pop(0)

    if "output" not in found:
        for group in (websocket_saves, save_nodes, preview_nodes):
            if group:
                found["output"] = group[0]
                break


def detect_slots(graph: dict) -> dict:
    """Guess which node id fills each slot of an API-format graph.

    Titles are trusted first (an exact label match beats a substring match, and
    both beat ``class_type`` heuristics). Returns ``{slot_key: node_id}`` for
    whatever was found, and never raises: a malformed graph simply yields fewer
    usable slots.
    """
    found = {}
    try:
        if not isinstance(graph, dict):
            return found
        nodes = {
            str(node_id): node
            for node_id, node in graph.items()
            if isinstance(node, dict)
        }
        _detect_by_title(nodes, found)
        if len(found) < len(SLOT_LABELS):
            _detect_by_class_type(nodes, found)
    except Exception:  # noqa: BLE001 — detection is a convenience, not a guarantee
        return found
    return found


# ---------------------------------------------------------------------------
# Patching
# ---------------------------------------------------------------------------


def _write_first(inputs: dict, names: tuple, value) -> None:
    """Write ``value`` to the first of ``names`` the node already declares."""
    for name in names:
        if name in inputs:
            inputs[name] = value
            return
    inputs[names[0]] = value  # node declares none of them; create the preferred one


def apply_slots(graph: dict, slots: dict, values: dict) -> dict:
    """Return a patched deep copy of ``graph``.

    For every slot present in both ``slots`` and ``values`` the value is written
    into that node's ``inputs``: images try ``("image", "filename")``, text tries
    ``("text", "prompt", "string")`` and numbers use their exact slot key. The
    input graph is never mutated.
    """
    patched = copy.deepcopy(graph)
    if not isinstance(patched, dict) or not isinstance(slots, dict) or not isinstance(values, dict):
        return patched

    for slot, value in values.items():
        node_id = slots.get(slot)
        if node_id is None:
            continue
        node = patched.get(str(node_id))
        if not isinstance(node, dict):
            continue

        inputs = node.get("inputs")
        if not isinstance(inputs, dict):
            inputs = {}
            node["inputs"] = inputs

        if slot in _IMAGE_SLOTS:
            _write_first(inputs, _IMAGE_INPUT_NAMES, value)
        elif slot in _TEXT_SLOTS:
            _write_first(inputs, _TEXT_INPUT_NAMES, value)
        elif slot in _NUMBER_SLOTS:
            inputs[slot] = value
        # Other slots (e.g. "output") carry no value to write.

    return patched


# ---------------------------------------------------------------------------
# Default graphs
# ---------------------------------------------------------------------------

# SDXL text-to-image. Node titles that match slot_map() labels are how
# detect_slots() finds the knobs; the other titles are plain descriptions.
DEFAULT_TXT2IMG = {
    "1": {
        "class_type": "CheckpointLoaderSimple",
        "inputs": {"ckpt_name": "sd_xl_base_1.0.safetensors"},
        "_meta": {"title": "Checkpoint"},
    },
    "2": {
        "class_type": "CLIPTextEncode",
        "inputs": {
            "text": "a detailed photograph, sharp focus, natural lighting",
            "clip": ["1", 1],
        },
        "_meta": {"title": "Positive prompt"},
    },
    "3": {
        "class_type": "CLIPTextEncode",
        "inputs": {
            "text": "blurry, low quality, watermark, text",
            "clip": ["1", 1],
        },
        "_meta": {"title": "Negative prompt"},
    },
    "4": {
        "class_type": "EmptyLatentImage",
        "inputs": {"width": 1024, "height": 1024, "batch_size": 1},
        "_meta": {"title": "Empty latent"},
    },
    "5": {
        "class_type": "KSampler",
        "inputs": {
            "model": ["1", 0],
            "seed": 0,
            "steps": 28,
            "cfg": 6.5,
            "sampler_name": "dpmpp_2m",
            "scheduler": "karras",
            "denoise": 1.0,
            "positive": ["2", 0],
            "negative": ["3", 0],
            "latent_image": ["4", 0],
        },
        "_meta": {"title": "Sampler"},
    },
    "6": {
        "class_type": "VAEDecode",
        "inputs": {"samples": ["5", 0], "vae": ["1", 2]},
        "_meta": {"title": "Decode"},
    },
    # SaveImageWebsocket streams the result over /ws; SaveImage also writes it to
    # the output folder so the HTTP polling fallback can retrieve it.
    "7": {
        "class_type": "SaveImageWebsocket",
        "inputs": {"images": ["6", 0]},
        "_meta": {"title": "Output"},
    },
    "8": {
        "class_type": "SaveImage",
        "inputs": {"images": ["6", 0], "filename_prefix": "projlayer/txt2img"},
        "_meta": {"title": "Save image (disk)"},
    },
}

# SDXL img2img + ControlNet. The reference image is scaled to the slider size and
# encoded straight into the latent, a depth and a normal ControlNet (both driven
# from one union model) are applied to the prompts, and the mask feeds
# SetLatentNoiseMask so masked areas are regenerated. Covers every slot in
# slot_map(). The LoadImage files are placeholders: projlayer uploads the real
# renders and patches the names in via apply_slots().
DEFAULT_IMG2IMG = {
    "1": {
        "class_type": "CheckpointLoaderSimple",
        "inputs": {"ckpt_name": "sd_xl_base_1.0.safetensors"},
        "_meta": {"title": "Checkpoint"},
    },
    "2": {
        "class_type": "CLIPTextEncode",
        "inputs": {
            "text": "a detailed photograph, sharp focus, natural lighting",
            "clip": ["1", 1],
        },
        "_meta": {"title": "Positive prompt"},
    },
    "3": {
        "class_type": "CLIPTextEncode",
        "inputs": {
            "text": "blurry, low quality, watermark, text",
            "clip": ["1", 1],
        },
        "_meta": {"title": "Negative prompt"},
    },
    "4": {
        "class_type": "LoadImage",
        "inputs": {"image": "reference.png"},
        "_meta": {"title": "Reference image"},
    },
    "5": {
        "class_type": "LoadImage",
        "inputs": {"image": "depth.png"},
        "_meta": {"title": "Depth map"},
    },
    "6": {
        "class_type": "LoadImage",
        "inputs": {"image": "normal.png"},
        "_meta": {"title": "Normal map"},
    },
    "7": {
        "class_type": "LoadImage",
        "inputs": {"image": "mask.png"},
        "_meta": {"title": "Mask"},
    },
    # ImageScale gives the Width/Height slots a home. The ControlNet hints do not
    # need scaling — ControlNetApplyAdvanced resizes them internally — and
    # SetLatentNoiseMask resizes the mask to the latent.
    "8": {
        "class_type": "ImageScale",
        "inputs": {
            "image": ["4", 0],
            "upscale_method": "lanczos",
            "width": 1024,
            "height": 1024,
            "crop": "disabled",
        },
        "_meta": {"title": "Scale to output size"},
    },
    "9": {
        "class_type": "VAEEncode",
        "inputs": {"pixels": ["8", 0], "vae": ["1", 2]},
        "_meta": {"title": "Encode latent"},
    },
    "10": {
        "class_type": "SetLatentNoiseMask",
        "inputs": {"samples": ["9", 0], "mask": ["7", 1]},
        "_meta": {"title": "Noised latent"},
    },
    "11": {
        "class_type": "ControlNetLoader",
        "inputs": {"control_net_name": "controlnet-union-sdxl-1.0.safetensors"},
        "_meta": {"title": "ControlNet loader"},
    },
    # SetUnionControlNetType (comfyui_controlnet_aux) picks the control mode of a
    # union model, so one checkpoint serves both maps.
    "12": {
        "class_type": "SetUnionControlNetType",
        "inputs": {"control_net": ["11", 0], "type": "depth"},
        "_meta": {"title": "ControlNet type A"},
    },
    "13": {
        "class_type": "SetUnionControlNetType",
        "inputs": {"control_net": ["11", 0], "type": "normal"},
        "_meta": {"title": "ControlNet type B"},
    },
    "14": {
        "class_type": "ControlNetApplyAdvanced",
        "inputs": {
            "positive": ["2", 0],
            "negative": ["3", 0],
            "control_net": ["12", 0],
            "image": ["5", 0],
            "strength": 0.75,
            "start_percent": 0.0,
            "end_percent": 0.85,
        },
        "_meta": {"title": "ControlNet apply A"},
    },
    "15": {
        "class_type": "ControlNetApplyAdvanced",
        "inputs": {
            "positive": ["14", 0],
            "negative": ["14", 1],
            "control_net": ["13", 0],
            "image": ["6", 0],
            "strength": 0.5,
            "start_percent": 0.0,
            "end_percent": 0.85,
        },
        "_meta": {"title": "ControlNet apply B"},
    },
    "16": {
        "class_type": "KSampler",
        "inputs": {
            "model": ["1", 0],
            "seed": 0,
            "steps": 28,
            "cfg": 6.5,
            "sampler_name": "dpmpp_2m",
            "scheduler": "karras",
            "denoise": 0.65,
            "positive": ["15", 0],
            "negative": ["15", 1],
            "latent_image": ["10", 0],
        },
        "_meta": {"title": "Sampler"},
    },
    "17": {
        "class_type": "VAEDecode",
        "inputs": {"samples": ["16", 0], "vae": ["1", 2]},
        "_meta": {"title": "Decode"},
    },
    "18": {
        "class_type": "SaveImageWebsocket",
        "inputs": {"images": ["17", 0]},
        "_meta": {"title": "Output"},
    },
    "19": {
        "class_type": "SaveImage",
        "inputs": {"images": ["17", 0], "filename_prefix": "projlayer/img2img"},
        "_meta": {"title": "Save image (disk)"},
    },
}


# ── bypassing branches the add-on is not feeding ─────────────────────────────


def _link_to(value, node_id):
    """Is this input value a link coming from node_id?"""
    return (
        isinstance(value, (list, tuple))
        and len(value) == 2
        and str(value[0]) == str(node_id)
    )


def _pass_through(inputs, wanted):
    """The removed node's own input to feed that consumer with.

    ComfyUI's Ctrl+B matches by socket type. The API format does not carry
    types, but names work for the shapes that matter here: a sampler consuming
    ``positive`` gets the bypassed ControlNet-apply's own ``positive``.
    """
    direct = inputs.get(wanted)
    if direct is not None:
        return direct
    for name, value in inputs.items():
        if wanted in name or name in wanted:
            return value
    return None


def bypass_node(graph, node_id):
    """Reconnect a node's consumers to its own inputs.

    Mirrors ComfyUI's bypass: an output link is re-routed to the equivalent
    input, so a ControlNetApplyAdvanced between a conditioning source and the
    sampler can disappear without breaking the chain.
    """
    node = graph.get(node_id)
    if not isinstance(node, dict):
        return False
    inputs = node.get("inputs") or {}
    changed = False
    for other_id, other in graph.items():
        if str(other_id) == str(node_id) or not isinstance(other, dict):
            continue
        for name, value in list((other.get("inputs") or {}).items()):
            if not _link_to(value, node_id):
                continue
            source = _pass_through(inputs, name)
            if source is None:
                other["inputs"].pop(name, None)
            else:
                other["inputs"][name] = source
            changed = True
    return changed


def _referenced(graph):
    """Every node id still used as an input somewhere."""
    used = set()
    for node in graph.values():
        if not isinstance(node, dict):
            continue
        for value in (node.get("inputs") or {}).values():
            if isinstance(value, (list, tuple)) and len(value) == 2:
                used.add(str(value[0]))
    return used


def disable_unused_guides(graph, slots, provided):
    """Remove the depth / normal / mask branches the add-on is not feeding.

    Skipping a slot is not enough. The workflow keeps its own LoadImage with
    whatever filename was last saved in it, so an unused guide still runs — and
    a stale filename makes ComfyUI reject the entire prompt with an
    "invalid ....png" error instead of quietly ignoring the branch.

    Returns the slot keys whose branches were removed.
    """
    provided = set(provided or ())
    removed = []

    for key in ("depth", "normal", "mask"):
        if key in provided or key not in slots:
            continue
        start = str(slots[key])
        if start not in graph:
            continue

        branch = [start]
        for node_id, node in graph.items():
            if not isinstance(node, dict) or str(node_id) == start:
                continue
            feeds = any(_link_to(value, start)
                        for value in (node.get("inputs") or {}).values())
            if feeds and "controlnet" in str(node.get("class_type", "")).lower():
                branch.append(str(node_id))

        # Bypass before deleting, so each node hands its consumers back to its
        # own inputs and nothing is left dangling.
        for node_id in branch:
            bypass_node(graph, node_id)
        for node_id in branch:
            graph.pop(node_id, None)

        # A ControlNet loader that nothing references would still be evaluated.
        used = _referenced(graph)
        for node_id, node in list(graph.items()):
            if not isinstance(node, dict):
                continue
            if "controlnetloader" in str(node.get("class_type", "")).lower():
                if str(node_id) not in used:
                    graph.pop(node_id, None)

        removed.append(key)

    return removed
