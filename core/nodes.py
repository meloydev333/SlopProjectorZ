"""Builds the layer stack into the object's material node tree.

The material tree stays readable no matter how many layers you have: everything
the add-on builds lives inside a single **mother group** (``PL_Stack``) wired to
the Principled BSDF, and inside that group each layer is its own small group
(``PL_LayerColor_0``, ``PL_LayerNormal_0``, â€¦) chained one into the next.

    material tree            PL_Stack group
    â”Śâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”       â”Śâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
    â”‚ [PL_Stack]â”€â”€â”€â”€â”€â”Ľâ”€â”€Colorâ”¤ BaseColor â†’[L0]â†’[L1]â†’[L2]â”€â”€â”€â”€â”€â”€â†’ Colorâ”‚
    â”‚      â”‚         â”‚       â”‚                                      â”‚
    â”‚  Principled    â”‚Normal â”‚ BaseNormalâ†’[N0]â†’[N1]â”€â”€â”€â”€â”€â”€â”€â”€â”€â†’ Normal â”‚
    â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”       â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”

Chain order is bottom-to-top: ``color_layers[0]`` is the bottom layer.

The artist's material is otherwise untouched. The add-on only writes the Base
Color and Normal inputs of the existing Principled BSDF, and remembers whatever
was plugged in there before (in ``PL_Original*`` reroute nodes) so removing the
last layer restores the original material exactly.
"""

import bpy
import numpy as np

from . import render
from .. import props

PREFIX = "PL_"

_ORIGINAL_SURFACE = f"{PREFIX}OriginalSurface"
_ORIGINAL_BASECOLOR = f"{PREFIX}OriginalBaseColor"
_ORIGINAL_NORMAL = f"{PREFIX}OriginalNormal"
_FALLBACK_SHADER = f"{PREFIX}FallbackShader"
_RNM_GROUP = f"{PREFIX}NormalRNM"

STACK_GROUP = f"{PREFIX}Stack"
COLOR_LAYER_GROUP = f"{PREFIX}LayerColor"
NORMAL_LAYER_GROUP = f"{PREFIX}LayerNormal"

#: The node Blender paints into, kept alive across rebuilds (see :func:`clear`).
PAINT_TARGET_NODE = f"{PREFIX}PaintTarget"

FLAT_NORMAL = (0.5, 0.5, 1.0)

_KEEP_IN_GROUP = {"NodeGroupInput", "NodeGroupOutput"}


# â”€â”€ socket helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _socket(collection, identifier):
    """Look a socket up by identifier (names are ambiguous on ShaderNodeMix)."""
    for sock in collection:
        if sock.identifier == identifier:
            return sock
    raise KeyError(f"socket '{identifier}' not found")


def _in(node, identifier):
    return _socket(node.inputs, identifier)


def _out(node, identifier):
    return _socket(node.outputs, identifier)


def _break(links, socket):
    for link in list(socket.links):
        links.remove(link)


# â”€â”€ material-tree plumbing â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def find_output(material):
    for node in material.node_tree.nodes:
        if node.type == "OUTPUT_MATERIAL" and getattr(node, "is_active_output", True):
            return node
    for node in material.node_tree.nodes:
        if node.type == "OUTPUT_MATERIAL":
            return node
    return None


def find_principled(material):
    for node in material.node_tree.nodes:
        if node.type == "BSDF_PRINCIPLED":
            return node
    return None


def clear(material):
    """Remove every node and node group this add-on created.

    ``PL_PaintTarget`` is spared on purpose. It is the node Blender paints into,
    and the debounced rebuild fires shortly *after* entering paint mode â€” wiping
    it there would silently break painting.
    """
    if material is not None and material.use_nodes and material.node_tree is not None:
        keep = {
            _ORIGINAL_SURFACE, _ORIGINAL_BASECOLOR, _ORIGINAL_NORMAL,
            _FALLBACK_SHADER, PAINT_TARGET_NODE,
        }
        doomed = [
            node
            for node in material.node_tree.nodes
            if node.name.startswith(PREFIX) and node.name not in keep
        ]
        for node in doomed:
            material.node_tree.nodes.remove(node)

    for group in list(bpy.data.node_groups):
        if group.name.startswith(PREFIX) and group.library is None:
            bpy.data.node_groups.remove(group)


def _remember_original(material, node, socket, marker_name):
    """Stash whatever currently feeds ``socket`` behind a reroute node."""
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    marker = nodes.get(marker_name)
    if marker is not None:
        return marker
    marker = nodes.new("NodeReroute")
    marker.name = marker_name
    marker.label = "Projection Layers: original input"
    marker.location = (node.location.x - 320, node.location.y + 120)
    if socket.links:
        links.new(socket.links[0].from_socket, marker.inputs[0])
    return marker


def _ensure_scaffold(material):
    """Make sure there is an output, a Principled, and original-input markers."""
    nodes = material.node_tree.nodes
    links = material.node_tree.links

    output = find_output(material)
    if output is None:
        output = nodes.new("ShaderNodeOutputMaterial")
        output.location = (600, 0)

    _remember_original(material, output, output.inputs["Surface"], _ORIGINAL_SURFACE)

    bsdf = find_principled(material)
    if bsdf is None:
        bsdf = nodes.new("ShaderNodeBsdfPrincipled")
        bsdf.name = _FALLBACK_SHADER
        fallback = True
    else:
        fallback = False

    _remember_original(material, bsdf, bsdf.inputs["Base Color"], _ORIGINAL_BASECOLOR)
    _remember_original(material, bsdf, bsdf.inputs["Normal"], _ORIGINAL_NORMAL)

    if fallback or not output.inputs["Surface"].links:
        _break(links, output.inputs["Surface"])
        links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])
    return output, bsdf


# â”€â”€ node factories â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _node(nodes, kind, name, location):
    node = nodes.new(kind)
    node.name = name
    node.label = name.replace(PREFIX, "")
    node.location = location
    return node


def _math(nodes, name, operation, location, defaults=None):
    """A Math node. ``defaults`` maps input index -> default value."""
    node = _node(nodes, "ShaderNodeMath", name, location)
    node.operation = operation
    for index, value in (defaults or {}).items():
        node.inputs[index].default_value = value
    return node


def _vector_math(nodes, name, operation, location):
    node = _node(nodes, "ShaderNodeVectorMath", name, location)
    node.operation = operation
    return node


def _const_vector(nodes, name, vector, x, y):
    """A CombineXYZ holding a constant vector."""
    node = _node(nodes, "ShaderNodeCombineXYZ", name, (x, y))
    node.inputs[0].default_value = vector[0]
    node.inputs[1].default_value = vector[1]
    node.inputs[2].default_value = vector[2]
    return node


# â”€â”€ node groups â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _reset_group(name, inputs, outputs):
    """Get (or create) a shader node group and wipe its innards.

    The interface is left alone between rebuilds, so dragging a slider does not
    churn datablocks or invalidate the sockets other nodes are linked to.
    """
    group = bpy.data.node_groups.get(name)
    if group is None:
        group = bpy.data.node_groups.new(name, "ShaderNodeTree")
    elif group.bl_idname != "ShaderNodeTree":
        bpy.data.node_groups.remove(group)
        group = bpy.data.node_groups.new(name, "ShaderNodeTree")

    for node in list(group.nodes):
        if node.bl_idname not in _KEEP_IN_GROUP:
            group.nodes.remove(node)
    for link in list(group.links):
        group.links.remove(link)

    existing = {}
    for item in group.interface.items_tree:
        if getattr(item, "item_type", "") == "SOCKET":
            existing[(item.name, item.in_out)] = item
    for socket_name, socket_type in inputs:
        if (socket_name, "INPUT") not in existing:
            group.interface.new_socket(socket_name, in_out="INPUT",
                                       socket_type=socket_type)
    for socket_name, socket_type in outputs:
        if (socket_name, "OUTPUT") not in existing:
            group.interface.new_socket(socket_name, in_out="OUTPUT",
                                       socket_type=socket_type)
    return group


def _group_io(group):
    nodes = group.nodes
    gi = next((n for n in nodes if n.bl_idname == "NodeGroupInput"), None)
    go = next((n for n in nodes if n.bl_idname == "NodeGroupOutput"), None)
    if gi is None:
        gi = nodes.new("NodeGroupInput")
        gi.location = (-1800, 0)
    if go is None:
        go = nodes.new("NodeGroupOutput")
        go.location = (900, 0)
    return gi, go


# â”€â”€ per-layer groups â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _uv_source(nodes, links, layer, tag, x, y):
    """Where this layer's textures get their coordinates from.

    Paint layers read one of the mesh's own UV maps; projected layers read the
    camera projection stored in a corner attribute.
    """
    if props.is_paint_layer(layer):
        node = _node(nodes, "ShaderNodeUVMap", f"{tag}_UV", (x, y))
        node.uv_map = layer.uv_map
        return node.outputs["UV"]
    node = _node(nodes, "ShaderNodeAttribute", f"{tag}_UV", (x, y))
    node.attribute_type = "GEOMETRY"
    node.attribute_name = layer.uv_attr
    return node.outputs["Vector"]


def _mask_socket(nodes, links, layer, tag, x_base, y, uv_socket):
    """Mask * opacity * live projection limits, as a float socket.

    The angle cutoff and distance limit are evaluated here rather than baked into
    the mask image, so they respond instantly and the mask itself stays a clean
    black-and-white occlusion map.
    """
    if layer.mask_image is None:
        source = _node(nodes, "ShaderNodeValue", f"{tag}_Opacity", (x_base + 400, y - 260))
        source.outputs[0].default_value = layer.opacity
        source_socket = source.outputs[0]
    else:
        # The mask lives in the object's own UV space, not the projection.
        if layer.mask_uv:
            mask_uv_node = _node(nodes, "ShaderNodeUVMap", f"{tag}_MaskUV", (x_base, y - 260))
            mask_uv_node.uv_map = layer.mask_uv
            mask_vector = mask_uv_node.outputs["UV"]
        else:
            mask_vector = uv_socket

        mask = _node(nodes, "ShaderNodeTexImage", f"{tag}_Mask", (x_base + 200, y - 260))
        mask.image = layer.mask_image
        mask.extension = "CLIP"
        mask.interpolation = "Smart"
        links.new(mask_vector, mask.inputs["Vector"])

        bw = _node(nodes, "ShaderNodeRGBToBW", f"{tag}_MaskBW", (x_base + 400, y - 260))
        links.new(mask.outputs["Color"], bw.inputs["Color"])

        multiply = _math(nodes, f"{tag}_Alpha", "MULTIPLY", (x_base + 580, y - 260),
                         {1: layer.opacity})
        links.new(bw.outputs["Val"], multiply.inputs[0])
        source_socket = multiply.outputs[0]

    modifiers = None
    if not props.is_paint_layer(layer):
        # A paint layer has no camera, so there is no angle or distance to apply.
        modifiers = _projection_limits(nodes, links, layer, tag, x_base + 200, y - 620)
    if modifiers is None:
        return source_socket

    combined = _math(nodes, f"{tag}_Limit", "MULTIPLY", (x_base + 1900, y - 400))
    links.new(source_socket, combined.inputs[0])
    links.new(modifiers, combined.inputs[1])
    return combined.outputs[0]


def _projection_limits(nodes, links, layer, tag, x, y):
    """Live angle cutoff x distance limit for one layer, as a float socket.

    Returns None when the layer needs neither, so the common case adds no nodes.
    """
    projection = layer.projection
    needs_angle = projection.angle_cut < 89.5
    needs_distance = projection.max_distance > 0.0
    if not needs_angle and not needs_distance:
        return None

    matrix = render.matrix_from_projection(projection)
    cam_loc = matrix.translation

    geom = _node(nodes, "ShaderNodeNewGeometry", f"{tag}_LimGeom", (x, y))
    cam = _const_vector(nodes, f"{tag}_LimCam", cam_loc, x, y - 180)

    to_frag = _vector_math(nodes, f"{tag}_LimVec", "SUBTRACT", (x + 180, y))
    links.new(geom.outputs["Position"], to_frag.inputs[0])
    links.new(cam.outputs["Vector"], to_frag.inputs[1])

    result = None

    if needs_angle:
        unit = _vector_math(nodes, f"{tag}_LimUnit", "NORMALIZE", (x + 360, y))
        links.new(to_frag.outputs["Vector"], unit.inputs[0])

        facing = _vector_math(nodes, f"{tag}_LimFacing", "DOT_PRODUCT", (x + 540, y))
        links.new(geom.outputs["Normal"], facing.inputs[0])
        links.new(unit.outputs["Vector"], facing.inputs[1])

        negate = _math(nodes, f"{tag}_LimNeg", "MULTIPLY", (x + 720, y), {1: -1.0})
        links.new(facing.outputs["Value"], negate.inputs[0])
        clamp = _math(nodes, f"{tag}_LimClamp", "MAXIMUM", (x + 880, y), {1: 0.0})
        links.new(negate.outputs[0], clamp.inputs[0])

        # Full weight until just inside the cutoff, then a short soft lip. Only
        # this lip varies; the rest of the surface is a clean 1.
        lip = 12.0
        cos_cut = float(np.cos(np.radians(min(projection.angle_cut, 89.9))))
        cos_full = float(np.cos(np.radians(max(1.0, projection.angle_cut - lip))))
        span = max(1e-4, cos_full - cos_cut)

        over = _math(nodes, f"{tag}_LimOver", "SUBTRACT", (x + 1040, y), {1: cos_cut})
        links.new(clamp.outputs[0], over.inputs[0])
        normalised = _math(nodes, f"{tag}_LimNorm", "DIVIDE", (x + 1200, y), {1: span})
        links.new(over.outputs[0], normalised.inputs[0])
        lo = _math(nodes, f"{tag}_LimLo", "MAXIMUM", (x + 1360, y), {1: 0.0})
        links.new(normalised.outputs[0], lo.inputs[0])
        hi = _math(nodes, f"{tag}_LimHi", "MINIMUM", (x + 1520, y), {1: 1.0})
        links.new(lo.outputs[0], hi.inputs[0])
        result = hi.outputs[0]

    if needs_distance:
        length = _vector_math(nodes, f"{tag}_LimLen", "LENGTH", (x + 360, y - 360))
        links.new(to_frag.outputs["Vector"], length.inputs[0])
        within = _math(nodes, f"{tag}_LimDist", "LESS_THAN", (x + 540, y - 360),
                       {1: float(projection.max_distance)})
        links.new(length.outputs["Value"], within.inputs[0])
        if result is None:
            result = within.outputs[0]
        else:
            both = _math(nodes, f"{tag}_LimBoth", "MULTIPLY", (x + 1700, y - 180))
            links.new(result, both.inputs[0])
            links.new(within.outputs[0], both.inputs[1])
            result = both.outputs[0]

    return result


def _is_neutral_hsv(layer) -> bool:
    """True when the colour controls would leave the image unchanged."""
    return (
        abs(layer.hue - 0.5) < 1e-4
        and abs(layer.saturation - 1.0) < 1e-4
        and abs(layer.value - 1.0) < 1e-4
    )


#: Object Info makes EEVEE Next fail to compile the material inside a nested
#: group, which renders the whole object magenta. Cycles is fine with it. The
#: gate is therefore switched off when EEVEE is the active engine, so a confined
#: layer degrades to "applies everywhere" instead of painting the mesh pink.
OWNER_GATE_ENABLED = True


def owner_gate_allowed(engine):
    """False for EEVEE, which cannot compile Object Info inside a nested group."""
    return "EEVEE" not in str(engine or "").upper()


def _owner_gate(nodes, links, layer, tag, socket, x, y):
    """Zero this layer out on objects it does not belong to.

    A UV-space mask cannot separate two meshes that share UVs, which is exactly
    what duplicated objects sharing a material do - they occupy the same texels.
    Object Index can tell them apart, so a layer flagged with an owner pass index
    is multiplied by (Object Index == owner) and disappears everywhere else.
    """
    owner = float(getattr(layer, "owner_id", 0.0) or 0.0)
    if owner <= 0.0 or not OWNER_GATE_ENABLED:
        return socket

    # The object ALPHA channel, not its RGB. Object Index breaks EEVEE Next's
    # shader compile inside a nested node group (Cycles tolerates it), and the
    # RGB of Object Colour is the viewport object colour - so the id lives in
    # alpha, which nothing visible uses and which stays engine-agnostic.
    info = _node(nodes, "ShaderNodeObjectInfo", f"{tag}_OwnerInfo", (x, y))
    compare = _math(nodes, f"{tag}_OwnerCmp", "COMPARE", (x + 170, y - 160))
    links.new(info.outputs["Alpha"], compare.inputs[0])
    compare.inputs[1].default_value = owner
    compare.inputs[2].default_value = 0.004  # ids step by 0.01

    gate = _math(nodes, f"{tag}_OwnerGate", "MULTIPLY", (x + 470, y - 160))
    links.new(socket, gate.inputs[0])
    links.new(compare.outputs[0], gate.inputs[1])
    return gate.outputs[0]


def _build_color_layer_group(group, layer, index):
    """One colour layer: texture -> optional HSV -> mix over what is below."""
    nodes, links = group.nodes, group.links
    gi, go = _group_io(group)
    tag = f"{PREFIX}C{index:02d}"
    x_base, y = -1700, 0

    uv_socket = _uv_source(nodes, links, layer, tag, x_base, y)

    # "Show coverage" swaps the layer's picture for flat red, so its footprint
    # on the mesh is obvious. The mask, opacity and projection limits still
    # apply, so what you see is exactly where the layer lands.
    coverage = bool(getattr(layer, "show_coverage", False))
    if coverage:
        red = _node(nodes, "ShaderNodeRGB", f"{tag}_Coverage", (x_base + 200, y + 120))
        red.outputs[0].default_value = (1.0, 0.04, 0.04, 1.0)
        color_socket = red.outputs[0]
        alpha = _mask_socket(nodes, links, layer, tag, x_base, y, uv_socket)
        factor_socket = alpha
    else:
        tex = _node(nodes, "ShaderNodeTexImage", f"{tag}_Tex", (x_base + 200, y + 120))
        tex.image = layer.source_image
        tex.extension = "CLIP"
        tex.interpolation = "Smart"
        links.new(uv_socket, tex.inputs["Vector"])

        # Only insert Hue/Saturation/Value when the controls are actually
        # off-centre, so an untouched layer is a byte-for-byte passthrough of the
        # image ComfyUI returned.
        color_socket = tex.outputs["Color"]
        if not _is_neutral_hsv(layer):
            hsv = _node(nodes, "ShaderNodeHueSaturation", f"{tag}_HSV", (x_base + 400, y + 120))
            hsv.inputs["Hue"].default_value = layer.hue
            hsv.inputs["Saturation"].default_value = layer.saturation
            hsv.inputs["Value"].default_value = layer.value
            links.new(color_socket, hsv.inputs["Color"])
            color_socket = hsv.outputs["Color"]

        alpha = _mask_socket(nodes, links, layer, tag, x_base, y, uv_socket)
        # Honour the texture's own alpha, so a blank paint layer stays
        # transparent instead of covering everything with its fill colour.
        src_alpha = _math(nodes, f"{tag}_SrcAlpha", "MULTIPLY", (x_base + 2100, y + 320))
        links.new(alpha, src_alpha.inputs[0])
        links.new(tex.outputs["Alpha"], src_alpha.inputs[1])
        factor_socket = src_alpha.outputs[0]

    factor_socket = _owner_gate(nodes, links, layer, tag, factor_socket,
                                x_base + 2300, y + 360)

    mix = _node(nodes, "ShaderNodeMix", f"{tag}_Mix", (x_base + 1040, y + 60))
    mix.data_type = "RGBA"
    mix.blend_type = layer.blend
    mix.clamp_result = True
    links.new(factor_socket, _in(mix, "Factor_Float"))
    links.new(gi.outputs["Below"], _in(mix, "A_Color"))
    links.new(color_socket, _in(mix, "B_Color"))
    links.new(_out(mix, "Result_Color"), go.inputs["Color"])


def _build_normal_layer_group(group, layer, index, blend_mode):
    """One normal layer: texture -> NormalMap -> whiteout/RNM blend."""
    nodes, links = group.nodes, group.links
    gi, go = _group_io(group)
    tag = f"{PREFIX}N{index:02d}"
    x_base, y = -1700, 0

    uv_socket = _uv_source(nodes, links, layer, tag, x_base, y)

    tex = _node(nodes, "ShaderNodeTexImage", f"{tag}_Tex", (x_base + 200, y + 120))
    tex.image = layer.source_image
    tex.extension = "CLIP"
    tex.interpolation = "Smart"
    # Colour space lives on the image datablock: a normal map must be Non-Color
    # or Blender gamma-decodes it and the lighting goes subtly wrong.
    links.new(uv_socket, tex.inputs["Vector"])

    color_socket = tex.outputs["Color"]
    if layer.normal_flip_y:
        sep = _node(nodes, "ShaderNodeSeparateColor", f"{tag}_Sep", (x_base + 380, y + 200))
        sep.mode = "RGB"
        links.new(color_socket, sep.inputs["Color"])
        flip = _math(nodes, f"{tag}_FlipY", "SUBTRACT", (x_base + 540, y + 200), {0: 1.0})
        links.new(sep.outputs["Green"], flip.inputs[1])
        comb = _node(nodes, "ShaderNodeCombineColor", f"{tag}_Comb", (x_base + 700, y + 200))
        comb.mode = "RGB"
        links.new(sep.outputs["Red"], comb.inputs["Red"])
        links.new(flip.outputs[0], comb.inputs["Green"])
        links.new(sep.outputs["Blue"], comb.inputs["Blue"])
        color_socket = comb.outputs["Color"]

    # The projected image is a normal map written in the *projection's* own
    # frame, so its tangent basis is the camera's, not any UV map's. Building
    # the frame from the stored camera also removes the dependence on a UV map,
    # which matters because Blender caps those at eight.
    normal_socket = _projection_normal(nodes, links, layer, tag, color_socket,
                                       x_base + 900, y + 120)

    alpha = _mask_socket(nodes, links, layer, tag, x_base, y, uv_socket)
    blended = _blend_normals(nodes, links, tag, blend_mode,
                             gi.outputs["Below"], normal_socket,
                             x_base + 1700, y + 120)

    alpha = _owner_gate(nodes, links, layer, tag, alpha, x_base + 2300, y - 200)

    mix = _node(nodes, "ShaderNodeMix", f"{tag}_Mix", (x_base + 2000, y + 60))
    mix.data_type = "VECTOR"
    links.new(alpha, _in(mix, "Factor_Float"))
    links.new(gi.outputs["Below"], _in(mix, "A_Vector"))
    links.new(blended, _in(mix, "B_Vector"))
    links.new(_out(mix, "Result_Vector"), go.inputs["Normal"])


def _projection_normal(nodes, links, layer, tag, color_socket, x, y):
    """Turn a projected tangent-space normal map into a world-space normal.

    Uses the camera's right/up as the tangent frame and the geometric normal as
    the base, which is the correct interpretation for a screen-space projection
    and needs no UV map.
    """
    matrix = render.matrix_from_projection(layer.projection)
    right, up, _forward = render.camera_basis(matrix)

    # colour -> vector in -1..1
    scale = _vector_math(nodes, f"{tag}_NScale", "SCALE", (x, y))
    scale.inputs["Scale"].default_value = 2.0
    links.new(color_socket, scale.inputs[0])

    bias = _vector_math(nodes, f"{tag}_NBias", "SUBTRACT", (x + 180, y))
    bias.inputs[1].default_value = (1.0, 1.0, 1.0)
    links.new(scale.outputs["Vector"], bias.inputs[0])

    sep = _node(nodes, "ShaderNodeSeparateXYZ", f"{tag}_NSep", (x + 360, y + 120))
    links.new(bias.outputs["Vector"], sep.inputs["Vector"])

    strength = _math(nodes, f"{tag}_NStr", "MULTIPLY", (x + 360, y - 120),
                     {1: layer.normal_strength})

    sx = _math(nodes, f"{tag}_NX", "MULTIPLY", (x + 540, y + 120))
    links.new(sep.outputs["X"], sx.inputs[0])
    links.new(strength.outputs[0], sx.inputs[1])

    sy = _math(nodes, f"{tag}_NY", "MULTIPLY", (x + 540, y - 20))
    links.new(sep.outputs["Y"], sy.inputs[0])
    links.new(strength.outputs[0], sy.inputs[1])

    t_vec = _const_vector(nodes, f"{tag}_NT", right, x + 540, y - 400)
    b_vec = _const_vector(nodes, f"{tag}_NB", up, x + 540, y - 540)
    geom = _node(nodes, "ShaderNodeNewGeometry", f"{tag}_NGeom", (x + 540, y - 200))

    tx = _vector_math(nodes, f"{tag}_NTX", "SCALE", (x + 740, y + 200))
    links.new(t_vec.outputs["Vector"], tx.inputs[0])
    links.new(sx.outputs[0], tx.inputs["Scale"])

    by = _vector_math(nodes, f"{tag}_NBY", "SCALE", (x + 740, y + 40))
    links.new(b_vec.outputs["Vector"], by.inputs[0])
    links.new(sy.outputs[0], by.inputs["Scale"])

    nz = _vector_math(nodes, f"{tag}_NNZ", "SCALE", (x + 740, y - 120))
    links.new(geom.outputs["Normal"], nz.inputs[0])
    links.new(sep.outputs["Z"], nz.inputs["Scale"])

    add = _vector_math(nodes, f"{tag}_NAdd", "ADD", (x + 940, y + 120))
    links.new(tx.outputs["Vector"], add.inputs[0])
    links.new(by.outputs["Vector"], add.inputs[1])

    add2 = _vector_math(nodes, f"{tag}_NAdd2", "ADD", (x + 1120, y + 80))
    links.new(add.outputs["Vector"], add2.inputs[0])
    links.new(nz.outputs["Vector"], add2.inputs[1])

    norm = _vector_math(nodes, f"{tag}_NNorm", "NORMALIZE", (x + 1300, y + 60))
    links.new(add2.outputs["Vector"], norm.inputs[0])
    return norm.outputs["Vector"]


def _blend_normals(nodes, links, tag, mode, base_socket, detail_socket, x, y):
    """Combine two normal vectors. Returns a vector socket."""
    if mode == "MIX":
        add = _vector_math(nodes, f"{tag}_Add", "ADD", (x, y))
        links.new(base_socket, add.inputs[0])
        links.new(detail_socket, add.inputs[1])
        half = _vector_math(nodes, f"{tag}_Half", "SCALE", (x + 180, y))
        half.inputs["Scale"].default_value = 0.5
        links.new(add.outputs["Vector"], half.inputs[0])
        return half.outputs["Vector"]

    if mode == "RNM":
        group = _rnm_group()
        node = _node(nodes, "ShaderNodeGroup", f"{tag}_RNM", (x, y))
        node.node_tree = group
        links.new(base_socket, node.inputs["Base"])
        links.new(detail_socket, node.inputs["Detail"])
        return node.outputs["Result"]

    # Whiteout: n = normalize(vec3(n1.xy + n2.xy, n1.z * n2.z))
    base_xyz = _node(nodes, "ShaderNodeSeparateXYZ", f"{tag}_BaseXYZ", (x, y + 200))
    detail_xyz = _node(nodes, "ShaderNodeSeparateXYZ", f"{tag}_DetXYZ", (x, y - 40))
    links.new(base_socket, base_xyz.inputs["Vector"])
    links.new(detail_socket, detail_xyz.inputs["Vector"])

    add_x = _math(nodes, f"{tag}_X", "ADD", (x + 180, y + 260))
    links.new(base_xyz.outputs["X"], add_x.inputs[0])
    links.new(detail_xyz.outputs["X"], add_x.inputs[1])

    add_y = _math(nodes, f"{tag}_Y", "ADD", (x + 180, y + 140))
    links.new(base_xyz.outputs["Y"], add_y.inputs[0])
    links.new(detail_xyz.outputs["Y"], add_y.inputs[1])

    mul_z = _math(nodes, f"{tag}_Z", "MULTIPLY", (x + 180, y + 20))
    links.new(base_xyz.outputs["Z"], mul_z.inputs[0])
    links.new(detail_xyz.outputs["Z"], mul_z.inputs[1])

    combine = _node(nodes, "ShaderNodeCombineXYZ", f"{tag}_CombXYZ", (x + 360, y + 140))
    links.new(add_x.outputs[0], combine.inputs["X"])
    links.new(add_y.outputs[0], combine.inputs["Y"])
    links.new(mul_z.outputs[0], combine.inputs["Z"])

    normalize = _vector_math(nodes, f"{tag}_Norm", "NORMALIZE", (x + 540, y + 140))
    links.new(combine.outputs["Vector"], normalize.inputs[0])
    return normalize.outputs["Vector"]


def _rnm_group():
    """Reoriented Normal Mapping as a reusable node group."""
    group = _reset_group(
        _RNM_GROUP,
        [("Base", "NodeSocketVector"), ("Detail", "NodeSocketVector")],
        [("Result", "NodeSocketVector")],
    )
    nodes, links = group.nodes, group.links
    gi, go = _group_io(group)

    def vm(op, x, y, label):
        node = nodes.new("ShaderNodeVectorMath")
        node.operation = op
        node.location = (x, y)
        node.label = label
        return node

    # t = n1 * 2 + (-1, -1, 0)
    t_scale = vm("SCALE", -600, 200, "n1 * 2")
    t_scale.inputs["Scale"].default_value = 2.0
    links.new(gi.outputs["Base"], t_scale.inputs[0])
    t_add = vm("ADD", -420, 200, "t")
    t_add.inputs[1].default_value = (-1.0, -1.0, 0.0)
    links.new(t_scale.outputs["Vector"], t_add.inputs[0])

    # u = n2 * (-2, -2, 2) + (1, 1, -1)
    u_scale = vm("MULTIPLY", -600, -200, "n2 * (-2,-2,2)")
    u_scale.inputs[1].default_value = (-2.0, -2.0, 2.0)
    links.new(gi.outputs["Detail"], u_scale.inputs[0])
    u_add = vm("ADD", -420, -200, "u")
    u_add.inputs[1].default_value = (1.0, 1.0, -1.0)
    links.new(u_scale.outputs["Vector"], u_add.inputs[0])

    dot = vm("DOT_PRODUCT", -200, -320, "dot(t, u)")
    links.new(t_add.outputs["Vector"], dot.inputs[0])
    links.new(u_add.outputs["Vector"], dot.inputs[1])

    t_dot = vm("SCALE", -20, 200, "t * dot")
    links.new(t_add.outputs["Vector"], t_dot.inputs[0])
    links.new(dot.outputs["Value"], t_dot.inputs["Scale"])

    sep = nodes.new("ShaderNodeSeparateXYZ")
    sep.location = (-200, 420)
    links.new(t_add.outputs["Vector"], sep.inputs["Vector"])

    u_tz = vm("SCALE", -20, -40, "u * t.z")
    links.new(u_add.outputs["Vector"], u_tz.inputs[0])
    links.new(sep.outputs["Z"], u_tz.inputs["Scale"])

    sub = vm("SUBTRACT", 200, 80, "")
    links.new(t_dot.outputs["Vector"], sub.inputs[0])
    links.new(u_tz.outputs["Vector"], sub.inputs[1])

    normalize = vm("NORMALIZE", 400, 80, "")
    links.new(sub.outputs["Vector"], normalize.inputs[0])
    links.new(normalize.outputs["Vector"], go.inputs["Result"])
    return group


# â”€â”€ the mother group â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _build_stack(group, color_layers, normal_layers, normal_blend):
    nodes, links = group.nodes, group.links
    gi, go = _group_io(group)

    # â”€â”€ colour chain, bottom to top â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # The pre-mask is a working overlay, not part of the artwork, so it is
    # always composited last â€” otherwise every new projection would land on top
    # of it and hide the red you are painting with.
    ordered = [l for l in color_layers if l.role == "LAYER"]
    ordered += [l for l in color_layers if l.role != "LAYER"]

    cursor = gi.outputs["Base Color"]
    for index, layer in enumerate(ordered):
        layer_group = _reset_group(
            f"{COLOR_LAYER_GROUP}_{index}",
            [("Below", "NodeSocketColor")],
            [("Color", "NodeSocketColor")],
        )
        _build_color_layer_group(layer_group, layer, index)
        node = _node(nodes, "ShaderNodeGroup", f"{PREFIX}LayerC{index:02d}",
                     (-800, 200 - index * 300))
        node.node_tree = layer_group
        node.label = layer.name
        links.new(cursor, node.inputs["Below"])
        cursor = node.outputs["Color"]
    links.new(cursor, go.inputs["Color"])

    # â”€â”€ normal chain, bottom to top â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    cursor = gi.outputs["Base Normal"]
    for index, layer in enumerate(normal_layers):
        layer_group = _reset_group(
            f"{NORMAL_LAYER_GROUP}_{index}",
            [("Below", "NodeSocketVector")],
            [("Normal", "NodeSocketVector")],
        )
        _build_normal_layer_group(layer_group, layer, index, normal_blend)
        node = _node(nodes, "ShaderNodeGroup", f"{PREFIX}LayerN{index:02d}",
                     (400, 200 - index * 300))
        node.node_tree = layer_group
        node.label = layer.name
        links.new(cursor, node.inputs["Below"])
        cursor = node.outputs["Normal"]
    links.new(cursor, go.inputs["Normal"])


# â”€â”€ entry point â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def rebuild(material, color_layers, normal_layers, normal_blend="WHITEOUT"):
    """Rebuild the whole add-on contribution into ``material``.

    ``color_layers`` and ``normal_layers`` are ordered bottom-to-top.
    """
    if material is None:
        return
    if not material.use_nodes:
        material.use_nodes = True
    if material.node_tree is None:
        return

    _ensure_scaffold(material)
    clear(material)

    nodes = material.node_tree.nodes
    links = material.node_tree.links

    def original(marker_name):
        marker = nodes.get(marker_name)
        if marker is not None and marker.inputs[0].links:
            return marker.inputs[0].links[0].from_socket
        return None

    original_color = original(_ORIGINAL_BASECOLOR)
    original_normal = original(_ORIGINAL_NORMAL)
    bsdf = find_principled(material)

    if not color_layers and not normal_layers:
        _break(links, bsdf.inputs["Base Color"])
        if original_color is not None:
            links.new(original_color, bsdf.inputs["Base Color"])
        _break(links, bsdf.inputs["Normal"])
        if original_normal is not None:
            links.new(original_normal, bsdf.inputs["Normal"])
        return

    stack = _reset_group(
        STACK_GROUP,
        [("Base Color", "NodeSocketColor"), ("Base Normal", "NodeSocketVector")],
        [("Color", "NodeSocketColor"), ("Normal", "NodeSocketVector")],
    )
    _build_stack(stack, color_layers, normal_layers, normal_blend)

    node = _node(nodes, "ShaderNodeGroup", f"{PREFIX}StackNode", (-600, 0))
    node.node_tree = stack
    node.label = "Projection Layers"

    grey = _node(nodes, "ShaderNodeRGB", f"{PREFIX}BaseGrey", (-900, 260))
    grey.outputs[0].default_value = (0.5, 0.5, 0.5, 1.0)
    flat = _node(nodes, "ShaderNodeRGB", f"{PREFIX}FlatNormal", (-900, -260))
    flat.outputs[0].default_value = (*FLAT_NORMAL, 1.0)

    links.new(original_color or grey.outputs[0], node.inputs["Base Color"])
    normal_start = original_normal
    if normal_start is None or normal_start.type != "VECTOR":
        normal_start = flat.outputs[0]
    links.new(normal_start, node.inputs["Base Normal"])

    _break(links, bsdf.inputs["Base Color"])
    if color_layers:
        links.new(node.outputs["Color"], bsdf.inputs["Base Color"])
    elif original_color is not None:
        links.new(original_color, bsdf.inputs["Base Color"])

    _break(links, bsdf.inputs["Normal"])
    if normal_layers:
        links.new(node.outputs["Normal"], bsdf.inputs["Normal"])
    elif original_normal is not None:
        links.new(original_normal, bsdf.inputs["Normal"])
