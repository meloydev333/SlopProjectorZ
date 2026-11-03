"""Does a generated layer survive save + reopen?

The smoke test never touched this, and that is exactly why masks "vanished":
every assertion ran in the same session that created them, so nothing ever
exercised Blender's save path.

Two phases, two Blender invocations:

    blender -b -P persistence_test.py -- write
    blender -b <out.blend> -P persistence_test.py -- read

Phase 1 runs the real pipeline.apply() with a real PNG, paints the mask the way
a brush stroke would, and saves. Phase 2 reopens that file and checks every
datablock the layer depends on, then rebuilds the node tree and checks the mask
is still wired into the shader.
"""
import sys
import json
import os
import time

sys.path.insert(0, r"C:\Users\Csabi\Desktop\Addons")

import bpy
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
STATE = os.path.join(HERE, "_persistence_state.json")
BLEND = os.path.join(HERE, "_persistence.blend")

FAIL = []


def check(label, ok, detail=""):
    print("%s | %s%s" % ("PASS" if ok else "FAIL", label,
                         " | " + detail if detail else ""))
    if not ok:
        FAIL.append(label)


def enable():
    if not hasattr(bpy.types.Material, "projlayer"):
        bpy.ops.preferences.addon_enable(module="projlayer")
    from projlayer.core import pipeline
    from projlayer import props
    return pipeline, props


# --------------------------------------------------------------------------
# write
# --------------------------------------------------------------------------
def phase_write():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    pipeline, props = enable()
    from projlayer.core import render as R

    bpy.ops.mesh.primitive_uv_sphere_add(location=(0, 0, 0))
    obj = bpy.context.active_object
    obj.name = "PL_PersistSphere"
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.uv.smart_project()
    bpy.ops.object.mode_set(mode="OBJECT")

    mat = bpy.data.materials.new("PL_PersistMat")
    mat.use_nodes = True
    obj.data.materials.append(mat)

    cam = R.ensure_camera(bpy.context)
    bpy.context.scene.camera = cam
    cam.location = (0, -8, 0)
    cam.rotation_euler = (1.5708, 0, 0)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)

    # A real PNG, like the one ComfyUI returns.
    src = bpy.data.images.new("PL_probe", 64, 64, alpha=False)
    src.pixels.foreach_set(
        np.tile(np.array([0.4, 0.6, 0.8, 1.0], dtype=np.float32), 64 * 64))
    png = R.temp_path("persist_generated.png")
    R.save_image_as_png(src, png)
    with open(png, "rb") as fh:
        data = fh.read()

    payload = pipeline.Payload()
    payload.obj_name = obj.name
    payload.material_name = mat.name
    payload.camera_name = cam.name
    payload.width = payload.height = 64
    payload.mode = "COLOR"
    payload.prompt = "persist probe"
    payload.started = time.time()
    payload.feather = 0

    layer = pipeline.apply(bpy.context, payload, [data])
    check("apply created a layer", layer is not None)
    check("the layer has a mask", layer.mask_image is not None)
    check("the layer has a source image", layer.source_image is not None)
    mask_name = layer.mask_image.name
    source_name = layer.source_image.name
    check("the mask really has packed pixels after apply",
          layer.mask_image.packed_file is not None,
          "packed=%s source=%s" % (layer.mask_image.packed_file is not None,
                                   layer.mask_image.source))

    # A brush stroke: paint the upper-left quadrant far darker, exactly the kind
    # of edit that used to be thrown away because the image was already packed.
    h = layer.mask_image.size[1]
    w = layer.mask_image.size[0]
    buf = np.ones((h, w, 4), dtype=np.float32)
    buf[:, :, :3] = 1.0
    buf[: h // 2, : w // 2, :3] = 0.25
    buf[:, :, 3] = 1.0
    layer.mask_image.pixels.foreach_set(buf.ravel())
    layer.mask_image.update()
    # What _activate_paint_target does when it hands the image to the brush.
    layer.mask_image["pl_needs_pack"] = True

    info = {
        "mask": mask_name,
        "source": source_name,
        "material": mat.name,
        "obj": obj.name,
        "mask_uv": layer.mask_uv,
        "uv_attr": layer.uv_attr,
        "layers": len(mat.projlayer.color_layers),
    }
    with open(STATE, "w") as fh:
        json.dump(info, fh)

    for name in (mask_name, source_name):
        im = bpy.data.images.get(name)
        print("BEFORE SAVE %-14s users=%d fake=%s packed=%s dirty=%s"
              % (name, im.users, im.use_fake_user,
                 im.packed_file is not None, im.is_dirty))
    print("BEFORE SAVE PL images: %s"
          % sorted(i.name for i in bpy.data.images if i.name.startswith("PL_")))

    bpy.ops.wm.save_as_mainfile(filepath=BLEND)
    print("saved %s" % BLEND)


# --------------------------------------------------------------------------
# read
# --------------------------------------------------------------------------
def phase_read():
    pipeline, props = enable()
    with open(STATE) as fh:
        want = json.load(fh)

    pl_images = sorted(i.name for i in bpy.data.images if i.name.startswith("PL_"))
    print("AFTER  OPEN PL images: %s" % pl_images)
    print("AFTER  OPEN all images: %s" % sorted(i.name for i in bpy.data.images))

    check("the generated source image is still in the file",
          bpy.data.images.get(want["source"]) is not None,
          "present=%s" % (bpy.data.images.get(want["source"]) is not None))
    check("the mask image is still in the file",
          bpy.data.images.get(want["mask"]) is not None,
          "present=%s" % (bpy.data.images.get(want["mask"]) is not None))

    mat = bpy.data.materials.get(want["material"])
    check("the material survived", mat is not None)
    if mat is None:
        return
    stack = mat.projlayer
    check("the layer stack survived", len(stack.color_layers) == want["layers"],
          "layers=%d want=%d" % (len(stack.color_layers), want["layers"]))
    if not len(stack.color_layers):
        return
    layer = stack.color_layers[0]
    check("the layer still points at its source image",
          layer.source_image is not None
          and layer.source_image.name == want["source"],
          repr(layer.source_image.name if layer.source_image else None))
    check("the layer still points at its mask",
          layer.mask_image is not None,
          repr(layer.mask_image.name if layer.mask_image else None))
    check("the mask UV map name survived", layer.mask_uv == want["mask_uv"],
          "%r vs %r" % (layer.mask_uv, want["mask_uv"]))
    # cam_matrix is a subtype="MATRIX" property, so iterating it yields rows.
    magnitude = sum(abs(float(v)) for row in layer.projection.cam_matrix
                    for v in row)
    check("the projection matrix survived", magnitude > 0.0,
          "sum=%.3f" % magnitude)

    mask = layer.mask_image
    if mask is not None:
        check("the mask is packed",
              mask.packed_file is not None,
              "packed=%s source=%s" % (mask.packed_file is not None, mask.source))
        px = np.empty(mask.size[0] * mask.size[1] * 4, dtype=np.float32)
        mask.pixels.foreach_get(px)
        px = px.reshape(mask.size[1], mask.size[0], 4)
        corner = float(px[0, 0, 0])
        centre = float(px[mask.size[1] // 2 + 4, mask.size[0] // 2 + 4, 0])
        check("the painted stroke survived the save",
              abs(corner - 0.25) < 0.02,
              "corner=%.3f (want 0.25)  centre=%.3f" % (corner, centre))

    # The shader has to pick the mask back up after a restart, or the layer
    # renders unrestricted even though the datablocks are all present.
    bpy.context.view_layer.objects.active = bpy.data.objects.get(want["obj"])
    pipeline.rebuild(bpy.context)
    tree = mat.node_tree

    # The texture nodes live inside the per-layer node groups, so search one
    # level down as well as the top level.
    def all_nodes(node_tree):
        found = list(node_tree.nodes)
        for node in found:
            if node.type == "GROUP" and node.node_tree is not None:
                found.extend(all_nodes(node.node_tree))
        return found

    mask_nodes = [n for n in all_nodes(tree)
                  if n.type == "TEX_IMAGE" and n.image is not None
                  and n.image.name == want["mask"]]
    check("the rebuilt shader re-binds the mask image", len(mask_nodes) >= 1,
          "nodes=%d" % len(mask_nodes))
    if mask_nodes:
        check("the mask texture is clipped, not repeated",
              mask_nodes[0].extension == "CLIP",
              mask_nodes[0].extension)
    groups = [n for n in tree.nodes if n.type == "GROUP"]
    check("the stack node group is intact after reload",
          all(n.node_tree is not None for n in groups) and len(groups) >= 1,
          "groups=%d missing=%d" % (
              len(groups),
              sum(1 for n in groups if n.node_tree is None)))


if sys.argv[-1] == "write":
    phase_write()
else:
    phase_read()

print("")
if FAIL:
    print("FAILURES (%d): %s" % (len(FAIL), FAIL))
    print("PERSISTENCE: FAILED")
else:
    print("PERSISTENCE: ALL CHECKS PASSED")
