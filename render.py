"""Render verification views from the saved model.

blender --background falcon9_block5.blend --python render.py -- --outdir renders [--cams Cam_Full,Cam_Base]
        [--samples 64] [--scale 100] [--engine CYCLES|BLENDER_EEVEE]

Per-camera custom props: res_x/res_y, deployed (rig at 1), ground (show ground plane),
hide (object-name prefixes), cut ("azimuth,half-angle" wedge section, undone after the
render), cut_keep (prefixes left whole), cut_z ("z0,z1": only meshes overlapping are cut).
"""
import argparse
import math
import os
import sys
import time

import bmesh
import bpy
from mathutils import Vector


def args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=os.path.join(os.path.dirname(bpy.data.filepath), "renders"))
    ap.add_argument("--cams", default="")
    ap.add_argument("--samples", type=int, default=64)
    ap.add_argument("--scale", type=int, default=100)
    ap.add_argument("--engine", default="CYCLES")
    ap.add_argument("--hide", default="", help="comma-separated object name prefixes to hide (cutaways)")
    return ap.parse_args(argv)


def setup_gpu():
    try:
        prefs = bpy.context.preferences.addons["cycles"].preferences
    except KeyError:
        return "CPU"
    for kind in ("OPTIX", "CUDA", "HIP", "ONEAPI", "METAL"):
        try:
            prefs.compute_device_type = kind
        except TypeError:
            continue
        prefs.get_devices()
        devs = [d for d in prefs.devices if d.type == kind]
        if devs:
            for d in prefs.devices:
                d.use = d.type == kind
            bpy.context.scene.cycles.device = "GPU"
            return kind + ": " + ", ".join(d.name for d in devs)
    bpy.context.scene.cycles.device = "CPU"
    return "CPU"


def wedge_cut(spec, keep, z_range):
    """Section view: temporarily remove the wedge (centre azimuth, half angle in degrees)
    about the vehicle axis from every mesh overlapping z_range. Returns swaps to undo."""
    c, w = (math.radians(float(v)) for v in spec.split(","))
    n0 = Vector((-math.sin(c - w), math.cos(c - w), 0.0))      # boundary normals, into the wedge
    n1 = Vector((math.sin(c + w), -math.cos(c + w), 0.0))
    z0, z1 = z_range
    swaps = []
    for ob in bpy.data.objects:
        if ob.type != "MESH" or ob.hide_render or any(ob.name.startswith(k) for k in keep):
            continue
        mw = ob.matrix_world
        zs = [(mw @ Vector(b)).z for b in ob.bound_box]
        if max(zs) < z0 or min(zs) > z1:
            continue
        orig, me = ob.data, ob.data.copy()
        bm = bmesh.new()
        bm.from_mesh(me)
        o_l, m3t = mw.inverted() @ Vector((0.0, 0.0, 0.0)), mw.to_3x3().transposed()
        for n in (n0, n1):
            bmesh.ops.bisect_plane(bm, geom=bm.verts[:] + bm.edges[:] + bm.faces[:], dist=1e-6,
                                   plane_co=o_l, plane_no=(m3t @ n).normalized())
        dead = []
        for f in bm.faces:
            p = mw @ f.calc_center_median()
            if n0.dot(p) > 1e-6 and n1.dot(p) > 1e-6:
                dead.append(f)
        bmesh.ops.delete(bm, geom=dead, context="FACES")
        bm.to_mesh(me)
        bm.free()
        ob.data = me
        swaps.append((ob, orig, me))
    return swaps


def main():
    a = args()
    scn = bpy.context.scene
    scn.render.engine = a.engine
    if a.engine == "CYCLES":
        print("[render] device", setup_gpu(), flush=True)
        scn.cycles.samples = a.samples
        scn.cycles.use_denoising = True
    scn.render.image_settings.file_format = "PNG"
    scn.render.resolution_percentage = a.scale
    os.makedirs(a.outdir, exist_ok=True)

    hide = [h for h in a.hide.split(",") if h]
    for ob in bpy.data.objects:
        if any(ob.name.startswith(h) for h in hide):
            ob.hide_render = True

    rig = bpy.data.objects.get("F9_Rig")
    ground = bpy.data.objects.get("F9_Ground")
    cams = [o for o in bpy.data.objects if o.type == "CAMERA"]
    if a.cams:
        want = a.cams.split(",")
        cams = [o for o in cams if o.name in want]
    for cam in sorted(cams, key=lambda o: o.name):
        t = time.time()
        scn.camera = cam
        scn.render.resolution_x = int(cam.get("res_x", 1600))
        scn.render.resolution_y = int(cam.get("res_y", 1000))
        dep = 1.0 if cam.get("deployed", False) else 0.0
        if rig is not None:
            rig["leg_deploy"] = dep
            rig["gridfin_deploy"] = dep
        if ground is not None:
            ground.hide_render = not bool(cam.get("ground", False))
        cam_hide = [h for h in str(cam.get("hide", "")).split(",") if h]
        hidden = []
        for ob in bpy.data.objects:
            if cam_hide and any(ob.name.startswith(h) for h in cam_hide) and not ob.hide_render:
                ob.hide_render = True
                hidden.append(ob)
        bpy.context.view_layer.update()
        swaps = []
        if cam.get("cut"):
            keep = [k for k in str(cam.get("cut_keep", "")).split(",") if k]
            z_range = [float(v) for v in str(cam.get("cut_z", "-1000,1000")).split(",")]
            swaps = wedge_cut(str(cam["cut"]), keep, z_range)
        scn.render.filepath = os.path.join(a.outdir, cam.name + ".png")
        bpy.ops.render.render(write_still=True)
        for ob in hidden:
            ob.hide_render = False
        for ob, orig, me in swaps:
            ob.data = orig
            bpy.data.meshes.remove(me)
        print(f"[render] {cam.name} {time.time() - t:.1f}s", flush=True)


if __name__ == "__main__":
    main()
