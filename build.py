"""Build the 1:1 Falcon 9 Block 5 model and save it.

blender --background --factory-startup --python build.py -- --out falcon9_block5.blend
"""
import argparse
import importlib
import os
import sys
import time

import bpy

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import f9  # noqa: E402
from f9 import (dims, engines, fairing, gridfins, internals, interstage, legs, livery,  # noqa: E402
                materials, mvac, rig, scene, stage1, stage2, util)

for mod in (dims, util, materials, stage1, engines, legs, gridfins, interstage, stage2, mvac, fairing,
            livery, internals, scene, rig):
    importlib.reload(mod)

U = util


def args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "falcon9_block5.blend"))
    return ap.parse_args(argv)


def clear_scene():
    for ob in list(bpy.data.objects):
        bpy.data.objects.remove(ob, do_unlink=True)
    for coll in list(bpy.data.collections):
        bpy.data.collections.remove(coll)
    for block in (bpy.data.meshes, bpy.data.materials, bpy.data.lights, bpy.data.cameras, bpy.data.worlds):
        for item in list(block):
            block.remove(item)


def main():
    a = args()
    t0 = time.time()
    clear_scene()
    M = materials.library()

    master = U.collection("Falcon9")
    keys = ["Stage1", "Engines", "Legs", "GridFins", "Interstage", "Stage2", "MVac", "Fairing", "Livery",
            "Internals"]
    colls = {k: U.collection("F9_" + k, master) for k in keys}
    colls["Master"] = master
    scene_coll = U.collection("F9_Scene")

    root = U.empty("F9_Root", master, size=6.0, kind="CIRCLE")
    s1 = U.empty("S1", colls["Stage1"], parent=root, size=4.0)
    s2 = U.empty("S2", colls["Stage2"], parent=root, size=4.0)
    fair = U.empty("Fairing", colls["Fairing"], parent=s2, size=4.0)

    def step(label, fn, *fa):
        t = time.time()
        out = fn(*fa)
        print(f"[f9] {label:<12s} {time.time() - t:6.2f}s", flush=True)
        return out

    step("stage1", stage1.build, colls, M, s1)
    step("engines", engines.build, colls, M, s1)
    leg_objs = step("legs", legs.build, colls, M, s1)
    fin_objs = step("gridfins", gridfins.build, colls, M, s1)
    step("interstage", interstage.build, colls, M, s1, s2)
    step("stage2", stage2.build, colls, M, s2)
    step("mvac", mvac.build, colls, M, s2)
    step("fairing", fairing.build, colls, M, fair)
    step("livery", livery.build, colls, M, s1)
    step("internals", internals.build, colls, M, s1, s2)
    cams, ground = step("scene", scene.build, scene_coll)
    step("rig", rig.build, colls, M, root, s1, leg_objs, fin_objs)

    bpy.context.view_layer.update()
    nobj = sum(1 for o in bpy.data.objects if o.type == "MESH")
    ntri = 0
    for o in bpy.data.objects:
        if o.type == "MESH":
            me = o.data
            me.calc_loop_triangles()
            ntri += len(me.loop_triangles)
    print(f"[f9] meshes={nobj} triangles={ntri} build={time.time() - t0:.1f}s", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(a.out), compress=True)
    print(f"[f9] saved {a.out}", flush=True)


if __name__ == "__main__":
    main()
