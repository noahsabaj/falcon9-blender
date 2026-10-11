"""Render a built shot (resumable: existing frames are skipped).

blender --background film/shots/<id>.blend --python film/render_shot.py --
        [--preview] [--still F] [--step N] [--samples N] [--scale PCT] [--frames A-B]
Final frames go to film/frames/<id>/ (existing frames kept); previews and stills go to
film/preview/<id>/ and are overwritten.
"""
import argparse
import os
import sys
import time

import bpy

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


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
            if kind == "OPTIX":
                bpy.context.scene.cycles.denoiser = "OPTIX"
            bpy.context.scene.render.compositor_device = "GPU"   # same frames, 0.2 s less per frame than on the CPU
            return kind
    bpy.context.scene.cycles.device = "CPU"
    return "CPU"


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--still", type=int, default=0)
    ap.add_argument("--step", type=int, default=0)
    ap.add_argument("--samples", type=int, default=0)
    ap.add_argument("--scale", type=int, default=0)
    ap.add_argument("--frames", default="")
    ap.add_argument("--part", default="0/1", help="I/N: render every Nth frame from offset I, so N processes "
                                                  "can share a shot")
    a = ap.parse_args(argv)
    scn = bpy.context.scene
    shot = os.path.splitext(os.path.basename(bpy.data.filepath))[0]
    print("[render] device", setup_gpu(), flush=True)
    scn.render.use_persistent_data = True
    if a.still:
        if a.scale:
            scn.render.resolution_percentage = a.scale
        if a.samples:
            scn.cycles.samples = a.samples
        scn.frame_set(a.still)
        scn.render.filepath = os.path.join(ROOT, "film", "preview", shot, f"still_{a.still:04d}.png")
        t = time.time()
        bpy.ops.render.render(write_still=True)
        print(f"[render] {shot} still {a.still} in {time.time() - t:.1f}s", flush=True)
        return
    if a.frames:
        f0, f1 = (int(v) for v in a.frames.split("-"))
        scn.frame_start, scn.frame_end = f0, f1
    if a.preview:
        scn.render.use_overwrite = True
        scn.render.use_placeholder = False
        scn.render.resolution_percentage = a.scale or 50
        scn.cycles.samples = a.samples or max(16, scn.cycles.samples // 4)
        scn.frame_step = a.step or 8
        scn.render.filepath = os.path.join(ROOT, "film", "preview", shot, "f_")
    else:
        if a.scale:
            scn.render.resolution_percentage = a.scale
        if a.samples:
            scn.cycles.samples = a.samples
        part, parts = (int(v) for v in a.part.split("/"))
        scn.frame_start += part * (a.step or 1)
        scn.frame_step = (a.step or 1) * parts
        scn.render.filepath = os.path.join(ROOT, "film", "frames", shot, "f_")
        # an interrupted render leaves an empty placeholder, which would be skipped: clear this part's
        out = os.path.dirname(scn.render.filepath)
        for f in range(scn.frame_start, scn.frame_end + 1, scn.frame_step):
            p = os.path.join(out, f"f_{f:04d}.png")
            if os.path.exists(p) and os.path.getsize(p) == 0:
                os.remove(p)
    t = time.time()
    bpy.ops.render.render(animation=True)
    n = len(range(scn.frame_start, scn.frame_end + 1, scn.frame_step))
    print(f"[render] {shot}: {n} frames in {time.time() - t:.1f}s", flush=True)


if __name__ == "__main__":
    main()
