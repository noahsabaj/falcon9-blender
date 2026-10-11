"""Preview renders for the film assets.

    blender --background --factory-startup --python film/test_assets.py -- --asset droneship
    blender --background --factory-startup --python film/test_assets.py -- --asset payload
    blender --background --factory-startup --python film/test_assets.py -- --asset payload_fairing
        [--samples 64] [--res 1280x720] [--views wide,deck,stern] [--outdir film/previews]

droneship        empty scene + calm dark water + sunset sky/sun -> previews/droneship_<view>.png
payload          the Starlink stack alone under studio lights   -> previews/payload_<view>.png
payload_fairing  opens falcon9_block5.blend (never saved), adds the stack inside the fairing
                 with one fairing half hidden                     -> previews/payload_fairing*.png
"""
import argparse
import importlib
import math
import os
import sys
import time

import bmesh
import bpy
from mathutils import Vector

FILM = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(FILM)
for p in (ROOT, FILM):
    if p not in sys.path:
        sys.path.insert(0, p)

VEHICLE_BLEND = os.path.join(ROOT, "falcon9_block5.blend")


def args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", default="droneship")
    ap.add_argument("--samples", type=int, default=64)
    ap.add_argument("--res", default="1280x720")
    ap.add_argument("--views", default="")
    ap.add_argument("--outdir", default=os.path.join(FILM, "previews"))
    ap.add_argument("--flood", type=float, default=None)
    return ap.parse_args(argv)


# ---------------------------------------------------------------------------------
# common
# ---------------------------------------------------------------------------------
def setup_render(samples, res):
    scn = bpy.context.scene
    scn.render.engine = "CYCLES"
    try:
        prefs = bpy.context.preferences.addons["cycles"].preferences
        for kind in ("OPTIX", "CUDA"):
            try:
                prefs.compute_device_type = kind
            except TypeError:
                continue
            prefs.get_devices()
            devs = [d for d in prefs.devices if d.type == kind]
            if devs:
                for d in prefs.devices:
                    d.use = d.type == kind
                scn.cycles.device = "GPU"
                print("[test] device", kind, [d.name for d in devs], flush=True)
                break
    except KeyError:
        pass
    scn.cycles.samples = samples
    scn.cycles.use_denoising = True
    scn.cycles.max_bounces = 8
    scn.cycles.sample_clamp_indirect = 8.0
    w, h = (int(v) for v in res.lower().split("x"))
    scn.render.resolution_x, scn.render.resolution_y = w, h
    scn.render.resolution_percentage = 100
    scn.render.image_settings.file_format = "PNG"
    scn.view_settings.view_transform = "AgX"
    try:
        scn.view_settings.look = "AgX - Medium High Contrast"
    except TypeError:
        pass
    scn.frame_set(1)


def clear_scene():
    for ob in list(bpy.data.objects):
        bpy.data.objects.remove(ob, do_unlink=True)
    for block in (bpy.data.meshes, bpy.data.materials, bpy.data.lights, bpy.data.cameras):
        for item in list(block):
            block.remove(item)


def camera(name, loc, target, lens=35.0, coll=None):
    cd = bpy.data.cameras.new(name)
    cd.lens = lens
    cd.sensor_width = 36.0
    cd.clip_start = 0.1
    cd.clip_end = 20000.0
    ob = bpy.data.objects.new(name, cd)
    (coll or bpy.context.scene.collection).objects.link(ob)
    ob.location = Vector(loc)
    ob.rotation_euler = (Vector(target) - Vector(loc)).to_track_quat("-Z", "Y").to_euler()
    return ob


def render(cam, path):
    scn = bpy.context.scene
    scn.camera = cam
    scn.render.filepath = path
    t = time.time()
    bpy.ops.render.render(write_still=True)
    print(f"[test] wrote {path} ({time.time() - t:.1f}s)", flush=True)


def tri_count(objs):
    n = 0
    for ob in objs:
        if ob.type == "MESH":
            ob.data.calc_loop_triangles()
            n += len(ob.data.loop_triangles)
    return n


def sun_dir(az_deg, el_deg):
    el, az = math.radians(el_deg), math.radians(az_deg)
    return Vector((math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)))


def add_sun(name, az, el, energy, color, angle=0.53, coll=None):
    ld = bpy.data.lights.new(name, "SUN")
    ld.energy = energy
    ld.angle = math.radians(angle)
    ld.color = color
    ob = bpy.data.objects.new(name, ld)
    (coll or bpy.context.scene.collection).objects.link(ob)
    d = sun_dir(az, el)
    ob.rotation_euler = d.to_track_quat("Z", "Y").to_euler()
    ob.location = d * 200.0
    return ob


def sky_world(sun_az, sun_el, strength=0.35, name="Preview_Sky", air=1.0, aerosol=1.6, ozone=1.0):
    w = bpy.data.worlds.new(name)
    bpy.context.scene.world = w
    w.use_nodes = True
    nt = w.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    out = nt.nodes.new("ShaderNodeOutputWorld")
    bg = nt.nodes.new("ShaderNodeBackground")
    sky = nt.nodes.new("ShaderNodeTexSky")
    sky.sky_type = "MULTIPLE_SCATTERING"
    sky.sun_disc = False
    sky.sun_elevation = math.radians(sun_el)
    sky.sun_rotation = math.radians(sun_az - 90.0)
    sky.altitude = 20.0
    sky.air_density = air
    sky.aerosol_density = aerosol
    sky.ozone_density = ozone
    bg.inputs["Strength"].default_value = strength
    nt.links.new(sky.outputs["Color"], bg.inputs["Color"])
    nt.links.new(bg.outputs["Background"], out.inputs["Surface"])
    return w


# ---------------------------------------------------------------------------------
# droneship
# ---------------------------------------------------------------------------------
def water_material():
    m = bpy.data.materials.new("Preview_Water")
    m.use_nodes = True
    nt = m.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    b = nt.nodes.new("ShaderNodeBsdfPrincipled")
    nt.links.new(b.outputs["BSDF"], out.inputs["Surface"])
    b.inputs["Base Color"].default_value = (0.004, 0.011, 0.014, 1.0)
    b.inputs["Roughness"].default_value = 0.025
    b.inputs["IOR"].default_value = 1.333
    tc = nt.nodes.new("ShaderNodeTexCoord")
    # long low swell (anisotropic) + fine wind ripples
    mp = nt.nodes.new("ShaderNodeVectorMath")
    mp.operation = "MULTIPLY"
    mp.inputs[1].default_value = (0.35, 1.0, 1.0)
    nt.links.new(tc.outputs["Object"], mp.inputs[0])
    sw = nt.nodes.new("ShaderNodeTexNoise")
    sw.inputs["Scale"].default_value = 0.03
    sw.inputs["Detail"].default_value = 3.0
    nt.links.new(mp.outputs[0], sw.inputs["Vector"])
    rp = nt.nodes.new("ShaderNodeTexNoise")
    rp.inputs["Scale"].default_value = 0.9
    rp.inputs["Detail"].default_value = 8.0
    rp.inputs["Roughness"].default_value = 0.55
    nt.links.new(mp.outputs[0], rp.inputs["Vector"])
    m1 = nt.nodes.new("ShaderNodeMath")
    m1.operation = "MULTIPLY"
    m1.inputs[1].default_value = 0.35
    nt.links.new(sw.outputs["Fac"], m1.inputs[0])
    m2 = nt.nodes.new("ShaderNodeMath")
    m2.operation = "MULTIPLY_ADD"
    m2.inputs[1].default_value = 0.022
    nt.links.new(rp.outputs["Fac"], m2.inputs[0])
    nt.links.new(m1.outputs[0], m2.inputs[2])
    bp = nt.nodes.new("ShaderNodeBump")
    bp.inputs["Strength"].default_value = 1.0
    bp.inputs["Distance"].default_value = 1.0
    nt.links.new(m2.outputs[0], bp.inputs["Height"])
    nt.links.new(bp.outputs["Normal"], b.inputs["Normal"])
    return m


def water_plane(z, size=40000.0, n=16):
    bm = bmesh.new()
    bmesh.ops.create_grid(bm, x_segments=n, y_segments=n, size=size / 2)
    me = bpy.data.meshes.new("Preview_Water")
    bm.to_mesh(me)
    bm.free()
    me.materials.append(water_material())
    ob = bpy.data.objects.new("Preview_Water", me)
    bpy.context.scene.collection.objects.link(ob)
    ob.location.z = z
    return ob


DRONESHIP_VIEWS = {
    # name: (camera location, target, lens)
    "wide": ((56.0, -139.0, 42.0), (-3.0, 0.0, 0.0), 50.0),
    "deck": ((8.0, -21.6, 1.3), (-10.5, 2.0, 2.3), 24.0),
    "stern": ((-16.0, 13.0, 2.0), (-40.0, -6.0, 4.5), 28.0),
}
EXTRA_VIEWS = {   # inspection only (--views top,wallc,deckc)
    "top": ((0.0, 0.0, 125.0), (0.0, 0.001, 0.0), 35.0),
    "wallc": ((-26.5, 11.0, 2.0), (-33.0, 15.5, 3.2), 35.0),
    "deckc": ((9.0, -15.0, 1.7), (1.0, -5.0, 0.0), 35.0),
}
DUSK_VIEW = ((44.0, -78.0, 16.0), (-6.0, 2.0, 2.0), 40.0)
DRONESHIP_DEFAULT = ["wide", "deck", "stern"]


def run_droneship(a):
    import droneship
    importlib.reload(droneship)
    clear_scene()
    coll = bpy.data.collections.new("ASDS")
    bpy.context.scene.collection.children.link(coll)
    t = time.time()
    root = droneship.build(coll)
    print(f"[test] droneship built in {time.time() - t:.2f}s", flush=True)
    if a.flood is not None:
        droneship.set_flood_intensity(root, a.flood)
    objs = [o for o in coll.objects]
    print(f"[test] droneship objects={len(objs)} meshes={sum(o.type == 'MESH' for o in objs)} "
          f"lights={sum(o.type == 'LIGHT' for o in objs)} triangles={tri_count(objs)}", flush=True)
    for o in sorted(objs, key=lambda o: o.name):
        if o.type == "MESH":
            print(f"    {o.name:28s} tris={tri_count([o]):7d} mats={[m.name for m in o.data.materials]}")
    water_plane(droneship.WATERLINE_Z)
    sun_az, sun_el = -15.0, 4.0
    sky_world(sun_az, sun_el, strength=0.32, aerosol=1.0)
    add_sun("Preview_Sun", sun_az, sun_el, 3.2, (1.0, 0.62, 0.36))
    setup_render(a.samples, a.res)
    os.makedirs(a.outdir, exist_ok=True)
    views = [v for v in a.views.split(",") if v] or DRONESHIP_DEFAULT + ["dusk"]
    for v in views:
        if v == "dusk":
            continue
        loc, tgt, lens = DRONESHIP_VIEWS.get(v) or EXTRA_VIEWS[v]
        cam = camera("Cam_" + v, loc, tgt, lens)
        render(cam, os.path.join(a.outdir, f"droneship_{v}.png"))
    if "dusk" in views:
        # after sunset: sun below the horizon, deck floodlights and nav lights on
        sun_az, sun_el = -15.0, -2.5
        sky_world(sun_az, sun_el, strength=0.5, aerosol=1.0, name="Preview_Sky_Dusk")
        bpy.data.objects["Preview_Sun"].hide_render = True
        droneship.set_flood_intensity(root, 1.0 if a.flood is None else a.flood, nav=1.0)
        loc, tgt, lens = DUSK_VIEW
        cam = camera("Cam_dusk", loc, tgt, lens)
        render(cam, os.path.join(a.outdir, "droneship_dusk.png"))


def main():
    a = args()
    a.outdir = os.path.abspath(a.outdir)
    if a.asset == "droneship":
        run_droneship(a)
    elif a.asset == "payload":
        run_payload(a)
    elif a.asset == "payload_fairing":
        run_payload_fairing(a)
    else:
        raise SystemExit("unknown asset " + a.asset)


# ---------------------------------------------------------------------------------
# payload
# ---------------------------------------------------------------------------------
def area_light(name, loc, target, size, energy, color=(1.0, 1.0, 1.0), size_y=None, coll=None, spread=180.0):
    ld = bpy.data.lights.new(name, "AREA")
    ld.energy = energy
    ld.color = color
    ld.spread = math.radians(spread)
    if size_y is None:
        ld.shape = "SQUARE"
        ld.size = size
    else:
        ld.shape = "RECTANGLE"
        ld.size, ld.size_y = size, size_y
    ob = bpy.data.objects.new(name, ld)
    (coll or bpy.context.scene.collection).objects.link(ob)
    ob.location = Vector(loc)
    ob.rotation_euler = (Vector(target) - Vector(loc)).to_track_quat("-Z", "Y").to_euler()
    return ob


def studio_backdrop(facing_az, color="#1C1D1F", rough=0.38):
    """Seamless floor + curved cyclorama wall; the wall stands on the far side of the
    origin as seen from azimuth `facing_az` (degrees)."""
    from f9 import materials as MT
    prof = [(-40.0, 0.0), (6.0, 0.0)]
    rc = 7.0
    for k in range(1, 13):
        a = math.radians(90.0 * k / 12)
        prof.append((6.0 + rc * math.sin(a), rc * (1.0 - math.cos(a))))
    prof.append((6.0 + rc, 40.0))
    bm = bmesh.new()
    half = 45.0
    rows = [[bm.verts.new((d, y, z)) for d, z in prof] for y in (-half, half)]
    for j in range(len(prof) - 1):
        bm.faces.new((rows[0][j], rows[0][j + 1], rows[1][j + 1], rows[1][j]))
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
    for f in bm.faces:
        f.smooth = True
    me = bpy.data.meshes.new("Studio_Cyc")
    bm.to_mesh(me)
    bm.free()
    m = bpy.data.materials.new("Studio_Cyc")
    m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = MT.srgb(color)
    b.inputs["Roughness"].default_value = rough
    me.materials.append(m)
    ob = bpy.data.objects.new("Studio_Cyc", me)
    bpy.context.scene.collection.objects.link(ob)
    ob.rotation_euler = (0.0, 0.0, math.radians(facing_az + 180.0))
    for f in me.polygons:      # make sure the floor faces up
        pass
    return ob


def glow_backdrop(cam_loc, target, dist=40.0, size=150.0, strength=0.06, color=(0.55, 0.62, 0.75)):
    """Large dim emissive card behind the subject with a radial falloff (soft glow)."""
    from mathutils import Matrix as Mx
    d = (Vector(target) - Vector(cam_loc)).normalized()
    bm = bmesh.new()
    bmesh.ops.create_grid(bm, x_segments=1, y_segments=1, size=size / 2)
    me = bpy.data.meshes.new("Studio_Glow")
    bm.to_mesh(me)
    bm.free()
    m = bpy.data.materials.new("Studio_Glow")
    m.use_nodes = True
    nt = m.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    em = nt.nodes.new("ShaderNodeEmission")
    tc = nt.nodes.new("ShaderNodeTexCoord")
    gr = nt.nodes.new("ShaderNodeTexGradient")
    gr.gradient_type = "SPHERICAL"
    mp = nt.nodes.new("ShaderNodeVectorMath")
    mp.operation = "MULTIPLY"
    mp.inputs[1].default_value = (1.0 / 25.0, 1.0 / 25.0, 1.0)      # glow radius ~25 m
    nt.links.new(tc.outputs["Object"], mp.inputs[0])
    nt.links.new(mp.outputs[0], gr.inputs["Vector"])
    pw = nt.nodes.new("ShaderNodeMath")
    pw.operation = "POWER"
    pw.inputs[1].default_value = 2.2
    nt.links.new(gr.outputs["Fac"], pw.inputs[0])
    ml = nt.nodes.new("ShaderNodeMath")
    ml.operation = "MULTIPLY"
    ml.inputs[1].default_value = strength
    nt.links.new(pw.outputs[0], ml.inputs[0])
    em.inputs["Color"].default_value = (*color, 1.0)
    nt.links.new(ml.outputs[0], em.inputs["Strength"])
    nt.links.new(em.outputs[0], out.inputs["Surface"])
    me.materials.append(m)
    ob = bpy.data.objects.new("Studio_Glow", me)
    bpy.context.scene.collection.objects.link(ob)
    ob.matrix_world = Mx.Translation(Vector(target) + d * dist) @ (-d).to_track_quat("Z", "Y").to_matrix().to_4x4()
    ob.visible_diffuse = False
    ob.visible_glossy = False
    ob.visible_shadow = False
    return ob


def studio_world(value=0.012):
    w = bpy.data.worlds.new("Studio_World")
    bpy.context.scene.world = w
    w.use_nodes = True
    bg = w.node_tree.nodes["Background"]
    bg.inputs["Color"].default_value = (value, value, value * 1.05, 1.0)
    bg.inputs["Strength"].default_value = 1.0
    return w


def orbit(target, az, el, dist):
    t = Vector(target)
    a, e = math.radians(az), math.radians(el)
    return tuple(t + Vector((math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e))) * dist)


PAYLOAD_VIEWS = {
    # name: (target, azimuth, elevation, distance, lens) around the stack (root at z = 0)
    "stack": ((0.0, 0.0, 3.95), -33.0, -3.0, 14.6, 30.0),
    "detail": ((1.15, -1.45, 4.05), -42.0, 9.0, 3.3, 50.0),
    "top": ((0.0, 0.0, 6.9), -48.0, 38.0, 9.0, 40.0),
    "low": ((0.0, 0.0, 2.6), -28.0, -6.0, 9.5, 24.0),
}
PAYLOAD_DEFAULT = ["stack", "detail", "top"]


def _report_payload(payload, root):
    objs = []
    stack = list(root.children)
    while stack:
        o = stack.pop()
        stack.extend(o.children)
        objs.append(o)
    print(f"[test] payload objects={len(objs)} triangles={payload.triangle_count(root)}", flush=True)
    seen = set()
    for o in sorted(objs, key=lambda o: o.name):
        if o.type == "MESH" and o.data.name not in seen:
            seen.add(o.data.name)
            users = sum(1 for x in objs if x.type == "MESH" and x.data == o.data)
            print(f"    {o.data.name:24s} x{users:<3d} tris={tri_count([o]):7d} "
                  f"mats={[m.name for m in o.data.materials]}")


def run_payload(a):
    import payload
    importlib.reload(payload)
    clear_scene()
    coll = bpy.data.collections.new("Starlink")
    bpy.context.scene.collection.children.link(coll)
    t = time.time()
    root = payload.build(coll)
    print(f"[test] payload built in {time.time() - t:.2f}s", flush=True)
    _report_payload(payload, root)
    payload.clearance_report(root)            # at the default (flight) position
    root.location = (0.0, 0.0, 0.0)           # studio: stand the stack on the floor
    views = [v for v in a.views.split(",") if v] or PAYLOAD_DEFAULT
    cam_az = PAYLOAD_VIEWS[views[0]][1]
    studio_world(0.006)
    c = Vector((0.0, 0.0, 3.9))
    # key: tall strip softbox at the long face's mirror angle -> a sparkling streak on the MLI
    area_light("Key", orbit(c, cam_az + 74.0, 10.0, 11.0), c, 1.2, 800.0, (1.0, 0.95, 0.88), size_y=9.0,
               spread=70.0)
    area_light("Key2", orbit(c, cam_az + 30.0, 35.0, 12.0), c, 5.0, 260.0, (1.0, 0.97, 0.92), spread=80.0)
    area_light("Fill", orbit(c, cam_az - 40.0, 8.0, 14.0), c, 8.0, 140.0, (0.85, 0.9, 1.0), spread=90.0)
    area_light("Rim_L", orbit(c, cam_az - 125.0, 16.0, 9.5), c, 0.6, 1500.0, (0.9, 0.95, 1.0), size_y=10.0,
               spread=50.0)
    area_light("Rim_R", orbit(c, cam_az + 140.0, 24.0, 9.5), c, 0.6, 1200.0, (1.0, 0.93, 0.85), size_y=10.0,
               spread=50.0)
    area_light("Top", (0.0, 0.0, 13.0), (0.0, 0.0, 0.0), 5.0, 300.0, (1.0, 1.0, 1.0), spread=60.0)
    setup_render(a.samples, a.res)
    os.makedirs(a.outdir, exist_ok=True)
    glow = None
    for v in views:
        tgt, az, el, dist, lens = PAYLOAD_VIEWS[v]
        loc = orbit(tgt, az, el, dist)
        if glow is not None:
            bpy.data.objects.remove(glow, do_unlink=True)
        glow = glow_backdrop(loc, tgt)
        cam = camera("Cam_" + v, loc, tgt, lens)
        render(cam, os.path.join(a.outdir, f"payload_{v}.png"))


FAIRING_VIEWS = {
    # name: (target, azimuth, elevation, distance, lens) in vehicle/world coordinates
    "fairing": ((0.0, 0.0, 63.2), -24.0, -13.0, 21.5, 35.0),
    "fairing_close": ((0.3, 0.2, 61.6), -34.0, -12.0, 8.6, 30.0),
}
FAIRING_DEFAULT = ["fairing", "fairing_close"]


def run_payload_fairing(a):
    # open the vehicle file; it is never saved (nothing in this script writes .blend files)
    bpy.ops.wm.open_mainfile(filepath=VEHICLE_BLEND)
    for name in list(sys.modules):
        if name == "f9" or name.startswith("f9."):
            importlib.reload(sys.modules[name])
    import payload
    importlib.reload(payload)
    s2 = bpy.data.objects["S2"]
    coll = bpy.data.collections.new("Starlink")
    bpy.context.scene.collection.children.link(coll)
    root = payload.build(coll, parent=s2)
    _report_payload(payload, root)
    payload.clearance_report(root)
    half_b = bpy.data.objects.get("Fairing_Half_B")
    if half_b is not None:
        half_b.hide_render = True
        half_b.hide_viewport = True
    setup_render(a.samples, a.res)
    os.makedirs(a.outdir, exist_ok=True)
    views = [v for v in a.views.split(",") if v] or FAIRING_DEFAULT
    for v in views:
        tgt, az, el, dist, lens = FAIRING_VIEWS[v]
        cam = camera("Cam_payload_" + v, orbit(tgt, az, el, dist), tgt, lens)
        render(cam, os.path.join(a.outdir, f"payload_{v}.png"))


if __name__ == "__main__":
    main()
