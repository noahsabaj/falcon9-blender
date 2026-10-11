"""Shared helpers for the film: model loading, render/compositor setup, cameras and
keyframed motion sampled from Python functions (dense keys so motion blur works)."""
import math
import os
import random
import sys

import bpy
from mathutils import Matrix, Quaternion, Vector

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
MODEL = os.path.join(ROOT, "falcon9_block5.blend")
FPS = 24
RES = (1920, 1080)


# --- easing / noise -----------------------------------------------------------------
def clamp01(t):
    return max(0.0, min(1.0, t))


def smooth(t):
    t = clamp01(t)
    return t * t * (3.0 - 2.0 * t)


def smoother(t):
    t = clamp01(t)
    return t * t * t * (t * (6.0 * t - 15.0) + 10.0)


def ease_out(t, p=2.0):
    return 1.0 - (1.0 - clamp01(t)) ** p


def ease_in(t, p=2.0):
    return clamp01(t) ** p


def lerp(a, b, t):
    return a + (b - a) * t


def vlerp(a, b, t):
    return Vector(a).lerp(Vector(b), t)


def window(t, t0, t1):
    """0 before t0, ramps to 1 at t1 (smooth)."""
    return smooth((t - t0) / max(1e-9, t1 - t0))


class Shake:
    """Smooth pseudo-random 3D wobble (sum of sines), for handheld / long-lens camera."""

    def __init__(self, seed, n=4, base=0.35):
        rnd = random.Random(seed)
        self.terms = [[(rnd.uniform(0.6, 1.4) * base * (1.9 ** k), rnd.uniform(0, math.tau), 0.55 ** k)
                       for k in range(n)] for _ in range(3)]

    def __call__(self, t):
        return Vector([sum(a * math.sin(f * math.tau * t + p) for f, p, a in axis) for axis in self.terms])


# --- scene ------------------------------------------------------------------------------
def open_model():
    bpy.ops.wm.open_mainfile(filepath=MODEL)
    sc = bpy.data.collections.get("F9_Scene")
    if sc is not None:
        for ob in list(sc.objects):
            bpy.data.objects.remove(ob, do_unlink=True)
        bpy.data.collections.remove(sc)
    for w in list(bpy.data.worlds):
        bpy.data.worlds.remove(w)
    scn = bpy.context.scene
    scn.frame_current = 1
    return scn


def vehicle():
    names = ("F9_Root", "S1", "S2", "Fairing", "F9_Rig")
    return {n: bpy.data.objects[n] for n in names}


def descendants(ob):
    out = []
    for ch in ob.children:
        out.append(ch)
        out.extend(descendants(ch))
    return out


def set_visible(objs, vis):
    for o in objs:
        o.hide_render = not vis
        o.hide_viewport = not vis


def collection(name, parent=None):
    c = bpy.data.collections.get(name) or bpy.data.collections.new(name)
    if c.name not in (parent or bpy.context.scene.collection).children:
        (parent or bpy.context.scene.collection).children.link(c)
    return c


def fast_sampling(scn, ray_march=True, light_tree=True):
    """Faster sampling than setup_render's defaults, found by benchmarking every shot against them: the
    film renders about 2x faster (162 -> 78 min of frames on an RTX 4060 Ti), and each shot's frames
    differ from the defaults' no more than a re-render with another noise seed does.  The looser noise
    threshold is most of it.  Ray-marched volumes halve the cost of the plumes, steam and light beams,
    but shift the faint haze behind the livery shot, which turns them off.  Sampling without the light
    tree is faster again in the wide flight and space shots; in close-ups it leaves grain in dark metal
    and glossy detail, so they keep it.  (Fewer bounces saved nothing and brightened the engine bells.)"""
    c = scn.cycles
    c.adaptive_threshold = 0.04
    if ray_march:
        c.volume_biased, c.volume_step_rate = True, 2.0
    c.use_light_tree = light_tree


def setup_render(scn, frames, samples=128, motion_blur=False, shutter=0.5, exposure=0.0,
                 look="AgX - Medium High Contrast", res=RES, bounces=8, clamp_indirect=8.0, view="AgX"):
    r = scn.render
    r.engine = "CYCLES"
    r.resolution_x, r.resolution_y = res
    r.resolution_percentage = 100
    r.fps, r.fps_base = FPS, 1.0
    scn.frame_start, scn.frame_end = 1, frames
    r.use_motion_blur = motion_blur
    r.motion_blur_shutter = shutter
    c = scn.cycles
    c.samples = samples
    c.use_adaptive_sampling = True
    c.adaptive_threshold = 0.01
    c.use_denoising = True
    c.max_bounces = bounces
    c.glossy_bounces = 4
    c.transmission_bounces = 6
    c.volume_bounces = 1
    c.transparent_max_bounces = 16
    c.caustics_reflective = False
    c.caustics_refractive = False
    c.sample_clamp_indirect = clamp_indirect
    c.blur_glossy = 1.0
    scn.view_settings.view_transform = view
    try:
        scn.view_settings.look = look
    except TypeError:
        pass
    scn.view_settings.exposure = exposure
    r.image_settings.file_format = "PNG"
    r.image_settings.color_depth = "8"
    r.use_overwrite = False
    r.use_placeholder = True
    r.film_transparent = False
    for screen in bpy.data.screens:
        for area in screen.areas:
            for space in area.spaces:
                if space.type == "VIEW_3D":
                    space.clip_start, space.clip_end = 0.05, 1e7


def setup_compositor(scn, glare="Fog Glow", glare_strength=0.6, glare_threshold=1.2, glare_size=0.7,
                     vignette=0.22, dispersion=0.004, distortion=-0.006):
    """Bloom/glare on hot highlights, slight lens distortion with chromatic dispersion, vignette."""
    ng = bpy.data.node_groups.new("F9_FilmComp", "CompositorNodeTree")
    ng.interface.new_socket("Image", in_out="OUTPUT", socket_type="NodeSocketColor")
    N, L = ng.nodes, ng.links
    rl = N.new("CompositorNodeRLayers")
    rl.location = (-900, 0)
    cur = rl.outputs["Image"]
    if glare:
        g = N.new("CompositorNodeGlare")
        g.location = (-650, 0)
        g.inputs["Type"].default_value = glare
        g.inputs["Quality"].default_value = "High"
        g.inputs["Threshold"].default_value = glare_threshold
        g.inputs["Strength"].default_value = glare_strength
        g.inputs["Size"].default_value = glare_size
        L.new(cur, g.inputs["Image"])
        cur = g.outputs["Image"]
    if distortion or dispersion:
        ld = N.new("CompositorNodeLensdist")
        ld.location = (-400, 0)
        ld.inputs["Distortion"].default_value = distortion
        ld.inputs["Dispersion"].default_value = dispersion
        ld.inputs["Fit"].default_value = True
        L.new(cur, ld.inputs["Image"])
        cur = ld.outputs["Image"]
    if vignette > 0:
        em = N.new("CompositorNodeEllipseMask")
        em.location = (-400, -300)
        em.inputs["Size"].default_value = (1.25, 1.25)
        bl = N.new("CompositorNodeBlur")
        bl.location = (-200, -300)
        bl.inputs["Size"].default_value = (500.0, 500.0)
        L.new(em.outputs["Mask"], bl.inputs["Image"])
        mix = N.new("ShaderNodeMix")
        mix.data_type = "RGBA"
        mix.blend_type = "MULTIPLY"
        mix.location = (0, 0)
        mix.inputs[0].default_value = vignette
        a = [s for s in mix.inputs if s.name == "A" and s.type == "RGBA"][0]
        b = [s for s in mix.inputs if s.name == "B" and s.type == "RGBA"][0]
        L.new(cur, a)
        inv = N.new("ShaderNodeMix")
        inv.data_type = "RGBA"
        inv.location = (-50, -300)
        ia = [s for s in inv.inputs if s.name == "A" and s.type == "RGBA"][0]
        ib = [s for s in inv.inputs if s.name == "B" and s.type == "RGBA"][0]
        ia.default_value = (0.0, 0.0, 0.0, 1.0)
        ib.default_value = (1.0, 1.0, 1.0, 1.0)
        L.new(bl.outputs["Image"], inv.inputs[0])
        L.new([s for s in inv.outputs if s.type == "RGBA"][0], b)
        cur = [s for s in mix.outputs if s.type == "RGBA"][0]
    out = N.new("NodeGroupOutput")
    out.location = (250, 0)
    L.new(cur, out.inputs[0])
    scn.compositing_node_group = ng
    scn.render.use_compositing = True
    return ng


# --- cameras and keyed motion -----------------------------------------------------------------
def camera(name, lens=50.0, fstop=None, clip=(0.1, 2.0e7), sensor=36.0, coll=None):
    cd = bpy.data.cameras.new(name)
    cd.lens = lens
    cd.sensor_width = sensor
    cd.sensor_fit = "HORIZONTAL"
    cd.clip_start, cd.clip_end = clip
    if fstop:
        cd.dof.use_dof = True
        cd.dof.aperture_fstop = fstop
        cd.dof.aperture_blades = 7
    ob = bpy.data.objects.new(name, cd)
    (coll or bpy.context.scene.collection).objects.link(ob)
    bpy.context.scene.camera = ob
    return ob


def look_quat(pos, tgt, up=None):
    """Rotation pointing local -Z from pos at tgt with local +Y toward `up` (default: the
    world-Z-preferring track quaternion, unstable when looking straight up or down)."""
    f = Vector(tgt) - Vector(pos)
    if up is None:
        return f.to_track_quat("-Z", "Y")
    f.normalize()
    r = f.cross(Vector(up))
    if r.length < 1e-8:
        r = f.cross(Vector((1.0, 0.0, 0.0)))
    r.normalize()
    u = r.cross(f)
    return Matrix((r, u, -f)).transposed().to_quaternion()


def look_euler(pos, tgt, roll=0.0, prev=None, up=None):
    q = look_quat(pos, tgt, up)
    if roll:
        q = q @ Quaternion((0.0, 0.0, 1.0), roll)
    e = q.to_euler()
    if prev is not None:
        e.make_compatible(prev)
    return e


def aim(ob, tgt, up=None):
    ob.rotation_mode = "XYZ"
    ob.rotation_euler = look_euler(ob.location, tgt, up=up)


def _linear(idblock):
    ad = idblock.animation_data
    if ad is None or ad.action is None:
        return
    for fc in _fcurves(ad.action):
        for kp in fc.keyframe_points:
            kp.interpolation = "LINEAR"


def _fcurves(action):
    if hasattr(action, "fcurves") and len(getattr(action, "fcurves", [])):
        return list(action.fcurves)
    out = []
    for layer in getattr(action, "layers", []):
        for strip in layer.strips:
            for bag in strip.channelbags:
                out.extend(bag.fcurves)
    return out


def key_camera(cam, frames, pos_fn, tgt_fn, roll_fn=None, focus_fn=None, lens_fn=None, shake=None,
               up_fn=None, shift_fn=None):
    """pos_fn/tgt_fn(t in 0..1) -> world vectors; keys every frame (plus the frame after the
    last one so motion blur at the final frame has data).  up_fn(t) gives an explicit world
    up vector (for shots looking straight up); shift_fn(t) -> (shift_x, shift_y)."""
    prev = None
    n = frames
    for f in range(1, n + 2):
        t = (f - 1) / max(1, n - 1)
        p = Vector(pos_fn(t))
        q = Vector(tgt_fn(t))
        if shake is not None:
            q = q + shake((f - 1) / FPS)
        e = look_euler(p, q, roll_fn(t) if roll_fn else 0.0, prev, up_fn(t) if up_fn else None)
        prev = e.copy()
        cam.location = p
        cam.rotation_euler = e
        cam.keyframe_insert("location", frame=f)
        cam.keyframe_insert("rotation_euler", frame=f)
        if focus_fn is not None:
            cam.data.dof.focus_distance = max(0.05, focus_fn(t, p, q))
            cam.data.keyframe_insert("dof.focus_distance", frame=f)
        if lens_fn is not None:
            cam.data.lens = lens_fn(t)
            cam.data.keyframe_insert("lens", frame=f)
        if shift_fn is not None:
            cam.data.shift_x, cam.data.shift_y = shift_fn(t)
            cam.data.keyframe_insert("shift_x", frame=f)
            cam.data.keyframe_insert("shift_y", frame=f)
    _linear(cam)
    _linear(cam.data)


def key_object(ob, frames, fn, rot_mode="QUATERNION"):
    """fn(t, f) -> (location, rotation) with rotation a Quaternion or Euler; dense keys."""
    ob.rotation_mode = rot_mode
    prevq = None
    for f in range(1, frames + 2):
        t = (f - 1) / max(1, frames - 1)
        loc, rot = fn(t, f)
        ob.location = loc
        if rot_mode == "QUATERNION":
            q = rot if isinstance(rot, Quaternion) else rot.to_quaternion()
            if prevq is not None and prevq.dot(q) < 0:
                q = -q
            prevq = q.copy()
            ob.rotation_quaternion = q
            ob.keyframe_insert("rotation_quaternion", frame=f)
        else:
            ob.rotation_euler = rot
            ob.keyframe_insert("rotation_euler", frame=f)
        ob.keyframe_insert("location", frame=f)
    _linear(ob)


def key_prop(idblock, path, frames, fn, linear=True):
    """Key a property (e.g. '["glow"]' or 'energy') from fn(t, f)."""
    for f in range(1, frames + 2):
        t = (f - 1) / max(1, frames - 1)
        val = fn(t, f)
        if path.startswith('["'):
            idblock[path[2:-2]] = val
        else:
            setattr(idblock, path, val)
        idblock.keyframe_insert(path, frame=f)
    if linear:
        _linear(idblock)


def key_value(socket, frames, fn):
    """Key a node socket default_value."""
    for f in range(1, frames + 2):
        t = (f - 1) / max(1, frames - 1)
        socket.default_value = fn(t, f)
        socket.keyframe_insert("default_value", frame=f)


def area_light(name, size, energy, color=(1, 1, 1), loc=(0, 0, 0), tgt=(0, 0, 0), shape="RECTANGLE",
               size_y=None, spread=180.0, camera_visible=False, coll=None, up=None, glossy=True):
    ld = bpy.data.lights.new(name, "AREA")
    ld.shape = shape
    ld.size = size
    if size_y is not None:
        ld.size_y = size_y
    ld.energy = energy
    ld.color = color
    ld.spread = math.radians(spread)
    ob = bpy.data.objects.new(name, ld)
    (coll or bpy.context.scene.collection).objects.link(ob)
    ob.location = loc
    ob.rotation_euler = look_euler(loc, tgt, up=up)
    ob.visible_camera = camera_visible
    ob.visible_glossy = glossy
    return ob


def point_light(name, energy, radius=0.5, color=(1, 1, 1), loc=(0, 0, 0), coll=None):
    ld = bpy.data.lights.new(name, "POINT")
    ld.energy = energy
    ld.shadow_soft_size = radius
    ld.color = color
    ob = bpy.data.objects.new(name, ld)
    (coll or bpy.context.scene.collection).objects.link(ob)
    ob.location = loc
    return ob


def sun_light(name, energy, angle_deg=0.53, color=(1, 1, 1), direction=(0, 0, 1), coll=None):
    ld = bpy.data.lights.new(name, "SUN")
    ld.energy = energy
    ld.angle = math.radians(angle_deg)
    ld.color = color
    ob = bpy.data.objects.new(name, ld)
    (coll or bpy.context.scene.collection).objects.link(ob)
    d = Vector(direction).normalized()
    ob.rotation_euler = d.to_track_quat("Z", "Y").to_euler()
    ob.location = d * 50.0
    return ob


def sun_dir(az_deg, el_deg):
    az, el = math.radians(az_deg), math.radians(el_deg)
    return Vector((math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)))


def save(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=path, compress=True)
