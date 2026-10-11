"""Shot builders.  SHOTS[id](scn, frames) dresses the freshly opened model file with a world,
lights, a keyed camera and vehicle animation; inside a builder t runs 0..1 across the shot.
Shot ids, timings and descriptions live in cues.json."""
import json
import math
import os

import bpy
from mathutils import Euler, Matrix, Quaternion, Vector

from . import common as C
from f9 import dims as D

from . import droneship, envs, fx, payload

with open(os.path.join(C.HERE, "cues.json"), encoding="utf-8") as fh:
    CUES = json.load(fh)
SHOTS = {}

WARM = (1.0, 0.86, 0.72)
NEUTRAL = (1.0, 0.96, 0.92)
COOL = (0.66, 0.79, 1.0)


def shot(fn):
    SHOTS[fn.__name__] = fn
    return fn


def frames(sid):
    s = next(s for s in CUES["shots"] if s["id"] == sid)
    return round((s["end"] - s["start"]) * C.FPS)


def cyl(r, az_deg, z):
    a = math.radians(az_deg)
    return Vector((r * math.cos(a), r * math.sin(a), z))


def glide(t, k=0.5):
    """Mostly constant-velocity move with softened ends, so cuts land mid-motion."""
    return (1.0 - k) * t + k * C.smooth(t)


def stowed():
    rig = bpy.data.objects["F9_Rig"]
    rig["leg_deploy"] = 0.0
    rig["gridfin_deploy"] = 0.0


def studio_setup(scn, n, samples=128, exposure=0.0, vignette=0.28, glare=0.35, threshold=2.0, horizon=None,
                 ray_march=True):
    C.setup_render(scn, n, samples=samples, exposure=exposure)
    C.fast_sampling(scn, ray_march=ray_march)
    C.setup_compositor(scn, glare_strength=glare, glare_threshold=threshold, vignette=vignette,
                       dispersion=0.003, distortion=-0.004)
    envs.studio(scn, horizon=horizon)
    stowed()
    return C.collection("Film_Studio")


# --- Act I: the machine (dark studio) ------------------------------------------------------
@shot
def s01_nine(scn, n):
    """Looking up into the nine Merlins while a strip light sweeps beneath the bells."""
    coll = studio_setup(scn, n, vignette=0.38)
    cam = C.camera("Cam_s01", lens=24.0, fstop=4.0, coll=coll)
    a0, a1 = math.radians(-24.0), math.radians(10.0)
    C.key_camera(
        cam, n,
        pos_fn=lambda t: C.vlerp((0.65, -1.10, -5.2), (0.30, -0.55, -3.8), C.ease_out(t, 1.5)),
        tgt_fn=lambda t: Vector((0.0, 0.2, 1.2)),
        up_fn=lambda t: (math.cos(C.lerp(a0, a1, t)), math.sin(C.lerp(a0, a1, t)), 0.0),
        focus_fn=lambda t, p, q: -p.z + 0.25)
    sweep = C.area_light("s01_Sweep", 0.5, 600.0, NEUTRAL, size_y=10.0, coll=coll)
    tgt = Vector((0.0, 0.0, 0.7))

    def sweep_fn(t, f):
        u = C.smoother((t - 0.08) / 0.84)
        loc = Vector((C.lerp(-6.5, 6.5, u), -0.8, -2.4))
        return loc, C.look_quat(loc, tgt, up=(0.0, 1.0, 0.0))

    C.key_object(sweep, n, sweep_fn)
    C.area_light("s01_Rim", 3.0, 260.0, COOL, loc=(-4.5, 4.0, -0.6), tgt=(0, 0, 1.0), coll=coll)
    C.area_light("s01_Skirt", 2.0, 120.0, COOL, size_y=6.0, loc=cyl(6.5, 200.0, -1.2), tgt=(0, 0, 1.6), coll=coll)
    C.area_light("s01_Glow", 4.0, 18.0, WARM, loc=(0.0, 0.0, -8.0), tgt=(0, 0, 0), shape="DISK", coll=coll)


@shot
def s02_lattice(scn, n):
    """Glide across stowed grid fin 1 (az 45): titanium lattice in shallow focus, the camera
    crossing the fin so the blades shift against each other."""
    coll = studio_setup(scn, n, vignette=0.30)
    a = math.radians(45.0)
    er, et = Vector((math.cos(a), math.sin(a), 0.0)), Vector((-math.sin(a), math.cos(a), 0.0))

    def P(r, s, z):
        return r * er + s * et + Vector((0.0, 0.0, z))

    cam = C.camera("Cam_s02", lens=50.0, fstop=2.8, coll=coll)
    C.key_camera(
        cam, n,
        pos_fn=lambda t: P(C.lerp(3.95, 3.80, t), C.lerp(-1.30, 0.90, glide(t)), C.lerp(40.55, 41.45, glide(t))),
        tgt_fn=lambda t: P(2.12, C.lerp(-0.55, 0.35, glide(t)), C.lerp(41.00, 41.25, glide(t))),
        focus_fn=lambda t, p, q: (q - p).length)
    C.area_light("s02_Key", 1.6, 320.0, WARM, loc=P(4.2, 1.8, 43.6), tgt=P(2.0, 0.0, 41.2), coll=coll)
    C.area_light("s02_Rim", 1.0, 220.0, COOL, size_y=2.5, loc=P(2.6, -2.6, 40.2), tgt=P(2.0, 0.0, 41.0),
                 coll=coll)
    C.area_light("s02_Fill", 3.0, 25.0, NEUTRAL, loc=P(6.0, -0.5, 40.5), tgt=P(2.0, 0.0, 41.0), coll=coll)


@shot
def s03_livery(scn, n):
    """Crane down the stacked SPACEX letters on the +Y side, reading S to X."""
    coll = studio_setup(scn, n, vignette=0.30, ray_march=False)
    cam = C.camera("Cam_s03", lens=35.0, fstop=4.0, coll=coll)

    def zc(t):
        return C.lerp(28.6, 21.4, glide(t, 0.6))

    C.key_camera(
        cam, n,
        pos_fn=lambda t: cyl(C.lerp(9.6, 9.0, t), C.lerp(101.0, 95.0, t), zc(t)),
        tgt_fn=lambda t: cyl(1.83, 90.0, zc(t) - 0.35),
        focus_fn=lambda t, p, q: (q - p).length)
    C.area_light("s03_Key", 4.0, 1500.0, WARM, size_y=12.0, loc=cyl(9.0, 138.0, 30.0), tgt=(0, 0, 25.0),
                 coll=coll)
    C.area_light("s03_RimL", 1.2, 1600.0, COOL, size_y=26.0, loc=cyl(9.0, 345.0, 25.0), tgt=(0, 0, 25.0),
                 coll=coll)
    C.area_light("s03_RimR", 1.2, 700.0, COOL, size_y=26.0, loc=cyl(9.0, 215.0, 25.0), tgt=(0, 0, 25.0),
                 coll=coll)
    envs.haze(density=0.002, size=(60.0, 60.0, 50.0), center=(0.0, 0.0, 25.0))


@shot
def s04_interstage(scn, n):
    """Low orbit around the black interstage past grid fin 1, the GN2 thruster pod and grid fin 2."""
    coll = studio_setup(scn, n, vignette=0.30)
    cam = C.camera("Cam_s04", lens=24.0, fstop=5.6, coll=coll)
    C.key_camera(
        cam, n,
        pos_fn=lambda t: cyl(C.lerp(6.4, 6.0, t), C.lerp(22.0, 150.0, glide(t)), C.lerp(40.8, 43.6, glide(t))),
        tgt_fn=lambda t: Vector((0.0, 0.0, C.lerp(42.8, 45.2, glide(t)))),
        focus_fn=lambda t, p, q: p.xy.length - 1.83)
    C.area_light("s04_Key", 8.0, 4500.0, NEUTRAL, loc=cyl(14.0, 75.0, 50.0), tgt=(0, 0, 45.0), coll=coll)
    C.area_light("s04_RimL", 1.5, 2600.0, COOL, size_y=16.0, loc=cyl(10.0, 330.0, 45.0), tgt=(0, 0, 45.0),
                 coll=coll)
    C.area_light("s04_RimR", 1.5, 2600.0, COOL, size_y=16.0, loc=cyl(10.0, 230.0, 45.0), tgt=(0, 0, 45.0),
                 coll=coll)
    C.area_light("s04_Top", 6.0, 3000.0, NEUTRAL, loc=cyl(9.0, 100.0, 58.0), tgt=(0, 0, 47.0), coll=coll)
    envs.haze(density=0.002, size=(60.0, 60.0, 40.0), center=(0.0, 0.0, 45.0))


@shot
def s05_reveal(scn, n):
    """Pull back from the fairing to the whole 70 m vehicle standing on black lacquer,
    silhouetted against a soft backlight and rim lit; the title lands at 19.0 s (t = 0.29) in
    the edit.  The camera stays level and frames with lens shift so the vehicle stands
    truly vertical, on the left third."""
    az_end = 140.0
    back = az_end + 180.0
    coll = studio_setup(scn, n, vignette=0.30, glare=0.4, horizon=((0.010, 0.012, 0.016), 0.06))
    envs.backlight(scn.world, C.sun_dir(back, 12.0), (0.030, 0.036, 0.046), 6.0)
    envs.studio_floor(rough=0.06)
    cam = C.camera("Cam_s05", lens=45.0, coll=coll)

    def u(t):
        return C.smoother(t / 0.72) * 0.9 + 0.1 * t

    def pos(t):
        k = u(t)
        return cyl(C.lerp(12.0, 92.0, k), C.lerp(122.0, az_end, k), C.lerp(60.0, 5.0, k))

    C.key_camera(
        cam, n,
        pos_fn=pos,
        tgt_fn=lambda t: Vector((0.0, 0.0, pos(t).z)),
        lens_fn=lambda t: C.lerp(45.0, 24.0, u(t)),
        shift_fn=lambda t: (C.lerp(0.0, 0.14, u(t)), C.lerp(0.0, 0.21, u(t))))
    for side, d in (("L", -28.0), ("R", 28.0)):
        C.area_light("s05_Rim" + side, 2.0, 40000.0, COOL, size_y=70.0, loc=cyl(24.0, back + d, 45.0),
                     tgt=(0, 0, 45.0), coll=coll)
    C.area_light("s05_Key", 12.0, 7000.0, WARM, size_y=30.0, loc=cyl(50.0, az_end + 35.0, 25.0),
                 tgt=(0, 0, 30.0), coll=coll)
    envs.haze(density=0.0006, size=(120.0, 120.0, 90.0))


# --- Act II: ascent (sunset) ---------------------------------------------------------------
SUN_AZ, SUN_EL = 180.0, 3.0          # sunset in the west (-X); +Y is north
SUN_COLOR = (1.0, 0.56, 0.30)


def flight_setup(scn, n, samples=96, exposure=0.0, mb=True, glare=0.55, threshold=1.6, vignette=0.22,
                 light_tree=False):
    C.setup_render(scn, n, samples=samples, exposure=exposure, motion_blur=mb, shutter=0.5,
                   view="Khronos PBR Neutral", look="None")
    C.fast_sampling(scn, light_tree=light_tree)
    C.setup_compositor(scn, glare_strength=glare, glare_threshold=threshold, vignette=vignette,
                       dispersion=0.004, distortion=-0.006)
    stowed()
    return C.collection("Film_Flight")


def sunset_world(scn, coll, sky_strength=1.0, sun_energy=2.0):
    w, sk = envs.sky(scn, SUN_EL, SUN_AZ, strength=sky_strength, air=1.0, aerosol=1.6)
    C.sun_light("Sun", sun_energy, angle_deg=0.6, color=SUN_COLOR, direction=C.sun_dir(SUN_AZ, SUN_EL), coll=coll)
    envs.ocean(sk, strength=sky_strength, coll=coll)
    return sk


@shot
def s06_ascent(scn, n):
    """Hard cut into the climb: a chase camera ~45 m off the vehicle at ~1.5 km rises a little
    slower than the vehicle, so the engines and blazing plume start in frame and the vehicle
    slides up and away while the plume streams past the shaking camera."""
    coll = flight_setup(scn, n, light_tree=True)
    sunset_world(scn, coll, sky_strength=SKY_K, sun_energy=SUN_K)
    V = C.vehicle()
    root = V["F9_Root"]
    view_az = 125.0
    root.rotation_euler = (0.0, 0.0, math.radians(35.0))
    dur = (n - 1) / C.FPS

    def tail(T):
        return 1475.0 + 55.0 * T + 3.0 * T * T

    def cam_z(T):
        return tail(T) + 6.0 - 16.0 * T

    C.key_object(root, n, lambda t, f: (Vector((0.0, 0.0, tail(t * dur))), root.rotation_euler.copy()),
                 rot_mode="XYZ")
    fx.plume(root, n, length=80.0, r_end=12.0, intensity=PLUME_K, flow=110.0, smoke=SMOKE_K, coll=coll)
    fx.plume_light(root, z=-10.0, energy=LIGHT_K, radius=6.0, coll=coll)
    def d(t):
        a = math.radians(view_az + C.lerp(-10.0, 10.0, t))
        return Vector((math.cos(a), math.sin(a), 0.0))

    cam = C.camera("Cam_s06", lens=24.0, coll=coll)
    shake = C.Shake(6, n=4, base=2.2)
    amp = lambda s: 0.10 + 0.35 * math.exp(-((s / dur - 0.5) / 0.25) ** 2)
    C.key_camera(
        cam, n,
        pos_fn=lambda t: -45.0 * d(t) + Vector((0.0, 0.0, cam_z(t * dur))),
        tgt_fn=lambda t: Vector((0.0, 0.0, tail(t * dur) + C.lerp(8.0, -6.0, t))),
        shake=lambda s: amp(s) * shake(s))


SKY_K, SUN_K, PLUME_K, LIGHT_K = 0.08, 2.0, 1.5, 1.5e6
SMOKE_K = 0.0


def hide_vehicle():
    for ob in C.descendants(bpy.data.objects["F9_Root"]):
        ob.hide_render = True
        ob.hide_viewport = True


def sph(dist, az_deg, el_deg):
    return dist * C.sun_dir(az_deg, el_deg)


def bezier(p0, p1, p2, p3, k):
    return [((1 - t) ** 3) * p0 + 3 * ((1 - t) ** 2) * t * p1 + 3 * (1 - t) * t * t * p2 + (t ** 3) * p3
            for t in (i / (k - 1) for i in range(k))]


@shot
def s07_jellyfish(scn, n):
    """Twilight wide shot from the ground (scene unit = 100 m): the sun has set, but the
    vehicle ~120 km up is still sunlit and its exhaust blooms into a translucent jellyfish
    over a sunlit trail that drops into Earth's shadow; slow push in."""
    C.setup_render(scn, n, samples=64, view="Khronos PBR Neutral", look="None")
    C.fast_sampling(scn, light_tree=False)
    C.setup_compositor(scn, glare_strength=0.8, glare_threshold=0.9, glare_size=0.85, vignette=0.32,
                       dispersion=0.003, distortion=-0.004)
    hide_vehicle()
    coll = C.collection("Film_Twilight")
    w, sk = envs.sky(scn, -4.5, 5.0, strength=TW_SKY, aerosol=1.2)      # afterglow off to the right
    envs.stars(w, brightness=TW_STARS, scale=500.0)
    envs.treeline(dist=2600.0, az0=0.0, az1=120.0, coll=coll)
    pad = sph(4800.0, 40.0, 0.0)
    rocket = sph(5200.0, 70.0, 15.5)
    vel = ((rocket - pad).normalized() + Vector((0.0, 0.0, 0.15))).normalized()
    q = vel.to_track_quat("Z", "Y")
    # bell, inner bell and hot core
    def bell(name, L, R, color, k, limb, apex=None, streaks=7.0):
        ob = fx.paraboloid(name, L, R, coll=coll)
        ob.location = rocket
        ob.rotation_mode = "QUATERNION"
        ob.rotation_quaternion = q
        m, tsock, gsock = fx.glow_surface(name, L, color, k, limb=limb, streaks=streaks, apex_color=apex)
        ob.data.materials.append(m)
        fx.key_socket(tsock, n, lambda t, f: (f - 1) / C.FPS)
        return ob

    halo = bell("Jelly_Halo", 1000.0, 950.0, (0.55, 0.75, 1.0), TW_BELL * 0.22, 0.8, streaks=3.0)
    outer = bell("Jelly_Bell", 700.0, 620.0, (0.62, 0.82, 1.0), TW_BELL, 2.2, apex=(1.0, 0.92, 0.82))
    inner = bell("Jelly_Inner", 420.0, 300.0, (0.85, 0.90, 1.0), TW_BELL * 0.6, 1.6, apex=(1.0, 0.9, 0.75), streaks=5.0)
    core = bell("Jelly_Core", 90.0, 45.0, (1.0, 0.85, 0.62), TW_BELL * 2.5, 1.2, streaks=3.0)
    for ob, s0, s1 in ((halo, 0.7, 1.15), (outer, 0.78, 1.12), (inner, 0.85, 1.08), (core, 0.95, 1.05)):
        C.key_prop(ob, "scale", n, lambda t, f, a=s0, b=s1: Vector([C.lerp(a, b, C.ease_out(t, 1.4))] * 3))
    # engine point
    import bmesh as _bm
    me = bpy.data.meshes.new("Jelly_Engine")
    bm = _bm.new()
    _bm.ops.create_uvsphere(bm, u_segments=16, v_segments=8, radius=2.5)
    bm.to_mesh(me)
    bm.free()
    eng = bpy.data.objects.new("Jelly_Engine", me)
    coll.objects.link(eng)
    eng.location = rocket
    em = bpy.data.materials.new("Jelly_Engine")
    em.use_nodes = True
    ent = em.node_tree
    for nd in list(ent.nodes):
        ent.nodes.remove(nd)
    eo = ent.nodes.new("ShaderNodeOutputMaterial")
    es = ent.nodes.new("ShaderNodeEmission")
    es.inputs["Color"].default_value = (1.0, 0.9, 0.75, 1.0)
    es.inputs["Strength"].default_value = TW_ENGINE
    ent.links.new(es.outputs["Emission"], eo.inputs["Surface"])
    me.materials.append(em)
    # trail: vertical rise from the pad, pitching over toward the vehicle's heading
    pts = bezier(pad, pad + Vector((0, 0, 650.0)), rocket - vel * 1300.0, rocket - vel * 25.0, 110)
    radii = [C.lerp(75.0, 6.0, (i / 109) ** 0.5) for i in range(110)]
    tr = fx.tube("Trail", pts, radii, segs=28, coll=coll)
    tm, tt = fx.trail_glow_material("Trail", TW_TRAIL, z_shadow=360.0, z_soft=45.0)
    tr.data.materials.append(tm)
    fx.key_socket(tt, n, lambda t, f: (f - 1) / C.FPS)
    # camera on the ground, slow push in
    cam = C.camera("Cam_s07", lens=26.0, coll=coll)
    C.key_camera(
        cam, n,
        pos_fn=lambda t: Vector((0.0, 0.0, 2.0)),
        tgt_fn=lambda t: sph(1000.0, C.lerp(57.0, 60.0, t), C.lerp(12.5, 13.5, C.smooth(t))),
        lens_fn=lambda t: C.lerp(26.0, 31.0, C.smooth(t)))


TW_SKY, TW_STARS, TW_BELL, TW_TRAIL, TW_ENGINE = 1.0, 6.0, 0.75, 0.32, 600.0


# --- Act III: space -----------------------------------------------------------------------
SPACE_SUN = C.sun_dir(215.0, 22.0)


def space_setup(scn, n, samples=96, coll_name="Film_Space", mb=True, light_tree=False):
    C.setup_render(scn, n, samples=samples, motion_blur=mb, shutter=0.5, view="Khronos PBR Neutral", look="None")
    C.fast_sampling(scn, light_tree=light_tree)
    C.setup_compositor(scn, glare_strength=0.6, glare_threshold=1.4, glare_size=0.8, vignette=0.25,
                       dispersion=0.004, distortion=-0.005)
    stowed()
    coll = C.collection(coll_name)
    envs.space(scn, stars_brightness=SPACE_STARS)
    C.sun_light("Sun", SPACE_SUN_K, angle_deg=0.53, color=(1.0, 0.96, 0.9), direction=SPACE_SUN, coll=coll)
    envs.earth(SPACE_SUN, altitude=80e3, coll=coll)
    return coll


def earth_drift(n, speed):
    """Fly over the ground at `speed` m/s toward +X: turn the Earth under the camera."""
    e = bpy.data.objects["Earth"]
    dur = (n - 1) / C.FPS
    loc = e.location.copy()
    C.key_object(e, n, lambda t, f: (loc, Euler((0.0, -speed * t * dur / envs.R_EARTH, 0.0))), rot_mode="XYZ")


def pitch_vehicle(root, pitch_deg=25.0):
    """Point the vehicle axis toward +X, pitch_deg above the local horizontal."""
    root.rotation_mode = "XYZ"
    root.rotation_euler = (0.0, math.radians(90.0 - pitch_deg), 0.0)
    a = math.radians(pitch_deg)
    return Vector((math.cos(a), 0.0, math.sin(a)))


@shot
def s08_separation(scn, n):
    """Above Earth's limb: the nine-engine plume cuts off (MECO), the stages separate, and
    after a beat of silence the MVac lights and the second stage pulls away."""
    coll = space_setup(scn, n)
    earth_drift(n, 2200.0)
    V = C.vehicle()
    axis = pitch_vehicle(V["F9_Root"])
    dur = (n - 1) / C.FPS
    T_SEP, T_IGN = 1.0, 3.0

    def d(T):          # second-stage travel along the axis (m)
        if T < T_SEP:
            return 0.0
        if T < T_IGN:
            return 3.0 * (T - T_SEP)
        u = T - T_IGN
        return 3.0 * (T_IGN - T_SEP) + 3.0 * u + 4.5 * u * u

    s2 = V["S2"]
    C.key_object(s2, n, lambda t, f: (Vector((0.0, 0.0, d(t * dur))), s2.rotation_quaternion.copy()))
    s1 = V["S1"]
    C.key_object(s1, n, lambda t, f: (Vector((0.0, 0.0, -0.35 * max(0.0, t * dur - T_SEP))),
                                      s1.rotation_quaternion.copy()))
    pl = fx.plume(V["S1"], n, length=60.0, r_end=34.0, intensity=0.9, flow=200.0, coll=coll)
    fx.key_socket(pl["power"], n, lambda t, f: 1.0 - C.smooth((t * dur - 0.12) / 0.25))
    shells = fx.mvac_plume(s2, n, coll=coll)
    fx.key_shells(shells, n, lambda t, f: C.smooth((t * dur - T_IGN) / 0.3)
                  * (1.0 + 0.6 * math.exp(-((t * dur - T_IGN - 0.25) / 0.12) ** 2)))
    fx.niobium_glow(n, lambda t, f: C.smooth((t * dur - T_IGN - 0.3) / 3.2), strength=3.5)
    ign = fx.plume_light(s2, "MVac_Light", z=D.MVAC_Z_EXIT - 1.5, energy=0.0, radius=1.0,
                         color=(1.0, 0.66, 0.38), coll=coll)
    C.key_prop(ign.data, "energy", n, lambda t, f: 4000.0 * C.smooth((t * dur - T_IGN) / 0.3)
               * (1.0 + 2.0 * math.exp(-((t * dur - T_IGN - 0.25) / 0.12) ** 2)))

    def station(z):
        return axis * z

    cam = C.camera("Cam_s08", lens=35.0, coll=coll)
    side = Vector((0.0, -1.0, 0.0))

    def pos(t):        # wide side view -> in close at the interstage -> rise with the departing S2
        T = t * dur
        a = C.vlerp(station(36.0) + side * 125.0 + Vector((-6.0, 0.0, 5.0)),
                    station(47.0) + side * 40.0 + Vector((-5.0, 0.0, 3.0)), C.smoother(T / 3.2))
        return a.lerp(Vector((74.0, -82.0, 58.0)), C.smoother((T - 3.0) / (dur - 3.0)))

    def tgt(t):
        T = t * dur
        k = C.smoother(T / 3.2)
        base = C.vlerp(station(34.0), station(47.0), k)
        follow = station(49.0 + d(T))
        return base.lerp(follow, C.smooth((T - T_IGN - 0.2) / 2.0))

    C.key_camera(cam, n, pos, tgt, lens_fn=lambda t: C.lerp(35.0, 30.0, C.smoother(t * dur / 3.2)))


SPACE_STARS, SPACE_SUN_K = 1.5, 4.0


def mvac_burning(s2, n, coll):
    """Second-stage engine at steady full thrust: plume shells, glowing nozzle, warm light."""
    fx.key_shells(fx.mvac_plume(s2, n, coll=coll), n, lambda t, f: 1.0)
    fx.niobium_glow(n, lambda t, f: 1.0, strength=3.5)
    fx.plume_light(s2, "MVac_Light", z=D.MVAC_Z_EXIT - 1.5, energy=4000.0, radius=1.0,
                   color=(1.0, 0.66, 0.38), coll=coll)


@shot
def s09_fairing(scn, n):
    """Alongside the burning second stage: the fairing halves hinge open, let go and fall away
    behind the accelerating stage, uncovering the stack of Starlink satellites in full sun."""
    coll = space_setup(scn, n, light_tree=True)
    earth_drift(n, 3000.0)
    V = C.vehicle()
    root = V["F9_Root"]
    pitch_vehicle(root)
    R = root.rotation_euler.to_matrix()
    C.set_visible([V["S1"]] + C.descendants(V["S1"]), False)
    s2 = V["S2"]
    payload.build(coll, parent=s2, yaw=math.radians(90.0))
    mvac_burning(s2, n, coll)
    dur = (n - 1) / C.FPS
    z0, r0 = D.Z_FAIRING_BASE, 2.6
    # half A (-x, the space side) lets go first, B (+x, toward Earth) 0.15 s later: two pops
    for name, sx, t_rel, spin in (("Fairing_Half_A", -1.0, 1.0, 1.0), ("Fairing_Half_B", 1.0, 1.15, -0.8)):
        ob = bpy.data.objects[name]
        rest = ob.location.copy()
        pivot = Vector((sx * r0, 0.0, z0))

        def fn(t, f, sx=sx, t_rel=t_rel, spin=spin, rest=rest, pivot=pivot):
            tau = max(0.0, t * dur - t_rel)
            free = max(0.0, tau - 0.45)            # hinged at the base for the first 0.45 s
            hinge = math.radians(24.0) * C.smooth(tau / 0.5) + math.radians(16.0) * free
            q = (Quaternion((0.0, 0.0, 1.0), spin * math.radians(6.0) * free)
                 @ Quaternion((0.0, 1.0, 0.0), sx * hinge))
            off = Vector((sx * (0.3 * C.smooth(tau / 0.45) + 2.4 * free), 0.0, -4.25 * free * free))
            return pivot + q @ (rest - pivot) + off, q
        C.key_object(ob, n, fn)
    cam = C.camera("Cam_s09", lens=35.0, coll=coll)

    def pos(t):
        k = C.smoother(t)
        az = math.radians(C.lerp(252.0, 278.0, k))
        r = C.lerp(25.0, 19.0, k)
        return R @ Vector((r * math.cos(az), r * math.sin(az), C.lerp(59.5, 62.0, k)))

    C.key_camera(cam, n, pos, lambda t: R @ Vector((0.0, 0.0, C.lerp(62.6, 61.4, C.smoother(t)))))


@shot
def s10_flip(scn, n):
    """The booster alone above the limb: cold-gas thrusters fire in bursts, it turns end over
    end to point its engines along its path, and the grid fins swing out."""
    coll = space_setup(scn, n)
    earth_drift(n, 2000.0)
    V = C.vehicle()
    root, s1 = V["F9_Root"], V["S1"]
    C.set_visible([V["S2"]] + C.descendants(V["S2"]), False)
    dur = (n - 1) / C.FPS
    PIV = 16.0                     # centre of mass above the nozzle exits (m)
    th0 = math.radians(65.0)       # pitch_vehicle(25 deg)
    PUFFS = ((0.4, 0.30, 1), (0.9, 0.25, 1), (1.3, 0.35, 1), (1.9, 0.20, -1), (2.3, 0.25, -1))
    W_UP, W_DN = 17.0, 9.0         # pitch-rate change per burst (deg/s): three spin it up, two trim

    def omega(T):
        return sum((W_UP if sg > 0 else -W_DN) * C.smooth((T - t0) / d) for t0, d, sg in PUFFS)

    steps = 2400
    h = dur / steps
    tab = [th0]
    for i in range(steps):
        tab.append(tab[-1] - math.radians(omega((i + 0.5) * h)) * h)

    def theta(T):
        x = min(max(T / h, 0.0), steps - 1e-6)
        i = int(x)
        return tab[i] + (tab[i + 1] - tab[i]) * (x - i)

    pivot_w = Matrix.Rotation(th0, 3, "Y") @ Vector((0.0, 0.0, PIV))

    def at(T, z):                  # world position of booster station z at time T
        return pivot_w + Matrix.Rotation(theta(T), 3, "Y") @ Vector((0.0, 0.0, z - PIV))

    C.key_object(root, n, lambda t, f: (at(t * dur, 0.0), Euler((0.0, theta(t * dur), 0.0))), rot_mode="XYZ")
    C.key_prop(V["F9_Rig"], '["gridfin_deploy"]', n, lambda t, f: C.smoother((t * dur - 2.6) / 0.8))
    for side, tag in ((1.0, "L"), (-1.0, "R")):
        for sg, kind in ((1, "spin"), (-1, "trim")):
            fx.gas_puff(s1, n, "GN2_" + tag + kind, loc=(0.0, side * 2.02, 42.1), direction=(float(sg), 0.0, 0.0),
                        times=[(t0, d) for t0, d, g in PUFFS if g == sg], coll=coll)
    cam = C.camera("Cam_s10", lens=32.0, coll=coll)

    def pos(t):                    # rides with the interstage, easing back: the body swings past
        k = C.smoother(t)
        return at(t * dur, 43.0) + Vector((C.lerp(-4.0, -8.0, k), -C.lerp(22.0, 58.0, k), C.lerp(3.0, 7.0, k)))

    def tgt(t):
        T = t * dur
        return at(T, 43.0).lerp(at(T, 22.0), C.lerp(0.15, 0.4, C.smoother(t)))

    C.key_camera(cam, n, pos, tgt)


def soot(amount=1.0):
    """Entry and landing burns leave the booster sooty: darken F9_paint_white toward the base
    and in long vertical streaks.  The material is shared with S2 and the fairing, so this is
    only for shots where those are hidden."""
    m = bpy.data.materials["F9_paint_white"]
    nt = m.node_tree
    b = next(nd for nd in nt.nodes if nd.type == "BSDF_PRINCIPLED")
    nb = fx.NB(nt)
    tc = nb._node("ShaderNodeTexCoord")
    tc.object = bpy.data.objects["F9_Root"]
    x, y, z = nb.sep(tc.outputs["Object"])
    ang = nb.math("ARCTAN2", y, x)
    ring = nb.comb(nb.mul(nb.math("COSINE", ang), 6.0), nb.mul(nb.math("SINE", ang), 6.0), nb.mul(z, 0.05))
    streak = nb.noise(ring, scale=1.0, detail=6.0, rough=0.65)
    fine = nb.noise(nb.comb(x, y, nb.mul(z, 0.25)), scale=3.0, detail=4.0, rough=0.6)
    sv = nb.add(nb.mul(nb.smooth(30.0, 3.0, z), 0.85), nb.mul(nb.sub(streak, 0.42), 1.6))
    sv = nb.clamp01(nb.mul(nb.clamp01(nb.add(sv, nb.mul(nb.sub(fine, 0.5), 0.4))), amount))
    col = b.inputs["Base Color"]
    src = col.links[0].from_socket if col.is_linked else tuple(col.default_value)
    nt.links.new(nb.mixrgb(sv, src, (0.035, 0.032, 0.03, 1.0)), col)
    rgh = b.inputs["Roughness"]
    src = rgh.links[0].from_socket if rgh.is_linked else float(rgh.default_value)
    nt.links.new(nb.add(nb.mul(src, nb.sub(1.0, sv)), nb.mul(sv, 0.85)), rgh)


def droneship_at_sea(coll, flood=0.0):
    ship = droneship.build(coll)
    deck = -droneship.WATERLINE_Z
    ship.location = (0.0, 0.0, deck)               # bow toward +X, blast walls at the -X (sunset) end
    droneship.set_flood_intensity(ship, flood, nav=1.0)
    return ship, deck


FOOT = 3.36                     # deployed leg feet below the nozzle exits (m)


@shot
def s11_landing(scn, n):
    """Sunset at sea: the booster drops onto the droneship on one engine, legs swinging down,
    flame splashing across the deck, and settles in a burst of steam."""
    coll = flight_setup(scn, n)
    sunset_world(scn, coll, sky_strength=SKY_K, sun_energy=SUN_K)
    ship, deck = droneship_at_sea(coll)
    V = C.vehicle()
    root, s1 = V["F9_Root"], V["S1"]
    C.set_visible([V["S2"]] + C.descendants(V["S2"]), False)
    soot()
    dur = (n - 1) / C.FPS
    T_LEGS, T_LATCH, T_TD, T_CUT = 3.0, 4.4, 6.6, 6.8
    A, V_TD = 7.0, 1.6

    def tau(T):
        return max(0.0, T_TD - T)

    def hn(T):                  # nozzle-exit height above the deck
        return FOOT + 0.5 * A * tau(T) ** 2 + V_TD * tau(T)

    def side(T):
        return (tau(T) / T_TD) ** 2

    def power(T):
        return 1.0 - C.smooth((T - T_CUT) / 0.12)

    def root_fn(t, f):
        T = t * dur
        return (Vector((6.0 * side(T), -3.0 * side(T), deck + hn(T))),
                Euler((math.radians(1.0) * side(T), math.radians(-1.8) * side(T), 0.0)))

    C.key_object(root, n, root_fn, rot_mode="XYZ")
    rig = V["F9_Rig"]
    rig["gridfin_deploy"] = 1.0
    C.key_prop(rig, '["leg_deploy"]', n, lambda t, f: C.clamp01((t * dur - T_LEGS) / (T_LATCH - T_LEGS)) ** 1.7)
    pl = fx.plume(s1, n, name="LandingPlume", length=36.0, r_end=6.5, intensity=LAND_PLUME_K, flow=150.0,
                  engines="one", coll=coll)
    fx.key_socket(pl["power"], n, lambda t, f: power(t * dur))
    lt = fx.plume_light(s1, "LandingLight", z=-6.0, energy=0.0, radius=2.5, coll=coll)
    C.key_object(lt, n, lambda t, f: (Vector((0.0, 0.0, -min(6.0, hn(t * dur) - 1.0))), Quaternion()))
    C.key_prop(lt.data, "energy", n, lambda t, f: LAND_LIGHT_K * power(t * dur))

    def splash(T):
        return power(T) * C.smooth((34.0 - hn(T)) / 24.0)

    fx.flame_pool(n, "DeckFlame", Vector((0.0, 0.0, deck + 0.05)), 15.0,
                  lambda t, f: DECK_FLAME_K * splash(t * dur), coll=coll)
    dl = C.point_light("DeckFlameLight", 0.0, radius=4.0, color=(1.0, 0.6, 0.3), loc=(0.0, 0.0, deck + 1.5),
                       coll=coll)
    C.key_prop(dl.data, "energy", n, lambda t, f: 0.45 * LAND_LIGHT_K * splash(t * dur))
    fx.steam(n, "Steam", Vector((0.0, 0.0, deck)), 16.0, 20.0,
             lambda t, f: STEAM_K * C.smooth((t * dur - 6.0) / 1.2), rise=2.5, grow=0.6,
             color=(0.86, 0.84, 0.8), coll=coll)
    cam = C.camera("Cam_s11", lens=50.0, coll=coll)
    sh = C.Shake(11, 4, 0.18)

    def tgt(t):
        T = t * dur
        zb, zf = deck + hn(T) + 16.0, deck + 19.0
        return Vector((2.0 * side(T), -1.0 * side(T), 0.5 * (zb + zf + math.sqrt((zb - zf) ** 2 + 36.0))))

    # off the bow quarter looking back toward the sunset, the walls behind the booster
    C.key_camera(cam, n, lambda t: Vector((163.0, 76.0, 14.0)), tgt, shake=lambda s: 0.25 * sh(s))


LAND_PLUME_K, LAND_LIGHT_K, DECK_FLAME_K, STEAM_K = 1.2, 1.0e5, 0.7, 0.1
END_SKY_K, END_KEY_K = 1.0, 6.0e4


@shot
def s12_end(scn, n):
    """Blue hour: the booster stands on the droneship under its floodlights, the last glow of
    sunset behind it, as a drone drifts around; titles land on the sky to the right."""
    coll = flight_setup(scn, n, mb=False, glare=0.5, threshold=1.4)
    w, sk = envs.sky(scn, -4.0, SUN_AZ, strength=END_SKY_K, air=1.0, aerosol=1.6)
    envs.ocean(sk, strength=END_SKY_K, coll=coll)
    ship, deck = droneship_at_sea(coll, flood=1.0)
    V = C.vehicle()
    root = V["F9_Root"]
    C.set_visible([V["S2"]] + C.descendants(V["S2"]), False)
    soot()
    root.location = (0.0, 0.0, deck + FOOT)
    rig = V["F9_Rig"]
    rig["leg_deploy"] = 1.0
    rig["gridfin_deploy"] = 1.0
    fx.steam(n, "Vapour", Vector((0.0, 0.0, deck)), 10.0, 12.0, lambda t, f: 0.05, rise=0.8, grow=0.3, coll=coll)
    # deck-edge work lights picking the booster out of the dusk
    for name, (x, y), e in (("BoosterLight_A", (24.0, -18.0), END_KEY_K), ("BoosterLight_B", (-4.0, 26.0), 0.5 * END_KEY_K)):
        ld = bpy.data.lights.new(name, "SPOT")
        ld.energy = e
        ld.spot_size = math.radians(70.0)
        ld.spot_blend = 0.5
        ld.shadow_soft_size = 0.4
        ld.color = (1.0, 0.86, 0.68)
        ob = bpy.data.objects.new(name, ld)
        coll.objects.link(ob)
        ob.location = (x, y, deck + 5.0)
        C.aim(ob, Vector((0.0, 0.0, deck + 22.0)))
    cam = C.camera("Cam_s12", lens=35.0, coll=coll)

    def pos(t):
        k = glide(t)
        return cyl(C.lerp(150.0, 134.0, k), C.lerp(340.0, 352.0, k), C.lerp(22.0, 18.0, k))

    C.key_camera(cam, n, pos, lambda t: Vector((0.0, 0.0, deck + 23.0)), shift_fn=lambda t: (0.13, 0.0))
