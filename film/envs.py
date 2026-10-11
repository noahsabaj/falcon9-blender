"""Film environments: dark studio, sunset/twilight sky, space with Earth, ocean."""
import math

import bmesh
import bpy
from mathutils import Vector

from . import common as C
from f9 import materials as M


def _world(name):
    w = bpy.data.worlds.new(name)
    bpy.context.scene.world = w
    w.use_nodes = True
    nt = w.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    out = nt.nodes.new("ShaderNodeOutputWorld")
    out.location = (600, 0)
    return w, nt, out


def studio(scn, color=(0.0011, 0.0012, 0.0015), horizon=None):
    """Near-black void; the shots light the vehicle with area lights.  horizon=(rgb, width)
    adds a faint glow band at the horizon (an infinity-cove look behind a floor)."""
    w, nt, out = _world("F9_Studio")
    bg = nt.nodes.new("ShaderNodeBackground")
    bg.location = (350, 0)
    bg.inputs["Color"].default_value = (*color, 1.0)
    bg.inputs["Strength"].default_value = 1.0
    nt.links.new(bg.outputs["Background"], out.inputs["Surface"])
    if horizon:
        glow, width = horizon
        tc = nt.nodes.new("ShaderNodeTexCoord")
        tc.location = (-800, 0)
        sep = nt.nodes.new("ShaderNodeSeparateXYZ")
        sep.location = (-600, 0)
        nt.links.new(tc.outputs["Generated"], sep.inputs[0])
        k = nt.nodes.new("ShaderNodeMath")
        k.operation = "ABSOLUTE"
        k.location = (-420, 0)
        nt.links.new(sep.outputs["Z"], k.inputs[0])
        m = nt.nodes.new("ShaderNodeMath")
        m.operation = "MULTIPLY"
        m.location = (-260, 0)
        m.inputs[1].default_value = -1.0 / width
        nt.links.new(k.outputs[0], m.inputs[0])
        e = nt.nodes.new("ShaderNodeMath")
        e.operation = "EXPONENT"
        e.location = (-100, 0)
        nt.links.new(m.outputs[0], e.inputs[0])
        mix = nt.nodes.new("ShaderNodeMix")
        mix.data_type = "RGBA"
        mix.location = (120, 0)
        a = [s for s in mix.inputs if s.name == "A" and s.type == "RGBA"][0]
        b = [s for s in mix.inputs if s.name == "B" and s.type == "RGBA"][0]
        a.default_value = (*color, 1.0)
        b.default_value = (*glow, 1.0)
        nt.links.new(e.outputs[0], mix.inputs[0])
        nt.links.new([s for s in mix.outputs if s.type == "RGBA"][0], bg.inputs["Color"])
    return w


def _rgba_io(node):
    a = [s for s in node.inputs if s.name == "A" and s.type == "RGBA"][0]
    b = [s for s in node.inputs if s.name == "B" and s.type == "RGBA"][0]
    out = [s for s in node.outputs if s.type == "RGBA"][0]
    return a, b, out


def backlight(world, direction, color, power):
    """Add a soft glow to the world around `direction` (a big light far behind the subject):
    colour * max(0, dot(view, direction)) ** power, added to the existing background."""
    nt = world.node_tree
    bg = next(nd for nd in nt.nodes if nd.type == "BACKGROUND")
    src = bg.inputs["Color"].links[0].from_socket if bg.inputs["Color"].is_linked else None
    base = tuple(bg.inputs["Color"].default_value)
    tc = nt.nodes.new("ShaderNodeTexCoord")
    tc.location = (-800, -300)
    dot = nt.nodes.new("ShaderNodeVectorMath")
    dot.operation = "DOT_PRODUCT"
    dot.location = (-600, -300)
    nt.links.new(tc.outputs["Generated"], dot.inputs[0])
    dot.inputs[1].default_value = tuple(Vector(direction).normalized())
    mx = nt.nodes.new("ShaderNodeMath")
    mx.operation = "MAXIMUM"
    mx.location = (-420, -300)
    mx.inputs[1].default_value = 0.0
    nt.links.new(dot.outputs["Value"], mx.inputs[0])
    pw = nt.nodes.new("ShaderNodeMath")
    pw.operation = "POWER"
    pw.location = (-260, -300)
    pw.inputs[1].default_value = power
    nt.links.new(mx.outputs[0], pw.inputs[0])
    sc = nt.nodes.new("ShaderNodeMix")
    sc.data_type = "RGBA"
    sc.blend_type = "MULTIPLY"
    sc.location = (0, -350)
    sc.inputs[0].default_value = 1.0
    sa, sb, sout = _rgba_io(sc)
    sa.default_value = (*color, 1.0)
    cv = nt.nodes.new("ShaderNodeCombineXYZ")
    cv.location = (-100, -450)
    for i in range(3):
        nt.links.new(pw.outputs[0], cv.inputs[i])
    nt.links.new(cv.outputs[0], sb)
    add = nt.nodes.new("ShaderNodeMix")
    add.data_type = "RGBA"
    add.blend_type = "ADD"
    add.location = (180, -200)
    add.inputs[0].default_value = 1.0
    a, b, out = _rgba_io(add)
    if src is not None:
        nt.links.new(src, a)
    else:
        a.default_value = base
    nt.links.new(sout, b)
    nt.links.new(out, bg.inputs["Color"])


def _quad(name, half, z=0.0, coll=None):
    me = bpy.data.meshes.new(name)
    me.from_pydata([(-half, -half, 0.0), (half, -half, 0.0), (half, half, 0.0), (-half, half, 0.0)],
                   [], [(0, 1, 2, 3)])
    ob = bpy.data.objects.new(name, me)
    (coll or bpy.context.scene.collection).objects.link(ob)
    ob.location.z = z
    return ob


def studio_floor(z=-0.02, half=400.0, color=(0.0035, 0.0037, 0.0042), rough=0.14):
    """Black lacquer floor: a soft mirror that grounds the vehicle in the void."""
    ob = _quad("Studio_Floor", half, z)
    m, nt, b = M._base("Studio_Floor")
    b.inputs["Base Color"].default_value = (*color, 1.0)
    b.inputs["Roughness"].default_value = rough
    nz = M._noise(nt, 0.15, detail=4.0, coord="Object")
    mr = M._maprange(nt, nz.outputs["Fac"], rough * 0.7, rough * 1.5)
    nt.links.new(mr.outputs["Result"], b.inputs["Roughness"])
    ob.data.materials.append(m)
    return ob


def haze(density=0.0015, size=(260.0, 260.0, 120.0), anisotropy=0.55, color=(0.85, 0.9, 1.0), center=None):
    """Thin scattering volume around the vehicle so backlights bloom into the air.
    `center` defaults to the box standing on z = -0.5."""
    import bmesh
    me = bpy.data.meshes.new("Haze")
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    bm.to_mesh(me)
    bm.free()
    ob = bpy.data.objects.new("Haze", me)
    bpy.context.scene.collection.objects.link(ob)
    ob.scale = size
    ob.location = center if center is not None else (0.0, 0.0, size[2] / 2 - 0.5)
    m = bpy.data.materials.new("Haze")
    m.use_nodes = True
    nt = m.node_tree
    for nd in list(nt.nodes):
        nt.nodes.remove(nd)
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    vol = nt.nodes.new("ShaderNodeVolumePrincipled")
    vol.inputs["Density"].default_value = density
    vol.inputs["Anisotropy"].default_value = anisotropy
    vol.inputs["Color"].default_value = (*color, 1.0)
    nt.links.new(vol.outputs["Volume"], out.inputs["Volume"])
    me.materials.append(m)
    ob.visible_shadow = False
    return ob


# --- daylight / sunset sky ----------------------------------------------------------------
def sky(scn, sun_el, sun_az, strength=1.0, altitude=0.0, air=1.0, aerosol=1.0, ozone=1.0,
        sun_disc=False, kind="MULTIPLE_SCATTERING"):
    """Physical sky.  sun_az is measured like dims.py azimuths (from +X toward +Y, degrees),
    the same convention as common.sun_dir; Blender's sun_rotation puts the sun at azimuth
    rotation - 90 deg (measured: rotation = az - 90 put the glow at az + 180)."""
    w, nt, out = _world("F9_Sky")
    sk = nt.nodes.new("ShaderNodeTexSky")
    sk.location = (0, 0)
    sk.sky_type = kind
    sk.sun_elevation = math.radians(sun_el)
    sk.sun_rotation = math.radians(sun_az + 90.0)
    sk.altitude = altitude
    sk.air_density = air
    sk.aerosol_density = aerosol
    sk.ozone_density = ozone
    sk.sun_disc = sun_disc
    bg = nt.nodes.new("ShaderNodeBackground")
    bg.location = (350, 0)
    bg.inputs["Strength"].default_value = strength
    nt.links.new(sk.outputs["Color"], bg.inputs["Color"])
    nt.links.new(bg.outputs["Background"], out.inputs["Surface"])
    # below the horizon show the horizon colour instead of the sky model's black ground
    tc = nt.nodes.new("ShaderNodeTexCoord")
    tc.location = (-700, 0)
    sp = nt.nodes.new("ShaderNodeSeparateXYZ")
    sp.location = (-520, 0)
    nt.links.new(tc.outputs["Generated"], sp.inputs[0])
    mx = nt.nodes.new("ShaderNodeMath")
    mx.operation = "MAXIMUM"
    mx.location = (-360, -80)
    mx.inputs[1].default_value = 0.002
    nt.links.new(sp.outputs["Z"], mx.inputs[0])
    cb = nt.nodes.new("ShaderNodeCombineXYZ")
    cb.location = (-200, 0)
    nt.links.new(sp.outputs["X"], cb.inputs[0])
    nt.links.new(sp.outputs["Y"], cb.inputs[1])
    nt.links.new(mx.outputs[0], cb.inputs[2])
    nm = nt.nodes.new("ShaderNodeVectorMath")
    nm.operation = "NORMALIZE"
    nm.location = (-100, -150)
    nt.links.new(cb.outputs[0], nm.inputs[0])
    nt.links.new(nm.outputs["Vector"], sk.inputs["Vector"])
    return w, sk


def sky_copy(nt, sky_node, loc=(-600, 300)):
    """Duplicate the world's sky node settings into another node tree (for aerial haze)."""
    sk = nt.nodes.new("ShaderNodeTexSky")
    sk.location = loc
    for attr in ("sky_type", "sun_elevation", "sun_rotation", "altitude", "air_density", "aerosol_density",
                 "ozone_density"):
        setattr(sk, attr, getattr(sky_node, attr))
    sk.sun_disc = False
    return sk


def ocean(sky_node, strength=1.0, half=2000e3, color=(0.003, 0.010, 0.018), rough=0.06,
          haze_km=30.0, wave_scale=0.004, wave_bump=0.25, coll=None, name="Ocean"):
    """Sea to the horizon: glossy dark water whose colour blends into the sky's horizon colour
    with distance (aerial perspective), so the far edge melts into the sunset."""
    ob = _quad(name, half, 0.0, coll)
    m, nt, b = M._base(name)
    b.inputs["Base Color"].default_value = (*color, 1.0)
    b.inputs["Roughness"].default_value = rough
    b.inputs["Specular IOR Level"].default_value = 0.5
    b.inputs["IOR"].default_value = 1.33
    # waves: two noise scales into bump (world space so the sea does not swim with objects)
    geo = nt.nodes.new("ShaderNodeNewGeometry")
    geo.location = (-1400, -400)
    nz = nt.nodes.new("ShaderNodeTexNoise")
    nz.location = (-1150, -400)
    nz.inputs["Scale"].default_value = wave_scale
    nz.inputs["Detail"].default_value = 8.0
    nz.inputs["Roughness"].default_value = 0.6
    nt.links.new(geo.outputs["Position"], nz.inputs["Vector"])
    bp = nt.nodes.new("ShaderNodeBump")
    bp.location = (-900, -400)
    bp.inputs["Strength"].default_value = wave_bump
    nt.links.new(nz.outputs["Fac"], bp.inputs["Height"])
    nt.links.new(bp.outputs["Normal"], b.inputs["Normal"])
    # aerial perspective
    cam = nt.nodes.new("ShaderNodeCameraData")
    cam.location = (-1400, 300)
    k = nt.nodes.new("ShaderNodeMath")
    k.operation = "MULTIPLY"
    k.location = (-1150, 300)
    k.inputs[1].default_value = -1.0 / (haze_km * 1000.0)
    nt.links.new(cam.outputs["View Distance"], k.inputs[0])
    ex = nt.nodes.new("ShaderNodeMath")
    ex.operation = "EXPONENT"
    ex.location = (-950, 300)
    nt.links.new(k.outputs[0], ex.inputs[0])
    fac = nt.nodes.new("ShaderNodeMath")
    fac.operation = "SUBTRACT"
    fac.location = (-750, 300)
    fac.inputs[0].default_value = 1.0
    nt.links.new(ex.outputs[0], fac.inputs[1])
    # horizon colour in the viewing direction: -Incoming with z pinned just above 0
    inc = nt.nodes.new("ShaderNodeVectorMath")
    inc.operation = "SCALE"
    inc.location = (-1150, 600)
    inc.inputs["Scale"].default_value = -1.0
    nt.links.new(geo.outputs["Incoming"], inc.inputs[0])
    sp = nt.nodes.new("ShaderNodeSeparateXYZ")
    sp.location = (-950, 600)
    nt.links.new(inc.outputs["Vector"], sp.inputs[0])
    cb = nt.nodes.new("ShaderNodeCombineXYZ")
    cb.location = (-800, 600)
    nt.links.new(sp.outputs["X"], cb.inputs[0])
    nt.links.new(sp.outputs["Y"], cb.inputs[1])
    cb.inputs[2].default_value = 0.035
    nm = nt.nodes.new("ShaderNodeVectorMath")
    nm.operation = "NORMALIZE"
    nm.location = (-650, 600)
    nt.links.new(cb.outputs[0], nm.inputs[0])
    sk = sky_copy(nt, sky_node, loc=(-450, 600))
    nt.links.new(nm.outputs["Vector"], sk.inputs["Vector"])
    em = nt.nodes.new("ShaderNodeEmission")
    em.location = (100, 400)
    em.inputs["Strength"].default_value = strength
    nt.links.new(sk.outputs["Color"], em.inputs["Color"])
    out = next(nd for nd in nt.nodes if nd.type == "OUTPUT_MATERIAL")
    out.location = (800, 0)
    mix = nt.nodes.new("ShaderNodeMixShader")
    mix.location = (600, 0)
    nt.links.new(fac.outputs[0], mix.inputs["Fac"])
    nt.links.new(b.outputs["BSDF"], mix.inputs[1])
    nt.links.new(em.outputs["Emission"], mix.inputs[2])
    nt.links.new(mix.outputs["Shader"], out.inputs["Surface"])
    ob.data.materials.append(m)
    return ob


def stars(world, density=0.9, brightness=0.6, horizon_fade=0.25, scale=900.0):
    """Add a procedural star field to the world (Voronoi points on the view direction),
    fading toward the horizon where the twilight glow and extinction hide them."""
    nt = world.node_tree
    bg = next(nd for nd in nt.nodes if nd.type == "BACKGROUND")
    src = bg.inputs["Color"].links[0].from_socket if bg.inputs["Color"].is_linked else None
    tc = nt.nodes.new("ShaderNodeTexCoord")
    tc.location = (-1200, -700)
    vo = nt.nodes.new("ShaderNodeTexVoronoi")
    vo.location = (-1000, -700)
    vo.feature = "F1"
    vo.inputs["Scale"].default_value = scale
    vo.inputs["Randomness"].default_value = 1.0
    nt.links.new(tc.outputs["Generated"], vo.inputs["Vector"])
    # point size: bright where the distance to the cell point is tiny
    pt = nt.nodes.new("ShaderNodeMapRange")
    pt.location = (-800, -700)
    pt.inputs["From Min"].default_value = 0.075
    pt.inputs["From Max"].default_value = 0.0
    nt.links.new(vo.outputs["Distance"], pt.inputs["Value"])
    # random magnitude per star: keep only a fraction, with a steep brightness distribution
    mag = nt.nodes.new("ShaderNodeMapRange")
    mag.location = (-800, -900)
    mag.inputs["From Min"].default_value = 1.0 - density * 0.5
    mag.inputs["From Max"].default_value = 1.0
    sepc = nt.nodes.new("ShaderNodeSeparateColor")
    sepc.location = (-1000, -950)
    nt.links.new(vo.outputs["Color"], sepc.inputs["Color"])
    nt.links.new(sepc.outputs[0], mag.inputs["Value"])
    pw = nt.nodes.new("ShaderNodeMath")
    pw.operation = "POWER"
    pw.location = (-600, -900)
    pw.inputs[1].default_value = 3.0
    nt.links.new(mag.outputs["Result"], pw.inputs[0])
    mul = nt.nodes.new("ShaderNodeMath")
    mul.operation = "MULTIPLY"
    mul.location = (-450, -750)
    nt.links.new(pt.outputs["Result"], mul.inputs[0])
    nt.links.new(pw.outputs[0], mul.inputs[1])
    # fade near the horizon
    sp = nt.nodes.new("ShaderNodeSeparateXYZ")
    sp.location = (-1000, -1100)
    nt.links.new(tc.outputs["Generated"], sp.inputs[0])
    hf = nt.nodes.new("ShaderNodeMapRange")
    hf.location = (-800, -1100)
    hf.inputs["From Min"].default_value = 0.02 if horizon_fade is not None else -2.0
    hf.inputs["From Max"].default_value = horizon_fade if horizon_fade is not None else -1.0
    nt.links.new(sp.outputs["Z"], hf.inputs["Value"])
    mul2 = nt.nodes.new("ShaderNodeMath")
    mul2.operation = "MULTIPLY"
    mul2.location = (-300, -800)
    nt.links.new(mul.outputs[0], mul2.inputs[0])
    nt.links.new(hf.outputs["Result"], mul2.inputs[1])
    mul3 = nt.nodes.new("ShaderNodeMath")
    mul3.operation = "MULTIPLY"
    mul3.location = (-150, -800)
    mul3.inputs[1].default_value = brightness
    nt.links.new(mul2.outputs[0], mul3.inputs[0])
    add = nt.nodes.new("ShaderNodeMix")
    add.data_type = "RGBA"
    add.blend_type = "ADD"
    add.location = (150, -500)
    add.inputs[0].default_value = 1.0
    a, b, out = _rgba_io(add)
    if src is not None:
        nt.links.new(src, a)
    else:
        a.default_value = tuple(bg.inputs["Color"].default_value)
    tint = nt.nodes.new("ShaderNodeCombineColor")
    tint.location = (0, -800)
    for i in range(3):
        nt.links.new(mul3.outputs[0], tint.inputs[i])
    nt.links.new(tint.outputs["Color"], b)
    nt.links.new(out, bg.inputs["Color"])


def treeline(dist=2500.0, az0=-40.0, az1=200.0, h0=8.0, h1=26.0, seed=3, coll=None, name="Treeline"):
    """Black silhouette of a distant treeline: an arc strip at `dist` whose top edge is
    noisy (tree crowns), plus a black ground disc out to it."""
    import random
    rnd = random.Random(seed)
    bm = bmesh.new()
    n = 1600
    bottom, top = [], []
    h = (h0 + h1) / 2
    for i in range(n + 1):
        a = math.radians(az0 + (az1 - az0) * i / n)
        h += rnd.uniform(-1.6, 1.6)
        h = max(h0, min(h1, h))
        crown = h + abs(math.sin(i * 0.9 + rnd.uniform(0, 3))) * rnd.uniform(0.0, 3.5)
        x, y = dist * math.cos(a), dist * math.sin(a)
        bottom.append(bm.verts.new((x, y, -2.0)))
        top.append(bm.verts.new((x, y, crown)))
    for i in range(n):
        bm.faces.new((bottom[i], bottom[i + 1], top[i + 1], top[i]))
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    ob = bpy.data.objects.new(name, me)
    (coll or bpy.context.scene.collection).objects.link(ob)
    m, nt, b = M._base(name)
    b.inputs["Base Color"].default_value = (0.0, 0.0, 0.0, 1.0)
    b.inputs["Roughness"].default_value = 1.0
    b.inputs["Specular IOR Level"].default_value = 0.0
    ob.data.materials.append(m)
    ob.visible_shadow = False
    g = _quad(name + "_Ground", dist * 1.05, -0.5, coll)
    g.data.materials.append(m)
    g.visible_shadow = False
    return ob


# --- space: black sky, procedural Earth, atmosphere limb ----------------------------------
R_EARTH = 6.371e6


def space(scn, stars_brightness=2.0):
    w, nt, out = _world("F9_Space")
    bg = nt.nodes.new("ShaderNodeBackground")
    bg.location = (350, 0)
    bg.inputs["Color"].default_value = (0.0, 0.0, 0.0, 1.0)
    nt.links.new(bg.outputs["Background"], out.inputs["Surface"])
    if stars_brightness > 0:
        stars(w, brightness=stars_brightness, scale=500.0, horizon_fade=None)
    return w


def _uv_sphere(name, radius, segs, rings, coll):
    """Smooth UV sphere (poles on Z, outward normals) built with numpy: bmesh's
    create_uvsphere takes minutes at Earth resolution."""
    import numpy as np
    th = np.linspace(0.0, np.pi, rings + 1)[1:-1]
    ph = np.linspace(0.0, 2.0 * np.pi, segs, endpoint=False)
    rows = len(th)
    ring = np.column_stack([np.outer(np.sin(th), np.cos(ph)).ravel(), np.outer(np.sin(th), np.sin(ph)).ravel(),
                            np.repeat(np.cos(th), segs)])
    verts = np.vstack([ring, [[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]]]) * radius
    top, bot = rows * segs, rows * segs + 1
    j = np.arange(segs)
    jn = (j + 1) % segs
    r = np.arange(rows - 1)[:, None]
    quads = np.stack([r * segs + j, (r + 1) * segs + j, (r + 1) * segs + jn, r * segs + jn], -1).reshape(-1, 4)
    last = (rows - 1) * segs
    caps = np.concatenate([np.stack([np.full(segs, top), j, jn], -1), np.stack([np.full(segs, bot), last + jn, last + j], -1)])
    loops = np.concatenate([quads.ravel(), caps.ravel()]).astype(np.int32)
    sizes = np.concatenate([np.full(len(quads), 4), np.full(len(caps), 3)])
    me = bpy.data.meshes.new(name)
    me.vertices.add(len(verts))
    me.vertices.foreach_set("co", verts.astype(np.float32).ravel())
    me.loops.add(len(loops))
    me.loops.foreach_set("vertex_index", loops)
    me.polygons.add(len(sizes))
    me.polygons.foreach_set("loop_start", np.concatenate([[0], np.cumsum(sizes)[:-1]]).astype(np.int32))
    me.update(calc_edges=True)
    me.validate()
    me.shade_smooth()
    ob = bpy.data.objects.new(name, me)
    (coll or bpy.context.scene.collection).objects.link(ob)
    return ob


def earth(sun_dir, altitude=80e3, coll=None, cloud_seed=0.0, city=0.25):
    """Earth below an observer at the origin: ocean + fractal clouds lit by the real sun
    lamp (so the terminator falls where it should), faint city lights on the night side, and
    a thin atmosphere shell whose limb glows blue by day and orange at the terminator.
    Returns (earth, atmosphere)."""
    from film.fx import NB
    centre = Vector((0.0, 0.0, -(R_EARTH + altitude)))
    ob = _uv_sphere("Earth", R_EARTH, 1024, 512, coll)
    ob.location = centre
    ob.visible_shadow = False
    m, nt, b = M._base("Earth")
    nb = NB(nt)
    p = nb.coord("Object")
    pc = nb.vmath("ADD", p, (cloud_seed * 1e6, 0.0, 0.0))
    warp = nb.noise(pc, scale=2.0e-6, detail=3.0, rough=0.5)
    pw = nb.vmath("ADD", pc, nb.vmath("SCALE", nb.comb(warp, warp, warp), scale=4.0e5))
    c1 = nb.noise(pw, scale=4.5e-6, detail=10.0, rough=0.62, distortion=0.0)
    c2 = nb.noise(pw, scale=2.6e-5, detail=6.0, rough=0.6)
    clouds = nb.mul(nb.smooth(0.50, 0.66, c1), nb.add(0.55, nb.mul(c2, 0.6)))
    clouds = nb.clamp01(clouds)
    ocean = (0.004, 0.018, 0.055, 1.0)
    col = nb.mixrgb(clouds, ocean, (0.82, 0.84, 0.86, 1.0))
    nt.links.new(col, b.inputs["Base Color"])
    nt.links.new(nb.maprange(clouds, 0.0, 1.0, 0.22, 0.95), b.inputs["Roughness"])
    b.inputs["Specular IOR Level"].default_value = 0.5
    # city lights on the night side (clustered noise, masked to where the sun is down)
    geo = nb._node("ShaderNodeNewGeometry")
    sd = nb.vmath("DOT_PRODUCT", geo.outputs["Normal"], tuple(Vector(sun_dir).normalized()))
    night = nb.sub(1.0, nb.smooth(-0.12, 0.02, sd))
    lights = nb.mul(nb.smooth(0.62, 0.8, nb.noise(p, scale=6e-5, detail=8.0, rough=0.7)),
                    nb.sub(1.0, clouds))
    em = nb.mul(nb.mul(lights, night), city)
    nt.links.new(em, b.inputs["Emission Strength"])
    b.inputs["Emission Color"].default_value = (1.0, 0.62, 0.28, 1.0)
    ob.data.materials.append(m)
    # atmosphere shell
    atm = _uv_sphere("Atmosphere", R_EARTH + 60e3, 1024, 512, coll)
    atm.location = centre
    atm.visible_shadow = False
    am = bpy.data.materials.new("Atmosphere")
    am.use_nodes = True
    at = am.node_tree
    for nd in list(at.nodes):
        at.nodes.remove(nd)
    ao = at.nodes.new("ShaderNodeOutputMaterial")
    ao.location = (900, 0)
    ab = NB(at)
    lw = ab._node("ShaderNodeLayerWeight")
    lw.inputs["Blend"].default_value = 0.5
    limb = ab.pow(lw.outputs["Facing"], 6.0)
    haze = ab.mul(ab.pow(lw.outputs["Facing"], 2.5), 0.035)
    g2 = ab._node("ShaderNodeNewGeometry")
    sda = ab.vmath("DOT_PRODUCT", g2.outputs["Normal"], tuple(Vector(sun_dir).normalized()))
    lit = ab.smooth(-0.18, 0.12, sda)
    dusk = ab.mul(ab.sub(1.0, ab.smooth(0.0, 0.16, sda)), ab.smooth(-0.18, -0.02, sda))
    acol = ab.mixrgb(dusk, (0.28, 0.52, 1.0, 1.0), (1.0, 0.42, 0.14, 1.0))
    e = ab.mul(ab.add(limb, haze), ab.mul(lit, 2.6))
    aem = at.nodes.new("ShaderNodeEmission")
    aem.location = (500, 100)
    at.links.new(acol, aem.inputs["Color"])
    at.links.new(e, aem.inputs["Strength"])
    atr = at.nodes.new("ShaderNodeBsdfTransparent")
    atr.location = (500, -100)
    add = at.nodes.new("ShaderNodeAddShader")
    add.location = (700, 0)
    at.links.new(aem.outputs["Emission"], add.inputs[0])
    at.links.new(atr.outputs["BSDF"], add.inputs[1])
    at.links.new(add.outputs["Shader"], ao.inputs["Surface"])
    atm.data.materials.append(am)
    return ob, atm
