"""SpaceX-style Autonomous Spaceport Drone Ship (ASDS) - generic, no names or logos.

Modelled on a Marmac 300-class deck barge (300 x 100 x 20 ft = 91.4 x 30.5 x 6.1 m) as
converted by SpaceX: outboard "wing" extensions widen the deck to 52 m (they taper into
the bow corners), tall corrugated blast walls at the stern shield two equipment pods
(generator / control containers, fuel tanks, a pedestal crane) and a robot garage with
the antenna mast behind it, a low wall closes the bow, light masts stand along the long
edges, and the dark, sooty steel deck carries the white landing target (39 m ring + X).

Frame: 1 BU = 1 m, Z up, long axis on X with the bow toward +X, deck top surface at
z = 0 and the deck centre at the origin of the root empty.  The sea surface belongs at
z = WATERLINE_Z (freeboard 2.9 m, draft 3.2 m).

    import droneship
    root = droneship.build(coll, M=None)            # -> "ASDS_Root" empty (all parts parented)
    droneship.set_flood_intensity(root, 1.0, nav=1.0)   # 0 = off; drives lenses + spot lights
    # ...or keyframe the custom properties root["flood"] / root["nav"] directly
    droneship.triangle_count(root)

Materials are procedural (Principled BSDF + noise / brick / math nodes, object space);
the deck shader draws the landing target analytically so it stays crisp at any range.
"""
import math
import os
import sys

import bmesh
import bpy
from mathutils import Matrix, Vector

_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

from f9 import materials as MT  # noqa: E402
from f9 import util as U  # noqa: E402

# ---------------------------------------------------------------------------------
# dimensions (metres)
# ---------------------------------------------------------------------------------
L_HULL = 91.44
B_HULL = 30.48
D_HULL = 6.10
XS, XB = -L_HULL / 2, L_HULL / 2        # stern / bow
YH = B_HULL / 2                         # hull side
YW = 26.0                               # wing outer edge -> 52 m deck
X_CHAMFER = 36.0                        # wing edge tapers from here into the bow corners
WATERLINE_Z = -2.9
DECK_T = 0.03
WING_D_IN, WING_D_OUT = 2.05, 1.25      # wing depth at the hull side / at the outer edge
RAKE_BOW, RAKE_STERN = 10.0, 7.0
X_WALL = -33.0                          # forward face of the stern blast walls
WALL_H = 7.0
Y_POD = 4.6                             # inboard face of the stern equipment pods
X_BOW_WALL = 40.5
BOW_WALL_H = 1.5
RING_R, RING_HW = 18.9, 0.6             # target ring: 39.0 m outer diameter, 1.2 m line
X_HL, X_HW = 11.5, 1.6                  # target X: half-length along a diagonal, bar half-width

DECK_OUTLINE = [(XS, -YW), (X_CHAMFER, -YW), (XB, -YH), (XB, YH), (X_CHAMFER, YW), (XS, YW)]

Z_UP = Vector((0.0, 0.0, 1.0))


def deck_half_width(x):
    """Half-width of the deck at station x."""
    if x <= X_CHAMFER:
        return YW
    return YW - (x - X_CHAMFER) / (XB - X_CHAMFER) * (YW - YH)


# ---------------------------------------------------------------------------------
# shader-graph builder
# ---------------------------------------------------------------------------------
def _sock(coll, ident):
    for s in coll:
        if s.identifier == ident:
            return s
    raise KeyError(ident)


def _lin(hexstr, k=1.0):
    c = MT.srgb(hexstr)
    return (c[0] * k, c[1] * k, c[2] * k, 1.0)


class _G:
    """Tiny node-graph builder.  Inputs may be sockets or constants."""

    def __init__(self, mat):
        self.nt = mat.node_tree
        self.k = 0

    def node(self, idn, **props):
        nd = self.nt.nodes.new(idn)
        nd.location = (-400 - 230 * (self.k // 12), 900 - 160 * (self.k % 12))
        self.k += 1
        for key, val in props.items():
            setattr(nd, key, val)
        return nd

    def feed(self, sock, val):
        if val is None:
            return
        if isinstance(val, bpy.types.NodeSocket):
            self.nt.links.new(val, sock)
        else:
            sock.default_value = val

    def math(self, op, a, b=None, c=None, clamp=False):
        nd = self.node("ShaderNodeMath", operation=op, use_clamp=clamp)
        for i, v in enumerate((a, b, c)):
            self.feed(nd.inputs[i], v)
        return nd.outputs[0]

    def add(self, a, b):
        return self.math("ADD", a, b)

    def sub(self, a, b):
        return self.math("SUBTRACT", a, b)

    def mul(self, a, b, clamp=False):
        return self.math("MULTIPLY", a, b, clamp=clamp)

    def maxi(self, a, b):
        return self.math("MAXIMUM", a, b)

    def mini(self, a, b):
        return self.math("MINIMUM", a, b)

    def absf(self, a):
        return self.math("ABSOLUTE", a)

    def vec(self, op, a, b=None, scale=None):
        nd = self.node("ShaderNodeVectorMath", operation=op)
        self.feed(nd.inputs[0], a)
        if b is not None:
            self.feed(nd.inputs[1], b)
        if scale is not None:
            self.feed(_sock(nd.inputs, "Scale"), scale)
        return nd.outputs["Value"] if op in ("LENGTH", "DOT_PRODUCT", "DISTANCE") else nd.outputs["Vector"]

    def coords(self, kind="Object"):
        return self.node("ShaderNodeTexCoord").outputs[kind]

    def sep(self, v):
        nd = self.node("ShaderNodeSeparateXYZ")
        self.feed(nd.inputs[0], v)
        return nd.outputs[0], nd.outputs[1], nd.outputs[2]

    def comb(self, x, y, z):
        nd = self.node("ShaderNodeCombineXYZ")
        for i, v in enumerate((x, y, z)):
            self.feed(nd.inputs[i], v)
        return nd.outputs[0]

    def noise(self, v, scale, detail=4.0, rough=0.5, distortion=0.0, lac=2.0, out=0):
        nd = self.node("ShaderNodeTexNoise")
        self.feed(nd.inputs["Vector"], v)
        nd.inputs["Scale"].default_value = scale
        nd.inputs["Detail"].default_value = detail
        nd.inputs["Roughness"].default_value = rough
        nd.inputs["Lacunarity"].default_value = lac
        nd.inputs["Distortion"].default_value = distortion
        return nd.outputs[out]

    def voronoi(self, v, scale, feature="F1", out="Distance", rand=1.0):
        nd = self.node("ShaderNodeTexVoronoi", feature=feature)
        self.feed(nd.inputs["Vector"], v)
        nd.inputs["Scale"].default_value = scale
        nd.inputs["Randomness"].default_value = rand
        return nd.outputs[out]

    def maprange(self, v, a, b, c=0.0, d=1.0, interp="LINEAR", clamp=True):
        nd = self.node("ShaderNodeMapRange", interpolation_type=interp, clamp=clamp)
        self.feed(nd.inputs[0], v)
        self.feed(nd.inputs[1], a)
        self.feed(nd.inputs[2], b)
        self.feed(nd.inputs[3], c)
        self.feed(nd.inputs[4], d)
        return nd.outputs[0]

    def mix(self, fac, a, b, blend="MIX"):
        nd = self.node("ShaderNodeMix", data_type="RGBA", blend_type=blend)
        self.feed(_sock(nd.inputs, "Factor_Float"), fac)
        self.feed(_sock(nd.inputs, "A_Color"), a)
        self.feed(_sock(nd.inputs, "B_Color"), b)
        return _sock(nd.outputs, "Result_Color")

    def mixf(self, fac, a, b):
        nd = self.node("ShaderNodeMix", data_type="FLOAT")
        self.feed(_sock(nd.inputs, "Factor_Float"), fac)
        self.feed(_sock(nd.inputs, "A_Float"), a)
        self.feed(_sock(nd.inputs, "B_Float"), b)
        return _sock(nd.outputs, "Result_Float")

    def ramp(self, fac, stops, interp="LINEAR"):
        nd = self.node("ShaderNodeValToRGB")
        cr = nd.color_ramp
        cr.interpolation = interp
        els = cr.elements
        cols = [(p, c if len(c) == 4 else (c[0], c[1], c[2], 1.0)) for p, c in stops]
        els[0].position, els[0].color = cols[0]
        els[1].position, els[1].color = cols[-1]
        for p, c in cols[1:-1]:
            e = els.new(p)
            e.color = c
        self.feed(nd.inputs[0], fac)
        return nd.outputs[0]

    def gray(self, fac, stops):
        """Ramp of plain values -> float socket."""
        col = self.ramp(fac, [(p, (v, v, v, 1.0)) for p, v in stops])
        nd = self.node("ShaderNodeSeparateColor")
        self.feed(nd.inputs[0], col)
        return nd.outputs[0]

    def bump(self, height, strength=1.0, distance=1.0, normal=None):
        nd = self.node("ShaderNodeBump")
        nd.inputs["Strength"].default_value = strength
        nd.inputs["Distance"].default_value = distance
        self.feed(nd.inputs["Height"], height)
        if normal is not None:
            self.feed(nd.inputs["Normal"], normal)
        return nd.outputs["Normal"]


def _new_mat(name):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    out.location = (500, 300)
    b = nt.nodes.new("ShaderNodeBsdfPrincipled")
    b.location = (150, 300)
    nt.links.new(b.outputs["BSDF"], out.inputs["Surface"])
    return m, _G(m), b


def _existing(name):
    return bpy.data.materials.get(name)


def _drive(idb, path, root, prop, expr):
    """Scripted driver `expr` of variable v = root[prop] (simple expression, no Python needed)."""
    fc = idb.driver_add(path)
    d = fc.driver
    d.type = "SCRIPTED"
    v = d.variables.new()
    v.name = "v"
    v.type = "SINGLE_PROP"
    v.targets[0].id_type = "OBJECT"
    v.targets[0].id = root
    v.targets[0].data_path = '["%s"]' % prop
    d.expression = expr
    return fc


# ---------------------------------------------------------------------------------
# materials
# ---------------------------------------------------------------------------------
def _mat_deck(name="ASDS_Deck"):
    """Dark painted steel deck plating with seams, wear, rust, puddles, the white target
    (ring + X, analytic in object space) and a soot-scorched landing zone."""
    m = _existing(name)
    if m is not None:
        return m
    m, g, b = _new_mat(name)
    co = g.coords()
    x, y, _z = g.sep(co)

    # --- landing target: signed distance in metres (negative inside the paint) ----
    r = g.vec("LENGTH", g.comb(x, y, 0.0))
    d_ring = g.sub(g.absf(g.sub(r, RING_R)), RING_HW)
    u = g.mul(g.add(x, y), 0.70710678)
    v = g.mul(g.sub(x, y), 0.70710678)
    au, av = g.absf(u), g.absf(v)
    d_x = g.mini(g.maxi(g.sub(av, X_HW), g.sub(au, X_HL)), g.maxi(g.sub(au, X_HW), g.sub(av, X_HL)))
    d_paint = g.mini(d_x, d_ring)
    ragged = g.noise(co, 2.2, detail=5.0, rough=0.6)
    d_paint = g.add(d_paint, g.mul(g.sub(ragged, 0.5), 0.05))
    paint = g.maprange(d_paint, 0.006, -0.006)
    chips = g.maprange(g.noise(co, 5.0, detail=10.0, rough=0.72), 0.645, 0.69)
    worn = g.maprange(g.noise(g.vec("MULTIPLY", co, (0.3, 1.0, 1.0)), 0.9, detail=8.0, rough=0.65), 0.6, 0.8)
    paint_amt = g.mul(paint, g.sub(1.0, g.maxi(chips, g.mul(worn, 0.5))))

    # --- steel plating: 40 x 8 ft plates (per-plate tone), welded repair doublers ---
    def brick(w, hgt, mortar, smooth, vec=co, offset=0.5):
        nd = g.node("ShaderNodeTexBrick", offset=offset, offset_frequency=2, squash=1.0, squash_frequency=2)
        g.feed(nd.inputs["Vector"], vec)
        nd.inputs["Color1"].default_value = (0.0, 0.0, 0.0, 1.0)     # Color -> per-brick random value
        nd.inputs["Color2"].default_value = (1.0, 1.0, 1.0, 1.0)
        nd.inputs["Mortar"].default_value = (0.5, 0.5, 0.5, 1.0)
        nd.inputs["Scale"].default_value = 1.0
        nd.inputs["Mortar Size"].default_value = mortar
        nd.inputs["Mortar Smooth"].default_value = smooth
        nd.inputs["Bias"].default_value = 0.0
        nd.inputs["Brick Width"].default_value = w
        nd.inputs["Row Height"].default_value = hgt
        sc = g.node("ShaderNodeSeparateColor")
        g.feed(sc.inputs[0], nd.outputs["Color"])
        return sc.outputs[0], nd.outputs["Fac"]

    plate_rnd, _f0 = brick(12.19, 2.44, 0.0, 0.0)
    _r1, seam = brick(12.19, 2.44, 0.018, 0.65)
    _r2, near_seam = brick(12.19, 2.44, 0.30, 1.0)
    co_p = g.vec("ADD", co, (3.1, 0.9, 0.0))
    patch_rnd, _f3 = brick(2.1, 1.35, 0.0, 0.0, co_p, offset=0.37)
    _r4, patch_seam = brick(2.1, 1.35, 0.008, 0.4, co_p, offset=0.37)
    patch = g.maprange(patch_rnd, 0.97, 0.975)
    patch_seam = g.mul(patch_seam, patch)

    base = _lin("#323538")
    col = g.mix(1.0, base, g.maprange(plate_rnd, 0.0, 1.0, 0.86, 1.12), blend="MULTIPLY")
    col = g.mix(1.0, col, g.maprange(near_seam, 0.0, 1.0, 1.0, 0.8), blend="MULTIPLY")
    col = g.mix(g.mul(patch, 0.6), col, _lin("#3E4245"))
    tone = g.noise(co, 0.22, detail=3.0, rough=0.5)
    col = g.mix(1.0, col, g.maprange(tone, 0.3, 0.7, 0.9, 1.08), blend="MULTIPLY")
    grime = g.noise(co, 2.6, detail=9.0, rough=0.7)
    col = g.mix(1.0, col, g.maprange(grime, 0.38, 0.75, 1.1, 0.72), blend="MULTIPLY")
    dirt = g.noise(co, 0.7, detail=6.0, rough=0.6, distortion=0.3)
    col = g.mix(1.0, col, g.maprange(dirt, 0.42, 0.7, 1.06, 0.7), blend="MULTIPLY")
    # longitudinal scuffs / drag marks (lighter) and oil stains (dark, slightly glossy)
    scuff = g.maprange(g.noise(g.vec("MULTIPLY", co, (0.12, 1.0, 1.0)), 3.0, detail=6.0, rough=0.7,
                               distortion=0.3), 0.66, 0.74)
    scuff = g.mul(scuff, g.maprange(g.noise(co, 0.3, detail=2.0), 0.45, 0.65))
    col = g.mix(g.mul(scuff, 0.45), col, _lin("#6E7172"))
    oil = g.maprange(g.noise(co, 0.55, detail=6.0, rough=0.62, distortion=0.6), 0.74, 0.77)
    col = g.mix(g.mul(oil, 0.75), col, _lin("#17130F"))

    # rust: bleeding from the seams and a scatter of spots
    rust_n = g.noise(co, 1.3, detail=10.0, rough=0.72)
    rust = g.maprange(g.add(rust_n, g.mul(near_seam, 0.13)), 0.70, 0.82)
    rust = g.maxi(rust, g.mul(patch_seam, 0.6))
    rust_col = g.ramp(g.noise(co, 9.0, detail=5.0), [(0.3, _lin("#3A1D10")), (0.7, _lin("#6E3618"))])
    col = g.mix(g.mul(rust, 0.8), col, rust_col)

    # white paint (dirty, chipped)
    paint_col = g.ramp(grime, [(0.3, _lin("#DADAD3")), (0.75, _lin("#AEAEA6"))])
    col = g.mix(paint_amt, col, paint_col)

    # --- soot: dense scorch round the touchdown point, irregular radial streaks -----
    rn = g.mul(r, 1.0 / 30.0)
    s_r = g.gray(rn, [(0.0, 0.88), (0.15, 0.97), (0.3, 0.92), (0.45, 0.75), (0.6, 0.5),
                      (0.75, 0.28), (0.92, 0.1), (1.1, 0.0)])
    inv = g.math("DIVIDE", 1.0, g.maxi(r, 0.05))
    pv = g.comb(g.mul(g.mul(x, inv), 11.0), g.mul(g.mul(y, inv), 11.0), g.mul(r, 0.09))
    streak = g.maprange(g.noise(pv, 1.0, detail=7.0, rough=0.66, distortion=0.8), 0.3, 0.78)
    blot = g.maprange(g.noise(co, 0.13, detail=4.0, rough=0.6, distortion=0.4), 0.28, 0.72, 0.72, 1.0)
    t_out = g.maprange(r, 3.0, 17.0, 0.15, 0.95)
    soot = g.mul(g.mul(s_r, g.mixf(t_out, 1.0, g.add(0.15, streak))), blot, clamp=True)
    soot_fine = g.maprange(g.noise(co, 5.0, detail=8.0, rough=0.7), 0.3, 0.7, 0.85, 1.12)
    soot = g.mul(soot, soot_fine, clamp=True)
    col = g.mix(g.mul(soot, 0.96), col, _lin("#0E0C0A"))
    # grime collects round walls and fittings (non-local AO)
    aon = g.node("ShaderNodeAmbientOcclusion", samples=6, only_local=False)
    aon.inputs["Distance"].default_value = 1.2
    cav = g.sub(1.0, aon.outputs["AO"])
    col = g.mix(g.mul(cav, 0.75), col, _lin("#15130F"))

    # --- puddles: a few irregular wet patches in low spots --------------------------
    wet = g.maprange(g.noise(co, 0.22, detail=5.0, rough=0.6, distortion=0.7), 0.735, 0.75)
    wet = g.mul(wet, g.maprange(g.noise(co, 0.035, detail=2.0), 0.55, 0.62))
    col = g.mix(g.mul(wet, 0.45), col, (0.0, 0.0, 0.0, 1.0))

    rough_steel = g.maprange(g.noise(co, 9.0, detail=4.0), 0.3, 0.7, 0.64, 0.8)
    # patchy wear / grime -> broad sheen variation visible at grazing angles
    patchy = g.maprange(g.noise(co, 0.45, detail=6.0, rough=0.62, distortion=0.25), 0.3, 0.72, -0.16, 0.12)
    rough_steel = g.add(rough_steel, patchy)
    rough = g.mixf(paint_amt, rough_steel, 0.58)
    rough = g.mixf(oil, rough, 0.3)
    rough = g.mixf(rust, rough, 0.88)
    rough = g.mixf(soot, rough, 1.0)
    rough = g.mixf(wet, rough, 0.06)

    # --- relief: plate dishing between stiffeners, weld beads, grit, paint edge -----
    fy = g.math("FRACT", g.mul(y, 1.0 / 0.61))
    fx = g.math("FRACT", g.mul(x, 1.0 / 2.44))
    dish = g.mul(g.math("SINE", g.mul(fy, math.pi)),
                 g.add(0.35, g.mul(g.math("SINE", g.mul(fx, math.pi)), 0.65)))
    amp = g.maprange(g.noise(co, 0.12, detail=1.0), 0.3, 0.7, 0.0009, 0.0026)
    h = g.mul(g.mul(dish, amp), -1.0)
    h = g.add(h, g.mul(g.maxi(seam, patch_seam), 0.0022))
    h = g.add(h, g.mul(patch, 0.004))
    h = g.add(h, g.mul(g.noise(co, 55.0, detail=3.0), 0.00025))
    h = g.add(h, g.mul(paint_amt, 0.0003))
    h = g.add(h, g.mul(rust, g.mul(g.noise(co, 45.0, detail=6.0), 0.0012)))
    h = g.mul(h, g.sub(1.0, g.mul(wet, 0.95)))
    g.feed(b.inputs["Base Color"], col)
    g.feed(b.inputs["Roughness"], rough)
    g.feed(b.inputs["Normal"], g.bump(h, 1.0, 1.0))
    g.feed(b.inputs["Specular IOR Level"], g.mixf(g.maprange(soot, 0.04, 0.5), 0.5, 0.0))
    m.diffuse_color = _lin("#3A3D40")
    return m


def _mat_weathered(name, hexcol, rust=0.5, rough=0.55, metallic=0.0, waterline=None, streak_top=2.6,
                   period=2.6, soot=0.0, dish=0.0, streak_scale=1.0, panel=0.0, ao=1.0, jitter=0.0):
    """Weathered painted steel: tonal mottling, per-panel tone (panel = cell size), rust
    and dark runoff streaks that start at z = streak_top - k * period (edges, seams,
    container roofs) and fade downward over a random length, rust spots, scrapes;
    optional waterline (grime band + antifouling below) and soot rising from the deck."""
    m = _existing(name)
    if m is not None:
        return m
    m, g, b = _new_mat(name)
    co = g.coords()
    rnd = g.node("ShaderNodeObjectInfo").outputs["Random"]
    co_r = g.vec("ADD", co, g.comb(g.mul(rnd, 37.0), g.mul(rnd, 71.0), 0.0))
    _x, _y, z = g.sep(co)
    base = MT.srgb(hexcol)
    lo = tuple(c * 0.87 for c in base[:3]) + (1.0,)
    hi = tuple(min(1.0, c * 1.09) for c in base[:3]) + (1.0,)
    mott = g.noise(co_r, 0.5, detail=6.0, rough=0.6)
    col = g.ramp(mott, [(0.25, lo), (0.75, hi)])
    grime = g.noise(co_r, 1.7, detail=8.0, rough=0.65)
    col = g.mix(1.0, col, g.maprange(grime, 0.4, 0.8, 1.04, 0.8), blend="MULTIPLY")

    if panel > 0.0:
        cell = g.vec("FLOOR", g.vec("DIVIDE", co, (panel, panel, panel * 3.0)))
        wn = g.node("ShaderNodeTexWhiteNoise", noise_dimensions="3D")
        g.feed(wn.inputs["Vector"], cell)
        col = g.mix(1.0, col, g.maprange(wn.outputs["Value"], 0.0, 1.0, 0.86, 1.1), blend="MULTIPLY")
    # streaks: thin vertical columns (noise of the horizontal position only) that start
    # at an origin line and fade downward over a per-column random length
    flat = g.vec("MULTIPLY", co_r, (1.0, 1.0, 0.015))
    org = g.sub(streak_top, z)
    if jitter > 0.0:
        org = g.add(org, g.mul(g.noise(g.vec("ADD", flat, (3.0, 11.0, 0.0)), 3.0, detail=1.0), period * jitter))
    d_org = g.math("FLOORED_MODULO", org, period)
    L = g.maprange(g.noise(flat, 1.3, detail=2.0), 0.3, 0.7, 0.25, min(period, 2.8))
    fade = g.maprange(g.math("DIVIDE", d_org, L), 0.0, 1.0, 1.0, 0.0)
    fade = g.mul(fade, fade)
    runc = g.maprange(g.noise(flat, 4.0 * streak_scale, detail=5.0, rough=0.6), 0.48, 0.64)
    run = g.mul(runc, g.maprange(fade, 0.0, 1.0, 0.3, 1.0))
    col = g.mix(1.0, col, g.maprange(run, 0.0, 1.0, 1.0, 0.45), blend="MULTIPLY")
    rc = g.noise(g.vec("ADD", flat, (17.0, 5.0, 0.0)), 6.0 * streak_scale, detail=4.0, rough=0.6)
    streak = g.mul(g.maprange(rc, 0.6 - 0.06 * rust, 0.72), fade)
    spots = g.maprange(g.noise(co_r, 4.2, detail=10.0, rough=0.72), 0.735 - 0.05 * rust, 0.8)
    rust_m = g.mul(g.maxi(g.mul(streak, 0.45 + 0.55 * rust), spots), 1.0, clamp=True)
    rcol = g.ramp(g.noise(co_r, 8.0, detail=5.0), [(0.25, _lin("#3E2414")), (0.55, _lin("#64391E")),
                                                   (0.8, _lin("#83502C"))])
    col = g.mix(g.mul(rust_m, 0.85), col, rcol)
    if ao > 0.0:
        # dirt in crevices (local AO) and worn, lighter convex edges (inside AO)
        a1 = g.node("ShaderNodeAmbientOcclusion", samples=6, only_local=True)
        a1.inputs["Distance"].default_value = 0.35
        cav = g.mul(g.sub(1.0, a1.outputs["AO"]), ao)
        col = g.mix(g.mul(cav, 0.55), col, _lin("#1A1712"))
        a2 = g.node("ShaderNodeAmbientOcclusion", samples=6, only_local=True, inside=True)
        a2.inputs["Distance"].default_value = 0.006
        edge = g.maprange(g.sub(1.0, a2.outputs["AO"]), 0.3, 0.6)
        edge = g.mul(edge, g.maprange(g.noise(co_r, 6.0, detail=6.0, rough=0.65), 0.4, 0.62))
        col = g.mix(g.mul(edge, 0.7 * ao), col, g.mix(0.5, _lin("#8A8C8B"), rcol))
    # scrapes: horizontal light marks (fenders, chains, boots)
    scr = g.noise(g.vec("MULTIPLY", co_r, (0.12, 0.12, 2.5)), 2.0, detail=6.0, rough=0.7)
    scr = g.maprange(scr, 0.71, 0.76)
    col = g.mix(g.mul(scr, 0.55), col, _lin("#7B7D7C"))
    rgh = g.maprange(g.noise(co_r, 4.0, detail=5.0), 0.3, 0.7, rough - 0.08, rough + 0.1)
    rgh = g.mixf(rust_m, rgh, 0.86)
    if soot > 0.0:
        s = g.maprange(z, 0.0, 3.8, 1.0, 0.0)
        s = g.mul(s, g.maprange(g.noise(co_r, 0.9, detail=6.0), 0.3, 0.7, 0.35, 1.0))
        s = g.mul(s, soot, clamp=True)
        col = g.mix(g.mul(s, 0.85), col, _lin("#121110"))
        rgh = g.mixf(s, rgh, 0.9)
    if waterline is not None:
        dz = g.sub(z, waterline)
        below = g.maprange(dz, -0.05, -0.25, 0.0, 1.0)
        col = g.mix(below, col, _lin("#4A1E16"))
        band = g.maprange(g.absf(g.sub(dz, 0.12)), 0.1, 0.55, 1.0, 0.0)
        band = g.mul(band, g.maprange(g.noise(g.vec("MULTIPLY", co_r, (1.0, 1.0, 0.3)), 3.0, detail=5.0),
                                      0.3, 0.7, 0.5, 1.0))
        col = g.mix(g.mul(band, 0.85), col, _lin("#2A2C1E"))
        rgh = g.mixf(band, rgh, 0.35)
    h = g.mul(g.noise(co_r, 25.0, detail=4.0), 0.0004)
    h = g.add(h, g.mul(rust_m, g.mul(g.noise(co_r, 40.0, detail=6.0), 0.0012)))
    if dish > 0.0:
        fx = g.math("FRACT", g.mul(g.add(_x, _y), 1.0 / 0.61))
        h = g.sub(h, g.mul(g.math("SINE", g.mul(fx, math.pi)), dish))
    g.feed(b.inputs["Base Color"], col)
    g.feed(b.inputs["Roughness"], rgh)
    b.inputs["Metallic"].default_value = metallic
    g.feed(b.inputs["Normal"], g.bump(h, 1.0, 1.0))
    m.diffuse_color = base
    return m


def _mat_simple(name, hexcol, rough=0.5, metallic=0.0, spec=0.5, coat=0.0):
    m = _existing(name)
    if m is not None:
        return m
    m, g, b = _new_mat(name)
    b.inputs["Base Color"].default_value = MT.srgb(hexcol)
    b.inputs["Roughness"].default_value = rough
    b.inputs["Metallic"].default_value = metallic
    b.inputs["Specular IOR Level"].default_value = spec
    if coat > 0:
        b.inputs["Coat Weight"].default_value = coat
        b.inputs["Coat Roughness"].default_value = 0.05
    m.diffuse_color = MT.srgb(hexcol)
    return m


def _mat_louvre(name="ASDS_Louvre"):
    """Dark louvred grille (horizontal slats via bump)."""
    m = _existing(name)
    if m is not None:
        return m
    m, g, b = _new_mat(name)
    co = g.coords()
    _x, _y, z = g.sep(co)
    f = g.math("FRACT", g.mul(z, 1.0 / 0.06))
    h = g.mul(g.math("SINE", g.mul(f, math.pi)), 0.006)
    b.inputs["Base Color"].default_value = _lin("#232526")
    b.inputs["Roughness"].default_value = 0.6
    g.feed(b.inputs["Normal"], g.bump(h, 1.0, 1.0))
    return m


def _mat_lamp(name, root, prop, color, strength, base="#1B1C1E"):
    """Lamp lens: dark glass whose emission strength = strength * root[prop] (driver)."""
    m = _existing(name)
    if m is not None:
        return m
    m, g, b = _new_mat(name)
    b.inputs["Base Color"].default_value = MT.srgb(base)
    b.inputs["Roughness"].default_value = 0.12
    b.inputs["Emission Color"].default_value = color
    sock = b.inputs["Emission Strength"]
    sock.default_value = strength
    _drive(m.node_tree, sock.path_from_id("default_value"), root, prop, "v*%.4f" % strength)
    return m


def _materials(M, root):
    M = M or {}
    mt = {}
    mt["deck"] = _mat_deck()
    mt["hull"] = _mat_weathered("ASDS_HullPaint", "#1E2022", rust=0.6, rough=0.58, waterline=WATERLINE_Z,
                                streak_top=0.0, period=10.0, dish=0.003)
    mt["steel"] = _mat_weathered("ASDS_DarkSteel", "#303336", rust=0.45, rough=0.55)
    mt["wall"] = _mat_weathered("ASDS_BlastWall", "#3A3D40", rust=0.55, rough=0.6, soot=0.9, streak_top=WALL_H,
                                period=2.3, panel=3.04, ao=0.7, jitter=0.9)
    mt["frame"] = _mat_weathered("ASDS_WallFrame", "#34373A", rust=0.5, rough=0.58, soot=0.9,
                                 streak_top=WALL_H + 0.7, period=9.0, ao=0.6, jitter=0.25)
    mt["grey"] = _mat_weathered("ASDS_GreyPaint", "#7A7E80", rust=0.35, rough=0.5)
    mt["white"] = _mat_weathered("ASDS_WhitePaint", "#CDCFCB", rust=0.4, rough=0.45, streak_scale=1.3)
    mt["blue"] = _mat_weathered("ASDS_BluePaint", "#2C4766", rust=0.4, rough=0.5)
    mt["yellow"] = _mat_weathered("ASDS_SafetyYellow", "#D9A21E", rust=0.45, rough=0.45)
    mt["green"] = _mat_weathered("ASDS_TankGreen", "#46503F", rust=0.45, rough=0.5)
    mt["galv"] = _mat_weathered("ASDS_Galvanized", "#9C9F9E", rust=0.25, rough=0.42, metallic=0.7)
    mt["metal"] = M.get("steel") or _mat_simple("ASDS_BareSteel", "#8E8F90", rough=0.35, metallic=1.0)
    mt["rubber"] = _mat_simple("ASDS_Rubber", "#151515", rough=0.85)
    mt["black"] = _mat_simple("ASDS_BlackPlastic", "#18191A", rough=0.45)
    mt["glass"] = _mat_simple("ASDS_DarkGlass", "#0A0C0E", rough=0.04, spec=0.6)
    mt["radome"] = _mat_simple("ASDS_Radome", "#E4E4DF", rough=0.4)
    mt["soot"] = _mat_simple("ASDS_ExhaustSoot", "#100F0E", rough=0.95)
    mt["louvre"] = _mat_louvre()
    mt["mark"] = _mat_simple("ASDS_MarkingWhite", "#D8D8D2", rough=0.55)
    mt["orange"] = _mat_weathered("ASDS_LifeRingOrange", "#E2581C", rust=0.1, rough=0.5, ao=0.0)
    mt["lens"] = _mat_lamp("ASDS_FloodLens_" + root.name, root, "flood", (1.0, 0.88, 0.72, 1.0), 60.0)
    mt["nav_red"] = _mat_lamp("ASDS_NavRed_" + root.name, root, "nav", (1.0, 0.05, 0.03, 1.0), 40.0, "#3A0806")
    mt["nav_green"] = _mat_lamp("ASDS_NavGreen_" + root.name, root, "nav", (0.05, 1.0, 0.25, 1.0), 40.0,
                                "#06300E")
    return mt


# ---------------------------------------------------------------------------------
# geometry helpers
# ---------------------------------------------------------------------------------
def _bx(bm, x0, x1, y0, y1, z0, z1, mat=0):
    """Axis-aligned box from bounds."""
    return U.add_box(bm, (abs(x1 - x0), abs(y1 - y0), abs(z1 - z0)),
                     Matrix.Translation(((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2)), mat)


def _frame(x_axis, up=Z_UP, origin=(0.0, 0.0, 0.0)):
    """Matrix with local X along x_axis and local Z as close to `up` as possible."""
    x = Vector(x_axis).normalized()
    upv = Vector(up)
    if abs(x.dot(upv.normalized())) > 0.98:
        upv = Vector((1.0, 0.0, 0.0)) if abs(x.x) < 0.9 else Vector((0.0, 1.0, 0.0))
    y = upv.cross(x).normalized()
    z = x.cross(y)
    m = Matrix((x, y, z)).transposed().to_4x4()
    m.translation = Vector(origin)
    return m


def _beam(bm, p0, p1, w, h, mat=0, up=Z_UP):
    """Box beam from p0 to p1 (w across, h up)."""
    p0, p1 = Vector(p0), Vector(p1)
    L = (p1 - p0).length
    if L < 1e-6:
        return []
    return U.add_box(bm, (L, w, h), _frame(p1 - p0, up, (p0 + p1) / 2), mat)


def _cyl(bm, p0, p1, r, segs=12, mat=0, caps=True, r1=None):
    """Cylinder (or cone: radius r1 at p0, r at p1) between two points."""
    p0, p1 = Vector(p0), Vector(p1)
    d = p1 - p0
    L = d.length
    if L < 1e-6:
        return []
    dn = d / L
    q = dn.to_track_quat("Z", "X" if abs(dn.y) > 0.95 else "Y")
    m = Matrix.Translation((p0 + p1) / 2) @ q.to_matrix().to_4x4()
    return U.add_cyl(bm, r if r1 is None else r1, r, L, segs, m, mat=mat, caps=caps)


def _sweep(bm, path, prof, closed=False, mat=0, up=Z_UP):
    """Extrude a 2D profile (a = left of travel, b = up) along a polyline, mitred joints."""
    P = [Vector(p) for p in path]
    n = len(P)
    secs = []
    for i in range(n):
        if closed:
            d_in = (P[i] - P[i - 1]).normalized()
            d_out = (P[(i + 1) % n] - P[i]).normalized()
        else:
            d_in = (P[i] - P[i - 1]).normalized() if i > 0 else (P[1] - P[0]).normalized()
            d_out = (P[i + 1] - P[i]).normalized() if i < n - 1 else d_in
        t = d_in + d_out
        if t.length < 1e-6:
            t = d_out.copy()
        t.normalize()
        side = up.cross(t).normalized()
        k = 1.0 / max(0.25, side.dot(up.cross(d_in).normalized()))
        secs.append([tuple(P[i] + side * (a * k) + up * bb) for a, bb in prof])
    if closed:
        secs.append(secs[0])
    rows, faces = U.loft(bm, secs, mat=mat, cap_start=not closed, cap_end=not closed)
    if closed:
        bmesh.ops.remove_doubles(bm, verts=[v for row in (rows[0], rows[-1]) for v in row], dist=1e-5)
    return faces


def _strip(bm, pts, z0, z1, mat=0, out=None):
    """Single-sided vertical strip through 2D points; faces turned toward `out`."""
    lo = [bm.verts.new((x, y, z0)) for x, y in pts]
    hi = [bm.verts.new((x, y, z1)) for x, y in pts]
    faces = []
    for i in range(len(pts) - 1):
        f = bm.faces.new((lo[i], lo[i + 1], hi[i + 1], hi[i]))
        f.material_index = mat
        f.normal_update()
        if out is not None and f.normal.x * out[0] + f.normal.y * out[1] < 0:
            f.normal_flip()
        faces.append(f)
    return faces


def _strip_solid(bm, front, back, z0, z1, mat=0):
    """Closed vertical solid between a front and a back polyline (same point count)."""
    F0 = [bm.verts.new((x, y, z0)) for x, y in front]
    F1 = [bm.verts.new((x, y, z1)) for x, y in front]
    B0 = [bm.verts.new((x, y, z0)) for x, y in back]
    B1 = [bm.verts.new((x, y, z1)) for x, y in back]
    faces = []
    for i in range(len(front) - 1):
        faces.append(bm.faces.new((F0[i], F0[i + 1], F1[i + 1], F1[i])))
        faces.append(bm.faces.new((B0[i + 1], B0[i], B1[i], B1[i + 1])))
        faces.append(bm.faces.new((F1[i], F1[i + 1], B1[i + 1], B1[i])))
        faces.append(bm.faces.new((B0[i], B0[i + 1], F0[i + 1], F0[i])))
    faces.append(bm.faces.new((B0[0], F0[0], F1[0], B1[0])))
    faces.append(bm.faces.new((F0[-1], B0[-1], B1[-1], F1[-1])))
    for f in faces:
        f.material_index = mat
    bmesh.ops.recalc_face_normals(bm, faces=faces)
    return faces


def _corr_profile(L, pitch, depth, crest=0.3):
    """Trapezoidal corrugation (s along the sheet, t outward) over length L."""
    n = max(1, int(round(L / pitch)))
    p = L / n
    c = p * crest
    w = (p - 2 * c) / 2
    pts = []
    for k in range(n):
        s0 = k * p
        pts += [(s0, 0.0), (s0 + c, 0.0), (s0 + c + w, depth), (s0 + 2 * c + w, depth)]
    pts.append((L, 0.0))
    return pts


def _line_frame(p0, p1, front):
    """Unit direction d, front normal nrm and length for a wall from p0 to p1 (plan)."""
    a, b = Vector((p0[0], p0[1], 0.0)), Vector((p1[0], p1[1], 0.0))
    d = b - a
    L = d.length
    d.normalize()
    nrm = Vector((-d.y, d.x, 0.0))
    if nrm.x * front[0] + nrm.y * front[1] < 0:
        nrm = -nrm
    return a, d, nrm, L


def _corrugated(bm, p0, p1, z0, z1, front, pitch=0.62, depth=0.16, thick=0.012, mat=0):
    """Corrugated sheet wall from p0 to p1; corrugation crests toward `front`."""
    a, d, nrm, L = _line_frame(p0, p1, front)
    prof = _corr_profile(L, pitch, depth)
    fr = [tuple(a + d * s + nrm * t)[:2] for s, t in prof]
    bk = [tuple(a + d * s + nrm * (t - thick))[:2] for s, t in prof]
    return _strip_solid(bm, fr, bk, z0, z1, mat)


def _railing(bm, p0, p1, mat=0, h=1.07, spacing=1.85, r_post=0.024, r_rail=0.022, mid=0.55):
    """Pipe railing (stanchions + top and knee rails) on the deck from p0 to p1."""
    p0 = Vector((p0[0], p0[1], 0.0))
    p1 = Vector((p1[0], p1[1], 0.0))
    L = (p1 - p0).length
    n = max(1, int(round(L / spacing)))
    up = Vector((0.0, 0.0, 1.0))
    for k in range(n + 1):
        p = p0.lerp(p1, k / n)
        _cyl(bm, p, p + up * h, r_post, 8, mat)
        U.add_box(bm, (0.11, 0.11, 0.012), Matrix.Translation(p + up * 0.006), mat)
    _cyl(bm, p0 + up * h, p1 + up * h, r_rail * 1.1, 8, mat)
    _cyl(bm, p0 + up * mid, p1 + up * mid, r_rail, 8, mat)


def _ladder(bm, base, height, along, mat=0, width=0.44, standoff=0.16, out=None):
    """Vertical ladder: rails + rungs, with stand-off brackets toward the wall (-out)."""
    base = Vector(base)
    along = Vector(along).normalized()
    up = Vector((0.0, 0.0, 1.0))
    for s in (-1.0, 1.0):
        p = base + along * (s * width / 2)
        _cyl(bm, p, p + up * height, 0.024, 8, mat)
        if out is not None:
            o = Vector(out).normalized()
            for zb in (0.4, height - 0.3):
                _beam(bm, p + up * zb, p + up * zb - o * standoff, 0.05, 0.012, mat)
    n = int(height / 0.3)
    for k in range(1, n + 1):
        z = k * 0.3
        _cyl(bm, base - along * (width / 2) + up * z, base + along * (width / 2) + up * z, 0.012, 6, mat)


def _bitt(bm, x, y, yaw=0.0, mat=0, r=0.16, h=0.62, spread=0.62):
    """Double mooring bitt on a base plate (posts along the yaw direction)."""
    m = Matrix.Translation((x, y, 0.0)) @ Matrix.Rotation(yaw, 4, "Z")
    U.add_box(bm, (spread + 2 * r + 0.3, 2 * r + 0.24, 0.035), m @ Matrix.Translation((0, 0, 0.0175)), mat)
    for s in (-0.5, 0.5):
        mp = m @ Matrix.Translation((s * spread, 0.0, 0.0))
        U.add_cyl(bm, r, r * 0.94, h, 18, mp @ Matrix.Translation((0, 0, h / 2)), mat)
        U.add_cyl(bm, r * 1.28, r * 1.28, 0.05, 18, mp @ Matrix.Translation((0, 0, h + 0.02)), mat)
    U.add_box(bm, (spread, 0.05, 0.1), m @ Matrix.Translation((0, 0, h * 0.62)), mat)


def _chock(bm, x, y, yaw, mat=0):
    """Closed (panama) chock: an oval ring standing at the deck edge."""
    sub = bmesh.new()
    prof = []
    n = 20
    for k in range(n):
        a = U.TAU * k / n
        prof.append((0.30 * math.cos(a), 0.36 + 0.2 * math.sin(a)))
    inner = [(0.21 * math.cos(-U.TAU * k / n), 0.36 + 0.11 * math.sin(-U.TAU * k / n)) for k in range(n)]
    U.flat_prism(sub, [prof, inner], -0.06, 0.06, 0, 0, 0)
    U.add_box(sub, (0.66, 0.42, 0.12), Matrix.Translation((0.0, 0.08, 0.0)))
    # (x, y, w) -> local (x, w, y): ring stands in the XZ plane
    for v in sub.verts:
        v.co = Vector((v.co.x, v.co.z, v.co.y))
    bmesh.ops.recalc_face_normals(sub, faces=sub.faces[:])
    sub.transform(Matrix.Translation((x, y, 0.0)) @ Matrix.Rotation(yaw, 4, "Z"))
    U.merge_into(bm, sub, mat)


def _manhole(bm, x, y, mat=0, mat_bolt=0, r=0.42):
    U.add_cyl(bm, r, r - 0.01, 0.025, 28, Matrix.Translation((x, y, 0.0125)), mat)
    for k in range(12):
        a = U.TAU * k / 12
        U.add_cyl(bm, 0.022, 0.02, 0.018, 6,
                  Matrix.Translation((x + (r - 0.05) * math.cos(a), y + (r - 0.05) * math.sin(a), 0.032)), mat_bolt)
    U.add_box(bm, (0.12, 0.03, 0.03), Matrix.Translation((x, y, 0.04)), mat_bolt)


def _tyre(bm, p, axis, mat=0, major=0.42, minor=0.16):
    """Hung tyre fender: torus with its axis along `axis`."""
    q = Vector(axis).normalized().to_track_quat("Z", "Y")
    m = Matrix.Translation(p) @ q.to_matrix().to_4x4()
    U.add_torus(bm, major, minor, 20, 8, m, mat)


def _flood(bm_body, bm_lens, pos, aim, mat_body=0, mat_lens=0, w=0.56, h=0.42):
    """LED floodlight head facing `aim`; returns its frame (local +Z = beam axis)."""
    pos = Vector(pos)
    z = (Vector(aim) - pos).normalized()
    x = Z_UP.cross(z)
    if x.length < 1e-4:
        x = Vector((1.0, 0.0, 0.0))
    x.normalize()
    y = z.cross(x)
    m = Matrix((x, y, z)).transposed().to_4x4()
    m.translation = pos
    U.add_box(bm_body, (w, h, 0.13), m @ Matrix.Translation((0.0, 0.0, -0.07)), mat_body)
    for k in range(7):
        yy = -h / 2 + 0.05 + (h - 0.1) * k / 6
        U.add_box(bm_body, (w - 0.08, 0.012, 0.07), m @ Matrix.Translation((0.0, yy, -0.17)), mat_body)
    for s in (-1.0, 1.0):        # yoke arms
        U.add_box(bm_body, (0.025, 0.07, 0.36), m @ Matrix.Translation((s * (w / 2 + 0.03), 0.0, -0.2)), mat_body)
    U.add_box(bm_lens, (w - 0.06, h - 0.06, 0.012), m @ Matrix.Translation((0.0, 0.0, 0.0)), mat_lens)
    return m


def _spot(coll, root, name, frame, energy, size_deg=78.0, color=(1.0, 0.88, 0.72)):
    """Spot light along the frame's +Z, energy = energy * root['flood'] (driver)."""
    ld = bpy.data.lights.new(name, "SPOT")
    ld.energy = energy
    ld.color = color
    ld.spot_size = math.radians(size_deg)
    ld.spot_blend = 0.45
    ld.shadow_soft_size = 0.18
    _drive(ld, "energy", root, "flood", "v*%.3f" % energy)
    ob = bpy.data.objects.new(name, ld)
    coll.objects.link(ob)
    ob.parent = root
    ob.matrix_basis = frame @ Matrix.Translation((0.0, 0.0, 0.06)) @ Matrix.Rotation(math.pi, 4, "X")
    return ob


def _container(bm, cx, cy, yaw, length, mi, z0=0.0, height=2.591, width=2.438, door_end=1, genset=False):
    """ISO container (corrugated sides, corner posts, rails, door end with locking bars).

    mi: material indices (body, frame, dark interior, metal, louvre)."""
    body, frame, dark, metal, louvre = mi
    s = bmesh.new()
    hl, hw, H = length / 2, width / 2, height
    p = 0.17
    _bx(s, -hl + 0.06, hl - 0.06, -hw + 0.06, hw - 0.06, 0.12, H - 0.06, dark)
    for sx in (-1, 1):
        for sy in (-1, 1):
            x0, x1 = sorted((sx * hl, sx * (hl - p)))
            y0, y1 = sorted((sy * hw, sy * (hw - p)))
            _bx(s, x0, x1, y0, y1, 0.0, H, frame)
    for sy in (-1, 1):
        y0, y1 = sorted((sy * hw, sy * (hw - 0.12)))
        _bx(s, -hl + p, hl - p, y0, y1, 0.0, 0.17, frame)
        _bx(s, -hl + p, hl - p, y0, y1, H - 0.11, H, frame)
    for sx in (-1, 1):
        x0, x1 = sorted((sx * hl, sx * (hl - 0.12)))
        _bx(s, x0, x1, -hw + p, hw - p, 0.0, 0.22, frame)
        _bx(s, x0, x1, -hw + p, hw - p, H - 0.2, H, frame)
    # corrugated long sides
    prof = _corr_profile(length - 2 * p, 0.278, 0.036, 0.27)
    for sy in (-1, 1):
        pts = [(-hl + p + u, sy * (hw - 0.05 + t)) for u, t in prof]
        _strip(s, pts, 0.17, H - 0.11, body, out=(0.0, sy))
    # closed end
    sgn = -door_end
    prof_e = _corr_profile(width - 2 * p, 0.23, 0.03, 0.3)
    pts = [(sgn * (hl - 0.06 + t), -hw + p + u) for u, t in prof_e]
    _strip(s, pts, 0.22, H - 0.2, body, out=(sgn, 0.0))
    # door end: two leaves, four locking bars with cams and handles
    dx = door_end * (hl - 0.03)
    for side in (-1, 1):
        y0, y1 = sorted((0.0, side * (hw - p)))
        _bx(s, dx - 0.025, dx + 0.025, y0 + 0.005, y1 - 0.005, 0.22, H - 0.2, body)
        for yb in (0.3, 0.85):
            yy = side * yb
            px = dx + door_end * 0.05
            _cyl(s, (px, yy, 0.18), (px, yy, H - 0.16), 0.017, 8, metal)
            for zc in (0.22, H - 0.2):
                _bx(s, px - 0.04, px + 0.04, yy - 0.05, yy + 0.05, zc - 0.05, zc + 0.05, metal)
            _bx(s, px, px + door_end * 0.05, yy, yy + side * 0.25, 1.15, 1.2, metal)
    # roof
    _bx(s, -hl + p, hl - p, -hw + 0.12, hw - 0.12, H - 0.075, H - 0.035, body)
    if genset:
        # louvred intake panels on both long sides and a radiator hood at one end
        for sy in (-1, 1):
            for xc in (-hl * 0.45, hl * 0.2):
                _bx(s, xc - 0.9, xc + 0.9, sy * (hw + 0.005), sy * (hw - 0.03), 0.6, 2.1, louvre)
        _bx(s, hl - 2.2, hl - 0.25, -hw + 0.25, hw - 0.25, H, H + 0.55, body)
        _bx(s, hl - 2.1, hl - 0.35, -hw + 0.32, hw - 0.32, H + 0.55, H + 0.57, louvre)
    s.transform(Matrix.Translation((cx, cy, z0)) @ Matrix.Rotation(yaw, 4, "Z"))
    U.merge_into(bm, s)


def _exhaust(bm, x, y, z0, z1, mi_pipe, mi_soot, r=0.16):
    _cyl(bm, (x, y, z0), (x, y, z1 - 0.5), r, 16, mi_pipe)
    _cyl(bm, (x, y, z1 - 0.5), (x, y, z1), r * 1.05, 16, mi_soot)
    _cyl(bm, (x, y, z0 + 0.3), (x, y, z0 + 1.6), r * 1.6, 16, mi_pipe)        # silencer
    _cyl(bm, (x, y, z1 - 2.0), (x, y, z1 - 1.9), r * 1.5, 16, mi_pipe)        # bracket collar


def _lattice(bm, cx, cy, z0, H, wb, wt, bays, mat, r_leg=0.06, r_br=0.03):
    """Four-legged tapered lattice tower."""
    def hw(z):
        return (wb + (wt - wb) * (z - z0) / H) / 2
    zs = [z0 + H * k / bays for k in range(bays + 1)]
    corners = [(1, 1), (-1, 1), (-1, -1), (1, -1)]
    for i in range(bays):
        za, zb = zs[i], zs[i + 1]
        for sx, sy in corners:
            _cyl(bm, (cx + sx * hw(za), cy + sy * hw(za), za), (cx + sx * hw(zb), cy + sy * hw(zb), zb),
                 r_leg, 10, mat)
        for j in range(4):
            (ax, ay), (bx_, by_) = corners[j], corners[(j + 1) % 4]
            _cyl(bm, (cx + ax * hw(zb), cy + ay * hw(zb), zb), (cx + bx_ * hw(zb), cy + by_ * hw(zb), zb),
                 r_br, 6, mat)
            if (i + j) % 2:
                _cyl(bm, (cx + ax * hw(za), cy + ay * hw(za), za), (cx + bx_ * hw(zb), cy + by_ * hw(zb), zb),
                     r_br, 6, mat)
            else:
                _cyl(bm, (cx + bx_ * hw(za), cy + by_ * hw(za), za), (cx + ax * hw(zb), cy + ay * hw(zb), zb),
                     r_br, 6, mat)


# ---------------------------------------------------------------------------------
# parts
# ---------------------------------------------------------------------------------
def _hull(coll, mt, root):
    bm = bmesh.new()
    prof = [(XS, -DECK_T), (XS, -1.5), (XS + RAKE_STERN, -D_HULL), (XB - RAKE_BOW, -D_HULL), (XB, -1.2),
            (XB, -DECK_T)]
    U.flat_prism(bm, [prof], -YH, YH, 0, 0, 0)
    for v in bm.verts:
        v.co = Vector((v.co.x, v.co.z, v.co.y))
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
    # stern skegs
    for sy in (-9.0, 9.0):
        sub = bmesh.new()
        U.flat_prism(sub, [[(XS + 2.5, -2.4), (XS + RAKE_STERN + 4.0, -D_HULL), (XS + RAKE_STERN - 0.5, -D_HULL)]],
                     sy - 0.03, sy + 0.03, 0, 0, 0)
        for v in sub.verts:
            v.co = Vector((v.co.x, v.co.z, v.co.y))
        bmesh.ops.recalc_face_normals(sub, faces=sub.faces[:])
        U.merge_into(bm, sub, 0)
    # wings: deck extensions with a sloping underside, tapered into the bow corners
    for sgn in (1.0, -1.0):
        sub = bmesh.new()
        poly = [(XS, sgn * YH), (XS, sgn * YW), (X_CHAMFER, sgn * YW), (XB, sgn * YH)]
        U.flat_prism(sub, [poly], -3.0, -DECK_T, 0, 0, 0)
        for v in sub.verts:
            if v.co.z < -1.0:
                f = (abs(v.co.y) - YH) / (YW - YH)
                v.co.z = -WING_D_IN + (WING_D_IN - WING_D_OUT) * f
        U.merge_into(bm, sub, 0)
        # knee brackets under the wing against the hull side
        x = XS + 2.2
        while x < XB - 4.0:
            if deck_half_width(x) - YH > 2.8:
                zb = -WING_D_IN
                z_out = -WING_D_IN + (WING_D_IN - WING_D_OUT) * (2.4 / (YW - YH))
                k = bmesh.new()
                U.flat_prism(k, [[(YH, zb + 0.02), (YH, zb - 0.55), (YH + 2.4, z_out + 0.02)]],
                             x - 0.014, x + 0.014, 0, 0, 0)
                for v in k.verts:
                    v.co = Vector((v.co.z, sgn * v.co.x, v.co.y))
                bmesh.ops.recalc_face_normals(k, faces=k.faces[:])
                U.merge_into(bm, k, 0)
            x += 4.57
    # rub rail / fender strake all round the deck edge, a lower one on the hull ends
    outline = [Vector((px, py, -0.36)) for px, py in DECK_OUTLINE]
    prof_r = [(0.02, -0.12), (-0.10, -0.12), (-0.15, -0.06), (-0.15, 0.06), (-0.10, 0.12), (0.02, 0.12)]
    _sweep(bm, outline, prof_r, closed=True, mat=0)
    for xe, ys in ((XB, (-YH, YH)), (XS, (-YW, YW))):
        _sweep(bm, [(xe, ys[0], -1.0), (xe, ys[1], -1.0)],
               [(a, bb) for a, bb in ((0.02, -0.1), (-0.12, -0.1), (-0.12, 0.1), (0.02, 0.1))]
               if xe > 0 else [(-a, bb) for a, bb in ((0.02, -0.1), (-0.12, -0.1), (-0.12, 0.1), (0.02, 0.1))],
               mat=0)
    U.finish("ASDS_Hull", bm, coll, [mt["hull"]], parent=root, sharp=30.0)

    # draft marks and fender tyres
    bm = bmesh.new()
    for sy in (-1.0, 1.0):
        for xm in (XB - 0.6, XS + 0.6):
            for k in range(8):
                z = WATERLINE_Z + 0.25 * k
                if z > -1.3:
                    continue
                y0 = sy * (YH + 0.004)
                w = 0.32 if k % 2 == 0 else 0.18
                _bx(bm, xm - 0.03, xm + 0.03, y0, y0 + sy * 0.006, z, z + 0.05, 0)
                _bx(bm, xm - w if xm > 0 else xm, xm if xm > 0 else xm + w, y0, y0 + sy * 0.006, z, z + 0.05, 0)
    U.finish("ASDS_DraftMarks", bm, coll, [mt["mark"]], parent=root, smooth=False)


def _deck(coll, mt, root):
    bm = bmesh.new()
    U.flat_prism(bm, [DECK_OUTLINE], -DECK_T, 0.0, 0, 0, 0)
    U.finish("ASDS_Deck", bm, coll, [mt["deck"]], parent=root, smooth=False)


def _edges(coll, mt, root):
    """Toe bar, railings, bitts, chocks, manholes, tyres along the deck edges."""
    mats = [mt["steel"], mt["galv"], mt["rubber"], mt["metal"], mt["orange"], mt["mark"]]
    bm = bmesh.new()
    # toe bar (bull rail) round the deck edge
    path = [Vector((px, py, 0.0)) for px, py in DECK_OUTLINE]
    _sweep(bm, path, [(0.0, 0.0), (0.12, 0.0), (0.12, 0.16), (0.0, 0.16)], closed=True, mat=0)

    rail = bmesh.new()
    bitt_x = (-22.0, 2.0, 24.0)
    for sgn in (1.0, -1.0):
        y = sgn * (YW - 0.2)
        cuts = [X_WALL + 0.9]
        for bx_ in bitt_x:
            cuts += [bx_ - 1.6, bx_ + 1.6]
        cuts.append(X_CHAMFER - 0.15)
        for a, b in zip(cuts[0::2], cuts[1::2]):
            _railing(rail, (a, y), (b, y), 1)
        # chamfer edge toward the bow wall
        p0 = Vector((X_CHAMFER + 0.25, sgn * (YW - 0.25), 0.0))
        xe = X_BOW_WALL - 0.5
        p1 = Vector((xe, sgn * (deck_half_width(xe) - 0.25), 0.0))
        _railing(rail, p0, p1, 1)
        for bx_ in bitt_x:
            _bitt(bm, bx_, sgn * (YW - 1.05), 0.0, 0)
            _chock(bm, bx_, sgn * (YW - 0.06), 0.0 if sgn > 0 else math.pi, 0)
        # stern-corner bitts
        _bitt(bm, XS + 2.2, sgn * (YW - 1.05), 0.0, 0)
        # tyre fenders hung from the toe bar along the wing faces
        for xt in (-38.0, -27.5, -15.0, -4.0, 9.0, 20.0, 31.0):
            _tyre(bm, (xt, sgn * (YW + 0.2), -0.95), (0.0, 1.0, 0.0), 2)
            _cyl(bm, (xt, sgn * (YW + 0.06), 0.14), (xt, sgn * (YW + 0.2), -0.55), 0.018, 6, 3)
    # stern edge railing behind the equipment, bow edge railing beside the tow bitts
    for a, b in ((-25.5, -18.0), (-14.0, -5.6), (5.6, 14.0), (18.0, 25.5)):
        _railing(rail, (XS + 0.2, a), (XS + 0.2, b), 1)
    for a, b in ((-15.0, -5.4), (5.4, 15.0)):
        _railing(rail, (XB - 0.2, a), (XB - 0.2, b), 1)
    U.merge_into(bm, rail)
    # life-ring stations hung on the railings (orange rings with reflective bands)
    for (lx, ly) in ((-27.0, YW - 0.2), (19.0, YW - 0.2), (-8.0, -(YW - 0.2)), (31.0, -(YW - 0.2))):
        sg = 1.0 if ly > 0 else -1.0
        cx_, cy_ = lx, ly - sg * 0.06
        U.add_box(bm, (0.05, 0.05, 0.25), Matrix.Translation((cx_, cy_ - sg * 0.04, 0.92)), 1)
        U.add_torus(bm, 0.29, 0.06, 24, 8, Matrix.Translation((cx_, cy_ - sg * 0.1, 0.62)) @
                    Matrix.Rotation(math.pi / 2, 4, "X"), 4)
        for k in range(4):
            a = U.TAU * k / 4 + math.pi / 4
            U.add_box(bm, (0.07, 0.135, 0.07), Matrix.Translation((cx_ + 0.29 * math.cos(a), cy_ - sg * 0.1,
                                                                    0.62 + 0.29 * math.sin(a))) @
                      Matrix.Rotation(-a, 4, "Y"), 5)

    # manholes (raised flush covers) outside the paint
    for (mx, my) in ((-27.5, 10.5), (-27.5, -10.5), (-27.5, 21.0), (-27.5, -21.0), (27.0, 10.5), (27.0, -10.5),
                     (27.0, 21.5), (27.0, -21.5), (-1.0, 23.0), (-1.0, -23.0), (31.5, 0.0), (-25.0, 0.0),
                     (12.0, 22.5), (-14.0, -22.5)):
        _manhole(bm, mx, my, 0, 3)
    # bow: tow bitts and bridle pad-eyes
    for sy in (-1.0, 1.0):
        _bitt(bm, XB - 1.3, sy * 3.2, math.pi / 2, 0, r=0.22, h=0.75, spread=0.8)
        for yy in (sy * 6.5, sy * 13.5):
            U.add_box(bm, (0.5, 0.06, 0.34), Matrix.Translation((XB - 0.5, yy, 0.17)), 0)
            U.add_torus(bm, 0.11, 0.035, 12, 6, Matrix.Translation((XB - 0.5, yy, 0.34)) @
                        Matrix.Rotation(math.pi / 2, 4, "X"), 3)
        _tyre(bm, (XB + 0.2, sy * 9.0, -0.75), (1.0, 0.0, 0.0), 2)
        _tyre(bm, (XS - 0.2, sy * 9.0, -0.75), (1.0, 0.0, 0.0), 2)
    U.finish("ASDS_DeckFittings", bm, coll, mats, parent=root, sharp=35.0)


def _blast_wall(bm, p0, p1, front, h, wall=0, frame=1, struts=True, post_step=3.1, deflector=True):
    """Blast wall: corrugated infill between H-columns on the front face, kick plate,
    cap beam with a forward-flaring deflector lip, back posts and raking struts."""
    a, d, nrm, L = _line_frame(p0, p1, front)
    up = Vector((0.0, 0.0, 1.0))
    s = 1.0 if up.cross(d).dot(nrm) > 0 else -1.0       # profile a-axis -> front normal
    D = 0.16
    _corrugated(bm, p0, p1, 0.3, h - 0.3, front, depth=D, mat=wall)
    path = [a, a + d * L]
    _sweep(bm, path, [(s * -0.12, 0.0), (s * (D + 0.06), 0.0), (s * (D + 0.06), 0.32), (s * -0.12, 0.32)],
           mat=frame)
    _sweep(bm, path, [(s * -0.22, h - 0.32), (s * (D + 0.36), h - 0.32), (s * (D + 0.36), h), (s * -0.22, h)],
           mat=frame)
    if deflector:
        _sweep(bm, path, [(s * (D + 0.30), h - 0.03), (s * (D + 1.05), h + 0.62), (s * (D + 1.02), h + 0.66),
                          (s * (D + 0.26), h + 0.02)], mat=wall)
    n = max(1, int(round(L / post_step)))
    for k in range(n + 1):
        p = a + d * (L * k / n)
        fm = _frame(d, up, p)
        for (t0, t1, w) in ((D, D + 0.024, 0.34), (D + 0.024, D + 0.30, 0.018), (D + 0.30, D + 0.324, 0.34)):
            U.add_box(bm, (w, t1 - t0, h), fm @ Matrix.Translation((0.0, s * (t0 + t1) / 2, h / 2)), frame)
        U.add_box(bm, (0.5, 0.42, 0.025), fm @ Matrix.Translation((0.0, s * (D + 0.17), 0.0125)), frame)
        pb = p - nrm * 0.17
        _beam(bm, pb + up * 0.3, pb + up * (h - 0.3), 0.22, 0.26, frame, up=nrm)
        if struts and 0 < k < n:
            top = pb + up * (h - 1.0)
            foot = pb - nrm * 3.0
            _beam(bm, top, foot + up * 0.1, 0.16, 0.16, frame, up=d)
            U.add_box(bm, (0.5, 0.5, 0.03), Matrix.Translation(foot + up * 0.015), frame)


def _stern(coll, mt, root, lights):
    """Blast walls, equipment pods, garage, crane and antenna mast at the stern."""
    # --- blast walls -----------------------------------------------------------
    bm = bmesh.new()
    for sgn in (1.0, -1.0):
        _blast_wall(bm, (X_WALL, sgn * Y_POD), (X_WALL, sgn * (YW - 0.15)), (1.0, 0.0), WALL_H, 0, 1)
        _blast_wall(bm, (X_WALL - 0.3, sgn * Y_POD), (X_WALL - 9.3, sgn * Y_POD), (0.0, -sgn), WALL_H, 0, 1,
                    struts=False, deflector=False)
        # corner and end columns
        _bx(bm, X_WALL - 0.55, X_WALL + 0.1, sgn * (Y_POD - 0.1), sgn * (Y_POD + 0.55), 0.0, WALL_H + 0.25, 1)
        _bx(bm, X_WALL - 0.45, X_WALL + 0.1, sgn * (YW - 0.55), sgn * (YW - 0.05), 0.0, WALL_H + 0.1, 1)
        _bx(bm, X_WALL - 9.6, X_WALL - 9.15, sgn * (Y_POD - 0.1), sgn * (Y_POD + 0.4), 0.0, WALL_H + 0.1, 1)
        # low side screens along the deck edge behind the walls
        _corrugated(bm, (X_WALL - 0.4, sgn * (YW - 0.3)), (XS + 0.4, sgn * (YW - 0.3)), 0.05, 2.4, (0.0, sgn),
                    pitch=0.5, depth=0.1, mat=0)
        _beam(bm, (X_WALL - 0.4, sgn * (YW - 0.33), 2.45), (XS + 0.4, sgn * (YW - 0.33), 2.45), 0.2, 0.1, 1)
        # access ladder on the back of the forward wall
        _ladder(bm, (X_WALL - 0.75, sgn * 15.6, 0.0), WALL_H + 0.9, (0.0, 1.0, 0.0), 2, out=(-1.0, 0.0, 0.0))
    U.finish("ASDS_BlastWalls", bm, coll, [mt["wall"], mt["frame"], mt["galv"]], parent=root, sharp=35.0)

    # --- equipment in the pods --------------------------------------------------
    bm = bmesh.new()
    # material slots: 0 white, 1 grey frame, 2 dark, 3 metal, 4 louvre, 5 blue, 6 green, 7 soot, 8 galv,
    #                 9 black, 10 glass
    mats = [mt["white"], mt["grey"], mt["black"], mt["metal"], mt["louvre"], mt["blue"], mt["green"], mt["soot"],
            mt["galv"], mt["black"], mt["glass"]]
    xr1, xr2 = -38.0, -41.55
    for sgn in (1.0, -1.0):
        _container(bm, xr1, sgn * 12.3, math.pi / 2, 12.192, (0, 1, 2, 3, 4), genset=True,
                   door_end=1 if sgn > 0 else -1)
        _exhaust(bm, xr1 - 0.3, sgn * 15.6, 2.59, 9.4, 3, 7)
        _exhaust(bm, xr1 + 0.3, sgn * 9.2, 2.59, 9.0, 3, 7)
        _container(bm, xr1, sgn * 22.0, math.pi / 2, 6.058, (5 if sgn > 0 else 0, 1, 2, 3, 4), genset=sgn < 0,
                   door_end=-1)
        if sgn < 0:
            _exhaust(bm, xr1 + 0.2, sgn * 23.4, 2.59, 8.6, 3, 7)
        _container(bm, xr2, sgn * 9.7, math.pi / 2, 6.058, (0, 1, 2, 3, 4), door_end=1)
        # control container: door, window, AC units on the roof
        yc = sgn * 9.7
        _bx(bm, xr2 + 1.22, xr2 + 1.25, yc - 1.2, yc - 0.3, 0.25, 2.3, 1)
        _bx(bm, xr2 + 1.22, xr2 + 1.26, yc + 0.5, yc + 1.5, 1.2, 1.9, 10)
        for k in range(2):
            ya = yc - 1.6 + 2.6 * k
            _bx(bm, xr2 - 0.6, xr2 + 0.6, ya - 0.5, ya + 0.5, 2.59, 3.25, 1)
            U.add_cyl(bm, 0.36, 0.36, 0.03, 20, Matrix.Translation((xr2, ya, 3.26)), 9)
        # fuel tank on saddles
        yt = sgn * 19.3
        _cyl(bm, (xr2, yt - 3.6, 1.45), (xr2, yt + 3.6, 1.45), 1.15, 28, 6)
        for e in (-1.0, 1.0):
            _cyl(bm, (xr2, yt + e * 3.6, 1.45), (xr2, yt + e * 3.75, 1.45), 1.0, 28, 6, r1=1.15)
            _bx(bm, xr2 - 1.0, xr2 + 1.0, yt + e * 2.3 - 0.15, yt + e * 2.3 + 0.15, 0.0, 0.9, 1)
        _cyl(bm, (xr2, yt - 1.0, 2.5), (xr2, yt - 1.0, 2.95), 0.12, 12, 3)
        _cyl(bm, (xr2 + 0.4, yt + 1.2, 2.5), (xr2 + 0.4, yt + 1.2, 2.75), 0.07, 10, 3)
        # cable tray from the containers to the garage
        _bx(bm, X_WALL - 3.3, X_WALL - 3.0, sgn * 4.2, sgn * 21.0, 2.9, 3.0, 8)
        _bx(bm, X_WALL - 3.3, X_WALL - 3.0, sgn * 4.2, sgn * 21.0, 3.0, 3.1, 8)
        for yy in range(5, 21, 3):
            _cyl(bm, (X_WALL - 3.15, sgn * yy, 0.0), (X_WALL - 3.15, sgn * yy, 2.9), 0.04, 6, 8)
    # hydraulic power unit by the crane
    _bx(bm, -43.2, -41.0, -17.4, -15.6, 0.0, 1.6, 5)
    _bx(bm, -43.0, -41.2, -17.2, -15.8, 1.6, 1.7, 1)
    U.finish("ASDS_SternEquipment", bm, coll, mats, parent=root, sharp=35.0)

    # --- garage for the deck robot, with roll-up door ------------------------------
    bm = bmesh.new()
    gx0, gx1, gy, gh = -43.6, -35.4, 4.1, 4.6
    _bx(bm, gx0, gx1 - 0.25, -gy, gy, 0.0, gh, 0)                 # shell
    _bx(bm, gx0 - 0.15, gx1 + 0.15, -gy - 0.15, gy + 0.15, gh, gh + 0.18, 1)   # roof slab
    door_y, door_h = 3.2, 3.9
    _bx(bm, gx1 - 0.25, gx1 + 0.05, -gy, -door_y, 0.0, gh, 1)
    _bx(bm, gx1 - 0.25, gx1 + 0.05, door_y, gy, 0.0, gh, 1)
    _bx(bm, gx1 - 0.25, gx1 + 0.05, -door_y, door_y, door_h, gh, 1)
    _bx(bm, gx1 + 0.05, gx1 + 0.6, -door_y - 0.2, door_y + 0.2, door_h, door_h + 0.55, 1)  # roll housing
    # door slats: zig-zag profile in (x, z), lofted across the opening
    slats = []
    z = 0.0
    k = 0
    while z < door_h - 1e-6:
        slats.append((gx1 - 0.15 + (0.02 if k % 2 else 0.0), z))
        z += 0.05
        k += 1
    slats.append((gx1 - 0.15, door_h))
    secs = [[(xx, yy, zz) for xx, zz in slats] for yy in (-door_y, door_y)]
    U.loft(bm, secs, mat=2, cap_start=False, cap_end=False, closed_loop=False)
    # side louvres and a wall pack lamp over the door
    for sy in (-1.0, 1.0):
        _bx(bm, gx0 + 1.5, gx0 + 3.0, sy * (gy + 0.01), sy * (gy - 0.02), 2.6, 3.6, 3)
    U.finish("ASDS_Garage", bm, coll, [mt["grey"], mt["steel"], mt["white"], mt["louvre"]], parent=root,
             sharp=35.0)

    # --- pedestal crane (starboard pod) ----------------------------------------------
    _crane(coll, mt, root, (-42.7, -21.6))

    # --- antenna / radar mast behind the garage ----------------------------------------
    _antenna_mast(coll, mt, root, (-44.65, 0.0))

    # --- floodlights on the wall tops, facing the deck -------------------------------------
    for sgn in (1.0, -1.0):
        for yy in (sgn * 9.0, sgn * 20.0):
            lights.append(((X_WALL + 1.35, yy, WALL_H + 0.95), (0.0, yy * 0.3, 0.0), True))


def _crane(coll, mt, root, base):
    bx_, by_ = base
    bm = bmesh.new()
    # slots: 0 yellow, 1 metal, 2 black, 3 glass
    zt = 6.7
    U.add_cyl(bm, 0.9, 0.9, 0.06, 24, Matrix.Translation((bx_, by_, 0.03)), 0)
    for k in range(8):
        a = U.TAU * k / 8
        p = Vector((bx_ + 0.62 * math.cos(a), by_ + 0.62 * math.sin(a), 0.0))
        _beam(bm, p, p + Vector((0.28 * math.cos(a), 0.28 * math.sin(a), 0.0)), 0.03, 0.5, 0)
    _cyl(bm, (bx_, by_, 0.0), (bx_, by_, zt), 0.55, 24, 0)
    U.add_cyl(bm, 0.78, 0.78, 0.22, 28, Matrix.Translation((bx_, by_, zt + 0.11)), 1)
    yaw = math.radians(78.0)
    fw = Vector((math.cos(yaw), math.sin(yaw), 0.0))
    sd = Vector((-fw.y, fw.x, 0.0))
    c0 = Vector((bx_, by_, zt + 0.22))
    hm = _frame(fw, Z_UP, c0 - fw * 0.35 + Vector((0, 0, 0.7)))
    U.add_box(bm, (2.3, 1.6, 1.4), hm, 0)                              # slewing house
    U.add_box(bm, (0.9, 0.85, 1.25), hm @ Matrix.Translation((0.3, -1.25, 0.05)), 0)   # control cab
    U.add_box(bm, (0.6, 0.02, 0.6), hm @ Matrix.Translation((0.45, -1.68, 0.2)), 3)
    U.add_box(bm, (0.02, 0.6, 0.55), hm @ Matrix.Translation((0.76, -1.25, 0.2)), 3)
    # boom: two telescopic box sections, luffed up 32 deg
    pivot = c0 + fw * 0.55 + Vector((0, 0, 1.25))
    el = math.radians(32.0)
    bd = (fw * math.cos(el) + Vector((0, 0, math.sin(el)))).normalized()
    upb = bd.cross(sd).normalized()
    if upb.z < 0:
        upb = -upb
    L1, L2 = 7.0, 5.0
    _beam(bm, pivot - bd * 0.6, pivot + bd * L1, 0.55, 0.62, 0, up=upb)
    _beam(bm, pivot + bd * (L1 - 0.4), pivot + bd * (L1 + L2), 0.42, 0.46, 0, up=upb)
    tip = pivot + bd * (L1 + L2)
    for s in (-1.0, 1.0):                                              # pivot lugs
        _cyl(bm, pivot + sd * (s * 0.2), pivot + sd * (s * 0.42), 0.16, 14, 1)
    _cyl(bm, tip + sd * -0.25 - upb * 0.1, tip + sd * 0.25 - upb * 0.1, 0.26, 18, 1)   # sheave
    # luffing cylinder from the house front to under the boom
    c_base = c0 + fw * 0.6 + Vector((0, 0, 0.35))
    c_top = pivot + bd * 3.4 - upb * 0.31
    mid = c_base.lerp(c_top, 0.55)
    _cyl(bm, c_base, mid, 0.14, 14, 0)
    _cyl(bm, mid - (c_top - c_base).normalized() * 0.2, c_top, 0.075, 12, 1)
    # hook block on its wire
    hook = tip - upb * 0.1 + Vector((0, 0, -4.2))
    _cyl(bm, tip - upb * 0.15, hook + Vector((0, 0, 0.35)), 0.012, 6, 2)
    U.add_box(bm, (0.35, 0.22, 0.5), Matrix.Translation(hook + Vector((0, 0, 0.1))), 0)
    U.add_torus(bm, 0.09, 0.03, 12, 6, Matrix.Translation(hook + Vector((0, 0, -0.25))) @
                Matrix.Rotation(math.pi / 2, 4, "X"), 1)
    U.finish("ASDS_Crane", bm, coll, [mt["yellow"], mt["metal"], mt["black"], mt["glass"]], parent=root,
             sharp=35.0)


def _antenna_mast(coll, mt, root, base):
    cx, cy = base
    bm = bmesh.new()
    # slots: 0 grey steel, 1 radome, 2 black, 3 galv
    H = 16.5
    U.add_box(bm, (1.9, 1.9, 0.05), Matrix.Translation((cx, cy, 0.025)), 0)
    _lattice(bm, cx, cy, 0.05, H, 1.6, 0.7, 8, 3)
    zt = H + 0.05
    U.add_box(bm, (2.2, 2.2, 0.06), Matrix.Translation((cx, cy, zt)), 3)     # platform grating
    for (ax, ay), (bx2, by2) in (((-1, -1), (1, -1)), ((1, -1), (1, 1)), ((1, 1), (-1, 1)), ((-1, 1), (-1, -1))):
        p0 = Vector((cx + ax * 1.08, cy + ay * 1.08, zt))
        p1 = Vector((cx + bx2 * 1.08, cy + by2 * 1.08, zt))
        _cyl(bm, p0, p0 + Vector((0, 0, 1.0)), 0.02, 6, 3)
        _cyl(bm, p0 + Vector((0, 0, 1.0)), p1 + Vector((0, 0, 1.0)), 0.02, 6, 3)
        _cyl(bm, p0 + Vector((0, 0, 0.5)), p1 + Vector((0, 0, 0.5)), 0.016, 6, 3)
    # radar: pedestal + open-array scanner
    _cyl(bm, (cx, cy, zt), (cx, cy, zt + 1.7), 0.08, 12, 0)
    U.add_box(bm, (0.5, 0.42, 0.45), Matrix.Translation((cx, cy, zt + 1.9)), 0)
    U.add_box(bm, (0.32, 2.6, 0.26), Matrix.Translation((cx + 0.05, cy, zt + 2.28)) @
              Matrix.Rotation(math.radians(20.0), 4, "Z"), 2)
    # radomes (satcom), flat-panel terminals, whips, anemometer
    for sy in (-1.0, 1.0):
        p = Vector((cx - 0.75, cy + sy * 0.75, zt))
        _cyl(bm, p, p + Vector((0, 0, 0.6)), 0.06, 10, 0)
        U.add_cyl(bm, 0.36, 0.36, 0.25, 24, Matrix.Translation(p + Vector((0, 0, 0.72))), 1)
        sub = bmesh.new()
        U.revolve(sub, [(0.36, 0.0), (0.34, 0.12), (0.26, 0.25), (0.14, 0.33), (0.0, 0.36)], 24)
        sub.transform(Matrix.Translation(p + Vector((0, 0, 0.845))))
        U.merge_into(bm, sub, 1)
        q = Vector((cx + 0.8, cy + sy * 0.8, zt))
        _cyl(bm, q, q + Vector((0, 0, 0.5)), 0.04, 8, 0)
        U.add_box(bm, (0.57, 0.51, 0.04), Matrix.Translation(q + Vector((0, 0, 0.62))) @
                  Matrix.Rotation(math.radians(-25.0), 4, "Y"), 1)
    for (wx, wy, hh) in ((1.0, 0.0, 3.0), (-1.0, 0.25, 2.4), (0.0, -1.0, 2.0)):
        p = Vector((cx + wx * 1.0, cy + wy * 1.0, zt + 1.0))
        _cyl(bm, p, p + Vector((0, 0, hh)), 0.018, 6, 2, r1=0.03)
    p = Vector((cx, cy, zt + 2.6))
    _cyl(bm, p, p + Vector((0, 0, 0.6)), 0.03, 8, 0)
    for k in range(3):
        a = U.TAU * k / 3
        e = p + Vector((0.3 * math.cos(a), 0.3 * math.sin(a), 0.6))
        _cyl(bm, p + Vector((0, 0, 0.6)), e, 0.01, 6, 0)
        U.add_cyl(bm, 0.05, 0.05, 0.06, 8, Matrix.Translation(e), 2)
    # livestream camera housings on the mast
    for zc, yaw in ((9.0, 0.0), (12.5, 0.4)):
        m = Matrix.Translation((cx + 0.6, cy, zc)) @ Matrix.Rotation(yaw, 4, "Z")
        U.add_box(bm, (0.4, 0.18, 0.18), m @ Matrix.Translation((0.25, 0, 0)), 1)
        U.add_box(bm, (0.44, 0.24, 0.02), m @ Matrix.Translation((0.27, 0, 0.11)), 1)
        U.add_cyl(bm, 0.05, 0.05, 0.03, 12, m @ Matrix.Translation((0.46, 0, 0)) @
                  Matrix.Rotation(math.pi / 2, 4, "Y"), 2)
    U.finish("ASDS_AntennaMast", bm, coll, [mt["grey"], mt["radome"], mt["black"], mt["galv"]], parent=root,
             sharp=35.0)
    # beacon + all-round light at the top
    bm = bmesh.new()
    U.add_cyl(bm, 0.07, 0.07, 0.16, 12, Matrix.Translation((cx, cy, zt + 3.4)), 0)
    U.finish("ASDS_MastBeacon", bm, coll, [mt["nav_red"]], parent=root)


def _bow(coll, mt, root, lights):
    bm = bmesh.new()
    # slots: 0 wall, 1 steel, 2 louvre, 3 metal, 4 soot, 5 grey
    hw = deck_half_width(X_BOW_WALL) - 0.35
    fr = [(X_BOW_WALL, -hw), (X_BOW_WALL, hw)]
    bk = [(X_BOW_WALL + 0.025, -hw), (X_BOW_WALL + 0.025, hw)]
    _strip_solid(bm, fr, bk, 0.0, BOW_WALL_H, 0)
    _cyl(bm, (X_BOW_WALL + 0.01, -hw, BOW_WALL_H + 0.05), (X_BOW_WALL + 0.01, hw, BOW_WALL_H + 0.05), 0.07, 12, 1)
    n = int(2 * hw / 1.5)
    for k in range(n + 1):
        y = -hw + 2 * hw * k / n
        _bx(bm, X_BOW_WALL + 0.025, X_BOW_WALL + 0.24, y - 0.01, y + 0.01, 0.0, BOW_WALL_H, 1)
        if k % 2 == 0:
            _beam(bm, (X_BOW_WALL + 0.24, y, BOW_WALL_H - 0.1), (X_BOW_WALL + 1.3, y, 0.05), 0.08, 0.08, 1)
    # bow deckhouses (thruster engine rooms) with louvres and exhausts
    for sy in (-1.0, 1.0):
        yc = sy * 10.8
        _bx(bm, 41.6, 45.0, yc - 2.7, yc + 2.7, 0.0, 2.5, 5)
        _bx(bm, 41.5, 45.1, yc - 2.8, yc + 2.8, 2.5, 2.62, 1)
        _bx(bm, 41.58, 41.62, yc - 1.8, yc + 0.2, 0.6, 1.9, 2)
        _bx(bm, 41.58, 41.62, yc + 0.8, yc + 1.8, 0.0, 2.1, 1)            # door
        _exhaust(bm, 44.2, yc - sy * 1.6, 2.62, 4.6, 3, 4, r=0.12)
        lights.append(((41.4, yc - sy * 2.4, 3.3), (5.0, yc * 0.2, 0.0), False))
        # navigation light posts at the bow corners
        p = Vector((XB - 0.45, sy * (YH - 0.35), 0.0))
        _cyl(bm, p, p + Vector((0, 0, 1.6)), 0.05, 10, 1)
    U.finish("ASDS_Bow", bm, coll, [mt["wall"], mt["steel"], mt["louvre"], mt["metal"], mt["soot"], mt["grey"]],
             parent=root, sharp=35.0)
    for sy, key in ((1.0, "nav_red"), (-1.0, "nav_green")):
        bm = bmesh.new()
        U.add_box(bm, (0.18, 0.14, 0.2), Matrix.Translation((XB - 0.45, sy * (YH - 0.35), 1.72)), 0)
        U.finish("ASDS_NavLight_" + ("Port" if sy > 0 else "Stbd"), bm, coll, [mt[key]], parent=root,
                 smooth=False)


def _light_masts(coll, mt, root, lights):
    """Light masts along the long edges + all floodlight heads and their spot lights."""
    bm = bmesh.new()
    H = 10.5
    for sgn in (1.0, -1.0):
        for xm in (-12.5, 14.0):
            y = sgn * (YW - 0.75)
            p = Vector((xm, y, 0.0))
            U.add_box(bm, (0.7, 0.7, 0.035), Matrix.Translation(p + Vector((0, 0, 0.0175))), 0)
            for k in range(4):
                a = U.TAU * k / 4 + math.pi / 4
                dvec = Vector((math.cos(a), math.sin(a), 0.0))
                _beam(bm, p + dvec * 0.17 + Vector((0, 0, 0.2)), p + dvec * 0.33 + Vector((0, 0, 0.03)), 0.015,
                      0.3, 0)
            _cyl(bm, p, p + Vector((0, 0, H)), 0.085, 16, 0, r1=0.15)
            U.add_box(bm, (0.24, 0.3, 0.42), Matrix.Translation(p + Vector((-0.0, -sgn * 0.22, 1.4))), 0)
            # crossarm at the top, perpendicular to the aim
            tgt = Vector((xm * 0.25, sgn * 4.0, 0.0))
            aim = (tgt - p)
            aim.z = 0.0
            aim.normalize()
            side = Vector((-aim.y, aim.x, 0.0))
            top = p + Vector((0, 0, H - 0.25))
            _beam(bm, top - side * 0.95, top + side * 0.95, 0.12, 0.12, 0)
            for s in (-0.75, 0.75):
                lights.append((top + side * s + aim * 0.25 + Vector((0, 0, 0.1)), tgt + side * (s * 8.0), True))
            # small camera housing on the pole
            cm = _frame(aim, Z_UP, p + aim * 0.25 + Vector((0, 0, 4.2)))
            U.add_box(bm, (0.38, 0.17, 0.17), cm, 0)
            U.add_box(bm, (0.42, 0.22, 0.02), cm @ Matrix.Translation((0.02, 0, 0.1)), 0)
    U.finish("ASDS_LightMasts", bm, coll, [mt["galv"]], parent=root, sharp=35.0)

    body = bmesh.new()
    lens = bmesh.new()
    k = 0
    for pos, aim, has_light in lights:
        fr = _flood(body, lens, pos, aim, 0, 0)
        # bracket down to whatever it is mounted on
        _cyl(body, Vector(pos) - Vector((0, 0, 0.25)), Vector(pos) - Vector((0, 0, 0.75)), 0.04, 8, 0)
        if has_light:
            _spot(coll, root, "ASDS_FloodLight_%02d" % k, fr, 2600.0)
        k += 1
    U.finish("ASDS_FloodHeads", body, coll, [mt["grey"]], parent=root, sharp=35.0)
    U.finish("ASDS_FloodLenses", lens, coll, [mt["lens"]], parent=root, smooth=False)


# ---------------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------------
def build(coll, M=None):
    """Build the drone ship into `coll`; returns the root empty "ASDS_Root".

    M: optional material dict from f9.materials.library() (only "steel" is borrowed);
    everything else uses the ship's own procedural materials."""
    root = U.empty("ASDS_Root", coll, size=12.0, kind="ARROWS")
    root["flood"] = 1.0
    root["nav"] = 1.0
    try:
        ui = root.id_properties_ui("flood")
        ui.update(min=0.0, soft_max=3.0, description="Floodlight intensity (lamp lenses and spot lights)")
        ui = root.id_properties_ui("nav")
        ui.update(min=0.0, soft_max=3.0, description="Navigation light / beacon intensity")
    except (AttributeError, TypeError):
        pass
    root["waterline_z"] = WATERLINE_Z
    mt = _materials(M, root)
    lights = []
    _hull(coll, mt, root)
    _deck(coll, mt, root)
    _edges(coll, mt, root)
    _stern(coll, mt, root, lights)
    _bow(coll, mt, root, lights)
    _light_masts(coll, mt, root, lights)
    return root


def set_flood_intensity(root, value, nav=None):
    """Floodlight (and optionally nav-light) intensity multiplier: 0 = off, 1 = nominal.

    The lights and lens emission follow root["flood"] / root["nav"] through drivers, so
    the properties can also be keyframed directly.  Setting a custom property from
    Python does not tag the depsgraph, hence the explicit update tag."""
    root["flood"] = float(value)
    if nav is not None:
        root["nav"] = float(nav)
    root.update_tag()


def triangle_count(root):
    """Total triangles of all meshes parented (recursively) to root."""
    n = 0
    stack = list(root.children)
    while stack:
        ob = stack.pop()
        stack.extend(ob.children)
        if ob.type == "MESH":
            ob.data.calc_loop_triangles()
            n += len(ob.data.loop_triangles)
    return n
