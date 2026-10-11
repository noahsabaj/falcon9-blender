"""Starlink v2-mini-style flat-pack satellite stack, as carried inside the Falcon 9 fairing.

22 flat satellites (4.0 x 2.6 x 0.28 m each) stacked with 40 mm gaps on corner fittings
and held by four corner tension rods that run the full stack height, from a base frame
+ plate on an adapter ring clamped to the PAF up to a machined top retention plate.

Each satellite (one mesh, 22 linked duplicates): phased-array antenna panels and
laser-link terminals on the underside; a bus whose pillowed sides carry crinkled MLI
(aluminised on the long faces with gold Kapton patches round the thrusters, black
Kapton on the ends) between dark edge rails; two folded solar-array packs stowed flat on
top (dark blue cells, silver fingers/busbars, cover-glass sheen; layered panel edges with
close-out strips) with hold-down brackets and pin-pullers; an argon Hall thruster on each
long side, two star trackers, OSR mirror-tile radiators and connector / fill-valve panels
on the ends; four corner fittings with stacking bosses (the rods sit in their notches).

Frame: 1 BU = 1 m, stack axis on Z, satellite long axis on Y (the long faces look at the
fairing halves, which separate along X).  The root empty sits at the PAF top: world
z = D.Z_FAIRING_BASE + 0.82 when parent is None, otherwise the same position expressed in
the parent's frame (e.g. the vehicle's "S2" empty).

    import payload
    root = payload.build(coll, M=None, parent=None)     # -> "SL_StackRoot"
    payload.clearance_report(root)                       # min radial clearance to the fairing
    payload.triangle_count(root)
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

from f9 import dims as D  # noqa: E402
from f9 import fairing as FR  # noqa: E402
from f9 import materials as MT  # noqa: E402
from f9 import util as U  # noqa: E402

# ---------------------------------------------------------------------------------
# dimensions (metres)
# ---------------------------------------------------------------------------------
N_SATS = 22
SAT_W, SAT_L, SAT_H = 2.6, 4.0, 0.28       # X, Y, Z
GAP = 0.04
PITCH = SAT_H + GAP
PAF_TOP = 0.82                             # above the fairing base plane
ADAPTER_H = 0.27                           # adapter ring on the PAF
FRAME_H = 0.12                             # base / top frame beam depth
Z_FIRST = ADAPTER_H + FRAME_H + 0.03       # bottom of satellite 1 above the PAF top
HX, HY = SAT_W / 2 - 0.02, SAT_L / 2 - 0.02    # bus half extents
NX, NY = 1.03, 1.73                        # corner notches start here
ROD_X, ROD_Y = 1.235, 1.935
ROD_R = 0.032
FIT_C = (1.16, 1.86)                       # corner fitting centre (+,+ quadrant)
Z_ANT, Z_BUS0, Z_BUS1, Z_ARR1 = 0.024, 0.024, 0.186, 0.274


def stack_height(n_sats=N_SATS):
    """Height of the highest point (tension-rod ends) above the PAF top."""
    return Z_FIRST + (n_sats - 1) * PITCH + SAT_H + 0.03 + FRAME_H + 0.16


# ---------------------------------------------------------------------------------
# shader-graph builder (compact)
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

    def math(self, op, a, b=None, clamp=False):
        nd = self.node("ShaderNodeMath", operation=op, use_clamp=clamp)
        self.feed(nd.inputs[0], a)
        if b is not None:
            self.feed(nd.inputs[1], b)
        return nd.outputs[0]

    def vec(self, op, a, b=None):
        nd = self.node("ShaderNodeVectorMath", operation=op)
        self.feed(nd.inputs[0], a)
        if b is not None:
            self.feed(nd.inputs[1], b)
        return nd.outputs["Value"] if op in ("LENGTH", "DOT_PRODUCT", "DISTANCE") else nd.outputs["Vector"]

    def coords(self, kind="Object", random_offset=0.0):
        co = self.node("ShaderNodeTexCoord").outputs[kind]
        if random_offset:
            rnd = self.node("ShaderNodeObjectInfo").outputs["Random"]
            off = self.node("ShaderNodeCombineXYZ")
            for i, k in enumerate((1.0, 1.7, 2.3)):
                self.feed(off.inputs[i], self.math("MULTIPLY", rnd, random_offset * k))
            co = self.vec("ADD", co, off.outputs[0])
        return co

    def sep(self, v):
        nd = self.node("ShaderNodeSeparateXYZ")
        self.feed(nd.inputs[0], v)
        return nd.outputs[0], nd.outputs[1], nd.outputs[2]

    def noise(self, v, scale, detail=4.0, rough=0.5, distortion=0.0, out=0):
        nd = self.node("ShaderNodeTexNoise")
        self.feed(nd.inputs["Vector"], v)
        nd.inputs["Scale"].default_value = scale
        nd.inputs["Detail"].default_value = detail
        nd.inputs["Roughness"].default_value = rough
        nd.inputs["Distortion"].default_value = distortion
        return nd.outputs[out]

    def voronoi(self, v, scale, feature="F1", out="Distance"):
        nd = self.node("ShaderNodeTexVoronoi", feature=feature)
        self.feed(nd.inputs["Vector"], v)
        nd.inputs["Scale"].default_value = scale
        return nd.outputs[out]

    def maprange(self, v, a, b, c=0.0, d=1.0):
        nd = self.node("ShaderNodeMapRange", clamp=True)
        for i, x in enumerate((v, a, b, c, d)):
            self.feed(nd.inputs[i], x)
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

    def bump(self, height, strength=1.0, distance=1.0, normal=None):
        nd = self.node("ShaderNodeBump")
        nd.inputs["Strength"].default_value = strength
        nd.inputs["Distance"].default_value = distance
        self.feed(nd.inputs["Height"], height)
        if normal is not None:
            self.feed(nd.inputs["Normal"], normal)
        return nd.outputs["Normal"]

    def brick(self, v, w, h, mortar, smooth=0.0, offset=0.0):
        nd = self.node("ShaderNodeTexBrick", offset=offset, offset_frequency=2, squash=1.0, squash_frequency=2)
        self.feed(nd.inputs["Vector"], v)
        nd.inputs["Color1"].default_value = (0.0, 0.0, 0.0, 1.0)
        nd.inputs["Color2"].default_value = (1.0, 1.0, 1.0, 1.0)
        nd.inputs["Mortar"].default_value = (0.5, 0.5, 0.5, 1.0)
        nd.inputs["Scale"].default_value = 1.0
        nd.inputs["Mortar Size"].default_value = mortar
        nd.inputs["Mortar Smooth"].default_value = smooth
        nd.inputs["Bias"].default_value = 0.0
        nd.inputs["Brick Width"].default_value = w
        nd.inputs["Row Height"].default_value = h
        sc = self.node("ShaderNodeSeparateColor")
        self.feed(sc.inputs[0], nd.outputs["Color"])
        return sc.outputs[0], nd.outputs["Fac"]


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


# ---------------------------------------------------------------------------------
# materials
# ---------------------------------------------------------------------------------
def _band(g, v, lo, hi, soft=0.003):
    """1 inside [lo, hi] with soft edges."""
    return g.math("MULTIPLY", g.maprange(v, lo - soft, lo), g.maprange(v, hi + soft, hi))


def _facet_normal(g, co, layers, base=None):
    """Crumpled-foil normal: a random tilt vector per Voronoi cell (flat facets with sharp
    creases), summed over (scale, strength) layers.  Returns (tilt, base normal)."""
    n = base if base is not None else g.node("ShaderNodeNewGeometry").outputs["Normal"]
    acc = None
    for sc, k in layers:
        vc = g.voronoi(co, sc, out="Color")
        p = g.vec("MULTIPLY", g.vec("SUBTRACT", vc, (0.5, 0.5, 0.5)), (k, k, k))
        acc = p if acc is None else g.vec("ADD", acc, p)
    return acc, n


def _mat_mli(name="SL_MLI"):
    """Bus thermal blankets.  Long faces: crinkled aluminised MLI with gold (aluminised
    Kapton) panels around the thrusters and star trackers; end faces: black Kapton.
    Vertical aluminised-tape seams, tape along the blanket edges, crevice darkening."""
    m = bpy.data.materials.get(name)
    if m is not None:
        return m
    m, g, b = _new_mat(name)
    co = g.coords()
    cr = g.coords(random_offset=40.0)
    x, y, z = g.sep(co)
    nx, ny, _nz = g.sep(g.node("ShaderNodeTexCoord").outputs["Normal"])
    end = g.maprange(g.math("ABSOLUTE", ny), 0.55, 0.75)
    px = g.maprange(nx, 0.5, 0.7)
    mx = g.maprange(nx, -0.5, -0.7)
    gold = g.math("ADD", g.math("MULTIPLY", px, _band(g, y, -1.47, -1.03)),
                  g.math("MULTIPLY", mx, _band(g, y, 1.03, 1.47)))
    gold = g.math("MINIMUM", gold, 1.0)
    rnd = g.node("ShaderNodeObjectInfo").outputs["Random"]
    u = g.math("DIVIDE", g.math("ADD", g.math("ADD", x, y), g.math("MULTIPLY", rnd, 3.1)), 0.69)
    dseam = g.math("MULTIPLY", g.math("ABSOLUTE", g.math("SUBTRACT", g.math("FRACT", u), 0.5)), 0.69)
    tape = g.maprange(dseam, 0.0105, 0.0085)
    tape = g.math("MAXIMUM", tape, g.maprange(g.math("ABSOLUTE", g.math("SUBTRACT", z, 0.105)), 0.0565, 0.0585))
    # colours
    tone = g.maprange(g.noise(cr, 1.4, detail=3.0), 0.3, 0.7)
    tone = g.math("MULTIPLY", tone, g.maprange(rnd, 0.0, 1.0, 0.55, 1.0))
    blanket = g.node("ShaderNodeTexWhiteNoise", noise_dimensions="1D")      # one tone per blanket panel
    g.feed(blanket.inputs["W"], g.math("ADD", g.math("FLOOR", u), g.math("MULTIPLY", rnd, 53.0)))
    tone = g.math("MULTIPLY", tone, g.maprange(blanket.outputs["Value"], 0.0, 1.0, 0.6, 1.1))
    silver = g.mix(tone, (0.58, 0.58, 0.59, 1.0), (0.80, 0.79, 0.77, 1.0))
    goldc = g.mix(tone, _lin("#8C6630"), _lin("#B58A4E"))
    col = g.mix(gold, silver, goldc)
    col = g.mix(g.math("MULTIPLY", tape, 0.7), col, (0.86, 0.86, 0.88, 1.0))
    col = g.mix(end, col, _lin("#0C0C0D"))
    ao = g.node("ShaderNodeAmbientOcclusion", only_local=True, samples=8)
    ao.inputs["Distance"].default_value = 0.035
    col = g.mix(1.0, col, g.maprange(ao.outputs["AO"], 0.2, 1.0, 0.35, 1.0), blend="MULTIPLY")
    g.feed(b.inputs["Base Color"], col)
    g.feed(b.inputs["Metallic"], g.math("SUBTRACT", 1.0, end))
    rough = g.maprange(g.noise(cr, 6.0, detail=4.0), 0.3, 0.7, 0.14, 0.32)
    rough = g.mixf(end, rough, 0.3)
    rough = g.mixf(tape, rough, 0.07)
    g.feed(b.inputs["Roughness"], rough)
    # crinkles: random facet tilts at two scales (weaker on tape), soft folds via bump
    wob = g.vec("ADD", cr, g.vec("MULTIPLY", g.noise(cr, 12.0, out=1), (0.012, 0.012, 0.012)))
    pert, n0 = _facet_normal(g, wob, ((38.0, 0.055), (110.0, 0.035)))
    k = g.math("SUBTRACT", 1.0, g.math("MULTIPLY", tape, 0.85))
    nd = g.node("ShaderNodeVectorMath", operation="SCALE")
    g.feed(nd.inputs[0], pert)
    g.feed(nd.inputs["Scale"], k)
    nrm = g.vec("NORMALIZE", g.vec("ADD", n0, nd.outputs["Vector"]))
    folds = g.vec("MULTIPLY", cr, (2.5, 2.5, 7.0))
    h = g.math("MULTIPLY", g.noise(folds, 1.0, detail=3.0, rough=0.55), 0.0035)
    for sc, amp in ((9.0, 0.0011), (31.0, 0.0004)):
        rn = g.noise(g.vec("MULTIPLY", cr, (1.0, 1.0, 1.6)), sc, detail=2.0, rough=0.5, distortion=0.4)
        ridge = g.math("SUBTRACT", 1.0, g.math("ABSOLUTE", g.math("SUBTRACT", g.math("MULTIPLY", rn, 2.0), 1.0)))
        h = g.math("ADD", h, g.math("MULTIPLY", g.math("POWER", ridge, 3.0), amp))
    g.feed(b.inputs["Normal"], g.bump(h, 1.0, 1.0, normal=nrm))
    m.diffuse_color = (0.6, 0.58, 0.55, 1.0)
    return m


def _mat_solar(name="SL_Solar"):
    """Solar cells: dark blue, per-cell tone, silver fingers and busbars, light gaps,
    cover glass (coat) with an AR-coating sheen (thin film)."""
    m = bpy.data.materials.get(name)
    if m is not None:
        return m
    m, g, b = _new_mat(name)
    co = g.coords()
    x, y, _z = g.sep(co)
    cell_rnd, gap = g.brick(co, 0.124, 0.124, 0.0035, 0.1, offset=0.0)
    col = g.mix(g.maprange(cell_rnd, 0.0, 1.0), _lin("#0A1230"), _lin("#13204A"))
    fy = g.math("FRACT", g.math("MULTIPLY", y, 3.0 / 0.124))
    bus = g.maprange(g.math("ABSOLUTE", g.math("SUBTRACT", fy, 0.5)), 0.03, 0.018)
    fx = g.math("FRACT", g.math("MULTIPLY", x, 1.0 / 0.0062))
    fing = g.maprange(g.math("ABSOLUTE", g.math("SUBTRACT", fx, 0.5)), 0.12, 0.06)
    col = g.mix(g.math("MULTIPLY", fing, 0.25), col, _lin("#4A5263"))
    col = g.mix(bus, col, _lin("#A9ADB3"))
    col = g.mix(gap, col, _lin("#B9BBBC"))
    g.feed(b.inputs["Base Color"], col)
    g.feed(b.inputs["Metallic"], g.math("MAXIMUM", g.math("MULTIPLY", bus, 0.9), g.math("MULTIPLY", gap, 0.3)))
    b.inputs["Roughness"].default_value = 0.32
    b.inputs["Coat Weight"].default_value = 1.0
    b.inputs["Coat Roughness"].default_value = 0.035
    b.inputs["Coat IOR"].default_value = 1.5
    b.inputs["Thin Film Thickness"].default_value = 260.0
    b.inputs["Thin Film IOR"].default_value = 1.9
    m.diffuse_color = _lin("#0E1838")
    return m


def _mat_array_edge(name="SL_ArrayEdge"):
    """Edges of the folded array stack: composite panel edges, thin dark gaps (object z),
    a slightly different tone per panel."""
    m = bpy.data.materials.get(name)
    if m is not None:
        return m
    m, g, b = _new_mat(name)
    co = g.coords()
    _x, _y, z = g.sep(co)
    zz = g.math("MULTIPLY", g.math("SUBTRACT", z, Z_BUS1), 1.0 / 0.0146)
    f = g.math("FRACT", zz)
    gap = g.maprange(f, 0.82, 0.88)
    strip = g.math("MULTIPLY", g.maprange(f, 0.66, 0.69), g.maprange(f, 0.78, 0.75))
    wn = g.node("ShaderNodeTexWhiteNoise", noise_dimensions="1D")
    g.feed(wn.inputs["W"], g.math("ADD", g.math("FLOOR", zz),
                                  g.math("MULTIPLY", g.node("ShaderNodeObjectInfo").outputs["Random"], 97.0)))
    lay = wn.outputs["Value"]
    col = g.mix(lay, _lin("#15171A"), _lin("#2B2E33"))
    col = g.mix(1.0, col, g.maprange(g.noise(co, 25.0, detail=3.0), 0.3, 0.7, 0.85, 1.12), blend="MULTIPLY")
    col = g.mix(strip, col, _lin("#55585A"))
    col = g.mix(gap, col, (0.003, 0.003, 0.003, 1.0))
    g.feed(b.inputs["Base Color"], col)
    rgh = g.mixf(strip, g.maprange(lay, 0.0, 1.0, 0.34, 0.5), 0.55)
    g.feed(b.inputs["Roughness"], g.mixf(gap, rgh, 0.85))
    g.feed(b.inputs["Specular IOR Level"], g.mixf(gap, 0.5, 0.05))
    g.feed(b.inputs["Normal"], g.bump(g.math("MULTIPLY", gap, -0.0012), 1.0, 1.0))
    m.diffuse_color = _lin("#202226")
    return m


def _mat_osr(name="SL_OSR"):
    """Radiator of optical solar reflector tiles (second-surface mirrors) on the end faces."""
    m = bpy.data.materials.get(name)
    if m is not None:
        return m
    m, g, b = _new_mat(name)
    co = g.coords()
    x, y, z = g.sep(co)
    uv = g.node("ShaderNodeCombineXYZ")
    g.feed(uv.inputs[0], g.math("ADD", x, y))
    g.feed(uv.inputs[1], z)
    tile_rnd, grout = g.brick(uv.outputs[0], 0.0405, 0.0405, 0.0007, 0.2, offset=0.0)
    wn = g.node("ShaderNodeTexWhiteNoise", noise_dimensions="1D")
    g.feed(wn.inputs["W"], g.math("MULTIPLY", tile_rnd, 311.0))
    tilt = g.vec("MULTIPLY", g.vec("SUBTRACT", wn.outputs["Color"], (0.5, 0.5, 0.5)), (0.025, 0.025, 0.025))
    n0 = g.node("ShaderNodeNewGeometry").outputs["Normal"]
    g.feed(b.inputs["Normal"], g.vec("NORMALIZE", g.vec("ADD", n0, tilt)))
    g.feed(b.inputs["Base Color"], g.mix(grout, (0.90, 0.91, 0.93, 1.0), (0.008, 0.008, 0.008, 1.0)))
    g.feed(b.inputs["Metallic"], g.math("SUBTRACT", 1.0, grout))
    g.feed(b.inputs["Roughness"], g.mixf(grout, g.maprange(tile_rnd, 0.0, 1.0, 0.015, 0.05), 0.7))
    g.feed(b.inputs["Specular IOR Level"], g.mixf(grout, 0.5, 0.0))
    m.diffuse_color = (0.7, 0.7, 0.72, 1.0)
    return m


def _mat_antenna(name="SL_Antenna"):
    """Phased-array radome: light grey with a fine element grid."""
    m = bpy.data.materials.get(name)
    if m is not None:
        return m
    m, g, b = _new_mat(name)
    co = g.coords()
    _r, gr = g.brick(co, 0.055, 0.055, 0.004, 0.6, offset=0.0)
    col = g.mix(g.math("MULTIPLY", gr, 0.5), _lin("#9A9C9C"), _lin("#7C7E7F"))
    g.feed(b.inputs["Base Color"], col)
    b.inputs["Roughness"].default_value = 0.62
    g.feed(b.inputs["Normal"], g.bump(g.math("MULTIPLY", gr, -0.0004), 1.0, 1.0))
    m.diffuse_color = _lin("#9A9C9C")
    return m


def _mat_simple(name, hexcol, rough=0.5, metallic=0.0, aniso=0.0, coat=0.0, rough_var=0.0, scale=20.0):
    m = bpy.data.materials.get(name)
    if m is not None:
        return m
    m, g, b = _new_mat(name)
    b.inputs["Base Color"].default_value = MT.srgb(hexcol)
    b.inputs["Metallic"].default_value = metallic
    if rough_var > 0:
        g.feed(b.inputs["Roughness"], g.maprange(g.noise(g.coords(), scale, detail=4.0), 0.3, 0.7,
                                                 rough - rough_var, rough + rough_var))
    else:
        b.inputs["Roughness"].default_value = rough
    if aniso > 0:
        b.inputs["Anisotropic"].default_value = aniso
    if coat > 0:
        b.inputs["Coat Weight"].default_value = coat
        b.inputs["Coat Roughness"].default_value = 0.05
    m.diffuse_color = MT.srgb(hexcol)
    return m


def _materials(M):
    M = M or {}
    mt = {}
    mt["mli"] = _mat_mli()
    mt["solar"] = _mat_solar()
    mt["edge"] = _mat_array_edge()
    mt["antenna"] = _mat_antenna()
    mt["bottom"] = _mat_simple("SL_BlackKapton", "#121212", rough=0.34, rough_var=0.1)
    mt["rail"] = _mat_simple("SL_PanelEdge", "#232527", rough=0.45, rough_var=0.08)
    mt["fit"] = _mat_simple("SL_FittingGrey", "#5A5D61", rough=0.42, metallic=0.9, rough_var=0.07, scale=12.0)
    mt["alu"] = _mat_simple("SL_Aluminium", "#B4B7BA", rough=0.3, metallic=1.0, rough_var=0.03, scale=3.0)
    mt["rod"] = _mat_simple("SL_TensionRod", "#2B2C2E", rough=0.3, metallic=1.0)
    mt["black"] = _mat_simple("SL_BlackAnodised", "#101011", rough=0.45, metallic=0.4)
    mt["ceramic"] = _mat_simple("SL_BoronNitride", "#D9D4C9", rough=0.85)
    mt["steel"] = M.get("steel") or _mat_simple("SL_Steel", "#9C9EA0", rough=0.25, metallic=1.0)
    mt["glass"] = _mat_simple("SL_OpticGlass", "#0A1018", rough=0.03, coat=1.0)
    mt["gold"] = _mat_simple("SL_GoldFoil", "#C9973F", rough=0.25, metallic=1.0, rough_var=0.08, scale=30.0)
    mt["osr"] = _mat_osr()
    return mt


# slot indices for the satellite mesh
S_MLI, S_SOLAR, S_EDGE, S_ANT, S_BOTTOM, S_RAIL, S_FIT, S_BLACK, S_CERAMIC, S_STEEL, S_GLASS, S_GOLD, S_OSR = \
    range(13)
SAT_SLOTS = ("mli", "solar", "edge", "antenna", "bottom", "rail", "fit", "black", "ceramic", "steel", "glass",
             "gold", "osr")


# ---------------------------------------------------------------------------------
# geometry helpers
# ---------------------------------------------------------------------------------
def _fillet(poly, r, n=4):
    """Round every corner of a closed 2D polygon with radius r (n segments per arc)."""
    out = []
    m = len(poly)
    for i in range(m):
        p = Vector(poly[i])
        a = Vector(poly[i - 1])
        b = Vector(poly[(i + 1) % m])
        u = (a - p).normalized()
        v = (b - p).normalized()
        cosang = max(-1.0, min(1.0, u.dot(v)))
        half = math.acos(cosang) / 2.0
        if half < 1e-3 or abs(half - math.pi / 2) < 1e-3:
            out.append(tuple(p))
            continue
        t = r / math.tan(half)
        t1 = p + u * t
        t2 = p + v * t
        bis = (u + v).normalized()
        c = p + bis * (r / math.sin(half))
        a1 = math.atan2(t1.y - c.y, t1.x - c.x)
        a2 = math.atan2(t2.y - c.y, t2.x - c.x)
        da = (a2 - a1 + math.pi) % (2 * math.pi) - math.pi
        for k in range(n + 1):
            ang = a1 + da * k / n
            out.append((c.x + r * math.cos(ang), c.y + r * math.sin(ang)))
    return out


def notched_outline(hx, hy, nx, ny):
    """Rectangle |x|<=hx, |y|<=hy with square corner notches beyond (nx, ny); CCW."""
    return [(nx, hy), (-nx, hy), (-nx, ny), (-hx, ny), (-hx, -ny), (-nx, -ny), (-nx, -hy), (nx, -hy),
            (nx, -ny), (hx, -ny), (hx, ny), (nx, ny)]


def _box(bm, c, size, mat=0):
    return U.add_box(bm, size, Matrix.Translation(c), mat)


def _bx(bm, x0, x1, y0, y1, z0, z1, mat=0):
    return U.add_box(bm, (abs(x1 - x0), abs(y1 - y0), abs(z1 - z0)),
                     Matrix.Translation(((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2)), mat)


def _cyl(bm, p0, p1, r, segs=16, mat=0, caps=True, r1=None):
    p0, p1 = Vector(p0), Vector(p1)
    d = p1 - p0
    L = d.length
    if L < 1e-7:
        return []
    dn = d / L
    q = dn.to_track_quat("Z", "X" if abs(dn.y) > 0.95 else "Y")
    m = Matrix.Translation((p0 + p1) / 2) @ q.to_matrix().to_4x4()
    return U.add_cyl(bm, r if r1 is None else r1, r, L, segs, m, mat=mat, caps=caps)


def _ring(bm, center, axis, r_in, r_out, t0, t1, segs=24, mat=0):
    """Annulus (flat washer) between t0 and t1 along `axis` from `center`."""
    sub = bmesh.new()
    U.flat_prism(sub, [U.ellipse(0, 0, r_out, r_out, segs), U.ellipse(0, 0, r_in, r_in, segs)], t0, t1,
                 mat, mat, mat)
    ax = Vector(axis).normalized()
    q = ax.to_track_quat("Z", "X" if abs(ax.y) > 0.95 else "Y")
    sub.transform(Matrix.Translation(center) @ q.to_matrix().to_4x4())
    U.merge_into(bm, sub)


def _pillow_prism(bm, loop, z0, z1, bulge, mat_side, mat_bot, mat_top, rows=(0.0, 0.12, 0.5, 0.88, 1.0)):
    """Prism from a CCW loop whose side walls bulge outward by `bulge` at mid height."""
    n = len(loop)
    # outward normals at the loop vertices (CCW -> outward is to the right of travel)
    nrm = []
    for i in range(n):
        a, p, b = Vector(loop[i - 1]), Vector(loop[i]), Vector(loop[(i + 1) % n])
        t = ((p - a).normalized() + (b - p).normalized())
        if t.length < 1e-6:
            t = (b - p).normalized()
        t.normalize()
        nrm.append(Vector((t.y, -t.x)))
    secs = []
    for f in rows:
        z = z0 + (z1 - z0) * f
        o = bulge * math.sin(math.pi * f)
        secs.append([(p[0] + nv.x * o, p[1] + nv.y * o, z) for p, nv in zip(loop, nrm)])
    rows_v, faces = U.loft(bm, secs, mat=mat_side, cap_start=False, cap_end=False, recalc=False)
    bot = U.flat_fill(bm, [loop], w=z0, mat=mat_bot)
    for f in bot:
        f.normal_flip()
    U.flat_fill(bm, [loop], w=z1, mat=mat_top)
    return faces


# ---------------------------------------------------------------------------------
# satellite
# ---------------------------------------------------------------------------------
def _hall_thruster(bm, x_face, y, z, out):
    """Argon Hall thruster on a long side; out = +1 / -1 (face normal along X)."""
    ax = Vector((out, 0.0, 0.0))
    c = Vector((x_face, y, z))
    _cyl(bm, c - ax * 0.004, c + ax * 0.008, 0.07, 28, S_BLACK)                  # mounting flange
    _cyl(bm, c + ax * 0.008, c + ax * 0.040, 0.058, 28, S_GOLD)                  # body (foil wrapped)
    _ring(bm, c + ax * 0.040, ax, 0.044, 0.058, 0.0, 0.006, 28, S_FIT)          # outer pole
    _ring(bm, c + ax * 0.034, ax, 0.029, 0.044, 0.0, 0.009, 28, S_CERAMIC)      # discharge channel
    _cyl(bm, c + ax * 0.034, c + ax * 0.046, 0.029, 24, S_FIT)                   # inner pole
    _cyl(bm, c + ax * 0.046, c + ax * 0.048, 0.010, 12, S_BLACK)                 # centre cap
    # hollow cathode beside the channel
    cc = c + Vector((0.0, 0.085 * (1 if y < 0 else -1), 0.022))
    _cyl(bm, cc - ax * 0.002, cc + ax * 0.044, 0.011, 12, S_STEEL)
    _cyl(bm, cc + ax * 0.044, cc + ax * 0.051, 0.008, 12, S_BLACK)
    _bx(bm, *sorted((x_face, x_face + out * 0.018)), cc.y - 0.018, cc.y + 0.018, cc.z - 0.03, cc.z - 0.01, S_BLACK)


def _star_tracker(bm, x_face, y, z, tilt_y):
    d = Vector((-0.78, tilt_y, 0.5)).normalized()
    base = Vector((x_face + 0.01, y, z))
    _bx(bm, x_face - 0.03, x_face + 0.005, y - 0.05, y + 0.05, z - 0.045, z + 0.035, S_BLACK)   # bracket
    _cyl(bm, base, base + d * 0.05, 0.03, 18, S_BLACK)                           # camera head
    _cyl(bm, base + d * 0.05, base + d * 0.095, 0.036, 18, S_BLACK, r1=0.028)    # baffle (flared)
    _cyl(bm, base + d * 0.046, base + d * 0.05, 0.022, 18, S_GLASS)              # lens


def _holddown(bm, x, y, out):
    """Low hold-down bracket over the array edge (long sides), pin-puller and bolt."""
    x0, x1 = sorted((x, x + out * 0.022))
    _bx(bm, x0, x1, y - 0.035, y + 0.035, 0.168, Z_ARR1 + 0.004, S_BLACK)
    xo = x1 if out > 0 else x0
    _cyl(bm, (xo, y, 0.214), (xo + out * 0.016, y, 0.214), 0.015, 12, S_FIT)
    _cyl(bm, (xo + out * 0.016, y, 0.214), (xo + out * 0.021, y, 0.214), 0.009, 10, S_STEEL)
    _cyl(bm, ((x0 + x1) / 2, y, Z_ARR1 + 0.004), ((x0 + x1) / 2, y, Z_ARR1 + 0.011), 0.011, 6, S_STEEL)


def _fitting_loop(sx, sy):
    """L-shaped corner fitting outline for quadrant (sx, sy); the rod sits in the notch."""
    pts = [(1.05, 1.75), (1.27, 1.75), (1.27, 1.885), (1.19, 1.885), (1.19, 1.97), (1.05, 1.97)]
    loop = [(sx * x, sy * y) for x, y in pts]
    if sx * sy < 0:
        loop.reverse()
    return loop


def build_satellite_mesh(name="SL_Satellite"):
    """One satellite in its own frame (origin at the bottom centre); returns the bmesh."""
    bm = bmesh.new()
    outline = _fillet(notched_outline(HX, HY, NX, NY), 0.03, 3)
    # bus (pillowed MLI sides) and its edge rails
    _pillow_prism(bm, outline, Z_BUS0 + 0.012, Z_BUS1 - 0.012, 0.006, S_MLI, S_BOTTOM, S_RAIL)
    rail = _fillet(notched_outline(HX + 0.006, HY + 0.006, NX - 0.006, NY - 0.006), 0.03, 3)
    U.flat_prism(bm, [rail], Z_BUS0, Z_BUS0 + 0.012, S_RAIL, S_BOTTOM, S_RAIL)
    U.flat_prism(bm, [rail], Z_BUS1 - 0.012, Z_BUS1, S_RAIL, S_RAIL, S_RAIL)
    # stowed solar arrays: two packs, cells up
    for sy in (1.0, -1.0):
        hx, hy, y0 = HX + 0.02, HY + 0.01, 0.022
        loop = [(hx, y0), (hx, NY), (NX, NY), (NX, hy), (-NX, hy), (-NX, NY), (-hx, NY), (-hx, y0)]
        loop = _fillet(loop, 0.012, 2)
        loop = [(x, sy * y) for x, y in loop]
        if sy < 0:
            loop.reverse()
        U.flat_prism(bm, [loop], Z_BUS1, Z_ARR1, S_SOLAR, S_EDGE, S_EDGE)
        # hinge / deployment hardware at the root (outer end) of each pack
        for xh in (-0.75, 0.0, 0.75):
            _bx(bm, xh - 0.06, xh + 0.06, sy * (hy - 0.005), sy * (hy + 0.022), 0.172, Z_ARR1 - 0.012, S_BLACK)
            _cyl(bm, (xh - 0.075, sy * (hy + 0.012), 0.2), (xh + 0.075, sy * (hy + 0.012), 0.2), 0.01, 10, S_FIT)
        # hold-down brackets on the long sides
        for yy in (0.48, 1.42):
            for sx in (1.0, -1.0):
                _holddown(bm, sx * (HX + 0.02), sy * yy, sx)
    _cyl(bm, (-1.0, 0.0, Z_ARR1 - 0.02), (1.0, 0.0, Z_ARR1 - 0.02), 0.01, 8, S_BLACK)      # centre tie
    for xh in (-0.9, 0.0, 0.9):
        _bx(bm, xh - 0.04, xh + 0.04, -0.022, 0.022, Z_BUS1, Z_ARR1 + 0.004, S_BLACK)
    # underside: phased-array antennas, laser-link terminals
    for sx in (1.0, -1.0):
        for sy in (1.0, -1.0):
            _bx(bm, sx * 0.07, sx * 1.13, sy * 0.09, sy * 0.95, 0.0, Z_ANT, S_ANT)
    for sy in (1.0, -1.0):
        _bx(bm, -0.39, 0.39, sy * 1.14, sy * 1.58, 0.003, Z_ANT, S_ANT)
        for sx in (1.0, -1.0):
            c = Vector((sx * 0.92, sy * 1.42, Z_ANT))
            _cyl(bm, c, c - Vector((0, 0, 0.016)), 0.075, 20, S_BLACK)
            _cyl(bm, c - Vector((0, 0, 0.016)), c - Vector((0, 0, 0.02)), 0.05, 20, S_GLASS)
    # corner fittings with stacking bosses (the boss spans the gap to the next satellite)
    for sx in (1.0, -1.0):
        for sy in (1.0, -1.0):
            U.flat_prism(bm, [_fitting_loop(sx, sy)], 0.0, SAT_H + 0.004, S_FIT, S_FIT, S_FIT)
            c = Vector((sx * FIT_C[0], sy * FIT_C[1], SAT_H + 0.004))
            _cyl(bm, c, c + Vector((0, 0, GAP - 0.004)), 0.045, 16, S_FIT)
            _cyl(bm, c + Vector((0, 0, GAP - 0.006)), c + Vector((0, 0, GAP - 0.004)), 0.05, 16, S_BLACK)
    # Hall thrusters and star trackers on the long sides
    _hall_thruster(bm, HX + 0.006, -1.25, 0.105, 1.0)
    _hall_thruster(bm, -(HX + 0.006), 1.25, 0.105, -1.0)
    _star_tracker(bm, -(HX + 0.006), -0.52, 0.11, 0.25)
    _star_tracker(bm, -(HX + 0.006), -0.28, 0.11, -0.25)
    # ends: OSR radiator panels, connector panel (+Y), fill / drain valves (-Y)
    _bx(bm, -0.95, 0.55, HY + 0.004, HY + 0.0095, 0.05, 0.158, S_OSR)
    _bx(bm, -0.55, 0.95, -(HY + 0.0095), -(HY + 0.004), 0.05, 0.158, S_OSR)
    _bx(bm, 0.62, 0.92, HY + 0.006, HY + 0.014, 0.065, 0.145, S_BLACK)
    for k in range(3):
        xc = 0.68 + 0.09 * k
        _cyl(bm, (xc, HY + 0.014, 0.105), (xc, HY + 0.026, 0.105), 0.016, 12, S_BLACK)
        _cyl(bm, (xc, HY + 0.026, 0.105), (xc, HY + 0.029, 0.105), 0.013, 12, S_FIT)
    _bx(bm, -0.92, -0.62, -(HY + 0.014), -(HY + 0.006), 0.065, 0.145, S_BLACK)
    for xc in (-0.84, -0.70):
        _cyl(bm, (xc, -(HY + 0.014), 0.105), (xc, -(HY + 0.032), 0.105), 0.01, 10, S_FIT)
        _cyl(bm, (xc, -(HY + 0.032), 0.105), (xc, -(HY + 0.04), 0.105), 0.014, 6, S_BLACK)
    return bm


# ---------------------------------------------------------------------------------
# stack structure
# ---------------------------------------------------------------------------------
PLATE_T = 0.008


def _plate(bm, z0, t=PLATE_T, mat=0):
    """Machined retention plate with lightening cut-outs (inside the rod lugs)."""
    outer = _fillet(notched_outline(1.2, 1.9, 1.06, 1.76), 0.04, 3)
    holes = []
    for sx in (1.0, -1.0):
        for sy in (1.0, -1.0):
            for cx, cy, w, h in ((0.62, 0.5, 0.74, 0.6), (0.62, 1.33, 0.74, 0.62)):
                holes.append(U.rounded_rect(sx * cx, sy * cy, w, h, 0.08, 4))
    U.flat_prism(bm, [outer] + holes, z0, z0 + t, mat, mat, mat)


def _frame_grid(bm, z0, top=False):
    """Base / top retention frame: perimeter beams, diagonals to the rod lugs, hub ring,
    and the retention plate on top of the beams."""
    h = FRAME_H
    px, py = 1.12, 1.82
    w = 0.1
    # perimeter
    _bx(bm, -px - w / 2, px + w / 2, py - w / 2, py + w / 2, z0, z0 + h, 0)
    _bx(bm, -px - w / 2, px + w / 2, -py - w / 2, -py + w / 2, z0, z0 + h, 0)
    _bx(bm, px - w / 2, px + w / 2, -py, py, z0, z0 + h, 0)
    _bx(bm, -px - w / 2, -px + w / 2, -py, py, z0, z0 + h, 0)
    # cross members and diagonals
    _bx(bm, -px, px, -w / 2, w / 2, z0, z0 + h, 0)
    _bx(bm, -w / 2, w / 2, -py, py, z0, z0 + h, 0)
    for sx in (1.0, -1.0):
        for sy in (1.0, -1.0):
            a = Vector((sx * 0.55, sy * 0.55, z0 + h / 2))
            b = Vector((sx * (ROD_X - 0.1), sy * (ROD_Y - 0.1), z0 + h / 2))
            d = b - a
            fr = d.normalized()
            yv = Vector((0, 0, 1)).cross(fr).normalized()
            mm = Matrix((fr, yv, Vector((0, 0, 1)))).transposed().to_4x4()
            mm.translation = (a + b) / 2
            U.add_box(bm, (d.length, w * 0.9, h * 0.9), mm, 0)
            # rod lug block
            _bx(bm, sx * 1.08, sx * 1.29, sy * 1.78, sy * 1.99, z0 - 0.01, z0 + h + 0.01, 0)
            # lightening holes rendered as dark inset discs on the lugs' outer faces
            _cyl(bm, (sx * 1.29, sy * 1.86, z0 + h / 2), (sx * 1.295, sy * 1.86, z0 + h / 2), 0.03, 12, 1)
    # hub ring
    sub = bmesh.new()
    U.revolve(sub, [(0.62, z0), (0.86, z0), (0.86, z0 + h), (0.62, z0 + h)], 48, closed=True)
    U.merge_into(bm, sub, 0)
    _plate(bm, z0 + h)
    if top:
        # centre fitting and harness/pyro boxes on the top plate
        zp = z0 + h + PLATE_T
        _cyl(bm, (0, 0, zp), (0, 0, zp + 0.08), 0.22, 32, 1)
        _cyl(bm, (0, 0, zp + 0.08), (0, 0, zp + 0.1), 0.16, 32, 0)
        for sx in (1.0, -1.0):
            _bx(bm, sx * 0.25, sx * 0.55, -0.12, 0.12, zp, zp + 0.09, 1)
            _bx(bm, -0.12, 0.12, sx * 0.9, sx * 1.3, zp, zp + 0.07, 1)
            for sy in (1.0, -1.0):      # plate bolts along the beams
                for k in range(5):
                    t = (k + 0.5) / 5
                    _cyl(bm, (sx * 1.12, sy * (0.15 + 1.5 * t), zp), (sx * 1.12, sy * (0.15 + 1.5 * t), zp + 0.006),
                         0.011, 6, 1)


def build(coll, M=None, parent=None, yaw=0.0, n_sats=N_SATS):
    """Build the stack into `coll`; returns the root empty "SL_StackRoot".

    The root sits at the PAF top (world z = D.Z_FAIRING_BASE + 0.82 without a parent,
    else that position in the parent's frame); yaw rotates the stack about Z."""
    mt = _materials(M)
    root = U.empty("SL_StackRoot", coll, parent=parent, location=(0.0, 0.0, D.Z_FAIRING_BASE + PAF_TOP),
                   size=3.0, kind="PLAIN_AXES")
    root.rotation_euler = (0.0, 0.0, yaw)
    root["n_sats"] = n_sats
    root["stack_height"] = stack_height(n_sats)

    # --- adapter ring on the PAF + base frame ----------------------------------------
    bm = bmesh.new()
    prof = [(0.80, 0.0), (0.865, 0.0), (0.865, 0.05), (0.84, 0.07), (0.84, ADAPTER_H - 0.05), (0.9, ADAPTER_H - 0.03),
            (0.9, ADAPTER_H), (0.62, ADAPTER_H), (0.62, ADAPTER_H - 0.03), (0.80, ADAPTER_H - 0.05)]
    U.revolve(bm, prof, 64, closed=True, mat=0)
    for k in range(16):
        a = U.TAU * (k + 0.5) / 16
        c = Vector((0.855 * math.cos(a), 0.855 * math.sin(a), ADAPTER_H / 2 + 0.01))
        fr = Matrix.Translation(c) @ Matrix.Rotation(a, 4, "Z")
        U.add_box(bm, (0.04, 0.012, ADAPTER_H - 0.1), fr, 0)
    for k in range(32):
        a = U.TAU * k / 32
        c = Vector((0.88 * math.cos(a), 0.88 * math.sin(a), ADAPTER_H + 0.006))
        U.add_cyl(bm, 0.011, 0.011, 0.012, 6, Matrix.Translation(c), 2)
    _frame_grid(bm, ADAPTER_H, top=False)
    # separation-nut housings under the rod lugs
    for sx in (1.0, -1.0):
        for sy in (1.0, -1.0):
            c = Vector((sx * ROD_X, sy * ROD_Y, ADAPTER_H))
            _cyl(bm, c - Vector((0, 0, 0.07)), c, 0.05, 16, 1)
            _cyl(bm, c + Vector((0, 0, FRAME_H)), c + Vector((0, 0, Z_FIRST - ADAPTER_H)), 0.05, 16, 0)
            fc = Vector((sx * FIT_C[0], sy * FIT_C[1], ADAPTER_H + FRAME_H))
            _cyl(bm, fc, fc + Vector((0, 0, Z_FIRST - ADAPTER_H - FRAME_H)), 0.045, 16, 0)
    U.finish("SL_BaseAdapter", bm, coll, [mt["alu"], mt["black"], mt["steel"]], parent=root, sharp=35.0)

    # --- satellites (linked duplicates of one mesh) ----------------------------------
    bm = build_satellite_mesh()
    me = bpy.data.meshes.new("SL_Satellite")
    bm.normal_update()
    bm.to_mesh(me)
    bm.free()
    for key in SAT_SLOTS:
        me.materials.append(mt[key])
    me.shade_smooth()
    me.set_sharp_from_angle(angle=math.radians(35.0))
    for i in range(n_sats):
        ob = bpy.data.objects.new("SL_Satellite_%02d" % (i + 1), me)
        coll.objects.link(ob)
        ob.parent = root
        ob.matrix_basis = Matrix.Translation((0.0, 0.0, Z_FIRST + i * PITCH))

    # --- top retention frame ------------------------------------------------------------
    z_top0 = Z_FIRST + (n_sats - 1) * PITCH + SAT_H + 0.03
    bm = bmesh.new()
    _frame_grid(bm, z_top0, top=True)
    U.finish("SL_TopFrame", bm, coll, [mt["alu"], mt["black"]], parent=root, sharp=35.0)

    # --- tension rods with their top tensioners -------------------------------------------
    bm = bmesh.new()
    z_rod0 = ADAPTER_H - 0.06
    z_rod1 = z_top0 + FRAME_H + 0.16
    for sx in (1.0, -1.0):
        for sy in (1.0, -1.0):
            p = Vector((sx * ROD_X, sy * ROD_Y, 0.0))
            _cyl(bm, p + Vector((0, 0, z_rod0)), p + Vector((0, 0, z_rod1)), ROD_R, 16, 0)
            zt = z_top0 + FRAME_H
            _cyl(bm, p + Vector((0, 0, zt)), p + Vector((0, 0, zt + 0.07)), 0.058, 20, 1)     # spring housing
            _cyl(bm, p + Vector((0, 0, zt + 0.07)), p + Vector((0, 0, zt + 0.1)), 0.05, 6, 2)  # hex nut
            _cyl(bm, p + Vector((0, 0, zt + 0.1)), p + Vector((0, 0, zt + 0.13)), 0.044, 20, 1)
    U.finish("SL_TensionRods", bm, coll, [mt["rod"], mt["black"], mt["steel"]], parent=root, sharp=35.0)
    return root


# ---------------------------------------------------------------------------------
# clearance check against the fairing's inner mould line
# ---------------------------------------------------------------------------------
def fairing_inner_profile():
    """(r, z_f) inner wall of the fairing, z_f above the fairing base plane."""
    outer = FR.outer_profile()
    nrm = FR.profile_normals(outer)
    inner = []
    for p, nv in zip(outer, nrm):
        q = Vector(p) - nv * D.FAIRING_WALL
        inner.append((max(0.0, q.x), q.y))
    return inner


def _inner_table(inner, step=0.002, z_max=None):
    """Dense lookup of the smallest inner radius at each z_f (step metres)."""
    z_max = z_max if z_max is not None else D.FAIRING_LEN
    n = int(z_max / step) + 2
    tab = [None] * n
    for (r0, z0), (r1, z1) in zip(inner[:-1], inner[1:]):
        lo, hi = min(z0, z1), max(z0, z1)
        k0, k1 = max(0, int(math.ceil(lo / step))), min(n - 1, int(math.floor(hi / step)))
        for k in range(k0, k1 + 1):
            if abs(z1 - z0) < 1e-9:
                r = min(r0, r1)
            else:
                r = r0 + (r1 - r0) * (k * step - z0) / (z1 - z0)
            tab[k] = r if tab[k] is None else min(tab[k], r)
    last = 0.0
    for k in range(n):            # fill any holes (shouldn't happen) with the last value
        if tab[k] is None:
            tab[k] = last
        last = tab[k]
    return tab


def clearance_report(root, verbose=True):
    """Minimum radial clearance (m) between every stack vertex and the fairing inner wall.

    Works in world coordinates, so the stack must be at its flight position (vehicle axis
    on world Z, fairing base plane at D.Z_FAIRING_BASE).  Returns
    (min_clearance, z_f at the minimum, object name)."""
    import numpy as np
    bpy.context.view_layer.update()
    step = 0.002
    tab = np.array(_inner_table(fairing_inner_profile(), step), dtype=np.float64)
    best = (1e9, 0.0, "")
    stack = list(root.children)
    while stack:
        ob = stack.pop()
        stack.extend(ob.children)
        if ob.type != "MESH" or not len(ob.data.vertices):
            continue
        co = np.empty(len(ob.data.vertices) * 3, dtype=np.float64)
        ob.data.vertices.foreach_get("co", co)
        co = co.reshape(-1, 3)
        mw = np.array(ob.matrix_world, dtype=np.float64)
        w = co @ mw[:3, :3].T + mw[:3, 3]
        zf = w[:, 2] - D.Z_FAIRING_BASE
        k = np.clip(zf / step, 0, len(tab) - 2)
        k0 = np.floor(k).astype(np.int64)
        t = k - k0
        r_in = tab[k0] * (1 - t) + tab[k0 + 1] * t
        r_in = np.where((zf < 0) | (zf > D.FAIRING_LEN), 0.0, r_in)
        c = r_in - np.hypot(w[:, 0], w[:, 1])
        i = int(np.argmin(c))
        if c[i] < best[0]:
            best = (float(c[i]), float(zf[i]), ob.name)
    if verbose:
        print(f"[payload] min radial clearance to the fairing inner wall: {best[0] * 1000:.0f} mm "
              f"at z_f = {best[1]:.2f} m ({best[2]}); stack top z_f = "
              f"{PAF_TOP + stack_height(int(root.get('n_sats', N_SATS))):.2f} m "
              f"(fairing barrel ends at z_f = {D.FAIRING_CYL_TOP:.2f} m)", flush=True)
    return best


def triangle_count(root):
    n = 0
    stack = list(root.children)
    while stack:
        ob = stack.pop()
        stack.extend(ob.children)
        if ob.type == "MESH":
            ob.data.calc_loop_triangles()
            n += len(ob.data.loop_triangles)
    return n
