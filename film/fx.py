"""Film effects: engine plumes, smoke trail, MVac glow, cold-gas puffs, steam.

All effects are procedural volumes (closed meshes with Principled Volume shaders) whose
noise is advected by a keyed `time` value, so they flow when the film plays.  Each builder
returns a dict of the objects and the node sockets a shot keys (power, time, top ...)."""
import math

import bmesh
import bpy
from mathutils import Vector

from . import common as C
from f9 import dims as D


# --- tiny node-expression helper ---------------------------------------------------------
class NB:
    """Build shader math with less boilerplate: values are sockets or numbers."""

    def __init__(self, nt):
        self.nt = nt
        self.i = 0

    def _node(self, kind):
        n = self.nt.nodes.new(kind)
        n.location = (-1800 + 180 * (self.i % 12), 900 - 140 * (self.i // 12))
        self.i += 1
        return n

    def _set(self, sock, v):
        if isinstance(v, (int, float)):
            sock.default_value = v
        elif isinstance(v, (tuple, list, Vector)):
            sock.default_value = tuple(v)
        else:
            self.nt.links.new(v, sock)

    def math(self, op, a, b=None, c=None, clamp=False):
        n = self._node("ShaderNodeMath")
        n.operation = op
        n.use_clamp = clamp
        self._set(n.inputs[0], a)
        if b is not None:
            self._set(n.inputs[1], b)
        if c is not None:
            self._set(n.inputs[2], c)
        return n.outputs[0]

    def add(self, a, b):
        return self.math("ADD", a, b)

    def sub(self, a, b):
        return self.math("SUBTRACT", a, b)

    def mul(self, a, b, clamp=False):
        return self.math("MULTIPLY", a, b, clamp=clamp)

    def div(self, a, b):
        return self.math("DIVIDE", a, b)

    def pow(self, a, b):
        return self.math("POWER", a, b)

    def clamp01(self, a):
        return self.math("ADD", a, 0.0, clamp=True)

    def smooth(self, e0, e1, x):
        n = self._node("ShaderNodeMapRange")
        n.interpolation_type = "SMOOTHSTEP"
        self._set(n.inputs["Value"], x)
        self._set(n.inputs["From Min"], e0)
        self._set(n.inputs["From Max"], e1)
        n.inputs["To Min"].default_value = 0.0
        n.inputs["To Max"].default_value = 1.0
        return n.outputs["Result"]

    def maprange(self, x, a0, a1, b0, b1, clamp=True):
        n = self._node("ShaderNodeMapRange")
        n.clamp = clamp
        self._set(n.inputs["Value"], x)
        self._set(n.inputs["From Min"], a0)
        self._set(n.inputs["From Max"], a1)
        self._set(n.inputs["To Min"], b0)
        self._set(n.inputs["To Max"], b1)
        return n.outputs["Result"]

    def vmath(self, op, a, b=None, scale=None):
        n = self._node("ShaderNodeVectorMath")
        n.operation = op
        self._set(n.inputs[0], a)
        if b is not None:
            self._set(n.inputs[1], b)
        if scale is not None:
            self._set(n.inputs["Scale"], scale)
        if op in ("LENGTH", "DOT_PRODUCT", "DISTANCE"):
            return n.outputs["Value"]
        return n.outputs["Vector"]

    def sep(self, v):
        n = self._node("ShaderNodeSeparateXYZ")
        self._set(n.inputs[0], v)
        return n.outputs["X"], n.outputs["Y"], n.outputs["Z"]

    def comb(self, x, y, z):
        n = self._node("ShaderNodeCombineXYZ")
        for i, v in enumerate((x, y, z)):
            self._set(n.inputs[i], v)
        return n.outputs[0]

    def value(self, v=0.0, label=""):
        n = self._node("ShaderNodeValue")
        n.outputs[0].default_value = v
        n.label = label
        return n.outputs[0]

    def noise(self, vec, w=None, scale=1.0, detail=4.0, rough=0.55, distortion=0.0, kind="FBM"):
        n = self._node("ShaderNodeTexNoise")
        n.noise_dimensions = "4D" if w is not None else "3D"
        n.noise_type = kind
        self._set(n.inputs["Vector"], vec)
        if w is not None:
            self._set(n.inputs["W"], w)
        self._set(n.inputs["Scale"], scale)
        n.inputs["Detail"].default_value = detail
        n.inputs["Roughness"].default_value = rough
        n.inputs["Distortion"].default_value = distortion
        return n.outputs["Fac"]

    def ramp(self, fac, stops):
        n = self._node("ShaderNodeValToRGB")
        cr = n.color_ramp
        cr.elements[0].position, cr.elements[0].color = stops[0][0], (*stops[0][1], 1.0)
        cr.elements[1].position, cr.elements[1].color = stops[-1][0], (*stops[-1][1], 1.0)
        for pos, col in stops[1:-1]:
            e = cr.elements.new(pos)
            e.color = (*col, 1.0)
        self._set(n.inputs["Fac"], fac)
        return n.outputs["Color"]

    def coord(self, kind="Object"):
        n = self._node("ShaderNodeTexCoord")
        return n.outputs[kind]

    def position(self):
        n = self._node("ShaderNodeNewGeometry")
        return n.outputs["Position"]

    def mixrgb(self, fac, a, b, blend="MIX"):
        n = self._node("ShaderNodeMix")
        n.data_type = "RGBA"
        n.blend_type = blend
        self._set(n.inputs[0], fac)
        ia = [s for s in n.inputs if s.name == "A" and s.type == "RGBA"][0]
        ib = [s for s in n.inputs if s.name == "B" and s.type == "RGBA"][0]
        self._set(ia, a)
        self._set(ib, b)
        return [s for s in n.outputs if s.type == "RGBA"][0]


def volume_material(name):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    for nd in list(nt.nodes):
        nt.nodes.remove(nd)
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    out.location = (600, 0)
    vol = nt.nodes.new("ShaderNodeVolumePrincipled")
    vol.location = (300, 0)
    nt.links.new(vol.outputs["Volume"], out.inputs["Volume"])
    vol.inputs["Density"].default_value = 0.0
    return m, nt, vol


def revolve_mesh(name, profile, segs=48, coll=None):
    """Closed solid of revolution about local Z from [(r, z), ...] (top to bottom)."""
    bm = bmesh.new()
    rings = []
    for r, z in profile:
        ring = [bm.verts.new((r * math.cos(2 * math.pi * k / segs), r * math.sin(2 * math.pi * k / segs), z))
                for k in range(segs)]
        rings.append(ring)
    for a, b in zip(rings, rings[1:]):
        for k in range(segs):
            bm.faces.new((a[k], a[(k + 1) % segs], b[(k + 1) % segs], b[k]))
    bm.faces.new(list(reversed(rings[0])))
    bm.faces.new(rings[-1])
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    ob = bpy.data.objects.new(name, me)
    (coll or bpy.context.scene.collection).objects.link(ob)
    ob.visible_shadow = False
    return ob


def key_socket(sock, frames, fn):
    """Key a Value-node output (or any socket default) from fn(t, f), linear."""
    for f in range(1, frames + 2):
        t = (f - 1) / max(1, frames - 1)
        sock.default_value = fn(t, f)
        sock.keyframe_insert("default_value", frame=f)
    nt = sock.id_data
    if nt.animation_data and nt.animation_data.action:
        for fc in C._fcurves(nt.animation_data.action):
            for kp in fc.keyframe_points:
                kp.interpolation = "LINEAR"


# --- first-stage plume -------------------------------------------------------------------
PLUME_RAMP = [(0.0, (1.0, 0.93, 0.78)), (0.04, (1.0, 0.78, 0.38)), (0.12, (1.0, 0.60, 0.17)),
              (0.3, (1.0, 0.42, 0.08)), (0.55, (0.85, 0.22, 0.04)), (1.0, (0.25, 0.05, 0.01))]


def plume(parent, frames, name="Plume", length=70.0, r_end=10.0, intensity=60.0, flow=90.0,
          engines="nine", smoke=0.0, coll=None):
    """Exhaust plume below the nozzle exits (local z < 0 of `parent`), emission volume.

    engines: "nine" (octaweb: 9 jets merging within ~8 m) or "one" (centre engine only).
    Returns dict(ob, power, time, mat).  Key `power` (0..1) for ignition / cutoff and
    `time` (seconds) so the turbulence flows downstream at `flow` m/s."""
    one = engines == "one"
    r0 = 0.55 if one else 2.0
    prof = [(r0 * 0.9, 0.35), (r0 * 1.15, -2.0), (C.lerp(r0, r_end, 0.35), -0.25 * length),
            (C.lerp(r0, r_end, 0.7), -0.6 * length), (r_end, -length)]
    ob = revolve_mesh(name, prof, segs=40, coll=coll)
    ob.parent = parent
    m, nt, vol = volume_material(name)
    ob.data.materials.append(m)
    nb = NB(nt)
    time = nb.value(0.0, "time")
    power = nb.value(1.0, "power")
    p = nb.coord("Object")
    x, y, z = nb.sep(p)
    s = nb.clamp01(nb.div(nb.mul(z, -1.0), length))                    # 0 at nozzles .. 1 at the end
    rr = nb.vmath("LENGTH", nb.comb(x, y, 0.0))
    # turbulence advected downstream: n1 = big puffs, n2 = ragged boundary detail
    flowz = nb.add(z, nb.mul(time, flow))
    pn = nb.comb(x, y, flowz)
    sc = 80.0 / length
    n1 = nb.noise(pn, w=nb.mul(time, 0.8), scale=0.07 * sc, detail=5.0, rough=0.6, distortion=0.6)
    n2 = nb.noise(pn, w=nb.mul(time, 1.6), scale=0.35 * sc, detail=4.0, rough=0.55)
    edge = nb.add(0.72, nb.mul(n2, 0.6))
    # local plume radius: jets near the nozzles, merged column downstream
    R = nb.add(r0 * 0.75, nb.mul(nb.pow(s, 0.75), r_end - r0 * 0.75))
    q = nb.div(nb.div(rr, R), edge)
    merged = nb.smooth(1.0, 0.45, q)
    if one:
        radial = merged
    else:
        # distance to the nearest of the nine engine axes
        ang = nb.math("ARCTAN2", y, x)
        step = math.radians(45.0)
        k = nb.math("ROUND", nb.div(nb.sub(ang, math.radians(22.5)), step))
        an = nb.add(nb.mul(k, step), math.radians(22.5))
        ex, ey = nb.mul(nb.math("COSINE", an), D.M1D_RING_R), nb.mul(nb.math("SINE", an), D.M1D_RING_R)
        d_out = nb.vmath("LENGTH", nb.comb(nb.sub(x, ex), nb.sub(y, ey), 0.0))
        d_jet = nb.math("MINIMUM", d_out, rr)
        rj = nb.add(0.42, nb.mul(s, length * 0.09))
        qj = nb.div(d_jet, rj)
        jets = nb.pow(nb.clamp01(nb.sub(1.0, nb.mul(qj, qj))), 1.3)
        w = nb.smooth(0.02, 0.14, s)
        radial = nb.add(nb.mul(jets, nb.sub(1.0, w)), nb.mul(merged, w))
    puffs = nb.add(0.18, nb.mul(nb.smooth(0.38, 0.78, n1), 1.6))
    # axial brightness: hot near field with shock-diamond banding, decaying downstream
    axial = nb.add(nb.math("EXPONENT", nb.mul(s, -5.5)), nb.mul(nb.math("EXPONENT", nb.mul(s, -1.0 / 0.035)), 3.0))
    band = nb.add(1.0, nb.mul(nb.mul(nb.add(0.5, nb.mul(nb.math("COSINE", nb.mul(z, 2 * math.pi / 2.3)), 0.5)), 0.45),
                              nb.sub(1.0, nb.smooth(0.0, 0.22, s))))
    e = nb.mul(nb.mul(nb.mul(radial, axial), nb.mul(puffs, band)), nb.mul(power, intensity))
    cfac = nb.clamp01(nb.add(s, nb.mul(nb.sub(n1, 0.5), 0.25)))
    col = nb.ramp(cfac, PLUME_RAMP)
    nt.links.new(e, vol.inputs["Emission Strength"])
    nt.links.new(col, vol.inputs["Emission Color"])
    if smoke > 0:
        shell = nb.mul(nb.smooth(0.35, 0.9, q), nb.sub(1.0, nb.smooth(0.95, 1.1, q)))
        dens = nb.mul(nb.mul(nb.add(shell, nb.mul(radial, 0.3)), nb.smooth(0.1, 0.7, s)), nb.mul(n1, smoke))
        nt.links.new(dens, vol.inputs["Density"])
        vol.inputs["Color"].default_value = (0.5, 0.45, 0.4, 1.0)
    key_socket(time, frames, lambda t, f: (f - 1) / C.FPS)
    return {"ob": ob, "power": power, "time": time, "mat": m}


def plume_light(parent, name="PlumeLight", z=-8.0, energy=4.0e6, radius=4.0, color=(1.0, 0.62, 0.32), coll=None):
    ob = C.point_light(name, energy, radius=radius, color=color, loc=(0.0, 0.0, z), coll=coll)
    ob.parent = parent
    return ob


# --- smoke trail -------------------------------------------------------------------------
def trail(frames, top_fn, name="Trail", r_top=6.0, spread=0.12, z_bottom=0.0, density=0.08,
          color=(0.92, 0.9, 0.88), anisotropy=0.45, coll=None, half=700.0):
    """World-space vertical smoke column from z_bottom up to top_fn(t) (the plume's end),
    widening with age below the vehicle.  The noise is fixed in world space, so the smoke
    hangs in the air while the vehicle climbs out of it."""
    zt0 = top_fn(0.0)
    zt1 = top_fn(1.0)
    rmax = r_top + spread * (max(zt0, zt1) - z_bottom)
    prof = [(r_top * 1.4, max(zt0, zt1) + 2.0), (rmax * 1.2, z_bottom - 1.0)]
    ob = revolve_mesh(name, prof, segs=48, coll=coll)
    m, nt, vol = volume_material(name)
    ob.data.materials.append(m)
    nb = NB(nt)
    top = nb.value(zt0, "top")
    time = nb.value(0.0, "time")
    p = nb.position()
    x, y, z = nb.sep(p)
    below = nb.math("MAXIMUM", nb.sub(top, z), 0.0)                       # metres below the plume end
    rad = nb.add(r_top, nb.mul(below, spread))
    rr = nb.vmath("LENGTH", nb.comb(x, y, 0.0))
    # billowing edge: displace the radius with low-frequency noise
    pn = nb.comb(x, y, nb.add(z, nb.mul(time, 1.5)))
    nlow = nb.noise(pn, w=nb.mul(time, 0.05), scale=0.025, detail=3.0, rough=0.5)
    nhi = nb.noise(pn, w=nb.mul(time, 0.1), scale=0.12, detail=5.0, rough=0.6, distortion=0.2)
    q = nb.div(rr, nb.mul(rad, nb.add(0.75, nb.mul(nlow, 0.5))))
    core = nb.clamp01(nb.sub(1.0, nb.pow(q, 2.0)))
    above = nb.sub(1.0, nb.smooth(-4.0, 6.0, nb.sub(z, top)))           # fade in just under the plume end
    dens = nb.mul(nb.mul(core, above), nb.mul(nb.add(0.25, nb.mul(nhi, 1.1)), density))
    nt.links.new(dens, vol.inputs["Density"])
    vol.inputs["Color"].default_value = (*color, 1.0)
    vol.inputs["Anisotropy"].default_value = anisotropy
    key_socket(top, frames, lambda t, f: top_fn(t))
    key_socket(time, frames, lambda t, f: (f - 1) / C.FPS)
    return {"ob": ob, "top": top, "time": time, "mat": m}


# --- twilight "jellyfish" -----------------------------------------------------------------
def paraboloid(name, length, radius, segs=72, rings=48, coll=None):
    """Open paraboloid shell: apex at the origin, opening along local -Z (r = radius * u,
    z = -length * u^2)."""
    bm = bmesh.new()
    apex = bm.verts.new((0.0, 0.0, 0.0))
    prev = None
    for k in range(1, rings + 1):
        u = k / rings
        r, z = radius * u, -length * u * u
        ring = [bm.verts.new((r * math.cos(2 * math.pi * j / segs), r * math.sin(2 * math.pi * j / segs), z))
                for j in range(segs)]
        if prev is None:
            for j in range(segs):
                bm.faces.new((apex, ring[(j + 1) % segs], ring[j]))
        else:
            for j in range(segs):
                bm.faces.new((prev[j], prev[(j + 1) % segs], ring[(j + 1) % segs], ring[j]))
        prev = ring
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    for poly in me.polygons:
        poly.use_smooth = True
    ob = bpy.data.objects.new(name, me)
    (coll or bpy.context.scene.collection).objects.link(ob)
    ob.visible_shadow = False
    return ob


def glow_surface(name, length, color, strength, limb=2.2, streaks=7.0, rim_fade=(0.55, 1.0), apex_color=None,
                 body=False):
    """Additive translucent glow for a thin luminous shell: brightness rises toward the limb
    (grazing view), modulated by streaks along the meridians, dissolving toward the open rim.
    body=True instead brightens toward the centre and fades to nothing at the silhouette, so the
    shell reads as a soft filled glow (limb is then the falloff exponent).
    Returns (material, time socket, strength socket)."""
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    for nd in list(nt.nodes):
        nt.nodes.remove(nd)
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    out.location = (900, 0)
    nb = NB(nt)
    time = nb.value(0.0, "time")
    gain = nb.value(strength, "strength")
    lw = nb._node("ShaderNodeLayerWeight")
    lw.inputs["Blend"].default_value = 0.5
    facing = lw.outputs["Facing"]
    limbf = nb.pow(nb.sub(1.0, facing), limb) if body else nb.add(0.08, nb.pow(facing, limb))
    p = nb.coord("Object")
    x, y, z = nb.sep(p)
    u = nb.math("SQRT", nb.clamp01(nb.div(nb.mul(z, -1.0), length)))        # 0 apex .. 1 rim
    ang = nb.math("ARCTAN2", y, x)
    sv = nb.comb(nb.mul(nb.math("COSINE", ang), streaks), nb.mul(nb.math("SINE", ang), streaks), nb.mul(u, 1.2))
    st = nb.noise(sv, w=nb.mul(time, 0.05), scale=1.0, detail=5.0, rough=0.6, distortion=0.3)
    stk = nb.add(0.25, nb.mul(nb.smooth(0.35, 0.75, st), 1.5))
    rim = nb.sub(1.0, nb.smooth(rim_fade[0], rim_fade[1], u))
    e = nb.mul(nb.mul(nb.mul(limbf, stk), rim), gain)
    if apex_color is not None:
        col = nb.mixrgb(nb.smooth(0.0, 0.45, u), (*apex_color, 1.0), (*color, 1.0))
    else:
        col = (*color, 1.0)
    em = nt.nodes.new("ShaderNodeEmission")
    em.location = (500, 100)
    nb._set(em.inputs["Color"], col)
    nt.links.new(e, em.inputs["Strength"])
    tr = nt.nodes.new("ShaderNodeBsdfTransparent")
    tr.location = (500, -100)
    add = nt.nodes.new("ShaderNodeAddShader")
    add.location = (700, 0)
    nt.links.new(em.outputs["Emission"], add.inputs[0])
    nt.links.new(tr.outputs["BSDF"], add.inputs[1])
    nt.links.new(add.outputs["Shader"], out.inputs["Surface"])
    return m, time, gain


def tube(name, pts, radii, segs=24, coll=None):
    """Tube mesh along a polyline with per-point radius (closed ends)."""
    bm = bmesh.new()
    rings = []
    n = len(pts)
    for i, (p, r) in enumerate(zip(pts, radii)):
        t = (pts[min(i + 1, n - 1)] - pts[max(i - 1, 0)]).normalized()
        a = t.orthogonal().normalized()
        b = t.cross(a).normalized()
        rings.append([bm.verts.new(p + r * (math.cos(2 * math.pi * j / segs) * a + math.sin(2 * math.pi * j / segs) * b))
                      for j in range(segs)])
    for r0, r1 in zip(rings, rings[1:]):
        for j in range(segs):
            bm.faces.new((r0[j], r0[(j + 1) % segs], r1[(j + 1) % segs], r1[j]))
    bm.faces.new(list(reversed(rings[0])))
    bm.faces.new(rings[-1])
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    for poly in me.polygons:
        poly.use_smooth = True
    ob = bpy.data.objects.new(name, me)
    (coll or bpy.context.scene.collection).objects.link(ob)
    ob.visible_shadow = False
    return ob


def trail_glow_material(name, strength, z_shadow, z_soft, lit=(1.0, 0.93, 0.85), dusk=(1.0, 0.55, 0.30),
                        shade=(0.10, 0.13, 0.22)):
    """Glowing exhaust trail seen at twilight: brightest along the tube's centre line, sunlit
    (white, reddening near the shadow line) above world z_shadow, dim and blue below it."""
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    for nd in list(nt.nodes):
        nt.nodes.remove(nd)
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    out.location = (900, 0)
    nb = NB(nt)
    time = nb.value(0.0, "time")
    lw = nb._node("ShaderNodeLayerWeight")
    lw.inputs["Blend"].default_value = 0.5
    core = nb.pow(nb.sub(1.0, lw.outputs["Facing"]), 2.5)
    pos = nb.position()
    _, _, z = nb.sep(pos)
    sun = nb.smooth(z_shadow - z_soft, z_shadow + z_soft, z)
    redden = nb.sub(1.0, nb.smooth(z_shadow, z_shadow + 6 * z_soft, z))
    n1 = nb.noise(pos, w=nb.mul(time, 0.03), scale=0.008, detail=6.0, rough=0.62, distortion=0.8)
    var = nb.add(0.15, nb.mul(nb.smooth(0.32, 0.78, n1), 1.6))
    litc = nb.mixrgb(redden, (*lit, 1.0), (*dusk, 1.0))
    col = nb.mixrgb(sun, (*shade, 1.0), litc)
    e = nb.mul(nb.mul(core, var), nb.add(0.06, nb.mul(sun, strength)))
    em = nt.nodes.new("ShaderNodeEmission")
    em.location = (500, 100)
    nt.links.new(col, em.inputs["Color"])
    nt.links.new(e, em.inputs["Strength"])
    tr = nt.nodes.new("ShaderNodeBsdfTransparent")
    tr.location = (500, -100)
    add = nt.nodes.new("ShaderNodeAddShader")
    add.location = (700, 0)
    nt.links.new(em.outputs["Emission"], add.inputs[0])
    nt.links.new(tr.outputs["BSDF"], add.inputs[1])
    nt.links.new(add.outputs["Shader"], out.inputs["Surface"])
    return m, time


# --- second stage: MVac vacuum plume and glowing niobium extension -------------------------
def mvac_plume(s2, frames, coll=None, scale=1.0):
    """Vacuum plume below the MVac exit (S2 local z = MVAC_Z_EXIT): nested luminous shells,
    a hot narrow core inside a huge faint bell.  Returns dict(power socket list, objects)."""
    shells = []
    spec = (("MVac_Core", 16.0, 2.3, (1.0, 0.86, 0.66), 1.6, 1.0, 3.0, (0.55, 1.0)),
            ("MVac_Mid", 48.0, 13.0, (0.80, 0.86, 1.0), 0.18, 2.0, 4.0, (0.25, 0.95)),
            ("MVac_Bell", 130.0, 70.0, (0.60, 0.72, 1.0), 0.06, 2.5, 5.0, (0.15, 0.9)))
    for name, L, R, col, k, falloff, streaks, rim in spec:
        ob = paraboloid(name, L * scale, R * scale, coll=coll)
        ob.parent = s2
        ob.location = (0.0, 0.0, D.MVAC_Z_EXIT + 0.4)
        m, tsock, gsock = glow_surface(name, L * scale, col, k, limb=falloff, streaks=streaks, rim_fade=rim,
                                       apex_color=(1.0, 0.95, 0.85), body=name != "MVac_Core")
        ob.data.materials.append(m)
        key_socket(tsock, frames, lambda t, f: (f - 1) / C.FPS)
        shells.append((ob, gsock, k))
    return shells


def key_shells(shells, frames, power_fn):
    """power_fn(t, f) -> 0..1 scales every shell's strength."""
    for ob, gsock, k in shells:
        key_socket(gsock, frames, lambda t, f, k=k: k * power_fn(t, f))


def niobium_glow(frames, glow_fn, z_exit=D.MVAC_Z_EXIT, z_top=D.MVAC_Z_EXT_TOP, strength=6.0):
    """Make the MVac niobium extension glow after ignition: emission from dull red at the
    exit to orange-yellow near the top, scaled by glow_fn(t, f) (0..1, keyed)."""
    m = bpy.data.materials["F9_niobium_c103"]
    nt = m.node_tree
    b = next(nd for nd in nt.nodes if nd.type == "BSDF_PRINCIPLED")
    nb = NB(nt)
    p = nb.coord("Object")
    _, _, z = nb.sep(p)
    u = nb.clamp01(nb.div(nb.sub(z, z_exit), z_top - z_exit))
    heat = nb.pow(u, 1.4)
    col = nb.ramp(heat, [(0.0, (0.55, 0.06, 0.01)), (0.5, (1.0, 0.22, 0.03)), (1.0, (1.0, 0.55, 0.15))])
    g = nb.value(0.0, "glow")
    e = nb.mul(nb.mul(nb.add(0.15, heat), g), strength)
    nt.links.new(col, b.inputs["Emission Color"])
    nt.links.new(e, b.inputs["Emission Strength"])
    key_socket(g, frames, glow_fn)
    return g


# --- cold gas, deck flame, vapour -----------------------------------------------------------
def gas_puff(parent, frames, name, loc, direction, times, length=5.0, radius=2.4, strength=1.6,
             coll=None):
    """Cold-gas thruster bursts: a soft white jet from `loc` (parent space) along `direction`,
    flashing on for each (start, duration) in `times` (seconds) and swelling as it fades."""
    ob = paraboloid(name, length, radius, segs=40, rings=20, coll=coll)
    ob.parent = parent
    ob.location = loc
    ob.rotation_mode = "QUATERNION"
    ob.rotation_quaternion = Vector((0.0, 0.0, -1.0)).rotation_difference(Vector(direction).normalized())
    m, tsock, gsock = glow_surface(name, length, (0.92, 0.95, 1.0), 0.0, limb=1.6, streaks=5.0,
                                   rim_fade=(0.2, 1.0), body=True)
    ob.data.materials.append(m)

    def env(T):
        a, s = 0.0, 0.0
        for t0, d in times:
            x = T - t0
            if x < 0.0:
                continue
            v = C.smooth(x / 0.04) * math.exp(-max(0.0, x - d) / 0.12)
            if v > a:
                a, s = v, C.smooth(x / (d + 0.3))
        return a, s

    key_socket(tsock, frames, lambda t, f: (f - 1) / C.FPS)
    key_socket(gsock, frames, lambda t, f: strength * env((f - 1) / C.FPS)[0])
    for f in range(1, frames + 2):
        s = env((f - 1) / C.FPS)[1]
        k = 0.45 + 0.85 * s
        ob.scale = (k, k, 0.5 + 0.9 * s)
        ob.keyframe_insert("scale", frame=f)
    C._linear(ob)
    return ob


def flame_pool(frames, name, center, radius, power_fn, height=5.0, intensity=1.0, coll=None):
    """Exhaust splashing across a deck: a ragged, flickering pancake of flame at `center`;
    power_fn(t, f) keys its strength."""
    prof = [(radius * 0.35, height), (radius * 0.75, height * 0.6), (radius, height * 0.25), (radius, 0.0)]
    ob = revolve_mesh(name, prof, segs=48, coll=coll)
    ob.location = center
    m, nt, vol = volume_material(name)
    ob.data.materials.append(m)
    nb = NB(nt)
    time = nb.value(0.0, "time")
    power = nb.value(0.0, "power")
    x, y, z = nb.sep(nb.coord("Object"))
    rr = nb.vmath("LENGTH", nb.comb(x, y, 0.0))
    s = nb.clamp01(nb.div(rr, radius))
    n1 = nb.noise(nb.comb(x, y, nb.mul(z, 1.5)), w=nb.mul(time, 1.4), scale=0.22, detail=5.0,
                  rough=0.6, distortion=0.8)
    vert = nb.smooth(nb.mul(height, nb.sub(1.0, nb.mul(s, 0.75))), 0.0, z)
    edge = nb.smooth(1.0, 0.55, nb.add(s, nb.mul(nb.sub(n1, 0.5), 0.5)))
    lumps = nb.add(0.25, nb.mul(nb.smooth(0.4, 0.75, n1), 1.5))
    body = nb.mul(edge, vert)
    nt.links.new(nb.mul(nb.mul(body, lumps), nb.mul(power, intensity)), vol.inputs["Emission Strength"])
    cf = nb.clamp01(nb.add(nb.mul(s, 0.7), nb.add(0.12, nb.mul(nb.sub(n1, 0.5), 0.3))))
    nt.links.new(nb.ramp(cf, PLUME_RAMP), vol.inputs["Emission Color"])
    vol.inputs["Color"].default_value = (0.35, 0.32, 0.3, 1.0)
    nt.links.new(nb.mul(body, nb.mul(power, 0.04)), vol.inputs["Density"])
    key_socket(time, frames, lambda t, f: (f - 1) / C.FPS)
    key_socket(power, frames, power_fn)
    return ob


def steam(frames, name, center, radius, height, dens_fn, rise=1.5, grow=0.5, color=(0.92, 0.92, 0.94),
          coll=None):
    """Billowing vapour (white scattering volume) rising from `center`; dens_fn(t, f) keys the
    density and the cloud swells by `grow` (a fraction) across the shot."""
    prof = [(radius * 0.55, height), (radius, height * 0.55), (radius * 0.95, 0.0)]
    ob = revolve_mesh(name, prof, segs=40, coll=coll)
    ob.location = center
    m, nt, vol = volume_material(name)
    ob.data.materials.append(m)
    nb = NB(nt)
    time = nb.value(0.0, "time")
    dens = nb.value(0.0, "density")
    x, y, z = nb.sep(nb.coord("Object"))
    rr = nb.vmath("LENGTH", nb.comb(x, y, 0.0))
    zr = nb.clamp01(nb.div(z, height))
    q = nb.div(rr, nb.mul(radius, nb.sub(1.0, nb.mul(zr, 0.45))))
    n1 = nb.noise(nb.comb(x, y, nb.sub(z, nb.mul(time, rise))), w=nb.mul(time, 0.25), scale=1.4 / radius,
                  detail=6.0, rough=0.62, distortion=0.4)
    shape = nb.smooth(1.0, 0.5, nb.add(q, nb.mul(nb.sub(n1, 0.5), 0.8)))
    top = nb.smooth(1.0, 0.55, nb.add(zr, nb.mul(nb.sub(n1, 0.5), 0.5)))
    d = nb.mul(nb.mul(nb.mul(shape, top), nb.smooth(0.38, 0.72, n1)), dens)
    nt.links.new(d, vol.inputs["Density"])
    vol.inputs["Color"].default_value = (*color, 1.0)
    vol.inputs["Anisotropy"].default_value = 0.35
    key_socket(time, frames, lambda t, f: (f - 1) / C.FPS)
    key_socket(dens, frames, dens_fn)
    for f in range(1, frames + 2):
        k = 1.0 + grow * (f - 1) / max(1, frames)
        ob.scale = (k, k, k)
        ob.keyframe_insert("scale", frame=f)
    C._linear(ob)
    return ob
