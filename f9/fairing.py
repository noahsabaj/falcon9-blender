"""Two-piece composite payload fairing (standard, 5.2 m x 13.15 m).

Half A spans theta 90..270 deg (-X side), half B 270..450 deg (+X side); the vertical
seams sit at 90 / 270 deg.  Outline from the Falcon User's Guide fairing figure.
"""
import math

import bmesh
from mathutils import Matrix, Vector

from . import dims as D
from . import util as U

SEGS_HALF = 160
GAP = 0.003          # half-gap at each seam (m)
CAP_R = 0.68         # dark nose cap below this radius


def quad_bezier(p0, c, p2, n):
    out = []
    for k in range(1, n):
        t = k / n
        out.append(tuple((1 - t) ** 2 * Vector(p0) + 2 * (1 - t) * t * Vector(c) + t * t * Vector(p2)))
    return out


def outer_profile():
    """(r, z_f) outline from the base ring to the tip."""
    pts = [(D.R, 0.0), (D.R, 0.17), (2.07, 0.21)]
    # conical boattail (slope from (2.07, 0.21) to (RF, 1.22)) with a fillet into the barrel
    slope = (D.RF - 2.07) / (D.FAIRING_BOATTAIL_TOP - 0.21)
    z_corner = D.FAIRING_BOATTAIL_TOP
    p0 = (2.07 + slope * (1.03 - 0.21), 1.03)
    pts += [(2.07 + slope * (z - 0.21), z) for z in (0.45, 0.70, 0.90)]
    pts += [p0] + quad_bezier(p0, (D.RF, z_corner), (D.RF, 1.42), 6) + [(D.RF, 1.42)]
    n = 16
    pts += [(D.RF, 1.42 + (D.FAIRING_CYL_TOP - 1.42) * k / n) for k in range(1, n + 1)]
    nose = U.catmull_rom([(z, r) for z, r in D.FAIRING_NOSE], samples_per_seg=8)
    pts += [(r, z) for z, r in nose[1:]]
    pts[-1] = (0.0, D.FAIRING_LEN)
    return pts


def profile_normals(pts):
    out = []
    n = len(pts)
    for i in range(n):
        a = Vector(pts[max(0, i - 1)])
        b = Vector(pts[min(n - 1, i + 1)])
        t = (b - a).normalized()
        out.append(Vector((t.y, -t.x)))          # right of travel = outward
    return out


def arc_samples(pts, step, z_min, r_min):
    """Points (r, z, normal) every `step` of arc length along the outline."""
    nrm = profile_normals(pts)
    out = []
    acc = 0.0
    nxt = 0.0
    for i in range(len(pts) - 1):
        a, b = Vector(pts[i]), Vector(pts[i + 1])
        L = (b - a).length
        while nxt <= acc + L and L > 0:
            t = (nxt - acc) / L
            p = a.lerp(b, t)
            nv = nrm[i].lerp(nrm[i + 1], t).normalized()
            if p.y >= z_min and p.x >= r_min:
                out.append((p.x, p.y, nv))
            nxt += step
        acc += L
    return out


def radius_at(pts, z):
    for i in range(len(pts) - 1):
        (r0, z0), (r1, z1) = pts[i], pts[i + 1]
        if z0 <= z <= z1 and z1 > z0:
            return r0 + (r1 - r0) * (z - z0) / (z1 - z0)
    return 0.0


def surface_frame(theta, r, z, nv, lift=0.0):
    """World matrix on the fairing skin: Z = surface normal, Y = tangential."""
    c, s = math.cos(theta), math.sin(theta)
    n3 = Vector((nv.x * c, nv.x * s, nv.y))
    tan = Vector((-s, c, 0.0))
    x = tan.cross(n3).normalized()
    m = Matrix((x, tan, n3)).transposed().to_4x4()
    m.translation = Vector((r * c, r * s, z)) + n3 * lift
    return m


def shell_half(a0, a1, z_base):
    outer = outer_profile()
    N = len(outer)
    nrm = profile_normals(outer)
    inner = []
    for i, (p, nv) in enumerate(zip(outer, nrm)):
        q = Vector(p) - nv * D.FAIRING_WALL
        inner.append((max(0.004, q.x), q.y))
    inner[-1] = (0.0, D.FAIRING_LEN - D.FAIRING_WALL)
    prof = outer + list(reversed(inner))
    bm = bmesh.new()
    rings, faces = U.revolve(bm, prof, SEGS_HALF, a0=a0, a1=a1, closed=True)
    # end caps at the seams
    caps = []
    for j in (0, len(rings[0]) - 1):
        for i in range(N - 1):
            q = [rings[i][j], rings[i + 1][j], rings[2 * N - 2 - i][j], rings[2 * N - 1 - i][j]]
            q = list(dict.fromkeys(q))
            if len(q) < 3:
                continue
            try:
                caps.append(bm.faces.new(q))
            except ValueError:
                pass
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
    caps = set(caps)
    # materials: 0 white, 1 nose cap, 2 seam / edges, 3 interior
    idx = {}
    for i, ring in enumerate(rings):
        for v in ring:
            idx[v] = i
    for f in bm.faces:
        ids = {idx.get(v, -1) for v in f.verts}
        if f in caps:
            f.material_index = 2
        elif all(0 <= i < N for i in ids):
            cen = f.calc_center_median()
            r = math.hypot(cen.x, cen.y)
            f.material_index = 1 if (cen.z > 11.0 and r < CAP_R) else 0
        elif all(i >= N for i in ids):
            f.material_index = 3
        else:
            f.material_index = 2
    bm.transform(Matrix.Translation((0.0, 0.0, z_base)))
    return bm, outer


def details(bm, outer, seam_lo, seam_hi, z_base, mat_bolt, mat_vent, mat_seam):
    """Fasteners, bolt rows, vents, access door and nose thrusters for one half spanning
    theta seam_lo..seam_hi (radians)."""
    zb = z_base
    # seam fasteners: one row along each seam edge, every 0.15 m of outline
    for th_edge, sgn in ((seam_lo, 1.0), (seam_hi, -1.0)):
        for r, z, nv in arc_samples(outer, 0.15, 0.30, 0.45):
            th = th_edge + sgn * 0.045 / r
            m = surface_frame(th, r, z + zb, nv, lift=0.002)
            U.add_cyl(bm, 0.013, 0.010, 0.008, 8, m, mat=mat_bolt)
    # horizontal bolt rows near the seams
    for zr in (5.63, 10.72):
        r = radius_at(outer, zr)
        nv = profile_normals(outer)[min(range(len(outer)), key=lambda i: abs(outer[i][1] - zr))]
        k = 0
        while k * 0.10 <= 1.6:
            for th_edge, sgn in ((seam_lo, 1.0), (seam_hi, -1.0)):
                th = th_edge + sgn * (0.10 + k * 0.10) / r
                m = surface_frame(th, r, zr + zb, nv, lift=0.002)
                U.add_cyl(bm, 0.011, 0.009, 0.007, 8, m, mat=mat_bolt)
            k += 1
    # slanted vent louvres near the base of the barrel
    vent = bmesh.new()
    for k in range(10):
        cu = (k - 4.5) * 0.13
        cz = 1.95
        L, W = 0.75, 0.055
        d = Vector((math.cos(math.radians(45)), math.sin(math.radians(45))))
        nrm2 = Vector((-d.y, d.x))
        c0 = Vector((cu, cz))
        loop = [tuple(c0 + d * (-L / 2) + nrm2 * (-W / 2)), tuple(c0 + d * (L / 2) + nrm2 * (-W / 2)),
                tuple(c0 + d * (L / 2) + nrm2 * (W / 2)), tuple(c0 + d * (-L / 2) + nrm2 * (W / 2))]
        U.flat_prism(vent, [loop], -0.004, 0.0025, mat_vent, mat_vent, mat_vent)
    U.slice_u(vent, 0.05)
    for v in vent.verts:
        v.co.y += zb
    U.wrap(vent, seam_lo + math.radians(55.0), D.RF)
    U.merge_into(bm, vent)
    # circular access door with bolt ring
    door = bmesh.new()
    th_d = seam_lo + math.radians(20.0)
    zc = 3.4
    U.flat_prism(door, [U.ellipse(0.0, zc, 0.305, 0.305, 64), U.ellipse(0.0, zc, 0.298, 0.298, 64)],
                 -0.002, 0.0015, mat_seam, mat_seam, mat_seam)
    for k in range(16):
        a = U.TAU * k / 16
        U.flat_prism(door, [U.ellipse(0.335 * math.cos(a), zc + 0.335 * math.sin(a), 0.011, 0.011, 8)],
                     -0.002, 0.004, mat_bolt, mat_bolt, mat_bolt)
    U.slice_u(door, 0.05)
    for v in door.verts:
        v.co.y += zb
    U.wrap(door, th_d, D.RF)
    U.merge_into(bm, door)
    # cold-gas thruster ports near the nose, close to each seam
    zt = 11.55
    r = radius_at(outer, zt)
    nv = profile_normals(outer)[min(range(len(outer)), key=lambda i: abs(outer[i][1] - zt))]
    for th_edge, sgn in ((seam_lo, 1.0), (seam_hi, -1.0)):
        th = th_edge + sgn * 0.20 / r
        m = surface_frame(th, r, zt + zb, nv, lift=0.0)
        U.add_cyl(bm, 0.07, 0.07, 0.012, 20, m, mat=mat_seam)
        U.add_cyl(bm, 0.028, 0.040, 0.03, 16, m @ Matrix.Translation((0.0, 0.0, 0.012)), mat=mat_vent)


def build(colls, M, parent):
    c = colls["Fairing"]
    mats = [M["white"], M["nosecap"], M["seam"], M["light_grey"], M["bolt"], M["vent"]]
    halves = []
    g = GAP / D.RF
    for name, lo, hi in (("Fairing_Half_A", math.radians(90.0) + g, math.radians(270.0) - g),
                         ("Fairing_Half_B", math.radians(270.0) + g, math.radians(450.0) - g)):
        bm, outer = shell_half(lo, hi, D.Z_FAIRING_BASE)
        details(bm, outer, lo, hi, D.Z_FAIRING_BASE, 4, 5, 2)
        frame = Matrix.Translation((0.0, 0.0, D.Z_FAIRING_BASE))
        ob = U.finish(name, bm, c, mats, parent=parent, frame=frame, sharp=40.0)
        halves.append(ob)
    return halves
