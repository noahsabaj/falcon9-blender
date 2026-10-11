"""Four titanium grid fins (stowed, hanging below their hinges) + actuator housings.

Each fin object's origin is on its hinge axis with local +Y tangential; rotating about
local Y by -90 deg swings it out to the deployed (horizontal) position.
"""
import math

import bmesh
from mathutils import Matrix, Vector

from . import dims as D
from . import util as U


def clip_line(p0, d, poly):
    """Clip infinite line p0 + t d against convex polygon (CCW, 2D). Returns (t0, t1) or None."""
    t0, t1 = -1e9, 1e9
    n = len(poly)
    for i in range(n):
        a, b = Vector(poly[i]), Vector(poly[(i + 1) % n])
        e = b - a
        nrm = Vector((e.y, -e.x))            # outward normal for CCW polygon
        denom = nrm.dot(d)
        num = nrm.dot(Vector(p0) - a)
        if abs(denom) < 1e-12:
            if num > 0:
                return None
            continue
        t = -num / denom
        if denom > 0:
            t1 = min(t1, t)
        else:
            t0 = max(t0, t)
    return (t0, t1) if t1 > t0 else None


def inset(poly, d):
    """Inset a convex CCW polygon by distance d."""
    n = len(poly)
    lines = []
    for i in range(n):
        a, b = Vector(poly[i]), Vector(poly[(i + 1) % n])
        e = (b - a).normalized()
        nrm = Vector((-e.y, e.x))            # inward normal
        lines.append((a + nrm * d, e))
    out = []
    for i in range(n):
        (p1, e1), (p2, e2) = lines[i - 1], lines[i]
        # intersect p1 + s e1 with p2 + t e2
        det = e1.x * (-e2.y) - e1.y * (-e2.x)
        r = p2 - p1
        s = (r.x * (-e2.y) - r.y * (-e2.x)) / det
        out.append(tuple(p1 + e1 * s))
    return out


def fin_bm():
    """Fin in a local 2D frame: (y tangential, z vertical) with hinge at (0, 0); depth along x."""
    bm = bmesh.new()
    hw = D.GF_WIDTH / 2
    z_root = -D.GF_ROOT
    z_bot = z_root - D.GF_LATTICE_LEN
    outer = [(-hw, z_bot), (hw, z_bot), (hw, z_root), (0.0, -0.03), (-hw, z_root)]   # CCW in (y, z)
    inner = inset(outer, D.GF_FRAME)
    depth = D.GF_DEPTH

    def box_between(a, b, thick, extra=0.0, mat=0):
        a, b = Vector(a), Vector(b)
        dvec = (b - a)
        L = dvec.length + extra
        ang = math.atan2(dvec.y, dvec.x)
        mid = (a + b) / 2
        # local: X = depth, Y along the segment, Z = thickness  (in fin frame x=depth, y, z)
        m = Matrix.Translation((0.0, mid.x, mid.y)) @ Matrix.Rotation(ang, 4, "X")
        U.add_box(bm, (depth, L, thick), m, mat=mat)

    # frame: members centred on the mid-line between outer and inner outlines
    n = len(outer)
    for i in range(n):
        a = (Vector(outer[i]) + Vector(inner[i])) / 2
        b = (Vector(outer[(i + 1) % n]) + Vector(inner[(i + 1) % n])) / 2
        box_between(a, b, D.GF_FRAME, extra=D.GF_FRAME, mat=0)
    # lattice blades at +-45 deg, clipped to the inner outline
    cy = 0.0
    cz = (z_bot + z_root) / 2
    step = D.GF_PITCH * math.sqrt(2.0)
    for sgn in (1.0, -1.0):
        d = Vector((1.0, sgn)).normalized()
        for k in range(-12, 13):
            p0 = (cy, cz + (k + 0.5) * step)
            seg = clip_line(p0, d, inner)
            if seg is None:
                continue
            t0, t1 = seg
            if t1 - t0 < 0.03:
                continue
            a = Vector(p0) + d * t0
            b = Vector(p0) + d * t1
            box_between(a, b, D.GF_BLADE, extra=D.GF_FRAME * 0.8, mat=0)
    # hinge hub (cylinder along tangential Y) at the gable apex
    U.add_cyl(bm, 0.075, 0.075, 0.34, 24, Matrix.Translation((0.0, 0.0, -0.03)) @
              Matrix.Rotation(math.pi / 2, 4, "X"), mat=1)
    # root web
    U.add_box(bm, (depth * 0.9, 0.05, D.GF_ROOT), Matrix.Translation((0.0, 0.0, z_root / 2 - 0.02)), mat=0)
    return bm


def build(colls, M, parent):
    c = colls["GridFins"]
    fins = []
    x_h = D.GF_R_IN + D.GF_DEPTH / 2
    for i, a in enumerate(D.AZ_GRIDFINS):
        th = math.radians(a)
        frame = U.rot_z(th) @ Matrix.Translation((x_h, 0.0, D.GF_Z_HINGE))
        bm = fin_bm()
        bm.transform(frame)
        ob = U.finish(f"GridFin_{i + 1}", bm, c, [M["titanium"], M["steel"]], parent=parent,
                      frame=frame, sharp=40.0)
        ob["hinge_axis"] = "local Y"
        fins.append(ob)

        # actuator housing on the interstage, above the hinge
        hb = bmesh.new()
        sec = [(-0.36, D.R - 0.02), (-0.36, D.R + 0.10), (-0.30, D.R + 0.17), (0.30, D.R + 0.17),
               (0.36, D.R + 0.10), (0.36, D.R - 0.02)]
        sections = [U.section_cyl(th, [(y * 0.7, D.R - 0.02) for y, _ in sec], 41.70, D.R),
                    U.section_cyl(th, sec, 41.86, D.R), U.section_cyl(th, sec, 42.40, D.R),
                    U.section_cyl(th, [(y * 0.8, x - (x - D.R) * 0.75) for y, x in sec], 42.62, D.R)]
        U.loft(hb, sections, mat=0)
        # radial shaft stub to the hub
        m = frame @ Matrix.Translation((-0.09, 0.0, -0.03)) @ Matrix.Rotation(math.pi / 2, 4, "Y")
        U.add_cyl(hb, 0.06, 0.06, 0.16, 20, m, mat=1)
        U.finish(f"GridFin_{i + 1}_Actuator", hb, colls["Interstage"], [M["black"], M["steel"]],
                 parent=parent, sharp=35.0)
    return fins
