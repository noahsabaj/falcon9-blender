"""Four carbon-fibre landing legs, stowed.

Planform is taken from the AXM Block 5 leg sheet (sheet mm, 1:100) and wrapped onto the
tank.  Each leg object has its origin on the hinge line through the two prong lugs with
local +Y along that line, so rotating about local Y deploys it (see rig.py).
"""
import math

import bmesh
from mathutils import Matrix, Vector

from . import dims as D
from . import util as U

R_OUT = D.LEG_R_IN + D.LEG_DEPTH
X_C = 78.2                      # sheet x of the leg centre line
Y_BOT, Y_TOP = 133.70, 46.33    # sheet y of hinge line / face top
K = (D.LEG_Z_FACE_TOP - D.LEG_Z_HINGE) / ((Y_BOT - Y_TOP) * 0.1)


def sheet(x, y):
    return ((x - X_C) * 0.1, D.LEG_Z_HINGE + (Y_BOT - y) * 0.1 * K)


def bez(p0, p1, p2, p3, n=10):
    out = []
    for i in range(1, n + 1):
        t = i / n
        mt = 1 - t
        out.append((mt ** 3 * p0[0] + 3 * mt * mt * t * p1[0] + 3 * mt * t * t * p2[0] + t ** 3 * p3[0],
                    mt ** 3 * p0[1] + 3 * mt * mt * t * p1[1] + 3 * mt * t * t * p2[1] + t ** 3 * p3[1]))
    return out


def face_outline():
    pts = [(81.845, 46.389), (92.114, 133.697), (90.842, 132.872), (80.744, 100.828)]
    pts += bez((80.744, 100.828), (78.046, 99.953), (75.820, 100.887), (75.820, 100.887), 8)
    pts += [(65.721, 132.931), (64.289, 133.671), (74.606, 46.328), (76.541, 51.291)]
    pts += bez((76.541, 51.291), (76.541, 51.291), (78.327, 52.196), (80.025, 51.291), 8)
    return [sheet(x, y) for x, y in pts]


def cap_outline():
    """Footpad cap outline in cap-sheet mm (top dome + tapered tail)."""
    pts = bez((74.167, 33.878), (74.167, 33.878), (74.483, 26.961), (78.161, 27.040), 12)
    pts += [(78.429, 27.040)]
    pts += bez((78.429, 27.040), (82.107, 26.961), (82.423, 33.878), (82.423, 33.878), 12)
    pts += [(79.878, 41.317)]
    pts += bez((79.878, 41.317), (78.444, 42.135), (76.797, 41.341), (76.797, 41.341), 8)
    return [(74.167, 33.878)] + pts


def half_width(outline, y):
    xs = []
    n = len(outline)
    for i in range(n):
        (x0, y0), (x1, y1) = outline[i], outline[(i + 1) % n]
        if (y0 - y) * (y1 - y) <= 0 and abs(y1 - y0) > 1e-9:
            xs.append(x0 + (x1 - x0) * (y - y0) / (y1 - y0))
    if len(xs) < 2:
        return None
    return min(xs), max(xs)


def to_world(theta, u, z, r):
    t = theta + u / R_OUT
    return Vector((r * math.cos(t), r * math.sin(t), z))


def build_leg(theta, M):
    """Return bmesh of one leg (world space) and its key points."""
    bm = bmesh.new()
    # --- body: wrapped prism -------------------------------------------------------
    U.flat_prism(bm, [face_outline()], -D.LEG_DEPTH, 0.0)
    U.slice_u(bm, 0.06)
    U.wrap(bm, theta, R_OUT, R_OUT)

    # --- raised face details -------------------------------------------------------
    det = bmesh.new()
    zf = lambda y: sheet(78.2, y)[1]
    spine = [(-0.157, zf(100.26)), (0.157, zf(100.26)), (0.157, zf(52.0)), (-0.157, zf(52.0))]
    U.flat_prism(det, [spine], -0.01, 0.008)
    for su in (-1, 1):
        U.flat_prism(det, [U.ellipse(su * 0.54, zf(93.45), 0.275, 0.275, 40)], -0.01, 0.016)
        U.flat_prism(det, [U.ellipse(su * 0.54, zf(93.45), 0.19, 0.19, 32)], 0.016, 0.03)
        for (cu, y, w, h) in ((0.67, 89.37, 0.41, 0.22), (0.41, 67.40, 0.41, 0.22), (0.90, 101.3, 0.25, 0.42),
                              (0.62, 77.35, 0.25, 0.42), (0.90, 113.3, 0.49, 0.27)):
            U.flat_prism(det, [U.rounded_rect(su * cu, zf(y), w, h, 0.06)], -0.01, 0.007)
    U.slice_u(det, 0.06)
    U.wrap(det, theta, R_OUT, R_OUT)
    U.merge_into(bm, det, mat=0)

    # --- inner A-frame panel filling the V cut-out ---------------------------------
    af = bmesh.new()
    apex_z = sheet(78.2, 100.2)[1] - 0.22
    base_z = D.LEG_Z_HINGE + 0.22
    sx = (2.0 * 1.0) / (35.594 - 16.617)              # scale A-frame sheet to ~2 m base
    sy = (apex_z - base_z) / (101.17 - 72.74)

    def afp(x, y):
        return ((x - 26.1) * sx * 1.0, apex_z - (y - 72.74) * sy)
    tri = [afp(16.617, 100.964), afp(25.130, 74.335)]
    tri += [afp(*p) for p in bez((25.130, 74.335), (25.130, 74.335), (26.080, 72.737), (27.271, 74.354), 6)]
    tri += [afp(35.594, 101.171), afp(34.506, 101.016)]
    tri += [afp(*p) for p in bez((34.506, 101.016), (25.334, 94.226), (17.595, 100.887), (17.595, 100.887), 10)]
    U.flat_prism(af, [tri], -D.LEG_DEPTH + 0.03, -0.09, 1, 1, 1)
    U.flat_prism(af, [U.ellipse(0.0, afp(26.17, 89.36)[1], 0.16, 0.19, 32)], -0.09, -0.07, 3, 3, 3)
    U.slice_u(af, 0.06)
    U.wrap(af, theta, R_OUT, R_OUT)
    U.merge_into(bm, af)

    # --- footpad cap --------------------------------------------------------------
    co = cap_outline()
    y_top, y_bot = 27.04, 41.73
    cap_y_bot_in_leg = 51.97                       # cap tail tip sits in the face notch
    ofs = cap_y_bot_in_leg - y_bot
    sections = []
    ys = [y_top + 0.06] + [y_top + (y_bot - 0.05 - y_top) * (k / 30) ** 1.0 for k in range(1, 31)]
    nu = 18
    for y in ys:
        hw = half_width(co, y)
        if hw is None:
            continue
        x0, x1 = hw
        hwid = max(0.02, (x1 - x0) * 0.05)
        cu = ((x0 + x1) / 2 - 78.295) * 0.1
        z = sheet(78.2, y + ofs)[1]
        # dome height: highest around the footpad disc
        tz = (y - y_top) / (y_bot - y_top)
        h_c = D.LEG_DEPTH + 0.13 * math.sin(math.pi * min(1.0, 0.25 + tz * 0.9)) + 0.02
        h_e = D.LEG_DEPTH * 0.55
        sec = []
        for k in range(nu + 1):
            s = -1.0 + 2.0 * k / nu
            u = cu + s * hwid
            h = h_e + (h_c - h_e) * (1.0 - s * s) ** 0.6
            sec.append(to_world(theta, u, z, D.LEG_R_IN + h))
        sec.append(to_world(theta, cu + hwid, z, D.LEG_R_IN + 0.004))
        sec.append(to_world(theta, cu - hwid, z, D.LEG_R_IN + 0.004))
        sections.append(sec)
    cap = bmesh.new()
    U.loft(cap, sections, mat=0)
    U.merge_into(bm, cap)
    # footpad disc + bolts on the dome
    yd = 34.038
    zd = sheet(78.2, yd + ofs)[1]
    td = (yd - y_top) / (y_bot - y_top)
    hd = D.LEG_DEPTH + 0.13 * math.sin(math.pi * min(1.0, 0.25 + td * 0.9)) + 0.02
    pd = to_world(theta, 0.0, zd, D.LEG_R_IN + hd - 0.01)
    radial = Vector((math.cos(theta), math.sin(theta), 0.0))
    rot = radial.to_track_quat("Z", "Y").to_matrix().to_4x4()
    U.add_cyl(bm, 0.227, 0.215, 0.05, 40, Matrix.Translation(pd) @ rot, mat=2)
    for (bx, by) in ((77.355, 34.791), (77.345, 33.203), (79.210, 34.822), (79.200, 33.234)):
        u = (bx - 78.335) * 0.1
        z = sheet(78.2, by + ofs)[1]
        p = to_world(theta, u, z, D.LEG_R_IN + hd + 0.015)
        U.add_cyl(bm, 0.018, 0.014, 0.02, 8, Matrix.Translation(p) @ rot, mat=3)
    # side ears
    for pts in ([(76.421, 27.594)] + bez((76.421, 27.594), (76.421, 27.594), (73.660, 29.670), (71.593, 32.934), 8)
                + [(74.072, 33.832)] + bez((74.072, 33.832), (74.072, 33.832), (75.034, 28.797), (76.421, 27.594), 8),):
        for mirror in (1, -1):
            ear = bmesh.new()
            loop = [((x - 78.295) * 0.1 * mirror, sheet(78.2, y + ofs)[1]) for x, y in pts]
            U.flat_prism(ear, [loop], -D.LEG_DEPTH + 0.02, -0.05)
            U.slice_u(ear, 0.05)
            U.wrap(ear, theta, R_OUT, R_OUT)
            U.merge_into(bm, ear, mat=0)

    # --- prong lugs at the hinge line ----------------------------------------------
    hinge_pts = []
    for su in (-1, 1):
        u = su * 1.328
        p = to_world(theta, u, D.LEG_Z_HINGE + 0.06, D.LEG_R_IN + D.LEG_DEPTH / 2)
        hinge_pts.append(p)
    return bm, hinge_pts


def hinge_frame(theta, hinge_pts):
    a, b = hinge_pts
    mid = (a + b) / 2
    y = (b - a).normalized()
    z = Vector((0.0, 0.0, 1.0))
    x = y.cross(z).normalized()
    radial = Vector((math.cos(theta), math.sin(theta), 0.0))
    if x.dot(radial) < 0:
        x = -x
        y = -y
    m = Matrix((x, y, z)).transposed().to_4x4()
    m.translation = mid
    return m


def build(colls, M, parent):
    c = colls["Legs"]
    legs = []
    for i, a in enumerate(D.AZ_LEGS):
        th = math.radians(a)
        bm, hp = build_leg(th, M)
        frame = hinge_frame(th, hp)
        ob = U.finish(f"Leg_{i + 1}", bm, c, [M["carbon"], M["grey"], M["alu"], M["bolt"]],
                      parent=parent, frame=frame, sharp=40.0)
        ob["hinge_axis"] = "local Y"
        legs.append(ob)

        # body-side clevis fittings on the skirt (stay with the stage)
        cb = bmesh.new()
        for p in hp:
            t = math.atan2(p.y, p.x)
            m = frame.copy()
            m.translation = p
            U.add_box(cb, (0.30, 0.20, 0.30), Matrix.Translation(Vector((math.cos(t), math.sin(t), 0)) * -0.06) @ m)
            U.add_cyl(cb, 0.055, 0.055, 0.34, 16, m @ Matrix.Rotation(math.pi / 2, 4, "X"), mat=1)
        U.finish(f"Leg_{i + 1}_HingeFittings", cb, c, [M["black"], M["steel"]], parent=parent, sharp=40.0)
    return legs
