"""First stage: tank barrel, aft skirt, base heat shield, raceway, small fittings."""
import math

import bmesh

from . import dims as D
from . import util as U

SEGS = 256


def raceway_section(base_w=0.20, top_w=0.16, h=0.085, sink=0.012):
    R = D.R
    hw, tw = base_w / 2, top_w / 2
    return [(-hw, R - sink), (-hw, R + 0.02), (-tw - 0.012, R + h - 0.008), (-tw, R + h),
            (tw, R + h), (tw + 0.012, R + h - 0.008), (hw, R + 0.02), (hw, R - sink)]


def build_raceway(coll, M, parent, z0, z1, name, mat, theta=0.0, base_w=0.20, top_w=0.16, h=0.085,
                  band_step=1.25, end_slope_bottom=0.35, end_slope_top=0.45):
    """Raceway cover with sloped end ramps and clamp bands."""
    bm = bmesh.new()
    sec = raceway_section(base_w, top_w, h)
    flat = [(y, D.R - 0.012) for y, _ in sec]           # collapsed section for ramp ends
    sections = [U.section_at(theta, flat, z0 - end_slope_bottom),
                U.section_at(theta, sec, z0),
                U.section_at(theta, sec, z1),
                U.section_at(theta, flat, z1 + end_slope_top)]
    U.loft(bm, sections, mat=0)
    # clamp bands
    z = z0 + band_step * 0.5
    while z < z1 - 0.2:
        m = U.frame_at(theta, D.R + h / 2 + 0.002, z)
        U.add_box(bm, (h + 0.03, base_w + 0.016, 0.035), m, mat=0)
        z += band_step
    return U.finish(name, bm, coll, [mat], parent=parent, sharp=30.0)


def build(colls, M, parent):
    c = colls["Stage1"]
    R = D.R
    objs = {}

    # --- tank barrel (white) ---------------------------------------------------
    bm = bmesh.new()
    U.revolve(bm, [(R, D.Z_SKIRT_TOP), (R, 41.45)], SEGS)
    objs["tank"] = U.finish("S1_Tank", bm, c, [M["white"]], parent=parent)

    # --- aft skirt / octaweb fairing (black) -----------------------------------
    bm = bmesh.new()
    prof = [(R - 0.03, D.Z_HEATSHIELD + 0.005), (R + 0.018, D.Z_HEATSHIELD + 0.005),
            (R + 0.018, D.Z_HEATSHIELD + 0.07), (R + 0.004, D.Z_HEATSHIELD + 0.09),
            (R + 0.004, D.Z_SKIRT_TOP - 0.02), (R, D.Z_SKIRT_TOP)]
    U.revolve(bm, prof, SEGS)
    objs["skirt"] = U.finish("S1_AftSkirt", bm, c, [M["black"]], parent=parent, sharp=40.0)

    # bolt ring on the skirt's lower flange
    bm = bmesh.new()
    n = 96
    for k in range(n):
        t = U.TAU * (k + 0.5) / n
        m = U.frame_at(t, R + 0.018, D.Z_HEATSHIELD + 0.04) @ U.Matrix.Rotation(math.pi / 2, 4, "Y")
        U.add_cyl(bm, 0.011, 0.009, 0.012, 6, m)
    U.finish("S1_AftSkirt_Bolts", bm, c, [M["bolt"]], parent=parent, sharp=40.0)

    # --- base heat shield: disc with nine engine openings ------------------------
    bm = bmesh.new()
    loops = [U.ellipse(0.0, 0.0, R - 0.005, R - 0.005, n=SEGS)]
    holes = [(0.0, 0.0)] + [(D.M1D_RING_R * math.cos(math.radians(a)), D.M1D_RING_R * math.sin(math.radians(a)))
                            for a in D.M1D_RING_AZ]
    for hx, hy in holes:
        loops.append(U.ellipse(hx, hy, 0.335, 0.335, n=48))
    U.flat_prism(bm, loops, D.Z_HEATSHIELD, D.Z_HEATSHIELD + 0.04)
    objs["heatshield"] = U.finish("S1_HeatShield", bm, c, [M["heatshield"]], parent=parent, sharp=30.0)

    # --- raceway (black on S1, continues 1.5 m onto the interstage) ---------------
    objs["raceway"] = build_raceway(c, M, parent, D.Z_SKIRT_TOP + 0.05, 43.11, "S1_Raceway", M["raceway"],
                                    base_w=0.20, top_w=0.16, h=0.085)

    # --- grid fin rest pads (dark ovals under each stowed fin) -------------------
    bm = bmesh.new()
    for a in D.AZ_GRIDFINS:
        U.flat_prism(bm, [U.ellipse(0.0, 40.815, 0.175, 0.285, n=40)], -0.01, 0.065)
        verts = [v for v in bm.verts if not v.tag]
        U.wrap(bm, math.radians(a), D.R, verts=verts)
        for v in verts:
            v.tag = True
    U.finish("S1_GridFin_RestPads", bm, c, [M["pad"]], parent=parent, sharp=40.0)

    # --- aft skirt fittings (AXM lower-stage sheet) ---------------------------------
    # hold-down / T-E interface fittings between the legs, umbilical (QD) fairings at
    # -24 / 156 deg; laid out in the unrolled (u, z, w) frame and wrapped onto the skirt
    hs = D.Z_HEATSHIELD
    sk, grey, white = bmesh.new(), bmesh.new(), bmesh.new()
    for a in (0.0, 90.0, 180.0, 270.0):
        part, gpart, wpart = bmesh.new(), bmesh.new(), bmesh.new()
        U.flat_prism(part, [[(-0.415, hs + 0.10), (0.415, hs + 0.10), (0.25, hs + 0.52), (-0.25, hs + 0.52)]],
                     0.0, 0.014)
        U.flat_prism(part, [U.rounded_rect(0.0, hs + 0.72, 0.085, 0.42, 0.02)], 0.0, 0.04)
        U.flat_prism(gpart, [U.rounded_rect(0.0, hs + 0.17, 0.26, 0.15, 0.015)], 0.0, 0.10)
        U.flat_prism(wpart, [U.rounded_rect(0.0, hs + 0.40, 0.165, 0.10, 0.01)], 0.014, 0.018)
        if a != 0.0:
            U.flat_prism(gpart, [U.rounded_rect(0.0, hs + 0.86, 0.315, 0.30, 0.02)], 0.0, 0.035)
        for b, dst in ((part, sk), (gpart, grey), (wpart, white)):
            U.slice_u(b, 0.05)
            U.wrap(b, math.radians(a), D.R + 0.004)
            U.merge_into(dst, b)
    for a in (-24.0, 156.0):
        part = bmesh.new()
        bottle = [(-0.365, hs + 0.08), (0.365, hs + 0.08), (0.30, hs + 0.55), (0.17, hs + 0.80),
                  (0.17, hs + 0.93), (-0.17, hs + 0.93), (-0.17, hs + 0.80), (-0.30, hs + 0.55)]
        U.flat_prism(part, [bottle], 0.0, 0.06)
        U.flat_prism(part, [U.rounded_rect(0.0, hs + 0.40, 0.30, 0.30, 0.03)], 0.06, 0.075)
        U.slice_u(part, 0.05)
        U.wrap(part, math.radians(a), D.R + 0.004)
        U.merge_into(sk, part)
    U.finish("S1_AftSkirt_Fittings", sk, c, [M["black"]], parent=parent, sharp=35.0)
    U.finish("S1_HoldDown_Blocks", grey, c, [M["grey"]], parent=parent, sharp=35.0)
    U.finish("S1_AftSkirt_Markings", white, c, [M["white"]], parent=parent, sharp=35.0)

    # --- bolted hoods on the tank receiving each stowed leg's footpad cap ----------
    bm = bmesh.new()
    zc = 11.0
    for a in D.AZ_LEGS:
        part = bmesh.new()
        n = 24
        outer = [(0.50 * math.cos(math.pi * k / n), zc + 0.47 * math.sin(math.pi * k / n)) for k in range(n + 1)]
        inner = [(0.39 * math.cos(math.pi * k / n), zc + 0.355 * math.sin(math.pi * k / n)) for k in range(n + 1)]
        U.flat_prism(part, [outer + list(reversed(inner))], -0.005, 0.025, 0, 0, 0)
        for k in range(9):
            t = math.pi * (k + 0.5) / 9
            U.flat_prism(part, [U.ellipse(0.445 * math.cos(t), zc + 0.41 * math.sin(t), 0.016, 0.016, 8)],
                         0.025, 0.032, 1, 1, 1)
        U.slice_u(part, 0.04)
        U.wrap(part, math.radians(a), D.R)
        U.merge_into(bm, part)
    U.finish("S1_LegHoods", bm, c, [M["black"], M["light_grey"]], parent=parent, sharp=35.0)

    # --- small grey patches flanking the leg azimuths and a ring of teardrop ports,
    # around the E / X of SPACEX (AXM sheet positions, sheet mm -> u, z) -------------
    x_ref, y_ref = 145.235, 19.11
    def uz(x, y):
        return ((x - x_ref) * 0.1, D.Z_S1_TOP - (y - y_ref) * 0.1)
    bm = bmesh.new()
    pairs_b = (94.24, 107.95, 123.08, 135.64, 150.26, 164.58, 179.71, 192.27)
    pairs_c = (96.09, 106.10, 124.93, 133.79, 152.11, 162.73, 181.56, 190.42)
    for xs, ys in ((pairs_b, (205.92, 221.09)), (pairs_c, (210.12, 225.29))):
        for x0 in xs:
            for y0 in ys:
                (u0, z0), (u1, z1) = uz(x0, y0), uz(x0 + 1.85, y0 + 2.79)
                U.flat_prism(bm, [U.rounded_rect((u0 + u1) / 2, (z0 + z1) / 2, abs(u1 - u0), abs(z1 - z0), 0.02)],
                             -0.001, 0.0018)
    for x0 in (91.21, 103.14, 114.85, 126.78, 138.50, 149.30, 161.23, 172.95, 184.88, 196.59):
        (u0, z0), (u1, z1) = uz(x0, 201.46), uz(x0 + 1.41, 203.30)
        cu, cz, hw, hh = (u0 + u1) / 2, (z0 + z1) / 2, abs(u1 - u0) / 2, abs(z1 - z0) / 2
        drop = [(cu + hw * math.cos(a) * (1.0 if math.sin(a) < 0 else 1.0 - 0.6 * math.sin(a)),
                 cz + hh * math.sin(a)) for a in (U.TAU * k / 24 for k in range(24))]
        U.flat_prism(bm, [drop], -0.001, 0.0018)
    U.slice_u(bm, 0.04)
    U.wrap(bm, 0.0, D.R)
    U.finish("S1_TankPatches", bm, c, [M["grey"]], parent=parent, sharp=40.0)

    # --- round service port (concentric rings) at +102 deg ----------------------------
    bm = bmesh.new()
    th = math.radians(102.0)
    for (rx, ry, w0, w1, mat) in ((0.445, 0.33, 0.0, 0.006, 0), (0.38, 0.28, 0.006, 0.010, 1),
                                  (0.30, 0.22, 0.010, 0.016, 0)):
        U.flat_prism(bm, [U.ellipse(0.0, 40.66, rx, ry, n=56)], w0, w1, mat, mat, mat)
    U.wrap(bm, th, D.R)
    U.finish("S1_ServicePort", bm, c, [M["light_grey"], M["grey"]], parent=parent, sharp=40.0)
    return objs
