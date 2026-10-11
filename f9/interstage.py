"""Black composite interstage: shell, S1 joint bolt pads, latch fairings, GN2 thruster
pods, grid-fin access ports."""
import math

import bmesh
from mathutils import Matrix

from . import dims as D
from . import util as U

SEGS = 256


def build(colls, M, parent, parent_s2=None):
    c = colls["Interstage"]
    R = D.R
    z0, z1 = 41.45, D.Z_IS_TOP

    # --- shell (includes the 0.13 m dark joint band at the top of S1) ---------------
    bm = bmesh.new()
    prof = [(R, z0), (R, z1 - 0.05), (R + 0.006, z1 - 0.04), (R + 0.006, z1), (R - 0.03, z1),
            (R - 0.03, z0)]
    U.revolve(bm, prof, SEGS, closed=True)
    U.finish("Interstage_Shell", bm, c, [M["black"]], parent=parent, sharp=40.0)

    # --- S1/interstage joint: white bolt pads with three bolts each ----------------
    pads, bolts = bmesh.new(), bmesh.new()
    npad = 30
    for k in range(npad):
        u0 = (k + 0.5) * (U.TAU * R / npad)
        U.flat_prism(pads, [U.rounded_rect(u0, 41.497, 0.374, 0.091, 0.02)], -0.004, 0.006)
        # AXM pattern: washered bolts at both ends, three small bolts between
        for du, rr in ((-0.151, 0.032), (0.151, 0.032), (-0.075, 0.0185), (0.0, 0.0185), (0.072, 0.0185)):
            t = (u0 + du) / R
            m = U.frame_at(t, R + 0.0075, 41.497) @ Matrix.Rotation(math.pi / 2, 4, "Y")
            U.add_cyl(bolts, rr, rr * 0.85, 0.005, 10, m)
    U.slice_u(pads, 0.06)
    U.wrap(pads, 0.0, R)
    U.finish("Interstage_JointPads", pads, c, [M["white"]], parent=parent, sharp=40.0)
    U.finish("Interstage_JointBolts", bolts, c, [M["bolt"]], parent=parent, sharp=40.0)

    # --- separation latch fairings (3) straddling the stage joint -------------------
    # black lower part on the interstage, white rounded cap on the second stage
    lat, caps = bmesh.new(), bmesh.new()
    for a in D.AZ_LATCHES:
        th = math.radians(a)
        U.blister(lat, th, z1 + 0.004, z1 - 1.76, 0.26, 0.14, R, round_bot=0.26, nz=30)
        U.blister(caps, th, z1 + 0.415, z1 + 0.006, 0.265, 0.141, R, round_top=0.40, nz=16)
        # latch access panel outline (raised strip) on the blister crown
        for dz in (0.35, 1.05):
            m = U.frame_at(th, R + 0.142, z1 - dz)
            U.add_box(lat, (0.012, 0.16, 0.03), m, mat=1)
    U.finish("Interstage_LatchFairings", lat, c, [M["black"], M["grey"]], parent=parent, sharp=35.0)
    U.finish("S2_LatchCaps", caps, colls["Stage2"], [M["white"]], parent=parent_s2 or parent, sharp=35.0)

    # --- GN2 cold-gas thruster pods (2) ---------------------------------------------
    pod = bmesh.new()
    for a in D.AZ_GN2_PODS:
        th = math.radians(a)
        sec = [(-0.315, R - 0.02), (-0.315, R + 0.16), (-0.25, R + 0.27), (0.25, R + 0.27),
               (0.315, R + 0.16), (0.315, R - 0.02)]

        def scaled(fu, fr):
            return [(u * fu, R + (r - R) * fr) for u, r in sec]
        sections = [scaled(0.60, 0.05), scaled(0.85, 0.60), scaled(1.0, 1.0), scaled(1.0, 1.0),
                    scaled(0.70, 0.50), scaled(0.40, 0.05)]
        zs = [41.74, 41.82, 41.95, 42.28, 42.40, 42.46]
        U.loft(pod, [U.section_cyl(th, s, z, R) for s, z in zip(sections, zs)], mat=0)
        # roll nozzles on both sides, pitch/yaw nozzle on the face
        for side in (-1, 1):
            for dz in (-0.07, 0.07):
                m = U.frame_at(th + side * 0.33 / R, R + 0.15, 42.12 + dz) @ \
                    Matrix.Rotation(-side * math.pi / 2, 4, "X")
                U.add_cyl(pod, 0.032, 0.048, 0.07, 16, m, mat=1)
        m = U.frame_at(th, R + 0.29, 42.12) @ Matrix.Rotation(math.pi / 2, 4, "Y")
        U.add_cyl(pod, 0.032, 0.05, 0.06, 16, m, mat=1)
    U.finish("Interstage_GN2_Thrusters", pod, c, [M["black"], M["steel"]], parent=parent, sharp=35.0)

    # --- access ports above each grid fin (white ring markings) ----------------------
    ports = bmesh.new()
    for a in D.AZ_GRIDFINS:
        U.flat_prism(ports, [U.ellipse(0.0, 42.80, 0.125, 0.125, 40), U.ellipse(0.0, 42.80, 0.098, 0.098, 40)],
                     0.0, 0.004, 0, 0, 0)
        U.flat_prism(ports, [U.ellipse(0.0, 42.80, 0.098, 0.098, 40)], -0.005, 0.002, 1, 1, 1)
        pv = [v for v in ports.verts if not v.tag]
        U.wrap(ports, math.radians(a), R, verts=pv)
        for v in pv:
            v.tag = True
    U.finish("Interstage_Ports", ports, c, [M["white"], M["grey"]], parent=parent, sharp=40.0)
