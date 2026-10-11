"""Plausible internal structure (hidden by the skins; for cutaways): tank domes,
LOX downcomer, octaweb thrust structure, COPVs, second-stage thrust cone, PAF 1575."""
import math

import bmesh
from mathutils import Matrix

from . import dims as D
from . import util as U

SEGS = 96


def dome(bm, z_eq, depth, up=True, r=None, n=16, mat=0):
    """Elliptical tank dome (single surface) with equator at z_eq."""
    r = (D.R - 0.012) if r is None else r
    prof = []
    for k in range(n + 1):
        phi = (math.pi / 2) * k / n
        prof.append((r * math.cos(phi), z_eq + (depth if up else -depth) * math.sin(phi)))
    prof[-1] = (0.0, prof[-1][1])
    if not up:
        prof.reverse()
    U.revolve(bm, prof, SEGS, mat=mat)


def build(colls, M, parent_s1, parent_s2):
    c = colls["Internals"]
    mats = [M["alu"], M["grey"], M["steel"], M["carbon"]]

    # --- first stage --------------------------------------------------------------
    bm = bmesh.new()
    dome(bm, 40.20, 1.05, up=True)                 # LOX forward dome
    dome(bm, 16.00, 1.00, up=False)                # common bulkhead (LOX above / RP-1 below)
    dome(bm, 3.20, 1.05, up=False)                 # RP-1 aft dome
    # LOX downcomer through the RP-1 tank to the centre engine
    U.add_cyl(bm, 0.25, 0.25, 15.0 - 2.15, 32, Matrix.Translation((0.0, 0.0, (15.0 + 2.15) / 2)), mat=2)
    # COPVs (helium) near the top of the LOX tank
    for k in range(3):
        t = math.radians(60.0 + 120.0 * k)
        m = Matrix.Translation((1.05 * math.cos(t), 1.05 * math.sin(t), 38.4))
        U.add_cyl(bm, 0.33, 0.33, 0.9, 24, m, mat=3)
        for s in (-1, 1):
            sub = bmesh.new()
            prof = [(0.0, -0.68), (0.23, -0.62), (0.33, -0.45)] if s < 0 else [(0.33, 0.45), (0.23, 0.62), (0.0, 0.68)]
            U.revolve(sub, prof, 24, mat=3)
            sub.transform(m)
            U.merge_into(bm, sub, mat=3)
    # octaweb: centre ring, radial beams and outer ring between the engines
    U.add_cyl(bm, 0.42, 0.42, 0.85, 32, Matrix.Translation((0.0, 0.0, 1.62)), mat=1, caps=False)
    U.add_cyl(bm, 1.62, 1.62, 0.85, 64, Matrix.Translation((0.0, 0.0, 1.62)), mat=1, caps=False)
    for k in range(8):
        t = math.radians(45.0 * k)
        m = U.frame_at(t, 1.02, 1.62)
        U.add_box(bm, (1.2, 0.08, 0.85), m, mat=1)
    for a in D.M1D_RING_AZ:
        t = math.radians(a)
        m = Matrix.Translation((D.M1D_RING_R * math.cos(t), D.M1D_RING_R * math.sin(t), 2.05))
        U.add_cyl(bm, 0.26, 0.26, 0.06, 24, m, mat=1)
    U.finish("S1_Internals", bm, c, mats, parent=parent_s1, sharp=40.0)

    # stage-separation centre pusher on top of the S1 forward dome
    bm = bmesh.new()
    U.add_cyl(bm, 0.16, 0.16, 0.9, 24, Matrix.Translation((0.0, 0.0, 41.25 + 0.45)), mat=2)
    U.add_cyl(bm, 0.30, 0.30, 0.05, 24, Matrix.Translation((0.0, 0.0, 41.25 + 0.92)), mat=1)
    U.finish("Interstage_CentrePusher", bm, c, mats, parent=parent_s1, sharp=40.0)

    # --- second stage ---------------------------------------------------------------
    bm = bmesh.new()
    dome(bm, 49.60, 1.15, up=False)                # aft dome
    dome(bm, 53.10, 0.95, up=False)                # common bulkhead
    dome(bm, 55.60, 1.00, up=True)                 # forward dome
    # MVac thrust cone from the aft skirt to the gimbal mount
    U.revolve(bm, [(1.80, 48.40), (0.30, 47.98)], SEGS, mat=1)
    U.revolve(bm, [(0.30, 47.98), (1.80, 48.40)], SEGS, mat=1)
    U.add_cyl(bm, 0.30, 0.30, 0.06, 32, Matrix.Translation((0.0, 0.0, 47.98)), mat=2)
    # helium COPVs in the aft skirt
    for k in range(2):
        t = math.radians(90.0 + 180.0 * k)
        sub = bmesh.new()
        prof = [(0.0, -0.38)] + [(0.38 * math.sin(math.pi * j / 12), -0.38 * math.cos(math.pi * j / 12))
                                 for j in range(1, 12)] + [(0.0, 0.38)]
        U.revolve(sub, prof, 24, mat=3)
        sub.transform(Matrix.Translation((1.25 * math.cos(t), 1.25 * math.sin(t), 48.75)))
        U.merge_into(bm, sub, mat=3)
    U.finish("S2_Internals", bm, c, mats, parent=parent_s2, sharp=40.0)

    # PAF 1575 payload attach fitting on the S2 forward skirt
    bm = bmesh.new()
    zf = D.Z_FAIRING_BASE
    prof = [(D.R - 0.04, zf - 0.02), (D.R - 0.04, zf + 0.06), (0.86, zf + 0.72), (0.86, zf + 0.80),
            (0.78, zf + 0.80), (0.78, zf + 0.74), (D.R - 0.10, zf + 0.06), (D.R - 0.10, zf - 0.02)]
    U.revolve(bm, prof, SEGS, closed=True, mat=0)
    U.add_torus(bm, 0.84, 0.035, 64, 10, Matrix.Translation((0.0, 0.0, zf + 0.82)), mat=2)
    U.finish("PAF_1575", bm, c, mats, parent=parent_s2, sharp=40.0)
