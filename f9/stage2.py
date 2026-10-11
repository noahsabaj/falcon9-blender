"""Second stage exterior: white tank barrel above the interstage and its raceway."""
import math

import bmesh
from mathutils import Matrix

from . import dims as D
from . import util as U
from .stage1 import build_raceway

SEGS = 256


def build(colls, M, parent):
    c = colls["Stage2"]
    R = D.R
    objs = {}

    # barrel: starts inside the interstage (aft skirt) and runs to the fairing base
    bm = bmesh.new()
    U.revolve(bm, [(R - 0.02, 47.55), (R - 0.02, D.Z_IS_TOP - 0.02), (R, D.Z_IS_TOP),
                   (R, D.Z_FAIRING_BASE)], SEGS)
    objs["tank"] = U.finish("S2_Tank", bm, c, [M["white"]], parent=parent, sharp=40.0)

    # raceway (white) from the latch cap at 0 deg up to just under the fairing
    objs["raceway"] = build_raceway(c, M, parent, 48.56, 56.55, "S2_Raceway", M["white"],
                                    theta=math.radians(D.AZ_RACEWAY), base_w=0.18, top_w=0.14, h=0.07,
                                    band_step=1.0, end_slope_bottom=0.03, end_slope_top=0.22)

    # umbilical plate (T/E side, 180 deg) just above the interstage
    bm = bmesh.new()
    th = math.radians(180.0)
    U.flat_prism(bm, [U.rounded_rect(0.0, 48.62, 0.46, 0.36, 0.05)], -0.004, 0.012, 0, 0, 0)
    U.flat_prism(bm, [U.rounded_rect(0.0, 48.62, 0.34, 0.24, 0.04)], 0.012, 0.020, 1, 1, 1)
    for (du, dz) in ((-0.2, 0.15), (0.2, 0.15), (-0.2, -0.15), (0.2, -0.15)):
        U.flat_prism(bm, [U.ellipse(du, 48.62 + dz, 0.012, 0.012, 8)], 0.012, 0.022, 2, 2, 2)
    U.slice_u(bm, 0.05)
    U.wrap(bm, th, R)
    U.finish("S2_UmbilicalPlate", bm, c, [M["light_grey"], M["grey"], M["bolt"]], parent=parent, sharp=40.0)
    return objs
