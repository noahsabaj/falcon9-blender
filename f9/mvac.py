"""Merlin Vacuum (inside the interstage): niobium nozzle extension, turbine-exhaust
manifold, regeneratively cooled nozzle, chamber and turbomachinery."""
import math

import bmesh
from mathutils import Matrix

from . import dims as D
from . import util as U


def ext_r(z):
    """Niobium extension contour: bell-like, nearly straight at the exit."""
    t = (D.MVAC_Z_EXT_TOP - z) / (D.MVAC_Z_EXT_TOP - D.MVAC_Z_EXIT)     # 0 at joint, 1 at exit
    t = min(1.0, max(0.0, t))
    a = 0.25
    f = a * t + (1.0 - a) * (1.0 - (1.0 - t) ** 1.6)
    return D.MVAC_EXT_TOP_R + (D.MVAC_EXIT_R - D.MVAC_EXT_TOP_R) * f


def regen_r(z):
    """Regen section from the extension joint up to the throat."""
    t = (z - D.MVAC_Z_EXT_TOP) / (D.MVAC_Z_THROAT - D.MVAC_Z_EXT_TOP)  # 0 at joint, 1 at throat
    t = min(1.0, max(0.0, t))
    f = 1.0 - (1.0 - t) ** 1.8
    return D.MVAC_EXT_TOP_R + (0.13 - D.MVAC_EXT_TOP_R) * f


def build(colls, M, parent):
    c = colls["MVac"]
    z0, z1 = D.MVAC_Z_EXIT, D.MVAC_Z_EXT_TOP
    wall = 0.006

    # --- niobium extension (thin shell with exit stiffener lip) ----------------------
    bm = bmesh.new()
    n = 60
    zs = [z0 + (z1 - z0) * k / n for k in range(n + 1)]
    outer = [(ext_r(z0) + 0.025, z0), (ext_r(z0) + 0.025, z0 + 0.04), (ext_r(z0 + 0.06) + wall, z0 + 0.06)]
    outer += [(ext_r(z) + wall, z) for z in zs if z > z0 + 0.07]
    inner = [(ext_r(z), z) for z in reversed(zs)]
    rings, faces = U.revolve(bm, outer + inner, 192, closed=True)
    for f in faces:
        cen = f.calc_center_median()
        f.material_index = 0
    U.finish("MVac_NozzleExtension", bm, c, [M["niobium"]], parent=parent, sharp=50.0)

    # --- turbine exhaust manifold torus at the joint + regen nozzle -----------------
    bm = bmesh.new()
    U.add_torus(bm, 0.62, 0.08, 96, 16, Matrix.Translation((0.0, 0.0, z1 + 0.02)), mat=1)
    zt = D.MVAC_Z_THROAT
    zr = [z1 + (zt - z1) * k / 30 for k in range(31)]
    prof = [(regen_r(z) + 0.012, z) for z in zr]
    prof += [(0.15, zt + 0.07), (0.20, zt + 0.16), (0.20, zt + 0.45), (0.18, zt + 0.52), (0.10, zt + 0.57),
             (0.0, zt + 0.58)]
    U.revolve(bm, [(regen_r(z1) - 0.002, z1 - 0.01)] + prof, 96, mat=0)
    # inner wall of the regen section (seen from below)
    U.revolve(bm, [(regen_r(z), z) for z in reversed(zr)], 96, mat=2)
    # turbopump, gas generator and propellant lines (simplified)
    U.add_cyl(bm, 0.16, 0.16, 0.48, 24, Matrix.Translation((0.34, 0.0, zt + 0.32)) @
              Matrix.Rotation(math.pi / 2, 4, "X"), mat=1)
    U.add_cyl(bm, 0.07, 0.07, 0.36, 12, Matrix.Translation((0.30, 0.20, zt + 0.10)), mat=1)
    U.add_cyl(bm, 0.055, 0.055, 0.55, 12, Matrix.Translation((-0.26, 0.10, zt + 0.25)), mat=1)
    # turbine exhaust duct down to the manifold
    U.add_cyl(bm, 0.06, 0.06, 0.70, 12, Matrix.Translation((0.52, 0.0, z1 + 0.40)) @
              Matrix.Rotation(0.35, 4, "Y"), mat=1)
    # gimbal actuators
    for a in (0.0, 90.0):
        t = math.radians(a)
        m = Matrix.Translation((0.45 * math.cos(t + 2.4), 0.45 * math.sin(t + 2.4), zt + 0.35)) @ \
            Matrix.Rotation(0.5, 4, "X")
        U.add_cyl(bm, 0.035, 0.035, 0.6, 12, m, mat=1)
    U.finish("MVac_Engine", bm, c, [M["regen"], M["engine_dark"], M["nozzle_in"]], parent=parent, sharp=45.0)
