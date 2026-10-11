"""Nine Merlin 1D sea-level engines (one shared mesh, nine linked objects)."""
import math

import bmesh
from mathutils import Matrix, Vector

from . import dims as D
from . import util as U

SEGS = 96


def bell_r(z):
    """Inner contour radius of the Merlin 1D bell at height z (exit at z=0)."""
    s = min(1.0, max(0.0, (D.M1D_Z_THROAT - z) / D.M1D_Z_THROAT))
    f = 0.3 * s + 0.7 * (1.0 - (1.0 - s) ** 2)
    return D.M1D_THROAT_R + (D.M1D_EXIT_R - D.M1D_THROAT_R) * f


def nozzle_profile(wall=0.007, lip=0.022, lip_h=0.055, z_top=None):
    z_top = D.M1D_Z_THROAT if z_top is None else z_top
    hoops = (0.33, 0.62)
    outer = [(bell_r(0.0), 0.0), (bell_r(0.0) + lip, 0.0), (bell_r(0.0) + lip, lip_h * 0.75),
             (bell_r(lip_h) + lip * 0.55, lip_h), (bell_r(lip_h + 0.02) + wall, lip_h + 0.02)]
    zs = [lip_h + 0.02 + (z_top - lip_h - 0.02) * k / 40 for k in range(1, 41)]
    for z in zs:
        r = bell_r(z) + wall
        for hz in hoops:
            if abs(z - hz) < 0.018:
                r += 0.011 * math.cos((z - hz) / 0.018 * math.pi / 2)
        outer.append((r, z))
    inner = [(bell_r(z), z) for z in reversed([z_top * k / 48 for k in range(1, 49)])]
    return outer + inner


def build(colls, M, parent):
    c = colls["Engines"]
    bm = bmesh.new()
    # 0 nozzle outer, 1 nozzle inner, 2 boot, 3 dark engine parts
    rings, faces = U.revolve(bm, nozzle_profile(), SEGS, closed=True)
    for f in faces:
        cen = f.calc_center_median()
        radial = Vector((cen.x, cen.y, 0.0)).normalized()
        f.material_index = 1 if f.normal.dot(radial) < -0.2 and cen.z > 0.004 else 0

    # throat / chamber / injector (inside the octaweb, visible in cutaways)
    zt = D.M1D_Z_THROAT
    chamber = [(D.M1D_THROAT_R + 0.007, zt), (0.15, zt + 0.08), (0.19, zt + 0.2), (0.19, zt + 0.52),
               (0.17, zt + 0.6), (0.11, zt + 0.66), (0.0, zt + 0.68)]
    U.revolve(bm, chamber, 48, mat=3)
    # turbopump + gas generator lines (simplified)
    U.add_cyl(bm, 0.15, 0.15, 0.5, 24, Matrix.Translation((0.26, 0.0, zt + 0.82)) @
              Matrix.Rotation(math.pi / 2, 4, "X"), mat=3)
    U.add_cyl(bm, 0.05, 0.05, 0.42, 12, Matrix.Translation((0.2, 0.18, zt + 0.5)), mat=3)
    U.add_cyl(bm, 0.05, 0.05, 0.42, 12, Matrix.Translation((0.2, -0.18, zt + 0.5)), mat=3)
    U.add_cyl(bm, 0.07, 0.07, 0.25, 12, Matrix.Translation((0.0, 0.0, zt + 0.78)), mat=3)

    # flexible heat-shield boot (bellows) from the shield opening down to the bell
    z_hs = D.Z_HEATSHIELD - D.Z_NOZZLE_EXIT
    zb = 0.90
    boot = []
    n = 24
    for k in range(n + 1):
        t = k / n
        z = zb + (z_hs + 0.02 - zb) * t
        r0 = bell_r(zb) + 0.012
        r = r0 + (0.335 - r0) * (t ** 0.85) + 0.010 * math.sin(t * math.pi * 7.0) * (0.3 + 0.7 * t)
        boot.append((r, z))
    U.revolve(bm, boot, 72, mat=2)

    mats = [M["nozzle"], M["nozzle_in"], M["boot"], M["engine_dark"]]
    proto = U.finish("Merlin1D_Center", bm, c, mats, parent=parent, sharp=50.0)
    proto.location = (0.0, 0.0, D.Z_NOZZLE_EXIT)
    engines = [proto]
    for k, a in enumerate(D.M1D_RING_AZ):
        t = math.radians(a)
        mtx = Matrix.Translation((D.M1D_RING_R * math.cos(t), D.M1D_RING_R * math.sin(t), D.Z_NOZZLE_EXIT)) @ \
            U.rot_z(t)
        engines.append(U.linked_copy(proto, f"Merlin1D_{k + 1}", c, mtx, parent=parent))
    return engines
