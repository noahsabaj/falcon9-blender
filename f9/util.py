"""Geometry helpers for the Falcon 9 build.

Everything is built with bmesh / the data API (no operators), so the build runs the same
in background mode as in the UI.

Conventions used by the "flat" helpers: geometry is first authored in an unrolled
(u, z, w) space stored in vertex (x, y, z): u = arc length around the axis measured at a
reference radius, z = height, w = radial offset.  `wrap()` then maps it onto the
cylinder.  Increasing u = increasing azimuth.
"""
import math

import bmesh
import bpy
from mathutils import Matrix, Vector
from mathutils import geometry as mgeo

TAU = 2.0 * math.pi


# ---------------------------------------------------------------------------------
# scene plumbing
# ---------------------------------------------------------------------------------
def collection(name, parent=None):
    coll = bpy.data.collections.get(name)
    if coll is None:
        coll = bpy.data.collections.new(name)
        (parent if parent is not None else bpy.context.scene.collection).children.link(coll)
    return coll


def empty(name, coll, parent=None, location=(0.0, 0.0, 0.0), size=1.0, kind="PLAIN_AXES"):
    ob = bpy.data.objects.new(name, None)
    ob.empty_display_type = kind
    ob.empty_display_size = size
    ob.location = location
    coll.objects.link(ob)
    if parent is not None:
        ob.parent = parent
    return ob


def finish(name, bm, coll, mats, smooth=True, sharp=35.0, parent=None, frame=None):
    """Write `bm` (world-space coordinates) into a new mesh object.

    frame: optional world matrix for the object origin; the mesh is re-expressed in it.
    """
    if frame is not None:
        bm.transform(frame.inverted())
    me = bpy.data.meshes.new(name)
    bm.normal_update()
    bm.to_mesh(me)
    bm.free()
    for m in mats:
        me.materials.append(m)
    if smooth:
        me.shade_smooth()
        if sharp is not None:
            me.set_sharp_from_angle(angle=math.radians(sharp))
    else:
        me.shade_flat()
    ob = bpy.data.objects.new(name, me)
    coll.objects.link(ob)
    if parent is not None:
        ob.parent = parent
    if frame is not None:
        ob.matrix_world = frame
    return ob


def linked_copy(src, name, coll, matrix, parent=None):
    ob = bpy.data.objects.new(name, src.data)
    coll.objects.link(ob)
    if parent is not None:
        ob.parent = parent
    ob.matrix_world = matrix
    return ob


# ---------------------------------------------------------------------------------
# primitives
# ---------------------------------------------------------------------------------
def revolve(bm, profile, segs, a0=0.0, a1=None, mat=0, closed=False):
    """Revolve an (r, z) profile about Z.

    Faces point to the right of the direction of travel in the (r right, z up) plane,
    so an upward profile faces outward and a counter-clockwise closed profile is an
    outward-facing solid.  a1=None revolves a full turn.
    """
    full = a1 is None
    if full:
        angs = [a0 + TAU * j / segs for j in range(segs)]
    else:
        angs = [a0 + (a1 - a0) * j / segs for j in range(segs + 1)]
    cs = [(math.cos(a), math.sin(a)) for a in angs]
    rings = []
    for r, z in profile:
        if r < 1e-7:
            v = bm.verts.new((0.0, 0.0, z))
            rings.append([v] * len(angs))
        else:
            rings.append([bm.verts.new((r * c, r * s, z)) for c, s in cs])
    n = len(angs)
    jmax = n if full else n - 1
    pairs = [(i, i + 1) for i in range(len(rings) - 1)]
    if closed:
        pairs.append((len(rings) - 1, 0))
    faces = []
    for i0, i1 in pairs:
        A, B = rings[i0], rings[i1]
        for j in range(jmax):
            j1 = (j + 1) % n
            q = []
            for v in (A[j], A[j1], B[j1], B[j]):
                if v not in q:
                    q.append(v)
            if len(q) < 3:
                continue
            try:
                f = bm.faces.new(q)
            except ValueError:
                continue
            f.material_index = mat
            faces.append(f)
    return rings, faces


def cap_ring(bm, ring, mat=0, flip=False):
    """Fill a closed vertex ring with an n-gon."""
    verts = list(dict.fromkeys(ring))
    if flip:
        verts.reverse()
    f = bm.faces.new(verts)
    f.material_index = mat
    return f


def add_box(bm, size, matrix=Matrix(), mat=0):
    sx, sy, sz = size
    ret = bmesh.ops.create_cube(bm, size=1.0, matrix=matrix @ Matrix.Diagonal((sx, sy, sz, 1.0)))
    for f in {f for v in ret["verts"] for f in v.link_faces}:
        f.material_index = mat
    return ret["verts"]


def add_cyl(bm, r1, r2, depth, segs, matrix=Matrix(), mat=0, caps=True):
    """Cylinder/cone along local Z, centred on the origin of `matrix`."""
    ret = bmesh.ops.create_cone(bm, cap_ends=caps, cap_tris=False, segments=segs,
                                radius1=r1, radius2=r2, depth=depth, matrix=matrix)
    for f in {f for v in ret["verts"] for f in v.link_faces}:
        f.material_index = mat
    return ret["verts"]


def add_torus(bm, major, minor, segs_major, segs_minor, matrix=Matrix(), mat=0):
    prof = [(major + minor * math.cos(TAU * k / segs_minor), minor * math.sin(TAU * k / segs_minor))
            for k in range(segs_minor)]
    sub = bmesh.new()
    revolve(sub, prof, segs_major, closed=True)
    sub.transform(matrix)
    return merge_into(bm, sub, mat)


def merge_into(bm, sub, mat=None):
    """Append bmesh `sub` into `bm` (consumes sub).  Returns new verts."""
    me = bpy.data.meshes.new("_tmp")
    sub.to_mesh(me)
    sub.free()
    n0 = len(bm.verts)
    nf0 = len(bm.faces)
    bm.from_mesh(me)
    bpy.data.meshes.remove(me)
    bm.verts.ensure_lookup_table()
    bm.faces.ensure_lookup_table()
    if mat is not None:
        for f in bm.faces[nf0:]:
            f.material_index = mat
    return bm.verts[n0:]


def frame_at(theta, r, z, local=Matrix()):
    """Matrix whose X = outward radial, Y = tangential (+theta), Z = up, at (theta, r, z)."""
    c, s = math.cos(theta), math.sin(theta)
    m = Matrix(((c, -s, 0.0, r * c), (s, c, 0.0, r * s), (0.0, 0.0, 1.0, z), (0.0, 0.0, 0.0, 1.0)))
    return m @ local


def rot_z(theta):
    return Matrix.Rotation(theta, 4, "Z")


# ---------------------------------------------------------------------------------
# flat (unrolled) authoring
# ---------------------------------------------------------------------------------
def _clean_loop(pts):
    q = []
    for x, y in pts:
        if not q or abs(x - q[-1][0]) > 1e-7 or abs(y - q[-1][1]) > 1e-7:
            q.append((float(x), float(y)))
    if len(q) > 2 and abs(q[0][0] - q[-1][0]) < 1e-7 and abs(q[0][1] - q[-1][1]) < 1e-7:
        q.pop()
    return q if len(q) >= 3 else None


def flat_fill(bm, loops, w=0.0, mat=0):
    """Triangulate 2D loops (outer + holes, even-odd) into faces at height w facing +w."""
    loops = [q for q in (_clean_loop(p) for p in loops) if q]
    if not loops:
        return []
    tris = mgeo.tessellate_polygon([[Vector((x, y, 0.0)) for x, y in q] for q in loops])
    flat = [p for q in loops for p in q]
    verts = [bm.verts.new((x, y, w)) for x, y in flat]
    faces = []
    for t in tris:
        try:
            f = bm.faces.new([verts[i] for i in t])
        except ValueError:
            continue
        f.normal_update()
        if f.normal.z < 0:
            f.normal_flip()
        f.material_index = mat
        faces.append(f)
    return faces


def flat_prism(bm, loops, w0, w1, mat_top=0, mat_bottom=0, mat_side=0):
    """Solid prism from 2D loops between w0 (bottom) and w1 (top)."""
    loops = [q for q in (_clean_loop(p) for p in loops) if q]
    if not loops:
        return []
    tris = mgeo.tessellate_polygon([[Vector((x, y, 0.0)) for x, y in q] for q in loops])
    flat = [p for q in loops for p in q]
    top = [bm.verts.new((x, y, w1)) for x, y in flat]
    bot = [bm.verts.new((x, y, w0)) for x, y in flat]
    faces = []
    for t in tris:
        try:
            ft = bm.faces.new([top[i] for i in t])
            fb = bm.faces.new([bot[i] for i in reversed(t)])
        except ValueError:
            continue
        ft.normal_update()
        if ft.normal.z < 0:
            ft.normal_flip()
            fb.normal_flip()
        ft.material_index = mat_top
        fb.material_index = mat_bottom
        faces += [ft, fb]
    base = 0
    for q in loops:
        n = len(q)
        for i in range(n):
            a, b = base + i, base + (i + 1) % n
            try:
                f = bm.faces.new([bot[a], bot[b], top[b], top[a]])
            except ValueError:
                continue
            f.material_index = mat_side
            faces.append(f)
        base += n
    bmesh.ops.recalc_face_normals(bm, faces=faces)
    return faces


def slice_u(bm, step, axis=0):
    """Cut all geometry with planes perpendicular to `axis` every `step` (for wrapping)."""
    if not bm.verts:
        return
    lo = min(v.co[axis] for v in bm.verts)
    hi = max(v.co[axis] for v in bm.verts)
    k = math.floor(lo / step) + 1
    no = Vector((0.0, 0.0, 0.0))
    no[axis] = 1.0
    while k * step < hi - 1e-6:
        co = Vector((0.0, 0.0, 0.0))
        co[axis] = k * step
        bmesh.ops.bisect_plane(bm, geom=bm.verts[:] + bm.edges[:] + bm.faces[:],
                               plane_co=co, plane_no=no, dist=1e-6)
        k += 1


def wrap(bm, theta0, r_ref, r_base=None, verts=None):
    """Map flat (u, z, w) verts onto the cylinder: theta = theta0 + u / r_ref,
    radius = r_base + w (r_base defaults to r_ref)."""
    if r_base is None:
        r_base = r_ref
    for v in (verts if verts is not None else bm.verts):
        u, z, w = v.co
        t = theta0 + u / r_ref
        r = r_base + w
        v.co = Vector((r * math.cos(t), r * math.sin(t), z))


def ellipse(cx, cy, rx, ry, n=48, a0=0.0):
    return [(cx + rx * math.cos(a0 + TAU * k / n), cy + ry * math.sin(a0 + TAU * k / n)) for k in range(n)]


def rounded_rect(cx, cy, w, h, r, n=6):
    r = min(r, w / 2 - 1e-6, h / 2 - 1e-6)
    pts = []
    for (qx, qy, a0) in ((cx + w / 2 - r, cy + h / 2 - r, 0.0), (cx - w / 2 + r, cy + h / 2 - r, 90.0),
                         (cx - w / 2 + r, cy - h / 2 + r, 180.0), (cx + w / 2 - r, cy - h / 2 + r, 270.0)):
        for k in range(n + 1):
            a = math.radians(a0 + 90.0 * k / n)
            pts.append((qx + r * math.cos(a), qy + r * math.sin(a)))
    return pts


def catmull_rom(points, samples_per_seg=12, alpha=0.5):
    """Centripetal Catmull-Rom through 2D points (endpoints duplicated)."""
    P = [Vector(p) for p in points]
    P = [P[0] + (P[0] - P[1])] + P + [P[-1] + (P[-1] - P[-2])]
    out = []
    for i in range(1, len(P) - 2):
        p0, p1, p2, p3 = P[i - 1], P[i], P[i + 1], P[i + 2]
        t0 = 0.0
        t1 = t0 + max((p1 - p0).length, 1e-9) ** alpha
        t2 = t1 + max((p2 - p1).length, 1e-9) ** alpha
        t3 = t2 + max((p3 - p2).length, 1e-9) ** alpha
        for k in range(samples_per_seg):
            t = t1 + (t2 - t1) * k / samples_per_seg
            a1 = p0 * ((t1 - t) / (t1 - t0)) + p1 * ((t - t0) / (t1 - t0))
            a2 = p1 * ((t2 - t) / (t2 - t1)) + p2 * ((t - t1) / (t2 - t1))
            a3 = p2 * ((t3 - t) / (t3 - t2)) + p3 * ((t - t2) / (t3 - t2))
            b1 = a1 * ((t2 - t) / (t2 - t0)) + a2 * ((t - t0) / (t2 - t0))
            b2 = a2 * ((t3 - t) / (t3 - t1)) + a3 * ((t - t1) / (t3 - t1))
            out.append(tuple(b1 * ((t2 - t) / (t2 - t1)) + b2 * ((t - t1) / (t2 - t1))))
    out.append(tuple(P[-2]))
    return out


def set_material_by(bm, pred, mat):
    for f in bm.faces:
        if pred(f):
            f.material_index = mat


def loft(bm, sections, mat=0, cap_start=True, cap_end=True, closed_loop=True, recalc=True):
    """Skin consecutive 3D sections (equal point counts) with quads; optional end caps."""
    rows = [[bm.verts.new(p) for p in sec] for sec in sections]
    n = len(rows[0])
    jmax = n if closed_loop else n - 1
    faces = []
    for A, B in zip(rows[:-1], rows[1:]):
        for j in range(jmax):
            j1 = (j + 1) % n
            try:
                f = bm.faces.new((A[j], A[j1], B[j1], B[j]))
            except ValueError:
                continue
            f.material_index = mat
            faces.append(f)
    if closed_loop:
        for row, use in ((rows[0], cap_start), (rows[-1], cap_end)):
            if use:
                try:
                    f = bm.faces.new(row)
                    f.material_index = mat
                    faces.append(f)
                except ValueError:
                    pass
    if recalc:
        bmesh.ops.recalc_face_normals(bm, faces=faces)
    return rows, faces


def section_at(theta, pts_yx, z):
    """Place a 2D section given as (tangential y, radial x) at azimuth theta, height z."""
    c, s = math.cos(theta), math.sin(theta)
    return [(x * c - y * s, x * s + y * c, z) for y, x in pts_yx]


def section_cyl(theta, pts_ur, z, r_ref):
    """Like section_at, but the first coordinate is arc length u at r_ref and the second
    is the radius, so parts sitting on the skin follow the curvature."""
    out = []
    for u, r in pts_ur:
        t = theta + u / r_ref
        out.append((r * math.cos(t), r * math.sin(t), z))
    return out


def blister(bm, theta, z_top, z_bot, half_w, height, r_skin, round_top=0.0, round_bot=0.0,
            nz=24, nu=14, sink=0.012, mat=0, crown=0.7):
    """Rounded fairing bump lying on a cylinder (cable/latch covers, pods).

    round_top / round_bot: length over which the outline closes in a semicircle-like
    curve at that end (0 = square end)."""
    sections = []
    for k in range(nz + 1):
        z = z_top + (z_bot - z_top) * k / nz
        f = 1.0
        dt, db = z_top - z, z - z_bot
        if round_top > 0 and dt < round_top:
            f = min(f, math.sqrt(max(0.0, 1.0 - ((round_top - dt) / round_top) ** 2)))
        if round_bot > 0 and db < round_bot:
            f = min(f, math.sqrt(max(0.0, 1.0 - ((round_bot - db) / round_bot) ** 2)))
        hw = max(0.01, half_w * f)
        hh = max(0.002, height * f ** 0.6)
        sec = []
        for j in range(nu + 1):
            s = -1.0 + 2.0 * j / nu
            sec.append((s * hw, r_skin + 0.001 + hh * (1.0 - s * s) ** crown))
        sec += [(hw, r_skin - sink), (-hw, r_skin - sink)]
        sections.append(section_cyl(theta, sec, z, r_skin))
    return loft(bm, sections, mat=mat)
