"""First-stage livery: stacked SPACEX with swoosh, Falcon 9 "F9" logo, US flag.

Logo outlines come from data/livery_pd.json, converted by tools/svg_to_livery.py from
public-domain SVGs on Wikimedia Commons (SpaceX, Falcon 9 and their logos are trademarks
of SpaceX).  The flag is generated from the Executive Order 10834 proportions.
Each set is placed on both sides (centred on 90 and 270 deg) as thin decal meshes that
follow the tank curvature a few millimetres above the paint.
"""
import json
import math
import os

import bmesh

from . import dims as D
from . import util as U

DATA = os.path.join(os.path.dirname(__file__), "data", "livery_pd.json")

# placement on the vehicle (m): letters 0.80 m tall stacked top to bottom S P A C E X
CAP = 0.80
PITCH = 1.6126
X_BASELINE = 19.973
F9_CENTRE_Z, F9_HEIGHT = 38.90, 1.90
FLAG_CENTRE_Z, FLAG_HOIST = 36.687, 0.946

# SVG letter metrics (user units): shared baseline and cap line
SVG_BASE, SVG_CAP = 49.3, 18.6


def load():
    with open(DATA, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _inside(p, q):
    x, y = p
    hit = False
    for (x0, y0), (x1, y1) in zip(q, q[1:] + q[:1]):
        if (y0 > y) != (y1 > y) and x < x0 + (y - y0) * (x1 - x0) / (y1 - y0):
            hit = not hit
    return hit


def nest(loops):
    """Split loops into [outer, holes...] groups by even-odd nesting; scanfill mis-fills
    disjoint outers passed together (e.g. the two parts of the E)."""
    depth = [sum(_inside(a[0], b) for b in loops if b is not a) for a in loops]
    groups = {i: [loops[i]] for i, d in enumerate(depth) if d % 2 == 0}
    for i, d in enumerate(depth):
        if d % 2:
            owner = next((j for j in groups if depth[j] == d - 1 and _inside(loops[i][0], loops[j])), None)
            if owner is None:
                groups[i] = [loops[i]]
            else:
                groups[owner].append(loops[i])
    return list(groups.values())


def _bbox(loops):
    xs = [p[0] for q in loops for p in q]
    ys = [p[1] for q in loops for p in q]
    return min(xs), max(xs), min(ys), max(ys)


def spacex_loops(data):
    """[(loops in (u, z), ...)] for the six letters and the swoosh, u relative to the logo axis."""
    L = data["spacex"]["letters"]
    s = CAP / (SVG_BASE - SVG_CAP)
    out = []
    for k, key in enumerate("spacex"):
        x0, x1, _, _ = _bbox(L[key])
        xc = (x0 + x1) / 2
        zb = X_BASELINE + (5 - k) * PITCH
        parts = [L[key]] + ([L["swoosh"]] if key == "x" else [])
        for loops in parts:
            out.append([[((x - xc) * s, zb + (SVG_BASE - y) * s) for x, y in q] for q in loops])
    return out


def f9_loops(data):
    red, black = data["f9"]["red"], data["f9"]["black"]
    x0, x1, y0, y1 = _bbox(red + black)
    s = F9_HEIGHT / (y1 - y0)
    xc, yc = (x0 + x1) / 2, (y0 + y1) / 2

    def tr(loops):
        return [[((x - xc) * s, F9_CENTRE_Z + (yc - y) * s) for x, y in q] for q in loops]
    return tr(red), tr(black)


def flag_loops():
    """US flag (hoist A = 1, fly 1.9) scaled to FLAG_HOIST; returns field, red, canton, stars."""
    A, B, C, Dw, E, G, K = 1.0, 1.9, 7.0 / 13.0, 0.76, 0.054, 0.063, 0.0616
    s = FLAG_HOIST
    left, top = -B / 2, A / 2                        # flag-local origin at the top-left corner

    def pt(x, y):                                    # x along the fly, y down from the top
        return ((left + x) * s, FLAG_CENTRE_Z + (top - y) * s)

    def rect(x0, y0, x1, y1):
        return [pt(x0, y0), pt(x1, y0), pt(x1, y1), pt(x0, y1)]

    field = [rect(0.0, 0.0, B, A)]
    red = []
    for k in range(0, 13, 2):                        # 7 red stripes; the top 4 stop at the canton
        y0, y1 = k / 13.0, (k + 1) / 13.0
        red.append(rect(Dw if y1 <= C + 1e-9 else 0.0, y0, B, y1))
    canton = [rect(0.0, 0.0, Dw, C)]
    stars = []
    ro, ri = K / 2, K / 2 * 0.381966
    for j in range(1, 10):
        for i in (range(1, 12, 2) if j % 2 else range(2, 11, 2)):
            cx, cy = i * G, j * E
            stars.append([pt(cx + (ro if n % 2 == 0 else ri) * math.sin(math.pi * n / 5),
                             cy - (ro if n % 2 == 0 else ri) * math.cos(math.pi * n / 5)) for n in range(10)])
    return field, red, canton, stars


def build(colls, M, parent):
    c = colls["Livery"]
    data = load()
    u90 = D.R * math.pi / 2
    red, black = f9_loops(data)
    field, stripes, canton, stars = flag_loops()
    # name: [(list of shapes (each a list of loops), material key, radial offset)]
    layers = {
        "SPACEX": [(spacex_loops(data), "blue", 0.0025)],
        "Falcon9Logo": [([red], "red_bird", 0.0025), ([black], "ink", 0.0025)],
        "Flag": [([field], "flag_white", 0.0020), ([[q] for q in stripes], "flag_red", 0.0026),
                 ([canton], "flag_blue", 0.0026), ([[q] for q in stars], "flag_white", 0.0032)],
    }
    objs = []
    for side, th0 in (("R", 0.0), ("L", math.pi)):
        for name, items in layers.items():
            mats, bm = [], bmesh.new()
            for shapes, mkey, w in items:
                if M[mkey] not in mats:
                    mats.append(M[mkey])
                mi = mats.index(M[mkey])
                for loops in shapes:
                    for group in nest([[(u + u90, z) for u, z in q] for q in loops]):
                        U.flat_fill(bm, group, w, mat=mi)
            U.slice_u(bm, 0.03)
            U.wrap(bm, th0, D.R)
            objs.append(U.finish(f"Livery_{name}_{side}", bm, c, mats, parent=parent, sharp=None))
    return objs
