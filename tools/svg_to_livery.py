"""Convert the public-domain logo SVGs from Wikimedia Commons into livery polygons.

Sources (both tagged PD-textlogo / PD-shape on Commons; the marks are SpaceX trademarks):
  https://commons.wikimedia.org/wiki/File:SpaceX_logo_black.svg
  https://commons.wikimedia.org/wiki/File:Falcon_9_logo.svg

Curves are flattened adaptively; output stays in SVG user units (y down) and is laid
out on the vehicle by f9/livery.py.  The US flag is generated there from its spec.

usage: python -I svg_to_livery.py <SpaceX_logo_black.svg> <Falcon_9_logo.svg> <out.json>
"""
import json
import math
import re
import sys
import xml.etree.ElementTree as ET

NUM = r"[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?"
TOKEN = re.compile(r"[MmLlHhVvCcSsQqTtZz]|" + NUM)


def mul(m, t):
    a, b, c, d, e, f = m
    A, B, C, D, E, F = t
    return [a * A + c * B, b * A + d * B, a * C + c * D, b * C + d * D, a * E + c * F + e, b * E + d * F + f]


def apply(m, p):
    return (m[0] * p[0] + m[2] * p[1] + m[4], m[1] * p[0] + m[3] * p[1] + m[5])


def parse_transform(s):
    m = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
    for name, args in re.findall(r"(\w+)\s*\(([^)]*)\)", s or ""):
        v = [float(x) for x in re.findall(NUM, args)]
        if name == "translate":
            t = [1, 0, 0, 1, v[0], v[1] if len(v) > 1 else 0.0]
        elif name == "scale":
            t = [v[0], 0, 0, v[1] if len(v) > 1 else v[0], 0, 0]
        elif name == "matrix":
            t = v
        elif name == "rotate":
            a = math.radians(v[0])
            t = [math.cos(a), math.sin(a), -math.sin(a), math.cos(a), 0, 0]
            if len(v) == 3:
                t = mul(mul([1, 0, 0, 1, v[1], v[2]], t), [1, 0, 0, 1, -v[1], -v[2]])
        else:
            raise ValueError(f"unsupported transform {name}")
        m = mul(m, t)
    return m


def _dist_line(p, a, b):
    dx, dy = b[0] - a[0], b[1] - a[1]
    n = math.hypot(dx, dy)
    if n < 1e-12:
        return math.hypot(p[0] - a[0], p[1] - a[1])
    return abs((p[0] - a[0]) * dy - (p[1] - a[1]) * dx) / n


def flat_cubic(p0, p1, p2, p3, tol, out, depth=0):
    if depth > 18 or max(_dist_line(p1, p0, p3), _dist_line(p2, p0, p3)) <= tol:
        out.append(p3)
        return
    mid = lambda a, b: ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
    p01, p12, p23 = mid(p0, p1), mid(p1, p2), mid(p2, p3)
    p012, p123 = mid(p01, p12), mid(p12, p23)
    q = mid(p012, p123)
    flat_cubic(p0, p01, p012, q, tol, out, depth + 1)
    flat_cubic(q, p123, p23, p3, tol, out, depth + 1)


def path_loops(d, tol):
    """SVG path data -> list of closed loops (untransformed user units)."""
    toks = TOKEN.findall(d)
    loops, cur = [], []
    i, cmd = 0, None
    pos, start, last_ctrl = (0.0, 0.0), (0.0, 0.0), None

    def num():
        nonlocal i
        v = float(toks[i])
        i += 1
        return v

    def close():
        nonlocal cur
        if len(cur) >= 3:
            if math.hypot(cur[0][0] - cur[-1][0], cur[0][1] - cur[-1][1]) < 1e-9:
                cur = cur[:-1]
            loops.append(cur)
        cur = []

    while i < len(toks):
        if re.fullmatch(r"[A-Za-z]", toks[i]):
            cmd = toks[i]
            i += 1
            if cmd in "Zz":
                close()
                pos, last_ctrl = start, None
                continue
        rel = cmd.islower()
        c = cmd.upper()
        ox, oy = pos if rel else (0.0, 0.0)
        if c == "M":
            close()
            pos = (ox + num(), oy + num())
            start = pos
            cur = [pos]
            cmd = "l" if rel else "L"          # subsequent pairs are implicit linetos
            last_ctrl = None
        elif c == "L":
            pos = (ox + num(), oy + num())
            cur.append(pos)
            last_ctrl = None
        elif c == "H":
            pos = ((pos[0] if rel else 0.0) + num(), pos[1])
            cur.append(pos)
            last_ctrl = None
        elif c == "V":
            pos = (pos[0], (pos[1] if rel else 0.0) + num())
            cur.append(pos)
            last_ctrl = None
        elif c in "CS":
            if c == "C":
                p1 = (ox + num(), oy + num())
            else:
                p1 = (2 * pos[0] - last_ctrl[0], 2 * pos[1] - last_ctrl[1]) if last_ctrl else pos
            p2 = (ox + num(), oy + num())
            p3 = (ox + num(), oy + num())
            flat_cubic(pos, p1, p2, p3, tol, cur)
            pos, last_ctrl = p3, p2
        elif c in "QT":
            if c == "Q":
                q = (ox + num(), oy + num())
            else:
                q = (2 * pos[0] - last_ctrl[0], 2 * pos[1] - last_ctrl[1]) if last_ctrl else pos
            p3 = (ox + num(), oy + num())
            p1 = (pos[0] + 2 / 3 * (q[0] - pos[0]), pos[1] + 2 / 3 * (q[1] - pos[1]))
            p2 = (p3[0] + 2 / 3 * (q[0] - p3[0]), p3[1] + 2 / 3 * (q[1] - p3[1]))
            flat_cubic(pos, p1, p2, p3, tol, cur)
            pos, last_ctrl = p3, q
        else:
            raise ValueError(f"unsupported path command {cmd}")
    close()
    return loops


def element_loops(el, m, tol):
    tag = el.tag.split("}")[-1]
    if tag == "path":
        loops = path_loops(el.get("d"), tol)
    elif tag in ("polygon", "polyline"):
        v = [float(x) for x in re.findall(NUM, el.get("points"))]
        loops = [list(zip(v[0::2], v[1::2]))]
    elif tag == "rect":
        x, y = float(el.get("x", 0)), float(el.get("y", 0))
        w, h = float(el.get("width")), float(el.get("height"))
        loops = [[(x, y), (x + w, y), (x + w, y + h), (x, y + h)]]
    else:
        return []
    return [[apply(m, p) for p in q] for q in loops]


def walk(el, m, tol, out, ctx):
    m = mul(m, parse_transform(el.get("transform")))
    ctx = dict(ctx)
    if el.get("class"):
        ctx["class"] = el.get("class")
    style = el.get("style", "")
    fill = el.get("fill") or (re.search(r"fill:\s*(#[0-9a-fA-F]{6})", style) or [None, None])[1]
    if fill:
        ctx["fill"] = fill.lower()
    loops = element_loops(el, m, tol)
    if loops:
        out.append((ctx, loops))
    for ch in el:
        walk(ch, m, tol, out, ctx)


def load(path, tol):
    out = []
    walk(ET.parse(path).getroot(), [1.0, 0.0, 0.0, 1.0, 0.0, 0.0], tol, out, {})
    return out


def rounded(loops):
    return [[[round(x, 4), round(y, 4)] for x, y in q] for q in loops]


def main():
    spacex_svg, f9_svg, out = sys.argv[1:4]
    # tolerances in user units: ~1.5 mm chord error at the vehicle scale
    letters = {}
    for ctx, loops in load(spacex_svg, tol=0.05):
        key = ctx.get("class", "").replace("letter_", "")
        letters.setdefault(key, []).extend(loops)
    f9 = {"red": [], "black": []}
    for ctx, loops in load(f9_svg, tol=0.03):
        f9["red" if ctx.get("fill") == "#eb1747" else "black"].extend(loops)
    data = {
        "note": "Generated by tools/svg_to_livery.py from public-domain Wikimedia Commons SVGs. "
                "SpaceX, Falcon 9 and their logos are trademarks of SpaceX.",
        "spacex": {"source": "https://commons.wikimedia.org/wiki/File:SpaceX_logo_black.svg",
                   "units": "svg user units, y down",
                   "letters": {k: rounded(v) for k, v in letters.items()}},
        "f9": {"source": "https://commons.wikimedia.org/wiki/File:Falcon_9_logo.svg",
               "units": "mm, y down",
               "red": rounded(f9["red"]), "black": rounded(f9["black"])},
    }
    with open(out, "w") as fh:
        json.dump(data, fh, separators=(",", ":"))
    for k, v in letters.items():
        xs = [p[0] for q in v for p in q]
        ys = [p[1] for q in v for p in q]
        print(f"spacex {k:7s} loops {len(v)}  pts {sum(map(len, v)):4d}  x {min(xs):7.2f}..{max(xs):7.2f}"
              f"  y {min(ys):6.2f}..{max(ys):6.2f}")
    for k, v in f9.items():
        xs = [p[0] for q in v for p in q]
        ys = [p[1] for q in v for p in q]
        print(f"f9 {k:5s} loops {len(v)}  pts {sum(map(len, v)):4d}  x {min(xs):7.3f}..{max(xs):7.3f}"
              f"  y {min(ys):7.3f}..{max(ys):7.3f}")


if __name__ == "__main__":
    main()
