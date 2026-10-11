"""Assemble the film: rendered shots + titles + fades + score -> H.264/AAC MP4.

python film/edit.py                      # final: film/frames/<id>/ -> film/out/falcon9.mp4
python film/edit.py --no-music           # sound effects only -> film/out/falcon9_sfx_only.mp4
python film/edit.py --draft              # previews (film/preview/<id>/, any frame step) at 960x540

The shots play back to back on the cues.json timeline (hard cuts, except a short dissolve
into the end shot); the picture fades up from black and out at the end.  Frames are
composited in numpy and piped straight into ffmpeg, so the edit is exact to the frame and
fully reproducible.  Titles use Bahnschrift (Windows); any other TTF can be passed with --font.
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FONT = r"C:\Windows\Fonts\bahnschrift.ttf"
REPO = "github.com/noahsabaj/falcon9-blender"
DISSOLVE = {"s12_end": 16}          # incoming shot -> dissolve length in frames, centred on the cut

# (start, end, fade_in, fade_out, left_x_frac, [(text, size_px, weight, tracking_em, y_frac, opacity, delay_s)])
# sizes are for 1920 px wide frames; a block that would pass the right title-safe edge is moved left
TITLES = [
    (19.0, 23.5, 0.9, 0.6, 0.54, [("FALCON 9", 112, "Light", 0.30, 0.47, 0.95, 0.0)]),
    (59.0, 65.0, 0.8, 1.0, 0.58, [("FALCON 9", 88, "Light", 0.30, 0.255, 0.95, 0.0),
                                 ("A 1:1 model, film and score,", 36, "SemiLight", 0.03, 0.355, 0.90, 0.6),
                                 ("made entirely in code by Claude Opus 5.5", 36, "SemiLight", 0.03, 0.402, 0.90, 0.9),
                                 (REPO, 30, "Regular", 0.04, 0.478, 0.82, 1.5),
                                 ("Fan work. Not affiliated with or endorsed by SpaceX.", 22, "Regular", 0.03,
                                  0.527, 0.60, 2.1)]),
]
SAFE_RIGHT = 0.93                   # right edge of the title-safe area (fraction of width)


def smooth(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3.0 - 2.0 * x)


def load_font(font_path, size, weight):
    f = ImageFont.truetype(font_path, size)
    try:
        f.set_variation_by_name(weight)
    except (OSError, ValueError, AttributeError):
        pass
    return f


def text_width(text, f, size, tracking):
    """Width of tracked text in pixels (no tracking after the last glyph)."""
    return sum(f.getlength(ch) for ch in text) + tracking * size * (len(text) - 1)


def text_layer(W, H, x, text, f, size, tracking, y_frac):
    """Float (0..1) coverage of tracked text with its left edge at x, and a soft shadow."""
    img = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(img)
    y = y_frac * H - size * 0.5
    for ch in text:
        d.text((x, y), ch, font=f, fill=255)
        x += f.getlength(ch) + tracking * size
    a = np.asarray(img, np.float32) / 255.0
    sh = np.asarray(img.filter(ImageFilter.GaussianBlur(size * 0.12)), np.float32) / 255.0
    return a, sh


def build_titles(W, H, font_path):
    out = []
    for start, end, fin, fout, x_frac, lines in TITLES:
        sized = [(t, int(sz * W / 1920), wt, tr, y, op, dl) for t, sz, wt, tr, y, op, dl in lines]
        fonts = [load_font(font_path, sz, wt) for _, sz, wt, *_ in sized]
        widest = max(text_width(t, f, sz, tr) for (t, sz, _, tr, *_), f in zip(sized, fonts))
        x = min(x_frac * W, SAFE_RIGHT * W - widest)
        layers = [(text_layer(W, H, x, t, f, sz, tr, y), op, dl)
                  for (t, sz, _, tr, y, op, dl), f in zip(sized, fonts)]
        out.append((start, end, fin, fout, layers))
        print(f"[edit] titles at {start:.1f}s: left edge {x / W:.3f}, widest line {widest / W:.3f} of the width")
    return out


def frame_path(base, sid, f, draft):
    p = os.path.join(base, sid, f"f_{f:04d}.png")
    if os.path.exists(p) or not draft:
        return p
    # drafts use whatever exists: the last preview frame so far, else the final frame, else a still
    for g in range(f - 1, 0, -1):
        q = os.path.join(base, sid, f"f_{g:04d}.png")
        if os.path.exists(q):
            return q
    q = os.path.join(HERE, "frames", sid, f"f_{f:04d}.png")
    if os.path.exists(q) and os.path.getsize(q) > 0:
        return q
    d = os.path.join(base, sid)
    stills = sorted(int(x[6:10]) for x in os.listdir(d) if x.startswith("still_")) if os.path.isdir(d) else []
    if stills:
        g = max([s for s in stills if s <= f] or stills[:1])
        return os.path.join(d, f"still_{g:04d}.png")
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draft", action="store_true")
    ap.add_argument("--no-music", action="store_true")
    ap.add_argument("--out", default="")
    ap.add_argument("--font", default=FONT)
    ap.add_argument("--crf", type=int, default=16)
    a = ap.parse_args()
    with open(os.path.join(HERE, "cues.json"), encoding="utf-8") as fh:
        cues = json.load(fh)
    fps = cues["fps"]
    W, H = (960, 540) if a.draft else tuple(cues["resolution"])
    base = os.path.join(HERE, "preview" if a.draft else "frames")
    shots = [(s["id"], round(s["start"] * fps), round(s["end"] * fps)) for s in cues["shots"]]
    total = round(cues["duration"] * fps)
    titles = build_titles(W, H, a.font)
    audio = os.path.join(HERE, "audio", "sfx.wav" if a.no_music else "mix.wav")
    name = "falcon9_sfx_only" if a.no_music else "falcon9"
    out = a.out or os.path.join(HERE, "out", name + ("_draft" if a.draft else "") + ".mp4")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(fps), "-i", "-",
           "-i", audio,
           "-vf", "scale=out_color_matrix=bt709:out_range=tv,format=yuv420p",
           "-c:v", "libx264", "-preset", "slow", "-crf", str(a.crf), "-profile:v", "high",
           "-x264-params", "aq-mode=3", "-colorspace", "bt709", "-color_primaries", "bt709",
           "-color_trc", "bt709", "-c:a", "aac", "-b:a", "320k", "-ar", "48000",
           "-movflags", "+faststart", "-shortest", out]
    enc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    rng = np.random.default_rng(9)
    cache = {}

    def load(sid, f, n):
        f = min(max(f, 1), n)
        p = frame_path(base, sid, f, a.draft)
        if p not in cache:
            if not os.path.exists(p):
                sys.exit(f"missing frame {p}")
            im = Image.open(p).convert("RGB")
            if im.size != (W, H):
                im = im.resize((W, H), Image.LANCZOS)
            cache.clear()
            cache[p] = np.asarray(im, np.float32) / 255.0
        return cache[p]

    for i in range(total):
        t = i / fps
        k = next(j for j, (_, s0, s1) in enumerate(shots) if s0 <= i < s1 or j == len(shots) - 1)
        sid, s0, s1 = shots[k]
        img = load(sid, i - s0 + 1, s1 - s0).copy()
        # dissolves into a shot (held frames cover the overlap on both sides of the cut)
        for j, (sj, a0, a1) in enumerate(shots):
            L = DISSOLVE.get(sj, 0)
            if L and j > 0 and a0 - L // 2 <= i < a0 + L // 2:
                pj, p0, p1 = shots[j - 1]
                A = load(pj, i - p0 + 1, p1 - p0)
                B = load(sj, i - a0 + 1, a1 - a0)
                w = smooth((i - (a0 - L // 2) + 0.5) / L)
                img = A * (1.0 - w) + B * w
        # titles
        for start, end, fin, fout, layers in titles:
            if not (start <= t < end):
                continue
            for (txt, shadow), op, delay in layers:
                o = op * smooth((t - start - delay) / fin) * (1.0 - smooth((t - (end - fout)) / fout))
                if o <= 0.0:
                    continue
                img *= (1.0 - 0.45 * o * shadow)[..., None]
                img += (o * txt)[..., None] * (1.0 - img)
        # fade up from black, fade out at the end
        g = smooth(t / 1.5) * (1.0 - smooth((t - 64.0) / 1.0))
        if g < 1.0:
            img *= g
        # +-1 LSB triangular dither (luma only) so fades and dark gradients do not band in 8 bits
        dith = rng.random((H, W, 1), np.float32) - rng.random((H, W, 1), np.float32)
        enc.stdin.write(np.clip(img * 255.0 + 0.5 + dith, 0.0, 255.0).astype(np.uint8).tobytes())
        if i % 120 == 0:
            print(f"[edit] {i}/{total} {sid}", flush=True)
    enc.stdin.close()
    if enc.wait() != 0:
        sys.exit("ffmpeg failed")
    print(f"[edit] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
