"""Build one film shot (model + environment + cameras + animation) into film/shots/<id>.blend.

blender --background --factory-startup --python film/make_shot.py -- --shot s01_nine
"""
import argparse
import importlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import film  # noqa: E402
from film import common, fx, envs, shots  # noqa: E402

for mod in (common, fx, envs, shots):
    importlib.reload(mod)


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--shot", required=True)
    a = ap.parse_args(argv)
    scn = common.open_model()
    frames = shots.frames(a.shot)
    shots.SHOTS[a.shot](scn, frames)
    scn.render.filepath = os.path.join(ROOT, "film", "frames", a.shot, "f_")
    out = os.path.join(ROOT, "film", "shots", a.shot + ".blend")
    common.save(out)
    print(f"[film] built {a.shot}: {frames} frames -> {out}", flush=True)


if __name__ == "__main__":
    main()
