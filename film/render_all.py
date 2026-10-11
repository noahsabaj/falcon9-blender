"""Build and final-render the film's shots in order (resumable: finished frames are kept).

python film/render_all.py [--blender PATH] [--no-build] [shot ...]

Each shot is rebuilt from the model with film/make_shot.py (unless --no-build), then rendered
at full resolution into film/frames/<id>/ by film/render_shot.py.  With no shot ids, every
shot in cues.json is done in order.
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
BLENDER = r"C:\Program Files\Blender Foundation\Blender 5.2\blender.exe"


def run(cmd):
    t = time.time()
    with tempfile.TemporaryFile() as log:             # a file, not a pipe: Blender's output never blocks it
        p = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        log.seek(0)
        for line in log.read().decode("utf-8", "replace").splitlines():
            if line.startswith("[") or "Error" in line or "Traceback" in line:
                print("   ", line, flush=True)
    if p.returncode != 0:
        sys.exit(f"failed ({p.returncode}): {' '.join(cmd)}")
    return time.time() - t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blender", default=os.environ.get("BLENDER", BLENDER))
    ap.add_argument("--no-build", action="store_true")
    ap.add_argument("shots", nargs="*")
    a = ap.parse_args()
    with open(os.path.join(HERE, "cues.json"), encoding="utf-8") as fh:
        ids = [s["id"] for s in json.load(fh)["shots"]]
    for sid in a.shots or ids:
        if sid not in ids:
            sys.exit(f"unknown shot {sid}")
        print(f"[all] {sid}", flush=True)
        if not a.no_build:
            run([a.blender, "--background", "--factory-startup", "--python", "film/make_shot.py", "--", "--shot", sid])
        dt = run([a.blender, "--background", os.path.join("film", "shots", sid + ".blend"),
                  "--python", "film/render_shot.py"])
        print(f"[all] {sid} done in {dt / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
