"""Random collector for the file manager (Level A2): open folders, DRAG files onto folders, DOUBLE-CLICK to preview.

usage: python level0/files_collect.py [STEPS=30000]
"""
import random, sys
from pathlib import Path
from app_collect import collect

ROOT = Path(__file__).parent
STEPS = int(sys.argv[1]) if __name__ == "__main__" and len(sys.argv) > 1 else 30_000
FOLDERS = [(36, 39 + 34 * i) for i in range(4)]  # folder row centres (left 2..70, rows of 30 px from y=24, step 34)
CLOSE = (148, 191)  # preview Close button centre
key = lambda s: "".join(map(str, s["loc"])) + str(s["cur"]) + str(s["preview"] + 1)


def sample(page, s):
    tiles = page.evaluate("[...document.querySelectorAll('.tile')].map(e => { const r = e.getBoundingClientRect(); "
                          "return [r.left + r.width / 2, r.top + 24]; })")  # icon centre of each visible file
    jit = lambda x, y, d=6: (x + random.uniform(-d, d), y + random.uniform(-d, d))
    r = random.random()
    if s["preview"] >= 0 and r < 0.6:
        return [1, *jit(*CLOSE, 4), 0, 0]
    if r < 0.30:
        return [1, *jit(*random.choice(FOLDERS)), 0, 0]
    if r < 0.62 and tiles:  # drag a file onto a folder
        return [0, *jit(*random.choice(tiles)), *jit(*random.choice(FOLDERS))]
    if r < 0.80 and tiles:
        return [2, *jit(*random.choice(tiles)), 0, 0]
    if r < 0.93:
        return [1, random.uniform(0, 223), random.uniform(0, 223), 0, 0]
    return [0, *(random.uniform(0, 223) for _ in range(4))]


if __name__ == "__main__":
    collect(ROOT / "files.html", ROOT.parent / "data" / "files", STEPS, sample, key)
