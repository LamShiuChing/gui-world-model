"""Screenshots PNG -> lossless WebP (about 2x smaller on Tetris frames, bit-exact). Each file is checked pixel-for-pixel
before its PNG is deleted. Readers use img(path_without_suffix) which takes whichever file exists.

usage: python level0/webp.py DIR [DIR...]      e.g. data/tetris/w0/img
"""
import os, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from PIL import Image


def img(p):  # p: .../img/123.png (or without suffix) -> PIL image from the .png or the .webp
    p = Path(p)
    q = p.with_suffix(".png")
    return Image.open(q if q.exists() else p.with_suffix(".webp"))


def one(p):
    a = np.asarray(Image.open(p).convert("RGB"))
    w = p.with_suffix(".webp")
    Image.fromarray(a).save(w, "WEBP", lossless=True, quality=100, method=4)
    if not np.array_equal(np.asarray(Image.open(w).convert("RGB")), a):
        w.unlink(); return 0, 0
    n0, n1 = p.stat().st_size, w.stat().st_size
    p.unlink()
    return n0, n1


if __name__ == "__main__":
    for d in sys.argv[1:]:
        ps = sorted(Path(d).glob("*.png"))
        if not ps:
            continue  # already converted
        with ProcessPoolExecutor(max(1, os.cpu_count() - 8)) as ex:  # leave a few threads for other jobs
            r = np.array(list(ex.map(one, ps, chunksize=64)), dtype=np.int64)
        bad = int((r[:, 0] == 0).sum())
        print(f"{d}: {len(ps)} PNG -> WebP, {r[:, 0].sum() / 1e9:.2f} GB -> {r[:, 1].sum() / 1e9:.2f} GB"
              + (f", {bad} NOT exact (PNG kept)" if bad else ""), flush=True)
