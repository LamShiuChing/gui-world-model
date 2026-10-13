"""Compact storage for EYES features: int8 with a per-channel scale/offset, optionally on a frozen PCA basis.

int8 per-channel is 2x smaller than fp16 with no measurable loss; PCA-k shrinks it further (gate it:
store_gate.py). X.npy (fp16, N x T x 384) -> X.q8.npy (int8, N x T x k) + X.q8.npz (offset, scale [, mean, basis]).
load(path) gives the fp16 file if it still exists, else a reader whose [idx] returns fp16 (n, T, 384) like the array.

usage: python level0/store.py PATH.npy [K=384] [rm]   convert + report error; rm = delete the fp16 file afterwards
"""
import sys
from pathlib import Path
import numpy as np


def fit(x, k=384, rng=np.random.default_rng(0)):  # x: (N, T, d) array/memmap -> codec dict
    rows = np.sort(rng.choice(len(x), min(len(x), 2000), replace=False))
    cal = np.asarray(x[rows]).reshape(-1, x.shape[-1]).astype(np.float32)
    cal = cal[rng.choice(len(cal), min(len(cal), 50_000), replace=False)]
    c = {}
    if k < x.shape[-1]:
        c["mean"] = cal.mean(0)
        c["basis"] = np.linalg.svd(cal - c["mean"], full_matrices=False)[2][:k]
        cal = (cal - c["mean"]) @ c["basis"].T
    lo, hi = np.percentile(cal, [0.01, 99.99], axis=0)
    c["scale"] = (np.maximum(hi - lo, 1e-6) / 255).astype(np.float32)
    c["offset"] = (lo + 128 * c["scale"]).astype(np.float32)  # q in [-128, 127]
    return c


def encode(y, c):
    y = np.asarray(y, np.float32)
    if "basis" in c:
        y = (y - c["mean"]) @ c["basis"].T
    return np.clip(np.rint((y - c["offset"]) / c["scale"]), -128, 127).astype(np.int8)


def decode(q, c):
    x = q.astype(np.float32) * c["scale"] + c["offset"]
    return (x if "basis" not in c else x @ c["basis"] + c["mean"]).astype(np.float16)


class Q8:
    def __init__(self, p):
        p = Path(p)
        self.q = np.load(p.with_suffix(".q8.npy"), mmap_mode="r")
        self.c = dict(np.load(p.with_suffix(".q8.npz")))
        self.shape = (*self.q.shape[:2], 384)

    def __len__(self):
        return len(self.q)

    def __getitem__(self, i):
        idx = np.arange(len(self))[i]  # any numpy index -> frame numbers; decoded in chunks to keep RAM low
        if idx.ndim == 0:
            return decode(self.q[int(idx)], self.c)
        out = np.empty((len(idx), *self.shape[1:]), np.float16)
        for j in range(0, len(idx), 2048):
            out[j:j + 2048] = decode(self.q[idx[j:j + 2048]], self.c)
        return out


def load(p, mmap=True):
    p = Path(p)
    if p.exists():
        return np.load(p, mmap_mode="r" if mmap else None)
    f = Q8(p)
    return f if mmap else f[:]


def convert(p, k=384, rng=np.random.default_rng(0)):
    p = Path(p)
    x = np.load(p, mmap_mode="r")
    N, T, _ = x.shape
    c = fit(x, k, rng)
    np.savez(p.with_suffix(".q8.npz"), **c)
    q = np.lib.format.open_memmap(p.with_suffix(".q8.npy"), "w+", np.int8, (N, T, k))
    for i in range(0, N, 512):
        q[i:i + 512] = encode(x[i:i + 512], c)
    q.flush(); del q
    r, e = Q8(p), np.zeros(4)  # relative error on frames and on frame-to-frame changes (what the IMAGINER predicts)
    for i in rng.choice(N - 1, min(N - 1, 500), replace=False):
        a, b = np.asarray(x[i:i + 2], np.float32), r[i:i + 2].astype(np.float32)
        da, db = a[1] - a[0], b[1] - b[0]
        e += [((a - b) ** 2).sum(), ((a - a.mean((0, 1))) ** 2).sum(), ((da - db) ** 2).sum(), (da ** 2).sum()]
    print(f"{p.name}: {N} frames -> int8 k={k} | NMSE frames {e[0] / e[1]:.5f}, changes {e[2] / max(e[3], 1e-9):.5f} | "
          f"{p.stat().st_size / 1e9:.2f} GB -> {p.with_suffix('.q8.npy').stat().st_size / 1e9:.2f} GB", flush=True)
    return e[0] / e[1], e[2] / max(e[3], 1e-9)


if __name__ == "__main__":
    P = Path(sys.argv[1])
    nm, nd = convert(P, int(sys.argv[2]) if len(sys.argv) > 2 else 384)
    if "rm" in sys.argv[3:]:
        assert nm < 0.01 and nd < 0.05, "error too large - fp16 file kept"
        P.unlink(); print(f"deleted {P.name}")
