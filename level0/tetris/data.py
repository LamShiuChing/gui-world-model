"""Tetris data shared by train.py and mix.py: frame table, step kinds, held-out game split, fast chunked batches.

Fast loading: random single-frame reads from the 35 GB feature file cost ~150 ms per batch; a batch is instead built
from short CONTIGUOUS runs (L transitions + the frame before and after), read in one slice each (~ms), and the next
batch is prefetched in a background thread while the GPU trains.
"""
import json, sys, threading, queue
from pathlib import Path
import numpy as np, torch
from pieces import visible_cells
sys.path.insert(0, str(Path(__file__).parent.parent))
import store

D = Path(__file__).parent.parent.parent / "data" / "tetris"
SIZE, G = 336, 24
KEYS = ["none", "Left", "Right", "Down", "Z (rotate)", "X (rotate)", "Space (drop)", "C (hold)"]

states = sum((json.loads((w / "states.json").read_text()) for w in sorted(D.glob("w[0-9]"))), [])
_fr = np.load(D / "frames.npz")
acts, game = _fr["action"], _fr["game"]
feats = store.load(D / f"feats_{SIZE}.npy")
masks = torch.from_numpy(np.load(D / f"masks_{SIZE}.npy"))
N = len(states)
assert len(acts) == N == len(feats), (len(acts), N, len(feats))


def kind(t):
    s, s1, a = states[t], states[t + 1], acts[t]
    if s1["over"] and not s["over"]:
        return "game over"
    if s1["lines"] > s["lines"]:
        return "line clear"
    if a == 7 and s1["hold"] != s["hold"]:
        return "hold"
    if s1["spawns"] > s["spawns"]:
        return "lock + new piece"
    if s1["o"] != s["o"]:
        return "rotate"
    if s1["x"] != s["x"]:
        return "move sideways"
    if s1["y"] != s["y"]:
        return "falls (gravity / soft drop)"
    return "piece still (timer only)"


trans = np.where(acts >= 0)[0]
trans = trans[[not states[t]["over"] for t in trans]]
kinds = np.array([kind(t) for t in trans])
held = np.random.default_rng(0).random(game.max() + 1) < 0.2
test, train = trans[held[game[trans]]], trans[~held[game[trans]]]
prev_ok = np.r_[False, acts[:-1] >= 0]  # frame t has a previous frame in the same game
dev = "cuda"


def get(idx):  # random frames (sorted reads) -> GPU float
    idx = np.asarray(idx); o = np.argsort(idx)
    out = np.empty((len(idx), G * G, 384), np.float16); out[o] = feats[idx[o]]
    return torch.from_numpy(out).to(dev).float()


def prev_of(b):
    b = np.asarray(b)
    return np.where(prev_ok[b], b - 1, b)


def chunk_starts(pool, L, n=None):  # transitions t where t..t+L-1 are all in the pool (same game, valid)
    n = N if n is None else n
    m = np.zeros(n + L, bool); m[pool] = True
    ok = m[:n].copy()
    for j in range(1, L):
        ok &= m[j:n + j]
    return np.where(ok)[0]


def chunk_batch(starts, rng, n_chunks=16, L=4, F=None, P=None):
    """-> transition ids b, and CPU fp16 arrays z, z1, zprev (each chunk read as ONE contiguous slice).
    F = feature array (default: Tetris), P = prev_ok array (frame t has a previous frame in the same game)."""
    F = feats if F is None else F
    P = prev_ok if P is None else P
    zs, z1s, zps, bs = [], [], [], []
    for t in rng.choice(starts, n_chunks):
        lo = t - 1 if P[t] else t
        blk = np.asarray(F[lo:t + L + 1])  # frames lo .. t+L
        off = t - lo
        zs.append(blk[off:off + L]); z1s.append(blk[off + 1:off + L + 1])
        zps.append(blk[off - 1:off - 1 + L] if off else np.concatenate([blk[:1], blk[:L - 1]]))
        bs.append(np.arange(t, t + L))
    return np.concatenate(bs), np.concatenate(zs), np.concatenate(z1s), np.concatenate(zps)


def prefetch(make, n=3):
    """Run make() in a background thread, keep n batches ready."""
    q = queue.Queue(n)
    threading.Thread(target=lambda: [q.put(make()) for _ in iter(int, 1)], daemon=True).start()
    while True:
        yield q.get()


# board reader geometry: 10 x 20 cells of 29 px inside the 620 px crop, top-left at (225, 25)
CELLS = [(225 + 29 * c + 14.5, 25 + 29 * r + 14.5) for r in range(20) for c in range(10)]
PIDX = torch.tensor([int(y * SIZE / 620 // 14) * G + int(x * SIZE / 620 // 14) for x, y in CELLS], device=dev)
board = lambda t: torch.tensor(sum(visible_cells(states[t]), []), dtype=torch.float32)
