"""EYES over every frame of a multi-worker recorded dataset (fp16 memmap on disk) + per-transition change maps.
Used for Tetris (and its expert data) and for Paint: data/<DATASET>/w*/{img/*.png, states.json, actions.json[, episodes.json]}.

All 8 worker folders become one frame table. For frame i: action[i] = key pressed after it (-1 = episode boundary, no
transition), game[i] = global game id. Change maps use a pixel threshold because the renderer adds faint noise
(an exact diff marks ~61% of patches as changed).

usage: python level0/tetris/encode.py [SIZE=224] [THRESHOLD=8] [DATASET=tetris]
"""
import json, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from PIL import Image
sys.path.insert(0, str(Path(__file__).parent.parent))
from model import Eyes
from webp import img

D = Path(__file__).parent.parent.parent / "data" / (sys.argv[3] if len(sys.argv) > 3 else "tetris")
SIZE = int(sys.argv[1]) if len(sys.argv) > 1 else 224
THR = int(sys.argv[2]) if len(sys.argv) > 2 else 8
G = SIZE // 14

paths, acts, games = [], [], []
for w in sorted(D.glob("w[0-9]")):
    n = len(json.loads((w / "states.json").read_text()))
    a = [x if isinstance(x, int) else 0 for x in json.loads((w / "actions.json").read_text())] + [-1]  # -1: no transition
    ep = json.loads((w / "episodes.json").read_text()) if (w / "episodes.json").exists() else [0] * n
    paths += [w / "img" / f"{i}.png" for i in range(n)]
    acts += a
    games += [int(w.name[1:]) * 1000 + e for e in ep]
N = len(paths)
print(f"{N} frames from {len(set(games))} games, {SIZE}px -> {G * G} tokens, change threshold {THR}", flush=True)

feats = np.lib.format.open_memmap(D / f"feats_{SIZE}.npy", "w+", np.float16, (N, G * G, 384))
masks = np.zeros((N, G * G), bool)
def load(p):
    im = img(p).convert("RGB")
    return np.asarray(im if im.size == (SIZE, SIZE) else im.resize((SIZE, SIZE), Image.LANCZOS))



eyes, prev = Eyes(), None
with ThreadPoolExecutor(8) as pool:
    for i in range(0, N, 256):
        imgs = np.stack(list(pool.map(load, paths[i:i + 256])))
        feats[i:i + len(imgs)] = eyes(imgs).half().cpu().numpy()
        seq = imgs if prev is None else np.concatenate([prev[None], imgs])
        d = (np.abs(seq[1:].astype(np.int16) - seq[:-1]).max(-1) > THR).reshape(-1, G, 14, G, 14).any((2, 4)).reshape(-1, G * G)
        start = i if prev is None else i - 1
        masks[start:start + len(d)] = d
        prev = imgs[-1]
        if i % 10240 == 0:
            print(f"{i}/{N}", flush=True)
feats.flush()
acts = np.array(acts)
masks[acts < 0] = False  # no transition across episode boundaries
np.save(D / f"masks_{SIZE}.npy", masks)
np.savez(D / "frames.npz", action=acts, game=np.array(games))
t = acts >= 0
print(f"done: feats {feats.shape} | changed-patch fraction on transitions {masks[t].mean():.4f} | "
      f"transitions {t.sum()}", flush=True)
