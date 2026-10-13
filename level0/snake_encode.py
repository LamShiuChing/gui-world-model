"""EYES over every Snake frame (fp16 to disk) + per-transition pixel change maps (which 14x14 patches changed)."""
import json
from pathlib import Path
import numpy as np, torch
from PIL import Image
from model import Eyes

D = Path(__file__).parent.parent / "data" / "snake"
T1 = len(json.loads((D / "states.json").read_text()))  # frames = transitions + 1
feats = np.lib.format.open_memmap(D / "feats.npy", "w+", np.float16, (T1, 256, 384))
masks = np.zeros((T1 - 1, 256), bool)
eyes = Eyes()
load = lambda i: np.asarray(Image.open(D / "img" / f"{i}.png").convert("RGB"))
prev = None
for i in range(0, T1, 256):
    imgs = np.stack([load(j) for j in range(i, min(i + 256, T1))])
    feats[i:i + len(imgs)] = eyes(imgs).half().cpu().numpy()
    seq = imgs if prev is None else np.concatenate([prev[None], imgs])
    d = (seq[1:] != seq[:-1]).any(-1).reshape(-1, 16, 14, 16, 14).any((2, 4)).reshape(-1, 256)
    masks[(i - 1 if prev is not None else 0):(i - 1 if prev is not None else 0) + len(d)] = d
    prev = imgs[-1]
    if i % 5120 == 0:
        print(f"{i}/{T1}", flush=True)
feats.flush()
np.save(D / "masks.npy", masks)
print("feats", feats.shape, "| changed-patch fraction", masks.mean().round(4))
