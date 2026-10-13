"""Tetris T0 step 1: which input size does EYES need? (decide BEFORE encoding 80k frames)

For 224 px and 336 px: encode ~3000 frames, train one shared linear reader "is this board cell filled?" on the patch token
under each of the 200 visible cells, test on held-out games. Also: does a frame-to-frame pixel diff need a threshold?
"""
import json, sys
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image
sys.path.insert(0, str(Path(__file__).parent.parent))
from model import Eyes, dev
from pieces import visible_cells
from webp import img

D = Path(__file__).parent.parent.parent / "data" / "tetris" / "w0"
states, eps = json.loads((D / "states.json").read_text()), json.loads((D / "episodes.json").read_text())
rng = np.random.default_rng(0)
frames = np.sort(rng.choice(len(states), 3000, replace=False))
test = np.array([eps[i] >= 20 for i in frames])  # later games held out
# board geometry inside the 620x620 crop: 10 x 20 cells of 29 px, top-left at (225, 25)
cells = [(225 + 29 * c + 14.5, 25 + 29 * r + 14.5) for r in range(20) for c in range(10)]
filled = np.array([sum(visible_cells(states[i]), []) for i in frames], np.float32)  # 3000 x 200, falling piece included
print(f"{len(frames)} frames, filled-cell rate {filled.mean():.1%}; test frames {test.sum()}")

imgs = [img(D / "img" / f"{i}.png").convert("RGB") for i in frames]
eyes = Eyes()
res = {}
for size in (224, 336):
    g = size // 14
    pidx = torch.tensor([int(y * size / 620 // 14) * g + int(x * size / 620 // 14) for x, y in cells], device=dev)
    feats = []
    for i in range(0, len(imgs), 64):
        batch = np.stack([np.asarray(im.resize((size, size), Image.LANCZOS)) for im in imgs[i:i + 64]])
        feats.append(eyes(batch)[:, pidx].half())  # tokens under the 200 cells
    X = torch.cat(feats).float()  # 3000,200,384
    Y = torch.from_numpy(filled).to(dev)
    tr, te = torch.from_numpy(~test).to(dev), torch.from_numpy(test).to(dev)
    reader = nn.Linear(384, 1).to(dev)
    opt = torch.optim.Adam(reader.parameters(), 1e-3)
    for _ in range(1500):
        b = torch.where(tr)[0][torch.randint(int(tr.sum()), (128,), device=dev)]
        loss = F.binary_cross_entropy_with_logits(reader(X[b]).squeeze(-1), Y[b])
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        pred = reader(X[te]).squeeze(-1) > 0
    y = Y[te] > 0.5
    acc = (pred == y).float().mean().item()
    bal = 0.5 * ((pred & y).sum() / y.sum() + (~pred & ~y).sum() / (~y).sum()).item()
    exact = (pred == y).all(1).float().mean().item()
    res[size] = {"cell_acc": acc, "balanced_acc": bal, "whole_board_exact": exact, "tokens": g * g}
    print(f"{size}px ({g * g} tokens): cell accuracy {acc:.2%}  balanced {bal:.2%}  whole board exactly right {exact:.1%}", flush=True)

# pixel-diff noise: how many 14px patches change between consecutive frames, exact vs threshold 8
a = np.asarray(img(D / "img" / "100.png").convert("RGB").resize((224, 224), Image.LANCZOS)).astype(int)
b = np.asarray(img(D / "img" / "101.png").convert("RGB").resize((224, 224), Image.LANCZOS)).astype(int)
for thr in (0, 8, 16):
    m = (np.abs(a - b).max(-1) > thr).reshape(16, 14, 16, 14).any((1, 3))
    print(f"patches changed 100->101 with pixel threshold {thr}: {m.mean():.1%}")
(D.parent / "probe_resolution.json").write_text(json.dumps(res, indent=2))
