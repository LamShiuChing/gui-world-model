"""Gate for compressed features (store.py): does the trained Tetris IMAGINER (v2, history) predict as well when its
inputs are compressed? Held-out games; truth = the fp16 next screen. Also: is the codec's error on a TARGET screen
small next to the model's own error (so training on compressed targets loses nothing)?

PASS per codec: prediction error within +2% of fp16 inputs AND target codec error < 5% of the model error.
usage: python level0/store_gate.py
"""
import sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).parent)); sys.path.insert(0, str(Path(__file__).parent / "tetris"))
import data as T
import store
from model import Imaginer, dev

rng = np.random.default_rng(1)
model = Imaginer(keys=8, tokens=T.G * T.G, history=True).to(dev)
model.load_state_dict(torch.load(T.D / "imaginer_t0_hist.pt")); model.eval()
ev = np.sort(rng.choice(T.test, 1500, replace=False))
moved = T.kinds[np.searchsorted(T.trans, ev)] != "piece still (timer only)"
fit_rows = T.feats[np.sort(rng.choice(T.train, 2000, replace=False))]  # codecs fitted on training games only


@torch.no_grad()
def errors(codec):
    f = (lambda x: x) if codec is None else (lambda x: store.decode(store.encode(x, codec), codec))
    e, e0, et = [], [], []
    for i in range(0, len(ev), 50):
        b = ev[i:i + 50]
        z, z1, zp = (np.asarray(T.feats[j]) for j in (b, b + 1, T.prev_of(b)))
        g = lambda x: torch.from_numpy(np.asarray(x)).to(dev).float()
        with torch.autocast("cuda", torch.bfloat16):
            p, _ = model(g(f(z)), k=torch.from_numpy(T.acts[b]).long().to(dev), z_prev=g(f(zp)))
        e.append((p.float() - g(z1)).pow(2).mean((1, 2)).cpu()); e0.append((g(z) - g(z1)).pow(2).mean((1, 2)).cpu())
        et.append((g(f(z1)) - g(z1)).pow(2).mean((1, 2)).cpu())
    return torch.cat(e).numpy(), torch.cat(e0).numpy(), torch.cat(et).numpy()


base, e0, _ = errors(None)
print(f"{len(ev)} held-out Tetris steps | model error / nothing-changes: all {base.mean() / e0.mean():.3f}, "
      f"piece moved {base[moved].mean() / e0[moved].mean():.3f}")
print(f"{'codec':<14} {'bytes/token':>11} {'pred err vs fp16':>17} {'(piece moved)':>14} {'target err / model err':>23}  gate")
for k in (384, 256, 192, 128):
    c = store.fit(fit_rows, k)
    e, _, et = errors(c)
    r, rm, rt = e.mean() / base.mean() - 1, e[moved].mean() / base[moved].mean() - 1, et.mean() / base.mean()
    ok = r < 0.02 and rm < 0.02 and rt < 0.05
    print(f"{'int8' if k == 384 else f'PCA-{k} int8':<14} {k:>11} {r:>+16.2%} {rm:>+14.2%} {rt:>22.2%}   {'PASS' if ok else 'FAIL'}", flush=True)
