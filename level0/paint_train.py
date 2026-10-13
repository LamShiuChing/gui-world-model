"""Paint P1 (predict): IMAGINER with DRAG STROKES. screen + stroke/click (x1,y1,x2,y2) -> next screen. Held-out worker.

Checks on held-out data:
  - latent error vs 'nothing changes' per step type, and with a WRONG stroke (does it use the stroke?)
  - WHERE: overlap (IoU) of predicted vs true change map on strokes that draw
  - COLOUR: a colour reader trained on REAL screens (mean token over the drawn patches -> 4 colours) applied to the
    IMAGINED screen at the same patches: does the imagined line have the right colour (= the selected swatch)?

usage: python level0/paint_train.py [STEPS=8000]   (STEPS=0: evaluate the saved model)
"""
import json, sys
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image, ImageDraw
from model import Imaginer, dev
import store
from webp import img

D = Path(__file__).parent.parent / "data" / "paint"
R = Path(__file__).parent.parent / "results"
STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
torch.manual_seed(0)
rng = np.random.default_rng(0)

feats = store.load(D / "feats_224.npy")
masks = torch.from_numpy(np.load(D / "masks_224.npy"))
fr = np.load(D / "frames.npz")
valid, worker = fr["action"] >= 0, fr["game"] // 1000
seg, kind, color, paths = [], [], [], []
for w in sorted(D.glob("w[0-9]")):
    a, st = json.loads((w / "actions.json").read_text()), json.loads((w / "states.json").read_text())
    for x in a + [[1, 0, 0, 0, 0]]:  # last frame of a worker: placeholder (no transition)
        seg.append([x[1], x[2], x[3], x[4]] if x[0] == 0 else [x[1], x[2], x[1], x[2]])
        kind.append(x[0])
    color += [s["color"] for s in st]
    paths += [w / "img" / f"{i}.png" for i in range(len(st))]
seg = torch.tensor(seg, dtype=torch.float32) / 224
kind, color = torch.tensor(kind), np.array(color)
assert len(seg) == len(feats) == len(color), (len(seg), len(feats), len(color))
canvas = torch.zeros(16, 16, dtype=torch.bool); canvas[2:, :] = True  # patch rows of the canvas (starts at y=30 -> row 2)
canvas = canvas.flatten()


def step_type(t):
    drew = bool(masks[t][canvas].any())
    if kind[t] == 0:
        return "stroke draws" if drew else "stroke does nothing"
    if color[t + 1] != color[t]:
        return "colour pick"
    x, y = seg[t, 0] * 224, seg[t, 1] * 224
    if 150 <= x <= 216 and 4 <= y <= 24 and drew:
        return "clear canvas"
    return "click, nothing happens"


trans = np.where(valid)[0]
types = np.array([step_type(t) for t in trans])
test_w = worker.max()  # hold out the last worker's whole sequence
train, test = trans[worker[trans] != test_w], trans[worker[trans] == test_w]
print(f"{len(trans)} transitions | train {len(train)} | test {len(test)} (held-out worker {test_w})", flush=True)
for k_, n_ in zip(*np.unique(types, return_counts=True)):
    print(f"   {k_:<26} {n_:>6}")


def get(idx):
    idx = np.asarray(idx); o = np.argsort(idx)
    out = np.empty((len(idx), 256, 384), np.float16); out[o] = feats[idx[o]]
    return torch.from_numpy(out).to(dev).float()


model = Imaginer(keys=2, segments=True).to(dev)
opt = torch.optim.AdamW(model.parameters(), 3e-4, weight_decay=0.01)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-4, total_steps=max(STEPS, 1), pct_start=0.05)
rho = masks[train].float().mean().item()
lam = min((1 - rho) / rho, 50.0)
print(f"changed-patch fraction rho={rho:.3f} -> change weight {lam:.1f}", flush=True)
if STEPS == 0:  # evaluate the saved model only
    model.load_state_dict(torch.load(D / "imaginer_p1.pt"))
for step in range(STEPS):
    b = rng.choice(train, 128)
    z, z1, m = get(b), get(b + 1), masks[b].float().to(dev)
    with torch.autocast("cuda", torch.bfloat16):
        pred, logit = model(z, k=kind[b].to(dev), s=seg[b].to(dev))
    w = 1 + lam * m
    loss = (w * (pred.float() - z1).pow(2).mean(-1)).sum() / w.sum() + F.binary_cross_entropy_with_logits(logit.float(), m)
    opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
    if step % 1000 == 0 or step == STEPS - 1:
        print(f"step {step:>5}  loss {loss.item():.4f}", flush=True)
if STEPS:
    torch.save(model.state_dict(), D / "imaginer_p1.pt")
model.eval()

# ---------------------------------------------------------------- COLOUR READER on real screens: drawn patches -> which colour
draws_tr = train[types[np.searchsorted(trans, train)] == "stroke draws"]
reader = nn.Linear(384, 4).to(dev)
ropt = torch.optim.Adam(reader.parameters(), 1e-3)
pool = lambda zz, m: (zz * m[..., None]).sum(1) / m.sum(1, keepdim=True).clamp(min=1)  # mean token over drawn patches
for _ in range(1500):
    b = rng.choice(draws_tr, 256)
    m = (masks[b] & canvas).float().to(dev)
    loss = F.cross_entropy(reader(pool(get(b + 1), m)), torch.from_numpy(color[b]).long().to(dev))
    ropt.zero_grad(); loss.backward(); ropt.step()

# ---------------------------------------------------------------- evaluation on the held-out worker
ev = np.sort(rng.choice(test, min(6000, len(test)), replace=False))
et = types[np.searchsorted(trans, ev)]
s_true = seg[ev].to(dev)
s_wrong = s_true[torch.randperm(len(ev), device=dev)]
k_ev = kind[ev].to(dev)
E = {n: [] for n in ("err", "err_w", "err_id", "mask", "col", "col_real")}
with torch.no_grad():
    for i in range(0, len(ev), 128):
        b = ev[i:i + 128]
        z, z1 = get(b), get(b + 1)
        with torch.autocast("cuda", torch.bfloat16):
            p, lg = model(z, k=k_ev[i:i + 128], s=s_true[i:i + 128])
            pw, _ = model(z, k=k_ev[i:i + 128], s=s_wrong[i:i + 128])
        p, pw = p.float(), pw.float()
        E["err"].append((p - z1).pow(2).mean((1, 2)).cpu()); E["err_w"].append((pw - z1).pow(2).mean((1, 2)).cpu())
        E["err_id"].append((z - z1).pow(2).mean((1, 2)).cpu()); E["mask"].append((lg.float() > 0).cpu())
        m = (masks[b] & canvas).float().to(dev)
        E["col"].append(reader(pool(p, m)).argmax(1).cpu()); E["col_real"].append(reader(pool(z1, m)).argmax(1).cpu())
err, err_w, err_id, mp = (torch.cat(E[n]) for n in ("err", "err_w", "err_id", "mask"))
col, col_real = torch.cat(E["col"]).numpy(), torch.cat(E["col_real"]).numpy()
m_true = masks[ev]

res = {"n_test": len(ev), "groups": {}}
print("\n=== Paint P1 on the held-out worker: latent error vs 'nothing changes' (1.00), lower = better ===")
print(f"{'step type':<26} {'n':>5} {'model':>7} {'wrong stroke':>13} {'change-map IoU':>15}")
for g_ in ["stroke draws", "colour pick", "clear canvas", "stroke does nothing", "click, nothing happens"]:
    sel = torch.from_numpy(et == g_)
    if not sel.any():
        continue
    base = err_id[sel].sum()
    inter, union = (mp[sel] & m_true[sel]).sum().item(), (mp[sel] | m_true[sel]).sum().item()
    row = {"n": int(sel.sum()), "model": (err[sel].sum() / base).item() if base > 0 else None,
           "wrong_stroke": (err_w[sel].sum() / base).item() if base > 0 else None, "iou": inter / union if union else None}
    res["groups"][g_] = row
    f = lambda v: f"{v:.3f}" if v is not None else "  -"
    print(f"{g_:<26} {row['n']:>5} {f(row['model']):>7} {f(row['wrong_stroke']):>13} {f(row['iou']):>15}")
dr = et == "stroke draws"
truth = color[ev]
res["colour"] = {"imagined": float((col[dr] == truth[dr]).mean()), "real_next_screen": float((col_real[dr] == truth[dr]).mean())}
print(f"\nColour of the newly drawn line, read from the IMAGINED screen: {res['colour']['imagined']:.1%} "
      f"(reader on the REAL next screen: {res['colour']['real_next_screen']:.1%}; chance 25%)")
g = res["groups"]["stroke draws"]
ok = g["model"] < 0.5 and g["wrong_stroke"] >= 2 * g["model"] and g["iou"] >= 0.5 and res["colour"]["imagined"] >= 0.8
print(f"\nKILL TEST: {'PASS' if ok else 'FAIL'}  (needs stroke error < 0.5, wrong stroke >= 2x worse, change-map IoU >= 0.5, colour >= 80%)")
(R / "paint_p1_metrics.json").write_text(json.dumps(res, indent=2))

# ---------------------------------------------------------------- picture: before + stroke | true after | true change | predicted change
def overlay(i, mask, col_):
    im = img(paths[i]).convert("RGB")
    mm = Image.fromarray((mask.view(16, 16).numpy() * 255).astype(np.uint8)).resize((224, 224), Image.NEAREST)
    return Image.composite(Image.new("RGB", (224, 224), col_), im, mm.point(lambda v: v // 2))


pick = [i for g_ in ["stroke draws", "stroke draws", "stroke draws", "colour pick", "clear canvas"] for i in np.where(et == g_)[0][:1 + (g_ == "stroke draws") * 2]]
pick = list(dict.fromkeys(pick))[:6]
sheet = Image.new("RGB", (4 * 230, len(pick) * 230), "white")
for r_, i in enumerate(pick):
    t = int(ev[i]); s_ = (seg[t] * 224).tolist()
    b4 = Image.open(paths[t]).convert("RGB")
    d = ImageDraw.Draw(b4)
    d.line(s_, fill=(255, 0, 255), width=2); d.ellipse([s_[2] - 4, s_[3] - 4, s_[2] + 4, s_[3] + 4], outline=(255, 0, 255), width=2)
    for c_, im in enumerate([b4, Image.open(paths[t + 1]).convert("RGB"), overlay(t, m_true[i], (255, 60, 60)), overlay(t, mp[i], (60, 60, 255))]):
        sheet.paste(im, (c_ * 230, r_ * 230))
sheet.save(R / "paint_p1_examples.png")
print(f"saved {R / 'paint_p1_examples.png'}  (cols: before + stroke (magenta, circle = end) | true after | true change | predicted change)")
