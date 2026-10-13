"""Level 0a: train the IMAGINER on tic-tac-toe, then test it on boards it has NEVER seen.

IN : z_t (frozen DINOv2-S patch tokens of the screen) + click (x, y)
OUT: z_{t+1} prediction (256 x 384)  +  "where it changes" map (256 patch logits)

Pass = beats "nothing changes" and "shuffled click" on changed transitions of held-out boards.
"""
import json, os, sys
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
from PIL import Image, ImageDraw
from model import Imaginer, dev
import store

D = Path(__file__).parent.parent / "data" / "level0"
R = Path(__file__).parent.parent / "results"
STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 6000
MODE = sys.argv[2] if len(sys.argv) > 2 else "full"  # full | small (10% labelled only) | pseudo (10% real + 90% INVERSE labels)
TAG = "" if MODE == "full" else f"_{MODE}"
FRAC = float(os.environ.get("FRAC", 1))  # data-efficiency curve: train only on the first FRAC of the collected actions
TAG += "" if FRAC == 1 else f"_f{FRAC:g}"
torch.manual_seed(0)
rng = np.random.default_rng(0)
torch.backends.cuda.matmul.allow_tf32 = True

states = json.loads((D / "states.json").read_text())
feats = torch.from_numpy(store.load(D / "feats.npy", mmap=False)).to(dev).float()  # S,256,384
imgs = torch.from_numpy(np.load(D / "images.npy")).to(dev)  # S,224,224,3 uint8
t = np.load(D / "transitions.npz")
src, dst = torch.from_numpy(t["src"]).long(), torch.from_numpy(t["dst"]).long()
xy = torch.from_numpy(np.stack([t["x"], t["y"]], 1) / 224).float()
S = len(feats)

# Split by STATE: a held-out board never appears in training, not even as a target.
held = torch.from_numpy(rng.random(S) < 0.3)
held[0] = False  # keep the empty board (start of every game) in training
tr = torch.where(~held[src] & ~held[dst])[0]
te = torch.where(held[src])[0]
xy_train = xy.clone()  # click labels the IMAGINER trains on; evaluation always uses the true clicks
if MODE != "full":  # made by inverse.py
    lab = np.load(D / "inverse_labels.npz")
    if MODE == "small":
        tr = torch.from_numpy(lab["labeled"]).long()
    else:
        unl = torch.ones(len(xy), dtype=torch.bool); unl[lab["labeled"]] = False
        xy_train[unl] = torch.from_numpy(lab["xy_pseudo"]).float()[unl]
tr = tr[tr < int(FRAC * len(src))]
print(f"{S} states ({held.sum()} held out) | train {len(tr)} | test {len(te)} transitions | labels: {MODE}")


def change_mask(a, b):  # which 14x14 patches differ in pixels -> (B,256) in {0,1}
    d = (imgs[a] != imgs[b]).any(-1)
    return d.view(-1, 16, 14, 16, 14).any(4).any(2).flatten(1).float()


model = Imaginer().to(dev)
print(f"trainable params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
opt = torch.optim.AdamW(model.parameters(), 3e-4, weight_decay=0.01)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-4, total_steps=STEPS, pct_start=0.05)

probe = tr[torch.randint(len(tr), (4096,))]
rho = change_mask(src[probe].to(dev), dst[probe].to(dev)).mean().item()
lam = min((1 - rho) / rho, 50.0)  # balance changed vs unchanged patches (report 31)
print(f"changed-patch fraction rho={rho:.3f} -> change weight {lam:.1f}")

for step in range(STEPS):
    b = tr[torch.randint(len(tr), (256,))]
    s, d1, a = src[b].to(dev), dst[b].to(dev), xy_train[b].to(dev)
    z, z1, m = feats[s], feats[d1], change_mask(s, d1)
    with torch.autocast("cuda", torch.bfloat16):
        pred, logit = model(z, a)
    w = 1 + lam * m
    loss_z = (w * (pred.float() - z1).pow(2).mean(-1)).sum() / w.sum()
    loss_m = F.binary_cross_entropy_with_logits(logit.float(), m)
    opt.zero_grad(set_to_none=True)
    (loss_z + loss_m).backward()
    opt.step(), sched.step()
    if step % 500 == 0 or step == STEPS - 1:
        print(f"step {step:>5}  latent {loss_z.item():.4f}  mask {loss_m.item():.4f}", flush=True)

# ---------------------------------------------------------------- evaluation on held-out boards
model.eval()
torch.save({"model": model.state_dict(), "held": held}, D / f"imaginer{TAG}.pt")  # Level 1 acts with the full one
ev = te[torch.randperm(len(te))[:6000]]
s, d1, a = src[ev].to(dev), dst[ev].to(dev), xy[ev].to(dev)
a_shuf = a[torch.randperm(len(a), device=dev)]
flat = feats.flatten(1)  # S, 98304
fnorm = flat.pow(2).sum(1)


@torch.no_grad()
def run(acts):
    """Returns per-transition latent error, decoded board id (nearest state), and mask logits."""
    err, near, logits = [], [], []
    for i in range(0, len(s), 250):
        sl = slice(i, i + 250)
        with torch.autocast("cuda", torch.bfloat16):
            p, lg = model(feats[s[sl]], acts[sl])
        p = p.float().flatten(1)
        err.append((p - flat[d1[sl]]).pow(2).sum(1))
        near.append((fnorm[None] - 2 * p @ flat.T).argmin(1))  # nearest known screen = "what the model thinks the board is"
        logits.append(lg.float())
    return torch.cat(err), torch.cat(near), torch.cat(logits)


err, near, logits = run(a)
err_sh, near_sh, _ = run(a_shuf)
err_id = (feats[s] - feats[d1]).flatten(1).pow(2).sum(1)

changed = s != d1
win = torch.tensor([states[i]["winner"] in ("X", "O") for i in d1.tolist()], device=dev) & torch.tensor(
    [states[i]["winner"] is None for i in s.tolist()], device=dev)
reset = (d1 == 0) & changed
groups = {"all": torch.ones_like(changed), "no-op": ~changed, "changed": changed, "winning move": win, "reset": reset}

m_true = change_mask(s, d1)
m_pred = (logits > 0).float()
tp = (m_pred * m_true)[changed].sum()
f1 = (2 * tp / (m_pred[changed].sum() + m_true[changed].sum())).item()

res = {"n_test": len(s), "mask_F1_changed": f1,
       "latent_err_vs_nothing_changes": (err[changed].sum() / err_id[changed].sum()).item(),
       "latent_err_shuffled_vs_nothing_changes": (err_sh[changed].sum() / err_id[changed].sum()).item(), "board_acc": {}}
print("\n=== Level 0a results on held-out boards (never seen in training) ===")
print(f"latent error, changed transitions (nothing-changes baseline = 1.00): "
      f"model {res['latent_err_vs_nothing_changes']:.3f} | shuffled click {res['latent_err_shuffled_vs_nothing_changes']:.3f}")
print(f"change-map F1 on changed transitions: {f1:.3f}")
print(f"{'board accuracy':<14} {'n':>6} {'model':>7} {'shuffled':>9} {'nothing-changes':>16}")
for g, sel in groups.items():
    acc = lambda nr: (nr[sel] == d1[sel]).float().mean().item()
    row = {"n": int(sel.sum()), "model": acc(near), "shuffled": acc(near_sh), "nothing_changes": acc(s)}
    res["board_acc"][g] = row
    print(f"{g:<14} {row['n']:>6} {row['model']:>7.1%} {row['shuffled']:>9.1%} {row['nothing_changes']:>16.1%}")

ok = res["latent_err_vs_nothing_changes"] < 0.5 and res["board_acc"]["changed"]["model"] > res["board_acc"]["changed"]["shuffled"] + 0.3
print(f"\nKILL TEST: {'PASS' if ok else 'FAIL'}  (needs latent error < 0.5 x baseline and +30 pts board accuracy over shuffled click)")

# ---------------------------------------------------------------- picture: before | true after | model's board | true / predicted change map
R.mkdir(exist_ok=True)
(R / f"level0a_metrics{TAG}.json").write_text(json.dumps(res, indent=2))


def pic(i, mask=None, dot=None):
    im = Image.fromarray(imgs[i].cpu().numpy())
    if mask is not None:
        m = Image.fromarray((mask.view(16, 16).cpu().numpy() * 255).astype(np.uint8)).resize((224, 224), Image.NEAREST)
        im = Image.composite(Image.new("RGB", (224, 224), (255, 60, 60)), im, m.point(lambda v: v // 2))
    if dot is not None:
        x, y = dot[0] * 224, dot[1] * 224
        ImageDraw.Draw(im).ellipse([x - 5, y - 5, x + 5, y + 5], outline=(255, 0, 255), width=3)
    return im


pick = torch.cat([torch.where(win)[0][:3], torch.where(changed & ~win & ~reset)[0][:4], torch.where(~changed)[0][:1]])
sheet = Image.new("RGB", (5 * 230, len(pick) * 230), "white")
for r, j in enumerate(pick.tolist()):
    cols = [pic(s[j], dot=a[j]), pic(d1[j]), pic(near[j]), pic(s[j], m_true[j]), pic(s[j], m_pred[j])]
    for c_, im in enumerate(cols):
        sheet.paste(im, (c_ * 230, r * 230))
sheet.save(R / f"level0a_examples{TAG}.png")
print(f"saved {R / f'level0a_examples{TAG}.png'}  (cols: before+click | true after | model's predicted board | true change map | predicted change map)")
