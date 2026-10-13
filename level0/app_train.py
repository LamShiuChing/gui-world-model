"""Apps A1-A3 (predict): IMAGINER on a small deterministic app - clicks, drags, double-clicks (typing for the sheet).

Like tic-tac-toe Level 0a: each unique app state has one screenshot. Apps can have HIDDEN state (files: what sits in
closed folders), so many states share the same screenshot; everything is scored on SCREENS (image-equality classes),
the only thing the model sees. 30% of screens are held out (never seen in training, not even as a target). Test =
actions starting from held-out screens. "App accuracy": the imagined screen decoded to the nearest known screen (a
measuring tool, as in 0a) vs the true next screen, per kind of effect.
(settings_train.py is the A1-only original of this script.)

usage: python level0/app_train.py APP [STEPS=6000]        APP = settings | files | sheet
"""
import json, os, sys
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
from PIL import Image
from model import Imaginer, Eyes, dev

APP = sys.argv[1] if len(sys.argv) > 1 else "settings"
D = Path(__file__).parent.parent / "data" / APP
R = Path(__file__).parent.parent / "results"
STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 6000
NKEYS = 4  # action kinds: 0 drag, 1 click, 2 double-click, 3 type
torch.manual_seed(0)
rng = np.random.default_rng(0)

states = json.loads((D / "states.json").read_text())
tr_ = np.array(json.loads((D / "transitions.json").read_text()))  # src, kind, x1, y1, x2, y2, dst
src, dst = torch.from_numpy(tr_[:, 0]).long(), torch.from_numpy(tr_[:, 6]).long()
kind = torch.from_numpy(tr_[:, 1]).long()
seg = torch.from_numpy(tr_[:, 2:6] / 224).float()
S = len(states)
imgs_np = np.stack([np.asarray(Image.open(D / "img" / f"{i}.png").convert("RGB")) for i in range(S)])
cls = torch.from_numpy(np.unique([x.tobytes() for x in imgs_np], return_inverse=True)[1]).long()  # state -> screen id
C = int(cls.max()) + 1
eyes = Eyes()
feats = torch.cat([eyes(imgs_np[i:i + 128]) for i in range(0, S, 128)]).half().float()
np.save(D / "feats.npy", feats.half().cpu().numpy())
imgs = torch.from_numpy(imgs_np).to(dev)
del eyes; torch.cuda.empty_cache()

held_c = torch.from_numpy(rng.random(C) < 0.3)
held_c[cls[0]] = False  # default screen stays in training
held = held_c[cls]  # per state
tr = torch.where(~held[src] & ~held[dst])[0]
te = torch.where(held[src])[0]
FRAC = float(os.environ.get("FRAC", 1))  # data-efficiency curve: train only on the first FRAC of the collected actions
tr = tr[tr < int(FRAC * len(src))]
TAG = "" if FRAC == 1 else f"_f{FRAC:g}"
print(f"{S} states = {C} distinct screens ({int(held_c.sum())} held out) | train {len(tr)} | test {len(te)} actions", flush=True)


def what(t):  # name of the effect an action had (for the report), from which state fields changed
    a, b = states[int(src[t])], states[int(dst[t])]
    if cls[src[t]] == cls[dst[t]]:
        return "nothing happens"  # on screen (hidden state may still change)
    diff = [k for k in a if k != "key" and a[k] != b[k]]
    if APP == "settings":
        x, y = float(seg[t, 0]) * 224, float(seg[t, 1]) * 224
        if kind[t] == 1 and not a["menu"] and 150 <= x <= 216 and 196 <= y <= 218:
            return "reset"
        if "menu" in diff:
            return "pick from menu" if a["menu"] else "open menu"
        return {"dark": "dark mode", "wifi": "wi-fi", "volume": "volume slider", "font": "font", "notif": "notifications"}[diff[0]]
    if APP == "files":
        if "loc" in diff:
            return "move file (drag)"
        if "preview" in diff:
            return "open preview (dbl-click)" if b["preview"] >= 0 else "close preview"
        return "open folder"
    return " + ".join(sorted(diff))


change_mask = lambda a, b: (imgs[a] != imgs[b]).any(-1).view(-1, 16, 14, 16, 14).any(4).any(2).flatten(1).float()
model = Imaginer(keys=NKEYS, segments=True).to(dev)
opt = torch.optim.AdamW(model.parameters(), 3e-4, weight_decay=0.01)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-4, total_steps=STEPS, pct_start=0.05)
probe = tr[torch.randint(len(tr), (4096,))]
rho = change_mask(src[probe].to(dev), dst[probe].to(dev)).mean().item()
lam = min((1 - rho) / rho, 50.0)
print(f"changed-patch fraction rho={rho:.3f} -> change weight {lam:.1f}", flush=True)
for step in range(STEPS):
    b = tr[torch.randint(len(tr), (256,))]
    s_, d_ = src[b].to(dev), dst[b].to(dev)
    z, z1, m = feats[s_], feats[d_], change_mask(s_, d_)
    with torch.autocast("cuda", torch.bfloat16):
        pred, logit = model(z, k=kind[b].to(dev), s=seg[b].to(dev))
    w = 1 + lam * m
    loss = (w * (pred.float() - z1).pow(2).mean(-1)).sum() / w.sum() + F.binary_cross_entropy_with_logits(logit.float(), m)
    opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
    if step % 1000 == 0 or step == STEPS - 1:
        print(f"step {step:>5}  loss {loss.item():.4f}", flush=True)
torch.save({"model": model.state_dict(), "held": held, "cls": cls}, D / f"imaginer_app{TAG}.pt")
model.eval()

# ---------------------------------------------------------------- evaluation on held-out states
ev = te[torch.randperm(len(te))[:6000]]
flat = feats.flatten(1); fnorm = flat.pow(2).sum(1)
perm = torch.randperm(len(ev))


@torch.no_grad()
def run(sg, kd):
    near = []
    for i in range(0, len(ev), 500):
        b = ev[i:i + 500]
        with torch.autocast("cuda", torch.bfloat16):
            p, _ = model(feats[src[b].to(dev)], k=kd[i:i + 500].to(dev), s=sg[i:i + 500].to(dev))
        near.append((fnorm[None] - 2 * p.float().flatten(1) @ flat.T).argmin(1).cpu())
    return torch.cat(near)


near = cls[run(seg[ev], kind[ev])]  # compare screens, not hidden states
near_sh = cls[run(seg[ev][perm], kind[ev][perm])]  # wrong action
truth, start = cls[dst[ev]], cls[src[ev]]
types = np.array([what(t) for t in ev.tolist()])
res = {"n_test": len(ev), "groups": {}}
print(f"\n=== {APP} app on held-out screens: imagined screen = correct next screen? ===")
print(f"{'action effect':<26} {'n':>5} {'model':>7} {'wrong action':>13} {'nothing changes':>16}")
for g in sorted(set(types), key=lambda g: (g == "nothing happens", g)):
    sel = torch.from_numpy(types == g)
    row = {"n": int(sel.sum()), "model": (near[sel] == truth[sel]).float().mean().item(),
           "wrong_action": (near_sh[sel] == truth[sel]).float().mean().item(), "nothing_changes": (start[sel] == truth[sel]).float().mean().item()}
    res["groups"][g] = row
    print(f"{g:<26} {row['n']:>5} {row['model']:>7.1%} {row['wrong_action']:>13.1%} {row['nothing_changes']:>16.1%}")
ch = torch.from_numpy(types != "nothing happens")
res["changed"] = {"model": (near[ch] == truth[ch]).float().mean().item(), "wrong_action": (near_sh[ch] == truth[ch]).float().mean().item()}
print(f"{'ALL changes':<26} {int(ch.sum()):>5} {res['changed']['model']:>7.1%} {res['changed']['wrong_action']:>13.1%}")
ok = res["changed"]["model"] >= 0.9 and res["changed"]["model"] >= res["changed"]["wrong_action"] + 0.3
print(f"\nKILL TEST: {'PASS' if ok else 'FAIL'}  (needs >= 90% correct next state on changes and +30 pts over a wrong action)")
(R / f"app_{APP}_metrics{TAG}.json").write_text(json.dumps(res, indent=2))
