"""Level A1 (predict): IMAGINER on the Settings app - clicks AND drags (slider), menus, dark mode, font size.

Like tic-tac-toe Level 0a: each unique app state has one screenshot; 30% of states are held out (never seen in training,
not even as a target). Test = actions starting from held-out states. Board accuracy -> "settings accuracy": the
imagined screen decoded to the nearest known state screen (a measuring tool, as in 0a), vs the true next state.

usage: python level0/settings_train.py [STEPS=6000]
"""
import json, sys
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
from PIL import Image
from model import Imaginer, Eyes, dev

D = Path(__file__).parent.parent / "data" / "settings"
R = Path(__file__).parent.parent / "results"
STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 6000
torch.manual_seed(0)
rng = np.random.default_rng(0)

states = json.loads((D / "states.json").read_text())
tr_ = np.array(json.loads((D / "transitions.json").read_text()))  # src, kind, x1, y1, x2, y2, dst
src, dst = torch.from_numpy(tr_[:, 0]).long(), torch.from_numpy(tr_[:, 6]).long()
kind = torch.from_numpy(tr_[:, 1]).long()
seg = torch.from_numpy(tr_[:, 2:6] / 224).float()
S = len(states)
imgs_np = np.stack([np.asarray(Image.open(D / "img" / f"{i}.png").convert("RGB")) for i in range(S)])
eyes = Eyes()
feats = torch.cat([eyes(imgs_np[i:i + 128]) for i in range(0, S, 128)]).half().float()
np.save(D / "feats.npy", feats.half().cpu().numpy())
imgs = torch.from_numpy(imgs_np).to(dev)
del eyes; torch.cuda.empty_cache()

held = torch.from_numpy(rng.random(S) < 0.3)
held[0] = False  # default screen stays in training
tr = torch.where(~held[src] & ~held[dst])[0]
te = torch.where(held[src])[0]
print(f"{S} states ({int(held.sum())} held out) | train {len(tr)} | test {len(te)} actions", flush=True)


def what(t):  # what kind of change this action made (for the report)
    a, b = states[int(src[t])], states[int(dst[t])]
    if a["key"] == b["key"]:
        return "nothing happens"
    diff = [k for k in ("dark", "wifi", "volume", "font", "notif", "menu") if a[k] != b[k]]
    x, y = float(seg[t, 0]) * 224, float(seg[t, 1]) * 224
    if kind[t] == 1 and not a["menu"] and 150 <= x <= 216 and 196 <= y <= 218:
        return "reset"
    if "menu" in diff:
        return "pick from menu" if a["menu"] else "open menu"
    return {"dark": "dark mode", "wifi": "wi-fi", "volume": "volume slider", "font": "font", "notif": "notifications"}[diff[0]]


change_mask = lambda a, b: (imgs[a] != imgs[b]).any(-1).view(-1, 16, 14, 16, 14).any(4).any(2).flatten(1).float()
model = Imaginer(keys=2, segments=True).to(dev)
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
torch.save({"model": model.state_dict(), "held": held}, D / "imaginer_a1.pt")
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


near = run(seg[ev], kind[ev])
near_sh = run(seg[ev][perm], kind[ev][perm])  # wrong action
truth, start = dst[ev], src[ev]
types = np.array([what(t) for t in ev.tolist()])
res = {"n_test": len(ev), "groups": {}}
print("\n=== A1 Settings on held-out app states: imagined screen = correct next state? ===")
print(f"{'action effect':<18} {'n':>5} {'model':>7} {'wrong action':>13} {'nothing changes':>16}")
for g in ["dark mode", "wi-fi", "notifications", "volume slider", "open menu", "pick from menu", "font", "reset", "nothing happens"]:
    sel = torch.from_numpy(types == g)
    if not sel.any():
        continue
    row = {"n": int(sel.sum()), "model": (near[sel] == truth[sel]).float().mean().item(),
           "wrong_action": (near_sh[sel] == truth[sel]).float().mean().item(), "nothing_changes": (start[sel] == truth[sel]).float().mean().item()}
    res["groups"][g] = row
    print(f"{g:<18} {row['n']:>5} {row['model']:>7.1%} {row['wrong_action']:>13.1%} {row['nothing_changes']:>16.1%}")
ch = torch.from_numpy(types != "nothing happens")
res["changed"] = {"model": (near[ch] == truth[ch]).float().mean().item(), "wrong_action": (near_sh[ch] == truth[ch]).float().mean().item()}
print(f"{'ALL changes':<18} {int(ch.sum()):>5} {res['changed']['model']:>7.1%} {res['changed']['wrong_action']:>13.1%}")
ok = res["changed"]["model"] >= 0.9 and res["changed"]["model"] >= res["changed"]["wrong_action"] + 0.3
print(f"\nKILL TEST: {'PASS' if ok else 'FAIL'}  (needs >= 90% correct next state on changes and +30 pts over a wrong action)")
(R / "settings_a1_metrics.json").write_text(json.dumps(res, indent=2))
