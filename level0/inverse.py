"""INVERSE model on tic-tac-toe: (screen before, screen after) -> which click happened.

1. Train INVERSE on only 10% of the labelled training clicks (pretend the rest are unlabelled screen recordings).
2. Score on held-out boards with FUNCTIONAL accuracy: does the guessed click give the same result in the real game?
   Compared: random click | trained INVERSE | IMAGINER-as-inverse (imagine all 256 clicks, pick the one matching 'after').
3. Write INVERSE's guesses for every transition -> train.py "pseudo" mode trains an IMAGINER on 10% real + 90% guessed labels.
"""
import json, sys
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
from model import Imaginer, Inverse, CENTERS, dev
from ttt_rules import step, hit
import store

D, R = Path(__file__).parent.parent / "data" / "level0", Path(__file__).parent.parent / "results"
STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
FRAC = 0.1
torch.manual_seed(0)
rng = np.random.default_rng(1)

states = json.loads((D / "states.json").read_text())
feats = torch.from_numpy(store.load(D / "feats.npy", mmap=False)).to(dev).float()
t = np.load(D / "transitions.npz")
src, dst = torch.from_numpy(t["src"]).long(), torch.from_numpy(t["dst"]).long()
ck = torch.load(D / "imaginer.pt")  # same held-out boards as train.py
held = ck["held"]
tr = torch.where(~held[src] & ~held[dst])[0]
te = torch.where(held[src])[0]
labeled = torch.from_numpy(rng.choice(tr.numpy(), int(FRAC * len(tr)), replace=False))
patch = torch.from_numpy((t["y"] // 14).astype(np.int64) * 16 + (t["x"] // 14).astype(np.int64))  # true click -> patch id
print(f"INVERSE trains on {len(labeled)} labelled clicks ({FRAC:.0%} of {len(tr)}); test {len(te)} transitions on held-out boards")

inv = Inverse().to(dev)
opt = torch.optim.AdamW(inv.parameters(), 3e-4, weight_decay=0.01)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-4, total_steps=STEPS, pct_start=0.05)
for it in range(STEPS):
    b = labeled[torch.randint(len(labeled), (256,))]
    with torch.autocast("cuda", torch.bfloat16):
        logits, _ = inv(feats[src[b].to(dev)], feats[dst[b].to(dev)])
    loss = F.cross_entropy(logits.float(), patch[b].to(dev))
    opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
    if it % 500 == 0 or it == STEPS - 1:
        print(f"step {it:>5}  loss {loss.item():.3f}", flush=True)
inv.eval()


@torch.no_grad()
def guess_inverse(idx):  # -> patch id per transition
    out = []
    for i in range(0, len(idx), 1000):
        b = idx[i:i + 1000]
        with torch.autocast("cuda", torch.bfloat16):
            out.append(inv(feats[src[b].to(dev)], feats[dst[b].to(dev)])[0].float().argmax(1).cpu())
    return torch.cat(out)


imag = Imaginer().to(dev)
imag.load_state_dict(ck["model"])
imag.eval()


@torch.no_grad()
def guess_imagination(idx):  # imagine all 256 clicks, pick the one whose imagined screen matches the real 'after'
    out = []
    for i in range(0, len(idx), 8):
        b = idx[i:i + 8]
        z = feats[src[b].to(dev)].repeat_interleave(256, 0)
        with torch.autocast("cuda", torch.bfloat16):
            p = imag(z, CENTERS.repeat(len(b), 1))[0].float()
        d = (p - feats[dst[b].to(dev)].repeat_interleave(256, 0)).pow(2).mean((1, 2)).view(len(b), 256)
        out.append(d.argmin(1).cpu())
    return torch.cat(out)


def works(idx, patches):  # functional accuracy: guessed click reproduces the real result?
    ok = []
    for i, pch in zip(idx.tolist(), patches.tolist()):
        x, y = (pch % 16) * 14 + 7, (pch // 16) * 14 + 7
        s = {k: states[int(src[i])][k] for k in ("board", "turn", "winner")}
        nxt = step(s, hit(x, y))
        ok.append("".join(c or "." for c in nxt["board"]) + nxt["turn"] + (nxt["winner"] or "-") == states[int(dst[i])]["key"])
    return np.array(ok)


ev = te[torch.randperm(len(te))[:6000]]
changed = (src[ev] != dst[ev]).numpy()
reset = ((dst[ev] == 0) & (src[ev] != dst[ev])).numpy()
methods = {"random click": torch.randint(256, (len(ev),)), "INVERSE (10% labels)": guess_inverse(ev),
           "IMAGINER as inverse (100% labels)": guess_imagination(ev)}
res = {"n_test": len(ev), "labeled": len(labeled), "functional_accuracy": {}}
print(f"\n=== Guess the click from (before, after) on held-out boards: functional accuracy ===")
print(f"{'method':<36} {'all':>7} {'changed':>8} {'reset':>7} {'no-op':>7}")
for name, g in methods.items():
    w = works(ev, g)
    row = {"all": w.mean(), "changed": w[changed].mean(), "reset": w[reset].mean(), "no-op": w[~changed].mean()}
    res["functional_accuracy"][name] = {k: float(v) for k, v in row.items()}
    print(f"{name:<36} {row['all']:>7.1%} {row['changed']:>8.1%} {row['reset']:>7.1%} {row['no-op']:>7.1%}")

# ---------------------------------------------------------------- pseudo-labels for the unlabelled 90%
g_all = guess_inverse(torch.arange(len(src)))
xy_pseudo = torch.stack([(g_all % 16) * 14 + 7, (g_all // 16) * 14 + 7], 1).float() / 224
unl = tr[~torch.isin(tr, labeled)]
pl = works(unl, g_all[unl])
res["pseudo_label_functional_accuracy_on_unlabelled_train"] = float(pl.mean())
print(f"\nINVERSE labels for the {len(unl)} 'unlabelled' training clicks: {pl.mean():.1%} functionally correct")
np.savez(D / "inverse_labels.npz", labeled=labeled.numpy(), xy_pseudo=xy_pseudo.numpy())
(R / "inverse_metrics.json").write_text(json.dumps(res, indent=2))
print("saved inverse_labels.npz -> next: python train.py 6000 small | python train.py 6000 pseudo")
