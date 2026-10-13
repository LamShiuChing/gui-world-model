"""Step 3 test: does the IMAGINER GENERALIZE to apps it has never seen, and does that grow with the NUMBER of apps?

Train on the first N generated apps (seeds 0..N-1), same model and same number of training steps for every N;
test on 100 never-seen apps (seeds 1000-1099). Score on changed steps: the imagined screen decoded to the nearest
screen OF THAT APP = the right next screen (screen-image ids, as A2); plus latent error vs "nothing changes" and vs a
wrong (shuffled) click. Features stay int8 on the GPU and are decoded per batch.

usage: python level0/tinyapps_train.py [N_APPS=10,30,100,300,1000] [STEPS=8000]
"""
import json, sys
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
from model import Imaginer, dev

D = Path(__file__).parent.parent / "data" / "tinyapps"
R = Path(__file__).parent.parent / "results"
NS = [int(n) for n in (sys.argv[1] if len(sys.argv) > 1 else "10,30,100,300,1000").split(",")]
STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 8000

q = torch.from_numpy(np.load(D / "feats.q8.npy", mmap_mode="r")[:]).to(dev)  # int8 on the GPU
c = np.load(D / "feats.q8.npz")
scale, off = (torch.from_numpy(c[k]).to(dev) for k in ("scale", "offset"))
get = lambda i: q[i].float() * scale + off
scr = np.load(D / "screens.npz")
cls = torch.from_numpy(scr["cls"]).to(dev)
tr = torch.from_numpy(np.load(D / "trans.npy")).to(dev)
app, src, dst, xy = tr[:, 0].long(), tr[:, 1].long(), tr[:, 4].long(), tr[:, 2:4] / 224
changed = cls[src] != cls[dst]
TEST = (app >= 1000) & (app < 1100)
print(f"{len(scr['app'])} screens, {len(tr)} actions, {int(changed.float().mean() * 100)}% change the screen", flush=True)

# change map target: patches whose features move a lot (no pixels stored); threshold from changed training steps
pr = torch.where(changed & ~TEST)[0][:2000]
e = torch.cat([(get(dst[pr[i:i + 500]]) - get(src[pr[i:i + 500]])).pow(2).mean(-1) for i in range(0, len(pr), 500)])
THR = 0.2 * e.mean().item()
rho = (e > THR).float().mean().item()
lam = min((1 - rho) / rho, 50.0)
print(f"change map: patch energy > {THR:.3f} -> {rho:.3f} of patches changed, weight {lam:.1f}", flush=True)

te = torch.where(TEST)[0]
te = te[torch.randperm(len(te), generator=torch.Generator().manual_seed(0))[:20000].to(dev)]
app_rows = torch.from_numpy(scr["app"]).to(dev)
rows_of = {a: torch.where(app_rows == a)[0] for a in app[te].unique().tolist()}  # test app -> its screen rows
results = {}
for N in NS:
    torch.manual_seed(0)
    pool = torch.where(app < N)[0]
    model = Imaginer().to(dev)
    opt = torch.optim.AdamW(model.parameters(), 3e-4, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-4, total_steps=STEPS, pct_start=0.05)
    for step in range(STEPS):
        b = pool[torch.randint(len(pool), (256,), device=dev)]
        z, z1 = get(src[b]), get(dst[b])
        m = ((z1 - z).pow(2).mean(-1) > THR).float()
        with torch.autocast("cuda", torch.bfloat16):
            pred, logit = model(z, xy[b])
        w = 1 + lam * m
        loss = (w * (pred.float() - z1).pow(2).mean(-1)).sum() / w.sum() + F.binary_cross_entropy_with_logits(logit.float(), m)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
    model.eval()

    @torch.no_grad()
    def score(a_xy):  # -> nearest screen of the same app is the true next screen; latent error
        ok, err = [], []
        for i in range(0, len(te), 500):
            b = te[i:i + 500]
            with torch.autocast("cuda", torch.bfloat16):
                p = model(get(src[b]), a_xy[i:i + 500])[0].float()
            err.append((p - get(dst[b])).pow(2).mean((1, 2)))
            for j, t in enumerate(b.tolist()):  # candidates: every screen of this app
                rows = rows_of[int(app[t])]
                ok.append(cls[rows[(get(rows) - p[j]).pow(2).mean((1, 2)).argmin()]] == cls[dst[t]])
        return torch.stack(ok), torch.cat(err)

    ok, err = score(xy[te])
    ok_w, err_w = score(xy[te][torch.randperm(len(te), generator=torch.Generator().manual_seed(1)).to(dev)])
    e0 = torch.cat([(get(src[te[i:i + 500]]) - get(dst[te[i:i + 500]])).pow(2).mean((1, 2)) for i in range(0, len(te), 500)])
    ch = changed[te]
    r = {"train_apps": N, "train_actions": len(pool), "right_screen_changed": ok[ch].float().mean().item(),
         "right_screen_changed_wrong_click": ok_w[ch].float().mean().item(), "right_screen_noop": ok[~ch].float().mean().item(),
         "err_vs_nochange_changed": (err[ch].mean() / e0[ch].mean()).item(), "err_wrong_click_changed": (err_w[ch].mean() / e0[ch].mean()).item()}
    results[N] = r
    print(f"N={N:>5} apps ({len(pool):>6} actions): unseen apps, right next screen on changes {r['right_screen_changed']:.1%} "
          f"(wrong click {r['right_screen_changed_wrong_click']:.1%}), no-ops {r['right_screen_noop']:.1%} | "
          f"error vs nothing-changes {r['err_vs_nochange_changed']:.3f} (wrong click {r['err_wrong_click_changed']:.3f})", flush=True)
    torch.save(model.state_dict(), D / f"imaginer_n{N}.pt")
    (R / "tinyapps_metrics.json").write_text(json.dumps(results, indent=2))
    del model, opt; torch.cuda.empty_cache()

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(4.5, 3.2))
ns = sorted(results)
ax.plot(ns, [results[n]["right_screen_changed"] for n in ns], color="#2a78d6", lw=2, marker="o", ms=8, mec="#fcfcfb", mew=2)
ax.set_xscale("log"); ax.set_ylim(0, 1); ax.grid(alpha=0.25); ax.spines[["top", "right"]].set_visible(False)
ax.set_xlabel("number of different training apps", color="#52514e"); ax.set_ylabel("right next screen, unseen apps", color="#52514e")
ax.set_title("TinyApps: more apps -> better on new apps?", color="#0b0b0b")
fig.tight_layout(); fig.savefig(R / "tinyapps_curve.png", dpi=120, facecolor="#fcfcfb")
print(f"saved {R / 'tinyapps_curve.png'}")
