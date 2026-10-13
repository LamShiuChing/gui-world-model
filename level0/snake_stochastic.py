"""Level 3b: STOCHASTIC IMAGINER on Snake - imagine several sharp futures instead of one blurred average.

MultiImaginer: 64 possible futures + how likely each is. Trained winner-takes-all (only the closest future is corrected).
Check with a FOOD READER (linear probe trained on REAL screens: which of the 64 cells has the food?) applied to imagined screens:
  - P(true cell): probability the model gives to where the food really appeared (uniform guess = 1/free cells)
  - sharpness  : does ONE imagined future show ONE clear food? (reader's top probability)
  - valid      : is the imagined food on an empty cell (not on the snake)?
Same held-out games as Level 3a; compared with the 3a deterministic IMAGINER.

Step 4 (mode "cvae"): the winner-takes-all futures collapsed to ~3 food cells. CVAE-style fix: a POSTERIOR sees what
the base prediction missed (true next screen - base) and picks ONE of the 64 codes (Gumbel-softmax, straight-through);
the future is decoded from that code; the PRIOR (screen + key only) learns to predict the posterior's code (KL).
At test time futures come from the prior, as before - so the codes have to carry "where the food went".

usage: python level0/snake_stochastic.py [STEPS=8000] [wta|cvae]
"""
import json, sys
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image, ImageDraw
from model import Imaginer, MultiImaginer, dev
from snake_data import D, KEYS, states, acts, T, kinds, train, test
import store

R = Path(__file__).parent.parent / "results"
STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
K, B = 64, 64
MODE = sys.argv[2] if len(sys.argv) > 2 else "wta"
TAG = "" if MODE == "wta" else "_cvae"
BETA = 0.02  # KL weight (cvae)
torch.manual_seed(0)
torch.backends.cuda.matmul.allow_tf32 = True

feats = torch.from_numpy(store.load(D / "feats.npy", mmap=False))  # T+1,256,384 fp16, kept in RAM (7.9 GB)
masks = torch.from_numpy(np.load(D / "masks.npy")).to(dev).float()
get = lambda idx: feats[idx].to(dev, non_blocking=True).float()
EAT = "eat (food reappears at random)"

# ---------------------------------------------------------------- train the stochastic IMAGINER
model = MultiImaginer(K=K, keys=6, concat=MODE == "cvae").to(dev)
# v2: start from the working 3a IMAGINER (it already handles everything predictable); the 64 futures only learn the
# LEFTOVER (base prediction is detached for them), so they compete on what 3a could not predict - mostly the random food.
missing = model.load_state_dict(torch.load(D / "imaginer.pt"), strict=False).missing_keys
print(f"started from 3a IMAGINER; new parts: {sorted({k.split('.')[0] for k in missing})}")
post = nn.Sequential(nn.LayerNorm(384), nn.Linear(384, 256), nn.GELU(), nn.Linear(256, K)).to(dev)  # per patch -> code logits
post_pos = nn.Parameter(torch.randn(1, 256, 384, device=dev) * 0.5)  # v3: WHERE the leftover is (v1/v2 had no position -> 1 food cell)
opt = torch.optim.AdamW(list(model.parameters()) + list(post.parameters()) + [post_pos], 2e-4, weight_decay=0.01)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-4, total_steps=STEPS, pct_start=0.05)
rho = masks[train].mean().item()
lam = min((1 - rho) / rho, 50.0)
wmse = lambda p, t, w: ((p - t).pow(2).mean(-1) * w).sum(-1) / w.sum(-1)
# v3: half of every batch comes from the 10% of training steps where the 3a IMAGINER was most SURPRISED (largest error).
# Generic "practise what surprised you" - nothing about food is hand-coded; rare random events land there by themselves.
with torch.no_grad():
    surprise = []
    for i in range(0, len(train), 256):
        bb = train[i:i + 256]
        with torch.autocast("cuda", torch.bfloat16):
            z0 = get(bb)
            p0 = z0 + model.delta(model.features(z0, k=acts[bb].to(dev)))
        surprise.append(wmse(p0.float(), get(bb + 1), 1 + lam * masks[bb.to(dev)]).cpu())
    surprise = torch.cat(surprise)
hard = train[surprise >= surprise.quantile(0.9)]
print(f"surprising steps: {len(hard)}; of them 'eat': {np.mean(kinds[hard.numpy()] == EAT):.0%} "
      f"(vs {np.mean(kinds[train.numpy()] == EAT):.1%} of all steps)")
for step in range(STEPS):
    b = torch.cat([train[torch.randint(len(train), (B // 2,))], hard[torch.randint(len(hard), (B // 2,))]])
    z, z1, m, k = get(b), get(b + 1), masks[b.to(dev)], acts[b].to(dev)
    with torch.autocast("cuda", torch.bfloat16):
        h = model.features(z, k=k)
        base = z + model.delta(h)
        hd = h.detach()
        logit, prior = model.mask(h).squeeze(-1), model.prior(hd.mean(1))
        if MODE == "cvae":
            r = (z1 - base.detach()).float()  # what the base prediction missed
            att = (2 * (r.pow(2).mean(-1) + 1e-6).log()).softmax(1)  # look where the miss is biggest (~ energy^2)
            ql = (att[..., None] * post(r + post_pos).float()).sum(1)  # B,K: which code explains the leftover
            oh = F.gumbel_softmax(ql, tau=1.0, hard=True)
            fut = base.detach() + model.head(model.mix(hd, (oh.to(hd.dtype) @ model.code.weight.to(hd.dtype))[:, None]))
        else:
            fut = base.detach()[:, None] + model.head(model.mix(hd[:, None], model.code.weight[None, :, None]))
    w = 1 + lam * m
    loss_base = wmse(base.float(), z1, w).mean()
    if MODE == "cvae":
        L = wmse(fut.float(), z1, w)[:, None]  # B,1 (the posterior's code)
        best = oh.argmax(1)
        q = ql.softmax(-1)
        kl = (q * (q.clamp_min(1e-8).log() - prior.float().log_softmax(-1))).sum(-1).mean()
        loss = loss_base + L.mean() + BETA * kl + F.binary_cross_entropy_with_logits(logit.float(), m)
    else:
        L = wmse(fut.float(), z1[:, None], w[:, None])  # B,K
        best = L.argmin(1)
        loss = loss_base + L.gather(1, best[:, None]).mean() + 0.01 * L.mean() + F.cross_entropy(prior.float(), best)             + F.binary_cross_entropy_with_logits(logit.float(), m)
    opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
    if step % 1000 == 0 or step == STEPS - 1:
        print(f"step {step:>5}  base {loss_base.item():.4f}  best-future {(L.mean() if MODE == 'cvae' else L.gather(1, best[:, None]).mean()).item():.4f}"
              f"  futures used in batch {len(best.unique())}", flush=True)
torch.save(model.state_dict(), D / f"imaginer_multi{TAG}.pt")
model.eval()

# ---------------------------------------------------------------- FOOD READER: real screen latent -> which cell has the food
PIDX = torch.tensor([((38 + 24 * (c // 8)) // 14) * 16 + (28 + 24 * (c % 8)) // 14 for c in range(64)], device=dev)
reader = nn.Linear(384, 1).to(dev)
ropt = torch.optim.Adam(reader.parameters(), 1e-3)
fr = [t for t in train.tolist() if states[t]["food"]]
food_cell = lambda t: states[t]["food"][1] * 8 + states[t]["food"][0]
read = lambda zz: reader(zz[:, PIDX]).squeeze(-1)  # (N,64) logits
for _ in range(1500):
    b = torch.tensor(np.random.choice(fr, 256))
    loss = F.cross_entropy(read(get(b)), torch.tensor([food_cell(t) for t in b.tolist()], device=dev))
    ropt.zero_grad(); loss.backward(); ropt.step()
ft = [t + 1 for t in test.tolist() if states[t + 1]["food"]][:3000]
with torch.no_grad():
    racc = (read(get(torch.tensor(ft))).argmax(1).cpu() == torch.tensor([food_cell(t) for t in ft])).float().mean().item()
print(f"food reader on REAL held-out screens: {racc:.1%} correct cell")

# ---------------------------------------------------------------- evaluation on held-out games
det = Imaginer(keys=6).to(dev)
det.load_state_dict(torch.load(D / "imaginer.pt"))
det.eval()
tk = kinds[test.numpy()]
err_det, err_map, err_best, err_id = [], [], [], []
eat_rows = []
with torch.no_grad():
    for i in range(0, len(test), 32):
        b = test[i:i + 32]
        z, z1, k = get(b), get(b + 1), acts[b].to(dev)
        with torch.autocast("cuda", torch.bfloat16):
            pd, _ = det(z, k=k)
            fut, _, prior = model(z, k=k)
        pd, fut, p = pd.float(), fut.float(), prior.float().softmax(-1)
        e_all = (fut - z1[:, None]).pow(2).mean(-1)  # B,K,256
        err_det.append((pd - z1).pow(2).mean(-1).cpu())
        err_map.append(e_all[torch.arange(len(b)), p.argmax(1)].cpu())
        err_best.append(e_all[torch.arange(len(b)), e_all.mean(-1).argmin(1)].cpu())
        err_id.append((z - z1).pow(2).mean(-1).cpu())
        for j in np.where(tk[i:i + 32] == EAT)[0]:
            t = int(b[j])
            snake = {c[1] * 8 + c[0] for c in states[t + 1]["snake"]}
            pr_det = read(pd[j:j + 1]).softmax(-1)[0]  # 64
            pr_fut = read(fut[j]).softmax(-1)  # K,64
            truth = food_cell(t + 1)
            top = pr_fut.argmax(1)
            eat_rows.append({
                "t": t, "free": 64 - len(snake),
                "p_true_det": pr_det[truth].item(), "p_true_stoch": (p[j] @ pr_fut[:, truth]).item(),
                "sharp_det": pr_det.max().item(), "sharp_stoch": (p[j] @ pr_fut.max(1).values).item(),
                "valid_det": float(int(pr_det.argmax()) not in snake),
                "covered_det": float(int(pr_det.argmax()) == truth), "covered_stoch": float((top == truth).any().item()),
                "distinct_cells_stoch": len(set(top.tolist())),
                "valid_stoch": sum(p[j, kk].item() for kk in range(K) if int(top[kk]) not in snake),
                "map_det": pr_det.cpu().numpy(), "map_stoch": (p[j] @ pr_fut).cpu().numpy(),
                "top3": [(int(top[kk]), p[j, kk].item()) for kk in p[j].argsort(descending=True)[:3].tolist()]})
err_det, err_map, err_best, err_id = map(torch.cat, (err_det, err_map, err_best, err_id))

res = {"food_reader_real_acc": racc, "groups": {}, "eat": {}}
print("\n=== Level 3b (Snake) on held-out games: latent error vs 'nothing changes' (1.00), lower = better ===")
print(f"{'step type':<32} {'n':>5} {'3a single':>10} {'3b likeliest':>13} {'3b best of 64':>14}")
for g in ["no key: time passes", "turn", "key, no turn", EAT, "death", "restart (random new game)"]:
    sel = torch.from_numpy(tk == g)
    base = err_id[sel].sum()
    row = {"n": int(sel.sum()), **{n: (e[sel].sum() / base).item() for n, e in
                                   [("single_3a", err_det), ("likeliest_3b", err_map), ("best_of_64", err_best)]}}
    res["groups"][g] = row
    print(f"{g:<32} {row['n']:>5} {row['single_3a']:>10.3f} {row['likeliest_3b']:>13.3f} {row['best_of_64']:>14.3f}")

E = {k_: float(np.mean([r[k_] for r in eat_rows])) for k_ in eat_rows[0] if k_ not in ("t", "map_det", "map_stoch", "top3")}
E["uniform_guess"] = float(np.mean([1 / r["free"] for r in eat_rows]))
res["eat"] = E
print(f"\nWhere does the NEW FOOD appear? (n={len(eat_rows)} eat steps; reader applied to imagined screens)")
print(f"{'':<36} {'3a single':>10} {'3b 64 futures':>14}")
print(f"{'P(true food cell)  [uniform guess ' + format(E['uniform_guess'], '.3f') + ']':<36} {E['p_true_det']:>10.3f} {E['p_true_stoch']:>14.3f}")
print(f"{'sharpness (one clear food?)':<36} {E['sharp_det']:>10.3f} {E['sharp_stoch']:>14.3f}")
print(f"{'imagined food on an empty cell':<36} {E['valid_det']:>10.1%} {E['valid_stoch']:>14.1%}")
print(f"{'true cell among the imagined futures':<36} {E['covered_det']:>10.1%} {E['covered_stoch']:>14.1%}   "
      f"(64 futures show {E['distinct_cells_stoch']:.0f} different food cells on average)")

# Food is uniformly random, so the best possible P(true cell) is ~1/free cells: judge sharp + valid + diverse instead.
g_ok = all(res["groups"][g]["likeliest_3b"] <= 1.15 * res["groups"][g]["single_3a"] for g in ["no key: time passes", "turn"])
ok = E["sharp_stoch"] >= 0.8 and E["valid_stoch"] >= 0.8 and E["covered_stoch"] >= 3 * E["covered_det"] and g_ok
print(f"\nKILL TEST: {'PASS' if ok else 'FAIL'}  (needs each future sharp >= 0.8, food on empty cell >= 80%, "
      f"true cell among futures >= 3x single model, no more than 15% worse on 'time passes'/'turn')")
(R / f"level3b{TAG}_metrics.json").write_text(json.dumps(res, indent=2))

# ---------------------------------------------------------------- picture: before | true after | 3a food belief | 3b food belief + top-3 futures
def board_heat(t, pmap, col, top3=None):
    im = Image.open(D / "img" / f"{t}.png").convert("RGB")
    hm = np.zeros((224, 224), np.uint8)
    pm = pmap / (pmap.max() + 1e-9)
    for c in range(64):
        x0, y0 = 16 + 24 * (c % 8), 26 + 24 * (c // 8)
        hm[y0:y0 + 24, x0:x0 + 24] = int(pm[c] * 200)
    im = Image.composite(Image.new("RGB", (224, 224), col), im, Image.fromarray(hm))
    if top3:
        dr = ImageDraw.Draw(im)
        for r_, (c, pr) in enumerate(top3):
            x0, y0 = 16 + 24 * (c % 8), 26 + 24 * (c // 8)
            dr.rectangle([x0 + 2, y0 + 2, x0 + 21, y0 + 21], outline=(255, 0, 255), width=2)
            dr.text((x0 + 4, y0 + 6), str(r_ + 1), fill=(255, 0, 255))
    return im


rows = eat_rows[:6]
sheet = Image.new("RGB", (4 * 230, len(rows) * 230), "white")
for r_, row in enumerate(rows):
    t = row["t"]
    before = Image.open(D / "img" / f"{t}.png").convert("RGB")
    ImageDraw.Draw(before).text((4, 210), f"key: {KEYS[int(acts[t])]}", fill=(200, 0, 200))
    cols = [before, Image.open(D / "img" / f"{t + 1}.png").convert("RGB"),
            board_heat(t + 1, row["map_det"], (60, 60, 255)), board_heat(t + 1, row["map_stoch"], (0, 170, 0), row["top3"])]
    for c_, im in enumerate(cols):
        sheet.paste(im, (c_ * 230, r_ * 230))
sheet.save(R / f"level3b{TAG}_examples.png")
print(f"saved {R / 'level3b_examples.png'}  (cols: before + key | true after | 3a: where it thinks food is | "
      f"3b: food belief over all futures, boxes = 3 likeliest futures)")
