"""Level 3a: IMAGINER on Snake. New vs tic-tac-toe: TIME (the snake moves with no key) and RANDOMNESS (new food).

IN : z_t (frozen DINOv2-S patch tokens) + key (Up/Down/Left/Right/none/Enter)
OUT: z_{t+1} prediction + "where it changes" map
Test = held-out games (never seen in training). Pass = learns time + keys; food placement is MEASURED, not required.
"""
import json, os, sys
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
from PIL import Image, ImageDraw
from model import Imaginer, dev
from snake_data import D, KEYS, states, acts, T, kinds, ep, train, test
import store
FRAC = float(os.environ.get("FRAC", 1))  # data-efficiency curve: train only on the first FRAC of the collected steps
train = train[:int(FRAC * len(train))]
TAG = "" if FRAC == 1 else f"_f{FRAC:g}"

R = Path(__file__).parent.parent / "results"
STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 6000
torch.manual_seed(0)
torch.backends.cuda.matmul.allow_tf32 = True

feats = torch.from_numpy(store.load(D / "feats.npy", mmap=False)).to(dev)  # T+1,256,384 fp16
masks = torch.from_numpy(np.load(D / "masks.npy")).to(dev).float()  # T,256
print(f"{T} transitions, {ep.max() + 1} games | train {len(train)} | test {len(test)} (held-out games)")

model = Imaginer(keys=6).to(dev)
opt = torch.optim.AdamW(model.parameters(), 3e-4, weight_decay=0.01)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-4, total_steps=STEPS, pct_start=0.05)
rho = masks[train].mean().item()
lam = min((1 - rho) / rho, 50.0)
print(f"changed-patch fraction rho={rho:.3f} -> change weight {lam:.1f}")

for step in range(STEPS):
    b = train[torch.randint(len(train), (256,))].to(dev)
    z, z1, m, k = feats[b].float(), feats[b + 1].float(), masks[b], acts[b.cpu()].to(dev)
    with torch.autocast("cuda", torch.bfloat16):
        pred, logit = model(z, k=k)
    w = 1 + lam * m
    loss_z = (w * (pred.float() - z1).pow(2).mean(-1)).sum() / w.sum()
    loss_m = F.binary_cross_entropy_with_logits(logit.float(), m)
    opt.zero_grad(set_to_none=True)
    (loss_z + loss_m).backward()
    opt.step(), sched.step()
    if step % 500 == 0 or step == STEPS - 1:
        print(f"step {step:>5}  latent {loss_z.item():.4f}  mask {loss_m.item():.4f}", flush=True)
torch.save(model.state_dict(), D / f"imaginer{TAG}.pt")

# ---------------------------------------------------------------- evaluation on held-out games
model.eval()
k_test = acts[test].to(dev)
k_shuf = k_test[torch.randperm(len(test), device=dev)]


@torch.no_grad()
def run(keys):  # per-patch squared error (N,256) and change-map probabilities (N,256)
    err, prob = [], []
    for i in range(0, len(test), 500):
        b = test[i:i + 500].to(dev)
        with torch.autocast("cuda", torch.bfloat16):
            p, lg = model(feats[b].float(), k=keys[i:i + 500])
        err.append((p.float() - feats[b + 1].float()).pow(2).mean(-1))
        prob.append(lg.float().sigmoid())
    return torch.cat(err), torch.cat(prob)


err, prob = run(k_test)
err_sh, _ = run(k_shuf)
tb = test.to(dev)
err_id = (feats[tb].float() - feats[tb + 1].float()).pow(2).mean(-1)  # "nothing changes" baseline
m_true = masks[tb]
m_pred = (prob > 0.5).float()
tk = kinds[test.numpy()]

res = {"n_test": len(test), "groups": {}}
print("\n=== Level 3a (Snake) on held-out games: latent error vs 'nothing changes' (1.00), lower = better ===")
print(f"{'step type':<32} {'n':>5} {'model':>7} {'wrong key':>10} {'change-map F1':>14}")
for g in ["no key: time passes", "turn", "key, no turn", "eat (food reappears at random)", "death",
          "restart (random new game)", "game over, idle"]:
    sel = torch.from_numpy(tk == g).to(dev)
    if not sel.any():
        continue
    base = err_id[sel].sum()
    tp = (m_pred[sel] * m_true[sel]).sum()
    f1 = (2 * tp / (m_pred[sel].sum() + m_true[sel].sum() + 1e-9)).item() if m_true[sel].sum() > 0 else None
    row = {"n": int(sel.sum()), "model": (err[sel].sum() / base).item() if base > 0 else None,
           "wrong_key": (err_sh[sel].sum() / base).item() if base > 0 else None, "mask_F1": f1,
           "predicts_no_change": (m_pred[sel].sum(1) == 0).float().mean().item()}
    res["groups"][g] = row
    fmt = lambda v: f"{v:.3f}" if v is not None else "  -"
    print(f"{g:<32} {row['n']:>5} {fmt(row['model']):>7} {fmt(row['wrong_key']):>10} {fmt(f1):>14}"
          + (f"   (predicts 'no change' {row['predicts_no_change']:.1%})" if g == "game over, idle" else ""))

# ---------------------------------------------------------------- the random part: where does the new food go?
eat = np.where(tk == "eat (food reappears at random)")[0]
food_p = torch.zeros(len(eat), 256, dtype=torch.bool, device=dev)
for r, i in enumerate(eat):
    fx, fy = states[int(test[i]) + 1]["food"]
    x0, y0 = 16 + 24 * fx + 4, 26 + 24 * fy + 4
    for py in range(y0 // 14, (y0 + 15) // 14 + 1):
        for px in range(x0 // 14, (x0 + 15) // 14 + 1):
            food_p[r, py * 16 + px] = True
e, ei, mt, pr = err[eat], err_id[eat], m_true[eat].bool(), prob[eat]
snake_p = mt & ~food_p
food_ratio = (e[food_p].sum() / ei[food_p].sum()).item()
snake_ratio = (e[snake_p].sum() / ei[snake_p].sum()).item()
board = torch.zeros(256, dtype=torch.bool, device=dev)
board.view(16, 16)[1:16, 1:15] = True  # patches over the board (rows = y, cols = x)
empty_p = board[None] & ~mt
res["eat_split"] = {"new_food_patches": food_ratio, "snake_and_score_patches": snake_ratio,
                    "mean_change_prob_on_new_food": pr[food_p].mean().item(),
                    "mean_change_prob_on_unchanged_board": pr[empty_p].mean().item()}
print(f"\nWhen the snake EATS (n={len(eat)}): error vs 'nothing changes'")
print(f"  snake + score patches (predictable) : {snake_ratio:.3f}")
print(f"  new-food patches (random)           : {food_ratio:.3f}")
print(f"  change probability it gives the real new-food spot: {res['eat_split']['mean_change_prob_on_new_food']:.3f}"
      f"  vs other unchanged board spots: {res['eat_split']['mean_change_prob_on_unchanged_board']:.3f}")

g1, g2 = res["groups"]["no key: time passes"], res["groups"]["turn"]
ok = g1["model"] < 0.5 and g2["model"] < 0.5 and g2["wrong_key"] > 2 * g2["model"]
print(f"\nKILL TEST: {'PASS' if ok else 'FAIL'}  (needs error < 0.5 on 'time passes' and 'turn', and wrong key >= 2x worse on turns)")
(R / f"level3a_metrics{TAG}.json").write_text(json.dumps(res, indent=2))

# ---------------------------------------------------------------- picture: before + key | true after | true change map | predicted change probability
def img(i):
    return Image.open(D / "img" / f"{i}.png").convert("RGB")


def overlay(i, mask, col):
    m = Image.fromarray((mask.view(16, 16).cpu().numpy() * 255).astype(np.uint8)).resize((224, 224), Image.NEAREST)
    return Image.composite(Image.new("RGB", (224, 224), col), img(i), m.point(lambda v: int(v * 0.6)))


pick = [*np.where(tk == "no key: time passes")[0][:2], *np.where(tk == "turn")[0][:2], *eat[:3],
        *np.where(tk == "death")[0][:1]]
sheet = Image.new("RGB", (4 * 230, len(pick) * 230), "white")
for r, i in enumerate(pick):
    t = int(test[i])
    before = img(t)
    ImageDraw.Draw(before).text((4, 210), f"key: {KEYS[int(acts[t])]}", fill=(200, 0, 200))
    p_ = prob[i] / prob[i].max()  # scale so faint 'maybe here' guesses are visible
    for c, im in enumerate([before, img(t + 1), overlay(t, m_true[i], (255, 60, 60)), overlay(t, p_, (60, 60, 255))]):
        sheet.paste(im, (c * 230, r * 230))
sheet.save(R / f"level3a_examples{TAG}.png")
print(f"saved {R / 'level3a_examples.png'}  (cols: before + key | true after | true change map | predicted change map, scaled)")
