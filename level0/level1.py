"""Level 1: ACT BY PREDICTION ONLY, one step.

IN : current screen + goal screen (one click away), both seen through frozen EYES
DO : IMAGINER imagines all 256 clicks (one per patch centre); click the one whose
     imagined screen is closest to the goal, in the REAL browser
OUT: success = real next state == goal state

No policy is trained: acting = prediction + argmin. Start boards are held-out (never seen in training).
Baselines: random click; "click where the goal looks most different" (pixel diff).
"""
import io, json, random, sys, time
from pathlib import Path
import numpy as np, torch
from PIL import Image, ImageDraw
from playwright.sync_api import sync_playwright
from model import Imaginer, Eyes, CENTERS, dev
from collect import key
import store

ROOT = Path(__file__).parent
D, R = ROOT.parent / "data" / "level0", ROOT.parent / "results"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
random.seed(1)

states = json.loads((D / "states.json").read_text())
ids = {s["key"]: i for i, s in enumerate(states)}
feats = store.load(D / "feats.npy", mmap=False)
ck = torch.load(D / "imaginer.pt")
held = ck["held"]
model = Imaginer().to(dev)
model.load_state_dict(ck["model"])
model.eval()
eyes = Eyes()

CELL = [(52 + 60 * (j % 3), 54 + 60 * (j // 3)) for j in range(9)]  # cell centres, px
RESET = (112, 214)
CAND = (CENTERS * 224).tolist()  # the 256 clicks the model may choose from, px
starts = [i for i in range(len(states)) if held[i]]
starts = random.sample(starts, min(N, len(starts)))


@torch.no_grad()
def choose(z0, zg):  # imagine every candidate click, return (best index, distance-to-goal map)
    with torch.autocast("cuda", torch.bfloat16):
        pred, _ = model(z0.expand(256, -1, -1), CENTERS)
    dist = (pred.float() - zg).pow(2).mean((1, 2))
    return int(dist.argmin()), dist


rows, ex, ms = [], [], []
with sync_playwright() as p:
    page = p.chromium.launch().new_page(viewport={"width": 224, "height": 224})
    page.goto((ROOT / "tictactoe.html").as_uri())
    shot = lambda: np.asarray(Image.open(io.BytesIO(page.screenshot())).convert("RGB"))
    set_state = lambda st: page.evaluate("st => { s = st; render(); }", st)

    def click_from(st, xy):  # put the real app in state st, click, return the real next state
        set_state(st)
        page.mouse.click(*xy)
        return page.evaluate("window.__state")

    for n, i in enumerate(starts):
        st = {k: states[i][k] for k in ("board", "turn", "winner")}
        targets = [CELL[j] for j in range(9) if not st["winner"] and not st["board"][j]] + [RESET]
        tgt = random.choice(targets)
        set_state(st)
        o = shot()
        g = click_from(st, tgt)
        og = shot()
        kind = "reset" if tgt == RESET else "winning move" if g["winner"] in ("X", "O") else "place mark"
        unseen_goal = key(g) not in ids or bool(held[ids[key(g)]])

        torch.cuda.synchronize(); t0 = time.perf_counter()
        z = eyes(np.stack([o, og]))
        best, dist = choose(z[:1], z[1:])
        torch.cuda.synchronize(); ms.append((time.perf_counter() - t0) * 1000)
        if n == 0:
            print(f"check: live EYES vs cached latent, max abs diff {np.abs(z[0].cpu().numpy() - feats[i]).max():.4f}")

        diff = (o != og).any(-1).reshape(16, 14, 16, 14).sum((1, 3)).flatten()
        res = {"kind": kind, "unseen_goal": unseen_goal}
        for name, j in [("model", best), ("random", random.randrange(256)), ("pixel diff", int(diff.argmax()))]:
            res[name] = key(click_from(st, CAND[j])) == key(g)
        rows.append(res)
        if len(ex) < 8 and (sum(e[5] == kind for e in ex) < 2 or not res["model"]):
            set_state(st); page.mouse.click(*CAND[best])
            ex.append((o, og, dist, best, shot(), kind, res["model"]))
        if n % 200 == 0:
            print(f"{n:>5}/{len(starts)}  model success so far {np.mean([r['model'] for r in rows]):.1%}", flush=True)

# ---------------------------------------------------------------- results
groups = {"all": lambda r: True, "place mark": lambda r: r["kind"] == "place mark",
          "winning move": lambda r: r["kind"] == "winning move", "reset": lambda r: r["kind"] == "reset",
          "goal screen never trained on": lambda r: r["unseen_goal"]}
out = {"n": len(rows), "decision_ms_median": float(np.median(ms)), "success": {}}
print("\n=== Level 1: reach a goal screen one click away, by prediction only (held-out start boards) ===")
print(f"{'goal type':<30} {'n':>5} {'model':>7} {'random':>7} {'pixel diff':>11}")
for g_, f in groups.items():
    sel = [r for r in rows if f(r)]
    row = {"n": len(sel), **{m: float(np.mean([r[m] for r in sel])) for m in ("model", "random", "pixel diff")}}
    out["success"][g_] = row
    print(f"{g_:<30} {row['n']:>5} {row['model']:>7.1%} {row['random']:>7.1%} {row['pixel diff']:>11.1%}")
print(f"decision time (see both screens + imagine 256 clicks + pick), median: {out['decision_ms_median']:.1f} ms")
a = out["success"]["all"]
ok = a["model"] >= 0.9 and a["model"] >= a["random"] + 0.5
print(f"\nKILL TEST: {'PASS' if ok else 'FAIL'}  (needs >= 90% success and +50 pts over random click)")
(R / "level1_metrics.json").write_text(json.dumps(out, indent=2))

# ---------------------------------------------------------------- picture: now + chosen click | goal | imagination map | real result
sheet = Image.new("RGB", (4 * 230, len(ex) * 230), "white")
for r_, (o, og, dist, best, after, kind, okk) in enumerate(ex):
    s_ = (dist.max() - dist) / (dist.max() - dist.min() + 1e-9)  # 1 = imagined screen closest to goal
    m = Image.fromarray((s_.view(16, 16).cpu().numpy() ** 2 * 255).astype(np.uint8)).resize((224, 224), Image.NEAREST)
    hm = Image.composite(Image.new("RGB", (224, 224), (0, 200, 0)), Image.fromarray(o), m.point(lambda v: int(v * 0.7)))
    now = Image.fromarray(o)
    x, y = CAND[best]
    ImageDraw.Draw(now).ellipse([x - 6, y - 6, x + 6, y + 6], outline=(255, 0, 255), width=3)
    res_ = Image.fromarray(after)
    ImageDraw.Draw(res_).rectangle([0, 0, 223, 223], outline=(0, 170, 0) if okk else (220, 0, 0), width=4)
    for c_, im in enumerate([now, Image.fromarray(og), hm, res_]):
        sheet.paste(im, (c_ * 230, r_ * 230))
sheet.save(R / "level1_examples.png")
print(f"saved {R / 'level1_examples.png'}  (cols: now + chosen click | goal | imagination map, green = closer to goal | real result, green border = reached)")
