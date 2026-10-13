"""Level 2: CHAIN clicks to reach a goal screen 2-5 clicks away, by prediction only.

Loop:  see screen -> imagine -> click the best first step -> see the new screen -> ...
       stop when the agent itself thinks "my screen now looks like the goal".
Success = the REAL app is in the goal state when the agent says stop (budget: shortest path + 4 clicks).

Planners (same IMAGINER, no extra training):
  greedy    : imagine 1 click ahead (= Level 1, repeated)
  lookahead : imagine chains of 3 clicks (the model's own predictions fed back in), beam of 8
Rules (python copy of the page) are used ONLY to pick test goals and their true shortest distance, never by the agent.
"""
import io, json, random, sys, time
from collections import deque
from pathlib import Path
import numpy as np, torch
from PIL import Image, ImageDraw
from playwright.sync_api import sync_playwright
from model import Imaginer, Eyes, CENTERS, dev
from collect import key
from ttt_rules import step, hit
import store
import os

ROOT = Path(__file__).parent
D, R = ROOT.parent / "data" / "level0", ROOT.parent / "results"
PER_D = int(sys.argv[1]) if len(sys.argv) > 1 else 100
random.seed(2)

# ---------------------------------------------------------------- shortest paths, for picking goals only
def bfs(s, allow_reset=True):  # shortest number of clicks from s to every reachable state
    dist, q, seen = {key(s): 0}, deque([s]), {key(s): s}
    while q:
        u = q.popleft()
        for m in range(10 if allow_reset else 9):
            v = step(u, m)
            if key(v) not in dist:
                dist[key(v)], seen[key(v)] = dist[key(u)] + 1, v
                q.append(v)
    return dist, seen


states = json.loads((D / "states.json").read_text())
t = np.load(D / "transitions.npz")
EDGES_X, EDGES_Y = (22, 82, 142, 202), (24, 84, 144, 204, 206, 222)
for a, x, y, b in list(zip(t["src"], t["x"], t["y"], t["dst"]))[:20000]:  # self-check: python rules == real page
    if min(abs(x - e) for e in EDGES_X) < 1 or min(abs(y - e) for e in EDGES_Y) < 1:
        continue  # Chromium's sub-pixel hit-testing differs from ours by < 1 px at borders
    s = {k: states[a][k] for k in ("board", "turn", "winner")}
    assert key(step(s, hit(x, y))) == states[b]["key"], (states[a]["key"], x, y, states[b]["key"])
print("check: python rules match the real page on 20,000 recorded clicks (border +-1px skipped)")

# ---------------------------------------------------------------- model + thresholds
ck = torch.load(D / "imaginer.pt")
held = ck["held"]
model = Imaginer().to(dev)
model.load_state_dict(ck["model"])
model.eval()
eyes = Eyes()
CAND = (CENTERS * 224).tolist()
feats = store.load(D / "feats.npy")
ch = np.where(t["src"] != t["dst"])[0][:5000]
f_ = lambda ix: torch.from_numpy(np.stack([feats[i] for i in ix])).float()
real_change = (f_(t["src"][ch]) - f_(t["dst"][ch])).pow(2).mean((1, 2))
TAU = 0.3 * float(real_change.quantile(0.01))  # "same screen" = closer than 30% of the smallest real change
print(f"smallest real change {TAU / 0.3:.4f} -> same-screen threshold {TAU:.4f}")


def dist(a, b):  # a: n,256,384  b: 1,256,384 -> n
    return (a - b).pow(2).mean((1, 2))


@torch.no_grad()
def imagine(Z):  # every node x every candidate click -> n*256 predicted screens
    n = len(Z)
    out = []
    for i in range(0, n * 256, 2048):
        idx = torch.arange(i, min(i + 2048, n * 256), device=dev)
        with torch.autocast("cuda", torch.bfloat16):
            out.append(model(Z[idx // 256], CENTERS[idx % 256])[0].float())
    return torch.cat(out)


JUDGE = os.environ.get("JUDGE") == "1"  # step 1: learned step-distance judge (judge.py) instead of latent MSE
if JUDGE:
    import judge as judge_mod
    judge, actor = judge_mod.load("ttt"), judge_mod.load_actor("ttt")


@torch.no_grad()
def act_plan(z0, zg, top=8):  # step 2: the actor proposes; top=1 -> its best click, else imagine its top clicks + judge picks
    with torch.autocast("cuda", torch.bfloat16):
        c = actor(z0, zg).float()[0].topk(top).indices
    if top == 1:
        return int(c[0])
    with torch.autocast("cuda", torch.bfloat16):
        P = model(z0.expand(len(c), -1, -1), CENTERS[c])[0].float()
    sc = torch.where(dist(P, z0) > TAU, judge.steps(P, zg), torch.inf)
    return int(c[int(sc.argmin())])


@torch.no_grad()
def plan(z0, zg, H, J=False, B=8):
    """Beam search in imagination. Returns the first click of the imagined chain that gets closest to the goal.
    Imagined steps that change nothing are dropped (a no-op never helps); near-identical imagined screens are merged.
    J: score = imagined steps so far + the judge's steps-to-goal (else: latent MSE to the goal)."""
    Z, first, best = z0, None, (float("inf"), None)
    for h in range(H):
        P = imagine(Z)
        f = torch.arange(256, device=dev) if first is None else first.repeat_interleave(256)
        moved = dist(P, Z.repeat_interleave(256, 0)) > TAU
        d = torch.where(moved, judge.steps(P, zg) + h + 1 if J else dist(P, zg), torch.inf)
        i = int(d.argmin())
        if d[i] < best[0] - (0.25 if J else TAU):  # deeper chain must be clearly better (prefer short plans)
            best = (float(d[i]), int(f[i]))
        order = d.argsort()[:512]
        order = order[torch.isfinite(d[order])]
        keep = []
        for j in order.tolist():  # greedy dedupe: keep B distinct imagined screens
            if all(dist(P[j:j + 1], P[k:k + 1]).item() > TAU for k in keep):
                keep.append(j)
            if len(keep) == B:
                break
        if not keep:
            break
        Z, first = P[keep], f[keep]
    return best[1] if best[1] is not None else random.randrange(256)


# ---------------------------------------------------------------- test episodes: held-out start, goal at exact shortest distance d
episodes = []
starts = [i for i in range(len(states)) if held[i]]
for d_ in (2, 3, 4, 5):
    for need in (True, False):  # half the goals need a Reset first, half do not
        while sum(e["d"] == d_ and e["needs_reset"] == need for e in episodes) < PER_D // 2:
            s = {k: states[random.choice(starts)][k] for k in ("board", "turn", "winner")}
            dist_all, seen = bfs(s)
            no_reset = bfs(s, allow_reset=False)[0]
            far = [k for k, v in dist_all.items() if v == d_ and (k not in no_reset) == need]
            if far:
                episodes.append({"s": s, "g": seen[random.choice(far)], "d": d_, "needs_reset": need})

# default: greedy vs lookahead-3.  "Thinking budget" mode: python level2.py PER_D 1,2,3,4,5  -> one planner per depth
PLANNERS = {"greedy": 1, "lookahead": 3} if len(sys.argv) < 3 else {f"think {h}": int(h) for h in sys.argv[2].split(",")}
TAG = "" if len(sys.argv) < 3 else "_budget"
USE_J = {}
if JUDGE:  # MSE lookahead (old best) vs the judge, greedy and lookahead
    PLANNERS, USE_J, TAG = {"mse look3": 3, "judge 1": 1, "judge 3": 3, "actor": 1, "actor8+judge": 1}, {"judge 1": True, "judge 3": True}, "_judge"
MS = {name: [] for name in PLANNERS}
rows, strips = [], []
with sync_playwright() as p:
    page = p.chromium.launch().new_page(viewport={"width": 224, "height": 224})
    page.goto((ROOT / "tictactoe.html").as_uri())
    shot = lambda: np.asarray(Image.open(io.BytesIO(page.screenshot())).convert("RGB"))
    set_state = lambda st: page.evaluate("st => { s = st; render(); }", st)

    for n, ep in enumerate(episodes):
        set_state(ep["g"]); og = shot()
        zg = eyes(og[None])
        budget = ep["d"] + 4
        res = {"d": ep["d"], "needs_reset": ep["needs_reset"]}
        for name, H in PLANNERS.items():
            set_state(ep["s"]); frames, ok, used = [], False, 0
            for used in range(budget + 1):
                o = shot()
                z = eyes(o[None])
                stopped = dist(z, zg).item() < TAU
                if stopped:  # agent's own judgement: "I'm there"
                    ok = key(page.evaluate("window.__state")) == key(ep["g"])
                    break
                if used == budget:
                    break
                torch.cuda.synchronize(); t0 = time.perf_counter()
                j = act_plan(z, zg, 1 if name == "actor" else 8) if name.startswith("actor") else plan(z, zg, H, USE_J.get(name, False))
                torch.cuda.synchronize(); MS[name].append((time.perf_counter() - t0) * 1000)
                frames.append((o, CAND[j]))
                page.mouse.click(*CAND[j])
            res[name], res[name + "_clicks"] = ok, used
            res[name + "_fail"] = None if ok else ("stopped at a wrong board" if stopped else "ran out of clicks")
            if len(strips) < 12 and ep["needs_reset"] and n % 7 == 0:
                strips.append((name, ok, og, frames + [(shot(), None)]))
        set_state(ep["s"])  # random baseline, generous: success if it ever passes through the goal
        hitg = False
        for _ in range(budget):
            page.mouse.click(*random.choice(CAND))
            hitg |= key(page.evaluate("window.__state")) == key(ep["g"])
        res["random"] = hitg
        rows.append(res)
        if n % 50 == 0:
            print(f"{n:>4}/{len(episodes)}  " + "  ".join(f"{m} {np.mean([r[m] for r in rows]):.1%}" for m in PLANNERS), flush=True)

# ---------------------------------------------------------------- results
out = {"n": len(rows), "same_screen_threshold": TAU, "success": {}, "decision_ms_median": {m: float(np.median(MS[m])) for m in PLANNERS},
       "failures": {m: {k: sum(r[m + "_fail"] == k for r in rows) for k in ("stopped at a wrong board", "ran out of clicks")} for m in PLANNERS}}
names = list(PLANNERS) + ["random"]
deep = list(PLANNERS)[-1]
print("\n=== Level 2: reach a goal screen 2-5 clicks away (held-out start boards), agent decides when it is done ===")
print(f"{'goal':<26} {'n':>4} " + " ".join(f"{m:>10}" for m in names) + f"   extra clicks ({deep}, successes)")
groups = {"all": lambda r: True, **{f"{d_} clicks away": (lambda r, d_=d_: r["d"] == d_) for d_ in (2, 3, 4, 5)},
          "needs Reset first": lambda r: r["needs_reset"], "no Reset needed": lambda r: not r["needs_reset"]}
for g_, f in groups.items():
    sel = [r for r in rows if f(r)]
    row = {"n": len(sel), **{m: float(np.mean([r[m] for r in sel])) for m in names}}
    wins = [r[deep + "_clicks"] - r["d"] for r in sel if r[deep]]
    row["extra_clicks_" + deep] = float(np.mean(wins)) if wins else None
    out["success"][g_] = row
    print(f"{g_:<26} {row['n']:>4} " + " ".join(f"{row[m]:>10.1%}" for m in names) + f"   {np.mean(wins) if wins else float('nan'):.2f}")
print("* random = clicks at random for the same budget; counted as success if it EVER passes through the goal (generous)")
print("decision time per click (median): " + "  ".join(f"{m} {out['decision_ms_median'][m]:.0f} ms" for m in PLANNERS))
print("why failures happened: " + "  |  ".join(f"{m}: {out['failures'][m]}" for m in PLANNERS))
(R / f"level2{TAG}_metrics.json").write_text(json.dumps(out, indent=2))
if TAG == "_budget":  # thinking budget curve (the judge run used to overwrite it)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    hs = list(PLANNERS.values())
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
    for g_ in ["all", "2 clicks away", "3 clicks away", "4 clicks away", "5 clicks away"]:
        ax[0].plot(hs, [out["success"][g_][m] for m in PLANNERS], "o-", lw=2.5 if g_ == "all" else 1, label=g_)
    ax[0].set(xlabel="imagined steps ahead (thinking budget)", ylabel="goals reached", ylim=(0, 1.02)); ax[0].legend(fontsize=7)
    ax[1].plot(hs, [out["decision_ms_median"][m] for m in PLANNERS], "o-", color="tab:red")
    ax[1].set(xlabel="imagined steps ahead", ylabel="ms per click (median)")
    fig.suptitle("Tic-tac-toe: thinking longer in imagination"); fig.tight_layout(); fig.savefig(R / "level2_budget_curve.png", dpi=120)

# ---------------------------------------------------------------- picture: one row per run = goal | screens with the click taken
W = max(len(fr) for *_, fr in strips) + 1
sheet = Image.new("RGB", (W * 230, len(strips) * 230), "white")
for r_, (name, ok, og, fr) in enumerate(strips):
    gi = Image.fromarray(og)
    dr = ImageDraw.Draw(gi)
    dr.rectangle([0, 0, 223, 223], outline=(0, 0, 255), width=4)
    dr.text((4, 208), f"GOAL ({name})", fill=(0, 0, 255))
    sheet.paste(gi, (0, r_ * 230))
    for c_, (o, xy) in enumerate(fr):
        im = Image.fromarray(o)
        dr = ImageDraw.Draw(im)
        if xy:
            dr.ellipse([xy[0] - 6, xy[1] - 6, xy[0] + 6, xy[1] + 6], outline=(255, 0, 255), width=3)
        else:
            dr.rectangle([0, 0, 223, 223], outline=(0, 170, 0) if ok else (220, 0, 0), width=4)
        sheet.paste(im, ((c_ + 1) * 230, r_ * 230))
sheet.save(R / f"level2{TAG}_examples.png")
print(f"saved {R / f'level2{TAG}_examples.png'}  (row: goal | each screen + click taken | final screen, green = reached, red = failed)")
