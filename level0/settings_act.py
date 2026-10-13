"""Level A1 (act): reach a GOAL SETTINGS SCREEN 1-4 actions away, by imagination only, in the real browser.

e.g. start: light, volume 50, font Medium  ->  goal: dark, volume 30, font Large (menu open + pick + drag + toggle).
Candidate actions: a click at each of the 256 patch centres + 11 drags along the volume slider (track centre -> each
step). Planner = imagine chains of actions (beam), take the first action of the chain closest to the goal, look again,
stop when the screen looks like the goal (the agent's own judgement). Success = real app state == goal state.
Goal distances come from the graph of transitions seen in the collected data. Start states: held-out states.

usage: python level0/settings_act.py [PER_D=50] [DEPTHS=1,2,3]
"""
import io, json, random, sys, time
from collections import deque
from pathlib import Path
import numpy as np, torch
from PIL import Image
from playwright.sync_api import sync_playwright
from model import Imaginer, Eyes, CENTERS, dev
import store

ROOT = Path(__file__).parent
D, R = ROOT.parent / "data" / "settings", ROOT.parent / "results"
PER_D = int(sys.argv[1]) if len(sys.argv) > 1 else 50
DEPTHS = [int(h) for h in (sys.argv[2] if len(sys.argv) > 2 else "1,2,3").split(",")]
random.seed(3)

states = json.loads((D / "states.json").read_text())
keys = {s["key"]: i for i, s in enumerate(states)}
trans = json.loads((D / "transitions.json").read_text())
feats = torch.from_numpy(store.load(D / "feats.npy", mmap=False)).to(dev).float()
ck = torch.load(D / "imaginer_a1.pt")
model = Imaginer(keys=2, segments=True).to(dev)
model.load_state_dict(ck["model"]); model.eval()
held = ck["held"]
eyes = Eyes()

# candidate actions: 256 clicks (kind 1, segment of length 0) + 11 slider drags (kind 0)
TRACK = (69, 110)  # left end of the volume track (page px) and its y centre (checked against the DOM below); 1.1 px per unit
clicks = (CENTERS * 224).tolist()
drags = [[TRACK[0] + 55, TRACK[1], TRACK[0] + v * 1.1, TRACK[1]] for v in range(0, 101, 10)]
CAND = [(1, [x, y, x, y]) for x, y in clicks] + [(0, d) for d in drags]
K_ = torch.tensor([c[0] for c in CAND], device=dev)
S_ = torch.tensor([c[1] for c in CAND], device=dev, dtype=torch.float32) / 224
NC = len(CAND)

# same-screen threshold: 30% of the smallest real change between two different states
d_real = torch.stack([(feats[a] - feats[b]).pow(2).mean() for a, _, _, _, _, _, b in trans[:5000] if a != b])
TAU = 0.3 * float(d_real.quantile(0.01))
dist = lambda a, b: (a - b).pow(2).mean((1, 2))


@torch.no_grad()
def imagine(Z):  # every node x every candidate
    out = []
    for i in range(0, len(Z) * NC, 2048):
        idx = torch.arange(i, min(i + 2048, len(Z) * NC), device=dev)
        with torch.autocast("cuda", torch.bfloat16):
            out.append(model(Z[idx // NC], k=K_[idx % NC], s=S_[idx % NC])[0].float())
    return torch.cat(out)


@torch.no_grad()
def plan(z0, zg, H, B=8):  # beam search in imagination (as in Level 2), no-op steps dropped, near-duplicates merged
    Z, first, best = z0, None, (float("inf"), None)
    for h in range(H):
        P = imagine(Z)
        f = torch.arange(NC, device=dev) if first is None else first.repeat_interleave(NC)
        moved = dist(P, Z.repeat_interleave(NC, 0)) > TAU
        d = torch.where(moved, dist(P, zg), torch.inf)
        i = int(d.argmin())
        if d[i] < best[0] - TAU:
            best = (float(d[i]), int(f[i]))
        keep = []
        for j in d.argsort()[:512].tolist():
            if not torch.isfinite(d[j]):
                break
            if all(dist(P[j:j + 1], P[k:k + 1]).item() > TAU for k in keep):
                keep.append(j)
            if len(keep) == B:
                break
        if not keep:
            break
        Z, first = P[keep], f[keep]
    return best[1] if best[1] is not None else random.randrange(NC)


# goal distances on the graph of observed transitions
adj = {}
for a, *_, b in trans:
    adj.setdefault(a, set()).add(b)


def bfs(s):
    dd, q = {s: 0}, deque([s])
    while q:
        u = q.popleft()
        for v in adj.get(u, ()):
            if v not in dd:
                dd[v] = dd[u] + 1; q.append(v)
    return dd


starts = [i for i in range(len(states)) if held[i]]
episodes = []
for d_ in (1, 2, 3, 4):
    while sum(e["d"] == d_ for e in episodes) < PER_D:
        s = random.choice(starts)
        far = [g for g, v in bfs(s).items() if v == d_]
        if far:
            episodes.append({"s": s, "g": random.choice(far), "d": d_})

rows, ms = [], {h: [] for h in DEPTHS}
with sync_playwright() as p:
    page = p.chromium.launch().new_page(viewport={"width": 224, "height": 224})
    page.goto((ROOT / "settings.html").as_uri())
    shot = lambda: np.asarray(Image.open(io.BytesIO(page.screenshot())).convert("RGB"))
    state_key = lambda: (lambda s: f"{int(s['dark'])}{int(s['wifi'])}{s['volume']:03d}{s['font']}{int(s['notif'])}{int(s['menu'])}")(page.evaluate("window.__state"))
    set_state = lambda i: page.evaluate("st => window.__set(st)", {k: v for k, v in states[i].items() if k != "key"})

    def act(c):
        kind_, (x1, y1, x2, y2) = CAND[c]
        if kind_ == 0:
            page.mouse.move(x1, y1); page.mouse.down(); page.mouse.move(x2, y2, steps=6); page.mouse.up()
        else:
            page.mouse.click(x1, y1)

    for n, ep in enumerate(episodes):
        zg = feats[ep["g"]][None]
        res = {"d": ep["d"]}
        for H in DEPTHS:
            set_state(ep["s"]); ok, budget = False, ep["d"] + 3
            for used in range(budget + 1):
                z = eyes(shot()[None])
                if dist(z, zg).item() < TAU:
                    ok = state_key() == states[ep["g"]]["key"]; break
                if used == budget:
                    break
                torch.cuda.synchronize(); t0 = time.perf_counter()
                c = plan(z, zg, H)
                torch.cuda.synchronize(); ms[H].append((time.perf_counter() - t0) * 1000)
                act(c)
            res[f"think {H}"] = ok
        set_state(ep["s"]); hit = False  # random baseline (generous: success if it ever passes through the goal)
        for _ in range(ep["d"] + 3):
            act(random.randrange(NC)); hit |= state_key() == states[ep["g"]]["key"]
        res["random"] = hit
        rows.append(res)
        if n % 50 == 0:
            print(f"{n:>4}/{len(episodes)}  " + "  ".join(f"think {h} {np.mean([r[f'think {h}'] for r in rows]):.0%}" for h in DEPTHS), flush=True)

names = [f"think {h}" for h in DEPTHS] + ["random"]
out = {"n": len(rows), "success": {}, "decision_ms_median": {h: float(np.median(ms[h])) for h in DEPTHS}}
print("\n=== A1 Settings: reach a goal settings screen by imagination (start states never seen in training) ===")
print(f"{'goal':<16} {'n':>4} " + " ".join(f"{m:>9}" for m in names))
for g_, f in [("all", lambda r: True)] + [(f"{d_} actions away", lambda r, d_=d_: r["d"] == d_) for d_ in (1, 2, 3, 4)]:
    sel = [r for r in rows if f(r)]
    row = {"n": len(sel), **{m: float(np.mean([r[m] for r in sel])) for m in names}}
    out["success"][g_] = row
    print(f"{g_:<16} {row['n']:>4} " + " ".join(f"{row[m]:>9.1%}" for m in names))
print("decision ms (median): " + "  ".join(f"think {h} {out['decision_ms_median'][h]:.0f}" for h in DEPTHS))
a = out["success"]["all"]
ok = max(a[f"think {h}"] for h in DEPTHS) >= 0.8 and max(a[f"think {h}"] for h in DEPTHS) >= a["random"] + 0.5
print(f"\nKILL TEST: {'PASS' if ok else 'FAIL'}  (needs >= 80% goals reached and +50 pts over random)")
(R / "settings_a1_act_metrics.json").write_text(json.dumps(out, indent=2))
