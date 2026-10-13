"""Apps A1-A3 (act): reach a GOAL APP SCREEN 1-4 actions away, by imagination only, in the real browser.

settings: e.g. dark on + volume 30 + font Large (toggle, slider drag, menu open + pick)
files:    e.g. report.txt moved to Archive and Archive open (drag-and-drop onto a folder, click a folder, double-click)
Candidate actions per app (below). Planner = imagine chains of actions (beam), take the first action of the chain
closest to the goal, look again, stop when the screen looks like the goal (the agent's own judgement).
Success = the real app shows the goal SCREEN (screen id of the reached state; pixel check if the state was never
collected) - apps can have hidden state, so the hidden part of the goal is not required. Goal distance: fewest actions
from the start state to any state with that screen, on the graph of transitions seen in the data. Starts: held-out
screens.
Uses the model from app_train.py (settings_act.py is the A1-only original of this script).

usage: python level0/app_act.py APP [PER_D=50] [DEPTHS=1,2,3]
"""
import io, json, os, random, sys, time
from collections import deque
from pathlib import Path
import numpy as np, torch
from PIL import Image
from playwright.sync_api import sync_playwright
from model import Imaginer, Eyes, CENTERS, dev
from app_collect import do
import store

ROOT = Path(__file__).parent
APP = sys.argv[1] if len(sys.argv) > 1 else "settings"
D, R = ROOT.parent / "data" / APP, ROOT.parent / "results"
PER_D = int(sys.argv[2]) if len(sys.argv) > 2 else 50
DEPTHS = [int(h) for h in (sys.argv[3] if len(sys.argv) > 3 else "1,2,3").split(",")]
key = __import__(f"{APP}_collect").key  # the app's own state -> unique string
random.seed(3)
JUDGE = os.environ.get("JUDGE") == "1"  # step 1: learned step-distance judge (judge.py) instead of latent MSE
TAG = "_judge" if JUDGE else ""

states = json.loads((D / "states.json").read_text())
keys = {s["key"]: i for i, s in enumerate(states)}
trans = json.loads((D / "transitions.json").read_text())
feats = torch.from_numpy(store.load(D / "feats.npy", mmap=False)).to(dev).float()
ck = torch.load(D / "imaginer_app.pt")
model = Imaginer(keys=4, segments=True).to(dev)
model.load_state_dict(ck["model"]); model.eval()
held, cls = ck["held"], ck["cls"].tolist()  # cls: state -> screen id
rep = {c: i for i, c in reversed(list(enumerate(cls)))}  # screen id -> first state showing it
eyes = Eyes()
if JUDGE:
    import judge as judge_mod
    judge = judge_mod.load(APP)

# candidate actions: a click (kind 1) at each patch centre, plus app-specific drags (kind 0) / double-clicks (kind 2)
clicks = (CENTERS * 224).tolist()
CAND = [(1, [x, y, x, y]) for x, y in clicks]
if APP == "settings":  # 11 drags along the volume slider: track centre -> each step (track x 69..179, y 110)
    CAND += [(0, [124, 110, 69 + v * 1.1, 110]) for v in range(0, 101, 10)]
if APP == "files":  # double-click every patch centre + drag each of the 4 file-tile slots onto each of the 4 folders
    CAND += [(2, [x, y, x, y]) for x, y in clicks]
    CAND += [(0, [tx, ty, 36, 39 + 34 * f]) for tx, ty in [(112, 55), (184, 55), (112, 139), (184, 139)] for f in range(4)]
K_ = torch.tensor([c[0] for c in CAND], device=dev)
S_ = torch.tensor([c[1] for c in CAND], device=dev, dtype=torch.float32) / 224
NC = len(CAND)

# same-screen threshold: 30% of the smallest real change between two different states
d_real = torch.stack([(feats[a] - feats[b]).pow(2).mean() for a, _, _, _, _, _, b in trans[:5000] if a != b])
TAU = 0.3 * float(d_real.quantile(0.01))
dist = lambda a, b: (a - b).pow(2).mean((1, 2))


@torch.no_grad()
def imagine(Z, zg):  # every node x every candidate -> imagined screens (fp16), "did it move?", goal score
    P, mv, sc = [], [], []  # in chunks of 1024: the planner used to hold every imagined screen in fp32 (~15 GB VRAM)
    for i in range(0, len(Z) * NC, 1024):
        idx = torch.arange(i, min(i + 1024, len(Z) * NC), device=dev)
        z = Z[idx // NC].float()
        with torch.autocast("cuda", torch.bfloat16):
            p = model(z, k=K_[idx % NC], s=S_[idx % NC])[0].float()
        mv.append(dist(p, z) > TAU)
        sc.append(judge.steps(p, zg) if JUDGE else dist(p, zg))
        P.append(p.half())
    return torch.cat(P), torch.cat(mv), torch.cat(sc)


@torch.no_grad()
def plan(z0, zg, H, B=8):  # beam search in imagination (as in Level 2), no-op steps dropped, near-duplicates merged
    Z, first, best = z0, None, (float("inf"), None)
    for h in range(H):
        P, moved, sc = imagine(Z, zg)
        f = torch.arange(NC, device=dev) if first is None else first.repeat_interleave(NC)
        d = torch.where(moved, sc + h + 1 if JUDGE else sc, torch.inf)
        i = int(d.argmin())
        if d[i] < best[0] - (0.25 if JUDGE else TAU):
            best = (float(d[i]), int(f[i]))
        keep = []
        for j in d.argsort()[:512].tolist():
            if not torch.isfinite(d[j]):
                break
            if all(dist(P[j:j + 1].float(), P[k:k + 1].float()).item() > TAU for k in keep):
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
        sd = {}  # screen id -> fewest actions
        for g, v in bfs(s).items():
            sd[cls[g]] = min(v, sd.get(cls[g], v))
        far = [c for c, v in sd.items() if v == d_]
        if far:
            episodes.append({"s": s, "g": rep[random.choice(far)], "d": d_})

rows, ms = [], {h: [] for h in DEPTHS}
with sync_playwright() as p:
    page = p.chromium.launch().new_page(viewport={"width": 224, "height": 224})
    page.goto((ROOT / f"{APP}.html").as_uri())
    shot = lambda: np.asarray(Image.open(io.BytesIO(page.screenshot())).convert("RGB"))
    state_key = lambda: key(page.evaluate("window.__state"))
    set_state = lambda i: page.evaluate("st => window.__set(st)", {k: v for k, v in states[i].items() if k != "key"})
    act = lambda c: do(page, [CAND[c][0], *CAND[c][1]])

    def on_goal(g):  # does the real app show the goal screen?
        i = keys.get(state_key())
        if i is not None:
            return cls[i] == cls[g]
        d = np.abs(shot().astype(int) - np.asarray(Image.open(D / "img" / f"{g}.png").convert("RGB")).astype(int)).max(-1)
        return (d > 0).sum() <= 50 and d.max() <= 96  # same anti-aliasing tolerance as the collector

    for n, ep in enumerate(episodes):
        zg = feats[ep["g"]][None]
        res = {"d": ep["d"]}
        for H in DEPTHS:
            set_state(ep["s"]); ok, budget = False, ep["d"] + 3
            for used in range(budget + 1):
                z = eyes(shot()[None])
                if dist(z, zg).item() < TAU:
                    ok = on_goal(ep["g"]); break
                if used == budget:
                    break
                torch.cuda.synchronize(); t0 = time.perf_counter()
                c = plan(z, zg, H)
                torch.cuda.synchronize(); ms[H].append((time.perf_counter() - t0) * 1000)
                act(c)
            res[f"think {H}"] = ok
        set_state(ep["s"]); hit = False  # random baseline (generous: success if it ever passes through the goal)
        for _ in range(ep["d"] + 3):
            act(random.randrange(NC)); hit |= on_goal(ep["g"])
        res["random"] = hit
        rows.append(res)
        if n % 50 == 0:
            print(f"{n:>4}/{len(episodes)}  " + "  ".join(f"think {h} {np.mean([r[f'think {h}'] for r in rows]):.0%}" for h in DEPTHS), flush=True)

names = [f"think {h}" for h in DEPTHS] + ["random"]
out = {"n": len(rows), "success": {}, "decision_ms_median": {h: float(np.median(ms[h])) for h in DEPTHS}}
print(f"\n=== {APP}: reach a goal app screen by imagination (start states never seen in training) ===")
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
(R / f"app_{APP}_act{TAG}_metrics.json").write_text(json.dumps(out, indent=2))
