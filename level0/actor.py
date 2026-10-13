"""Step 2: an ACTOR that PROPOSES actions (generative over actions) instead of trying every click in imagination.

ACTOR(screen, goal screen) -> a probability for clicking each of the 256 patches. Trained by hindsight on the
agent's own experience graph: for a start a and any screen g it has reached, the right clicks are the recorded clicks
from a that land on a shortest route to g (labels from experience only - never the game rules).
Then: actor alone (1 forward pass), or actor proposes its top 8, the IMAGINER imagines them and the JUDGE (step 1)
picks (= the planner, but searching 8 actions instead of 256).

Test (offline, tic-tac-toe): held-out start boards, goals 2-5 clicks away (true distance from the rules, as Level 2),
budget = distance + 4. Screens are the cached real screenshots of each board; the agent stops itself when its screen
matches the goal (latent MSE < threshold, as Level 2). Rules only move the board and score.

usage: python level0/actor.py [STEPS=6000] [PER_D=100]
"""
import json, random, sys, time
from collections import deque
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from model import Imaginer, CENTERS, dev
from collect import key
from ttt_rules import step, hit
import judge as J

STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 6000
PER_D = int(sys.argv[2]) if len(sys.argv) > 2 else 100
R = Path(__file__).parent.parent / "results"
torch.manual_seed(0); rng = np.random.default_rng(0); random.seed(2)


D, feats, held, src, dst = J.graph("ttt")
t = np.load(D / "transitions.npz")
patch = (np.minimum(t["y"] // 14, 15) * 16 + np.minimum(t["x"] // 14, 15)).astype(int)
S = len(feats)
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
tr_e = ~held[src] & ~held[dst]
dd = shortest_path(csr_matrix((np.ones(tr_e.sum()), (src[tr_e], dst[tr_e])), (S, S)), unweighted=True)
dd = np.minimum(np.nan_to_num(dd, posinf=99), 99).astype(np.int16)
out = {}  # training start -> (patches, next states) of its recorded clicks, unique
for a, p_, b in zip(src[tr_e], patch[tr_e], dst[tr_e]):
    out.setdefault(a, set()).add((p_, b))
out = {a: (np.array([p_ for p_, _ in v]), np.array([b for _, b in v])) for a, v in out.items()}
tr_s = np.array(sorted(out))


def batch(n=128):
    a, g, y = [], [], torch.zeros(n, 256)
    while len(a) < n:
        s = rng.choice(tr_s)
        k = rng.integers(1, 8)
        cand = np.where(dd[s] == k)[0]
        if not len(cand):
            continue
        goal = rng.choice(cand)
        ps, bs = out[s]
        good = ps[dd[bs, goal] == k - 1]
        y[len(a)].index_add_(0, torch.from_numpy(good), torch.ones(len(good)))
        a.append(s); g.append(goal)
    return np.array(a), np.array(g), y / y.sum(1, keepdim=True)


actor = J.Actor().to(dev)
opt = torch.optim.AdamW(actor.parameters(), 3e-4, weight_decay=0.01)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-4, total_steps=STEPS, pct_start=0.05)
for it in range(STEPS):
    a, g, y = batch()
    with torch.autocast("cuda", torch.bfloat16):
        loss = -(F.log_softmax(actor(feats[a], feats[g]).float(), -1) * y.to(dev)).sum(-1).mean()
    opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
    if it % 1000 == 0 or it == STEPS - 1:
        print(f"step {it:>5}  loss {loss.item():.4f}", flush=True)
torch.save(actor.state_dict(), D / "actor.pt")
actor.eval()

# ---------------------------------------------------------------- offline goal test
judge = J.load("ttt")
ck = torch.load(D / "imaginer.pt", weights_only=False)
model = Imaginer().to(dev); model.load_state_dict(ck["model"]); model.eval()
states = json.loads((D / "states.json").read_text())
idx = {s["key"]: i for i, s in enumerate(states)}
ch = np.where(src != dst)[0][:5000]
TAU = 0.3 * float((feats[src[ch]] - feats[dst[ch]]).pow(2).mean((1, 2)).quantile(0.01))
XY = (CENTERS * 224).tolist()


def bfs(s):
    dist_, q, seen = {key(s): 0}, deque([s]), {key(s): s}
    while q:
        u = q.popleft()
        for m in range(10):
            v = step(u, m)
            if key(v) not in dist_:
                dist_[key(v)], seen[key(v)] = dist_[key(u)] + 1, v; q.append(v)
    return dist_, seen


@torch.no_grad()
def choose(mode, z, zg):
    if mode == "actor":
        with torch.autocast("cuda", torch.bfloat16):
            return int(actor(z, zg).argmax())
    if mode == "actor top8 + judge":
        with torch.autocast("cuda", torch.bfloat16):
            c = actor(z, zg).float()[0].topk(8).indices
    else:  # "judge, all 256"
        c = torch.arange(256, device=dev)
    with torch.autocast("cuda", torch.bfloat16):
        P = model(z.expand(len(c), -1, -1), CENTERS[c])[0].float()
    moved = (P - z).pow(2).mean((1, 2)) > TAU
    sc = torch.where(moved, judge.steps(P, zg), torch.inf)
    return int(c[int(sc.argmin())])


starts = [s for s in states if held[idx[s["key"]]]]
episodes = []
for d_ in (2, 3, 4, 5):
    while sum(e["d"] == d_ for e in episodes) < PER_D:
        s = {k: v for k, v in random.choice(starts).items() if k in ("board", "turn", "winner")}
        dist_, seen = bfs(s)
        far = [k for k, v in dist_.items() if v == d_ and k in idx]
        if far:
            episodes.append({"s": s, "g": seen[random.choice(far)], "d": d_})

MODES = ["actor", "actor top8 + judge", "judge, all 256"]
res = {m: [] for m in MODES}; ms = {m: [] for m in MODES}; off = 0
for ep in episodes:
    zg = feats[idx[key(ep["g"])]][None]
    for m in MODES:
        s, ok = ep["s"], False
        for used in range(ep["d"] + 5):
            i = idx.get(key(s))
            if i is None:
                off += 1; break  # a board with no cached screenshot (never recorded): counted as a failure
            z = feats[i][None]
            if (z - zg).pow(2).mean().item() < TAU:  # the agent's own "I'm there"
                ok = key(s) == key(ep["g"]); break
            if used == ep["d"] + 4:
                break
            torch.cuda.synchronize(); t0 = time.perf_counter()
            c = choose(m, z, zg)
            torch.cuda.synchronize(); ms[m].append((time.perf_counter() - t0) * 1000)
            s = step(s, hit(*XY[c]))
        res[m].append((ep["d"], ok))

out_ = {"n": len(episodes), "off_data_boards": off, "success": {}, "decision_ms_median": {m: float(np.median(ms[m])) for m in MODES}}
print(f"\n=== tic-tac-toe: reach a goal board 2-5 clicks away (held-out starts), offline on cached screens ===")
print(f"{'goal':<14} " + " ".join(f"{m:>20}" for m in MODES))
for g_ in ["all", 2, 3, 4, 5]:
    row = {m: float(np.mean([ok for d_, ok in res[m] if g_ == "all" or d_ == g_])) for m in MODES}
    out_["success"][str(g_)] = row
    print(f"{(g_ if g_ == 'all' else f'{g_} clicks away'):<14} " + " ".join(f"{row[m]:>20.1%}" for m in MODES))
print("decision ms (median): " + "  ".join(f"{m} {out_['decision_ms_median'][m]:.1f}" for m in MODES) + f" | off-data boards {off}")
(R / "actor_ttt_metrics.json").write_text(json.dumps(out_, indent=2))
