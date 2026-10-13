"""Step 1: a learned STEP-DISTANCE JUDGE - "how many actions from this screen to that screen?" - to replace latent MSE
as the planners' goal distance (the bottleneck in Level 2, the thinking-budget run and A1).

Labels come only from the agent's own experience: shortest paths on the graph of recorded transitions (every screen it
reached is a goal it could have aimed for = hindsight, with the shortest route found so far). Classes 0..7 and 8 = "8+
or not reached"; expected steps = sum p_k * k. Trained on pairs whose START is a training screen (both ends for the
split used by the IMAGINER); tested on held-out start screens. Inputs get a little noise so imagined screens (slightly
off the real ones) are judged sensibly.

usage: python level0/judge.py APP [STEPS=6000]        APP = ttt | settings | files
"""
import json, sys
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from model import dev
import store

DATA = Path(__file__).parent.parent / "data"
K = 9  # distance classes 0..7, 8 = 8+ / unreached


class Judge(nn.Module):
    def __init__(self, d=384, w=256, layers=4, tokens=256):
        super().__init__()
        self.inp, self.pos = nn.Linear(2 * d, w), nn.Parameter(torch.zeros(1, tokens, w))
        block = nn.TransformerEncoderLayer(w, 4, 4 * w, 0.0, batch_first=True, norm_first=True)
        self.tf = nn.TransformerEncoder(block, layers, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.LayerNorm(w), nn.Linear(w, K))

    def forward(self, z, g):  # (B,T,d) now, (B,T,d) goal -> (B,K) logits
        return self.head(self.tf(self.inp(torch.cat([z, g - z], -1)) + self.pos).mean(1))

    @torch.no_grad()
    def steps(self, z, g):  # expected number of actions; g may be (1,T,d)
        with torch.autocast("cuda", torch.bfloat16):
            p = self(z, g.expand_as(z)).float().softmax(-1)
        return p @ torch.arange(K, device=p.device, dtype=torch.float32)


class Actor(Judge):  # same body as the judge, one logit per patch: which click to make (actor.py trains it)
    def __init__(self, **kw):
        super().__init__(**kw)
        self.head = nn.Sequential(nn.LayerNorm(256), nn.Linear(256, 1))

    def forward(self, z, g):
        return self.head(self.tf(self.inp(torch.cat([z, g - z], -1)) + self.pos)).squeeze(-1)


def load_actor(app):
    a = Actor().to(dev)
    a.load_state_dict(torch.load(DATA / ("level0" if app == "ttt" else app) / "actor.pt", weights_only=True))
    return a.eval()


def load(app):
    j = Judge().to(dev)
    j.load_state_dict(torch.load(DATA / ("level0" if app == "ttt" else app) / "judge.pt", weights_only=True))
    return j.eval()


def graph(app):  # -> feats (S,T,d) on GPU, held (S,) bool, src, dst arrays
    D = DATA / ("level0" if app == "ttt" else app)
    feats = torch.from_numpy(np.asarray(store.load(D / "feats.npy", mmap=False))).to(dev).float()
    if app == "ttt":
        t = np.load(D / "transitions.npz"); src, dst = t["src"], t["dst"]
        held = torch.load(D / "imaginer.pt", weights_only=False)["held"].numpy()
    else:
        tr = np.array(json.loads((D / "transitions.json").read_text())); src, dst = tr[:, 0].astype(int), tr[:, 6].astype(int)
        held = torch.load(D / "imaginer_app.pt", weights_only=False)["held"].numpy()
    return D, feats, held, src, dst


if __name__ == "__main__":
    APP = sys.argv[1] if len(sys.argv) > 1 else "ttt"
    STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 6000
    torch.manual_seed(0); rng = np.random.default_rng(0)
    D, feats, held, src, dst = graph(APP)
    S = len(feats)
    tr_edges = ~held[src] & ~held[dst]  # the judge only learns from experience inside the training screens
    full = shortest_path(csr_matrix((np.ones(len(src)), (src, dst)), (S, S)), unweighted=True)
    train_d = shortest_path(csr_matrix((np.ones(tr_edges.sum()), (src[tr_edges], dst[tr_edges])), (S, S)), unweighted=True)
    lab = lambda m: torch.from_numpy(np.minimum(np.nan_to_num(m, posinf=K - 1), K - 1).astype(np.int64))
    full, train_d = lab(full), lab(train_d)
    tr_s, te_s = np.where(~held)[0], np.where(held)[0]
    near = [[np.where(train_d[a].numpy() == k)[0] for k in range(K)] for a in tr_s]  # goals by distance, per start

    def batch(n=128):  # half: goal at a uniformly drawn distance (so short ones are seen), half: any training screen
        ia = rng.integers(len(tr_s), size=n)
        a = tr_s[ia]
        b = rng.choice(tr_s, n)
        for i in range(n // 2):
            ks = [k for k in range(K) if len(near[ia[i]][k])]
            b[i] = rng.choice(near[ia[i]][rng.choice(ks)])
        return a, b

    j = Judge().to(dev)
    opt = torch.optim.AdamW(j.parameters(), 3e-4, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-4, total_steps=STEPS, pct_start=0.05)
    for step in range(STEPS):
        a, b = batch()
        za, zb = feats[a], feats[b]
        za = za + 0.05 * za.std() * torch.randn_like(za)
        with torch.autocast("cuda", torch.bfloat16):
            loss = F.cross_entropy(j(za, zb).float(), train_d[a, b].to(dev))
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
        if step % 1000 == 0 or step == STEPS - 1:
            print(f"step {step:>5}  loss {loss.item():.4f}", flush=True)
    torch.save(j.state_dict(), D / "judge.pt")
    j.eval()

    # held-out start screens, goals at true distance 1..7 (true = full experience graph); MSE ranking as the baseline
    res = {}
    print(f"\n=== {APP}: judge on held-out start screens (true steps from the full experience graph) ===")
    print(f"{'true steps':>10} {'n':>5} {'exact':>7} {'within 1':>9} {'mean guess':>11} | ranking: judge  vs  MSE")
    for k in range(1, 8):
        pairs = [(a, rng.choice(np.where(full[a].numpy() == k)[0])) for a in rng.choice(te_s, 400) if (full[a] == k).any()][:300]
        if not pairs:
            continue
        a, b = map(np.array, zip(*pairs))
        g = j.steps(feats[a], feats[b]).cpu()
        # ranking test: is a goal k steps away judged nearer than a goal k+2 steps away (same start)?
        far = [rng.choice(np.where(full[x].numpy() == k + 2)[0]) if (full[x] == k + 2).any() else -1 for x in a]
        ok = np.array(far) >= 0
        rj = rm = float("nan")
        if ok.any():
            fa = np.array(far)[ok]
            rj = (g[ok] < j.steps(feats[a[ok]], feats[fa]).cpu()).float().mean().item()
            mse = lambda x, y: (feats[x] - feats[y]).pow(2).mean((1, 2)).cpu()
            rm = (mse(a[ok], b[ok]) < mse(a[ok], fa)).float().mean().item()
        res[k] = {"n": len(a), "exact": (g.round() == k).float().mean().item(), "within1": ((g - k).abs() <= 1).float().mean().item(),
                  "mean_guess": g.mean().item(), "rank_judge": rj, "rank_mse": rm}
        r = res[k]
        print(f"{k:>10} {r['n']:>5} {r['exact']:>7.1%} {r['within1']:>9.1%} {r['mean_guess']:>11.2f} | {rj:>13.1%} {rm:>7.1%}")
    (Path(__file__).parent.parent / "results" / f"judge_{APP}_metrics.json").write_text(json.dumps(res, indent=2))
