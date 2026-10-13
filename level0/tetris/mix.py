"""MIX: does pretraining on OTHER apps (tic-tac-toe + Snake + Paint) cut the Tetris data needed?

A) Tetris IMAGINER from scratch                       with 5% / 20% / 100% of the Tetris training games
B) IMAGINER pretrained on tic-tac-toe + Snake + Paint, then Tetris with the same 5% / 20% / 100%
Same architecture, same Tetris steps, same held-out games. One action vocabulary for all apps:
key ids 0-5 Snake keys, 6 stroke, 7 click (Paint + tic-tac-toe, with a segment), 8-15 Tetris keys.
Pretraining is at 224px (256 patches); for Tetris (336px, 576 patches) the position map is stretched 16x16 -> 24x24.

usage: python level0/tetris/mix.py [PRETRAIN_STEPS=8000] [TETRIS_STEPS=5000]
"""
import json, sys, time
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).parent.parent))
import data as T
from model import Imaginer
import store

PRE = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
TET = int(sys.argv[2]) if len(sys.argv) > 2 else 5000
FRACS = (0.05, 0.2, 1.0)
ROOT = Path(__file__).parent.parent.parent
R, dev = ROOT / "results", "cuda"
torch.backends.cuda.matmul.allow_tf32 = True
wmse = lambda p, t, w: (w * (p - t).pow(2).mean(-1)).sum() / w.sum()
new_model = lambda tokens: Imaginer(keys=16, tokens=tokens, history=True, segments=True).to(dev)

# ---------------------------------------------------------------- pretraining sources
L0 = ROOT / "data" / "level0"  # tic-tac-toe: unique states + (src, click, dst)
ttt_f = torch.from_numpy(store.load(L0 / "feats.npy", mmap=False)).to(dev)  # fp16 on GPU (1 GB)
ttt_img = torch.from_numpy(np.load(L0 / "images.npy")).to(dev)
tt = np.load(L0 / "transitions.npz")
ttt_src, ttt_dst = torch.from_numpy(tt["src"]).long(), torch.from_numpy(tt["dst"]).long()
ttt_seg = torch.from_numpy(np.stack([tt["x"], tt["y"], tt["x"], tt["y"]], 1) / 224).float()
ttt_held = torch.load(L0 / "imaginer.pt")["held"]
ttt_train = torch.where(~ttt_held[ttt_src] & ~ttt_held[ttt_dst])[0]

SN = ROOT / "data" / "snake"
sn_f, sn_m = store.load(SN / "feats.npy"), torch.from_numpy(np.load(SN / "masks.npy"))
sn_acts = np.array(json.loads((SN / "actions.json").read_text()))
sn_states = json.loads((SN / "states.json").read_text())
sn_ep = np.cumsum([0] + [sn_states[t]["dead"] and not sn_states[t + 1]["dead"] for t in range(len(sn_acts) - 1)])
sn_held = np.random.default_rng(0).random(sn_ep.max() + 1) < 0.2  # same split as snake_data.py
sn_train = np.where(~sn_held[sn_ep])[0]
sn_prev = np.r_[False, sn_ep[1:] == sn_ep[:-1]]
sn_starts = T.chunk_starts(sn_train, 4, len(sn_acts))

PA = ROOT / "data" / "paint"
pa_f, pa_m = store.load(PA / "feats_224.npy"), torch.from_numpy(np.load(PA / "masks_224.npy"))
pa_fr = np.load(PA / "frames.npz")
pa_valid, pa_worker = pa_fr["action"] >= 0, pa_fr["game"] // 1000
pa_seg, pa_kind = [], []
for w in sorted(PA.glob("w[0-9]")):
    for x in json.loads((w / "actions.json").read_text()) + [[1, 0, 0, 0, 0]]:
        pa_seg.append([x[1], x[2], x[3], x[4]] if x[0] == 0 else [x[1], x[2], x[1], x[2]]); pa_kind.append(x[0])
pa_seg, pa_kind = torch.tensor(pa_seg, dtype=torch.float32) / 224, torch.tensor(pa_kind)
pa_train = np.where(pa_valid & (pa_worker != pa_worker.max()))[0]  # same held-out worker as paint_train.py
pa_prev = np.r_[False, pa_valid[:-1]]
pa_starts = T.chunk_starts(pa_train, 4, len(pa_valid))
LAM = {"ttt": 21.6, "snake": 13.4, "paint": 21.1, "tetris": 14.2}  # (1-rho)/rho measured per dataset


def ttt_batch(rng):
    b = ttt_train[torch.from_numpy(rng.integers(len(ttt_train), size=64))]
    s, d = ttt_src[b].to(dev), ttt_dst[b].to(dev)
    m = (ttt_img[s] != ttt_img[d]).any(-1).view(-1, 16, 14, 16, 14).any(4).any(2).flatten(1).float()
    z = ttt_f[s].float()
    return "ttt", z, ttt_f[d].float(), z, m, torch.full((64,), 7, device=dev), ttt_seg[b].to(dev)


def seq_batch(name, F_, M, P, starts, rng, keys, segs=None):
    b, z, z1, zp = T.chunk_batch(starts, rng, F=F_, P=P)
    g = lambda x: torch.from_numpy(x).to(dev).float()
    return name, g(z), g(z1), g(zp), M[b].float().to(dev), keys(b).to(dev), (segs[b].to(dev) if segs is not None else None)


def pretrain_batches(rng):
    while True:
        yield ttt_batch(rng)
        yield seq_batch("snake", sn_f, sn_m, sn_prev, sn_starts, rng, lambda b: torch.from_numpy(sn_acts[b]).long())
        yield seq_batch("paint", pa_f, pa_m, pa_prev, pa_starts, rng, lambda b: 6 + pa_kind[b], pa_seg)


def train(model, batches, steps, tag):
    opt = torch.optim.AdamW(model.parameters(), 3e-4, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-4, total_steps=steps, pct_start=0.05)
    t0 = time.time()
    for step in range(steps):
        name, z, z1, zp, m, k, s = next(batches)
        with torch.autocast("cuda", torch.bfloat16):
            p, lg = model(z, k=k, z_prev=zp, s=s)
        loss = wmse(p.float(), z1, 1 + LAM[name] * m) + F.binary_cross_entropy_with_logits(lg.float(), m)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
        if step % 1000 == 0 or step == steps - 1:
            print(f"  [{tag}] step {step:>5}  loss {loss.item():.4f}  ({time.time() - t0:.0f} s)", flush=True)


def tetris_batches(pool, rng):
    starts = T.chunk_starts(pool, 4)
    def make():
        b, z, z1, zp = T.chunk_batch(starts, rng)
        return b, z, z1, zp
    for b, z, z1, zp in T.prefetch(make):
        g = lambda x: torch.from_numpy(x).to(dev).float()
        yield "tetris", g(z), g(z1), g(zp), T.masks[b].float().to(dev), torch.from_numpy(T.acts[b] + 8).long().to(dev), None


def stretch(state):  # pretrained 256-patch position map -> 576 patches
    pos = state["pos"]
    w = pos.shape[-1]
    state["pos"] = F.interpolate(pos.view(1, 16, 16, w).permute(0, 3, 1, 2), size=(24, 24), mode="bilinear",
                                 align_corners=False).permute(0, 2, 3, 1).reshape(1, 576, w)
    return state


# ---------------------------------------------------------------- shared evaluation on held-out Tetris games
rng0 = np.random.default_rng(0)
ev = np.sort(rng0.choice(T.test, 3000, replace=False))
ek = T.kinds[np.searchsorted(T.trans, ev)]
reader = nn.Linear(384, 1).to(dev)
ropt = torch.optim.Adam(reader.parameters(), 1e-3)
read = lambda zz: reader(zz[:, T.PIDX]).squeeze(-1)
r_starts = T.chunk_starts(T.train, 4)
for _ in range(1500):  # board reader on REAL train screens
    b, z, _, _ = T.chunk_batch(r_starts, rng0)
    loss = F.binary_cross_entropy_with_logits(read(torch.from_numpy(z).to(dev).float()), torch.stack([T.board(t) for t in b]).to(dev))
    ropt.zero_grad(); loss.backward(); ropt.step()
EV = []
for i in range(0, len(ev), 64):  # cache the eval set once (random reads, done one time)
    b = ev[i:i + 64]
    EV.append((b, T.get(b).half().cpu(), T.get(b + 1).half().cpu(), T.get(T.prev_of(b)).half().cpu()))  # CPU: 4 GB
MAIN = ["falls (gravity / soft drop)", "move sideways", "rotate", "lock + new piece", "line clear", "hold"]


@torch.no_grad()
def evaluate(model):
    model.eval()
    err, base, ok, n = {g: 0.0 for g in MAIN}, {g: 0.0 for g in MAIN}, 0.0, 0.0
    for b, z, z1, zp in EV:
        z, z1, zp = z.to(dev).float(), z1.to(dev).float(), zp.to(dev).float()
        with torch.autocast("cuda", torch.bfloat16):
            p, _ = model(z, k=torch.from_numpy(T.acts[b] + 8).long().to(dev), z_prev=zp)
        e, e0 = (p.float() - z1).pow(2).mean((1, 2)), (z - z1).pow(2).mean((1, 2))
        kk = ek[np.searchsorted(ev, b)]
        for g in MAIN:
            sel = torch.from_numpy(kk == g).to(dev)
            err[g] += e[sel].sum().item(); base[g] += e0[sel].sum().item()
        b0, b1 = torch.stack([T.board(t) for t in b]).to(dev), torch.stack([T.board(t + 1) for t in b]).to(dev)
        ch = b0 != b1
        ok += ((read(p.float()) > 0) == (b1 > 0.5))[ch].float().sum().item(); n += ch.sum().item()
    model.train()
    out = {g: err[g] / base[g] for g in MAIN if base[g] > 0}
    out["moves+rotations+falls"] = sum(err[g] for g in MAIN[:3]) / sum(base[g] for g in MAIN[:3])
    out["changed_cells"] = ok / n
    return out


# ---------------------------------------------------------------- run
res = {"pretrain_steps": PRE, "tetris_steps": TET, "runs": []}
print(f"pretraining on tic-tac-toe ({len(ttt_train)}) + Snake ({len(sn_train)}) + Paint ({len(pa_train)}) transitions", flush=True)
torch.manual_seed(0)
pre = new_model(256)
train(pre, pretrain_batches(np.random.default_rng(1)), PRE, "pretrain")
pre_state = {k: v.detach().clone() for k, v in pre.state_dict().items()}
del pre; torch.cuda.empty_cache()

games = np.unique(T.game[T.train])
for frac in FRACS:
    sub = np.random.default_rng(5).choice(games, max(1, int(round(frac * len(games)))), replace=False)
    pool = T.train[np.isin(T.game[T.train], sub)]
    for cond in ("scratch", "pretrained"):
        torch.manual_seed(0)
        model = new_model(576)
        if cond == "pretrained":
            model.load_state_dict(stretch(dict(pre_state)))
        train(model, tetris_batches(pool, np.random.default_rng(2)), TET, f"{cond} {frac:.0%}")
        m = evaluate(model)
        m.update(condition=cond, fraction=frac, tetris_games=len(sub), tetris_transitions=len(pool))
        res["runs"].append(m)
        print(f"{cond:>10} {frac:>5.0%} ({len(pool):>6} transitions): moves/rotations/falls {m['moves+rotations+falls']:.3f}  "
              f"changed cells {m['changed_cells']:.1%}", flush=True)
        (R / "mix_metrics.json").write_text(json.dumps(res, indent=2))
        del model; torch.cuda.empty_cache()

print("\n=== MIX: Tetris prediction on held-out games (error vs 'nothing changes', lower = better | changed board cells right) ===")
print(f"{'Tetris data':>12} {'scratch':>22} {'pretrained on 3 apps':>24}")
for frac in FRACS:
    a, b_ = [r for r in res["runs"] if r["fraction"] == frac]
    print(f"{frac:>12.0%} {a['moves+rotations+falls']:>10.3f} | {a['changed_cells']:>7.1%}   {b_['moves+rotations+falls']:>10.3f} | {b_['changed_cells']:>7.1%}")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
for cond, c in (("scratch", "tab:gray"), ("pretrained", "tab:blue")):
    rs = [r for r in res["runs"] if r["condition"] == cond]
    ax[0].plot([r["fraction"] for r in rs], [r["moves+rotations+falls"] for r in rs], "o-", color=c, label=cond)
    ax[1].plot([r["fraction"] for r in rs], [r["changed_cells"] for r in rs], "o-", color=c, label=cond)
for a_, yl in zip(ax, ("error vs 'nothing changes' (lower = better)", "changed board cells right")):
    a_.set(xscale="log", xlabel="fraction of Tetris training data", ylabel=yl); a_.legend()
fig.suptitle("Does pretraining on other apps help Tetris?"); fig.tight_layout(); fig.savefig(R / "mix_curve.png", dpi=120)
print(f"saved {R / 'mix_curve.png'}")
