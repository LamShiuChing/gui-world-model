"""Level 3d: SELF-PLAY on Snake - the model teaches itself (model-based RL). No expert, no human data.

Round 0: IMAGINER (3b) + SENSES learned from the random-bot data only.
Each round:  PLAY 40 games (imagine -> act, 5% random keys to explore), recording screens/keys/outcomes
          -> RETRAIN IMAGINER + SENSES on bot data + ALL self-play so far
             VALUE = discounted future reward from ITS OWN games (+1 eat, -1 death, gamma 0.95)
          -> MEASURE on 20 fresh games (no randomness)
Planner: imagine every 2-key sequence; score = rewards along the way (EAT +1, DEATH -1 stops) + gamma^2 * VALUE(end).

usage: python level0/snake_selfplay.py [ROUNDS=5]
"""
import io, json, random, sys, time
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image
from playwright.sync_api import sync_playwright
from model import MultiImaginer, Eyes, dev
from snake_data import D, states as B_states, acts as B_acts, train as B_train
import store

ROOT = Path(__file__).parent
R = ROOT.parent / "results"
OUT = ROOT.parent / "data" / "snake_selfplay"
ROUNDS = int(sys.argv[1]) if len(sys.argv) > 1 else 5
PLAY_GAMES, EVAL_GAMES, MAX_STEPS, H, EPS, GAMMA = 40, 20, 500, 2, 0.05, 0.95
KEYS = ["ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight", None]
torch.manual_seed(0); random.seed(0); np.random.seed(0)
OUT.mkdir(parents=True, exist_ok=True)


def returns(rew, alive_after):  # discounted future reward per step, cut at death / end of a game segment
    G, out = 0.0, np.zeros(len(rew), np.float32)
    for t in range(len(rew) - 1, -1, -1):
        G = rew[t] + (GAMMA * G if alive_after[t] else 0.0)
        out[t] = G
    return out


# ---------------------------------------------------------------- bot data (train games only), as one transition table
B_feats = torch.from_numpy(store.load(D / "feats.npy", mmap=False))  # T+1,256,384 fp16
B_masks = torch.from_numpy(np.load(D / "masks.npy"))
T = len(B_acts)
b_alive = np.array([not B_states[t]["dead"] for t in range(T)])
b_eat = np.array([float(B_states[t + 1]["score"] > B_states[t]["score"]) * b_alive[t] for t in range(T)], np.float32)
b_die = np.array([float(B_states[t + 1]["dead"]) * b_alive[t] for t in range(T)], np.float32)
b_ret = returns(b_eat - b_die, b_alive & (b_die == 0))
b_idx = B_train.numpy()
b_alive_idx = b_idx[b_alive[b_idx]]

# self-play store (grows every round): frames z (fp16), per-transition (i -> i+1) masks, keys, eat/die/return labels
S = {"z": [], "trans": [], "mask": [], "key": [], "eat": [], "die": [], "ret": []}

# ---------------------------------------------------------------- models
imag = MultiImaginer(K=64, keys=6).to(dev)
imag.load_state_dict(torch.load(D / "imaginer_multi.pt"))
rho = B_masks[b_idx].float().mean().item()
lam = min((1 - rho) / rho, 50.0)


class Value(nn.Module):  # screen tokens -> discounted future reward
    def __init__(self, w=128):
        super().__init__()
        self.inp, self.pos = nn.Linear(384, w), nn.Parameter(torch.zeros(1, 256, w))
        self.tf = nn.TransformerEncoder(nn.TransformerEncoderLayer(w, 4, 2 * w, 0.0, batch_first=True, norm_first=True), 2)
        self.out = nn.Linear(w, 1)

    def forward(self, z):
        return self.out(self.tf(self.inp(z) + self.pos).mean(1)).squeeze(-1)


status = lambda z: z.view(-1, 16, 16, 384)[:, :2].mean((1, 2))  # tokens over the status line (score text)
eat_feat = lambda z, z1: torch.cat([status(z1) - status(z), (z1 - z).abs().mean(1)], -1)
dead_feat = lambda z: torch.cat([status(z), z.mean(1)], -1)
eat_net, dead_net, value = nn.Linear(768, 1).to(dev), nn.Linear(768, 1).to(dev), Value().to(dev)
s_opt = torch.optim.AdamW([*eat_net.parameters(), *dead_net.parameters(), *value.parameters()], 1e-3)


def batch(n_bot, n_sp, alive_only=False):
    """Mixed batch of transitions: (z, z1, mask, key, eat, die, ret) on the GPU."""
    bi = np.random.choice(b_alive_idx if alive_only else b_idx, n_bot)
    parts = [(B_feats[bi], B_feats[bi + 1], B_masks[bi], B_acts[bi], b_eat[bi], b_die[bi], b_ret[bi])]
    if n_sp and S["trans"]:
        si = np.random.randint(len(S["trans"]), size=n_sp)
        i = torch.from_numpy(np.array(S["trans"])[si])
        parts.append((SZ[i], SZ[i + 1], torch.from_numpy(np.array(S["mask"])[si]), torch.tensor(np.array(S["key"])[si]),
                      np.array(S["eat"])[si], np.array(S["die"])[si], np.array(S["ret"])[si]))
    cols = list(zip(*parts))
    z, z1, m, k = (torch.cat([torch.as_tensor(c) for c in col]).to(dev) for col in cols[:4])
    lab = [torch.from_numpy(np.concatenate(col)).float().to(dev) for col in cols[4:]]
    return z.float(), z1.float(), m.float(), k.long(), *lab


def train_imaginer(steps):
    opt = torch.optim.AdamW(imag.parameters(), 1e-4, weight_decay=0.01)
    wmse = lambda p, t, w: ((p - t).pow(2).mean(-1) * w).sum(-1) / w.sum(-1)
    for _ in range(steps):
        z, z1, m, k, *_ = batch(32, 32)
        with torch.autocast("cuda", torch.bfloat16):
            h = imag.features(z, k=k)
            base = z + imag.delta(h)
            hd = h.detach()
            fut = base.detach()[:, None] + imag.head(hd[:, None] + imag.code.weight[None, :, None])
            logit, prior = imag.mask(h).squeeze(-1), imag.prior(hd.mean(1))
        w = 1 + lam * m
        L = wmse(fut.float(), z1[:, None], w[:, None])
        best = L.argmin(1)
        loss = wmse(base.float(), z1, w).mean() + L.gather(1, best[:, None]).mean() + 0.01 * L.mean() \
            + F.cross_entropy(prior.float(), best) + F.binary_cross_entropy_with_logits(logit.float(), m)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()


def train_senses(steps):
    for _ in range(steps):
        n_sp = 128 if S["trans"] else 0
        z, z1, _, _, eat, die, ret = batch(256 - n_sp, n_sp, alive_only=True)
        loss = F.binary_cross_entropy_with_logits(eat_net(eat_feat(z, z1)).squeeze(-1), eat, pos_weight=torch.tensor(5.0, device=dev)) \
            + F.binary_cross_entropy_with_logits(dead_net(dead_feat(z1)).squeeze(-1), die, pos_weight=torch.tensor(5.0, device=dev)) \
            + F.smooth_l1_loss(value(z), ret)
        s_opt.zero_grad(); loss.backward(); s_opt.step()


@torch.no_grad()
def plan(z0):
    """Imagine every key sequence of length H; return the first key of the best one."""
    Z, score, alive, first = z0, torch.zeros(1, device=dev), torch.ones(1, dtype=torch.bool, device=dev), None
    for h in range(H):
        keys = torch.arange(5, device=dev).repeat(len(Z))
        with torch.autocast("cuda", torch.bfloat16):
            fut, _, prior = imag(Z.repeat_interleave(5, 0), k=keys)
        Z1 = fut.float()[torch.arange(len(keys)), prior.float().argmax(1)]  # likeliest future
        Zp = Z.repeat_interleave(5, 0)
        eat = (eat_net(eat_feat(Zp, Z1)).squeeze(-1) > 0).float()
        die = dead_net(dead_feat(Z1)).squeeze(-1) > 0
        score, alive = score.repeat_interleave(5), alive.repeat_interleave(5)
        first = keys if first is None else first.repeat_interleave(5)
        score = score + alive * (GAMMA ** h) * (eat - die.float())
        alive = alive & ~die
        Z = Z1
    score = score + alive * (GAMMA ** H) * value(Z)
    return int(first[score.argmax()])


def play(page, eyes, eps, record):
    page.evaluate("reset()")
    food, ms, frames, prev_img = 0, [], [], None
    for step in range(MAX_STEPS):
        s = page.evaluate("window.__state")
        img = np.asarray(Image.open(io.BytesIO(page.screenshot())).convert("RGB"))
        t0 = time.perf_counter()
        z = eyes(img[None])
        k = random.randrange(5) if random.random() < eps else plan(z)
        torch.cuda.synchronize(); ms.append((time.perf_counter() - t0) * 1000)
        if record:
            frames.append((z[0].half().cpu(), img, k, s))
        if s["dead"]:
            break
        if KEYS[k]:
            page.keyboard.press(KEYS[k])
        page.evaluate("step()")
        food += page.evaluate("window.__state")["score"] > s["score"]
    died = page.evaluate("window.__state")["dead"]
    if record and len(frames) > 1:  # store transitions of this game
        base = sum(len(x) for x in S["z"]) if S["z"] else 0
        zs = torch.stack([f[0] for f in frames])
        rew, alive_after = [], []
        for t in range(len(frames) - 1):
            s0, s1 = frames[t][3], frames[t + 1][3]
            e, d = float(s1["score"] > s0["score"]), float(s1["dead"])
            rew.append(e - d); alive_after.append(not s1["dead"])
            S["trans"].append(base + t)
            S["mask"].append((frames[t][1] != frames[t + 1][1]).any(-1).reshape(16, 14, 16, 14).any((1, 3)).reshape(256))
            S["key"].append(frames[t][2]); S["eat"].append(e); S["die"].append(d)
        S["ret"].extend(returns(np.array(rew, np.float32), np.array(alive_after)).tolist())
        S["z"].append(zs)
    return food, step + 1, died, ms


def evaluate(page, eyes, games, eps=0.0, record=False):
    runs = [play(page, eyes, eps, record) for _ in range(games)]
    f, st, dd, ms = zip(*runs)
    return {"food_per_game": float(np.mean(f)), "food_max": int(max(f)), "steps_survived": float(np.mean(st)),
            "died": float(np.mean(dd)), "decision_ms_median": float(np.median([m for x in ms for m in x]))}


log = []
eyes = Eyes()
with sync_playwright() as p:
    page = p.chromium.launch().new_page(viewport={"width": 224, "height": 224})
    page.goto((ROOT / "snake.html").as_uri())
    t_all = time.time()
    train_senses(3000)  # round 0: senses from bot data only
    for r in range(ROUNDS + 1):
        if r > 0:
            t0 = time.time()
            evaluate(page, eyes, PLAY_GAMES, eps=EPS, record=True)  # self-play games -> data
            SZ = torch.cat(S["z"])
            torch.cuda.empty_cache()  # free play-phase blocks: a full 16 GB card makes Windows page GPU memory (5x slower)
            train_imaginer(1500)
            train_senses(1500)
            torch.cuda.empty_cache()
            print(f"  round {r}: self-play data {len(S['trans'])} transitions, retrained in {time.time() - t0:.0f} s", flush=True)
        else:
            SZ = None
        m = evaluate(page, eyes, EVAL_GAMES)
        m.update(round=r, selfplay_transitions=len(S["trans"]))
        log.append(m)
        print(f"round {r}: food/game {m['food_per_game']:5.2f} (best {m['food_max']:2d})  survived {m['steps_survived']:6.1f}  "
              f"died {m['died']:4.0%}  decision {m['decision_ms_median']:.0f} ms  [{(time.time() - t_all) / 60:.0f} min]", flush=True)
        (R / "level3d_metrics.json").write_text(json.dumps(log, indent=2))

torch.save({"imag": imag.state_dict(), "eat": eat_net.state_dict(), "dead": dead_net.state_dict(), "value": value.state_dict()},
           OUT / "selfplay_models.pt")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
ax[0].plot([m["round"] for m in log], [m["food_per_game"] for m in log], "o-"); ax[0].set(xlabel="self-play round", ylabel="food per game")
ax[1].plot([m["round"] for m in log], [m["steps_survived"] for m in log], "o-", color="tab:green"); ax[1].set(xlabel="self-play round", ylabel="steps survived")
fig.suptitle("Snake: the model teaches itself (no expert, no human data)"); fig.tight_layout(); fig.savefig(R / "level3d_curve.png", dpi=120)
print(f"saved {R / 'level3d_curve.png'}")
