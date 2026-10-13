"""Level 3c: PLAY Snake by imagination, in the real browser, with time running.

No goal screen and no hand-written rules about food or walls. Three small SENSES are learned from the recorded games
(real screens only) and then used to judge IMAGINED screens:
  EAT   : did the score just go up?          (before, after) -> probability
  DEATH : is this a game-over screen?        screen -> probability
  VALUE : how many steps until I eat?        screen -> steps (learned from past games: the 'practice' signal)
Each step: imagine every key sequence H steps ahead (5 keys: Up/Down/Left/Right/none) with the 3b IMAGINER's likeliest
future; a sequence that eats scores high, one that dies scores very low, otherwise lower VALUE is better.
Press the first key of the best sequence, tick the clock, look again.
Agents: random keys | imagine 1 step | imagine 3 steps.  Fresh games (new random food), not from the dataset.
"""
import io, json, random, sys, time
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image
from playwright.sync_api import sync_playwright
from model import MultiImaginer, Eyes, dev
from snake_data import D, states, train, kinds
import store

ROOT = Path(__file__).parent
R = ROOT.parent / "results"
GAMES = int(sys.argv[1]) if len(sys.argv) > 1 else 20
MAX_STEPS = 200
KEYS = ["ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight", None]  # ids 0..4 match the training key ids
torch.manual_seed(0); random.seed(0); np.random.seed(0)

feats = torch.from_numpy(store.load(D / "feats.npy", mmap=False))
get = lambda idx: feats[idx].to(dev).float()
imag = MultiImaginer(K=64, keys=6).to(dev)
imag.load_state_dict(torch.load(D / "imaginer_multi.pt"))
imag.eval()

# ---------------------------------------------------------------- learn the three senses from real recorded screens
T = len(kinds)
alive_tr = [t for t in train.tolist() if not states[t]["dead"]]
eat_lbl = torch.tensor([float(states[t + 1]["score"] > states[t]["score"] and not states[t]["dead"]) for t in range(T)])
dead_lbl = torch.tensor([float(s["dead"]) for s in states])
to_eat = np.full(T + 1, 30.0)  # steps until the next eat in the same game (30 = not soon / dies first)
for t in range(T - 1, -1, -1):
    if states[t]["dead"] or states[t + 1]["dead"]:
        to_eat[t] = 30.0
    elif eat_lbl[t]:
        to_eat[t] = 1.0
    else:
        to_eat[t] = min(30.0, to_eat[t + 1] + 1)
to_eat = torch.from_numpy(to_eat).float()


class Value(nn.Module):  # small transformer: screen tokens -> steps until eat
    def __init__(self, w=128):
        super().__init__()
        self.inp, self.pos = nn.Linear(384, w), nn.Parameter(torch.zeros(1, 256, w))
        self.tf = nn.TransformerEncoder(nn.TransformerEncoderLayer(w, 4, 2 * w, 0.0, batch_first=True, norm_first=True), 2)
        self.out = nn.Linear(w, 1)

    def forward(self, z):
        return self.out(self.tf(self.inp(z) + self.pos).mean(1)).squeeze(-1)


status = lambda z: z.view(-1, 16, 16, 384)[:, :2].mean((1, 2))  # tokens over the status line (score text)
eat_net = nn.Linear(2 * 384, 1).to(dev)  # (status-line change, overall change) -> did we eat?
dead_net = nn.Linear(2 * 384, 1).to(dev)  # (status line, whole screen) -> game over?
value = Value().to(dev)
eat_feat = lambda z, z1: torch.cat([status(z1) - status(z), (z1 - z).abs().mean(1)], -1)
dead_feat = lambda z: torch.cat([status(z), z.mean(1)], -1)
opt = torch.optim.AdamW([*eat_net.parameters(), *dead_net.parameters(), *value.parameters()], 1e-3)
eats = [t for t in alive_tr if eat_lbl[t]]
deaths = [t for t in alive_tr if dead_lbl[t + 1]]
for it in range(3000):
    b = torch.tensor(random.sample(alive_tr, 160) + random.sample(eats, 64) + random.sample(deaths, 32))  # oversample rare steps
    z, z1 = get(b), get(b + 1)
    loss = F.binary_cross_entropy_with_logits(eat_net(eat_feat(z, z1)).squeeze(-1), eat_lbl[b].to(dev)) \
        + F.binary_cross_entropy_with_logits(dead_net(dead_feat(z1)).squeeze(-1), dead_lbl[b + 1].to(dev)) \
        + F.smooth_l1_loss(value(z), to_eat[b].to(dev))
    opt.zero_grad(); loss.backward(); opt.step()
print(f"senses trained (last loss {loss.item():.3f})", flush=True)


@torch.no_grad()
def plan(z0, H):
    """Imagine every key sequence of length H. Returns the best first key."""
    Z, first = z0, None
    score = torch.full((5 ** H,), float("nan"), device=dev)
    done = None
    for h in range(H):
        n = len(Z)
        keys = torch.arange(5, device=dev).repeat(n)
        with torch.autocast("cuda", torch.bfloat16):
            fut, _, prior = imag(Z.repeat_interleave(5, 0), k=keys)
        Z1 = fut.float()[torch.arange(len(keys)), prior.float().argmax(1)]  # likeliest future
        Zp = Z.repeat_interleave(5, 0)
        p_eat = eat_net(eat_feat(Zp, Z1)).squeeze(-1).sigmoid()
        p_dead = dead_net(dead_feat(Z1)).squeeze(-1).sigmoid()
        first = keys if first is None else first.repeat_interleave(5)
        done = torch.zeros(len(keys), dtype=torch.bool, device=dev) if done is None else done.repeat_interleave(5)
        rep = 5 ** (H - h - 1)  # each node stands for this many full-length sequences
        s = torch.full((len(keys),), float("nan"), device=dev)
        s[(p_dead > 0.5) & ~done] = -100.0 + h  # dying later is a little less bad
        s[(p_eat > 0.5) & (p_dead <= 0.5) & ~done] = 100.0 - h  # eating sooner is better
        new_done = ~torch.isnan(s)
        score_view = score.view(len(keys), rep)
        score_view[new_done] = s[new_done, None]
        done = done | new_done
        Z = Z1
    open_ = torch.isnan(score)
    if open_.any():
        v = value(Z).clamp(1, 30)  # steps until eat at the end of the imagined path
        score[open_] = -v[open_]
    first_full = first
    best = torch.full((5,), -1e9, device=dev)
    for k in range(5):
        best[k] = score[first_full == k].max()
    return int(best.argmax())


def play(agent, H, page, eyes):
    page.evaluate("reset()")
    food, ms = 0, []
    for step in range(MAX_STEPS):
        s = page.evaluate("window.__state")
        if s["dead"]:
            return food, step, True, ms
        if agent == "random":
            k = random.randrange(5)
        else:
            t0 = time.perf_counter()
            o = np.asarray(Image.open(io.BytesIO(page.screenshot())).convert("RGB"))
            k = plan(eyes(o[None]), H)
            torch.cuda.synchronize(); ms.append((time.perf_counter() - t0) * 1000)
        if KEYS[k]:
            page.keyboard.press(KEYS[k])
        page.evaluate("step()")
        food += page.evaluate("window.__state").get("score", 0) > s["score"]
    return food, MAX_STEPS, False, ms


eyes = Eyes()
res = {}
with sync_playwright() as p:
    page = p.chromium.launch().new_page(viewport={"width": 224, "height": 224})
    page.goto((ROOT / "snake.html").as_uri())
    for agent, H in [("random", 0), ("imagine 1 step", 1), ("imagine 3 steps", 3)]:
        runs = [play(agent, H, page, eyes) for _ in range(GAMES)]
        f, st, dd, ms = zip(*runs)
        allms = [m for x in ms for m in x]
        res[agent] = {"food_per_game": float(np.mean(f)), "food_per_100_steps": 100 * sum(f) / sum(st),
                      "died": float(np.mean(dd)), "steps_survived": float(np.mean(st)),
                      "decision_ms_median": float(np.median(allms)) if allms else None}
        print(f"{agent:<16} food/game {res[agent]['food_per_game']:5.2f}  food/100 steps {res[agent]['food_per_100_steps']:5.2f}  "
              f"died {res[agent]['died']:5.0%}  survived {res[agent]['steps_survived']:6.1f} steps"
              + (f"  decision {res[agent]['decision_ms_median']:.0f} ms" if allms else ""), flush=True)

bot = {"food_per_100_steps": 100 * 2664 / 40000}
print(f"(reference: the data bot, which reads the true game state, ate {bot['food_per_100_steps']:.1f} food / 100 steps)")
r, a = res["random"], res["imagine 3 steps"]
ok = a["food_per_game"] >= 3 * max(r["food_per_game"], 0.1) and a["died"] <= r["died"]
print(f"\nKILL TEST: {'PASS' if ok else 'FAIL'}  (needs imagine-3 to eat >= 3x random's food per game and die no more often)")
(R / "level3c_metrics.json").write_text(json.dumps(res, indent=2))
