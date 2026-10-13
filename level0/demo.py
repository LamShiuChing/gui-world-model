"""Demo: watch the model play, live, and save results/demo.gif.

  1. Tic-tac-toe  - reach a goal board 3 clicks away by imagining chains of clicks (Level 2, lookahead 3)
  2. Snake        - play a real game with the self-play-trained model (Level 3d: imagine 2 keys ahead + learned senses)
  3. Tetris       - imagination vs reality: the board the model IMAGINES after each real key vs what really happens
                    (Tetris cannot be played yet: the T1 'act' level is not built)
Starting boards / games are never seen in training.

usage: python level0/demo.py [live]       live = also show an enlarged window while it runs
"""
import io, json, random, sys, time
from collections import deque
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from playwright.sync_api import sync_playwright
from model import Imaginer, MultiImaginer, Eyes, CENTERS, dev
from ttt_rules import step as ttt_step
from collect import key as ttt_key
import store
from webp import img

ROOT = Path(__file__).parent
DATA, R = ROOT.parent / "data", ROOT.parent / "results"
LIVE = "live" in sys.argv
W, H = 760, 420
random.seed(7); torch.manual_seed(0)
font = lambda n: ImageFont.truetype("arial.ttf", n)
frames = []
if LIVE:
    import cv2


def show(title, tiles, note="", ms=500):
    """tiles: list of (PIL image, caption). Draws one demo frame, shows it live, keeps it for the GIF."""
    canvas = Image.new("RGB", (W, H), (24, 26, 30))
    d = ImageDraw.Draw(canvas)
    d.text((16, 10), title, fill=(255, 255, 255), font=font(20))
    tw = (W - 32 - 16 * (len(tiles) - 1)) // len(tiles)
    for i, (im, cap) in enumerate(tiles):
        im = im.copy(); im.thumbnail((tw, H - 110), Image.NEAREST)
        x = 16 + i * (tw + 16) + (tw - im.width) // 2
        canvas.paste(im, (x, 48))
        d.text((16 + i * (tw + 16), 52 + im.height), cap, fill=(200, 200, 200), font=font(15))
    d.text((16, H - 30), note, fill=(150, 220, 150), font=font(15))
    frames.append((canvas, ms))
    if LIVE:
        cv2.imshow("agent (demo)", cv2.cvtColor(np.asarray(canvas), cv2.COLOR_RGB2BGR)); cv2.waitKey(max(1, ms))


def mark(img, xy, col=(255, 0, 255)):
    im = Image.fromarray(img) if isinstance(img, np.ndarray) else img.copy()
    if xy:
        ImageDraw.Draw(im).ellipse([xy[0] - 7, xy[1] - 7, xy[0] + 7, xy[1] + 7], outline=col, width=3)
    return im


eyes = Eyes()
dist = lambda a, b: (a - b).pow(2).mean((1, 2))

# ======================================================================== 1. tic-tac-toe: reach a goal board
ck = torch.load(DATA / "level0" / "imaginer.pt")
ttt = Imaginer().to(dev); ttt.load_state_dict(ck["model"]); ttt.eval()
held = ck["held"]
states = json.loads((DATA / "level0" / "states.json").read_text())
feats0 = store.load(DATA / "level0" / "feats.npy", mmap=False)
tr0 = np.load(DATA / "level0" / "transitions.npz")
ch = [i for i in range(5000) if tr0["src"][i] != tr0["dst"][i]]
TAU = 0.3 * float(np.quantile([((feats0[tr0["src"][i]].astype(np.float32) - feats0[tr0["dst"][i]]) ** 2).mean() for i in ch], 0.01))
CAND = (CENTERS * 224).tolist()


@torch.no_grad()
def ttt_plan(z0, zg, depth=3, B=8):
    Z, first, best = z0, None, (float("inf"), None)
    for h in range(depth):
        with torch.autocast("cuda", torch.bfloat16):
            P = ttt(Z.repeat_interleave(256, 0), CENTERS.repeat(len(Z), 1))[0].float()
        f = torch.arange(256, device=dev) if first is None else first.repeat_interleave(256)
        d = torch.where(dist(P, Z.repeat_interleave(256, 0)) > TAU, dist(P, zg), torch.inf)
        i = int(d.argmin())
        if d[i] < best[0] - TAU:
            best = (float(d[i]), int(f[i]))
        keep = []
        for j in d.argsort()[:256].tolist():
            if torch.isfinite(d[j]) and all(dist(P[j:j + 1], P[k:k + 1]).item() > TAU for k in keep):
                keep.append(j)
            if len(keep) == B:
                break
        if not keep:
            break
        Z, first = P[keep], f[keep]
    return best[1] if best[1] is not None else random.randrange(256)


def ttt_goal(s, d_):
    dd, q, seen = {ttt_key(s): 0}, deque([s]), {ttt_key(s): s}
    while q:
        u = q.popleft()
        for m in range(10):
            v = ttt_step(u, m)
            if ttt_key(v) not in dd:
                dd[ttt_key(v)] = dd[ttt_key(u)] + 1; seen[ttt_key(v)] = v; q.append(v)
    far = [k for k, v in dd.items() if v == d_]
    return seen[random.choice(far)] if far else None


with sync_playwright() as p:
    page = p.chromium.launch().new_page(viewport={"width": 224, "height": 224})
    page.goto((ROOT / "tictactoe.html").as_uri())
    shot = lambda: np.asarray(Image.open(io.BytesIO(page.screenshot())).convert("RGB"))
    for ep in range(3):
        while True:
            s0 = {k: states[random.choice([i for i in range(len(states)) if held[i]])][k] for k in ("board", "turn", "winner")}
            g = ttt_goal(s0, 3)
            if g:
                break
        page.evaluate("st => { s = st; render(); }", g); og = shot(); zg = eyes(og[None])
        page.evaluate("st => { s = st; render(); }", s0)
        title = f"1/3  Tic-tac-toe: reach the goal board by imagination  (game {ep + 1}/3)"
        for k in range(8):
            o = shot(); z = eyes(o[None])
            if dist(z, zg).item() < TAU:
                ok = ttt_key(page.evaluate("window.__state")) == ttt_key(g)
                show(title, [(Image.fromarray(o), "now"), (Image.fromarray(og), "goal")],
                     ("REACHED the goal" if ok else "stopped (wrong board)") + f" in {k} clicks", 1800)
                break
            c = ttt_plan(z, zg)
            show(title, [(mark(o, CAND[c]), f"now  -> click {k + 1} (magenta)"), (Image.fromarray(og), "goal")],
                 "imagining 256 clicks x 3 steps ahead, picking the chain that ends closest to the goal", 900)
            page.mouse.click(*CAND[c])
del ttt; torch.cuda.empty_cache()

# ======================================================================== 2. Snake: self-play-trained player
SP = torch.load(DATA / "snake_selfplay" / "selfplay_models.pt")
imag = MultiImaginer(K=64, keys=6).to(dev); imag.load_state_dict(SP["imag"]); imag.eval()


class Value(nn.Module):  # same as snake_selfplay.py
    def __init__(self, w=128):
        super().__init__()
        self.inp, self.pos = nn.Linear(384, w), nn.Parameter(torch.zeros(1, 256, w))
        self.tf = nn.TransformerEncoder(nn.TransformerEncoderLayer(w, 4, 2 * w, 0.0, batch_first=True, norm_first=True), 2)
        self.out = nn.Linear(w, 1)

    def forward(self, z):
        return self.out(self.tf(self.inp(z) + self.pos).mean(1)).squeeze(-1)


eat_net, dead_net, value = nn.Linear(768, 1).to(dev), nn.Linear(768, 1).to(dev), Value().to(dev)
eat_net.load_state_dict(SP["eat"]); dead_net.load_state_dict(SP["dead"]); value.load_state_dict(SP["value"]); value.eval()
status = lambda z: z.view(-1, 16, 16, 384)[:, :2].mean((1, 2))
eat_feat = lambda z, z1: torch.cat([status(z1) - status(z), (z1 - z).abs().mean(1)], -1)
dead_feat = lambda z: torch.cat([status(z), z.mean(1)], -1)
SKEYS = ["ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight", None]
ARROW = ["Up", "Down", "Left", "Right", "(no key)"]


@torch.no_grad()
def snake_plan(z0, depth=2, gamma=0.95):
    Z, score, alive, first = z0, torch.zeros(1, device=dev), torch.ones(1, dtype=torch.bool, device=dev), None
    for h in range(depth):
        keys = torch.arange(5, device=dev).repeat(len(Z))
        with torch.autocast("cuda", torch.bfloat16):
            fut, _, prior = imag(Z.repeat_interleave(5, 0), k=keys)
        Z1 = fut.float()[torch.arange(len(keys)), prior.float().argmax(1)]
        eat = (eat_net(eat_feat(Z.repeat_interleave(5, 0), Z1)).squeeze(-1) > 0).float()
        die = dead_net(dead_feat(Z1)).squeeze(-1) > 0
        score, alive = score.repeat_interleave(5), alive.repeat_interleave(5)
        first = keys if first is None else first.repeat_interleave(5)
        score = score + alive * (gamma ** h) * (eat - die.float()); alive = alive & ~die; Z = Z1
    score = score + alive * (gamma ** depth) * value(Z)
    return int(first[score.argmax()])


with sync_playwright() as p:
    page = p.chromium.launch().new_page(viewport={"width": 224, "height": 224})
    page.goto((ROOT / "snake.html").as_uri())
    best = None
    for attempt in range(3):  # play up to 3 games, keep the frames of the longest one for the GIF
        page.evaluate("reset()")
        run = []
        for t in range(220):
            s = page.evaluate("window.__state")
            o = np.asarray(Image.open(io.BytesIO(page.screenshot())).convert("RGB"))
            if s["dead"]:
                run.append((o, None, s, t)); break
            t0 = time.perf_counter(); k = snake_plan(eyes(o[None])); torch.cuda.synchronize()
            run.append((o, k, s, (time.perf_counter() - t0) * 1000))
            if LIVE:
                show("2/3  Snake: playing by imagination (self-play trained)", [(Image.fromarray(o), f"score {s['score']}")],
                     f"step {t}: imagined 25 futures (2 keys ahead) in {run[-1][3]:.0f} ms -> {ARROW[k]}", 90)
                frames.pop()
            if SKEYS[k]:
                page.keyboard.press(SKEYS[k])
            page.evaluate("step()")
        if best is None or len(run) > len(best):
            best = run
    for i, (o, k, s, ms) in enumerate(best):
        if k is None:
            show("2/3  Snake: playing by imagination (self-play trained)", [(Image.fromarray(o), f"score {s['score']}")],
                 f"game over after {i} steps, score {s['score']}  (longest of 3 games shown)", 1500)
        elif i % 2 == 0:
            show("2/3  Snake: playing by imagination (self-play trained; longest of 3 games)", [(Image.fromarray(o), f"score {s['score']}")],
                 f"step {i}: imagined 25 futures (2 keys ahead) in {ms:.0f} ms -> {ARROW[k]}", 110)
del imag; torch.cuda.empty_cache()

# ======================================================================== 3. Tetris: imagination vs reality
sys.path.insert(0, str(ROOT / "tetris"))
import data as T
tet = Imaginer(keys=8, tokens=576, history=True).to(dev)
tet.load_state_dict(torch.load(DATA / "tetris" / "imaginer_t0_hist.pt")); tet.eval()
reader = nn.Linear(384, 1).to(dev)
ropt = torch.optim.Adam(reader.parameters(), 1e-3)
read = lambda zz: reader(zz[:, T.PIDX]).squeeze(-1)
rs, rng = T.chunk_starts(T.train, 4), np.random.default_rng(0)
for _ in range(800):  # board reader on real training screens (~1 min)
    b, z, _, _ = T.chunk_batch(rs, rng)
    loss = F.binary_cross_entropy_with_logits(read(torch.from_numpy(z).to(dev).float()), torch.stack([T.board(t) for t in b]).to(dev))
    ropt.zero_grad(); loss.backward(); ropt.step()
first_of = {w: int(np.where(T.game // 1000 == w)[0][0]) for w in np.unique(T.game // 1000)}
png = lambda t: img(DATA / "tetris" / f"w{T.game[t] // 1000}" / "img" / f"{t - first_of[T.game[t] // 1000]}.png").convert("RGB")


def board_img(cells, col):
    im = Image.new("RGB", (100, 200), (250, 245, 230)); dr = ImageDraw.Draw(im)
    for j, on in enumerate(cells):
        r_, c_ = divmod(j, 10)
        dr.rectangle([c_ * 10, r_ * 10, c_ * 10 + 9, r_ * 10 + 9], fill=col if on else None, outline=(205, 205, 205))
    return im


starts = T.chunk_starts(T.test, 48)
moving = [t for t in starts[::50] if len({T.kinds[np.searchsorted(T.trans, u)] for u in range(t, t + 48)}) >= 5]
t0 = int(moving[0] if moving else starts[0])
with torch.no_grad():
    for i, t in enumerate(range(t0, t0 + 48)):
        z, z1, zp = T.get([t]), T.get([t + 1]), T.get(T.prev_of([t]))
        with torch.autocast("cuda", torch.bfloat16):
            pz, _ = tet(z, k=torch.tensor([int(T.acts[t])], device=dev), z_prev=zp)
        imagined = (read(pz.float())[0] > 0).cpu().tolist()
        real = T.board(t + 1).bool().tolist()
        show("3/3  Tetris: IMAGINED vs REAL next board (not playable yet)",
             [(png(t), f"real screen, key: {T.KEYS[int(T.acts[t])]}"), (board_img(imagined, (70, 90, 255)), "imagined next board"),
              (board_img(real, (40, 40, 40)), "real next board")],
             f"{T.kinds[np.searchsorted(T.trans, t)]}  |  held-out game, model never saw it", 450)

if LIVE:
    cv2.destroyAllWindows()
out = R / "demo.gif"
frames[0][0].save(out, save_all=True, append_images=[f for f, _ in frames[1:]], duration=[ms for _, ms in frames], loop=0, optimize=True)
print(f"saved {out}  ({len(frames)} frames, {sum(ms for _, ms in frames) / 1000:.0f} s, {out.stat().st_size / 1e6:.1f} MB)")
