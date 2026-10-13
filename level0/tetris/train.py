"""Tetris T0: IMAGINER predicts the next Tetris screen (336px -> 576 tokens) from screen + key. Held-out games only.

New vs Snake: gravity at 21 speeds (levels 1-21 incl. 20G), 8 keys (none, Left, Right, Down, Z, X, Space, C), rotations,
holds, line clears, a ticking timer, and mild randomness (newest preview piece). Deterministic IMAGINER first.
Scores per step type + a BOARD READER (linear probe trained on real screens: which of the 200 cells are filled, falling
piece included) applied to the IMAGINED next screen: are the cells that really changed predicted right?

usage: python level0/tetris/train.py [STEPS=8000] [hist]
"""
import json, os, sys
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image, ImageDraw
sys.path.insert(0, str(Path(__file__).parent.parent))
from model import Imaginer, dev
from pieces import visible_cells
import store
from webp import img

D = Path(__file__).parent.parent.parent / "data" / "tetris"
R = Path(__file__).parent.parent.parent / "results"
STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
HIST = len(sys.argv) > 2 and sys.argv[2] == "hist"  # v2: also see the previous frame (hidden gravity / lock timers)
TAG = "_hist" if HIST else ""
FRAC = float(os.environ.get("FRAC", 1))  # data-efficiency curve: the IMAGINER trains on a random FRAC of training games
TAG += "" if FRAC == 1 else f"_f{FRAC:g}"
SIZE, G = 336, 24
KEYS = ["none", "Left", "Right", "Down", "Z (rotate)", "X (rotate)", "Space (drop)", "C (hold)"]
torch.manual_seed(0)
rng = np.random.default_rng(0)
torch.backends.cuda.matmul.allow_tf32 = True

states = sum((json.loads((w / "states.json").read_text()) for w in sorted(D.glob("w[0-9]"))), [])
fr = np.load(D / "frames.npz")
acts, game = fr["action"], fr["game"]
feats = store.load(D / f"feats_{SIZE}.npy")  # 80k x 576 x 384 fp16 on disk (NVMe)
masks = torch.from_numpy(np.load(D / f"masks_{SIZE}.npy"))
N = len(states)
assert len(acts) == N == len(feats), (len(acts), N, len(feats))


def kind(t):
    s, s1, a = states[t], states[t + 1], acts[t]
    if s1["over"] and not s["over"]:
        return "game over"
    if s1["lines"] > s["lines"]:
        return "line clear"
    if a == 7 and s1["hold"] != s["hold"]:
        return "hold"
    if s1["spawns"] > s["spawns"]:
        return "lock + new piece"
    if s1["o"] != s["o"]:
        return "rotate"
    if s1["x"] != s["x"]:
        return "move sideways"
    if s1["y"] != s["y"]:
        return "falls (gravity / soft drop)"
    return "piece still (timer only)"


trans = np.where(acts >= 0)[0]
trans = trans[[not states[t]["over"] for t in trans]]
kinds = np.array([kind(t) for t in trans])
held = rng.random(game.max() + 1) < 0.2
test, train = trans[held[game[trans]]], trans[~held[game[trans]]]
g_tr = np.unique(game[train])
train_im = train[np.isin(game[train], np.random.default_rng(5).choice(g_tr, max(1, round(FRAC * len(g_tr))), replace=False))]
print(f"IMAGINER trains on {len(train_im)} transitions ({FRAC:.0%} of {len(g_tr)} training games)", flush=True)
print(f"{len(trans)} transitions | train {len(train)} | test {len(test)} (held-out games)", flush=True)
for k_, n_ in zip(*np.unique(kinds, return_counts=True)):
    print(f"   {k_:<30} {n_:>6}")


def get(idx):  # sorted reads from the disk memmap
    idx = np.asarray(idx)
    o = np.argsort(idx)
    out = np.empty((len(idx), G * G, 384), np.float16)
    out[o] = feats[idx[o]]
    return torch.from_numpy(out).to(dev).float()


model = Imaginer(keys=8, tokens=G * G, history=HIST).to(dev)
prev_ok = np.r_[False, acts[:-1] >= 0]  # frame t has a previous frame in the same game
def prev(b):  # previous screen (or the same screen at the start of a game: "no motion")
    if not HIST:
        return None
    b = np.asarray(b)
    return get(np.where(prev_ok[b], b - 1, b))


print(f"trainable params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M", flush=True)
opt = torch.optim.AdamW(model.parameters(), 3e-4, weight_decay=0.01)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-4, total_steps=STEPS, pct_start=0.05)
rho = masks[train_im].float().mean().item()
lam = min((1 - rho) / rho, 50.0)
print(f"changed-patch fraction rho={rho:.3f} -> change weight {lam:.1f}", flush=True)
for step in range(STEPS):
    b = rng.choice(train_im, 64)
    z, z1, m, k = get(b), get(b + 1), masks[b].float().to(dev), torch.from_numpy(acts[b]).long().to(dev)
    with torch.autocast("cuda", torch.bfloat16):
        pred, logit = model(z, k=k, z_prev=prev(b))
    w = 1 + lam * m
    loss_z = (w * (pred.float() - z1).pow(2).mean(-1)).sum() / w.sum()
    loss_m = F.binary_cross_entropy_with_logits(logit.float(), m)
    opt.zero_grad(set_to_none=True); (loss_z + loss_m).backward(); opt.step(); sched.step()
    if step % 1000 == 0 or step == STEPS - 1:
        print(f"step {step:>5}  latent {loss_z.item():.4f}  mask {loss_m.item():.4f}", flush=True)
torch.save(model.state_dict(), D / f"imaginer_t0{TAG}.pt")
model.eval()

# ---------------------------------------------------------------- BOARD READER on real screens (train games)
cells = [(225 + 29 * c + 14.5, 25 + 29 * r + 14.5) for r in range(20) for c in range(10)]
PIDX = torch.tensor([int(y * SIZE / 620 // 14) * G + int(x * SIZE / 620 // 14) for x, y in cells], device=dev)
board = lambda t: torch.tensor(sum(visible_cells(states[t]), []), dtype=torch.float32)
reader = nn.Linear(384, 1).to(dev)
ropt = torch.optim.Adam(reader.parameters(), 1e-3)
read = lambda zz: reader(zz[:, PIDX]).squeeze(-1)  # (N,200) logits
for _ in range(2000):
    b = rng.choice(train, 128)
    loss = F.binary_cross_entropy_with_logits(read(get(b)), torch.stack([board(t) for t in b]).to(dev))
    ropt.zero_grad(); loss.backward(); ropt.step()

# ---------------------------------------------------------------- evaluation on held-out games
ev = np.sort(rng.choice(test, min(6000, len(test)), replace=False))
ek = kinds[np.searchsorted(trans, ev)]
k_true = torch.from_numpy(acts[ev]).long().to(dev)
k_wrong = (k_true + torch.randint(1, 8, k_true.shape, device=dev)) % 8  # always a DIFFERENT key
err, err_w, err_id, mp, cell_ok, cell_ok_w, cell_n, real_ok = [], [], [], [], [], [], [], []
with torch.no_grad():
    for i in range(0, len(ev), 64):
        b = ev[i:i + 64]
        z, z1, zp = get(b), get(b + 1), prev(b)
        with torch.autocast("cuda", torch.bfloat16):
            p, lg = model(z, k=k_true[i:i + 64], z_prev=zp)
            pw, _ = model(z, k=k_wrong[i:i + 64], z_prev=zp)
        p, pw = p.float(), pw.float()
        err.append((p - z1).pow(2).mean((1, 2)).cpu()); err_w.append((pw - z1).pow(2).mean((1, 2)).cpu())
        err_id.append((z - z1).pow(2).mean((1, 2)).cpu()); mp.append((lg.float() > 0).cpu())
        b0, b1 = torch.stack([board(t) for t in b]).to(dev), torch.stack([board(t + 1) for t in b]).to(dev)
        ch = b0 != b1  # cells that really changed
        cell_ok.append(((read(p) > 0) == (b1 > 0.5))[ch].float().sum().cpu())
        cell_ok_w.append(((read(pw) > 0) == (b1 > 0.5))[ch].float().sum().cpu())
        cell_n.append(ch.sum().cpu())
        real_ok.append(((read(z1) > 0) == (b1 > 0.5))[ch].float().sum().cpu())
err, err_w, err_id, mp = torch.cat(err), torch.cat(err_w), torch.cat(err_id), torch.cat(mp)
m_true = masks[ev].bool()

res = {"n_test": len(ev), "groups": {}}
print("\n=== Tetris T0 on held-out games: latent error vs 'nothing changes' (1.00), lower = better ===")
print(f"{'step type':<30} {'n':>5} {'model':>7} {'wrong key':>10} {'change-map F1':>14}")
for g_ in ["falls (gravity / soft drop)", "move sideways", "rotate", "hold", "lock + new piece", "line clear",
           "game over", "piece still (timer only)"]:
    sel = torch.from_numpy(ek == g_)
    if not sel.any():
        continue
    base = err_id[sel].sum()
    tp = (mp[sel] & m_true[sel]).sum()
    f1 = (2 * tp / (mp[sel].sum() + m_true[sel].sum()).clamp(min=1)).item()
    row = {"n": int(sel.sum()), "model": (err[sel].sum() / base).item(), "wrong_key": (err_w[sel].sum() / base).item(), "mask_F1": f1}
    res["groups"][g_] = row
    print(f"{g_:<30} {row['n']:>5} {row['model']:>7.3f} {row['wrong_key']:>10.3f} {f1:>14.3f}")
cn = sum(cell_n).item()
res["changed_cells"] = {"n": cn, "model": sum(cell_ok).item() / cn, "wrong_key": sum(cell_ok_w).item() / cn,
                        "reader_on_real_next_screen": sum(real_ok).item() / cn, "nothing_changes": 0.0}
c = res["changed_cells"]
print(f"\nBoard cells that REALLY changed (n={cn}): read correctly from the imagined screen")
print(f"  model {c['model']:.1%} | wrong key {c['wrong_key']:.1%} | 'nothing changes' 0% | ceiling (reader on the REAL next screen) {c['reader_on_real_next_screen']:.1%}")
mv = [res["groups"][g_] for g_ in ("falls (gravity / soft drop)", "move sideways", "rotate") if g_ in res["groups"]]
ok = all(r["model"] < 0.5 for r in mv) and all(r["wrong_key"] >= 2 * r["model"] for r in mv[1:]) and c["model"] >= 0.7 * c["reader_on_real_next_screen"]
print(f"\nKILL TEST: {'PASS' if ok else 'FAIL'}  (needs error < 0.5 on falls/moves/rotations, wrong key >= 2x worse on moves/rotations, "
      f"changed cells >= 70% of the reader's ceiling)")
(R / f"tetris_t0{TAG}_metrics.json").write_text(json.dumps(res, indent=2))

# ---------------------------------------------------------------- picture: before + key | true after | imagined board | true board
def draw_board(cells_on, col):
    im = Image.new("RGB", (120, 240), (250, 245, 230))
    d = ImageDraw.Draw(im)
    for j, on in enumerate(cells_on):
        r_, c_ = divmod(j, 10)
        d.rectangle([c_ * 12, r_ * 12, c_ * 12 + 11, r_ * 12 + 11], fill=col if on else None, outline=(200, 200, 200))
    return im


pick = [int(np.where(ek == g_)[0][0]) for g_ in ["move sideways", "rotate", "falls (gravity / soft drop)", "line clear", "hold", "lock + new piece"]
        if (ek == g_).any()]
sheet = Image.new("RGB", (310 + 310 + 130 + 130, len(pick) * 310), "white")
with torch.no_grad():
    for r_, i in enumerate(pick):
        t = int(ev[i])
        z = get([t])
        with torch.autocast("cuda", torch.bfloat16):
            p, _ = model(z, k=torch.tensor([acts[t]], device=dev), z_prev=prev([t]))
        imag = (read(p.float())[0] > 0).cpu().tolist()
        b4 = img(D / f"w{game[t] // 1000}" / "img" / f"{t - int(np.where(game // 1000 == game[t] // 1000)[0][0])}.png").convert("RGB").resize((300, 300))
        ImageDraw.Draw(b4).text((6, 285), f"{ek[i]} | key {KEYS[acts[t]]}", fill=(200, 0, 200))
        af = img(D / f"w{game[t] // 1000}" / "img" / f"{t + 1 - int(np.where(game // 1000 == game[t] // 1000)[0][0])}.png").convert("RGB").resize((300, 300))
        for c_, im in enumerate([b4, af]):
            sheet.paste(im, (c_ * 310, r_ * 310))
        sheet.paste(draw_board(imag, (60, 60, 255)), (620, r_ * 310 + 30))
        sheet.paste(draw_board(sum(visible_cells(states[t + 1]), []), (40, 40, 40)), (750, r_ * 310 + 30))
sheet.save(R / f"tetris_t0{TAG}_examples.png")
print(f"saved {R / f'tetris_t0{TAG}_examples.png'}  (cols: before + key | true after | board READ from the imagined screen | true board)")
