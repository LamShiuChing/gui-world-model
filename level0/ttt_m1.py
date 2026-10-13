"""Mastery M1: see EVERY tic-tac-toe screen, and check the screen latents carry the board.

1. All 5,478 legal boards (python rules enumerate them - harness only); each is rendered on the real page via set_state.
2. Own experience on all of them: from every board, click the 9 cell centres, Reset and 2 off-board points, and record
   where the page goes (66k clicks; the page decides, not the rules) -> the exact click graph M2 checks against.
3. EYES (frozen DINOv2-S) on every screen.
4. Probes (measuring tools, trained on half of the boards, tested on the other half, then swapped):
   board readout (9 cells x empty/X/O) -> PASS = 5,478 / 5,478 boards fully right;
   outcome readout on finished boards (X wins / O wins / draw) -> PASS = all 958 right.

Out: data/ttt_all/ states.json, img/*.png, clicks.npy [src, x, y, dst], feats.npy (fp16)
usage: python level0/ttt_m1.py [WORKERS=8]
"""
import io, json, sys
from collections import deque
from multiprocessing import Process
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path(__file__).parent
D = ROOT.parent / "data" / "ttt_all"
R = ROOT.parent / "results"
CELL = [(52 + 60 * (j % 3), 54 + 60 * (j // 3)) for j in range(9)]
CLICKS = CELL + [(112, 214), (8, 8), (216, 120)]  # 9 cells, Reset, 2 off-board points
sys.path.insert(0, str(ROOT))
from ttt_rules import step, EMPTY
from collect import key


def all_boards():
    seen, q = {key(EMPTY): EMPTY}, deque([EMPTY])
    while q:
        u = q.popleft()
        for m in range(9):
            v = step(u, m)
            if key(v) not in seen:
                seen[key(v)] = v; q.append(v)
    return list(seen.values())


def work(k, nw, boards):
    from playwright.sync_api import sync_playwright
    ids = {key(b): i for i, b in enumerate(boards)}
    clicks = []
    with sync_playwright() as p:
        page = p.chromium.launch(args=["--disable-dev-shm-usage"]).new_page(viewport={"width": 224, "height": 224})
        page.goto((ROOT / "tictactoe.html").as_uri())
        put = lambda b: page.evaluate("st => { s = st; render(); }", b)
        for i in range(k, len(boards), nw):
            put(boards[i])
            page.screenshot(path=D / "img" / f"{i}.png")
            for x, y in CLICKS:
                put(boards[i])
                page.mouse.click(x, y)
                clicks.append([i, x, y, ids[key(page.evaluate("window.__state"))]])
    np.save(D / f"clicks_w{k}.npy", np.array(clicks, np.int32))


if __name__ == "__main__":
    nw = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    (D / "img").mkdir(parents=True, exist_ok=True)
    boards = all_boards()
    assert len(boards) == 5478, len(boards)
    (D / "states.json").write_text(json.dumps(boards))
    ps = [Process(target=work, args=(k, nw, boards)) for k in range(nw)]
    [p.start() for p in ps]; [p.join() for p in ps]
    clicks = np.concatenate([np.load(D / f"clicks_w{k}.npy") for k in range(nw)])
    np.save(D / "clicks.npy", clicks)
    [(D / f"clicks_w{k}.npy").unlink() for k in range(nw)]
    print(f"{len(boards)} boards rendered, {len(clicks)} clicks recorded", flush=True)

    import torch, torch.nn as nn, torch.nn.functional as F
    from model import Eyes, dev
    eyes = Eyes()
    load = lambda i: np.asarray(Image.open(D / "img" / f"{i}.png").convert("RGB"))
    feats = np.concatenate([eyes(np.stack([load(i) for i in range(s, min(s + 256, len(boards)))])).half().cpu().numpy()
                            for s in range(0, len(boards), 256)])
    np.save(D / "feats.npy", feats)
    del eyes; torch.cuda.empty_cache()

    # ---------------------------------------------------------------- probes (two-fold: train on one half, test on the other)
    Z = torch.from_numpy(feats).to(dev).float()
    cells = torch.tensor([[{None: 0, "X": 1, "O": 2}[c] for c in b["board"]] for b in boards], device=dev)
    fin = [i for i, b in enumerate(boards) if b["winner"]]
    outc = torch.tensor([{"X": 0, "O": 1, "draw": 2}[b["winner"]] if b["winner"] else -1 for b in boards], device=dev)
    half = torch.from_numpy(np.random.default_rng(0).random(len(boards)) < 0.5).to(dev)

    def probe(n_out, x_of, y, train, test, steps=3000):
        net = nn.Sequential(nn.LayerNorm(384), nn.Linear(384, n_out)).to(dev)
        opt = torch.optim.Adam(net.parameters(), 1e-3)
        tr = torch.where(train)[0]
        for _ in range(steps):
            b = tr[torch.randint(len(tr), (256,), device=dev)]
            loss = F.cross_entropy(net(x_of(b)).flatten(0, -2), y[b].flatten())
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            te = torch.where(test)[0]
            return net(x_of(te)).argmax(-1), te

    # board: each cell read from its own patch tokens (mean of the 4x4 patches inside the cell) -> 3 classes
    cell_patches = [[(54 + 60 * (j // 3) - 30 + dy * 14 + 7) // 14 * 16 + (52 + 60 * (j % 3) - 30 + dx * 14 + 7) // 14
                     for dy in range(4) for dx in range(4)] for j in range(9)]
    CP = torch.tensor(cell_patches, device=dev)  # 9 x 16
    cell_in = lambda b: Z[b][:, CP].mean(2)  # (n, 9, 384)
    pooled = lambda b: Z[b].mean(1)  # (n, 384) for the outcome probe (status line + board)
    ok_board, ok_out = torch.zeros(len(boards), dtype=torch.bool, device=dev), {}
    isfin = outc >= 0
    for tr_half in (half, ~half):
        pred, te = probe(3, cell_in, cells, tr_half, ~tr_half)
        ok_board[te] = (pred == cells[te]).all(1)
        pred, te = probe(3, pooled, outc.clamp(min=0), tr_half & isfin, ~tr_half & isfin)
        ok_out.update(dict(zip(te.tolist(), (pred == outc[te]).tolist())))
    nb, no = int(ok_board.sum()), sum(ok_out.values())
    bad = [boards[i] for i in torch.where(~ok_board)[0].tolist()][:5]
    res = {"boards": len(boards), "clicks": len(clicks), "board_readout_right": nb, "finished_boards": len(fin),
           "outcome_readout_right": no, "examples_wrong_board": bad}
    (R / "ttt_m1_metrics.json").write_text(json.dumps(res, indent=2))
    print(f"\nM1 board readout (held-out half each time): {nb} / {len(boards)} boards fully right")
    print(f"M1 outcome readout on finished boards:       {no} / {len(fin)}")
    print(f"M1: {'PASS' if nb == len(boards) and no == len(fin) else 'FAIL'}  (needs {len(boards)}/{len(boards)} and {len(fin)}/{len(fin)})")
