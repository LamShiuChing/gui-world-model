"""Random collector for the Settings app (Level A1). Like tic-tac-toe: the picture depends only on the state, so each
unique state is screenshotted once and every action is logged as (src state, action, dst state).

Action = [kind, x1, y1, x2, y2] (page px): kind 0 = drag (press, move, release), kind 1 = click (x2,y2 = x1,y1).
The bot aims at real controls most of the time (positions read from the page, with jitter), sometimes clicks/drags anywhere.
Determinism check: 1% of revisits must match the stored PNG (exactly, or within a tiny anti-aliasing difference on
rounded corners: <=50 faint pixels; Chromium repaints partially after clicks).

usage: python level0/settings_collect.py [STEPS=30000]
"""
import io, json, random, sys, time
import numpy as np
from PIL import Image
from pathlib import Path
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).parent
OUT = ROOT.parent / "data" / "settings"
STEPS = int(sys.argv[1]) if __name__ == "__main__" and len(sys.argv) > 1 else 30_000
random.seed(0)
key = lambda s: f"{int(s['dark'])}{int(s['wifi'])}{s['volume']:03d}{s['font']}{int(s['notif'])}{int(s['menu'])}"


def main():
    (OUT / "img").mkdir(parents=True, exist_ok=True)
    ids, states, trans, checked, tiny_aa = {}, [], [], 0, 0
    with sync_playwright() as p:
        page = p.chromium.launch().new_page(viewport={"width": 224, "height": 224})
        page.goto((ROOT / "settings.html").as_uri())
        box = lambda sel: page.evaluate(f"(() => {{ const r = document.querySelector('{sel}').getBoundingClientRect(); return [r.left, r.top, r.right, r.bottom]; }})()")
        targets = {n: box(sel) for n, sel in [("dark", "#dark"), ("wifi", "#wifi"), ("track", "#track"), ("font", "#font"),
                                              ("notif", "#notif"), ("reset", "#reset")]}
        page.evaluate("window.__set({...window.__state, menu: true})")
        items = [box(f"#menu div:nth-child({i})") for i in (1, 2, 3)]
        page.evaluate("window.__set({...window.__state, menu: false})")
        inside = lambda b: (random.uniform(b[0] + 1, b[2] - 1), random.uniform(b[1] + 1, b[3] - 1))

        def sample(s):
            r = random.random()
            if s["menu"] and r < 0.7:  # an open menu: usually pick an item
                return [1, *inside(random.choice(items)), 0, 0]
            if r < 0.55:
                n = random.choice(list(targets))
                b = targets[n]
                x, y = (random.uniform(b[0], b[2]), (b[1] + b[3]) / 2) if n == "track" else inside(b)
                return [1, x, y, 0, 0]
            if r < 0.80:  # drag along the volume slider, starting at the knob
                t = targets["track"]
                kx = t[0] + s["volume"] * 1.1
                return [0, kx + random.uniform(-4, 4), (t[1] + t[3]) / 2, random.uniform(t[0] - 10, t[2] + 10), (t[1] + t[3]) / 2 + random.uniform(-6, 6)]
            if r < 0.93:
                return [1, random.uniform(0, 223), random.uniform(0, 223), 0, 0]
            return [0, *(random.uniform(0, 223) for _ in range(4))]

        def observe():
            nonlocal checked, tiny_aa
            s = page.evaluate("window.__state")
            k = key(s)
            if k not in ids:
                ids[k] = len(states)
                states.append({"key": k, **s})
                page.screenshot(path=OUT / "img" / f"{ids[k]}.png")
            elif random.random() < 0.01:
                shot = page.screenshot()
                if shot != (OUT / "img" / f"{ids[k]}.png").read_bytes():
                    # Chromium anti-aliases rounded toggle corners slightly differently after partial repaints
                    # (seen: 14 faint pixels). Tolerate that; anything bigger is a real bug -> keep evidence, stop.
                    a = np.asarray(Image.open(io.BytesIO(shot)).convert("RGB")).astype(int)
                    b = np.asarray(Image.open(OUT / "img" / f"{ids[k]}.png").convert("RGB")).astype(int)
                    d = np.abs(a - b).max(-1)
                    if (d > 0).sum() > 50 or d.max() > 96:
                        (OUT / "mismatch_now.png").write_bytes(shot)
                        (OUT / "mismatch_last_actions.json").write_text(json.dumps(trans[-5:]))
                        raise AssertionError(f"non-deterministic render: {k} (see mismatch_now.png vs img/{ids[k]}.png)")
                    tiny_aa += 1
                checked += 1
            return ids[k]

        cur, t0 = observe(), time.time()
        for step in range(STEPS):
            a = sample(states[cur])
            if a[0] == 0:
                page.mouse.move(a[1], a[2]); page.mouse.down(); page.mouse.move(a[3], a[4], steps=6); page.mouse.up()
            else:
                page.mouse.click(a[1], a[2])
                a[3], a[4] = a[1], a[2]
            nxt = observe()
            trans.append([cur, *a, nxt])
            cur = nxt
            if step % 5000 == 0:
                print(f"step {step:>6}  states {len(states):>4}  {step / (time.time() - t0 + 1e-9):.0f} steps/s", flush=True)
    (OUT / "states.json").write_text(json.dumps(states))
    (OUT / "transitions.json").write_text(json.dumps(trans))
    noop = sum(t[0] == t[-1] for t in trans) / len(trans)
    print(f"done: {STEPS} actions, {len(states)} unique states, {noop:.1%} no-op, {checked} determinism checks passed ({tiny_aa} with a tiny anti-aliasing difference, <=50 faint pixels)")


if __name__ == "__main__":
    main()
