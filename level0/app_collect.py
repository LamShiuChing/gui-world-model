"""Shared collector for small deterministic apps (A2 file manager, A3 spreadsheet; A1 settings predates it).

The page must expose window.__state (JSON) and window.__set(state). Each unique state is screenshotted once; every
action is logged as [src, kind, x1, y1, x2, y2, dst]. Action kinds: 0 = drag (press, move, release), 1 = click,
2 = double-click, 3 = type text (x1 = index into TEXTS; the app decides where it goes). Determinism check: 1% of
revisits must match the stored PNG, exactly or within a tiny anti-aliasing difference (<= 50 faint pixels).
"""
import io, json, random, time
from pathlib import Path
import numpy as np
from PIL import Image
from playwright.sync_api import sync_playwright

TEXTS = ["5", "12", "abc", "Total", "=A1+B1", "0"]  # what kind-3 actions may type (then Enter)


def do(page, a):
    k, x1, y1, x2, y2 = a
    if k == 0:
        page.mouse.move(x1, y1); page.mouse.down(); page.mouse.move(x2, y2, steps=6); page.mouse.up()
    elif k == 1:
        page.mouse.click(x1, y1)
    elif k == 2:
        page.mouse.dblclick(x1, y1)
    else:
        page.keyboard.type(TEXTS[int(x1)]); page.keyboard.press("Enter")


def collect(html, out, steps, sample, key, seed=0):
    """sample(page, state) -> action [kind, x1, y1, x2, y2]; key(state) -> unique string."""
    random.seed(seed)
    out = Path(out); (out / "img").mkdir(parents=True, exist_ok=True)
    ids, states, trans, checked, tiny = {}, [], [], 0, 0
    with sync_playwright() as p:
        page = p.chromium.launch().new_page(viewport={"width": 224, "height": 224})
        page.goto(Path(html).resolve().as_uri())

        def observe():
            nonlocal checked, tiny
            s = page.evaluate("window.__state")
            k = key(s)
            if k not in ids:
                ids[k] = len(states)
                states.append({"key": k, **s})
                page.screenshot(path=out / "img" / f"{ids[k]}.png")
            elif random.random() < 0.01:
                shot = page.screenshot()
                if shot != (out / "img" / f"{ids[k]}.png").read_bytes():
                    a = np.asarray(Image.open(io.BytesIO(shot)).convert("RGB")).astype(int)
                    b = np.asarray(Image.open(out / "img" / f"{ids[k]}.png").convert("RGB")).astype(int)
                    d = np.abs(a - b).max(-1)
                    if (d > 0).sum() > 50 or d.max() > 96:
                        (out / "mismatch_now.png").write_bytes(shot)
                        (out / "mismatch_last_actions.json").write_text(json.dumps(trans[-5:]))
                        raise AssertionError(f"non-deterministic render: {k} (see mismatch_now.png vs img/{ids[k]}.png)")
                    tiny += 1
                checked += 1
            return ids[k]

        cur, t0 = observe(), time.time()
        for step in range(steps):
            a = [float(v) for v in sample(page, states[cur])]
            a[0] = int(a[0])
            do(page, a)
            if a[0] in (1, 2):
                a[3], a[4] = a[1], a[2]
            nxt = observe()
            trans.append([cur, *a, nxt])
            cur = nxt
            if step % 5000 == 0:
                print(f"step {step:>6}  states {len(states):>5}  {step / (time.time() - t0 + 1e-9):.0f} steps/s", flush=True)
    (out / "states.json").write_text(json.dumps(states))
    (out / "transitions.json").write_text(json.dumps(trans))
    noop = sum(t[0] == t[-1] for t in trans) / len(trans)
    print(f"done: {steps} actions, {len(states)} unique states, {noop:.1%} no-op, {checked} determinism checks passed "
          f"({tiny} with a tiny anti-aliasing difference)")
