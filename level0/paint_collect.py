"""Random-stroke collector for the Paint app (Level P1).

Action = [kind, x1, y1, x2, y2] in page pixels. kind 0 = drag stroke (press at x1,y1, move to x2,y2, release),
kind 1 = click at (x1,y1) (palette swatch, Clear button, or anywhere). Frames are a sequence like Snake:
transition t = (frame t, action t, frame t+1). Determinism check at the end: replay the first 50 actions on a fresh
page and require a pixel-identical screenshot.

usage: python level0/paint_collect.py WORKER_ID STEPS     (run several workers in parallel)
"""
import json, math, random, sys, time
from pathlib import Path
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).parent
WID = int(sys.argv[1]) if len(sys.argv) > 1 else 0
STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 10_000
OUT = ROOT.parent / "data" / "paint" / f"w{WID}"
random.seed(WID)
SWATCH = [(18, 14), (52, 14), (86, 14), (120, 14)]  # palette centres
CLEAR = (183, 14)


def sample():  # dumb random bot: mostly strokes inside the canvas, some colour picks, rare Clear
    r = random.random()
    if r < 0.75:
        x1, y1 = random.uniform(10, 214), random.uniform(32, 216)
        ang, ln = random.uniform(0, 2 * math.pi), random.uniform(8, 120)
        return [0, x1, y1, min(max(x1 + ln * math.cos(ang), 0), 223), min(max(y1 + ln * math.sin(ang), 0), 223)]
    if r < 0.93:
        x, y = random.choice(SWATCH)
        return [1, x + random.uniform(-7, 7), y + random.uniform(-7, 7), 0, 0]
    if r < 0.97:
        return [1, CLEAR[0] + random.uniform(-25, 25), CLEAR[1] + random.uniform(-6, 6), 0, 0]
    return [0, *(random.uniform(0, 223) for _ in range(4))]  # stroke anywhere (may start off-canvas: does nothing)


def do(page, a):
    if a[0] == 0:
        page.mouse.move(a[1], a[2]); page.mouse.down()
        page.mouse.move(a[3], a[4], steps=8); page.mouse.up()
    else:
        page.mouse.click(a[1], a[2])


def main():
    (OUT / "img").mkdir(parents=True, exist_ok=True)
    acts, states = [], []
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 224, "height": 224})
        page.goto((ROOT / "paint.html").as_uri())
        states.append(page.evaluate("window.__state()"))
        page.screenshot(path=OUT / "img" / "0.png")
        t0 = time.time()
        for step in range(STEPS):
            a = sample()
            do(page, a)
            acts.append(a)
            states.append(page.evaluate("window.__state()"))
            page.screenshot(path=OUT / "img" / f"{step + 1}.png")
            if step % 2000 == 0:
                print(f"w{WID} step {step:>6}  {step / (time.time() - t0 + 1e-9):.1f} steps/s", flush=True)
        fresh = browser.new_page(viewport={"width": 224, "height": 224})  # determinism check
        fresh.goto((ROOT / "paint.html").as_uri())
        for a in acts[:50]:
            do(fresh, a)
        assert fresh.screenshot() == (OUT / "img" / "50.png").read_bytes(), "replay of 50 actions is not pixel-identical"
    (OUT / "actions.json").write_text(json.dumps(acts))
    (OUT / "states.json").write_text(json.dumps(states))
    kinds = [a[0] for a in acts]
    print(f"w{WID} done: {STEPS} steps ({kinds.count(0)} strokes, {kinds.count(1)} clicks); determinism check passed")


if __name__ == "__main__":
    main()
