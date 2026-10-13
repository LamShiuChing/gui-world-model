"""TinyApps collector: random clicks on many generated apps (tinyapp.html#SEED). Stores STATE + ACTION only, no
screenshots (tinyapps_encode.py re-renders each unique state once to make its features) -> a few KB per app.

85% of clicks land on a random clickable element (jittered inside it), 15% anywhere. Output per worker:
data/tinyapps/apps_wK.json = [{seed, spec, states: [state...], trans: [[src, x, y, dst], ...]}, ...]

usage: python level0/tinyapps_collect.py FIRST_SEED LAST_SEED [STEPS=200] [WORKERS=6]
"""
import json, random, sys, time
from multiprocessing import Process
from pathlib import Path
from playwright.sync_api import sync_playwright

HTML = (Path(__file__).parent / "tinyapp.html").resolve().as_uri()
OUT = Path(__file__).parent.parent / "data" / "tinyapps"
HITS = "[...document.querySelectorAll('[data-hit]')].map(e => { const r = e.getBoundingClientRect(); return [r.left, r.top, r.width, r.height]; })"


def work(k, seeds, steps):
    random.seed(k)
    apps, t0 = [], time.time()
    with sync_playwright() as p:
        page = p.chromium.launch().new_page(viewport={"width": 224, "height": 224})
        for n, seed in enumerate(seeds):
            page.goto("about:blank"); page.goto(f"{HTML}#{seed}")
            ids, states, trans = {}, [], []

            def observe():
                s = page.evaluate("window.__state")
                key = json.dumps(s, sort_keys=True)
                if key not in ids:
                    ids[key] = len(states); states.append(s)
                return ids[key]

            cur = observe()
            for _ in range(steps):
                boxes = page.evaluate(HITS)
                if boxes and random.random() < 0.85:
                    x, y, w, h = random.choice(boxes)
                    x, y = x + w * random.uniform(0.15, 0.85), y + h * random.uniform(0.15, 0.85)
                else:
                    x, y = random.uniform(0, 223), random.uniform(0, 223)
                page.mouse.click(x, y)
                nxt = observe()
                trans.append([cur, round(x, 1), round(y, 1), nxt]); cur = nxt
            apps.append({"seed": seed, "spec": page.evaluate("window.__spec"), "states": states, "trans": trans})
            if n % 50 == 0:
                print(f"w{k}: {n}/{len(seeds)} apps, {(n + 1) * steps / (time.time() - t0):.0f} actions/s", flush=True)
    (OUT / f"apps_w{k}.json").write_text(json.dumps(apps))


if __name__ == "__main__":
    a, b = int(sys.argv[1]), int(sys.argv[2])
    steps = int(sys.argv[3]) if len(sys.argv) > 3 else 200
    nw = int(sys.argv[4]) if len(sys.argv) > 4 else 6
    OUT.mkdir(parents=True, exist_ok=True)
    seeds = list(range(a, b + 1))
    ps = [Process(target=work, args=(k, seeds[k::nw], steps)) for k in range(nw)]
    [p.start() for p in ps]; [p.join() for p in ps]
    apps = sum((json.loads((OUT / f"apps_w{k}.json").read_text()) for k in range(nw)), [])
    print(f"done: {len(apps)} apps, {sum(len(x['states']) for x in apps)} unique states, "
          f"{sum(len(x['trans']) for x in apps)} actions, {sum(x['trans'][i][0] == x['trans'][i][3] for x in apps for i in range(len(x['trans']))) / sum(len(x['trans']) for x in apps):.1%} no-op")
