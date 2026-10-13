"""TinyApps encoder: re-render every unique state once (window.__set) and store only its EYES features, int8
(store.py codec, fitted on the first 2048 screens) - the screenshots themselves are never written to disk.

Out (data/tinyapps): feats.q8.npy/.q8.npz (screens x 256 x 384, read with store.load("feats.npy")),
screens.npz: app (seed per screen row), cls (screen-image id: states that render identically share it),
trans.npy: [app_seed, src_row, x, y, dst_row] float32.

usage: python level0/tinyapps_encode.py [WORKERS=6]
"""
import hashlib, io, json, sys
from multiprocessing import Process, Queue
from pathlib import Path
import numpy as np
from PIL import Image

D = Path(__file__).parent.parent / "data" / "tinyapps"
HTML = (Path(__file__).parent / "tinyapp.html").resolve().as_uri()


def render(k, nw, q):
    from playwright.sync_api import sync_playwright
    apps = sorted(sum((json.loads(p.read_text()) for p in D.glob("apps_w*.json")), []), key=lambda a: a["seed"])
    def shots(page, app):
        page.goto("about:blank"); page.goto(f"{HTML}#{app['seed']}")
        imgs = []
        for s in app["states"]:
            page.evaluate("s => window.__set(s)", s)
            imgs.append(np.asarray(Image.open(io.BytesIO(page.screenshot())).convert("RGB")))
        return np.stack(imgs)

    try:
        with sync_playwright() as p:
            new = lambda: p.chromium.launch(args=["--disable-dev-shm-usage"]).new_page(viewport={"width": 224, "height": 224})
            page = new()  # (Docker's small /dev/shm made a Chromium crash on screenshot)
            for i, app in enumerate(apps):
                if i % nw != k:
                    continue
                for attempt in range(3):  # a crashed browser is replaced and the app re-rendered
                    try:
                        q.put((i, shots(page, app))); break
                    except Exception as e:
                        print(f"worker {k}: app {app['seed']} attempt {attempt + 1} failed: {str(e)[:80]}", flush=True)
                        page = new()
                else:
                    raise RuntimeError(f"app {app['seed']} failed 3 times")
    finally:
        q.put(None)  # always tell the main process this worker is finished


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))
    import store
    from model import Eyes
    nw = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    apps = sorted(sum((json.loads(p.read_text()) for p in D.glob("apps_w*.json")), []), key=lambda a: a["seed"])
    off = np.cumsum([0] + [len(a["states"]) for a in apps])
    N = int(off[-1])
    print(f"{len(apps)} apps, {N} screens to render", flush=True)
    q = Queue(64)
    ps = [Process(target=render, args=(k, nw, q)) for k in range(nw)]
    [p.start() for p in ps]
    eyes, codec, buf = Eyes(), None, []
    qf = np.lib.format.open_memmap(D / "feats.q8.npy", "w+", np.int8, (N, 256, 384))
    cls_hash = [None] * N
    done, n_done = 0, 0

    def write(i, f):
        qf[off[i]:off[i + 1]] = store.encode(f, codec)

    while done < nw:
        item = q.get()
        if item is None:
            done += 1; continue
        i, imgs = item
        for j, im in enumerate(imgs):
            cls_hash[off[i] + j] = hashlib.md5(im.tobytes()).hexdigest()
        f = np.concatenate([eyes(imgs[b:b + 256]).cpu().numpy() for b in range(0, len(imgs), 256)])
        if codec is None:  # hold the first apps until 2048 screens are there to fit the int8 codec
            buf.append((i, f))
            if sum(len(x) for _, x in buf) >= 2048:
                codec = store.fit(np.concatenate([x for _, x in buf]))
                np.savez(D / "feats.q8.npz", **codec)
                [write(*b) for b in buf]; buf = []
        else:
            write(i, f)
        n_done += 1
        if n_done % 100 == 0:
            print(f"{n_done}/{len(apps)} apps encoded", flush=True)
    if codec is None:  # tiny run
        codec = store.fit(np.concatenate([x for _, x in buf])); np.savez(D / "feats.q8.npz", **codec)
        [write(*b) for b in buf]
    qf.flush(); [p.join() for p in ps]
    assert n_done == len(apps), f"only {n_done}/{len(apps)} apps rendered - see worker errors above"
    _, cls = np.unique(cls_hash, return_inverse=True)
    app_of = np.repeat([a["seed"] for a in apps], np.diff(off))
    np.savez(D / "screens.npz", app=app_of, cls=cls)
    tr = np.array([[a["seed"], off[n] + s, x, y, off[n] + d] for n, a in enumerate(apps) for s, x, y, d in a["trans"]], np.float32)
    np.save(D / "trans.npy", tr)
    print(f"done: {N} screens ({cls.max() + 1} distinct images), {len(tr)} actions, features "
          f"{(D / 'feats.q8.npy').stat().st_size / 1e9:.2f} GB int8")
