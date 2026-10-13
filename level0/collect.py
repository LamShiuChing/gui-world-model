"""Random-click collector for tic-tac-toe (Level 0a).

The page renders deterministically, so a screenshot is a pure function of the state:
store one PNG per unique state and log every click as (src_state, x, y, dst_state).
"""
import json, random, sys, time
from pathlib import Path
import numpy as np
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).parent
OUT = ROOT.parent / "data" / "level0"
STEPS = int(sys.argv[1]) if __name__ == "__main__" and len(sys.argv) > 1 else 100_000
random.seed(0)


def key(s):
    return "".join(c or "." for c in s["board"]) + s["turn"] + (s["winner"] or "-")


def sample_click():
    r = random.random()
    if r < 0.7:  # on the board
        return random.uniform(22, 202), random.uniform(24, 204)
    if r < 0.8:  # on the reset button
        return random.uniform(83, 141), random.uniform(207, 221)
    return random.uniform(0, 224), random.uniform(0, 224)  # anywhere


def main():
    (OUT / "img").mkdir(parents=True, exist_ok=True)
    ids, states, trans, checked = {}, [], [], 0
    with sync_playwright() as p:
        page = p.chromium.launch().new_page(viewport={"width": 224, "height": 224})
        page.goto((ROOT / "tictactoe.html").as_uri())

        def observe():
            nonlocal checked
            s = page.evaluate("window.__state")
            k = key(s)
            if k not in ids:
                ids[k] = len(states)
                states.append({"key": k, **s})
                page.screenshot(path=OUT / "img" / f"{ids[k]}.png")
            elif random.random() < 0.01:  # determinism check: same state -> identical pixels
                assert page.screenshot() == (OUT / "img" / f"{ids[k]}.png").read_bytes(), f"non-deterministic render: {k}"
                checked += 1
            return ids[k]

        cur, t0 = observe(), time.time()
        for step in range(STEPS):
            x, y = sample_click()
            page.mouse.click(x, y)
            nxt = observe()
            trans.append((cur, x, y, nxt))
            cur = nxt
            if step % 10_000 == 0:
                print(f"step {step:>7}  states {len(states):>5}  {step / (time.time() - t0 + 1e-9):.0f} steps/s", flush=True)

    t = np.array(trans, dtype=np.float32)
    np.savez(OUT / "transitions.npz", src=t[:, 0].astype(np.int32), x=t[:, 1], y=t[:, 2], dst=t[:, 3].astype(np.int32))
    (OUT / "states.json").write_text(json.dumps(states))
    noop = (t[:, 0] == t[:, 3]).mean()
    print(f"done: {len(trans)} transitions, {len(states)} unique states, {noop:.1%} no-op, {checked} determinism checks passed")


if __name__ == "__main__":
    main()
