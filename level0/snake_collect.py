"""Random-key collector for Snake (Level 3a).

Every env step = press one key (or none) -> one clock tick -> screenshot.
Snake states rarely repeat, so frames are stored as a sequence: transition t = (frame t, key t, frame t+1).
"""
import json, random, sys, time
from pathlib import Path
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).parent
OUT = ROOT.parent / "data" / "snake"
STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 40_000
KEYS = ["ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight", None, "Enter"]  # action ids 0..5
random.seed(0)


MOVE = {0: (0, -1), 1: (0, 1), 2: (-1, 0), 3: (1, 0)}
DIRS = {"U": 0, "D": 1, "L": 2, "R": 3}


def policy(s):  # data bot (not the model): mostly safe random moves, sometimes chases food, sometimes dies
    if s["dead"]:
        return 5 if random.random() < 0.5 else random.randrange(5)
    if random.random() < 0.05:
        return random.randrange(6)
    head, body, cur = s["snake"][0], s["snake"][:-1], DIRS[s["dir"]]

    def where(a):  # cell the head lands on after action a (4 = none = keep going; reversing is ignored)
        d = cur if a == 4 or MOVE[a] == tuple(-v for v in MOVE[cur]) else a
        return head[0] + MOVE[d][0], head[1] + MOVE[d][1]

    safe = [a for a in range(5) if 0 <= where(a)[0] < 8 and 0 <= where(a)[1] < 8 and list(where(a)) not in body]
    if not safe:
        return random.randrange(5)
    if s["food"] and random.random() < 0.4:
        return min(safe, key=lambda a: abs(where(a)[0] - s["food"][0]) + abs(where(a)[1] - s["food"][1]))
    return random.choices(safe, weights=[3 if a == 4 else 1 for a in safe])[0]


def main():
    (OUT / "img").mkdir(parents=True, exist_ok=True)
    states, acts = [], []
    with sync_playwright() as p:
        page = p.chromium.launch().new_page(viewport={"width": 224, "height": 224})
        page.goto((ROOT / "snake.html").as_uri())

        def observe():
            states.append(page.evaluate("window.__state"))
            page.screenshot(path=OUT / "img" / f"{len(states) - 1}.png")

        observe()
        t0 = time.time()
        for step in range(STEPS):
            a = policy(states[-1])
            if KEYS[a]:
                page.keyboard.press(KEYS[a])
            page.evaluate("step()")
            acts.append(a)
            observe()
            if step % 5000 == 0:
                print(f"step {step:>6}  {step / (time.time() - t0 + 1e-9):.0f} steps/s", flush=True)

    (OUT / "states.json").write_text(json.dumps(states))
    (OUT / "actions.json").write_text(json.dumps(acts))
    eats = sum(states[i + 1]["score"] > states[i]["score"] for i in range(STEPS))
    deaths = sum(states[i + 1]["dead"] and not states[i]["dead"] for i in range(STEPS))
    print(f"done: {STEPS} steps, {eats} food eaten, {deaths} deaths, "
          f"{sum(s['dead'] for s in states) / len(states):.1%} frames on game-over screen")


if __name__ == "__main__":
    main()
