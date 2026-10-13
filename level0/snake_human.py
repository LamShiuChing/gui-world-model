"""Record YOU playing Snake (human demonstrations + human baseline score).

A visible browser window opens (3x zoom). Arrow keys steer, Enter restarts after game over, Esc (or closing the window) ends.
Smooth controls: the game clock runs INSIDE the browser (no Python round-trips per tick), and key presses are QUEUED, so two
quick presses (e.g. Up then Left for a U-turn) both happen, one per tick. The game waits for your first arrow key and pauses
on game over until you press Enter.
Only states + applied keys are logged during play; frames are re-rendered headless at 1x afterwards, pixel-identical to the
bot data. Each run saves to its own session folder.

usage: python level0/snake_human.py [TICK_MS=200]
"""
import json, sys, time
from pathlib import Path
from playwright.sync_api import sync_playwright, Error

ROOT = Path(__file__).parent
OUT = ROOT.parent / "data" / "snake_human" / time.strftime("session_%Y%m%d_%H%M%S")
TICK = int(sys.argv[1]) if len(sys.argv) > 1 else 200

# Key ids match the bot data: 0 Up, 1 Down, 2 Left, 3 Right, 4 no key, 5 Enter.
GAME_LOOP = """tick => {
  const ID = { ArrowUp: 0, ArrowDown: 1, ArrowLeft: 2, ArrowRight: 3, Enter: 5 }, DIRS = 'UDLR';
  const OPP = { U: 'D', D: 'U', L: 'R', R: 'L' };
  window.__queue = []; window.__log = [JSON.parse(JSON.stringify(window.__state))]; window.__quit = false;
  let started = false;
  // capture phase on window: runs before (and blocks) the page's own instant-steer handler
  window.addEventListener('keydown', e => {
    if (e.key === 'Escape') { window.__quit = true; return; }
    const k = ID[e.key]; if (k === undefined) return;
    e.preventDefault(); e.stopImmediatePropagation();
    if (k !== 5) started = true;
    if (window.__queue.length < 3) window.__queue.push(k);
  }, true);
  setInterval(() => {
    if (!started || window.__quit) return;
    let a = 4;
    if (s.dead) {  // paused on game over until Enter
      const i = window.__queue.indexOf(5);
      if (i < 0) { window.__queue = []; return; }
      window.__queue = []; reset(); a = 5;
    } else {
      while (window.__queue.length) {  // first queued key that actually turns the snake
        const k = window.__queue.shift(); if (k === 5) continue;
        const d = DIRS[k];
        if (d !== s.dir && d !== OPP[s.dir]) { s.next = d; a = k; break; }
      }
    }
    step();
    window.__log.push({ a, s: JSON.parse(JSON.stringify(window.__state)) });
  }, tick);
}"""


def record():
    states, acts, scores = [], [], []
    with sync_playwright() as p:
        page = p.chromium.launch(headless=False).new_page(viewport={"width": 224, "height": 224}, device_scale_factor=3)
        page.goto((ROOT / "snake.html").as_uri())
        page.evaluate(GAME_LOOP, TICK)
        print("Click the game window, press an arrow key to start. Enter = new game, Esc = finish.", flush=True)
        try:
            while True:  # collect the log in the background; the browser keeps the clock
                time.sleep(0.3)
                new = page.evaluate("window.__log.splice(0, window.__log.length)")
                for e in new:
                    if not states:  # initial state
                        states.append(e)
                        continue
                    acts.append(e["a"])
                    states.append(e["s"])
                    if e["s"]["dead"] and not states[-2]["dead"]:
                        scores.append(e["s"]["score"])
                        print(f"game {len(scores)}: score {e['s']['score']}  (Enter to play again, Esc to finish)", flush=True)
                if page.evaluate("window.__quit"):
                    break
        except Error:
            pass  # window closed: everything up to the last poll is kept
    return states, acts, scores


def render(states):  # same page, headless, 1x: identical pixels to the bot data
    (OUT / "img").mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        page = p.chromium.launch().new_page(viewport={"width": 224, "height": 224})
        page.goto((ROOT / "snake.html").as_uri())
        for i, s in enumerate(states):
            page.evaluate("st => { s = st; render(); }", s)
            page.screenshot(path=OUT / "img" / f"{i}.png")


if __name__ == "__main__":
    states, acts, scores = record()
    if len(states) < 2:
        sys.exit("nothing recorded")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "states.json").write_text(json.dumps(states))
    (OUT / "actions.json").write_text(json.dumps(acts))
    print(f"rendering {len(states)} frames ...", flush=True)
    render(states)
    eats = sum(states[i + 1]["score"] > states[i]["score"] for i in range(len(acts)))
    print(f"saved {len(acts)} steps to {OUT} | games finished: {len(scores)} | scores: {scores} | "
          f"mean {sum(scores) / max(len(scores), 1):.1f} | food per 100 steps {100 * eats / len(acts):.1f}")
