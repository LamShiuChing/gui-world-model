"""Tetris (Monochrome Rush, local copy) data collector. Never touches the live site.

Every env step = tap one key (or none) -> advance the game clock 50 ms -> screenshot the game area.
Transition t = (frame t, key t, frame t+1) inside one episode. True game state is logged per frame.
Data bot (NOT the model): plays a simple placement heuristic with per-episode random noise, random start level 1-20,
so the data has line clears, long games, all gravity speeds incl. 20G, and plenty of random mistakes.

usage: python collect.py WORKER_ID STEPS [expert]      (run several workers in parallel)
"expert" = labelled EXPERT data (no noise, El-Tetris features + 1-piece lookahead) -> data/tetris_expert/
"""
import functools, http.server, json, random, sys, threading, time
from pathlib import Path
from playwright.sync_api import sync_playwright

HERE = Path(__file__).parent
WID = int(sys.argv[1]) if len(sys.argv) > 1 else 0
STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 20_000
EXPERT = len(sys.argv) > 3 and sys.argv[3] == "expert"  # labelled EXPERT data: no noise, El-Tetris + lookahead, hard drops
OUT = HERE.parent.parent / "data" / ("tetris_expert" if EXPERT else "tetris") / f"w{WID}"
KEYS = [None, "ArrowLeft", "ArrowRight", "ArrowDown", "KeyZ", "KeyX", "Space", "KeyC"]  # action ids 0..7
TICK = 50  # ms of game time per step
CLIP = {"x": 270, "y": 140, "width": 620, "height": 620}  # hold + stats + board + next
SEED = WID + (100 if EXPERT else 0)  # expert games get their own piece sequences and levels
random.seed(SEED)

http.server.SimpleHTTPRequestHandler.log_message = lambda *a: None
srv = http.server.ThreadingHTTPServer(("127.0.0.1", 8800 + WID), functools.partial(http.server.SimpleHTTPRequestHandler, directory=HERE))
threading.Thread(target=srv.serve_forever, daemon=True).start()

INIT = f"""
let _s = {1000 + SEED}; Math.random = () => ((_s = (_s * 1664525 + 1013904223) >>> 0) / 4294967296);  // reproducible
localStorage.setItem('setting_shake', '0'); localStorage.setItem('setting_lineclear', 'false'); localStorage.setItem('setting_sfx', '0');
"""
HOOKS = """() => {
  const g = window.__game, sp = g.spawnPiece.bind(g);
  g.spawnPiece = function () { g.__spawns = (g.__spawns || 0) + 1; return sp(); };
  window.__state = () => ({
    board: g.board.slice(20).map(r => r.map(v => v ? '#' : '.').join('')),
    piece: g.player.piece && g.player.piece.type, o: g.player.piece ? g.player.piece.orientation : null,
    x: g.player.pos.x, y: g.player.pos.y, hold: g.player.holdPiece, next: g.bag.slice(0, 5),
    score: Math.floor(g.player.score), level: g.player.level, lines: g.player.totalLines || 0,
    over: g.gameOver, clearing: g.clearAnimationTimer > 0, ms: Math.round(g.elapsedTime), spawns: g.__spawns || 0 });
  window.__best = () => {  // data bot only: best (orientation, x) by a classic stacking heuristic
    const p = g.player.piece; if (!p) return null;
    const W = 10, H = 40; let best = null;
    const fits = (x, y, o) => p.getBlocks(x, y, o).every(b => b.x >= 0 && b.x < W && b.y < H && (b.y < 0 || g.board[b.y][b.x] === 0));
    for (let o = 0; o < 4; o++) for (let x = -3; x < W + 3; x++) {
      let y = g.player.pos.y; if (!fits(x, y, o)) continue;
      while (fits(x, y + 1, o)) y++;
      const b = g.board.map(r => r.slice()); p.getBlocks(x, y, o).forEach(c => { if (c.y >= 0) b[c.y][c.x] = 1; });
      const keep = b.filter(r => !r.every(v => v)), full = H - keep.length;
      let agg = 0, holes = 0, bump = 0, prev = null;
      for (let c = 0; c < W; c++) {
        let h = 0, seen = false;
        for (let r = 0; r < keep.length; r++) { if (keep[r][c]) { if (!seen) h = keep.length - r; seen = true; } else if (seen) holes++; }
        agg += h; if (prev !== null) bump += Math.abs(h - prev); prev = h;
      }
      const s = -0.51 * agg + 0.76 * full - 0.36 * holes - 0.18 * bump;
      if (!best || s > best.s) best = { s, o, x };
    }
    return best;
  };
  window.__bestExpert = () => {  // EXPERT data bot: El-Tetris features + 1-piece lookahead with the first preview piece
    const p = g.player.piece; if (!p) return null;
    const W = 10, H = 40, Piece = p.constructor;
    // El-Tetris weights (Yiyuan Lee's tuned Dellacherie features; quoted from memory, fine for a data bot)
    const score = (b, landing, cleared) => {
      let rowT = 0, colT = 0, holes = 0, wells = 0;
      for (let r = 0; r < H; r++) { let prev = 1; for (let c = 0; c < W; c++) { const v = b[r][c] ? 1 : 0; if (v !== prev) rowT++; prev = v; } if (!prev) rowT++; }
      for (let c = 0; c < W; c++) {
        let prev = 0, seen = false, depth = 0;
        for (let r = 0; r < H; r++) {
          const v = b[r][c] ? 1 : 0; if (v !== prev) colT++; prev = v;
          if (v) seen = true; else if (seen) holes++;
          const wellCell = !v && (c === 0 || b[r][c - 1]) && (c === W - 1 || b[r][c + 1]);
          if (wellCell) { depth++; wells += depth; } else depth = 0;
        }
        if (!prev) colT++;
      }
      return -4.500158825082766 * landing + 3.4181268101392694 * cleared - 3.2178882868487753 * rowT
             - 9.348695305445199 * colT - 7.899265427351652 * holes - 3.3855972247263626 * wells;
    };
    const placements = (board, type, y0, reach) => {  // reach: set of "o,x" the piece can actually get to (null = any)
      const t = new Piece(type), out = [];
      const fits = (x, y, o) => t.getBlocks(x, y, o).every(c => c.x >= 0 && c.x < W && c.y < H && (c.y < 0 || board[c.y][c.x] === 0));
      for (let o = 0; o < (type === 'O' ? 1 : 4); o++) for (let x = -3; x < W + 3; x++) {
        if (reach && !reach.has(o + ',' + x)) continue;
        let y = y0; if (!fits(x, y, o)) continue;
        while (fits(x, y + 1, o)) y++;
        const b = board.map(r => r.slice()); t.getBlocks(x, y, o).forEach(c => { if (c.y >= 0) b[c.y][c.x] = 1; });
        const keep = b.filter(r => !r.every(v => v)), cleared = H - keep.length;
        while (keep.length < H) keep.unshift(Array(W).fill(0));
        out.push({ o, x, b: keep, s: score(keep, H - y, cleared) });
      }
      return out;
    };
    // which (orientation, x) can the falling piece reach from where it is now? moves/rotations without wall kicks;
    // at 20G (dropInterval 0) the piece falls to the stack after every move, so tall columns block sideways paths
    const fitsNow = (x, y, o) => p.getBlocks(x, y, o).every(c => c.x >= 0 && c.x < W && c.y < H && (c.y < 0 || g.board[c.y][c.x] === 0));
    const settle = (x, y, o) => { if (g.dropInterval === 0) while (fitsNow(x, y + 1, o)) y++; return y; };
    const reach = new Set(), queue = [[p.orientation, g.player.pos.x, settle(g.player.pos.x, g.player.pos.y, p.orientation)]];
    reach.add(queue[0][0] + ',' + queue[0][1]);
    while (queue.length) {
      const [o, x, y] = queue.shift();
      for (const [o2, x2] of [[o, x - 1], [o, x + 1], [(o + 1) % 4, x], [(o + 3) % 4, x]]) {
        if (reach.has(o2 + ',' + x2) || !fitsNow(x2, y, o2)) continue;
        reach.add(o2 + ',' + x2); queue.push([o2, x2, settle(x2, y, o2)]);
      }
    }
    let best = null;
    const nxt = g.bag[0];
    for (const c1 of placements(g.board, p.type, g.player.pos.y, reach)) {
      const c2 = nxt ? placements(c1.b, nxt, 18) : [];
      const s = c1.s + (c2.length ? Math.max(...c2.map(c => c.s)) : -1e6);
      if (!best || s > best.s) best = { s, o: c1.o, x: c1.x };
    }
    return best;
  };
}"""


def main():
    (OUT / "img").mkdir(parents=True, exist_ok=True)
    states, acts, eps = [], [], []
    with sync_playwright() as p:
        page = p.chromium.launch().new_page(viewport={"width": 1280, "height": 800})
        page.route("**/api/**", lambda r: r.abort())  # no leaderboard calls, ever
        page.add_init_script(INIT)
        page.clock.install()
        page.goto(f"http://127.0.0.1:{8800 + WID}/index.html")
        page.evaluate(HOOKS)
        page.get_by_text("START GAME").click()
        page.clock.run_for(4000)  # countdown

        def new_episode():
            level = random.choice([1, 1, 1] + list(range(2, 21)))
            page.evaluate(f"(() => {{ const g = window.__game; g.isRunning = false; g.restart({level}); }})()")
            page.clock.run_for(TICK)
            return 0.0 if EXPERT else random.uniform(0.03, 0.25)  # this episode's bot noise

        ep, noise, ep_len, plan, last_spawn = 0, new_episode(), 0, None, -1

        def observe():
            states.append(page.evaluate("window.__state()"))
            eps.append(ep)
            page.screenshot(path=OUT / "img" / f"{len(states) - 1}.png", clip=CLIP)

        observe()
        t0 = time.time()
        for step in range(STEPS):
            s = states[-1]
            if s["over"] or ep_len >= 3000:  # episode boundary: harness restarts (not an action, no transition recorded)
                ep += 1
                noise, ep_len, plan = new_episode(), 0, None
                acts.append(-1)  # marks "no transition" between frame t and t+1
                observe()
                continue
            if EXPERT or s["spawns"] != last_spawn or plan is None:  # expert re-plans every tick (adapts to blocked moves)
                plan, last_spawn = page.evaluate("window.__bestExpert()" if EXPERT else "window.__best()"), s["spawns"]
            if random.random() < noise or plan is None or s["piece"] is None:
                a = random.randrange(8)
            elif s["o"] != plan["o"]:
                a = 4 if EXPERT and (s["o"] - plan["o"]) % 4 == 1 else 5  # rotate the short way (Z = CCW, X = CW)
            elif s["x"] != plan["x"]:
                a = 2 if s["x"] < plan["x"] else 1
            else:
                a = 6 if EXPERT else random.choices([6, 3, 0], weights=[0.4, 0.2, 0.4])[0]  # hard drop, soft drop, or let gravity work
            if KEYS[a]:
                page.keyboard.press(KEYS[a])
            page.clock.run_for(TICK)
            acts.append(a)
            ep_len += 1
            observe()
            if step % 2000 == 0:
                print(f"w{WID} step {step:>6}  {step / (time.time() - t0 + 1e-9):.1f} steps/s  episodes {ep + 1}", flush=True)

    (OUT / "states.json").write_text(json.dumps(states))
    (OUT / "actions.json").write_text(json.dumps(acts))
    (OUT / "episodes.json").write_text(json.dumps(eps))
    lines = sum(max(0, states[i + 1]["lines"] - states[i]["lines"]) for i in range(len(acts)) if acts[i] >= 0)
    print(f"w{WID} done: {STEPS} steps, {ep + 1} episodes, {lines} lines cleared, "
          f"levels {sorted(set(s['level'] for s in states))[:3]}..{max(s['level'] for s in states)}")


if __name__ == "__main__":
    main()
