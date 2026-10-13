"""Data efficiency: how many recorded actions does the IMAGINER need per app? Same training (steps, model) on the first
1-30% of each app's data (Snake: first games; Tetris: random games) vs all of it; test always on the same held-out
screens / games. Runs every missing point (results/curve/), then a table + small-multiples plot.

score per app (held-out): tic-tac-toe / settings / files = imagined screen is the right next screen, on changes;
Snake = latent error on turns vs nothing-changes (lower = better); Tetris = changed board cells read right.

usage: python level0/data_curve.py [APPS=ttt,settings,files,snake,tetris]
"""
import json, os, re, subprocess, sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

L0 = Path(__file__).parent
R = L0.parent / "results"
C = R / "curve"; C.mkdir(exist_ok=True)
APPS = sys.argv[1].split(",") if len(sys.argv) > 1 else ["ttt", "settings", "files", "snake", "tetris"]
FR = [0.01, 0.03, 0.1, 0.3, 1.0]
tag = lambda f: "" if f == 1 else f"_f{f:g}"
# app -> (command, metrics file for a fraction, log for a fraction, score from metrics, higher is better, score name)
JOBS = {
    "ttt": (["train.py", "6000"], lambda f: R / f"level0a_metrics{tag(f)}.json",
            lambda f: R / ("level0a_train_log.txt" if f == 1 else f"curve/ttt{tag(f)}.txt"),
            lambda m: m["board_acc"]["changed"]["model"], True, "right next board (changes)"),
    "settings": (["app_train.py", "settings"], lambda f: R / f"app_settings_metrics{tag(f)}.json",
                 lambda f: R / f"curve/settings{tag(f)}.txt", lambda m: m["changed"]["model"], True, "right next screen (changes)"),
    "files": (["app_train.py", "files"], lambda f: R / f"app_files_metrics{tag(f)}.json",
              lambda f: R / ("app_files_log.txt" if f == 1 else f"curve/files{tag(f)}.txt"),
              lambda m: m["changed"]["model"], True, "right next screen (changes)"),
    "snake": (["snake_train.py", "6000"], lambda f: R / f"level3a_metrics{tag(f)}.json",
              lambda f: R / ("level3a_log.txt" if f == 1 else f"curve/snake{tag(f)}.txt"),
              lambda m: m["groups"]["turn"]["model"], False, "turn error vs no-change (lower = better)"),
    "tetris": (["tetris/train.py", "16000", "hist"], lambda f: R / f"tetris_t0_hist{tag(f)}_metrics.json",
               lambda f: R / ("tetris_t0_hist_log.txt" if f == 1 else f"curve/tetris{tag(f)}.txt"),
               lambda m: m["changed_cells"]["model"], True, "changed board cells right"),
}
FRACS = {"tetris": [0.01, 0.05, 0.2, 1.0]}  # Tetris: 16k steps per point, fewer points

for app in APPS:
    cmd, met, log, *_ = JOBS[app]
    for f in FRACS.get(app, FR):
        if met(f).exists():
            continue
        print(f"--- {app} {f:.0%}", flush=True)
        with open(log(f), "w") as fh:
            subprocess.run([sys.executable, str(L0 / cmd[0]), *cmd[1:]], env={**os.environ, "FRAC": str(f)},
                           stdout=fh, stderr=subprocess.STDOUT, check=True)

rows = {}
for app in APPS:
    _, met, log, score, up, name = JOBS[app]
    for f in FRACS.get(app, FR):
        if not met(f).exists():
            continue
        txt = log(f).read_text(errors="ignore") if log(f).exists() else ""
        n = re.search(r"trains on (\d+)", txt) or re.search(r"train (\d+)", txt)
        rows.setdefault(app, []).append({"frac": f, "train_actions": int(n.group(1)) if n else None,
                                         "score": score(json.loads(met(f).read_text()))})
(R / "data_curve.json").write_text(json.dumps(rows, indent=2))

print("\napp       data used: score (training actions)")
for app, rs in rows.items():
    cell = lambda r: f"{r['frac']:.0%}: {r['score']:.3f} ({r['train_actions']})"
    print(f"{app:<9} " + "  ".join(f"{cell(r):>21}" for r in rs) + f"   <- {JOBS[app][5]}")

fig, axs = plt.subplots(1, len(rows), figsize=(3.2 * len(rows), 3), squeeze=False)
for ax, (app, rs) in zip(axs[0], rows.items()):
    x = [r["train_actions"] for r in rs]
    ax.plot(x, [r["score"] for r in rs], color="#2a78d6", lw=2, marker="o", ms=8, mec="#fcfcfb", mew=2)
    ax.set_xscale("log"); ax.set_title(app, color="#0b0b0b"); ax.set_xlabel("training actions", color="#52514e")
    ax.set_ylabel(JOBS[app][5], color="#52514e", fontsize=8)
    ax.grid(alpha=0.25); ax.spines[["top", "right"]].set_visible(False)
fig.suptitle("How much data does one app need? (held-out screens / games)", color="#0b0b0b")
fig.tight_layout(); fig.savefig(R / "data_curve.png", dpi=120, facecolor="#fcfcfb")
print(f"saved {R / 'data_curve.png'}")
