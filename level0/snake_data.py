"""Snake data shared by snake_train.py and snake_stochastic.py: states, keys, step kinds, held-out game split."""
import json
from pathlib import Path
import numpy as np, torch

D = Path(__file__).parent.parent / "data" / "snake"
KEYS = ["Up", "Down", "Left", "Right", "none", "Enter"]
states = json.loads((D / "states.json").read_text())
acts = torch.tensor(json.loads((D / "actions.json").read_text()))
T = len(acts)


def kind(t):  # what kind of step is transition t?
    s, s1, a = states[t], states[t + 1], int(acts[t])
    if s["dead"]:
        return "restart (random new game)" if not s1["dead"] else "game over, idle"
    if s1["dead"]:
        return "death"
    if s1["score"] > s["score"]:
        return "eat (food reappears at random)"
    if s1["dir"] != s["dir"]:
        return "turn"
    return "no key: time passes" if a == 4 else "key, no turn"


kinds = np.array([kind(t) for t in range(T)])
ep = np.cumsum([0] + [states[t]["dead"] and not states[t + 1]["dead"] for t in range(T - 1)])  # game id per transition
held_games = np.random.default_rng(0).random(ep.max() + 1) < 0.2
test = torch.from_numpy(np.where(held_games[ep])[0])
train = torch.from_numpy(np.where(~held_games[ep])[0])
