"""Tic-tac-toe rules in python (a copy of tictactoe.html) - for picking test goals and scoring, never used by the agent."""

LINES = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6)]
EMPTY = {"board": [None] * 9, "turn": "X", "winner": None}


def step(s, move):  # move: 0-8 = cell, 9 = reset, None = nothing
    if move == 9:
        return EMPTY
    if move is None or s["winner"] or s["board"][move]:
        return s
    b = list(s["board"]); b[move] = s["turn"]
    w = next((b[a] for a, c, e in LINES if b[a] and b[a] == b[c] == b[e]), None) or ("draw" if all(b) else None)
    return {"board": b, "turn": "O" if s["turn"] == "X" else "X", "winner": w}


def hit(x, y):  # which element a pixel click lands on
    if 22 <= x < 202 and 24 <= y < 204:
        return int((y - 24) // 60) * 3 + int((x - 22) // 60)
    return 9 if 82 <= x < 142 and 206 <= y < 222 else None
