"""Tetris piece shapes read straight from the game's own src/engine/Tetromino.js (no hand-copied tables)."""
import json, re
from pathlib import Path

_src = (Path(__file__).parent / "src" / "engine" / "Tetromino.js").read_text()
_src = _src[_src.index("export const TETROMINOES"):_src.index("\n};")]  # shapes only (KICK_TABLE also has an "I:" key)
OFFSETS = {}
for t, body in re.findall(r"\n    ([IJLOSTZ]): \{(.*?)\n    \}", _src, re.S):
    OFFSETS[t] = {k: json.loads(v) for k, v in re.findall(r"\b(0|R|2|L): (\[\[.*?\]\])", body)}
ORIENT = [0, "R", 2, "L"]  # orientation index -> key, as in Tetromino.getBlocks


def visible_cells(s):
    """Board with the falling piece drawn in: 20 x 10 list of bools (visible rows only)."""
    grid = [[ch == "#" for ch in row] for row in s["board"]]
    if s["piece"]:
        for dx, dy in OFFSETS[s["piece"]][str(ORIENT[s["o"]])]:
            r, c = s["y"] + dy - 20, s["x"] + dx
            if 0 <= r < 20 and 0 <= c < 10:
                grid[r][c] = True
    return grid


if __name__ == "__main__":  # self-check: 7 pieces x 4 orientations x 4 blocks
    assert sorted(OFFSETS) == list("IJLOSTZ"), OFFSETS.keys()
    assert all(len(OFFSETS[t]) == 4 and all(len(b) == 4 for b in OFFSETS[t].values()) for t in OFFSETS)
    print("ok", OFFSETS["J"])
