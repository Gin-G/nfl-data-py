"""Matching a FanDuel slate to the board, where no shared id exists.

Everything inside the package joins on the gsis id. FanDuel's export carries
its own ids and a display name, so a slate is the one place a name has to do
the work -- and it has to be done in ONE place, the same way, by everything
that reads a slate (the optimizer, /dfs/targets).

The rules, in order, for each slate row:

  1. name key + team             the ordinary case
  2. name key alone              only when exactly one board row has it (a
                                 player traded since the board was built)
  3. last name + team + position only when exactly one board row qualifies and
                                 the first names agree on a prefix ("Josh" /
                                 "Joshua", "Matt" / "Matthew") or an initial

A name key drops suffixes, punctuation and middle initials: "Tre' Harris" and
"Tre Harris", "D.J. Moore" and "DJ Moore", "James Cook III" and "James Cook"
are each one key. Anything still unmatched is reported, never guessed: an
unmatched player shows up as "no projection", which is visible; a wrong match
puts another man's projection on him, which is not.
"""
import re

import pandas as pd

from .utils import NAME_SUFFIXES

# FanDuel's abbreviations where nflverse's differ.
TEAM_ALIASES = {"JAC": "JAX", "LAR": "LA", "WSH": "WAS", "LVR": "LV", "OAK": "LV",
                "SD": "LAC", "STL": "LA"}


def team_key(team):
    if team is None or (isinstance(team, float) and pd.isna(team)):
        return ""
    t = str(team).strip().upper()
    return TEAM_ALIASES.get(t, t)


def name_key(name):
    """Lower-case, suffix-free, punctuation-free, middle initials dropped."""
    if name is None or (isinstance(name, float) and pd.isna(name)):
        return ""
    n = str(name).strip()
    for suffix in NAME_SUFFIXES:
        n = re.sub(suffix, "", n, flags=re.IGNORECASE)
    n = re.sub(r"\s+[A-Z]\.(?=\s)", " ", n)          # "Patrick A. Mahomes"
    n = re.sub(r"[.'’`\-]", "", n).lower()
    return re.sub(r"\s+", " ", n).strip()


def _first_names_agree(a, b):
    if not a or not b:
        return False
    return a.startswith(b) or b.startswith(a) or (len(a) <= 2 or len(b) <= 2) and a[0] == b[0]


class Matcher:
    """The rules above, built once over a board and asked one player at a time."""

    def __init__(self, board, *, name="player_name", team="team", pos="position"):
        board = board if board is not None else pd.DataFrame()
        n = len(board)
        b = pd.DataFrame({
            "key": board[name].map(name_key) if n else pd.Series(dtype=object),
            "team": board[team].map(team_key) if n and team in board else "",
            "pos": (board[pos].astype(str).str.upper() if n and pos in board else ""),
        }, index=board.index)
        b["last"] = b["key"].str.split().str[-1].fillna("")
        b["first"] = b["key"].str.split().str[0].fillna("")
        self._first = b["first"].to_dict()
        self._by_key_team = b.groupby(["key", "team"]).groups if n else {}
        self._by_key = b.groupby("key").groups if n else {}
        self._by_last = b.groupby(["last", "team", "pos"]).groups if n else {}

    def match(self, name, team=None, position=None):
        """The board index this player is, or None when no rule is sure."""
        key = name_key(name)
        if not key:
            return None
        team = team_key(team)
        hit = self._by_key_team.get((key, team))
        if hit is not None and len(hit) == 1:
            return hit[0]
        hit = self._by_key.get(key)
        if hit is not None and len(hit) == 1:
            return hit[0]
        parts = key.split()
        pos = str(position or "").upper().split("/")[0]
        cands = self._by_last.get((parts[-1], team, pos))
        if cands is None:
            return None
        cands = [c for c in cands if _first_names_agree(parts[0], self._first[c])]
        return cands[0] if len(cands) == 1 else None


def match_slate(slate, board, *, slate_name="Nickname", slate_team="Team",
                slate_pos="Position", board_name="player_name", board_team="team",
                board_pos="position"):
    """Board row index for every slate row (a Series on the slate's index, None
    where nothing matched safely)."""
    out = pd.Series(None, index=slate.index if slate is not None else None,
                    dtype="object")
    if board is None or len(board) == 0 or slate is None or len(slate) == 0:
        return out
    m = Matcher(board, name=board_name, team=board_team, pos=board_pos)
    for i, row in slate.iterrows():
        out[i] = m.match(row.get(slate_name), row.get(slate_team), row.get(slate_pos))
    return out


def attach_board(slate, board, columns, **kw):
    """The slate with `columns` from its matched board row (left join)."""
    idx = match_slate(slate, board, **kw)
    out = slate.copy()
    for c in columns:
        if c not in board.columns:
            continue
        target = c if c not in out.columns else f"{c}_board"
        values = board[c].to_dict()
        out[target] = [values.get(j) if j is not None and pd.notna(j) else None
                       for j in idx]
        out[target] = out[target].infer_objects()
    return out
