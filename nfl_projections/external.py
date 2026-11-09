"""Hand-exported third-party matchup tables, normalised onto our own ids.

The first source is FantasyGuru's weekly "smash" tables, which a subscriber
exports by hand. **Nothing in this module fetches anything.** There is no URL,
no session, no client — by design, because the source's terms are personal use
only, so the only way data arrives is a human exporting a CSV and uploading it.
A future contributor looking for the fetcher should stop looking: its absence
is the feature.

**Nothing here feeds a projection.** That is a gate, not an oversight. The
house rule is that a signal earns its way in by beating the production board's
own residual, and this one has not been tested yet — it cannot be, until
several weeks of snapshots exist. Until then these are comparison numbers for a
human reading a slate, in exactly the spirit of `coverage.py` and
`trenches.py`, and `predict.py` does not import this file. See EXPERIMENTS.md,
"FantasyGuru snapshots".

There is prior reason for scepticism about the part of it we can already test.
Our own WR coverage matchups failed three separate tests (coverage.py's
docstring has the numbers: wide receiver was actively *harmful* at −0.0032 MAE
with the interval clear of zero on the wrong side), and the sharp-edge alt-line
backtest found matchup added nothing beyond role and cushion. What is genuinely
untested is the **assignment** — which corner a receiver actually draws, and
whether he is shadowed. nflverse publishes no such field, so this is the one
column here that is not a different estimate of something we already have.

Two export kinds are canonical and get shredded into rows:

    wr_coverage   25 rows, one per corner/receiver pairing, with the coverage
                  shell the corner plays and an "advantage" score
    team_lines    32 rows, an offensive-line and defensive-front rating a team

A third export, the per-game line matchup table, is **entirely derived** from
``team_lines`` plus the schedule: its ADV columns are one side's O-line minus
the other's D-front, verified to 0.02 across all 30 sides of week 9. It is
accepted and checked rather than stored as its own fact, so the two cannot
drift apart.

**The duplicate-header trap.** The WR coverage export has two columns both
headed ``Team`` — the corner's and the receiver's. ``csv.DictReader`` keeps the
last and silently drops the corner's team; ``pandas.read_csv`` renames it to
``Team.1``. Both are wrong in a way that produces a plausible-looking frame, so
no name-keyed mapping is ever built over that header.

**A file identifies itself.** Nothing here reads the filename: these exports
arrive called ``table.csv``, ``table (1).csv`` and whatever the browser names
the next one, so the kind is worked out from the header and the filename is
kept only as a label on the stored snapshot. The header is found by *content*
rather than assumed to be the first line — exports grow title rows, blank
leading lines and extra columns — and columns are resolved by name, so a
reordered or widened export still reads correctly.

The two ``Team`` columns are the one case a name cannot resolve alone, and they
are handled by what they follow: the corner's team is the first ``Team`` after
``Defensive Back``, the receiver's the first after ``Primary Receiver``. That
is the file's actual grammar — each name followed by its team — rather than an
absolute column number, which an added column would quietly shift. If the two
resolve to the same column, parsing raises instead of guessing.
"""
from __future__ import annotations

import csv
import difflib
import io
import logging
import re

import pandas as pd

from .utils import normalize_player_name

logger = logging.getLogger(__name__)

# FantasyGuru's spellings -> nflverse abbreviations. Their team-line export
# uses city-style names ("LA Chargers", "NY Jets") and their per-game export
# uses abbreviations that differ from nflverse on two clubs (JAC for JAX, LVR
# for LV). Both forms resolve here so a caller never has to care which export a
# name came from.
TEAM_ALIASES = {
    # city-style full names, as the team-lines export writes them
    "arizona": "ARI", "atlanta": "ATL", "baltimore": "BAL", "buffalo": "BUF",
    "carolina": "CAR", "chicago": "CHI", "cincinnati": "CIN",
    "cleveland": "CLE", "dallas": "DAL", "denver": "DEN", "detroit": "DET",
    "green bay": "GB", "houston": "HOU", "indianapolis": "IND",
    "jacksonville": "JAX", "kansas city": "KC", "las vegas": "LV",
    "la chargers": "LAC", "la rams": "LAR", "miami": "MIA",
    "minnesota": "MIN", "new england": "NE", "new orleans": "NO",
    "ny giants": "NYG", "ny jets": "NYJ", "philadelphia": "PHI",
    "pittsburgh": "PIT", "san francisco": "SF", "seattle": "SEA",
    "tampa bay": "TB", "tennessee": "TEN", "washington": "WAS",
    # abbreviations that disagree with nflverse
    "jac": "JAX", "lvr": "LV", "wsh": "WAS", "lv r": "LV",
    # spellings seen elsewhere, mapped so an export from another page still lands
    "arizona cardinals": "ARI", "los angeles chargers": "LAC",
    "los angeles rams": "LAR", "new york giants": "NYG",
    "new york jets": "NYJ", "oakland": "LV", "st. louis": "LAR",
    "san diego": "LAC", "washington commanders": "WAS",
}

# Every abbreviation nflverse itself uses, so a normalised value is always one
# of these or None. "LA" is deliberately absent: it is ambiguous between the
# Rams and the Chargers, and guessing would be worse than refusing.
NFLVERSE_TEAMS = {
    "ARI", "ATL", "BAL", "BUF", "CAR", "CHI", "CIN", "CLE", "DAL", "DEN",
    "DET", "GB", "HOU", "IND", "JAX", "KC", "LV", "LAC", "LAR", "MIA", "MIN",
    "NE", "NO", "NYG", "NYJ", "PHI", "PIT", "SEA", "SF", "TB", "TEN", "WAS",
}

# Coverage shells, mapped onto nflverse's own `defense_coverage_type`
# vocabulary where one exists so the two can be compared directly. "Quarters"
# is Cover 4 under another name. "Match" has no nflverse equivalent — it is a
# pattern-match zone that charts as several different shells — so it keeps its
# own value rather than being forced into one.
COVERAGE_SHELLS = {
    "cover-0": "COVER_0", "cover-1": "COVER_1", "cover-2": "COVER_2",
    "cover-3": "COVER_3", "cover-4": "COVER_4", "cover-6": "COVER_6",
    "cover-9": "COVER_9", "quarters": "COVER_4", "2-man": "2_MAN",
    "match": "MATCH", "combo": "COMBO",
}

# How close two names on the SAME team have to be to be the same player.
# FantasyGuru writes "Jacob Parish" where nflverse has "Jacob Parrish" — a
# single letter, and a new one of these turns up most weeks, so a hand-kept
# alias list would be a weekly chore that silently rots. Scoped to one team's
# ~60 players a near-exact match is safe, and every such match is reported as
# `team_fuzzy` rather than being passed off as exact.
FUZZY_CUTOFF = 0.88

# The canonical export kinds. The derived one is listed so an upload of it is
# recognised and checked rather than rejected.
KINDS = ("wr_coverage", "team_lines", "line_matchups")


def normalize_team(value) -> str | None:
    """A FantasyGuru team spelling as an nflverse abbreviation, or None.

    None rather than a guess: an unmapped name is a mapping to fix, and a
    silently wrong team would attach a matchup to the wrong defence.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = re.sub(r"\s+", " ", str(value)).strip()
    if not text:
        return None
    upper = text.upper()
    if upper in NFLVERSE_TEAMS:
        return upper
    return TEAM_ALIASES.get(text.lower())


def parse_coverage_type(value) -> dict:
    """Split "45% Cover-3 Zone" into its parts.

    Returns ``shell_pct`` (how often the corner plays that shell),
    ``shell`` (nflverse vocabulary where it maps), ``man_zone``
    (MAN_COVERAGE / ZONE_COVERAGE, nflverse's own labels) and ``shadow``.

    "100% Shadow Coverage" is the one value that names an assignment rather
    than a shell: it says the corner travels with the receiver. It carries no
    shell, so shell is None and the percentage belongs to the shadowing.
    """
    out = {"coverage_type": None, "shell_pct": None, "shell": None,
           "man_zone": None, "shadow": False}
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return out
    text = re.sub(r"\s+", " ", str(value)).strip()
    if not text:
        return out
    out["coverage_type"] = text
    low = text.lower()

    m = re.search(r"(\d+(?:\.\d+)?)\s*%", low)
    if m:
        out["shell_pct"] = float(m.group(1))

    if "shadow" in low:
        out["shadow"] = True
        return out              # an assignment, not a shell

    for token, shell in COVERAGE_SHELLS.items():
        if token in low:
            out["shell"] = shell
            break

    # The label's own word wins; a bare shell falls back to how it is played.
    # Cover 0/1 and 2-man are man coverage, the rest zone.
    if "zone" in low:
        out["man_zone"] = "ZONE_COVERAGE"
    elif "man" in low:
        out["man_zone"] = "MAN_COVERAGE"
    elif out["shell"] in ("COVER_0", "COVER_1", "2_MAN"):
        out["man_zone"] = "MAN_COVERAGE"
    elif out["shell"] is not None:
        out["man_zone"] = "ZONE_COVERAGE"
    return out


def _rows(csv_text: str) -> list:
    """A CSV's rows as lists, header included, BOM tolerated."""
    text = csv_text.lstrip("﻿") if isinstance(csv_text, str) else csv_text
    return [r for r in csv.reader(io.StringIO(text)) if any(c.strip() for c in r)]


def _number(value):
    """A float, or None for a blank or unparseable cell."""
    try:
        return float(str(value).replace(",", "").strip())
    except (TypeError, ValueError):
        return None


def _norm_header(value) -> str:
    """A header cell flattened for comparison: case, padding, underscores.

    Deliberately not aggressive: hyphens and parentheses are kept, because
    "O-LINE (AWAY)" and "O-LINE ADV (AWAY)" differ only in a word and running
    them together would make two columns indistinguishable.
    """
    text = "" if value is None else str(value)
    return re.sub(r"[\s_]+", " ", text).strip().lower()


# What makes a row the header of a known export: columns that must all be
# present, by name, in any order. A set rather than a sequence because the
# position of a column is not what identifies it -- and because an export
# that gains a column should still be recognisable as itself.
_SIGNATURES = {
    "wr_coverage": frozenset({"rank", "defensive back", "team",
                              "coverage type", "advantage",
                              "primary receiver"}),
    "team_lines": frozenset({"team", "offensive line", "defensive front"}),
    "line_matchups": frozenset({"away", "home",
                                "o-line (away)", "d-line (away)",
                                "o-line (home)", "d-line (home)",
                                "o-line adv (away)", "o-line adv (home)"}),
}

# How far into a file to look for the header. Exports pick up title rows and
# blank lines; a header that is not in the first few rows is a different file.
HEADER_SEARCH_ROWS = 12

# What to call each kind in a message aimed at a person. The slugs are an
# internal key and have no business in an error someone has to read.
_KIND_NAMES = {"wr_coverage": "WR coverage", "team_lines": "team-lines",
               "line_matchups": "line-matchup"}


def find_header(csv_text: str):
    """``(row_index, kind, cells)`` for the first row that heads a known export.

    ``(None, None, [])`` when nothing in the first ``HEADER_SEARCH_ROWS``
    non-empty rows matches. Scanning rather than taking row 0 is what lets a
    file with a title line, a blank line or an export banner above the header
    still be read.
    """
    rows = _rows(csv_text)
    for i, row in enumerate(rows[:HEADER_SEARCH_ROWS]):
        cells = [_norm_header(c) for c in row]
        present = set(cells)
        for kind, required in _SIGNATURES.items():
            if required <= present:
                return i, kind, cells
    return None, None, []


def _column(cells, name, *, after=None):
    """The index of column ``name``, optionally the first one after ``after``.

    ``after`` is how the two ``Team`` columns are told apart: by the named
    column each one follows, which is the export's own grammar.
    """
    start = 0 if after is None else after + 1
    for i in range(start, len(cells)):
        if cells[i] == name:
            return i
    return None


def _body(csv_text: str, kind: str):
    """``(column map, data rows)`` for one export, or raise with the reason.

    One place where a file is matched against what it has to contain, so every
    parser reports a wrong file the same way and none of them reads a column
    by an absolute number.
    """
    rows = _rows(csv_text)
    if not rows:
        raise ValueError("that file has no rows")
    index, found, cells = find_header(csv_text)
    if found is None:
        missing = sorted(_SIGNATURES[kind] - {_norm_header(c) for c in rows[0]})
        raise ValueError(
            f"not a FantasyGuru {_KIND_NAMES[kind]} export: no header row in "
            f"the first {HEADER_SEARCH_ROWS} rows carries the columns it "
            f"needs ({', '.join(sorted(_SIGNATURES[kind]))}). The first row "
            f"is {rows[0]}, missing {', '.join(missing) or 'nothing'}.")
    if found != kind:
        raise ValueError(
            f"that file is a {_KIND_NAMES[found]} export, not "
            f"{_KIND_NAMES[kind]}: its header is {rows[index]}")
    return cells, rows[index + 1:]


def parse_wr_coverage(csv_text: str) -> pd.DataFrame:
    """The corner-vs-receiver export, resolved by column name.

    Rank, Defensive Back, Team, Coverage Type, Advantage, Primary Receiver,
    Team — note the two columns called Team. Neither is read by an absolute
    number: each is the first ``Team`` following the name it belongs to, so an
    added or reordered column cannot silently swap the corner's team for the
    receiver's. See the module docstring for why no name-keyed mapping is
    built over this header at all.
    """
    cells, rows = _body(csv_text, "wr_coverage")
    if not rows:
        return pd.DataFrame()
    i_def = _column(cells, "defensive back")
    i_rec = _column(cells, "primary receiver")
    i_def_team = _column(cells, "team", after=i_def)
    i_rec_team = _column(cells, "team", after=i_rec)
    if i_def_team is None or i_rec_team is None or i_def_team == i_rec_team:
        raise ValueError(
            "that WR coverage export does not have a Team column after each "
            "of Defensive Back and Primary Receiver, so there is no way to "
            f"tell the corner's team from the receiver's; header is {cells}")
    cols = {
        "fg_rank": _column(cells, "rank"),
        "defender": i_def,
        "defender_team": i_def_team,
        "coverage": _column(cells, "coverage type"),
        "advantage": _column(cells, "advantage"),
        "receiver": i_rec,
        "receiver_team": i_rec_team,
    }
    widest = max(cols.values())
    out = []
    for r in rows:
        if len(r) <= widest:
            continue
        cov = parse_coverage_type(r[cols["coverage"]])
        out.append({
            "fg_rank": int(_number(r[cols["fg_rank"]]) or 0) or None,
            "defender": r[cols["defender"]].strip(),
            "defender_team": normalize_team(r[cols["defender_team"]]),
            "defender_team_raw": r[cols["defender_team"]].strip(),
            **cov,
            "advantage": _number(r[cols["advantage"]]),
            "receiver": r[cols["receiver"]].strip(),
            "receiver_team": normalize_team(r[cols["receiver_team"]]),
            "receiver_team_raw": r[cols["receiver_team"]].strip(),
        })
    return pd.DataFrame(out)


def parse_team_lines(csv_text: str) -> pd.DataFrame:
    """The 32-row team export: Team, Offensive Line, Defensive Front."""
    cells, rows = _body(csv_text, "team_lines")
    if not rows:
        return pd.DataFrame()
    i_team = _column(cells, "team")
    i_o = _column(cells, "offensive line")
    i_d = _column(cells, "defensive front")
    widest = max(i_team, i_o, i_d)
    return pd.DataFrame([{
        "team": normalize_team(r[i_team]),
        "team_raw": r[i_team].strip(),
        "offensive_line": _number(r[i_o]),
        "defensive_front": _number(r[i_d]),
    } for r in rows if len(r) > widest])


def parse_line_matchups(csv_text: str) -> pd.DataFrame:
    """The per-game export, one row per SIDE, with its own ADV recomputed.

    Kept as a check rather than a fact: ``adv`` is the side's own O-line minus
    the opponent's D-front, so ``adv_stated`` and ``adv_derived`` must agree.
    Where they do, the file adds nothing to ``team_lines`` but the pairing.
    """
    cells, rows = _body(csv_text, "line_matchups")
    if not rows:
        return pd.DataFrame()
    c = {name: _column(cells, name) for name in (
        "away", "home", "o-line (away)", "d-line (away)", "o-line (home)",
        "d-line (home)", "o-line adv (away)", "o-line adv (home)")}
    widest = max(c.values())
    out = []
    for r in rows:
        if len(r) <= widest:
            continue
        away, home = normalize_team(r[c["away"]]), normalize_team(r[c["home"]])
        # Each side paired with the opponent D-front its own ADV is built from.
        sides = (
            (away, home, False, _number(r[c["o-line (away)"]]),
             _number(r[c["d-line (away)"]]), _number(r[c["o-line adv (away)"]]),
             _number(r[c["d-line (home)"]])),
            (home, away, True, _number(r[c["o-line (home)"]]),
             _number(r[c["d-line (home)"]]), _number(r[c["o-line adv (home)"]]),
             _number(r[c["d-line (away)"]])),
        )
        for team, opp, is_home, oline, dline, adv, opp_dline in sides:
            derived = (None if oline is None or opp_dline is None
                       else round(oline - opp_dline, 3))
            out.append({
                "team": team, "opponent": opp, "home": is_home,
                "offensive_line": oline, "defensive_front": dline,
                "adv_stated": adv, "adv_derived": derived,
                "adv_agrees": (derived is not None and adv is not None
                               and abs(derived - adv) <= 0.02),
            })
    return pd.DataFrame(out)


_PARSERS = {"wr_coverage": parse_wr_coverage, "team_lines": parse_team_lines,
            "line_matchups": parse_line_matchups}


def parse(kind: str, csv_text: str) -> pd.DataFrame:
    """Shred one export. Raises ValueError on an unknown kind or wrong shape."""
    if kind not in _PARSERS:
        raise ValueError(f"unknown export kind {kind!r}; expected one of "
                         f"{', '.join(KINDS)}")
    return _PARSERS[kind](csv_text)


def detect_kind(csv_text: str) -> str | None:
    """Which export a file is, from its own contents, or None.

    The filename is never consulted: these arrive as ``table.csv``,
    ``table (1).csv`` and whatever the browser calls the next one, so an
    upload cannot depend on what a file is called, nor on the person picking
    the right kind from a menu. A mismatch with what they did pick is worth
    refusing rather than storing under the wrong name.
    """
    return find_header(csv_text)[1]


def _roster_index(roster: pd.DataFrame):
    """``{(name, team): id}`` and ``{name: id}`` from a season roster frame.

    A season roster is the right scope rather than the all-time player list,
    and the reason is concrete. Normalising a name strips generational
    suffixes, so "Marvin Harrison Jr" collides with his Hall of Fame father;
    "Mike Evans" is an active receiver and a 1992 defensive tackle. Scoped to
    the players on a roster this season both resolve on their own.

    What scope alone cannot fix is two active players of the same name --
    there is a Justin Jefferson at receiver and another at linebacker, both
    rostered in 2026 -- which is why the team-keyed index exists and is tried
    first.
    """
    name_col = next((c for c in ("full_name", "player_name", "display_name")
                     if c in roster.columns), None)
    id_col = next((c for c in ("gsis_id", "player_id") if c in roster.columns),
                  None)
    if name_col is None or id_col is None:
        return {}, {}
    df = roster[[c for c in {name_col, id_col, "team"} if c in roster.columns]]
    df = df.dropna(subset=[name_col, id_col]).copy()
    df["_key"] = df[name_col].map(normalize_player_name)
    df = df[df["_key"] != ""]
    by_team = {}
    team_keys = {}
    if "team" in df.columns:
        by_team = {(k, t): i for k, t, i
                   in zip(df["_key"], df["team"], df[id_col])}
        for k, t in zip(df["_key"], df["team"]):
            team_keys.setdefault(t, []).append(k)
    counts = df.groupby("_key")[id_col].nunique()
    unambiguous = df[df["_key"].isin(set(counts[counts == 1].index))]
    by_name = dict(zip(unambiguous["_key"], unambiguous[id_col]))
    return by_team, by_name, team_keys


def attach_player_ids(df: pd.DataFrame, columns, roster: pd.DataFrame | None = None,
                      season: int | None = None) -> pd.DataFrame:
    """Add ``<column>_id`` beside each named player column.

    ``columns`` is either a list of name columns, or a mapping of name column
    to the column holding that player's team — pass the mapping where the
    export has one, because the team is what separates two active players of
    the same name.

    Matching is on the normalised key `utils.normalize_player_name` builds,
    which strips the suffixes nflverse drops and rosters keep — the same
    mismatch that once projected a 1,600-yard back off the rookie prior.
    An unmatched name keeps a null id: it is a mapping to fix, never a row to
    drop, and the caller reports it.
    """
    if df.empty:
        return df
    pairs = (columns.items() if hasattr(columns, "items")
             else [(c, None) for c in columns])
    if roster is None:
        import nflreadpy as nfl

        from .utils import to_pandas

        roster = to_pandas(nfl.load_rosters(seasons=[season]))
    by_team, by_name, team_keys = _roster_index(roster)
    out = df.copy()
    for name_col, team_col in pairs:
        if name_col not in out.columns:
            continue
        ids, hows = [], []
        for i, raw in enumerate(out[name_col]):
            key = normalize_player_name(raw)
            team = (out[team_col].iloc[i]
                    if team_col and team_col in out.columns else None)
            found, how = by_team.get((key, team)), "team"
            if found is None:
                found, how = by_name.get(key), "name"
            if found is None and team in team_keys:
                near = difflib.get_close_matches(key, team_keys[team], n=1,
                                                 cutoff=FUZZY_CUTOFF)
                if near:
                    found, how = by_team.get((near[0], team)), "team_fuzzy"
            ids.append(found)
            hows.append(how if found is not None else None)
        # dtype=object, explicitly. Handed a plain list of strings-and-Nones,
        # pandas 3 infers its new `str` dtype and turns the None into NaN, so
        # "no id for this name" stopped being `None` and started being a float
        # — on the newer pandas only, which is how this passed locally and
        # failed in CI. An id column is identifiers or nothing; object dtype
        # says so and keeps the null the callers test for.
        out[f"{name_col}_id"] = pd.Series(ids, index=out.index, dtype=object)
        out[f"{name_col}_match"] = pd.Series(hows, index=out.index, dtype=object)
    return out
