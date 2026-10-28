"""Projected DFS roster percentages — what share of the field will own a player.

A projection says how many points a player scores. It says nothing about how
many other entries also have him, and in a tournament that second number is
half the game: being right about a player everyone owns wins you nothing, and
being right about one nobody owns wins you the week. Nothing in this package
measured the field until this module.

Footballguys publishes projected roster percentages per site and week, free, as
plain HTML tables. The numbers are internally consistent in the way a real
projection has to be — across a FanDuel classic roster the positions sum to
very nearly 100/240/320/120/100 for QB/RB/WR/TE/DEF, which is one QB, one
defence and 2.4/3.2/1.2 of the flexible spots. That accounting is checked on
every parse (`check_totals`), because it is the one cheap way to notice the
page has changed shape under us.

    from nfl_projections import ownership
    own = ownership.load(week=7)                     # name, position, ownership
    pool = ownership.attach(pool, own)               # adds `ownership`
    pool = ownership.add_leverage(pool, "fanduel_fantasy_points")

`ownership` is a percentage (0-100), the same units the source prints, because
that is how the number gets talked about. `leverage` is where our board sits on
a player minus where the field sits, both as within-position percentiles, so it
runs -1 (the field loves him, we do not) to +1 (we love him, the field has not
noticed).

The page is only populated from about the Friday before each slate; asking for
a week before then returns a page with no rows, which `load` reports as an
empty frame rather than an error, since "not out yet" is a normal Tuesday.
"""

import re
import warnings

import numpy as np
import pandas as pd

from .optimizer import normalize_name

URL = "https://www.footballguys.com/dfs-roster-percentages"

# The table id on the page -> the position it holds. "td" is theirs for team
# defence; everything else is the obvious abbreviation.
TABLE_POSITIONS = {"qb": "QB", "rb": "RB", "wr": "WR", "te": "TE", "td": "D"}

# How many of each position a FanDuel classic roster holds once the FLEX is
# shared out at its historical rate. Used only to sanity-check a parse.
EXPECTED_SHARE = {"QB": 100.0, "RB": 240.0, "WR": 320.0, "TE": 120.0, "D": 100.0}

_ROW = re.compile(r'class="la">([^<]*)</td>\s*<td[^>]*>\s*([\d.]+)\s*%')
_USER_AGENT = "nfl-projections/1.6 (+https://github.com/Gin-G/nfl-data-py)"


def fetch_html(week, *, site="fanduel", season=None, timeout=20, session=None):
    """The raw roster-percentage page for a week."""
    import requests

    params = {"site": site, "week": int(week)}
    if season is not None:
        params["season"] = int(season)
    getter = session.get if session is not None else requests.get
    response = getter(URL, params=params, timeout=timeout,
                      headers={"User-Agent": _USER_AGENT})
    response.raise_for_status()
    return response.text


def parse(html):
    """Pull the per-position tables out of the page.

    Returns a frame of name / position / ownership. An empty frame means the
    page rendered but holds no rows, which is what a week looks like before the
    numbers are published.
    """
    rows = []
    for table_id, position in TABLE_POSITIONS.items():
        block = re.search(r'id="data-%s"(.*?)</table>' % table_id, html, re.S)
        if block is None:
            continue
        for name, pct in _ROW.findall(block.group(1)):
            name = name.strip()
            if name:
                rows.append((name, position, float(pct)))
    frame = pd.DataFrame(rows, columns=["name", "position", "ownership"])
    if not frame.empty:
        frame["merge_name"] = [normalize_name(n) for n in frame["name"]]
    else:
        frame["merge_name"] = pd.Series(dtype=object)
    return frame


def check_totals(own, tolerance=0.25):
    """Per-position sums against what a full field has to add up to.

    A FanDuel classic lineup holds one QB, so every entry's QB ownership sums
    to 100%. If a position comes back at half its expected share the page has
    changed, or a table was missed, and the right move is to notice rather than
    to quietly optimise against a third of the field.
    """
    report = {}
    for position, expected in EXPECTED_SHARE.items():
        got = float(own.loc[own["position"] == position, "ownership"].sum())
        report[position] = {"total": got, "expected": expected,
                            "ok": abs(got - expected) <= tolerance * expected}
    return report


def load(week, *, site="fanduel", season=None, timeout=20, session=None,
         html=None, warn=True):
    """Fetch and parse one week, checking the totals look like a whole field."""
    if html is None:
        html = fetch_html(week, site=site, season=season, timeout=timeout,
                          session=session)
    own = parse(html)
    if own.empty:
        if warn:
            warnings.warn(
                f"No roster percentages published yet for week {week}; "
                "they appear on the Friday before the slate.",
                RuntimeWarning, stacklevel=2)
        return own
    if warn:
        bad = [p for p, r in check_totals(own).items() if not r["ok"]]
        if bad:
            warnings.warn(
                f"Roster percentages for {', '.join(bad)} do not sum to a full "
                "field; the page may have changed shape.",
                RuntimeWarning, stacklevel=2)
    return own


# A player the page does not list is NOT cheap leverage. The source covers one
# slate -- the Sunday 13:00-16:25 windows, which is the main slate -- and inside
# that slate an omission carries information we do not have: it means the field
# has already written the player off. On 2026 week 6, of the five main-slate
# players our board projected above 10 points and the page did not list, every
# single one was inactive, and our own `exp_games` had all five at 1.00. Across
# weeks 6 and 7 together it is 2 of 9 who played. So an unlisted player's
# ownership is unknown, not zero, and `attach` leaves it as NaN.
UNKNOWN = np.nan


def attach(pool, own, *, name_col=None, floor=None, warn=True):
    """Add an `ownership` column to a player pool, matched on normalized name.

    Unmatched players get NaN unless a `floor` is passed. `ownership_listed`
    records which is which, because a build that charges for ownership must be
    able to say how much of its pool the field was actually measured on.
    """
    pool = pool.copy()
    if name_col is None:
        name_col = "Nickname" if "Nickname" in pool.columns else "player_name"
    keys = [normalize_name(str(n)) for n in pool[name_col]]
    if own is None or own.empty:
        pool["ownership"] = np.nan
        pool["ownership_listed"] = False
        return pool
    lookup = dict(zip(own["merge_name"], own["ownership"]))
    values = [lookup.get(k) for k in keys]
    pool["ownership_listed"] = [v is not None for v in values]
    fill = UNKNOWN if floor is None else float(floor)
    pool["ownership"] = [fill if v is None else v for v in values]
    if warn:
        report = coverage(pool)
        if report["players"] and report["listed_share"] < 0.8:
            warnings.warn(
                f"only {report['listed_share']:.0%} of the {report['players']} "
                "players this pool projects above "
                f"{report['bar']:g} points have published ownership — the pool "
                "is probably a wider slate than the source projects, which is "
                "the Sunday 13:00-16:25 games only. Filter the pool to that "
                "slate first, or every team the source skips will read as "
                "contrarian.",
                RuntimeWarning, stacklevel=2)
    return pool


# Where to look for a projection when judging coverage, in order of preference.
_POINTS_COLUMNS = ("fanduel_fantasy_points", "lineup_points", "projected_points",
                   "FPPG")


def coverage(pool, *, points_col=None, bar=6.0):
    """How much of the pool the field was actually measured on.

    Judged on the players the board rates, not on the pool as a whole: a slate
    file holds hundreds of third-string names that no ownership source lists
    anywhere, so a raw match rate is near-meaningless. Judged the other way
    round -- what share of PUBLISHED names we found -- it misses the mismatch
    that matters, because a too-wide pool is a superset and matches all of
    them. On the real 2026 week 6 files that reading was 145 of 146 while a
    third of the board's playable names had no ownership at all.
    """
    if points_col is None:
        points_col = next((c for c in _POINTS_COLUMNS if c in pool.columns), None)
    if points_col is None:
        return {"players": 0, "listed": 0, "listed_share": float("nan"), "bar": bar}
    points = pd.to_numeric(pool[points_col], errors="coerce")
    rated = pool.loc[points >= bar]
    listed = int(rated["ownership_listed"].sum()) if len(rated) else 0
    return {"players": int(len(rated)), "listed": listed,
            "listed_share": (listed / len(rated)) if len(rated) else float("nan"),
            "bar": bar}


def flag_unlisted(pool, points_col="fanduel_fantasy_points", *, threshold=10.0):
    """Players the board likes that the field has not rostered at all.

    Almost always this means the field knows something the board does not —
    usually that he is not playing. Worth surfacing before a build, and worth
    checking rather than trusting in either direction: in 2026 week 7 Kirk
    Cousins was unlisted at a projected 18.4 because the field expected Penix
    to start, and he went for 20.4.
    """
    points = pd.to_numeric(pool[points_col], errors="coerce")
    hit = (~pool["ownership_listed"]) & (points >= threshold)
    return pool.loc[hit].assign(**{points_col: points[hit]}).sort_values(
        points_col, ascending=False)


def add_leverage(pool, points_col="fanduel_fantasy_points", *,
                 position_col=None, ownership_col="ownership"):
    """Where our board has a player, minus where the field has him.

    Both sides are within-position percentiles, so the comparison does not care
    that quarterbacks score more than tight ends or that running backs are
    rostered more than anyone. +1 is a player we rank top of his position and
    the field ranks bottom; -1 is the reverse. Players with unknown ownership
    get NaN leverage rather than a flattering number.
    """
    pool = pool.copy()
    if position_col is None:
        position_col = "Position" if "Position" in pool.columns else "position"
    points = pd.to_numeric(pool[points_col], errors="coerce")
    owned = pd.to_numeric(pool[ownership_col], errors="coerce")
    groups = pool[position_col].astype(str).str.upper().str.replace("/FLEX", "", regex=False)
    # `pct=True` on a single-row group gives 1.0, which is the right reading:
    # the only player at his position is both our favourite and the field's.
    pool["projection_pct"] = points.groupby(groups).rank(pct=True)
    pool["ownership_pct"] = owned.groupby(groups).rank(pct=True)
    pool["leverage"] = pool["projection_pct"] - pool["ownership_pct"]
    return pool
