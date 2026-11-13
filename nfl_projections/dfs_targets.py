"""Group a published board into price tiers a human can read down.

The shape is borrowed from how a DFS columnist writes a slate up: split each
position into price bands, order inside a band, and give every name a sentence
saying why it is there and what would change it. That shape is good because it
answers the question actually being asked at build time -- "who is worth this
money" -- which a single 500-row board sorted by points does not.

What is borrowed is only the shape. Everything that orders a player here comes
from our own board:

    value = projected points / (salary / 1000)

and nothing else. In particular:

**No Vegas.** Implied team totals and spreads are the spine of most published
write-ups and they are not an input here, by standing rule: a line may be a
benchmark but never a regressor in an edge-hunting pipeline. A slate read that
ranks by implied total is largely re-reading the market.

**No matchup in the ordering.** Opponent strength, coverage grades, shell mix
and any third-party matchup score ride along as CONTEXT, in their own block,
flagged as such. Our own residual tests found WR coverage matchups actively
harmful and the alt-line backtest found matchup added nothing over role and
cushion, so letting them tilt an ordering would be re-importing something that
has already failed. They are here because a person reading a slate uses them
to break ties and to decide what to look up, which is not the same as a model
using them to score.

**No manual overrides.** There is no list of names in this file. A player
moves because the board moved.

The tiers are price bands, not quality bands: "premium" means expensive, and
an expensive player in the wrong spot is exactly what the value column is for.
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)

# FanDuel classic bands. A published write-up usually tiers on DraftKings,
# where prices sit lower; these are the FanDuel equivalents and are overridable
# because FanDuel moves its own pricing between seasons.
PREMIUM_MIN = 7500
VALUE_MAX = 6000

TIER_ORDER = ("premium", "mid", "value")

# FanDuel's own designations for a player who cannot be rostered. The
# optimizer already refuses these; a targets list that still showed them would
# be recommending someone the builder cannot use.
UNROSTERABLE = ("IR", "O", "D")

# FanDuel spells the defence "D" in its Position column while calling the
# roster slot "DEF". Both are accepted so a caller can ask for either.
POSITIONS = ("QB", "RB", "WR", "TE", "D")
POSITION_ALIASES = {"DEF": "D", "DST": "D", "D/ST": "D"}


def canonical_positions(wanted) -> tuple:
    """Caller-supplied position names as the slate spells them."""
    return tuple(dict.fromkeys(
        POSITION_ALIASES.get(str(p).strip().upper(), str(p).strip().upper())
        for p in wanted if str(p).strip()))

# How thin a sample has to be before a projection is worth a warning rather
# than a recommendation. Mirrors the band the coverage work uses.
THIN_SNAPS = 4.0


def tier_of(salary, premium_min: int = PREMIUM_MIN, value_max: int = VALUE_MAX):
    """Which price band a salary falls in, or None when there is no salary."""
    if salary is None or pd.isna(salary):
        return None
    salary = float(salary)
    if salary >= premium_min:
        return "premium"
    if salary < value_max:
        return "value"
    return "mid"


def _clean_flag(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip().upper()


def merge_board_and_slate(board: pd.DataFrame, slate: pd.DataFrame,
                          name_key) -> pd.DataFrame:
    """Join a published board onto a FanDuel slate on normalised names.

    The slate is the left side on purpose: it is the list of people who can
    actually be rostered, and a board row with no salary is not a target. The
    reverse join would quietly offer players who are not in the contest.
    """
    if slate is None or len(slate) == 0:
        return pd.DataFrame()
    s = slate.copy()
    s.columns = [str(c).strip() for c in s.columns]
    s["_key"] = s["Nickname"].map(name_key)
    out = s
    if board is not None and len(board):
        # The one slate matcher (names.match_slate): name + team first, so a
        # name two players share cannot hand one of them the other's line.
        from .names import attach_board

        keep = [c for c in ("projected_points", "floor", "median", "ceiling",
                            "position", "team", "player_id") if c in board.columns]
        out = attach_board(s, board.reset_index(drop=True), keep)
    return out


def build(board: pd.DataFrame, slate: pd.DataFrame, *, name_key,
          premium_min: int = PREMIUM_MIN, value_max: int = VALUE_MAX,
          positions=POSITIONS, history=None) -> pd.DataFrame:
    """One row per rosterable player, with tier, value and ranks attached.

    `history` is each player's own weekly results, attached for reading only --
    it reaches `summarize` and nothing else. See `attach_history` for why it is
    not allowed near the ordering.
    """
    merged = merge_board_and_slate(board, slate, name_key)
    if merged.empty:
        return merged
    merged = attach_history(merged, history, name_key=name_key)

    merged["injury_flag"] = merged.get(
        "Injury Indicator", pd.Series("", index=merged.index)).map(_clean_flag)
    merged["rosterable"] = ~merged["injury_flag"].isin(UNROSTERABLE)

    merged["salary"] = pd.to_numeric(merged.get("Salary"), errors="coerce")
    merged["fppg"] = pd.to_numeric(merged.get("FPPG"), errors="coerce")
    merged["games_played"] = pd.to_numeric(merged.get("Played"), errors="coerce")
    merged["slate_position"] = merged.get("Position")
    merged["opponent"] = merged.get("Opponent")
    merged["game"] = merged.get("Game")

    proj = pd.to_numeric(merged.get("projected_points"), errors="coerce")
    merged["projected_points"] = proj
    merged["value"] = (proj / (merged["salary"] / 1000.0)).round(3)
    # FanDuel's own season average, on the same scale, so a board that has
    # drifted from the player's actual production is visible rather than
    # implied.
    merged["fppg_value"] = (merged["fppg"] / (merged["salary"] / 1000.0)).round(3)
    merged["value_vs_fppg"] = (merged["value"] - merged["fppg_value"]).round(3)

    merged["tier"] = merged["salary"].map(
        lambda s: tier_of(s, premium_min, value_max))

    out = merged[merged["slate_position"].isin(positions)
                 & merged["rosterable"]
                 & merged["value"].notna()].copy()
    if out.empty:
        return out

    out["position_rank"] = out.groupby("slate_position")["value"].rank(
        ascending=False, method="min").astype(int)
    out["tier_rank"] = out.groupby(["slate_position", "tier"])["value"].rank(
        ascending=False, method="min").astype(int)
    out["points_rank"] = out.groupby("slate_position")["projected_points"].rank(
        ascending=False, method="min").astype(int)
    return out.sort_values(["slate_position", "tier", "value"],
                           ascending=[True, True, False])


#: Below this many games a range is one or two outings, not a range.
HISTORY_MIN_GAMES = 3

#: Columns `attach_history` adds, so a caller can tell "no history supplied"
#: from "history supplied and this player has none".
HISTORY_COLUMNS = ("hist_games", "hist_low", "hist_high", "hist_median")


def attach_history(frame: pd.DataFrame, history, *, name_key) -> pd.DataFrame:
    """Add each player's own weekly range: games, low, high, median.

    **Description, not prediction.** Tested on 2026 weeks 5-9, n=617
    player-weeks: a band of the lowest and highest a player has scored covers
    71.0% of his next result against the model band's 81.7%, and at a matched
    width 67.7% against 73.7% — because a min and max over *n* games is not a
    range, it is an unstable estimate of the 1/(n+1) and n/(n+1) quantiles. At
    five prior games "the lowest he has scored" is his 17th percentile and gets
    undercut 16% of the time, and both ends drift outward all season as *n*
    grows rather than as the player changes.

    So this never feeds `value`, a tier or a rank, and `floor`/`ceiling` stay
    the model's. What a range is genuinely good for is being read: it says in
    one line what the player has actually done, which a projection and a band
    both hide. See EXPERIMENTS.md "a player's own high and low make a worse
    band than the model's".

    `history` is a long frame, one row per player-week, carrying a name column
    (`player_name`, `Nickname` or `name`) and a points column
    (`fantasy_points`, `actual_points`, `points` or `fd_points`).
    `name_key` is the same normaliser the board and slate were joined on, so a
    history that spells a name FanDuel's way still lands on the right row.
    """
    frame = frame.copy()
    if history is None or len(history) == 0:
        return frame
    points_column = next(
        (c for c in ("fantasy_points", "actual_points", "points", "fd_points")
         if c in history.columns), None)
    if points_column is None:
        raise ValueError(
            "history needs a points column, one of fantasy_points, "
            f"actual_points, points or fd_points; it has {list(history.columns)}")
    name_column = next((c for c in ("player_name", "Nickname", "name")
                        if c in history.columns), None)
    if name_column is None:
        raise ValueError(
            "history needs a name column, one of player_name, Nickname or "
            f"name; it has {list(history.columns)}")

    if "_key" not in frame.columns:
        raise ValueError("attach_history needs the joined frame from "
                         "merge_board_and_slate, which carries `_key`")
    # By player id wherever both sides have one: by name, the Browns'
    # linebacker Justin Jefferson's scoreless weeks were the receiver's too.
    by_id = "player_id" in history.columns and "player_id" in frame.columns
    rows = pd.DataFrame({
        "_key": (history["player_id"].astype("object") if by_id
                 else history[name_column].map(name_key)),
        "_p": pd.to_numeric(history[points_column], errors="coerce"),
    }).dropna()
    stats = rows.groupby("_key")["_p"].agg(
        hist_games="size", hist_low="min", hist_high="max", hist_median="median")
    if by_id:
        merged = frame.merge(stats.rename_axis("_hid").reset_index(),
                             left_on="player_id", right_on="_hid", how="left"
                             ).drop(columns="_hid")
    else:
        merged = frame.merge(stats.reset_index(), on="_key", how="left")
    merged.index = frame.index
    return merged


def _spread(row) -> float | None:
    """Ceiling minus floor, the board's own uncertainty about a player."""
    lo, hi = row.get("floor"), row.get("ceiling")
    if lo is None or hi is None or pd.isna(lo) or pd.isna(hi):
        return None
    return round(float(hi) - float(lo), 1)


def summarize(row, context: dict | None = None) -> str:
    """One sentence about a player, built only from numbers we hold.

    Deliberately plain: it states the board's number, what that is worth at
    the price, how the two compare with the player's own season, and the one
    thing most likely to make it wrong. No adjectives that imply a confidence
    the board has not earned.
    """
    context = context or {}
    name = row.get("Nickname", "This player")
    pos = row.get("slate_position", "")
    salary = row.get("salary")
    proj = row.get("projected_points")
    value = row.get("value")
    tier = row.get("tier", "")
    tier_rank = row.get("tier_rank")
    pos_rank = row.get("position_rank")

    bits = [
        f"We project {proj:.1f} at ${int(salary):,}"
        f" — {value:.2f} points per $1k, {_ordinal(tier_rank)} among"
        f" {tier} {pos}s and {_ordinal(pos_rank)} at the position."
    ]

    spread = _spread(row)
    if spread is not None:
        bits.append(f"Floor {row['floor']:.1f} to ceiling {row['ceiling']:.1f}"
                    f" (a {spread:.1f}-point spread).")

    # What he has actually done, which neither the projection nor the band
    # shows. Phrased as a record rather than a range he will stay inside,
    # because measured as a band it is worse than the model's -- see
    # `attach_history`. The projection sitting outside it is worth saying
    # outright: it is the board claiming something new about him.
    games, low = row.get("hist_games"), row.get("hist_low")
    high, mid = row.get("hist_high"), row.get("hist_median")
    if (games is not None and not pd.isna(games) and games >= HISTORY_MIN_GAMES
            and low is not None and not pd.isna(low)
            and high is not None and not pd.isna(high)):
        line = (f"Has scored {float(low):.1f} to {float(high):.1f} in"
                f" {int(games)} games")
        if mid is not None and not pd.isna(mid):
            line += f", median {float(mid):.1f}"
        line += "."
        if proj is not None and not pd.isna(proj):
            if proj > float(high):
                line += " We project above anything done this season."
            elif proj < float(low):
                line += " We project below anything done this season."
        bits.append(line)

    fppg, vvf = row.get("fppg"), row.get("value_vs_fppg")
    played = row.get("games_played")
    if fppg is not None and not pd.isna(fppg) and vvf is not None and not pd.isna(vvf):
        if played is not None and not pd.isna(played) and played < THIN_SNAPS:
            bits.append(f"Only {int(played)} games played, so his {fppg:.1f}"
                        " season average is thin.")
        elif abs(vvf) >= 0.25:
            direction = "above" if vvf > 0 else "below"
            bits.append(f"That is {abs(vvf):.2f} per $1k {direction} his own"
                        f" season average of {fppg:.1f}.")

    flag = row.get("injury_flag") or ""
    if flag == "Q":
        bits.append("FanDuel lists him questionable, and the board does not"
                    " discount for that.")

    opp = context.get("opponent_fpa")
    if opp and opp.get("band"):
        bits.append(f"Context only: we grade {row.get('opponent')} {opp['band']}"
                    f" against {pos}s ({opp['grade']:.0f}/100,"
                    f" {_ordinal(opp.get('rank'))} softest) after adjusting for"
                    " the schedule they have faced.")
    return " ".join(bits)


def _ordinal(n) -> str:
    if n is None or (isinstance(n, float) and pd.isna(n)):
        return "unranked"
    n = int(n)
    if 10 <= n % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"
