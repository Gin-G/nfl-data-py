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
        b = board.copy()
        b["_key"] = b["player_name"].map(name_key)
        keep = [c for c in ("_key", "projected_points", "floor", "median",
                            "ceiling", "position", "team", "player_id")
                if c in b.columns]
        # Two rows for one name would duplicate a slate entry; the board is
        # one row per player, so this only guards a bad upstream join.
        out = s.merge(b[keep].drop_duplicates("_key"), on="_key", how="left",
                      suffixes=("", "_board"))
    return out


def build(board: pd.DataFrame, slate: pd.DataFrame, *, name_key,
          premium_min: int = PREMIUM_MIN, value_max: int = VALUE_MAX,
          positions=POSITIONS) -> pd.DataFrame:
    """One row per rosterable player, with tier, value and ranks attached."""
    merged = merge_board_and_slate(board, slate, name_key)
    if merged.empty:
        return merged

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
