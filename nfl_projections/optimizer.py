"""FanDuel lineup optimizer.

Works from a FanDuel main-slate player list CSV merged with our projections
(see merge_fanduel_salaries), building diverse lineups under the salary cap
with a per-player usage limit.
"""

import re
from collections import Counter

import numpy as np
import pandas as pd

ROSTER_SLOTS = {
    "QB": 1,
    "RB/FLEX": 2,
    "WR/FLEX": 3,
    "TE/FLEX": 1,
    "DEF": 1,
    "FLEX": 1,
}
FLEX_ELIGIBLE = ["RB/FLEX", "WR/FLEX", "TE/FLEX"]

# Which roster slots each position can fill. FanDuel's export has used two
# conventions for "Roster Position" over the years — bare positions ("RB") and
# flex-qualified ones ("RB/FLEX") — and matching one of them exactly produces NO
# lineups at all when the file uses the other, with nothing to say why. So both
# are parsed, and `Position` is the fallback when the roster column is unhelpful.
_POSITION_SLOTS = {
    "QB": ("QB",),
    "RB": ("RB/FLEX", "FLEX"),
    "FB": ("RB/FLEX", "FLEX"),
    "WR": ("WR/FLEX", "FLEX"),
    "TE": ("TE/FLEX", "FLEX"),
    "D": ("DEF",),
    "DEF": ("DEF",),
    "DST": ("DEF",),
    "D/ST": ("DEF",),
    "K": (),            # FanDuel's NFL roster has no kicker
}


def eligible_slots(roster_position, position=None) -> frozenset:
    """The roster slots a player can fill, from either CSV convention."""
    tokens = []
    for raw in (roster_position, position):
        if raw is None or raw != raw:        # None / NaN
            continue
        tokens = [t.strip().upper() for t in str(raw).split("/") if t.strip()]
        # "RB/FLEX" splits to RB + FLEX; "D/ST" is one position, not a flex.
        if str(raw).strip().upper() == "D/ST":
            tokens = ["D/ST"]
        if any(t in _POSITION_SLOTS for t in tokens):
            break
    slots = set()
    for token in tokens:
        slots.update(_POSITION_SLOTS.get(token, ()))
    return frozenset(slots)


def add_eligibility(df):
    """Add a `_slots` column of roster slots each row can fill (in place)."""
    df["_slots"] = [
        eligible_slots(rp, pos)
        for rp, pos in zip(df.get("Roster Position", pd.Series(index=df.index, dtype=object)),
                           df.get("Position", pd.Series(index=df.index, dtype=object)))
    ]
    return df


def slot_counts(df) -> dict:
    """How many rows can fill each slot — what to report when no lineup fits."""
    frame = df if "_slots" in df.columns else add_eligibility(df.copy())
    return {slot: int(sum(slot in s for s in frame["_slots"]))
            for slot in list(ROSTER_SLOTS) + ["FLEX"] if slot != "FLEX" or True}
EXCLUDED_INJURY_STATUSES = ["IR", "O", "D"]

# Which projection column drives lineup value. "mean" is the expected-points
# projection; floor/median/ceiling come from the quantile model (use ceiling for
# tournament upside, floor for cash-game safety).
OBJECTIVE_COLUMNS = {
    "mean": "fanduel_fantasy_points",
    "median": "projection_median",
    "floor": "floor",
    "ceiling": "ceiling",
}


def _objective_column(df, objective):
    """Resolve the objective to a present column, falling back to the mean."""
    col = OBJECTIVE_COLUMNS.get(objective, "fanduel_fantasy_points")
    if col not in df.columns:
        if objective not in (None, "mean"):
            print(f"'{objective}' column not found; using mean projection instead")
        return "fanduel_fantasy_points"
    return col


def normalize_name(name):
    """Normalize FanDuel nicknames for matching against our projections."""
    name = re.sub(r"\s+(Jr\.|Sr\.|I{2,}|IV)$", "", name)
    name = re.sub(r"\s+[A-Z]\.\s+", " ", name)  # middle initials
    return name.replace(".", "").lower().strip()


def merge_fanduel_salaries(fanduel_df, predictions_df, opportunity_context=None):
    """Join a FanDuel salary export with our projections by normalized name.

    Adds a 'value' column (projected points per $1000 of salary).

    `opportunity_context` is the frame from `nfl_projections.opportunity`, and
    it is **context only**: it adds columns beside `value` and never enters the
    objective. That boundary is measured, not cautious — opportunity share and
    volume-implied points are both null against the board's own error, because
    the board already carries volume as input features, and the one piece left
    over (`volume_td_gap`) buys 0.2 points of top-24 overlap in 4 weeks out of
    83. The optimizer reads nothing but the order, so it stays outside.
    """
    fanduel_df = fanduel_df.copy()
    predictions_df = predictions_df.copy()
    fanduel_df["Normalized_Name"] = fanduel_df["Nickname"].apply(normalize_name)
    predictions_df["Normalized_Name"] = predictions_df["player_name"].apply(normalize_name)

    merged = pd.merge(fanduel_df, predictions_df, on="Normalized_Name", how="left")
    merged = merged.drop(columns=["Normalized_Name"])
    merged["value"] = (merged["fanduel_fantasy_points"] / merged["Salary"]) * 1000
    if opportunity_context is not None:
        from . import opportunity

        merged = opportunity.attach(merged, opportunity_context)
    return merged


def _get_top_n_by_position(df, position, n=5):
    return df[[position in s for s in df["_slots"]]].nlargest(n, "lineup_points")


def _is_lineup_unique(new_lineup, existing_lineups):
    new_set = set(new_lineup["Nickname"].values)
    return all(set(l["Nickname"].values) != new_set for l in existing_lineups)


def calculate_player_usage(lineups):
    """Usage counts and percentages for players across generated lineups."""
    all_players = []
    for lineup in lineups:
        all_players.extend(lineup["Nickname"].tolist())
    counts = Counter(all_players)

    usage = pd.DataFrame.from_dict(counts, orient="index", columns=["Count"])
    usage["Usage_Percentage"] = (usage["Count"] / len(lineups) * 100).round(2)
    return usage.sort_values("Usage_Percentage", ascending=False)


def _check_usage_limit(nickname, current_lineups, num_lineups, max_usage_percentage):
    if not current_lineups:
        return True
    current = sum(1 for l in current_lineups if nickname in l["Nickname"].values)
    return (current + 1) / num_lineups * 100 <= max_usage_percentage


def _pareto(frontier):
    """Drop entries a cheaper-and-better one dominates.

    Safe for combining: if A costs less than B and scores more, swapping B for A
    keeps every budget feasible and raises the total, so B cannot be part of any
    optimal lineup. On a real slate this collapses the table hard, which is what
    keeps the exact solve fast enough to sit inside a request.
    """
    out, best = {}, float("-inf")
    for cost in sorted(frontier):
        points, picks = frontier[cost]
        if points > best:
            out[cost] = (points, picks)
            best = points
    return out


def _slot_frontier(pool, k, budget, step):
    """{cost_buckets: (points, picks)} for EXACTLY `k` players from `pool`.

    The whole cost profile, not just the best under one cap — the profile is
    what lets slots be combined exactly, because the cheapest way to fill one
    slot is not generally part of the best way to fill the roster.
    """
    if k <= 0:
        return {0: (0.0, ())}
    rows = [(idx, int(row["Salary"] // step), float(row["lineup_points"]))
            for idx, row in pool.iterrows()
            if row["Salary"] == row["Salary"] and row["Salary"] >= 0]
    if len(rows) < k:
        return {}
    best = [dict() for _ in range(k + 1)]
    best[0][0] = (0.0, ())
    for idx, cost, points in rows:
        for count in range(k - 1, -1, -1):
            for spent, (total, picks) in list(best[count].items()):
                new_spent = spent + cost
                if new_spent > budget:
                    continue
                current = best[count + 1].get(new_spent)
                if current is None or total + points > current[0]:
                    best[count + 1][new_spent] = (total + points, picks + (idx,))
    return _pareto(best[k])


def _combine(a, b, budget):
    """Max-plus convolution of two cost profiles under a shared budget."""
    out = {}
    if not a or not b:
        return {}
    bs = sorted(b)
    for cost_a, (points_a, picks_a) in a.items():
        for cost_b in bs:
            total_cost = cost_a + cost_b
            if total_cost > budget:
                break               # bs ascending, so nothing further fits
            points_b, picks_b = b[cost_b]
            current = out.get(total_cost)
            if current is None or points_a + points_b > current[0]:
                out[total_cost] = (points_a + points_b, picks_a + picks_b)
    return _pareto(out)


def _flex_expansions(roster_slots):
    """Every way to turn the FLEX units into concrete slots.

    After the expansion the slot pools are disjoint by position, which is what
    makes the rest exact: QB, RB, WR, TE and DEF selections then interact only
    through the shared salary budget.
    """
    import itertools

    flex = int(roster_slots.get("FLEX", 0))
    base = {slot: int(n) for slot, n in roster_slots.items() if slot != "FLEX"}
    targets = [slot for slot in base if slot in FLEX_ELIGIBLE]
    if not flex or not targets:
        yield base
        return
    seen = set()
    for combo in itertools.combinations_with_replacement(targets, flex):
        need = dict(base)
        for slot in combo:
            need[slot] += 1
        key = tuple(sorted(need.items()))
        if key not in seen:
            seen.add(key)
            yield need


def best_lineup(df, cap, roster_slots=None, salary_step=100):
    """The highest-scoring legal lineup at or under `cap`, or None if none fits.

    Exact, where the old fill was greedy. It filled slot by slot taking the best
    points-per-dollar player that fit the salary left, with no budget held back
    for the slots still to come and no way to back up when one came up empty --
    so it reported "the salary cap or the exclusions are what nothing fits
    under" for slates where plenty fits. A pool with expensive players who also
    have the best value is enough to trigger it: the early slots eat the budget,
    the tight end slot finds nothing under what is left, and the attempt is
    abandoned rather than retried a little cheaper.

    Returns (indices, salary, points).
    """
    roster_slots = roster_slots or ROSTER_SLOTS
    budget = int(cap // salary_step)
    if "_slots" not in df.columns:
        add_eligibility(df)
    best = None
    for need in _flex_expansions(roster_slots):
        # Hardest slot first: it prunes the table soonest.
        order = sorted(need, key=lambda s: -need[s])
        profile = {0: (0.0, ())}
        for slot in order:
            pool = df[[slot in s for s in df["_slots"]]]
            frontier = _slot_frontier(pool, need[slot], budget, salary_step)
            profile = _combine(profile, frontier, budget)
            if not profile:
                break
        if not profile:
            continue
        cost, (points, picks) = max(profile.items(), key=lambda kv: kv[1][0])
        if best is None or points > best[2]:
            best = (list(picks), cost * salary_step, points)
    return best


def cheapest_lineup_cost(df, roster_slots=None):
    """What the cheapest legal lineup on this slate costs, or None if the slate
    cannot field one at any price.

    So that "nothing fits under the cap" can say by how much. After a FLEX
    expansion the slot pools are disjoint, so the minimum is just the cheapest
    `n` players in each slot summed -- no search needed.
    """
    roster_slots = roster_slots or ROSTER_SLOTS
    if "_slots" not in df.columns:
        df = add_eligibility(df.copy())
    best = None
    for need in _flex_expansions(roster_slots):
        total = 0
        for slot, n in need.items():
            if n <= 0:
                continue
            pool = df[[slot in sl for sl in df["_slots"]]]
            salaries = sorted(pd.to_numeric(pool["Salary"], errors="coerce").dropna())
            if len(salaries) < n:
                total = None
                break
            total += sum(salaries[:n])
        if total is not None and (best is None or total < best):
            best = total
    return int(best) if best is not None else None


def optimize_lineups(df, num_lineups=5, salary_cap=60000, exclude_players=None,
                     max_usage_percentage=50, objective="mean", roster_slots=None,
                     salary_step=100, price_step=None):
    """Build FanDuel lineups: the best one under the cap, then cheaper ones.

    Args:
        df: output of merge_fanduel_salaries (needs Salary, FPPG,
            Roster Position, Nickname, fanduel_fantasy_points columns)
        num_lineups: how many lineups to generate
        salary_cap: FanDuel salary cap — a hard constraint on every lineup
        exclude_players: nicknames to leave out entirely
        max_usage_percentage: cap on how often one player appears
        objective: which projection drives value - "mean" (default),
            "ceiling" (GPP upside), "floor" (cash safety), or "median".
            Non-mean objectives need a projection made with the quantile model.

    Each lineup is the **highest-scoring** one that fits its budget, solved
    exactly (`best_lineup`). Lineup 1 gets the whole cap; each one after it is
    capped `price_step` below what the previous lineup actually spent, so the
    set walks down the price range instead of returning variations on the same
    expensive core. That is also what makes them diverse: a strictly smaller
    budget cannot return the same roster.

    `price_step` defaults to 1% of the cap ($600 of $60,000). Stepping by the
    $100 salary granularity instead is technically enough to force a different
    roster, but it buys a one-player swap -- the first three lineups came back
    at $58,800, $58,700 and $57,200 with eight names in common. A step with
    some width in it is what makes the cheaper lineups actually cheaper.

    The old builder filled greedily by points per dollar and could neither hold
    budget back for the slots still to come nor back up when one came up empty,
    so it returned nothing at all on slates where plenty fits. It also left
    money on the cap for the reason `_best_under_cap` documents: cheap and
    efficient beats expensive and better on a per-dollar measure every time,
    and a lineup is scored on points.

    Returns a list of lineup DataFrames, most expensive first.
    """
    roster_slots = roster_slots or ROSTER_SLOTS
    df = df.copy()

    df["Injury Indicator"] = df.get("Injury Indicator", pd.Series("", index=df.index)).fillna("")
    df = df[~df["Injury Indicator"].str.upper().isin(EXCLUDED_INJURY_STATUSES)]
    df = df.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["Salary", "FPPG", "Roster Position", "Nickname"]
    )
    if exclude_players:
        df = df[~df["Nickname"].isin(set(exclude_players))]

    # Optimize toward the chosen projection; DEF has no model projection so it
    # always falls back to FanDuel's FPPG.
    proj_col = _objective_column(df, objective)
    df["lineup_points"] = df[proj_col].fillna(df["fanduel_fantasy_points"]).fillna(0)
    add_eligibility(df)
    is_defense = [("DEF" in s) for s in df["_slots"]]
    df.loc[is_defense, "lineup_points"] = df.loc[is_defense, "FPPG"]

    step = int(price_step) if price_step else max(salary_step, salary_cap // 100)
    all_lineups = []
    budget = salary_cap
    while len(all_lineups) < num_lineups and budget >= 0:
        # Players already at their exposure limit are dropped from the pool
        # rather than rejected after the fact, so the solve stays exact over
        # whoever is actually still available.
        pool = df[[_check_usage_limit(n, all_lineups, num_lineups, max_usage_percentage)
                   for n in df["Nickname"]]]
        found = best_lineup(pool, budget, roster_slots, salary_step)
        if found is None:
            break
        picks, spent, _ = found
        lineup_df = pool.loc[picks]
        if _is_lineup_unique(lineup_df, all_lineups):
            all_lineups.append(lineup_df)
            print(f"Created lineup {len(all_lineups)} (${spent:,})")
        # Next one has to be cheaper, which is both the variation that was asked
        # for and the reason this terminates.
        budget = spent - step

    if len(all_lineups) < num_lineups:
        print(f"Warning: only generated {len(all_lineups)}/{num_lineups} lineups "
              f"under a ${salary_cap:,} cap")
    return all_lineups


def optimize_from_csv(csv_file, num_lineups=5, salary_cap=60000, exclude_players=None,
                      max_usage_percentage=50, objective="mean"):
    """Optimize lineups from a merged salary+projection CSV file."""
    df = pd.read_csv(csv_file)
    return optimize_lineups(df, num_lineups=num_lineups, salary_cap=salary_cap,
                            exclude_players=exclude_players,
                            max_usage_percentage=max_usage_percentage,
                            objective=objective)


def display_lineups(lineups):
    """Print each lineup with salary and projected point totals."""
    if not lineups:
        print("No valid lineups found.")
        return
    for i, lineup in enumerate(lineups, 1):
        print(f"\nLineup {i}:")
        cols = [c for c in ["Roster Position", "Nickname", "Salary", "lineup_points",
                            "floor", "ceiling", "opp_per_game", "opp_share",
                            "volume_td_gap", "Injury Indicator"]
                if c in lineup.columns]
        print(lineup[cols])
        print(f"Total Salary: ${lineup['Salary'].sum():,.0f}")
        print(f"Projected Points: {lineup['lineup_points'].sum():.2f}")
        if {"floor", "ceiling"} <= set(lineup.columns):
            print(f"Range: floor {lineup['floor'].sum():.1f} - "
                  f"ceiling {lineup['ceiling'].sum():.1f}")

    print("\nTop 20 most used players:")
    print(calculate_player_usage(lineups).head(20))


# ── Showdown (single game) ────────────────────────────────────────────────────
#
# A different contest, not a variant of the classic one: five players from ONE
# game, with one named MVP who scores 1.5x and costs 1.5x. FanDuel's export says
# so in two places — every row's Roster Position reads "MVP - 1.5X
# Points/AnyFLEX", and there is an extra "MVP 1.5x Salary" column — and no
# classic slot appears anywhere in the file, which is why running it through the
# classic optimizer yields nothing at all.
SHOWDOWN_ROSTER_SIZE = 5
MVP_MULTIPLIER = 1.5
MVP_SALARY_COLUMN = "MVP 1.5x Salary"


def is_showdown(df) -> bool:
    """Whether a FanDuel export is a single-game Showdown slate."""
    if MVP_SALARY_COLUMN in df.columns:
        return True
    if "Roster Position" not in df.columns:
        return False
    values = df["Roster Position"].dropna().astype(str).str.upper()
    return bool(len(values)) and values.str.contains("MVP|ANYFLEX").all()


def _best_under_cap(pool, k, cap, salary_step=100, position_limits=None):
    """Indices of exactly ``k`` rows maximising ``lineup_points`` with salaries
    summing to no more than ``cap`` — a cardinality-constrained knapsack, solved
    exactly.

    Filling by points-per-dollar instead leaves money on the table: on a real
    Showdown pool it spent $38k of a $60k cap and rostered both kickers, because
    cheap-and-efficient beats expensive-and-better on that measure every time. A
    lineup is scored on points, not on value, so the cap should be spent.

    Salaries are bucketed to ``salary_step`` to keep the table small; FanDuel
    prices in hundreds, so nothing is lost.

    ``position_limits`` ({"K": 1} and so on) caps how many of a position may
    appear. It is enforced by solving, then dropping the weakest offender and
    solving again — near-exact rather than exact, which is the right trade for
    a rule that exists to stop a lineup rostering two kickers because FanDuel's
    season-average FPPG makes both look efficient.
    """
    limits = position_limits or {}
    pool = pool.copy()
    if k <= 0:
        return None
    budget = int(cap // salary_step)
    dropped: set = set()

    while True:
        rows = [(idx, int(row["Salary"] // salary_step), float(row["lineup_points"]))
                for idx, row in pool.iterrows()
                if idx not in dropped and row["Salary"] == row["Salary"]
                and row["Salary"] <= cap]
        if len(rows) < k:
            return None

        # best[c][s] = (points, picks) for exactly c players costing s buckets
        best = [dict() for _ in range(k + 1)]
        best[0][0] = (0.0, ())
        for idx, cost, points in rows:
            for count in range(k - 1, -1, -1):
                for spent, (total, picks) in list(best[count].items()):
                    new_spent = spent + cost
                    if new_spent > budget:
                        continue
                    current = best[count + 1].get(new_spent)
                    if current is None or total + points > current[0]:
                        best[count + 1][new_spent] = (total + points, picks + (idx,))
        if not best[k]:
            return None
        picks = max(best[k].values(), key=lambda v: v[0])[1]
        if not limits:
            return picks

        over = None
        for position, limit in limits.items():
            same = [i for i in picks
                    if str(pool.loc[i].get("Position", "")).upper() == position.upper()]
            if len(same) > limit:
                over = min(same, key=lambda i: pool.loc[i]["lineup_points"])
                break
        if over is None:
            return picks
        dropped.add(over)


def optimize_showdown(df, num_lineups=5, salary_cap=60000, exclude_players=None,
                      max_usage_percentage=50, objective="mean",
                      roster_size=SHOWDOWN_ROSTER_SIZE, position_limits=None):
    """Build Showdown lineups: one MVP at 1.5x points and 1.5x salary, plus
    ``roster_size - 1`` others from the same game.

    Kickers and defenses are eligible here, unlike the classic slate, and both
    appear in a real Showdown pool. Neither is projected by the model, so both
    fall back to FanDuel's FPPG the same way classic defenses do — which also
    makes them look cheap and reliable, so ``position_limits`` (e.g. {"K": 1})
    is how a caller stops a lineup taking both kickers in a game.

    Each MVP candidate is tried in turn and the rest of the lineup filled by
    points per dollar — the same greedy shape as the classic optimizer, which
    keeps the two reading alike.
    """
    df = df.copy()
    df["Injury Indicator"] = df.get(
        "Injury Indicator", pd.Series("", index=df.index)).fillna("")
    df = df[~df["Injury Indicator"].str.upper().isin(EXCLUDED_INJURY_STATUSES)]
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["Salary", "FPPG", "Nickname"])
    if exclude_players:
        df = df[~df["Nickname"].isin(set(exclude_players))]
    if df.empty:
        return []

    proj_col = _objective_column(df, objective)
    df["lineup_points"] = df[proj_col].fillna(df["fanduel_fantasy_points"]).fillna(df["FPPG"])
    # Positions the model never projects — kickers and defenses — keep FanDuel's.
    unprojected = df["fanduel_fantasy_points"].isna()
    df.loc[unprojected, "lineup_points"] = df.loc[unprojected, "FPPG"]
    df["mvp_salary"] = (df[MVP_SALARY_COLUMN] if MVP_SALARY_COLUMN in df.columns
                        else df["Salary"] * MVP_MULTIPLIER)

    lineups = []
    # Best MVP candidates first: the 1.5x applies to their points, so the
    # ordering that matters is points, not value.
    for _, mvp in df.sort_values("lineup_points", ascending=False).iterrows():
        if len(lineups) >= num_lineups:
            break
        if not _check_usage_limit(mvp["Nickname"], lineups, num_lineups, max_usage_percentage):
            continue
        remaining = salary_cap - mvp["mvp_salary"]
        if remaining < 0:
            continue

        pool = df[df["Nickname"] != mvp["Nickname"]]
        pool = pool[[_check_usage_limit(n, lineups, num_lineups, max_usage_percentage)
                     for n in pool["Nickname"]]]
        best = _best_under_cap(pool, roster_size - 1, remaining,
                               position_limits=position_limits)
        if best is None:
            continue
        picked = [mvp] + [pool.loc[i] for i in best]
        lineup = pd.DataFrame(picked)
        # MVP first, and his row carries the 1.5x on both sides.
        lineup.loc[lineup.index[0], "lineup_points"] = mvp["lineup_points"] * MVP_MULTIPLIER
        lineup.loc[lineup.index[0], "Salary"] = mvp["mvp_salary"]
        lineup["Roster Position"] = ["MVP"] + ["FLEX"] * (len(lineup) - 1)
        if _is_lineup_unique(lineup, lineups):
            lineups.append(lineup)

    return lineups
