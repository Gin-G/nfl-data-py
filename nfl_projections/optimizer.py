"""FanDuel lineup optimizer.

Works from a FanDuel main-slate player list CSV merged with our projections
(see merge_fanduel_salaries), building diverse lineups under the salary cap
with a per-player usage limit.
"""

import re
import warnings
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


def _prune_dominated(rows, k):
    """Drop players that cannot be in any optimal choice of `k` from `rows`.

    A player is useless if `k` others are all at least as cheap and at least as
    good: at most k-1 of them can share a roster with him, so one is always
    free to take his place at no cost and no loss. Sorting by salary and then
    by points descending makes "at least as cheap" mean "earlier", and the
    sort order breaks ties so that identical players form a chain instead of
    eliminating each other.

    This is pure speed — the answer is unchanged — but it is a lot of speed:
    the frontier DP is linear in the pool and a main-slate WR pool is ~180 deep
    behind maybe 40 players that can matter.
    """
    if k <= 0 or len(rows) <= k:
        return rows
    ordered = sorted(rows, key=lambda r: (r[1], -r[2]))
    kept, points_seen = [], []
    for row in ordered:
        if sum(1 for p in points_seen if p >= row[2]) < k:
            kept.append(row)
            points_seen.append(row[2])
    return kept


def _rows_for(pool, step):
    """(index, cost in salary buckets, points) for a slot pool."""
    return [(idx, int(row["Salary"] // step), float(row["lineup_points"]))
            for idx, row in pool.iterrows()
            if row["Salary"] == row["Salary"] and row["Salary"] >= 0]


def _frontiers(rows, kmax, budget):
    """{k: {cost_buckets: (points, picks)}} for every k up to `kmax`.

    The whole cost profile, not just the best under one cap — the profile is
    what lets slots be combined exactly, because the cheapest way to fill one
    slot is not generally part of the best way to fill the roster. Every count
    comes out of the one pass because the correlation rules need "two from this
    team and one from everyone else" as well as the plain total.
    """
    rows = _prune_dominated(rows, kmax)
    best = [dict() for _ in range(kmax + 1)]
    best[0][0] = (0.0, ())
    for idx, cost, points in rows:
        for count in range(min(kmax, len(best) - 2), -1, -1):
            for spent, (total, picks) in list(best[count].items()):
                new_spent = spent + cost
                if new_spent > budget:
                    continue
                current = best[count + 1].get(new_spent)
                if current is None or total + points > current[0]:
                    best[count + 1][new_spent] = (total + points, picks + (idx,))
    return {k: _pareto(best[k]) for k in range(kmax + 1)}


def _slot_frontier(pool, k, budget, step):
    """{cost_buckets: (points, picks)} for EXACTLY `k` players from `pool`."""
    if k <= 0:
        return {0: (0.0, ())}
    rows = _rows_for(pool, step)
    if len(rows) < k:
        return {}
    return _frontiers(rows, k, budget)[k]


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


PASS_CATCHER_SLOTS = ("WR/FLEX", "TE/FLEX")


def _plain_expansion(df, need, budget, step):
    """Best lineup for one concrete slot shape, with no correlation rules."""
    profile = {0: (0.0, ())}
    # Hardest slot first: it prunes the table soonest.
    for slot in sorted(need, key=lambda s: -need[s]):
        pool = df[[slot in s for s in df["_slots"]]]
        profile = _combine(profile, _slot_frontier(pool, need[slot], budget, step), budget)
        if not profile:
            return None
    cost, (points, picks) = max(profile.items(), key=lambda kv: kv[1][0])
    return list(picks), cost, points


def _prefix_best(profile):
    """(costs ascending, best points at or under each cost) for fast lookup."""
    costs = sorted(profile)
    running, out = float("-inf"), []
    for cost in costs:
        running = max(running, profile[cost][0])
        out.append(running)
    return costs, out


def _best_at_or_under(costs, prefix, budget):
    """The best points reachable spending no more than `budget`."""
    import bisect

    at = bisect.bisect_right(costs, budget) - 1
    return prefix[at] if at >= 0 else float("-inf")


def _merge_profiles(a, b):
    """The better of two cost profiles at every price."""
    if not a:
        return dict(b)
    out = dict(a)
    for cost, entry in b.items():
        current = out.get(cost)
        if current is None or entry[0] > current[0]:
            out[cost] = entry
    return _pareto(out)


def _compositions(total, parts):
    """Every way to split `total` picks across `parts` pools."""
    if parts == 1:
        yield (total,)
        return
    for first in range(total + 1):
        for rest in _compositions(total - first, parts - 1):
            yield (first,) + rest


def _grouped_expansion(rows_of, need, budget, specials, team_of, *,
                       limits=None, stack_team=None, stack_qb=0, no_rb_team=None):
    """Best lineup for one slot shape while counting certain teams' players.

    The correlation rules are all statements about how many of one team end up
    in the lineup, which the plain slot-by-slot convolution cannot see: it
    tracks price and nothing else. So the teams a rule names — `specials` — get
    their players split out of every slot pool, and the running state carries
    how many of each have been taken. Teams nobody is asking about stay in one
    undifferentiated pool and cost nothing extra, which is what keeps this
    affordable: a cap only ever binds on a handful of teams.

    Returns (picks, cost_buckets, points) or None.
    """
    limits = limits or {}
    index = {team: n for n, team in enumerate(specials)}
    stack_at = index.get(stack_team) if stack_team is not None else None
    no_rb_at = index.get(no_rb_team) if no_rb_team is not None else None
    want = stack_qb if stack_at is not None else 0

    # state: (picks per special team, pass catchers from the QB's team) -> profile
    states = {((0,) * len(specials), 0): {0: (0.0, ())}}
    for slot in sorted(need, key=lambda s: -need[s]):
        count = need[slot]
        if count <= 0:
            continue
        parts = [[] for _ in range(len(specials) + 1)]
        for row in rows_of.get(slot, ()):
            parts[index.get(team_of.get(row[0]), len(specials))].append(row)
        frontiers = [_frontiers(part, count, budget) for part in parts]
        following = {}
        for (taken, catchers), profile in states.items():
            for split in _compositions(count, len(parts)):
                if no_rb_at is not None and slot == "RB/FLEX" and split[no_rb_at]:
                    continue
                running = list(taken)
                over = False
                for n, picked in enumerate(split[:-1]):
                    running[n] += picked
                    ceiling = limits.get(specials[n])
                    # A team can be split out for the stack rules alone, with
                    # no cap of its own -- then there is nothing to exceed.
                    if ceiling is not None and running[n] > ceiling:
                        over = True
                        break
                if over:
                    continue
                combined = profile
                for n, picked in enumerate(split):
                    if not picked:
                        continue
                    frontier = frontiers[n].get(picked)
                    if not frontier:
                        combined = None
                        break
                    combined = _combine(combined, frontier, budget)
                    if not combined:
                        break
                if not combined:
                    continue
                held = catchers
                if stack_at is not None and slot in PASS_CATCHER_SLOTS:
                    held = min(want, catchers + split[stack_at])
                key = (tuple(running), held)
                following[key] = _merge_profiles(following.get(key), combined)
        states = following
        if not states:
            return None

    best = None
    for (_, catchers), profile in states.items():
        if catchers < want:
            continue
        for cost, (points, picks) in profile.items():
            if best is None or points > best[2]:
                best = (list(picks), cost, points)
    return best


def _ruled_expansion(df, need, budget, step, stack_qb, rb_with_qb, limits, incumbent):
    """Best lineup for one slot shape under every rule in force.

    The quarterback rules hang off which team he comes from, so the solve runs
    once per team and keeps the best. Teams are tried best-bound first against
    an incumbent — the bound is the same roster with every rule relaxed, so a
    team that cannot win even unconstrained is skipped rather than solved.
    """
    if need.get("QB", 0) != 1:
        return None                     # not a shape these rules describe
    if stack_qb > need.get("WR/FLEX", 0) + need.get("TE/FLEX", 0):
        return None                     # more partners than slots to hold them

    rows_of = {}
    for slot, count in need.items():
        pool = df[[slot in s for s in df["_slots"]]]
        rows_of[slot] = _rows_for(pool, step)
        if len(rows_of[slot]) < count:
            return None
    team_of = df["Team"].astype(str).to_dict()
    per_qb = not rb_with_qb or bool(stack_qb)

    if not per_qb:
        return _grouped_expansion(rows_of, need, budget, sorted(limits), team_of,
                                  limits=limits)

    # The relaxed remainder, for the per-team bound.
    relaxed = {0: (0.0, ())}
    for slot, count in need.items():
        if slot == "QB" or not count:
            continue
        relaxed = _combine(relaxed, _frontiers(rows_of[slot], count, budget)[count], budget)
        if not relaxed:
            return None
    costs, prefix = _prefix_best(relaxed)

    by_team = {}
    for row in rows_of["QB"]:
        by_team.setdefault(team_of.get(row[0]), []).append(row)
    bounds = sorted(
        ((max(points + _best_at_or_under(costs, prefix, budget - cost)
              for _, cost, points in rows), team)
         for team, rows in by_team.items()), reverse=True)

    best = incumbent
    for bound, team in bounds:
        if best is not None and bound <= best[2]:
            break                       # sorted, so nothing after it can win
        specials = sorted({team} | set(limits))
        # The quarterback slot is narrowed to this team, which is what makes
        # the rules below be about him rather than about some other team's.
        found = _grouped_expansion(
            dict(rows_of, QB=by_team[team]), need, budget, specials, team_of,
            limits=limits, stack_team=team if stack_qb else None,
            stack_qb=stack_qb, no_rb_team=None if rb_with_qb else team)
        if found is not None and (best is None or found[2] > best[2]):
            # The quarterback has to be the one this team was solved for.
            best = found
    return best


def _team_counts(df, picks):
    counts = Counter()
    teams = df["Team"].astype(str)
    for idx in picks:
        counts[teams.get(idx)] += 1
    return counts


def best_lineup(df, cap, roster_slots=None, salary_step=100, *,
                max_per_team=None, stack_qb=0, rb_with_qb=True):
    """The highest-scoring legal lineup at or under `cap`, or None if none fits.

    Exact, where the old fill was greedy. It filled slot by slot taking the best
    points-per-dollar player that fit the salary left, with no budget held back
    for the slots still to come and no way to back up when one came up empty --
    so it reported "the salary cap or the exclusions are what nothing fits
    under" for slates where plenty fits. A pool with expensive players who also
    have the best value is enough to trigger it: the early slots eat the budget,
    the tight end slot finds nothing under what is left, and the attempt is
    abandoned rather than retried a little cheaper.

    The optional rules say how the nine picks may relate to each other, which
    is the one thing a per-player projection cannot express. They are all still
    solved exactly:

      max_per_team: at most this many of the nine from one team. FanDuel's own
          limit is 4, and the defense is counted here as one of its team's
          players -- the stricter reading, and no loss, since four of a team's
          skill players alongside their own defense is a bet against itself.
      stack_qb: at least this many WR/TE from the quarterback's team.
      rb_with_qb: when False, no running back from the quarterback's team.

    All three need a `Team` column, which every FanDuel player list has.

    Returns (indices, salary, points).
    """
    roster_slots = roster_slots or ROSTER_SLOTS
    budget = int(cap // salary_step)
    if "_slots" not in df.columns:
        add_eligibility(df)
    correlated = bool(stack_qb) or not rb_with_qb
    if "Team" not in df.columns:
        # A real FanDuel player list always names the team. A hand-made frame
        # may not, and the team cap is on by default -- so an explicit request
        # is an error, while the default quietly stands down rather than
        # refusing to build at all.
        if correlated:
            raise ValueError(
                "stack_qb / rb_with_qb need a 'Team' column; the FanDuel "
                "player list has one, a bare projection frame does not.")
        if max_per_team:
            warnings.warn("no 'Team' column, so max_per_team is not applied",
                          RuntimeWarning, stacklevel=2)
            max_per_team = None

    if max_per_team:
        shape = sum(int(n) for n in roster_slots.values())
        if shape > df["Team"].astype(str).nunique() * max_per_team:
            return None                 # the slate cannot spread a roster that thin

    def solve(limits):
        best = None
        for need in _flex_expansions(roster_slots):
            if correlated or limits:
                found = _ruled_expansion(df, need, budget, salary_step, stack_qb,
                                         rb_with_qb, limits, best)
            else:
                found = _plain_expansion(df, need, budget, salary_step)
            if found is not None and (best is None or found[2] > best[2]):
                best = found
        return best

    # Counting a team costs something, so no team is counted until it has
    # actually broken the cap. Each pass adds the offenders and solves again,
    # and since every pass only ever removes lineups, the first answer that
    # breaks no cap is the best answer that breaks no cap.
    limits, found = {}, None
    for _ in range(sum(int(n) for n in roster_slots.values()) + 1):
        found = solve(limits)
        if found is None or not max_per_team:
            break
        over = [team for team, n in _team_counts(df, found[0]).items()
                if n > max_per_team]
        if not over:
            break
        limits.update({team: max_per_team for team in over})
    else:
        found = None
    if found is None:
        return None
    picks, cost, points = found
    return list(picks), cost * salary_step, points


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
                     salary_step=100, price_step=None, max_per_team=4,
                     stack_qb=0, rb_with_qb=True, ownership_weight=0.0,
                     target=None, sims=None, candidates=None, diversify=None):
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
        max_per_team: at most this many of the nine from one team, defence
            included. 4 by default, which is FanDuel's own limit — the builder
            could previously return a lineup the site would refuse to enter.
        stack_qb: require at least this many WR/TE from the quarterback's team.
        rb_with_qb: when False, no running back from the quarterback's team.
        ownership_weight: points to charge per percent of projected ownership.
            0 ignores the field. The charge is linear and additive, so the
            solve stays exact; at 0.1 a 30%-owned player has to out-project a
            1%-owned one by 2.9 points to keep his place.
        target: a score to aim at (125, say). Given one, the builder solves
            `candidates` lineups on the linear objective and then re-ranks them
            by the chance of CLEARING the target, which needs `sims`.
        sims: a per-simulation points matrix aligned row-for-row with `df`,
            from `nfl_projections.simulate.pool_sims`.
        candidates: how many lineups to solve before re-ranking on `target`.
            Defaults to eight times what was asked for, floored at 40.
        diversify: whether `max_usage_percentage` still binds after a re-rank.
            Defaults to False when a target is set and True otherwise, because
            spreading is a tournament idea: it buys more shots at an unlikely
            outcome. A cash target wants the opposite -- there is usually one
            best way to clear it, and the next-best lineups are eight-ninths
            the same roster. Forcing five different ones costs real points: on
            the 2026 week 6 main slate the top five by P(>=125) scored 123.4,
            and the same five spread under a 50% usage cap scored 94.0.

    A target changes what "best" means, and it is the more honest objective of
    the two. "ceiling" adds up nine players' individual 90th percentiles, and
    because a lineup's mean grows with nine players while its spread grows with
    three, that sum lands around the 99th percentile of what the lineup can
    actually do -- on the 2026 week 6 main slate, an eight-player core summed to
    238.7 against a true p90 of 169.9. Ranking on it ranks by volatility. The
    chance of clearing a target is a probability of something that happens.

    The three rules above are the only place the builder knows that players
    share a game. The projections are made one player at a time, so nothing
    else stops nine independently sensible picks from being one bet on one
    offence: in 2026 week 7 the board put Carolina in 16% of every slot it
    filled, and Carolina lost 43-13.

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
    # `sims` is built from the pool as handed in, so a simulation's row number
    # is that frame's POSITION. Capture the mapping before any filtering drops
    # rows, or an injury scratch silently shifts every player's simulation by
    # one and the probabilities come back confidently wrong.
    df_rows_at_entry = list(df.index)
    positions = {label: i for i, label in enumerate(df_rows_at_entry)}
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
    # The defence slot takes a real projection when one has been joined on, and
    # falls back to FanDuel's FPPG when it has not. FPPG is a bad number here
    # and measurably so -- over 2019-2026 it trails a flat league average on
    # error, because a defence's season mean chases sacks and takeaways that
    # barely repeat. See nfl_projections.defense.
    is_defense = [("DEF" in s) for s in df["_slots"]]
    if "defense_points" in df.columns:
        projected = pd.to_numeric(df.loc[is_defense, "defense_points"], errors="coerce")
        df.loc[is_defense, "lineup_points"] = projected.fillna(df.loc[is_defense, "FPPG"])
    else:
        df.loc[is_defense, "lineup_points"] = df.loc[is_defense, "FPPG"]

    # The field is charged for before the solve, so it is part of the exact
    # objective rather than a filter applied to lineups that were already
    # chosen. A player missing from the ownership source is NOT cheap leverage:
    # on the 2026 week 6 main slate every one of the five unlisted players the
    # board projected above 10 points was inactive, so an unknown is left
    # uncharged rather than rewarded.
    if ownership_weight:
        if "ownership" not in df.columns:
            raise ValueError("ownership_weight needs an `ownership` column; see "
                             "nfl_projections.ownership.attach")
        owned = pd.to_numeric(df["ownership"], errors="coerce").fillna(0.0)
        df["lineup_points"] = df["lineup_points"] - ownership_weight * owned

    # A usage cap is a spreading rule, and spreading only pays when there are
    # enough lineups to spread over. Asking for five under a 50% cap lets each
    # player appear twice, which forces lineups 3-5 down the price ladder: on
    # the 2026 week 6 main slate that set averaged 104.8 against 125.1 for the
    # five best, while building 25 under the same cap and playing the top five
    # got 123.4. Fewer lineups should not mean worse lineups.
    if (max_usage_percentage and max_usage_percentage < 100 and num_lineups > 1
            and num_lineups * max_usage_percentage / 100 < 3):
        warnings.warn(
            f"a {max_usage_percentage}% usage cap over {num_lineups} lineups "
            f"lets each player appear in at most "
            f"{int(num_lineups * max_usage_percentage / 100)}, which pushes the "
            "later lineups well down the price ladder. Build more and play the "
            "best few, or pass max_usage_percentage=100.",
            RuntimeWarning, stacklevel=2)

    wanted = num_lineups
    if target is not None:
        if sims is None:
            raise ValueError("target needs `sims`; see simulate.pool_sims")
        if len(set(df_rows_at_entry)) != len(df_rows_at_entry):
            raise ValueError("the pool's index has duplicate labels, so a "
                             "lineup cannot be mapped back to its simulations; "
                             "reset_index() before building sims and lineups")
        if len(sims) != len(df_rows_at_entry):
            raise ValueError(
                f"sims has {len(sims)} rows but the pool had "
                f"{len(df_rows_at_entry)}; pool_sims must be built from the "
                "same frame that is passed here, before any filtering")
        wanted = int(candidates) if candidates else max(40, num_lineups * 8)
        if diversify is None:
            diversify = False
        # Walk the budget down in the smallest step the salaries allow, so the
        # candidates are all near-optimal variants rather than a ladder whose
        # bottom half is too cheap to clear anything.
        if price_step is None:
            price_step = salary_step

    step = int(price_step) if price_step else max(salary_step, salary_cap // 100)
    all_lineups = []
    budget = salary_cap
    while len(all_lineups) < wanted and budget >= 0:
        # Players already at their exposure limit are dropped from the pool
        # rather than rejected after the fact, so the solve stays exact over
        # whoever is actually still available.
        pool = df[[_check_usage_limit(n, all_lineups, wanted, max_usage_percentage)
                   for n in df["Nickname"]]]
        found = best_lineup(pool, budget, roster_slots, salary_step,
                            max_per_team=max_per_team, stack_qb=stack_qb,
                            rb_with_qb=rb_with_qb)
        if found is None:
            break
        picks, spent, _ = found
        lineup_df = pool.loc[picks]
        if _is_lineup_unique(lineup_df, all_lineups):
            all_lineups.append(lineup_df)
            if target is None:
                print(f"Created lineup {len(all_lineups)} (${spent:,})")
        # Next one has to be cheaper, which is both the variation that was asked
        # for and the reason this terminates.
        budget = spent - step

    if target is not None:
        all_lineups = rank_by_target(all_lineups, sims, target, positions)
        if diversify:
            all_lineups = _select_diverse(all_lineups, num_lineups,
                                          max_usage_percentage)

    if len(all_lineups) < num_lineups:
        print(f"Warning: only generated {len(all_lineups)}/{num_lineups} lineups "
              f"under a ${salary_cap:,} cap")
    return all_lineups[:num_lineups]


def rank_by_target(lineups, sims, target, positions):
    """Sort lineups by how often they clear `target`, best first.

    `positions` maps a lineup row's index label to its row in `sims`. Each
    lineup's total is summed across the SAME simulations, so a quarterback and
    his receiver rise together instead of being averaged apart.
    """
    from .simulate import lineup_totals

    scored = []
    for lineup in lineups:
        rows = [positions[i] for i in lineup.index]
        totals = lineup_totals(sims, rows)
        lineup = lineup.copy()
        lineup.attrs["p_target"] = float((totals >= target).mean())
        lineup.attrs["sim_mean"] = float(totals.mean())
        lineup.attrs["sim_p90"] = float(np.percentile(totals, 90))
        scored.append(lineup)
    return sorted(scored, key=lambda l: l.attrs["p_target"], reverse=True)


def _select_diverse(ranked, num_lineups, max_usage_percentage):
    """Take the best lineups in order, keeping anyone's usage under the cap.

    Re-ranking happens over a candidate pool many times the size of what was
    asked for, and the best few by probability are often eight-ninths the same
    roster. Without this the usage cap would be silently lost in the re-rank.
    """
    if not max_usage_percentage or max_usage_percentage >= 100:
        return ranked[:num_lineups]
    limit = max_usage_percentage / 100 * num_lineups
    chosen, used = [], Counter()
    for lineup in ranked:
        names = list(lineup["Nickname"])
        if any(used[n] + 1 > limit for n in names):
            continue
        chosen.append(lineup)
        used.update(names)
        if len(chosen) == num_lineups:
            break
    # A cap too tight for the candidate pool should not return nothing.
    for lineup in ranked:
        if len(chosen) >= num_lineups:
            break
        if not any(l is lineup for l in chosen):
            chosen.append(lineup)
    return chosen


def optimize_from_csv(csv_file, num_lineups=5, salary_cap=60000, exclude_players=None,
                      max_usage_percentage=50, objective="mean", max_per_team=4,
                      stack_qb=0, rb_with_qb=True):
    """Optimize lineups from a merged salary+projection CSV file."""
    df = pd.read_csv(csv_file)
    return optimize_lineups(df, num_lineups=num_lineups, salary_cap=salary_cap,
                            exclude_players=exclude_players,
                            max_usage_percentage=max_usage_percentage,
                            objective=objective, max_per_team=max_per_team,
                            stack_qb=stack_qb, rb_with_qb=rb_with_qb)


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


# --- FanDuel bulk upload --------------------------------------------------
#
# FanDuel's own "lineup upload template" is a CSV whose first nine columns are
# the roster, in this exact order, and whose instructions say that anything to
# the right of the ninth is ignored. A cell holds either the player's id or
# "id:Name". Up to 250 lineups go in one upload.
FANDUEL_UPLOAD_COLUMNS = ("QB", "RB", "RB", "WR", "WR", "WR", "TE", "FLEX", "DEF")
FANDUEL_UPLOAD_LIMIT = 250

# How many of each position the nine columns hold before the FLEX is filled.
_UPLOAD_QUOTA = {"QB": 1, "RB": 2, "WR": 3, "TE": 1, "D": 1}
_UPLOAD_BASE = {"QB": "QB", "RB": "RB", "FB": "RB", "WR": "WR", "TE": "TE",
                "D": "D", "DEF": "D", "DST": "D", "D/ST": "D"}


def _base_position(row):
    """The position a row counts as for the upload columns."""
    for raw in (row.get("Position"), row.get("Roster Position")):
        if raw is None or raw != raw:
            continue
        token = str(raw).strip().upper()
        if token in _UPLOAD_BASE:
            return _UPLOAD_BASE[token]
        head = token.split("/")[0]
        if head in _UPLOAD_BASE:
            return _UPLOAD_BASE[head]
    return None


def assign_upload_slots(lineup):
    """Order a lineup's nine rows into FanDuel's upload columns.

    A classic roster is one QB, two RB, three WR, one TE, one DEF and a FLEX,
    and the solver does not record which of the flexible players it treated as
    the FLEX -- nor does it need to, since FanDuel scores the column the same
    either way. So the quotas are filled first and the one surplus player goes
    to FLEX. Which one that is is cosmetic; taking the lowest projected of its
    group keeps the named starters reading as the starters.

    Returns a list of nine rows in `FANDUEL_UPLOAD_COLUMNS` order.
    """
    rows = [r for _, r in lineup.iterrows()]
    if len(rows) != 9:
        raise ValueError(f"a classic lineup has nine players, got {len(rows)}")
    points = "lineup_points" if "lineup_points" in lineup.columns else None
    groups = {}
    for row in rows:
        base = _base_position(row)
        if base is None:
            raise ValueError(f"cannot place {row.get('Nickname')!r} in an upload "
                             f"column: position {row.get('Position')!r} is not one "
                             "of QB/RB/WR/TE/D")
        groups.setdefault(base, []).append(row)
    for base, group in groups.items():
        if points:
            group.sort(key=lambda r: (pd.to_numeric(r.get(points), errors="coerce")
                                      if pd.notna(r.get(points)) else 0.0),
                       reverse=True)
        quota = _UPLOAD_QUOTA.get(base, 0)
        if len(group) < quota:
            raise ValueError(f"the lineup has {len(group)} {base} where the upload "
                             f"columns need {quota}")

    placed, flex = {}, []
    for base, group in groups.items():
        quota = _UPLOAD_QUOTA.get(base, 0)
        placed[base] = list(group[:quota])
        flex.extend(group[quota:])
    if len(flex) != 1:
        raise ValueError(f"expected exactly one FLEX player, got {len(flex)} "
                         "— is this a classic lineup?")
    # FanDuel's FLEX takes a running back, receiver or tight end. A spare
    # quarterback or defence satisfies the arithmetic and would be rejected on
    # upload, which is the worst place to find out.
    if _base_position(flex[0]) not in ("RB", "WR", "TE"):
        raise ValueError(
            f"{flex[0].get('Nickname')!r} cannot fill the FLEX: it takes a "
            "running back, receiver or tight end, not a "
            f"{_base_position(flex[0])}")

    ordered, taken = [], {base: 0 for base in placed}
    for column in FANDUEL_UPLOAD_COLUMNS:
        if column == "FLEX":
            ordered.append(flex[0])
            continue
        base = "D" if column == "DEF" else column
        ordered.append(placed[base][taken[base]])
        taken[base] += 1
    return ordered


def _upload_cell(row, contest_id, include_names):
    """One cell: the FanDuel id, optionally with the name FanDuel prints beside it."""
    raw = row.get("Id")
    if raw is None or raw != raw or not str(raw).strip():
        raise ValueError(
            f"{row.get('Nickname')!r} has no FanDuel `Id`. The upload needs the "
            "ids from the contest's own player list, so build from a FanDuel "
            "CSV rather than from a projection export.")
    player_id = str(raw).strip()
    if contest_id:
        # An id is "<contest>-<player>", and the contest half differs between
        # the player list you priced from and the contest you are entering. Get
        # it wrong and FanDuel rejects the whole file, so it can be overridden
        # in one place instead of re-downloading everything.
        player_id = f"{contest_id}-{player_id.split('-', 1)[-1]}"
    if not include_names:
        return player_id
    return f"{player_id}:{row.get('Nickname')}"


def fanduel_upload_rows(lineups, *, contest_id=None, include_names=True,
                        header=True):
    """Lineups as the rows of a FanDuel bulk-upload CSV."""
    rows = [list(FANDUEL_UPLOAD_COLUMNS)] if header else []
    for lineup in lineups:
        rows.append([_upload_cell(r, contest_id, include_names)
                     for r in assign_upload_slots(lineup)])
    return rows


def write_fanduel_upload(lineups, path=None, *, contest_id=None,
                         include_names=True):
    """Write lineups as a CSV FanDuel's bulk entry will accept.

    Returns the CSV text, and writes it to `path` when one is given. More than
    `FANDUEL_UPLOAD_LIMIT` lineups is refused rather than silently truncated,
    because a partial upload looks like a successful one.
    """
    import csv
    import io

    lineups = list(lineups)
    if len(lineups) > FANDUEL_UPLOAD_LIMIT:
        raise ValueError(f"FanDuel takes {FANDUEL_UPLOAD_LIMIT} lineups per "
                         f"upload; this is {len(lineups)}. Split the build.")
    buffer = io.StringIO()
    # FanDuel's own files are CRLF, and its parser is the one that has to read
    # this back.
    writer = csv.writer(buffer, lineterminator="\r\n")
    writer.writerows(fanduel_upload_rows(lineups, contest_id=contest_id,
                                         include_names=include_names))
    text = buffer.getvalue()
    if path is not None:
        with open(path, "w", newline="") as handle:
            handle.write(text)
    return text


def read_fanduel_csv(source):
    """Read a FanDuel player list, from either file FanDuel hands out.

    The "players list" export is a plain table. The "lineup upload template" is
    the same table pushed ten columns to the right, with the roster columns and
    four lines of instructions occupying the left -- and it is the more useful
    of the two here, because its ids carry the contest you are actually
    entering. Reading both means an upload cannot end up prefixed with a
    different contest's id.

    Parsed row by row rather than with `read_csv`, because the template is
    ragged: its first rows are eleven fields wide and its player rows are
    twenty-eight, which pandas treats as a broken file.
    """
    import csv

    if hasattr(source, "read"):
        if hasattr(source, "seek"):
            source.seek(0)
        text = source.read()
        if isinstance(text, bytes):
            text = text.decode("utf-8-sig")
    else:
        with open(source, newline="", encoding="utf-8-sig") as handle:
            text = handle.read()

    rows = list(csv.reader(text.splitlines()))
    header_at = offset = None
    for i, row in enumerate(rows):
        if "Id" in row and "Nickname" in row:
            header_at, offset = i, row.index("Id")
            break
    if header_at is None:
        raise ValueError("this does not look like a FanDuel player list or "
                         "upload template: no row holds both `Id` and `Nickname`")

    names = rows[header_at][offset:]
    # FanDuel's own export has three unnamed columns before "Roster Position".
    keep = [j for j, name in enumerate(names) if name]
    body = []
    for row in rows[header_at + 1:]:
        cells = row[offset:]
        if not any(cell.strip() for cell in cells):
            continue
        cells = cells + [""] * (len(names) - len(cells))
        body.append([cells[j] for j in keep])
    frame = pd.DataFrame(body, columns=[names[j] for j in keep])
    frame = frame[frame["Id"].astype(str).str.strip() != ""].reset_index(drop=True)
    return _numeric_slate_columns(frame)


def _numeric_slate_columns(frame):
    """Put the numbers back after a string-typed read."""
    frame = frame.copy()
    for column in ("FPPG", "Salary", "Played", "Tier"):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame
