"""Full-season projection assembly.

The weekly model gives a matchup-neutral "expected points given current form" number
(the same every week, since its only inputs are the player's own rolling stats). This
module turns that into a full-season projection that VARIES BY OPPONENT: each of a team's
scheduled games gets a matchup multiplier from the team power ratings (own offense grade
vs the opponent's defense grade), so a player projects for more against a soft defense and
less against a stingy one.

Honest scope (EXPERIMENTS.md): the opponent-defense side of a matchup is a weak, dampened
per-game lever — the base rolling-form projection is the backbone; the matchup nudge is
small and clipped. Rookies come in via the draft-capital prior (predict.py). This is a
preseason / forward-looking season view; regenerate as real games update both the base
form and the grades.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import ratings as ratings_mod
from . import roles as roles_mod

_TEAM_COLS = ("team", "recent_team", "team_abbr", "club_code")


def _snap_factor(snap_share, player_id, player_name) -> float:
    """Look up a manual snap-share factor (two-way / part-time players) by id or
    name. ``snap_share`` maps a player id or name to a factor in (0, 1]; returns
    1.0 when absent. Used for cases with no data to learn from — e.g. a rookie
    expected to split snaps between offense and defense (Travis Hunter)."""
    if not snap_share:
        return 1.0
    for key in (player_id, player_name):
        if key is not None and key in snap_share:
            return float(snap_share[key])
    if player_name:
        low = str(player_name).lower()
        for k, v in snap_share.items():
            if isinstance(k, str) and k.lower() == low:
                return float(v)
    return 1.0


def _team_col(df: pd.DataFrame) -> str:
    for c in _TEAM_COLS:
        if c in df.columns:
            return c
    raise KeyError(f"no team column in projections (looked for {_TEAM_COLS})")


def _schedule_opponents(season: int, schedule=None) -> pd.DataFrame:
    """Long form: one row per team per scheduled game -> (season, week, team, opp, home)."""
    if schedule is None:
        import nflreadpy as nfl
        schedule = nfl.load_schedules(seasons=[season]).to_pandas()
    sch = schedule[schedule["season"] == season] if "season" in schedule.columns else schedule
    rows = []
    for _, g in sch.iterrows():
        h, a = g.get("home_team"), g.get("away_team")
        if pd.isna(h) or pd.isna(a):
            continue
        rows.append((season, g["week"], h, a, True))
        rows.append((season, g["week"], a, h, False))
    return pd.DataFrame(rows, columns=["season", "week", "team", "opp", "home"])


def assemble_season(base_projections: pd.DataFrame, season: int, *, grades=None,
                    league_avg=None, schedule=None, damp: float = 0.75,
                    use_roles: bool = True, budgets=None, snap_share=None,
                    shares=None, share_blend: float = 0.5) -> pd.DataFrame:
    """Expand matchup-neutral base projections into a per-game season projection.

    Args:
        base_projections: one row per player with a team column, ``position``,
            ``depth_rank`` and ``fanduel_fantasy_points`` (and optionally floor/ceiling).
        grades: a ratings.grades() frame (off_rating/def_rating by team); built from the
            prior season's preseason prior if omitted.
        league_avg: league avg points/team/game; derived from the prior season if omitted.
        use_roles: apply depth-chart role corrections (roles.py) — availability
            (``expected_games``) so non-starters play fewer games, and a finite
            per-team-position budget so teammates SHARE a pool instead of each
            projecting for a full workload. On by default; set False for the raw
            (every player, 17 full games) behaviour.
        budgets: per-position per-game team fantasy-point pool for the finite cap;
            defaults to roles.DEFAULT_BUDGETS. Pass roles.position_budgets(dataset)
            to derive from history.
        snap_share: optional {player id or name: factor in (0,1]} to hand-cap
            part-time / two-way players (no data exists to learn this).

    Returns a long DataFrame: one row per player per scheduled game with base_projection,
    matchup_multiplier, opponent, the (role/budget-adjusted) projection, an ``exp_games``
    weight per row (sums to the player's expected games), and floor/ceiling if present.
    """
    if grades is None:
        grades = ratings_mod.grades(season, schedule=schedule)
    if league_avg is None:
        league_avg = ratings_mod.league_avg_points(season - 1, schedule=schedule)
    if use_roles and budgets is None:
        budgets = dict(roles_mod.DEFAULT_BUDGETS)

    off = grades["off_rating"].to_dict()
    dff = grades["def_rating"].to_dict()
    tcol = _team_col(base_projections)
    sched = _schedule_opponents(season, schedule=schedule)

    have_band = {"floor", "ceiling"} <= set(base_projections.columns)
    out = []
    for _, p in base_projections.iterrows():
        team = p[tcol]
        games = sched[sched["team"] == team]
        n_games = len(games)
        base = float(p["fanduel_fantasy_points"])
        position = p.get("position")
        depth_rank = p.get("depth_rank")

        # Availability: fraction of scheduled games this depth role actually plays,
        # times any manual snap-share cap. Multiplies each game so the season TOTAL
        # reflects expected games while per-game matchup variation is preserved.
        if use_roles and n_games:
            exp_games = roles_mod.expected_games(position, depth_rank, scheduled_games=n_games)
            play_w = (exp_games / n_games) * _snap_factor(
                snap_share, p.get("player_id"),
                p.get("player_name") or p.get("player_display_name"))
        else:
            play_w = 1.0

        for _, g in games.iterrows():
            mult = ratings_mod.scoring_multiplier(
                off.get(team, 0.0), dff.get(g["opp"], 0.0), league_avg, damp=damp)
            eff = mult * play_w
            row = {
                "player_id": p.get("player_id"),
                "player_name": p.get("player_name") or p.get("player_display_name"),
                "position": position,
                "team": team,
                "depth_rank": roles_mod.norm_rank(depth_rank),
                "week": int(g["week"]),
                "opponent": g["opp"],
                "home": bool(g["home"]),
                "base_projection": round(base, 2),
                "matchup_multiplier": round(mult, 3),
                "exp_games": round(play_w, 4),
                "projection": round(base * eff, 2),
            }
            if have_band:
                row["floor"] = round(float(p["floor"]) * eff, 2)
                row["ceiling"] = round(float(p["ceiling"]) * eff, 2)
            out.append(row)

    weekly = pd.DataFrame(out)
    if not weekly.empty and shares is not None:
        weekly = _allocate_by_share(weekly, shares, have_band, blend=share_blend)
    if use_roles and not weekly.empty:
        weekly = _apply_team_budget(weekly, budgets, have_band)
    return weekly


def _volume_weight(row) -> float:
    """A player's volume weight from predicted shares: RBs are carry-led (plus receiving),
    WR/TE target-led. QB/unknown -> NaN (left on rolling form)."""
    pos = row.get("position")
    cs = float(row.get("carry_share", 0.0) or 0.0)
    ts = float(row.get("target_share", 0.0) or 0.0)
    if pos == "RB":
        return cs + 0.5 * ts
    if pos in ("WR", "TE"):
        return ts
    return float("nan")


def _allocate_by_share(weekly: pd.DataFrame, shares: pd.DataFrame, have_band: bool,
                       blend: float = 0.5) -> pd.DataFrame:
    """Redistribute each (team, position, week) group's projected fantasy TOTAL by a blend of
    rolling-form weight and the validated share model's volume weight (shares.py). Keeps the
    group total unchanged (no points invented) — it only moves volume toward who the share
    model says earns it, which fixes vacated-share / committee cases (e.g. a lead back gets his
    ~56% instead of an even split). QB and no-share players stay on pure form."""
    key = ["player_id", "team"]
    cols = [c for c in ("carry_share", "target_share") if c in shares.columns]
    w = weekly.merge(shares[key + cols].drop_duplicates(key), on=key, how="left")
    w["_vw"] = w.apply(_volume_weight, axis=1)

    grp = w.groupby(["team", "position", "week"])
    form_sum = grp["projection"].transform("sum")
    form_w = np.where(form_sum > 0, w["projection"] / form_sum, 0.0)
    vw = w["_vw"].fillna(0.0)
    vw_sum = grp["_vw"].transform("sum")
    share_w = np.where(vw_sum > 0, vw / vw_sum, form_w)
    # QB / positions the share model doesn't cover stay on form
    share_w = np.where(w["position"].isin(["RB", "WR", "TE"]) & (vw_sum > 0), share_w, form_w)
    blended = blend * share_w + (1 - blend) * form_w
    scale = np.where(form_w > 0, blended / np.where(form_w > 0, form_w, 1.0), 1.0)

    for c in ["projection"] + (["floor", "ceiling"] if have_band else []):
        weekly[c] = (weekly[c].to_numpy() * scale).round(2)
    return weekly


def _apply_team_budget(weekly: pd.DataFrame, budgets: dict, have_band: bool) -> pd.DataFrame:
    """Enforce a finite team-position pool per (team, position, week) so teammates
    SHARE — two RBs on one team can't both project as bell cows. See
    roles.apply_team_budget; only ever scales DOWN, preserving the role-based split."""
    cols = ["projection"] + (["floor", "ceiling"] if have_band else [])
    roles_mod.apply_team_budget(
        weekly, budgets, points_col="projection",
        group_cols=("team", "position", "week"), scale_cols=cols)
    for c in cols:
        weekly[c] = weekly[c].round(2)
    return weekly


def season_totals(weekly: pd.DataFrame) -> pd.DataFrame:
    """Aggregate a per-game season projection to per-player totals.

    ``proj_total`` is the season sum; ``exp_games`` the expected games actually
    played (role-weighted, so a backup QB reads ~1-2, not 17); ``proj_per_game``
    is per game actually played (proj_total / exp_games), i.e. the rate when on
    the field, not diluted by the games they sit."""
    aggs = {"proj_total": ("projection", "sum"),
            "games": ("week", "count")}
    if "exp_games" in weekly.columns:
        aggs["exp_games"] = ("exp_games", "sum")
    if "floor" in weekly.columns:
        aggs["floor_total"] = ("floor", "sum")
        aggs["ceiling_total"] = ("ceiling", "sum")
    g = (weekly.groupby(["player_id", "player_name", "position", "team"], dropna=False)
         .agg(**aggs)
         .reset_index())
    if "exp_games" in g.columns:
        g["exp_games"] = g["exp_games"].round(1)
        g["proj_per_game"] = (g["proj_total"] / g["exp_games"].where(g["exp_games"] > 0)).round(2)
    else:
        g["proj_per_game"] = (g["proj_total"] / g["games"]).round(2)
    return g.sort_values("proj_total", ascending=False).reset_index(drop=True)


def project_season(service, season: int, *, base_week: int = 1, grades=None,
                   league_avg=None, schedule=None, damp: float = 0.75,
                   positions=None, use_injuries: bool = False,
                   use_roles: bool = True, budgets=None, snap_share=None) -> pd.DataFrame:
    """Convenience: build matchup-neutral base projections from a ProjectionService
    (projecting `base_week` for current form) and assemble the season. Returns the
    per-game long DataFrame; call season_totals() for per-player totals.

    Role corrections (``use_roles``) are on by default; team-position budgets are
    derived from the service's historical dataset. Pass ``snap_share`` to hand-cap
    part-time / two-way players (e.g. {"Travis Hunter": 0.5})."""
    base = service.project(season, base_week, positions=positions,
                           use_injuries=use_injuries, as_frame=True, schedule=schedule)
    if use_roles and budgets is None:
        budgets = roles_mod.position_budgets(getattr(service, "dataset", None))
    return assemble_season(base, season, grades=grades, league_avg=league_avg,
                           schedule=schedule, damp=damp, use_roles=use_roles,
                           budgets=budgets, snap_share=snap_share)
