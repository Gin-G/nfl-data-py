"""Play-by-play scheme features (free nflverse pbp).

Two products, both leakage-safe (only plays from weeks *before* a target week
feed that week's features):

- defense scheme splits: EPA a defense allows on inside vs outside runs, short
  vs deep passes, and overall run vs pass (front vs secondary). One profile per
  (season, week, defense_team), rolled over a trailing window.
- player tendencies: how a player is used - outside-run share, average depth of
  target/throw (aDOT), and deep involvement rate - per (player_id, season, week).
  These are merged into the dataset and then trailing-averaged like the other
  rolling features.

pbp has no blocking-scheme or coverage labels (those are paid: PFF/SIS/FTN), so
"outside runs" (run_gap == end / edge) is a proxy for outside-zone exposure, not
a true scheme tag.
"""

import logging

import numpy as np
import pandas as pd

from .utils import to_pandas

logger = logging.getLogger(__name__)

# Opponent (defense) scheme-split columns produced here
SCHEME_DEFENSE_FEATURES = [
    "opp_rush_epa_inside",
    "opp_rush_epa_outside",
    "opp_pass_epa_short",
    "opp_pass_epa_deep",
    "opp_rush_epa",
    "opp_pass_epa",
]

# Neutral fallbacks (EPA ~0 = league-average defense)
SCHEME_DEFENSE_NEUTRAL = {c: 0.0 for c in SCHEME_DEFENSE_FEATURES}

# Player tendency columns merged into the dataset (then rolled)
PLAYER_TENDENCY_FEATURES = ["ten_outside_run_share", "ten_adot", "ten_deep_rate"]


def load_pbp(seasons):
    import nflreadpy as nfl

    return to_pandas(nfl.load_pbp(seasons=[int(s) for s in seasons]))


def _run_masks(pbp):
    """Boolean masks for inside vs outside runs (run_gap first, then location)."""
    gap = pbp.get("run_gap")
    loc = pbp.get("run_location")
    is_run = (pbp.get("play_type") == "run") | (pbp.get("rush_attempt") == 1)
    gap = gap if gap is not None else pd.Series(np.nan, index=pbp.index)
    loc = loc if loc is not None else pd.Series(np.nan, index=pbp.index)

    outside = is_run & (
        (gap == "end") | (gap.isna() & loc.isin(["left", "right"]))
    )
    inside = is_run & (
        gap.isin(["guard", "tackle"]) | (gap.isna() & (loc == "middle"))
    )
    return is_run, inside, outside


def _pass_masks(pbp):
    is_pass = (pbp.get("play_type") == "pass") | (pbp.get("pass_attempt") == 1)
    length = pbp.get("pass_length")
    length = length if length is not None else pd.Series(np.nan, index=pbp.index)
    short = is_pass & (length == "short")
    deep = is_pass & (length == "deep")
    return is_pass, short, deep


def defense_splits_weekly(pbp):
    """Per (season, week, defense_team): mean EPA allowed by play category."""
    df = pbp[pbp["defteam"].notna() & pbp["epa"].notna()].copy()
    is_run, run_in, run_out = _run_masks(df)
    is_pass, pass_short, pass_deep = _pass_masks(df)

    df = df.assign(
        _run=is_run.values, _run_in=run_in.values, _run_out=run_out.values,
        _pass=is_pass.values, _pass_short=pass_short.values, _pass_deep=pass_deep.values,
    )

    def mean_epa(g, mask_col):
        sel = g[g[mask_col]]
        return sel["epa"].mean() if len(sel) else np.nan

    rows = []
    for (season, week, defteam), g in df.groupby(["season", "week", "defteam"]):
        rows.append({
            "season": season, "week": int(week), "defense_team": defteam,
            "rush_epa_inside": mean_epa(g, "_run_in"),
            "rush_epa_outside": mean_epa(g, "_run_out"),
            "pass_epa_short": mean_epa(g, "_pass_short"),
            "pass_epa_deep": mean_epa(g, "_pass_deep"),
            "rush_epa": mean_epa(g, "_run"),
            "pass_epa": mean_epa(g, "_pass"),
        })
    return pd.DataFrame(rows)


def _roll_entering_week(weekly, key_cols, value_cols, window):
    """Trailing mean of value_cols over the `window` weeks strictly before each
    week, per key group. Mirrors opponent.build_defense_form semantics."""
    out = []
    for key, grp in weekly.groupby(key_cols):
        vals = grp.set_index("week")[value_cols].sort_index()
        full = vals.reindex(range(1, int(vals.index.max()) + 1))
        windowed = full.rolling(window, min_periods=1).mean().shift(1)
        windowed = windowed.dropna(how="all")
        if windowed.empty:
            continue
        key = key if isinstance(key, tuple) else (key,)
        for col, val in zip(key_cols, key):
            windowed[col] = val
        out.append(windowed.reset_index())
    if not out:
        return pd.DataFrame(columns=["week", *value_cols, *key_cols])
    return pd.concat(out, ignore_index=True)


def build_scheme_defense_form(pbp, window=6):
    """Rolling entering-week defensive scheme splits per (season, week, team)."""
    weekly = defense_splits_weekly(pbp)
    raw_cols = ["rush_epa_inside", "rush_epa_outside", "pass_epa_short",
                "pass_epa_deep", "rush_epa", "pass_epa"]
    rolled = _roll_entering_week(
        weekly, ["season", "defense_team"], raw_cols, window
    )
    rename = dict(zip(raw_cols, SCHEME_DEFENSE_FEATURES))
    return rolled.rename(columns=rename)[
        ["season", "week", "defense_team", *SCHEME_DEFENSE_FEATURES]
    ]


def player_tendencies_weekly(pbp):
    """Per (player_id, season, week): usage tendencies from pbp.

    - outside-run share (rushers)
    - aDOT (receiver air yards per target; passer air yards per attempt)
    - deep rate (receiver deep-target rate; passer deep-attempt rate)
    A player fills only the columns for their role; the rest stay NaN -> 0.
    """
    _, run_in, run_out = _run_masks(pbp)
    is_pass, pass_short, pass_deep = _pass_masks(pbp)

    # Rushers: outside-run share
    rush = pbp[pbp["rusher_player_id"].notna()].assign(
        _out=run_out[pbp["rusher_player_id"].notna()].values,
        _run=((run_in | run_out)[pbp["rusher_player_id"].notna()]).values,
    )
    rush_g = rush.groupby(["rusher_player_id", "season", "week"]).agg(
        _out=("_out", "sum"), _run=("_run", "sum")
    ).reset_index()
    rush_g["ten_outside_run_share"] = np.where(
        rush_g["_run"] > 0, rush_g["_out"] / rush_g["_run"], np.nan
    )
    rush_g = rush_g.rename(columns={"rusher_player_id": "player_id"})[
        ["player_id", "season", "week", "ten_outside_run_share"]
    ]

    # Receivers: aDOT + deep-target rate
    rec = pbp[pbp["receiver_player_id"].notna() & is_pass].assign(
        _deep=pass_deep[pbp["receiver_player_id"].notna() & is_pass].values,
    )
    rec_g = rec.groupby(["receiver_player_id", "season", "week"]).agg(
        _adot=("air_yards", "mean"), _deep=("_deep", "mean")
    ).reset_index().rename(columns={"receiver_player_id": "player_id"})

    # Passers: aDOT + deep-attempt rate
    pas = pbp[pbp["passer_player_id"].notna() & is_pass].assign(
        _deep=pass_deep[pbp["passer_player_id"].notna() & is_pass].values,
    ) if "passer_player_id" in pbp.columns else pd.DataFrame(
        columns=["passer_player_id", "season", "week", "air_yards", "_deep"]
    )
    if not pas.empty:
        pas_g = pas.groupby(["passer_player_id", "season", "week"]).agg(
            _adot=("air_yards", "mean"), _deep=("_deep", "mean")
        ).reset_index().rename(columns={"passer_player_id": "player_id"})
    else:
        pas_g = pd.DataFrame(columns=["player_id", "season", "week", "_adot", "_deep"])

    # Receiver and passer aDOT/deep don't overlap per player -> combine
    pass_side = pd.concat([rec_g, pas_g], ignore_index=True)
    pass_side = pass_side.rename(columns={"_adot": "ten_adot", "_deep": "ten_deep_rate"})
    pass_side = pass_side.groupby(["player_id", "season", "week"], as_index=False)[
        ["ten_adot", "ten_deep_rate"]
    ].mean()

    out = pd.merge(rush_g, pass_side, on=["player_id", "season", "week"], how="outer")
    out["week"] = out["week"].astype(int)
    return out


def attach_player_tendencies(dataset, tendencies):
    """Merge per-(player_id, season, week) tendencies onto the player dataset.

    Only real games carry a numeric week; AVG rows and unmatched games get NaN
    (later trailing-averaged / zero-filled by features.add_rolling_features).
    """
    df = dataset.copy()
    week_num = pd.to_numeric(df["week"], errors="coerce")
    merged = df.assign(_week_num=week_num).merge(
        tendencies.rename(columns={"week": "_week_num"}),
        on=["player_id", "season", "_week_num"], how="left",
    ).drop(columns=["_week_num"])
    return merged
