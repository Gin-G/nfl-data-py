"""Does FTN charting predict a player's USAGE, rather than his efficiency?

Next Gen Stats failed because they are efficiency ratings, and efficiency
regresses — the strongest residual signal there (RYOE at r -0.13) added nothing
once a shrunk baseline was controlled for. Everything that has worked here has
been opportunity: share of carries, share of targets, role.

FTN charting is play-level *context* rather than a player rating: whether a
snap was play-action, a screen, an RPO, had motion, how many were in the
backfield, how many in the box. The hypothesis is that it describes a player's
ROLE, and role is the thing that persists.

So the test is not "does FTN predict yards". It is:

  1. STABILITY  does a player's FTN role profile repeat season to season? A
     profile that does not persist cannot predict anything, and this is the
     check that should have come first for success-rate-allowed.

  2. INCREMENTAL  does the profile predict next season's target/carry SHARE
     beyond the player's own current share? The control matters more than the
     result — a regression on current share alone will improve MAE by shrinking,
     exactly as it did for NGS.
"""
import pathlib
import numpy as np
import pandas as pd

D = pathlib.Path(__file__).parent
CACHE = pathlib.Path.home() / ".sharp-edge" / "nfl"
WEEKLY = ("https://github.com/nflverse/nflverse-data/releases/download/"
          "stats_player/stats_player_week_{s}.parquet")
FTN = ("https://github.com/nflverse/nflverse-data/releases/download/"
       "ftn_charting/ftn_charting_{s}.parquet")
PBP = ("https://github.com/nflverse/nflverse-data/releases/download/pbp/"
       "play_by_play_{s}.parquet")
SEASONS = (2022, 2023, 2024, 2025)

# Context flags that plausibly describe a role rather than a result.
FLAGS = ["is_play_action", "is_screen_pass", "is_rpo", "is_motion", "is_no_huddle"]


def weekly(seasons):
    out = []
    for s in seasons:
        p = CACHE / f"week_{s}.parquet"
        out.append(pd.read_parquet(p) if p.exists() else pd.read_parquet(WEEKLY.format(s=s)))
    return pd.concat(out, ignore_index=True)


def _cached(url, name, **kw):
    p = D / name
    if p.exists():
        return pd.read_parquet(p)
    df = pd.read_parquet(url, **kw)
    df.to_parquet(p)
    return df


def profiles(season):
    """Per-player role profile: the share of his touches in each context.

    Targets and carries are pooled into one 'touch' so a back's screen role and
    a receiver's are measured the same way.
    """
    ftn = _cached(FTN.format(s=season), f"ftn_{season}.parquet",
                  columns=["nflverse_game_id", "nflverse_play_id", "n_defense_box",
                           "n_offense_backfield"] + FLAGS)
    pbp = _cached(PBP.format(s=season), f"pbpftn_{season}.parquet",
                  columns=["game_id", "play_id", "receiver_player_id",
                           "rusher_player_id", "posteam"])
    d = ftn.rename(columns={"nflverse_game_id": "game_id",
                            "nflverse_play_id": "play_id"}).merge(
        pbp, on=["game_id", "play_id"], how="inner")

    d["player_id"] = d.receiver_player_id.fillna(d.rusher_player_id)
    d = d[d.player_id.notna()]
    for f in FLAGS:
        d[f] = d[f].astype("float")

    g = d.groupby("player_id").agg(
        touches=("player_id", "size"),
        **{f: (f, "mean") for f in FLAGS},
        box=("n_defense_box", "mean"),
        backfield=("n_offense_backfield", "mean"),
    ).reset_index()
    g["season"] = season
    return g[g.touches >= 40]


def shares(season):
    """Each player's share of his team's targets+carries, and his position."""
    w = weekly([season])
    w = w[w.season_type == "REG"].copy()
    for c in ("targets", "carries"):
        w[c] = w[c].fillna(0)
    w["touch"] = w.targets + w.carries
    team = w.groupby("team").touch.sum().rename("team_touch")
    p = w.groupby(["player_id", "team", "position"]).touch.sum().reset_index()
    p = p.merge(team, left_on="team", right_index=True)
    p["share"] = p.touch / p.team_touch
    return p[["player_id", "position", "share", "touch"]]


def build():
    rows = []
    for s in SEASONS[:-1]:
        prof = profiles(s)
        cur = shares(s).rename(columns={"share": "share_now", "touch": "touch_now"})
        nxt = shares(s + 1).rename(columns={"share": "share_next"})[["player_id", "share_next"]]
        d = prof.merge(cur, on="player_id").merge(nxt, on="player_id")
        d["from_season"] = s
        rows.append(d)
    return pd.concat(rows, ignore_index=True)


def stability(df):
    """Does a player's profile repeat from one season to the next?"""
    print("1. STABILITY — does the role profile persist season to season?\n")
    print(f"{'feature':22s}{'r (season N -> N+1)':>22}{'n':>7}")
    prof = pd.concat([profiles(s) for s in SEASONS], ignore_index=True)
    for f in FLAGS + ["box", "backfield"]:
        a = prof[["player_id", "season", f]].copy()
        b = a.copy()
        b["season"] = b.season - 1
        m = a.merge(b, on=["player_id", "season"], suffixes=("", "_next")).dropna()
        if len(m) < 50:
            print(f"{f:22s}{'too few':>22}{len(m):>7}")
            continue
        r = np.corrcoef(m[f], m[f + "_next"])[0, 1]
        print(f"{f:22s}{r:22.3f}{len(m):>7}")


def incremental(df, positions, label):
    d = df[df.position.isin(positions)].dropna(
        subset=["share_now", "share_next"] + FLAGS + ["box", "backfield"]).copy()
    if len(d) < 80:
        print(f"\n{label}: only {len(d)} rows, skipping")
        return
    feats = FLAGS + ["box", "backfield"]
    for f in feats:
        d[f + "_z"] = d.groupby("from_season")[f].transform(
            lambda s: (s - s.mean()) / (s.std() or 1))
    zf = [f + "_z" for f in feats]

    def fit(cols):
        preds = []
        for s in d.from_season.unique():
            tr, te = d[d.from_season != s], d[d.from_season == s]
            if len(tr) < 60 or len(te) < 20:
                continue
            X = np.column_stack([np.ones(len(tr))] + [tr[c].values for c in cols])
            beta, *_ = np.linalg.lstsq(X, tr.share_next.values, rcond=None)
            Xt = np.column_stack([np.ones(len(te))] + [te[c].values for c in cols])
            preds.append(te.assign(pred=Xt @ beta))
        return pd.concat(preds) if preds else None

    base, full = fit(["share_now"]), fit(["share_now"] + zf)
    if base is None or full is None:
        return
    f = lambda p: ((p.pred - p.share_next).abs().mean() * 100,
                   np.corrcoef(p.pred, p.share_next)[0, 1])
    mb, rb = f(base)
    mf, rf = f(full)
    raw = ((d.share_now - d.share_next).abs().mean() * 100,
           np.corrcoef(d.share_now, d.share_next)[0, 1])
    print(f"\n{label}  (n={len(d)})")
    print(f"  {'model':<28}{'MAE (share pts)':>18}{'corr':>9}")
    print(f"  {'current share, raw':<28}{raw[0]:18.3f}{raw[1]:9.3f}")
    print(f"  {'current share, fitted':<28}{mb:18.3f}{rb:9.3f}")
    print(f"  {'+ FTN role profile':<28}{mf:18.3f}{rf:9.3f}")
    print(f"  {'FTN adds':<28}{mf-mb:+18.3f}{rf-rb:+9.3f}")


if __name__ == "__main__":
    df = build()
    stability(df)
    print("\n\n2. INCREMENTAL — does it predict next season's share beyond"
          " the current one?")
    incremental(df, ["RB"], "RB share of team touches")
    incremental(df, ["WR"], "WR share of team touches")
    incremental(df, ["TE"], "TE share of team touches")
