"""Weekly projection engine.

The Projector ties together the historical dataset, the trained network,
current rosters/depth charts, and (optionally) live injury data, and produces
per-player projections for a given week.

    projector = Projector(dataset_df, trained_model, season=2025, week=13)
    results = projector.run()                          # every position
    results = projector.run(positions=["RB"])          # one position
    one = projector.predict_player("Bijan Robinson")   # single player
"""

import logging
import os

import pandas as pd

from . import config, features
from .injuries import integrate_injuries
from .utils import to_pandas

logger = logging.getLogger(__name__)

TEAM_COLUMNS = ["team", "team_abbr", "club_code", "recent_team"]


def _extract_team(row):
    """Pull a team abbreviation out of a roster row, whatever it's called."""
    for col in TEAM_COLUMNS:
        if col in row.index:
            val = row[col]
            if val is not None and not pd.isna(val):
                return str(val)
    return "Unknown"


class DepthChartAnalyzer:
    """Current depth chart roles: starter / backup / deep_backup per player."""

    OFFENSIVE_POSITIONS = ["QB", "RB", "WR", "TE", "FB"]

    def __init__(self, depth_chart_data):
        self.depth_charts = depth_chart_data
        self._process_depth_charts()

    def _process_depth_charts(self):
        self.offensive_depth = self.depth_charts[
            self.depth_charts["pos_abb"].isin(self.OFFENSIVE_POSITIONS)
        ].copy()
        self.offensive_depth["depth_role"] = self.offensive_depth["pos_rank"].apply(
            lambda x: "starter" if x == 1 else "backup" if x == 2 else "deep_backup"
        )

        self.team_position_depth = {}
        for (team, position), group in self.offensive_depth.groupby(["team", "pos_abb"]):
            sorted_group = group.sort_values("pos_rank")
            self.team_position_depth[(team, position)] = {
                "total_players": len(sorted_group),
                "starter": sorted_group.iloc[0]["player_name"] if len(sorted_group) > 0 else None,
                "backup": sorted_group.iloc[1]["player_name"] if len(sorted_group) > 1 else None,
            }

        self.player_roles = {}
        for _, row in self.offensive_depth.iterrows():
            if not isinstance(row["player_name"], str):
                continue
            self.player_roles[row["player_name"]] = {
                "team": row["team"],
                "position": row["pos_abb"],
                "depth_rank": row["pos_rank"],
                "role": row["depth_role"],
            }
        logger.info("Processed depth charts: %d offensive players", len(self.offensive_depth))

    def get_player_role(self, player_name):
        if player_name in self.player_roles:
            return self.player_roles[player_name]
        for depth_name, role_info in self.player_roles.items():
            if self._names_similar(player_name, depth_name):
                return role_info
        return None

    def get_team_competition(self, team, position):
        info = self.team_position_depth.get((team, position))
        if not info:
            return None
        total = info["total_players"]
        return {
            **info,
            "competition_level": "high" if total >= 4 else "medium" if total >= 3 else "low",
        }

    def analyze_rookie_opportunity(self, player_name, team, position):
        """Rate a rookie's opportunity from where they sit on the depth chart."""
        team_depth = self.get_team_competition(team, position)
        player_role = self.get_player_role(player_name)

        if not team_depth:
            return {"opportunity": "unknown", "context": "No team depth chart data"}
        if not player_role:
            return {"opportunity": "low", "context": f"Not found on {team} depth chart"}

        analysis = {
            "depth_rank": player_role["depth_rank"],
            "role": player_role["role"],
            "team_depth": team_depth["total_players"],
            "competition_level": team_depth["competition_level"],
        }
        if player_role["role"] == "starter":
            analysis.update(opportunity="high",
                            context=f"Listed as starter on {team} depth chart")
        elif player_role["role"] == "backup" and position == "RB":
            analysis.update(opportunity="medium",
                            context="Backup RB, could see rotation work")
        elif player_role["role"] == "backup":
            analysis.update(
                opportunity="low",
                context=f"Backup behind {team_depth.get('starter', 'established starter')}",
            )
        else:
            analysis.update(opportunity="very_low", context="Deep on depth chart")
        return analysis

    @staticmethod
    def _names_similar(name1, name2):
        if not isinstance(name1, str) or not isinstance(name2, str):
            return False
        parts1, parts2 = name1.lower().split(), name2.lower().split()
        if len(parts1) >= 2 and len(parts2) >= 2:
            return parts1[-1] == parts2[-1] and (
                parts1[0] == parts2[0] or parts1[0][0] == parts2[0][0]
            )
        return False


class InjuryStatusAnalyzer:
    """Answer injury questions about players, with fuzzy name matching."""

    def __init__(self, injury_overrides, backup_situations):
        self.injury_overrides = injury_overrides
        self.backup_situations = backup_situations
        self.injury_by_normalized = {
            self._normalize_name(name): (name, info)
            for name, info in injury_overrides.items()
        }
        self.backup_by_normalized = {
            self._normalize_name(name): (name, info)
            for name, info in backup_situations.items()
        }

    @staticmethod
    def _normalize_name(name):
        if not name:
            return ""
        name_clean = name.lower().strip()
        for suffix in [" jr.", " sr.", " iii", " ii", " iv", " jr", " sr"]:
            name_clean = name_clean.replace(suffix, "")
        return " ".join(name_clean.split())

    def check_player_status(self, player_name):
        if player_name in self.injury_overrides:
            return self.injury_overrides[player_name]
        if player_name in self.backup_situations:
            return self.backup_situations[player_name]

        normalized = self._normalize_name(player_name)
        if normalized in self.injury_by_normalized:
            return self.injury_by_normalized[normalized][1]
        if normalized in self.backup_by_normalized:
            return self.backup_by_normalized[normalized][1]

        # Last resort: last name + first initial
        last_name = normalized.split()[-1] if normalized else ""
        if last_name and len(last_name) > 3:
            for lookup in (self.injury_by_normalized, self.backup_by_normalized):
                for norm_key, (_, info) in lookup.items():
                    if norm_key.endswith(last_name) and normalized[0] == norm_key[0]:
                        return info
        return None

    def should_zero_out_player(self, player_name):
        status_info = self.check_player_status(player_name)
        if status_info and "status" in status_info:
            return status_info["status"] in ["OUT", "DOUBTFUL"]
        return False

    def should_boost_backup(self, player_name):
        if player_name in self.backup_situations:
            return True
        return self._normalize_name(player_name) in self.backup_by_normalized

    def get_adjustment_info(self, player_name):
        status_info = self.check_player_status(player_name)
        if not status_info:
            return None
        if self.should_zero_out_player(player_name):
            return {
                "type": "injury_zero",
                "reason": f"OUT due to {status_info.get('reason', 'injury')}",
                "original_role": "injured_starter",
            }
        if self.should_boost_backup(player_name):
            return {
                "type": "backup_boost",
                "reason": status_info.get("reason", "replacing injured starter"),
                "replacing": status_info.get("replacing"),
                "original_role": "backup_now_starting",
            }
        return None


def _pick_tier(pick):
    """Draft-capital label for a pick number (None/0 = undrafted)."""
    if not pick or pick <= 0:
        return "undrafted"
    if pick <= 10:
        return "elite"
    if pick <= 32:
        return "high"
    if pick <= 64:
        return "medium"
    if pick <= 100:
        return "late"
    return "very_late"


class RookiePredictor:
    """Baseline projections for rookies with no NFL games, scaled by draft
    capital and depth chart opportunity. Off by default (Projector's
    rookie_fallback flag) to match prior behavior."""

    def __init__(self, historical_df, depth_analyzer, current_season):
        self.historical_df = historical_df
        self.depth_analyzer = depth_analyzer
        self.current_season = current_season
        self.draft_data = self._load_draft_data()
        self.rookie_baselines = self._calculate_baselines()
        # Draft-capital prior (measured win over the old multiplier heuristic;
        # EXPERIMENTS.md). Fit on rookies drafted strictly before this season.
        self.rookie_prior = self._fit_prior()

    def _fit_prior(self):
        from .rookies import RookiePrior
        try:
            return RookiePrior.fit(max_year=self.current_season - 1)
        except Exception as e:
            logger.warning("Could not fit rookie draft-capital prior: %s", e)
            return None

    def _load_draft_data(self):
        import nflreadpy as nfl

        try:
            draft_data = to_pandas(nfl.load_draft_picks(seasons=[self.current_season]))
            logger.info("Loaded %d draft picks for %s", len(draft_data), self.current_season)
            return draft_data
        except Exception as e:
            logger.warning("Could not load draft data: %s", e)
            return pd.DataFrame()

    def _calculate_baselines(self):
        """Average rookie-season production by position across history."""
        first_seasons = self.historical_df.groupby("player_id")["season"].min()

        rookie_frames = []
        for player_id, first_season in first_seasons.items():
            games = self.historical_df[
                (self.historical_df["player_id"] == player_id)
                & (self.historical_df["season"] == first_season)
                & (self.historical_df["week"] != "AVG")
            ]
            if len(games) >= 3:
                rookie_frames.append(games)
        if not rookie_frames:
            return {}

        all_rookies = pd.concat(rookie_frames)
        baselines = {}
        for pos in config.POSITIONS:
            pos_rookies = all_rookies[all_rookies["position"] == pos]
            if len(pos_rookies) > 20:
                baselines[pos] = {
                    "avg_fppg": pos_rookies["fanduel_fantasy_points"].mean(),
                    "avg_snaps": pos_rookies["offensive_snap_pct"].mean(),
                    "sample_size": len(pos_rookies),
                }
        return baselines

    def get_draft_info(self, player_name):
        if self.draft_data.empty:
            return None
        matches = self.draft_data[
            self.draft_data["pfr_player_name"].str.lower() == player_name.lower()
        ]
        if matches.empty:
            matches = self.draft_data[
                self.draft_data["pfr_player_name"].str.contains(
                    player_name, case=False, na=False
                )
            ]
        if matches.empty:
            return None
        draft_info = matches.iloc[0]
        return {
            "draft_position": int(draft_info["pick"]),
            "round": int(draft_info["round"]),
            "team": draft_info["team"],
        }

    def predict_rookie(self, player_name, position, team):
        draft_info = self.get_draft_info(player_name)

        # Preferred path: calibrated draft-capital prior (pick -> expected PPG +
        # floor/ceiling + component estimates). Falls back to the legacy multiplier
        # heuristic below only when the prior is unavailable for this position.
        if self.rookie_prior is not None:
            pick = draft_info["draft_position"] if draft_info else None
            projected = self.rookie_prior.project(position, pick)
            if projected is not None:
                projected["draft_tier"] = _pick_tier(pick)
                return projected

        baseline = self.rookie_baselines.get(position)
        if not baseline:
            return None
        base_points = baseline["avg_fppg"]

        if draft_info:
            pick = draft_info["draft_position"]
            if pick <= 10:
                draft_multiplier, tier = 1.6, "elite"
            elif pick <= 32:
                draft_multiplier, tier = 1.3, "high"
            elif pick <= 64:
                draft_multiplier, tier = 1.0, "medium"
            elif pick <= 100:
                draft_multiplier, tier = 0.8, "late"
            else:
                draft_multiplier, tier = 0.6, "very_late"
        else:
            draft_multiplier, tier = 0.4, "undrafted"

        opportunity = self.depth_analyzer.analyze_rookie_opportunity(
            player_name, team, position
        )
        depth_multiplier = {
            "high": 1.5, "medium": 0.8, "low": 0.3,
        }.get(opportunity["opportunity"], 0.1)

        if position == "QB":
            position_multiplier = 1.0 if opportunity["opportunity"] == "high" else 0.2
        elif position == "RB":
            position_multiplier = 1.2
        elif position in ["WR", "TE"]:
            position_multiplier = 1.0 if tier in ["elite", "high"] else 0.7
        else:
            position_multiplier = 1.0

        final_points = base_points * draft_multiplier * depth_multiplier * position_multiplier
        final_points = max(1.0, min(25.0, final_points))

        return {
            "fanduel_fantasy_points": round(final_points, 1),
            "prediction_type": "rookie",
            "draft_tier": tier,
        }


def dedupe_rosters(rosters):
    """One row per player, keeping the most recent week's roster entry."""
    rosters = rosters.copy()
    if "player_name" not in rosters.columns and "full_name" in rosters.columns:
        rosters["player_name"] = rosters["full_name"]
    if rosters["player_name"].duplicated().any():
        rosters = rosters.sort_values(["player_name", "week"], ascending=[True, False])
        rosters = rosters.drop_duplicates(subset=["player_name"], keep="first")
    return rosters


class Projector:
    """Generate weekly projections for players.

    Args:
        dataset: historical dataset DataFrame (from dataset.build_dataset/load_dataset)
        trained: model.TrainedModel
        season: season to project (e.g. 2025)
        week: week to project
        use_injuries: fetch Sportradar injuries and zero out OUT players
        rookie_fallback: give baseline projections to rookies with no games
        rosters / depth_charts: pass pre-loaded frames (mainly for tests);
            loaded from nflreadpy when omitted
        blend_form: blend each projection with the player's trailing-5 scoring
            form (see blend.py). On by default — measured to improve per-game
            MAE, cross-player ranking and spread at the same time. Pass False
            for the raw network output.
    """

    def __init__(self, dataset, trained, season, week,
                 use_injuries=True, rookie_fallback=False,
                 rosters=None, depth_charts=None, schedule=None,
                 injury_source="nflverse", scheme_form=None, include_coarse=True,
                 quantile_model=None, blend_form=True):
        self.trained = trained
        self.quantile_model = quantile_model
        self.season = season
        self.week = week
        self.blend_form = blend_form
        self._preseason = None  # resolved lazily from the history

        if rosters is None or depth_charts is None:
            import nflreadpy as nfl

            if rosters is None:
                rosters = to_pandas(nfl.load_rosters_weekly(seasons=[season]))
            if depth_charts is None:
                depth_charts = to_pandas(nfl.load_depth_charts(seasons=[season]))

        self.rosters = dedupe_rosters(rosters)
        self.depth_charts = depth_charts
        self.depth_analyzer = DepthChartAnalyzer(depth_charts)

        # Opponent-defense features, only when the model was trained with them
        from .opponent import OPPONENT_FEATURES

        self.week_features = None
        if any(c in OPPONENT_FEATURES for c in trained.numerical_features):
            from . import opponent

            schedule_map = opponent.build_schedule_map([season], schedule=schedule)
            matchup_table = opponent.build_matchup_table(
                dataset, schedule_map, scheme_form=scheme_form,
                include_coarse=include_coarse,
            )
            self.week_features = opponent.lookup_week_features(matchup_table, season, week)
            print(f"Loaded opponent matchup features for {len(self.week_features)} "
                  f"team-position slots in week {week}")

        if use_injuries:
            overrides, backups = integrate_injuries(
                week=week, roster_data=self.rosters,
                depth_charts=depth_charts, season=season, source=injury_source,
            )
        else:
            overrides, backups = {}, {}
        self.injury_analyzer = InjuryStatusAnalyzer(overrides, backups)

        # History table with derived features for latest-game lookups
        print("Preparing player history features...")
        self.history = features.prepare_prediction_base(dataset)

        self.rookie_predictor = (
            RookiePredictor(self.history, self.depth_analyzer, season)
            if rookie_fallback else None
        )

    def _opponent_features(self, team, position):
        """One-row DataFrame of upcoming-opponent features, or None if the model
        doesn't use them / the team has no game this week (neutral fallback)."""
        if self.week_features is None:
            return None
        from .opponent import NEUTRAL_VALUES, OPPONENT_FEATURES

        opp_cols = [c for c in OPPONENT_FEATURES if c in self.week_features.columns]
        try:
            row = self.week_features.loc[(team, position)]
            if isinstance(row, pd.DataFrame):  # duplicate key: take first
                row = row.iloc[0]
            return pd.DataFrame([row[opp_cols].to_dict()])
        except KeyError:
            return pd.DataFrame([{c: NEUTRAL_VALUES[c] for c in opp_cols}])

    # -- single player ----------------------------------------------------

    def predict_player(self, player_name):
        """Project one player. Returns a dict of stats or None."""
        matches = self.rosters[
            self.rosters["player_name"].str.contains(player_name, case=False, na=False)
        ]
        if matches.empty:
            logger.info("%s not found on any roster", player_name)
            return None
        return self._predict_roster_row(matches.iloc[0])

    def _is_preseason(self):
        """True when the season being projected has no games in the history —
        a preseason board (e.g. projecting 2026 off 2025 data). Those blend
        against the prior season's average rather than its last five games."""
        if self._preseason is None:
            seasons = pd.to_numeric(self.history.get("season"), errors="coerce")
            self._preseason = bool(seasons.notna().any()
                                   and (seasons == self.season).sum() == 0)
        return self._preseason

    def _predict_roster_row(self, player_info):
        from . import model as model_mod

        player_name = player_info["player_name"]
        position = player_info["position"]
        team = _extract_team(player_info)
        player_id = player_info.get("player_id", player_info.get("gsis_id", ""))
        if pd.isna(player_id):
            player_id = ""

        # Injured players are zeroed before anything else
        if self.injury_analyzer.should_zero_out_player(player_name):
            adjustment = self.injury_analyzer.get_adjustment_info(player_name)
            return {
                "player_id": player_id,
                "player_name": player_name,
                "position": position,
                "team": team,
                "fanduel_fantasy_points": 0.0,
                "prediction_type": "injured_out",
                "injury_status": "OUT",
                "injury_reason": adjustment["reason"],
            }

        player_data = self.history[
            self.history["player_display_name"].str.contains(player_name, case=False, na=False)
            | self.history["player_name"].str.contains(player_name, case=False, na=False)
        ]

        # "Recent" = this season or last; need 2+ games for an ML projection
        recent_data = player_data[player_data["season"] >= self.season - 1]
        actual_games = recent_data[recent_data["week"] != "AVG"]
        if len(actual_games) < 2:
            if self.rookie_predictor:
                rookie_pred = self.rookie_predictor.predict_rookie(
                    player_name, position, team
                )
                if rookie_pred:
                    return {
                        "player_id": player_id,
                        "player_name": player_name,
                        "position": position,
                        "team": team,
                        **rookie_pred,
                    }
            logger.info("%s has no recent data, skipping", player_name)
            return None

        recent_stats = recent_data.sort_values(["season", "week"]).iloc[[-1]]
        opp_features = self._opponent_features(team, position)
        input_df = model_mod.build_input_rows(
            self.trained, recent_stats, [position], [team],
            opponent_features=opp_features,
        )
        prediction = model_mod.predict_batch(self.trained, input_df).iloc[0]
        result = prediction.to_dict()

        # Blend with the player's own recent scoring form. The network shrinks
        # hard toward the positional mean (it is minimizing per-game error on a
        # very noisy target), which flattens the board — Josh Allen and Jared
        # Goff land within 0.01 points of each other. See blend.py.
        blend_ratio = 1.0
        if self.blend_form:
            from . import blend as blend_mod

            preseason = self._is_preseason()
            form = blend_mod.recent_form_from_rows(
                recent_stats, preseason=preseason
            ).iloc[0]
            # How many recent games back the form window — too few and the
            # "trailing average" is one game wearing a five-game label.
            n_form_games = len(actual_games)
            blended = blend_mod.blend_value(
                result["fanduel_fantasy_points"], form, position,
                weights=blend_mod.weights_for_mode(preseason),
                n_games=n_form_games,
            )
            if result["fanduel_fantasy_points"]:
                blend_ratio = blended / result["fanduel_fantasy_points"]
            # Each component is blended against its OWN trailing average rather
            # than scaled by the points ratio — the components shrink toward the
            # positional mean independently of the total, which is how a QB who
            # gained 1 rushing yard all season projected for ~160. See blend.py.
            result = blend_mod.blend_components(
                result, recent_stats, fallback_ratio=blend_ratio,
                n_games=n_form_games)
            result["fanduel_fantasy_points"] = blended

        # Depth-chart role adjustment: scale the per-game number to a snap-share
        # proxy for the player's depth rank, so a non-starter isn't read at a
        # starter's rate. This uses the full rank (not just role == "backup", so
        # deep backups no longer escape) and does NOT gate on production (the old
        # avg_fppg < 8.0 gate let inflated injury fill-ins through — exactly the
        # players we most want to discount). See roles.py.
        from . import roles

        depth_role = self.depth_analyzer.get_player_role(player_name)
        depth_rank = depth_role["depth_rank"] if depth_role else None
        role_mult = roles.per_game_role_multiplier(position, depth_rank)
        if role_mult < 1.0:
            result["fanduel_fantasy_points"] *= role_mult  # floor/ceiling scaled below too
            adjustment = f"{position} depth-rank {roles.norm_rank(depth_rank)} x{role_mult:.2f}"
        else:
            adjustment = "no adjustment"
        result["fanduel_fantasy_points"] = round(result["fanduel_fantasy_points"], 1)

        rookie = features.is_rookie(player_name, player_id, self.history, self.season)
        result.update({
            "player_id": player_id,
            "player_name": player_name,
            "position": position,
            "team": team,
            "prediction_type": "rookie_ml" if rookie else "veteran_ml",
            "depth_chart_role": depth_role["role"] if depth_role else "unknown",
            "depth_rank": depth_role["depth_rank"] if depth_role else "N/A",
            "role_adjustment": adjustment,
        })

        # Floor / median / ceiling range from the quantile model, if provided
        if self.quantile_model is not None:
            from . import quantiles as q_mod

            q_input = model_mod.build_input_rows(
                self.quantile_model, recent_stats, [position], [team],
                opponent_features=opp_features,
            )
            qp = q_mod.predict_quantiles(self.quantile_model, q_input).iloc[0]
            qcols = [f"q{int(round(q * 100))}" for q in self.quantile_model.quantiles]
            median_col = "q50" if "q50" in qcols else qcols[len(qcols) // 2]
            # Same depth-rank and form-blend scaling as the mean, so the band
            # travels with the projection instead of drifting away from it.
            band_mult = role_mult * blend_ratio
            result["floor"] = round(float(qp[qcols[0]]) * band_mult, 1)
            result["projection_median"] = round(float(qp[median_col]) * band_mult, 1)
            result["ceiling"] = round(float(qp[qcols[-1]]) * band_mult, 1)

        return result

    # -- position / week --------------------------------------------------

    def predict_position(self, position, players=None):
        """Project every rostered player at a position. Returns a DataFrame.

        Args:
            players: optional list of names to restrict to (substring match)
        """
        from tqdm import tqdm

        roster_rows = self.rosters[self.rosters["position"] == position]
        if players:
            mask = pd.Series(False, index=roster_rows.index)
            for name in players:
                mask |= roster_rows["player_name"].str.contains(name, case=False, na=False)
            roster_rows = roster_rows[mask]

        predictions = []
        processed = set()
        print(f"\nGenerating predictions for {len(roster_rows)} {position}s...")

        for _, player in tqdm(roster_rows.iterrows(), total=len(roster_rows),
                              desc=f"Predicting {position}"):
            player_name = player["player_name"]
            if player_name in processed:
                continue

            if self.injury_analyzer.should_zero_out_player(player_name):
                adjustment = self.injury_analyzer.get_adjustment_info(player_name)
                processed.add(player_name)
                predictions.append({
                    "player_id": player.get("player_id", player.get("gsis_id", "")),
                    "player_name": player_name,
                    "position": position,
                    "team": _extract_team(player),
                    "fanduel_fantasy_points": 0.0,
                    "prediction_type": "injured_out",
                    "injury_status": "OUT",
                    "injury_reason": adjustment["reason"],
                    "role_adjustment": adjustment["reason"],
                })
                continue

            # Skip inactive players (practice squad etc.) unless they're a
            # backup elevated by an injury
            if player.get("status") != "ACT" and not self.injury_analyzer.should_boost_backup(player_name):
                continue

            pred = self._predict_roster_row(player)
            if pred:
                processed.add(player_name)
                predictions.append(pred)

        if not predictions:
            return pd.DataFrame()

        df = pd.DataFrame(predictions)
        df = df.drop_duplicates(subset=["player_name"], keep="first")
        df = df.sort_values("fanduel_fantasy_points", ascending=False).reset_index(drop=True)
        df["rank"] = range(1, len(df) + 1)
        return df

    def run(self, positions=None, players=None, output_dir=config.PREDICTIONS_DIR,
            save=True):
        """Project a full week. Returns {position: DataFrame}.

        Args:
            positions: subset of positions (default: QB/RB/WR/TE)
            players: restrict to specific player names (substring match)
            output_dir: where CSVs are written when save=True
        """
        positions = positions or config.POSITIONS
        results = {}

        for position in positions:
            df = self.predict_position(position, players=players)
            if df.empty:
                continue
            results[position] = df

            if save:
                os.makedirs(output_dir, exist_ok=True)
                filename = os.path.join(
                    output_dir,
                    f"{position}_predictions_week{self.week}_season{self.season}.csv",
                )
                df.to_csv(filename, index=False)
                print(f"Saved {len(df)} {position} projections to {filename}")

            healthy = df[~df["prediction_type"].str.contains("injured", na=False)]
            has_range = "floor" in df.columns
            print(f"\nTop {position}s for week {self.week}:")
            for _, p in healthy.head(10).iterrows():
                range_str = ""
                if has_range and not pd.isna(p.get("floor")):
                    range_str = f"  range {p['floor']:.0f}-{p['ceiling']:.0f}"
                print(f"  {p['rank']:3d}. {p['player_name']:24s} "
                      f"{p['fanduel_fantasy_points']:5.1f} pts{range_str} "
                      f"[{p['prediction_type']}]")

        return results
