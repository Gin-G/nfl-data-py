"""Projected roster percentages: the parse, and the traps in using them."""

import warnings

import numpy as np
import pandas as pd
import pytest

from nfl_projections import ownership


def page(rows=None, *, qb=None, missing=()):
    """A stand-in for the Footballguys page, same shape as the real one."""
    qb = qb if qb is not None else [("Jordan Love", 60.0), ("C.J. Stroud", 39.1)]
    blocks = {
        "qb": qb,
        "rb": [("Jahmyr Gibbs", 35.1), ("Woody Marks", 31.5), ("Bucky Irving", 173.4)],
        "wr": [("Matthew Golden", 25.5), ("Puka Nacua", 294.5)],
        "te": [("George Kittle", 27.7), ("Sam LaPorta", 92.3)],
        "td": [("Tennessee Titans", 31.6), ("Cleveland Browns", 68.4)],
    }
    if rows is not None:
        blocks = rows
    html = ["<html><body>"]
    for table_id, entries in blocks.items():
        if table_id in missing:
            continue
        html.append(f"<p id='{table_id}' class='bar'>x</p><table class='table' id=\"data-{table_id}\">")
        for name, pct in entries:
            html.append(f'<tr> <td class="la">{name}</td>'
                        f'<td style="background:rgba(255,204,204,1);">{pct}%</td></tr>')
        html.append("</table>")
    html.append("</body></html>")
    return "\n".join(html)


class TestParse:
    def test_pulls_every_position_table(self):
        own = ownership.parse(page())
        assert len(own) == 11
        assert set(own.position) == {"QB", "RB", "WR", "TE", "D"}

    def test_keeps_the_percentage_as_a_percentage(self):
        own = ownership.parse(page())
        assert own.loc[own.name == "Jordan Love", "ownership"].iloc[0] == 60.0

    def test_defence_rows_keep_their_full_team_name(self):
        own = ownership.parse(page())
        # FanDuel's own CSV writes "Tennessee Titans" in Nickname, so these join
        # without a team-abbreviation lookup.
        assert "Tennessee Titans" in set(own.name)

    def test_adds_a_normalized_merge_key(self):
        own = ownership.parse(page())
        assert "cj stroud" in set(own.merge_name)

    def test_a_page_with_no_tables_is_empty_not_an_error(self):
        own = ownership.parse("<html><body><p>Available Friday.</p></body></html>")
        assert own.empty
        assert list(own.columns) == ["name", "position", "ownership", "merge_name"]

    def test_a_missing_table_does_not_lose_the_others(self):
        own = ownership.parse(page(missing=("wr",)))
        assert "WR" not in set(own.position)
        assert "QB" in set(own.position)


class TestTotals:
    def test_a_whole_field_passes(self):
        report = ownership.check_totals(ownership.parse(page()))
        assert all(r["ok"] for r in report.values())

    def test_half_a_field_is_caught(self):
        # One QB per lineup means QB ownership sums to 100%. Half of that means
        # we are looking at part of a slate, and would read every unlisted
        # player as a contrarian gem.
        own = ownership.parse(page(qb=[("Jordan Love", 30.0), ("C.J. Stroud", 19.0)]))
        assert ownership.check_totals(own)["QB"]["ok"] is False

    def test_load_warns_when_the_page_does_not_add_up(self):
        html = page(qb=[("Jordan Love", 30.0)])
        with pytest.warns(RuntimeWarning, match="full field"):
            ownership.load(7, html=html)

    def test_load_warns_when_nothing_is_published_yet(self):
        with pytest.warns(RuntimeWarning, match="not published yet|No roster"):
            own = ownership.load(8, html="<html><body>Friday.</body></html>")
        assert own.empty


class TestAttach:
    @staticmethod
    def pool():
        return pd.DataFrame({
            "Nickname": ["Jordan Love", "George Kittle", "Tennessee Titans", "Nobody Here"],
            "Position": ["QB", "TE", "D", "WR"],
            "fanduel_fantasy_points": [18.0, 11.0, 7.0, 14.0],
        })

    def test_matches_on_the_normalized_name(self):
        out = ownership.attach(self.pool(), ownership.parse(page()), warn=False)
        assert out.loc[0, "ownership"] == 60.0
        assert out.loc[2, "ownership"] == 31.6

    def test_an_unlisted_player_is_unknown_not_zero(self):
        # The source covers one slate, and inside it an omission means the field
        # has written the player off -- on 2026 week 6 every main-slate player
        # we projected above 10 points and the page did not list was inactive.
        # Reading that as 0% ownership turns a scratch into free leverage.
        out = ownership.attach(self.pool(), ownership.parse(page()), warn=False)
        assert np.isnan(out.loc[3, "ownership"])
        assert out.loc[3, "ownership_listed"] == False      # noqa: E712
        assert out.loc[0, "ownership_listed"] == True       # noqa: E712

    def test_a_floor_is_available_but_opt_in(self):
        out = ownership.attach(self.pool(), ownership.parse(page()), floor=0.1,
                              warn=False)
        assert out.loc[3, "ownership"] == 0.1

    def test_no_ownership_at_all_leaves_the_column_empty(self):
        out = ownership.attach(self.pool(), pd.DataFrame(
            columns=["name", "position", "ownership", "merge_name"]))
        assert out["ownership"].isna().all()
        assert not out["ownership_listed"].any()

    def test_a_wider_slate_than_the_source_is_called_out(self):
        # Footballguys projects the Sunday 13:00-16:25 games. Handing it a pool
        # that also holds Thursday, London, Sunday night and Monday leaves eight
        # teams unmatched, and every one of their players looking contrarian.
        wrong_slate = pd.DataFrame({
            "Nickname": ["Brock Purdy", "Dak Prescott", "Jordan Love", "C.J. Stroud"],
            "Position": ["QB"] * 4,
            "fanduel_fantasy_points": [19.6, 19.3, 17.7, 15.9],
        })
        with pytest.warns(RuntimeWarning, match="wider slate"):
            ownership.attach(wrong_slate, ownership.parse(page()))

    def test_a_matching_slate_is_quiet(self):
        matching = pd.DataFrame({
            "Nickname": ["Jordan Love", "George Kittle", "Tennessee Titans"],
            "Position": ["QB", "TE", "D"],
            "fanduel_fantasy_points": [18.0, 11.0, 7.0],
        })
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            ownership.attach(matching, ownership.parse(page()))

    def test_coverage_is_judged_on_the_players_the_board_rates(self):
        # A slate file's third-stringers are unlisted everywhere, so counting
        # them would condemn every pool.
        pool = pd.DataFrame({
            "Nickname": ["Jordan Love"] + [f"Scrub {i}" for i in range(200)],
            "Position": ["QB"] * 201,
            "fanduel_fantasy_points": [18.0] + [0.4] * 200,
        })
        out = ownership.attach(pool, ownership.parse(page()), warn=False)
        assert ownership.coverage(out) == {"players": 1, "listed": 1,
                                           "listed_share": 1.0, "bar": 6.0}

    def test_coverage_without_a_projection_column_is_unknown(self):
        out = ownership.attach(pd.DataFrame({"Nickname": ["Jordan Love"]}),
                               ownership.parse(page()), warn=False)
        assert np.isnan(ownership.coverage(out)["listed_share"])


class TestFlagUnlisted:
    def test_surfaces_players_the_board_likes_and_the_field_has_not(self):
        pool = pd.DataFrame({
            "Nickname": ["Lamar Jackson", "Jordan Love", "Deep Bench"],
            "ownership_listed": [False, True, False],
            "fanduel_fantasy_points": [16.0, 17.7, 1.2],
        })
        flagged = ownership.flag_unlisted(pool, threshold=10.0)
        assert list(flagged["Nickname"]) == ["Lamar Jackson"]

    def test_orders_by_how_much_the_board_likes_them(self):
        pool = pd.DataFrame({
            "Nickname": ["Breece Hall", "Lamar Jackson"],
            "ownership_listed": [False, False],
            "fanduel_fantasy_points": [11.6, 16.0],
        })
        assert list(ownership.flag_unlisted(pool)["Nickname"]) == ["Lamar Jackson", "Breece Hall"]


class TestLeverage:
    def test_is_positive_when_we_are_high_and_the_field_is_not(self):
        pool = pd.DataFrame({
            "Nickname": ["Ours", "Theirs"],
            "Position": ["WR", "WR"],
            "fanduel_fantasy_points": [20.0, 10.0],
            "ownership": [2.0, 30.0],
        })
        out = ownership.add_leverage(pool)
        assert out.loc[0, "leverage"] > 0
        assert out.loc[1, "leverage"] < 0

    def test_compares_within_position_only(self):
        # A quarterback outscoring a tight end is not leverage, it is the
        # position. Percentiles are taken inside each group.
        pool = pd.DataFrame({
            "Nickname": ["QB hi", "QB lo", "TE hi", "TE lo"],
            "Position": ["QB", "QB", "TE", "TE"],
            "fanduel_fantasy_points": [22.0, 15.0, 11.0, 6.0],
            "ownership": [30.0, 2.0, 25.0, 1.0],
        })
        out = ownership.add_leverage(pool)
        assert out.loc[1, "leverage"] == pytest.approx(out.loc[3, "leverage"])

    def test_flex_qualified_positions_group_with_their_base(self):
        pool = pd.DataFrame({
            "Nickname": ["a", "b"],
            "Position": ["WR/FLEX", "WR"],
            "fanduel_fantasy_points": [20.0, 10.0],
            "ownership": [1.0, 20.0],
        })
        out = ownership.add_leverage(pool)
        assert out["projection_pct"].tolist() == [1.0, 0.5]

    def test_unknown_ownership_gives_unknown_leverage(self):
        pool = pd.DataFrame({
            "Nickname": ["known", "unknown"],
            "Position": ["WR", "WR"],
            "fanduel_fantasy_points": [20.0, 18.0],
            "ownership": [10.0, np.nan],
        })
        out = ownership.add_leverage(pool)
        assert np.isnan(out.loc[1, "leverage"])
