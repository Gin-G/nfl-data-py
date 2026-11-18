"""ESPN's injuries feed as an in-week overlay on the nflverse report.

2026 week 10: Kenneth Walker was Questionable on Friday's report, ruled out on
Sunday, and nflverse still said Questionable after the game. ESPN had him Out.
"""
import pandas as pd
import pytest

from nfl_projections import injuries

SINCE = pd.Timestamp("2026-11-11T10:00Z")
UNTIL = pd.Timestamp("2026-11-18T10:00Z")


def _entry(name, espn_id, team, pos, status, date, injury="Groin"):
    return {
        "status": status, "date": date, "shortComment": f"{name} news",
        "details": {"type": injury},
        "athlete": {
            "displayName": name,
            "links": [{"href": f"https://www.espn.com/nfl/player/_/id/{espn_id}/x"}],
            "position": {"abbreviation": pos},
            "team": {"abbreviation": team},
        },
    }


def _payload(*entries):
    return {"injuries": [{"displayName": "Everyone", "injuries": list(entries)}]}


CROSSWALK = {"4567048": "00-0038134", "1111": "00-0011111"}


class TestParse:
    def test_maps_status_team_and_id(self):
        frame = injuries.parse_espn_injuries(_payload(
            _entry("Kenneth Walker III", 4567048, "KC", "RB", "Out", "2026-11-15T16:30Z"),
            _entry("Some Receiver", 1111, "WSH", "WR", "Injured Reserve", "2026-11-12T00:00Z"),
            _entry("A Lineman", 2222, "LAR", "G", "Out", "2026-11-12T00:00Z"),
            _entry("Odd Status", 3333, "LAR", "TE", "Day-To-Day", "2026-11-12T00:00Z"),
        ), CROSSWALK)
        assert list(frame.player_name) == ["Kenneth Walker III", "Some Receiver"]
        walker, wr = frame.iloc[0], frame.iloc[1]
        assert walker.status == "OUT" and walker.player_id == "00-0038134"
        assert walker.espn_id == "4567048" and walker.updated == pd.Timestamp(
            "2026-11-15T16:30Z")
        assert wr.team == "WAS" and wr.status == "OUT"       # ESPN's WSH, IR

    def test_an_unmapped_id_keeps_an_empty_gsis(self):
        frame = injuries.parse_espn_injuries(_payload(
            _entry("New Guy", 9999, "LAR", "WR", "Questionable", "2026-11-12T00:00Z")))
        assert frame.iloc[0].player_id == "" and frame.iloc[0].team == "LA"

    def test_an_empty_or_missing_payload(self):
        assert injuries.parse_espn_injuries({}).empty
        assert injuries.parse_espn_injuries(None).empty


class TestWindow:
    def test_from_the_day_after_last_week_to_the_day_after_this_one(self):
        sch = pd.DataFrame({"season": 2026, "game_type": "REG",
                            "week": [9, 9, 10, 10, 10],
                            "gameday": ["2026-11-05", "2026-11-09", "2026-11-12",
                                        "2026-11-15", "2026-11-16"]})
        since, until = injuries.week_window(2026, 10, sch)
        assert since == pd.Timestamp("2026-11-10T10:00Z")
        assert until == pd.Timestamp("2026-11-17T10:00Z")
        assert injuries.week_window(2026, 11, sch) is None


def _espn(*rows):
    return pd.DataFrame([{"player_name": n, "team": t, "position": p, "status": s,
                          "injury_type": "Groin", "espn_id": "", "player_id": pid,
                          "updated": pd.Timestamp(d), "comment": ""}
                         for n, t, p, s, pid, d in rows])


REPORT = {
    "Kenneth Walker III": {"status": "QUESTIONABLE", "injury_type": "Groin",
                           "position": "RB", "team": "KC",
                           "practice_status": "Did Not Participate",
                           "player_id": "00-0038134"},
    "Doubtful Dan": {"status": "DOUBTFUL", "injury_type": "Ankle", "position": "WR",
                     "team": "LA", "practice_status": "", "player_id": "00-2"},
}


class TestOverlay:
    def test_sunday_out_benches_a_friday_questionable(self):
        merged, changes = injuries.overlay_espn(REPORT, _espn(
            ("Kenneth Walker", "KC", "RB", "OUT", "00-0038134", "2026-11-15T16:30Z")),
            SINCE, UNTIL)
        assert merged["Kenneth Walker III"]["status"] == "OUT"     # by id, not name
        assert merged["Kenneth Walker III"]["practice_status"] == "Did Not Participate"
        assert changes == [{"player": "Kenneth Walker III", "team": "KC",
                            "position": "RB", "player_id": "00-0038134",
                            "from": "QUESTIONABLE", "to": "OUT"}]
        assert REPORT["Kenneth Walker III"]["status"] == "QUESTIONABLE"   # not mutated

    def test_news_outside_the_week_changes_nothing(self):
        for when in ("2026-11-10T09:00Z", "2026-11-18T12:00Z"):
            merged, changes = injuries.overlay_espn(REPORT, _espn(
                ("Kenneth Walker III", "KC", "RB", "OUT", "00-0038134", when)),
                SINCE, UNTIL)
            assert merged == REPORT and changes == []

    def test_active_clears_a_doubtful(self):
        merged, changes = injuries.overlay_espn(REPORT, _espn(
            ("Doubtful Dan", "LA", "WR", "ACTIVE", "", "2026-11-15T15:00Z")), SINCE, UNTIL)
        assert "Doubtful Dan" not in merged
        assert changes[0]["from"] == "DOUBTFUL" and changes[0]["to"] == "ACTIVE"

    def test_active_for_someone_never_listed_is_no_change(self):
        merged, changes = injuries.overlay_espn(REPORT, _espn(
            ("Healthy Hal", "KC", "TE", "ACTIVE", "", "2026-11-15T15:00Z")), SINCE, UNTIL)
        assert merged == REPORT and changes == []

    def test_a_player_off_the_report_is_added(self):
        merged, changes = injuries.overlay_espn(REPORT, _espn(
            ("Late Scratch", "KC", "WR", "OUT", "00-9", "2026-11-15T16:00Z")), SINCE, UNTIL)
        assert merged["Late Scratch"]["status"] == "OUT"
        assert merged["Late Scratch"]["source"] == "espn"
        assert changes[0]["from"] is None

    def test_the_same_status_is_not_a_change(self):
        merged, changes = injuries.overlay_espn(REPORT, _espn(
            ("Doubtful Dan", "LA", "WR", "DOUBTFUL", "00-2", "2026-11-14T15:00Z")),
            SINCE, UNTIL)
        assert changes == [] and merged == REPORT


class TestForWeek:
    def test_a_failed_fetch_leaves_the_report_alone(self, monkeypatch):
        sch = pd.DataFrame({"season": 2026, "game_type": "REG", "week": [9, 10],
                            "gameday": ["2026-11-09", "2026-11-15"]})

        def boom(*a, **k):
            raise ConnectionError("espn down")
        monkeypatch.setattr(injuries, "fetch_espn_payload", boom)
        merged, prov = injuries.espn_overlay_for_week(REPORT, 2026, 10, sch)
        assert merged is REPORT and "espn down" in prov["espn_error"]

    def test_the_overlay_runs_only_when_asked(self, monkeypatch):
        calls = []
        monkeypatch.setattr(injuries, "_nflverse_injuries_with_coverage",
                            lambda s, w: (dict(REPORT), {"report_available": True}))
        monkeypatch.setattr(injuries, "espn_overlay_for_week",
                            lambda inj, s, w, sch: (calls.append(w) or inj,
                                                    {"espn_rows": 0, "espn_fresh": 0,
                                                     "espn_changes": []}))
        roster = pd.DataFrame(columns=["player_name", "team", "position", "status"])
        depth = pd.DataFrame(columns=["team", "pos_abb", "pos_rank", "player_name"])
        _, _, prov = injuries.integrate_injuries_detailed(10, roster, depth, 2026)
        assert calls == [] and prov["source"] == "nflverse"
        _, _, prov = injuries.integrate_injuries_detailed(
            10, roster, depth, 2026, source="nflverse+espn")
        assert calls == [10] and prov["source"] == "nflverse+espn"
        assert prov["espn_changes"] == []


@pytest.mark.parametrize("raw,norm", [("Out", "OUT"), ("Doubtful", "DOUBTFUL"),
                                      ("Questionable", "QUESTIONABLE"),
                                      ("Active", "ACTIVE")])
def test_status_names(raw, norm):
    frame = injuries.parse_espn_injuries(_payload(
        _entry("X", 1, "KC", "QB", raw, "2026-11-12T00:00Z")))
    assert frame.iloc[0].status == norm
