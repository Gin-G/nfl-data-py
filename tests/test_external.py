"""Parsing and id-mapping for hand-exported third-party matchup tables.

Every fixture below is the real shape of a FantasyGuru week 9 2026 export,
including the two columns both headed `Team`.
"""
import pandas as pd
import pytest

from nfl_projections import external as ext

WR_CSV = (
    "Rank,Defensive Back,Team,Coverage Type,Advantage,Primary Receiver,Team\n"
    "1,Jason Marshall,MIA,45% Cover-3 Zone,65.38,Jameson Williams,DET\n"
    "2,Paulson Adebo,NYG,45% Quarters Zone,62.86,Darius Cooper,PHI\n"
    "3,Amik Robertson,WAS,85% Match Zone,60.49,Tutu Atwell,LAR\n"
    "4,Jaylon Johnson,CHI,55% Cover-1 Man,55.99,Emeka Egbuka,TB\n"
    "5,Travis Hunter,JAC,100% Shadow Coverage,51.20,Zay Flowers,BAL\n"
)

TEAM_CSV = ("Team,Offensive Line,Defensive Front\n"
            "Arizona,24.51,22.54\n"
            "Baltimore,35.97,38.69\n"
            "LA Chargers,40.10,31.00\n"
            "NY Jets,52.55,47.67\n"
            "Las Vegas,30.00,25.00\n"
            "Jacksonville,49.54,50.45\n")

LINE_CSV = ("AWAY,O-LINE (AWAY),D-LINE (AWAY),O-LINE ADV (AWAY),"
            "O-LINE ADV (HOME),O-LINE (HOME),D-LINE (HOME),HOME\n"
            "JAC,49.54,50.45,10.848,-14.486,35.97,38.69,BAL\n")


class TestTheDuplicateTeamColumn:
    """The trap: both the corner's team and the receiver's are called `Team`.

    csv.DictReader keeps the last and drops the corner's team without error;
    pandas.read_csv renames it `Team.1`. Both produce a frame that looks fine.
    """

    def test_both_teams_survive(self):
        df = ext.parse_wr_coverage(WR_CSV)
        assert df.defender_team.tolist() == ["MIA", "NYG", "WAS", "CHI", "JAX"]
        assert df.receiver_team.tolist() == ["DET", "PHI", "LAR", "TB", "BAL"]

    def test_dictreader_really_does_lose_it(self):
        """Pinning the reason this module reads by position."""
        import csv
        import io

        row = next(csv.DictReader(io.StringIO(WR_CSV)))
        assert row["Team"] == "DET"          # the RECEIVER's team
        assert "MIA" not in row.values()     # the corner's is simply gone

    def test_a_different_shape_is_refused_not_guessed(self):
        with pytest.raises(ValueError, match="not a FantasyGuru WR coverage"):
            ext.parse_wr_coverage("Rank,Corner,Team\n1,A,MIA\n")


class TestNormalizeTeam:
    def test_city_style_full_names(self):
        assert ext.normalize_team("LA Chargers") == "LAC"
        assert ext.normalize_team("NY Jets") == "NYJ"
        assert ext.normalize_team("Green Bay") == "GB"

    def test_the_two_abbreviations_that_disagree_with_nflverse(self):
        assert ext.normalize_team("JAC") == "JAX"
        assert ext.normalize_team("LVR") == "LV"

    def test_nflverse_abbreviations_pass_through(self):
        assert ext.normalize_team("BAL") == "BAL"
        assert ext.normalize_team("lac") == "LAC"

    def test_whitespace_and_case(self):
        assert ext.normalize_team("  green   bay ") == "GB"

    def test_an_ambiguous_name_is_refused(self):
        """"LA" is the Rams or the Chargers; guessing attaches a matchup to the
        wrong defence, which is worse than leaving it null."""
        assert ext.normalize_team("LA") is None

    def test_unknown_and_empty(self):
        assert ext.normalize_team("Toronto") is None
        assert ext.normalize_team("") is None
        assert ext.normalize_team(None) is None


class TestParseCoverageType:
    def test_a_zone_shell(self):
        out = ext.parse_coverage_type("45% Cover-3 Zone")
        assert out["shell_pct"] == 45.0
        assert out["shell"] == "COVER_3"
        assert out["man_zone"] == "ZONE_COVERAGE"
        assert out["shadow"] is False

    def test_a_man_shell(self):
        out = ext.parse_coverage_type("55% Cover-1 Man")
        assert (out["shell"], out["man_zone"]) == ("COVER_1", "MAN_COVERAGE")

    def test_quarters_is_cover_four(self):
        """Same shell under another name, so it compares with nflverse."""
        assert ext.parse_coverage_type("45% Quarters Zone")["shell"] == "COVER_4"
        assert ext.parse_coverage_type("55% Cover-4 Quarters")["shell"] == "COVER_4"

    def test_match_zone_keeps_its_own_value(self):
        """Pattern-match coverage charts as several shells; forcing it into one
        would invent agreement with nflverse that does not exist."""
        out = ext.parse_coverage_type("85% Match Zone")
        assert out["shell"] == "MATCH"
        assert out["man_zone"] == "ZONE_COVERAGE"

    def test_shadow_is_an_assignment_not_a_shell(self):
        out = ext.parse_coverage_type("100% Shadow Coverage")
        assert out["shadow"] is True
        assert out["shell_pct"] == 100.0
        assert out["shell"] is None and out["man_zone"] is None

    def test_the_raw_string_is_kept(self):
        assert ext.parse_coverage_type("45% Cover-3 Zone")["coverage_type"] == \
            "45% Cover-3 Zone"

    def test_blank(self):
        out = ext.parse_coverage_type("")
        assert out["shell"] is None and out["shadow"] is False


class TestParseTeamLines:
    def test_rows_and_mapping(self):
        df = ext.parse_team_lines(TEAM_CSV)
        assert len(df) == 6
        assert df.team.tolist() == ["ARI", "BAL", "LAC", "NYJ", "LV", "JAX"]
        assert df.offensive_line.iloc[1] == 35.97
        assert df.defensive_front.iloc[1] == 38.69

    def test_the_raw_spelling_is_kept(self):
        df = ext.parse_team_lines(TEAM_CSV)
        assert df.team_raw.iloc[2] == "LA Chargers"

    def test_wrong_shape_refused(self):
        with pytest.raises(ValueError, match="not a FantasyGuru team-lines"):
            ext.parse_team_lines("Team,OL,DL\nArizona,1,2\n")


class TestParseLineMatchups:
    def test_one_row_per_side_with_adv_rederived(self):
        df = ext.parse_line_matchups(LINE_CSV)
        assert len(df) == 2
        away, home = df.iloc[0], df.iloc[1]
        assert (away.team, away.opponent, away.home) == ("JAX", "BAL", False)
        assert (home.team, home.opponent, home.home) == ("BAL", "JAX", True)

    def test_adv_is_oline_minus_opposing_dfront(self):
        """The check that keeps this file from becoming its own version of the
        truth: it is derived from team_lines, so it must agree with it."""
        df = ext.parse_line_matchups(LINE_CSV)
        assert df.adv_agrees.all()
        assert df.adv_derived.tolist() == [10.85, -14.48]

    def test_a_disagreement_is_flagged_not_silently_kept(self):
        tampered = LINE_CSV.replace("10.848", "99.000")
        df = ext.parse_line_matchups(tampered)
        assert df.adv_agrees.tolist() == [False, True]


class TestDetectKind:
    def test_each_export_identifies_itself(self):
        assert ext.detect_kind(WR_CSV) == "wr_coverage"
        assert ext.detect_kind(TEAM_CSV) == "team_lines"
        assert ext.detect_kind(LINE_CSV) == "line_matchups"

    def test_something_else(self):
        assert ext.detect_kind("a,b,c\n1,2,3\n") is None
        assert ext.detect_kind("") is None

    def test_a_bom_does_not_hide_the_header(self):
        assert ext.detect_kind("﻿" + TEAM_CSV) == "team_lines"


class TestParseDispatch:
    def test_unknown_kind(self):
        with pytest.raises(ValueError, match="unknown export kind"):
            ext.parse("smash_tables", TEAM_CSV)

    def test_dispatches(self):
        assert len(ext.parse("team_lines", TEAM_CSV)) == 6


class TestAttachPlayerIds:
    def _roster(self):
        return pd.DataFrame([
            # Two active players of the same name: only the team separates them.
            {"full_name": "Justin Jefferson", "team": "MIN", "position": "WR",
             "gsis_id": "00-0036322"},
            {"full_name": "Justin Jefferson", "team": "SEA", "position": "LB",
             "gsis_id": "00-0041075"},
            # nflverse keeps the suffix; normalising strips it.
            {"full_name": "Marvin Harrison Jr.", "team": "ARI", "position": "WR",
             "gsis_id": "00-0039849"},
            # A one-letter spelling difference from the export.
            {"full_name": "Jacob Parrish", "team": "TB", "position": "DB",
             "gsis_id": "00-0040143"},
            {"full_name": "Zay Flowers", "team": "BAL", "position": "WR",
             "gsis_id": "00-0039337"},
        ])

    def test_team_separates_two_active_players_of_one_name(self):
        df = pd.DataFrame({"receiver": ["Justin Jefferson", "Justin Jefferson"],
                           "receiver_team": ["MIN", "SEA"]})
        out = ext.attach_player_ids(df, {"receiver": "receiver_team"},
                                    self._roster())
        assert out.receiver_id.tolist() == ["00-0036322", "00-0041075"]
        assert out.receiver_match.tolist() == ["team", "team"]

    def test_a_stripped_suffix_still_matches(self):
        df = pd.DataFrame({"receiver": ["Marvin Harrison Jr"],
                           "receiver_team": ["ARI"]})
        out = ext.attach_player_ids(df, {"receiver": "receiver_team"},
                                    self._roster())
        assert out.receiver_id.iloc[0] == "00-0039849"

    def test_a_near_miss_on_the_same_team_resolves_and_says_so(self):
        df = pd.DataFrame({"defender": ["Jacob Parish"],
                           "defender_team": ["TB"]})
        out = ext.attach_player_ids(df, {"defender": "defender_team"},
                                    self._roster())
        assert out.defender_id.iloc[0] == "00-0040143"
        # Never passed off as exact.
        assert out.defender_match.iloc[0] == "team_fuzzy"

    def test_a_near_miss_on_a_different_team_does_not_resolve(self):
        """The fuzzy fallback is scoped to one roster for exactly this reason."""
        df = pd.DataFrame({"defender": ["Jacob Parish"],
                           "defender_team": ["BAL"]})
        out = ext.attach_player_ids(df, {"defender": "defender_team"},
                                    self._roster())
        assert out.defender_id.iloc[0] is None
        assert out.defender_match.iloc[0] is None

    def test_an_unknown_name_keeps_a_null_id(self):
        df = pd.DataFrame({"receiver": ["Nobody At All"],
                           "receiver_team": ["BAL"]})
        out = ext.attach_player_ids(df, {"receiver": "receiver_team"},
                                    self._roster())
        assert out.receiver_id.iloc[0] is None

    def test_without_a_team_column_an_ambiguous_name_is_refused(self):
        df = pd.DataFrame({"receiver": ["Justin Jefferson", "Zay Flowers"]})
        out = ext.attach_player_ids(df, ["receiver"], self._roster())
        assert out.receiver_id.tolist() == [None, "00-0039337"]

    def test_an_empty_frame_is_returned_untouched(self):
        assert ext.attach_player_ids(pd.DataFrame(), ["receiver"],
                                     self._roster()).empty


class TestNothingFetches:
    """The source's terms are personal use only, so the only way data arrives
    is a human exporting a CSV. The absence of a fetcher is the feature, and
    these tests are what keep a well-meaning contributor from adding one."""

    # nflreadpy is deliberately absent: this module does load OUR OWN roster
    # data from nflverse to map names onto gsis ids. What it must never be able
    # to do is reach the third-party source itself.
    NETWORK_MODULES = {"requests", "urllib", "urllib3", "http", "httpx",
                       "aiohttp", "socket", "ftplib", "selenium",
                       "bs4", "playwright", "scrapy", "mechanize"}

    def _tree(self):
        import ast
        import inspect

        return ast.parse(inspect.getsource(ext))

    def test_no_network_library_is_imported(self):
        import ast

        imported = set()
        for node in ast.walk(self._tree()):
            if isinstance(node, ast.Import):
                imported |= {a.name.split(".")[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        assert not (imported & self.NETWORK_MODULES), (
            f"external.py imports {sorted(imported & self.NETWORK_MODULES)}; "
            "this module must not be able to fetch anything")

    def test_no_url_literal_outside_the_docs(self):
        """Prose may name the source; code may not hold an address for it."""
        import ast

        tree = self._tree()
        docstrings = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef)):
                doc = ast.get_docstring(node, clean=False)
                if doc:
                    docstrings.add(doc)
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                if node.value in docstrings:
                    continue
                assert "//" not in node.value and "www." not in node.value, (
                    f"URL-like literal in external.py: {node.value!r}")


class TestAFileIdentifiesItself:
    """Detection must not depend on the filename or on the header being row 1.

    These exports arrive called `table.csv`, `table (1).csv` and whatever the
    browser names the next one, so the only thing that can identify a file is
    what is inside it. Before this, the header had to be the very first row
    and every column had to be in a fixed position: a title row, a blank line
    or one added column and a perfectly good export was refused.
    """

    def test_a_title_row_above_the_header(self):
        text = ("FantasyGuru Weekly Matchups -- Week 9\n"
                "\n" + WR_CSV)
        assert ext.detect_kind(text) == "wr_coverage"
        frame = ext.parse("wr_coverage", text)
        plain = ext.parse("wr_coverage", WR_CSV)
        # A preamble must change nothing at all, not merely be survived.
        pd.testing.assert_frame_equal(frame, plain)

    def test_a_blank_leading_line_on_the_team_table(self):
        text = "\n\n" + TEAM_CSV
        assert ext.detect_kind(text) == "team_lines"
        pd.testing.assert_frame_equal(ext.parse("team_lines", text),
                                      ext.parse("team_lines", TEAM_CSV))

    def test_columns_in_a_different_order(self):
        text = ("Primary Receiver,Team,Advantage,Coverage Type,"
                "Defensive Back,Team,Rank\n"
                "Jameson Williams,DET,65.38,45% Cover-3 Zone,"
                "Jason Marshall,MIA,1\n")
        assert ext.detect_kind(text) == "wr_coverage"
        row = ext.parse("wr_coverage", text).iloc[0]
        # The point of the exercise: the receiver's Team column now comes
        # FIRST, so an absolute-position read would hand the corner DET.
        assert row.defender == "Jason Marshall"
        assert row.defender_team == "MIA"
        assert row.receiver == "Jameson Williams"
        assert row.receiver_team == "DET"
        assert row.fg_rank == 1

    def test_an_added_column_does_not_shift_anything(self):
        text = ("Rank,Defensive Back,Team,Coverage Type,Advantage,"
                "Primary Receiver,Team,Notes\n"
                "1,Jason Marshall,MIA,45% Cover-3 Zone,65.38,"
                "Jameson Williams,DET,some new field\n")
        row = ext.parse("wr_coverage", text).iloc[0]
        assert row.defender_team == "MIA"
        assert row.receiver_team == "DET"
        assert row.advantage == 65.38

    def test_a_column_inserted_between_the_two_team_columns(self):
        """The case absolute positions got wrong and anchoring gets right."""
        text = ("Rank,Defensive Back,Team,Coverage Type,Advantage,Snaps,"
                "Primary Receiver,Team\n"
                "1,Jason Marshall,MIA,45% Cover-3 Zone,65.38,312,"
                "Jameson Williams,DET\n")
        row = ext.parse("wr_coverage", text).iloc[0]
        assert row.defender_team == "MIA"
        assert row.receiver_team == "DET"

    def test_case_padding_and_underscores_in_the_header(self):
        text = ("  RANK , Defensive_Back ,TEAM, Coverage_Type ,ADVANTAGE,"
                "Primary_Receiver,Team\n"
                "1,Jason Marshall,MIA,45% Cover-3 Zone,65.38,"
                "Jameson Williams,DET\n")
        assert ext.detect_kind(text) == "wr_coverage"
        assert ext.parse("wr_coverage", text).iloc[0].defender_team == "MIA"

    def test_a_reordered_line_matchup_still_pairs_the_right_d_front(self):
        """ADV is own O-line minus OPPONENT D-front, so pairing by name matters."""
        text = ("HOME,O-LINE (HOME),D-LINE (HOME),O-LINE ADV (HOME),"
                "AWAY,O-LINE (AWAY),D-LINE (AWAY),O-LINE ADV (AWAY)\n"
                "BAL,35.97,38.69,-14.486,JAC,49.54,50.45,10.848\n")
        assert ext.detect_kind(text) == "line_matchups"
        frame = ext.parse("line_matchups", text)
        assert len(frame) == 2
        assert frame.adv_agrees.all(), frame.to_dict("records")

    def test_the_filename_is_never_consulted(self):
        """There is no filename argument to consult, and that is the point."""
        import inspect
        for fn in (ext.detect_kind, ext.parse,
                   ext.parse_wr_coverage, ext.parse_team_lines,
                   ext.parse_line_matchups):
            params = set(inspect.signature(fn).parameters)
            assert not {"filename", "name", "path"} & params, fn.__name__

    def test_a_header_too_far_down_is_still_refused(self):
        text = "\n".join(["junk line"] * 20) + "\n" + TEAM_CSV
        assert ext.detect_kind(text) is None

    def test_a_file_that_is_a_different_export_says_which(self):
        with pytest.raises(ValueError, match="is a team-lines export, not"):
            ext.parse("wr_coverage", TEAM_CSV)

    def test_one_team_column_is_refused_rather_than_guessed(self):
        """Losing a Team column must not quietly reuse the other one."""
        text = ("Rank,Defensive Back,Coverage Type,Advantage,"
                "Primary Receiver,Team\n"
                "1,Jason Marshall,45% Cover-3 Zone,65.38,Jameson Williams,DET\n")
        with pytest.raises(ValueError, match="no way to tell the corner's team"):
            ext.parse("wr_coverage", text)
