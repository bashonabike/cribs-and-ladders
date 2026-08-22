"""
Phase 5 (integration tests) -- board-build + Optimizer round trip.

Phase 3 covered `setBoardFromDb` against a temp `AllBoards.db`
(tests/test_board_setter.py) and Phase 4 covered `Optimizer.__init__`/
`retrievePairingsSettings` against a temp `Optimizer.db`
(tests/test_optimizer.py) -- but as two separate DBs exercised in
isolation. This file chains them: build a real Board from a temp
AllBoards.db, feed that board's real tracks into a real Optimizer
pointed at a *second* temp db (Optimizer.db), then run one full
parameter-adjustment cycle (runIncrIteration -> setBestIterParams ->
setupFminParamsList -> getFminStarterParams/getFminBounds) and check the
numbers come out consistent end to end.

Everything here resolves through `GameConfig(data_root=tmp_path)` --
`config.db_path` / `config.optimizer_db_path` -- rather than the
historical hardcoded 'Boards/AllBoards.db' / 'etc/Optimizer.db' literals
(see config.py's module docstring), which is what makes this runnable
against a disposable temp directory instead of a real machine's board
library.

Note: `setBoardFromDb`'s non-findmode branch (used here) does not
populate `Track.Track_ID` -- only the findmode branch does (see
BoardSetter.hydrate_tracks_from_dataframes / test_board_setter.py). So
this test keys Optimizer params by `track.num`, not `track.Track_ID`,
same as every real call site building `freshParams` would have to.
"""
import sqlite3

import pytest

from cribsandladders.Board import Board
from cribsandladders.BoardSetter import setBoardFromDb
from cribsandladders.config import GameConfig
from cribsandladders.Optimizer import Optimizer

pytestmark = pytest.mark.integration


def _create_board_schema(conn):
    conn.executescript(
        """
        CREATE TABLE Board (
            Board_ID INTEGER PRIMARY KEY,
            Board_Name TEXT,
            Num_Tracks INTEGER,
            Two_Deck INTEGER,
            Track1BoardPath TEXT,
            Track2BoardPath TEXT,
            Track3BoardPath TEXT,
            TwoDeckLineBoardPath TEXT,
            Width REAL,
            Height REAL
        );
        CREATE TABLE Track (
            Track_ID INTEGER PRIMARY KEY AUTOINCREMENT,
            Board_ID INTEGER,
            Num_On_Board INTEGER,
            Length INTEGER,
            Two_Deck_Length INTEGER,
            Colour TEXT
        );
        CREATE TABLE Chute (
            Chute_ID INTEGER PRIMARY KEY AUTOINCREMENT,
            Board_ID INTEGER,
            Track_ID INTEGER,
            Start INTEGER,
            End INTEGER
        );
        CREATE TABLE Ladder (
            Ladder_ID INTEGER PRIMARY KEY AUTOINCREMENT,
            Board_ID INTEGER,
            Track_ID INTEGER,
            Start INTEGER,
            End INTEGER
        );
        """
    )


def _create_optimizer_schema(conn):
    conn.executescript(
        """
        CREATE TABLE OptimizerParamPairings (
            Result TEXT, Param TEXT, Trackwise INTEGER, Inverse INTEGER, Active INTEGER
        );
        CREATE TABLE BoardTrackHints (
            Board_ID INTEGER, Track_ID INTEGER, Param TEXT, LBound REAL, UBound REAL, Active INTEGER
        );
        """
    )


def _build_board_from_temp_db(tmp_path, board_id=1):
    """Real Board, built via the real setBoardFromDb() I/O path against a
    throwaway AllBoards.db under tmp_path -- not a hand-built Board."""
    (tmp_path / "Boards").mkdir(exist_ok=True)
    config = GameConfig(data_root=tmp_path, findmode=False, twodecks=False)

    conn = sqlite3.connect(config.db_path)
    _create_board_schema(conn)
    conn.execute(
        "INSERT INTO Board (Board_ID, Board_Name, Num_Tracks, Two_Deck, Width, Height) "
        "VALUES (?, 'IntegrationTestBoard', 2, 0, 300.0, 400.0)",
        [board_id],
    )
    conn.execute("INSERT INTO Track (Board_ID, Num_On_Board, Length, Two_Deck_Length) VALUES (?, 1, 100, 150)",
                 [board_id])
    conn.execute("INSERT INTO Track (Board_ID, Num_On_Board, Length, Two_Deck_Length) VALUES (?, 2, 120, 180)",
                 [board_id])
    conn.execute("INSERT INTO Chute (Board_ID, Track_ID, Start, End) VALUES (?, 1, 40, 25)", [board_id])
    conn.execute("INSERT INTO Ladder (Board_ID, Track_ID, Start, End) VALUES (?, 2, 30, 55)", [board_id])
    conn.commit()
    conn.close()

    board = Board(config=config)
    setBoardFromDb(board, "IntegrationTestBoard", config=config)
    return board, config


def _seed_optimizer_db(config, board_id, lbound=0, ubound=50):
    """Second temp db, same tmp_path/data_root, used only by Optimizer."""
    (config.data_root / "etc").mkdir(exist_ok=True)
    conn = sqlite3.connect(config.optimizer_db_path)
    _create_optimizer_schema(conn)
    conn.execute(
        "INSERT INTO OptimizerParamPairings VALUES ('twohits', 'maxtwohitnetgainloss', 0, 0, 1)"
    )
    # A Trackwise=0 'ALL' pairing is what setupFminParamsList() collects
    # its param list from.
    conn.execute(
        "INSERT INTO OptimizerParamPairings VALUES ('ALL', 'maxtwohitnetgainloss', 0, 0, 1)"
    )
    conn.execute(
        "INSERT INTO BoardTrackHints VALUES (?, -1, 'maxtwohitnetgainloss', ?, ?, 1)",
        [board_id, lbound, ubound],
    )
    conn.commit()
    conn.close()


class TestBoardBuildAndOptimizerRoundTrip:
    def test_config_paths_resolve_under_the_temp_data_root_not_the_repo(self, tmp_path):
        # The whole point of GameConfig.data_root (see its module
        # docstring): no path here should ever touch the real repo's
        # Boards/AllBoards.db or etc/Optimizer.db.
        board, config = _build_board_from_temp_db(tmp_path)
        _seed_optimizer_db(config, board_id=board.boardID)

        assert str(config.db_path).startswith(str(tmp_path))
        assert str(config.optimizer_db_path).startswith(str(tmp_path))

    def test_board_loads_real_tracks_chutes_and_ladders_from_temp_db(self, tmp_path):
        board, _config = _build_board_from_temp_db(tmp_path)

        assert board.boardName == "IntegrationTestBoard"
        assert len(board.tracks) == 2
        track1 = board.getTrackByNum(1)
        track2 = board.getTrackByNum(2)
        assert [(c.start, c.end) for c in track1.chutes] == [(40, 25)]
        assert [(l.start, l.end) for l in track2.ladders] == [(30, 55)]
        # non-findmode hydration doesn't set Track_ID -- see module docstring.
        assert track1.Track_ID == 0

    def test_full_param_adjustment_cycle_against_temp_optimizer_db(self, tmp_path):
        board, config = _build_board_from_temp_db(tmp_path)
        _seed_optimizer_db(config, board_id=board.boardID, lbound=0, ubound=50)

        opt = Optimizer(board, optimizerRunSet=1, config=config)

        # __init__ already ran retrievePairingsSettings() against the temp
        # Optimizer.db -- confirm it actually read what we seeded, keyed
        # by this board's real (temp-db-assigned) boardID.
        assert opt.pairings_df.loc["twohits"]["Param"] == "maxtwohitnetgainloss"
        assert opt.absoluteBounds.loc["maxtwohitnetgainloss"]["UBound"] == 50

        freshParams = [
            {"track_id": t.num, "param": "maxtwohitnetgainloss", "value": v}
            for t, v in zip(board.tracks, (25, 30))
        ]
        freshResults = [
            {"Result": "twohits", "ResultValue": 0.5, "ResultValueIterative": 0.5, "Weighting": 30}
        ]

        adjusted = opt.runIncrIteration(freshParams, freshResults)

        # Hand-computed expected shift, same formula as
        # tests/test_optimizer.py::test_run_incr_iteration_adjusts_param_toward_bound_by_configured_increment:
        # changeAmt = (UBound - LBound) * changebaseincrperiter * (|ResultValue| * Weighting)
        change_amt = (50 - 0) * config.changebaseincrperiter * (abs(0.5) * 30)
        expected = {t.num: v - change_amt for t, v in zip(board.tracks, (25, 30))}

        assert len(adjusted) == 2
        by_track = {p["track_id"]: p["value"] for p in adjusted}
        for track_num, expected_val in expected.items():
            assert by_track[track_num] == pytest.approx(expected_val)
            # stayed within the bounds we seeded, not clamped away
            assert 0 <= by_track[track_num] <= 50

        # Round-trip the adjusted params back through the fmin-prep methods.
        opt.setBestIterParams(adjusted)
        opt.setupFminParamsList(sampleParams=adjusted)
        assert opt.fminParamsList == ["maxtwohitnetgainloss"]

        starters = opt.getFminStarterParams()
        assert sorted(starters) == sorted(by_track.values())

        bounds = opt.getFminBounds(adjusted)
        assert bounds == [(0, 50), (0, 50)]

    def test_out_of_range_result_clamps_instead_of_exceeding_seeded_bounds(self, tmp_path):
        board, config = _build_board_from_temp_db(tmp_path)
        _seed_optimizer_db(config, board_id=board.boardID, lbound=0, ubound=50)
        opt = Optimizer(board, optimizerRunSet=1, config=config)

        freshParams = [{"track_id": board.tracks[0].num, "param": "maxtwohitnetgainloss", "value": 25}]
        # A huge weighted result would push newVal outside [0, 50];
        # runIncrIteration should leave the value unchanged (paramMaxed)
        # rather than write an out-of-bounds number back.
        freshResults = [
            {"Result": "twohits", "ResultValue": 1000, "ResultValueIterative": 1000, "Weighting": 30}
        ]

        adjusted = opt.runIncrIteration(freshParams, freshResults)
        assert adjusted[0]["value"] == 25
