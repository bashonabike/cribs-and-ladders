"""
Phase 3 (Board/geometry) tests for cribsandladders.BoardSetter.

setBoardFromDb used to be 100% sqlite -- no seam for testing the
"turn database rows into Track/Chute/Ladder objects" logic without a
real db file at a hardcoded 'Boards/AllBoards.db' path. The non-findmode
branch of that function has been pulled out into
`hydrate_tracks_from_dataframes`, which takes already-fetched
DataFrames and does no I/O at all, so most of this file tests that
directly with small hand-built DataFrames.

The findmode branch still does real INSERT/DELETE statements to
generate stub Track rows the first time a board is loaded (a real
persistence side effect, not just a read), so that part is covered
by an @pytest.mark.integration test against a temp sqlite db instead
of being extracted -- there's no pure subset of "delete then insert
autoincrement rows and read back their IDs" worth pulling out on its
own.
"""
import sqlite3

import pandas as pd
import pytest

from cribsandladders.Board import Board
from cribsandladders.BoardSetter import hydrate_tracks_from_dataframes, setBoardFromDb
from cribsandladders.config import GameConfig


# ---------------------------------------------------------------------
# hydrate_tracks_from_dataframes (pure, unit)
# ---------------------------------------------------------------------

def _tracks_df():
    return pd.DataFrame([
        {"Track_ID": 1, "tracknum": 1, "length": 100, "twodeck": 0, "twodecklength": 150},
        {"Track_ID": 2, "tracknum": 2, "length": 120, "twodeck": 1, "twodecklength": 180},
    ])


def test_hydrate_builds_one_track_per_row_with_single_deck_length():
    board = Board()
    hydrate_tracks_from_dataframes(
        board, _tracks_df(), pd.DataFrame(), pd.DataFrame(), config=GameConfig(twodecks=False)
    )
    assert [t.num for t in board.tracks] == [1, 2]
    assert [t.length for t in board.tracks] == [100, 120]
    # twodecks=False -> efflength uses plain length even for the track
    # that has a twodeck length available.
    assert [t.efflength for t in board.tracks] == [100, 120]


def test_hydrate_uses_two_deck_length_for_efflength_when_configured():
    board = Board()
    hydrate_tracks_from_dataframes(
        board, _tracks_df(), pd.DataFrame(), pd.DataFrame(), config=GameConfig(twodecks=True)
    )
    # Track 1's row has twodeck=0 -> twodeckslength mirrors plain length
    # regardless of config; track 2's row has twodeck=1 -> it gets the
    # real two-deck length. Both then use twodeckslength for efflength
    # because config.twodecks=True.
    assert [t.twodeckslength for t in board.tracks] == [100, 180]
    assert [t.efflength for t in board.tracks] == [100, 180]


def test_hydrate_falls_back_to_single_deck_length_when_track_has_no_two_deck_flag():
    # twodeck=0 on the row itself means twodeckslength mirrors length,
    # regardless of the global config.twodecks setting.
    df = pd.DataFrame([{"Track_ID": 1, "tracknum": 1, "length": 100, "twodeck": 0, "twodecklength": 999}])
    board = Board()
    hydrate_tracks_from_dataframes(board, df, pd.DataFrame(), pd.DataFrame(), config=GameConfig(twodecks=True))
    assert board.tracks[0].twodeckslength == 100
    assert board.tracks[0].efflength == 100


def test_hydrate_assigns_track_specific_chutes_and_ladders():
    chutes_df = pd.DataFrame([{"Track_ID": 1, "Chute_ID": 1, "start": 10, "end": 5}])
    ladders_df = pd.DataFrame([{"Track_ID": 2, "Ladder_ID": 1, "start": 20, "end": 30}])
    board = Board()
    hydrate_tracks_from_dataframes(board, _tracks_df(), chutes_df, ladders_df, config=GameConfig())

    track1 = board.getTrackByNum(1)
    track2 = board.getTrackByNum(2)
    assert [(c.start, c.end) for c in track1.chutes] == [(10, 5)]
    assert track1.ladders == []
    assert [(l.start, l.end) for l in track2.ladders] == [(20, 30)]
    assert track2.chutes == []
    # setEventChutes/setEventLadders/setEventImpedance were called too:
    assert track1.eventsListChute == [10]
    assert track2.eventsListLadder == [20]


def test_hydrate_shares_track_id_zero_events_across_all_tracks():
    # A chute/ladder row with Track_ID == 0 in the *domain object* sense
    # (i.e. curtrack.num == 0 semantics via the {0, curtrack.num} filter
    # in the merge step) applies to every track. We exercise that merge
    # directly since it only triggers when a chute/ladder's `.track`
    # attribute is 0, which happens when the owning track being iterated
    # has num == 0.
    tracks_df = pd.DataFrame([{"Track_ID": 5, "tracknum": 0, "length": 50, "twodeck": 0, "twodecklength": 50}])
    chutes_df = pd.DataFrame([{"Track_ID": 5, "Chute_ID": 1, "start": 8, "end": 2}])
    board = Board()
    hydrate_tracks_from_dataframes(board, tracks_df, chutes_df, pd.DataFrame(), config=GameConfig())
    assert len(board.tracks) == 1
    assert [(c.start, c.end) for c in board.tracks[0].chutes] == [(8, 2)]


def test_hydrate_with_no_chutes_or_ladders_leaves_tracks_eventless():
    board = Board()
    hydrate_tracks_from_dataframes(board, _tracks_df(), pd.DataFrame(), pd.DataFrame(), config=GameConfig())
    for t in board.tracks:
        assert t.chutes == []
        assert t.ladders == []
        assert t.simplEventImpedance == 0


# ---------------------------------------------------------------------
# setBoardFromDb -- integration (real sqlite db at config.db_path)
# ---------------------------------------------------------------------

def _make_config_pointing_at(tmp_path, **kwargs):
    boards_dir = tmp_path / "Boards"
    boards_dir.mkdir(exist_ok=True)
    return GameConfig(data_root=tmp_path, **kwargs)


def _create_schema(conn):
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


@pytest.mark.integration
def test_set_board_from_db_non_findmode_loads_tracks_chutes_and_ladders(tmp_path):
    config = _make_config_pointing_at(tmp_path, findmode=False, twodecks=False)
    conn = sqlite3.connect(config.db_path)
    _create_schema(conn)
    conn.execute(
        "INSERT INTO Board (Board_ID, Board_Name, Num_Tracks, Two_Deck, Width, Height) "
        "VALUES (1, 'TestBoard', 2, 0, 300.0, 400.0)"
    )
    conn.execute(
        "INSERT INTO Track (Board_ID, Num_On_Board, Length, Two_Deck_Length) VALUES (1, 1, 100, 150)"
    )
    conn.execute(
        "INSERT INTO Track (Board_ID, Num_On_Board, Length, Two_Deck_Length) VALUES (1, 2, 120, 180)"
    )
    conn.execute("INSERT INTO Chute (Board_ID, Track_ID, Start, End) VALUES (1, 1, 10, 5)")
    conn.execute("INSERT INTO Ladder (Board_ID, Track_ID, Start, End) VALUES (1, 2, 20, 30)")
    conn.commit()
    conn.close()

    board = Board(config=config)
    setBoardFromDb(board, "TestBoard", config=config)

    assert board.boardName == "TestBoard"
    assert board.width == 300.0
    assert board.height == 400.0
    assert len(board.tracks) == 2
    track1 = board.getTrackByNum(1)
    assert [(c.start, c.end) for c in track1.chutes] == [(10, 5)]


@pytest.mark.integration
def test_set_board_from_db_findmode_generates_stub_tracks_when_none_exist(tmp_path):
    config = _make_config_pointing_at(tmp_path, findmode=True)
    conn = sqlite3.connect(config.db_path)
    _create_schema(conn)
    conn.execute(
        "INSERT INTO Board (Board_ID, Board_Name, Num_Tracks, Two_Deck, "
        "Track1BoardPath, Track2BoardPath, Width, Height) "
        "VALUES (1, 'StubBoard', 2, 0, 'track1.svg', 'track2.svg', 300.0, 400.0)"
    )
    conn.commit()
    conn.close()

    board = Board(config=config)
    setBoardFromDb(board, "StubBoard", config=config)

    assert len(board.tracks) == 2
    assert {t.num for t in board.tracks} == {1, 2}
    assert board.getTrackByNum(1).holesetfilepath == "track1.svg"
    assert board.getTrackByNum(2).holesetfilepath == "track2.svg"
    # Track_IDs were assigned by the db (autoincrement), not left as 0.
    assert all(t.Track_ID for t in board.tracks)


@pytest.mark.integration
def test_set_board_from_db_raises_when_db_file_missing(tmp_path):
    config = GameConfig(data_root=tmp_path)  # Boards/AllBoards.db doesn't exist
    board = Board(config=config)
    with pytest.raises(Exception):
        setBoardFromDb(board, "NoSuchBoard", config=config)


@pytest.mark.integration
def test_set_board_from_db_raises_when_board_name_not_found(tmp_path):
    config = _make_config_pointing_at(tmp_path)
    conn = sqlite3.connect(config.db_path)
    _create_schema(conn)
    conn.commit()
    conn.close()

    board = Board(config=config)
    with pytest.raises(Exception):
        setBoardFromDb(board, "DoesNotExist", config=config)
