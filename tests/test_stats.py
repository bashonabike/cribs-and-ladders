"""
Phase 4 (Optimizer/board-design subsystem) tests for cribsandladders.Stats.

Stats.py is fully importable in a minimal environment (pandas, seaborn,
matplotlib, stdlib -- no scipy/shapely/mystic/compiled extensions), so
unlike Evaluator/Optimizer/EventSetBuilder this file is exercised
end-to-end, not just at the seams.

Covers:
- `Stats`/`Move` now take/use an injected `config: GameConfig` instead
  of reading `game_params` directly.
- `build_insert_stat_stub`, pulled out of `insertStatsRecord` (where it
  used to read the lazily-computed `gp.insertstatstub` global -- itself
  built from a *second*, separate sqlite connection opened behind the
  scenes by game_params.py). It's now a plain function over an explicit
  cursor, unit-tested against a temp sqlite db with a matching `Stat`
  table instead of the real one.
- `insertStatsRecord` end-to-end against a temp sqlite db built at
  `config.db_path` (via `data_root`), replacing the previously
  hardcoded `'Boards/AllBoards.db'` (which didn't even agree with the
  module's own `boardDBName`-style literal -- same class of bug fixed
  in BoardSetter during Phase 3).
- `calc_metrics`, which is pure computation over `self.moves`/
  `self.board` plus `config.numtrials` -- run against a small
  synthetic set of `Move`s.
- `print_metrics`/`print_temp_maps` gained an `output_dir` parameter
  (replacing hardcoded `"./Board_Results"` / `"./Board_Results/images"`)
  so tests could point them at a temp directory. Both methods have
  pre-existing bugs unrelated to this migration (see the TODO comments
  left in Stats.py) that make them raise before writing anything, so
  they aren't exercised end-to-end here -- see
  `test_print_metrics_and_print_temp_maps_have_known_pre_existing_bugs`
  for what's confirmed broken and why that's out of scope for Phase 4.
"""
import sqlite3
from pathlib import Path

import pandas as pd
import pytest

import Enums as en
from cribsandladders.Board import Board, Track
from cribsandladders.config import GameConfig
from cribsandladders.Stats import Stats, Move, build_insert_stat_stub


class FakePlayer:
    def __init__(self, num, tracknum, wins=0):
        self.num = num
        self.tracknum = tracknum
        self.wins = wins


class FakeSquad:
    def __init__(self, players):
        self.players = players


def _board_with_one_track():
    board = Board()
    board.boardID = 1
    board.boardName = "TestBoard"
    t1 = Track()
    t1.num = 1
    board.tracks = [t1]
    return board, t1


# ---------------------------------------------------------------------
# Move -- config-independent, but shares this module
# ---------------------------------------------------------------------

def test_move_computes_ladder_amount_on_ladder_event():
    track = Track()
    track.num = 1
    track.Track_ID = 1
    m = Move(0, 0, track, 1, 1, 1, oldScore=0, baseScore=0, reason="peg",
             event=en.Event.LADDER, newScore=10, soexcite=False, pegMove=True)
    assert m.ladderamt == 10
    assert m.chuteamt == 0
    assert m.ladderorchuteamt == 10
    assert m.hasEvent is True


def test_move_computes_chute_amount_on_chute_event():
    track = Track()
    track.num = 1
    track.Track_ID = 1
    m = Move(0, 0, track, 1, 1, 1, oldScore=10, baseScore=0, reason="peg",
             event=en.Event.CHUTE, newScore=2, soexcite=False, pegMove=True)
    assert m.chuteamt == -8
    assert m.ladderamt == 0
    assert m.hasEvent is True


def test_move_to_dict_round_trips_key_fields():
    track = Track()
    track.num = 1
    track.Track_ID = 1
    m = Move(0, 0, track, 1, 1, 1, oldScore=0, baseScore=0, reason="peg",
             event=en.Event.NONE, newScore=5, soexcite=True, pegMove=True)
    d = m.to_dict()
    assert d["track"] == 1
    assert d["score"] == 5
    assert d["soexcite"] is True


# ---------------------------------------------------------------------
# Stats -- config injection
# ---------------------------------------------------------------------

def test_stats_stores_injected_config():
    board, _ = _board_with_one_track()
    squad = FakeSquad([FakePlayer(1, 1)])
    config = GameConfig(numtrials=50, numplayers=2)
    stats = Stats(board, squad, optimizerRunSet=1, optimizerRun=1, config=config)
    assert stats.config is config


def test_stats_defaults_to_default_config_when_not_given():
    board, _ = _board_with_one_track()
    squad = FakeSquad([FakePlayer(1, 1)])
    stats = Stats(board, squad, 1, 1)
    from cribsandladders.config import DEFAULT_CONFIG
    assert stats.config is DEFAULT_CONFIG


def test_clear_stats_and_set_moves_resets_and_sets_moves():
    board, t1 = _board_with_one_track()
    squad = FakeSquad([FakePlayer(1, 1)])
    stats = Stats(board, squad, 1, 1, config=GameConfig())
    stats.ladders = 99
    moves = [Move(0, 0, t1, 1, 1, 1, 0, 0, "peg", en.Event.NONE, 3, False, True)]
    stats.clearStatsAndSetMoves(moves)
    assert stats.ladders == 0
    assert stats.moves == moves


# ---------------------------------------------------------------------
# calc_metrics -- pure computation, uses config.numtrials
# ---------------------------------------------------------------------

def _two_move_game(track):
    return [
        Move(0, 0, track, 1, 1, 1, oldScore=0, baseScore=0, reason="peg",
             event=en.Event.NONE, newScore=5, soexcite=False, pegMove=True),
        Move(0, 0, track, 2, 1, 2, oldScore=0, baseScore=0, reason="peg",
             event=en.Event.NONE, newScore=3, soexcite=False, pegMove=True),
    ]


def test_calc_metrics_runs_without_game_params_and_uses_config_numtrials():
    board, t1 = _board_with_one_track()
    squad = FakeSquad([FakePlayer(1, 1), FakePlayer(2, 1)])
    config = GameConfig(numtrials=2, numplayers=2)
    stats = Stats(board, squad, 1, 1, config=config)
    stats.clearStatsAndSetMoves(_two_move_game(t1))

    stats.calc_metrics()

    # No events configured on this bare track -> zero ladders/chutes,
    # but avglengthinrounds should reflect config.numtrials=2 dividing
    # the max round (1) summed once.
    assert stats.ladders == 0
    assert stats.chutes == 0
    assert stats.avglengthinrounds == pytest.approx(1 / 2)


def test_calc_metrics_scales_inversely_with_numtrials():
    board, t1 = _board_with_one_track()
    squad = FakeSquad([FakePlayer(1, 1), FakePlayer(2, 1)])

    stats_a = Stats(board, squad, 1, 1, config=GameConfig(numtrials=1, numplayers=2))
    stats_a.clearStatsAndSetMoves(_two_move_game(t1))
    stats_a.calc_metrics()

    board2, t2 = _board_with_one_track()
    stats_b = Stats(board2, squad, 1, 1, config=GameConfig(numtrials=4, numplayers=2))
    stats_b.clearStatsAndSetMoves(_two_move_game(t2))
    stats_b.calc_metrics()

    assert stats_a.avglengthinrounds == pytest.approx(stats_b.avglengthinrounds * 4)


# ---------------------------------------------------------------------
# build_insert_stat_stub
# ---------------------------------------------------------------------

def test_build_insert_stat_stub_lists_columns_excluding_stat_id():
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE Stat (Stat_ID INTEGER PRIMARY KEY, Board_ID INTEGER, NumTrials INTEGER)")
    stub = build_insert_stat_stub(conn.cursor())
    assert stub == "INSERT INTO Stat (Board_ID,NumTrials) Values ("
    conn.close()


# ---------------------------------------------------------------------
# insertStatsRecord -- integration, real temp sqlite db
# ---------------------------------------------------------------------

_STAT_SCHEMA = """
CREATE TABLE Stat (
    Stat_ID INTEGER PRIMARY KEY AUTOINCREMENT,
    Board_ID INTEGER, Timestamp TEXT, NumTrials INTEGER, NumPlayers INTEGER, NumDecks INTEGER,
    TracksUsed TEXT, BoardName TEXT, AvgLengthRounds REAL,
    Bal1 REAL, Bal2 REAL, Bal3 REAL, Bal4 REAL,
    SoExT1 REAL, SoExT2 REAL, SoExT3 REAL, SoExT4 REAL, SoExOverall REAL,
    Rep1 REAL, Rep2 REAL, Rep3 REAL, Rep4 REAL, RepOverall REAL,
    Chu1 REAL, Chu2 REAL, Chu3 REAL, Chu4 REAL, ChuOverall REAL,
    Lad1 REAL, Lad2 REAL, Lad3 REAL, Lad4 REAL, LadOverall REAL,
    Ev1 REAL, Ev2 REAL, Ev3 REAL, Ev4 REAL, EvOverall REAL,
    ChuIn1_1 REAL, ChuIn1_2 REAL, ChuIn1_3 REAL, ChuIn1_4 REAL, ChuIn1Overall REAL,
    LadIn1_1 REAL, LadIn1_2 REAL, LadIn1_3 REAL, LadIn1_4 REAL, LadIn1Overall REAL,
    EvIn1_1 REAL, EvIn1_2 REAL, EvIn1_3 REAL, EvIn1_4 REAL, EvIn1Overall REAL,
    ChuIn2_1 REAL, ChuIn2_2 REAL, ChuIn2_3 REAL, ChuIn2_4 REAL, ChuIn2Overall REAL,
    LadIn2_1 REAL, LadIn2_2 REAL, LadIn2_3 REAL, LadIn2_4 REAL, LadIn2Overall REAL,
    EvIn2_1 REAL, EvIn2_2 REAL, EvIn2_3 REAL, EvIn2_4 REAL, EvIn2Overall REAL
);
"""


def _config_with_stat_table(tmp_path):
    (tmp_path / "Boards").mkdir()
    config = GameConfig(data_root=tmp_path, numtrials=2, numplayers=2)
    conn = sqlite3.connect(config.db_path)
    conn.executescript(_STAT_SCHEMA)
    conn.commit()
    conn.close()
    return config


@pytest.mark.integration
def test_insert_stats_record_writes_a_row_to_temp_db(tmp_path):
    config = _config_with_stat_table(tmp_path)
    board, t1 = _board_with_one_track()
    squad = FakeSquad([FakePlayer(1, 1), FakePlayer(2, 1)])
    stats = Stats(board, squad, 1, 1, config=config)
    stats.clearStatsAndSetMoves(_two_move_game(t1))
    stats.calc_metrics()

    stats.insertStatsRecord()

    conn = sqlite3.connect(config.db_path)
    row = conn.execute("SELECT Board_ID, BoardName, NumTrials, NumPlayers FROM Stat").fetchone()
    conn.close()
    assert row == (1, "TestBoard", 2, 2)


@pytest.mark.integration
def test_insert_stats_record_raises_on_invalid_tracksused(tmp_path):
    config = _config_with_stat_table(tmp_path)
    config.tracksused = "not-a-list-or-none"  # neither None nor `list`
    board, t1 = _board_with_one_track()
    squad = FakeSquad([FakePlayer(1, 1)])
    stats = Stats(board, squad, 1, 1, config=config)
    stats.clearStatsAndSetMoves(_two_move_game(t1))
    stats.calc_metrics()

    with pytest.raises(Exception):
        stats.insertStatsRecord()


# ---------------------------------------------------------------------
# Known pre-existing bugs in print_metrics / print_temp_maps
# ---------------------------------------------------------------------

def test_print_metrics_and_print_temp_maps_have_known_pre_existing_bugs(tmp_path):
    """Not a regression test for Phase 4 -- documents that both methods
    are currently unreachable due to bugs unrelated to the config
    migration (see the TODO comments on each in Stats.py):

    - print_metrics reads self.soexcites_pegs / self.lengths_in_rounds,
      neither of which Stats ever sets.
    - print_temp_maps calls .to_list() on a DataFrameGroupBy, which
      doesn't have that method.

    Both still accept the new `output_dir` param fine -- they just
    never get far enough to use it. Pinned down here so a future fix
    has a starting point instead of rediscovering this from scratch.
    """
    board, t1 = _board_with_one_track()
    squad = FakeSquad([FakePlayer(1, 1, wins=1), FakePlayer(2, 1)])
    stats = Stats(board, squad, 1, 1, config=GameConfig(numtrials=2, numplayers=2))
    stats.clearStatsAndSetMoves(_two_move_game(t1))
    stats.calc_metrics()

    with pytest.raises(AttributeError):
        stats.print_metrics(output_dir=str(tmp_path))

    stats.hist_df = pd.DataFrame({"normmove": [0.1], "ladderorchuteamt": [1]})
    stats.hist_by_track_df = pd.DataFrame({"track": [1], "normmove": [0.1], "ladderorchuteamt": [1]})
    with pytest.raises(AttributeError):
        stats.print_temp_maps(output_dir=str(tmp_path))
