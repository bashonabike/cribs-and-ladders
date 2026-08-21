"""
Phase 3 (Board/geometry) tests for cribsandladders.Board.

Track's own methods (`setEffLandingForHoles`, `setEventImpedance`,
`getHoleByCoords`, `getHoleByNum`, the `getXAsDF` helpers) were already
pure in-memory logic with no I/O -- they just needed tests. Board's
`setBoardAfterSetter` is covered here for a different reason: it's the
regression test for the Phase 3 import-hygiene fix that made the
`cribsandladders.PossibleEvents` import lazy (moved inside the
findmode branch) so `import cribsandladders.Board` no longer requires
matplotlib to be installed. `test_set_board_after_setter_does_not_import_possible_events_when_findmode_is_false`
below asserts that directly via sys.modules.
"""
import sys

import pandas as pd
import pytest

from cribsandladders.Board import Board, Track, Event, Ladder, Chute
from cribsandladders.config import GameConfig


# ---------------------------------------------------------------------
# Track.setEffLandingForHoles
# ---------------------------------------------------------------------

def test_set_eff_landing_for_holes_without_trackholes_falls_back_to_plain_range():
    t = Track()
    t.length = 5
    t.trackholes = None
    t.setEffLandingForHoles()
    assert t.effLandingForHoles == [1, 2, 3, 4, 5]


def test_set_eff_landing_for_holes_with_no_events_lands_on_self():
    t = Track()
    t.trackholes = [object()] * 4  # only length matters here
    t.eventsListChute = []
    t.eventsListLadder = []
    t.setEffLandingForHoles()
    assert t.effLandingForHoles == [1, 2, 3, 4]


def test_set_eff_landing_for_holes_redirects_through_ladder_but_not_chute():
    """Characterization test, not a correctness claim.

    The ladder branch redirects to `ladder.end` (2 -> 5, forward), which
    is what you'd expect. The chute branch redirects to `chute.start`
    (Board.py, setEffLandingForHoles) instead of `chute.end` -- since
    eventsListChute is built from chute *start* positions too, landing
    on a chute's start hole maps back to that same start hole, i.e. a
    no-op. That looks like a copy-paste asymmetry with the ladder
    branch rather than intentional behavior, but fixing game-logic bugs
    is out of scope for Phase 3 (testability/extraction); this test
    pins down the current behavior as found so a future fix has a
    safety net and an explicit note of what changes.
    """
    t = Track()
    t.trackholes = [object()] * 6
    chute = Chute(4, 1, track=1)
    ladder = Ladder(2, 5, track=1)
    t.chutes = [chute]
    t.eventsListChute = [4]
    t.ladders = [ladder]
    t.eventsListLadder = [2]
    t.setEffLandingForHoles()
    # hole4 (chute trigger) currently maps to chute.start (4), i.e. itself.
    assert t.effLandingForHoles == [1, 5, 3, 4, 5, 6]


# ---------------------------------------------------------------------
# Track.setEventImpedance
# ---------------------------------------------------------------------

def test_set_event_impedance_combines_ladder_and_chute_lengths():
    t = Track()
    t.efflength = 100
    t.ladders = [Ladder(1, 10, track=1)]  # length 9
    t.chutes = [Chute(20, 15, track=1)]  # length -5
    t.setEventImpedance()
    # (sumLadders + sumChutes) * numEvents / efflength = (9 + -5) * 2 / 100
    assert t.simplEventImpedance == pytest.approx((9 + -5) * 2 / 100)


def test_set_event_impedance_zero_with_no_events():
    t = Track()
    t.efflength = 50
    t.setEventImpedance()
    assert t.simplEventImpedance == 0


# ---------------------------------------------------------------------
# Track.getHoleByCoords / getHoleByNum
# ---------------------------------------------------------------------

class FakeHole:
    def __init__(self, num, coords):
        self.num = num
        self.coords = coords


def test_get_hole_by_coords_finds_matching_hole():
    t = Track()
    t.trackholes = [FakeHole(1, (0, 0)), FakeHole(2, (1, 1))]
    found = t.getHoleByCoords((1, 1))
    assert found.num == 2


def test_get_hole_by_coords_returns_none_when_missing():
    t = Track()
    t.trackholes = [FakeHole(1, (0, 0))]
    assert t.getHoleByCoords((9, 9)) is None


def test_get_hole_by_num_uses_holeset_indexer():
    t = Track()
    t.trackholes = [FakeHole(5, (0, 0)), FakeHole(10, (1, 1)), FakeHole(15, (2, 2))]
    t.setHolesetIndexer()
    assert t.getHoleByNum(10).coords == (1, 1)
    assert t.getHoleByNum(99) is None


# ---------------------------------------------------------------------
# Track DataFrame helpers
# ---------------------------------------------------------------------

def test_get_ladders_as_df_round_trips_start_end_track():
    t = Track()
    t.ladders = [Ladder(1, 5, track=2), Ladder(3, 8, track=2)]
    df = t.getLaddersAsDF()
    assert list(df["start"]) == [1, 3]
    assert list(df["end"]) == [5, 8]
    assert list(df["track"]) == [2, 2]


def test_get_events_as_df_combines_ladders_and_chutes():
    t = Track()
    t.ladders = [Ladder(1, 5, track=2)]
    t.chutes = [Chute(9, 3, track=2)]
    df = t.getEventsAsDF()
    assert len(df) == 2
    assert set(df["start"]) == {1, 9}


# ---------------------------------------------------------------------
# Track add/set/clear helpers
# ---------------------------------------------------------------------

def test_clear_track_events_resets_events_but_not_length():
    t = Track()
    t.length = 42
    t.addLadder(Ladder(1, 2, track=1))
    t.addChute(Chute(3, 1, track=1))
    t.addEventLadder(1)
    t.addEventChute(3)
    t.instLocked = True

    t.clearTrackEvents()

    assert t.ladders == []
    assert t.chutes == []
    assert t.eventsListLadder == []
    assert t.eventsListChute == []
    assert t.instLocked is False
    assert t.length == 42  # not touched by clearTrackEvents


# ---------------------------------------------------------------------
# Board.getTrackByNum / clearBoard / clearTrackEvents
# ---------------------------------------------------------------------

def test_get_track_by_num_finds_and_misses():
    board = Board()
    t1, t2 = Track(), Track()
    t1.num, t2.num = 1, 2
    board.tracks = [t1, t2]
    assert board.getTrackByNum(2) is t2
    assert board.getTrackByNum(99) is None


def test_clear_board_resets_fields_but_keeps_config():
    config = GameConfig(numplayers=2)
    board = Board(config=config)
    board.boardName = "test"
    board.tracks = [Track()]
    board.clearBoard()
    assert board.boardName == ""
    assert board.tracks == []
    assert board.possibleEvents is None
    assert board.config is config  # config is not part of "board data"


def test_clear_track_events_defaults_to_all_tracks():
    board = Board()
    t1, t2 = Track(), Track()
    t1.ladders = [Ladder(1, 2, track=1)]
    t2.chutes = [Chute(3, 1, track=2)]
    board.tracks = [t1, t2]
    board.clearTrackEvents()
    assert t1.ladders == []
    assert t2.chutes == []


def test_clear_track_events_honours_specific_tracks_subset():
    board = Board()
    t1, t2 = Track(), Track()
    t1.ladders = [Ladder(1, 2, track=1)]
    t2.ladders = [Ladder(3, 4, track=2)]
    board.tracks = [t1, t2]
    board.clearTrackEvents(specificTracks=[t1])
    assert t1.ladders == []
    assert len(t2.ladders) == 1
    assert t2.ladders[0].start == 3 and t2.ladders[0].end == 4


# ---------------------------------------------------------------------
# Board.setBoardAfterSetter -- config-driven findmode branch + the lazy
# PossibleEvents import (Phase 3 import-hygiene fix)
# ---------------------------------------------------------------------

def test_set_board_after_setter_noop_when_findmode_false(monkeypatch):
    board = Board(config=GameConfig(findmode=False))

    called = []
    monkeypatch.setattr(
        "cribsandladders.BaseLayout.setTrackHolesets",
        lambda *a, **k: called.append((a, k)),
    )

    board.setBoardAfterSetter()

    assert called == []
    assert board.possibleEvents is None


def test_set_board_after_setter_does_not_import_possible_events_when_findmode_is_false():
    sys.modules.pop("cribsandladders.PossibleEvents", None)
    board = Board(config=GameConfig(findmode=False))
    board.setBoardAfterSetter()
    assert "cribsandladders.PossibleEvents" not in sys.modules


def test_set_board_after_setter_builds_holesets_and_possible_events_when_findmode_true(monkeypatch):
    board = Board(config=GameConfig(findmode=True))
    board.tracks = [Track()]

    holeset_calls = []
    monkeypatch.setattr(
        "cribsandladders.BaseLayout.setTrackHolesets",
        lambda tracks, height, path, config: holeset_calls.append((tracks, height, path, config)),
    )

    sentinel = object()

    class FakePossibleEvents:
        def __init__(self, board_arg):
            self.board_arg = board_arg

    import cribsandladders.PossibleEvents as pe_module
    monkeypatch.setattr(pe_module, "PossibleEvents", FakePossibleEvents)

    board.setBoardAfterSetter()

    assert len(holeset_calls) == 1
    tracks_arg, height_arg, path_arg, config_arg = holeset_calls[0]
    assert tracks_arg == board.tracks
    assert config_arg is board.config
    assert isinstance(board.possibleEvents, FakePossibleEvents)
    assert board.possibleEvents.board_arg is board
