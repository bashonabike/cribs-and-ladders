"""
Refactor Mk II, Phase 7 -- characterization test for
PossibleEvents.buildSet().

buildSet is Tier 3 in tdd-refactor-assessment.md: heavy, DB-coupled,
real geometry, and (per test_possible_events.py's own module
docstring) explicitly left untested through Phase 4 -- "not a good
target for unit tests in one pass." Phase 7 splits it into three named
pieces (direct/multi-track-loop search, orthogonal/loopy sidestep
search, and DB persistence -- see PossibleEvents.py's module-level
comment on buildSet for the full breakdown), which is pure code motion
and shouldn't change what buildSet actually produces.

This file exists to make that claim checkable: it builds a small but
real Board (one zigzagging 9-hole track -- collinear holes turned out
to make every pair either always- or never-eligible depending on the
branch, so the holes zigzag left/right to get a genuine mix of
accepted/rejected candidates in the ortho-sidestep branch), runs the
real buildSet() end to end against a real (in-memory) sqlite
TempCandidateEvents table, and pins down the exact candidate events and
DB rows that come out -- captured by running the *pre-refactor* code
once and hand-verifying the shape looked sane (a mix of accepted/
rejected candidates, not "everything" or "nothing"). If a future change
to buildSet (or the pieces it gets split into) changes this output,
this test will fail and the diff will say exactly what changed.

Not a spec of "correct" board-design behavior -- like the other
characterization tests in this suite, it's a safety net for refactoring,
not a claim that this geometry is what buildSet *should* produce.
"""
import sqlite3

import pytest

from cribsandladders.Board import Board, Track
from cribsandladders.BaseLayout import Hole
from cribsandladders.PossibleEvents import PossibleEvents
from cribsandladders.config import GameConfig

pytestmark = pytest.mark.integration

_TEMP_CANDIDATE_EVENTS_SCHEMA = """
CREATE TABLE TempCandidateEvents (
    Board_ID INTEGER NOT NULL, Track_ID INTEGER NOT NULL, CandidateEvent_ID INTEGER NOT NULL,
    Timestamp TEXT, trackNum INTEGER, startHole INTEGER, endHole INTEGER, midPointNum REAL,
    crowVector TEXT, length INTEGER, canBeLadder INTEGER, isOrtho INTEGER, orthoVector TEXT,
    orthoFwdMinIncr INTEGER, orthoFwdMaxIncr INTEGER, orthoRevMinIncr INTEGER, orthoRevMaxIncr INTEGER,
    sharedWithTracks TEXT, sharedSubHash INTEGER, instanceIncr INTEGER, instanceRev INTEGER,
    linkedEvent1 INTEGER, linkedEvent2 INTEGER, isShared INTEGER, FinderHash INTEGER,
    PRIMARY KEY (Board_ID, Track_ID, CandidateEvent_ID)
);
"""


def _make_single_track_board():
    """One 9-hole track, zigzagging between x=50/x=55 every other hole
    (a straight line made every pair either uniformly eligible or
    uniformly ineligible for the ortho-sidestep branch -- not useful
    for a characterization fixture that wants to see both accept and
    reject cases)."""
    board = Board(config=GameConfig(findmode=True))
    board.boardID = 1
    board.width = 200.0
    board.height = 200.0

    track = Track()
    track.num = 1
    track.Track_ID = 1
    xs = [50, 55, 50, 55, 50, 55, 50, 55, 50]
    track.trackholes = [
        Hole(float(x), float(y * 15), num=i + 1, tracknum=1, lastHole=(i == len(xs) - 1))
        for i, (x, y) in enumerate(zip(xs, range(len(xs))))
    ]
    track.setHolesetIndexer()

    board.tracks = [track]
    return board


def _bare_possible_events(board, config):
    """Same object.__new__ pattern test_possible_events.py already uses
    to skip __init__'s live sqlite connection -- buildSet only reads
    self.board/self.config/self.byTrackCandidateSets/etc, all set here
    by hand instead."""
    pe = object.__new__(PossibleEvents)
    pe.allTracksCandidateSet, pe.byTrackCandidateSets = None, []
    pe.multiTrackCandidateSet = None
    pe.board = board
    pe.config = config
    return pe


# Captured by running buildSet() once against the fixture above (before
# any Phase 7 extraction) and confirming by hand that the result is a
# genuine mix of accepted (21) and rejected (15) candidate pairs out of
# all 36 possible pairs on a 9-hole track -- not "everything passes" or
# "nothing passes", which would make this a weak regression signal.
_EXPECTED_EVENTS = [
    (1, 2, True, False, 1, True),
    (1, 3, True, False, 2, True),
    (1, 4, True, False, 3, True),
    (1, 5, True, False, 4, True),
    (1, 7, True, False, 6, True),
    (2, 3, True, False, 1, True),
    (2, 4, True, False, 2, True),
    (2, 5, True, False, 3, True),
    (3, 4, True, False, 1, True),
    (3, 5, True, False, 2, True),
    (3, 6, True, False, 3, True),
    (3, 7, True, False, 4, True),
    (4, 5, True, False, 1, True),
    (4, 6, True, False, 2, True),
    (4, 7, True, False, 3, True),
    (5, 6, True, False, 1, True),
    (5, 7, True, False, 2, True),
    (5, 8, True, False, 3, True),
    (6, 7, True, False, 1, True),
    (6, 8, True, False, 2, True),
    (7, 8, True, False, 1, True),
]


def test_build_set_produces_the_expected_candidate_events_and_db_rows():
    board = _make_single_track_board()
    conn = sqlite3.connect(":memory:")
    conn.executescript(_TEMP_CANDIDATE_EVENTS_SCHEMA)
    conn.commit()

    config = GameConfig(findmode=True, maxloopyorthoeventdisplacementincrements=4)
    pe = _bare_possible_events(board, config)

    pe.buildSet(board, conn)

    track = board.tracks[0]
    events = sorted(
        (c.startHole.num, c.endHole.num, c.isOrtho, c.isShared, c.length, c.canBeLadder)
        for c in track.candidateEvents.candidateEvents
    )
    assert events == _EXPECTED_EVENTS

    # All 21 accepted candidates round-tripped into the temp db too.
    row_count = conn.execute("SELECT COUNT(*) FROM TempCandidateEvents").fetchone()[0]
    assert row_count == 21

    db_pairs = sorted(
        conn.execute("SELECT startHole, endHole FROM TempCandidateEvents").fetchall()
    )
    assert db_pairs == sorted((e[0], e[1]) for e in _EXPECTED_EVENTS)

    # DB rows are scoped to this board/track -- a re-run (buildSet is
    # called once per board load) deletes and re-inserts rather than
    # accumulating duplicates.
    board2 = _make_single_track_board()
    pe2 = _bare_possible_events(board2, GameConfig(findmode=True, maxloopyorthoeventdisplacementincrements=4))
    pe2.buildSet(board2, conn)
    row_count_after_rerun = conn.execute("SELECT COUNT(*) FROM TempCandidateEvents").fetchone()[0]
    assert row_count_after_rerun == 21

    conn.close()


def test_build_set_rejects_more_candidates_with_a_tighter_displacement_budget():
    """Same fixture, smaller maxloopyorthoeventdisplacementincrements --
    confirms the ortho-sidestep branch's accept/reject decision actually
    responds to config (not just a fixed pass-everything/fail-everything
    outcome), which is what makes the fixture above a meaningful
    regression signal rather than a coincidence of these specific
    holes."""
    board = _make_single_track_board()
    conn = sqlite3.connect(":memory:")
    conn.executescript(_TEMP_CANDIDATE_EVENTS_SCHEMA)
    conn.commit()

    config = GameConfig(findmode=True, maxloopyorthoeventdisplacementincrements=2)
    pe = _bare_possible_events(board, config)
    pe.buildSet(board, conn)

    assert len(board.tracks[0].candidateEvents.candidateEvents) == 16
    conn.close()
