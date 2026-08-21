"""
Phase 4 (Optimizer/board-design subsystem) tests for
cribsandladders.PossibleEvents.

PossibleEvents.__init__ does real I/O (opens a sqlite connection, then
either reads a cache or runs the full candidate-event search algorithm
against real board geometry) -- that full search (`buildSet`,
`tryRetrieveCache`'s DB-facing half) is Tier 3 per the TDD refactor
assessment: heavy, DB-coupled, and not a good target for unit tests in
one pass. What *is* tractable and covered here:

- The class's many geometry/math helper methods (ccw, intersect,
  orientation, orthogonal_vector, bounding boxes, angle checks, ...)
  are pure -- no I/O -- so they're tested directly by constructing a
  bare instance with `object.__new__(PossibleEvents)` and setting just
  the attributes each method reads (self.config, occasionally
  self.board), bypassing __init__'s DB connection entirely.
- `hydrate_candidate_events_from_dataframe`, extracted from
  tryRetrieveCache specifically for this reason, is unit-tested with a
  small hand-built DataFrame instead of a real Temp.db cache.
- `CandidateEvent`/`CandidateEvents` (module-level, not methods of
  PossibleEvents) are tested directly, including that
  `CandidateEvent.canBeLadder` respects the injected `GameConfig`
  instead of the old `game_params` global.

PossibleEvents now also takes an injected `config: GameConfig` (mirrors
Board/BoardSetter's Phase 3 migration and Stats's below), and the
`etc/Temp.db` / `Boards/AllBoards.db` paths it opens now come from
`config.temp_events_db_path` / `config.db_path` instead of hardcoded
literals.
"""
import math

import pandas as pd
import pytest

from cribsandladders.PossibleEvents import PossibleEvents, CandidateEvent, CandidateEvents
from cribsandladders.config import GameConfig


def _bare_possible_events(config=None, board=None):
    """A PossibleEvents instance with __init__ (and its sqlite connect)
    skipped -- only self.config/self.board are set, which is all these
    geometry helper methods read."""
    pe = object.__new__(PossibleEvents)
    pe.config = config or GameConfig()
    pe.board = board
    return pe


# ---------------------------------------------------------------------
# ccw / orientation / onSegment / doIntersect / intersect
# ---------------------------------------------------------------------

def test_ccw_true_for_counterclockwise_triangle():
    pe = _bare_possible_events()
    assert pe.ccw((0, 0), (1, 0), (1, 1)) is True


def test_intersect_true_for_crossing_segments():
    pe = _bare_possible_events()
    assert pe.intersect(((0, 0), (10, 10)), ((0, 10), (10, 0))) is True


def test_intersect_false_for_parallel_segments():
    pe = _bare_possible_events()
    assert pe.intersect(((0, 0), (10, 0)), ((0, 5), (10, 5))) is False


def test_orientation_collinear_clockwise_counterclockwise():
    pe = _bare_possible_events()
    assert pe.orientation((0, 0), (1, 1), (2, 2)) == 0  # collinear
    assert pe.orientation((0, 0), (0, 1), (1, 0)) == 1  # clockwise
    assert pe.orientation((0, 0), (1, 0), (0, 1)) == 2  # counterclockwise


def test_do_intersect_general_case():
    pe = _bare_possible_events()
    assert pe.doIntersect((0, 0), (10, 10), (0, 10), (10, 0)) is True


def test_do_intersect_collinear_overlap():
    pe = _bare_possible_events()
    # p2 lies on segment p1q1
    assert pe.doIntersect((0, 0), (10, 0), (5, 0), (15, 0)) is True


def test_do_intersect_no_intersection():
    pe = _bare_possible_events()
    assert pe.doIntersect((0, 0), (1, 0), (5, 5), (6, 6)) is False


# ---------------------------------------------------------------------
# orthogonal_vector / midpoint / calculate_distance
# ---------------------------------------------------------------------

def test_orthogonal_vector_perpendicular_and_scaled():
    pe = _bare_possible_events()
    dx, dy = pe.orthogonal_vector((0, 0), (1, 0), 5, False)
    assert dx == pytest.approx(0.0, abs=1e-9)
    assert dy == pytest.approx(5.0)


def test_orthogonal_vector_reverse_flips_direction():
    pe = _bare_possible_events()
    fwd = pe.orthogonal_vector((0, 0), (1, 0), 5, False)
    rev = pe.orthogonal_vector((0, 0), (1, 0), 5, True)
    assert rev == (-fwd[0], -fwd[1])


def test_orthogonal_vector_zero_length_segment_returns_zero_vector():
    pe = _bare_possible_events()
    assert pe.orthogonal_vector((1, 1), (1, 1), 5, False) == (0, 0)


def test_midpoint():
    pe = _bare_possible_events()
    assert pe.midpoint((0, 0), (4, 6)) == (2.0, 3.0)


def test_calculate_distance_3_4_5():
    ce_holder = _bare_possible_events()
    assert ce_holder.calculate_distance((0, 0), (3, 4)) == pytest.approx(5.0)


# ---------------------------------------------------------------------
# bounding boxes
# ---------------------------------------------------------------------

def test_cartesian_bounding_box_pads_by_ten():
    pe = _bare_possible_events()
    box = pe.cartesian_bounding_box([(0, 0), (10, 10)])
    assert box == [(-10, -10), (20, -10), (20, 20), (-10, 20)]


def test_cartesian_bounding_box_rejects_empty_points():
    pe = _bare_possible_events()
    with pytest.raises(ValueError):
        pe.cartesian_bounding_box([])


def test_determine_rectangle_corners():
    pe = _bare_possible_events()
    corners = pe.determine_rectangle_corners((0, 0), (4, 4))
    assert corners == [(0, 0), (0, 4), (4, 4), (4, 0)]


def test_ortho_bounding_box_returns_four_corners():
    pe = _bare_possible_events(config=GameConfig(eventminspacing=5))
    corners = pe.orthoBoundingBox(((0, 0), (10, 0)))
    assert len(corners) == 4
    # The two original endpoints must be among the corners.
    assert (0, 0) in corners
    assert (10, 0) in corners


# ---------------------------------------------------------------------
# checkAngleForOrtho -- config-injected minanglefromtracktangent
# ---------------------------------------------------------------------

def test_check_angle_for_ortho_true_near_zero_degrees():
    pe = _bare_possible_events(config=GameConfig(minanglefromtracktangent=30))
    assert pe.checkAngleForOrtho(0) is True


def test_check_angle_for_ortho_false_near_ninety_degrees():
    pe = _bare_possible_events(config=GameConfig(minanglefromtracktangent=30))
    assert pe.checkAngleForOrtho(90) is False


def test_check_angle_for_ortho_boundary_respects_config_value():
    # With a tighter tolerance, an angle that used to fail the ortho
    # check now passes it -- proves the value comes from the injected
    # config, not a hardcoded constant.
    tight = _bare_possible_events(config=GameConfig(minanglefromtracktangent=5))
    loose = _bare_possible_events(config=GameConfig(minanglefromtracktangent=80))
    assert tight.checkAngleForOrtho(10) is False
    assert loose.checkAngleForOrtho(10) is True


# ---------------------------------------------------------------------
# find_longest_line / ordered_by_proximity / extend_line
# ---------------------------------------------------------------------

def test_find_longest_line_picks_extreme_points():
    pe = _bare_possible_events()
    coords = [(5, 5), (0, 0), (10, 10), (3, 3)]
    p1, p2 = pe.find_longest_line(coords)
    assert p1 == (0, 0)
    assert p2 == (10, 10)


def test_find_longest_line_requires_at_least_two_points():
    pe = _bare_possible_events()
    with pytest.raises(ValueError):
        pe.find_longest_line([(0, 0)])


class _FakeHole:
    def __init__(self, coords):
        self.coords = coords


def test_ordered_by_proximity_sorts_by_distance_to_reference():
    pe = _bare_possible_events()
    holes = [_FakeHole((10, 10)), _FakeHole((1, 1)), _FakeHole((5, 5))]
    ordered = pe.ordered_by_proximity(holes, (0, 0))
    assert [h.coords for h in ordered] == [(1, 1), (5, 5), (10, 10)]


def test_extend_line_extends_both_directions_by_distance():
    pe = _bare_possible_events()
    (p1, new_p1), (p2, new_p2) = pe.extend_line((0, 0), (10, 0), 5)
    assert new_p1 == pytest.approx((-5, 0))
    assert new_p2 == pytest.approx((15, 0))


# ---------------------------------------------------------------------
# eventAngleWithInstantSlope / points_in_rectangle
# ---------------------------------------------------------------------

def test_event_angle_with_instant_slope_zero_for_same_direction():
    pe = _bare_possible_events()
    angle = pe.eventAngleWithInstantSlope((0, 0), (1, 0), (2, 0))
    assert angle == pytest.approx(0.0, abs=1e-6)


def test_event_angle_with_instant_slope_ninety_for_perpendicular():
    pe = _bare_possible_events()
    angle = pe.eventAngleWithInstantSlope((0, 0), (1, 0), (0, 1))
    assert angle == pytest.approx(90.0, abs=1e-6)


def test_points_in_rectangle_filters_correctly():
    pe = _bare_possible_events()
    rect = [(0, 0), (10, 0), (10, 10), (0, 10)]
    points = [(5, 5), (20, 20), (1, 1)]
    inside = pe.points_in_rectangle(points, rect)
    assert set(inside) == {(5, 5), (1, 1)}


# ---------------------------------------------------------------------
# build_interception_test_vector_set / build_proximity_test_vector_set
# ---------------------------------------------------------------------

def test_build_interception_test_vector_set_links_subset_to_neighbours():
    pe = _bare_possible_events()
    main_set = [(0, 0), (1, 1), (2, 2), (3, 3)]
    vectors = pe.build_interception_test_vector_set(main_set, [(1, 1)])
    # (1,1) -> next point (2,2), plus prev point (0,0) -> (1,1) since
    # (0,0) is not itself in the subset.
    assert ((1, 1), (2, 2)) in vectors
    assert ((0, 0), (1, 1)) in vectors


# ---------------------------------------------------------------------
# hydrate_candidate_events_from_dataframe (extracted, pure w.r.t. I/O)
# ---------------------------------------------------------------------

class _Hole:
    def __init__(self, num, coords):
        self.num = num
        self.coords = coords
        self.tracknum = 1
        self.lastHole = False


class _Track:
    def __init__(self, num, holes):
        self.num = num
        self.trackholes = holes
        self.candidateEvents = None

    def getHoleByNum(self, n):
        for h in self.trackholes:
            if h.num == n:
                return h
        return None


class _Board:
    def __init__(self, tracks):
        self.tracks = tracks

    def getTrackByNum(self, n):
        for t in self.tracks:
            if t.num == n:
                return t
        return None


def _cached_row(track_num, start, end, cand_id, finder_hash, shared_with=None, linked1=None, linked2=None):
    return {
        "trackNum": track_num, "startHole": start, "endHole": end, "isOrtho": False,
        "orthoFwdMinIncr": 0, "orthoRevMinIncr": 0, "orthoFwdMaxIncr": 0, "orthoRevMaxIncr": 0,
        "CandidateEvent_ID": cand_id, "FinderHash": finder_hash, "instanceIncr": -1,
        "instanceRev": False, "isShared": shared_with is not None,
        "orthoVector": "0.0,0.0", "sharedWithTracks": shared_with,
        "linkedEvent1": linked1, "linkedEvent2": linked2,
    }


def test_hydrate_candidate_events_builds_events_per_track():
    holes = [_Hole(1, (0, 0)), _Hole(2, (1, 1)), _Hole(3, (2, 2))]
    track = _Track(1, holes)
    board = _Board([track])
    pe = _bare_possible_events(board=board)

    df = pd.DataFrame([
        _cached_row(1, 1, 2, 100, 111),
        _cached_row(1, 2, 3, 101, 222),
    ])
    pe.hydrate_candidate_events_from_dataframe(df)

    cs = track.candidateEvents
    assert len(cs.candidateEvents) == 2
    assert {(e.startHole.num, e.endHole.num) for e in cs.candidateEvents} == {(1, 2), (2, 3)}


def test_hydrate_candidate_events_sets_up_linked_events():
    holes = [_Hole(1, (0, 0)), _Hole(2, (1, 1)), _Hole(3, (2, 2))]
    track = _Track(1, holes)
    board = _Board([track])
    pe = _bare_possible_events(board=board)

    df = pd.DataFrame([
        _cached_row(1, 1, 2, 100, 111, shared_with="2", linked1=222),
        _cached_row(1, 2, 3, 101, 222, shared_with="1", linked1=111),
    ])
    pe.hydrate_candidate_events_from_dataframe(df)

    events = {(e.startHole.num, e.endHole.num): e for e in track.candidateEvents.candidateEvents}
    ev_1_2 = events[(1, 2)]
    ev_2_3 = events[(2, 3)]
    assert ev_1_2.linkedEvents == [ev_2_3]
    assert ev_2_3.linkedEvents == [ev_1_2]


def test_hydrate_candidate_events_respects_injected_config_for_can_be_ladder():
    holes = [_Hole(1, (0, 0)), _Hole(2, (1, 1)), _Hole(30, (29, 29))]
    track = _Track(1, holes)
    board = _Board([track])
    pe = _bare_possible_events(config=GameConfig(maxladderlength=5), board=board)

    df = pd.DataFrame([_cached_row(1, 1, 30, 100, 111)])  # length 29 > maxladderlength 5
    pe.hydrate_candidate_events_from_dataframe(df)

    event = track.candidateEvents.candidateEvents[0]
    assert event.canBeLadder is False


# ---------------------------------------------------------------------
# CandidateEvent / CandidateEvents
# ---------------------------------------------------------------------

def test_candidate_event_can_be_ladder_respects_config_maxladderlength():
    short = CandidateEvent(1, _Hole(1, (0, 0)), _Hole(3, (2, 2)), False, config=GameConfig(maxladderlength=5))
    long_ = CandidateEvent(1, _Hole(1, (0, 0)), _Hole(20, (19, 19)), False, config=GameConfig(maxladderlength=5))
    assert short.canBeLadder is True
    assert long_.canBeLadder is False


def test_candidate_event_equality_and_hash_based_on_key():
    a = CandidateEvent(1, _Hole(1, (0, 0)), _Hole(2, (1, 1)), False)
    b = CandidateEvent(1, _Hole(1, (0, 0)), _Hole(2, (1, 1)), False)
    c = CandidateEvent(2, _Hole(1, (0, 0)), _Hole(2, (1, 1)), False)
    assert a == b
    assert hash(a) == hash(b)
    assert a != c


def test_candidate_event_ordering_by_track_then_start_then_end():
    a = CandidateEvent(1, _Hole(1, (0, 0)), _Hole(2, (1, 1)), False)
    b = CandidateEvent(1, _Hole(2, (1, 1)), _Hole(3, (2, 2)), False)
    assert a < b


def test_candidate_event_set_linked_events_updates_shared_tracks():
    main = CandidateEvent(1, _Hole(1, (0, 0)), _Hole(2, (1, 1)), False)
    other = CandidateEvent(2, _Hole(5, (4, 4)), _Hole(6, (5, 5)), False)
    main.setLinkedEvents([other])
    assert main.sharedWithTracks == [2]
    assert main.isShared is True


def test_candidate_events_add_sets_hash_and_remove_duplicates_dedupes():
    holder = CandidateEvents([_Hole(1, (0, 0)), _Hole(2, (1, 1))], trackNum=1)
    e1 = CandidateEvent(1, _Hole(1, (0, 0)), _Hole(2, (1, 1)), False)
    e2 = CandidateEvent(1, _Hole(1, (0, 0)), _Hole(2, (1, 1)), False)  # duplicate key
    holder.addCandidateEvent(e1)
    holder.addCandidateEvent(e2)
    assert len(holder.candidateEvents) == 2
    holder.removeDuplicates()
    assert len(holder.candidateEvents) == 1
