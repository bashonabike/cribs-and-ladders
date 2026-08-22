"""
Refactor Mk II, Phase 6: unit tests for cribsandladders.evaluator_metrics
-- the per-result-block functions pulled out of Evaluator.detMetrics().
detMetrics()'s own black-box behavior (calling these in the right order,
unchanged dict shapes) is already covered by tests/test_evaluator.py;
these test each function directly with the specific small inputs it
needs, rather than a full Evaluator/EventSetBuilder/Board/Stats stack.
"""
import pytest

from cribsandladders.config import GameConfig
from cribsandladders import evaluator_metrics as em


class FakeTrack:
    def __init__(self, num, track_id, length=100):
        self.num = num
        self.Track_ID = track_id
        self.length = length


class FakeBoard:
    def __init__(self, tracks):
        self.tracks = tracks

    def getTrackByNum(self, n):
        for t in self.tracks:
            if t.Track_ID == n or t.num == n:
                return t
        return None


class FakeMove:
    def __init__(self, track, hasEvent=False, ladderorchuteamt=0, movenum=1):
        self.track = track
        self.hasEvent = hasEvent
        self.ladderorchuteamt = ladderorchuteamt
        self.movenum = movenum


# ---------------------------------------------------------------------
# structure_scalar_stats
# ---------------------------------------------------------------------

def test_structure_scalar_stats_falls_back_to_config_targets_with_zero_events():
    config = GameConfig(optorthospct=0.2, optmultispct=0.05, idealcancelspct=0.75)
    results = em.structure_scalar_stats(events=0, orthos=0, multis=0, cancels=0, config=config)
    by_result = {r["Result"]: r for r in results}
    assert by_result["orthos"]["ResultValue"] == pytest.approx(0.2)
    assert by_result["multis"]["ResultValue"] == pytest.approx(0.05)
    assert by_result["cancels"]["ResultValue"] == 0


def test_structure_scalar_stats_computes_distance_from_target_with_events():
    config = GameConfig(optorthospct=0.5, optmultispct=0.5, idealcancelspct=0.5)
    results = em.structure_scalar_stats(events=10, orthos=8, multis=2, cancels=5, config=config)
    by_result = {r["Result"]: r for r in results}
    assert by_result["orthos"]["ResultValue"] == pytest.approx(abs(0.5 - 0.8))
    assert by_result["multis"]["ResultValue"] == pytest.approx(abs(0.5 - 0.2))
    assert by_result["cancels"]["ResultValue"] == pytest.approx(0.0)


def test_structure_scalar_stats_clamps_negative_cancels_to_zero():
    config = GameConfig(idealcancelspct=0.9)
    results = em.structure_scalar_stats(events=10, orthos=0, multis=0, cancels=1, config=config)
    by_result = {r["Result"]: r for r in results}
    assert by_result["cancels"]["ResultValue"] == 0


# ---------------------------------------------------------------------
# early_termination_stats
# ---------------------------------------------------------------------

def test_early_termination_stats_uses_config_finishlinelength():
    config = GameConfig(finishlinelength=15)
    track = FakeTrack(num=1, track_id=1, length=100)
    board = FakeBoard([track])
    results = em.early_termination_stats([{"tracknum": 1, "nodes": [50]}], board, config)
    expected = 1.0 - min(50 / 85, 1.0)
    assert results[0]["Result"] == "earlytermination_T1"
    assert results[0]["ResultValue"] == pytest.approx(expected)


def test_early_termination_stats_one_result_per_track():
    config = GameConfig(finishlinelength=0)
    tracks = [FakeTrack(num=1, track_id=1, length=100), FakeTrack(num=2, track_id=2, length=50)]
    board = FakeBoard(tracks)
    event_nodes = [{"tracknum": 1, "nodes": [10, 90]}, {"tracknum": 2, "nodes": [50]}]
    results = em.early_termination_stats(event_nodes, board, config)
    assert {r["Result"] for r in results} == {"earlytermination_T1", "earlytermination_T2"}


# ---------------------------------------------------------------------
# balance_stats
# ---------------------------------------------------------------------

def test_balance_stats_includes_overall_stdev_and_per_track_values():
    track1 = FakeTrack(num=1, track_id=10)
    track2 = FakeTrack(num=2, track_id=20)
    board = FakeBoard([track1, track2])
    partial_balance_set = [(1, 0.1), (2, -0.1)]

    results = em.balance_stats(partial_balance_set, board)
    by_result = {r["Result"]: r for r in results}
    assert "balance" in by_result
    assert by_result["balance_T10"]["ResultValue"] == 0.1
    assert by_result["balance_T20"]["ResultValue"] == -0.1


# ---------------------------------------------------------------------
# gamelength_stat
# ---------------------------------------------------------------------

def test_gamelength_stat_computes_relative_distance_from_ideal():
    config = GameConfig(idealgamelength=10)
    result = em.gamelength_stat(avglengthinrounds=12, config=config)
    assert result["ResultValueIterative"] == pytest.approx(0.2)
    assert result["ResultValue"] == pytest.approx(0.2)


def test_gamelength_stat_defaults_to_one_when_no_ideal_configured():
    config = GameConfig(idealgamelength=0)
    result = em.gamelength_stat(avglengthinrounds=12, config=config)
    assert result["ResultValueIterative"] == 1


# ---------------------------------------------------------------------
# twohits_stat
# ---------------------------------------------------------------------

def test_twohits_stat_counts_consecutive_events_on_same_track():
    config = GameConfig(opttwohitspct=0.0)
    track = FakeTrack(num=1, track_id=1)
    moves = [
        FakeMove(track=1, hasEvent=True, movenum=1),
        FakeMove(track=1, hasEvent=True, movenum=2),  # back-to-back event -> a two-hit
        FakeMove(track=1, hasEvent=False, movenum=3),
    ]
    result = em.twohits_stat(moves, [track], config)
    # 1 two-hit / 3 moves - 0.0 target = 1/3
    assert result["ResultValueIterative"] == pytest.approx(1 / 3)


def test_twohits_stat_defaults_to_one_when_no_moves():
    config = GameConfig()
    result = em.twohits_stat([], [], config)
    assert result["ResultValueIterative"] == 1


# ---------------------------------------------------------------------
# soexcite_stat / repeats_stat / events_hit_skew_stat
# ---------------------------------------------------------------------

def test_soexcite_stat_inverts_the_rate():
    result = em.soexcite_stat(soexcitespegging=0.25)
    assert result["ResultValue"] == pytest.approx(4.0)


def test_soexcite_stat_caps_at_one_at_or_below_threshold():
    result = em.soexcite_stat(soexcitespegging=0.05)
    assert result["ResultValue"] == 1.0


def test_repeats_stat_normalizes_by_move_count():
    result = em.repeats_stat(repeats=4, num_moves=8)
    assert result["ResultValue"] == pytest.approx(0.5)


def test_repeats_stat_defaults_to_one_when_no_moves():
    result = em.repeats_stat(repeats=0, num_moves=0)
    assert result["ResultValue"] == 1


def test_events_hit_skew_stat_is_zero_when_balanced():
    moves = [
        FakeMove(track=1, hasEvent=True, ladderorchuteamt=5),
        FakeMove(track=1, hasEvent=True, ladderorchuteamt=-5),
    ]
    result = em.events_hit_skew_stat(moves)
    assert result["ResultValue"] == 0


def test_events_hit_skew_stat_measures_imbalance():
    moves = [
        FakeMove(track=1, hasEvent=True, ladderorchuteamt=5),
        FakeMove(track=1, hasEvent=True, ladderorchuteamt=5),
        FakeMove(track=1, hasEvent=True, ladderorchuteamt=-5),
    ]
    result = em.events_hit_skew_stat(moves)
    # 2 ladder hits, 1 chute hit -> |2-1| / 3
    assert result["ResultValue"] == pytest.approx(1 / 3)


# ---------------------------------------------------------------------
# event_length_distribution_stats
# ---------------------------------------------------------------------

def _fake_hist_fn(values):
    return [[v, values.count(v)] for v in sorted(set(values))] if values else []


def _fake_regression_fn(curve_file, hist):
    return sum(h[1] for h in hist)  # arbitrary deterministic stand-in


def test_event_length_distribution_stats_overall_and_no_bottomheavy_flag():
    config = GameConfig()
    track = FakeTrack(num=1, track_id=1)
    moves = [FakeMove(track=1, hasEvent=True, ladderorchuteamt=10)]  # length 10, not <= 4

    results = em.event_length_distribution_stats(moves, [track], _fake_hist_fn, _fake_regression_fn, config)
    by_result = {r["Result"]: r for r in results}
    assert "eventsHitLengthDistribution_curvefit" in by_result
    assert "eventsHitLengthDistribution_bottomheavy_T1" not in by_result


def test_event_length_distribution_stats_flags_bottom_heavy_track():
    config = GameConfig()
    track = FakeTrack(num=1, track_id=1)
    moves = [
        FakeMove(track=1, hasEvent=True, ladderorchuteamt=2),
        FakeMove(track=1, hasEvent=True, ladderorchuteamt=3),
        FakeMove(track=1, hasEvent=True, ladderorchuteamt=10),
    ]  # 2 of 3 are <= 4 -> more than half -> flagged

    results = em.event_length_distribution_stats(moves, [track], _fake_hist_fn, _fake_regression_fn, config)
    by_result = {r["Result"]: r for r in results}
    assert by_result["eventsHitLengthDistribution_bottomheavy_T1"]["ResultValue"] == pytest.approx(2 / 3 - 0.5)


def test_event_length_distribution_stats_empty_moves_gives_result_one():
    config = GameConfig()
    results = em.event_length_distribution_stats([], [], _fake_hist_fn, _fake_regression_fn, config)
    assert results[0]["ResultValue"] == 1
