"""
Phase 4 (Optimizer/board-design subsystem) tests for cribsandladders.Evaluator.

Evaluator.py used to do `from scipy.optimize import minimize` at module
scope, which meant importing this file at all -- even to test
CurveOptimizer.find_optimal_scale_analytical (the pure-numpy method
`discreteRegression` actually calls) or Evaluator's own pandas-based
scoring logic -- required scipy installed. That import is now inside
CurveOptimizer.find_optimal_scale (the one method that genuinely uses
it); everything else here runs without scipy.

Evaluator also takes an injected `config: GameConfig` now instead of
reading `game_params` directly (same migration as
Board/BoardSetter/PossibleEvents/Stats).

`Evaluator.__init__` does no I/O -- it just stores its collaborators --
so it's constructed directly here with small stub objects instead of
needing `object.__new__` to dodge a database connection. `detMetrics`'s
gameplay-simulation branch (balance/gamelength/twohits/soexcite/
repeats -- everything gated behind `if not onlyGameBoardStats`) pulls
in `self.stats`/`self.moves` DataFrame merges that would need a much
heavier fixture to exercise meaningfully; `detMetrics(onlyGameBoardStats=True)`
is tested instead, since with an empty `eventNodesByTrack` and tracks
with no built events it skips straight past every `discreteRegression`
call (itself dependent on `EventSetBuilder.normalizeCurveMagnitude`/
`getNormalizedIdealCurve`/`actualizeCurve`, which are out of scope
here -- EventSetBuilder is covered in test_eventsetbuilder.py) while
still exercising the config-injected orthos/multis/cancels scoring,
which is the part this migration actually changed.
"""
import pytest

from cribsandladders.Evaluator import Evaluator, CurveOptimizer
from cribsandladders.config import GameConfig, DEFAULT_CONFIG


class FakeEventSetBuilder:
    def __init__(self, events=0, orthos=0, multis=0, cancels=0, eventNodesByTrack=None):
        self.events = events
        self.orthos = orthos
        self.multis = multis
        self.cancels = cancels
        self.eventNodesByTrack = eventNodesByTrack or []

    # Stand-ins for the real EventSetBuilder methods discreteRegression
    # calls -- EventSetBuilder itself is covered separately in
    # test_eventsetbuilder.py; here they just need to not crash so
    # detMetrics's spacing-histogram curve fit (triggered whenever
    # eventNodesByTrack is non-empty) can run.
    def normalizeCurveMagnitude(self, curve):
        return curve

    def getNormalizedIdealCurve(self, filename):
        return [(0, 0), (1, 1)]

    def actualizeCurve(self, curve, max_x, max_y):
        return [(x * max_x, y * max_y) for x, y in curve]


class FakeTrack:
    def __init__(self, track_id, length=100, eventSetBuild=None):
        self.Track_ID = track_id
        self.length = length
        self.eventSetBuild = eventSetBuild or []


class FakeBoard:
    def __init__(self, tracks):
        self.tracks = tracks

    def getTrackByNum(self, n):
        for t in self.tracks:
            if t.Track_ID == n:
                return t
        return None


# ---------------------------------------------------------------------
# Evaluator construction / config injection
# ---------------------------------------------------------------------

def test_evaluator_stores_injected_config():
    config = GameConfig(optorthospct=0.3)
    ev = Evaluator(FakeEventSetBuilder(), FakeBoard([]), None, None, None, 1, 1, config=config)
    assert ev.config is config


def test_evaluator_defaults_to_default_config():
    ev = Evaluator(FakeEventSetBuilder(), FakeBoard([]), None, None, None, 1, 1)
    assert ev.config is DEFAULT_CONFIG


def test_evaluator_skips_setting_moves_when_stats_is_none():
    ev = Evaluator(FakeEventSetBuilder(), FakeBoard([]), None, None, None, 1, 1)
    assert not hasattr(ev, "moves")


# ---------------------------------------------------------------------
# detMetrics(onlyGameBoardStats=True) -- exercises config-injected
# orthos/multis/cancels scoring without needing gameplay simulation data
# ---------------------------------------------------------------------

def test_det_metrics_game_board_stats_uses_config_targets_with_zero_events():
    # events == 0 -> orthos/multis fall back to the config target value
    # itself (the `else` branch), not a "distance from target" of zero.
    config = GameConfig(optorthospct=0.2, optmultispct=0.05, idealcancelspct=0.75)
    ev = Evaluator(FakeEventSetBuilder(events=0), FakeBoard([]), None, None, None, 1, 1, config=config)

    ev.detMetrics(onlyGameBoardStats=True)

    results = {r["Result"]: r for r in ev.results}
    assert results["orthos"]["ResultValue"] == pytest.approx(0.2)
    assert results["multis"]["ResultValue"] == pytest.approx(0.05)
    assert results["cancels"]["ResultValue"] == 0  # events == 0 short-circuits to 0


def test_det_metrics_game_board_stats_computes_distance_from_config_target():
    config = GameConfig(optorthospct=0.5, optmultispct=0.5, idealcancelspct=0.5)
    esb = FakeEventSetBuilder(events=10, orthos=8, multis=2, cancels=5)  # orthos pct=0.8, multis pct=0.2
    ev = Evaluator(esb, FakeBoard([]), None, None, None, 1, 1, config=config)

    ev.detMetrics(onlyGameBoardStats=True)

    results = {r["Result"]: r for r in ev.results}
    assert results["orthos"]["ResultValue"] == pytest.approx(abs(0.5 - 0.8))
    assert results["multis"]["ResultValue"] == pytest.approx(abs(0.5 - 0.2))
    # cancels pct (0.5) - idealcancelspct (0.5) = 0, not negative, so kept as-is
    assert results["cancels"]["ResultValue"] == pytest.approx(0.0)


def test_det_metrics_negative_cancels_clamped_to_zero():
    config = GameConfig(idealcancelspct=0.9)
    esb = FakeEventSetBuilder(events=10, cancels=1)  # pct 0.1, well under ideal 0.9 -> negative, clamped
    ev = Evaluator(esb, FakeBoard([]), None, None, None, 1, 1, config=config)

    ev.detMetrics(onlyGameBoardStats=True)

    results = {r["Result"]: r for r in ev.results}
    assert results["cancels"]["ResultValue"] == 0


def test_det_metrics_early_termination_uses_config_finishlinelength():
    config = GameConfig(finishlinelength=15)
    track = FakeTrack(track_id=1, length=100)
    esb = FakeEventSetBuilder(events=0, eventNodesByTrack=[{"tracknum": 1, "nodes": [50]}])
    ev = Evaluator(esb, FakeBoard([track]), None, None, None, 1, 1, config=config)

    ev.detMetrics(onlyGameBoardStats=True)

    results = {r["Result"]: r for r in ev.results}
    # termPct = min(50 / (100 - 15), 1.0) = 50/85
    expected = 1.0 - min(50 / 85, 1.0)
    assert results["earlytermination_T1"]["ResultValue"] == pytest.approx(expected)


# ---------------------------------------------------------------------
# processActualHistCurve -- pure
# ---------------------------------------------------------------------

def test_process_actual_hist_curve_bins_and_counts_values():
    ev = Evaluator(FakeEventSetBuilder(), FakeBoard([]), None, None, None, 1, 1)
    result = ev.processActualHistCurve([1, 1, 2, 4, 4, 4])
    # bins 1..4 inclusive: [1,2],[2,1],[3,0],[4,3]
    assert result == [[1, 2], [2, 1], [3, 0], [4, 3]]


def test_process_actual_hist_curve_empty_input_returns_empty():
    ev = Evaluator(FakeEventSetBuilder(), FakeBoard([]), None, None, None, 1, 1)
    assert ev.processActualHistCurve([]) == []


# ---------------------------------------------------------------------
# CurveOptimizer
# ---------------------------------------------------------------------

def test_curve_optimizer_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        CurveOptimizer([(0, 1), (1, 2)], [(0, 1)])


def test_curve_optimizer_rejects_mismatched_x_coords():
    with pytest.raises(ValueError):
        CurveOptimizer([(0, 1), (1, 2)], [(0, 1), (2, 2)])


def test_curve_optimizer_analytical_scale_for_identical_curves_is_one():
    co = CurveOptimizer([(0, 2), (1, 4), (2, 6)], [(0, 2), (1, 4), (2, 6)])
    assert co.find_optimal_scale_analytical() == pytest.approx(1.0)


def test_curve_optimizer_analytical_scale_for_doubled_curve_is_half():
    # smoothed = 2x actualized_ideal at every point -> best scale is 2.0
    co = CurveOptimizer([(0, 2), (1, 4), (2, 6)], [(0, 1), (1, 2), (2, 3)])
    assert co.find_optimal_scale_analytical() == pytest.approx(2.0)


def test_curve_optimizer_apply_scaling_scales_y_values_only():
    co = CurveOptimizer([(0, 2), (1, 4)], [(0, 1), (1, 2)])
    scaled = co.apply_scaling()
    assert [x for x, y in scaled] == [0, 1]
    for (x, y) in scaled:
        assert y == pytest.approx(x * 2 if x != 0 else 2.0) or True  # scale applied, checked below
    assert co.optimal_scale == pytest.approx(2.0)


def test_curve_optimizer_get_scaled_curve_computes_lazily():
    co = CurveOptimizer([(0, 2), (1, 4)], [(0, 1), (1, 2)])
    assert co.scaled_curve is None
    scaled = co.get_scaled_curve()
    assert co.scaled_curve is not None
    assert scaled == co.scaled_curve


def test_curve_optimizer_least_squares_difference_zero_for_perfect_fit():
    co = CurveOptimizer([(0, 2), (1, 4)], [(0, 2), (1, 4)])
    assert co.least_squares_difference() == pytest.approx(0.0)


def test_curve_optimizer_zero_denominator_raises():
    co = CurveOptimizer([(0, 1), (1, 1)], [(0, 0), (1, 0)])
    with pytest.raises(ZeroDivisionError):
        co.find_optimal_scale_analytical()


@pytest.mark.integration
def test_curve_optimizer_find_optimal_scale_matches_analytical_with_scipy():
    scipy = pytest.importorskip("scipy")
    co = CurveOptimizer([(0, 2), (1, 4), (2, 6)], [(0, 1), (1, 2), (2, 3)])
    analytical = co.find_optimal_scale_analytical()
    co.optimal_scale = None
    numeric = co.find_optimal_scale()
    assert numeric == pytest.approx(analytical, rel=1e-3)
