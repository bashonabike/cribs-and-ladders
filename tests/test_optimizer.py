"""
Phase 4 (Optimizer/board-design subsystem) tests for cribsandladders.Optimizer.

Optimizer.py used to import lightgbm and three sklearn modules at
module scope purely for two methods (runGBM, testGBMOnPairings) --
meaning importing this file at all, even to test the plain
pandas/numpy parameter-search logic in runIncrIteration,
detWeighedScoring, getFminStarterParams, etc., required both packages
installed. Those imports are now inside the two methods that actually
use them (mirrors the Phase 2 fix to Player.py's scoretree import and
the other Phase 4 import-hygiene fixes in this same pass), so
everything below runs without lightgbm/sklearn.

Optimizer also takes an injected `config: GameConfig` now (the
db path -- config.optimizer_db_path, replacing the hardcoded
'etc/Optimizer.db' literal -- and changebaseincrperiter /
testtotraindataratio_bnds / etc. all come from it instead of
game_params).

`Optimizer.__init__` does real I/O (opens a sqlite connection, then
calls retrievePairingsSettings(), which runs two queries against it),
so most methods here are tested by constructing a bare instance with
`object.__new__(Optimizer)` and setting only the attributes each method
reads -- the same pattern used for PossibleEvents in this phase. One
`@pytest.mark.integration` test builds a temp sqlite db with the
expected schema and exercises __init__ end-to-end.
"""
import sqlite3

import pandas as pd
import pytest

from cribsandladders.Optimizer import Optimizer
from cribsandladders.config import GameConfig


def _bare_optimizer(config=None):
    opt = object.__new__(Optimizer)
    opt.config = config or GameConfig()
    opt.prevParams = []
    opt.params = []
    opt.prevResults = []
    opt.freshResults = []
    opt.bestPostIterParams = []
    opt.bestPostFminParams = []
    opt.fminParamsList = []
    return opt


# ---------------------------------------------------------------------
# setParamFromBounds
# ---------------------------------------------------------------------

def test_set_param_from_bounds_returns_int_within_bounds_when_flagged():
    opt = _bare_optimizer()
    for _ in range(20):
        val = opt.setParamFromBounds((1, 5, True))
        assert isinstance(val, int)
        assert 1 <= val <= 5


def test_set_param_from_bounds_returns_float_within_bounds_when_not_flagged():
    opt = _bare_optimizer()
    for _ in range(20):
        val = opt.setParamFromBounds((1.0, 2.0, False))
        assert 1.0 <= val <= 2.0


# ---------------------------------------------------------------------
# detWeighedScoring
# ---------------------------------------------------------------------

def test_det_weighed_scoring_sums_absolute_value_times_weighting():
    opt = _bare_optimizer()
    results = [
        {"ResultValue": 2, "Weighting": 3},
        {"ResultValue": -1, "Weighting": 5},
    ]
    assert opt.detWeighedScoring(results) == pytest.approx(2 * 3 + 1 * 5)


# ---------------------------------------------------------------------
# setBestIterParams / getFminStarterParams / getFminBounds / setupFminParamsList
# ---------------------------------------------------------------------

def test_set_best_iter_params_stores_list_verbatim():
    opt = _bare_optimizer()
    params = [{"track_id": 1, "param": "x", "value": 5}]
    opt.setBestIterParams(params)
    assert opt.bestPostIterParams is params


def test_get_fmin_starter_params_pulls_values_for_each_fmin_param():
    opt = _bare_optimizer()
    opt.bestPostIterParams = [
        {"param": "alpha", "track_id": 1, "value": 10},
        {"param": "alpha", "track_id": 2, "value": 20},
        {"param": "beta", "track_id": 1, "value": 99},
    ]
    opt.fminParamsList = ["alpha"]
    starters = opt.getFminStarterParams()
    assert sorted(starters) == [10, 20]


def test_get_fmin_bounds_looks_up_absolute_bounds_per_param():
    opt = _bare_optimizer()
    opt.absoluteBounds = pd.DataFrame(
        [{"Param": "alpha", "LBound": 0, "UBound": 100}]
    ).set_index("Param")
    opt.fminParamsList = ["alpha"]
    sample_params = [
        {"param": "alpha", "track_id": 1, "value": 10},
        {"param": "alpha", "track_id": 2, "value": 20},
    ]
    bounds = opt.getFminBounds(sample_params)
    assert bounds == [(0, 100), (0, 100)]


def test_setup_fmin_params_list_collects_all_pairings():
    opt = _bare_optimizer()
    opt.pairings_df = pd.DataFrame(
        [
            {"Result": "ALL", "Param": "alpha"},
            {"Result": "ALL", "Param": "beta"},
            {"Result": "twohits", "Param": "gamma"},
        ]
    ).set_index("Result")
    opt.setupFminParamsList(sampleParams=[])
    assert opt.fminParamsList == ["alpha", "beta"]


# ---------------------------------------------------------------------
# runIncrIteration -- the main pure param-search logic, uses config.changebaseincrperiter
# ---------------------------------------------------------------------

def _optimizer_for_incr_iteration(config):
    opt = _bare_optimizer(config=config)
    opt.pairings_df = pd.DataFrame(
        [{"Result": "twohits", "Param": "maxtwohitnetgainloss", "Trackwise": 0, "Inverse": 0}]
    ).set_index("Result")
    opt.absoluteBounds = pd.DataFrame(
        [{"Param": "maxtwohitnetgainloss", "LBound": 0, "UBound": 50}]
    ).set_index("Param")
    return opt


def test_run_incr_iteration_adjusts_param_toward_bound_by_configured_increment():
    config = GameConfig(changebaseincrperiter=0.02)
    opt = _optimizer_for_incr_iteration(config)

    freshParams = [{"track_id": 1, "param": "maxtwohitnetgainloss", "value": 25}]
    freshResults = [{"Result": "twohits", "ResultValue": 0.5, "ResultValueIterative": 0.5, "Weighting": 30}]

    result = opt.runIncrIteration(freshParams, freshResults)

    weighed_result = abs(0.5) * 30
    change_amt = (50 - 0) * config.changebaseincrperiter * weighed_result
    expected_new_val = 25 - change_amt

    assert len(result) == 1
    assert result[0]["param"] == "maxtwohitnetgainloss"
    assert result[0]["value"] == pytest.approx(expected_new_val)
    assert opt.prevParams == opt.params
    assert opt.prevResults is freshResults


def test_run_incr_iteration_scales_change_with_config_changebaseincrperiter():
    freshParams = [{"track_id": 1, "param": "maxtwohitnetgainloss", "value": 25}]
    freshResults = [{"Result": "twohits", "ResultValue": 0.5, "ResultValueIterative": 0.5, "Weighting": 30}]

    # Both increments must keep newVal within [0, 50] or the change gets
    # clamped away entirely (paramMaxed) instead of scaling -- see
    # test_run_incr_iteration_clamps_to_bound_marks_maxed_without_crashing.
    small = _optimizer_for_incr_iteration(GameConfig(changebaseincrperiter=0.01))
    big = _optimizer_for_incr_iteration(GameConfig(changebaseincrperiter=0.02))

    result_small = small.runIncrIteration(list(freshParams), list(freshResults))
    result_big = big.runIncrIteration(list(freshParams), list(freshResults))

    change_small = 25 - result_small[0]["value"]
    change_big = 25 - result_big[0]["value"]
    assert change_big == pytest.approx(change_small * 2)


def test_run_incr_iteration_clamps_to_bound_marks_maxed_without_crashing():
    # A huge weighted result pushes newVal out of bounds -- code should
    # just leave the param unchanged (paramMaxed=True) rather than error.
    config = GameConfig(changebaseincrperiter=1.0)
    opt = _optimizer_for_incr_iteration(config)
    freshParams = [{"track_id": 1, "param": "maxtwohitnetgainloss", "value": 25}]
    freshResults = [{"Result": "twohits", "ResultValue": 100, "ResultValueIterative": 100, "Weighting": 30}]

    result = opt.runIncrIteration(freshParams, freshResults)
    assert result[0]["value"] == 25  # unchanged, since the computed newVal was out of bounds


# ---------------------------------------------------------------------
# __init__ / retrievePairingsSettings -- integration, real temp sqlite db
# ---------------------------------------------------------------------

@pytest.mark.integration
def test_optimizer_init_reads_pairings_and_bounds_from_temp_db(tmp_path):
    (tmp_path / "etc").mkdir()
    config = GameConfig(data_root=tmp_path)
    conn = sqlite3.connect(config.optimizer_db_path)
    conn.executescript(
        """
        CREATE TABLE OptimizerParamPairings (Result TEXT, Param TEXT, Trackwise INTEGER, Inverse INTEGER, Active INTEGER);
        CREATE TABLE BoardTrackHints (Board_ID INTEGER, Track_ID INTEGER, Param TEXT, LBound REAL, UBound REAL, Active INTEGER);
        """
    )
    conn.execute("INSERT INTO OptimizerParamPairings VALUES ('twohits', 'maxtwohitnetgainloss', 0, 0, 1)")
    conn.execute("INSERT INTO BoardTrackHints VALUES (1, -1, 'maxtwohitnetgainloss', 0, 50, 1)")
    conn.commit()
    conn.close()

    class FakeBoard:
        boardID = 1

    opt = Optimizer(FakeBoard(), optimizerRunSet=1, config=config)

    assert opt.pairings_df.loc["twohits"]["Param"] == "maxtwohitnetgainloss"
    assert opt.absoluteBounds.loc["maxtwohitnetgainloss"]["UBound"] == 50
