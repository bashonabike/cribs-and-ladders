"""
Refactor Mk II, Phase 6: unit tests for cribsandladders.stats_metrics --
the per-statistic functions pulled out of Stats.calc_metrics(). See that
module's docstring for why each function exists, and
tests/test_stats.py for calc_metrics()'s own (still-passing, unchanged)
black-box coverage of the orchestrator.

Each function is exercised with a small, hand-built scenario (2-3
Move/Track objects), built the same way tests/test_stats.py already
does, rather than hand-built raw DataFrames -- prep_joined_moves_df's
output shape (which columns even exist depends on whether any
ladders/chutes/events are present at all) is exactly what these
functions need and isn't worth hand-guessing when building it from real
domain objects is one function call away.
"""
import pytest

import Enums as en
from cribsandladders.Board import Track, Chute, Ladder
from cribsandladders.Stats import Move
from cribsandladders import stats_metrics as sm


def _track_with_one_chute_and_one_ladder(num=1):
    # chute: 10 -> 4 (land on 10, end up at 4); ladder: 6 -> 12
    track = Track()
    track.num = num
    track.Track_ID = num
    track.setChutes([Chute(10, 4, num)])
    track.setLadders([Ladder(6, 12, num)])
    return track


# ---------------------------------------------------------------------
# prep_joined_moves_df
# ---------------------------------------------------------------------

def test_prep_joined_moves_df_joins_moves_against_track_events():
    track = _track_with_one_chute_and_one_ladder()
    moves = [
        Move(0, 1, track, 1, 1, 1, oldScore=0, baseScore=0, reason="peg",
             event=en.Event.CHUTE, newScore=4, soexcite=False, pegMove=True),
        Move(0, 1, track, 2, 1, 1, oldScore=0, baseScore=0, reason="peg",
             event=en.Event.LADDER, newScore=12, soexcite=True, pegMove=True),
    ]

    joined_df, moves_df, ladders_df, chutes_df, games_df, games_by_track_df = (
        sm.prep_joined_moves_df(moves, [track]))

    assert len(joined_df) == 2
    assert len(moves_df) == 2
    # the chute move landed on currpos 4 == the chute's *end* -> joins against end_c
    assert joined_df.loc[joined_df['movenum'] == 1, 'end_c'].iloc[0] == 4
    # the ladder move landed on currpos 12 == the ladder's end -> joins against end_l
    assert joined_df.loc[joined_df['movenum'] == 2, 'end_l'].iloc[0] == 12
    assert games_df['moves'].sum() == 2
    assert len(games_by_track_df) == 1


def test_prep_joined_moves_df_handles_track_with_no_events():
    track = Track()
    track.num = 1
    moves = [Move(0, 1, track, 1, 1, 1, 0, 0, "peg", en.Event.NONE, 5, False, True)]

    joined_df, *_ = sm.prep_joined_moves_df(moves, [track])

    assert len(joined_df) == 1
    assert joined_df['end_c'].isna().all()
    assert joined_df['end_l'].isna().all()


# ---------------------------------------------------------------------
# calc_soexcites_and_repeats
# ---------------------------------------------------------------------

def test_calc_soexcites_and_repeats_counts_soexcite_moves_per_track():
    track = _track_with_one_chute_and_one_ladder()
    moves = [
        Move(0, 1, track, 1, 1, 1, 0, 0, "peg", en.Event.NONE, 1, soexcite=True, pegMove=True),
        Move(0, 1, track, 2, 1, 1, 1, 0, "peg", en.Event.NONE, 2, soexcite=False, pegMove=True),
    ]
    joined_df, *_ = sm.prep_joined_moves_df(moves, [track])

    soexcitespeggingbytrack, soexcitespegging, repeatsbytrack, repeats = (
        sm.calc_soexcites_and_repeats(joined_df, numtrials=2))

    assert soexcitespeggingbytrack == [0.5]  # 1 soexcite move / 2 trials
    assert soexcitespegging == 0.5
    assert repeatsbytrack == []
    assert repeats == 0


def test_calc_soexcites_and_repeats_counts_a_repeat_when_same_event_hit_twice_in_one_trial():
    track = _track_with_one_chute_and_one_ladder()
    # Both moves are in the same trial and land on the ladder's end (12)
    # -> same (oldpos, newpos) eventhit tuple hit twice -> one "repeat".
    moves = [
        Move(0, 1, track, 1, 1, 1, oldScore=0, baseScore=0, reason="peg",
             event=en.Event.LADDER, newScore=12, soexcite=False, pegMove=True),
        Move(0, 1, track, 2, 1, 2, oldScore=0, baseScore=0, reason="peg",
             event=en.Event.LADDER, newScore=12, soexcite=False, pegMove=True),
    ]
    joined_df, *_ = sm.prep_joined_moves_df(moves, [track])

    _, _, repeatsbytrack, repeats = sm.calc_soexcites_and_repeats(joined_df, numtrials=1)
    assert repeatsbytrack == [1.0]
    assert repeats == 1.0


# ---------------------------------------------------------------------
# calc_avg_length_in_rounds
# ---------------------------------------------------------------------

def test_calc_avg_length_in_rounds_averages_max_round_per_trial():
    track = Track()
    track.num = 1
    moves = [
        Move(0, 1, track, 1, 1, 1, 0, 0, "peg", en.Event.NONE, 1, False, True),   # trial 1, round 1
        Move(0, 1, track, 2, 3, 1, 0, 0, "peg", en.Event.NONE, 2, False, True),   # trial 1, round 3 (max)
        Move(0, 2, track, 1, 1, 1, 0, 0, "peg", en.Event.NONE, 1, False, True),   # trial 2, round 1
    ]
    joined_df, *_ = sm.prep_joined_moves_df(moves, [track])

    # trial 1 -> max round 3; trial 2 -> max round 1; (3 + 1) / numtrials(2) = 2.0
    assert sm.calc_avg_length_in_rounds(joined_df, numtrials=2) == 2.0


# ---------------------------------------------------------------------
# calc_events_by_track
# ---------------------------------------------------------------------

def test_calc_events_by_track_counts_chute_and_ladder_hits():
    track = _track_with_one_chute_and_one_ladder()
    moves = [
        Move(0, 1, track, 1, 1, 1, oldScore=0, baseScore=0, reason="peg",
             event=en.Event.CHUTE, newScore=4, soexcite=False, pegMove=True),
        Move(0, 2, track, 1, 1, 1, oldScore=0, baseScore=0, reason="peg",
             event=en.Event.LADDER, newScore=12, soexcite=False, pegMove=True),
    ]
    joined_df, *_ = sm.prep_joined_moves_df(moves, [track])

    chutesbytrack, laddersbytrack, eventsbytrack, chutes, ladders, events = (
        sm.calc_events_by_track(joined_df, numtrials=2))

    assert chutesbytrack == [0.5]
    assert laddersbytrack == [0.5]
    assert eventsbytrack == [1.0]
    assert chutes == pytest.approx(0.5)
    assert ladders == pytest.approx(0.5)
    assert events == pytest.approx(1.0)


# ---------------------------------------------------------------------
# calc_move_histograms
# ---------------------------------------------------------------------

def test_calc_move_histograms_only_includes_moves_that_hit_an_event():
    track = _track_with_one_chute_and_one_ladder()
    moves = [
        Move(0, 1, track, 1, 1, 1, 0, 0, "peg", en.Event.NONE, 1, False, True),
        Move(0, 1, track, 2, 1, 1, oldScore=0, baseScore=0, reason="peg",
             event=en.Event.LADDER, newScore=12, soexcite=False, pegMove=True),
    ]
    joined_df, moves_df, ladders_df, chutes_df, games_df, games_by_track_df = (
        sm.prep_joined_moves_df(moves, [track]))

    hist_df, hist_by_track_df = sm.calc_move_histograms(joined_df, games_df, games_by_track_df)

    # only the ladder move (movenum=2 of 2 total moves in the trial) has a
    # non-null end_e -> the one row in the histogram.
    assert len(hist_df) == 1
    assert hist_df['normmove'].iloc[0] == pytest.approx(1.0)
    assert len(hist_by_track_df) == 1


# ---------------------------------------------------------------------
# calc_lookforward_events_by_track
# ---------------------------------------------------------------------

def test_calc_lookforward_events_by_track_counts_ladder_starting_one_space_ahead():
    track = _track_with_one_chute_and_one_ladder()  # ladder starts at hole 6
    # newScore=5 -> currpos=5 -> posin1=6 (matches the ladder's start)
    moves = [Move(0, 1, track, 1, 1, 1, oldScore=0, baseScore=0, reason="peg",
                  event=en.Event.NONE, newScore=5, soexcite=False, pegMove=True)]
    joined_df, moves_df, ladders_df, chutes_df, games_df, games_by_track_df = (
        sm.prep_joined_moves_df(moves, [track]))

    result = sm.calc_lookforward_events_by_track(moves_df, ladders_df, chutes_df, numtrials=1)
    (laddersin1bytrack, laddersin2bytrack, chutesin1bytrack, chutesin2bytrack,
     eventsin1bytrack, eventsin2bytrack, laddersin1, laddersin2, chutesin1, chutesin2,
     eventsin1, eventsin2) = result

    assert laddersin1bytrack == [1.0]
    assert laddersin1 == 1.0
    assert chutesin1bytrack == []
    # CHARACTERIZATION, not a spec of intended behavior: eventsin1bytrack
    # is built as `[l + c for l, c in zip(laddersin1bytrack,
    # chutesin1bytrack)]` -- zip() truncates to the *shorter* of the two
    # lists, so whenever only one of ladders/chutes has any per-track
    # hits at all (as here: laddersin1bytrack=[1.0], chutesin1bytrack=[]),
    # eventsin1bytrack silently comes out empty instead of [1.0], even
    # though the *scalar* eventsin1 total below is computed independently
    # (laddersin1 + chutesin1) and is correct. Pre-existing quirk in the
    # original calc_metrics code, carried over verbatim by this
    # extraction -- not fixed here, per the same policy applied to the
    # other pre-existing bugs documented elsewhere in this test suite
    # (see e.g. test_board.py's chute/ladder eff-landing asymmetry).
    assert eventsin1bytrack == []
    assert laddersin2bytrack == []
    assert eventsin1 == 1.0
    assert eventsin2 == 0


def test_calc_lookforward_events_by_track_finds_nothing_when_not_adjacent_to_an_event():
    track = _track_with_one_chute_and_one_ladder()
    # currpos=0 -> posin1=1, posin2=2, neither matches chute start (10) or ladder start (6)
    moves = [Move(0, 1, track, 1, 1, 1, 0, 0, "peg", en.Event.NONE, 0, False, True)]
    joined_df, moves_df, ladders_df, chutes_df, games_df, games_by_track_df = (
        sm.prep_joined_moves_df(moves, [track]))

    result = sm.calc_lookforward_events_by_track(moves_df, ladders_df, chutes_df, numtrials=1)
    assert result == ([], [], [], [], [], [], 0, 0, 0, 0, 0, 0)
