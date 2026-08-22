"""
Pure per-result-block helpers pulled out of Evaluator.detMetrics().

detMetrics() used to be one 193-line method appending one named "result"
dict (or a few) to `self.results` per statistic, all inline. Most of
those blocks read only a handful of specific inputs (an event count, a
config target, a list of moves) rather than needing the whole
`Evaluator` instance -- so each becomes its own function here, taking
exactly what it needs and returning the result dict(s) it used to
append directly.

A few blocks (`eventSpacingHist_curvefit`, `eventsOverTime_curvefit`,
`energy_curvefit`, `velocity_curvefit`, and the onlyGameBoardStats-only
`trackEventLengthDistribution_curvefit_T*` block) genuinely do need
`Evaluator.processActualHistCurve`/`Evaluator.discreteRegression`,
which delegate to `self.eventSetBuilder`'s curve-math methods -- those
stay as private methods on `Evaluator` itself (see `_event_spacing_
histogram_result` etc. in Evaluator.py) rather than moving here, since
turning them into free functions too would mean either passing the
whole `EventSetBuilder` through as a parameter or duplicating its
curve-math delegation. Worth revisiting in a later pass (Evaluator
could call `cribsandladders.event_curve_math` directly instead of
proxying through `self.eventSetBuilder`, since that's exactly what
EventSetBuilder's own delegating wrappers already do) -- out of scope
for this one.

`Evaluator.detMetrics()` is now a thin orchestrator, same as
`Stats.calc_metrics()`: call each function/method below in the same
order the original inline code ran in, and extend/append `self.results`
with what comes back. No logic changed -- every function body here is
the original inline code, unindented and given a name and a return
value instead of `self.results.append(...)` mid-method.
"""
import statistics as stt


def structure_scalar_stats(events, orthos, multis, cancels, config):
    """
    The "GAME BOARD STRUCTURE SCALAR STATS" scalar block: orthos,
    multis, cancels -- how far the actual orthogonal/multi-track/
    cancelled-event rates are from their configured targets.
    """
    if events > 0:
        orthos_val = abs(config.optorthospct - orthos / events)
        orthos_it = config.optorthospct - orthos / events
    else:
        orthos_val, orthos_it = config.optorthospct, config.optorthospct

    if events > 0:
        multis_val = abs(config.optmultispct - multis / events)
        multis_it = config.optmultispct - multis / events
    else:
        multis_val, multis_it = config.optmultispct, config.optmultispct

    if events > 0:
        cancels_val = (cancels / events) - config.idealcancelspct
        # If too few cancels, don't sweat it, meshes with iter model
        if cancels_val < 0:
            cancels_val = 0
    else:
        cancels_val = 0

    return [
        dict(Result="orthos", ResultFlavour="GAME BOARD STRUCTURE SCALAR STATS",
             ResultValue=orthos_val, ResultValueIterative=orthos_it, Weighting=0.5),
        dict(Result="multis", ResultFlavour="GAME BOARD STRUCTURE SCALAR STATS",
             ResultValue=multis_val, ResultValueIterative=multis_it, Weighting=1),
        dict(Result="cancels", ResultFlavour="GAME BOARD STRUCTURE SCALAR STATS",
             ResultValue=cancels_val, Weighting=5),
    ]


def early_termination_stats(event_nodes_by_track, board, config):
    """One `earlytermination_T{track_id}` result per track in
    `event_nodes_by_track` -- how far short of the finish line
    (minus `config.finishlinelength`) the furthest-placed event node
    got."""
    results = []
    for n in event_nodes_by_track:
        track = board.getTrackByNum(n['tracknum'])
        track_id = track.Track_ID
        maxNode = max(n['nodes'])
        termPct = min(maxNode / (track.length - config.finishlinelength), 1.0)
        results.append(dict(Result="earlytermination_T{}".format(track_id), ResultFlavour="GAMEPLAY SCALAR STATS",
                            ResultValue=1.0 - termPct, Weighting=8))
    return results


def balance_stats(partial_balance_set, board):
    """Overall + per-track `balance` results from `Stats.partialBalanceSet`
    (a list of `(tracknum, balance_fraction)` pairs)."""
    results = [dict(Result="balance", ResultFlavour="GAMEPLAY SCALAR STATS",
                    ResultValue=stt.stdev([b[1] for b in partial_balance_set]), Weighting=0)]
    for b in partial_balance_set:
        track_id = board.getTrackByNum(b[0]).Track_ID
        results.append(dict(Result="balance_T{}".format(track_id), ResultFlavour="GAMEPLAY SCALAR STATS",
                            ResultValue=b[1], Weighting=0))
    return results


def gamelength_stat(avglengthinrounds, config):
    """How far `avglengthinrounds` is from `config.idealgamelength`."""
    if config.idealgamelength > 0:
        gamelengthstatit = (avglengthinrounds - config.idealgamelength) / config.idealgamelength
    else:
        gamelengthstatit = 1
    return dict(Result="gamelength", ResultFlavour="GAMEPLAY SCALAR STATS",
               ResultValue=abs(gamelengthstatit), ResultValueIterative=gamelengthstatit, Weighting=0)


def twohits_stat(moves, tracks, config):
    """Fraction of moves that landed on an event immediately after
    another event on the same track (a "two-hit"), vs.
    `config.opttwohitspct`."""
    trackNumChecks = [dict(tracknum=t.num, prevwasevent=False) for t in tracks]

    twoHits = []
    for m in moves:
        curTrack = None
        for tn in trackNumChecks:
            if tn['tracknum'] == m.track:
                curTrack = tn
                break
        if m.hasEvent:
            if curTrack['prevwasevent']:
                twoHits.append(dict(tracknum=curTrack['tracknum'], movenum=m.movenum))
            curTrack['prevwasevent'] = True
        else:
            curTrack['prevwasevent'] = False

    if len(moves) > 0:
        twohitsstatit = len(twoHits) / len(moves) - config.opttwohitspct
    else:
        twohitsstatit = 1
    return dict(Result="twohits", ResultFlavour="GAMEPLAY SCALAR STATS",
               ResultValue=abs(twohitsstatit), ResultValueIterative=twohitsstatit, Weighting=30)


def soexcite_stat(soexcitespegging):
    """Inverse of `Stats.soexcitespegging` (maximize so-excites), capped
    at 1.0 once the rate drops to/under 0.1."""
    return dict(Result="soexcite", ResultFlavour="GAMEPLAY SCALAR STATS",
               ResultValue=1.0 / soexcitespegging if soexcitespegging > 0.1 else 1.0, Weighting=0.1)


def repeats_stat(repeats, num_moves):
    """`Stats.repeats` normalized by move count (minimize)."""
    return dict(Result="repeats", ResultFlavour="GAMEPLAY SCALAR STATS",
               ResultValue=repeats / num_moves if num_moves > 0 else 1, Weighting=400)


def events_hit_skew_stat(moves):
    """How skewed ladder-hits vs. chute-hits are, as a fraction of all
    event hits."""
    laddersHit = len([m for m in moves if m.hasEvent and m.ladderorchuteamt > 0])
    chutesHit = len([m for m in moves if m.hasEvent and m.ladderorchuteamt < 0])
    skewEvents = 0 if laddersHit + chutesHit == 0 else abs(laddersHit - chutesHit) / (laddersHit + chutesHit)
    return dict(Result="eventshitskew", ResultFlavour="GAMEPLAY SCALAR STATS", ResultValue=skewEvents, Weighting=1)


def event_length_distribution_stats(moves, tracks, hist_fn, regression_fn, config):
    """
    Overall `eventsHitLengthDistribution_curvefit` plus a per-track
    `eventsHitLengthDistribution_bottomheavy_T{track_id}` result
    whenever more than half of a track's hit events were length <= 4.

    `hist_fn`/`regression_fn` are `Evaluator.processActualHistCurve`/
    `Evaluator.discreteRegression` (or fakes with the same signature in
    tests) -- see this module's docstring for why those didn't move
    here too.
    """
    eventsLengthHist_l = hist_fn([abs(m.ladderorchuteamt) for m in moves if m.hasEvent])
    if len(eventsLengthHist_l) == 0:
        result = 1
    else:
        result = regression_fn(config.eventlengthdisthistcurvefile, eventsLengthHist_l)
    results = [dict(Result="eventsHitLengthDistribution_curvefit",
                    ResultFlavour="GAMEPLAY STATISTICAL STATS (lol)",
                    ResultValue=result, Weighting=10)]

    # If more than 50% of events are between 2 & 4 spaces, penalize (track-wise)
    for t in tracks:
        tracksByLength_l = [abs(m.ladderorchuteamt) for m in moves if (m.hasEvent and m.track == t.num)]
        shortTracks_l = [e for e in tracksByLength_l if e <= 4]
        if len(tracksByLength_l) > 0 and len(shortTracks_l) * 2 > len(tracksByLength_l):
            results.append(dict(Result="eventsHitLengthDistribution_bottomheavy_T{}".format(t.Track_ID),
                                ResultFlavour="GAMEPLAY STATISTICAL STATS (lol)",
                                ResultValue=len(shortTracks_l) / len(tracksByLength_l) - 0.5, Weighting=10))
    return results
