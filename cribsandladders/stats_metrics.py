"""
Pure per-statistic helpers pulled out of Stats.calc_metrics().

calc_metrics() used to be one 137-line method: build a chain of merged/
joined pandas DataFrames, then compute ~20 named statistics from them,
all inline, all writing straight to instance attributes buried in the
middle of the method. That made it impossible to test any single
statistic without a full `Stats` instance (a real `Board`, `CribSquad`,
and move list) -- and impossible to tell, from a stack trace or a diff,
which of those ~20 statistics a change actually touched.

Each function here does exactly one part of that computation, given
already-built DataFrames rather than `self.board`/`self.moves`, so each
is independently unit-testable with small hand-built DataFrames -- the
same pattern already proven on `hydrate_tracks_from_dataframes`
(BoardSetter.py) and `hydrate_candidate_events_from_dataframe`
(PossibleEvents.py).

`Stats.calc_metrics()` is now a thin orchestrator: build the joined
DataFrames via `prep_joined_moves_df`, call each function below in the
same order the original inline code ran in, and assign the results to
`self.*`. No logic changed here -- every function body is the original
inline code, unindented and given a name and a return statement instead
of `self.x = ...` mid-method.
"""
import pandas as pd


def prep_joined_moves_df(moves, tracks):
    """
    Builds the moves DataFrame joined against every track's ladders/
    chutes/all-events tables, plus the game- and (game, track)-level
    move counts several of the statistics below need.

    Args:
        moves (list[Move]): moves to summarize.
        tracks (list[Track]): board.tracks -- source of the ladders/
            chutes/events tables moves get joined against.

    Returns:
        tuple: (joined_df, moves_df, ladders_df, chutes_df, games_df,
        games_by_track_df). `moves_df`/`ladders_df`/`chutes_df` are
        returned alongside `joined_df` since
        `calc_lookforward_events_by_track` below needs the un-joined
        versions.
    """
    moves_df = pd.DataFrame.from_records([m.to_dict() for m in moves])
    ladders_df = pd.concat([t.getLaddersAsDF() for t in tracks])
    chutes_df = pd.concat([t.getChutesAsDF() for t in tracks])
    events_df = pd.concat([t.getEventsAsDF() for t in tracks])

    # set up chutes & ladder df's if none defined in input
    if ladders_df.columns is None or len(ladders_df.columns) == 0:
        ladders_df = pd.DataFrame(columns=['track', 'start_l', 'end_l'])
    else:
        ladders_df.rename(columns={"start": "start_l", "end": "end_l"}, inplace=True)
    if chutes_df.columns is None or len(chutes_df.columns) == 0:
        chutes_df = pd.DataFrame(columns=['track', 'start_c', 'end_c'])
    else:
        chutes_df.rename(columns={"start": "start_c", "end": "end_c"}, inplace=True)
    if events_df.columns is None or len(events_df.columns) == 0:
        events_df = pd.DataFrame(columns=['track', 'start_e', 'end_e'])
    else:
        events_df.rename(columns={"start": "start_e", "end": "end_e"}, inplace=True)

    moves_df.sort_values(['trialmux', 'track', 'player', 'movenum'], inplace=True)
    ladders_df.sort_values(['track', 'end_l'], inplace=True)
    chutes_df.sort_values(['track', 'end_c'], inplace=True)
    events_df.sort_values(['track', 'end_e'], inplace=True)

    # Join into data analytics tables
    joined_df = pd.merge(moves_df, ladders_df, left_on=['track', 'currpos'], right_on=['track', 'end_l'],
                         how='left')
    joined_df = pd.merge(joined_df, chutes_df, left_on=['track', 'currpos'], right_on=['track', 'end_c'],
                         how='left')
    joined_df = pd.merge(joined_df, events_df, left_on=['track', 'currpos'], right_on=['track', 'end_e'],
                         how='left').reset_index()

    # Rollup into game & track-level stats
    games_df = joined_df[['trialmux']].assign(moves=1).groupby('trialmux').agg('sum').reset_index()
    games_by_track_df = (joined_df[['trialmux', 'track']].assign(moves=1).groupby(['trialmux', 'track']).agg('sum')
                         .reset_index())

    return joined_df, moves_df, ladders_df, chutes_df, games_df, games_by_track_df


def calc_soexcites_and_repeats(joined_df, numtrials):
    """
    Returns (soexcitespeggingbytrack, soexcitespegging, repeatsbytrack, repeats).
    """
    soexcitespeggingbytrack = [float(e) / float(numtrials) for e in
                               ((joined_df.query('soexcite == True'))[['track']]
                                .assign(moves=1).groupby(['track'])
                                .agg('sum')['moves'].to_list())]
    soexcitespegging = sum(soexcitespeggingbytrack)
    repeatsbytrack = ([float(r) / float(numtrials) for r in joined_df.query('not end_e.isnull()')
    [['trialmux', 'track', 'eventhit']].assign(hits=1).groupby(['trialmux', 'track', 'eventhit']).agg('sum')
    .query('hits > 1').assign(repeats=1).groupby(['track']).agg('sum')['repeats'].to_list()])
    repeats = sum(repeatsbytrack)
    return soexcitespeggingbytrack, soexcitespegging, repeatsbytrack, repeats


def calc_avg_length_in_rounds(joined_df, numtrials):
    """Average number of rounds per trial (game)."""
    return (float(sum(joined_df[['trialmux', 'round']].groupby(['trialmux']).agg('max')['round']
                       .to_list())) / float(numtrials))


def calc_events_by_track(joined_df, numtrials):
    """
    Returns (chutesbytrack, laddersbytrack, eventsbytrack, chutes,
    ladders, events) -- per-track and overall chute/ladder/event hit
    rates per trial.
    """
    chutesbytrack = ([float(c) / float(numtrials) for c in joined_df[['track', 'end_c']]
    .dropna().assign(chutes=1).groupby(['track']).agg('sum').sort_values(['track'])['chutes'].to_list()])
    laddersbytrack = ([float(l) / float(numtrials) for l in joined_df[['track', 'end_l']]
    .dropna().assign(ladders=1).groupby(['track']).agg('sum').sort_values(['track'])['ladders'].to_list()])
    eventsbytrack = [l + c for (l, c) in zip(laddersbytrack, chutesbytrack)]
    ladders = sum(laddersbytrack)
    chutes = sum(chutesbytrack)
    events = sum(eventsbytrack)
    return chutesbytrack, laddersbytrack, eventsbytrack, chutes, ladders, events


def calc_move_histograms(joined_df, games_df, games_by_track_df):
    """Returns (hist_df, hist_by_track_df)."""
    # NOTE: stripping out columns from joined_df so doesn't wipe all records on dropna
    raw_hist_df = (pd.merge(games_df, joined_df[['end_e', 'trialmux', 'ladderorchuteamt', 'movenum']].reset_index(),
                            on=['trialmux'], suffixes=('_sum', '')).dropna())
    raw_hist_df['normmove'] = raw_hist_df['movenum'] / raw_hist_df['moves']
    hist_df = raw_hist_df[['normmove', 'ladderorchuteamt']].sort_values(['normmove'])

    raw_hist_by_track_df = (pd.merge(games_by_track_df, joined_df[['end_e', 'trialmux', 'track',
                                                                   'ladderorchuteamt', 'movenum']].reset_index(),
                                     on=['trialmux', 'track'], suffixes=('_sum', '')).dropna())
    raw_hist_by_track_df['normmove'] = raw_hist_by_track_df['movenum'] / raw_hist_by_track_df['moves']
    hist_by_track_df = (raw_hist_by_track_df[['track', 'normmove', 'ladderorchuteamt']]
                        .sort_values(['track', 'normmove']))
    return hist_df, hist_by_track_df


def calc_lookforward_events_by_track(moves_df, ladders_df, chutes_df, numtrials):
    """
    "Look forward" stats: how often a ladder/chute starts 1 or 2 spaces
    ahead of a move's current position (`posin1`/`posin2` on `Move`).

    Returns (laddersin1bytrack, laddersin2bytrack, chutesin1bytrack,
    chutesin2bytrack, eventsin1bytrack, eventsin2bytrack, laddersin1,
    laddersin2, chutesin1, chutesin2, eventsin1, eventsin2).
    """
    laddersin1_df = pd.merge(moves_df, ladders_df, left_on=['track', 'posin1'], right_on=['track', 'start_l'])
    chutesin1_df = pd.merge(moves_df, chutes_df, left_on=['track', 'posin1'], right_on=['track', 'start_c'])
    laddersin2_df = pd.merge(moves_df, ladders_df, left_on=['track', 'posin2'], right_on=['track', 'start_l'])
    chutesin2_df = pd.merge(moves_df, chutes_df, left_on=['track', 'posin2'], right_on=['track', 'start_c'])

    laddersin1_df.sort_values(['track', 'trialmux'])
    chutesin1_df.sort_values(['track', 'trialmux'])
    laddersin2_df.sort_values(['track', 'trialmux'])
    chutesin2_df.sort_values(['track', 'trialmux'])

    laddersin1bytrack = ([float(i) / float(numtrials) for i in
                          (laddersin1_df.groupby(['track', 'trialmux']).size().reset_index(name='counts').
                           groupby('track').agg('sum')['counts'].to_list())])
    laddersin2bytrack = ([float(i) / float(numtrials) for i in
                          (laddersin2_df.groupby(['track', 'trialmux']).size().reset_index(name='counts').
                           groupby('track').agg('sum')['counts'].to_list())])
    chutesin1bytrack = ([float(i) / float(numtrials) for i in
                         (chutesin1_df.groupby(['track', 'trialmux']).size().reset_index(name='counts').
                          groupby('track').agg('sum')['counts'].to_list())])
    chutesin2bytrack = ([float(i) / float(numtrials) for i in
                         (chutesin2_df.groupby(['track', 'trialmux']).size().reset_index(name='counts').
                          groupby('track').agg('sum')['counts'].to_list())])
    eventsin1bytrack = [l + c for l, c in zip(laddersin1bytrack, chutesin1bytrack)]
    eventsin2bytrack = [l + c for l, c in zip(laddersin2bytrack, chutesin2bytrack)]
    laddersin1 = sum(laddersin1bytrack)
    laddersin2 = sum(laddersin2bytrack)
    chutesin1 = sum(chutesin1bytrack)
    chutesin2 = sum(chutesin2bytrack)
    eventsin1 = laddersin1 + chutesin1
    eventsin2 = laddersin2 + chutesin2
    return (laddersin1bytrack, laddersin2bytrack, chutesin1bytrack, chutesin2bytrack,
            eventsin1bytrack, eventsin2bytrack, laddersin1, laddersin2, chutesin1, chutesin2,
            eventsin1, eventsin2)
