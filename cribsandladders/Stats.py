import game_params as gp
import bisect
import cribsandladders.Board as bd
import cribsandladders.CribSquad as sq
import statistics as st
import collections as col
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import pandas as pd
import Enums as en
import inspect as it
import functools as fu
import sqlite3 as sql
import datetime as dt
from io import StringIO
import os
class Stats:

    def __init__(self, board, squad):
        self.board = board
        self.squad = squad

        self.moves = []

        self.partialBalanceSet = []

        self.chutesbytrack = []
        self.laddersbytrack = []
        self.eventsbytrack = []
        self.ladders = 0
        self.chutes = 0
        self.events = 0

        self.laddersin1bytrack = []
        self.laddersin2bytrack = []
        self.chutesin1bytrack = []
        self.chutesin2bytrack = []
        self.eventsin1bytrack = []
        self.eventsin2bytrack = []        
        self.laddersin1 = 0
        self.laddersin2 = 0
        self.chutesin1 =  0
        self.chutesin2 =  0
        self.eventsin1 =  0
        self.eventsin2 =  0

        self.soexcitespeggingbytrack = []
        self.repeatsbytrack = []
        self.soexcitespegging = 0
        self.repeats = 0

        self.avglengthinrounds = 0

        self.hist_df = None
        self.hist_by_track_df = None


    #slice & dice data at end of trials, group by trial to compute stuff pointwise per trial
    # def addMoveStat(self, trial, track, moveNum, round, playerNum, oldScore, baseScore, reason, event, newScore, soexcite):
    #     eventAmt = newScore- oldScore - baseScore
    #     curMove = Move()
    #     curMove.trial = trial
    #     curMove.movenum = moveNum
    #     curMove.player = playerNum
    #     curMove.track = track.num
    #     curMove.soexcite = soexcite
    #     curMove.round = round
    #     curMove.hasEvent = event != en.Event.NONE
    #
    #     if event != en.Event.NONE:
    #         if eventAmt == 0:
    #             wtf='wtf'
    #         if event == en.Event.CHUTE:
    #             curMove.chuteamt = eventAmt
    #             curMove.chutehit = (oldScore + baseScore, newScore)
    #             curMove.eventhit = curMove.chutehit
    #         elif event == en.Event.LADDER:
    #             curMove.ladderamt = eventAmt
    #             curMove.ladderhit = (oldScore + baseScore, newScore)
    #             curMove.eventhit = curMove.ladderhit
    #
    #         curMove.ladderorchuteamt = eventAmt
    #
    #     curMove.score = newScore - oldScore
    #     if   curMove.score == 0:
    #             wtf='wtf'
    #
    #     curMove.basescore = baseScore
    #     curMove.reason = reason
    #     curMove.currpos = newScore
    #     curMove.posin1 = curMove.currpos + 1
    #     curMove.posin2 = curMove.currpos + 2
    #
    #     self.moves.append(curMove)

    def clearStatsAndSetMoves(self, curMoveSet):
        self.partialBalanceSet = []

        self.chutesbytrack = []
        self.laddersbytrack = []
        self.eventsbytrack = []
        self.ladders = 0
        self.chutes = 0
        self.events = 0

        self.laddersin1bytrack = []
        self.laddersin2bytrack = []
        self.chutesin1bytrack = []
        self.chutesin2bytrack = []
        self.eventsin1bytrack = []
        self.eventsin2bytrack = []
        self.laddersin1 = 0
        self.laddersin2 = 0
        self.chutesin1 =  0
        self.chutesin2 =  0
        self.eventsin1 =  0
        self.eventsin2 =  0

        self.soexcitespeggingbytrack = []
        self.repeatsbytrack = []
        self.soexcitespegging = 0
        self.repeats = 0

        self.avglengthinrounds = 0

        self.hist_df = None
        self.hist_by_track_df = None

        self.moves = curMoveSet

    def calc_metrics(self):
        # Build dataframes & indices
        moves_df = pd.DataFrame.from_records([m.to_dict() for m in self.moves])
        ladders_df = pd.concat([t.getLaddersAsDF() for t in self.board.tracks])
        chutes_df = pd.concat([t.getChutesAsDF() for t in self.board.tracks])
        events_df = pd.concat([t.getEventsAsDF() for t in self.board.tracks])

        #set up chutes & ladder df's if none defined in input
        if ladders_df.columns is None or len(ladders_df.columns) == 0:
            ladders_df = pd.DataFrame(columns=['track','start_l','end_l'])
        else:
            ladders_df.rename(columns={"start": "start_l", "end": "end_l"}, inplace=True)
        if chutes_df.columns is None or len(chutes_df.columns) == 0:
            chutes_df = pd.DataFrame(columns=['track','start_c','end_c'])
        else:
            chutes_df.rename(columns={"start": "start_c", "end": "end_c"}, inplace=True)
        if events_df.columns is None or len(events_df.columns) == 0:
            events_df = pd.DataFrame(columns=['track','start_e','end_e'])
        else:
            events_df.rename(columns={"start": "start_e", "end": "end_e"}, inplace=True)

        #TODO: figure out how inde3xintg works???
        # moves_df.set_index(['track', 'currpos'], inplace=True, drop=False)
        # moves_df.index.name = 'idx_track_currpos'
        # moves_df.set_index(['trial', 'track', 'player', 'movenum'], inplace=True, drop=False)
        # moves_df.index.name = 'idx_main'
        # ladders_df.set_index(['track', 'end_l'], inplace=True, drop=False)
        # ladders_df.index.name = 'idx_main'
        # chutes_df.set_index(['track', 'end_c'], inplace=True, drop=False)
        # chutes_df.index.name = 'idx_main'
        # events_df.set_index(['track', 'end_e'], inplace=True, drop=False)
        # events_df.index.name = 'idx_main'
        #
        # moves_df.sort_index(inplace=True)
        # ladders_df.sort_index(inplace=True)
        # chutes_df.sort_index(inplace=True)
        # events_df.sort_index(inplace=True)

        moves_df.sort_values(['trialmux', 'track', 'player', 'movenum'], inplace=True)
        ladders_df.sort_values(['track', 'end_l'], inplace=True)
        chutes_df.sort_values(['track', 'end_c'], inplace=True)
        events_df.sort_values(['track', 'end_e'], inplace=True)

        #Join into data analytics tables
        joined_df = pd.merge(moves_df, ladders_df, left_on=['track', 'currpos'], right_on=['track', 'end_l'],
                             how = 'left')
        joined_df = pd.merge(joined_df, chutes_df, left_on=['track', 'currpos'], right_on=['track', 'end_c'],
                             how = 'left')
        joined_df = pd.merge(joined_df, events_df, left_on=['track', 'currpos'], right_on=['track', 'end_e'],
                             how = 'left').reset_index()
        # joined_df.set_index(['trialmux', 'track', 'player', 'movenum'], inplace=True, drop=False)

        #Rollup into game & track-level stats
        #NOTE: include columns needed at game level in spec
        games_df = joined_df[['trialmux']].assign(moves=1).groupby('trialmux').agg('sum').reset_index()
        games_by_track_df = (joined_df[['trialmux', 'track']].assign(moves=1).groupby(['trialmux', 'track']).agg('sum')
                             .reset_index())

        #Compute game & track-level stats
        self.soexcitespeggingbytrack = [float(e)/float(gp.numtrials) for e in ((joined_df.query('soexcite == True'))[['track']]
                                                                 .assign(moves=1).groupby(['track'])
                                                                 .agg('sum')['moves'].to_list())]
        self.soexcitespegging = sum(self.soexcitespeggingbytrack)
        self.repeatsbytrack  = ([float(r)/float(gp.numtrials) for r in joined_df.query('not end_e.isnull()')
        [['trialmux', 'track', 'eventhit']].assign(hits=1).groupby(['trialmux', 'track', 'eventhit']).agg('sum')
        .query('hits > 1').assign(repeats=1).groupby(['track']).agg('sum')['repeats'].to_list()])
        self.repeats = sum(self.repeatsbytrack)

        self.avglengthinrounds = (float(sum(joined_df[['trialmux', 'round']].groupby(['trialmux']).agg('max')['round']
                                            .to_list()))/float(gp.numtrials))

        self.chutesbytrack = ([float(c)/float(gp.numtrials) for c in joined_df[['track', 'end_c']]
        .dropna().assign(chutes=1).groupby(['track']).agg('sum').sort_values(['track'])['chutes'].to_list()])
        self.laddersbytrack = ([float(l)/float(gp.numtrials) for l in joined_df[['track', 'end_l']]
        .dropna().assign(ladders=1).groupby(['track']).agg('sum').sort_values(['track'])['ladders'].to_list()])
        self.eventsbytrack = [l + c for (l, c) in zip(self.laddersbytrack, self.chutesbytrack)]
        self.ladders = sum(self.laddersbytrack)
        self.chutes = sum(self.chutesbytrack)
        self.events = sum(self.eventsbytrack)

        #Build histagrams of events
        #NOTE: stripping out columns from joined_df so doesn't wipe all records on dropna
        raw_hist_df = (pd.merge(games_df, joined_df[['end_e', 'trialmux', 'ladderorchuteamt', 'movenum']].reset_index(),
                                on=['trialmux'],suffixes=('_sum', '')).dropna())
        raw_hist_df['normmove'] = raw_hist_df['movenum'] / raw_hist_df['moves']
        # self.hist_df = raw_hist_df[['normmove', 'ladderorchuteamt']].set_index(['normmove']).sort_index()
        self.hist_df = raw_hist_df[['normmove', 'ladderorchuteamt']].sort_values(['normmove'])
        raw_hist_by_track_df = (pd.merge(games_by_track_df, joined_df[['end_e', 'trialmux', 'track',
        'ladderorchuteamt', 'movenum']].reset_index(), on=['trialmux', 'track'], suffixes=('_sum', '')).dropna())
        raw_hist_by_track_df['normmove'] = raw_hist_by_track_df['movenum'] / raw_hist_by_track_df['moves']
        # self.hist_by_track_df = (raw_hist_by_track_df[['track', 'normmove', 'ladderorchuteamt']]
        #                     .set_index(['track', 'normmove']).sort_index())
        self.hist_by_track_df = (raw_hist_by_track_df[['track', 'normmove', 'ladderorchuteamt']]
                            .sort_values(['track', 'normmove']))




        #Build look forward events dataframes
        laddersin1_df = pd.merge(moves_df, ladders_df, left_on=['track', 'posin1'], right_on=['track', 'start_l'])
        chutesin1_df = pd.merge(moves_df, chutes_df, left_on=['track', 'posin1'], right_on=['track', 'start_c'])
        laddersin2_df = pd.merge(moves_df, ladders_df, left_on=['track', 'posin2'], right_on=['track', 'start_l'])
        chutesin2_df = pd.merge(moves_df, chutes_df, left_on=['track', 'posin2'], right_on=['track', 'start_c'])

        laddersin1_df.sort_values(['track', 'trialmux'])
        chutesin1_df.sort_values(['track', 'trialmux'])
        laddersin2_df.sort_values(['track', 'trialmux'])
        chutesin2_df.sort_values(['track', 'trialmux'])

        self.laddersin1bytrack = ([float(i)/float(gp.numtrials) for i in
                                   (laddersin1_df.groupby(['track', 'trialmux']).size().reset_index(name='counts').
                                  groupby('track').agg('sum')['counts'].to_list())])
        self.laddersin2bytrack = ([float(i)/float(gp.numtrials) for i in
                                   (laddersin2_df.groupby(['track', 'trialmux']).size().reset_index(name='counts').
                                  groupby('track').agg('sum')['counts'].to_list())])
        self.chutesin1bytrack = ([float(i)/float(gp.numtrials) for i in
                                  (chutesin1_df.groupby(['track', 'trialmux']).size().reset_index(name='counts').
                                 groupby('track').agg('sum')['counts'].to_list())])
        self.chutesin2bytrack = ([float(i)/float(gp.numtrials) for i in
                                  (chutesin2_df.groupby(['track', 'trialmux']).size().reset_index(name='counts').
                                 groupby('track').agg('sum')['counts'].to_list())])
        self.eventsin1bytrack = [l + c for l, c in zip(self.laddersin1bytrack, self.chutesin1bytrack)]
        self.eventsin2bytrack = [l + c for l, c in zip(self.laddersin2bytrack, self.chutesin2bytrack)]
        self.laddersin1 = sum(self.laddersin1bytrack)
        self.laddersin2 = sum(self.laddersin2bytrack)
        self.chutesin1 = sum(self.chutesin1bytrack)
        self.chutesin2 = sum(self.chutesin2bytrack)
        self.eventsin1 = self.laddersin1 + self.chutesin1
        self.eventsin2 = self.laddersin2 + self.chutesin2

    def buildSet4PlusInsertSnippet(self, partialSet, overall = None, end = False):
        snippet_sb = StringIO()
        if overall is not None:
            snippet_sb.write("{},".format(overall))
        snippet_sb.write("".join(["{},".format(v) for v in partialSet]))
        snippet_sb.write("".join(["NULL," for n in range(len(partialSet) + 1, 5)]))

        snippet = snippet_sb.getvalue()
        snippet_sb.close()
        if end:
            return snippet[:-1]
        return snippet

    def insertStatsRecord(self):
        sqliteConn = sql.connect('Boards/AllBoards.db')
        sqliteCursor = sqliteConn.cursor()

        #Prepend columns to write to, all except auto-incrementing Stat_ID
        insertstatquery_sb = StringIO()
        insertstatquery_sb.write(gp.insertstatstub)

        #board level setup stats
        tracksList_sb = StringIO()
        if gp.tracksused is None:
            for p in self.squad.players:
                tracksList_sb.write(str(p.tracknum))
                tracksList_sb.write(",")
        elif gp.tracksused is list:
            for t in gp.tracksused:
                tracksList_sb.write(t)
                tracksList_sb.write(",")
        else:
            raise Exception("Invalid tracksused setting in game_params: {}".format(gp.tracksused))
        pos = tracksList_sb.tell()
        tracksList_sb.seek(0, os.SEEK_END)
        if tracksList_sb.tell() == 0:
            raise Exception("Tracks list was blank")
        tracksList_sb.seek(pos)
        tracksList = tracksList_sb.getvalue()[:-1]
        tracksList_sb.close()
        #NOTE: this is less efficient than including values directly using '?'s but this insert only fires infrequently
        insertstatquery_sb.write("{},\'{}\',{},{},{},\'{}\',\'{}\',{},".format(self.board.boardID, dt.datetime.now(),
                                                                    gp.numtrials,gp.numplayers, gp.numdecks,
                                                                    tracksList,
                                                                    self.board.boardName, self.avglengthinrounds))

        #balance stats
        #NOTE: we cannot use player.wins with the multprocessing!
        winsByPlayer = col.Counter([m.player for m in self.moves if m.winningMove])
        self.partialBalanceSet = [float(winsByPlayer[p.num]) - (float(gp.numtrials) / float(gp.numplayers)) /
                                  float(gp.numtrials) for p in self.squad.players]
        insertstatquery_sb.write(self.buildSet4PlusInsertSnippet(self.partialBalanceSet))
        self.partialBalanceSet = [s for s in zip([p.tracknum for p in self.squad.players], self.partialBalanceSet)]

        #track stats
        insertstatquery_sb.write(self.buildSet4PlusInsertSnippet(self.soexcitespeggingbytrack, self.soexcitespegging))
        insertstatquery_sb.write(self.buildSet4PlusInsertSnippet(self.repeatsbytrack, self.repeats))

        insertstatquery_sb.write(self.buildSet4PlusInsertSnippet(self.chutesbytrack, self.chutes))
        insertstatquery_sb.write(self.buildSet4PlusInsertSnippet(self.laddersbytrack, self.ladders))
        insertstatquery_sb.write(self.buildSet4PlusInsertSnippet(self.eventsbytrack, self.events))

        insertstatquery_sb.write(self.buildSet4PlusInsertSnippet(self.chutesin1bytrack, self.chutesin1))
        insertstatquery_sb.write(self.buildSet4PlusInsertSnippet(self.laddersin1bytrack, self.laddersin1))
        insertstatquery_sb.write(self.buildSet4PlusInsertSnippet(self.eventsin1bytrack, self.eventsin1))

        insertstatquery_sb.write(self.buildSet4PlusInsertSnippet(self.chutesin2bytrack, self.chutesin2))
        insertstatquery_sb.write(self.buildSet4PlusInsertSnippet(self.laddersin2bytrack, self.laddersin2))
        insertstatquery_sb.write(self.buildSet4PlusInsertSnippet(self.eventsin2bytrack, self.eventsin2, True))

        #finalize query and commit data
        insertstatquery_sb.write(")")
        query = insertstatquery_sb.getvalue()
        insertstatquery_sb.close()
        sqliteCursor.execute(query)
        sqliteConn.commit()
        #TODO: insert file links to heatmaps once generated


    def print_metrics(self):
        #TODO: also commit this to data table in sql
        with open(("./Board_Results/" + self.board.boardName + "-" + str(gp.numplayers) + "-" + str(gp.numdecks) +"-" +
                   str(gp.numtrials) + "-" +  datetime.datetime.now().strftime("%y%m%d%H%M%S") + ".txt"), "w") as results:
            results.write("\t" + self.board.boardName + "\n")
            results.write("Lengths\t")

            strTemp = ""
            for player in self.squad.players:
                strTemp += "{}/".format(self.board.getTrackByNum(player.tracknum).efflength)
            results.write(strTemp[:-1] + "\n")

            results.write("# trials\t" + str(gp.numtrials) + "\n")
            results.write("# decks\t" + str(gp.numdecks) + "\n")
            results.write("# players\t" + str(gp.numplayers) + "\n")

            results.write("balance (<5% reasonable for traditional board, for comparison) \t")
            strTemp = ""
            for player in self.squad.players:
                winDif = (float(player.wins) - (float(gp.numtrials)/float(gp.numplayers)))/float(gp.numtrials)
                strTemp += "{:.2%} (Player {}), ".format(winDif, player.num)
            results.write(strTemp[:-2] + "\n")

            results.write("So excites (when some pegging options yield events and others don't)\t" + str(st.mean(self.soexcites_pegs)) + "\n")
            results.write("Boring repeats\t" + str(st.mean(self.repeats)) + "\n")
            results.write("Snakes/Ladders\t" + str(st.mean(self.events)) + "\n")
            results.write("Snakes/Ladders in 1 incr\t" + str(st.mean(self.eventsin1)) + "\n")
            results.write("Snakes/Ladders in 2 incrs\t" + str(st.mean(self.eventsin2)) + "\n")
            results.write("# rounds\t" + str(st.mean(self.lengths_in_rounds)) + "\n")
            results.write("Likelihood So excites (pegging) per round\t{:.2%}\n".
                          format(st.mean(self.soexcites_pegs)/st.mean(self.lengths_in_rounds)))
            results.write("Likelihood Boring repeats per round\t{:.2%}\n".
                          format(st.mean(self.repeats)/st.mean(self.lengths_in_rounds)))
            results.write("Likelihood Snakes/Ladders per round\t{:.2%}\n".
                          format(st.mean(self.events)/st.mean(self.lengths_in_rounds)))
            results.write("Likelihood Snakes/Ladders in 1 incr per round\t{:.2%}\n".
                          format(st.mean(self.eventsin1)/st.mean(self.lengths_in_rounds)))
            results.write("Likelihood Snakes/Ladders in 2 incrs per round\t{:.2%}\n".
                          format(st.mean(self.eventsin2)/st.mean(self.lengths_in_rounds)))


            # event_norm_mags_sorted = sorted(event_norm_mags, key = lambda e: (e[0], e[1]))
            # results.write("\n--------------------------------------------------------------\n")
            # results.write("Game_Duration\tEvent_Magnitude\n")
            # for bin, mag in event_norm_mags_sorted:
            #     results.write("{}\t{}\n".format(bin,mag))

    def print_temp_maps(self):
        if self.hist_df is None or self.hist_df.empty:
            return

        for tracknum in set(self.hist_by_track_df.groupby('track').to_list('track').append(0)):
            cur_df = (self.hist_by_track_df.loc[self.hist_by_track_df['track'] == tracknum]
                      [['normmove', 'ladderorchuteamt']] if tracknum > 0
                      else self.hist_df)
            cur_df.rename(columns={'normmove':'Game_Duration', 'ladderorchuteamt':'Event_Magnitude'}, inplace=True)
            title = 'Played Heat: Track {}'.format(tracknum) if tracknum > 0 else 'Played Heat: Overall'
            filename = ("./Board_Results/images/{}-{}-{}-{}-{}-{}.png".
                        format(self.board.boardName,str(gp.numplayers),str(gp.numdecks),
                               str(gp.numtrials),"Overall" if tracknum > 0 else "Track {}".format(tracknum),
                               datetime.datetime.now().strftime("%y%m%d%H%M%S")))

            fig, ax1 = plt.subplots(ncols=1, figsize=(30, 15), sharex=True, sharey=True)

            sns.set_style('darkgrid')
            sns.kdeplot(x=cur_df['Game_Duration'], y=cur_df['Event_Magnitude'], fill=True, ax=ax1)
            ax1.set_title(title)

            plt.tight_layout()
            plt.savefig(filename)

class Move:
    def __init__(self, threadnum, trial, track, moveNum, round, playerNum, oldScore, baseScore, reason, event, newScore,
                        soexcite):
        eventAmt = newScore - oldScore - baseScore
        self.threadnum = threadnum
        self.trial = trial
        self.trialmux = 10000*threadnum + trial
        self.movenum = moveNum
        self.player = playerNum
        self.track = track.num
        self.soexcite = soexcite
        self.round = round
        self.hasEvent = event != en.Event.NONE

        self.ladderamt = 0
        self.chuteamt = 0
        self.ladderorchuteamt = 0
        self.ladderhit = None
        self.chutehit = None
        self.eventhit = None

        if event != en.Event.NONE:
            if event == en.Event.CHUTE:
                self.chuteamt = eventAmt
                self.chutehit = (oldScore + baseScore, newScore)
                self.eventhit = self.chutehit
            elif event == en.Event.LADDER:
                self.ladderamt = eventAmt
                self.ladderhit = (oldScore + baseScore, newScore)
                self.eventhit = self.ladderhit

            self.ladderorchuteamt = eventAmt

        self.score = newScore - oldScore
        if self.score == 0:
            wtf = 'wtf'

        self.basescore = baseScore
        self.reason = reason
        self.currpos = newScore
        self.posin1 = self.currpos + 1
        self.posin2 = self.currpos + 2

        self.winningMove = False

    def to_dict(self):
        return {
     'movenum'     : self.movenum
     ,'threadnum'        : self.threadnum
     ,'trial'        : self.trial
     ,'trialmux'        : self.trialmux
     ,'player'      : self.player
     ,'round'      : self.round
     ,'track' : self.track
     ,'score'  : self.score
     ,'basescore' : self.basescore
     ,'ladderamt'    : self.ladderamt
     ,'chuteamt'     : self.chuteamt
     ,'ladderorchuteamt' : self.ladderorchuteamt
     ,'ladderhit' : self.ladderhit
     ,'chutehit' : self.chutehit
     ,'eventhit' : self.eventhit
     ,'currpos'      : self.currpos
     ,'posin1'       : self.posin1
     ,'posin2'       : self.posin2
     ,'soexcite'       : self.soexcite
     ,'reason'       : self.reason
     ,'winningMove'       : self.winningMove
        }