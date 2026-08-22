import statistics as st
import collections as col
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import pandas as pd
import Enums as en
import sqlite3 as sql
import datetime as dt
from io import StringIO
import os
from cribsandladders.config import GameConfig, DEFAULT_CONFIG
from cribsandladders import stats_metrics as sm


def build_insert_stat_stub(cursor):
    """
    Builds the "INSERT INTO Stat (col1,col2,...) VALUES (" prefix by
    introspecting the live Stat table schema (skipping the
    auto-incrementing Stat_ID column).

    This used to be `gp.insertstatstub`, a lazily-computed module-level
    global in game_params.py, built from a second live sqlite
    connection opened behind the scenes the first time any of
    gp.sqliteConn/sqliteCursor/insertstatstub was read (see that
    module's __getattr__). Pulling it into Stats.py as a plain function
    over an explicit cursor means insertStatsRecord no longer depends
    on game_params at all, and this piece is unit-testable against a
    temp sqlite db with a matching Stat table instead of the real one.

    Args:
        cursor (sqlite3.Cursor): cursor for a connection with a Stat
            table already defined.

    Returns:
        str: e.g. "INSERT INTO Stat (Board_ID,Timestamp,...) Values ("
    """
    cursor.execute("SELECT name FROM pragma_table_info('Stat') as tblInfo")
    result = cursor.fetchall()
    result.remove(('Stat_ID',))
    sb = StringIO()
    sb.write("INSERT INTO Stat (")
    sb.write("".join([c[0] + "," for c in result])[:-1])
    sb.write(") Values (")
    stub = sb.getvalue()
    sb.close()
    return stub


class Stats:

    def __init__(self, board, squad, optimizerRunSet, optimizerRun, config: GameConfig = DEFAULT_CONFIG):
        """
        Initialize the statistics object.

        Args:
            board: The Board object to evaluate.
            squad: The CribSquad object containing the players to evaluate.
            optimizerRunSet: The identifier for the current optimizer run set.
            optimizerRun: The identifier for the current optimizer run.
            config (GameConfig): game configuration (defaults to the
                module-level DEFAULT_CONFIG).

        Attributes:
            moves (list): A list of moves made in the games.
            partialBalanceSet (list): A list of partial balance values by player.
            chutesbytrack (list): A list of chutes by track.
            laddersbytrack (list): A list of ladders by track.
            eventsbytrack (list): A list of events by track.
            ladders (int): The total number of ladders.
            chutes (int): The total number of chutes.
            events (int): The total number of events.
            laddersin1bytrack (list): A list of ladders in track 1 by player.
            laddersin2bytrack (list): A list of ladders in track 2 by player.
            chutesin1bytrack (list): A list of chutes in track 1 by player.
            chutesin2bytrack (list): A list of chutes in track 2 by player.
            eventin1bytrack (list): A list of events in track 1 by player.
            eventin2bytrack (list): A list of events in track 2 by player.
            laddersin1 (int): The total number of ladders in track 1.
            laddersin2 (int): The total number of ladders in track 2.
            chutesin1 (int): The total number of chutes in track 1.
            chutesin2 (int): The total number of chutes in track 2.
            eventin1 (int): The total number of events in track 1.
            eventin2 (int): The total number of events in track 2.
            soexcitespeggingbytrack (list): A list of soexcites pegging by track.
            repeats (int): The total number of repeats.
            avglengthinrounds (int): The average length of games in rounds.
            hist_df (DataFrame): A DataFrame containing the histogram of lengths.
            hist_by_track_df (DataFrame): A DataFrame containing the histogram of lengths by track.
        """
        self.board = board
        self.squad = squad
        self.optimizerRunSet = optimizerRunSet
        self.optimizerRun = optimizerRun
        self.config = config

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
        self.chutesin1 = 0
        self.chutesin2 = 0
        self.eventsin1 = 0
        self.eventsin2 = 0

        self.soexcitespeggingbytrack = []
        self.repeatsbytrack = []
        self.soexcitespegging = 0
        self.repeats = 0

        self.avglengthinrounds = 0

        self.hist_df = None
        self.hist_by_track_df = None

    def clearStatsAndSetMoves(self, curMoveSet):
        """
        Resets all statistics and sets the current set of moves.

        Args:
            curMoveSet (list): A list of moves made in the games.

        Returns:
            None
        """
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
        self.chutesin1 = 0
        self.chutesin2 = 0
        self.eventsin1 = 0
        self.eventsin2 = 0

        self.soexcitespeggingbytrack = []
        self.repeatsbytrack = []
        self.soexcitespegging = 0
        self.repeats = 0

        self.avglengthinrounds = 0

        self.hist_df = None
        self.hist_by_track_df = None

        self.moves = curMoveSet

    def calc_metrics(self):
        """
        Calculate statistics on the given moves.

        Thin orchestrator (Refactor Mk II, Phase 6) -- see
        cribsandladders/stats_metrics.py for the actual per-statistic
        logic. This used to be one 137-line method building a chain of
        merged DataFrames and computing ~20 named statistics from them
        all inline; each is now a separate, independently unit-tested
        function over already-built DataFrames instead of `self.board`/
        `self.moves`. Calls run in the same order the original inline
        code did and no behavior changed -- see stats_metrics.py's
        module docstring for details.

        Parameters:
            moves (list): A list of Move objects to calculate statistics on.
        Returns:
            None
        """
        (joined_df, moves_df, ladders_df, chutes_df,
         games_df, games_by_track_df) = sm.prep_joined_moves_df(self.moves, self.board.tracks)

        (self.soexcitespeggingbytrack, self.soexcitespegging,
         self.repeatsbytrack, self.repeats) = sm.calc_soexcites_and_repeats(joined_df, self.config.numtrials)

        self.avglengthinrounds = sm.calc_avg_length_in_rounds(joined_df, self.config.numtrials)

        (self.chutesbytrack, self.laddersbytrack, self.eventsbytrack,
         self.chutes, self.ladders, self.events) = sm.calc_events_by_track(joined_df, self.config.numtrials)

        self.hist_df, self.hist_by_track_df = sm.calc_move_histograms(joined_df, games_df, games_by_track_df)

        (self.laddersin1bytrack, self.laddersin2bytrack, self.chutesin1bytrack, self.chutesin2bytrack,
         self.eventsin1bytrack, self.eventsin2bytrack, self.laddersin1, self.laddersin2,
         self.chutesin1, self.chutesin2, self.eventsin1, self.eventsin2) = (
            sm.calc_lookforward_events_by_track(moves_df, ladders_df, chutes_df, self.config.numtrials))

    def buildSet4PlusInsertSnippet(self, partialSet, overall=None, end=False):
        """
        Build a SQL snippet for inserting a set of values into a table.

        Args:
            partialSet (list): A list of values to be inserted into the table.
            overall (int): An optional overall value to be inserted into the table.
            end (bool): If True, the trailing comma will be removed from the snippet.

        Returns:
            str: The SQL snippet for inserting the values into the table.
        """
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
        """
        Insert a stats record into the database.

        This method inserts a stats record into the database given the
        game parameters, board, and moves. It also calculates
        various board level setup stats and balance stats.

        Args:
            None

        Returns:
            None
        """
        sqliteConn = sql.connect(self.config.db_path)
        sqliteCursor = sqliteConn.cursor()

        # Prepend columns to write to, all except auto-incrementing Stat_ID
        insertstatquery_sb = StringIO()
        insertstatquery_sb.write(build_insert_stat_stub(sqliteCursor))

        # board level setup stats
        tracksList_sb = StringIO()
        if self.config.tracksused is None:
            for p in self.squad.players:
                tracksList_sb.write(str(p.tracknum))
                tracksList_sb.write(",")
        elif self.config.tracksused is list:
            for t in self.config.tracksused:
                tracksList_sb.write(t)
                tracksList_sb.write(",")
        else:
            raise Exception("Invalid tracksused setting in game_params: {}".format(self.config.tracksused))
        pos = tracksList_sb.tell()
        tracksList_sb.seek(0, os.SEEK_END)
        if tracksList_sb.tell() == 0:
            raise Exception("Tracks list was blank")
        tracksList_sb.seek(pos)
        tracksList = tracksList_sb.getvalue()[:-1]
        tracksList_sb.close()
        # NOTE: this is less efficient than including values directly using '?'s but this insert only fires infrequently
        insertstatquery_sb.write("{},\'{}\',{},{},{},\'{}\',\'{}\',{},".format(self.board.boardID, dt.datetime.now(),
                                                                               self.config.numtrials, self.config.numplayers,
                                                                               self.config.numdecks,
                                                                               tracksList,
                                                                               self.board.boardName,
                                                                               self.avglengthinrounds))

        # balance stats
        # NOTE: we cannot use player.wins with the multprocessing!
        winsByPlayer = col.Counter([m.player for m in self.moves if m.winningMove])
        self.partialBalanceSet = [(float(winsByPlayer[p.num]) - (float(self.config.numtrials) / float(self.config.numplayers))) /
                                  float(self.config.numtrials) for p in self.squad.players]
        insertstatquery_sb.write(self.buildSet4PlusInsertSnippet(self.partialBalanceSet))
        self.partialBalanceSet = [s for s in zip([p.tracknum for p in self.squad.players], self.partialBalanceSet)]

        # track stats
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

        # finalize query and commit data
        insertstatquery_sb.write(")")
        query = insertstatquery_sb.getvalue()
        insertstatquery_sb.close()
        sqliteCursor.execute(query)
        sqliteConn.commit()
        # TODO: insert file links to heatmaps once generated

    def print_metrics(self, output_dir="./Board_Results"):
        # TODO: also commit this to data table in sql
        """
        Print game metrics to a text file.

        This method prints out various metrics from the game, including
        the balance, so excites, boring repeats, and snakes/ladders, as
        well as the likelihood of these events per round. The metrics are
        written to a text file, with the filename being a combination of the
        board name, number of players, number of decks, number of trials, and
        the current date and time.

        Args:
            output_dir (str): directory the results file is written
                into. Defaults to "./Board_Results" (previous hardcoded
                behavior). Parameterized so tests can point this at a
                temp directory.

        Returns:
            None

        TODO(liam): this method reads self.soexcites_pegs and
        self.lengths_in_rounds, neither of which this class ever sets
        (the actual attributes are self.soexcitespeggingbytrack /
        self.soexcitespegging and self.avglengthinrounds) -- calling
        print_metrics as written raises AttributeError immediately.
        Looks pre-existing and unrelated to the Phase 4 config-injection
        work; needs investigation into what this was meant to read
        before it's called anywhere for real.
        """
        os.makedirs(output_dir, exist_ok=True)
        with open((output_dir + "/" + self.board.boardName + "-" + str(self.config.numplayers) + "-" +
                   str(self.config.numdecks) + "-" +
                   str(self.config.numtrials) + "-" + datetime.datetime.now().strftime("%y%m%d%H%M%S") + ".txt"),
                  "w") as results:
            results.write("\t" + self.board.boardName + "\n")
            results.write("Lengths\t")

            strTemp = ""
            for player in self.squad.players:
                strTemp += "{}/".format(self.board.getTrackByNum(player.tracknum).efflength)
            results.write(strTemp[:-1] + "\n")

            results.write("# trials\t" + str(self.config.numtrials) + "\n")
            results.write("# decks\t" + str(self.config.numdecks) + "\n")
            results.write("# players\t" + str(self.config.numplayers) + "\n")

            results.write("balance (<5% reasonable for traditional board, for comparison) \t")
            strTemp = ""
            for player in self.squad.players:
                winDif = (float(player.wins) - (float(self.config.numtrials) / float(self.config.numplayers))) / float(self.config.numtrials)
                strTemp += "{:.2%} (Player {}), ".format(winDif, player.num)
            results.write(strTemp[:-2] + "\n")

            results.write("So excites (when some pegging options yield events and others don't)\t" + str(
                st.mean(self.soexcites_pegs)) + "\n")
            results.write("Boring repeats\t" + str(st.mean(self.repeats)) + "\n")
            results.write("Snakes/Ladders\t" + str(st.mean(self.events)) + "\n")
            results.write("Snakes/Ladders in 1 incr\t" + str(st.mean(self.eventsin1)) + "\n")
            results.write("Snakes/Ladders in 2 incrs\t" + str(st.mean(self.eventsin2)) + "\n")
            results.write("# rounds\t" + str(st.mean(self.lengths_in_rounds)) + "\n")
            results.write("Likelihood So excites (pegging) per round\t{:.2%}\n".
                          format(st.mean(self.soexcites_pegs) / st.mean(self.lengths_in_rounds)))
            results.write("Likelihood Boring repeats per round\t{:.2%}\n".
                          format(st.mean(self.repeats) / st.mean(self.lengths_in_rounds)))
            results.write("Likelihood Snakes/Ladders per round\t{:.2%}\n".
                          format(st.mean(self.events) / st.mean(self.lengths_in_rounds)))
            results.write("Likelihood Snakes/Ladders in 1 incr per round\t{:.2%}\n".
                          format(st.mean(self.eventsin1) / st.mean(self.lengths_in_rounds)))
            results.write("Likelihood Snakes/Ladders in 2 incrs per round\t{:.2%}\n".
                          format(st.mean(self.eventsin2) / st.mean(self.lengths_in_rounds)))

            # event_norm_mags_sorted = sorted(event_norm_mags, key = lambda e: (e[0], e[1]))
            # results.write("\n--------------------------------------------------------------\n")
            # results.write("Game_Duration\tEvent_Magnitude\n")
            # for bin, mag in event_norm_mags_sorted:
            #     results.write("{}\t{}\n".format(bin,mag))

    def print_temp_maps(self, output_dir="./Board_Results/images"):
        """
        Print a heat map of the game, with the x-axis being Game_Duration and
        the y-axis being Event_Magnitude. The heat map shows the likelihood of
        each combination of Game_Duration and Event_Magnitude.

        Each heat map is saved as a PNG file in the Board_Results/images
        directory, with the filename being a combination of the board name,
        number of players, number of decks, number of trials, and the current
        date and time.

        Args:
            output_dir (str): directory PNG heat maps are written into.
                Defaults to "./Board_Results/images" (previous hardcoded
                behavior). Parameterized so tests can point this at a
                temp directory.

        Returns:
            None

        TODO(liam): `self.hist_by_track_df.groupby('track').to_list('track')`
        below calls .to_list() on a DataFrameGroupBy, which doesn't have
        that method -- this raises AttributeError any time hist_by_track_df
        is non-empty. Looks pre-existing and unrelated to the Phase 4
        config-injection work; needs investigation into what grouping/
        listing was actually intended (probably
        `list(self.hist_by_track_df['track'].unique())`) before this is
        exercised for real.
        """
        if self.hist_df is None or self.hist_df.empty:
            return

        os.makedirs(output_dir, exist_ok=True)
        for tracknum in set(self.hist_by_track_df.groupby('track').to_list('track').append(0)):
            cur_df = (self.hist_by_track_df.loc[self.hist_by_track_df['track'] == tracknum]
                      [['normmove', 'ladderorchuteamt']] if tracknum > 0
                      else self.hist_df)
            cur_df.rename(columns={'normmove': 'Game_Duration', 'ladderorchuteamt': 'Event_Magnitude'}, inplace=True)
            title = 'Played Heat: Track {}'.format(tracknum) if tracknum > 0 else 'Played Heat: Overall'
            filename = (output_dir + "/{}-{}-{}-{}-{}-{}.png".
                        format(self.board.boardName, str(self.config.numplayers), str(self.config.numdecks),
                               str(self.config.numtrials), "Overall" if tracknum > 0 else "Track {}".format(tracknum),
                               datetime.datetime.now().strftime("%y%m%d%H%M%S")))

            fig, ax1 = plt.subplots(ncols=1, figsize=(30, 15), sharex=True, sharey=True)

            sns.set_style('darkgrid')
            sns.kdeplot(x=cur_df['Game_Duration'], y=cur_df['Event_Magnitude'], fill=True, ax=ax1)
            ax1.set_title(title)

            plt.tight_layout()
            plt.savefig(filename)


class Move:
    def __init__(self, threadnum, trial, track, moveNum, round, playerNum, oldScore, baseScore, reason, event, newScore,
                 soexcite, pegMove):
        """
        Initializes a Move object with given parameters.

        Parameters:
            threadnum (int): Thread number
            trial (int): Trial number
            track (Track): Track object
            moveNum (int): Move number
            round (int): Round number
            playerNum (int): Player number
            oldScore (int): Old score
            baseScore (int): Base score
            reason (str): Reason for the move
            event (en.Event): Event type
            newScore (int): New score
            soexcite (bool): Whether the move was a "so excite" move
            pegMove (bool): Whether the move was a pegging move

        Returns:
            None
        """
        eventAmt = newScore - oldScore - baseScore
        self.threadnum = threadnum
        self.trial = trial
        self.trialmux = 10000 * threadnum + trial
        self.movenum = moveNum
        self.player = playerNum
        self.track = track.num
        self.track_id = track.Track_ID
        self.soexcite = soexcite
        self.round = round
        self.hasEvent = event != en.Event.NONE
        self.pegMove = pegMove

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
            'movenum': self.movenum
            , 'threadnum': self.threadnum
            , 'trial': self.trial
            , 'trialmux': self.trialmux
            , 'player': self.player
            , 'round': self.round
            , 'track': self.track
            , 'track_id': self.track_id
            , 'score': self.score
            , 'basescore': self.basescore
            , 'ladderamt': self.ladderamt
            , 'chuteamt': self.chuteamt
            , 'ladderorchuteamt': self.ladderorchuteamt
            , 'ladderhit': self.ladderhit
            , 'chutehit': self.chutehit
            , 'eventhit': self.eventhit
            , 'currpos': self.currpos
            , 'posin1': self.posin1
            , 'posin2': self.posin2
            , 'soexcite': self.soexcite
            , 'reason': self.reason
            , 'winningmove': self.winningMove
            , 'pegmove': self.pegMove
        }
