
from cribsandladders.Player import Player
from cribsandladders.CribbageGame import CribbageGame
from cribsandladders.Board import Board
from cribsandladders.CribSquad import CribSquad
import os
import game_params as gp
import cribsandladders.Stats as crst
import cribsandladders.PossibleEvents as ps
import cribsandladders.EventSetBuilder as op
import cribsandladders.Evaluator as evl
import cribsandladders.Optimizer as opt
import sqlite3 as sql
import cProfile as cprf
import multiprocessing as mp
import pandas as pd
import cribsandladders.BoardSetter as bstr
from itertools import repeat
import math
import time
import mystic
from mystic.solvers import fmin
from mystic.monitors import VerboseMonitor

#NOTE: this is outside tha class called staticly to avoic sql pickling issues (Optimizer & Eval have stored conns)
def trialsPerThread(board, squad, threadNum, trialsOnThread):
    movesForThread = []
    for i in range(trialsOnThread):
        # print("Running game " + str(i + 1) + " of " + str(gp.numtrials) + "\r")
        squad.resetRisks()
        squad.resetScores()
        squad.resetCanPlay()
        # NOTE: we retain the same track per agent, even tho they are swapping leads
        game = CribbageGame(board, squad, i, threadNum=threadNum)
        movesForThread.extend(game.play_game())

    return movesForThread
class Routines:
    def __init__(self, optimizerRunSet):

        #Set up lookups table in DF, index
        sqliteConn = sql.connect('etc/Lookups.db')
        query = ("SELECT * FROM HandDiscards{}Player{}Rank"
                     .format("Two" if gp.numplayers == 2 else "Three", "TwoDeck" if gp.twodecks else "OneDeck"))
        self.rankLookupTable = pd.read_sql_query(query, sqliteConn)
        self.rankLookupTable.set_index(['HandHash', 'DiscardHash', 'HasCrib'], inplace=True)
        self.rankLookupTable.sort_index(inplace=True)
        sqliteConn.close()

        self.optimizer = None
        self.board = None
        self.squad = None
        self.posEvents = None
        self.eventSetBuilder = None
        self.optimizerRunSet, self.optimizerRun = optimizerRunSet, 0

        #TODO: refactor & docu https://www.sitepoint.com/refactor-code-gpt/#h-refactoring-code-with-gpt-4-and-chatgpt


    #TODO: run purely probabilistic mini model, run for iterative bit
    #Maybe bounds likely peg round, likely score round, vary it bit, still do multiproc, maybe fewer trials tho
    #Since can control variability
    def run_trials(self, board, squad, stats, debug=False):
        squad.resetWins()
        pool = mp.Pool(processes=gp.nummaxthreads)
        trialsOnThread = math.floor(gp.numtrials/gp.nummaxthreads)
        #Unsure why __debug__ is flagged when run, no -o flag in command...0_o
        if debug:
            movesOnSet=trialsPerThread(board, squad, 0, 5)
        else:
            boardsToIter = [board]*gp.nummaxthreads
            squadsToIter = [squad]*gp.nummaxthreads
            movesOnSetSplit = pool.starmap(trialsPerThread, zip(boardsToIter, squadsToIter, range(gp.nummaxthreads),
                                                            repeat(trialsOnThread)))
            movesOnSet = []
            for s in movesOnSetSplit: movesOnSet.extend(s)
        stats.clearStatsAndSetMoves(movesOnSet)
        stats.calc_metrics()
        # stats.print_temp_maps()
        stats.insertStatsRecord()

    def genTrainSet(self):
        self.setUpBoard()
        eventSetBuilder = op.EventSetBuilder(self.board, self.posevents)
        self.board.setEffLandingForHolesAllTracks()
        sqlOptimizerCon = sql.connect("etc/Optimizer")
        for i in range(2553, 3001):
            stats = crst.Stats(self.board, self.squad, self.optimizerRunSet, self.optimizerRun)
            eventSetBuilder.runMonteCarlo(i)
            self.board.setEffLandingForHolesAllTracks()
            start = time.time()
            self.run_trials(self.board, self.squad, stats)
            end = time.time()
            print(end - start)
            eval = evl.Evaluator(eventSetBuilder, self.board, self.posevents, stats, sqlOptimizerCon, self.optimizerRunSet, i)
            eval.detMetrics()
            eval.writeMetricsToDb()
            eventSetBuilder.paramSet.tempWriteMetricsToDb(eval)

    def attemptOptimalLayout(self):
        self.setUpBoard()
        self.optimizer = opt.Optimizer(self.board, 1)
        self.optimizer.retrieveTrainData()
        tentOpt, model, sum_squared_diff = self.optimizer.runGBM()
        eventSetBuilder = op.EventSetBuilder(self.board, self.posevents)
        eventSetBuilder.buildBoardFromParams(tentOpt)
        self.board.setEffLandingForHolesAllTracks()
        eventSetBuilder.plotBoard()
        eventSetBuilder.setParamsIntoDb(2, 100001)
        stats = crst.Stats(self.board, self.squad, self.optimizerRunSet, self.optimizerRun)
        self.run_trials(self.board, self.squad, stats)
        sqlOptimizerCon = sql.connect("etc/Optimizer")
        eval = evl.Evaluator(eventSetBuilder, self.board, self.posevents, stats, sqlOptimizerCon, self.optimizerRunSet, 100001)
        eval.detMetrics()
        eval.writeMetricsToDb()
        eventSetBuilder.paramSet.tempWriteMetricsToDb(eval)


    def checkBoardBestTrial(self, optimizerRunSet, optimizerRun):
        self.setUpBoard()
        eventSetBuilder = op.EventSetBuilder(self.board, self.posevents)
        eventSetBuilder.buildBoardFromParamsDb(optimizerRunSet, optimizerRun)
        self.board.setEffLandingForHolesAllTracks()
        eventSetBuilder.plotBoard()

    def microRegressionsUsingInputOutputPairings(self):
        board = Board()
        bstr.setBoardFromDb(board, gp.boardname)
        board.setBoardAfterSetter()
        posevents = ps.PossibleEvents(board)
        squad = CribSquad(self.rankLookupTable, board.tracks, tracksUsed=gp.tracksused)
        self.optimizer = opt.Optimizer(board, 1)
        self.optimizer.retrieveTrainData()
        self.optimizer.testGBMOnPairings(["gamelength", "balance"], "ladderscanstartat")

    def fminCostFunction(self, paramsSubset):
        self.optimizerRun += 1
        self.eventSetBuilder.modParamsForFmin(paramsSubset, self.optimizer.bestPostIterParams,
                                              self.optimizerRunSet, self.optimizerRun)
        self.board.setEffLandingForHolesAllTracks()
        stats = crst.Stats(self.board, self.squad, self.optimizerRunSet, self.optimizerRun)
        self.run_trials(self.board, self.squad, stats)
        sqlOptimizerCon = sql.connect("etc/Optimizer")
        eval = evl.Evaluator(self.eventSetBuilder, self.board, self.posevents, stats, sqlOptimizerCon,
                             self.optimizerRunSet,self.optimizerRun)
        sqlOptimizerCon.close()
        eval.detMetrics()
        eval.writeMetricsToDb()
        self.eventSetBuilder.paramSet.tempWriteMetricsToDb(eval)
        return self.optimizer.detWeighedScoring(eval.results)

    def setUpBoard(self):
        self.board = Board()
        bstr.setBoardFromDb(self.board, gp.boardname)
        self.board.setBoardAfterSetter()
        self.posevents = ps.PossibleEvents(self.board)
        self.eventSetBuilder = op.EventSetBuilder(self.board, self.posevents)
        self.optimizer = opt.Optimizer(self.board, self.optimizerRunSet)
        self.squad = CribSquad(self.rankLookupTable, self.board.tracks, tracksUsed=gp.tracksused)

    def runNormalCribGame(self, debug=False):
        self.board = Board()
        bstr.setBoardFromDb(self.board, gp.boardname)
        self.board.setEffLandingForHolesAllTracks()
        self.squad = CribSquad(self.rankLookupTable, self.board.tracks)
        stats = crst.Stats(self.board, self.squad, self.optimizerRunSet, self.optimizerRun)
        self.run_trials(self.board, self.squad, stats, debug=debug)



    def runFmin(self):
        # Initial guesses & bounds for the parameters
        self.optimizer.setupFminParamsList(self.eventSetBuilder.paramSet.params)
        initial_guess = self.optimizer.getFminStarterParams()
        bounds = self.optimizer.getFminBounds(self.eventSetBuilder.paramSet.params)
        # Set up a monitor to observe the optimization progress
        monitor = VerboseMonitor(10)

        # Run the optimization
        result = fmin(self.fminCostFunction, initial_guess, bounds=bounds, itermon=monitor)

        # Extract optimized parameters
        optimized_params = result
        self.optimizer.setBestFminParams(optimized_params)
        return self.optimizer.bestPostFminParams

    def runIter(self, debug=False):
        sqlOptimizerCon = sql.connect("etc/Optimizer")
        weighedScoring = 9999999
        self.eventSetBuilder.runMidpointInitParams(self.optimizerRunSet, self.optimizerRun)
        while weighedScoring > gp.iterscorecutoff:
            self.board.setEffLandingForHolesAllTracks()
            stats = crst.Stats(self.board, self.squad, self.optimizerRunSet, self.optimizerRun)
            self.run_trials(self.board, self.squad, stats, debug)
            eval = evl.Evaluator(self.eventSetBuilder, self.board, self.posevents, stats, sqlOptimizerCon,
                                 self.optimizerRunSet, self.optimizerRun)




            eval.detMetrics()
            eval.writeMetricsToDb()
            self.eventSetBuilder.paramSet.tempWriteMetricsToDb(eval)
            # self.eventSetBuilder.paramSet.tempWriteEvents(stats, self.optimizerRunSet, self.optimizerRun)
            weighedScoring = self.optimizer.detWeighedScoring(eval.results)
            print(weighedScoring)
            if weighedScoring <= gp.iterscorecutoff:
                self.optimizer.setBestIterParams(self.eventSetBuilder.paramSet.params)
                break

            self.optimizerRun += 1
            freshParams = self.optimizer.runIncrIteration(self.eventSetBuilder.paramSet.params, eval.results)
            self.eventSetBuilder.buildBoardFromParams(pd.DataFrame.from_records(freshParams),
                                                      self.optimizerRunSet, self.optimizerRun)

        sqlOptimizerCon.close()
        return self.optimizer.bestPostIterParams


if __name__ == "__main__":
    #TODO: mult-processing
    #TODO: time events, look for heavy hitters, maybe GPT it into c+++ bind?
    # for batch in range(gp.batchnum):
    #     # curBoardName = input("Please enter board name: ")
    #     curBoardName = "LM Init Trial #2"
    #     boards.append(Board(curBoardName))
    # squad = CribSquad(gp.tracksused)
    # for board in boards:
    #     run_trials(board, squad, False)
    ### human play
    # game = CribbageGame(AdvancedAgent(-1), HumanAgent())
    # game.play_game()
    # checkBoardBestTrial(1, 807)
    # genTrainSet()

    routines = Routines(optimizerRunSet=1)
    # routines.runNormalCribGame(debug=False)



    routines.setUpBoard()
    bestIterParams = routines.runIter(debug=True)
    print(bestIterParams)
    bestFminParams = routines.runFmin()
    print(bestFminParams)