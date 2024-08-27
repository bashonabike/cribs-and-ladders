# Evaluate:
# balance
# distribution of events
# regression fitting of events over time to ideal curve
# fit of the other curves
# game length (maybe allow chute cancelling)
# 2-hits (maybe fit curve?
# So excite
# repeats
# minimize othos
# maximize multis
import statistics as stt
import game_params as gp
import math
import pandas as pd
from io import StringIO
import sqlite3 as sql


class Evaluator:
    def __init__(self, eventSetBuilder, board, possibleEvents, stats ,sqlOptimizerCon, optimizerRunSet, optimizerRun):
        self.eventSetBuilder = eventSetBuilder
        self.board = board
        self.possibleEvents = possibleEvents
        self.stats = stats
        self.moves = stats.moves
        self.sqlOptimizerCon = sqlOptimizerCon
        self.optimizerRunSet = optimizerRunSet
        self.optimizerRun = optimizerRun
        self.results = []

    def lossFunction(self):
        temp1 = 0

    def writeMetricsToDb(self):
        #Input metric info
        # metrics_df = pd.DataFrame.from_records(self.results)
        # metrics_df['OptimizerRun'] = self.optimizerRun
        # metrics_df['Board_ID'] = self.board.boardID
        #
        # # Insert into data table
        # metrics_l = metrics_df.values.tolist()
        # metricsQuery_sb = StringIO()
        # metricsQuery_sb.write("INSERT INTO Testtest (")
        # cols = metrics_df.columns.values.tolist()
        # for c in range(0, len(cols) - 1):
        #     metricsQuery_sb.write(cols[c])
        #     metricsQuery_sb.write(", ")
        # metricsQuery_sb.write(cols[len(cols) - 1])
        # metricsQuery_sb.write(") VALUES (")
        # metricsQuery_sb.write(", ".join(['?'] * len(cols)))
        # metricsQuery_sb.write(")")
        # test = metricsQuery_sb.getvalue()
        # sqlcon2 = sql.connect("C:\\Users\\Dell 5290\\Documents\\cribs-and-ladders\\etc\\Optimizer.db")
        # sqliteCursor = sqlcon2.cursor()
        # sqliteCursor.execute("BEGIN TRANSACTION")
        # sqliteCursor.execute("select * from  Testtest")
        # for index, record in metrics_df.iterrows():
        #     sqliteCursor.execute(metricsQuery_sb.getvalue(), [record['Result'], record['ResultFlavour'], record['ResultValue'],
        #                                  record['OptimizerRun'], record['Board_ID']])
        #
        # sqliteCursor.execute("END TRANSACTION")
        # # sqliteCursor.executemany(metricsQuery_sb.getvalue(), metrics_df)
        # sqlcon2.commit()
        # metricsQuery_sb.close()
        for record in self.results:
            print("RUN {}: Result={}, ResultFlavour={}, ResultValue={}".format(
                self.optimizerRun, record['Result'], record['ResultFlavour'], record['ResultValue']))



    def detMetrics(self):

        #GAME BOARD STRUCTURE SCALAR STATS
        if self.eventSetBuilder.events > 0:
            orthos = abs(gp.optorthospct - self.eventSetBuilder.orthos/self.eventSetBuilder.events)
            orthosit = gp.optorthospct - self.eventSetBuilder.orthos/self.eventSetBuilder.events
        else:
            orthos, orthosit = gp.optorthospct, gp.optorthospct
        self.results.append(dict(Result="orthos", ResultFlavour="GAME BOARD STRUCTURE SCALAR STATS",
                                 ResultValue=orthos, IterativeResultValue=orthosit, Weighting=0.5))
        if self.eventSetBuilder.events > 0:
            multis = abs(gp.optmultispct - self.eventSetBuilder.multis/self.eventSetBuilder.events)
            multisit = gp.optmultispct - self.eventSetBuilder.multis/self.eventSetBuilder.events
        else:
            multis, multisit = gp.optmultispct, gp.optmultispct
        self.results.append(dict(Result="multis", ResultFlavour="GAME BOARD STRUCTURE SCALAR STATS",
                                 ResultValue=multis, IterativeResultValue=multisit, Weighting=1))

        if self.eventSetBuilder.events > 0:
            cancels = self.eventSetBuilder.cancels/self.eventSetBuilder.events
        else:
            cancels = 0
        self.results.append(dict(Result="cancels", ResultFlavour="GAME BOARD STRUCTURE SCALAR STATS",
                                 ResultValue=cancels, Weighting=4))


        #GAME BOARD STRUCTURE STATISTIC STATS

        #EVENT SPACING HISTOGRAM
        spacingsRaw_l = []
        for nds in self.eventSetBuilder.eventNodesByTrack:
            prvNode = 0
            for nd in nds['nodes']:
                spacingsRaw_l.append(nd - prvNode)
                prvNode = nd
        if len(spacingsRaw_l) > 0:
            spacingsHist_l = self.processActualHistCurve(spacingsRaw_l)
            result = self.discreteRegression(gp.eventspacingsdisthistcurvefile,
                                                                            spacingsHist_l)
        else:
            result = 1
        self.results.append(dict(Result="eventSpacingHist_curvefit", ResultFlavour="GAME BOARD STRUCTURE STATISTIC STATS",
                                 ResultValue=result, Weighting=0.1))

        #GAMEPLAY SCALAR STATS

        #Balance
        self.results.append(dict(Result="balance", ResultFlavour="GAMEPLAY SCALAR STATS",
                                 ResultValue=stt.stdev(self.stats.partialBalanceSet), Weighting=30))
        for b in self.stats.partialBalanceSet:
            track_id = self.board.getTrackByNum(b[0]).Track_ID
            self.results.append(dict(Result="balance_T{}".format(track_id), ResultFlavour="GAMEPLAY SCALAR STATS",
                                 ResultValue=b[1], Weighting=30))


        #Game length
        if gp.idealgamelength > 0:
            gamelengthstatit = (self.stats.avglengthinrounds - gp.idealgamelength)/gp.idealgamelength
        else: gamelengthstatit = 1

        self.results.append(dict(Result="gamelength", ResultFlavour="GAMEPLAY SCALAR STATS",
                                 ResultValue=abs(gamelengthstatit), ResultValueIterative=gamelengthstatit, Weighting=4))

        #Calculate two-hits
        trackNumChecks = []
        for t in self.board.tracks:
            trackNumChecks.append(dict(tracknum=t.num, prevwasevent=False))

        twoHits = []
        for m in self.moves:
            curTrack = None
            for tn in trackNumChecks:
                if tn['tracknum'] == m.track:
                    curTrack = tn
                    break
            if m.hasEvent:
                if curTrack['prevwasevent']: twoHits.append(dict(tracknum=curTrack['tracknum'], movenum=m.movenum))
                curTrack['prevwasevent'] = True
            else:
                curTrack['prevwasevent'] = False

        if len(self.moves) > 0:
            twohitsstatit = len(twoHits)/len(self.moves) - gp.opttwohitspct
        else: twohitsstatit = 1
        self.results.append(dict(Result="twohits", ResultFlavour="GAMEPLAY SCALAR STATS",
                                 ResultValue=abs(twohitsstatit), ResultValueIterative=twohitsstatit, Weighting=0.5))

        #Calculate so-excites (maximize)
        self.results.append(dict(Result="soexcite", ResultFlavour="GAMEPLAY SCALAR STATS",
                                 ResultValue=1.0/self.stats.soexcitespegging if self.stats.soexcitespegging > 0 else 1.0,
                                 Weighting=1))

        #Calculate repeats (minimize)
        self.results.append(dict(Result="repeats", ResultFlavour="GAMEPLAY SCALAR STATS",
                                 ResultValue=self.stats.repeats/len(self.moves) if len(self.moves) > 0 else 1,
                                 Weighting=20))

        #GAMEPLAY STATISTICAL STATS (lol)
        moves_df = pd.DataFrame.from_records([dict(movenum=m.movenum, trial=m.trial) for m in self.moves])
        movesPerTrial_df = moves_df[['trial']].assign(moves=1).groupby('trial').agg('sum').reset_index()
        movesPerTrial_df.sort_values(['trial'])

        #Events over time
        eventsOverTime_df = pd.DataFrame.from_records([dict(movenum=m.movenum, hasevent=1 if m.hasEvent else 0,
                                                            trial=m.trial) for m in self.moves if m.hasEvent])
        if len(eventsOverTime_df) == 0:
            result = 1
        else:
            eventsOverTime_l = self.processActualTimeCurve(movesPerTrial_df, eventsOverTime_df, "hasevent")
            result = self.discreteRegression(gp.eventsovertimecurvefile, eventsOverTime_l)
        self.results.append(dict(Result="eventsOverTime_curvefit", ResultFlavour="GAMEPLAY STATISTICAL STATS (lol)",
                                 ResultValue=result, Weighting=0.1))

        #Energy over time
        energyOverTime_df = pd.DataFrame.from_records([dict(movenum=m.movenum, eventmag=abs(m.ladderorchuteamt),
                                                            trial=m.trial) for m in self.moves])
        if len(energyOverTime_df) == 0:
            result = 1
        else:
            eventsOverTime_l = self.processActualTimeCurve(movesPerTrial_df, energyOverTime_df, "eventmag")
            result =self.discreteRegression(gp.eventenergyfile, eventsOverTime_l)
        self.results.append(dict(Result="energy_curvefit", ResultFlavour="GAMEPLAY STATISTICAL STATS (lol)",
                                 ResultValue=result, Weighting=0.1))

        #Velocity over time
        velocityOverTime_df = pd.DataFrame.from_records([dict(movenum=m.movenum, score=m.score,
                                                            trial=m.trial) for m in self.moves])
        if len(velocityOverTime_df) == 0:
            result = 1
        else:
            velocityOverTime_l = self.processActualTimeCurve(movesPerTrial_df, velocityOverTime_df, "score")
            result =self.discreteRegression(gp.velocityovertimecurvefile, velocityOverTime_l)
        self.results.append(dict(Result="velocity_curvefit", ResultFlavour="GAMEPLAY STATISTICAL STATS (lol)",
                                 ResultValue=result, Weighting=0.1))

        #Event length distribution as histogram
        eventsLengthHist_l = self.processActualHistCurve([abs(m.ladderorchuteamt) for m in self.moves if m.hasEvent])
        if len(eventsLengthHist_l) == 0:
            result = 1
        else:
            result = self.discreteRegression(gp.eventlengthdisthistcurvefile, eventsLengthHist_l)
        self.results.append(dict(Result="eventsHitLengthDistribution_curvefit",
                                 ResultFlavour="GAMEPLAY STATISTICAL STATS (lol)",
                                 ResultValue=result, Weighting=0.1))

    def processActualTimeCurve(self,movesPerTrial_df, curve_df, y_field):
        merged_df = pd.merge(movesPerTrial_df, curve_df,
                                     on=['trial'], suffixes=('_sum', ''))
        merged_df['movePct'] = merged_df['movenum'] / merged_df['moves']
        merged_df.sort_values(['movePct'])

        merged_l = self.eventSetBuilder.discretizeCurve(list(zip(merged_df['movePct'].to_list(), merged_df[y_field].to_list())),
                                                                gp.effectiveboardlength, accumulate=True)
        #Need to re-normalize since accumulated with smoothing applied
        final_l = self.eventSetBuilder.normalizeCurveMagnitude(merged_l)
        return final_l

    def processActualHistCurve(self, curve_l):
        curve_l.sort()
        curve_idx = 0
        merged_l = []
        for b in range(1, max(curve_l) + 1):
            accum = 0
            while curve_idx < len(curve_l) and curve_l[curve_idx] == b:
                accum += 1
                curve_idx += 1
            merged_l.append([b, accum])
        return merged_l


    def discreteRegression(self, idealCurveFileName, actualCurve):
        #Retrieve & normalize ideal curve from file
        normIdealCurve = self.eventSetBuilder.getNormalizedIdealCurve(idealCurveFileName)
        max_x = max([a[0] for a in actualCurve])
        max_y = max([a[1] for a in actualCurve])
        actualizedIdealCurve = self.eventSetBuilder.actualizeCurve(normIdealCurve, max_x, max_y)

        #Iterate over actuals comparing to ideal (least squares)
        regressionCurve = []
        regressionSum = 0
        ideal_idx = 0
        for a in actualCurve:
            while ideal_idx < len(actualizedIdealCurve) - 1 and actualizedIdealCurve[ideal_idx][0] < a[0]: ideal_idx += 1
            diffSquard = math.pow(actualizedIdealCurve[ideal_idx][1] - a[1], 2)
            regressionCurve.append((a[0], diffSquard))
            regressionSum += diffSquard

        return regressionSum/(len(regressionCurve)*math.pow(max(abs(max(actualizedIdealCurve[1])),
                                                                abs(min(actualizedIdealCurve[1]))), 2))
