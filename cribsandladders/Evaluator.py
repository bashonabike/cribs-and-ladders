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
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.optimize import minimize



class Evaluator:
    def __init__(self, eventSetBuilder, board, possibleEvents, stats ,sqlOptimizerCon, optimizerRunSet, optimizerRun):
        self.eventSetBuilder = eventSetBuilder
        self.board = board
        self.possibleEvents = possibleEvents
        self.stats = stats
        if self.stats is not None: self.moves = stats.moves
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



    def detMetrics(self, onlyGameBoardStats = False):




        #GAME BOARD STRUCTURE SCALAR STATS
        if self.eventSetBuilder.events > 0:
            orthos = abs(gp.optorthospct - self.eventSetBuilder.orthos/self.eventSetBuilder.events)
            orthosit = gp.optorthospct - self.eventSetBuilder.orthos/self.eventSetBuilder.events
        else:
            orthos, orthosit = gp.optorthospct, gp.optorthospct
        self.results.append(dict(Result="orthos", ResultFlavour="GAME BOARD STRUCTURE SCALAR STATS",
                                 ResultValue=orthos, ResultValueIterative=orthosit, Weighting=0.5))
        if self.eventSetBuilder.events > 0:
            multis = abs(gp.optmultispct - self.eventSetBuilder.multis/self.eventSetBuilder.events)
            multisit = gp.optmultispct - self.eventSetBuilder.multis/self.eventSetBuilder.events
        else:
            multis, multisit = gp.optmultispct, gp.optmultispct
        self.results.append(dict(Result="multis", ResultFlavour="GAME BOARD STRUCTURE SCALAR STATS",
                                 ResultValue=multis, ResultValueIterative=multisit, Weighting=1))

        if self.eventSetBuilder.events > 0:
            cancels = (self.eventSetBuilder.cancels/self.eventSetBuilder.events) - gp.idealcancelspct
            #If too few cancels, don't sweat it, muchks with iter model
            if cancels < 0: cancels = 0
        else:
            cancels = 0
        self.results.append(dict(Result="cancels", ResultFlavour="GAME BOARD STRUCTURE SCALAR STATS",
                                 ResultValue=cancels, Weighting=5))

        for n in self.eventSetBuilder.eventNodesByTrack:
            track = self.board.getTrackByNum(n['tracknum'])
            track_id = track.Track_ID
            maxNode = max(n['nodes'])
            termPct = maxNode/track.length
            self.results.append(dict(Result="earlytermination_T{}".format(track_id), ResultFlavour="GAMEPLAY SCALAR STATS",
                                 ResultValue=1.0-termPct, Weighting=8))

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
                                 ResultValue=result, Weighting=15))

        if onlyGameBoardStats:
            #Event length track distribution as histogram
            #Only do this if in prelim track eval mode, since better appraised by gameplay stat below
            for t in self.board.tracks:
                eventsLengthHist_l = self.processActualHistCurve([e.length for e in t.eventSetBuild])
                if len(eventsLengthHist_l) == 0:
                    result = 1
                else:
                    result = self.discreteRegression(gp.eventlengthdisthistcurvefile, eventsLengthHist_l)
                self.results.append(dict(Result="trackEventLengthDistribution_curvefit_T{}".format(t.Track_ID),
                                         ResultFlavour="GAME BOARD STRUCTURE STATISTIC STATS",
                                         ResultValue=result, Weighting=8))


        if not onlyGameBoardStats:
            #GAMEPLAY SCALAR STATS

            #Balance OMITING FROM EVAL SINCE DEALT W/ IN SETTER
            self.results.append(dict(Result="balance", ResultFlavour="GAMEPLAY SCALAR STATS",
                                     ResultValue=stt.stdev([b[1] for b in self.stats.partialBalanceSet]), Weighting=0))
            for b in self.stats.partialBalanceSet:
                track_id = self.board.getTrackByNum(b[0]).Track_ID
                self.results.append(dict(Result="balance_T{}".format(track_id), ResultFlavour="GAMEPLAY SCALAR STATS",
                                     ResultValue=b[1], Weighting=0))


            #Game length OMITING FROM EVAL SINCE DEALT W/ IN SETTER
            if gp.idealgamelength > 0:
                gamelengthstatit = (self.stats.avglengthinrounds - gp.idealgamelength)/gp.idealgamelength
            else: gamelengthstatit = 1

            self.results.append(dict(Result="gamelength", ResultFlavour="GAMEPLAY SCALAR STATS",
                                     ResultValue=abs(gamelengthstatit), ResultValueIterative=gamelengthstatit, Weighting=0))

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
                                     ResultValue=abs(twohitsstatit), ResultValueIterative=twohitsstatit, Weighting=30))

            #Calculate so-excites (maximize)
            self.results.append(dict(Result="soexcite", ResultFlavour="GAMEPLAY SCALAR STATS",
                                     ResultValue=1.0/self.stats.soexcitespegging if self.stats.soexcitespegging > 0.1 else 1.0,
                                     Weighting=1))

            #Calculate repeats (minimize)
            self.results.append(dict(Result="repeats", ResultFlavour="GAMEPLAY SCALAR STATS",
                                     ResultValue=self.stats.repeats/len(self.moves) if len(self.moves) > 0 else 1,
                                     Weighting=400))

            #GAMEPLAY STATISTICAL STATS (lol)
            moves_df = pd.DataFrame.from_records([dict(movenum=m.movenum, trial=m.trial) for m in self.moves])
            movesPerTrial_df = moves_df[['trial']].assign(moves=1).groupby('trial').agg('sum').reset_index()
            movesPerTrial_df.sort_values(['trial'])

            #Events over time
            eventsOverTime_df = pd.DataFrame.from_records([dict(movenum=m.movenum, hasevent=1 if m.hasEvent else 0,
                                                                trial=m.trial) for m in self.moves])
            if len(eventsOverTime_df) == 0:
                result = 1
            else:
                eventsOverTime_l = self.processActualTimeCurve(movesPerTrial_df, eventsOverTime_df, "hasevent")
                result = self.discreteRegression(gp.eventsovertimecurvefile, eventsOverTime_l, smoothing=0.7)
            self.results.append(dict(Result="eventsOverTime_curvefit", ResultFlavour="GAMEPLAY STATISTICAL STATS (lol)",
                                     ResultValue=result, Weighting=20))

            #Energy over time
            energyOverTime_df = pd.DataFrame.from_records([dict(movenum=m.movenum, eventmag=abs(m.ladderorchuteamt),
                                                                trial=m.trial) for m in self.moves])
            if len(energyOverTime_df) == 0:
                result = 1
            else:
                eventsOverTime_l = self.processActualTimeCurve(movesPerTrial_df, energyOverTime_df, "eventmag")
                result =self.discreteRegression(gp.eventenergyfile, eventsOverTime_l, smoothing=0.6)
            self.results.append(dict(Result="energy_curvefit", ResultFlavour="GAMEPLAY STATISTICAL STATS (lol)",
                                     ResultValue=result, Weighting=14))

            #Velocity over time
            velocityOverTime_df = pd.DataFrame.from_records([dict(movenum=m.movenum, score=m.score,
                                                                trial=m.trial) for m in self.moves])
            if len(velocityOverTime_df) == 0:
                result = 1
            else:
                velocityOverTime_l = self.processActualTimeCurve(movesPerTrial_df, velocityOverTime_df, "score")
                #Smooth, since we want to curve match general trends
                result =self.discreteRegression(gp.velocityovertimecurvefile, velocityOverTime_l, smoothing=0.6)
            self.results.append(dict(Result="velocity_curvefit", ResultFlavour="GAMEPLAY STATISTICAL STATS (lol)",
                                     ResultValue=result, Weighting=4))

            #Event length distribution as histogram
            eventsLengthHist_l = self.processActualHistCurve([abs(m.ladderorchuteamt) for m in self.moves if m.hasEvent])
            if len(eventsLengthHist_l) == 0:
                result = 1
            else:
                result = self.discreteRegression(gp.eventlengthdisthistcurvefile, eventsLengthHist_l)
            self.results.append(dict(Result="eventsHitLengthDistribution_curvefit",
                                     ResultFlavour="GAMEPLAY STATISTICAL STATS (lol)",
                                     ResultValue=result, Weighting=10))

            #If more than 50% of events are between 2 & 4 spaces, penalize (track-wise)
            for t in self.board.tracks:
                tracksByLength_l = [abs(m.ladderorchuteamt) for m in self.moves if (m.hasEvent and m.track == t.num)]
                shortTracks_l = [e for e in tracksByLength_l if e <= 4]
                if len(tracksByLength_l) > 0 and len(shortTracks_l)*2 > len(tracksByLength_l):
                    self.results.append(dict(Result="eventsHitLengthDistribution_bottomheavy_T{}".format(t.Track_ID),
                                             ResultFlavour="GAMEPLAY STATISTICAL STATS (lol)",
                                             ResultValue=len(shortTracks_l)/len(tracksByLength_l) - 0.5, Weighting=10))


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


    def discreteRegression(self, idealCurveFileName, actualCurve, smoothing=0.0):
        #Smooth if specified
        if smoothing > 0:
            smoothedCurve = []
            for idx in range(len(actualCurve)):
                if idx == 0: smoothedCurve.append(actualCurve[idx])
                else:
                    smoothedCurve.append((actualCurve[idx][0],
                                          actualCurve[idx][1]*(1.0-smoothing) + smoothedCurve[idx-1][1]*smoothing))
        else: smoothedCurve = actualCurve

        #Normalize smoothed actual curve
        normSmoothedActCurve = self.eventSetBuilder.normalizeCurveMagnitude(smoothedCurve)

        #TODO: scale to this as init, then make   SLEs, where y scaling factor is constant and minimized, use that for regr

        #Retrieve & normalize ideal curve from file
        normIdealCurve = self.eventSetBuilder.getNormalizedIdealCurve(idealCurveFileName)
        max_x = max([a[0] for a in normSmoothedActCurve])
        max_y = max([a[1] for a in normSmoothedActCurve])
        actualizedIdealCurve = self.eventSetBuilder.actualizeCurve(normIdealCurve, max_x, max_y)

        #Discretize ideal
        discrIdeal = []
        ideal_idx = 0
        for a in normSmoothedActCurve:
            while ideal_idx < len(actualizedIdealCurve) - 1 and actualizedIdealCurve[ideal_idx][0] < a[0]: ideal_idx += 1
            discrIdeal.append((a[0], actualizedIdealCurve[ideal_idx][1]))


        # Determine optimal scaling for best fit
        curveOptimizer = CurveOptimizer(normSmoothedActCurve, discrIdeal)
        # optimal_scale_opt = curveOptimizer.find_optimal_scale()
        optimal_scale_opt = curveOptimizer.find_optimal_scale_analytical()

        optimizedIdealCurve = curveOptimizer.apply_scaling()

        #Iterate over actuals comparing to ideal (least squares)
        regressionCurve = []
        regressionSum = 0
        for a, o in zip(normSmoothedActCurve, optimizedIdealCurve) :
            diffSquard = math.pow(o[1] - a[1], 2)
            regressionCurve.append((a[0], diffSquard))
            regressionSum += diffSquard

        # ##########################################
        # #TEMP
        # # Plotting the points
        # x_smth= [x for x, y in normSmoothedActCurve]
        # y_smth = [y for x, y in normSmoothedActCurve]
        # plt.plot(x_smth, y_smth, marker='o', linestyle='-', color='b')
        #
        # x_values = [x for x, y in optimizedIdealCurve]
        # y_values = [y for x, y in optimizedIdealCurve]
        # plt.plot(x_values, y_values, marker='o', linestyle='-', color='r')
        #
        # x_regr = [x for x, y in regressionCurve]
        # y_regr  = [y for x, y in regressionCurve]
        # plt.plot(x_regr , y_regr , marker='o', linestyle='-', color='g')
        #
        # # Adding labels and title
        # plt.xlabel('X values')
        # plt.ylabel('Y values')
        # # Get the base name (file name with extension)
        # file_name_with_ext = os.path.basename(idealCurveFileName)
        #
        # # Remove the extension to get the file name
        # file_name = os.path.splitext(file_name_with_ext)[0]
        # plt.title(file_name)
        #
        # # Display the plot
        # plt.show()
        # plt.waitforbuttonpress()
        # plt.close()
        #
        # ###################################

        return regressionSum/len(regressionCurve)

    import numpy as np
    from scipy.optimize import minimize
    import matplotlib.pyplot as plt

class CurveOptimizer:
    def __init__(self, smoothed_curve, actualized_ideal_curve):
        """
        Initialize the CurveOptimizer with two lists of (x, y) coordinates.

        :param smoothed_curve: List of tuples [(x1, y1), (x2, y2), ...]
        :param actualized_ideal_curve: List of tuples [(x1, y1), (x2, y2), ...]
        """
        self.smoothed_curve = smoothed_curve
        self.actualized_ideal_curve = actualized_ideal_curve
        self.optimal_scale = None
        self.scaled_curve = None
        self.validate_curves()

    def validate_curves(self):
        """
        Validate that both curves have the same length and matching x coordinates.
        """
        if len(self.smoothed_curve) != len(self.actualized_ideal_curve):
            raise ValueError("Both curves must have the same number of points.")

        for idx, ((x1, _), (x2, _)) in enumerate(zip(self.smoothed_curve, self.actualized_ideal_curve)):
            if x1 != x2:
                raise ValueError(f"X coordinates do not match at index {idx}: {x1} != {x2}")
        print("Curves validated successfully. Both curves have the same length and matching x coordinates.")

    def find_optimal_scale(self):
        """
        Find the scaling factor for the y-values of actualized_ideal_curve that minimizes
        the least squares difference between smoothed_curve and scaled actualized_ideal_curve.

        :return: optimal scaling factor (float)
        """
        y_smoothed = np.array([y for x, y in self.smoothed_curve])
        y_actualized_ideal = np.array([y for x, y in self.actualized_ideal_curve])

        # Define the least squares error function
        def least_squares_error(scale):
            y_scaled = scale * y_actualized_ideal
            error = np.sum((y_smoothed - y_scaled) ** 2)
            return error

        # Initial guess for the scaling factor
        initial_guess = np.array(1.0)

        # Perform the optimization
        result = minimize(least_squares_error, x0=initial_guess, method='BFGS')

        if result.success:
            self.optimal_scale = result.x[0]
            return self.optimal_scale
        else:
            raise RuntimeError(f"Optimization failed: {result.message}")

    def find_optimal_scale_analytical(self):
        """
        Find the scaling factor using the analytical least squares solution.

        :return: optimal scaling factor (float)
        """
        y_smoothed = np.array([y for x, y in self.smoothed_curve])
        y_actualized_ideal = np.array([y for x, y in self.actualized_ideal_curve])

        numerator = np.dot(y_smoothed, y_actualized_ideal)
        denominator = np.dot(y_actualized_ideal, y_actualized_ideal)

        if denominator == 0:
            print("y_smoothed:")
            print(y_smoothed)
            print("y_actualized_ideal:")
            print(y_actualized_ideal)
            raise ZeroDivisionError("Denominator in scaling factor calculation is zero.")

        self.optimal_scale = numerator / denominator
        print(f"Analytical optimal scaling factor: {self.optimal_scale:.6f}")
        return self.optimal_scale

    def apply_scaling(self):
        """
        Apply the optimal scaling factor to the actualized_ideal_curve.
        """
        if self.optimal_scale is None:
            self.find_optimal_scale_analytical()  # Use analytical method by default

        self.scaled_curve = [
            (x, y * self.optimal_scale) for (x, y) in self.actualized_ideal_curve
        ]
        return self.scaled_curve

    def get_scaled_curve(self):
        """
        Get the scaled actualized ideal curve. If scaling has not been applied yet,
        it will apply scaling first.

        :return: List of tuples [(x1, y1_scaled), (x2, y2_scaled), ...]
        """
        if self.scaled_curve is None:
            self.apply_scaling()
        return self.scaled_curve

    def least_squares_difference(self):
        """
        Calculate the least squares difference between smoothed_curve and scaled actualized_ideal_curve.

        :return: sum of squared differences (float)
        """
        if self.optimal_scale is None:
            self.find_optimal_scale_analytical()

        y_smoothed = np.array([y for x, y in self.smoothed_curve])
        y_scaled = self.optimal_scale * np.array([y for x, y in self.actualized_ideal_curve])
        return np.sum((y_smoothed - y_scaled) ** 2)

    def plot_curves(self, show=True, save_path=None):
        """
        Plot the original smoothed curve and the scaled actualized ideal curve.

        :param show: If True, displays the plot.
        :param save_path: If provided, saves the plot to the specified path.
        """
        if self.scaled_curve is None:
            self.apply_scaling()

        x_smoothed = [x for x, y in self.smoothed_curve]
        y_smoothed = [y for x, y in self.smoothed_curve]

        x_scaled = [x for x, y in self.scaled_curve]
        y_scaled = [y for x, y in self.scaled_curve]

        plt.figure(figsize=(10, 6))
        plt.plot(x_smoothed, y_smoothed, label='Smoothed Curve', marker='o', linestyle='-', color='blue')
        plt.plot(x_scaled, y_scaled, label='Scaled Actualized Ideal Curve', marker='x', linestyle='--', color='red')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Curve Comparison')
        plt.legend()
        plt.grid(True)

        if save_path:
            plt.savefig(save_path)
            print(f"Plot saved to {save_path}.")

        if show:
            plt.show()

