# TODO: ensure all the following are evaluated: balance, # distribution of events, # regression fitting of events over time to ideal curve, # fit of the other curves, # game length (maybe allow chute cancelling), # 2-hits (maybe fit curve?), # So excite, # repeats, # minimize othos, # maximize multis
import math
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from cribsandladders.config import GameConfig, DEFAULT_CONFIG
from cribsandladders import evaluator_metrics as em

# NOTE: scipy.optimize.minimize is imported lazily inside
# CurveOptimizer.find_optimal_scale (the only place it's used) instead
# of at module scope. discreteRegression -- the only call site in this
# file -- actually calls find_optimal_scale_analytical, the closed-form
# numpy-only alternative below, so scipy was never on the hot path for
# anything currently exercised; it was just an import-time tax on this
# whole module. This mirrors the Phase 2 fix to Player.py's scoretree
# import and the Phase 4 fix to EventSetBuilder.py's markovgame import.


class Evaluator:
    """
    Evaluates the quality of a generated game board layout using statistical analysis,
    gameplay simulation results, and regression fitting against ideal game curves.
    """

    def __init__(self, eventSetBuilder, board, possibleEvents, stats, sqlOptimizerCon, optimizerRunSet, optimizerRun,
                 config: GameConfig = DEFAULT_CONFIG):
        self.eventSetBuilder = eventSetBuilder
        self.board = board
        self.possibleEvents = possibleEvents
        self.stats = stats
        if self.stats is not None:
            self.moves = stats.moves
        self.sqlOptimizerCon = sqlOptimizerCon
        self.optimizerRunSet = optimizerRunSet
        self.optimizerRun = optimizerRun
        self.config = config
        self.results = []

    def lossFunction(self):
        """Placeholder for a combined loss score calculation."""
        temp1 = 0

    def writeMetricsToDb(self):
        """Writes evaluation metrics to the console (and handles legacy DB logic)."""
        # Input metric info
        # [Legacy database insertion code omitted but structure preserved for logic]
        for record in self.results:
            try:
                print("RUN {}: Result={}, ResultFlavour={}, ResultValue={}".format(
                    self.optimizerRun, record['Result'], record['ResultFlavour'], record['ResultValue']))
            except Exception as e:
                print(e)

    def detMetrics(self, onlyGameBoardStats = False):
        """
        Calculates all board and gameplay metrics.

        Thin orchestrator (Refactor Mk II, Phase 6) -- see
        cribsandladders/evaluator_metrics.py for the self-contained
        per-result-block logic, and the private `_*_result(s)` methods
        below for the handful of blocks that still need
        `self.eventSetBuilder`'s curve-math delegation. This used to be
        one 193-line method appending result dicts inline throughout;
        calls run in the same order the original inline code did and no
        behavior changed -- see evaluator_metrics.py's module docstring
        for details.

        Args:
            onlyGameBoardStats (bool): If True, skips simulation-based gameplay stats.
        """
        # GAME BOARD STRUCTURE SCALAR STATS
        self.results.extend(em.structure_scalar_stats(
            self.eventSetBuilder.events, self.eventSetBuilder.orthos,
            self.eventSetBuilder.multis, self.eventSetBuilder.cancels, self.config))
        self.results.extend(em.early_termination_stats(
            self.eventSetBuilder.eventNodesByTrack, self.board, self.config))

        # GAME BOARD STRUCTURE STATISTIC STATS
        self.results.append(self._event_spacing_histogram_result())

        if onlyGameBoardStats:
            # Event length track distribution as histogram
            # Only do this if in prelim track eval mode, since better appraised by gameplay stat below
            self.results.extend(self._track_event_length_distribution_results())

        if not onlyGameBoardStats:
            # GAMEPLAY SCALAR STATS

            # Balance OMITING FROM EVAL SINCE DEALT W/ IN SETTER
            self.results.extend(em.balance_stats(self.stats.partialBalanceSet, self.board))

            # Game length OMITING FROM EVAL SINCE DEALT W/ IN SETTER
            self.results.append(em.gamelength_stat(self.stats.avglengthinrounds, self.config))

            self.results.append(em.twohits_stat(self.moves, self.board.tracks, self.config))

            # Calculate so-excites (maximize)
            self.results.append(em.soexcite_stat(self.stats.soexcitespegging))

            # Calculate repeats (minimize)
            self.results.append(em.repeats_stat(self.stats.repeats, len(self.moves)))

            # Calculate skew of chutes or ladders hit as compared to tot # events hit
            self.results.append(em.events_hit_skew_stat(self.moves))

            # GAMEPLAY STATISTICAL STATS (lol)
            moves_df = pd.DataFrame.from_records([dict(movenum = m.movenum, trial = m.trial) for m in self.moves])
            movesPerTrial_df = moves_df[['trial']].assign(moves = 1).groupby('trial').agg('sum').reset_index()
            movesPerTrial_df.sort_values(['trial'])

            self.results.append(self._events_over_time_result(movesPerTrial_df))
            self.results.append(self._energy_over_time_result(movesPerTrial_df))
            self.results.append(self._velocity_over_time_result(movesPerTrial_df))

            # Event length distribution as histogram (+ per-track bottom-heavy check)
            self.results.extend(em.event_length_distribution_stats(
                self.moves, self.board.tracks, self.processActualHistCurve, self.discreteRegression, self.config))

    def _event_spacing_histogram_result(self):
        """The 'EVENT SPACING HISTOGRAM' block of detMetrics -- needs
        self.processActualHistCurve/self.discreteRegression (which
        delegate to self.eventSetBuilder's curve-math methods), so it
        stays a method rather than moving to evaluator_metrics.py. See
        that module's docstring for why."""
        spacingsRaw_l = []
        for nds in self.eventSetBuilder.eventNodesByTrack:
            prvNode = 0
            for nd in nds['nodes']:
                spacingsRaw_l.append(nd - prvNode)
                prvNode = nd
        if len(spacingsRaw_l) > 0:
            spacingsHist_l = self.processActualHistCurve(spacingsRaw_l)
            result = self.discreteRegression(self.config.eventspacingsdisthistcurvefile,
                                             spacingsHist_l)
        else:
            result = 1
        return dict(Result = "eventSpacingHist_curvefit", ResultFlavour = "GAME BOARD STRUCTURE STATISTIC STATS",
                   ResultValue = result, Weighting = 15)

    def _track_event_length_distribution_results(self):
        """The onlyGameBoardStats-only 'Event length track distribution'
        block. Same curve-math-delegation reason as
        _event_spacing_histogram_result for staying a method."""
        results = []
        for t in self.board.tracks:
            eventsLengthHist_l = self.processActualHistCurve([e.length for e in t.eventSetBuild])
            if len(eventsLengthHist_l) == 0:
                result = 1
            else:
                result = self.discreteRegression(self.config.eventlengthdisthistcurvefile, eventsLengthHist_l)
            results.append(dict(Result = "trackEventLengthDistribution_curvefit_T{}".format(t.Track_ID),
                               ResultFlavour = "GAME BOARD STRUCTURE STATISTIC STATS",
                               ResultValue = result, Weighting = 8))
        return results

    def _events_over_time_result(self, movesPerTrial_df):
        """'Events over time' curve fit. Same curve-math-delegation
        reason as _event_spacing_histogram_result for staying a method."""
        eventsOverTime_df = pd.DataFrame.from_records([dict(movenum = m.movenum, hasevent = 1 if m.hasEvent else 0,
                                                            trial = m.trial) for m in self.moves])
        if len(eventsOverTime_df) == 0:
            result = 1
        else:
            eventsOverTime_l = self.processActualTimeCurve(movesPerTrial_df, eventsOverTime_df, "hasevent")
            result = self.discreteRegression(self.config.eventsovertimecurvefile, eventsOverTime_l, smoothing = 0.7)
        return dict(Result = "eventsOverTime_curvefit", ResultFlavour = "GAMEPLAY STATISTICAL STATS (lol)",
                   ResultValue = result, Weighting = 20)

    def _energy_over_time_result(self, movesPerTrial_df):
        """'Energy over time' curve fit. Same curve-math-delegation
        reason as _event_spacing_histogram_result for staying a method."""
        energyOverTime_df = pd.DataFrame.from_records([dict(movenum = m.movenum, eventmag = abs(m.ladderorchuteamt),
                                                            trial = m.trial) for m in self.moves])
        if len(energyOverTime_df) == 0:
            result = 1
        else:
            eventsOverTime_l = self.processActualTimeCurve(movesPerTrial_df, energyOverTime_df, "eventmag")
            result = self.discreteRegression(self.config.eventenergyfile, eventsOverTime_l, smoothing = 0.6)
        return dict(Result = "energy_curvefit", ResultFlavour = "GAMEPLAY STATISTICAL STATS (lol)",
                   ResultValue = result, Weighting = 14)

    def _velocity_over_time_result(self, movesPerTrial_df):
        """'Velocity over time' curve fit. Same curve-math-delegation
        reason as _event_spacing_histogram_result for staying a method."""
        velocityOverTime_df = pd.DataFrame.from_records([dict(movenum = m.movenum, score = m.score,
                                                              trial = m.trial) for m in self.moves])
        if len(velocityOverTime_df) == 0:
            result = 1
        else:
            velocityOverTime_l = self.processActualTimeCurve(movesPerTrial_df, velocityOverTime_df, "score")
            # Smooth, since we want to curve match general trends
            result = self.discreteRegression(self.config.velocityovertimecurvefile, velocityOverTime_l, smoothing = 0.6)
        return dict(Result = "velocity_curvefit", ResultFlavour = "GAMEPLAY STATISTICAL STATS (lol)",
                   ResultValue = result, Weighting = 4)

    def processActualTimeCurve(self, movesPerTrial_df, curve_df, y_field):
        """Discretizes and normalizes gameplay data over a standardized time curve."""
        merged_df = pd.merge(movesPerTrial_df, curve_df, on = ['trial'], suffixes = ('_sum', ''))
        merged_df['movePct'] = merged_df['movenum'] / merged_df['moves']
        merged_df.sort_values(['movePct'])

        merged_l = self.eventSetBuilder.discretizeCurve(list(zip(merged_df['movePct'].to_list(), merged_df[y_field].to_list())),
                                                        self.config.effectiveboardlength, accumulate = True)
        # Need to re-normalize since accumulated with smoothing applied
        final_l = self.eventSetBuilder.normalizeCurveMagnitude(merged_l)
        return final_l

    def processActualHistCurve(self, curve_l):
        """Converts a raw list of values into a frequency histogram."""
        curve_l.sort()
        curve_idx = 0
        merged_l = []
        if not curve_l:
            return merged_l
        for b in range(1, max(curve_l) + 1):
            accum = 0
            while curve_idx < len(curve_l) and curve_l[curve_idx] == b:
                accum += 1
                curve_idx += 1
            merged_l.append([b, accum])
        return merged_l

    def discreteRegression(self, idealCurveFileName, actualCurve, smoothing = 0.0):
        """Calculates the regression error between an actual curve and a normalized ideal curve."""
        # Smooth if specified
        if smoothing > 0:
            smoothedCurve = []
            for idx in range(len(actualCurve)):
                if idx == 0:
                    smoothedCurve.append(actualCurve[idx])
                else:
                    smoothedCurve.append((actualCurve[idx][0],
                                          actualCurve[idx][1] * (1.0 - smoothing) + smoothedCurve[idx - 1][1] * smoothing))
        else:
            smoothedCurve = actualCurve

        # Normalize smoothed actual curve
        normSmoothedActCurve = self.eventSetBuilder.normalizeCurveMagnitude(smoothedCurve)

        #TODO: scale to this as init, then make   SLEs, where y scaling factor is constant and minimized, use that for regr
        # Retrieve & normalize ideal curve from file
        normIdealCurve = self.eventSetBuilder.getNormalizedIdealCurve(idealCurveFileName)
        max_x = max([a[0] for a in normSmoothedActCurve])
        max_y = max([a[1] for a in normSmoothedActCurve])
        actualizedIdealCurve = self.eventSetBuilder.actualizeCurve(normIdealCurve, max_x, max_y)

        # Discretize ideal
        discrIdeal = []
        ideal_idx = 0
        for a in normSmoothedActCurve:
            while ideal_idx < len(actualizedIdealCurve) - 1 and actualizedIdealCurve[ideal_idx][0] < a[0]:
                ideal_idx += 1
            discrIdeal.append((a[0], actualizedIdealCurve[ideal_idx][1]))

        # Determine optimal scaling for best fit
        curveOptimizer = CurveOptimizer(normSmoothedActCurve, discrIdeal)
        optimal_scale_opt = curveOptimizer.find_optimal_scale_analytical()
        optimizedIdealCurve = curveOptimizer.apply_scaling()

        #Iterate over actuals comparing to ideal (least squares)
        regressionCurve = []
        regressionSum = 0
        for a, o in zip(normSmoothedActCurve, optimizedIdealCurve) :
            diffSquard = math.pow(o[1] - a[1], 2)
            regressionCurve.append((a[0], diffSquard))
            regressionSum += diffSquard

        return regressionSum / len(regressionCurve)


class CurveOptimizer:
    """
    Optimizes the vertical scaling of an ideal curve to best fit an actual data curve
    using least squares minimization.
    """

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
        """Validate that both curves have the same length and matching x coordinates."""
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
        from scipy.optimize import minimize

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
        """Apply the optimal scaling factor to the actualized_ideal_curve."""
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

    def plot_curves(self, show = True, save_path = None):
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

        plt.figure(figsize = (10, 6))
        plt.plot(x_smoothed, y_smoothed, label = 'Smoothed Curve', marker = 'o', linestyle = '-', color = 'blue')
        plt.plot(x_scaled, y_scaled, label = 'Scaled Actualized Ideal Curve', marker = 'x', linestyle = '--', color = 'red')
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