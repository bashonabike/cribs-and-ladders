# TODO: Maybe do set of events
# Run it, check scalar stats namely player balance
# Try to finesse that out by cancelling ladders
# Probably some math formula, maybe like each event length (+/-) * likelihood of hitting it (1/board length) = event value,
# use this to scale back into within 5% score of others
# hmm maybe also add or retract "two-hit" events, whereby you set up the person to immed after a ladder hit a chute, or
# 2 shoots in a row, very strategically, to help balance uneven tracks
# select best of lot once hit finesse limit
# compare shape to ideal histogram curve, plus factor in residual scalar quantities
# use to drive next set
# output specified best x number of trials
import math
import random as rd
from dataclasses import dataclass, field
from typing import Any, List

import cribsandladders.Board as bd
import numpy as np
import copy as cp
import Enums as en
import bisect as bsc
import pandas as pd
import sqlite3 as sql
import contextlib
from cribsandladders.config import GameConfig, DEFAULT_CONFIG
from cribsandladders import event_curve_math
from cribsandladders.event_set_plotter import EventSetPlotter
from cribsandladders.ortho_path import OrthoPath  # noqa: F401 -- backward-compat re-export
from cribsandladders.ortho_line_trace import OrthoLineTrace
from cribsandladders.param_set import ParamSet

# NOTE: markovgame (a compiled pybind11 extension, Tier 4 per the TDD
# refactor assessment -- outside Python unit-test reach) is imported
# lazily inside runPartialTrackEffLengthHoles, the only method that
# uses it, instead of at module scope. That's what makes `import
# cribsandladders.EventSetBuilder` -- and constructing an
# EventSetBuilder against a mocked board/possibleEvents, as
# test_eventsetbuilder.py already does -- possible without the
# extension built. Mirrors the Phase 2 fix to Player.py's scoretree
# import and the other Phase 4 import-hygiene fixes in this pass
# (Evaluator's scipy.optimize, Optimizer's lightgbm/sklearn).

# NOTE (Phase 4 decomposition follow-up): this file used to also
# define OrthoPath, OrthoLineTrace, and ParamSet inline (none of them
# actually needed anything off EventSetBuilder itself), plus a batch of
# pure curve/geometry helper methods and three matplotlib-calling
# plotting methods. Those have moved out into
# cribsandladders/ortho_path.py, ortho_line_trace.py, param_set.py,
# event_curve_math.py, and event_set_plotter.py respectively --
# EventSetBuilder now focuses on the actual event-search/placement
# algorithm (scoreEventsForHole, tryEventSet, tryGetEventForHole,
# runPartialTrackEffLengthHoles, etc.), which is still one large class
# because that algorithm's steps are too interdependent (shared
# mutable vector-test sets, in-place track/hole state) to split further
# without a much deeper redesign than "extract the already-separable
# pieces." The methods that moved keep thin delegating wrappers here
# (actualizeCurve, discretizeCurve, plotBoard, etc.) so every existing
# call site -- production code and tests alike -- keeps working
# unchanged.

import time


# TODO: curvify lines, order by length in holes, curve it bspline? then iterate out making sure curve does not interfere with any neighbours
# maybe do this as sep class, once have a board w/ tracks and events established, call Curvify
# factor in occluded space for logo etc


@dataclass
class TrackBuildState:
    """
    Per-track working state used throughout `tryEventSet`'s placement
    loop (built by `EventSetBuilder._build_track_state`, one instance
    per not-`instLocked` track). Refactor Mk II Phase 8 step 2 (see
    [[Refactor Mk ii]] in the Obsidian vault): this used to be a plain
    dict (`trackEventsOverview[i]`, ~45 keys, built as a `dict(...)`
    literal and threaded through nearly every method via `t['key']`
    string access) -- a real class makes the schema explicit and
    typo-proof, and lets tests build partial instances directly via
    `object.__new__(TrackBuildState)` + hand-set attributes (the same
    pattern `test_optimizer.py`/`test_possible_events.py` already use),
    without needing every field populated.

    Field values/semantics are unchanged from the original dict -- this
    is a structural change only, not a behavior change. See the
    `_build_track_state` docstring for how these get populated, and
    `scoreEventsForHole`/`tryGetEventForHole`/`tryEventSet` for how
    they're read and mutated during placement.
    """
    track: Any
    trackidx: int
    tracknum: int
    optevents: int = 0
    track_id: int = 0
    optfirstchute: int = 0
    trackfilled: bool = False
    tracklength: int = 0
    lengthdeviation: float = 0.0
    spacinghisto: List[Any] = field(default_factory=list)
    minspacectr: int = 0
    eventsetbuild: List[Any] = field(default_factory=list)
    candeventspecs: List[Any] = field(default_factory=list)
    lengthdistidealcurve: List[Any] = field(default_factory=list)
    lengthdistactualhist: List[Any] = field(default_factory=list)
    lengthovertimeideal: List[Any] = field(default_factory=list)
    maxlength: int = 0
    trackenergycurve: List[Any] = field(default_factory=list)
    trackenergyintegral: List[Any] = field(default_factory=list)
    candavgenergy: float = 0.0
    energybuffer: float = 0.0
    energybufferidx: int = 0
    candeventstartlookup: List[Any] = field(default_factory=list)
    candcursor: int = 0
    chutecursor: int = 0
    holecoords: List[Any] = field(default_factory=list)
    lasteventtop: int = 0
    previsladder: bool = False
    chutebases: List[Any] = field(default_factory=list)
    chutetops: List[Any] = field(default_factory=list)
    ladders: List[Any] = field(default_factory=list)
    chutes: List[Any] = field(default_factory=list)
    eventnodes: List[Any] = field(default_factory=list)
    twohitsthusfar: int = 0
    cancels: int = 0
    eventscount: int = 0
    ladderbases: List[Any] = field(default_factory=list)
    laddertops: List[Any] = field(default_factory=list)
    holescompletepct: float = 0.0
    chutescompletepct: float = 0.0
    curhole: int = 0
    compensationbuffer: float = 0.0
    trackstalledcounter: int = 0
    trackisstalled: bool = False
    multistack: List[Any] = field(default_factory=list)
    controllength: int = 0
    curestefflength: int = 0
    nomultis: bool = False
    numdenies: int = 0
    numnogos: int = 0


class VectorCollisionTracker:
    """
    Tracks vectors occupied by already-placed event instances, and tests
    candidate events for collisions against them.

    Refactor Mk II Phase 9b (see [[Refactor Mk ii]]/Phase 9 Findings in
    the Obsidian vault): replaces the pair of bare `allVectorsTest`/
    `baseVectorsTest` sets that `tryEventSet` used to construct and thread
    through `tryGetEventForHole`/`testInterceptLegality`/`updateVectorsTest`
    by hand. `would_collide` is `testInterceptLegality`'s original body,
    verbatim, operating on `self.all_vectors` -- the original
    `baseVectorsTest` parameter is dropped entirely, since it was never
    referenced anywhere in that method's body (confirmed by reading it).
    `commit` is the pure set-mutation half of the former
    `updateVectorsTest` (the other half, geometry derivation, is
    `EventSetBuilder._derive_instance_geometry`, Phase 9a -- callers must
    invoke that first so `event.crowVector`/`event.instanceStartVector`/
    `event.instanceEndVector` already exist).

    Note: `commit`'s `removal=True` path is preserved structurally (for
    possible future backtracking work) but, same as the original
    `updateVectorsTest`, has zero call sites today -- nothing in the live
    codebase ever undoes a placement -- so it has no characterization
    test.
    """

    def __init__(self, possibleEvents, config):
        self.possibleEvents = possibleEvents
        self.config = config
        self.all_vectors = set()
        self.base_vectors = set()

    def _bounding_box_plus_vector(self, vector):
        # Same computation as EventSetBuilder.boundingBoxPlusVector --
        # duplicated here (rather than reaching back into EventSetBuilder)
        # so this class only depends on possibleEvents/config, which it
        # already holds.
        ortho_dxdy = self.possibleEvents.orthogonal_vector(vector[0], vector[1], self.config.eventminspacing / 2.0, False)
        return event_curve_math.bounding_box_plus_vector(vector, ortho_dxdy)

    def would_collide(self, curEvent, t):
        """
        Test if an event's vector intercepts with existing tracked vectors.

        Returns:
            Tuple of (is_legal, result_dict) where is_legal indicates if the intercept is allowed.
        """
        if not curEvent.isOrtho:
            if self.possibleEvents.check_intersections({curEvent.crowVector}, self.all_vectors, postGenTest=True):
                return False, dict(incr=-1, rev=False)
            else:
                return True, dict(incr=-1, rev=False)
        else:
            bestIncr, revOrtho = 999, False
            runs = []
            if curEvent.orthoFwdMinIncr > 0:
                runs.append(dict(rev=False, minincr=curEvent.orthoFwdMinIncr, maxincr=curEvent.orthoFwdMaxIncr))
            if curEvent.orthoRevMinIncr > 0:
                runs.append(dict(rev=True, minincr=curEvent.orthoRevMinIncr, maxincr=curEvent.orthoRevMaxIncr))

            if len(self.all_vectors) == 150:
                sfds = ""
            for run in runs:
                ortho = self.possibleEvents.orthogonal_vector(curEvent.startHole.coords, curEvent.endHole.coords,
                                                              self.config.maxloopyorthoeventdisplacementincrements
                                                              * self.config.eventminspacing, run['rev'])
                floor, peak = self.possibleEvents.test_sidestep_events(
                    curEvent.startHole, curEvent.endHole, t.track.trackholes, t.holecoords,
                    ortho, self.config.maxloopyorthoeventdisplacementincrements * self.config.eventminspacing, self.config.eventminspacing,
                    self.all_vectors, run['rev'], minIncr=run['minincr'], maxIncr=run['maxincr'], ignoreProximity=True)
                if floor > 0 and floor < bestIncr:
                    bestIncr = floor
                    revOrtho = run['rev']
            if bestIncr == 999 or bestIncr == 0:
                return False, dict(incr=-1, rev=False)
            else:
                # TEMPP
                if len(runs) == 1 or runs[0]['rev'] == revOrtho:
                    run = runs[0]
                else:
                    run = runs[1]
                ortho = self.possibleEvents.orthogonal_vector(curEvent.startHole.coords, curEvent.endHole.coords,
                                                              self.config.maxloopyorthoeventdisplacementincrements
                                                              * self.config.eventminspacing, revOrtho)
                self.possibleEvents.test_sidestep_events(
                    curEvent.startHole, curEvent.endHole, t.track.trackholes, t.holecoords,
                    ortho, self.config.maxloopyorthoeventdisplacementincrements * self.config.eventminspacing, self.config.eventminspacing,
                    self.all_vectors, revOrtho, minIncr=run['minincr'], maxIncr=run['maxincr'], ignoreProximity=True,
                    debugTest=True)
                return True, dict(incr=bestIncr, rev=revOrtho)

    def commit(self, event, isOrtho, removal=False):
        """
        Add (or, structurally preserved for future backtracking, remove)
        an already-geometry-derived event's vectors to/from the tracked
        collision sets. Assumes
        `EventSetBuilder._derive_instance_geometry(event, isOrtho)` has
        already been called by the caller.

        Args:
            event: The event being added or removed.
            isOrtho: Boolean indicating if the event is orthogonal.
            removal: Boolean indicating whether to remove the event's
                vectors (True) or add them (False). No current call site
                passes True (see class docstring).
        """
        if len(self.all_vectors) == 150:
            # Debug breakpoint condition
            sfsd = ""
        if not removal:
            if not isOrtho:
                self.all_vectors.update(
                    set(tuple(
                        [v for v in self._bounding_box_plus_vector(event.crowVector)])))
                self.base_vectors.add(event.crowVector)
            else:
                self.all_vectors.update(
                    set(tuple(
                        [v for v in self._bounding_box_plus_vector(event.instanceStartVector)])))
                self.all_vectors.update(
                    set(tuple(
                        [v for v in self._bounding_box_plus_vector(event.instanceEndVector)])))
                self.base_vectors.add(event.instanceStartVector)
                self.base_vectors.add(event.instanceEndVector)
        else:
            if not isOrtho:
                self.all_vectors.difference_update(
                    set(tuple(
                        [v for v in self._bounding_box_plus_vector(event.crowVector)])))
                self.base_vectors.discard(event.crowVector)
            else:
                if event.instanceIncr > -1:
                    self.all_vectors.difference_update(
                        set(tuple(
                            [v for v in self._bounding_box_plus_vector(event.instanceStartVector)])))
                    self.all_vectors.difference_update(
                        set(tuple(
                            [v for v in self._bounding_box_plus_vector(event.instanceEndVector)])))
                self.base_vectors.discard(event.instanceStartVector)
                self.base_vectors.discard(event.instanceEndVector)


class EventSetBuilder:
    """
    A class to build and optimize event sets for a board game with tracks and events.

    The EventSetBuilder is responsible for generating and optimizing event configurations
    on game tracks, including ladders and chutes, to ensure balanced gameplay.

    Attributes:
        board: The game board containing tracks and holes.
        possibleEvents: Collection of possible events that can be placed on the board.
        allTentLengthHisto: List to store length histograms of tentative event sets.
        paramSet: ParamSet instance for managing event parameters.
        orthos: Counter for orthogonal events.
        multis: Counter for multi-segment events.
        events: Counter for total events.
        cancels: Counter for cancelled events.
        eventNodesByTrack: List to store event nodes by track.
        posHands: List of possible hand moves.
        posHandProbs: Probabilities for hand moves.
        posPegs: List of possible peg moves.
        posPegProbs: Probabilities for peg moves.
        pegRounds: List of peg rounds.
        pegRoundProbs: Probabilities for peg rounds.
        prevEffLengths_starter: List of effective lengths for each track.
        avgScoreSum: Sum of average scores for optimization.
        avgScoreDiv: Divisor for calculating average score.
        avgScore: Calculated average score.
    """

    def __init__(self, board, possibleEvents, config: GameConfig = DEFAULT_CONFIG, plotter=None):
        """
        Initialize the EventSetBuilder with a game board and possible events.

        Args:
            board: The game board object containing tracks and holes.
            possibleEvents: Object containing possible events that can be placed on the board.
            config (GameConfig): game configuration (defaults to the
                module-level DEFAULT_CONFIG).
            plotter: object implementing plot_board/test_plot_vectors_on_holes/
                plot_coordinates_and_vectors (defaults to a real
                `EventSetPlotter`). Tests can pass a
                `NoOpEventSetPlotter` instead to avoid matplotlib
                popups/file writes.
        """
        self.board = board
        self.possibleEvents = possibleEvents
        self.config = config
        self.plotter = plotter if plotter is not None else EventSetPlotter()
        self.allTentLengthHisto = []
        self.paramSet = ParamSet(self.board, self.board.tracks)
        self.orthos = 0
        self.multis = 0
        self.events = 0
        self.cancels = 0
        self.eventNodesByTrack = []
        self.posHands = [item["move"] for item in self.config.probHandHist]
        self.posHandProbs = [item["prob"] for item in self.config.probHandHist]
        self.posPegs = [item["move"] for item in self.config.probPegHist]
        self.posPegProbs = [item["prob"] for item in self.config.probPegHist]
        self.pegRounds = [item["rounds"] for item in self.config.probPegRounds]
        self.pegRoundProbs = [item["prob"] for item in self.config.probPegRounds]
        self.prevEffLengths_starter = [dict(track_id=t.Track_ID, efflength=len(t.trackholes))
                                       for t in self.board.tracks]
        self.avgScoreSum, self.avgScoreDiv, self.avgScore = 0, 0, 0

        # TEMPPP
        self.miniMarkovTime, self.scoringTime, self.totalTime, self.benchmarkSetupTime = 0, 0, 0, 0

        # Load benchmarks
        self.benchmarkMoves_df = None
        self.track_dict = None
        self.retrieveOrGenerateBenchmarkMoves()

    def clearEventSet(self):
        """
        Reset all event-related attributes to their initial state.

        Clears all temporary data structures and counters used during event set generation.
        """
        self.allTentLengthHisto = []
        self.orthos = 0
        self.multis = 0
        self.events = 0
        self.cancels = 0
        self.eventNodesByTrack = []
        for t in self.board.tracks:
            t.eventSetBuild = []
            t.instLocked = False
        self.avgScoreSum, self.avgScoreDiv, self.avgScore = 0, 0, 0

    def optimizeSetup(self):
        """
        Optimize the event set configuration for the game board.

        Performs Monte Carlo simulation to find an optimal set of events
        that provides balanced gameplay. Raises an exception if no valid
        configuration is found within the maximum allowed iterations.

        Raises:
            Exception: If no valid event set is found within the maximum iterations.
        """
        builditer = 0
        self.paramSet.monteCarlo()
        prevEffLengths = cp.deepcopy(self.prevEffLengths_starter)
        while not self.tryEventSet(self.paramSet, prevEffLengths):
            builditer += 1
            if builditer > self.config.maxitertrynewbuild:
                raise Exception(
                    "Passed max # iters ({}) to find an event set.  ".format(self.config.maxitertrynewbuild) +
                    "This board may not be feasible.  Try adding more folds in the tracks")
        self.buildSetIntoEvents()
        # TEMPPP!
        # self.plot_coordinates_and_vectors()

    def runMonteCarlo(self, optimizerRunSet, optimizerRun):
        """
        Run a Monte Carlo simulation to optimize event parameters.

        Args:
            optimizerRunSet: Identifier for the optimizer run set.
            optimizerRun: Identifier for the specific optimizer run.

        Returns:
            bool: True if a valid event set was found, False otherwise.
        """
        self.clearEventSet()
        self.paramSet.monteCarlo()
        self.paramSet.tempInsertParamsDb(optimizerRunSet, optimizerRun)
        builditer = 0
        prevEffLengths = cp.deepcopy(self.prevEffLengths_starter)
        while not self.tryEventSet(self.paramSet, prevEffLengths):
            builditer += 1
            if builditer > self.config.maxitertrynewbuild:
                raise Exception(
                    "Passed max # iters ({}) to find an event set.  ".format(self.config.maxitertrynewbuild) +
                    "This board may not be feasible.  Try adding more folds in the tracks")
        self.buildSetIntoEvents()
        # self.plot_coordinates_and_vectors()

    def runMidpointInitParams(self, optimizerRunSet, optimizerRun):
        """
        Initialize parameters using midpoint values and attempt to build a valid event set.
        
        Args:
            optimizerRunSet: Identifier for the optimizer run set.
            optimizerRun: Identifier for the specific optimizer run.
            
        Raises:
            Exception: If no valid event set is found within the maximum iterations.
        """
        self.clearEventSet()
        self.paramSet.midpointInitParams()
        self.paramSet.tempInsertParamsDb(optimizerRunSet, optimizerRun)
        builditer = 0
        prevEffLengths = cp.deepcopy(self.prevEffLengths_starter)
        while not self.tryEventSet(self.paramSet, prevEffLengths):
            # TEMPPPP
            # self.buildSetIntoEvents()
            # self.plot_coordinates_and_vectors()
            builditer += 1
            if builditer > self.config.maxitertrynewbuild:
                raise Exception(
                    "Passed max # iters ({}) to find an event set.  ".format(self.config.maxitertrynewbuild) +
                    "This board may not be feasible.  Try adding more folds in the tracks")
        self.buildSetIntoEvents()
        # self.plot_coordinates_and_vectors()

    def setParamsIntoDb(self, optimizerRunSet, optimizerRun):
        """
        Store the current parameter set in the database.
        
        Args:
            optimizerRunSet: Identifier for the optimizer run set.
            optimizerRun: Identifier for the specific optimizer run.
        """
        self.paramSet.tempInsertParamsDb(optimizerRunSet, optimizerRun)

    def buildBoardFromParamsDb(self, optimizerRunSet, optimizerRun):
        """
        Build a board configuration using parameters from the database.
        
        Args:
            optimizerRunSet: Identifier for the optimizer run set.
            optimizerRun: Identifier for the specific optimizer run.
            
        Raises:
            Exception: If no valid event set is found within the maximum iterations.
        """
        self.clearEventSet()
        self.paramSet.intakeParamsFromDb(optimizerRunSet, optimizerRun)
        # self.paramSet.tempInsertParamsDb(100000+optimizerRun)
        builditer = 0
        prevEffLengths = cp.deepcopy(self.prevEffLengths_starter)
        while not self.tryEventSet(self.paramSet, prevEffLengths):
            builditer += 1
            if builditer > self.config.maxitertrynewbuild:
                raise Exception(
                    "Passed max # iters ({}) to find an event set.  ".format(self.config.maxitertrynewbuild) +
                    "This board may not be feasible.  Try adding more folds in the tracks")
        self.buildSetIntoEvents()

    def modParamsForFmin(self, paramsSubset, fminParamsList, optimizerRunSet, optimizerRun):
        """
        Modify parameters for function minimization and attempt to build a valid event set.
        
        Args:
            paramsSubset: Subset of parameters to modify.
            fminParamsList: List of parameter values for function minimization.
            optimizerRunSet: Identifier for the optimizer run set.
            optimizerRun: Identifier for the specific optimizer run.
            
        Raises:
            Exception: If no valid event set is found within the maximum iterations.
        """
        self.paramSet.modParamsForFmin(paramsSubset, fminParamsList)

        self.clearEventSet()
        self.paramSet.tempInsertParamsDb(optimizerRunSet, optimizerRun)
        builditer = 0
        prevEffLengths = cp.deepcopy(self.prevEffLengths_starter)
        while not self.tryEventSet(self.paramSet, prevEffLengths):
            builditer += 1
            if builditer > self.config.maxitertrynewbuild:
                raise Exception(
                    "Passed max # iters ({}) to find an event set.  ".format(self.config.maxitertrynewbuild) +
                    "This board may not be feasible.  Try adding more folds in the tracks")
        self.buildSetIntoEvents()

    def buildBoardFromParams(self, instanceParams_df, optimizerRunSet, optimizerRun):
        """
        Build a board configuration using the provided parameters.
        
        Args:
            instanceParams_df: DataFrame containing parameter configurations.
            optimizerRunSet: Identifier for the optimizer run set.
            optimizerRun: Identifier for the specific optimizer run.
            
        Returns:
            List of event sets for each track if successful, None otherwise.
        """
        self.clearEventSet()
        self.paramSet.intakeParams(instanceParams_df)
        builditer = 0
        prevEffLengths = cp.deepcopy(self.prevEffLengths_starter)
        while not self.tryEventSet(self.paramSet, prevEffLengths):
            builditer += 1
            if builditer > self.config.maxitertrynewbuild:
                print("Passed max # iters ({}) to find an event set.  ".format(self.config.maxitertrynewbuild) +
                      "This board may not be feasible.  Try adding more folds in the tracks")
                return None
        self.paramSet.tempInsertParamsDb(optimizerRunSet, optimizerRun)
        self.buildSetIntoEvents()
        return [t.eventSetBuild for t in self.board.tracks]

    def plotBoard(self):
        """
        Generate a plot of the current board configuration.

        This creates a visual representation of the board with all tracks and events.
        Delegates to `self.plotter` (see event_set_plotter.py).
        """
        self.plotter.plot_board(self)

    def retrieveOrGenerateBenchmarkMoves(self):
        """
        Retrieve benchmark moves from the database or generate new ones if none exist.
        
        This method manages the benchmark moves used for evaluating board configurations.
        It will either load existing benchmarks from the database or generate new ones
        by simulating moves on the current track configuration.
        """
        start_time = time.time()
        with contextlib.closing(sql.connect(self.config.optimizer_db_path)) as sqlConn:
            with sqlConn:
                with contextlib.closing(sqlConn.cursor()) as sqliteCursor:
                    # Try retrieve benchmark moves
                    query = "SELECT * FROM BenchmarkMoves WHERE Board_ID = ?"
                    sqliteCursor.execute(query, [self.board.boardID])
                    self.benchmarkMoves_df = pd.DataFrame(sqliteCursor.fetchall(),
                                                          columns=[d[0] for d in sqliteCursor.description])
        if len(self.benchmarkMoves_df) == 0:
            # Generate new benchmark moves out to double track length
            insertQuery = "INSERT INTO BenchmarkMoves VALUES(?,?,?,?,?)"
            for t in self.board.tracks:
                with contextlib.closing(sql.connect(self.config.optimizer_db_path)) as sqlConn:
                    with sqlConn:
                        with contextlib.closing(sqlConn.cursor()) as sqliteCursor:
                            sqliteCursor.execute("BEGIN TRANSACTION")
                            effLength, sequences = self.runPartialTrackEffLengthHoles(t.Track_ID, [],
                                                                                      2 * len(t.trackholes))
                            for trial in range(len(sequences)):
                                for move in range(len(sequences[trial])):
                                    sqliteCursor.execute(insertQuery, [self.board.boardID, t.Track_ID, trial, move,
                                                                       sequences[trial][move]])
                            sqliteCursor.execute("END TRANSACTION")

            # Retrieve newly created benchmarks
            with contextlib.closing(sql.connect(self.config.optimizer_db_path)) as sqlConn:
                with sqlConn:
                    with contextlib.closing(sqlConn.cursor()) as sqliteCursor:
                        # Try retrieve benchmark moves
                        query = "SELECT * FROM BenchmarkMoves WHERE Board_ID = ?"
                        sqliteCursor.execute(query, [self.board.boardID])
                        self.benchmarkMoves_df = pd.DataFrame(sqliteCursor.fetchall(),
                                                              columns=[d[0] for d in sqliteCursor.description])

        # Index & sort
        # self.benchmarkMoves_df.set_index(['Track_ID', 'Trial', 'MoveNum'], inplace=True)
        # self.benchmarkMoves_df.sort_index(inplace=True)

        # DataFrame -> {track_id: [[move_val, ...], ...]} hydration is a
        # pure function of the DataFrame (see
        # test_event_curve_math.py::test_build_track_dict_from_benchmark_moves_df),
        # split out to event_curve_math.py in the Phase 4 decomposition
        # follow-up.
        self.track_dict = event_curve_math.build_track_dict_from_benchmark_moves_df(self.benchmarkMoves_df)
        end_time = time.time()
        self.benchmarkSetupTime += end_time - start_time

    def buildPartialSetIntoTrack(self, track, startPoint, stopPoint):
        """
        Build a partial set of events into a track between specified points.

        This method adds events to a track between the given start and stop points,
        handling both regular and shared events that may span multiple tracks.

        Args:
            track: The track to add events to.
            startPoint: Starting index for event selection.
            stopPoint: Ending index for event selection.

        TODO(liam): flagged by the Phase 9 design spike (see [[Refactor
        Mk ii]]/Phase 9 Findings in the Obsidian vault). This method has
        zero call sites anywhere in the repo -- dead code. Its own
        vector-intersection-checking logic is commented out inline
        below ("TODO: figure this out in possible events, for now just
        excluding here"), so as it stands it just picks a random
        candidate event (`rd.choice`) and adds it with no collision
        checking at all, unlike `tryEventSet`'s real placement loop
        (`tryGetEventForHole`/`VectorCollisionTracker`). Not part of the
        Phase 9 collision-tracker redesign -- worth deciding separately
        whether to revive this (wiring it into the commented-out check,
        or the real `VectorCollisionTracker`) or delete it.

        Note:
            For shared events, this will also add linked events to other tracks.
        """
        for e in range(startPoint, stopPoint):
            if len(track.candidateEvents.candidateEvents) == 0:
                dsfd = "sdfds"
            # TODO: maybe start at beginning of tracks, work way up, event by event, iterating thru tracks, base spacing events tot
            # Try to follow a given story arc!  Maybe take input curve, try to follow it best fir
            currEvent = rd.choice(track.candidateEvents.candidateEvents)
            # TEMP:
            # TODO: figure this out in possible events, for now just excluding here
            # if not currEvent.isOrtho:
            #     searchRect = self.cartesian_bounding_box(self.orthoBoundingBox(currEvent.crowVector))
            #     intercPoints = self.possibleEvents.points_in_rectangle([t.coords for t in track.trackholes], searchRect)
            #     intercVects = self.possibleEvents.build_interception_test_vector_set([t.coords for t in track.trackholes],
            #                                                           intercPoints)
            #     holesHit = []
            #     if not self.possibleEvents.check_intersections({(currEvent.startHole.coords, currEvent.endHole.coords)},
            #                                                intercVects, currEvent.startHole.coords, currEvent.endHole.coords,
            #                                                holesHit,track.num):
            track.addTentativeEvent(currEvent)

            if currEvent.isShared:
                for link in currEvent.linkedEvents:
                    # NOTE: this may push us a little above the max.  Overdrive baybeeeee 8-P
                    self.board.getTrackByNum(link.trackNum).addTentativeEvent(link)

    def boundingBoxPlusVector(self, vector):
        """
        Create a bounding box around a vector with additional space.

        Args:
            vector: A tuple of two points defining the vector.

        Returns:
            A tuple containing the original vector and the four sides of the bounding box.

        Delegates to event_curve_math.bounding_box_plus_vector (this
        method's own job is just supplying the possibleEvents-derived
        ortho_dxdy that the pure function needs).
        """
        ortho_dxdy = self.possibleEvents.orthogonal_vector(vector[0], vector[1], self.config.eventminspacing / 2.0, False)
        return event_curve_math.bounding_box_plus_vector(vector, ortho_dxdy)

    def orthoBoundingBox(self, vector):
        """
        Create an orthogonal bounding box around a vector.

        Args:
            vector: A tuple of two points defining the vector.

        Returns:
            A tuple of four points defining the corners of the bounding box.

        Delegates to event_curve_math.ortho_bounding_box.
        """
        ortho_dxdy = self.possibleEvents.orthogonal_vector(vector[0], vector[1], self.config.eventminspacing / 2.0, False)
        return event_curve_math.ortho_bounding_box(vector, ortho_dxdy)

    def getNormLengthDistCurve(self):
        """
        Get the normalized length distribution curve.

        Returns:
            List of (x, y) points representing the normalized length distribution curve.
        """
        return self.getNormalizedIdealCurve(self.config.eventlengthdisthistcurvefile)

    def getNormLengthOverTimeCurve(self):
        """
        Get the normalized length over time curve.

        Returns:
            List of (x, y) points representing the length over time curve.
        """
        return self.getNormalizedIdealCurve(self.config.eventlengthovertimeidealcurve1file)

    def getEnergyCurve(self):
        """
        Get the energy curve for event placement.

        Returns:
            Tuple containing the energy curve and its normalized integral.
        """
        # Normalize the coordinates
        energyCurve = self.getNormalizedIdealCurve(self.config.eventenergyfile)

        # Det integral
        normalizer = sum([e[1] for e in energyCurve])
        energyNormIntegral = self.integrateAndNormalizeCurve(energyCurve, normalizer)

        return energyCurve, energyNormIntegral

    def getNormalizedIdealCurve(self, curveFile):
        """
        Load and normalize a curve from an SVG file. Delegates to
        event_curve_math.get_normalized_ideal_curve.
        """
        return event_curve_math.get_normalized_ideal_curve(curveFile)

    def integrateAndNormalizeCurve(self, curve, normalizer):
        """Integrate and normalize a curve. Delegates to event_curve_math."""
        return event_curve_math.integrate_and_normalize_curve(curve, normalizer)

    def normalizeCurveMagnitude(self, curve):
        """Normalize the magnitude of a curve. Delegates to event_curve_math."""
        return event_curve_math.normalize_curve_magnitude(curve)

    def actualizeCurve(self, curve, x_actualizer, y_actualizer, integrate=False):
        """Scale a curve by given x and y factors. Delegates to event_curve_math."""
        return event_curve_math.actualize_curve(curve, x_actualizer, y_actualizer, integrate)

    def discretizeCurve(self, curve, numBuckets, accumulate=False):
        """Convert a continuous curve into discrete buckets. Delegates to event_curve_math."""
        return event_curve_math.discretize_curve(curve, numBuckets, accumulate)

    def getPointsInProximity(self, searchRange, searchPoints, inputPoint):
        """Find points within a range of an input point. Delegates to event_curve_math."""
        return event_curve_math.get_points_in_proximity(searchRange, searchPoints, inputPoint)

    def tryGetDispAllowance(self, dispAllowances, proxPoint):
        """Look up a displacement allowance. Delegates to event_curve_math."""
        return event_curve_math.try_get_disp_allowance(dispAllowances, proxPoint)

    def getEffectorsForDisps(self, basePoint, searchDisps, posEffectors, eventNodes, events=None, selfScaleLength=-1):
        """
        Get effectors for specified displacements from a base point.
        
        Args:
            basePoint: Starting point for displacement calculations.
            searchDisps: List of displacements to search for.
            posEffectors: List of possible effectors.
            eventNodes: List of event node positions.
            events: Optional list of events for scaling.
            selfScaleLength: Length to use for self-scaling if events not provided.
            
        Returns:
            List of effector configurations for the given displacements.
        """
        effectors = []
        for p in searchDisps:
            idx = self.searchOrderedListForVal(eventNodes, basePoint + p)
            if idx > -1:
                allowances = self.tryGetDispAllowance(posEffectors, dict(disp=abs(p)))
                if events is None and selfScaleLength > -1:
                    effectors.append(dict(effect=allowances['effect'], scaledmod=allowances['mod'] * selfScaleLength,
                                          scaledenergymod=abs(allowances['mod'] * selfScaleLength)))
                elif events is not None and selfScaleLength == -1:
                    effectors.append(
                        dict(effect=allowances['effect'], scaledmod=allowances['mod'] * events[idx]['length'],
                             scaledenergymod=abs(allowances['mod'] * events[idx]['length'])))
                else:
                    raise Exception("Must pass either self scale length, or oth events for searching")

        return effectors

    def runPartialTrackEffLengthHoles(self, track_id, partialEventSet, trackActualLength, tentNewLadder=None,
                                      tentNewChute=None,
                                      overrideIters=-1, readMode=False):
        """
        Calculate the effective track length considering the current event set.
        
        Uses Markov chain forecasting to simulate game play and determine the effective
        length of the track when accounting for ladders and chutes.
        
        Args:
            track_id: Identifier for the track.
            partialEventSet: Current set of events on the track.
            trackActualLength: The physical length of the track.
            tentNewLadder: Optional tuple of (start, end) for a new ladder to test.
            tentNewChute: Optional tuple of (start, end) for a new chute to test.
            overrideIters: Override the default number of iterations for simulation.
            readMode: If True, read moves from benchmark data instead of generating them.
            
        Returns:
            Tuple of (effective_length, move_sequences) where effective_length is the
            calculated effective length of the track and move_sequences contains the
            sequences of moves made during simulation.
        """
        partialEventMappings = [dict(start=e.startHole.num, end=e.endHole.num)
                                for e in partialEventSet if e.instanceIsLadder]
        partialEventMappings.extend([dict(start=e.endHole.num, end=e.startHole.num)
                                     for e in partialEventSet if e.instanceIsChute])
        if tentNewLadder is not None:
            partialEventMappings.append(dict(start=tentNewLadder[0], end=tentNewLadder[1]))
        if tentNewChute is not None:
            partialEventMappings.append(dict(start=tentNewChute[0], end=tentNewChute[1]))
        partialEventMappings.sort(key=lambda e: e['start'])

        effHoleMap = []
        eIdx = 0
        # Extend track by 2 to incorporate start & finish holes
        effHoleMap.append(0)
        for idx in range(trackActualLength):
            if eIdx < len(partialEventMappings) and partialEventMappings[eIdx]['start'] == idx + 1:
                effHoleMap.append(partialEventMappings[eIdx]['end'])
                eIdx += 1
            else:
                effHoleMap.append(idx + 1)
        effHoleMap.append(trackActualLength + 1)

        # Figure out length of partial game
        movesAllTrials = 0
        if overrideIters < 0:
            iters = self.config.probminimodeliters
        else:
            iters = overrideIters

        if not readMode:
            sequencesOfMoves = []
            moveCounter = 0
            curSequence = []
            curReadSeq = []
            for trial in range(iters):
                # Set up trial gameplay
                startLoc = 0
                if not readMode:
                    if trial > 0: sequencesOfMoves.append(curSequence)
                    curSequence = []
                else:
                    moveCounter = 0
                    # startLoc = self.benchmarkMoves_df.index.get_loc((track_id, trial, 0))
                    curReadSeq = self.track_dict[track_id][trial]
                dealer = rd.randint(1, self.config.numplayers)
                curPos = 0
                countLoops = 0
                trackPosSeq = []
                while curPos < trackActualLength:
                    if not readMode:
                        # Run pegging:
                        pegRounds = rd.choices(self.pegRounds, weights=self.pegRoundProbs, k=1)[0]
                        for r in range(pegRounds):
                            curMove = rd.choices(self.posPegs, weights=self.posPegProbs, k=1)[0]
                            curSequence.append(curMove)
                            if curPos + curMove > len(effHoleMap):
                                curPos += curMove
                            else:
                                curPos = effHoleMap[curPos + curMove - 1]
                            movesAllTrials += 1
                            if curPos >= trackActualLength: break
                        if curPos >= trackActualLength: break

                        # Score hand
                        curMove = rd.choices(self.posHands, weights=self.posHandProbs, k=1)[0]
                        curSequence.append(curMove)
                        if curPos + curMove > len(effHoleMap):
                            curPos += curMove
                        else:
                            curPos = effHoleMap[curPos + curMove - 1]
                        movesAllTrials += 1
                        if curPos >= trackActualLength: break

                        if dealer == 1:
                            # Score crib
                            curMove = rd.choices(self.posHands, weights=self.posHandProbs, k=1)[0]
                            curSequence.append(curMove)
                            if curPos + curMove > len(effHoleMap):
                                curPos += curMove
                            else:
                                curPos = effHoleMap[curPos + curMove - 1]
                            movesAllTrials += 1
                            if curPos >= trackActualLength: break
                        dealer = 1 + dealer % self.config.numplayers
                    else:
                        # curMove = self.benchmarkMoves_df.iloc[startLoc + moveCounter]['MoveVal']
                        curMove = curReadSeq[moveCounter]
                        if moveCounter == 0:
                            countLoops += 1
                        moveCounter = (moveCounter + 1) % len(curReadSeq)
                        if curPos + curMove > len(effHoleMap):
                            curPos += curMove
                        else:
                            curPos = effHoleMap[curPos + curMove - 1]
                        trackPosSeq.append(curPos)
                        movesAllTrials += 1
                        if countLoops > 10:
                            # Track is stuck in an infinite loop!!! This event is no bueno
                            return 9999999, []
                        if curPos >= trackActualLength: break
            if not readMode: sequencesOfMoves.append(curSequence)

            # Forecast length of game based on control-case ideal moves:hole ratio
            actualPartialMoves = (movesAllTrials / iters)
            eventlessCtrlPartialMoves = (self.config.ideallikelihoodholehit * trackActualLength)
            shiftPct = actualPartialMoves / eventlessCtrlPartialMoves
            forecastedTrackEffLengthHoles = trackActualLength * shiftPct
            return forecastedTrackEffLengthHoles, sequencesOfMoves
        else:
            import markovgame as mg

            start_time = time.time()
            test = [len(t) for t in self.track_dict[track_id]]
            forecastedTrackEffLengthHoles = mg.runPartialTrackEffLengthHoles(trackActualLength, self.config.probminimodeliters,
                                                                             self.track_dict[track_id],
                                                                             [len(t) for t in
                                                                              self.track_dict[track_id]],
                                                                             effHoleMap,
                                                                             self.config.numplayers, self.config.ideallikelihoodholehit)
            end_time = time.time()
            self.miniMarkovTime += (end_time - start_time)
            return forecastedTrackEffLengthHoles, None

    def _scan_two_hits_for_direction(self, ref_hole_num, p_values,
                                     primary_positions, primary_items, primary_match_key,
                                     primary_guarded, primary_net_length_fn,
                                     secondary_positions, secondary_items, secondary_match_key,
                                     secondary_guarded, secondary_net_length_fn,
                                     event_length):
        """
        Scans one direction (forward or backward, `p_values` is (1, 2, 4)
        or (-1, -2, -4)) of scoreEventsForHole's two-hit detection for
        both event types (a "primary" position list/items -- e.g.
        ladders when checking a ladder instance -- and a "secondary" one
        -- e.g. chutes). Phase 8 step 3 extraction (see [[Refactor Mk
        ii]] in the Obsidian vault): this is the one shape duplicated
        four times in the original inline code (ladder-instance forward/
        backward, chute-instance forward/backward), each time checking a
        "same event type as the one being placed" list and an "opposite
        type" list.

        `primary_guarded`/`secondary_guarded` control whether
        `config.onlysamedirtwohits` and the "matched item's length is
        within 3 of this event's length" rule can reject the candidate
        outright for that position list's matches. TODO(liam): in the
        original inline code (verbatim preserved here, not changed),
        these guards were applied inconsistently depending on which
        *event type* was being matched, not on direction: matches
        against the *same* type as the instance being placed (ladder
        matches when placing a ladder, chute matches when placing a
        chute) were never guarded, while matches against the *opposite*
        type (chute matches when placing a ladder, and vice versa) always
        were, in both directions. Whether that's intentional design (the
        guard is really about mixed-type two-hits specifically) or a
        copy-paste asymmetry that should apply uniformly is unclear from
        the code/comments alone -- see
        tests/test_eventsetbuilder.py::test_scan_two_hits_for_direction_guards_only_the_flagged_side
        for the passing characterization of this exact behavior.

        For each `p` in `p_values`, checks `primary_positions` (an
        ordered list, searched via `self.searchOrderedListForVal`) for
        `ref_hole_num + p`; on a match, either counts it as "loose"
        (`abs(p) == 4`) or as a strict two-hit, then looks up the
        matching item in `primary_items` by `primary_match_key` to
        compute a net length delta via `primary_net_length_fn(event_length,
        matched_item_length)`. Then does the same for
        `secondary_positions`/`secondary_items`/`secondary_match_key`/
        `secondary_net_length_fn`. Matches original control flow exactly:
        a guarded rejection stops scanning further `p` values immediately
        (mirrors the original's `break` out of the `for p` loop) --
        anything already counted for earlier `p` values in this call is
        still returned, since the caller treats `invalid=True` as
        "reject this whole candidate" regardless.

        Returns:
            tuple: (num_two_hits, num_two_hits_loose, net_lengths, invalid)
        """
        num_two_hits = 0
        num_two_hits_loose = 0
        net_lengths = []
        for p in p_values:
            if self.searchOrderedListForVal(primary_positions, ref_hole_num + p) > -1:
                if abs(p) == 4:
                    num_two_hits_loose += 1
                else:
                    if primary_guarded and self.config.onlysamedirtwohits:
                        return num_two_hits, num_two_hits_loose, net_lengths, True
                    num_two_hits += 1
                    for item in primary_items:
                        if item[primary_match_key] == ref_hole_num + p:
                            if primary_guarded and abs(event_length - item['length']) < 3:
                                return num_two_hits, num_two_hits_loose, net_lengths, True
                            net_lengths.append(primary_net_length_fn(event_length, item['length']))
                            break
            if self.searchOrderedListForVal(secondary_positions, ref_hole_num + p) > -1:
                if abs(p) == 4:
                    num_two_hits_loose += 1
                else:
                    if secondary_guarded and self.config.onlysamedirtwohits:
                        return num_two_hits, num_two_hits_loose, net_lengths, True
                    num_two_hits += 1
                    for item in secondary_items:
                        if item[secondary_match_key] == ref_hole_num + p:
                            if secondary_guarded and abs(event_length - item['length']) < 3:
                                return num_two_hits, num_two_hits_loose, net_lengths, True
                            net_lengths.append(secondary_net_length_fn(event_length, item['length']))
                            break
        return num_two_hits, num_two_hits_loose, net_lengths, False

    def _score_candidate_instance(self, t, hole, candEventSpecs, instType,
                                  canBeChute, canBeChuteOnly, canBeLadder, canBeLadderOnly,
                                  chutes, chuteBases, chuteTops, ladders, ladderBases, ladderTops,
                                  params, explicitEvent, explicitChute, explicitLadder):
        """
        Scores one (candidate event, instance type) combination -- Phase 8
        step 4 extraction (see [[Refactor Mk ii]] in the Obsidian vault) of
        scoreEventsForHole's per-instance-type scoring body (the CHUTEONLY/
        LADDERONLY/CHUTEANDLADDER loop body: balance scoring, energy-buffer
        scoring, two-hit detection via _scan_two_hits_for_direction, cancel
        impedance, end-of-track weighting, length-histogram scoring, and
        length-over-time scoring). Pure code motion -- every `continue` in
        the original loop body becomes a `return None` here (meaning: skip
        this instType, same as the original continuing to the next one), and
        the original `eventFitnesses.append(dict(...))` becomes `return
        dict(...)`. The candidate-gating/cursor-walking loop above this in
        scoreEventsForHole is unchanged -- `candEventSpecs` is passed in
        already resolved, rather than this method reaching back into
        `t.candcursor` itself.

        Mutates `self.avgScoreSum`/`self.avgScoreDiv`/`self.avgScore` and
        `t.numnogos` exactly as the original inline code did (these are real
        instance/track state, not local to one call).

        Returns:
            dict | None: a fitness dict (same shape the original code
            appended to `eventFitnesses`) if this instType/candidate
            combination scored and passed every rejection check, else
            None (caller should not append anything and move on to the
            next instType).
        """
        effEnergy, effCompModulation = 0, 0
        effLengthForecast, partialTrackEnd = 0, 0
        # modsForType = []
        match instType:
            case en.InstanceEventType.CHUTEONLY:
                if not canBeChute: return None
                if not canBeChuteOnly: return None
                if (explicitEvent is None and not candEventSpecs['event'].isOrtho and
                        candEventSpecs['event'].crowLength < self.config.mincrowvectordistcancel):
                    return None
                if explicitEvent is not None and explicitLadder: return None
                effEnergy = candEventSpecs['length']
                # modsForType = allModsIfChute
                effLengthForecast = \
                self.runPartialTrackEffLengthHoles(t.track_id, t.eventsetbuild, t.tracklength,
                                                   tentNewChute=(candEventSpecs['event'].endHole.num,
                                                                 candEventSpecs['event'].startHole.num),
                                                   readMode=True)[0]
            case en.InstanceEventType.LADDERONLY:
                if not canBeLadder: return None
                if not canBeLadderOnly: return None
                if (explicitEvent is None and not candEventSpecs['event'].isOrtho and
                        candEventSpecs['event'].crowLength < self.config.mincrowvectordistcancel):
                    return None
                if explicitEvent is not None and explicitChute: return None
                effEnergy = candEventSpecs['length']
                # modsForType = allModsIfLadder
                effLengthForecast = \
                self.runPartialTrackEffLengthHoles(t.track_id, t.eventsetbuild, t.tracklength,
                                                   tentNewLadder=(candEventSpecs['event'].startHole.num,
                                                                  candEventSpecs['event'].endHole.num),
                                                   readMode=True)[0]
            case en.InstanceEventType.CHUTEANDLADDER:
                if not (canBeChute and canBeLadder): return None
                effEnergy = 2 * candEventSpecs['length']
                # modsForType = allModsIfChute + allModsIfLadder
                effLengthForecast = \
                self.runPartialTrackEffLengthHoles(t.track_id, t.eventsetbuild, t.tracklength,
                                                   tentNewLadder=(candEventSpecs['event'].startHole.num,
                                                                  candEventSpecs['event'].endHole.num),
                                                   tentNewChute=(candEventSpecs['event'].endHole.num,
                                                                 candEventSpecs['event'].startHole.num),
                                                   readMode=True)[0]

        # Infinite looping observed!!  Not a good set
        if effLengthForecast >= 9999999: return None
        # Adjust length as per control length
        effLengthForecast *= t.tracklength / t.controllength
        # NOTE: impeders are (-), boosters are (+)
        effCompModulation = effLengthForecast - t.curestefflength
        # print(str(effCompModulation))

        # BASE SCORE ON BLEND MOD + ENERGY

        # NOTE: longer balanceandefflengthcontrolfactor for longer route
        balFactor = params.tryGetParam(t.track_id, 'balanceandefflengthcontrolfactor')
        lengtheningControl, shorteningControl = balFactor, 1.0 - balFactor
        curEstLengthDiscr = t.curestefflength - self.config.effectiveboardlength
        instEstLengthDiscr = effLengthForecast - self.config.effectiveboardlength
        if abs(curEstLengthDiscr) > 10:
            sdfd = ""
        instEstLengthDisp = effLengthForecast - t.curestefflength
        # Too much instability!  Nix this uber event
        if instEstLengthDisp > self.config.maxefflengthdisp: return None

        curScore = 1.0  # Base amt
        if curEstLengthDiscr != 0:
            # If board is perfect, leave it alone!  Highly unlikely tho except for inital run
            balScoreMod = abs(instEstLengthDiscr) / abs(curEstLengthDiscr)
            # If we are moving in correct direction, reward
            reward = abs(instEstLengthDiscr) < abs(curEstLengthDiscr)
            if curEstLengthDiscr > 0:
                # Apply shortening control, board is too long
                if reward:
                    # curScore = balScoreMod/self.config.gamelengthtightness
                    curScore = balScoreMod * math.pow((1.0 - shorteningControl), self.config.gamelengthtightness)
                else:
                    # curScore = balScoreMod*self.config.gamelengthtightness
                    curScore = balScoreMod * math.pow((1.0 + shorteningControl), self.config.gamelengthtightness)
            elif curEstLengthDiscr < 0:
                # Apply lengthening control, board is too short
                if reward:
                    # curScore = balScoreMod/self.config.gamelengthtightness
                    curScore = balScoreMod * math.pow((1.0 - lengtheningControl), self.config.gamelengthtightness)
                else:
                    # curScore = balScoreMod*self.config.gamelengthtightness
                    curScore = balScoreMod * math.pow((1.0 + lengtheningControl), self.config.gamelengthtightness)

        if abs(effEnergy) + abs(t.energybuffer) > 0:
            curScore *= (1.0 + (params.tryGetParam(t.track_id, 'energybufferenforcement')
                                * abs(effEnergy - t.energybuffer) / (
                                            abs(effEnergy) + abs(t.energybuffer))))
        effNetEnergy = effEnergy + abs(effCompModulation)

        # #NOTE: longer balanceandefflengthcontrolfactor for longer route
        # balFactor = params.tryGetParam(t.track_id, 'balanceandefflengthcontrolfactor')
        # #TODO: re-enable this??
        # # if instType == en.InstanceEventType.CHUTEONLY and balFactor > 0.5:
        # #     curScore *= (1.0 - balFactor)/0.2
        # # elif instType == en.InstanceEventType.LADDERONLY and balFactor < 0.5:
        # #     curScore *= balFactor/0.2
        # # elif instType == en.InstanceEventType.LADDERONLY and balFactor > 0.5:
        # #     curScore /= (1.0 - balFactor)/0.2
        # # elif instType == en.InstanceEventType.CHUTEONLY and balFactor < 0.5:
        # #     curScore /= balFactor/0.2
        # curEstLengthDiscr = t.curestefflength - self.config.effectiveboardlength
        # instEstLengthDiscr = effLengthForecast - self.config.effectiveboardlength
        # if abs(curEstLengthDiscr) > 10:
        #     sdfd=""
        # instEstLengthDisp = effLengthForecast - t.curestefflength
        # #Too much instability!  Nix this uber event
        # if instEstLengthDisp > self.config.maxefflengthdisp: continue
        #
        # if curEstLengthDiscr != 0:
        #     #If board is perfect, leave it alone!  Highly unlikely tho except for inital run
        #     balScoreMod = abs(instEstLengthDiscr)/abs(curEstLengthDiscr)
        #     #If we are moving in correct direction, reward
        #     reward = abs(instEstLengthDiscr) < abs(curEstLengthDiscr)
        #     lengtheningControl, shorteningControl = balFactor, 1.0 - balFactor
        #     if curEstLengthDiscr > 0:
        #         #Apply shortening control, board is too long
        #         if reward: curScore *= balScoreMod*(1.0 - shorteningControl)
        #         else: curScore *= balScoreMod*(1.0 + shorteningControl)
        #     elif curEstLengthDiscr > 0:
        #         #Apply lengthening control, board is too short
        #         if reward: curScore *= balScoreMod*(1.0 - lengtheningControl)
        #         else: curScore *= balScoreMod*(1.0 + lengtheningControl)

        # Check for two-hits
        numTwoHits = 0
        numTwoHitsLoose = 0
        twoHitNetLengths = []
        twoHitInvalid = False

        # def getTwoHitNetLength(p, chutesOrLadders, searchHoleNum, foundHoleType, eventLength):
        #     for l in chutesOrLadders:
        #         if l[foundHoleType] == searchHoleNum + p:
        #             return l['length'] + eventLength
        #     return 0

        if instType in (en.InstanceEventType.LADDERONLY, en.InstanceEventType.CHUTEANDLADDER):
            # Forward: same-type (ladder) matches unguarded, opposite-type
            # (chute) matches guarded -- see _scan_two_hits_for_direction's
            # docstring for the TODO on this asymmetry.
            deltaHits, deltaLoose, deltaLengths, invalid = self._scan_two_hits_for_direction(
                candEventSpecs['event'].endHole.num, (1, 2, 4),
                ladderBases, ladders, 'ladderbase', False, lambda el, ll: ll + el,
                chuteTops, chutes, 'chutetop', True, lambda el, cl: el - cl,
                candEventSpecs['event'].length)
            numTwoHits += deltaHits
            numTwoHitsLoose += deltaLoose
            twoHitNetLengths.extend(deltaLengths)
            twoHitInvalid = twoHitInvalid or invalid
            if twoHitInvalid: return None
            deltaHits, deltaLoose, deltaLengths, invalid = self._scan_two_hits_for_direction(
                candEventSpecs['event'].startHole.num, (-1, -2, -4),
                ladderTops, ladders, 'laddertop', False, lambda el, ll: el + ll,
                chuteBases, chutes, 'chutebase', True, lambda el, cl: el - cl,
                candEventSpecs['event'].length)
            numTwoHits += deltaHits
            numTwoHitsLoose += deltaLoose
            twoHitNetLengths.extend(deltaLengths)
            twoHitInvalid = twoHitInvalid or invalid
            if twoHitInvalid: return None

        if instType in (en.InstanceEventType.CHUTEONLY, en.InstanceEventType.CHUTEANDLADDER):
            # Forward: opposite-type (ladder) matches guarded, same-type
            # (chute) matches unguarded -- mirror image of the LADDERONLY
            # branch above; see _scan_two_hits_for_direction's docstring.
            deltaHits, deltaLoose, deltaLengths, invalid = self._scan_two_hits_for_direction(
                candEventSpecs['event'].startHole.num, (1, 2, 4),
                ladderBases, ladders, 'ladderbase', True, lambda el, ll: ll - el,
                chuteTops, chutes, 'chutetop', False, lambda el, cl: (-1) * cl - el,
                candEventSpecs['event'].length)
            numTwoHits += deltaHits
            numTwoHitsLoose += deltaLoose
            twoHitNetLengths.extend(deltaLengths)
            twoHitInvalid = twoHitInvalid or invalid
            if twoHitInvalid: return None

            deltaHits, deltaLoose, deltaLengths, invalid = self._scan_two_hits_for_direction(
                candEventSpecs['event'].endHole.num, (-1, -2, -4),
                ladderTops, ladders, 'laddertop', True, lambda el, ll: ll - el,
                chuteBases, chutes, 'chutebase', False, lambda el, cl: (-1) * cl - el,
                candEventSpecs['event'].length)
            numTwoHits += deltaHits
            numTwoHitsLoose += deltaLoose
            twoHitNetLengths.extend(deltaLengths)
            twoHitInvalid = twoHitInvalid or invalid
            if twoHitInvalid: return None

        if len(twoHitNetLengths) > 0 and (min(twoHitNetLengths) < (-1) * self.config.maxtwohitnetgainloss or
                                          max(twoHitNetLengths) > self.config.maxtwohitnetgainloss):
            return None

        if (numTwoHits > 0 and numTwoHits * params.tryGetParam(t.track_id, 'twohitfreqimpedance') >
                (self.config.allowabletwohits - t.twohitsthusfar)):
            return None

        curScore *= (1.0 + (numTwoHits + numTwoHitsLoose / 2) * t.twohitsthusfar *
                     params.tryGetParam(t.track_id, 'twohitfreqimpedance'))

        # Impede score if too many ladders/chutes are getting cancelled
        if instType != en.InstanceEventType.CHUTEANDLADDER and t.cancels >= self.config.whenstartworryingaboutcancels:
            if t.cancels >= 2.5 * self.config.whenstartworryingaboutcancels: return None
            curScore *= (1.0 + params.tryGetParam(t.track_id, 'cancelimpedance') * (t.cancels + 1)
                         / (t.eventscount + 1))

        # Preferentially weight based on proximity to end of track
        endTrackWeight = params.tryGetParam(t.track_id, 'eventstowardsendoftrackreward')
        eventPosRelMidpoints = candEventSpecs['event'].midPointNum / t.tracklength - 0.5
        if eventPosRelMidpoints < 0:
            curScore *= (1.0 + abs(eventPosRelMidpoints) * endTrackWeight)
        else:
            curScore /= (1.0 + abs(eventPosRelMidpoints) * endTrackWeight)

        # Factor in distribution of length histogram to help ensure distributed lengths
        # Try to curve fit specified ideal histo
        # NOTE: golf-stylee, lower score is better
        curLenPerc = 0.0
        curLength = candEventSpecs['length']
        if sum([h[1] for h in self.allTentLengthHisto]) > 0:
            curLenPerc = (self.allTentLengthHisto[curLength - 1][1] /
                          sum([h[1] for h in self.allTentLengthHisto]))
        idealPerc = t.lengthdistidealcurve[curLength - 1][1]
        lenDistDisp = curLenPerc - idealPerc
        if lenDistDisp < 0:
            # Need more!
            curScore /= (1.0 + abs(lenDistDisp)) * params.tryGetParam(t.track_id,
                                                                      'lengthhistogramscoringfactor')
        elif lenDistDisp > 0:
            # Too many of this length already, downshift
            curScore *= (1.0 + abs(lenDistDisp)) * params.tryGetParam(t.track_id,
                                                                      'lengthhistogramscoringfactor')

        # Factor in distribution of length over time
        # TEMP!!
        if len(t.lengthovertimeideal) < hole.num:
            print("FAILED LENGTH OVER TIME TEST: hole.num {}".format(hole.num))
        else:
            idealLengthForHole = t.lengthovertimeideal[hole.num - 1][1]
            scoreMod = (1.0 + (abs(curLength - idealLengthForHole) / t.maxlength) *
                        params.tryGetParam(t.track_id, 'lengthovertimescoringfactor'))
            if curScore >= 0:
                curScore *= scoreMod
            else:
                curScore /= scoreMod

        # Aggr into avg score
        self.avgScoreSum += curScore
        self.avgScoreDiv += 1
        self.avgScore = self.avgScoreSum / self.avgScoreDiv

        # Elminate options based on shortening & lengthening control
        if (curEstLengthDiscr > 0 and shorteningControl > 0.5 and curScore > self.avgScore
                * self.config.goodscorecutoffperc * 2 * (
                        1.0 - (2 * (shorteningControl - 0.5)))):
            t.numnogos += 1
            return None
        elif (curEstLengthDiscr < 0 and lengtheningControl > 0.5 and curScore > self.avgScore
              * self.config.goodscorecutoffperc * 2 * (
                      1.0 - (2 * (lengtheningControl - 0.5)))):
            t.numnogos += 1
            return None

        return dict(event=candEventSpecs['event'],
                                   eventspecs=candEventSpecs,
                                   score=curScore, effnetenergy=effNetEnergy,
                                   effcompmodulation=effCompModulation,
                                   insttype=instType,
                                   instchute=instType in (en.InstanceEventType.CHUTEANDLADDER,
                                                          en.InstanceEventType.CHUTEONLY),
                                   instladder=instType in (en.InstanceEventType.CHUTEANDLADDER,
                                                           en.InstanceEventType.LADDERONLY),
                                   lasteventtop=0
                                   ,
                                   twohits=numTwoHits, estefflength=effLengthForecast
                                   )

    def scoreEventsForHole(self, t, hole,
                           chutes, chuteBases, chuteTops, ladders, ladderBases, ladderTops, params, trackEventsOverview,
                           explicitEvent=None, explicitChute=False, explicitLadder=False):
        """
        Calculate a score for placing an event at a specific hole.

        Evaluates the impact of placing an event (ladder or chute) at the given hole
        based on various factors including proximity to other events and game balance.

        Per-candidate cursor-walking/gating happens here; the actual
        per-instance-type scoring (balance/energy-buffer/two-hit/cancel-
        impedance/length-histogram/length-over-time) is delegated to
        `_score_candidate_instance` (Phase 8 step 4 -- see [[Refactor Mk
        ii]] in the Obsidian vault).

        Args:
            t: `TrackBuildState` instance containing this track's working
                state (Phase 8 step 2 turned this from a plain dict into
                a real class -- see `TrackBuildState`'s docstring).
            hole: The hole being evaluated for event placement.
            chutes: List of existing chutes on the track.
            chuteBases: List of base positions of chutes.
            chuteTops: List of top positions of chutes.
            ladders: List of existing ladders on the track.
            ladderBases: List of base positions of ladders.
            ladderTops: List of top positions of ladders.
            params: Parameter set for scoring calculations.
            trackEventsOverview: Overview of events on the track.
            explicitEvent: Optional explicit event to evaluate.
            explicitChute: If True, evaluate as a chute.
            explicitLadder: If True, evaluate as a ladder.

        Returns:
            List of dictionaries containing scoring information for potential events.
        """
        start_time = time.time()
        if explicitEvent is None and hole.num < t.optfirstchute: return []

        # Passed gauntlet!  Let's try to find an event to deplete this energy
        eventFitnesses = []
        explicitEventCounter = 0
        while ((explicitEvent is not None and explicitEventCounter < 1) or
               (explicitEvent is None and t.candcursor < len(t.candeventspecs) and
                t.candeventspecs[t.candcursor]['eventtop'] == hole.num)):
            explicitEventCounter += 1
            # Scoring system: base score is scalar displacement between energy amount and energy buffer
            # Disallow (do not include) if no according bypass rule
            # and base falls within range of 2-hit from another event
            # Sqrt value for each appropriate 2-hit eligible
            if explicitEvent is None:
                candEventSpecs = t.candeventspecs[t.candcursor]
            else:
                candEventSpecs = explicitEvent

            if explicitEvent is None and candEventSpecs['isshared']:
                # if one or more tracks are locked, no multis!
                # TODO: get multis working w/ elim mode
                # if t.nomultis:
                t.candcursor += 1
                continue

                # Check if linked event is legal
                assertLegal = True
                for ev in candEventSpecs['event'].linkedEvents:
                    linkedStart = ev.startHole.num
                    linkedEnd = ev.endHole.num
                    linkedTrackNum = ev.trackNum
                    t_sub = None
                    for t_match in trackEventsOverview:
                        if t_match.tracknum == linkedTrackNum:
                            t_sub = t_match
                            break
                    for n in [linkedStart, linkedStart]:
                        idx = bsc.bisect_left(t_sub.eventnodes, n)
                        if idx < len(t_sub.eventnodes) and t_sub.eventnodes[idx] == n:
                            assertLegal = False
                            break
                    if not assertLegal: break
                if not assertLegal:
                    # Cannot have multiple events landing or starting on same space!
                    t.candcursor += 1
                    continue

            if (explicitEvent is None and
                    (self.searchOrderedListForVal(t.eventnodes, candEventSpecs['eventbase']) > -1 or
                     self.searchOrderedListForVal(t.eventnodes, candEventSpecs['eventtop']) > -1)):
                # Cannot have multiple events landing or starting on same space!
                t.candcursor += 1
                continue

            # If hella override, check it
            if (explicitEvent is None and
                    (params.tryGetParam(t.track_id, 'disallowbelowsetlength', optional=True) > 0 and
                     candEventSpecs['length'] < params.tryGetParam(t.track_id, 'disallowbelowsetlength'))):
                t.candcursor += 1
                continue

            # Check if ortho ratio is exceeded
            if (explicitEvent is None and
                    (candEventSpecs['event'].isOrtho and len(t.eventsetbuild) > 0 and
                     len([e for e in t.eventsetbuild if e.isOrtho]) / len(t.eventsetbuild) >
                     params.tryGetParam(t.track_id, 'maxorthoratio'))):
                t.candcursor += 1
                continue

            if explicitEvent is None:
                canBeChute = True
                canBeLadder = candEventSpecs['canbeladder']
            else:
                canBeChute = explicitChute
                # MIGHT BE TOO DRACONIAN to force??
                # canBeLadder = candEventSpecs['canbeladder'] and explicitLadder
                canBeLadder = explicitLadder

            canBeLadderOnly, canBeChuteOnly = canBeLadder, canBeChute
            # Check if can be chute only
            if explicitEvent is None and (canBeChute and len(ladderBases) / (len(chuteBases) + 1)
                                          < params.tryGetParam(t.track_id, "minladdertochuteratio")):
                canBeChuteOnly = False
                if not canBeLadder:
                    t.candcursor += 1
                    continue

            # Check if can be ladder only
            if explicitEvent is None and (canBeLadder and len(chuteBases) / (len(ladderBases) + 1)
                                          < params.tryGetParam(t.track_id, "minchutetoladderratio")):
                canBeLadderOnly = False
                if not canBeChute:
                    t.candcursor += 1
                    continue

            # Insert event score as chute, ladder, and both
            for instType in (en.InstanceEventType.CHUTEONLY, en.InstanceEventType.LADDERONLY,
                             en.InstanceEventType.CHUTEANDLADDER):
                fitness = self._score_candidate_instance(
                    t, hole, candEventSpecs, instType,
                    canBeChute, canBeChuteOnly, canBeLadder, canBeLadderOnly,
                    chutes, chuteBases, chuteTops, ladders, ladderBases, ladderTops,
                    params, explicitEvent, explicitChute, explicitLadder)
                if fitness is not None:
                    eventFitnesses.append(fitness)

            t.candcursor += 1

        eventFitnesses.sort(key=lambda f: f['score'])
        end_time = time.time()
        self.scoringTime += end_time - start_time
        if ((len(eventFitnesses) > 0 and eventFitnesses[0]['score'] <= self.avgScore * self.config.goodscorecutoffperc * 2)
                or explicitEvent is not None):
            return eventFitnesses
        if len(eventFitnesses) > 0: t.numdenies += 1
        return None

    def searchOrderedListForVal(self, orderedList, val):
        """
        Search for a value in a sorted list using binary search.
        Delegates to event_curve_math.search_ordered_list_for_val.
        """
        return event_curve_math.search_ordered_list_for_val(orderedList, val)

    def _derive_instance_geometry(self, event, isOrtho):
        """
        Compute an event instance's placement geometry -- instanceStartVector/
        instanceEndVector (orthogonal events only, via OrthoLineTrace) and
        instanceLump (either kind, when instanceIsChute != instanceIsLadder,
        via self.possibleEvents.calculate_distance) -- so it can be handed to
        a VectorCollisionTracker's commit().

        Refactor Mk II Phase 9a (see [[Refactor Mk ii]]/Phase 9 Findings in
        the Obsidian vault): geometry-derivation half of the former
        updateVectorsTest, split out verbatim (same computation, same
        mutation of `event` in place) from the pure collision-set
        bookkeeping half, which is now VectorCollisionTracker.commit.

        Args:
            event: The event instance to derive geometry for. Mutated in place.
            isOrtho: Boolean indicating if the event is orthogonal.
        """
        if not isOrtho:
            # Add lumps 20% of the way along so ppl know cant go that way
            if event.instanceIsChute != event.instanceIsLadder:
                start, end = np.array(event.crowVector[0]), np.array(event.crowVector[1])
                dist = self.possibleEvents.calculate_distance(event.crowVector[0], event.crowVector[1])
                if event.instanceIsChute:
                    event.instanceLump = (start + (end - start) * ((3 / dist) + math.pow(dist, 0.25) / 50)).tolist()
                else:
                    event.instanceLump = (end + (start - end) * ((3 / dist) + math.pow(dist, 0.25) / 50)).tolist()
        else:
            event.instanceStartVector = OrthoLineTrace(self.possibleEvents, event, event.instanceIncr,
                                                       event.instanceRev,
                                                       en.OrthoLineTraceType.START).vector
            event.instanceEndVector = OrthoLineTrace(self.possibleEvents, event, event.instanceIncr,
                                                     event.instanceRev,
                                                     en.OrthoLineTraceType.END).vector

            # Add lumps 20% of the way along so ppl know cant go that way
            if event.instanceIsChute != event.instanceIsLadder:
                vector = event.instanceStartVector if event.instanceIsChute else event.instanceEndVector
                start, end = np.array(vector[0]), np.array(vector[1])
                dist = self.possibleEvents.calculate_distance(vector[0], vector[1])
                event.instanceLump = (start + (end - start) * ((3 / dist) + math.pow(dist, 0.25) / 50)).tolist()

    def tryGetEventForHole(self, hole, t, tracker, params, trackEventsOverview):
        """
        Attempts to find and return the most suitable event for a given hole on a track.

        This method evaluates potential events for a specific hole based on various factors including
        energy buffer levels, event spacing, and distribution patterns. It ensures that events are
        placed in a way that maintains game balance and follows design constraints.

        Args:
            hole: The hole object to find an event for.
            t: Dictionary containing track-specific data including energy buffer and event candidates.
            tracker: VectorCollisionTracker used to test candidate events for
                collisions against already-placed event vectors (Phase 9b;
                replaces the former separate interceptsTestVectors/
                baseVectorsTest set params -- baseVectorsTest was never
                actually referenced by the legality check).
            params: Parameter set containing configuration values for event selection.
            trackEventsOverview: Overview of all track events for reference and scoring.

        Returns:
            dict: A dictionary containing the selected event and its fitness score, or None if no
                  suitable event is found. The dictionary includes:
                  - 'event': The selected event object
                  - 'score': The fitness score of the event
                  - 'lasteventtop': The position of the last event's top
                  - Other event-specific metadata
        """
        while (t.energybufferidx < len(t.trackenergycurve) and
               t.trackenergycurve[t.energybufferidx][0] < hole.num):
            t.energybuffer += t.trackenergycurve[t.energybufferidx][1]
            t.energybufferidx += 1
        if t.energybuffer < t.candavgenergy / params.tryGetParam(t.track_id,
                                                                       'candenergybufferdivider'): return None

        # Cursor to start of trackhole in event list
        while (t.candcursor < len(t.candeventspecs)
               and t.candeventspecs[t.candcursor]['eventtop'] < hole.num):
            t.candcursor += 1
        if (t.candcursor >= len(t.candeventspecs) or
                t.candeventspecs[t.candcursor]['eventtop'] != hole.num): return None

        # Omit every 8th or so, feathering to avoid getting stuck in optimizer endless loops
        if rd.randint(1, self.config.randomfeatheringamount) == 1: return None

        # Skip if need to enforce min spacing to flesh out track
        if t.minspacectr < params.tryGetParam(t.track_id, 'enforceminspacing'): return None

        # Factor in distribution of spacing histogram to help ensure even distribution of spacings
        prevNode = 0
        if len(t.eventnodes) > 0:
            for n in t.eventnodes:
                if n >= hole.num: break
                prevNode = n
        spacing = hole.num - prevNode
        if spacing > len(t.spacinghisto):
            for i in range(len(t.spacinghisto) - 1, spacing):
                # Add in further spacing histos
                t.spacinghisto.append([i + 1, 0])

        if sum([h[1] for h in t.spacinghisto]) > 0:
            curSpcPerc = (t.spacinghisto[spacing - 1][1] /
                          sum([h[1] for h in t.spacinghisto]))
            # NOTE: for spacing we use all spacings for avg even unpopulated one
            # This is in order to factor in specified ideal deviation
            avgPerc = 1.0 / len(t.spacinghisto)
            if curSpcPerc - avgPerc > params.tryGetParam(t.track_id,
                                                         'eventspacinghistogramscoringfactor'): return None

        # Determine viable event fitnesses
        eventFitnesses = self.scoreEventsForHole(t, hole, t.chutes, t.chutebases, t.chutetops,
                                                 t.ladders, t.ladderbases, t.laddertops, params,
                                                 trackEventsOverview)

        # Find fittest event
        if eventFitnesses is not None and len(eventFitnesses) > 0:
            for fitness in eventFitnesses:
                legal, orthoInst = tracker.would_collide(fitness['event'], t)
                if legal:
                    if orthoInst['incr'] > -1:
                        fitness['event'].instanceIncr = orthoInst['incr']
                        fitness['event'].instanceRev = orthoInst['rev']
                    fitness['lasteventtop'] = prevNode
                    # Score cutoff
                    # if fitness['score'] < t['chosenscorecutoff']:
                    return fitness
                    # else: return None

        return None

    def indexStartOfEachHoleInCands(self, holes, trackEventOverview):
        """
        Creates a lookup table mapping hole numbers to their starting indices in the candidate events list.
        Delegates to event_curve_math.index_start_of_each_hole_in_cands
        (which, like this method, mutates trackEventOverview in place).
        """
        event_curve_math.index_start_of_each_hole_in_cands(holes, trackEventOverview)

    def recalcTrackCompletionPcts(self, trackEventsOverview):
        """
        Recalculates completion percentages and stall status for all tracks.
        Delegates to event_curve_math.recalc_track_completion_pcts,
        passing self.config.maxitertrackstalled as the stall threshold.
        """
        return event_curve_math.recalc_track_completion_pcts(trackEventsOverview, self.config.maxitertrackstalled)

    def tryEventSet(self, params, prevEffLengths):
        """
        Attempts to generate a complete set of events for all tracks based on the given parameters.

        This is the main method that coordinates the event generation process across all tracks.
        It handles the core logic of placing events while respecting constraints and maintaining
        game balance.

        Args:
            params: Parameter set containing configuration values for event generation.
            prevEffLengths: List of effective lengths from previous generation attempts.

        Returns:
            bool: True if a valid event set was generated, False otherwise.

        Note:
            This method modifies the board state in-place by adding events to tracks.
            If generation fails, it may adjust parameters and retry automatically.

        TODO(liam): flagged by
        tests/test_eventsetbuilder.py::TestEventSetBuilder::test_try_event_set
        (skipped, not asserted). This method immediately dereferences
        `t.candidateEvents.candidateEvents` for every track in
        `self.board.tracks` -- it requires each track to already have a
        real `CandidateEvents` instance (built by
        `PossibleEvents.buildSet`) populated with actual `CandidateEvent`
        objects. Nothing in this method's signature or docstring
        documents that precondition; a track with `candidateEvents is
        None` (e.g. a bare `Track()` built for a test, or a board loaded
        without running `PossibleEvents` first) will crash here rather
        than failing with a clear error. Worth either asserting the
        precondition explicitly or documenting it, and building a real
        fixture for the skipped test above.
        """

        start_time = time.time()
        self.board.clearTrackEvents(specificTracks=[t for t in self.board.tracks if not t.instLocked])
        trackEventsOverview = self._build_track_state(params)

        # Initial pass, try to populate tracks in tandem
        avgHolePct, avgChutesPct = 0.0, 0.0
        tracker = VectorCollisionTracker(self.possibleEvents, self.config)
        allTentative, allDirectTentative, allOrthoTentative = [], [], []
        stallCounter = 0
        while (len([t for t in trackEventsOverview if t.holescompletepct <
                                                      1.0 * params.tryGetParam(t.track_id,
                                                                               'holescompletetrackallowablecutoff')]) > 0 and
               len([t for t in trackEventsOverview if t.chutescompletepct <
                                                      1.0 * params.tryGetParam(t.track_id,
                                                                               'maxchuteoverdrivepct')]) > 0 and
               stallCounter <= self.config.maxitertrynewbuild):
            # if len([t for t in trackEventsOverview if t.holescompletepct < 0.9]) == 0:
            #     sfds=""
            if avgHolePct > 0.5:
                #     test = [t for t in trackEventsOverview if t.holescompletepct < 1.0*params.maxchuteoverdrivepct]
                sdffds = ""
            allTracksStalled = True
            for t in trackEventsOverview:
                # NOTE: factoring for roundoff error w/ chutes pct
                t.trackstalledcounter += 1
                while t.chutescompletepct <= avgChutesPct + 0.001 and t.curhole < len(t.track.trackholes):
                    allTracksStalled = False
                    idealEventWithFitness = None
                    isSharePop = False
                    if len(t.multistack) == 0:
                        # Find new event
                        t.curhole += 1
                        t.minspacectr += 1
                        curHoleObj = t.track.getHoleByNum(t.curhole)
                        idealEventWithFitness = self.tryGetEventForHole(curHoleObj, t, tracker,
                                                                        params, trackEventsOverview)
                    else:
                        # Pop queued multi event
                        idealEventWithFitness = t.multistack.pop()
                        isSharePop = True
                        prevNode = 0
                        if len(t.eventnodes) > 0:
                            for n in t.eventnodes:
                                if n >= idealEventWithFitness['eventspecs']['eventtop']: break
                                prevNode = n
                        idealEventWithFitness['lasteventtop'] = prevNode
                        spacing = idealEventWithFitness['eventspecs']['eventtop'] - prevNode
                        if spacing > len(t.spacinghisto):
                            for i in range(len(t.spacinghisto) - 1, spacing):
                                # Add in further spacing histos
                                t.spacinghisto.append([i + 1, 0])

                    if idealEventWithFitness is not None:
                        t.trackstalledcounter = 0
                        t.trackisstalled = False
                        t.minspacectr = 0
                        # t['chosenscorecutoff'] = 0.3*idealEventWithFitness['score']*10.0 + 0.7*t['chosenscorecutoff']
                        # if len(allVectorsTest) > 50 and len(allVectorsTest) % 100 <= 5:
                        # if len(allVectorsTest) >  155:
                        #     sdf=""
                        #     self.testPlotVectorsOnHoles(allVectorsTest)

                        # Great success!  Add event & update sets
                        t.track.addTentativeEvent(idealEventWithFitness['event'])
                        self.allTentLengthHisto[idealEventWithFitness['eventspecs']['length'] - 1][1] += 1
                        t.lengthdistactualhist[idealEventWithFitness['eventspecs']['length'] - 1][1] += 1
                        # NOTE: we subtract energy, but add in modulation
                        t.energybuffer -= idealEventWithFitness['effnetenergy']
                        # NOTE: we SUBTRACT, since boosters are (+) (decrease eff board length)
                        # ...and impeders are (-) (increase eff board length)
                        t.compensationbuffer += idealEventWithFitness['effcompmodulation']
                        curEvent = idealEventWithFitness['event']
                        isOrtho = curEvent.isOrtho
                        allTentative.append(curEvent)
                        if not isOrtho:
                            allDirectTentative.append(curEvent)
                        else:
                            allOrthoTentative.append(curEvent)
                        curEvent.instanceIsChute = idealEventWithFitness['instchute']
                        curEvent.instanceIsLadder = idealEventWithFitness['instladder']
                        curEvent.instanceCancel = curEvent.instanceIsChute != curEvent.instanceIsLadder
                        if curEvent.instanceCancel: t.cancels += 1
                        t.eventscount += 1

                        self._derive_instance_geometry(curEvent, isOrtho)
                        tracker.commit(curEvent, isOrtho)
                        t.twohitsthusfar += idealEventWithFitness['twohits']
                        t.curestefflength = idealEventWithFitness['estefflength']
                        if idealEventWithFitness['instchute']:
                            t.chutes.append(dict(chutetop=idealEventWithFitness['eventspecs']['eventtop'],
                                                    chutebase=idealEventWithFitness['eventspecs']['eventbase'],
                                                    length=idealEventWithFitness['eventspecs']['length']))
                            t.chutes.sort(key=lambda l: l['chutetop'])
                            t.chutebases.append(idealEventWithFitness['eventspecs']['eventbase'])
                            t.chutebases.sort()
                            t.chutetops.append(idealEventWithFitness['eventspecs']['eventtop'])
                            t.chutetops.sort()
                        t.eventnodes.extend([idealEventWithFitness['eventspecs']['eventbase'],
                                                idealEventWithFitness['eventspecs']['eventtop']])
                        t.eventnodes.sort()
                        if idealEventWithFitness['instladder']:
                            t.ladders.append(dict(ladderbase=idealEventWithFitness['eventspecs']['eventbase'],
                                                     laddertop=idealEventWithFitness['eventspecs']['eventtop'],
                                                     length=idealEventWithFitness['eventspecs']['length']))
                            t.ladders.sort(key=lambda l: l['ladderbase'])
                            t.ladderbases.append(idealEventWithFitness['eventspecs']['eventbase'])
                            t.ladderbases.sort()
                            t.laddertops.append(idealEventWithFitness['eventspecs']['eventtop'])
                            t.laddertops.sort()
                        t.previsladder = idealEventWithFitness['instladder']
                        t.spacinghisto[(idealEventWithFitness['eventspecs']['eventtop']
                                           - idealEventWithFitness['lasteventtop']) - 1][1] += 1
                        t.lasteventtop = idealEventWithFitness['eventspecs']['eventtop']

                        if curEvent.instanceIsChute != curEvent.instanceIsLadder: self.cancels += 1
                        if curEvent.isOrtho: self.orthos += 1
                        if curEvent.isShared: self.multis += 1
                        self.events += 1

                        if not isSharePop and curEvent.isShared:
                            for ev in curEvent.linkedEvents:
                                t_sub = next((t_sub for t_sub in trackEventsOverview
                                              if t_sub.tracknum == ev.trackNum), None)
                                if t_sub is None:
                                    raise Exception("Multi event not linked up to track! 0_o")
                                linkedEventSpecs = next((specs for specs in t_sub.candeventspecs
                                                         if specs['event'] == ev), None)
                                if linkedEventSpecs is None:
                                    raise Exception("Candidate event specs not found for event 0_o")
                                topHole = curEvent.endHole
                                linkedEventWithScore = self.scoreEventsForHole(t_sub, topHole, t_sub.chutes,
                                                                               t_sub.chutebases, t_sub.chutetops,
                                                                               t_sub.ladders, t_sub.ladderbases,
                                                                               t_sub.laddertops, params,
                                                                               trackEventsOverview,
                                                                               linkedEventSpecs,
                                                                               idealEventWithFitness['instchute'],
                                                                               idealEventWithFitness['instladder'])[0]
                                t_sub.multistack.append(linkedEventWithScore)

                        # if len(allVectorsTest) == 150:
                        # plt.figure(figsize=(10, 10))
                        # temp = set()
                        # temp.update(allVectorsTest)
                        # temp.update()
                        # for vector in allVectorsTest:
                        #     x_values = [vector[0][0], vector[1][0]]
                        #     y_values = [vector[0][1], vector[1][1]]
                        #     plt.plot(x_values, y_values)
                        # plt.show()
                        # plt.waitforbuttonpress()

                    avgHolePct, avgChutesPct = self.recalcTrackCompletionPcts(trackEventsOverview)

            if allTracksStalled:
                stallCounter += 1
            else:
                stallCounter = 0

        effLengths = []
        for t in trackEventsOverview:
            effLength = self.runPartialTrackEffLengthHoles(t.track_id, t.eventsetbuild,
                                                           t.tracklength,
                                                           readMode=True)[0] * (t.tracklength / t.controllength)
            if abs(effLength - self.config.effectiveboardlength) <= self.config.minqualityboardlengthmatching:
                # Lock this in!
                t.track.instLocked = True

            effLengths.append(dict(trackeventoverview=t, track_id=t.track_id,
                                   efflength=effLength, tracklocked=t.track.instLocked))
            sortedNodes = t.eventnodes
            sortedNodes.sort()

            nodesFound = False
            if len(self.eventNodesByTrack) > 0:
                for t_node in self.eventNodesByTrack:
                    if t_node['tracknum'] == t.tracknum:
                        t_node['nodes'] = sortedNodes
                        nodesFound = True
                        break
            if not nodesFound: self.eventNodesByTrack.append(dict(tracknum=t.tracknum, nodes=sortedNodes))

            ctl = len(t.chutes) / len(t.ladders) if len(t.ladders) > 0 else 1.0
            ltc = len(t.ladders) / len(t.chutes) if len(t.chutes) > 0 else 1.0

            print("{} chutes, {} ladders, {} events; ctl: {} ltc: {}".format(len(t.chutes), len(t.ladders),
                                                                             len(t.eventsetbuild), ctl, ltc))
            print("Two hits: {}".format(t.twohitsthusfar))
            print("{} nogos, {} denies".format(t.numnogos, t.numdenies))

        avgEffLength = sum(l['efflength'] for l in effLengths) / len(effLengths)

        for l in effLengths:
            print("Track {} has effective length of {}, which should yield an approx {} balance"
                  .format(l['track_id'], l['efflength'],
                          (self.config.effectiveboardlength - l['efflength']) / self.config.effectiveboardlength))

        if (max([abs(l['efflength'] - self.config.effectiveboardlength) for l in effLengths])
                > self.config.minqualityboardlengthmatching):
            # Massage balanceandefflengthcontrolfactor and retry
            # Longer balanceandefflengthcontrolfactor for longer route
            for l in effLengths:
                if l['tracklocked']: continue
                oldVal = self.paramSet.tryGetParam(l['track_id'], "balanceandefflengthcontrolfactor")
                newVal = oldVal
                # Take avg of this eff length and prev, to smooth out jumps
                prevEffLength, prevEffIdx = self.config.effectiveboardlength, -1
                for l_prv_idx in range(0, len(prevEffLengths)):
                    if prevEffLengths[l_prv_idx]['track_id'] == l['track_id']:
                        prevEffIdx = l_prv_idx
                        prevEffLength = prevEffLengths[l_prv_idx]['efflength']
                        break

                if (prevEffLength + l['efflength']) / 2 < self.config.effectiveboardlength - self.config.minqualityboardlengthmatching:
                    # Increase factor to lengthen board
                    newVal = oldVal + self.config.minqualityboardlengthintervalsrpt
                elif (prevEffLength + l['efflength']) / 2 > self.config.effectiveboardlength + self.config.minqualityboardlengthmatching:
                    # Decrease factor to shorten board
                    newVal = oldVal - self.config.minqualityboardlengthintervalsrpt
                if newVal > 0.95:
                    newVal = 0.95
                elif newVal < 0.05:
                    newVal = 0.05
                self.paramSet.tryModParam(l['track_id'], "balanceandefflengthcontrolfactor", newVal)
                prevEffLengths[prevEffIdx]['efflength'] = l['efflength']

            print("Retry, not good enough board eff length quality\n")
            end_time = time.time()
            self.totalTime += end_time - start_time

            return False

        end_time = time.time()
        self.totalTime += end_time - start_time
        totMarkov = self.miniMarkovTime / self.totalTime
        totScore = self.scoringTime / self.totalTime
        totBench = self.benchmarkSetupTime / self.totalTime
        print("Minimarkov took up %s" % totMarkov)
        print("Scoring took up %s" % totScore)

        return True

    def _build_track_state(self, params):
        """
        Builds tryEventSet's per-track working-state list
        (`trackEventsOverview`, a `TrackBuildState` instance per active
        track -- see the TODO on tryEventSet's own docstring re: the
        CandidateEvents precondition). Phase 8 code motion (verbatim, no
        logic changed) of tryEventSet's setup preamble -- no branching
        back into the placement loop below it, which is what made this
        the safest possible first extraction out of this method (Phase 8
        step 1). Phase 8 step 2 turned the dict this used to build into
        the real `TrackBuildState` class (see its docstring) -- this
        method's own body is otherwise unchanged from step 1.

        Builds one `TrackBuildState` per track not already `instLocked`,
        from that track's real `CandidateEvents` (must already be
        populated -- see the precondition documented on `tryEventSet`),
        plus the energy/length-distribution curves retrieved via
        `self.getEnergyCurve()`/`self.getNormLengthDistCurve()`/
        `self.getNormLengthOverTimeCurve()`.

        Args:
            params: parameter set (same as tryEventSet's `params`).

        Returns:
            list[TrackBuildState]: one working-state instance per active
            track.
        """
        trackEventsOverview = [TrackBuildState(track=t, trackidx=t.num - 1, tracknum=t.num, optevents=0, track_id=t.Track_ID,
                                    optfirstchute=0, trackfilled=False, tracklength=len(t.trackholes),
                                    lengthdeviation=(
                                                                len(t.trackholes) - self.config.effectiveboardlength) / self.config.effectiveboardlength,
                                    spacinghisto=[], minspacectr=0,  # chosenscorecutoff=100,
                                    eventsetbuild=t.eventSetBuild, candeventspecs=[],
                                    lengthdistidealcurve=[], lengthdistactualhist=[],
                                    lengthovertimeideal=[], maxlength=0,
                                    trackenergycurve=[], trackenergyintegral=[],
                                    candavgenergy=0.0, energybuffer=0.0, energybufferidx=0, candeventstartlookup=[],
                                    candcursor=0, chutecursor=0, holecoords=[h.coords for h in t.trackholes],
                                    lasteventtop=0, previsladder=False, chutebases=[], chutetops=[],
                                    ladders=[], chutes=[],
                                    eventnodes=[], twohitsthusfar=0, cancels=0, eventscount=0,
                                    ladderbases=[], laddertops=[], holescompletepct=0.0, chutescompletepct=0.0,
                                    curhole=0,
                                    compensationbuffer=0.0, trackstalledcounter=0, trackisstalled=False,
                                    multistack=[], controllength=0, curestefflength=len(t.trackholes),
                                    nomultis=False,
                                    numdenies=0, numnogos=0)
                               for t in self.board.tracks if not t.instLocked]

        # Lock out multis if one or more tracks are locked
        if len(trackEventsOverview) != len(self.board.tracks):
            for t in trackEventsOverview: t.nomultis = True

        # Retrieve & normalize energy curve and det integral
        energyCurve, energyNormIntegral = self.getEnergyCurve()

        # Retrieve & normalize length dist hist curve
        normLengthHistDist = self.getNormLengthDistCurve()

        # Retrieve & normalize length dist over time curve
        normLengthOverTimeDist = self.getNormLengthOverTimeCurve()

        # Compute overall figures & charts
        allCands = [c for c in [t.track.candidateEvents.candidateEvents for t in trackEventsOverview] for c in c]
        self.allTentLengthHisto = []
        for i in range(0, max([c.length for c in allCands])):
            self.allTentLengthHisto.append([i + 1, 0])

        allCandsEnergyPotentialBuilder = 0.0
        for i in range(0, len(trackEventsOverview)):
            allCandsEnergyPotentialBuilder += sum([c.length * (
                2 if (True if c.startHole.num < params.tryGetParam(trackEventsOverview[i].track_id,
                                                                   'ladderscanstartat')
                      else c.canBeLadder) else 1)
                                                   for c in
                                                   trackEventsOverview[i].track.candidateEvents.candidateEvents])
        avgOverallCandEnergyPotential = allCandsEnergyPotentialBuilder / len(allCands)

        # Iterate over tracks, create event when energy buildup exceeds req

        for t in trackEventsOverview:
            # Determine control lengths with blank track
            t.controllength = self.runPartialTrackEffLengthHoles(t.track_id, [],
                                                                    t.tracklength,
                                                                    readMode=True)[0]
            if t.controllength == 9999999:
                raise Exception("Failed initial control length")
                sdfsd = ""

            # Create track-specific energy curve
            candEventSpecs = [dict(event=c, isshared=c.isShared, eventtop=c.endHole.num, eventbase=c.startHole.num,
                                   length=c.length,
                                   canbeladder=False if c.startHole.num <
                                                        params.tryGetParam(t.track_id, 'ladderscanstartat')
                                   else c.canBeLadder)
                              for c in t.track.candidateEvents.candidateEvents]
            candEventSpecs.sort(key=lambda c: (c['eventtop'], c['length']))
            t.candeventspecs = candEventSpecs

            # NOTE: energy counts double for chutes + ladders, since energy is defined by position modulation force
            candEnergyPotential = (sum([c['length'] for c in candEventSpecs]) +
                                   sum([c['length'] for c in candEventSpecs if c['canbeladder']]))
            candAvgEnergy = candEnergyPotential / len(candEventSpecs)
            t.candavgenergy = candAvgEnergy

            # If the avg cand nrg is more than global avg, fewer events & vice versa
            candEnergySkew = ((candAvgEnergy - avgOverallCandEnergyPotential) /
                              (avgOverallCandEnergyPotential * params.tryGetParam(t.track_id,
                                                                                  'candenergyskewdiminisher')))
            t.optevents = int(params.tryGetParam(t.track_id, 'baseopteventspertrack') * (1.0 - candEnergySkew))
            t.optfirstchute = int(params.tryGetParam(t.track_id, 'baseoptfirstchute') * (1.0 + candEnergySkew))

            # Set up ideal length distribution curve
            discrLengthDistCurve = self.discretizeCurve(normLengthHistDist, max([c['length'] for c in candEventSpecs]))
            t.lengthdistidealcurve = self.actualizeCurve(discrLengthDistCurve, 1,
                                                            t.optevents / sum([n[1] for n in discrLengthDistCurve]))
            t.lengthdistactualhist = []
            for i in range(0, max([c['length'] for c in candEventSpecs])):
                t.lengthdistactualhist.append([i + 1, 0])

            # Set up ideal length over time curve
            discrLengthOverTimeCurve = self.discretizeCurve(normLengthOverTimeDist,
                                                            len(t.track.trackholes))
            t.lengthovertimeideal = self.actualizeCurve(discrLengthOverTimeCurve, 1,
                                                           max([c['length'] for c in candEventSpecs]))
            t.maxlength = max([c['length'] for c in candEventSpecs])

            # Set up spacing histogram to help ensure even distribution
            t.spacinghisto = []
            for i in range(0, int((t.optevents / len(t.track.trackholes)) * params.tryGetParam(t.track_id,
                                                                                                     'eventspacingdeviationfactor'))):
                t.spacinghisto.append([i + 1, 0])

            normTrackCurveNetEnergy = sum([c[1] for c in energyCurve])
            trackEnergyCurve = self.actualizeCurve(energyCurve, t.track.length,
                                                   (candAvgEnergy * t.optevents) / normTrackCurveNetEnergy)
            t.trackenergycurve = trackEnergyCurve
            trackEnergyIntegral = self.actualizeCurve(energyNormIntegral, t.track.length,
                                                      candAvgEnergy * t.optevents, integrate=True)
            t.trackenergyintegral = trackEnergyIntegral
            t.compensationbuffer = t.lengthdeviation * self.config.effectiveboardlength
            t.track.setTentativeEvents([])
            t.eventsetbuild = t.track.eventSetBuild

        return trackEventsOverview

    def testPlotVectorsOnHoles(self, vectors):
        """
        Creates a visualization of vectors overlaid on the board's hole positions.
        Delegates to self.plotter (see event_set_plotter.py).
        """
        self.plotter.test_plot_vectors_on_holes(self, vectors)

    def buildSetIntoEvents(self):
        """
        Converts the current event set into actual game events on the board.
        
        This method takes the tentatively placed events and finalizes them by creating
        the corresponding Chute and Ladder objects on each track. It also updates the
        track's event impedance based on the placed events.
        
        Note:
            - Converts events in track.eventSetBuild to Chute/Ladder objects
            - Updates track's ladders and chutes lists
            - Recalculates event impedance for each track
        """
        for t in self.board.tracks:
            for e in t.eventSetBuild:
                if e.instanceIsLadder: t.addLadder(bd.Ladder(e.startHole.num, e.endHole.num, t.num, e.crowVector, e))
                if e.instanceIsChute: t.addChute(bd.Chute(e.endHole.num, e.startHole.num, t.num, e.crowVector, e))
            t.setEventLadders([l.start for l in t.ladders])
            t.setEventChutes([c.start for c in t.chutes])

            # Set descriptive stats
            t.setEventImpedance()

    def buildExplicitSetIntoEvents(self, explicitEventSet):
        """
        Builds a specific set of events into the board's tracks.
        
        This method allows for manual specification of events rather than generating
        them algorithmically. It's useful for testing specific configurations or
        implementing custom event placement logic.
        
        Args:
            explicitEventSet: A list of event sets, where each set corresponds to a track.
                             Each set contains Event objects to be placed on that track.
                             
        Note:
            - Clears any existing events before applying the new set
            - Creates Chute and Ladder objects for each event
            - Updates track's event impedance
        """
        self.clearEventSet()
        for es in explicitEventSet:
            t = self.board.getTrackByNum(es[0].trackNum)
            t.eventSetBuild = es
            for e in es:
                if e.instanceIsLadder: t.addLadder(bd.Ladder(e.startHole.num, e.endHole.num, t.num, e.crowVector, e))
                if e.instanceIsChute: t.addChute(bd.Chute(e.endHole.num, e.startHole.num, t.num, e.crowVector, e))
            t.setEventLadders([l.start for l in t.ladders])
            t.setEventChutes([c.start for c in t.chutes])

            # Set descriptive stats
            t.setEventImpedance()

    def plot_coordinates_and_vectors(self, bitmap_name='output_bitmap.png'):
        """
        Plots multiple sets of coordinates and vectors, and saves the plot as a bitmap image.
        Delegates to self.plotter (see event_set_plotter.py).

        Args:
            bitmap_name (str): The name of the output bitmap file.
        """
        self.plotter.plot_coordinates_and_vectors(self, bitmap_name)
