# Maybe do set of events
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
import sqlite3

import game_params as gp
import cribsandladders.PossibleEvents as ps
import cribsandladders.BaseLayout as bse
import cribsandladders.Board as bd
import matplotlib.pyplot as plt
import numpy as np
import copy as cp
import Enums as en
import bisect as bsc
import pandas as pd
from collections import Counter
import sqlite3 as sql
from datetime import datetime as dt
from io import StringIO
import contextlib
from collections import defaultdict
import markovgame as mg

#TODO: curvify lines, order by length in holes, curve it bspline? then iterate out making sure curve does not interfere with any neighbours
#maybe do this as sep class, once have a board w/ tracks and events established, call Curvify
#factor in occluded space for logo etc

class EventSetBuilder:
    def __init__(self, board, possibleEvents):
        self.board = board
        self.possibleEvents = possibleEvents
        self.allTentLengthHisto = []
        self.paramSet = ParamSet(self.board, self.board.tracks)
        self.orthos = 0
        self.multis = 0
        self.events = 0
        self.cancels = 0
        self.eventNodesByTrack = []
        self.posHands =  [item["move"] for item in gp.probHandHist]
        self.posHandProbs =  [item["prob"] for item in gp.probHandHist]
        self.posPegs =  [item["move"] for item in gp.probPegHist]
        self.posPegProbs =  [item["prob"] for item in gp.probPegHist]
        self.pegRounds =  [item["rounds"] for item in gp.probPegRounds]
        self.pegRoundProbs =  [item["prob"] for item in gp.probPegRounds]
        self.benchmarkMoves_df = None
        self.track_dict = None
        self.prevEffLengths_starter = [dict(track_id=t.Track_ID, efflength=len(t.trackholes))
                                       for t in self.board.tracks]
        self.avgScoreSum, self.avgScoreDiv, self.avgScore = 0, 0, 0

    def clearEventSet(self):
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
        builditer = 0
        self.paramSet.monteCarlo()
        prevEffLengths = cp.deepcopy(self.prevEffLengths_starter)
        while not self.tryEventSet(self.paramSet, prevEffLengths):
            builditer += 1
            if builditer > gp.maxitertrynewbuild:
                raise Exception(
                "Passed max # iters ({}) to find an event set.  ".format(gp.maxitertrynewbuild) +
                "This board may not be feasible.  Try adding more folds in the tracks")
        self.buildSetIntoEvents()
        #TEMPPP!
        # self.plot_coordinates_and_vectors()

    def runMonteCarlo(self, optimizerRunSet, optimizerRun):
        self.clearEventSet()
        self.paramSet.monteCarlo()
        self.paramSet.tempInsertParamsDb(optimizerRunSet, optimizerRun)
        builditer = 0
        prevEffLengths = cp.deepcopy(self.prevEffLengths_starter) 
        while not self.tryEventSet(self.paramSet, prevEffLengths):
            builditer += 1
            if builditer > gp.maxitertrynewbuild:
                raise Exception(
                    "Passed max # iters ({}) to find an event set.  ".format(gp.maxitertrynewbuild) +
                    "This board may not be feasible.  Try adding more folds in the tracks")
        self.buildSetIntoEvents()
        # self.plot_coordinates_and_vectors()

    def runMidpointInitParams(self, optimizerRunSet, optimizerRun):
        self.clearEventSet()
        self.paramSet.midpointInitParams()
        self.paramSet.tempInsertParamsDb(optimizerRunSet, optimizerRun)
        builditer = 0
        prevEffLengths = cp.deepcopy(self.prevEffLengths_starter) 
        while not self.tryEventSet(self.paramSet, prevEffLengths):
            #TEMPPPP
            # self.buildSetIntoEvents()
            # self.plot_coordinates_and_vectors()
            builditer += 1
            if builditer > gp.maxitertrynewbuild:
                raise Exception(
                    "Passed max # iters ({}) to find an event set.  ".format(gp.maxitertrynewbuild) +
                    "This board may not be feasible.  Try adding more folds in the tracks")
        self.buildSetIntoEvents()
        # self.plot_coordinates_and_vectors()


    def setParamsIntoDb(self,optimizerRunSet, optimizerRun ):
        self.paramSet.tempInsertParamsDb(optimizerRunSet, optimizerRun)

    def buildBoardFromParamsDb(self, optimizerRunSet, optimizerRun):
        self.clearEventSet()
        self.paramSet.intakeParamsFromDb(optimizerRunSet, optimizerRun)
        # self.paramSet.tempInsertParamsDb(100000+optimizerRun)
        builditer = 0
        prevEffLengths = cp.deepcopy(self.prevEffLengths_starter) 
        while not self.tryEventSet(self.paramSet, prevEffLengths):
            builditer += 1
            if builditer > gp.maxitertrynewbuild:
                raise Exception(
                    "Passed max # iters ({}) to find an event set.  ".format(gp.maxitertrynewbuild) +
                    "This board may not be feasible.  Try adding more folds in the tracks")
        self.buildSetIntoEvents()

    def modParamsForFmin(self, paramsSubset, fminParamsList, optimizerRunSet, optimizerRun):
        self.paramSet.modParamsForFmin(paramsSubset, fminParamsList)

        self.clearEventSet()
        self.paramSet.tempInsertParamsDb(optimizerRunSet, optimizerRun)
        builditer = 0
        prevEffLengths = cp.deepcopy(self.prevEffLengths_starter) 
        while not self.tryEventSet(self.paramSet, prevEffLengths):
            builditer += 1
            if builditer > gp.maxitertrynewbuild:
                raise Exception(
                    "Passed max # iters ({}) to find an event set.  ".format(gp.maxitertrynewbuild) +
                    "This board may not be feasible.  Try adding more folds in the tracks")
        self.buildSetIntoEvents()


    def buildBoardFromParams(self, instanceParams_df, optimizerRunSet, optimizerRun):
        self.clearEventSet()
        self.paramSet.intakeParams(instanceParams_df)
        builditer = 0
        prevEffLengths = cp.deepcopy(self.prevEffLengths_starter) 
        while not self.tryEventSet(self.paramSet, prevEffLengths):
            builditer += 1
            if builditer > gp.maxitertrynewbuild:
                raise Exception(
                    "Passed max # iters ({}) to find an event set.  ".format(gp.maxitertrynewbuild) +
                    "This board may not be feasible.  Try adding more folds in the tracks")
        self.paramSet.tempInsertParamsDb(optimizerRunSet, optimizerRun)
        self.buildSetIntoEvents()

    def plotBoard(self):
        self.plot_coordinates_and_vectors()

    def retrieveOrGenerateBenchmarkMoves(self):
        with contextlib.closing(sql.connect('etc/Optimizer.db')) as sqlConn:
            with sqlConn:
                with contextlib.closing(sqlConn.cursor()) as sqliteCursor:
                    #Try retrieve benchmark moves
                    query = "SELECT * FROM BenchmarkMoves WHERE Board_ID = ?"
                    sqliteCursor.execute(query, [self.board.boardID])
                    self.benchmarkMoves_df = pd.DataFrame(sqliteCursor.fetchall(),
                                                       columns=[d[0] for d in sqliteCursor.description])
        if len(self.benchmarkMoves_df) == 0:
            #Generate new benchmark moves out to double track length
            insertQuery = "INSERT INTO BenchmarkMoves VALUES(?,?,?,?,?)"
            for t in self.board.tracks:
                with contextlib.closing(sql.connect('etc/Optimizer.db')) as sqlConn:
                    with sqlConn:
                        with contextlib.closing(sqlConn.cursor()) as sqliteCursor:
                            sqliteCursor.execute("BEGIN TRANSACTION")
                            effLength, sequences = self.runPartialTrackEffLengthHoles(t.Track_ID,[],
                                                                                      2*len(t.trackholes))
                            for trial in range(len(sequences)):
                                for move in range(len(sequences[trial])):
                                    sqliteCursor.execute(insertQuery, [self.board.boardID, t.Track_ID, trial, move,
                                                                       sequences[trial][move]])
                            sqliteCursor.execute("END TRANSACTION")

            #Retrieve newly created benchmarks
            with contextlib.closing(sql.connect('etc/Optimizer.db')) as sqlConn:
                with sqlConn:
                    with contextlib.closing(sqlConn.cursor()) as sqliteCursor:
                        #Try retrieve benchmark moves
                        query = "SELECT * FROM BenchmarkMoves WHERE Board_ID = ?"
                        sqliteCursor.execute(query, [self.board.boardID])
                        self.benchmarkMoves_df = pd.DataFrame(sqliteCursor.fetchall(),
                                                           columns=[d[0] for d in sqliteCursor.description])

        #Index & sort
        # self.benchmarkMoves_df.set_index(['Track_ID', 'Trial', 'MoveNum'], inplace=True)
        # self.benchmarkMoves_df.sort_index(inplace=True)


        # Initialize a dictionary to hold the lists per Track_ID
        track_dict = defaultdict(list)

        # Group the DataFrame by Track_ID and Trial
        grouped = self.benchmarkMoves_df.groupby(['Track_ID', 'Trial'])

        # Iterate over each group
        for (track_id, trial), group in grouped:
            # Create a list of MoveNums with MoveVal stored
            moves_list = [(row['MoveNum'], row['MoveVal']) for _, row in group.iterrows()]

            # Sort moves_list by MoveNum
            moves_list.sort(key=lambda x: x[0])

            # Store only MoveVal in the correct order of MoveNum
            trial_list = [move_val for _, move_val in moves_list]

            # Append the trial_list to the corresponding Track_ID
            track_dict[track_id].append(trial_list)

        # Convert defaultdict to regular dict for easier access
        self.track_dict = dict(track_dict)

        # Convert each trial's list to just MoveVal list
        for track_id in self.track_dict:
            for i in range(len(self.track_dict[track_id])):
                self.track_dict[track_id][i] = [val for val in self.track_dict[track_id][i]]

    def buildPartialSetIntoTrack (self, track, startPoint, stopPoint):
            for e in range(startPoint, stopPoint):
                if len(track.candidateEvents.candidateEvents) == 0:
                    dsfd ="sdfds"
                #TODO: maybe start at beginning of tracks, work way up, event by event, iterating thru tracks, base spacing events tot
                #Try to follow a given story arc!  Maybe take input curve, try to follow it best fir
                currEvent = rd.choice(track.candidateEvents.candidateEvents)
                #TEMP:
                #TODO: figure this out in possible events, for now just excluding here
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
        intersects = [vector]
        corners = self.orthoBoundingBox(vector)
        for i in range(0,4):
            intersects.append((corners[i], corners[(i+1)%4]))
        return tuple(intersects)

    def orthoBoundingBox(self, vector):
        ortho_dxdy = self.possibleEvents.orthogonal_vector(vector[0], vector[1], gp.eventminspacing / 2.0, False)
        revOrtho_dxdy = [(-1) * d for d in ortho_dxdy]
        corners = []
        for o in [ortho_dxdy, revOrtho_dxdy]:
            for v in vector:
                corners.append(tuple([c + co for c, co in zip(v, o)]))
        # NOTE: we zigzagged in teh nested iterators, unzigging here
        corners = [corners[0], corners[1], corners[3], corners[2]]
        return tuple(corners)

    def getNormLengthDistCurve (self):
        return self.getNormalizedIdealCurve(gp.eventlengthdisthistcurvefile)

    def getNormLengthOverTimeCurve (self):
        return self.getNormalizedIdealCurve(gp.eventlengthovertimeidealcurve1file)

    def getEnergyCurve (self):
        # Normalize the coordinates
        energyCurve = self.getNormalizedIdealCurve(gp.eventenergyfile)

        # Det integral
        normalizer = sum([e[1] for e in energyCurve])
        energyNormIntegral = self.integrateAndNormalizeCurve(energyCurve, normalizer)

        return energyCurve, energyNormIntegral

    def getNormalizedIdealCurve(self, curveFile):
        rawCurve = np.array(bse.svgParserHoles(curveFile, returnRawCoords=True))
        # Extract all x and y values
        x_values = [coord[0] for coord in rawCurve]
        y_values = [coord[1] for coord in rawCurve]

        # Find the min & maximum x and y values
        min_x = min(x_values)
        min_y = min(y_values)
        max_x = max(x_values)
        max_y = max(y_values)
        scale_x = max_x - min_x
        scale_y = max_y - min_y

        # Normalize the coordinates
        normCurve = [((x - min_x) / scale_x, (y - min_y) / scale_y) for x, y in rawCurve]
        normCurve.sort(key=lambda e: e[0])

        return normCurve

    def integrateAndNormalizeCurve(self, curve, normalizer):
        integrated_curve = []
        for i in range(0, len(curve)):
            normx=curve[i][0]
            if i == 0:
                normy = curve[i][1]/normalizer
            else:
                normy = integrated_curve[i-1][1] + curve[i][1]/normalizer
            integrated_curve.append((normx, normy))

        return integrated_curve

    def normalizeCurveMagnitude(self, curve):
        normalizer = max(abs(max([c[1] for c in curve])), abs(min([c[1] for c in curve])))
        if normalizer > 0:
            normalized_curve = []
            for i in range(0, len(curve)):
                normy = curve[i][1] / normalizer
                normalized_curve.append((curve[i][0], normy))

            return normalized_curve
        return curve


    def actualizeCurve(self, curve, x_actualizer, y_actualizer, integrate=False):
        actualized_curve = []
        for i in range(0, len(curve)):
            actx = curve[i][0]*x_actualizer
            if integrate and i > 0:
                acty = (curve[i - 1][1] + curve[i][1]) * y_actualizer
            else:
                acty = curve[i][1]*y_actualizer
            actualized_curve.append((actx, acty))

        return actualized_curve

    def discretizeCurve(self, curve, numBuckets, accumulate=False):
        #If accumulating, NORMALIZE AFTER!!!
        discretized_curve = []
        curveIdx = 0
        discFactor = len(curve)/numBuckets
        for i in range(0, numBuckets):
            discx = i + 1
            accum_y = 0.0
            while curveIdx < len(curve)-1 and curveIdx < i*discFactor:
                if accumulate:
                    accum_y += curve[curveIdx][1]
                curveIdx += 1
            if accumulate:
                if i == 0:
                    discy = accum_y
                else:
                    discy = 0.7*accum_y + 0.3*discretized_curve[i-1][1]
            else:
                discy = curve[curveIdx][1]
            discretized_curve.append((discx, discy))

        return discretized_curve

    def getPointsInProximity(self, searchRange, searchPoints, inputPoint):
        #NOTE: searchPoints MUST BE SORTED!!
        pointPairs = []
        for p in searchPoints:
            if searchRange[0] <= (inputPoint - p) <= searchRange[1]:
                pointPairs.append(dict(point=p, disp=inputPoint - p))

        return pointPairs

    def tryGetDispAllowance (self, dispAllowances, proxPoint):
        allowance = next((allow for allow in dispAllowances
                          if allow['scalardisp'] == abs(proxPoint['disp'])), None)
        if allowance is not None:
            return dict(effect=allowance['isallowed'], mod=allowance['mod'])
        return dict(effect=False, mod=0)

    def getEffectorsForDisps (self, basePoint, searchDisps, posEffectors, eventNodes, events = None, selfScaleLength = -1):
        effectors = []
        for p in searchDisps:
            idx = self.searchOrderedListForVal(eventNodes, basePoint + p)
            if idx > -1:
                allowances = self.tryGetDispAllowance(posEffectors, dict(disp=abs(p)))
                if events is None and selfScaleLength > -1:
                    effectors.append(dict(effect=allowances['effect'], scaledmod=allowances['mod']*selfScaleLength,
                                          scaledenergymod=abs(allowances['mod']*selfScaleLength)))
                elif events is not None and selfScaleLength == -1:
                    effectors.append(dict(effect=allowances['effect'], scaledmod=allowances['mod']*events[idx]['length'],
                                          scaledenergymod=abs(allowances['mod']*events[idx]['length'])))
                else:
                    raise Exception("Must pass either self scale length, or oth events for searching")

        return effectors


    def runPartialTrackEffLengthHoles (self, track_id, partialEventSet, trackActualLength, tentNewLadder=None, tentNewChute=None,
                                       overrideIters = -1, readMode = False):
        #Markov chain forecasting
        #INCORPORATE CRIB EVERY N'TH TURNM!!
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
        for idx in range(trackActualLength):
            if eIdx < len(partialEventMappings) and partialEventMappings[eIdx]['start'] == idx + 1:
                effHoleMap.append(partialEventMappings[eIdx]['end'])
                eIdx += 1
            else:
                effHoleMap.append(idx + 1)


        #Figure out length of partial game
        movesAllTrials = 0
        if overrideIters < 0:
            iters = gp.probminimodeliters
        else:
            iters = overrideIters

        if not readMode:
            sequencesOfMoves = []
            moveCounter = 0
            curSequence = []
            curReadSeq = []
            for trial in range(iters):
                #Set up trial gameplay
                startLoc = 0
                if not readMode:
                    if trial > 0: sequencesOfMoves.append(curSequence)
                    curSequence = []
                else:
                    moveCounter = 0
                    # startLoc = self.benchmarkMoves_df.index.get_loc((track_id, trial, 0))
                    curReadSeq = self.track_dict[track_id][trial]
                dealer = rd.randint(1, gp.numplayers)
                curPos = 0
                countLoops = 0
                trackPosSeq = []
                while curPos < trackActualLength:
                    if not readMode:
                        #Run pegging:
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

                        #Score hand
                        curMove = rd.choices(self.posHands, weights=self.posHandProbs, k=1)[0]
                        curSequence.append(curMove)
                        if curPos + curMove > len(effHoleMap): curPos += curMove
                        else: curPos = effHoleMap[curPos + curMove - 1]
                        movesAllTrials += 1
                        if curPos >= trackActualLength: break

                        if dealer == 1:
                            #Score crib
                            curMove = rd.choices(self.posHands, weights=self.posHandProbs, k=1)[0]
                            curSequence.append(curMove)
                            if curPos + curMove > len(effHoleMap): curPos += curMove
                            else: curPos = effHoleMap[curPos + curMove - 1]
                            movesAllTrials += 1
                            if curPos >= trackActualLength: break
                        dealer = 1 + dealer%gp.numplayers
                    else:
                        # curMove = self.benchmarkMoves_df.iloc[startLoc + moveCounter]['MoveVal']
                        curMove = curReadSeq[moveCounter]
                        if moveCounter ==0:
                            countLoops += 1
                        moveCounter = (moveCounter + 1)%len(curReadSeq)
                        if curPos + curMove > len(effHoleMap):
                            curPos += curMove
                        else:
                            curPos = effHoleMap[curPos + curMove - 1]
                        trackPosSeq.append(curPos)
                        movesAllTrials += 1
                        if countLoops > 10:
                            #Track is stuck in an infinite loop!!! This event is no bueno
                            return 9999999, []
                        if curPos >= trackActualLength: break
            if not readMode: sequencesOfMoves.append(curSequence)

            #Forecast length of game based on control-case ideal moves:hole ratio
            actualPartialMoves = (movesAllTrials/iters)
            eventlessCtrlPartialMoves = (gp.ideallikelihoodholehit*trackActualLength)
            shiftPct = actualPartialMoves/eventlessCtrlPartialMoves
            forecastedTrackEffLengthHoles = trackActualLength*shiftPct
            return forecastedTrackEffLengthHoles, sequencesOfMoves
        else:
            forecastedTrackEffLengthHoles = mg.runPartialTrackEffLengthHoles(trackActualLength, gp.probminimodeliters,
                                                                             self.track_dict[track_id],effHoleMap,
                                                                             gp.numplayers, gp.ideallikelihoodholehit)
            return forecastedTrackEffLengthHoles, None


    def scoreEventsForHole (self, t, hole,
                            chutes, chuteBases, chuteTops, ladders, ladderBases, ladderTops, params,trackEventsOverview,
                            explicitEvent = None,
                            explicitChute = False, explicitLadder = False):
        if explicitEvent is None and hole.num < t['optfirstchute']: return []

        initCompModifier = 0.0
        # twohitimpeders, twohitboosts = [], []
        # twohitimpedanceeffector = (params.tryGetParam(t['track_id'], "twohitfreqimpedance")
        #                            *(t['twohitsthusfar']/t['optevents']))
        #
        # twohitimpeders.append(dict(scalardisp=1, isallowed=(
        #         (1.0 - (hole.num / len(t['track'].trackholes))) * t['compensationbuffer']
        #         / len(t['track'].trackholes)
        #         > params.tryGetParam(t['track_id'],'move1allowanceratio')), mod=(-1)*gp.likelihoodofonemove
        #                                                                         *gp.ideallikelihoodholehit))
        # twohitimpeders.append(dict(scalardisp=2, isallowed=t['compensationbuffer'] > 0, mod=(-1)*gp.likelihoodoftwomove
        #                                                                         *gp.ideallikelihoodholehit))
        # twohitimpeders.append(dict(scalardisp=4, isallowed=t['compensationbuffer'] > 0, mod=(-1)*gp.likelihoodoffourmove
        #                                                                         *gp.ideallikelihoodholehit))
        #
        # twohitboosts.append(dict(scalardisp=1, isallowed=(
        #         (1.0 - (hole.num / len(t['track'].trackholes))) * t['compensationbuffer']
        #         / len(t['track'].trackholes)
        #         < (-1)*params.tryGetParam(t['track_id'],'move1allowanceratio')), mod=gp.likelihoodofonemove
        #                                                                         *gp.ideallikelihoodholehit))
        # twohitboosts.append(dict(scalardisp=2, isallowed=t['compensationbuffer'] < 0, mod=gp.likelihoodoftwomove
        #                                                                         *gp.ideallikelihoodholehit))
        # twohitboosts.append(dict(scalardisp=4, isallowed=t['compensationbuffer'] < 0, mod=gp.likelihoodoffourmove
        #                                                                         *gp.ideallikelihoodholehit))


        # # NOTE that we cannot fall before a laadder, since we are building forward from top of event
        # match spaceSinceLastChute:
        #     case 1:
        #         bypassRules = bypassonehit
        #         if bypassRules: initCompModifier = (-1) * gp.likelihoodofonemove
        #     case 2:
        #         bypassRules = bypasstwohit
        #         if bypassRules: initCompModifier = (-1) * gp.likelihoodoftwomove
        #     case 4:
        #         bypassRules = bypassfourhit
        #         if bypassRules: initCompModifier = (-1) * gp.likelihoodoffourmove
        #
        # if not prevIsLadder and spaceSinceLastChute < gp.mindistancechuteonly: return []
        # if prevIsLadder and not bypassRules and spaceSinceLastChute < gp.mindistanceafterladder: return []
        # if (spaceSinceLastChute < gp.mindistanceeveneventspacing and prevIsLadder and not bypassRules
        #         and spaceSinceLastChute % 2 == 0): return []

        # Passed gauntlet!  Let's try to find an event to deplete this energy
        eventFitnesses = []
        explicitEventCounter = 0
        while ((explicitEvent is not None and explicitEventCounter < 1) or
               (explicitEvent is None and t['candcursor'] < len(t['candeventspecs']) and
                t['candeventspecs'][t['candcursor']]['eventtop'] == hole.num)):
            explicitEventCounter += 1
            # Scoring system: base score is scalar displacement between energy amount and energy buffer
            # Disallow (do not include) if no according bypass rule
            # and base falls within range of 2-hit from another event
            # Sqrt value for each appropriate 2-hit eligible
            if explicitEvent is None:
                candEventSpecs = t['candeventspecs'][t['candcursor']]
            else:
                candEventSpecs = explicitEvent

            if explicitEvent is None and candEventSpecs['isshared']:
                #if one or more tracks are locked, no multis!
                #TODO: get multis working w/ elim mode
                # if t['nomultis']:
                t['candcursor'] += 1
                continue

                #Check if linked event is legal
                assertLegal = True
                for ev in candEventSpecs['event'].linkedEvents:
                    linkedStart = ev.startHole.num
                    linkedEnd = ev.endHole.num
                    linkedTrackNum = ev.trackNum
                    t_sub = None
                    for t_match in trackEventsOverview:
                        if t_match['tracknum'] ==linkedTrackNum:
                            t_sub = t_match
                            break
                    for n in [linkedStart, linkedStart]:
                        idx = bsc.bisect_left(t_sub['eventnodes'], n)
                        if idx < len(t_sub['eventnodes']) and t_sub['eventnodes'][idx] == n:
                            assertLegal = False
                            break
                    if not assertLegal: break
                if not assertLegal:
                    #Cannot have multiple events landing or starting on same space!
                    t['candcursor'] += 1
                    continue

            if (explicitEvent is None and
                    (self.searchOrderedListForVal(t['eventnodes'], candEventSpecs['eventbase']) > -1 or
                self.searchOrderedListForVal(t['eventnodes'], candEventSpecs['eventtop']) > -1)) :
                #Cannot have multiple events landing or starting on same space!
                t['candcursor'] += 1
                continue

            #If hella override, check it
            if (explicitEvent is None and
                    (params.tryGetParam(t['track_id'],'disallowbelowsetlength', optional=True) > 0 and
                candEventSpecs['length'] < params.tryGetParam(t['track_id'],'disallowbelowsetlength'))):
                t['candcursor'] += 1
                continue

            #Check if ortho ratio is exceeded
            if (explicitEvent is None and
                    (candEventSpecs['event'].isOrtho and len(t['eventsetbuild']) > 0 and
                len([e for e in t['eventsetbuild'] if e.isOrtho])/len(t['eventsetbuild']) >
                    params.tryGetParam(t['track_id'],'maxorthoratio'))):
                t['candcursor'] += 1
                continue

            #Check for two-hits

            # boostsIfChute, impedersIfChute, boostsIfLadder, impedersIfLadder = [],[],[],[]
            # curEventLength = candEventSpecs['length']
            #
            # #NOTE: All events can be chutes, hence no need to check annything
            # boostsIfChute.extend(self.getEffectorsForDisps(candEventSpecs['eventbase'], (1,2,4),
            #                                            twohitboosts, ladderBases, events=ladders))
            # impedersIfChute.extend(self.getEffectorsForDisps(candEventSpecs['eventbase'], (1,2,4),
            #                                            twohitboosts, chuteTops, events=chutes))
            # impedersIfChute.extend(self.getEffectorsForDisps(candEventSpecs['eventtop'], (-1,-2,-4),
            #                                            twohitimpeders, chuteBases, selfScaleLength=curEventLength))
            # impedersIfChute.extend(self.getEffectorsForDisps(candEventSpecs['eventtop'], (-1,-2,-4),
            #                                            twohitimpeders, ladderTops, selfScaleLength=curEventLength))
            # if candEventSpecs['canbeladder'] or explicitLadder:
            #     boostsIfLadder.extend(self.getEffectorsForDisps(candEventSpecs['eventtop'], (1,2,4),
            #                                                twohitboosts, ladderBases, events=ladders))
            #     impedersIfLadder.extend(self.getEffectorsForDisps(candEventSpecs['eventtop'], (1,2,4),
            #                                                twohitimpeders, chuteTops, events=chutes))
            #     boostsIfLadder.extend(self.getEffectorsForDisps(candEventSpecs['eventbase'], (-1,-2,-4),
            #                                                twohitboosts, ladderTops, selfScaleLength=curEventLength))
            #     boostsIfLadder.extend(self.getEffectorsForDisps(candEventSpecs['eventbase'], (-1,-2,-4),
            #                                                twohitboosts, chuteBases, selfScaleLength=curEventLength))

            # allModsIfChute = boostsIfChute + impedersIfChute
            # allModsIfLadder = boostsIfLadder + impedersIfLadder

            if explicitEvent is None:
                # canBeChute = len(allModsIfChute) == 0 or next((eff for eff in allModsIfChute if not eff['effect']), None) is None
                # canBeLadder = (candEventSpecs['canbeladder'] and
                #                (len(allModsIfLadder) == 0
                #                or (next((eff for eff in allModsIfLadder if not eff['effect']), None) is None)))
                canBeChute = True
                canBeLadder = candEventSpecs['canbeladder']
            else:
                canBeChute = explicitChute
                #MIGHT BE TOO DRACONIAN to force??
                # canBeLadder = candEventSpecs['canbeladder'] and explicitLadder
                canBeLadder = explicitLadder

            #Check if can be chute only
            if explicitEvent is None and (canBeChute and not canBeLadder and len(ladderBases)/(len(chuteBases) + 1)
                    < params.tryGetParam(t['track_id'], "minladdertochuteratio")):
                t['candcursor'] += 1
                continue

            #Check if can be ladder only
            if explicitEvent is None and (canBeLadder and not canBeChute and len(chuteBases)/(len(ladderBases) + 1)
                < params.tryGetParam(t['track_id'], "minchutetoladderratio")):
                t['candcursor'] += 1
                continue

            #Insert event score as chute, ladder, and both
            for instType in (en.InstanceEventType.CHUTEONLY, en.InstanceEventType.LADDERONLY,
                             en.InstanceEventType.CHUTEANDLADDER):
                effEnergy, effCompModulation = 0, 0
                effLengthForecast, partialTrackEnd = 0, 0
                # modsForType = []
                match instType:
                    case en.InstanceEventType.CHUTEONLY:
                        if not canBeChute: continue
                        if (explicitEvent is None and not candEventSpecs['event'].isOrtho and
                            candEventSpecs['event'].crowLength < gp.mincrowvectordistcancel):
                            continue
                        if explicitEvent is not None and explicitLadder: continue
                        effEnergy = candEventSpecs['length']
                        # modsForType = allModsIfChute
                        effLengthForecast = self.runPartialTrackEffLengthHoles(t['track_id'], t['eventsetbuild'], t['tracklength'],
                                                                               tentNewChute=(candEventSpecs['event'].endHole.num,
                                                                                             candEventSpecs['event'].startHole.num),
                                                                               readMode=True)[0]
                    case en.InstanceEventType.LADDERONLY:
                        if not canBeLadder: continue
                        if (explicitEvent is None and not candEventSpecs['event'].isOrtho and
                            candEventSpecs['event'].crowLength < gp.mincrowvectordistcancel):
                            continue
                        if explicitEvent is not None and explicitChute: continue
                        effEnergy = candEventSpecs['length']
                        # modsForType = allModsIfLadder
                        effLengthForecast = self.runPartialTrackEffLengthHoles(t['track_id'], t['eventsetbuild'], t['tracklength'],
                                                                               tentNewLadder=(candEventSpecs['event'].startHole.num,
                                                                                             candEventSpecs['event'].endHole.num),
                                                                               readMode=True)[0]
                    case en.InstanceEventType.CHUTEANDLADDER:
                        if not (canBeChute and canBeLadder): continue
                        effEnergy = 2*candEventSpecs['length']
                        # modsForType = allModsIfChute + allModsIfLadder
                        effLengthForecast = self.runPartialTrackEffLengthHoles(t['track_id'], t['eventsetbuild'], t['tracklength'],
                                                                               tentNewLadder=(candEventSpecs['event'].startHole.num,
                                                                                             candEventSpecs['event'].endHole.num),
                                                                               tentNewChute=(candEventSpecs['event'].endHole.num,
                                                                                             candEventSpecs['event'].startHole.num),
                                                                               readMode=True)[0]

                # Adjust length as per control length
                effLengthForecast *= t['tracklength']/t['controllength']
                #NOTE: impeders are (-), boosters are (+)
                effCompModulation = effLengthForecast - t['curestefflength']
                # print(str(effCompModulation))

                # effEnergyModulation = 0
                # if len(modsForType) > 0:
                #     #Any two-hits increase nrg, no matter up or down
                #     effEnergyModulation += sum([m['scaledenergymod'] for m in modsForType])

                #BASE SCORE ON BLEND MOD + ENERGY

                # NOTE: longer balanceandefflengthcontrolfactor for longer route
                balFactor = params.tryGetParam(t['track_id'], 'balanceandefflengthcontrolfactor')
                lengtheningControl, shorteningControl = balFactor, 1.0 - balFactor
                curEstLengthDiscr = t['curestefflength'] - gp.effectiveboardlength
                instEstLengthDiscr = effLengthForecast - gp.effectiveboardlength
                if abs(curEstLengthDiscr) > 10:
                    sdfd = ""
                instEstLengthDisp = effLengthForecast - t['curestefflength']
                # Too much instability!  Nix this uber event
                if instEstLengthDisp > gp.maxefflengthdisp: continue

                curScore = 1.0 #Base amt
                if curEstLengthDiscr != 0:
                    # If board is perfect, leave it alone!  Highly unlikely tho except for inital run
                    balScoreMod = abs(instEstLengthDiscr) / abs(curEstLengthDiscr)
                    # If we are moving in correct direction, reward
                    reward = abs(instEstLengthDiscr) < abs(curEstLengthDiscr)
                    if curEstLengthDiscr > 0:
                        # Apply shortening control, board is too long
                        if reward:
                            # curScore = balScoreMod/gp.gamelengthtightness
                            curScore = balScoreMod * math.pow((1.0 - shorteningControl), gp.gamelengthtightness)
                        else:
                            # curScore = balScoreMod*gp.gamelengthtightness
                            curScore = balScoreMod * math.pow((1.0 + shorteningControl), gp.gamelengthtightness)
                    elif curEstLengthDiscr < 0:
                        # Apply lengthening control, board is too short
                        if reward:
                            # curScore = balScoreMod/gp.gamelengthtightness
                            curScore = balScoreMod * math.pow((1.0 - lengtheningControl), gp.gamelengthtightness)
                        else:
                            # curScore = balScoreMod*gp.gamelengthtightness
                            curScore = balScoreMod * math.pow((1.0 + lengtheningControl), gp.gamelengthtightness)







                # compBufDiv = abs(t['compensationbuffer'])
                # if compBufDiv == 0: compBufDiv = 1
                # if abs(t['compensationbuffer'] + effCompModulation) < abs(t['compensationbuffer']):
                #     curScore = 10*(1.0 - abs(effCompModulation)/compBufDiv)
                # elif abs(t['compensationbuffer'] + effCompModulation) > abs(t['compensationbuffer']):
                #     curScore = 10*(1.0 + 10*abs(effCompModulation)/compBufDiv)
                # else:
                #     curScore = 10
                # if curScore < 0:
                #     sdfsd=""
                if abs(effEnergy) + abs(t['energybuffer']) > 0:
                    curScore *= (1.0 + (params.tryGetParam(t['track_id'], 'energybufferenforcement')
                                        *abs(effEnergy - t['energybuffer'])/(abs(effEnergy) + abs(t['energybuffer']))))
                effNetEnergy = effEnergy + abs(effCompModulation)

                # #NOTE: longer balanceandefflengthcontrolfactor for longer route
                # balFactor = params.tryGetParam(t['track_id'], 'balanceandefflengthcontrolfactor')
                # #TODO: re-enable this??
                # # if instType == en.InstanceEventType.CHUTEONLY and balFactor > 0.5:
                # #     curScore *= (1.0 - balFactor)/0.2
                # # elif instType == en.InstanceEventType.LADDERONLY and balFactor < 0.5:
                # #     curScore *= balFactor/0.2
                # # elif instType == en.InstanceEventType.LADDERONLY and balFactor > 0.5:
                # #     curScore /= (1.0 - balFactor)/0.2
                # # elif instType == en.InstanceEventType.CHUTEONLY and balFactor < 0.5:
                # #     curScore /= balFactor/0.2
                # curEstLengthDiscr = t['curestefflength'] - gp.effectiveboardlength
                # instEstLengthDiscr = effLengthForecast - gp.effectiveboardlength
                # if abs(curEstLengthDiscr) > 10:
                #     sdfd=""
                # instEstLengthDisp = effLengthForecast - t['curestefflength']
                # #Too much instability!  Nix this uber event
                # if instEstLengthDisp > gp.maxefflengthdisp: continue
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
                if instType in (en.InstanceEventType.LADDERONLY, en.InstanceEventType.CHUTEANDLADDER):
                    for p in (1, 2 ,4):
                        if self.searchOrderedListForVal(ladderBases, candEventSpecs['event'].endHole.num + p) > -1:
                            if abs(p) == 4: numTwoHitsLoose += 1
                            else: numTwoHits += 1
                        if self.searchOrderedListForVal(chuteTops, candEventSpecs['event'].endHole.num + p) > -1:
                            if abs(p) == 4: numTwoHitsLoose += 1
                            else: numTwoHits += 1
                    for p in (-1, -2 ,-4):
                        if self.searchOrderedListForVal(ladderTops, candEventSpecs['event'].startHole.num + p) > -1:
                            if abs(p) == 4: numTwoHitsLoose += 1
                            else: numTwoHits += 1
                        if self.searchOrderedListForVal(chuteBases, candEventSpecs['event'].startHole.num + p) > -1:
                            if abs(p) == 4: numTwoHitsLoose += 1
                            else: numTwoHits += 1
                if instType in (en.InstanceEventType.CHUTEONLY, en.InstanceEventType.CHUTEANDLADDER):
                    for p in (1, 2, 4):
                        if self.searchOrderedListForVal(ladderBases, candEventSpecs['event'].startHole.num + p) > -1:
                            if abs(p) == 4: numTwoHitsLoose += 1
                            else: numTwoHits += 1
                        if self.searchOrderedListForVal(chuteTops, candEventSpecs['event'].startHole.num + p) > -1:
                            if abs(p) == 4: numTwoHitsLoose += 1
                            else: numTwoHits += 1
                    for p in (-1, -2, -4):
                        if self.searchOrderedListForVal(ladderTops, candEventSpecs['event'].endHole.num + p) > -1:
                            if abs(p) == 4: numTwoHitsLoose += 1
                            else: numTwoHits += 1
                        if self.searchOrderedListForVal(chuteBases, candEventSpecs['event'].endHole.num + p) > -1:
                            if abs(p) == 4: numTwoHitsLoose += 1
                            else: numTwoHits += 1

                if (numTwoHits > 0 and numTwoHits*params.tryGetParam(t['track_id'], 'twohitfreqimpedance') >
                        (gp.allowabletwohits - t['twohitsthusfar'])):
                    continue
                curScore *= (1.0 + (numTwoHits + numTwoHitsLoose/2)*t['twohitsthusfar']*
                             params.tryGetParam(t['track_id'], 'twohitfreqimpedance'))


                #Adjust score based on direction of effCompModulation (trying to get to 0)
                #Again, boosters are (+) and vice versa, so we attempt negation
                tempp = False
                # if effCompModulation != 0:
                #     # print("Score b4 comp, hist & cancel mods: {}".format(curScore))
                #     tempp = True
                #     #balanceandefflengthcontrolfactor shifts based on whether game needs more lengthening or shortening
                #     if t['compensationbuffer'] < 0:
                #         effControlFactor = (1.0 - params.tryGetParam(t['track_id'], 'balanceandefflengthcontrolfactor'))
                #     else:
                #         effControlFactor = params.tryGetParam(t['track_id'], 'balanceandefflengthcontrolfactor')
                #
                #     if abs(t['compensationbuffer'] + effCompModulation) < abs(t['compensationbuffer']):
                #         #Good, reward heavily!
                #         curScore *= effControlFactor
                #     else:
                #         #Punish
                #         curScore /= effControlFactor


                # #Boost score for each allowable 2-hit, factoring in two-hit impedance
                # for m in modsForType:
                #     curScore = math.sqrt(curScore)*(1.0 + twohitimpedanceeffector)

                #Impede score if too many ladders/chutes are getting cancelled
                if instType != en.InstanceEventType.CHUTEANDLADDER and t['cancels'] >= gp.whenstartworryingaboutcancels:
                    if t['cancels'] >= 2.5*gp.whenstartworryingaboutcancels: continue
                    curScore *= (1.0 + params.tryGetParam(t['track_id'],'cancelimpedance')*(t['cancels'] + 1)
                                 /(t['eventscount'] + 1))

                #Preferentially weight based on proximity to end of track
                endTrackWeight = params.tryGetParam(t['track_id'], 'eventstowardsendoftrackreward')
                eventPosRelMidpoints = candEventSpecs['event'].midPointNum/t['tracklength'] - 0.5
                if eventPosRelMidpoints < 0:
                    curScore *= (1.0 + abs(eventPosRelMidpoints)*endTrackWeight)
                else:
                    curScore /= (1.0 + abs(eventPosRelMidpoints)*endTrackWeight)

                #Factor in distribution of length histogram to help ensure distributed lengths
                #Try to curve fit specified ideal histo
                #NOTE: golf-stylee, lower score is better
                curLenPerc = 0.0
                curLength = candEventSpecs['length']
                if sum([h[1] for h in self.allTentLengthHisto]) > 0:
                    curLenPerc = (self.allTentLengthHisto[curLength - 1][1] /
                                                     sum([h[1] for h in self.allTentLengthHisto]))
                idealPerc = t['lengthdistidealcurve'][curLength - 1][1]
                lenDistDisp = curLenPerc - idealPerc
                if lenDistDisp < 0:
                    #Need more!
                    curScore /= (1.0 + abs(lenDistDisp))*params.tryGetParam(t['track_id'],'lengthhistogramscoringfactor')
                elif lenDistDisp > 0:
                    #Too many of this length already, downshift
                    curScore *= (1.0 + abs(lenDistDisp))*params.tryGetParam(t['track_id'],'lengthhistogramscoringfactor')

                #Factor in distribution of length over time
                #TEMP!!
                if len(t['lengthovertimeideal']) < hole.num:
                    print("FAILED LENGTH OVER TIME TEST: hole.num {}".format(hole.num))
                else:
                    idealLengthForHole = t['lengthovertimeideal'][hole.num-1][1]
                    scoreMod = (1.0 + (abs(curLength - idealLengthForHole)/t['maxlength']) *
                    params.tryGetParam(t['track_id'],'lengthovertimescoringfactor'))
                    if curScore >= 0:
                        curScore *= scoreMod
                    else:
                        curScore /= scoreMod

                #Aggr into avg score
                self.avgScoreSum += curScore
                self.avgScoreDiv += 1
                self.avgScore = self.avgScoreSum/self.avgScoreDiv

                #Elminate options based on shortening & lengthening control
                if (curEstLengthDiscr > 0 and shorteningControl > 0.5 and curScore > self.avgScore
                        *gp.goodscorecutoffperc*2 * (
                        1.0 - (2 * (shorteningControl - 0.5)))):
                    t['numnogos'] += 1
                    continue
                elif (curEstLengthDiscr < 0 and lengtheningControl > 0.5 and curScore > self.avgScore
                      *gp.goodscorecutoffperc*2 * (
                        1.0 - (2 * (lengtheningControl - 0.5)))):
                    t['numnogos'] += 1
                    continue


                # if sum([h[1] for h in self.allTentLengthHisto]) > 0:
                #     curLenPerc = (self.allTentLengthHisto[candEventSpecs['length'] - 1][1] /
                #                   sum([h[1] for h in self.allTentLengthHisto]))
                #     avgPerc = (1.0 / len([h for h in self.allTentLengthHisto if h[1] > 0]))
                #     curScore *= (1.0 + (curLenPerc-avgPerc)*params.tryGetParam(t['track_id'],'lengthhistogramscoringfactor)

                #Add event score to output list
                # if tempp: print("Score after hist & cancel mods: {}".format(curScore))
                # print("{} {}".format(instType, curScore))
                # print(curScore)
                eventFitnesses.append(dict(event=candEventSpecs['event'],
                                           eventspecs=candEventSpecs,
                                           score=curScore, effnetenergy=effNetEnergy, effcompmodulation=effCompModulation,
                                           insttype=instType,
                                           instchute=instType in (en.InstanceEventType.CHUTEANDLADDER,
                                                                  en.InstanceEventType.CHUTEONLY),
                                           instladder=instType in (en.InstanceEventType.CHUTEANDLADDER,
                                                                  en.InstanceEventType.LADDERONLY),
                                           lasteventtop=0
                                            ,
                                            twohits = numTwoHits, estefflength=effLengthForecast
                                           ))

            t['candcursor'] += 1

        eventFitnesses.sort(key=lambda f: f['score'])
        if ((len(eventFitnesses) > 0 and eventFitnesses[0]['score'] <= self.avgScore*gp.goodscorecutoffperc*2)
                or explicitEvent is not None) :
            return eventFitnesses
        if len(eventFitnesses) > 0: t['numdenies'] += 1
        return None

    def searchOrderedListForVal(self, orderedList, val):
        idx = bsc.bisect_left(orderedList, val)
        if idx < len(orderedList) and orderedList[idx] == val: return idx
        return -1

    def testInterceptLegality(self, curEvent, allVectorsTest, baseVectorsTest, t):
        if not curEvent.isOrtho:
            if self.possibleEvents.check_intersections({curEvent.crowVector}, allVectorsTest, postGenTest=True):
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

            if len(allVectorsTest) == 150:
                sfds=""
            for run in runs:
                ortho = self.possibleEvents.orthogonal_vector(curEvent.startHole.coords, curEvent.endHole.coords,
                                                              gp.maxloopyorthoeventdisplacementincrements
                                                              * gp.eventminspacing, run['rev'])
                floor, peak = self.possibleEvents.test_sidestep_events(
                    curEvent.startHole, curEvent.endHole, t['track'].trackholes, t['holecoords'],
                    ortho, gp.maxloopyorthoeventdisplacementincrements * gp.eventminspacing, gp.eventminspacing,
                    allVectorsTest, run['rev'], minIncr=run['minincr'], maxIncr=run['maxincr'], ignoreProximity=True)
                if floor > 0 and floor < bestIncr:
                    bestIncr = floor
                    revOrtho = run['rev']
            if bestIncr == 999 or bestIncr == 0:
                return False, dict(incr=-1, rev=False)
            else:
                #TEMPP
                if len(runs) == 1 or runs[0]['rev'] == revOrtho:
                    run = runs[0]
                else:
                    run = runs[1]
                ortho = self.possibleEvents.orthogonal_vector(curEvent.startHole.coords, curEvent.endHole.coords,
                                                              gp.maxloopyorthoeventdisplacementincrements
                                                              * gp.eventminspacing, revOrtho)
                self.possibleEvents.test_sidestep_events(
                    curEvent.startHole, curEvent.endHole, t['track'].trackholes, t['holecoords'],
                    ortho, gp.maxloopyorthoeventdisplacementincrements * gp.eventminspacing, gp.eventminspacing,
                    allVectorsTest, revOrtho, minIncr=run['minincr'], maxIncr=run['maxincr'], ignoreProximity=True, debugTest=True)
                return True, dict(incr=bestIncr, rev=revOrtho)

    def tryGetEventForHole(self, hole, t, interceptsTestVectors, baseVectorsTest, params, trackEventsOverview):
        # Walk through path, assigning events as per energy buffer
        #Det energy buildup
        while (t['energybufferidx'] < len(t['trackenergycurve']) and
               t['trackenergycurve'][t['energybufferidx']][0] < hole.num):
            t['energybuffer'] += t['trackenergycurve'][t['energybufferidx']][1]
            t['energybufferidx'] += 1
        if t['energybuffer'] < t['candavgenergy'] / params.tryGetParam(t['track_id'],'candenergybufferdivider'): return None

        #Cursor to start of trackhole in event list
        while (t['candcursor'] < len(t['candeventspecs'])
               and t['candeventspecs'][t['candcursor']]['eventtop'] < hole.num):
            t['candcursor'] += 1
        if (t['candcursor'] >= len(t['candeventspecs']) or
                t['candeventspecs'][t['candcursor']]['eventtop'] != hole.num): return None

        #Omit every 8th or so, feathering to avoid getting stuck in optimizer endless loops
        if rd.randint(1,gp.randomfeatheringamount) == 1: return None

        #Skip if need to enforce min spacing to flesh out track
        if t['minspacectr'] < params.tryGetParam(t['track_id'],'enforceminspacing'): return None

        # Factor in distribution of spacing histogram to help ensure even distribution of spacings
        prevNode = 0
        if len(t['eventnodes']) > 0:
            for n in t['eventnodes']:
                if n >= hole.num: break
                prevNode = n
        spacing = hole.num - prevNode
        if spacing > len(t['spacinghisto']):
            for i in range(len(t['spacinghisto']) - 1, spacing):
                #Add in further spacing histos
                t['spacinghisto'].append([i + 1, 0])

        if sum([h[1] for h in t['spacinghisto']]) > 0:
            curSpcPerc = (t['spacinghisto'][spacing - 1][1] /
                          sum([h[1] for h in t['spacinghisto']]))
            # NOTE: for spacing we use all spacings for avg even unpopulated one
            # This is in order to factor in specified ideal deviation
            avgPerc = 1.0 / len(t['spacinghisto'])
            if curSpcPerc - avgPerc > params.tryGetParam(t['track_id'],'eventspacinghistogramscoringfactor'): return None

        #Determine viable event fitnesses
        eventFitnesses = self.scoreEventsForHole(t, hole, t['chutes'], t['chutebases'], t['chutetops'],
                                                        t['ladders'], t['ladderbases'], t['laddertops'], params,
                                                 trackEventsOverview)

        #Find fittest event
        if eventFitnesses is not None and len(eventFitnesses) > 0:
            for fitness in eventFitnesses:
                legal, orthoInst = self.testInterceptLegality(fitness['event'], interceptsTestVectors,baseVectorsTest, t)
                if legal:
                    if orthoInst['incr'] > -1:
                        fitness['event'].instanceIncr = orthoInst['incr']
                        fitness['event'].instanceRev = orthoInst['rev']
                    fitness['lasteventtop'] = prevNode
                    #Score cutoff
                    # if fitness['score'] < t['chosenscorecutoff']:
                    return fitness
                    # else: return None

        return None



    def indexStartOfEachHoleInCands(self, holes, trackEventOverview):
        candEventCursor = 0
        candEventCursorStartLookups = [-1]*len(holes)

        for h in holes:
            while trackEventOverview[candEventCursor]['eventtop'] < h.num:
                candEventCursor += 1
                if candEventCursor >= len(trackEventOverview): break
            if candEventCursor >= len(trackEventOverview): break

            if trackEventOverview[candEventCursor]['eventtop'] == h.num:
                candEventCursorStartLookups[h.num-1] = candEventCursor

        trackEventOverview['candeventstartlookup'] = candEventCursorStartLookups

    def updateVectorsTest(self, allVectorsTest, baseVectorsTest, event, removal, isOrtho):
        if len(allVectorsTest) == 150:
            sfsd = ""
        if not removal:
            if not isOrtho:
                allVectorsTest.update(
                    set(tuple(
                        [v for v in self.boundingBoxPlusVector(event.crowVector)])))

                #Add lumps 20% of the way along so ppl know cant go that way
                if event.instanceIsChute != event.instanceIsLadder:
                    start, end = np.array(event.crowVector[0]), np.array(event.crowVector[1])
                    dist = self.possibleEvents.calculate_distance(event.crowVector[0], event.crowVector[1])
                    if event.instanceIsChute:
                        event.instanceLump = (start + (end - start)*((3/dist) + math.pow(dist, 0.25)/50)).tolist()
                    else:
                        event.instanceLump = (end + (start - end)*((3/dist) + math.pow(dist, 0.25)/50)).tolist()

                baseVectorsTest.add(event.crowVector)
            else:
                event.instanceStartVector = OrthoLineTrace(self.possibleEvents, event, event.instanceIncr,
                                                           event.instanceRev,
                                                               en.OrthoLineTraceType.START).vector
                event.instanceEndVector = OrthoLineTrace(self.possibleEvents, event, event.instanceIncr,
                                                             event.instanceRev,
                                                             en.OrthoLineTraceType.END).vector

                #Add lumps 20% of the way along so ppl know cant go that way
                if event.instanceIsChute != event.instanceIsLadder:
                    vector = event.instanceStartVector if event.instanceIsChute else event.instanceEndVector
                    start, end = np.array(vector[0]), np.array(vector[1])
                    dist = self.possibleEvents.calculate_distance(vector[0], vector[1])
                    event.instanceLump = (start + (end - start) * ((3 / dist) + math.pow(dist, 0.25) / 50)).tolist()

                allVectorsTest.update(
                    set(tuple(
                        [v for v in self.boundingBoxPlusVector(event.instanceStartVector)])))
                allVectorsTest.update(
                    set(tuple(
                        [v for v in self.boundingBoxPlusVector(event.instanceEndVector)])))
                baseVectorsTest.add(event.instanceStartVector)
                baseVectorsTest.add(event.instanceEndVector)
        else:
            if not isOrtho:
                allVectorsTest.difference_update(
                    set(tuple(
                        [v for v in self.boundingBoxPlusVector(event.crowVector)])))
                baseVectorsTest.discard(event.crowVector)
            else:
                if event.instanceIncr > -1:
                    allVectorsTest.difference_update(
                        set(tuple(
                            [v for v in self.boundingBoxPlusVector(event.instanceStartVector)])))
                    allVectorsTest.difference_update(
                        set(tuple(
                            [v for v in self.boundingBoxPlusVector(event.instanceEndVector)])))
                baseVectorsTest.discard(event.instanceStartVector)
                baseVectorsTest.discard(event.instanceEndVector)

    def recalcTrackCompletionPcts(self, trackEventsOverview):
        for t in trackEventsOverview:
            t['trackisstalled'] = t['trackstalledcounter'] > gp.maxitertrackstalled
            t['holescompletepct'] = t['curhole'] / len(t['track'].trackholes)
            t['chutescompletepct'] = len(t['eventsetbuild'])/t['optevents']
        viableTracks = [e for e in trackEventsOverview if not e['trackisstalled']]

        avgHolePct = sum([t['holescompletepct'] for t in viableTracks]) / len(viableTracks)
        avgChutesPct = sum([t['chutescompletepct'] for t in viableTracks])/len(viableTracks)
        return avgHolePct, avgChutesPct

    def tryEventSet(self, params, prevEffLengths):
        """
        Intiial flying blind set, no specs known
        """

        self.board.clearTrackEvents(specificTracks=[t for t in self.board.tracks if not t.instLocked])
        trackEventsOverview = [dict(track=t, trackidx = t.num-1, tracknum=t.num, optevents=0, track_id=t.Track_ID,
                                    optfirstchute=0, trackfilled=False, tracklength=len(t.trackholes),
                                    lengthdeviation=(len(t.trackholes)-gp.effectiveboardlength )/gp.effectiveboardlength,
                                    spacinghisto=[], minspacectr=0, #chosenscorecutoff=100,
                                    eventsetbuild=t.eventSetBuild, candeventspecs=[],
                                    lengthdistidealcurve=[], lengthdistactualhist=[],
                                    lengthovertimeideal=[], maxlength=0,
                                    trackenergycurve=[], trackenergyintegral=[],
                                    candavgenergy=0.0, energybuffer=0.0, energybufferidx = 0, candeventstartlookup=[],
                                    candcursor=0, chutecursor=0, holecoords=[h.coords for h in t.trackholes],
                                    lasteventtop = 0, previsladder=False, chutebases = [], chutetops = [],
                                    ladders=[], chutes=[],
                                    eventnodes=[],twohitsthusfar=0,cancels=0,eventscount=0,
                                    ladderbases=[], laddertops=[], holescompletepct=0.0, chutescompletepct=0.0, curhole=0,
                                    compensationbuffer = 0.0, trackstalledcounter=0, trackisstalled=False,
                                    multistack=[], controllength = 0, curestefflength=len(t.trackholes),
                                    nomultis = False,
                                    numdenies=0, numnogos=0)
                          for t in self.board.tracks if not t.instLocked]

        #Lock out multis if one or more tracks are locked
        if len(trackEventsOverview) != len(self.board.tracks):
            for t in trackEventsOverview: t['nomultis'] = True

        #Load benchmarks
        self.retrieveOrGenerateBenchmarkMoves()

        #Retrieve & normalize energy curve and det integral
        energyCurve, energyNormIntegral = self.getEnergyCurve()

        #Retrieve & normalize length dist hist curve
        normLengthHistDist = self.getNormLengthDistCurve()

        #Retrieve & normalize length dist over time curve
        normLengthOverTimeDist = self.getNormLengthOverTimeCurve()

        #Compute overall figures & charts
        allCands = [c for c in [t['track'].candidateEvents.candidateEvents for t in trackEventsOverview] for c in c]
        self.allTentLengthHisto = []
        for i in range(0,max([c.length for c in allCands])):
            self.allTentLengthHisto.append([i + 1, 0])

        allCandsEnergyPotentialBuilder = 0.0
        for i in range(0, len(trackEventsOverview)):
            allCandsEnergyPotentialBuilder += sum([c.length * (
                2 if (True if c.startHole.num < params.tryGetParam(trackEventsOverview[i]['track_id'],'ladderscanstartat')
                      else c.canBeLadder) else 1)
                 for c in trackEventsOverview[i]['track'].candidateEvents.candidateEvents])
        avgOverallCandEnergyPotential = allCandsEnergyPotentialBuilder/len(allCands)


        #Iterate over tracks, create event when energy buildup exceeds req

        for t in trackEventsOverview:
            #Determine control lengths with blank track
            t['controllength'] = self.runPartialTrackEffLengthHoles(t['track_id'], [],
                                                                                t['tracklength'],
                                                                                     readMode=True)[0]

            # Create track-specific energy curve
            candEventSpecs = [dict(event=c, isshared=c.isShared, eventtop=c.endHole.num, eventbase=c.startHole.num,
                                   length=c.length,
                                   canbeladder=False if c.startHole.num <
                                                        params.tryGetParam(t['track_id'],'ladderscanstartat')
                                   else c.canBeLadder)
                              for c in t['track'].candidateEvents.candidateEvents]
            candEventSpecs.sort(key=lambda c: (c['eventtop'], c['length']))
            t['candeventspecs'] = candEventSpecs

            # NOTE: energy counts double for chutes + ladders, since energy is defined by position modulation force
            candEnergyPotential = (sum([c['length'] for c in candEventSpecs]) +
                                   sum([c['length'] for c in candEventSpecs if c['canbeladder']]))
            candAvgEnergy = candEnergyPotential / len(candEventSpecs)
            t['candavgenergy'] = candAvgEnergy

            #If the avg cand nrg is more than global avg, fewer events & vice versa
            candEnergySkew = ((candAvgEnergy-avgOverallCandEnergyPotential)/
                              (avgOverallCandEnergyPotential*params.tryGetParam(t['track_id'],'candenergyskewdiminisher')))
            t['optevents'] = int(params.tryGetParam(t['track_id'],'baseopteventspertrack')*(1.0-candEnergySkew))
            t['optfirstchute'] = int(params.tryGetParam(t['track_id'],'baseoptfirstchute')*(1.0+candEnergySkew))

            #Set up ideal length distribution curve
            discrLengthDistCurve = self.discretizeCurve(normLengthHistDist, max([c['length'] for c in candEventSpecs]))
            t['lengthdistidealcurve'] = self.actualizeCurve(discrLengthDistCurve, 1,
                                                      t['optevents']/sum([n[1] for n in discrLengthDistCurve]))
            t['lengthdistactualhist'] = []
            for i in range(0,  max([c['length'] for c in candEventSpecs])):
                t['lengthdistactualhist'].append([i + 1, 0])

            # Set up ideal length over time curve
            discrLengthOverTimeCurve = self.discretizeCurve(normLengthOverTimeDist,
                                                        len(t['track'].trackholes))
            t['lengthovertimeideal'] = self.actualizeCurve(discrLengthOverTimeCurve, 1,
                                                            max([c['length'] for c in candEventSpecs]))
            t['maxlength'] = max([c['length'] for c in candEventSpecs])

            # Set up spacing histogram to help ensure even distribution
            t['spacinghisto'] = []
            for i in range(0, int((t['optevents']/len(t['track'].trackholes))*params.tryGetParam(t['track_id'],
                                                                                                 'eventspacingdeviationfactor'))):
                t['spacinghisto'].append([i+1, 0])

            normTrackCurveNetEnergy = sum([c[1] for c in energyCurve])
            trackEnergyCurve = self.actualizeCurve(energyCurve, t['track'].length,
                                                   (candAvgEnergy*t['optevents'])/normTrackCurveNetEnergy)
            t['trackenergycurve'] = trackEnergyCurve
            trackEnergyIntegral = self.actualizeCurve(energyNormIntegral, t['track'].length,
                                                      candAvgEnergy * t['optevents'], integrate=True)
            t['trackenergyintegral'] = trackEnergyIntegral
            t['compensationbuffer'] = t['lengthdeviation']*gp.effectiveboardlength
            t['track'].setTentativeEvents([])
            t['eventsetbuild'] = t['track'].eventSetBuild

        #Initial pass, try to populate tracks in tandem
        avgHolePct, avgChutesPct = 0.0, 0.0
        allVectorsTest = set()
        baseVectorsTest = set()
        allTentative, allDirectTentative, allOrthoTentative = [], [], []
        stallCounter = 0
        while (len([t for t in trackEventsOverview if t['holescompletepct'] <
                                                      1.0*params.tryGetParam(t['track_id'],
                                                                             'holescompletetrackallowablecutoff')]) > 0 and
               len([t for t in trackEventsOverview if t['chutescompletepct'] <
                                                      1.0*params.tryGetParam(t['track_id'],'maxchuteoverdrivepct')]) > 0 and
               stallCounter <= gp.maxitertrynewbuild):
            # if len([t for t in trackEventsOverview if t['holescompletepct'] < 0.9]) == 0:
            #     sfds=""
            if avgHolePct > 0.5:
            #     test = [t for t in trackEventsOverview if t['holescompletepct'] < 1.0*params.maxchuteoverdrivepct]
                 sdffds=""
            allTracksStalled = True
            for t in trackEventsOverview:
                #NOTE: factoring for roundoff error w/ chutes pct
                t['trackstalledcounter'] += 1
                while t['chutescompletepct'] <= avgChutesPct + 0.001 and t['curhole'] < len(t['track'].trackholes):
                    allTracksStalled = False
                    idealEventWithFitness = None
                    isSharePop = False
                    if len(t['multistack']) == 0:
                        #Find new event
                        t['curhole'] += 1
                        t['minspacectr'] += 1
                        curHoleObj = t['track'].getHoleByNum(t['curhole'])
                        idealEventWithFitness = self.tryGetEventForHole(curHoleObj, t, allVectorsTest, baseVectorsTest,
                                                                        params, trackEventsOverview)
                    else:
                        #Pop queued multi event
                        idealEventWithFitness = t['multistack'].pop()
                        isSharePop = True
                        prevNode = 0
                        if len(t['eventnodes']) > 0:
                            for n in t['eventnodes']:
                                if n >= idealEventWithFitness['eventspecs']['eventtop']: break
                                prevNode = n
                        idealEventWithFitness['lasteventtop'] = prevNode
                        spacing = idealEventWithFitness['eventspecs']['eventtop'] - prevNode
                        if spacing > len(t['spacinghisto']):
                            for i in range(len(t['spacinghisto']) - 1, spacing):
                                # Add in further spacing histos
                                t['spacinghisto'].append([i + 1, 0])

                    if idealEventWithFitness is not None:
                        t['trackstalledcounter'] = 0
                        t['trackisstalled'] = False
                        t['minspacectr'] = 0
                        # t['chosenscorecutoff'] = 0.3*idealEventWithFitness['score']*10.0 + 0.7*t['chosenscorecutoff']
                        # if len(allVectorsTest) > 50 and len(allVectorsTest) % 100 <= 5:
                        # if len(allVectorsTest) >  155:
                        #     sdf=""
                        #     self.testPlotVectorsOnHoles(allVectorsTest)



                        #Great success!  Add event & update sets
                        t['track'].addTentativeEvent(idealEventWithFitness['event'])
                        self.allTentLengthHisto[idealEventWithFitness['eventspecs']['length'] - 1][1] += 1
                        t['lengthdistactualhist'][idealEventWithFitness['eventspecs']['length'] - 1][1] += 1
                        #NOTE: we subtract energy, but add in modulation
                        t['energybuffer'] -= idealEventWithFitness['effnetenergy']
                        #NOTE: we SUBTRACT, since boosters are (+) (decrease eff board length)
                        #...and impeders are (-) (increase eff board length)
                        t['compensationbuffer'] += idealEventWithFitness['effcompmodulation']
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
                        if curEvent.instanceCancel: t['cancels'] += 1
                        t['eventscount'] += 1

                        self.updateVectorsTest(allVectorsTest, baseVectorsTest, curEvent, False, isOrtho)
                        t['twohitsthusfar'] += idealEventWithFitness['twohits']
                        t['curestefflength'] = idealEventWithFitness['estefflength']
                        if idealEventWithFitness['instchute']:
                            t['chutes'].append(dict(chutetop=idealEventWithFitness['eventspecs']['eventtop'],
                                                     length=idealEventWithFitness['eventspecs']['length']))
                            t['chutes'].sort(key=lambda l: l['chutetop'])
                            t['chutebases'].append(idealEventWithFitness['eventspecs']['eventbase'])
                            t['chutebases'].sort()
                            t['chutetops'].append(idealEventWithFitness['eventspecs']['eventtop'])
                            t['chutetops'].sort()
                        t['eventnodes'].extend([idealEventWithFitness['eventspecs']['eventbase'],
                                               idealEventWithFitness['eventspecs']['eventtop']])
                        t['eventnodes'].sort()
                        if idealEventWithFitness['instladder']:
                            t['ladders'].append(dict(ladderbase=idealEventWithFitness['eventspecs']['eventbase'],
                                                     length=idealEventWithFitness['eventspecs']['length']))
                            t['ladders'].sort(key=lambda l: l['ladderbase'])
                            t['ladderbases'].append(idealEventWithFitness['eventspecs']['eventbase'])
                            t['ladderbases'].sort()
                            t['laddertops'].append(idealEventWithFitness['eventspecs']['eventtop'])
                            t['laddertops'].sort()
                        t['previsladder'] = idealEventWithFitness['instladder']
                        t['spacinghisto'][(idealEventWithFitness['eventspecs']['eventtop']
                                                                     - idealEventWithFitness['lasteventtop']) - 1][1] += 1
                        t['lasteventtop'] = idealEventWithFitness['eventspecs']['eventtop']

                        if curEvent.instanceIsChute != curEvent.instanceIsLadder: self.cancels += 1
                        if curEvent.isOrtho: self.orthos += 1
                        if curEvent.isShared: self.multis += 1
                        self.events += 1

                        if not isSharePop and curEvent.isShared:
                            for ev in curEvent.linkedEvents:
                                t_sub = next((t_sub for t_sub in trackEventsOverview
                                              if t_sub['tracknum'] == ev.trackNum), None)
                                if t_sub is None:
                                    raise Exception("Multi event not linked up to track! 0_o")
                                linkedEventSpecs = next((specs for specs in t_sub['candeventspecs']
                                              if specs['event'] == ev), None)
                                if linkedEventSpecs is None:
                                    raise Exception("Candidate event specs not found for event 0_o")
                                topHole = curEvent.endHole
                                linkedEventWithScore = self.scoreEventsForHole(t_sub, topHole, t_sub['chutes'],
                                                                               t_sub['chutebases'], t_sub['chutetops'],
                                                                               t_sub['ladders'], t_sub['ladderbases'],
                                                                               t_sub['laddertops'], params,
                                                                               trackEventsOverview,
                                                                                linkedEventSpecs,
                                                                               idealEventWithFitness['instchute'],
                                                                               idealEventWithFitness['instladder'])[0]
                                t_sub['multistack'].append(linkedEventWithScore)

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
            effLength = self.runPartialTrackEffLengthHoles(t['track_id'], t['eventsetbuild'],
                                                           t['tracklength'],
                                                           readMode=True)[0] * (t['tracklength'] / t['controllength'])
            if abs(effLength - gp.effectiveboardlength) <= gp.minqualityboardlengthmatching:
                #Lock this in!
                t['track'].instLocked = True

            effLengths.append(dict(trackeventoverview=t, track_id=t['track_id'],
                                   efflength=effLength, tracklocked=t['track'].instLocked))
            sortedNodes = t['eventnodes']
            sortedNodes.sort()

            nodesFound = False
            if len(self.eventNodesByTrack) > 0:
                for t_node in self.eventNodesByTrack:
                    if t_node['tracknum'] == t['tracknum']:
                        t_node['nodes'] = sortedNodes
                        nodesFound = True
                        break
            if not nodesFound: self.eventNodesByTrack.append(dict(tracknum=t['tracknum'], nodes=sortedNodes))

            print("{} chutes, {} ladders, {} events; ctl: {} ltc: {}".format(len(t['chutes']), len(t['ladders']),
                                                                             len(t['eventsetbuild']),
                                                                  len(t['chutes'])/ len(t['ladders']),
                                                                  len(t['ladders'])/ len(t['chutes'])))
            print("Two hits: {}".format(t['twohitsthusfar']))
            print("{} nogos, {} denies".format(t['numnogos'], t['numdenies']))

        avgEffLength = sum(l['efflength'] for l in effLengths)/len(effLengths)

        for l in effLengths:
            print("Track {} has effective length of {}, which should yield an approx {} balance"
                  .format(l['track_id'], l['efflength'],
                          (gp.effectiveboardlength-l['efflength'])/gp.effectiveboardlength))

        if (max([abs(l['efflength'] - gp.effectiveboardlength) for l in effLengths])
                > gp.minqualityboardlengthmatching):
            #Massage balanceandefflengthcontrolfactor and retry
            #Longer balanceandefflengthcontrolfactor for longer route
            for l in effLengths:
                if l['tracklocked']: continue
                oldVal = self.paramSet.tryGetParam(l['track_id'], "balanceandefflengthcontrolfactor")
                newVal = oldVal
                #Take avg of this eff length and prev, to smooth out jumps
                prevEffLength, prevEffIdx = gp.effectiveboardlength, -1
                for l_prv_idx in range(0, len(prevEffLengths)):
                    if prevEffLengths[l_prv_idx]['track_id'] == l['track_id']:
                        prevEffIdx = l_prv_idx
                        prevEffLength = prevEffLengths[l_prv_idx]['efflength']
                        break

                if (prevEffLength + l['efflength'])/2 < gp.effectiveboardlength - gp.minqualityboardlengthmatching:
                    #Increase factor to lengthen board
                    newVal = oldVal + gp.minqualityboardlengthintervalsrpt
                elif (prevEffLength + l['efflength'])/2 > gp.effectiveboardlength + gp.minqualityboardlengthmatching:
                    #Decrease factor to shorten board
                    newVal = oldVal - gp.minqualityboardlengthintervalsrpt
                if newVal > 0.95: newVal = 0.95
                elif newVal < 0.05: newVal = 0.05
                self.paramSet.tryModParam(l['track_id'], "balanceandefflengthcontrolfactor", newVal)
                prevEffLengths[prevEffIdx]['efflength'] = l['efflength']

            print("Retry, not good enough board eff length quality\n")
            return False


        #Try to bring all sets down to lowest one! since cannot really decrease eff length at this point
        #Given set of ladders, assume liklihood of landing on that square is 1/trackholes
        #therefore try to find a ladder closest to the rem buffer to cancel
        # effShortestTrackBuff = min([t['compensationbuffer'] for t in trackEventsOverview])
        # for t in trackEventsOverview:
            # trackLadders = [l for l in t['eventsetbuild'] if l.instanceIsLadder]
            # trackLadders.sort(key = lambda l: l.length)
            # trackLadderLengths = [l.length for l in trackLadders]
            #
            # effIncrease = int(t['compensationbuffer'] - effShortestTrackBuff)
            # if effIncrease == 0:
            #     t['trackfilled'] = True
            #     continue
            #
            # idx = self.searchOrderedListForVal(trackLadderLengths, effIncrease)
            # if idx > -1:
            #     #Success! cancel this ladder and move on
            #     trackLadders[idx].instanceIsLadder = False
            #     t['trackfilled'] = True
            # else:
            #     effIncr1 = math.floor(effIncrease/2)
            #     effIncr2 = math.floor(effIncrease/2) + effIncrease % 2
            #     while effIncr1 > 0:
            #         idx1 = self.searchOrderedListForVal(trackLadderLengths, effIncr1)
            #         if idx1 > -1:
            #             if effIncr1 == effIncr2:
            #                 #Look left
            #                 if idx1 > 0 and trackLadderLengths[idx-1] == effIncr2:
            #                     trackLadders[idx1].instanceIsLadder = False
            #                     trackLadders[idx1-1].instanceIsLadder = False
            #                     t['trackfilled'] = True
            #                     break
            #                 #Look right
            #                 elif idx1 < len(trackLadderLengths) - 1 and trackLadderLengths[idx+1] == effIncr2:
            #                     trackLadders[idx1].instanceIsLadder = False
            #                     trackLadders[idx1+1].instanceIsLadder = False
            #                     t['trackfilled'] = True
            #                     break
            #             else:
            #                 idx2 = self.searchOrderedListForVal(trackLadderLengths, effIncr2)
            #                 if idx2 > -1:
            #                     trackLadders[idx1].instanceIsLadder = False
            #                     trackLadders[idx2].instanceIsLadder = False
            #                     t['trackfilled'] = True
            #                     break
            #         effIncr1 -= 1
            #         effIncr2 += 1


        # return len([t for t in trackEventsOverview if not t['trackfilled']]) == 0
        return True

    def testPlotVectorsOnHoles(self, vectors):
        plt.figure(figsize=(15, 10))
        for vector in vectors:
            x_values = [vector[0][0], vector[1][0]]
            y_values = [vector[0][1], vector[1][1]]
            plt.plot(x_values, y_values)

        for t in self.board.tracks:
            coordinates = [h.coords for h in t.trackholes]
            x_coords, y_coords = zip(*coordinates)
            plt.scatter(x_coords, y_coords, marker='o')

        #Add labels
        for t in self.board.tracks:
            plt.annotate(str(t.Track_ID), t.trackholes[0].coords)
            for c in t.trackholes:
                if c.num % 5 == 0:
                    plt.annotate(str(c.num), c.coords)

        plt.show()
        plt.waitforbuttonpress()

    def buildSetIntoEvents(self):
        for t in self.board.tracks:
            for e in t.eventSetBuild:
                if e.instanceIsLadder: t.addLadder(bd.Ladder(e.startHole.num, e.endHole.num, t.num, e.crowVector, e))
                t.addChute(bd.Chute(e.endHole.num, e.startHole.num, t.num, e.crowVector, e))
            t.setEventLadders([l.start for l in t.ladders])
            t.setEventChutes([c.start for c in t.chutes])

            #Set descriptive stats
            t.setEventImpedance()


    def plot_coordinates_and_vectors(self, bitmap_name='output_bitmap.png'):
        """
        Plots multiple sets of coordinates and vectors, and saves the plot as a bitmap image.

        Parameters:
            coordinate_sets (list of lists): A list where each element is a list of coordinates (x, y) to be plotted.
            vector_sets (list of lists): A list where each element is a list of vectors ((x1, y1), (x2, y2)) to be plotted.
            bitmap_name (str): The name of the output bitmap file.
        """


        plt.figure(figsize=(20, 20))
        coordinate_sets = []
        path_dot_vectors = []
        vector_sets = []
        x_marks = []
        for t in self.board.tracks:
            holes = [h.coords for h in t.trackholes]
            coordinate_sets.append(holes)
            for c_idx in range(len(holes) - 1):
                path_dot_vectors.append((holes[c_idx], holes[c_idx+1]))
            trackVectorSet = []
            for l, ch in zip([True, True, False], [True, False, True]):
                trackVectorSubset = []
                trackVectorSubset.extend([c.crowVector for c in t.chutes if not c.eventDete.isOrtho
                                       and c.eventDete.instanceIsLadder == l and c.eventDete.instanceIsChute == ch])
                trackVectorSubset.extend([c.eventDete.instanceStartVector for c in t.chutes if c.eventDete.isOrtho
                                       and c.eventDete.instanceIsLadder == l and c.eventDete.instanceIsChute == ch])
                trackVectorSubset.extend([c.eventDete.instanceEndVector for c in t.chutes if c.eventDete.isOrtho
                                       and c.eventDete.instanceIsLadder == l and c.eventDete.instanceIsChute == ch])
                trackVectorSet.append(trackVectorSubset)

                if l and not ch:
                    x_marks.extend([c.crowVector[1] for c in t.chutes if not c.eventDete.isOrtho
                                       and c.eventDete.instanceIsLadder == l and c.eventDete.instanceIsChute == ch])
                    x_marks.extend([c.eventDete.instanceEndVector[0] for c in t.chutes if c.eventDete.isOrtho
                                       and c.eventDete.instanceIsLadder == l and c.eventDete.instanceIsChute == ch])
                elif ch and not l:
                    x_marks.extend([c.crowVector[0] for c in t.chutes if not c.eventDete.isOrtho
                                       and c.eventDete.instanceIsLadder == l and c.eventDete.instanceIsChute == ch])
                    x_marks.extend([c.eventDete.instanceStartVector[0] for c in t.chutes if c.eventDete.isOrtho
                                       and c.eventDete.instanceIsLadder == l and c.eventDete.instanceIsChute == ch])

            vector_sets.append(trackVectorSet)

        # Plot each set of coordinates
        for coordinates in coordinate_sets:
            x_coords, y_coords = zip(*coordinates)
            plt.scatter(x_coords, y_coords, marker='o')

        # Plot tracks in fine dots
        for vector in path_dot_vectors:
            x_values = [vector[0][0], vector[1][0]]
            y_values = [vector[0][1], vector[1][1]]
            plt.plot(x_values, y_values, linestyle=':', color='black', linewidth=1)

        #Plot lumps for cannot enter
        lumps = []
        lumps.extend([tuple(c.eventDete.instanceLump) for c in [c for t in self.board.tracks for c in t.chutes]
                      if c.eventDete.instanceLump != (-1, -1)])
        lumps.extend([tuple(l.eventDete.instanceLump) for l in [l for t in self.board.tracks for l in t.ladders]
                      if l.eventDete.instanceLump != (-1, -1)])
        for coordinates in lumps:
            x_coords, y_coords = coordinates[0], coordinates[1]
            plt.scatter(x_coords, y_coords, marker="s")

        #Add labels
        for t in self.board.tracks:
            plt.annotate(str(t.Track_ID), t.trackholes[0].coords)
            for c in t.trackholes:
                if c.num % 5 == 0:
                    plt.annotate(str(c.num), c.coords)

        # Plot each set of vectors
        colourCounter = 0
        colours = [(.5, 0, 0), (1, 0.5, 0), (1, 0, 0.5),
                   (0, .5, 0), (0.5, 1, 0), (0, 1, 0.5),
                   (0, 0, .5), (0, 0.5, 1), (0.5, 0, 1)]
        for vector_subset in vector_sets:
            for vectors in vector_subset:
                for vector in vectors:
                    x_values = [vector[0][0], vector[1][0]]
                    y_values = [vector[0][1], vector[1][1]]
                    plt.plot(x_values, y_values, color=colours[colourCounter])  # 'r-' means red line
                colourCounter += 1

        #Ploy x's on vectors for ladders-only & chutes-onlly
        # for c in x_marks:
        #     plt.annotate("x", c, fontsize=18)

        # Set the axes' limits to fit all points and vectors nicely
        all_x = [coord[0] for coordinates in coordinate_sets for coord in coordinates]
        all_y = [coord[1] for coordinates in coordinate_sets for coord in coordinates]
        plt.xlim([min(all_x) - 1, max(all_x) + 1])
        plt.ylim([min(all_y) - 1, max(all_y) + 1])

        # Save the plot as a bitmap image
        plt.savefig(bitmap_name, format='png')
        plt.show()
        plt.waitforbuttonpress()
        # plt.close()

class OrthoPath:
    def __init__(self, start, mid, end, incr, rev, event):
        self.start = start
        self.mid = mid
        self.end = end

        self.incr = incr
        self.rev = rev
        self.event = event

class OrthoLineTrace:
    def __init__(self, possibleEvents, event, incr, rev, type):
        self.event = event
        self.incr = incr
        self.rev = rev
        self.type = type
        self.vector = ((-1,-1),(-1,-1))

        p1,p2=(-1,-1),(-1,-1)
        midpoint = tuple([sum(c) / 2 for c in zip(self.event.startHole.coords, self.event.endHole.coords)])
        orthogonal_vector = tuple([(-1 if rev else 1)* o for o in self.event.orthoVector])
        orthogonal_vector = possibleEvents.orthogonal_vector(self.event.startHole.coords, self.event.endHole.coords,
                                                             gp.maxloopyorthoeventdisplacementincrements
                                                             * gp.eventminspacing, rev)
        length_divider = incr/gp.maxloopyorthoeventdisplacementincrements
        match type:
            case en.OrthoLineTraceType.START:
                p1 = self.event.startHole.coords
            case en.OrthoLineTraceType.END:
                p1 = self.event.endHole.coords
            case _:
                raise Exception("No ortho line trace type specified!")

        p2 = (midpoint[0] + orthogonal_vector[0] * length_divider,
              midpoint[1] + orthogonal_vector[1] * length_divider)

        self.vector = (p1, p2)
    def __key(self):
        return (self.event, self.rev, self.incr)
    #NOTE: do not put objects of multiple types in a set!!!

    def __hash__(self):
        return hash(self.__key())

    def __eq__(self, other):
        if isinstance(other, OrthoLineTrace):
            return self.__key() == other.__key()
        return NotImplemented
    def __lt__(self, other):
        # Define the comparison order
        if self.event != other.event:
            return self.event < other.event
        if self.rev != other.rev:
            return self.rev < other.rev
        return self.incr < other.incr

class ParamSet:
    def __init__(self, board, tracks):
        # HOLE AND LENGTH BASED INTS:
        # baseopteventspertrack
        # ladderscanstartat
        # baseoptfirstchute
        # RELATIVE
        # candenergyskewdiminisher - Divides skew by this amount, maintain convergence
        # maxchuteoverdrivepct
        # holescompletetrackallowablecutoff
        # lengthhistogramscoringfactor
        # eventspacingdeviationfactor - Higher means more deviation in event spacing
        # eventspacinghistogramscoringfactor - Lower means less weight put upon it, MAX around 0.4 or 0.5!
        # candenergybufferdivider
        # move1allowanceratio INACTIVE
        # lengthhistogramscoringfactor - Lower means less weight put upon it
        # lengthovertimescoringfactor - Lower means less weight put upon it
        # disallowbelowsetlength - HELLA override! track spec only probs
        # maxorthoratio
        # minladdertochuteratio
        # minchutetoladderratio
        # twohitfreqimpedance - higher means more slowing of two-hit event combos
        self.board = board
        self.tracks = tracks
        self.params = []

    def intakeParams(self, instanceParams_df):
        self.params = []
        #Cursor thru params setting as needed
        for index, param_sr in instanceParams_df.iterrows():
            # Prioritize track override if exists
            self.params.append(dict(track_id=param_sr['track_id'], param=param_sr['param'], value=param_sr['value']))

    def intakeParamsFromDb(self, optimizerRunSet, optimizerRun):
        with contextlib.closing(sql.connect('etc/Optimizer.db')) as sqlConn:
            with sqlConn:
                self.params = []
                paramsQuery_sb = StringIO()
                paramsQuery_sb.write("SELECT p.*")
                paramsQuery_sb.write(" from OptimizerRunTestParams p ")
                paramsQuery_sb.write("inner join OptimizerRuns o ")
                paramsQuery_sb.write("on o.OptimizerRun = p.OptimizerRun ")
                paramsQuery_sb.write("inner join OptimizerRunSets os ")
                paramsQuery_sb.write("on os.OptimizerRunSet = o.OptimizerRunSet ")
                paramsQuery_sb.write("where os.OptimizerRunSet = ? ")
                paramsQuery_sb.write("and o.OptimizerRun = ? ")
                params_df = pd.read_sql_query(paramsQuery_sb.getvalue(), sqlConn,
                                                   params=[optimizerRunSet, optimizerRun])
                #Cursor thru params setting as needed
                for index, param_sr in params_df.iterrows():
                    # Prioritize track override if exists
                    self.params.append(dict(track_id=param_sr['Track_ID'], param=param_sr['Param'],
                                            value=param_sr['InstanceParamValue']))


    def midpointInitParams(self):
        with contextlib.closing(sql.connect('etc/Optimizer.db')) as sqlConn:
            with sqlConn:
                with contextlib.closing(sqlConn.cursor()) as sqliteCursor:
                    #Retrieve base values from db
                    query = "SELECT * FROM BoardTrackHints WHERE Board_ID = ? AND Track_ID = ? AND Active = ?"
                    sqliteCursor.execute(query, [self.board.boardID, 0, 1])
                    boardparamranges_df = pd.DataFrame(sqliteCursor.fetchall(),
                                                      columns=[d[0] for d in sqliteCursor.description])
                    if len(boardparamranges_df) == 0:
                        raise Exception("No param bounds found for board ID {}".format(self.board.boardID))

                    # Set params from ranges for board
                    self.params = []
                    # for index, param_sr in boardparamranges_df.iterrows():
                    #     # Prioritize track override if exists
                    #     if param_sr['isInt'] == 1:
                    #         curVal = rd.randint(int(param_sr['LBound']), int(param_sr['UBound']))
                    #     else:
                    #         curVal = rd.uniform(param_sr['LBound'], param_sr['UBound'])
                    #     self.params.append(dict(track_id=0, param=param_sr['Param'], value=curVal))

                    for t in self.tracks:
                        # Try to retrieve overrides if exist
                        query = "SELECT * FROM BoardTrackHints WHERE Board_ID = ? AND Track_ID = ? AND Active = ?"
                        sqliteCursor.execute(query, [self.board.boardID, t.Track_ID, 1])
                        trackparamranges_df = pd.concat([pd.DataFrame(sqliteCursor.fetchall(),
                                                          columns=[d[0] for d in sqliteCursor.description]),
                                                         boardparamranges_df])
                        trackparamranges_df.sort_values(['Param', 'Track_ID'], inplace=True,
                                                        ascending=False)

                        #Set params from ranges
                        prevParam = ""
                        for index, param_sr in trackparamranges_df.iterrows():
                            #Prioritize track override if exists
                            if param_sr['Param'] == prevParam: continue
                            if param_sr['isInt'] == 1:
                                curVal = (int(param_sr['LBound']) + int(param_sr['UBound']))//2
                            else:
                                curVal = (param_sr['LBound'] + param_sr['UBound'])/2
                            self.params.append(dict(track_id=t.Track_ID, param=param_sr['Param'], value=curVal))
                            prevParam = param_sr['Param']

    def monteCarlo(self):
        with contextlib.closing(sql.connect('etc/Optimizer.db')) as sqlConn:
            with sqlConn:
                with contextlib.closing(sqlConn.cursor()) as sqliteCursor:
                    #Retrieve base values from db
                    query = "SELECT * FROM BoardTrackHints WHERE Board_ID = ? AND Track_ID = ? AND Active = ?"
                    sqliteCursor.execute(query, [self.board.boardID, 0, 1])
                    boardparamranges_df = pd.DataFrame(sqliteCursor.fetchall(),
                                                      columns=[d[0] for d in sqliteCursor.description])
                    if len(boardparamranges_df) == 0:
                        raise Exception("No param bounds found for board ID {}".format(self.board.boardID))
            
                    # Set params from ranges for board
                    self.params = []
                    # for index, param_sr in boardparamranges_df.iterrows():
                    #     # Prioritize track override if exists
                    #     if param_sr['isInt'] == 1:
                    #         curVal = rd.randint(int(param_sr['LBound']), int(param_sr['UBound']))
                    #     else:
                    #         curVal = rd.uniform(param_sr['LBound'], param_sr['UBound'])
                    #     self.params.append(dict(track_id=0, param=param_sr['Param'], value=curVal))
            
                    for t in self.tracks:
                        # Try to retrieve overrides if exist
                        query = "SELECT * FROM BoardTrackHints WHERE Board_ID = ? AND Track_ID = ? AND Active = ?"
                        sqliteCursor.execute(query, [self.board.boardID, t.Track_ID, 1])
                        trackparamranges_df = pd.concat([pd.DataFrame(sqliteCursor.fetchall(),
                                                          columns=[d[0] for d in sqliteCursor.description]),
                                                         boardparamranges_df])
                        trackparamranges_df.sort_values(['Param', 'Track_ID'], inplace=True,
                                                        ascending=False)
            
                        #Set params from ranges
                        prevParam = ""
                        for index, param_sr in trackparamranges_df.iterrows():
                            #Prioritize track override if exists
                            if param_sr['Param'] == prevParam: continue
                            if param_sr['isInt'] == 1:
                                curVal = rd.randint(int(param_sr['LBound']), int(param_sr['UBound']))
                            else:
                                curVal = rd.uniform(param_sr['LBound'], param_sr['UBound'])
                            self.params.append(dict(track_id=t.Track_ID, param=param_sr['Param'], value=curVal))
                            prevParam = param_sr['Param']


    def tempInsertParamsDb(self, optimizerRunSet, optimizerRun):
        with contextlib.closing(sql.connect('etc/Optimizer.db')) as sqlConn:
            with sqlConn:
                with contextlib.closing(sqlConn.cursor()) as sqliteCursor:
                    query = "INSERT INTO OptimizerRuns (OptimizerRunSet, OptimizerRun, Board_ID, Timestamp) VALUES (?,?,?,?)"
                    sqliteCursor.execute(query, [optimizerRunSet, optimizerRun, self.board.boardID, dt.now().strftime('%m/%d/%y %H:%M:%S')])
                    sqlConn.commit()
            
                    params_df = pd.DataFrame.from_records(self.params)
                    # params_df.rename(columns={"track_id": "Track_ID", "param": "Param", "value": "InstanceParamValue"})
                    params_df['Board_ID'] = self.board.boardID
                    params_df['OptimizerRunSet'] = optimizerRunSet
                    params_df['OptimizerRun'] = optimizerRun
            
                    # Insert into data table
                    params_l = params_df.values.tolist()
                    paramsQuery_sb = StringIO()
                    paramsQuery_sb.write(
                        "INSERT INTO OptimizerRunTestParams(Track_ID, Param, InstanceParamValue, Board_ID, OptimizerRunSet, OptimizerRun) values(?,?,?,?,?,?)")
                    sqliteCursor.execute("BEGIN TRANSACTION")
                    for index, record in params_df.iterrows():
                        sqliteCursor.execute(paramsQuery_sb.getvalue(), [record['track_id'], record['param'],
                                                                              record['value'], record['Board_ID'],
                                                                              record['OptimizerRunSet'],record['OptimizerRun']])
                    sqliteCursor.execute("END TRANSACTION")
            
                    sqlConn.commit()
                    paramsQuery_sb.close()

    def tempWriteMetricsToDb(self, evaluator):
        with contextlib.closing(sql.connect('etc/Optimizer.db')) as sqlConn:
            with sqlConn:
                with contextlib.closing(sqlConn.cursor()) as sqliteCursor:

                    # Input metric info
                    metrics_df = pd.DataFrame.from_records(evaluator.results)
                    metrics_df['WeightedValue'] = metrics_df['Weighting']*metrics_df['ResultValue']
                    metrics_df.drop(['Weighting', 'ResultValueIterative'], axis=1, inplace=True)
                    metrics_df['OptimizerRunSet'] = evaluator.optimizerRunSet
                    metrics_df['OptimizerRun'] = evaluator.optimizerRun
                    metrics_df['Board_ID'] = evaluator.board.boardID

                    # Insert into data table
                    metrics_l = metrics_df.values.tolist()
                    metricsQuery_sb = StringIO()
                    metricsQuery_sb.write("INSERT INTO OptimizerRunResults (")
                    cols = metrics_df.columns.values.tolist()
                    for c in range(0, len(cols) - 1):
                        metricsQuery_sb.write(cols[c])
                        metricsQuery_sb.write(", ")
                    metricsQuery_sb.write(cols[len(cols) - 1])
                    metricsQuery_sb.write(") VALUES (")
                    metricsQuery_sb.write(", ".join(['?'] * len(cols)))
                    metricsQuery_sb.write(")")
                    sqliteCursor.execute("BEGIN TRANSACTION")
                    # sqliteCursor.execute("select * from  Testtest")
                    for index, record in metrics_df.iterrows():
                        sqliteCursor.execute(metricsQuery_sb.getvalue(), [record['Result'], record['ResultFlavour'],
                                                                          record['ResultValue'], record['WeightedValue'],
                                                                          record['OptimizerRunSet'],
                                                                          record['OptimizerRun'], record['Board_ID']])

                    sqliteCursor.execute("END TRANSACTION")
                    # sqliteCursor.executemany(metricsQuery_sb.getvalue(), metrics_df)
                    sqlConn.commit()
                    metricsQuery_sb.close()

    def tempWriteEvents(self, stats, optimizerRunSet, optimizerRun):
        with contextlib.closing(sql.connect('etc/Optimizer.db')) as sqlConn:
            with sqlConn:
                with contextlib.closing(sqlConn.cursor()) as sqliteCursor:
                    # Cache events hit in db
                    eventshit_df = pd.DataFrame.from_records([m.to_dict() for m in stats.moves if m.ladderorchuteamt != 0])
                    query = "INSERT INTO EventHit Values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
                    sqliteCursor.execute("BEGIN TRANSACTION")
                    for idx, move in eventshit_df.iterrows():
                        sqliteCursor.execute(query, (optimizerRunSet, optimizerRun, self.board.boardID,
                                                     move['trial'], move['track_id'], move['threadnum'],move['movenum'],
                                                     move['currpos'] - move['score'] + move['basescore'], move['currpos'],
                                                     move['score'] - move['basescore']))

                    sqliteCursor.execute("END TRANSACTION")
                    sqlConn.commit()

    def modParamsForFmin(self, paramsSubset, fminParamsList):
        allParams_df = pd.DataFrame.from_records(self.params)
        allParams_df.set_index(['param'], inplace=True)
        allParams_df.sort_index(inplace=True)
        for idx in range(0, len(paramsSubset)):
            param_df = allParams_df.loc[fminParamsList[idx]['param']]
            if isinstance(param_df, pd.Series): param_df = pd.DataFrame(param_df)
            for idx2, param_sr in param_df.iterrows():
                mask = (allParams_df.index == idx2) & (allParams_df['track_id'] == param_sr['track_id'])
                allParams_df.loc[mask, 'value'] = paramsSubset[idx]

        # Cursor thru params setting as needed
        self.params = []
        for index, param_sr in allParams_df.iterrows():
            # Prioritize track override if exists
            self.params.append(
                dict(track_id=param_sr['track_id'], param=index, value=param_sr['value']))



    def tryGetParam(self, track_ID, paramName, optional = False):
        record = next((param for param in self.params if param['param'] == paramName and
                       param['track_id'] == track_ID), None)
        if record is None:
            if optional: return 0
            raise Exception("{} not found for track_ID {}".format(paramName, track_ID))

        return record['value']

    def tryModParam(self, track_ID, paramName, newValue):
        # Iterate through the list of dictionaries
        for record in self.params :
            if record['track_id'] == track_ID and record['param'] == paramName:
                record['value'] = newValue
                break  # Exit the loop once the record is found and modified
