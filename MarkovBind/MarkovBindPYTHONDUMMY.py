import random as rd

def runPartialTrackEffLengthHoles(track_id, partialEventSet, trackActualLength, probminimodeliters, track_dict,
                                  numplayers, ideallikelihoodholehit,
                                  tentNewLadder=None, tentNewChute=None):
    # Markov chain forecasting
    # INCORPORATE CRIB EVERY N'TH TURNM!!
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

    # Figure out length of partial game
    movesAllTrials = 0
    iters = probminimodeliters

    for trial in range(iters):
        # Set up trial gameplay
        startLoc = 0
        moveCounter = 0
        curReadSeq = track_dict[track_id][trial]
        dealer = rd.randint(1, numplayers)
        curPos = 0
        countLoops = 0
        trackPosSeq = []
        while curPos < trackActualLength:
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

    # Forecast length of game based on control-case ideal moves:hole ratio
    actualPartialMoves = (movesAllTrials / iters)
    eventlessCtrlPartialMoves = (ideallikelihoodholehit * trackActualLength)
    shiftPct = actualPartialMoves / eventlessCtrlPartialMoves
    forecastedTrackEffLengthHoles = trackActualLength * shiftPct
    return forecastedTrackEffLengthHoles