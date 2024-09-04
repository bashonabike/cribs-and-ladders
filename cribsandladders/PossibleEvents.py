#region plan
#Build set of all combinations of two points on track where angle > x deg off from instant slope (approx orthogonality)
#For each prospective event, do search for all holes OF ANY TRACK within the rectangle of say 11 mm (give good amt of room)
#if any found, cache it for prospective 2 or 3 track wide event
#else, include in set of prospects
#next, build set of mult-track events
#continue trace of line (via lengthening rectangle) in both directions until symetry obtained
#may need more sophisticated appraoch eventually but should suffice initially
#If rectangle hits more than 1 hole on search, maybe spawn into seperate event options for each viable combination

#grrr rectangle not cartesian
#new plan: include all points on all tracks that are within rectangle constructed where each event point is corner
#maybe broaden the rectangle by like 5mm or something on all sides, to be safe
#draw line segment between each contenders to forward & backward respective neighbour on their path
#check intersection for each
#Search in each opposite direction maybe like max 20mm or so, if cannot aquire symmetry scrap it
#Order intersects by proximity

#if not orthogonal, draw ortho line to direct vector btwn points either dir containing bounding rectangle
#keep going until intersect edge of board or another track
#define this space broken into 11mm slices as possible side tracks
#set max loopy distance, maybe as function of spacing btwn event holes
#set bounding rect for line intersects based on max
#when determining set of events for each test, need to ensure enough space for all
#give priority to events closest together (eg: 3 space diff goes first)

#Set spec  # long events, all chutes  only
#Set cutoff of when ladders can start (eg: after Hole 10)
#endregion

import matplotlib.path as mpath
import numpy as np
import matplotlib.pyplot as plt
import math
import game_params as gp
import cribsandladders.BaseLayout as bs
import copy as cp
import sqlite3 as sql
import pandas as pd
from datetime import datetime as dt
import os
import bisect as bsc
from io import StringIO

class PossibleEvents:

    def __init__(self, board):
        todo = "figure this out"
        self.allTracksCandidateSet, self.byTrackCandidateSets = None, []
        self.multiTrackCandidateSet = None
        self.board = board
        sqlConn = sql.connect('etc/Temp.db')
        if not self.tryRetrieveCache(sqlConn):
            self.buildSet(board, sqlConn)

    def tryRetrieveCache(self, sqlConn):
        #Retrieve cached data if exists
        query = ("SELECT c.* FROM TempCandidateEvents c " +
                "WHERE c.Board_ID = ? " +
                "ORDER BY c.Board_ID, c.Track_ID, c.CandidateEvent_ID")
        sqliteCursor = sqlConn.cursor()
        sqliteCursor.execute(query, [self.board.boardID])
        candidateEvents_df = pd.DataFrame(sqliteCursor.fetchall(), columns=[d[0] for d in sqliteCursor.description])
        if len(candidateEvents_df) == 0: return False

        #Verify files have not been updated since last cache
        tempTableUpdated = dt.strptime(candidateEvents_df.iloc[0]['Timestamp'], '%m/%d/%y %H:%M:%S')
        sqlConnBoard = sql.connect('Boards/AllBoards.db')
        boardQuery = ("SELECT  b.Track1BoardPath, b.Track2BoardPath, b.Track3BoardPath, b.TwoDeckLineBoardPath " +
                 "FROM Board b WHERE b.Board_ID = ?")
        boardFiles = sqlConnBoard.execute(boardQuery, [self.board.boardID]).fetchall()
        for file in boardFiles[0]:
            if file in ("", None): continue
            fileModTimestamp = dt.fromtimestamp(os.path.getmtime(file))
            if fileModTimestamp > tempTableUpdated:
                print("Resetting board candidate events")
                sqliteCursor.execute("DELETE FROM TempCandidateEvents WHERE Board_ID = ?", [self.board.boardID])
                sqlConn.commit()
                return False

        #If passed gauntlet, then Great Job!  Parse cached data into candidate events
        curTrackNum = -1
        curTrack = None
        currCandSetList = []
        candSetIdx = -1

        linkList = []
        for index, event_sr in candidateEvents_df.iterrows():
            if event_sr['trackNum'] != curTrackNum:
                curTrackNum = event_sr['trackNum']
                curTrack = self.board.getTrackByNum(curTrackNum)
                currCandSetList.append(CandidateEvents(curTrack.trackholes, curTrack.num))
                candSetIdx += 1
            curEvent = CandidateEvent(event_sr['trackNum'], curTrack.getHoleByNum(event_sr['startHole']),
                                      curTrack.getHoleByNum(event_sr['endHole']),
                                      event_sr['isOrtho'],
                                      orthoFwdMinIncr=event_sr['orthoFwdMinIncr'], orthoRevMinIncr=event_sr['orthoRevMinIncr'],
                                      orthoFwdMaxIncr=event_sr['orthoFwdMaxIncr'], orthoRevMaxIncr=event_sr['orthoRevMaxIncr'])
            curEvent.db_hash = event_sr["CandidateEvent_ID"]
            curEvent.linkFinderHash = event_sr['FinderHash']
            curEvent.instanceIncr = event_sr['instanceIncr']
            curEvent.instanceRev = event_sr['instanceRev']
            curEvent.isShared = event_sr['isShared']
            curEvent.orthoVector = tuple([float(c) for c in event_sr['orthoVector'].split(',')])
            currCandSetList[candSetIdx].addCandidateEvent(curEvent)
            if event_sr['sharedWithTracks'] is not None:
                curEvent.sharedWithTracks = [int(t) for t in event_sr['sharedWithTracks'].split(",")]
                linkList.append(dict(mainevent=curEvent,
                                     mainhash=event_sr['FinderHash'],
                                     linkeventhash1=event_sr['linkedEvent1'] if event_sr['linkedEvent1'] is not None else -1,
                                     linkeventhash2=event_sr['linkedEvent2'] if event_sr['linkedEvent2'] is not None else -1))

        #Add candidate event sets to respective tracks
        for cs in currCandSetList:
            curTrack = self.board.getTrackByNum(cs.trackNum)
            curTrack.candidateEvents = cs

        #Set up linked events
        if len(linkList) > 0:
            linkList.sort(key = lambda l: l['mainhash'])
            linkListFinder = [l['mainhash'] for l in linkList]
            for m in linkList:
                links = []
                for l in [m['linkeventhash1'], m['linkeventhash2']]:
                    if l is not None and l != "":
                        idx = bsc.bisect_left(linkListFinder, l)
                        if idx < len(linkListFinder) and linkListFinder[idx] == l :
                            links.append(linkList[idx]['mainevent'])
                if len(links) > 0: m['mainevent'].setLinkedEvents(links)

        return True

    def buildSet(self, board, sqlConn):
        todo = "figure this out"
        #maybe keep data here, cross-ref to tracks
        #store sets + alltracks here, pointer to each track one inside track

        #Initialize all Candidate objects
        np.seterr(all='raise')
        allHoles = set()
        for t in board.tracks:
            if t.trackholes is None: raise Exception("No holes found for Track {}".format(t.num))
            allHoles.union(t.trackholes)
            t.candidateEvents = CandidateEvents(t.trackholes, t.num)
            self.byTrackCandidateSets.append(t.candidateEvents)
        self.allTracksCandidateSet = CandidateEvents(allHoles)
        self.multiTrackCandidateSet = CandidateEvents(allHoles)

        #Iterate, by track, determining all possible candidate events
        for t in self.byTrackCandidateSets:
            holes = t.holes_l
            for h_a_idx in range(0, len(holes) -1):
                h_a = holes[h_a_idx]
                for h_b_idx in range(h_a_idx+1, len(holes)):
                    h_b = holes[h_b_idx]
                    # if h_a.num == 5 and h_b.num == 30 and t.trackNum == 1:
                    #     p1 = np.array(h_a.coords)
                    #     p2 = np.array(holes[h_a_idx + 1].coords)
                    #     p3 = np.array(h_b.coords)
                    #
                    #     # Create vectors
                    #     v1 = p2 - p1
                    #     v2 = p3 - p1
                    #
                    #     # Normalize the vectors
                    #     unit_v1 = v1 / np.linalg.norm(v1)
                    #     unit_v2 = v2 / np.linalg.norm(v2)
                    #
                    #     # Calculate the dot product and magnitudes
                    #     dot_product = np.dot(unit_v1, unit_v2)
                    #     if dot_product > 1.0:
                    #         # Rounding error??? this means vectors are parallel, 0 deg
                    #         angle_rad = 0
                    #     elif dot_product < -1.0:
                    #         angle_rad = math.pi
                    #     else:
                    #         # Calculate the angle in radians between the two vectors
                    #         angle_rad = np.arccos(dot_product)
                    #     angle_deg = np.degrees(angle_rad)
                    #Determine if route is possible directly
                    if not self.checkAngleForOrtho(self.eventAngleWithInstantSlope(h_a.coords,
                                                                              holes[h_a_idx + 1].coords, h_b.coords)):
                        #Check if any intercepts
                        searchRect = self.cartesian_bounding_box(self.orthoBoundingBox((h_a.coords, h_b.coords)))
                        routePos = True
                        multiPos = True
                        tracksHit = []
                        holesHitAllTracks = []
                        for t_interc in self.byTrackCandidateSets:
                            intercPoints = self.points_in_rectangle(t_interc.holeCoords_l, searchRect)
                            intercVects = self.build_interception_test_vector_set(t_interc.holeCoords_l, intercPoints)
                            holesHit = []
                            if self.check_intersections({(h_a.coords, h_b.coords)}, intercVects, h_a, h_b, holesHit,
                                                        t_interc.trackNum):
                                routePos = False
                                tracksHit.extend([t_interc.trackNum]*len(holesHit))
                                holesHitAllTracks.extend(holesHit)

                            if self.check_proximity_intersects(h_a, h_b, holesHit, t_interc.holes, t_interc.holeCoords_l,
                                                           intercPoints):
                                # Check for proximity
                                routePos = False
                                multiPos = False

                        if routePos:
                            t.addCandidateEvent(CandidateEvent(t.trackNum, h_a, h_b, False))
                        elif multiPos and not t.trackNum in tracksHit:
                            #Check to see if multi-track event possible
                            #if contains only couples, we are good!
                            tracksForExtension = []
                            tracksClosed = [t.trackNum]
                            eligibleForExtension = False
                            completeMulti = False
                            #NOTE that if say we are on track 1 (inner) and there is an enclosing fold of say track 2
                            #this multi-event will be picked up when we run track 2
                            trackshit_cleaned = cp.deepcopy(tracksHit)
                            for t_sub_num in list(set(tracksHit)):
                                if tracksHit.count(t_sub_num) == 1:
                                    #Track only hit once, elig for ext!
                                    tracksForExtension.append(t_sub_num)
                                    eligibleForExtension = True
                                    completeMulti = False
                                elif tracksHit.count(t_sub_num) == 2:
                                    tracksClosed.append(t_sub_num)
                                    trackshit_cleaned = [t for t in trackshit_cleaned if t != t_sub_num]
                                    if not eligibleForExtension: completeMulti = True
                                elif tracksHit.count(t_sub_num) > 2:
                                    #Cannot ever hit track more than 2x!
                                    eligibleForExtension = False
                                    completeMulti = False
                                    break
                            tracksHit = trackshit_cleaned
                            if len(tracksHit) == 0: eligibleForExtension = False

                            if eligibleForExtension:
                                extensions = self.extend_line(h_a.coords, h_b.coords, gp.maxeventlineext)
                                rects = self.create_extended_rectangles(h_a.coords, h_b.coords,
                                                                                        gp.maxeventlineext)
                                holesHitList_l = []
                                # tracksSubset = [sub_t_c_set for sub_t_c_set in self.byTrackCandidateSets
                                #                 if sub_t_c_set.trackNum not in tracksClosed]
                                for l in range(0,2):
                                    holesHit = []
                                    for t_interc in self.byTrackCandidateSets:
                                        intercPoints = self.points_in_rectangle(t_interc.holeCoords_l, rects[l])
                                        intercVects = self.build_interception_test_vector_set(t_interc.holeCoords_l,
                                                                                              intercPoints)

                                        self.check_intersections({extensions[l]}, intercVects, h_a, h_b, holesHit,
                                                                 t_interc.trackNum)
                                    holesHitList_l.append(self.ordered_by_proximity(holesHit, extensions[l][0]))

                                #Trace along each ext, determine if can aquire symmetry
                                extPos = True
                                holesHitMultiList_l = [h_a, h_b]
                                holesHitMultiList_l.extend(holesHitAllTracks)
                                cnt = [0, 0]
                                mx = [len(holesHitList_l[0]), len(holesHitList_l[1])]
                                stuck = [False, False]
                                while cnt[0] < mx[0] or cnt[1] < mx[1]:
                                    if len(tracksClosed) == len(self.board.tracks): break
                                    for i in [0,1]:
                                        if len(tracksClosed) == len(self.board.tracks): break
                                        if cnt[i] >= mx[i]:
                                            stuck[i] = True
                                        else:
                                            if holesHitList_l[i][cnt[i]].tracknum in tracksHit:
                                                #Success!  We have closed a loop
                                                tracksHit.remove(holesHitList_l[i][cnt[i]].tracknum)
                                                tracksClosed.append(holesHitList_l[i][cnt[i]].tracknum)
                                                holesHitMultiList_l.append(holesHitList_l[i][cnt[i]])
                                                mx[i] += 1
                                            elif len([s for s in stuck if s == True]) < 2:
                                                if not stuck[i]:
                                                    #Line now stuck, save this point for later
                                                    stuck[i] = True
                                            else:
                                                #Tracks all stuck, ext not possible
                                                extPos = False
                                                break
                                    if not extPos: break

                                if extPos and len(tracksHit) == 0 and len(holesHitMultiList_l) % 2 == 0:
                                    #Re-check proximity for new line trace
                                    full_line = self.find_longest_line([h.coords for h in holesHitMultiList_l])
                                    for t_interc in self.byTrackCandidateSets:
                                        if not extPos: break
                                        searchRect = self.cartesian_bounding_box(
                                            self.orthoBoundingBox((full_line[0], full_line[1])))
                                        intercPoints = self.points_in_rectangle(t_interc.holeCoords_l, searchRect)
                                        if self.check_proximity_intersects(h_a, h_b, holesHitMultiList_l, t_interc.holes,
                                                                           t_interc.holeCoords_l,intercPoints):
                                            extPos = False
                                    #Ran the gauntlet and we are good!
                                    if extPos: self.addMultiTrackEvent(board, holesHitMultiList_l)

                            elif completeMulti:
                                #Already completed multi! Woohoo!
                                holesHitMultiList_l = cp.deepcopy(holesHitAllTracks)
                                holesHitMultiList_l.extend([h_a, h_b])
                                if len(holesHitMultiList_l) % 2 != 0:
                                    sdfsd=""
                                self.addMultiTrackEvent(board, holesHitMultiList_l)

#TODO: add in half-orthos, when direct to 1 but not the other
                    elif self.checkAngleForOrtho(self.eventAngleWithInstantSlope(h_b.coords, holes[h_b_idx - 1].coords,
                                                                                  h_a.coords)):
                        # Determine if can otherwise route indirectly along orthogonal space either side of track path
                        curEvent = CandidateEvent(t.trackNum, h_a, h_b, True)
                        for rev in (False, True):
                            orthoVectsSearchAreas = list(self.smallest_enclosing_rectangle(h_a.coords, h_b.coords,
                                                                      gp.maxloopyorthoeventdisplacementincrements
                                                                      * gp.eventminspacing, rev))
                            maxIncr = gp.maxloopyorthoeventdisplacementincrements
                            minIncr = 0
                            ortho = self.orthogonal_vector(h_a.coords, h_b.coords,
                                                           gp.maxloopyorthoeventdisplacementincrements
                                                                  * gp.eventminspacing, rev)
                            for t_interc in self.byTrackCandidateSets:
                                intercPoints = self.points_in_rectangle(t_interc.holeCoords_l, orthoVectsSearchAreas)
                                intercVects = self.build_interception_test_vector_set(t_interc.holeCoords_l, intercPoints)
                                floorIncr, testIncr, = self.test_sidestep_events(h_a, h_b, t_interc.holes, t_interc.holeCoords_l, ortho,
                                                          gp.maxloopyorthoeventdisplacementincrements
                                                                     * gp.eventminspacing,
                                                          gp.eventminspacing, intercVects, rev)
                                if testIncr < maxIncr: maxIncr = testIncr
                                if floorIncr > minIncr: minIncr = floorIncr
                                if maxIncr == 0: break

                            if maxIncr == 0: break
                            if not rev:
                                curEvent.orthoFwdMaxIncr = maxIncr
                                curEvent.orthoFwdMinIncr = minIncr
                            else:
                                curEvent.orthoRevMaxIncr = maxIncr
                                curEvent.orthoRevMinIncr = minIncr

                            curEvent.orthoVector = ortho

                        if curEvent.orthoFwdMaxIncr > 0 or curEvent.orthoRevMaxIncr > 0: t.addCandidateEvent(curEvent)

        # Calculate final sets, for uniqueness
        timestamp =  dt.now().strftime('%m/%d/%y %H:%M:%S')
        for t in self.byTrackCandidateSets:
            t.removeDuplicates()
            curTrack_ID = self.board.getTrackByNum(t.trackNum).Track_ID

            #Set into database temp table
            deleteQuery = "DELETE FROM TempCandidateEvents WHERE Board_ID = ? AND Track_ID = ?"
            sqlConn.execute(deleteQuery, (self.board.boardID,
                                               self.board.getTrackByNum(t.trackNum).Track_ID))

            #Retrieve columns into empty dataframe
            query = "SELECT * FROM TempCandidateEvents WHERE 1 = 0"
            sqliteCursor = sqlConn.cursor()
            sqliteCursor.execute(query)
            candidateEvents_df = pd.DataFrame.from_records([[None for d in sqliteCursor.description]]
                                                           *len(t.candidateEvents),
                                              columns=[d[0] for d in sqliteCursor.description])

            #Pop dataframe
            for e_idx in range(0, len(candidateEvents_df)):
                candidateEvents_df.loc[e_idx]['Board_ID'] = self.board.boardID
                candidateEvents_df.loc[e_idx]['Track_ID'] =  curTrack_ID
                candidateEvents_df.loc[e_idx]['CandidateEvent_ID'] =  t.candidateEvents[e_idx].db_hash
                candidateEvents_df.loc[e_idx]['FinderHash'] = (10E6*t.candidateEvents[e_idx].trackNum
                                                               +10E3*t.candidateEvents[e_idx].startHole.num
                                                               + t.candidateEvents[e_idx].endHole.num)
                candidateEvents_df.loc[e_idx]['Timestamp'] =   timestamp
                candidateEvents_df.loc[e_idx]['trackNum'] =t.candidateEvents[e_idx].trackNum
                candidateEvents_df.loc[e_idx]['startHole'] =t.candidateEvents[e_idx].startHole.num
                candidateEvents_df.loc[e_idx]['endHole'] =t.candidateEvents[e_idx].endHole.num
                candidateEvents_df.loc[e_idx]['midPointNum'] =t.candidateEvents[e_idx].midPointNum
                candidateEvents_df.loc[e_idx]['crowVector'] =("{};{}"
                .format(",".join([str(c) for c in t.candidateEvents[e_idx].crowVector[0]])
                        , ",".join([str(c) for c in t.candidateEvents[e_idx]
                                   .crowVector[1]])))
                candidateEvents_df.loc[e_idx]['length'] =t.candidateEvents[e_idx].length
                candidateEvents_df.loc[e_idx]['canBeLadder'] =t.candidateEvents[e_idx].canBeLadder
                candidateEvents_df.loc[e_idx]['isOrtho'] =t.candidateEvents[e_idx].isOrtho
                candidateEvents_df.loc[e_idx]['orthoVector'] =  ("{},{}"
                                                              .format(t.candidateEvents[e_idx].orthoVector[0],
                                                                      t.candidateEvents[e_idx].orthoVector[1]))
                candidateEvents_df.loc[e_idx]['orthoFwdMinIncr'] =t.candidateEvents[e_idx].orthoFwdMinIncr
                candidateEvents_df.loc[e_idx]['orthoRevMinIncr'] =t.candidateEvents[e_idx].orthoRevMinIncr
                candidateEvents_df.loc[e_idx]['orthoFwdMaxIncr'] =t.candidateEvents[e_idx].orthoFwdMaxIncr
                candidateEvents_df.loc[e_idx]['orthoRevMaxIncr'] =t.candidateEvents[e_idx].orthoRevMaxIncr
                if t.candidateEvents[e_idx].sharedWithTracks not in (None, -1, 0):
                    candidateEvents_df.loc[e_idx]['sharedWithTracks'] =",".join([str(sh) for sh in
                                                                                 t.candidateEvents[e_idx].sharedWithTracks])
                candidateEvents_df.loc[e_idx]['sharedSubHash'] =t.candidateEvents[e_idx].sharedSubHash
                candidateEvents_df.loc[e_idx]['instanceIncr'] =t.candidateEvents[e_idx].instanceIncr
                candidateEvents_df.loc[e_idx]['instanceRev'] =t.candidateEvents[e_idx].instanceRev
                if t.candidateEvents[e_idx].linkedEvents is not None and len(t.candidateEvents[e_idx].linkedEvents) > 0:
                    candidateEvents_df.loc[e_idx]['linkedEvent1'] = (10E6 * t.candidateEvents[e_idx].linkedEvents[0].trackNum
                     + 10E3 * t.candidateEvents[e_idx].linkedEvents[0].startHole.num
                     + t.candidateEvents[e_idx].linkedEvents[0].endHole.num)


                if t.candidateEvents[e_idx].linkedEvents is not None and len(t.candidateEvents[e_idx].linkedEvents) > 1:
                    candidateEvents_df.loc[e_idx]['linkedEvent2'] = (10E6 * t.candidateEvents[e_idx].linkedEvents[1].trackNum
                     + 10E3 * t.candidateEvents[e_idx].linkedEvents[1].startHole.num
                     + t.candidateEvents[e_idx].linkedEvents[1].endHole.num)
                candidateEvents_df.loc[e_idx]['isShared'] =t.candidateEvents[e_idx].isShared

            #Insert into data table
            #TODO: round off coords & vectors!
            candidateEvents_df_m = candidateEvents_df.values.tolist()
            candQuery_sb = StringIO()
            candQuery_sb.write("INSERT INTO TempCandidateEvents(")
            cols = [d[0] for d in sqliteCursor.description]
            for c in range(0, len(cols) - 1):
                candQuery_sb.write(cols[c])
                candQuery_sb.write(", ")
            candQuery_sb.write(cols[len(cols)-1])
            candQuery_sb.write(") VALUES (")
            candQuery_sb.write(", ".join(['?']*len(cols)))
            candQuery_sb.write(")")
            sqliteCursor.executemany(candQuery_sb.getvalue(), candidateEvents_df_m)
            sqlConn.commit()
            candQuery_sb.close()


    def addMultiTrackEvent(self, board, holesHitMultiList_l):
        linkedEventsSet = []
        hole_vectors = []

        holesHitMultiList_l.sort(key=lambda h: (h.tracknum, h.num))

        for h in range(0,len(holesHitMultiList_l),2):
           hole_vectors.append((holesHitMultiList_l[h], holesHitMultiList_l[h+1]))

        for hole_vector in hole_vectors:
            sharedWithTracks = [v[0].tracknum for v in hole_vectors if v[0].tracknum != hole_vector[0].tracknum]
            currMultiCand = CandidateEvent(hole_vector[0].tracknum, hole_vector[0], hole_vector[1],
                                           isOrtho=False, sharedWithTracks=sharedWithTracks)
            currTrack = board.getTrackByNum(hole_vector[0].tracknum)
            currTrack.candidateEvents.addCandidateEvent(currMultiCand)
            linkedEventsSet.append(currMultiCand)

        #set up links
        if len(linkedEventsSet) > 1:
            linkedEventsSet.sort(key=lambda m: m.trackNum)
            for linkedEvent in linkedEventsSet:
                copyLinks = cp.deepcopy(linkedEventsSet)
                copyLinks.remove(linkedEvent)
                linkedEvent.setLinkedEvents(copyLinks)

    def ccw(self, A, B, C):
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

    def onSegment(self, p, q, r):
        if ((q[0] <= max(p[0], r[0])) and (q[0] >= min(p[0], r[0])) and
                (q[1] <= max(p[1], r[1])) and (q[1] >= min(p[1], r[1]))):
            return True
        return False

    def orientation(self, p, q, r):
        # to find the orientation of an ordered triplet (p,q,r) 
        # function returns the following values: 
        # 0 : Collinear points 
        # 1 : Clockwise points 
        # 2 : Counterclockwise 

        # See https://www.geeksforgeeks.org/orientation-3-ordered-points/amp/  
        # for details of below formula.  

        val = (float(q[1] - p[1]) * (r[0] - q[0])) - (float(q[0] - p[0]) * (r[1] - q[1]))
        if (val > 0):

            # Clockwise orientation 
            return 1
        elif (val < 0):

            # Counterclockwise orientation 
            return 2
        else:

            # Collinear orientation 
            return 0

    # The main function that returns true if  
    # the line segment 'p1q1' and 'p2q2' intersect. 
    def doIntersect(self, p1, q1, p2, q2):

        # Find the 4 orientations required for  
        # the general and special cases 
        o1 = self.orientation(p1, q1, p2)
        o2 = self.orientation(p1, q1, q2)
        o3 = self.orientation(p2, q2, p1)
        o4 = self.orientation(p2, q2, q1)

        # General case 
        if ((o1 != o2) and (o3 != o4)):
            return True

        # Special Cases 

        # p1 , q1 and p2 are collinear and p2 lies on segment p1q1 
        if ((o1 == 0) and self.onSegment(p1, p2, q1)):
            return True

        # p1 , q1 and q2 are collinear and q2 lies on segment p1q1 
        if ((o2 == 0) and self.onSegment(p1, q2, q1)):
            return True

        # p2 , q2 and p1 are collinear and p1 lies on segment p2q2 
        if ((o3 == 0) and self.onSegment(p2, p1, q2)):
            return True

        # p2 , q2 and q1 are collinear and q1 lies on segment p2q2 
        if ((o4 == 0) and self.onSegment(p2, q1, q2)):
            return True

        # If none of the cases 
        return False

    # Return true if line segments AB and CD intersect
    def intersect(self, vec1, vec2):
        # return self.doIntersect(vec1[0], vec1[1], vec2[0], vec2[1])
        return (self.ccw(vec1[0], vec2[0], vec2[1]) != self.ccw(vec1[1], vec2[0], vec2[1]) and
                self.ccw(vec1[0], vec1[1], vec2[0]) != self.ccw(vec1[0], vec1[1], vec2[1]))

    def check_intersections(self, test_path_set, possible_intercepts_set, h_a=(-1,-1), h_b=(-1,-1),
                            holesHit = None, trackNum = -1, closestPoint = -1, testOrthos = False,
                            fwdOrthosTest = False, postGenTest = False):
        """
        Check if any vector in test_path_set intersects with any vector in possible_intercepts_set.

        Parameters:
            test_path_set (set of tuples): List of vectors in the format ((x1, y1), (x2, y2)).
            possible_intercepts_set (set of tuples): List of vectors in the format ((x1, y1), (x2, y2)).

        Returns:
            True if intersects found, False if none found
        """
        for vector1 in test_path_set:
            for vector2 in possible_intercepts_set:
                if not postGenTest and (vector1[0] == vector2[0] or vector1[0] == vector2[1] or
                                vector1[1] == vector2[0] or vector1[1] == vector2[1]): continue
                if vector1 != vector2 and vector2[0] not in [h_a, h_b] and self.intersect(vector1, vector2):
                    if holesHit is None:
                        #If testing orthos, omit touching nodes
                        closestPoint = vector1[0]
                        return True
                    else:
                        holesHit.append(self.board.getTrackByNum(trackNum).getHoleByCoords(vector2[0]))
        if holesHit is not None and len(holesHit) > 0: return True
        return False

    def build_interception_test_vector_set(self, main_set, subset):
        """
        Given an ordered main set of points and a subset of points,
        draw vectors between each point in the subset and the next point in the main set.
        If the point immediately before the point in question in the main set is not in the subset,
        draw a vector between that previous point and the point in question as well.

        Args:
            main_set (list of tuples): An ordered list of points (x, y).
            subset (list of tuples): A subset of the points from the main set.

        Returns:
            List of tuples: Each tuple represents a vector from one point to another.
        """
        vectors = set()

        # Ensure both sets are sorted as per the main set

        subset_sorted = sorted(subset, key=lambda p: main_set.index(p))

        for point in subset_sorted:
            # Find the index of the point in the main set
            idx = main_set.index(point)

            # Draw a vector to the next point in the main set, if it exists
            if idx + 1 < len(main_set):
                next_point = main_set[idx + 1]
                vectors.add((point, next_point))

            # Draw a vector from the previous point if it is not in the subset
            if idx > 0:
                previous_point = main_set[idx - 1]
                if previous_point not in subset_sorted:
                    vectors.add((previous_point, point))

        return vectors

    def build_proximity_test_vector_set(self, main_set, subset, subset_omit):
        """
        Given an ordered main set of points and a subset of points,
        draw vectors between each point in the subset and the next point in the main set.
        If the point immediately before the point in question in the main set is not in the subset,
        draw a vector between that previous point and the point in question as well.

        Args:
            main_set (list of tuples): An ordered list of points (x, y).
            subset (list of tuples): A subset of the points from the main set.
            subset_omit (list of tuples): A subset of the points from the main set to omit (i.e. hit points).

        Returns:
            List of tuples: Each tuple represents a vector from one point to another.
        """
         # Omit designated points & build base vector set
        subset_omitted = cp.deepcopy(subset)

        for c in subset_omit:
            if c in subset_omitted: subset_omitted.remove(c)

        base_vectors = self.build_interception_test_vector_set(main_set, subset_omitted)

        # Determine ortho bounding boxes
        prox_vectors = set()
        for bsv in base_vectors:
            prox_vectors.update(self.boundingBoxForVector(bsv))

        return prox_vectors

    def check_proximity_intersects(self, h_a, h_b, holesHit, holes, pertinentHoleCoords_l, intercPoints,
                                   orthoVects = None):
        # Check for proximity
        proxVectsOmit = []
        proxVectsIndices = []
        if h_a.tracknum == holes[0].tracknum:
            proxVectsOmit.extend([h_a.coords, h_b.coords])
            proxVectsIndices.extend([h_a.num - 1, h_b.num - 1])
        if len(holesHit) > 0:
            proxVectsOmit.extend([h.coords for h in holesHit])
            proxVectsIndices.extend([h.num - 1 for h in holesHit])
        for i in proxVectsIndices:
            if i > 0: proxVectsOmit.append(holes[i - 1])
        proxVects = self.build_proximity_test_vector_set(pertinentHoleCoords_l, intercPoints,
                                                         proxVectsOmit)
        if orthoVects is None:
            searchVects = {(h_a.coords, h_b.coords)}
            testOrthos = False
        else:
            searchVects = orthoVects
            testOrthos = True
        if self.check_intersections(searchVects, proxVects, h_a, h_b, testOrthos=testOrthos):
            return True
        return False

    def boundingBoxForVector(self, vector):
        intersects = []
        cornersPlusMainPoints = self.orthoBoundingBox(vector)
        numPoints = len(cornersPlusMainPoints)
        for i in range(0,numPoints):
            intersects.append((cornersPlusMainPoints[i], cornersPlusMainPoints[(i+1)%numPoints]))
        return set(tuple(intersects))

    def test_sidestep_events(self, h_a, h_b, holes, holeCoords_l, orthogonal_vector, max_ortho_length, increment,
                             possible_intercepts_set, reverse, minIncr = -1, maxIncr = -1, ignoreProximity = False,
                             debugTest = False):
        """
        Calculate the lines incrementally from the given points following the orthogonal vector.

        Parameters:
            point1, point2 (Hole): Holes of the two points in the format (x, y).
            holes (Hole)
            holeCoords_l (list)
            orthogonal_vector (tuple): The direction of the orthogonal vector in the format (dx, dy).
            max_ortho_length (float): The total length to which the lines should be extended.
            increment (float): The increment value for each step.
            reverse (bool)

        Returns:
            int: min number of increments for potential paths
            int: max number of increments for potential paths
        """
        #TODO: incorpoarte half-ortho, where one side ortho other diurect
        #TODO: maybe play with triangular stright line approximations if both sides ortho?
            #Would need to much w/ proximity sensing in this case!
        if debugTest:
            sfds = ""
        total_increments = int(max_ortho_length / increment)
        lines = []
        point1 = h_a.coords
        point2 = h_b.coords
        midpoint = tuple([sum(c)/2 for c in zip(point1, point2)])
        validIncrs = []

        incrStart, incrStop = 1, total_increments
        if minIncr > -1: incrStart = minIncr
        if maxIncr > -1: incrStop = maxIncr

        for i in range(incrStart, incrStop + 1):
            # Calculate the incremented length
            current_length = increment * i
            length_divider = current_length/max_ortho_length

            # Calculate triangle peak point based on the orthogonal vector
            peak_point = (midpoint[0] + orthogonal_vector[0] * length_divider,
                          midpoint[1] + orthogonal_vector[1] * length_divider)

            #Check if valid
            if (peak_point[0] < 0 or peak_point[0] > self.board.width or
                    peak_point[1] < 0 or peak_point[1] > self.board.height):
                break

            test_set = set()
            # Line 1: From point1 to peak
            test_set.add((point1, peak_point))

            # Line 2: From point2 to peak
            test_set.add((point2, peak_point))

            if (not self.check_intersections(test_set, possible_intercepts_set, postGenTest=ignoreProximity,
                                             testOrthos=True)
                    and (ignoreProximity or not self.check_proximity_intersects(h_a, h_b, [], holes,
                                               holeCoords_l, [c[0] for c in possible_intercepts_set], test_set))):
                #Success! This is a valid incr
                validIncrs.append(i)

        minIncr, maxIncr, prevIncr = 0, 0, 0
        if len(validIncrs) > 0:
            minIncr, maxIncr, prevIncr = validIncrs[0],validIncrs[0],validIncrs[0]

            if len(validIncrs) > 1:
                for incr in validIncrs[1:]:
                    if incr - prevIncr == 1:
                        maxIncr = incr
                    else:
                        break
                    prevIncr = incr
        if maxIncr < minIncr: minIncr = maxIncr
        return minIncr, maxIncr

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

        plt.show()
        plt.waitforbuttonpress()
    def midpoint(self, point1, point2):
        """
        Calculate the midpoint between two points.

        Parameters:
            point1, point2 (tuple): Coordinates of the two points in the format (x, y).

        Returns:
            tuple: The midpoint coordinates in the format (x, y).
        """
        return ((point1[0] + point2[0]) / 2, (point1[1] + point2[1]) / 2)

    def orthogonal_vector(self, point1, point2, length, reverse):
        """
        Calculate the orthogonal vector of a specified length from the midpoint.

        Parameters:
            point1, point2 (tuple): Coordinates of the two points in the format (x, y).
            length (float): The length of the orthogonal vector.
            reverse (bool): Spec if reverse dir or not

        Returns:
            tuple: The endpoint of the orthogonal vector in the format (x, y).
        """
        dx = point2[0] - point1[0]
        dy = point2[1] - point1[1]

        # Normalize the orthogonal vector
        orthogonal_dx = -dy
        orthogonal_dy = dx
        magnitude = math.sqrt(orthogonal_dx ** 2 + orthogonal_dy ** 2)

        #Return nil if magnitude is zero
        if magnitude == 0:
            return 0,0

        # Scale to the desired length
        orthogonal_dx = (orthogonal_dx / magnitude) * length
        orthogonal_dy = (orthogonal_dy / magnitude) * length

        #Flip direction if rev
        if reverse:
            orthogonal_dx *= -1
            orthogonal_dy *= -1


        return orthogonal_dx, orthogonal_dy

    def smallest_enclosing_rectangle(self, point1, point2, length, reverse):
        """
        Determine the smallest rectangle that contains both points and the orthogonal vector.
        Bloat outwards to check in wide search area

        Parameters:
            point1, point2 (tuple): Coordinates of the two points in the format (x, y).
            length (float): The length of the orthogonal vector.
            reverse (bool): Spec if reverse dir or not

        Returns:
            List of tuples: Coordinates of the bottom-left and top-right corners of the rectangle.
        """
        mid = self.midpoint(point1, point2)
        orthogonal_dx, orthogonal_dy = self.orthogonal_vector(point1, point2, length + gp.eventminspacing, reverse)
        ortho_dxdy = (orthogonal_dx, orthogonal_dy)

        # The orthogonal vector endpoints
        # ortho_point1 = (mid[0] + orthogonal_dx / 2, mid[1] + orthogonal_dy / 2)
        # ortho_point2 = (mid[0] - orthogonal_dx / 2, mid[1] - orthogonal_dy / 2)

        #Extend out p1 & p2 to widen search box
        (ext1, ext2) = self.extend_line(point1, point2, gp.eventminspacing/2)
        extp1, extp2 = ext1[1], ext2[1]
        ortho_vec1 = (extp1, [p + o for p, o in zip(extp1, ortho_dxdy)])
        ortho_vec2 = (extp2, [p + o for p, o in zip(extp2, ortho_dxdy)])
        orthorecpoint1 = self.orthoBoundingBox(ortho_vec1)
        orthorecpoint2 = self.orthoBoundingBox(ortho_vec2)
        orthoboundpoints = list(orthorecpoint1)
        orthoboundpoints.extend(list(orthorecpoint2))

        # # Determine the min and max x and y values to form the smallest rectangle
        # min_x = min(point1[0], point2[0], ortho_point1[0], ortho_point2[0])
        # min_y = min(point1[1], point2[1], ortho_point1[1], ortho_point2[1])
        # max_x = max(point1[0], point2[0], ortho_point1[0], ortho_point2[0])
        # max_y = max(point1[1], point2[1], ortho_point1[1], ortho_point2[1])

        # return self.determine_rectangle_corners((min_x, min_y), (max_x, max_y))
        return self.cartesian_bounding_box(tuple(orthoboundpoints))
    def calculate_distance(self, point1, point2):
        """
        Calculate the Euclidean distance between two points.

        Parameters:
            point1, point2 (tuple): Coordinates of the two points in the format (x, y).

        Returns:
            float: The Euclidean distance between the two points.
        """
        return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

    def find_longest_line(self, coordinates):
        """
        Find the longest line that can be traced through a set of coordinates.

        Parameters:
            coordinates (list of tuples): A list of points in the format [(x1, y1), (x2, y2), ...].

        Returns:
            tuple: Vector of longest line
        """
        if len(coordinates) < 2:
            raise ValueError("At least two points are required to form a line.")

        # Sort the coordinates by the x-axis (or y-axis if preferred)
        coordinates.sort(key=lambda p: (p[0], p[1]))

        # The longest line is between the first and last points in the sorted list
        p1 = coordinates[0]
        p2 = coordinates[-1]

        # Calculate the distance between these two points
        max_distance = self.calculate_distance(p1, p2)

        return (p1, p2)
    def ordered_by_proximity(self, holes, reference_point):
        """
        Order a set of coordinates by their proximity to a given reference point.

        Parameters:
            points (list of Hole): A list of coordinates in the format (x, y).
            reference_point (tuple): The reference coordinate in the format (x, y).

        Returns:
            list of Hole: The list of coordinates ordered by their proximity to the reference point.
        """
        # Sort the points based on the calculated distance from the reference point
        return sorted(holes, key=lambda hole: self.calculate_distance(hole.coords, reference_point))

    def extend_line(self, p1, p2, distance):
        """
        Extend the line segment between two points in both directions.

        Parameters:
            p1, p2 (tuple): Coordinates of the two points in the format (x, y).
            distance (float): The distance by which to extend the line in both directions.

        Returns:
            list of tuples: Two new vectors, one extending in each direction.
        """
        # Calculate the direction vector from p1 to p2
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]

        # Calculate the length of the original line segment
        length = math.sqrt(dx ** 2 + dy ** 2)

        # Normalize the direction vector
        dx /= length
        dy /= length

        # Calculate the two new points
        new_p1 = (p1[0] - dx * distance, p1[1] - dy * distance)
        new_p2 = (p2[0] + dx * distance, p2[1] + dy * distance)

        return [(p1, new_p1), (p2, new_p2)]

    def create_extended_rectangles(self, p1, p2, distance):
        """
        Create two new rectangles by extending the line segment between two points.

        Parameters:
            p1, p2 (tuple): Coordinates of the original diagonal points.
            distance (float): The distance by which to extend the line in both directions.

        Returns:
            list of tuples: Two sets of coordinates, each representing the four corners of a new rectangle.
        """
        # Extend the line in both directions
        extended_p1, extended_p2 = self.extend_line(p1, p2, distance)

        #Determine bounding boxes
        rect1_corners = self.cartesian_bounding_box(self.orthoBoundingBox((p1, extended_p1[1])))
        rect2_corners = self.cartesian_bounding_box(self.orthoBoundingBox((p2, extended_p2[1])))

        # # Determine the corners of the two new rectangles
        # rect1_corners = self.determine_rectangle_corners(p1, extended_p1[1])
        # rect2_corners = self.determine_rectangle_corners(p2, extended_p2[1])

        return [rect1_corners, rect2_corners]

    def orthoBoundingBox(self, vector):
        ortho_dxdy = self.orthogonal_vector(vector[0], vector[1], gp.eventminspacing / 2.0, False)
        revOrtho_dxdy = [(-1) * d for d in ortho_dxdy]
        midpoint =  tuple(sum(c)/2 for c in zip(vector[0], vector[1]))
        corners = [vector[0], tuple(sum(c) for c in zip(midpoint, ortho_dxdy))]
        corners.extend([vector[1], tuple(sum(c) for c in zip(midpoint, revOrtho_dxdy))])
        return tuple(corners)

    def cartesian_bounding_box(self, points):
        """
        Calculate the smallest possible bounding box that contains all the given points.

        Parameters:
            points (tuple of tuples): A list of coordinates, where each coordinate is a tuple (x, y).

        Returns:
            list: A list containing four tuples, each representing a corner of the bounding box
                   in the following order: (bottom-left, bottom-right, top-right, top-left).
        """
        if not points:
            raise ValueError("The list of points cannot be empty.")

        # Unzip the points into separate x and y coordinate lists
        x_coords, y_coords = zip(*points)

        # Find the minimum and maximum x and y coordinates
        min_x = min(x_coords)
        max_x = max(x_coords)
        min_y = min(y_coords)
        max_y = max(y_coords)

        #Increase by 10 for safety
        min_x -= 10
        max_x += 10
        min_y -= 10
        max_y += 10

        # Define the corners of the bounding box
        bottom_left = (min_x, min_y)
        bottom_right = (max_x, min_y)
        top_right = (max_x, max_y)
        top_left = (min_x, max_y)

        return [bottom_left, bottom_right, top_right, top_left]



    def determine_rectangle_corners(self, p1, p2):
        """
        Determine the four corners of a rectangle given two diagonal points.

        Parameters:
            p1, p2 (tuple): Coordinates of the two diagonal points in the format (x, y).

        Returns:
            list: Coordinates of the four corners of the rectangle.
        """
        # Extract coordinates
        x1, y1 = p1
        x2, y2 = p2

        # Calculate the other two corners
        p3 = (x1, y2)
        p4 = (x2, y1)

        # Return all four corners
        return [p1, p3, p2, p4]

    def eventAngleWithInstantSlope(self, point, nextPoint, eventPoint):
        # Convert points to numpy arrays
        p1 = np.array(point)
        p2 = np.array(nextPoint)
        p3 = np.array(eventPoint)

        # Create vectors
        v1 = p2 - p1
        v2 = p3 - p1

        # Normalize the vectors
        unit_v1 = v1 / np.linalg.norm(v1)
        unit_v2 = v2 / np.linalg.norm(v2)

        # Calculate the dot product and magnitudes
        dot_product = np.dot(unit_v1, unit_v2)
        if dot_product > 1.0:
            #Rounding error??? this means vectors are parallel, 0 deg
            angle_rad = 0
        elif dot_product < -1.0:
            angle_rad = math.pi
        else:
            # Calculate the angle in radians between the two vectors
            angle_rad = np.arccos(dot_product)
        angle_deg = np.degrees(angle_rad)
        return angle_deg

    def points_in_rectangle(self, points, rect_corners):
        """
        Determines which points lie within the rectangle.

        Parameters:
            points (list of tuples): List of (x, y) points to check.
            rect_corners (list of tuples): List of four (x, y) points defining the rectangle.

        Returns:
            list of tuples: Points that lie within the rectangle.
        """
        # Convert points and rectangle corners to numpy arrays
        points_ar = np.array(points)
        rect_corners_ar = np.array(rect_corners)

        # Create a path object for the rectangle
        path = mpath.Path(rect_corners_ar)

        # Use the path to determine which points are inside the rectangle
        inside = path.contains_points(points_ar)

        # Return the points that are inside the rectangle
        return list(map(tuple, points_ar[inside]))

    def checkAngleForOrtho(self, angle_deg):
        angle_deg_controlled = (360 + angle_deg) % 360
        if (angle_deg_controlled >= gp.minanglefromtracktangent and
                angle_deg_controlled <= (180-gp.minanglefromtracktangent)): return False
        if (angle_deg_controlled >= 180 + gp.minanglefromtracktangent and
                angle_deg_controlled <= (360-gp.minanglefromtracktangent)): return False
        return True

class CandidateEvents:
    def __init__(self, holes, trackNum = -1):
        self.trackNum = trackNum
        self.holes = holes
        self.holes_l = list(holes)
        self.holes_l.sort(key=lambda h: h.num)
        self.holeCoords_l = [h.coords for h in self.holes_l]
        self.candidateEvents = []

    def addCandidateEvent(self, candidateEvent):
        candidateEvent.db_hash = hash(candidateEvent)
        self.candidateEvents.append(candidateEvent)

    def removeDuplicates(self):
        self.candidateEvents = list(set(self.candidateEvents))

class CandidateEvent:
    def __init__(self, trackNum, startHole, endHole, isOrtho,
                 orthoFwdMinIncr=0, orthoRevMinIncr=0, orthoFwdMaxIncr=0, orthoRevMaxIncr=0,
                 orthoVector = (-1,-1),
                 sharedWithTracks = None, linkedEvents = None):
        self.db_hash = -1
        self.linkFinderHash = -1
        self.trackNum = trackNum
        self.startHole = startHole
        self.endHole = endHole
        self.midPointNum = (startHole.num + endHole.num)/2
        self.crowVector = (startHole.coords, endHole.coords)
        self.crowLength = self.calculate_distance(*self.crowVector)
        self.length = self.endHole.num - self.startHole.num
        self.canBeLadder = self.length <= gp.maxladderlength
        self.isOrtho = isOrtho
        self.orthoVector = orthoVector
        self.orthoFwdMinIncr = orthoFwdMinIncr
        self.orthoRevMinIncr = orthoRevMinIncr
        self.orthoFwdMaxIncr = orthoFwdMaxIncr
        self.orthoRevMaxIncr = orthoRevMaxIncr
        self.sharedWithTracks = sharedWithTracks
        self.sharedSubHash = 0
        self.instanceIncr = -1
        self.instanceRev = False
        self.instanceStartVector = ((-1,-1),(-1,-1))
        self.instanceEndVector = ((-1,-1),(-1,-1))
        self.instanceLump = (-1,-1)
        self.instanceIsChute = True
        self.instanceIsLadder = self.canBeLadder
        self.instanceCancel = False
        if self.sharedWithTracks is not None:
            for s in range(0, len(self.sharedWithTracks)):
                self.sharedSubHash += self.sharedWithTracks[s]*(pow(10,s))
        self.linkedEvents = linkedEvents
        self.isShared = self.sharedWithTracks is not None

    def calculate_distance(self, point1, point2):
        """
        Calculate the Euclidean distance between two points.

        Parameters:
            point1, point2 (tuple): Coordinates of the two points in the format (x, y).

        Returns:
            float: The Euclidean distance between the two points.
        """
        return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

    def __key(self):
        return (self.trackNum, self.startHole.num, self.endHole.num, self.sharedSubHash)

    def __hash__(self):
        return hash(self.__key())

    def __eq__(self, other):
        if isinstance(other, CandidateEvent):
            return self.__key() == other.__key()
        return NotImplemented

    def __lt__(self, other):
        # Define the comparison order
        if self.trackNum != other.trackNum:
            return self.trackNum < other.trackNum
        if self.startHole.num != other.startHole.num:
            return self.startHole.num < other.startHole.num
        if self.endHole.num != other.endHole.num:
            return self.endHole.num < other.endHole.num
        return self.sharedSubHash < other.sharedSubHash

    def setLinkedEvents(self, linkedEvents):
        self.linkedEvents = linkedEvents
        self.sharedWithTracks = [e.trackNum for e in self.linkedEvents]
        #NOTE: this WILL NOT RECOMPUTE THE HASH
        self.sharedSubHash = 0
        for s in range(0, len(self.sharedWithTracks)):
            self.sharedSubHash += self.sharedWithTracks[s] * (pow(10, s))
        self.isShared = True
