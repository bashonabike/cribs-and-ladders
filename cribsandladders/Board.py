from xml.dom import minidom
import game_params as gp
import pandas as pd
import os
import cribsandladders.BaseLayout as bs
import cribsandladders.PossibleEvents as ps
import bisect as bsc

boardDBName = 'Boards/AllBoards.db'

class Board:

    def __init__(self):
        # self.boardFileName = boardFileName
        self.boardName = ""
        self.boardID = 0
        self.width = 0.0
        self.height = 0.0
        self.corners = None
        self.tracks = []
        self.twoDeckLineBoardPath = ""
        # self.xmlParser()
        self.possibleEvents = None
        # if gp.findmode:
        #     bs.setTrackHolesets(self.tracks, self.height, self.twoDeckLineBoardPath)
        #     self.possibleEvents = ps.PossibleEvents(self)

    def setEffLandingForHolesAllTracks(self):
        for t in self.tracks:
            t.setEffLandingForHoles()
    def getTrackByNum(self, trackNum):
        for track in self.tracks:
            if track.num == trackNum:
                return track

        return None

    def setBoardAfterSetter(self):
        if gp.findmode:
            bs.setTrackHolesets(self.tracks, self.height, self.twoDeckLineBoardPath)
            self.possibleEvents = ps.PossibleEvents(self)

    def clearBoard(self):
        self.boardName = ""
        self.boardID = 0
        self.width = 0.0
        self.height = 0.0
        self.corners = None
        self.tracks = []
        self.twoDeckLineBoardPath = ""
        # self.xmlParser()
        self.possibleEvents = None





    # def xmlParser(self):
    #     # parse an xml file by name
    #     board_xml_file=minidom.parse(self.boardFileName)
    #
    #     # use getElementsByTagName() to get tag
    #     self.boardName = str(board_xml_file.getElementsByTagName('boardname')[0].firstChild.data).strip("\"")
    #     batch_ladders = []
    #     batch_chutes = []
    #
    #     for xml_track in board_xml_file.getElementsByTagName('track'):
    #         curtrack = Track()
    #         curtrack.num = int(xml_track.getElementsByTagName('tracknum')[0].firstChild.data)
    #         curtrack.length = int(xml_track.getElementsByTagName('length')[0].firstChild.data)
    #         try:
    #             curtrack.twodeckslength = int(xml_track.getElementsByTagName('twodecklength')[0].firstChild.data)
    #         except:
    #             curtrack.twodeckslength = int(xml_track.getElementsByTagName('length')[0].firstChild.data)
    #         curtrack.efflength = curtrack.twodeckslength if gp.twodecks else curtrack.length
    #
    #         for ladder in xml_track.getElementsByTagName('ladder'):
    #             start = int(ladder.getElementsByTagName('start')[0].firstChild.data)
    #             end = int(ladder.getElementsByTagName('end')[0].firstChild.data)
    #             batch_ladders.append(Ladder(start, end, curtrack.num))
    #         for chute in xml_track.getElementsByTagName('chute'):
    #             start = int(chute.getElementsByTagName('start')[0].firstChild.data)
    #             end = int(chute.getElementsByTagName('end')[0].firstChild.data)
    #             batch_chutes.append(Chute(start, end, curtrack.num))
    #
    #         self.tracks.append(curtrack)
    #
    #     for curtrack in self.tracks:
    #
    #         (curtrack.setLadders(sorted([full_tracks for full_tracks in batch_ladders if full_tracks.track in
    #                                              {0, curtrack.num}], key=lambda x: x.start)))
    #         (curtrack.setChutes(sorted([full_tracks for full_tracks in batch_chutes if full_tracks.track in
    #                                             {0, curtrack.num}],key=lambda x: x.start)))
    #         curtrack.setEventLadders([l.start for l in curtrack.ladders])
    #         curtrack.setEventChutes([c.start for c in curtrack.chutes])
    def clearTrackEvents(self):
        for t in self.tracks:
            t.eventSetBuild = set()
            t.ladders = []
            t.chutes = []
            t.eventsListLadder = []
            t.eventsListChute = []

class Track:
    def __init__(self):
        self.Track_ID = 0
        self.num = 0
        self.length = 0
        self.twodeckslength = 0
        self.efflength = 0
        self.ladders = []
        self.chutes = []
        self.eventsListLadder = []
        self.eventsListChute = []
        self.holesetfilepath = ""
        self.trackholes = None
        self.holesetIndexer = []
        self.candidateEvents = None
        self.eventSetBuild = []
        self.effLandingForHoles = []
        # This is pointwise sum of event value (+/-) * likelihood of hit (1/length)
        # So sum of event values * # events / length
        # This will always be negative since always more chutes than ladders
        self.simplEventImpedance = 0.0

    def addLadder(self, ladder):
        self.ladders.append(ladder)

    def addChute(self, chute):
        self.chutes.append(chute)

    def addEventLadder(self, eventPos):
        self.eventsListLadder.append(eventPos)

    def addEventChute(self, eventPos):
        self.eventsListChute.append(eventPos)

    def addTentativeEvent(self, eventBuild):
        self.eventSetBuild.append(eventBuild)

    def setTentativeEvents(self, eventSetBuild):
        self.eventSetBuild = eventSetBuild

    def setLadders(self, ladders):
        self.ladders = ladders

    def setChutes(self, chutes):
        self.chutes = chutes

    def setEventLadders(self, eventsListLadder):
        self.eventsListLadder = eventsListLadder

    def setEventChutes(self, eventsListChute):
        self.eventsListChute = eventsListChute

    def setEventImpedance (self):
        sumLadders = sum([l.length for l in self.ladders])
        sumChutes = sum([c.length for c in self.chutes])
        self.simplEventImpedance = (sumLadders + sumChutes) * (len(self.ladders) + len(self.chutes))/self.efflength

    def setEffLandingForHoles(self):
        self.effLandingForHoles = []
        for i in range(0, len(self.trackholes)):
            effLanding = i+1
            chute_index = bsc.bisect_left(self.eventsListChute, i+1)
            if (chute_index < len(self.eventsListChute) and
                    self.eventsListChute[chute_index] == i+1):
                effLanding = self.chutes[chute_index].start
                #There should never be both chute and ladder on same space!
            ladder_index = bsc.bisect_left(self.eventsListLadder, (i+1))
            if (ladder_index < len(self.eventsListLadder) and
                    self.eventsListLadder[ladder_index] == i+1):
                effLanding = self.ladders[ladder_index].end
            self.effLandingForHoles.append(effLanding)


    def setHolesetIndexer(self):
        self.holesetIndexer = [h.num for h in self.trackholes]
    def getHoleByCoords(self, coords):
        for h in self.trackholes:
            if h.coords == coords: return h
        return None

    def getHoleByNum(self, holeNum):
        idx = bsc.bisect_left(self.holesetIndexer, holeNum)
        if idx < len(self.holesetIndexer) and self.holesetIndexer[idx] == holeNum:
            return self.trackholes[idx]
        return None

    def getLaddersAsDF(self):
        return (pd.DataFrame.from_records([l.to_dict() for l in self.ladders]))

    def getChutesAsDF(self):
        return (pd.DataFrame.from_records([c.to_dict() for c in self.chutes]))

    def getEventsAsDF(self):
        templ = [l.to_dict() for l in self.ladders]
        tempc = [c.to_dict() for c in self.chutes]
        templ.extend(tempc)
        return pd.DataFrame.from_records(templ)


class Ladder:
    def __init__(self, start, end, track, vector = ((-1,-1),(-1,-1)), eventDete = None):
        self.start = start
        self.end = end
        self.length = self.end - self.start
        self.track = track
        self.crowVector = vector
        self.eventDete = eventDete

    def to_dict(self):
        return {
            'start': self.start
            ,'end': self.end
            ,'track': self.track
        }

class Chute:
    def __init__(self, start, end, track, vector = ((-1,-1),(-1,-1)), eventDete = None):
        self.start = start
        self.end = end
        self.length = self.end - self.start
        self.track = track
        self.crowVector = vector
        self.eventDete = eventDete

    def to_dict(self):
        return {
            'start': self.start
            ,'end': self.end
            ,'track': self.track
        }
