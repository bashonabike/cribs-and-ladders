import game_params as gp
import pandas as pd
import cribsandladders.BaseLayout as bs
import cribsandladders.PossibleEvents as ps
import bisect as bsc

boardDBName = 'Boards/AllBoards.db'

class Board:
    """
    Represents a game board consisting of multiple tracks, events, and spatial dimensions.
    """

    def __init__(self):
        """Initializes a new Board instance with default values."""
        self.boardName = ""
        self.boardID = 0
        self.width = 0.0
        self.height = 0.0
        self.corners = None
        self.tracks = []
        self.twoDeckLineBoardPath = ""
        self.possibleEvents = None

    def setEffLandingForHolesAllTracks(self):
        """Calculates and sets the effective landing positions for all tracks on the board."""
        for t in self.tracks:
            t.setEffLandingForHoles()

    def getTrackByNum(self, trackNum):
        """
        Retrieves a track object by its identification number.

        Args:
            trackNum (int): The number of the track to retrieve.

        Returns:
            Track: The requested track object if found, otherwise None.
        """
        for track in self.tracks:
            if track.num == trackNum:
                return track
        return None

    def setBoardAfterSetter(self):
        """
        Initializes track hole sets and possible events if findmode is enabled.
        Used after the board configuration has been set.
        """
        if gp.findmode:
            bs.setTrackHolesets(self.tracks, self.height, self.twoDeckLineBoardPath)
            self.possibleEvents = ps.PossibleEvents(self)

    def clearBoard(self):
        """Resets all board attributes to their default initial states."""
        self.boardName = ""
        self.boardID = 0
        self.width = 0.0
        self.height = 0.0
        self.corners = None
        self.tracks = []
        self.twoDeckLineBoardPath = ""
        self.possibleEvents = None

    def clearTrackEvents(self, specificTracks=None):
        """
        Clears ladders, chutes, and event lists for specified tracks or all tracks.

        Args:
            specificTracks (list, optional): A subset of tracks to clear.
                Defaults to all tracks in self.tracks.
        """
        tracksToIter = self.tracks if specificTracks is None else specificTracks
        for t in tracksToIter:
            t.eventSetBuild = set()
            t.ladders = []
            t.chutes = []
            t.eventsListLadder = []
            t.eventsListChute = []
            t.instLocked = False


class Track:
    """
    Represents an individual track on the board, managing its length,
    events (ladders/chutes), and hole indexing.
    """

    def __init__(self):
        """Initializes a new Track instance with empty event lists and zeroed dimensions."""
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
        self.instLocked = False
        # This is pointwise sum of event value (+/-) * likelihood of hit (1/length)
        # So sum of event values * # events / length
        # This will always be negative since always more chutes than ladders
        self.simplEventImpedance = 0.0

    def addLadder(self, ladder):
        """Adds a ladder to the track."""
        self.ladders.append(ladder)

    def addChute(self, chute):
        """Adds a chute to the track."""
        self.chutes.append(chute)

    def addEventLadder(self, eventPos):
        """Adds a ladder start position to the ladder event list."""
        self.eventsListLadder.append(eventPos)

    def addEventChute(self, eventPos):
        """Adds a chute start position to the chute event list."""
        self.eventsListChute.append(eventPos)

    def addTentativeEvent(self, eventBuild):
        """Appends a potential event to the temporary build set."""
        self.eventSetBuild.append(eventBuild)

    def setTentativeEvents(self, eventSetBuild):
        """Sets the entire list of tentative events."""
        self.eventSetBuild = eventSetBuild

    def setLadders(self, ladders):
        """Sets the track's ladder list."""
        self.ladders = ladders

    def setChutes(self, chutes):
        """Sets the track's chute list."""
        self.chutes = chutes

    def setEventLadders(self, eventsListLadder):
        """Sets the track's list of ladder event positions."""
        self.eventsListLadder = eventsListLadder

    def setEventChutes(self, eventsListChute):
        """Sets the track's list of chute event positions."""
        self.eventsListChute = eventsListChute

    def setEventImpedance(self):
        """
        Calculates the average movement impact of events per unit of track length.
        """
        sumLadders = sum([l.length for l in self.ladders])
        sumChutes = sum([c.length for c in self.chutes])
        self.simplEventImpedance = (sumLadders + sumChutes) * (len(self.ladders) + len(self.chutes)) / self.efflength

    def setEffLandingForHoles(self):
        """
        Calculates the actual landing square for every hole on the track,
        accounting for ladders and chutes.
        """
        self.effLandingForHoles = []
        if self.trackholes is None or len(self.trackholes) == 0:
            for i in range(self.length):
                self.effLandingForHoles.append(i + 1)
        else:
            for i in range(0, len(self.trackholes)):
                effLanding = i + 1
                chute_index = bsc.bisect_left(self.eventsListChute, i + 1)
                if (chute_index < len(self.eventsListChute) and
                        self.eventsListChute[chute_index] == i + 1):
                    effLanding = self.chutes[chute_index].start
                    #There should never be both chute and ladder on same space!
                ladder_index = bsc.bisect_left(self.eventsListLadder, (i+1))
                if (ladder_index < len(self.eventsListLadder) and
                        self.eventsListLadder[ladder_index] == i + 1):
                    effLanding = self.ladders[ladder_index].end
                self.effLandingForHoles.append(effLanding)

    def setHolesetIndexer(self):
        """Creates a sorted index of hole numbers for faster searching."""
        self.holesetIndexer = [h.num for h in self.trackholes]

    def getHoleByCoords(self, coords):
        """Finds a hole object based on spatial coordinates."""
        for h in self.trackholes:
            if h.coords == coords: return h
        return None

    def getHoleByNum(self, holeNum):
        """
        Finds a hole object by its number using binary search on the indexer.
        """
        idx = bsc.bisect_left(self.holesetIndexer, holeNum)
        if idx < len(self.holesetIndexer) and self.holesetIndexer[idx] == holeNum:
            return self.trackholes[idx]
        return None

    def getLaddersAsDF(self):
        """Returns the track's ladders as a pandas DataFrame."""
        return pd.DataFrame.from_records([l.to_dict() for l in self.ladders])

    def getChutesAsDF(self):
        """Returns the track's chutes as a pandas DataFrame."""
        return pd.DataFrame.from_records([c.to_dict() for c in self.chutes])

    def getEventsAsDF(self):
        """Returns a combined DataFrame of both ladders and chutes."""
        templ = [l.to_dict() for l in self.ladders]
        tempc = [c.to_dict() for c in self.chutes]
        templ.extend(tempc)
        return pd.DataFrame.from_records(templ)


class Ladder:
    """Represents a ladder event that moves a player forward."""
    def __init__(self, start, end, track, vector=((-1, -1), (-1, -1)), eventDete=None):
        self.start = start
        self.end = end
        self.length = self.end - self.start
        self.track = track
        self.crowVector = vector
        self.eventDete = eventDete

    def to_dict(self):
        """Converts the ladder properties to a dictionary."""
        return {'start': self.start, 'end': self.end, 'track': self.track}


class Chute:
    """Represents a chute event that moves a player backward."""
    def __init__(self, start, end, track, vector=((-1, -1), (-1, -1)), eventDete=None):
        self.start = start
        self.end = end
        self.length = self.end - self.start
        self.track = track
        self.crowVector = vector
        self.eventDete = eventDete

    def to_dict(self):
        """Converts the chute properties to a dictionary."""
        return {'start': self.start, 'end': self.end, 'track': self.track}