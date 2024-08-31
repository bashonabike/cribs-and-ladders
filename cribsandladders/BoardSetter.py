from xml.dom import minidom
import game_params as gp
import pandas as pd
import sqlite3 as sql
import os
import cribsandladders.BaseLayout as bs
import cribsandladders.PossibleEvents as ps
import bisect as bsc
import cribsandladders.Board as bd

boardDBName = 'Boards/AllBoards.db'

def setBoardFromDb(board,  boardName):
    if not os.path.isfile(boardDBName):
        raise Exception("Board DB file {} not found!".format(boardDBName))

    sqliteConn = sql.connect('Boards/AllBoards.db')
    sqliteCursor = sqliteConn.cursor()

    #get board/track-level data
    #TODO: allow for selection w/o board db entries, write the data into the db
    #Maybe left join, if isnull then write else obtain
    query = ("select b.Board_ID, t.Track_ID, b.Board_Name as boardname, b.Num_Tracks as numtracks," +
    "b.Two_Deck as twodeck,t.Num_On_Board as tracknum,t.Length as length,t.Two_Deck_Length as twodecklength," +
    "t.Colour as colour, b.Track1BoardPath, b.Track2BoardPath, b.Track3BoardPath, b.TwoDeckLineBoardPath, " +
    "b.Width, b.Height " +
             "from Board b LEFT join Track t " +
             "on t.Board_ID = b.Board_ID where b.Board_Name = \'{}\'".format(boardName))
    boardAndTracks_df = getData(sqliteCursor, query, "No board/tracks found for board name \"{}\""
                                     .format(boardName))

    #set board-level info
    board.boardName = boardAndTracks_df.iloc[0]['boardname']
    board.boardID = int(boardAndTracks_df.iloc[0]['Board_ID'])
    board.width = float(boardAndTracks_df.iloc[0]['Width'])
    board.height = float(boardAndTracks_df.iloc[0]['Height'])
    if gp.findmode:
        #just set up blank track stubs
        board.twoDeckLineBoardPath = boardAndTracks_df.iloc[0]['TwoDeckLineBoardPath']
        if (boardAndTracks_df.iloc[0]['Track_ID'] in (None, 0) or
                len(boardAndTracks_df) < boardAndTracks_df.iloc[0]['numtracks']):
            #No tracks found!  Or partial found. generate & insert into table
            sqliteCursor.execute("DELETE FROM Track WHERE Board_ID = ?", [board.boardID])
            sqliteConn.commit()
            for t in range(1, int(boardAndTracks_df.iloc[0]['numtracks'])+1):
                curtrack = bd.Track()
                # Ignoring all other info, twodeck length if exists will be set from SVG file
                curtrack.num = t
                curtrack.holesetfilepath = boardAndTracks_df.iloc[0]["Track{}BoardPath".format(curtrack.num)]
                sqliteCursor.execute("INSERT INTO Track (Board_ID, Num_On_Board) VALUES (?, ?)",
                                   (board.boardID, curtrack.num))
                sqliteConn.commit()
                curtrack.Track_ID = int(sqliteConn.execute(
                    "SELECT Track_ID FROM Track WHERE Board_ID = ? AND Num_On_Board = ?",
                    (board.boardID, curtrack.num)).fetchall()[0][0])
                board.tracks.append(curtrack)
        else:
            for index, trackstub_sr in boardAndTracks_df.iterrows():
                curtrack = bd.Track()
                #Ignoring all other info, twodeck length if exists will be set from SVG file
                curtrack.num = int(trackstub_sr['tracknum'])
                curtrack.Track_ID = int(trackstub_sr['Track_ID'])
                curtrack.holesetfilepath = trackstub_sr["Track{}BoardPath".format(curtrack.num)]
                board.tracks.append(curtrack)

    else:
        #get chutes & ladders for board
        query = ("select c.Track_ID, c.Chute_ID, c.Start as start, c.End as end from Chute c where c.Board_ID = {}"
                 .format(boardAndTracks_df.iloc[0]['Board_ID']))
        # chutes_df = board.getData(sqliteCursor, query, "No chutes found for board name \"{}\"".format(boardName))
        chutes_df = getData(sqliteCursor, query, "", True)
        if len(chutes_df) > 0:
            chutes_df.sort_values(['Track_ID', 'start'])
        query = ("select l.Track_ID, l.Ladder_ID, l.Start as start, l.End as end from Ladder l where l.Board_ID = {}"
                 .format(boardAndTracks_df.iloc[0]['Board_ID']))
        # ladders_df = board.getData(sqliteCursor, query, "No ladders found for board name \"{}\"".format(boardName))
        ladders_df = getData(sqliteCursor, query, "", True)
        if len(ladders_df) > 0:
            ladders_df.sort_values(['Track_ID', 'start'])

        #set track-level info
        batch_ladders = []
        batch_chutes = []

        for index, track_sr in boardAndTracks_df.iterrows():
            curtrack = bd.Track()
            curtrack.num = int(track_sr['tracknum'])
            curtrack.length = int(track_sr['length'])
            if int(track_sr.loc['twodeck']) > 0:
                curtrack.twodeckslength = int(track_sr['twodecklength'])
            else:
                curtrack.twodeckslength = int(track_sr['length'])
            curtrack.efflength = curtrack.twodeckslength if gp.twodecks else curtrack.length
            board.tracks.append(curtrack)

            if len(chutes_df) > 0:
                #set chute-level info
                trackchutes_df = chutes_df.query("Track_ID == {}".format(track_sr['Track_ID']))
                for index, chute_sr in trackchutes_df.iterrows():
                    start = int(chute_sr['start'])
                    end = int(chute_sr['end'])
                    batch_chutes.append(bd.Chute(start, end, curtrack.num))

            if len(ladders_df) > 0:
                #set ladder-level info
                trackladders_df = ladders_df.query("Track_ID == {}".format(track_sr.loc['Track_ID']))
                for index, ladder_sr in trackladders_df.iterrows():
                    start = int(ladder_sr['start'])
                    end = int(ladder_sr['end'])
                    batch_ladders.append(bd.Ladder(start, end, curtrack.num))

        #Set final chutes & ladders, merging all-track events with track-specific events for each track
        for curtrack in board.tracks:
            if len(chutes_df) > 0:
                (curtrack.setChutes(sorted([full_tracks for full_tracks in batch_chutes if full_tracks.track in
                                                    {0, curtrack.num}],key=lambda x: x.start)))
                curtrack.setEventChutes([c.start for c in curtrack.chutes])
            if len(ladders_df) > 0:
                (curtrack.setLadders(sorted([full_tracks for full_tracks in batch_ladders if full_tracks.track in
                                                     {0, curtrack.num}], key=lambda x: x.start)))
                curtrack.setEventLadders([l.start for l in curtrack.ladders])

            #Set descriptive stats
            curtrack.setEventImpedance()

def getData(cursor, query, errorText, overrideException = False):
    cursor.execute(query)
    temp_df = pd.DataFrame(cursor.fetchall(), columns=[d[0] for d in cursor.description])
    if len(temp_df.index) == 0 and not overrideException:
        raise Exception(errorText)
    return temp_df