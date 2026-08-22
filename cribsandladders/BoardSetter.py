import pandas as pd
import sqlite3 as sql
import os
import cribsandladders.Board as bd
from cribsandladders.config import GameConfig, DEFAULT_CONFIG

# Kept for backward compatibility with anything importing this name
# directly; setBoardFromDb itself now resolves the path from
# config.db_path (see GameConfig.data_root), which is what fixes the
# previously-hardcoded 'Boards/AllBoards.db' literal that was also
# duplicated inline in the sql.connect() call below.
boardDBName = 'Boards/AllBoards.db'


def setBoardFromDb(board, boardName, config: GameConfig = DEFAULT_CONFIG):
    """
    Populates a Board object with data retrieved from the SQLite database.

    Args:
        board (bd.Board): The board object instance to be populated.
        boardName (str): The unique name of the board to search for in the database.
        config (GameConfig): game configuration (defaults to the module-level
            DEFAULT_CONFIG). Determines the db path (config.db_path),
            whether findmode stub-tracks are built, and whether two-deck
            track lengths are used.

    Raises:
        Exception: If the database file is missing or if no board/track data is found.
    """
    db_path = config.db_path
    if not os.path.isfile(db_path):
        raise Exception("Board DB file {} not found!".format(db_path))

    sqliteConn = sql.connect(db_path)
    sqliteCursor = sqliteConn.cursor()

    # get board/track-level data
    # TODO: allow for selection w/o board db entries, write the data into the db
    # Maybe left join, if isnull then write else obtain
    query = ("select b.Board_ID, t.Track_ID, b.Board_Name as boardname, b.Num_Tracks as numtracks," +
             "b.Two_Deck as twodeck,t.Num_On_Board as tracknum,t.Length as length,t.Two_Deck_Length as twodecklength," +
             "t.Colour as colour, b.Track1BoardPath, b.Track2BoardPath, b.Track3BoardPath, b.TwoDeckLineBoardPath, " +
             "b.Width, b.Height " +
             "from Board b LEFT join Track t " +
             "on t.Board_ID = b.Board_ID where b.Board_Name = \'{}\'".format(boardName))

    boardAndTracks_df = getData(sqliteCursor, query, "No board/tracks found for board name \"{}\""
                                .format(boardName))

    # set board-level info
    board.boardName = boardAndTracks_df.iloc[0]['boardname']
    board.boardID = int(boardAndTracks_df.iloc[0]['Board_ID'])
    board.width = float(boardAndTracks_df.iloc[0]['Width'])
    board.height = float(boardAndTracks_df.iloc[0]['Height'])

    if config.findmode:
        # just set up blank track stubs
        board.twoDeckLineBoardPath = boardAndTracks_df.iloc[0]['TwoDeckLineBoardPath']
        if (boardAndTracks_df.iloc[0]['Track_ID'] in (None, 0) or
                len(boardAndTracks_df) < boardAndTracks_df.iloc[0]['numtracks']):
            # No tracks found!  Or partial found. generate & insert into table
            sqliteCursor.execute("DELETE FROM Track WHERE Board_ID = ?", [board.boardID])
            sqliteConn.commit()
            for t in range(1, int(boardAndTracks_df.iloc[0]['numtracks']) + 1):
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
                # Ignoring all other info, twodeck length if exists will be set from SVG file
                curtrack.num = int(trackstub_sr['tracknum'])
                curtrack.Track_ID = int(trackstub_sr['Track_ID'])
                curtrack.holesetfilepath = trackstub_sr["Track{}BoardPath".format(curtrack.num)]
                board.tracks.append(curtrack)

    else:
        # get chutes & ladders for board
        query = ("select c.Track_ID, c.Chute_ID, c.Start as start, c.End as end from Chute c where c.Board_ID = {}"
                 .format(boardAndTracks_df.iloc[0]['Board_ID']))
        chutes_df = getData(sqliteCursor, query, "", True)
        if len(chutes_df) > 0:
            chutes_df.sort_values(['Track_ID', 'start'])

        query = ("select l.Track_ID, l.Ladder_ID, l.Start as start, l.End as end from Ladder l where l.Board_ID = {}"
                 .format(boardAndTracks_df.iloc[0]['Board_ID']))
        ladders_df = getData(sqliteCursor, query, "", True)
        if len(ladders_df) > 0:
            ladders_df.sort_values(['Track_ID', 'start'])

        hydrate_tracks_from_dataframes(board, boardAndTracks_df, chutes_df, ladders_df, config=config)


def hydrate_tracks_from_dataframes(board, boardAndTracks_df, chutes_df, ladders_df, config: GameConfig = DEFAULT_CONFIG):
    """
    Builds Track/Chute/Ladder domain objects onto `board` (mutated in
    place) purely from already-fetched DataFrames -- no sqlite, no
    filesystem. This is the entire non-findmode branch of
    setBoardFromDb, pulled out so it's unit-testable with small
    hand-built DataFrames instead of a real database.

    Args:
        board (bd.Board): board to populate; board.tracks is appended to
            and is expected to start empty.
        boardAndTracks_df (pd.DataFrame): one row per track, with columns
            Track_ID, tracknum, length, twodeck, twodecklength.
        chutes_df (pd.DataFrame): columns Track_ID, Chute_ID, start, end.
            May be empty (no chutes for this board).
        ladders_df (pd.DataFrame): columns Track_ID, Ladder_ID, start, end.
            May be empty (no ladders for this board).
        config (GameConfig): only config.twodecks is read, to decide
            whether a track's efflength uses its two-deck or
            single-deck length.

    TODO(liam): flagged by
    tests/test_integration_board_optimizer.py (module docstring). This
    function reads `track_sr['Track_ID']` below (to query that track's
    chutes/ladders out of chutes_df/ladders_df) but never assigns it
    onto `curtrack.Track_ID` -- so every `Track` built via this
    (non-findmode) path is left with `Track_ID == 0` (Track's
    `__init__` default), unlike the findmode branch of
    `setBoardFromDb`, which does set it (from a fresh INSERT or an
    existing stub row). Any downstream code that assumes `Track_ID` is
    always populated (e.g. keying `Optimizer` params by track) will
    silently misbehave for boards loaded this way; real call sites
    currently have to fall back to `track.num` instead. Not fixed yet
    since board-identity semantics (is `Track_ID` supposed to be
    optional here, or is this a real gap?) need deciding first.
    """
    # set track-level info
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
        curtrack.efflength = curtrack.twodeckslength if config.twodecks else curtrack.length
        board.tracks.append(curtrack)

        if len(chutes_df) > 0:
            # set chute-level info
            trackchutes_df = chutes_df.query("Track_ID == {}".format(track_sr['Track_ID']))
            for index, chute_sr in trackchutes_df.iterrows():
                start = int(chute_sr['start'])
                end = int(chute_sr['end'])
                batch_chutes.append(bd.Chute(start, end, curtrack.num))

        if len(ladders_df) > 0:
            # set ladder-level info
            trackladders_df = ladders_df.query("Track_ID == {}".format(track_sr.loc['Track_ID']))
            for index, ladder_sr in trackladders_df.iterrows():
                start = int(ladder_sr['start'])
                end = int(ladder_sr['end'])
                batch_ladders.append(bd.Ladder(start, end, curtrack.num))

    # Set final chutes & ladders, merging all-track events with track-specific events for each track
    for curtrack in board.tracks:
        if len(chutes_df) > 0:
            (curtrack.setChutes(sorted([full_tracks for full_tracks in batch_chutes if full_tracks.track in
                                        {0, curtrack.num}], key=lambda x: x.start)))
            curtrack.setEventChutes([c.start for c in curtrack.chutes])
        if len(ladders_df) > 0:
            (curtrack.setLadders(sorted([full_tracks for full_tracks in batch_ladders if full_tracks.track in
                                         {0, curtrack.num}], key=lambda x: x.start)))
            curtrack.setEventLadders([l.start for l in curtrack.ladders])

        # Set descriptive stats
        curtrack.setEventImpedance()


def getData(cursor, query, errorText, overrideException=False):
    """
    Executes a SQL query and returns the results as a pandas DataFrame.

    Args:
        cursor (sqlite3.Cursor): The database cursor to use for execution.
        query (str): The SQL query string.
        errorText (str): The error message to raise if no results are found.
        overrideException (bool, optional): If True, returns an empty DataFrame instead
            of raising an Exception. Defaults to False.

    Returns:
        pd.DataFrame: A DataFrame containing the query results.

    Raises:
        Exception: If no data is found and overrideException is False.
    """
    cursor.execute(query)
    temp_df = pd.DataFrame(cursor.fetchall(), columns=[d[0] for d in cursor.description])
    if len(temp_df.index) == 0 and not overrideException:
        raise Exception(errorText)
    return temp_df