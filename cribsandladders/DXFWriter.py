import ezdxf
import numpy as np
import random as rd
from datetime import datetime
import os
import sqlite3 as sql

# Pure coordinate/vector math lives in dxf_geometry.py (no ezdxf
# dependency) so it can be unit tested without ezdxf installed -- see
# that module's docstring. Re-exported here under the same names so
# nothing else in this file (or any external caller) needs to change.
from cribsandladders.dxf_geometry import (
    convert_mm_to_in,
    euclidean_distance,
    remove_close_coordinates,
    midpoint,
    adjust_close_points,
    searchOrderedListForVal,
    rotate_vector_2d,
    compute_offset_curve,
    create_progress_marker_vectors,
)


def insert_dxf_record(board, optimizerRunSet, optimizerRun):
    """
    Inserts board and event data into the DXFOutLog and DXFOutEvents tables.
    """
    sqlConn = sql.connect("etc/Optimizer.db")
    sqliteCursor = sqlConn.cursor()
    sqliteCursor.execute(
        "INSERT INTO DXFOutLog (OptimizerRunSet, OptimizerRun, Board_ID, Timestamp) VALUES (?, ?, ?, ?)",
        [optimizerRunSet, optimizerRun, board.boardID,
         datetime.now().strftime('%m/%d/%y %H:%M:%S')])
    sqlConn.commit()
    DXF_ID = sqlConn.execute("SELECT MAX(DXF_ID) FROM DXFOutLog WHERE Board_ID = ?", [board.boardID]).fetchone()[0]

    sqliteCursor.execute("BEGIN TRANSACTION")
    for t in board.tracks:
        for e in t.eventSetBuild:
            sqliteCursor.execute(
                "INSERT INTO DXFOutEvents(DXF_ID, Board_ID, Track_ID, CandidateEvent_ID, instanceIsChute, " +
                "instanceIsLadder, instanceIncr, instanceRev) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                [DXF_ID, board.boardID, t.Track_ID, e.eventID, e.instanceIsChute,
                 e.instanceIsLadder, e.instanceIncr, e.instanceRev])
    sqliteCursor.execute("END TRANSACTION")
    sqlConn.commit()


def buildDXFFile(board, output_dir="Boards"):
    """
    Args:
        board: the Board to render.
        output_dir (str): directory the per-board DXF subfolder is
            created under. Defaults to "Boards" (previous hardcoded
            behavior). Parameterized so tests can point this at a temp
            directory instead of writing into the real Boards/ folder.
    """
    # Create a new DXF document
    print("Creating DXF file")
    doc = ezdxf.new('R2010')  # or another DXF version
    doc.header['$INSUNITS'] = 1  # 1 = Inches (imperial units)
    doc.header['$LUNITS'] = 2  # Decimal units (commonly used with inches)
    doc.header['$AUNITS'] = 0  # Angle units (0 = Decimal degrees)
    doc.header['$DIMLUNIT'] = 1  # Dimension length units (1 = Inches)

    msp = doc.modelspace()

    # Build in holes
    holeRadius = 1 / 16  # 1/8" diameter means 1/16" radius
    for t in board.tracks:
        doc.layers.add(name="Holes_T" + str(t.Track_ID), color=rd.randint(1, 30), linetype="DASHED")
        # Mixed units, I know, sue me
        holes_in = convert_mm_to_in([h.coords for h in t.trackholes])
        for h in holes_in:
            msp.add_circle(h, holeRadius, dxfattribs={'layer': "Holes_T" + str(t.Track_ID)})

        # Determine holes containing events
        holesWithEvents = [e.startHole.num for e in t.eventSetBuild] + [e.endHole.num for e in t.eventSetBuild]
        holesWithEvents.sort()

        # Every 5th hole draw marker line across
        doc.layers.add(name="NumMarks_T" + str(t.Track_ID), color=rd.randint(1, 30), linetype="DOTTED")
        slashVectors = create_progress_marker_vectors(np.array(holes_in),
                                                      0.24)  # NOTE this should be 2x offset dist of spline
        for s in slashVectors:
            msp.add_lwpolyline(s, dxfattribs={'layer': "NumMarks_T" + str(t.Track_ID)})

        # Build in spline following along either side of each track
        right_curve, left_curve, arrows = \
            compute_offset_curve(np.array(holes_in), holesWithEvents, slashVectors,
                                 0.12, 0.115)
        doc.layers.add(name="TrackPath_T" + str(t.Track_ID), color=rd.randint(1, 30), linetype="DOTTED")
        right_spline = msp.add_spline(right_curve, dxfattribs={'layer': "TrackPath_T" + str(t.Track_ID)})
        left_spline = msp.add_spline(left_curve, dxfattribs={'layer': "TrackPath_T" + str(t.Track_ID)})

        # Build marker arrows in with spline
        for arrow in arrows:
            msp.add_lwpolyline(arrow, dxfattribs={'layer': "TrackPath_T" + str(t.Track_ID)})

        # Extract spline points for intersection with marker lines
        # num_points_extract = len(t.trackholes)*10
        # right_curve_np = np.array([right_spline.fit_points(i / num_points_extract) for i in range(num_points_extract + 1)])
        # left_curve_np = np.array([left_spline.point(i / num_points_extract) for i in range(num_points_extract + 1)])
        right_curve_np, left_curve_np = np.array(right_curve), np.array(left_curve)

        # Add starter holes + circumference
        msp.add_circle([0, t.num * 0.2], holeRadius, dxfattribs={'layer': "Holes_T" + str(t.Track_ID)})
        msp.add_circle([6 / 25.4, t.num * 0.2], holeRadius, dxfattribs={'layer': "Holes_T" + str(t.Track_ID)})
        rev, starter_circ_points, numincrs, cornercuts = False, [], 9, 2
        x_cur, y_cur = 6 / 25.4 + 0.16, t.num * 0.2 + 0.16
        for x in [-0.16, 6 / 25.4 + 0.16]:
            if rev:
                y_vals = [t.num * 0.2 - 0.16, t.num * 0.2 + 0.16]
            else:
                y_vals = [t.num * 0.2 + 0.16, t.num * 0.2 - 0.16]
            for y in y_vals:
                x_incr, y_incr = (x - x_cur) / numincrs, (y - y_cur) / numincrs
                for i in range(numincrs - cornercuts):  # Don't plot corner, round them off
                    x_cur += x_incr
                    y_cur += y_incr
                    if i >= (cornercuts - 1): starter_circ_points.append((x_cur, y_cur))
                x_cur, y_cur = x, y  # Set to target corner
            rev = not rev
        starter_circ_points.append(starter_circ_points[0])

        circSpline = msp.add_spline(starter_circ_points, dxfattribs={'layer': "TrackPath_T" + str(t.Track_ID)})
        circSpline.closed = True

    # Add shared finish hole
    doc.layers.add(name="Holes_Finish", color=rd.randint(1, 30), linetype="DASHED")
    msp.add_circle(convert_mm_to_in([(board.width, board.height)])[0], holeRadius, dxfattribs={'layer': "Holes_Finish"})

    # Arrow to shared finish hole
    doc.layers.add(name="TrackPath_ALL", color=rd.randint(1, 30), linetype="DOTTED")
    arrow_head = np.array(convert_mm_to_in([(board.width - 4, board.height)])[0])
    arrow_base = arrow_head - np.array([(2 / 25.4), 0])
    arrow_dir_vector = (arrow_head - arrow_base) / np.linalg.norm((arrow_head - arrow_base))

    # Build arrow unit vectors
    left_arrow_vect = rotate_vector_2d((-1) * arrow_dir_vector, -30) * 0.08
    right_arrow_vect = rotate_vector_2d((-1) * arrow_dir_vector, 30) * 0.08

    # Build vectors and print lines into layer
    msp.add_lwpolyline([arrow_base.tolist(), arrow_head.tolist()], dxfattribs={'layer': "TrackPath_ALL"})
    msp.add_lwpolyline([arrow_head.tolist(), (arrow_head + left_arrow_vect).tolist()],
                       dxfattribs={'layer': "TrackPath_ALL"})
    msp.add_lwpolyline([arrow_head.tolist(), (arrow_head + right_arrow_vect).tolist()],
                       dxfattribs={'layer': "TrackPath_ALL"})

    # Build in events
    for t in board.tracks:
        doc.layers.add(name="NormEvents_T" + str(t.Track_ID), color=rd.randint(1, 30), linetype="DASHED")
        doc.layers.add(name="RampUpEvents_T" + str(t.Track_ID), color=rd.randint(1, 30), linetype="DASHED")
        doc.layers.add(name="RampDownEvents_T" + str(t.Track_ID), color=rd.randint(1, 30), linetype="DASHED")
        # Start bottom up, use set so discard duplicates
        for e in t.eventSetBuild:
            curLayer, curVect = "", []
            if e.isOrtho:
                # NOTE we use the [0] for instance end vector, since both vectors are drawn towards the midpint-aligned
                # triangle apex
                curVect = convert_mm_to_in([e.instanceStartVector[0], e.instanceStartVector[1],
                                            e.instanceEndVector[0]])
            else:
                curVect = convert_mm_to_in(e.crowVector)

            if e.instanceIsChute and e.instanceIsLadder:
                curLayer = "NormEvents_T" + str(t.Track_ID)
            elif e.instanceIsLadder:
                curLayer = "RampUpEvents_T" + str(t.Track_ID)
            elif e.instanceIsChute:
                curLayer = "RampDownEvents_T" + str(t.Track_ID)
                curVect.reverse()  # Reverse so ramps from end to start

            msp.add_lwpolyline(curVect, dxfattribs={'layer': curLayer})

    # Save the DXF file
    dirName = os.path.join(output_dir, board.boardName)
    os.makedirs(dirName, exist_ok=True)
    date_time_str = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    fileName = board.boardName + " " + date_time_str + ".dxf"
    doc.saveas(dirName + "/" + fileName)
    print("DXF file has been created: " + fileName)