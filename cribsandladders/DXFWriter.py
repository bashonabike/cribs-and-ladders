import ezdxf
import numpy as np
import random as rd
from datetime import datetime
import os
import matplotlib.pyplot as plt
import bisect as bsc
from shapely.geometry import LineString
import sqlite3 as sql

def insert_dxf_record(board, optimizerRunSet, optimizerRun):
    sqlConn = sql.connect("etc/Optimizer.db")
    sqliteCursor = sqlConn.cursor()
    sqliteCursor.execute("INSERT INTO DXFOutLog (OptimizerRunSet,OptimizerRun,Board_ID,Timestamp) VALUES (?,?,?,?)",
                         [optimizerRunSet, optimizerRun, board.boardID,
                          datetime.now().strftime('%m/%d/%y %H:%M:%S')])
    sqlConn.commit()
    DXF_ID = sqlConn.execute("SELECT MAX(DXF_ID) FROM DXFOutLog WHERE Board_ID = ?", [board.boardID]).fetchone()[0]

    sqliteCursor.execute("BEGIN TRANSACTION")
    for t in board.tracks:
        for e in t.eventSetBuild:
            sqliteCursor.execute("INSERT INTO DXFOutEvents(DXF_ID,Board_ID,Track_ID,CandidateEvent_ID,instanceIsChute," +
                                 "instanceIsLadder,instanceIncr,instanceRev) VALUES (?,?,?,?,?,?,?,?)",
                                 [DXF_ID, board.boardID, t.Track_ID, e.eventID, e.instanceIsChute,
                                  e.instanceIsLadder,e.instanceIncr,e.instanceRev])
    sqliteCursor.execute("END TRANSACTION")
    sqlConn.commit()


def convert_mm_to_in(coordinate_list):
    """
    Convert a list of (x, y) coordinate tuples from millimeters to inches.

    Args:
    coordinate_list (list of tuples): A list of (x, y) coordinate tuples in millimeters.

    Returns:
    list of tuples: A list of (x, y) coordinate tuples converted to inches.
    """
    return [(x / 25.4, y / 25.4) for (x, y) in coordinate_list]


# Function to calculate Euclidean distance between two points
def euclidean_distance(coord1, coord2):
    return np.linalg.norm(np.array(coord1) - np.array(coord2))


# Function to remove coordinates from list1 that are too close to any coordinate in list2
def remove_close_coordinates(list1, list2, threshold=0.125):
    result = []

    for coord1 in list1:
        too_close = False
        for coord2 in list2:
            if euclidean_distance(coord1, coord2) <= threshold:
                too_close = True
                break
        if not too_close:
            result.append(coord1)

    return result


# Function to calculate the midpoint between two points
def midpoint(coord1, coord2):
    return tuple((np.array(coord1) + np.array(coord2)) / 2)


# Function to adjust points that are too close by replacing them with their midpoint
def adjust_close_points(coords, threshold=0.125):
    modified = coords.copy()  # Copy the original list to avoid modifying it during iteration
    n = len(modified)
    i = 0

    while i < n:
        j = i + 1
        while j < n:
            if euclidean_distance(modified[i], modified[j]) <= threshold:
                mid = midpoint(modified[i], modified[j])
                modified[i] = mid
                modified[j] = mid
            j += 1
        i += 1

    return modified

def searchOrderedListForVal(orderedList, val):
    idx = bsc.bisect_left(orderedList, val)
    if idx < len(orderedList) and orderedList[idx] == val: return idx
    return -1


def rotate_vector_2d(direction_vector, angle_deg):
    # Convert the angle to radians
    angle_rad = np.radians(angle_deg)

    # Define the 2D rotation matrix (counterclockwise rotation)
    rotation_matrix = np.array([[np.cos(angle_rad), -np.sin(angle_rad)],
                                [np.sin(angle_rad), np.cos(angle_rad)]])

    # Apply the rotation
    new_direction_vector = np.dot(rotation_matrix, direction_vector)
    return new_direction_vector

def compute_offset_curve(points, holesWithEvents, progressMarkers, offset_distance, proximityThresh):
    # List to store points for the left and right offset curves
    left_curve, right_curve = [], []
    arrows = []
    direction_vector = np.array([(-1,-1),(-1,-1)])

    #Pad with pre-point and post-point so tracks extends past first and last hole
    if len(points) >= 2:
        p_pre = 1.6*points[0] - 0.6*points[1]
        p_post = 1.6*points[len(points)-1] - 0.6*points[len(points) - 2]
        aug_points = np.vstack([p_pre, points, p_post])
    else: aug_points = points

    triggerArrow = False

    # Loop through each pair of consecutive aug_points on the curve
    progMarkerIdx = 0
    for i in range(len(aug_points) - 1):
        # Get two consecutive aug_points
        if i > 0:
            #use prev point ideally
            p1 = aug_points[i-1]
        else:
            p1 = aug_points[i]
        p2 = aug_points[i + 1]

        # Compute the direction vector (from p1 to p2)
        direction_vector = p2 - p1

        # Normalize the direction vector
        direction_vector /= np.linalg.norm(direction_vector)

        # Compute the perpendicular vector (-dy, dx)
        norm_perp_vector = np.array([-direction_vector[1], direction_vector[0]])

        # Scale the perpendicular vector to the offset distance
        perp_vector = norm_perp_vector*offset_distance

        # Compute the aug_points for the left and right offset curves
        left_curve.append((aug_points[i] + perp_vector).tolist())
        right_curve.append((aug_points[i] - perp_vector).tolist())

        #Add progress marker in if in line
        if i > 0 and i%5 == 0 and progMarkerIdx < len(progressMarkers):
            left_curve.append((progressMarkers[progMarkerIdx][0]).tolist())
            right_curve.append((progressMarkers[progMarkerIdx][1]).tolist())
            progMarkerIdx += 1

        if i > 1 and i%7 == 0: triggerArrow = True
        if triggerArrow and i%5 != 0:
            #Check if event on next hole which would muddy up the arrow
            if searchOrderedListForVal(holesWithEvents, i + 1) == -1:
                # Det arrow directional vector
                arrow_dir_vector = aug_points[i + 1] - aug_points[i]
                arrow_dir_vector /= np.linalg.norm(arrow_dir_vector)

                # Det arrow start point
                arrow_base = aug_points[i] + (2/25.4)*arrow_dir_vector
                arrow_head = aug_points[i] + (4/25.4)*arrow_dir_vector

                # Build arrow unit vectors
                left_arrow_vect = rotate_vector_2d((-1)*arrow_dir_vector, -30)*0.05
                right_arrow_vect = rotate_vector_2d((-1)*arrow_dir_vector, 30)*0.05

                # Build vectors and append to lists
                arrows.append([arrow_base.tolist(), arrow_head.tolist()])
                arrows.append([arrow_head.tolist(), (arrow_head + left_arrow_vect).tolist()])
                arrows.append([arrow_head.tolist(), (arrow_head + right_arrow_vect).tolist()])

                triggerArrow = False

    # Add the last point offsets (for the endpoint of the curve)
    last_perp_vector = np.array([-direction_vector[1], direction_vector[0]]) * offset_distance
    left_curve.append((aug_points[-1] + last_perp_vector).tolist())
    right_curve.append((aug_points[-1] - last_perp_vector).tolist())

    #Check proximity to holes, if too close remove outright
    left_curve = remove_close_coordinates(left_curve, aug_points, threshold=proximityThresh)
    right_curve = remove_close_coordinates(right_curve, aug_points, threshold=proximityThresh)

    #Check proximity to neighbours, if too close set each to midpoint of each other
    left_curve = adjust_close_points(left_curve, threshold=proximityThresh - 0.050)
    right_curve = adjust_close_points(right_curve, threshold=proximityThresh - 0.05)

    #Run 2nd time to clean up
    left_curve = remove_close_coordinates(left_curve, aug_points, threshold=proximityThresh)
    right_curve = remove_close_coordinates(right_curve, aug_points, threshold=proximityThresh)
    left_curve = adjust_close_points(left_curve, threshold=proximityThresh - 0.05)
    right_curve = adjust_close_points(right_curve, threshold=proximityThresh - 0.05)

    return left_curve, right_curve, arrows

def create_progress_marker_vectors(hole_list, length):
    vectors = []  # List to store the orthogonal vectors

    # Loop through every 5th point (starting from index 4 for 0-based indexing)
    for i in range(4, len(hole_list), 5):
        # Get the current point and its neighbors
        p1 = hole_list[i - 1]  # Previous point
        p2 = hole_list[i]  # Current point
        last_hole = i + 1 >= len(hole_list)
        p3 = p2 if last_hole else hole_list[i + 1]  # Next point (handle last point)
        #
        # # Compute the direction vector from p1 to p3 (use p1 to p3 for smoother orthogonal vector)
        # direction_vector = p3 - p1

        # Compute literal dir vector since trying 5-slash between 5-hole and next hole
        if last_hole:
            direction_vector = p2 - p1
        else:
            direction_vector = p3 - p2

        # Normalize the direction vector
        direction_vector /= np.linalg.norm(direction_vector)

        # Compute the orthogonal (perpendicular) vector (-dy, dx)
        # Note this is 90º CCW rotation
        orthogonal_vector = np.array([-direction_vector[1], direction_vector[0]])

        # Normalize and scale the orthogonal vector to half the desired length (0.5 in total length)
        orthogonal_vector *= (length / 2)

        # Find midpoint between 5 hole and next TRY IT OUT
        if last_hole:
            mid_point = (p2 - p1)/2 + p2
        else:
            mid_point = (p2 + p3)/2

        # Store the vector as a tuple of (left_point, right_point)
        linepoints = [mid_point + orthogonal_vector, mid_point - orthogonal_vector]
        vectors.append(tuple(linepoints))

    return vectors

def buildDXFFile(board):
    # Create a new DXF document
    print("Creating DXF file")
    doc = ezdxf.new('R2010')  # or another DXF version
    doc.header['$INSUNITS'] = 1  # 1 = Inches (imperial units)
    doc.header['$LUNITS'] = 2    # Decimal units (commonly used with inches)
    doc.header['$AUNITS'] = 0    # Angle units (0 = Decimal degrees)
    doc.header['$DIMLUNIT'] = 1  # Dimension length units (1 = Inches)

    msp = doc.modelspace()

    #Build in holes
    holeRadius = 1/16    # 1/8" diameter means 1/16" radius
    for t in board.tracks:
        doc.layers.add(name="Holes_T"+str(t.Track_ID), color=rd.randint(1,30), linetype="DASHED")
        #Mixed units, I know, sue me
        holes_in = convert_mm_to_in([h.coords for h in t.trackholes])
        for h in holes_in:
            msp.add_circle(h, holeRadius, dxfattribs={'layer': "Holes_T"+str(t.Track_ID)})

        # Determine holes containing events
        holesWithEvents = [e.startHole.num for e in t.eventSetBuild] + [e.endHole.num for e in t.eventSetBuild]
        holesWithEvents.sort()

        # Every 5th hole draw marker line across
        doc.layers.add(name="NumMarks_T" + str(t.Track_ID), color=rd.randint(1, 30), linetype="DOTTED")
        slashVectors = create_progress_marker_vectors(np.array(holes_in),0.24) #NOTE this should be 2x offset dist of spline
        for s in slashVectors:
            msp.add_lwpolyline(s, dxfattribs={'layer': "NumMarks_T"+str(t.Track_ID)})

        #Build in spline following along either side of each track
        right_curve, left_curve, arrows =\
            compute_offset_curve(np.array(holes_in), holesWithEvents, slashVectors,
                                 0.12, 0.115)
        doc.layers.add(name="TrackPath_T"+str(t.Track_ID), color=rd.randint(1,30), linetype="DOTTED")
        right_spline = msp.add_spline(right_curve, dxfattribs={'layer': "TrackPath_T"+str(t.Track_ID)})
        left_spline = msp.add_spline(left_curve, dxfattribs={'layer': "TrackPath_T"+str(t.Track_ID)})

        #Build marker arrows in with spline
        for arrow in arrows:
            msp.add_lwpolyline(arrow, dxfattribs={'layer': "TrackPath_T"+str(t.Track_ID)})

        # #Extract spline points for intersection with marker lines
        # num_points_extract = len(t.trackholes)*10
        # right_curve_np = np.array([right_spline.fit_points(i / num_points_extract) for i in range(num_points_extract + 1)])
        # left_curve_np = np.array([left_spline.point(i / num_points_extract) for i in range(num_points_extract + 1)])
        right_curve_np, left_curve_np = np.array(right_curve), np.array(left_curve)

        #Add starter holes + circumference
        msp.add_circle([0, t.num*0.2], holeRadius, dxfattribs={'layer': "Holes_T"+str(t.Track_ID)})
        msp.add_circle([6/25.4, t.num*0.2], holeRadius, dxfattribs={'layer': "Holes_T"+str(t.Track_ID)})
        rev, starter_circ_points, numincrs, cornercuts = False, [], 9, 2
        x_cur, y_cur = 6/25.4 + 0.16, t.num * 0.2 + 0.16
        for x in [-0.16, 6/25.4 + 0.16]:
            if rev:
                y_vals = [t.num*0.2 - 0.16, t.num*0.2 + 0.16]
            else:
                y_vals = [t.num * 0.2 + 0.16, t.num * 0.2 - 0.16]
            for y in y_vals:
                x_incr, y_incr = (x - x_cur)/numincrs, (y - y_cur)/numincrs
                for i in range(numincrs - cornercuts): #Don't plot corner, round them off
                    x_cur += x_incr
                    y_cur += y_incr
                    if i >= (cornercuts - 1): starter_circ_points.append((x_cur, y_cur))
                x_cur, y_cur = x, y #Set to target corner
            rev = not rev
        starter_circ_points.append(starter_circ_points[0])
        #
        # x_vals, y_vals = zip(*starter_circ_points)
        #
        # # Create a plot
        # plt.plot(x_vals, y_vals, marker='o')
        #
        # # Add labels and title
        # plt.xlabel('X values')
        # plt.ylabel('Y values')
        # plt.title('Plot of (x, y) Coordinates')
        #
        # # Show the plot
        # plt.show()
        # plt.waitforbuttonpress()
        # plt.close()

        circSpline = msp.add_spline(starter_circ_points, dxfattribs={'layer': "TrackPath_T"+str(t.Track_ID)})
        circSpline.closed = True

    #Add shared finish hole
    doc.layers.add(name="Holes_Finish", color=rd.randint(1,30), linetype="DASHED")
    msp.add_circle(convert_mm_to_in([(board.width, board.height)])[0], holeRadius, dxfattribs={'layer': "Holes_Finish"})

    # Arrow to shared finish hole
    doc.layers.add(name="TrackPath_ALL", color=rd.randint(1,30), linetype="DOTTED")
    arrow_head = np.array(convert_mm_to_in([(board.width-4, board.height)])[0])
    arrow_base = arrow_head - np.array([(2 / 25.4), 0])
    arrow_dir_vector = (arrow_head - arrow_base) / np.linalg.norm((arrow_head - arrow_base))

    # Build arrow unit vectors
    left_arrow_vect = rotate_vector_2d((-1) * arrow_dir_vector, -30) * 0.05
    right_arrow_vect = rotate_vector_2d((-1) * arrow_dir_vector, 30) * 0.05

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
        #Start bottom up, use set so discard duplicates
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
                curVect.reverse() #Reverse so ramps from end to start
            elif e.instanceIsChute:
                curLayer = "RampDownEvents_T" + str(t.Track_ID)

            msp.add_lwpolyline(curVect, dxfattribs={'layer': curLayer})

    # Save the DXF file
    dirName = "Boards/" + board.boardName
    os.makedirs(dirName, exist_ok=True)
    date_time_str = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    fileName = board.boardName + " " + date_time_str + ".dxf"
    doc.saveas(dirName + "/" + fileName)
    print("DXF file has been created: " + fileName)