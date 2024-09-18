import ezdxf
import numpy as np
import random as rd
from datetime import datetime
import os

def convert_mm_to_in(coordinate_list):
    """
    Convert a list of (x, y) coordinate tuples from millimeters to inches.

    Args:
    coordinate_list (list of tuples): A list of (x, y) coordinate tuples in millimeters.

    Returns:
    list of tuples: A list of (x, y) coordinate tuples converted to inches.
    """
    return [(x / 25.4, y / 25.4) for (x, y) in coordinate_list]


def compute_offset_curve(points, offset_distance):
    # List to store points for the left and right offset curves
    left_curve = []
    right_curve = []
    direction_vector = np.array([(-1,-1),(-1,-1)])

    # Loop through each pair of consecutive points on the curve
    for i in range(len(points) - 1):
        # Get two consecutive points
        p1 = points[i]
        p2 = points[i + 1]

        # Compute the direction vector (from p1 to p2)
        direction_vector = p2 - p1

        # Normalize the direction vector
        direction_vector /= np.linalg.norm(direction_vector)

        # Compute the perpendicular vector (-dy, dx)
        perp_vector = np.array([-direction_vector[1], direction_vector[0]])

        # Scale the perpendicular vector to the offset distance
        perp_vector *= offset_distance

        # Compute the points for the left and right offset curves
        left_curve.append((p1 + perp_vector).tolist())
        right_curve.append((p1 - perp_vector).tolist())

    # Add the last point offsets (for the endpoint of the curve)
    last_perp_vector = np.array([-direction_vector[1], direction_vector[0]]) * offset_distance
    left_curve.append((points[-1] + last_perp_vector).tolist())
    right_curve.append((points[-1] - last_perp_vector).tolist())

    return left_curve, right_curve

def create_progress_marker_vectors(hole_list, length):
    vectors = []  # List to store the orthogonal vectors

    # Loop through every 5th point (starting from index 4 for 0-based indexing)
    for i in range(4, len(hole_list), 5):
        # Get the current point and its neighbors
        p1 = hole_list[i - 1]  # Previous point
        p2 = hole_list[i]  # Current point
        p3 = hole_list[i + 1] if i + 1 < len(hole_list) else p2  # Next point (handle last point)

        # Compute the direction vector from p1 to p3 (use p1 to p3 for smoother orthogonal vector)
        direction_vector = p3 - p1

        # Normalize the direction vector
        direction_vector /= np.linalg.norm(direction_vector)

        # Compute the orthogonal (perpendicular) vector (-dy, dx)
        orthogonal_vector = np.array([-direction_vector[1], direction_vector[0]])

        # Normalize and scale the orthogonal vector to half the desired length (0.5 in total length)
        orthogonal_vector *= (length / 2)

        # Compute the start and end points of the orthogonal vector
        start_point = p2 - orthogonal_vector
        end_point = p2 + orthogonal_vector

        # Store the vector as a tuple of (start_point, end_point)
        vectors.append((start_point, end_point))

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

        #Build in spline following along either side of each track
        right_curve, left_curve = compute_offset_curve(holes_in, 0.25)
        doc.layers.add(name="TrackPath_T"+str(t.Track_ID), color=rd.randint(1,30), linetype="DOTTED")
        msp.add_spline(right_curve, dxfattribs={'layer': "TrackPath_T"+str(t.Track_ID)})
        msp.add_spline(left_curve, dxfattribs={'layer': "TrackPath_T"+str(t.Track_ID)})

        # Every 5th hole draw marker line across
        doc.layers.add(name="NumMarks_T" + str(t.Track_ID), color=rd.randint(1, 30), linetype="DOTTED")
        slashVectors = create_progress_marker_vectors(holes_in, 0.5) #NOTE this should be 2x offset dist of spline
        for s in slashVectors:
            msp.add_lwpolyline(s, dxfattribs={'layer': "NumMarks_T"+str(t.Track_ID)})

    #Build in events
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
            elif e.instanceIsChute:
                curLayer = "RampDownEvents_T" + str(t.Track_ID)
                curVect.reverse() #Reverse so ramps from end to start

            msp.add_lwpolyline(curVect, dxfattribs={'layer': curLayer})

    # Save the DXF file
    dirName = "Boards/" + board.boardName
    os.makedirs(dirName, exist_ok=True)
    date_time_str = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    fileName = board.boardName + " " + date_time_str + ".dxf"
    doc.saveas(dirName + "/" + fileName)
    print("DXF file has been created: " + fileName)