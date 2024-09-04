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



def create_progress_marker_vectors(hole_list):
    vectors = []

    for i in range(1, len(hole_list) - 1):
        hole = hole_list[i]
        num = i+1

        # Check if the hole number is divisible by 5 or 10
        if num % 5 == 0 and num > 0:
            if i+1 >= len(hole_list):
                prev_hole = hole_list[i - 1]

                # Calculate the vectors
                vec_prev = np.array([hole[0] - prev_hole[0], hole[1] - prev_hole[1]])
                avg_vector = vec_prev
            else:
                prev_hole = hole_list[i - 1]
                next_hole = hole_list[i + 1]

                # Calculate the vectors
                vec_prev = np.array([hole[0] - prev_hole[0], hole[1] - prev_hole[1]])
                vec_next = np.array([next_hole[0] - hole[0], next_hole[1] - hole[1]])

                # Calculate the average vector
                avg_vector = (vec_prev + vec_next) / 2

            # Find the orthogonal vector (rotate 90 degrees)
            orthogonal_vector = np.array([-avg_vector[1], avg_vector[0]])

            # Normalize the orthogonal vector and scale to 1/4 inch
            scale = 1/8
            orthogonal_vector = orthogonal_vector / np.linalg.norm(orthogonal_vector) * scale

            # Calculate the start and end points of the vector for holes divisible by 5
            if num % 10 != 0:
                start_point = np.array([hole[0], hole[1]]) + orthogonal_vector
                end_point = np.array([hole[0], hole[1]]) - orthogonal_vector
                vectors.append((tuple(start_point), tuple(end_point)))

            # For holes divisible by 10, create two parallel vectors displaced by 0.1 inch on either side
            if num % 10 == 0:
                displacement = 1/64  # 0.1 inch
                orthogonal_unit_vector = orthogonal_vector / np.linalg.norm(orthogonal_vector)
                avg_unit_vector  = avg_vector / np.linalg.norm(avg_vector)

                # First vector displaced by +0.1 inch
                start_point1 = np.array(
                    [hole[0], hole[1]]) + orthogonal_vector + avg_unit_vector * displacement
                end_point1 = np.array(
                    [hole[0], hole[1]]) - orthogonal_vector + avg_unit_vector * displacement
                vectors.append((tuple(start_point1), tuple(end_point1)))

                # Second vector displaced by -0.1 inch
                start_point2 = np.array(
                    [hole[0], hole[1]]) + orthogonal_vector - avg_unit_vector * displacement
                end_point2 = np.array(
                    [hole[0], hole[1]]) - orthogonal_vector - avg_unit_vector * displacement
                vectors.append((tuple(start_point2), tuple(end_point2)))

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
        doc.layers.add(name="Holes_T"+str(t.Track_ID), color=rd.randint(1,255), linetype="DASHED")
        #Mixed units, I know, sue me
        holes_in = convert_mm_to_in([h.coords for h in t.trackholes])
        for h in holes_in:
            msp.add_circle(h, holeRadius, dxfattribs={'layer': "Holes_T"+str(t.Track_ID)})

        #Build in spline following each track
        doc.layers.add(name="TrackPath_T"+str(t.Track_ID), color=rd.randint(1,255), linetype="DOTTED")
        msp.add_spline(holes_in, dxfattribs={'layer': "TrackPath_T"+str(t.Track_ID)})

        # Every 5th & 10th hole draw slashes across
        doc.layers.add(name="NumMarks_T" + str(t.Track_ID), color=rd.randint(1, 30), linetype="DOTTED")
        slashVectors = create_progress_marker_vectors(holes_in)
        for s in slashVectors:
            msp.add_lwpolyline(s, dxfattribs={'layer': "NumMarks_T"+str(t.Track_ID)})

    #Build in events
    for t in board.tracks:
        doc.layers.add(name="Events_T" + str(t.Track_ID), color=rd.randint(1, 30), linetype="DASHED")
        #Start bottom up, use set so discard duplicates
        rampingMarkers = []
        for e in t.eventSetBuild:
            #Check if cancel, if so add ramping marker
            if e.instanceIsChute != e.instanceIsLadder:
                rampingMarkers.append(convert_mm_to_in([e.instanceLump])[0])

            if e.isOrtho:
                # NOTE we use the [0] for instance end vector, since both vectors are drawn towards the midpint-aligned
                # triangle apex
                msp.add_lwpolyline(convert_mm_to_in([e.instanceStartVector[0], e.instanceStartVector[1],
                                                 e.instanceEndVector[0]]), dxfattribs={'layer': "Events_T"+str(t.Track_ID)})
            else:
                msp.add_lwpolyline(convert_mm_to_in(e.crowVector), dxfattribs={'layer': "Events_T"+str(t.Track_ID)})


        # #Draw on ramping markers
        doc.layers.add(name="RampingMarks_T" + str(t.Track_ID), color=rd.randint(1, 30), linetype="DOTTED")
        for r in rampingMarkers:
            msp.add_circle(r, 1/32, dxfattribs={'layer': "RampingMarks_T"+str(t.Track_ID)})

    # Save the DXF file
    dirName = "Boards/" + board.boardName
    os.makedirs(dirName, exist_ok=True)
    date_time_str = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    fileName = board.boardName + " " + date_time_str + ".dxf"
    doc.saveas(dirName + "/" + fileName)
    print("DXF file has been created: " + fileName)
