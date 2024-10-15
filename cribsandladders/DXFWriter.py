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


def rotate_vector_2d(direction_vector, angle_deg):
    # Convert the angle to radians
    angle_rad = np.radians(angle_deg)

    # Define the 2D rotation matrix (counterclockwise rotation)
    rotation_matrix = np.array([[np.cos(angle_rad), -np.sin(angle_rad)],
                                [np.sin(angle_rad), np.cos(angle_rad)]])

    # Apply the rotation
    new_direction_vector = np.dot(rotation_matrix, direction_vector)
    return new_direction_vector

def compute_offset_curve(points, offset_distance, proximityThresh):
    # List to store points for the left and right offset curves
    left_curve, right_curve = [], []
    left_arrows, right_arrows = [], []
    direction_vector = np.array([(-1,-1),(-1,-1)])

    #Pad with pre-point and post-point so tracks extends past first and last hole
    if len(points) >= 2:
        p_pre = 1.6*points[0] - 0.6*points[1]
        p_post = 1.6*points[len(points)-1] - 0.6*points[len(points) - 2]
        aug_points = np.vstack([p_pre, points, p_post])
    else: aug_points = points

    # Loop through each pair of consecutive aug_points on the curve
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

        if i%10 == 0:
            # Build arrow unit vectors
            left_arrow_unit_vect = rotate_vector_2d(norm_perp_vector, 80)
            right_arrow_unit_vect = rotate_vector_2d((-1)*norm_perp_vector, -80)

            # Build vectors and append to lists
            left_arrows.append([(aug_points[i] + perp_vector).tolist(), (aug_points[i] + perp_vector +
                                                                         0.25*left_arrow_unit_vect).tolist()])
            right_arrows.append([(aug_points[i] - perp_vector).tolist(), (aug_points[i] - perp_vector +
                                                                          0.25*right_arrow_unit_vect).tolist()])

    # Add the last point offsets (for the endpoint of the curve)
    last_perp_vector = np.array([-direction_vector[1], direction_vector[0]]) * offset_distance
    left_curve.append((aug_points[-1] + last_perp_vector).tolist())
    right_curve.append((aug_points[-1] - last_perp_vector).tolist())

    #Check proximity to holes, if too close remove outright
    left_curve = remove_close_coordinates(left_curve, aug_points, threshold=proximityThresh)
    right_curve = remove_close_coordinates(right_curve, aug_points, threshold=proximityThresh)

    #Check proximity to neighbours, if too close set each to midpoint of each other
    left_curve = adjust_close_points(left_curve, threshold=proximityThresh)
    right_curve = adjust_close_points(right_curve, threshold=proximityThresh)

    #Run 2nd time to clean up
    left_curve = remove_close_coordinates(left_curve, aug_points, threshold=proximityThresh)
    right_curve = remove_close_coordinates(right_curve, aug_points, threshold=proximityThresh)
    left_curve = adjust_close_points(left_curve, threshold=proximityThresh)
    right_curve = adjust_close_points(right_curve, threshold=proximityThresh)

    return left_curve, right_curve, left_arrows, right_arrows

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
        orthogonal_vector = np.array([-direction_vector[1], direction_vector[0]])

        # Normalize and scale the orthogonal vector to half the desired length (0.5 in total length)
        orthogonal_vector *= (length / 2)

        # Find midpoint between 5 hole and next TRY IT OUT
        if last_hole:
            mid_point = (p2 - p1)/2 + p2
        else:
            mid_point = (p2 + p3)/2

        # # Compute the start and end points of the orthogonal vector
        # start_point = p2 - orthogonal_vector
        # end_point = p2 + orthogonal_vector

        # Compute the start and end points of the orthogonal vector TRY IT OUT with midpoint
        start_point = mid_point - orthogonal_vector
        end_point = mid_point + orthogonal_vector

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
        right_curve, left_curve, right_arrows, left_arrows =\
            compute_offset_curve(np.array(holes_in), 0.16, 0.127)
        doc.layers.add(name="TrackPath_T"+str(t.Track_ID), color=rd.randint(1,30), linetype="DOTTED")
        msp.add_spline(right_curve, dxfattribs={'layer': "TrackPath_T"+str(t.Track_ID)})
        msp.add_spline(left_curve, dxfattribs={'layer': "TrackPath_T"+str(t.Track_ID)})

        #Build marker arrows in with spline
        for arrow in right_arrows + left_arrows:
            msp.add_lwpolyline(arrow, dxfattribs={'layer': "TrackPath_T"+str(t.Track_ID)})

        # Every 5th hole draw marker line across
        doc.layers.add(name="NumMarks_T" + str(t.Track_ID), color=rd.randint(1, 30), linetype="DOTTED")
        slashVectors = create_progress_marker_vectors(np.array(holes_in), 0.35) #NOTE this should be 2x offset dist of spline
        for s in slashVectors:
            msp.add_lwpolyline(s, dxfattribs={'layer': "NumMarks_T"+str(t.Track_ID)})

        #Add starter holes + circumference
        msp.add_circle([0, t.num*0.2], holeRadius, dxfattribs={'layer': "Holes_T"+str(t.Track_ID)})
        msp.add_circle([6/25.4, t.num*0.2], holeRadius, dxfattribs={'layer': "Holes_T"+str(t.Track_ID)})
        msp.add_ellipse([3/25.4, t.num*0.2], [9/25.4, t.num*0.2], 0.5,
                        dxfattribs={'layer': "TrackPath_T"+str(t.Track_ID)})

    #Add shared finish hole
    doc.layers.add(name="Holes_Finish", color=rd.randint(1,30), linetype="DASHED")
    msp.add_circle(convert_mm_to_in([(board.width, board.height)])[0], holeRadius, dxfattribs={'layer': "Holes_Finish"})

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